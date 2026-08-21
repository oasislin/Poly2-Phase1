#!/usr/bin/env python3
"""
GEFS data fetcher (Task 1.2, T01-T05 slice).

Downloads GEFS reforecast and realtime data via Herbie (PyPI package `herbie-data`),
crops to a requested region, and returns an xarray Dataset.

T01 scope: minimal reforecast path (tmax_2m, basic cropping, cycle loop).
T02 scope: dual-variable (tmax_2m + tmin_2m) and 5-member ensemble
protocol (c00 + p01-p04) merged into a member dimension.
T03 scope: regional cropping and longitude wrapping (0-360 / +/-180).
T04 scope: realtime download mode (model='gefs', product='atmos.25').
T05 scope: completely contained 6h forecast window selection (subseteq local day).
Caching / retry belong to T06.
"""

import hashlib
import logging
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

def check_data_link_health(
    target_url: str = "https://noaa-gefs-retrospective.s3.amazonaws.com",
    timeout: float = 5.0,
) -> dict:
    """Probe network connectivity and latency to the GEFS AWS S3 data source.

    Returns a dict with 'healthy' (bool), 'status_code' (int or None),
    'rtt_ms' (int), and 'message' (str).
    """
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            target_url,
            headers={"User-Agent": "Mozilla/5.0 (Poly-Way2-HealthProbe)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            rtt_ms = int((time.perf_counter() - t0) * 1000)
            code = response.status
            return {
                "healthy": True,
                "status_code": code,
                "rtt_ms": rtt_ms,
                "message": f"正常连通 (HTTP {code}, 响应延迟 {rtt_ms}ms, 无封禁迹象)",
            }
    except urllib.error.HTTPError as exc:
        rtt_ms = int((time.perf_counter() - t0) * 1000)
        # S3 root may return 403 Forbidden or 200, which means S3 is reachable and responding!
        return {
            "healthy": True,
            "status_code": exc.code,
            "rtt_ms": rtt_ms,
            "message": f"源站可达 (HTTP {exc.code}, 响应延迟 {rtt_ms}ms)",
        }
    except Exception as exc:
        rtt_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "healthy": False,
            "status_code": None,
            "rtt_ms": rtt_ms,
            "message": f"连接异常: {exc}",
        }


import xarray as xr

from herbie import Herbie
from src.data_processing.time_aligner import select_contained_6h_windows

DEFAULT_REGIONS = {
    "shanghai": {"lat": (25, 35), "lon": (115, 125)},
    "denver": {"lat": (35, 45), "lon": (-110, -100)},
}

# Reforecast is 00Z-only (v5.9.1 §1): 06/12/18Z do not exist on AWS.
REFORECAST_CYCLES = (0,)

# 5-member ensemble protocol (v5.9.1): c00 (member 0) + p01-p04 (members 1-4).
# Training (reforecast) and prediction (realtime) MUST use the same set;
# any other member id is invalid.
VALID_MEMBERS = (0, 1, 2, 3, 4)

# Reforecast stores TMAX and TMIN in separate files; T02 downloads both and
# merges them into one Dataset (decoded names tmax / tmin).
REFORECAST_VARIABLES = ("tmax_2m", "tmin_2m")

# variable_level (file naming) -> regex matching wgrib2-style idx content
VARIABLE_SEARCH = {
    "tmax_2m": "TMAX:2 m above ground",
    "tmin_2m": "TMIN:2 m above ground",
}


class GEFSValidationError(Exception):
    """Raised when user-supplied arguments are invalid."""


class GEFSDownloadError(Exception):
    """Raised when a download/read from the data source fails."""


# Transient errors worth retrying: network/IO failures only. Programming errors
# (TypeError, KeyError, GEFSValidationError, ...) must propagate immediately and
# not be masked by a retry loop. OSError covers requests.exceptions
# (RequestException subclasses IOError). Herbie wraps those IO errors as
# RuntimeError, so `_is_retryable` unwraps that case too.
RETRYABLE_EXCEPTIONS = (OSError,)


class GEFSFetcher:
    """Fetch GEFS data and return regional xarray Datasets with retry, cache,
    and integrity verification."""

    def __init__(
        self,
        cache_dir="/tmp/gefs_cache",
        max_retries=3,
        backoff_base=0.5,
        verbose=False,
    ):
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.verbose = verbose
        self._cache = {}

    @staticmethod
    def check_link_health(timeout: float = 5.0) -> dict:
        """Probe AWS S3 link health."""
        return check_data_link_health(timeout=timeout)

    @staticmethod
    def _is_retryable(exc) -> bool:
        """True for transient network/IO failures.

        Herbie wraps requests/IO errors as RuntimeError (its ``__cause__`` is the
        original OSError); unwrap those so they retry too. Herbie also raises
        ValueError when S3 idx index file fetch times out or fails remotely.
        """
        if isinstance(exc, OSError):
            return True
        if isinstance(exc, RuntimeError) and isinstance(exc.__cause__, OSError):
            return True
        if isinstance(exc, ValueError) and any(
            msg in str(exc) for msg in ("No index file", "Cant open index", "index file was found")
        ):
            return True
        return False

    def _execute_with_retry(self, func):
        """Execute func with exponential backoff retry on transient network/IO errors."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return func()
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                last_exc = exc
                health = self.check_link_health()
                logger.warning(
                    f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed ({exc}). "
                    f"链路诊断: {health['message']}"
                )
                if attempt < self.max_retries - 1:
                    sleep_s = self.backoff_base * (2 ** attempt)
                    time.sleep(sleep_s)
        health = self.check_link_health()
        raise GEFSDownloadError(
            f"Operation failed after {self.max_retries} attempts: {last_exc}. "
            f"链路状态: {health['message']}"
        ) from last_exc

    def _download_with_retry(self, h, search=None, expected_md5=None):
        def _attempt():
            if getattr(h, "idx", None) is None and getattr(h, "grib", None):
                h.idx = f"{h.grib}.idx"
                h.IDX_STYLE = "wgrib2"
                h.idx_source = "aws"
            if "index_as_dataframe" in h.__dict__:
                del h.__dict__["index_as_dataframe"]
            path = h.download(search=search) if search is not None else h.download()
            if expected_md5 is not None:
                paths = path if isinstance(path, (list, tuple)) else [path]
                for p in paths:
                    if not self.verify_file_md5(p, expected_md5):
                        raise OSError(f"MD5 mismatch for downloaded file {p}")
            return path

        def _attempt_with_cleanup():
            try:
                return _attempt()
            except Exception:
                # Herbie leaves a partial subset file when a download dies
                # mid-stream; drop it so the retry re-downloads instead of
                # reusing the truncated file (Herbie skips existing files).
                try:
                    local = h.get_localFilePath(search)
                    if local.exists():
                        local.unlink()
                except Exception:
                    pass
                raise

        return self._execute_with_retry(_attempt_with_cleanup)

    def _xarray_with_retry(self, h, search=None):
        def _attempt():
            if getattr(h, "idx", None) is None and getattr(h, "grib", None):
                h.idx = f"{h.grib}.idx"
                h.IDX_STYLE = "wgrib2"
                h.idx_source = "aws"
            if "index_as_dataframe" in h.__dict__:
                del h.__dict__["index_as_dataframe"]
            if search is not None:
                return h.xarray(search=search)
            return h.xarray()

        return self._execute_with_retry(_attempt)

    def download_reforecast(
        self,
        region_bounds,
        date_range,
        members,
        cycles=None,
        forecast_hours=None,
        expected_md5=None,
    ):
        """Download GEFS reforecast for every (day, cycle) and merge both
        variables (tmax_2m, tmin_2m) across the requested members.

        Returns a Dataset with `tmax`/`tmin` data_vars and
        time/step/latitude/longitude/member coordinates (the `step` dim holds
        forecast windows, `time` is the reference/init time; see R03). Members
        must be a subset of the v5.9 5-member protocol {0,1,2,3,4}
        (c00 + p01-p04); anything else raises GEFSValidationError. For each
        (day, cycle) the members are concatenated along `member` first, then the
        (day, cycle) blocks are concatenated along `time`.

        If `expected_md5` is given it is verified against each freshly
        downloaded file; a mismatch raises OSError and triggers a re-download
        via the retry loop. Byte-level resume is delegated to Herbie's
        `save_dir` cache (Herbie skips files it already has).
        """
        self._validate(date_range=date_range, region_bounds=region_bounds, members=members)
        start, end = date_range
        if cycles is None:
            cycles = REFORECAST_CYCLES

        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        blocks = []
        for day in days:
            for cycle in cycles:
                init_time = datetime(day.year, day.month, day.day, cycle)
                member_dss = []
                for member in members:
                    var_dss = [
                        self._fetch_variable(
                            init_time,
                            member,
                            variable,
                            forecast_hours,
                            region_bounds,
                            expected_md5,
                        )
                        for variable in REFORECAST_VARIABLES
                    ]
                    member_ds = xr.merge(
                        var_dss, compat="override", combine_attrs="override"
                    ).expand_dims(member=[member])
                    member_dss.append(member_ds)
                blocks.append(
                    xr.concat(
                        member_dss,
                        dim="member",
                        coords="minimal",
                        compat="override",
                    )
                )
        return xr.concat(
            blocks, dim="time", coords="minimal", compat="override"
        )

    def _fetch_variable(
        self,
        init_time,
        member,
        variable,
        forecast_hours,
        region_bounds,
        expected_md5=None,
    ):
        """Download one (init, member, variable) reforecast file, crop it, and
        memoize the result.

        Byte-level resume is delegated to Herbie's own `save_dir` cache (Herbie
        skips files it already has); the in-process `self._cache` memo avoids
        re-downloading within a run. If `expected_md5` is provided the freshly
        downloaded file is verified against it, and a mismatch raises OSError so
        the retry loop re-downloads.
        """
        cache_key = (
            "reforecast",
            init_time,
            member,
            variable,
            tuple(forecast_hours or ()),
            region_bounds["lat"],
            region_bounds["lon"],
        )
        if cache_key in self._cache:
            return self._cache[cache_key].copy(deep=True)
        search = self._build_search(variable, forecast_hours)

        def _fetch_attempt():
            h = Herbie(
                init_time,
                model="gefs_reforecast",
                member=member,
                fxx=0,
                variable_level=variable,
                save_dir=self.cache_dir,
                verbose=self.verbose,
            )
            self._download_with_retry(h, search=search, expected_md5=expected_md5)
            ds = self._xarray_with_retry(h, search=search)
            return self.extract_region(ds, region_bounds["lat"], region_bounds["lon"])

        ds = self._execute_with_retry(_fetch_attempt)
        self._cache[cache_key] = ds
        return ds.copy(deep=True)

    def download_realtime(
        self,
        region_bounds,
        forecast_time,
        members,
        fxx_hours=None,
    ):
        """Download realtime GEFS data (model="gefs", product="atmos.25") and
        extract tmax_2m/tmin_2m 6h windows across the requested members and
        forecast hours (fxx).

        `atmos.25` is the 0.25-degree grid, matching reforecast's resolution
        (Shanghai crop = 41x41). `atmos.5` is 0.5-degree (21x21) and would
        break the train/predict grid-alignment contract (v5.9.1 / spec).

        Unlike reforecast (variable-per-file), realtime atmos.25 files are
        per-fxx and contain all variables, so tmax/tmin are selected with the
        same `_build_search` regex via Herbie's `search` argument (verified
        2026-08-15: realtime idx lines are identical to reforecast, e.g.
        ":TMAX:2 m above ground:0-6 hour max fcst:ENS=low-res ctl:").

        Returns a Dataset with `tmax`/`tmin` data_vars, a `step` dimension
        (forecast windows), a `member` dimension, and latitude/longitude.
        """
        self._validate(region_bounds=region_bounds, members=members)
        if fxx_hours is None:
            fxx_hours = [6]

        member_dss = []
        for member in members:
            fxx_dss = []
            for fxx in fxx_hours:
                h = Herbie(
                    forecast_time,
                    model="gefs",
                    product="atmos.25",
                    member=member,
                    fxx=fxx,
                    save_dir=self.cache_dir,
                    verbose=self.verbose,
                )
                var_dss = []
                for variable in REFORECAST_VARIABLES:
                    cache_key = (
                        "realtime",
                        forecast_time,
                        member,
                        fxx,
                        variable,
                        region_bounds["lat"],
                        region_bounds["lon"],
                    )
                    if cache_key in self._cache:
                        vds = self._cache[cache_key].copy(deep=True)
                    else:
                        search = self._build_search(variable, [fxx])
                        self._download_with_retry(h, search=search)
                        vds = self._xarray_with_retry(h, search=search)
                        vds = self.extract_region(
                            vds, region_bounds["lat"], region_bounds["lon"]
                        )
                        self._cache[cache_key] = vds
                        vds = vds.copy(deep=True)
                    var_dss.append(vds)
                fxx_dss.append(
                    xr.merge(var_dss, compat="override", combine_attrs="override")
                )
            member_ds = xr.concat(
                fxx_dss, dim="step", coords="minimal", compat="override"
            ).expand_dims(member=[member])
            member_dss.append(member_ds)
        if len(member_dss) == 1:
            return member_dss[0]
        return xr.concat(
            member_dss, dim="member", coords="minimal", compat="override"
        )

    @staticmethod
    def calculate_md5(data_or_path) -> str:
        """Calculate MD5 checksum of bytes or file path."""
        hasher = hashlib.md5()
        if isinstance(data_or_path, (str, Path)):
            p = Path(data_or_path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {data_or_path}")
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
        elif isinstance(data_or_path, bytes):
            hasher.update(data_or_path)
        else:
            raise TypeError("Expected bytes or file path for MD5 calculation")
        return hasher.hexdigest()

    @staticmethod
    def verify_file_md5(file_path, expected_md5: str) -> bool:
        """Verify MD5 checksum of a file against an expected hash."""
        try:
            return GEFSFetcher.calculate_md5(file_path) == expected_md5
        except (FileNotFoundError, TypeError):
            return False

    @staticmethod
    def _build_search(variable, forecast_hours):
        """Build a regex matching idx lines (wgrib2 style) for the requested
        variable and forecast hours; falls back to the variable name itself.

        Real idx lines differ by fcst type: TMAX is "X-Y hour max fcst" while
        TMIN is "X-Y hour min fcst"."""
        base = VARIABLE_SEARCH.get(variable, variable)
        if not forecast_hours:
            return base
        fcst = "min" if "tmin" in variable else "max"
        windows = "|".join(
            (f"{h - 6}-{h} hour {fcst} fcst" if h % 6 == 0 else f"{h - 3}-{h} hour {fcst} fcst")
            for h in forecast_hours
        )
        return rf"{base}:(?:{windows})"

    @staticmethod
    def extract_region(ds, lat_bounds, lon_bounds):
        """Crop a Dataset to the given lat/lon window (handling 0-360 and +/-180
        longitude representations).

        GEFS global grids have descending latitudes, so sort ascending first."""
        lat_lo, lat_hi = lat_bounds
        if not (-90 <= lat_lo < lat_hi <= 90):
            raise GEFSValidationError(
                f"invalid latitude bounds {lat_bounds}"
            )
        lon_lo, lon_hi = lon_bounds
        if not (lon_lo < lon_hi):
            raise GEFSValidationError(
                f"invalid longitude bounds {lon_bounds}"
            )

        ds = ds.sortby("latitude")

        ds_min_lon = float(ds.longitude.min())
        ds_max_lon = float(ds.longitude.max())
        is_0_360 = ds_min_lon >= 0 and ds_max_lon > 180

        if is_0_360 and (lon_lo < 0 or lon_hi < 0):
            slice_lon_lo = lon_lo % 360
            slice_lon_hi = lon_hi % 360
            if slice_lon_lo <= slice_lon_hi:
                sub_ds = ds.sel(
                    latitude=slice(lat_lo, lat_hi),
                    longitude=slice(slice_lon_lo, slice_lon_hi),
                )
            else:
                part1 = ds.sel(
                    latitude=slice(lat_lo, lat_hi),
                    longitude=slice(slice_lon_lo, 360),
                )
                part2 = ds.sel(
                    latitude=slice(lat_lo, lat_hi),
                    longitude=slice(0, slice_lon_hi),
                )
                sub_ds = xr.concat([part1, part2], dim="longitude")

            normalized_lons = ((sub_ds.longitude + 180) % 360) - 180
            return sub_ds.assign_coords(longitude=normalized_lons).sortby(
                "longitude"
            )
        else:
            return ds.sel(
                latitude=slice(lat_lo, lat_hi), longitude=slice(lon_lo, lon_hi)
            )

    @staticmethod
    def _validate(date_range=None, region_bounds=None, members=None):
        if date_range is not None:
            start, end = date_range
            if start > end:
                raise GEFSValidationError(
                    f"date_range start {start} is after end {end}"
                )
        if region_bounds is not None:
            lat_lo, lat_hi = region_bounds["lat"]
            if not (-90 <= lat_lo < lat_hi <= 90):
                raise GEFSValidationError(
                    f"invalid latitude bounds {region_bounds['lat']}"
                )
            lon_lo, lon_hi = region_bounds["lon"]
            if not (lon_lo < lon_hi):
                raise GEFSValidationError(
                    f"invalid longitude bounds {region_bounds['lon']}"
                )
        if members is not None:
            invalid = [m for m in members if m not in VALID_MEMBERS]
            if invalid:
                raise GEFSValidationError(
                    f"invalid ensemble member(s) {invalid}; "
                    f"allowed members (c00+p01-p04) are {list(VALID_MEMBERS)}"
                )

    @staticmethod
    def select_contained_windows(
        init_time_utc: datetime,
        target_date: date,
        tz_or_station: str,
        max_lead_hours: int = 120,
    ) -> list[int]:
        """Select forecast step hours where 6h intervals are completely contained
        within the target local calendar day."""
        return select_contained_6h_windows(
            init_time_utc, target_date, tz_or_station, max_lead_hours=max_lead_hours
        )
