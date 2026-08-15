#!/usr/bin/env python3
"""
GEFS data fetcher (Task 1.2, T01+T02 slice).

Downloads GEFS reforecast data via Herbie (PyPI package `herbie-data`),
crops to a requested region, and returns an xarray Dataset.

T01 scope: minimal reforecast path (tmax_2m, basic cropping, cycle loop).
T02 scope: dual-variable (tmax_2m + tmin_2m) and 5-member ensemble
protocol (c00 + p01-p04) merged into a member dimension.
Caching / retry / realtime / longitude-wrapping belong to T03-T06.
"""

from datetime import date, datetime, timedelta

import xarray as xr

from herbie import Herbie

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


class GEFSFetcher:
    """Fetch GEFS data and return regional xarray Datasets."""

    def __init__(self, cache_dir="/tmp/gefs_cache", verbose=False):
        self.cache_dir = cache_dir
        self.verbose = verbose

    def download_reforecast(
        self,
        region_bounds,
        date_range,
        members,
        cycles=None,
        forecast_hours=None,
    ):
        """Download GEFS reforecast for every (day, cycle) and merge both
        variables (tmax_2m, tmin_2m) across the requested members.

        Returns a Dataset with `tmax`/`tmin` data_vars and
        time/latitude/longitude/member coordinates. Members must be a subset
        of the v5.9 5-member protocol {0,1,2,3,4} (c00 + p01-p04); anything
        else raises GEFSValidationError. For each (day, cycle) the members are
        concatenated along the `member` dimension first, then the (day, cycle)
        blocks are concatenated along `time`.
        """
        self._validate(date_range, region_bounds, members)
        start, end = date_range
        if cycles is None:
            cycles = REFORECAST_CYCLES

        days = [
            start + timedelta(days=i) for i in range((end - start).days + 1)
        ]
        blocks = []
        for day in days:
            for cycle in cycles:
                init_time = datetime(day.year, day.month, day.day, cycle)
                member_dss = []
                for member in members:
                    var_dss = []
                    for variable in REFORECAST_VARIABLES:
                        h = Herbie(
                            init_time,
                            model="gefs_reforecast",
                            member=member,
                            fxx=0,
                            variable_level=variable,
                            save_dir=self.cache_dir,
                            verbose=self.verbose,
                        )
                        search = self._build_search(variable, forecast_hours)
                        h.download(search=search)
                        ds = h.xarray(search=search)
                        ds = self.extract_region(
                            ds, region_bounds["lat"], region_bounds["lon"]
                        )
                        var_dss.append(ds)
                    member_ds = xr.merge(var_dss).expand_dims(
                        member=[member]
                    )
                    member_dss.append(member_ds)
                blocks.append(xr.concat(member_dss, dim="member"))
        return xr.concat(blocks, dim="time")

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
        """Crop a Dataset to the given lat/lon window (basic slicing).
        GEFS global grids have descending latitudes, so sort ascending first."""
        ds = ds.sortby("latitude")
        return ds.sel(
            latitude=slice(*lat_bounds), longitude=slice(*lon_bounds)
        )

    @staticmethod
    def _validate(date_range, region_bounds, members=None):
        start, end = date_range
        if start > end:
            raise GEFSValidationError(
                f"date_range start {start} is after end {end}"
            )
        lat_lo, lat_hi = region_bounds["lat"]
        if not (-90 <= lat_lo < lat_hi <= 90):
            raise GEFSValidationError(
                f"invalid latitude bounds {region_bounds['lat']}"
            )
        if members is not None:
            invalid = [m for m in members if m not in VALID_MEMBERS]
            if invalid:
                raise GEFSValidationError(
                    f"invalid ensemble member(s) {invalid}; "
                    f"allowed members (c00+p01-p04) are {list(VALID_MEMBERS)}"
                )
