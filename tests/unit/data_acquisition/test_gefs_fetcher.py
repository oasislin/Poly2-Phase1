"""
Unit tests for the GEFS data fetcher (Task 1.2).

All network access is mocked: we patch `gefs_fetcher.Herbie` with a fake
that returns synthetic xarray Datasets. This keeps the suite fast and
deterministic. A single optional network smoke test is included but
skipped unless RUN_NETWORK_TESTS=1 is set.
"""

import os
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_acquisition import gefs_fetcher
from src.data_acquisition.gefs_fetcher import (
    DEFAULT_REGIONS,
    GEFSDownloadError,
    GEFSFetcher,
    GEFSValidationError,
    REFORECAST_CYCLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_forecast_ds(
    lat_range=(-90.0, 90.0),
    lon_range=(0.0, 360.0),
    init_time=None,
    member=None,
    step_hours=(6, 12),
    variable="t2m",
):
    """Build a synthetic reforecast-style Dataset mirroring the REAL GRIB2
    decode contract (verified 2026-08-15 against 2019010100/c00 6h windows):

    - `time` is the single reference/init time, NOT the forecast window;
    - forecast windows live on a `step` dimension (lead hours);
    - `valid_time` (window end) = time + step.

    This mirrors cfgrib's decode of 6h TMAX/TMIN accumulation messages, where
    multiple windows from one init share one reference time."""
    lats = np.arange(lat_range[0], lat_range[1] + 0.5, 0.5)
    lons = np.arange(lon_range[0], lon_range[1] + 0.5, 0.5)
    if init_time is None:
        init_time = pd.Timestamp("2019-01-01 00:00")
    steps = [pd.Timedelta(hours=h) for h in step_hours]
    rng = np.random.default_rng(42)
    data = rng.random((1, len(steps), len(lats), len(lons)))
    ds = xr.Dataset(
        {variable: (("time", "step", "latitude", "longitude"), data)},
        coords={
            "time": [init_time],
            "step": steps,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds = ds.assign_coords(valid_time=("step", [init_time + s for s in steps]))
    if member is not None:
        ds = ds.expand_dims(member=[member])
    return ds


def make_fake_realtime_ds(init_time, fxx, variable):
    """Mirror a single realtime per-fxx decode: (latitude, longitude) data with
    scalar time/step/valid_time coords (verified 2026-08-15 against
    GEFS atmos.25 f006: TMAX/TMIN decode to (lat, lon), step is scalar)."""
    lats = np.arange(-90.0, 90.1, 0.5)
    lons = np.arange(0.0, 360.1, 0.5)
    init = pd.Timestamp(init_time)
    step = pd.Timedelta(hours=fxx)
    rng = np.random.default_rng(42)
    data = rng.random((len(lats), len(lons)))
    ds = xr.Dataset(
        {variable: (("latitude", "longitude"), data)},
        coords={"time": init, "step": step, "latitude": lats, "longitude": lons},
    )
    return ds.assign_coords(valid_time=init + step)


# GEFS decodes tmax_2m -> tmax and tmin_2m -> tmin; the mock mirrors that.
VARIABLE_DECODE = {"tmax_2m": "tmax", "tmin_2m": "tmin"}


class MockHerbie:
    """Fake Herbie class recording constructor args and download counts."""

    instances = []

    def __init__(
        self,
        date,
        model=None,
        member=None,
        fxx=0,
        variable_level=None,
        product=None,
        save_dir=None,
        verbose=False,
        **kwargs,
    ):
        self.date = pd.to_datetime(date)
        self.model = model
        self.member = member
        self.fxx = fxx
        self.variable_level = variable_level
        self.product = product
        self.save_dir = save_dir
        self.download_calls = 0
        self.fail_downloads = 0  # number of times download() should raise
        self.fail_xarray = 0     # number of times xarray() should raise
        MockHerbie.instances.append(self)

    def download(self, search=None, **kwargs):
        self.download_calls += 1
        if self.fail_downloads > 0:
            self.fail_downloads -= 1
            raise ConnectionError("simulated transient network error")
        return Path("fake.grib2")

    def xarray(self, search=None, **kwargs):
        if self.fail_xarray > 0:
            self.fail_xarray -= 1
            raise ConnectionError("simulated transient network error")
        if self.model == "gefs" and search is not None:
            # realtime: per-fxx single-window decode, variable from the search
            var = "tmin" if "TMIN" in search else "tmax"
            return make_fake_realtime_ds(self.date, self.fxx, var)
        var = VARIABLE_DECODE.get(self.variable_level, "t2m")
        return make_fake_forecast_ds(
            init_time=self.date, step_hours=(6, 12), variable=var
        )


@pytest.fixture(autouse=True)
def mock_herbie(monkeypatch):
    MockHerbie.instances = []
    monkeypatch.setattr(gefs_fetcher, "Herbie", MockHerbie)
    yield MockHerbie


def make_fetcher(**kwargs):
    defaults = dict(cache_dir="/tmp/fake_gefs_cache", verbose=False)
    defaults.update(kwargs)
    return GEFSFetcher(**defaults)


SHANGHAI = DEFAULT_REGIONS["shanghai"]
DENVER = DEFAULT_REGIONS["denver"]


# ---------------------------------------------------------------------------
# Region presets (from the specification)
# ---------------------------------------------------------------------------

class TestDefaultRegions:
    def test_shanghai_region_matches_spec(self):
        assert SHANGHAI["lat"] == (25, 35)
        assert SHANGHAI["lon"] == (115, 125)

    def test_denver_region_matches_spec(self):
        assert DENVER["lat"] == (35, 45)
        assert DENVER["lon"] == (-110, -100)


# ---------------------------------------------------------------------------
# download_reforecast
# ---------------------------------------------------------------------------

class TestDownloadReforecast:
    def test_returns_xarray_dataset_with_expected_coords(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
        )
        assert isinstance(ds, xr.Dataset)
        assert "tmax" in ds.data_vars
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time" in ds.coords
        assert "step" in ds.coords
        assert "valid_time" in ds.coords
        assert ds.sizes["member"] == 1

    def test_returns_both_tmax_and_tmin(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
        )
        assert "tmax" in ds.data_vars
        assert "tmin" in ds.data_vars

    def test_five_members_build_member_dimension(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0, 1, 2, 3, 4],
            cycles=[0],
        )
        assert ds.sizes["member"] == 5
        assert list(ds.member.values) == [0, 1, 2, 3, 4]
        # one (day, cycle) block -> 1 reference time + 2 forecast steps
        assert ds.sizes["time"] == 1
        assert ds.sizes["step"] == 2

    def test_iterates_all_default_cycles(self):
        fetcher = make_fetcher()
        fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
        )
        # reforecast is 00Z-only (v5.9.1): 1 cycle x 1 member x 1 day x 2 variables
        assert len(MockHerbie.instances) == len(REFORECAST_CYCLES) * 2
        cycles_seen = sorted(set(h.date.hour for h in MockHerbie.instances))
        assert cycles_seen == list(REFORECAST_CYCLES)

    def test_uses_reforecast_model_and_both_variables(self):
        fetcher = make_fetcher()
        fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
        )
        variable_levels = sorted({h.variable_level for h in MockHerbie.instances})
        assert variable_levels == ["tmax_2m", "tmin_2m"]
        assert all(h.model == "gefs_reforecast" for h in MockHerbie.instances)

    def test_caches_repeated_calls(self):
        fetcher = make_fetcher()
        kwargs = dict(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
        )
        fetcher.download_reforecast(**kwargs)
        fetcher.download_reforecast(**kwargs)
        # 1 day x 1 cycle x 1 member x 2 variables = 2 instances on first call,
        # 0 new instances on second call
        assert len(MockHerbie.instances) == 2
        assert all(h.download_calls == 1 for h in MockHerbie.instances)

    def test_crops_to_region_bounds(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
        )
        assert ds.latitude.min() >= 25
        assert ds.latitude.max() <= 35
        assert ds.longitude.min() >= 115
        assert ds.longitude.max() <= 125

    def test_crops_to_denver_bounds(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=DENVER,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
        )
        assert ds.latitude.min() >= 35
        assert ds.latitude.max() <= 45
        assert ds.longitude.min() >= -110
        assert ds.longitude.max() <= -100

    def test_multi_day_concats_time(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 2)),
            members=[0],
            cycles=[0],
        )
        assert len(MockHerbie.instances) == 4  # 2 days x 2 variables
        assert ds.sizes["time"] == 2  # 2 reference times (one per day)
        assert ds.sizes["step"] == 2  # 2 forecast steps per init

    def test_invalid_member_raises(self):
        fetcher = make_fetcher()
        with pytest.raises(GEFSValidationError):
            fetcher.download_reforecast(
                region_bounds=SHANGHAI,
                date_range=(date(2019, 1, 1), date(2019, 1, 1)),
                members=[7],
                cycles=[0],
            )

    def test_invalid_date_range_raises(self):
        fetcher = make_fetcher()
        with pytest.raises(GEFSValidationError):
            fetcher.download_reforecast(
                region_bounds=SHANGHAI,
                date_range=(date(2019, 1, 2), date(2019, 1, 1)),
                members=[0],
            )

    def test_invalid_region_bounds_raise(self):
        fetcher = make_fetcher()
        with pytest.raises(GEFSValidationError):
            fetcher.download_reforecast(
                region_bounds={"lat": (-95, 35), "lon": (115, 125)},
                date_range=(date(2019, 1, 1), date(2019, 1, 1)),
                members=[0],
            )


# ---------------------------------------------------------------------------
# _build_search (idx search regex, pinned to observed real GEFS idx content)
# ---------------------------------------------------------------------------

class TestBuildSearch:
    """Contract tests pinning `_build_search` regex to REAL observed GEFS idx
    content (verified 2026-08-15 against
    2019010100/c00 Days:1-10 tmax_2m/tmin_2m .idx files).

    The same file contains BOTH 3h and 6h windows (e.g. "0-3 hour max fcst"
    and "0-6 hour max fcst"), so the regex must not cross-match them.

    Herbie matches with `df.search_this.str.contains(search)` (regex, partial
    match), where `search_this` is the idx line wrapped in colons:
        ":TMAX:2 m above ground:18-24 hour max fcst:ENS=low-res ctl:"
    so these tests use `re.search` against that exact shape.
    """

    @staticmethod
    def _search_this(variable, fcst_window, ens="ENS=low-res ctl"):
        return f":{variable}:2 m above ground:{fcst_window}:{ens}:"

    def test_tmax_uses_max_fcst(self):
        # Real idx: TMAX lines are "... hour max fcst".
        assert re.search(
            GEFSFetcher._build_search("tmax_2m", [3]),
            self._search_this("TMAX", "0-3 hour max fcst"),
        )

    def test_tmin_uses_min_fcst(self):
        # Real idx: TMIN lines are "... hour min fcst" (not max).
        assert re.search(
            GEFSFetcher._build_search("tmin_2m", [3]),
            self._search_this("TMIN", "0-3 hour min fcst"),
        )

    def test_six_hour_window(self):
        # 6h windows exist alongside 3h windows in the same file.
        assert re.search(
            GEFSFetcher._build_search("tmin_2m", [6]),
            self._search_this("TMIN", "0-6 hour min fcst"),
        )

    def test_shanghai_three_window_join_matches_real_lines(self):
        # Shanghai fxx [24, 30, 36] -> "18-24|24-30|30-36 hour ..." windows.
        pattern = GEFSFetcher._build_search("tmax_2m", [24, 30, 36])
        for win in (
            "18-24 hour max fcst",
            "24-30 hour max fcst",
            "30-36 hour max fcst",
        ):
            assert re.search(pattern, self._search_this("TMAX", win))

    def test_tmin_shanghai_three_window_join_matches_real_lines(self):
        pattern = GEFSFetcher._build_search("tmin_2m", [24, 30, 36])
        for win in (
            "18-24 hour min fcst",
            "24-30 hour min fcst",
            "30-36 hour min fcst",
        ):
            assert re.search(pattern, self._search_this("TMIN", win))

    def test_six_hour_window_does_not_cross_match_three_hour(self):
        # A 6h search must NOT match the 3h lines in the same file.
        pattern = GEFSFetcher._build_search("tmax_2m", [6])
        assert not re.search(pattern, self._search_this("TMAX", "0-3 hour max fcst"))
        assert not re.search(pattern, self._search_this("TMAX", "6-9 hour max fcst"))


# ---------------------------------------------------------------------------
# download_realtime
# ---------------------------------------------------------------------------

class TestDownloadRealtime:
    def test_returns_xarray_dataset(self):
        fetcher = make_fetcher()
        ds = fetcher.download_realtime(
            region_bounds=SHANGHAI,
            forecast_time=datetime(2023, 7, 1, 0, 0),
            members=[0],
            fxx_hours=[6],
        )
        assert isinstance(ds, xr.Dataset)
        assert "tmax" in ds.data_vars
        assert "tmin" in ds.data_vars
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords

    def test_uses_realtime_model_and_product(self):
        fetcher = make_fetcher()
        fetcher.download_realtime(
            region_bounds=SHANGHAI,
            forecast_time=datetime(2023, 7, 1, 0, 0),
            members=[0],
            fxx_hours=[6],
        )
        h = MockHerbie.instances[0]
        assert h.model == "gefs"
        assert h.product == "atmos.25"
        assert h.fxx == 6

    def test_crops_to_denver_region(self):
        fetcher = make_fetcher()
        ds = fetcher.download_realtime(
            region_bounds=DENVER,
            forecast_time=datetime(2023, 7, 1, 0, 0),
            members=[0],
            fxx_hours=[6],
        )
        assert ds.latitude.min() >= 35
        assert ds.latitude.max() <= 45
        # Denver is in the western hemisphere; wrapped longitudes normalize
        # to [-180, 180]
        lons = np.asarray(ds.longitude.values)
        assert lons.min() >= -110
        assert lons.max() <= -100

    def test_five_members_build_member_dimension(self):
        fetcher = make_fetcher()
        ds = fetcher.download_realtime(
            region_bounds=SHANGHAI,
            forecast_time=datetime(2023, 7, 1, 0, 0),
            members=[0, 1, 2, 3, 4],
            fxx_hours=[6],
        )
        assert ds.sizes["member"] == 5
        assert list(ds.member.values) == [0, 1, 2, 3, 4]

    def test_multiple_fxx_hours(self):
        fetcher = make_fetcher()
        ds = fetcher.download_realtime(
            region_bounds=SHANGHAI,
            forecast_time=datetime(2023, 7, 1, 0, 0),
            members=[0],
            fxx_hours=[6, 12],
        )
        assert len(MockHerbie.instances) == 2
        fxx_seen = [h.fxx for h in MockHerbie.instances]
        assert fxx_seen == [6, 12]
        # realtime concats per-fxx single-window decodes along `step`
        assert ds.sizes["step"] == 2
        assert ds.sizes["member"] == 1

    def test_invalid_member_raises(self):
        fetcher = make_fetcher()
        with pytest.raises(GEFSValidationError):
            fetcher.download_realtime(
                region_bounds=SHANGHAI,
                forecast_time=datetime(2023, 7, 1, 0, 0),
                members=[9],
                fxx_hours=[6],
            )

    def test_invalid_region_raises(self):
        fetcher = make_fetcher()
        with pytest.raises(GEFSValidationError):
            fetcher.download_realtime(
                region_bounds={"lat": (-95, 35), "lon": (115, 125)},
                forecast_time=datetime(2023, 7, 1, 0, 0),
                members=[0],
            )



# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------

class TestExtractRegion:
    def test_crops_to_bounds(self):
        ds = make_fake_forecast_ds(lat_range=(-90, 90), lon_range=(0, 360))
        region = GEFSFetcher.extract_region(ds, (25, 35), (115, 125))
        assert region.latitude.min() >= 25
        assert region.latitude.max() <= 35
        assert region.longitude.min() >= 115
        assert region.longitude.max() <= 125

    def test_wraps_negative_lon_for_0_360_dataset(self):
        ds = make_fake_forecast_ds(lat_range=(-90, 90), lon_range=(0, 360))
        region = GEFSFetcher.extract_region(ds, (35, 45), (-110, -100))
        # After wrapping + normalization to [-180, 180], longitudes must
        # fall inside the requested western-hemisphere window.
        lons = np.asarray(region.longitude.values)
        assert lons.min() >= -110
        assert lons.max() <= -100

    def test_handles_180_domain_without_wrapping(self):
        ds = make_fake_forecast_ds(lat_range=(-90, 90), lon_range=(-180, 180))
        region = GEFSFetcher.extract_region(ds, (35, 45), (-110, -100))
        assert region.longitude.min() >= -110
        assert region.longitude.max() <= -100

    def test_invalid_lat_bounds_raise(self):
        ds = make_fake_forecast_ds()
        with pytest.raises(GEFSValidationError):
            GEFSFetcher.extract_region(ds, (-95, 35), (115, 125))


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retries_transient_download_error(self):
        fetcher = make_fetcher(max_retries=3, backoff_base=0.01)
        MockHerbie.instances = []
        original_init = MockHerbie.__init__

        def init_with_failures(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.fail_downloads = 2  # fail twice, succeed on 3rd

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(MockHerbie, "__init__", init_with_failures)
        try:
            fetcher.download_reforecast(
                region_bounds=SHANGHAI,
                date_range=(date(2019, 1, 1), date(2019, 1, 1)),
                members=[0],
                cycles=[0],
            )
        finally:
            monkeypatch.undo()
        assert MockHerbie.instances[0].download_calls == 3

    def test_retry_exhausted_raises(self):
        fetcher = make_fetcher(max_retries=2, backoff_base=0.01)
        MockHerbie.instances = []
        original_init = MockHerbie.__init__

        def init_always_fails(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.fail_downloads = 99

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(MockHerbie, "__init__", init_always_fails)
        try:
            with pytest.raises(GEFSDownloadError):
                fetcher.download_reforecast(
                    region_bounds=SHANGHAI,
                    date_range=(date(2019, 1, 1), date(2019, 1, 1)),
                    members=[0],
                    cycles=[0],
                )
        finally:
            monkeypatch.undo()

    def test_retries_transient_xarray_error(self):
        fetcher = make_fetcher(max_retries=3, backoff_base=0.01)
        MockHerbie.instances = []
        original_init = MockHerbie.__init__

        def init_with_xarray_failures(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.fail_xarray = 1

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(MockHerbie, "__init__", init_with_xarray_failures)
        try:
            fetcher.download_reforecast(
                region_bounds=SHANGHAI,
                date_range=(date(2019, 1, 1), date(2019, 1, 1)),
                members=[0],
                cycles=[0],
            )
        finally:
            monkeypatch.undo()
        assert MockHerbie.instances[0].fail_xarray == 0

    def test_does_not_retry_programming_errors(self):
        fetcher = make_fetcher(max_retries=3, backoff_base=0.01)
        calls = []

        def raises_type_error():
            calls.append(1)
            raise TypeError("programming bug, not transient")

        with pytest.raises(TypeError):
            fetcher._execute_with_retry(raises_type_error)
        assert len(calls) == 1  # not retried

    def test_retries_herbie_runtime_error_wrapping_oserror(self):
        # Herbie wraps network/IO failures as RuntimeError (cause = OSError);
        # these must still be retried (regression from R04's OSError narrowing).
        fetcher = make_fetcher(max_retries=3, backoff_base=0.01)
        calls = []

        def raises_herbie_style():
            calls.append(1)
            try:
                raise OSError("network down")
            except OSError as exc:
                raise RuntimeError("Processing failed") from exc

        with pytest.raises(GEFSDownloadError):
            fetcher._execute_with_retry(raises_herbie_style)
        assert len(calls) == 3  # retried max_retries times

    def test_does_not_retry_runtime_error_without_oserror_cause(self):
        fetcher = make_fetcher(max_retries=3, backoff_base=0.01)
        calls = []

        def raises_runtime_bug():
            calls.append(1)
            raise RuntimeError("genuine bug, no IO cause")

        with pytest.raises(RuntimeError):
            fetcher._execute_with_retry(raises_runtime_bug)
        assert len(calls) == 1  # not retried


# ---------------------------------------------------------------------------
# Contained 6h window selection (T05)
# ---------------------------------------------------------------------------

class TestContainedWindows:
    def test_shanghai_contained_windows(self):
        init_time = datetime(2019, 7, 1, 0, 0)
        target_date = date(2019, 7, 2)
        windows = GEFSFetcher.select_contained_windows(
            init_time, target_date, "shanghai"
        )
        assert windows == [24, 30, 36]

    def test_denver_summer_contained_windows(self):
        init_time = datetime(2019, 7, 1, 0, 0)
        target_date = date(2019, 7, 2)
        windows = GEFSFetcher.select_contained_windows(
            init_time, target_date, "denver"
        )
        assert windows == [36, 42, 48, 54]

    def test_denver_winter_contained_windows(self):
        init_time = datetime(2019, 1, 1, 0, 0)
        target_date = date(2019, 1, 2)
        windows = GEFSFetcher.select_contained_windows(
            init_time, target_date, "denver"
        )
        assert windows == [42, 48, 54]


# ---------------------------------------------------------------------------
# Integrity & MD5 (T06)
# ---------------------------------------------------------------------------

class TestIntegrityAndResume:
    def test_calculate_md5_bytes(self):
        data = b"test payload"
        md5_hash = GEFSFetcher.calculate_md5(data)
        assert len(md5_hash) == 32
        assert md5_hash == "c737a42e8172ef241a45e18857b8e544"

    def test_calculate_md5_file(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"test file content")
        expected = GEFSFetcher.calculate_md5(b"test file content")
        assert GEFSFetcher.calculate_md5(p) == expected

    def test_verify_file_md5_success_and_failure(self, tmp_path):
        p = tmp_path / "data.grib2"
        p.write_bytes(b"grib2 content")
        correct_md5 = GEFSFetcher.calculate_md5(b"grib2 content")
        assert GEFSFetcher.verify_file_md5(p, correct_md5) is True
        assert GEFSFetcher.verify_file_md5(p, "wrong_hash") is False
        assert GEFSFetcher.verify_file_md5(tmp_path / "nonexistent.bin", correct_md5) is False

    def test_download_retries_on_md5_mismatch(self, tmp_path):
        good = tmp_path / "good.grib2"
        good.write_bytes(b"correct payload")
        expected = GEFSFetcher.calculate_md5(b"correct payload")

        class FakeHerbie:
            def __init__(self):
                self.calls = 0

            def download(self, search=None, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    # first download is corrupt/missing -> MD5 mismatch
                    return tmp_path / "missing.grib2"
                return good

        h = FakeHerbie()
        fetcher = make_fetcher(max_retries=3, backoff_base=0.01)
        result = fetcher._download_with_retry(h, search="X", expected_md5=expected)
        assert h.calls == 2  # mismatch triggered a re-download
        assert result == good



# ---------------------------------------------------------------------------
# Optional real-network smoke test (skipped unless RUN_NETWORK_TESTS=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="set RUN_NETWORK_TESTS=1 to run real network tests",
)
def test_network_reforecast_single_message(tmp_path):
    """Real download of a single 3h forecast message (small, ~0.7 MB)."""
    from herbie import Herbie

    fetcher = GEFSFetcher(cache_dir=str(tmp_path), verbose=False)
    # Temporarily un-patch Herbie for this test
    real_herbie = gefs_fetcher.Herbie
    gefs_fetcher.Herbie = Herbie
    try:
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
            forecast_hours=[3],
        )
    finally:
        gefs_fetcher.Herbie = real_herbie
    # Real GEFS TMAX/TMIN messages decode as "tmax"/"tmin" (mock mirrors this)
    assert "tmax" in ds.data_vars
    assert "tmin" in ds.data_vars
    assert ds.sizes["time"] >= 1
    assert ds.latitude.min() >= 25
    assert ds.longitude.max() <= 125
