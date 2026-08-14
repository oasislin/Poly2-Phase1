"""
Unit tests for the GEFS data fetcher (Task 1.2).

All network access is mocked: we patch `gefs_fetcher.Herbie` with a fake
that returns synthetic xarray Datasets. This keeps the suite fast and
deterministic. A single optional network smoke test is included but
skipped unless RUN_NETWORK_TESTS=1 is set.
"""

import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import gefs_fetcher
from gefs_fetcher import (
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
    times=None,
    member=None,
    step_hours=(6, 12),
):
    """Build a synthetic reforecast-style Dataset (t2m over time/lat/lon)."""
    lats = np.arange(lat_range[0], lat_range[1] + 0.5, 0.5)
    lons = np.arange(lon_range[0], lon_range[1] + 0.5, 0.5)
    if times is None:
        times = pd.date_range("2019-01-01 06:00", periods=len(step_hours), freq="6h")
    rng = np.random.default_rng(42)
    data = rng.random((len(times), len(lats), len(lons)))
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    if member is not None:
        ds = ds.expand_dims(member=[member])
    return ds


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

    def xarray(self, **kwargs):
        if self.fail_xarray > 0:
            self.fail_xarray -= 1
            raise ConnectionError("simulated transient network error")
        return make_fake_forecast_ds(step_hours=(6, 12))


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
        assert "t2m" in ds.data_vars
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time" in ds.coords
        assert ds.sizes["member"] == 1

    def test_iterates_all_default_cycles(self):
        fetcher = make_fetcher()
        fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
        )
        # 4 cycles (00/06/12/18) x 1 member x 1 day
        assert len(MockHerbie.instances) == len(REFORECAST_CYCLES)
        cycles_seen = sorted(h.date.hour for h in MockHerbie.instances)
        assert cycles_seen == list(REFORECAST_CYCLES)

    def test_uses_reforecast_model_and_variable(self):
        fetcher = make_fetcher()
        fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
            variable="tmp_2m",
        )
        h = MockHerbie.instances[0]
        assert h.model == "gefs_reforecast"
        assert h.variable_level == "tmp_2m"

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
        # Same (date, cycle, member) should not be fetched twice
        assert len(MockHerbie.instances) == 1
        assert MockHerbie.instances[0].download_calls == 1

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

    def test_multi_day_concats_time(self):
        fetcher = make_fetcher()
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 2)),
            members=[0],
            cycles=[0],
        )
        assert len(MockHerbie.instances) == 2  # one per day
        assert ds.sizes["time"] == 4  # 2 days x 2 forecast hours from fake

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
        assert "t2m" in ds.data_vars
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
        assert h.product == "atmos.5"
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


# ---------------------------------------------------------------------------
# Optional real-network smoke test (skipped unless RUN_NETWORK_TESTS=1)
# ---------------------------------------------------------------------------

pytestmark_network = pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="set RUN_NETWORK_TESTS=1 to run real network tests",
)


@pytest.mark.skipif(
    not os.environ.get("RUN_NETWORK_TESTS"),
    reason="set RUN_NETWORK_TESTS=1 to run real network tests",
)
def test_network_reforecast_single_message(tmp_path):
    """Real download of a single 3h forecast message (small, ~0.7 MB)."""
    from herbie import Herbie

    fetcher = GEFSFetcher(cache_dir=str(tmp_path), verbose=False)
    # Temporarily un-patch Herbie for this test
    import gefs_fetcher as gf

    real_herbie = gf.Herbie
    gf.Herbie = Herbie
    try:
        ds = fetcher.download_reforecast(
            region_bounds=SHANGHAI,
            date_range=(date(2019, 1, 1), date(2019, 1, 1)),
            members=[0],
            cycles=[0],
            forecast_hours=[3],
        )
    finally:
        gf.Herbie = real_herbie
    # Real GEFS TMAX messages decode as "tmax" (mock scaffolding uses "t2m")
    assert "tmax" in ds.data_vars
    assert ds.sizes["time"] >= 1
    assert ds.latitude.min() >= 25
    assert ds.longitude.max() <= 125
