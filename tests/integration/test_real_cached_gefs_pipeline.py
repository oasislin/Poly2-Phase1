"""Integration test using real cached GRIB2 files from data/raw/gefs_probe (Task 1.3 / Task 1.4).

Verifies that the entire DataProcessor -> DataValidator pipeline correctly works on
real meteorological GRIB2 data downloaded from NOAA AWS without network access.
"""

from datetime import date, datetime, timezone
from pathlib import Path

import cfgrib
import numpy as np
import pytest
import xarray as xr

from src.data_acquisition.gefs_fetcher import GEFSFetcher
from src.data_processing.data_processor import DataProcessor
from src.data_processing.data_validator import DataValidator

PROBE_DIR = Path("data/raw/gefs_probe")


def _decode_and_crop(grib_path: Path, lat_bounds: tuple, lon_bounds: tuple) -> xr.Dataset:
    """Decode a real local GRIB2 file and extract target region."""
    dss = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
        decode_timedelta=True,
    )
    return GEFSFetcher.extract_region(dss[0], lat_bounds, lon_bounds)


class TestRealCachedGEFSPipeline:
    """Test full data processing and validation on real cached NOAA GRIB2 subsets."""

    @pytest.fixture
    def shanghai_real_summer_dataset(self):
        """Load real 2019-07-01 00Z GRIB2 subsets for Shanghai (ZSPD)."""
        tmax_file = PROBE_DIR / "gefs_reforecast/20190701/subset_19efc1fa__tmax_2m_2019070100_c00.grib2"
        tmin_file = PROBE_DIR / "gefs_reforecast/20190701/subset_19efc1fa__tmin_2m_2019070100_c00.grib2"

        if not (tmax_file.exists() and tmin_file.exists()):
            pytest.skip(f"Real GRIB2 cache files not found at {PROBE_DIR}")

        region_lat = (25.0, 35.0)
        region_lon = (115.0, 125.0)

        tmax_ds = _decode_and_crop(tmax_file, region_lat, region_lon)
        tmin_ds = _decode_and_crop(tmin_file, region_lat, region_lon)

        # Merge tmax and tmin into unified dataset
        ds = xr.merge([tmax_ds, tmin_ds], compat="override")
        # Add member coordinate (c00) for ensemble processing
        if "member" not in ds.dims:
            ds = ds.expand_dims(member=["c00"])

        # Map step timedelta to forecast hour integers (fxx in [24, 30, 36])
        if "step" in ds.coords:
            # Step in timedelta (e.g. 1 days, 1 days 06:00:00, 1 days 12:00:00)
            step_hours = [int(td / np.timedelta64(1, "h")) for td in ds.step.values]
            ds = ds.assign_coords(step=step_hours)

        return ds

    @pytest.fixture
    def denver_real_winter_dataset(self):
        """Load real 2019-01-01 00Z GRIB2 subsets for Denver (KDEN)."""
        tmax_file = PROBE_DIR / "gefs_reforecast/20190101/subset_97ef3872__tmax_2m_2019010100_c00.grib2"
        tmin_file = PROBE_DIR / "gefs_reforecast/20190101/subset_97ef3872__tmin_2m_2019010100_c00.grib2"

        if not (tmax_file.exists() and tmin_file.exists()):
            pytest.skip(f"Real GRIB2 cache files not found at {PROBE_DIR}")

        region_lat = (35.0, 45.0)
        region_lon = (-110.0, -100.0)

        tmax_ds = _decode_and_crop(tmax_file, region_lat, region_lon)
        tmin_ds = _decode_and_crop(tmin_file, region_lat, region_lon)

        ds = xr.merge([tmax_ds, tmin_ds], compat="override")
        if "member" not in ds.dims:
            ds = ds.expand_dims(member=["c00"])

        if "step" in ds.coords:
            step_hours = [int(td / np.timedelta64(1, "h")) for td in ds.step.values]
            ds = ds.assign_coords(step=step_hours)

        return ds

    def test_shanghai_real_data_processing_and_validation(self, shanghai_real_summer_dataset):
        processor = DataProcessor()
        validator = DataValidator(strict=True)

        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        # 1. Process Max Temperature Feature
        df_max = processor.process_forecast_to_features(
            dataset=shanghai_real_summer_dataset,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="max",
            lead_time_bucket=30,
        )

        # 2. Strict Validation with DataValidator
        val_max = validator.validate_features(df_max)
        assert val_max.is_valid is True
        assert len(val_max.errors) == 0

        # 3. Assert real meteorological plausibility for Shanghai in July
        # Real summer max temp in Shanghai is typically between 25°C and 38°C
        tmax_mean = df_max["ensemble_mean"].iloc[0]
        assert 24.0 <= tmax_mean <= 38.0
        assert df_max["member_max"].iloc[0] >= df_max["member_min"].iloc[0]

        # 4. Process Min Temperature Feature
        df_min = processor.process_forecast_to_features(
            dataset=shanghai_real_summer_dataset,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="min",
            lead_time_bucket=24,
        )

        val_min = validator.validate_features(df_min)
        assert val_min.is_valid is True
        tmin_mean = df_min["ensemble_mean"].iloc[0]
        # Summer min temp in Shanghai is typically between 20°C and 30°C
        assert 18.0 <= tmin_mean <= 30.0
        assert tmin_mean < tmax_mean

    def test_denver_real_data_processing_and_validation(self, denver_real_winter_dataset):
        processor = DataProcessor()
        validator = DataValidator(strict=True)

        init_time = datetime(2019, 1, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 1, 2)

        # Process Denver winter max temp
        df_kden = processor.process_forecast_to_features(
            dataset=denver_real_winter_dataset,
            station_id="KDEN",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="max",
            lead_time_bucket=30,
        )

        val_kden = validator.validate_features(df_kden)
        assert val_kden.is_valid is True

        # Denver in January typically has max temp around -10°C to 15°C
        tmax_kden = df_kden["ensemble_mean"].iloc[0]
        assert -15.0 <= tmax_kden <= 20.0
