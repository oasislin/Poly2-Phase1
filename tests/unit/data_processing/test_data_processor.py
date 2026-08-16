"""Unit tests for integrated DataProcessor pipeline (Task 1.3 T1.3-05)."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_processing.data_processor import DataProcessor


class TestDataProcessorEndToEnd:
    """Test unified DataProcessor orchestrating time-align -> interpolate -> elevate -> extract."""

    @pytest.fixture
    def mock_gefs_dataset(self):
        """Create a mock 41x41 GEFS dataset covering Shanghai [25-35N, 115-125E]
        with 5 members (c00, p01-p04), multiple forecast windows (fxx=6..60).
        """
        lats = np.linspace(35.0, 25.0, 41)  # 0.25 deg grid
        lons = np.linspace(115.0, 125.0, 41)
        members = ["c00", "p01", "p02", "p03", "p04"]
        fxx_steps = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

        # Base temperature in Kelvin: ~295K (21.85°C)
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        spatial_base_k = 295.15 + 0.05 * (LAT - 25.0) + 0.1 * (LON - 115.0)

        # 5 members x 10 steps x 41 lat x 41 lon
        tmax_data = np.zeros((5, 10, 41, 41))
        tmin_data = np.zeros((5, 10, 41, 41))

        for m_idx, _ in enumerate(members):
            for s_idx, fxx in enumerate(fxx_steps):
                # Daily warming curve peak around fxx=30 (LT 14:00)
                temp_factor = 3.0 if fxx in (24, 30, 36) else 0.0
                tmax_data[m_idx, s_idx, :, :] = spatial_base_k + temp_factor + m_idx * 0.5
                tmin_data[m_idx, s_idx, :, :] = (spatial_base_k - 8.0) + (temp_factor * 0.2) + m_idx * 0.5

        # Model orography elevation: 10m everywhere in plain
        orography = np.full((41, 41), 10.0)

        ds = xr.Dataset(
            {
                "tmax": (["member", "step", "latitude", "longitude"], tmax_data, {"units": "K"}),
                "tmin": (["member", "step", "latitude", "longitude"], tmin_data, {"units": "K"}),
                "orography": (["latitude", "longitude"], orography),
            },
            coords={
                "member": members,
                "step": fxx_steps,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={"init_time": "2019-07-01T00:00:00Z"},
        )
        return ds

    def test_process_shanghai_max_temp(self, mock_gefs_dataset):
        processor = DataProcessor()
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        feature_df = processor.process_forecast_to_features(
            dataset=mock_gefs_dataset,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="max",
            lead_time_bucket=30,
        )

        assert isinstance(feature_df, pd.DataFrame)
        assert len(feature_df) == 1

        row = feature_df.iloc[0]
        assert row["station_id"] == "ZSPD"
        assert row["target_type"] == "max"
        assert row["target_date"] == "2019-07-02"
        assert row["lead_time_bucket"] == 30

        # Base in Celsius: 295.15 - 273.15 = 22.0°C + spatial offset + 3.0 window peak
        # Plus elevation correction: ZSPD elev=4m, model=10m -> delta_h = -6m -> +0.0065*6 = +0.039°C
        assert 25.0 <= row["ensemble_mean"] <= 30.0
        assert row["ensemble_variance"] > 0
        assert row["member_max"] >= row["ensemble_mean"] >= row["member_min"]

    def test_process_shanghai_min_temp(self, mock_gefs_dataset):
        processor = DataProcessor()
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        feature_df = processor.process_forecast_to_features(
            dataset=mock_gefs_dataset,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="min",
            lead_time_bucket=24,
        )

        assert len(feature_df) == 1
        row = feature_df.iloc[0]
        assert row["target_type"] == "min"
        assert row["lead_time_bucket"] == 24
        # Min temperature should be colder than max temperature
        assert row["ensemble_mean"] < 22.0

    def test_batch_process_multiple_targets(self, mock_gefs_dataset):
        processor = DataProcessor()
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)

        records = [
            {"station_id": "ZSPD", "target_date": date(2019, 7, 2), "target_type": "max", "lead_time_bucket": 30},
            {"station_id": "ZSPD", "target_date": date(2019, 7, 2), "target_type": "min", "lead_time_bucket": 24},
        ]

        df = processor.batch_process(mock_gefs_dataset, init_time_utc=init_time, requests=records)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df["target_type"]) == {"max", "min"}
