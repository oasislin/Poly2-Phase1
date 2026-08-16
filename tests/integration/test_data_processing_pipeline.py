"""Integration tests for the complete Data Processing Pipeline (Task 1.3 T1.3-05).

Verifies the entire data transformation flow from raw 0.25° GEFS grid to station features:
1. Spatial Bilinear Interpolation (0.25° grid -> Station lat/lon)
2. Time Alignment (UTC -> Local Day completely contained 6h windows)
3. Unit Standardization (Kelvin -> Celsius)
4. Elevation Correction (Lapse rate Γ = 0.0065 K/m)
5. Ensemble Feature Extraction (5 members -> {mean, variance, max, min})
"""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_processing import DataProcessor


class TestDataProcessingIntegrationPipeline:
    """Integration test suite for the complete data processing pipeline."""

    def test_shanghai_summer_pipeline_execution(self):
        # 1. Simulate a GEFS dataset for Shanghai (ZSPD) on 2019-07-01 00Z init
        # Covering Shanghai region: lat 25-35N, lon 115-125E (41x41 grid)
        lats = np.linspace(35.0, 25.0, 41)
        lons = np.linspace(115.0, 125.0, 41)
        members = ["c00", "p01", "p02", "p03", "p04"]
        fxx_steps = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

        # 300K ~ 26.85°C
        tmax_grid = np.zeros((5, 10, 41, 41))
        tmin_grid = np.zeros((5, 10, 41, 41))

        for m_idx in range(5):
            for s_idx, fxx in enumerate(fxx_steps):
                tmax_grid[m_idx, s_idx, :, :] = 300.0 + m_idx * 0.4 + (1.0 if fxx in (24, 30, 36) else 0.0)
                tmin_grid[m_idx, s_idx, :, :] = 290.0 + m_idx * 0.4

        ds = xr.Dataset(
            {
                "tmax": (["member", "step", "latitude", "longitude"], tmax_grid, {"units": "K"}),
                "tmin": (["member", "step", "latitude", "longitude"], tmin_grid, {"units": "K"}),
                "orography": (["latitude", "longitude"], np.full((41, 41), 10.0)),
            },
            coords={
                "member": members,
                "step": fxx_steps,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={"station_id": "ZSPD"},
        )

        processor = DataProcessor()
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        # 2. Process Max Temperature
        df_max = processor.process_forecast_to_features(
            dataset=ds,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="max",
            lead_time_bucket=30,
        )

        assert isinstance(df_max, pd.DataFrame)
        assert len(df_max) == 1
        assert df_max["station_id"].iloc[0] == "ZSPD"
        assert df_max["target_type"].iloc[0] == "max"
        # 301K - 273.15 = 27.85°C (+ elevation correction)
        assert 27.5 <= df_max["ensemble_mean"].iloc[0] <= 29.5

        # 3. Process Min Temperature
        df_min = processor.process_forecast_to_features(
            dataset=ds,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="min",
            lead_time_bucket=24,
        )

        assert df_min["target_type"].iloc[0] == "min"
        assert df_min["ensemble_mean"].iloc[0] < df_max["ensemble_mean"].iloc[0]

    def test_denver_winter_pipeline_execution(self):
        # Denver: KDEN lat 39.86, lon -104.67 (255.33E), elevation 1655m
        # Grid stored in 0..360 longitude [250..260]
        lats = np.linspace(45.0, 35.0, 41)
        lons_360 = np.linspace(250.0, 260.0, 41)
        members = ["c00", "p01", "p02", "p03", "p04"]
        fxx_steps = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

        # Winter Denver temp: ~270K (-3.15°C)
        tmax_grid = np.zeros((5, 10, 41, 41))
        tmin_grid = np.zeros((5, 10, 41, 41))
        for m_idx in range(5):
            for s_idx, fxx in enumerate(fxx_steps):
                tmax_grid[m_idx, s_idx, :, :] = 270.0 + m_idx * 0.5
                tmin_grid[m_idx, s_idx, :, :] = 260.0 + m_idx * 0.5

        # Model orography at 1500m -> station is at 1655m -> Δh = 155m -> drop = 0.0065*155 = ~1.0°C
        orography = np.full((41, 41), 1500.0)

        ds = xr.Dataset(
            {
                "tmax": (["member", "step", "latitude", "longitude"], tmax_grid, {"units": "K"}),
                "tmin": (["member", "step", "latitude", "longitude"], tmin_grid, {"units": "K"}),
                "orography": (["latitude", "longitude"], orography),
            },
            coords={
                "member": members,
                "step": fxx_steps,
                "latitude": lats,
                "longitude": lons_360,
            },
        )

        processor = DataProcessor()
        init_time = datetime(2019, 1, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 1, 2)

        df_kden = processor.process_forecast_to_features(
            dataset=ds,
            station_id="KDEN",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="max",
            lead_time_bucket=30,
        )

        assert df_kden["station_id"].iloc[0] == "KDEN"
        # 271K mean - 273.15 = -2.15°C - 1.0°C elev drop = ~ -3.15°C
        assert -5.0 <= df_kden["ensemble_mean"].iloc[0] <= -1.0
