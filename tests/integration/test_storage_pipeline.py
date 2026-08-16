"""Integration test for end-to-end Storage Pipeline (Task 1.4 T1.4-04).

Simulates the complete lifecycle:
1. Process simulated GEFS grid to station features via DataProcessor
2. Persist features to partitioned Parquet files via StorageManager
3. Save ground-truth weather station observations to SQLite
4. Load aligned (X, y) training dataset
5. Save and query mock Gaussian EMOS predictions and backtest validation metrics
"""

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_processing.data_processor import DataProcessor
from src.data_processing.database import TimeSeriesDatabase
from src.data_processing.parquet_store import ParquetFeatureStore
from src.data_processing.storage_manager import StorageManager


class TestEndToEndStoragePipeline:
    """Complete integration suite verifying data transformation, persistence, and retrieval."""

    @pytest.fixture
    def environment(self, tmp_path):
        feature_dir = tmp_path / "features"
        db_path = tmp_path / "poly_test.db"

        parquet_store = ParquetFeatureStore(base_dir=feature_dir)
        database = TimeSeriesDatabase(db_path=db_path)
        database.init_schema()

        manager = StorageManager(
            parquet_store=parquet_store,
            database=database,
        )
        processor = DataProcessor()

        return {
            "manager": manager,
            "processor": processor,
            "feature_dir": feature_dir,
            "db_path": db_path,
        }

    def test_full_pipeline_flow(self, environment):
        manager = environment["manager"]
        processor = environment["processor"]

        # Step 1: Generate simulated GEFS forecast dataset for Shanghai
        lats = np.linspace(35.0, 25.0, 41)
        lons = np.linspace(115.0, 125.0, 41)
        members = ["c00", "p01", "p02", "p03", "p04"]
        fxx_steps = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

        tmax_grid = np.zeros((5, 10, 41, 41))
        tmin_grid = np.zeros((5, 10, 41, 41))
        for m_idx in range(5):
            for s_idx, fxx in enumerate(fxx_steps):
                tmax_grid[m_idx, s_idx, :, :] = 301.0 + m_idx * 0.5
                tmin_grid[m_idx, s_idx, :, :] = 293.0 + m_idx * 0.5

        ds = xr.Dataset(
            {
                "tmax": (["member", "step", "latitude", "longitude"], tmax_grid, {"units": "K"}),
                "tmin": (["member", "step", "latitude", "longitude"], tmin_grid, {"units": "K"}),
            },
            coords={"member": members, "step": fxx_steps, "latitude": lats, "longitude": lons},
        )

        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        # Step 2: Extract features via DataProcessor
        df_feat = processor.process_forecast_to_features(
            dataset=ds,
            station_id="ZSPD",
            target_date=target_date,
            init_time_utc=init_time,
            target_type="max",
            lead_time_bucket=30,
        )
        assert len(df_feat) == 1

        # Step 3: Save features to Parquet
        saved_feat_count = manager.save_forecast_features(df_feat)
        assert saved_feat_count == 1
        assert (environment["feature_dir"] / "ZSPD" / "2019.parquet").exists()

        # Step 4: Save true observation to SQLite
        df_obs = pd.DataFrame({
            "station_id": ["ZSPD"],
            "date": ["2019-07-02"],
            "temp_max": [29.2],
            "temp_min": [20.5],
            "humidity": [78.0],
            "wind_speed": [5.0],
            "pressure": [1007.0],
            "precipitation": [0.0],
        })
        manager.save_observations(df_obs)

        # Step 5: Load aligned training dataset
        train_df = manager.load_training_dataset(
            station_id="ZSPD",
            target_type="max",
            lead_time_bucket=30,
            start_date="2019-07-01",
            end_date="2019-07-03",
        )
        assert len(train_df) == 1
        assert train_df["target_date"].iloc[0] == "2019-07-02"
        assert train_df["observed_temp"].iloc[0] == pytest.approx(29.2)
        assert train_df["ensemble_mean"].iloc[0] == pytest.approx(df_feat["ensemble_mean"].iloc[0])

        # Step 6: Log prediction distribution
        pred_df = pd.DataFrame({
            "station_id": ["ZSPD"],
            "target_date": ["2019-07-02"],
            "target_type": ["max"],
            "lead_time_bucket": [30],
            "model_version": ["v5.9.1_gaussian_emos"],
            "mean": [28.8],
            "variance": [1.4],
            "floor_active": [0],
            "degraded_to_climatology": [0],
        })
        manager.save_predictions(pred_df)

        pred_res = manager.get_predictions(station_id="ZSPD", target_date="2019-07-02")
        assert len(pred_res) == 1
        assert pred_res["mean"].iloc[0] == pytest.approx(28.8)

        # Step 7: Verify health
        health = manager.verify_storage_health()
        assert health["status"] == "HEALTHY"
        assert health["inventory_partitions"] == 1
