"""Unit tests for unified StorageManager (Task 1.4 T1.4-04)."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.database import TimeSeriesDatabase
from src.data_processing.parquet_store import ParquetFeatureStore
from src.data_processing.storage_manager import StorageManager


class TestStorageManager:
    """Test integrated StorageManager facade for features, observations, and training datasets."""

    @pytest.fixture
    def setup_manager(self, tmp_path):
        feature_dir = tmp_path / "features"
        db_path = tmp_path / "poly.db"

        parquet_store = ParquetFeatureStore(base_dir=feature_dir)
        db = TimeSeriesDatabase(db_path=db_path)
        db.init_schema()

        manager = StorageManager(
            parquet_store=parquet_store,
            database=db,
        )
        return manager, parquet_store, db

    def test_save_and_load_features_via_manager(self, setup_manager):
        manager, _, _ = setup_manager
        df_features = pd.DataFrame({
            "target_date": ["2019-07-01", "2019-07-02"],
            "station_id": ["ZSPD", "ZSPD"],
            "target_type": ["max", "max"],
            "lead_time_bucket": [30, 30],
            "ensemble_mean": [30.0, 32.0],
            "ensemble_variance": [1.5, 2.0],
            "member_max": [32.0, 34.0],
            "member_min": [28.0, 30.0],
        })

        saved = manager.save_forecast_features(df_features)
        assert saved == 2

        loaded = manager.load_features(station_id="ZSPD", start_date="2019-07-01")
        assert len(loaded) == 2
        assert list(loaded["target_date"]) == ["2019-07-01", "2019-07-02"]

    def test_load_aligned_training_dataset(self, setup_manager):
        manager, _, _ = setup_manager

        # 1. Save Features
        df_features = pd.DataFrame({
            "target_date": ["2019-07-01", "2019-07-02", "2019-07-03"],
            "station_id": ["ZSPD", "ZSPD", "ZSPD"],
            "target_type": ["max", "max", "max"],
            "lead_time_bucket": [30, 30, 30],
            "ensemble_mean": [30.0, 32.0, 31.0],
            "ensemble_variance": [1.5, 2.0, 1.8],
            "member_max": [32.0, 34.0, 33.0],
            "member_min": [28.0, 30.0, 29.0],
        })
        manager.save_forecast_features(df_features)

        # 2. Save Observations (2019-07-01 and 2019-07-02 only)
        df_obs = pd.DataFrame({
            "date": ["2019-07-01", "2019-07-02"],
            "station_id": ["ZSPD", "ZSPD"],
            "temp_max": [31.5, 33.0],
            "temp_min": [23.0, 24.5],
        })
        manager.save_observations(df_obs)

        # 3. Load merged training dataset for Max Temperature
        train_df = manager.load_training_dataset(
            station_id="ZSPD",
            target_type="max",
            lead_time_bucket=30,
            start_date="2019-07-01",
            end_date="2019-07-03",
        )

        # Only 2 rows should match (2019-07-01 and 2019-07-02) because 2019-07-03 has no observation yet
        assert len(train_df) == 2
        assert "observed_temp" in train_df.columns
        assert list(train_df["observed_temp"]) == [31.5, 33.0]
        assert list(train_df["ensemble_mean"]) == [30.0, 32.0]

    def test_storage_health_check(self, setup_manager):
        manager, _, _ = setup_manager
        health = manager.verify_storage_health()
        assert health["status"] == "HEALTHY"
        assert "tables" in health
        assert "observations" in health["tables"]
        assert "inventory_partitions" in health
