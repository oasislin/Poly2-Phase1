"""Unit tests for ParquetFeatureStore (Task 1.4 T1.4-02)."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.data_validator import ValidationError
from src.data_processing.parquet_store import ParquetFeatureStore


class TestParquetFeatureStore:
    """Test Parquet file storage, partitioning, atomic writes, and querying."""

    @pytest.fixture
    def temp_store_dir(self, tmp_path):
        store_dir = tmp_path / "features"
        store_dir.mkdir(parents=True, exist_ok=True)
        return store_dir

    @pytest.fixture
    def sample_features_df(self):
        return pd.DataFrame({
            "target_date": ["2019-01-01", "2019-01-02", "2019-07-01", "2019-07-02"],
            "station_id": ["ZSPD", "ZSPD", "ZSPD", "ZSPD"],
            "target_type": ["max", "max", "max", "max"],
            "lead_time_bucket": [30, 30, 30, 30],
            "ensemble_mean": [8.5, 9.0, 32.0, 33.5],
            "ensemble_variance": [1.2, 1.0, 2.5, 2.8],
            "member_max": [10.0, 10.5, 34.0, 35.0],
            "member_min": [7.0, 7.5, 29.5, 31.0],
        })

    def test_save_and_load_features(self, temp_store_dir, sample_features_df):
        store = ParquetFeatureStore(base_dir=temp_store_dir)
        count = store.save_features(sample_features_df)
        assert count == 4

        # Verify Parquet file was created under station partition
        expected_file = temp_store_dir / "ZSPD" / "2019.parquet"
        assert expected_file.exists()

        # Load all ZSPD features
        loaded_df = store.load_features(station_id="ZSPD")
        assert len(loaded_df) == 4
        assert "updated_at" in loaded_df.columns
        assert list(loaded_df["target_date"]) == ["2019-01-01", "2019-01-02", "2019-07-01", "2019-07-02"]

    def test_filter_queries(self, temp_store_dir, sample_features_df):
        store = ParquetFeatureStore(base_dir=temp_store_dir)
        store.save_features(sample_features_df)

        # Filter by date range
        res_date = store.load_features(
            station_id="ZSPD",
            start_date="2019-07-01",
            end_date="2019-07-31",
        )
        assert len(res_date) == 2
        assert set(res_date["target_date"]) == {"2019-07-01", "2019-07-02"}

        # Filter by target_type and lead_time_bucket
        res_type = store.load_features(
            station_id="ZSPD",
            target_type="max",
            lead_time_bucket=30,
        )
        assert len(res_type) == 4

    def test_deduplication_on_upsert(self, temp_store_dir, sample_features_df):
        store = ParquetFeatureStore(base_dir=temp_store_dir)
        store.save_features(sample_features_df)

        # Updated record with higher ensemble_mean for 2019-01-01
        updated_row = pd.DataFrame({
            "target_date": ["2019-01-01"],
            "station_id": ["ZSPD"],
            "target_type": ["max"],
            "lead_time_bucket": [30],
            "ensemble_mean": [11.5],  # Changed from 8.5
            "ensemble_variance": [1.5],
            "member_max": [13.0],
            "member_min": [9.0],
        })

        store.save_features(updated_row, deduplicate=True)

        loaded_df = store.load_features(station_id="ZSPD")
        # Total count should still be 4
        assert len(loaded_df) == 4
        # Updated record should reflect new value
        record = loaded_df[loaded_df["target_date"] == "2019-01-01"].iloc[0]
        assert record["ensemble_mean"] == pytest.approx(11.5)

    def test_multi_station_partitioning(self, temp_store_dir):
        store = ParquetFeatureStore(base_dir=temp_store_dir)
        df_mixed = pd.DataFrame({
            "target_date": ["2019-01-01", "2019-01-01"],
            "station_id": ["ZSPD", "KDEN"],
            "target_type": ["max", "max"],
            "lead_time_bucket": [30, 30],
            "ensemble_mean": [8.5, -2.0],
            "ensemble_variance": [1.2, 2.0],
            "member_max": [10.0, 0.0],
            "member_min": [7.0, -5.0],
        })

        store.save_features(df_mixed)

        assert (temp_store_dir / "ZSPD" / "2019.parquet").exists()
        assert (temp_store_dir / "KDEN" / "2019.parquet").exists()

        zspd_df = store.load_features(station_id="ZSPD")
        kden_df = store.load_features(station_id="KDEN")
        assert len(zspd_df) == 1
        assert len(kden_df) == 1

    def test_invalid_features_rejected(self, temp_store_dir):
        store = ParquetFeatureStore(base_dir=temp_store_dir)
        invalid_df = pd.DataFrame({
            "target_date": ["2019-01-01"],
            "station_id": ["ZSPD"],
            "target_type": ["max"],
            "lead_time_bucket": [30],
            "ensemble_mean": [120.0],  # 120°C unphysical
            "ensemble_variance": [1.0],
            "member_max": [130.0],
            "member_min": [110.0],
        })

        with pytest.raises(ValidationError):
            store.save_features(invalid_df)

    def test_list_available_inventory(self, temp_store_dir, sample_features_df):
        store = ParquetFeatureStore(base_dir=temp_store_dir)
        store.save_features(sample_features_df)

        inv = store.list_inventory()
        assert isinstance(inv, pd.DataFrame)
        assert len(inv) == 1
        assert inv["station_id"].iloc[0] == "ZSPD"
        assert inv["year"].iloc[0] == 2019
        assert inv["record_count"].iloc[0] == 4
