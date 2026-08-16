#!/usr/bin/env python3
"""
Unified StorageManager facade orchestrating feature Parquet store and SQLite database (Task 1.4 T1.4-04).

Provides seamless access for Phase 1B EMOS training, backtesting, and production inference:
- Features persistence and querying (Parquet)
- Historical observations alignment (SQLite)
- Predictions and validation metrics logging (SQLite)
- Storage health checks and inventory tracking
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data_processing.data_validator import DataValidator
from src.data_processing.database import TimeSeriesDatabase
from src.data_processing.parquet_store import ParquetFeatureStore

logger = logging.getLogger(__name__)


class StorageManager:
    """Central data storage gateway integrating Parquet feature store and SQLite database."""

    def __init__(
        self,
        parquet_store: Optional[ParquetFeatureStore] = None,
        database: Optional[TimeSeriesDatabase] = None,
        validator: Optional[DataValidator] = None,
    ):
        self.parquet_store = parquet_store or ParquetFeatureStore()
        self.db = database or TimeSeriesDatabase()
        self.validator = validator or DataValidator(strict=True)

    # 1. Feature Store Operations
    def save_forecast_features(self, df: pd.DataFrame, deduplicate: bool = True) -> int:
        """Validate and persist forecast features to Parquet."""
        return self.parquet_store.save_features(df, deduplicate=deduplicate)

    def load_features(
        self,
        station_id: str,
        years: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_type: Optional[str] = None,
        lead_time_bucket: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query forecast features from Parquet storage."""
        return self.parquet_store.load_features(
            station_id=station_id,
            years=years,
            start_date=start_date,
            end_date=end_date,
            target_type=target_type,
            lead_time_bucket=lead_time_bucket,
        )

    # 2. Observation Operations
    def save_observations(self, df: pd.DataFrame) -> int:
        """Validate and persist weather station observations to SQLite."""
        self.validator.validate_observations(df)
        return self.db.save_observations(df)

    def get_observations(
        self,
        station_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query station observations from SQLite."""
        return self.db.get_observations(station_id, start_date=start_date, end_date=end_date)

    # 3. Aligned Training Dataset Loader
    def load_training_dataset(
        self,
        station_id: str,
        target_type: str,
        lead_time_bucket: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and merge forecast features with verified observations for EMOS training.

        Returns an inner-joined DataFrame containing feature columns and the true `observed_temp`.
        """
        t_type = target_type.strip().lower()
        obs_col = "temp_max" if t_type == "max" else "temp_min"

        # Load X features
        feat_df = self.load_features(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            target_type=t_type,
            lead_time_bucket=lead_time_bucket,
        )
        if feat_df.empty:
            return pd.DataFrame()

        # Load y observations
        obs_df = self.get_observations(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
        )
        if obs_df.empty:
            return pd.DataFrame()

        obs_subset = obs_df[["date", obs_col]].rename(columns={obs_col: "observed_temp"})

        # Inner join on target_date == date
        merged = pd.merge(
            feat_df,
            obs_subset,
            left_on="target_date",
            right_on="date",
            how="inner",
        )
        if "date" in merged.columns and "target_date" in merged.columns:
            merged = merged.drop(columns=["date"])

        return merged.sort_values("target_date").reset_index(drop=True)

    # 4. Predictions and Validation Metrics
    def save_predictions(self, df: pd.DataFrame) -> int:
        """Save model prediction distributions to SQLite."""
        return self.db.save_predictions(df)

    def get_predictions(
        self,
        station_id: str,
        target_date: Optional[str] = None,
        target_type: Optional[str] = None,
        lead_time_bucket: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query predictions from SQLite."""
        return self.db.get_predictions(
            station_id=station_id,
            target_date=target_date,
            target_type=target_type,
            lead_time_bucket=lead_time_bucket,
        )

    def save_validation_metrics(self, df: pd.DataFrame) -> int:
        """Save model backtest evaluation scores to SQLite."""
        return self.db.save_validation_metrics(df)

    def get_validation_metrics(
        self,
        station_id: str,
        season: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query validation metrics from SQLite."""
        return self.db.get_validation_metrics(
            station_id=station_id,
            season=season,
            target_type=target_type,
        )

    # 5. Health and Inventory
    def verify_storage_health(self) -> Dict[str, Any]:
        """Perform system health check on data directories and databases."""
        tables = self.db.list_tables()
        inv_df = self.parquet_store.list_inventory()
        return {
            "status": "HEALTHY",
            "tables": tables,
            "feature_dir": str(self.parquet_store.base_dir),
            "db_path": str(self.db.db_path),
            "inventory_partitions": len(inv_df),
            "total_cached_records": int(inv_df["record_count"].sum()) if not inv_df.empty else 0,
        }
