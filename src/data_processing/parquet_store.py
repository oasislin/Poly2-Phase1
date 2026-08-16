#!/usr/bin/env python3
"""
High-performance Parquet Feature Store for processed GEFS station features (Task 1.4 T1.4-02).

Partitioning structure:
  {base_dir}/{station_id}/{year}.parquet

Features:
- Validated on write via DataValidator (strict mode)
- Timestamped versioning (updated_at)
- Atomic safe writes (temporary file -> atomic rename)
- Upsert & deduplication on composite key (target_date, station_id, target_type, lead_time_bucket)
- Multi-dimensional query filtering (station, years, date range, target_type, lead_time_bucket)
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data_processing.data_validator import DataValidator

DEFAULT_FEATURE_DIR = Path("data/processed/features")
PRIMARY_KEY_COLS = ["target_date", "station_id", "target_type", "lead_time_bucket"]


class ParquetFeatureStore:
    """Service for managing partitioned Parquet files containing calibrated weather features."""

    def __init__(
        self,
        base_dir: Union[str, Path] = DEFAULT_FEATURE_DIR,
        validator: Optional[DataValidator] = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.validator = validator or DataValidator(strict=True)

    def _get_partition_path(self, station_id: str, year: int) -> Path:
        """Get path to the specific station/year Parquet partition file."""
        station_dir = self.base_dir / station_id
        station_dir.mkdir(parents=True, exist_ok=True)
        return station_dir / f"{year}.parquet"

    def _write_atomic_parquet(self, df: pd.DataFrame, target_path: Path, deduplicate: bool) -> None:
        """Atomically merge and write a partition DataFrame to a Parquet file."""
        merged_df = df
        if target_path.exists():
            existing_df = pd.read_parquet(target_path)
            merged_df = pd.concat([existing_df, df], ignore_index=True)

        if deduplicate:
            # Sort by updated_at so newest records are preserved
            if "updated_at" in merged_df.columns:
                merged_df = merged_df.sort_values("updated_at")
            merged_df = merged_df.drop_duplicates(subset=PRIMARY_KEY_COLS, keep="last")

        merged_df = merged_df.sort_values(PRIMARY_KEY_COLS).reset_index(drop=True)

        tmp_path = target_path.with_suffix(".tmp.parquet")
        merged_df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        tmp_path.replace(target_path)

    def save_features(self, df: pd.DataFrame, deduplicate: bool = True) -> int:
        """Validate and save a DataFrame of features into partitioned Parquet files.

        Parameters
        ----------
        df : pd.DataFrame
            Features DataFrame conforming to REQUIRED_FEATURE_COLUMNS.
        deduplicate : bool, default True
            Whether to upsert/deduplicate on composite primary key.

        Returns
        -------
        int
            Number of rows in the input DataFrame that were successfully processed.
        """
        if df.empty:
            return 0

        # 1. Strict validation before persisting
        self.validator.validate_features(df)

        df_to_save = df.copy()
        # 2. Attach updated_at timestamp
        now_iso = datetime.now(timezone.utc).isoformat()
        df_to_save["updated_at"] = now_iso

        # 3. Derive year for partitioning
        df_to_save["year"] = pd.to_datetime(df_to_save["target_date"]).dt.year

        # 4. Group by (station_id, year) and write partitions
        total_rows = 0
        for (station_id, year), group_df in df_to_save.groupby(["station_id", "year"]):
            part_path = self._get_partition_path(str(station_id), int(year))
            # Drop temporary year column before writing
            clean_group = group_df.drop(columns=["year"])
            self._write_atomic_parquet(clean_group, part_path, deduplicate=deduplicate)
            total_rows += len(group_df)

        return total_rows

    def _filter_dataframe(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_type: Optional[str] = None,
        lead_time_bucket: Optional[int] = None,
    ) -> pd.DataFrame:
        """Apply query filter constraints to loaded DataFrame."""
        filtered = df
        if start_date is not None:
            filtered = filtered[filtered["target_date"] >= start_date]
        if end_date is not None:
            filtered = filtered[filtered["target_date"] <= end_date]
        if target_type is not None:
            filtered = filtered[filtered["target_type"] == target_type.strip().lower()]
        if lead_time_bucket is not None:
            filtered = filtered[filtered["lead_time_bucket"] == int(lead_time_bucket)]
        return filtered

    def load_features(
        self,
        station_id: str,
        years: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_type: Optional[str] = None,
        lead_time_bucket: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load features matching query criteria for a given station."""
        station_dir = self.base_dir / station_id
        if not station_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(station_dir.glob("*.parquet"))
        if years is not None:
            year_strs = {str(y) for y in years}
            parquet_files = [f for f in parquet_files if f.stem in year_strs]

        dfs = [pd.read_parquet(f) for f in parquet_files]
        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)
        return self._filter_dataframe(
            full_df,
            start_date=start_date,
            end_date=end_date,
            target_type=target_type,
            lead_time_bucket=lead_time_bucket,
        )

    def list_inventory(self) -> pd.DataFrame:
        """List summary inventory of all stored station partitions."""
        records = []
        for station_dir in self.base_dir.iterdir():
            if station_dir.is_dir():
                for p_file in station_dir.glob("*.parquet"):
                    if not p_file.name.endswith(".tmp.parquet"):
                        df = pd.read_parquet(p_file)
                        records.append({
                            "station_id": station_dir.name,
                            "year": int(p_file.stem),
                            "record_count": len(df),
                            "file_size_bytes": p_file.stat().st_size,
                            "latest_updated_at": df["updated_at"].max() if "updated_at" in df else None,
                        })

        return pd.DataFrame(records) if records else pd.DataFrame()
