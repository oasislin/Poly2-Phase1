#!/usr/bin/env python3
"""
SQLite Time-Series Database engine for observations, predictions, and validation metrics (Task 1.4 T1.4-03).

Schema:
- observations: Historical Wunderground station records with (station_id, date) index
- predictions: Gaussian EMOS probability distribution parameters (μ, σ²) and degradation flags
- validation_metrics: Backtesting and calibration performance scores (CRPS, PIT, extreme tails)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

DEFAULT_DB_PATH = Path("data/db/predictions.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    date TEXT NOT NULL,
    temp_max REAL NOT NULL,
    temp_min REAL NOT NULL,
    humidity REAL,
    wind_speed REAL,
    pressure REAL,
    precipitation REAL,
    created_at TEXT NOT NULL,
    UNIQUE(station_id, date)
);

CREATE INDEX IF NOT EXISTS idx_obs_station_date ON observations(station_id, date);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    target_date TEXT NOT NULL,
    target_type TEXT NOT NULL,
    lead_time_bucket INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    mean REAL NOT NULL,
    variance REAL NOT NULL,
    floor_active INTEGER DEFAULT 0,
    degraded_to_climatology INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(station_id, target_date, target_type, lead_time_bucket, model_version)
);

CREATE INDEX IF NOT EXISTS idx_pred_station_target ON predictions(station_id, target_date, target_type);

CREATE TABLE IF NOT EXISTS validation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    season TEXT NOT NULL,
    target_type TEXT NOT NULL,
    lead_time_bucket INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    crps_mean REAL NOT NULL,
    pit_ks_pvalue REAL,
    sample_count INTEGER NOT NULL,
    extreme_warm_crps REAL,
    extreme_cold_crps REAL,
    evaluated_at TEXT NOT NULL,
    UNIQUE(station_id, season, target_type, lead_time_bucket, model_version)
);

CREATE INDEX IF NOT EXISTS idx_val_station_season ON validation_metrics(station_id, season, target_type);
"""


class TimeSeriesDatabase:
    """SQLite time-series persistence manager."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Initialize database tables and indexes if they do not already exist."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def list_tables(self) -> List[str]:
        """List all user tables in the SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            return [row[0] for row in cursor.fetchall()]

    def save_observations(self, df: pd.DataFrame) -> int:
        """Batch upsert station observation records."""
        if df.empty:
            return 0
        self.init_schema()
        now_iso = datetime.now(timezone.utc).isoformat()

        sql = """
        INSERT INTO observations (station_id, date, temp_max, temp_min, humidity, wind_speed, pressure, precipitation, created_at)
        VALUES (:station_id, :date, :temp_max, :temp_min, :humidity, :wind_speed, :pressure, :precipitation, :created_at)
        ON CONFLICT(station_id, date) DO UPDATE SET
            temp_max=excluded.temp_max,
            temp_min=excluded.temp_min,
            humidity=excluded.humidity,
            wind_speed=excluded.wind_speed,
            pressure=excluded.pressure,
            precipitation=excluded.precipitation,
            created_at=excluded.created_at;
        """
        records = []
        for _, row in df.iterrows():
            records.append({
                "station_id": str(row["station_id"]),
                "date": str(row["date"]),
                "temp_max": float(row["temp_max"]),
                "temp_min": float(row["temp_min"]),
                "humidity": float(row["humidity"]) if "humidity" in row and pd.notna(row["humidity"]) else None,
                "wind_speed": float(row["wind_speed"]) if "wind_speed" in row and pd.notna(row["wind_speed"]) else None,
                "pressure": float(row["pressure"]) if "pressure" in row and pd.notna(row["pressure"]) else None,
                "precipitation": float(row["precipitation"]) if "precipitation" in row and pd.notna(row["precipitation"]) else None,
                "created_at": now_iso,
            })

        with self._get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return len(records)

    def get_observations(
        self,
        station_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query station observations over a date range."""
        query = "SELECT * FROM observations WHERE station_id = ?"
        params: List[Any] = [station_id]
        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date ASC"

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def save_predictions(self, df: pd.DataFrame) -> int:
        """Batch upsert model prediction distribution records."""
        if df.empty:
            return 0
        self.init_schema()
        now_iso = datetime.now(timezone.utc).isoformat()

        sql = """
        INSERT INTO predictions (station_id, target_date, target_type, lead_time_bucket, model_version, mean, variance, floor_active, degraded_to_climatology, created_at)
        VALUES (:station_id, :target_date, :target_type, :lead_time_bucket, :model_version, :mean, :variance, :floor_active, :degraded_to_climatology, :created_at)
        ON CONFLICT(station_id, target_date, target_type, lead_time_bucket, model_version) DO UPDATE SET
            mean=excluded.mean,
            variance=excluded.variance,
            floor_active=excluded.floor_active,
            degraded_to_climatology=excluded.degraded_to_climatology,
            created_at=excluded.created_at;
        """
        records = []
        for _, row in df.iterrows():
            records.append({
                "station_id": str(row["station_id"]),
                "target_date": str(row["target_date"]),
                "target_type": str(row["target_type"]),
                "lead_time_bucket": int(row["lead_time_bucket"]),
                "model_version": str(row["model_version"]),
                "mean": float(row["mean"]),
                "variance": float(row["variance"]),
                "floor_active": int(row.get("floor_active", 0)),
                "degraded_to_climatology": int(row.get("degraded_to_climatology", 0)),
                "created_at": now_iso,
            })

        with self._get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return len(records)

    def get_predictions(
        self,
        station_id: str,
        target_date: Optional[str] = None,
        target_type: Optional[str] = None,
        lead_time_bucket: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query predictions matching station and target filters."""
        query = "SELECT * FROM predictions WHERE station_id = ?"
        params: List[Any] = [station_id]
        if target_date is not None:
            query += " AND target_date = ?"
            params.append(target_date)
        if target_type is not None:
            query += " AND target_type = ?"
            params.append(target_type)
        if lead_time_bucket is not None:
            query += " AND lead_time_bucket = ?"
            params.append(lead_time_bucket)
        query += " ORDER BY target_date, lead_time_bucket ASC"

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def save_validation_metrics(self, df: pd.DataFrame) -> int:
        """Batch upsert model validation metrics."""
        if df.empty:
            return 0
        self.init_schema()
        now_iso = datetime.now(timezone.utc).isoformat()

        sql = """
        INSERT INTO validation_metrics (station_id, season, target_type, lead_time_bucket, model_version, crps_mean, pit_ks_pvalue, sample_count, extreme_warm_crps, extreme_cold_crps, evaluated_at)
        VALUES (:station_id, :season, :target_type, :lead_time_bucket, :model_version, :crps_mean, :pit_ks_pvalue, :sample_count, :extreme_warm_crps, :extreme_cold_crps, :evaluated_at)
        ON CONFLICT(station_id, season, target_type, lead_time_bucket, model_version) DO UPDATE SET
            crps_mean=excluded.crps_mean,
            pit_ks_pvalue=excluded.pit_ks_pvalue,
            sample_count=excluded.sample_count,
            extreme_warm_crps=excluded.extreme_warm_crps,
            extreme_cold_crps=excluded.extreme_cold_crps,
            evaluated_at=excluded.evaluated_at;
        """
        records = []
        for _, row in df.iterrows():
            records.append({
                "station_id": str(row["station_id"]),
                "season": str(row["season"]),
                "target_type": str(row["target_type"]),
                "lead_time_bucket": int(row["lead_time_bucket"]),
                "model_version": str(row["model_version"]),
                "crps_mean": float(row["crps_mean"]),
                "pit_ks_pvalue": float(row["pit_ks_pvalue"]) if "pit_ks_pvalue" in row and pd.notna(row["pit_ks_pvalue"]) else None,
                "sample_count": int(row["sample_count"]),
                "extreme_warm_crps": float(row["extreme_warm_crps"]) if "extreme_warm_crps" in row and pd.notna(row["extreme_warm_crps"]) else None,
                "extreme_cold_crps": float(row["extreme_cold_crps"]) if "extreme_cold_crps" in row and pd.notna(row["extreme_cold_crps"]) else None,
                "evaluated_at": now_iso,
            })

        with self._get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return len(records)

    def get_validation_metrics(
        self,
        station_id: str,
        season: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query validation metrics for a station."""
        query = "SELECT * FROM validation_metrics WHERE station_id = ?"
        params: List[Any] = [station_id]
        if season is not None:
            query += " AND season = ?"
            params.append(season)
        if target_type is not None:
            query += " AND target_type = ?"
            params.append(target_type)

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
