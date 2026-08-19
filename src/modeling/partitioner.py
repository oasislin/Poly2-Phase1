#!/usr/bin/env python3
"""
DatasetPartitioner: Lead time bucketing, nominal local time conversion, and seasonal matrix partitioner (Ticket 2.2-03 / Issue #16).

Implements:
    - 4-season grouping: Spring (3-5), Summer (6-8), Autumn (9-11), Winter (12-2)
    - Nominal occurrence hours: Max 15:00 Local Time, Min 06:00 Local Time
    - Station UTC offsets: ZSPD (+8 UTC), KDEN (-7 UTC / Mountain Time)
    - round_to_nearest_6h lead time bucketing
    - Discrete matrix training nodes: Max {54h, 30h, 6h}, Min {48h, 24h}
    - 40 standard training partition buckets (2 stations x 4 seasons x 5 nodes)
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Standard station offsets and nominal extrema hours
STATION_UTC_OFFSETS: Dict[str, int] = {
    "ZSPD": 8,   # Shanghai Pudong (UTC+8)
    "KDEN": -7,  # Denver International (UTC-7 Mountain Standard)
}

NOMINAL_LOCAL_HOURS: Dict[str, int] = {
    "max": 15,  # 15:00 Local Time (diurnal peak temperature)
    "min": 6,   # 06:00 Local Time (diurnal minimum temperature / near sunrise)
}

LEAD_TIME_NODES: Dict[str, List[int]] = {
    "max": [54, 30, 6],
    "min": [48, 24],
}

SEASONS: List[str] = ["Spring", "Summer", "Autumn", "Winter"]


class DatasetPartitioner:
    """Manages seasonal dataset splitting and lead-time discretization into standard training matrices."""

    @staticmethod
    def round_to_nearest_6h(lead_hours: Union[float, np.ndarray, pd.Series]) -> Union[int, np.ndarray]:
        """Round continuous forecast lead hours to the nearest 6-hour discrete multiple."""
        arr = np.asarray(lead_hours, dtype=np.float64)
        rounded = np.round(arr / 6.0) * 6.0
        if np.ndim(rounded) == 0:
            return int(rounded.item())
        return rounded.astype(int)

    @staticmethod
    def get_season(date_or_month: Union[int, date, datetime, str]) -> str:
        """Map a month number, date, or date string to its meteorological season."""
        if isinstance(date_or_month, int):
            month = date_or_month
        elif isinstance(date_or_month, (date, datetime)):
            month = date_or_month.month
        elif isinstance(date_or_month, str):
            dt = pd.to_datetime(date_or_month)
            month = dt.month
        else:
            raise TypeError(f"Unsupported type for season mapping: {type(date_or_month)}")

        if month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        elif month in (9, 10, 11):
            return "Autumn"
        elif month in (12, 1, 2):
            return "Winter"
        else:
            raise ValueError(f"Invalid month integer: {month}")

    @staticmethod
    def compute_nominal_lead_hours(
        station_id: str,
        target_type: str,
        init_datetime: Union[datetime, str, pd.Timestamp],
        target_date: Union[date, str, pd.Timestamp],
    ) -> float:
        """Compute continuous lead hours from forecast init UTC to the nominal diurnal extreme time in UTC.

        Args:
            station_id: 'ZSPD' or 'KDEN'.
            target_type: 'max' or 'min'.
            init_datetime: Model initialization UTC datetime (e.g. 2019-01-01 00:00:00).
            target_date: Calendar date of the target observation.

        Returns:
            Continuous lead time in hours.
        """
        station = station_id.upper()
        if station not in STATION_UTC_OFFSETS:
            raise ValueError(f"Unknown station_id: {station_id}, expected one of {list(STATION_UTC_OFFSETS.keys())}")

        target_t = target_type.lower()
        if target_t not in NOMINAL_LOCAL_HOURS:
            raise ValueError(f"Unknown target_type: {target_type}, expected 'max' or 'min'")

        init_dt = pd.to_datetime(init_datetime)
        t_date = pd.to_datetime(target_date).date()

        # Nominal local time of the temperature extreme
        nominal_local_hour = NOMINAL_LOCAL_HOURS[target_t]
        utc_offset = STATION_UTC_OFFSETS[station]

        # Construct nominal local timestamp and convert to UTC
        local_dt = datetime(t_date.year, t_date.month, t_date.day, nominal_local_hour, 0, 0)
        nominal_utc_dt = local_dt - timedelta(hours=utc_offset)

        lead_delta = nominal_utc_dt - init_dt.to_pydatetime()
        return float(lead_delta.total_seconds() / 3600.0)

    @classmethod
    def get_lead_time_nodes(cls, target_type: str) -> List[int]:
        """Return the list of standard discrete lead time training nodes for target type."""
        t_type = target_type.lower()
        if t_type not in LEAD_TIME_NODES:
            raise ValueError(f"Unknown target_type: {target_type}, expected 'max' or 'min'")
        return list(LEAD_TIME_NODES[t_type])

    @classmethod
    def split_by_season(
        cls,
        df: pd.DataFrame,
        date_col: str = "target_date",
    ) -> Dict[str, pd.DataFrame]:
        """Split a DataFrame into 4 seasonal sub-DataFrames based on the date column."""
        if date_col not in df.columns:
            raise KeyError(f"Date column '{date_col}' not found in DataFrame columns: {df.columns.tolist()}")

        dt_series = pd.to_datetime(df[date_col])
        seasons_series = dt_series.dt.month.map(lambda m: cls.get_season(m))

        return {
            season: df[seasons_series == season].copy()
            for season in SEASONS
        }

    @classmethod
    def get_all_matrix_keys(cls) -> List[Tuple[str, str, str, int]]:
        """Return the exhaustive list of 40 training matrix keys: (station_id, season, target_type, lead_bucket)."""
        keys = []
        for station in sorted(STATION_UTC_OFFSETS.keys()):
            for season in SEASONS:
                for target_type in ["max", "min"]:
                    for lead in LEAD_TIME_NODES[target_type]:
                        keys.append((station, season, target_type, lead))
        return keys
