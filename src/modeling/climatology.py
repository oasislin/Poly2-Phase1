#!/usr/bin/env python3
"""
ClimatologyCalculator: Historical baseline and variance floor calculator (Ticket 2.1-01 / Issue #11).

Calculates daily climatological mean (μ_clim) and variance (σ_clim²) using a 31-day sliding
window over historical observations (2000-2018). Strictly Out-Of-Sample (OOS): does not touch 2019+ data.
"""

from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data_processing.database import TimeSeriesDatabase

logger = logging.getLogger(__name__)


class ClimatologyCalculator:
    """Computes and queries historical daily climatological temperature distributions."""

    def __init__(
        self,
        train_start_year: int = 2000,
        train_end_year: int = 2018,
        window_days: int = 31,
        database: Optional[TimeSeriesDatabase] = None,
    ):
        if train_start_year > train_end_year:
            raise ValueError(
                f"train_start_year ({train_start_year}) cannot be greater than train_end_year ({train_end_year})"
            )
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.window_days = window_days
        self.half_window = window_days // 2
        self.db = database or TimeSeriesDatabase()

        # Cache of raw filtered observation data per station: {station_id: pd.DataFrame}
        self._training_data: Dict[str, pd.DataFrame] = {}

        # Lookup tables: {(station_id, target_type): pd.DataFrame(index=1..366)}
        self._climatology_tables: Dict[Tuple[str, str], pd.DataFrame] = {}

    def fit(
        self,
        observations_df: pd.DataFrame,
        station_id: str,
        target_types: Optional[List[str]] = None,
    ) -> "ClimatologyCalculator":
        """Fit climatology tables from a provided observations DataFrame.

        Filters strictly to [train_start_year, train_end_year] to guarantee OOS integrity.
        """
        if observations_df.empty:
            raise ValueError("Provided observations DataFrame is empty")

        df = observations_df.copy()
        if "temp_high" in df.columns and "temp_max" not in df.columns:
            df = df.rename(columns={"temp_high": "temp_max", "temp_low": "temp_min"})

        df["dt"] = pd.to_datetime(df["date"])
        df["year"] = df["dt"].dt.year

        # Enforce strict OOS time window
        mask = (df["year"] >= self.train_start_year) & (df["year"] <= self.train_end_year)
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            raise ValueError(
                f"No observation records found for station {station_id} in range {self.train_start_year}-{self.train_end_year}"
            )

        self._training_data[station_id] = filtered_df

        t_types = target_types or ["max", "min"]
        for t_type in t_types:
            self._climatology_tables[(station_id, t_type)] = self.compute_climatology(
                filtered_df, station_id=station_id, target_type=t_type
            )

        logger.info(
            "Fitted climatology for %s over %d-%d (%d records)",
            station_id,
            self.train_start_year,
            self.train_end_year,
            len(filtered_df),
        )
        return self

    def fit_from_db(
        self,
        station_ids: Union[str, List[str]],
        target_types: Optional[List[str]] = None,
    ) -> "ClimatologyCalculator":
        """Load observations from SQLite database and fit climatology tables."""
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        start_date = f"{self.train_start_year}-01-01"
        end_date = f"{self.train_end_year}-12-31"

        for st_id in station_ids:
            obs_df = self.db.get_observations(st_id, start_date=start_date, end_date=end_date)
            if obs_df.empty:
                raise ValueError(
                    f"No observation records found in database for station {st_id} between {start_date} and {end_date}"
                )
            self.fit(obs_df, station_id=st_id, target_types=target_types)

        return self

    def compute_climatology(
        self,
        observations_df: pd.DataFrame,
        station_id: str,
        target_type: str,
    ) -> pd.DataFrame:
        """Compute the 1-366 daily climatological table for a given station and target type."""
        col_name = "temp_max" if target_type.lower() == "max" else "temp_min"
        if col_name not in observations_df.columns:
            if "temp_high" in observations_df.columns and col_name == "temp_max":
                col_name = "temp_high"
            elif "temp_low" in observations_df.columns and col_name == "temp_min":
                col_name = "temp_low"
            else:
                raise KeyError(f"Target column '{col_name}' not found in observations DataFrame")

        df = observations_df.copy()
        df["dt"] = pd.to_datetime(df["date"])
        df["year"] = df["dt"].dt.year
        df["month"] = df["dt"].dt.month
        df["day"] = df["dt"].dt.day

        # Filter strictly to training years
        mask = (df["year"] >= self.train_start_year) & (df["year"] <= self.train_end_year)
        filtered_df = df[mask].dropna(subset=[col_name]).copy()

        # Reference leap year 2020 to map 1..366 days seamlessly
        ref_dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        records = []

        # Precompute day-of-year array for filtered_df
        # We define circular window distance around target (month, day)
        # Using reference year 2020 day-of-year representation:
        # Convert each observation to its 2020 equivalent day-of-year
        df_month = filtered_df["month"].values
        df_day = filtered_df["day"].values
        df_temps = filtered_df[col_name].values.astype(np.float64)

        # Vectorized mapping to 2020 day-of-year (1..366)
        # Month lengths in leap year: [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cum_days = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
        df_doy_366 = cum_days[df_month - 1] + df_day

        for doy in range(1, 367):
            # Compute shortest circular distance on 366-day circle
            diff = np.abs(df_doy_366 - doy)
            circ_dist = np.minimum(diff, 366 - diff)

            window_mask = circ_dist <= self.half_window
            window_values = df_temps[window_mask]

            n_samples = len(window_values)
            if n_samples < 2:
                mu = float(np.mean(window_values)) if n_samples == 1 else 0.0
                sigma = 1.0
                var = 1.0
            else:
                mu = float(np.mean(window_values))
                sigma = float(np.std(window_values, ddof=1))
                var = float(sigma ** 2)

            records.append({
                "day_of_year": doy,
                "mu_clim": mu,
                "sigma_clim": sigma,
                "variance_clim": var,
                "sample_count": n_samples,
            })

        table_df = pd.DataFrame(records)
        return table_df

    def get_climatology(
        self,
        station_id: str,
        target_type: str,
        target: Union[str, date, datetime, int],
    ) -> Tuple[float, float]:
        """Query (mu_clim, sigma_clim) for a specific station, target type, and target date or DOY.

        Returns:
            Tuple[float, float]: (mu_clim, sigma_clim)
        """
        doy = self._resolve_day_of_year(target)
        key = (station_id, target_type.lower())

        if key not in self._climatology_tables:
            # If not fitted yet, try fitting from database
            self.fit_from_db(station_ids=[station_id], target_types=[target_type.lower()])

        table = self._climatology_tables[key]
        row = table[table["day_of_year"] == doy]
        if row.empty:
            raise KeyError(f"Day of year {doy} not found in climatology table")

        mu = float(row["mu_clim"].values[0])
        sigma = float(row["sigma_clim"].values[0])
        return mu, sigma

    def get_climatology_variance(
        self,
        station_id: str,
        target_type: str,
        target: Union[str, date, datetime, int],
    ) -> float:
        """Query variance floor σ_clim²(d) for a given date or DOY."""
        _, sigma = self.get_climatology(station_id, target_type, target)
        return sigma ** 2

    def get_training_data(self, station_id: str) -> pd.DataFrame:
        """Retrieve the filtered training observations for a station (OOS audit)."""
        if station_id not in self._training_data:
            self.fit_from_db(station_ids=[station_id])
        return self._training_data[station_id]

    @staticmethod
    def _resolve_day_of_year(target: Union[str, date, datetime, int]) -> int:
        """Convert string, date, datetime, or int into a 1..366 reference day of year."""
        if isinstance(target, int):
            if 1 <= target <= 366:
                return target
            raise ValueError(f"Day of year integer must be in 1..366, got {target}")

        if isinstance(target, str):
            dt = pd.to_datetime(target)
            m, d = dt.month, dt.day
        elif isinstance(target, (datetime, date)):
            m, d = target.month, target.day
        else:
            raise TypeError(f"Unsupported target date type: {type(target)}")

        # Convert (m, d) to 2020 reference leap year day of year
        cum_days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        return cum_days[m - 1] + d
