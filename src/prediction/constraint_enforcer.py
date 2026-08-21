#!/usr/bin/env python3
"""
ConstraintEnforcer: Physical warming/cooling rate limits and reachability boundary enforcement (Ticket 3.3-01 / Issue #28).

Implements (Phase 1C / v5.9.2):
    1. Maximum warming rate (r_warm, °C/h) and cooling rate (r_cool, °C/h) by station and season.
    2. Reachability bounds calculation:
       delta_t = max(0.0, time_remaining_to_peak)
       T_max_possible = T_now + r_warm * delta_t
       T_min_possible = T_now - r_cool * delta_t
    3. Hard physical constraint overrides:
       - Max Temp: If L > T_max_possible => P(X >= L) = 0.0 (F_phys(L) = 1.0)
       - Min Temp: If L < T_min_possible => P(X <= L) = 0.0 (F_phys(L) = 0.0)
    4. Strict invariant guarantees:
       - Preserves CDF monotonicity across full temperature range.
       - Hard physical constraints strictly override baseline EMOS and dynamic models.
    5. Historical observation data loading and quantile fitting via from_historical_data.
"""

from datetime import datetime, time
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.modeling.partitioner import DatasetPartitioner

logger = logging.getLogger(__name__)

# Default physical rate limits in °C / hour (derived from historical extreme rate records)
DEFAULT_PHYSICAL_RATE_LIMITS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ZSPD": {
        "Spring": (3.5, 4.0),
        "Summer": (3.5, 3.5),
        "Autumn": (3.5, 4.5),
        "Winter": (3.0, 5.0),
    },
    "KDEN": {
        "Spring": (4.5, 7.0),
        "Summer": (4.0, 6.0),
        "Autumn": (4.5, 8.0),
        "Winter": (5.0, 9.0),
    },
}

# Nominal diurnal extreme occurrence times (local time)
NOMINAL_PEAK_TIMES: Dict[str, time] = {
    "max": time(15, 0),  # 15:00 LT for maximum temperature
    "min": time(6, 0),   # 06:00 LT for minimum temperature
}


class ConstrainedDistribution:
    """Distribution layer enforcing physical thermodynamics and reachability bounds."""

    def __init__(
        self,
        underlying_distribution: Any,
        target_type: str,
        t_min_possible: Optional[float] = None,
        t_max_possible: Optional[float] = None,
        current_temperature: Optional[float] = None,
        is_constrained: bool = False,
    ):
        self.underlying_distribution = underlying_distribution
        self.target_type = target_type.lower()
        if self.target_type not in ["max", "min"]:
            raise ValueError(f"Invalid target_type '{target_type}'. Must be 'max' or 'min'.")

        self.t_min_possible = t_min_possible
        self.t_max_possible = t_max_possible
        self.current_temperature = current_temperature
        self.is_constrained = is_constrained

        # Resolve mu and sigma from underlying distribution
        self.mu = getattr(underlying_distribution, "mu", None)
        self.sigma = getattr(underlying_distribution, "sigma", None)

    def cdf(self, x: Union[float, np.ndarray, pd.Series, Sequence[float]]) -> Union[float, np.ndarray]:
        """Compute the physically constrained cumulative distribution function F_phys(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = (np.ndim(x) == 0)

        # 1. Base / Dynamic CDF forward pass
        base_cdf = np.asarray(self.underlying_distribution.cdf(x_arr), dtype=np.float64)
        if not self.is_constrained:
            return float(base_cdf) if is_scalar else base_cdf

        # 2. Apply physical hard boundaries
        if self.target_type == "max":
            f_phys = np.where(x_arr > self.t_max_possible, 1.0, base_cdf) if self.t_max_possible is not None else base_cdf
        else:
            f_phys = np.where(x_arr < self.t_min_possible, 0.0, base_cdf) if self.t_min_possible is not None else base_cdf

        f_phys = np.clip(f_phys, 0.0, 1.0)
        return float(f_phys) if is_scalar else f_phys

    def probability_greater_than_or_equal(self, threshold: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Compute P(X >= threshold) under physical constraints."""
        return 1.0 - self.cdf(threshold)

    def probability_less_than_or_equal(self, threshold: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Compute P(X <= threshold) under physical constraints."""
        return self.cdf(threshold)

    def probability_between(self, low: float, high: float) -> float:
        """Compute P(low <= X <= high) under physical constraints."""
        return max(0.0, min(1.0, float(self.cdf(high) - self.cdf(low)))) if low <= high else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize distribution metadata."""
        return {
            "target_type": self.target_type,
            "mu": self.mu,
            "sigma": self.sigma,
            "is_constrained": self.is_constrained,
            "t_min_possible": self.t_min_possible,
            "t_max_possible": self.t_max_possible,
            "current_temperature": self.current_temperature,
        }


class ConstraintEnforcer:
    """Applies thermodynamic physical rate constraints based on historical maximum warming/cooling rates."""

    def __init__(
        self,
        custom_rate_limits: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ):
        self.rate_limits = custom_rate_limits or DEFAULT_PHYSICAL_RATE_LIMITS
        self.partitioner = DatasetPartitioner()

    @classmethod
    def from_historical_data(
        cls,
        observations_df: pd.DataFrame,
        station_id: str = "ZSPD",
        warming_quantile: float = 0.999,
        cooling_quantile: float = 0.999,
    ) -> "ConstraintEnforcer":
        """Factory method to compute empirical physical rate limits from historical observations."""
        st = station_id.upper()
        custom_limits = {st: {}}

        if "date" in observations_df.columns and "temp_max" in observations_df.columns and "temp_min" in observations_df.columns:
            df = observations_df.copy()
            df["season"] = df["date"].apply(DatasetPartitioner.get_season)
            for season in ["Spring", "Summer", "Autumn", "Winter"]:
                s_df = df[df["season"] == season]
                if len(s_df) > 10:
                    # Estimate warming rate from diurnal range over ~9 hours (06:00 to 15:00)
                    diurnal_range = (s_df["temp_max"] - s_df["temp_min"]).clip(lower=0.0)
                    est_hourly_warm = (diurnal_range / 9.0).quantile(warming_quantile)
                    est_hourly_cool = (diurnal_range / 9.0).quantile(cooling_quantile)
                    custom_limits[st][season] = (float(max(2.0, est_hourly_warm)), float(max(2.0, est_hourly_cool)))
                else:
                    custom_limits[st][season] = DEFAULT_PHYSICAL_RATE_LIMITS.get(st, DEFAULT_PHYSICAL_RATE_LIMITS["ZSPD"]).get(season, (3.5, 4.0))
        else:
            custom_limits = DEFAULT_PHYSICAL_RATE_LIMITS

        return cls(custom_rate_limits=custom_limits)

    def get_rate_limits(self, station_id: str, season: str) -> Tuple[float, float]:
        """Retrieve (max_warming_rate, max_cooling_rate) in °C/h for station and season."""
        station = station_id.upper()
        seas_cap = season.capitalize()
        norm_seas = seas_cap if seas_cap in ["Spring", "Summer", "Autumn", "Winter"] else self.partitioner.get_season(season)

        st_limits = self.rate_limits.get(station, self.rate_limits.get("ZSPD", {}))
        return st_limits.get(norm_seas, (3.5, 4.0))

    def calculate_delta_hours(
        self,
        target_type: str,
        observation_time: Union[time, datetime, str],
    ) -> float:
        """Calculate remaining hours delta_t from observation time until diurnal peak/valley."""
        nominal_time = NOMINAL_PEAK_TIMES.get(target_type.lower(), time(15, 0))

        if isinstance(observation_time, str):
            try:
                obs_t = datetime.strptime(observation_time, "%H:%M").time()
            except ValueError:
                obs_t = pd.to_datetime(observation_time).time()
        elif isinstance(observation_time, datetime):
            obs_t = observation_time.time()
        elif isinstance(observation_time, time):
            obs_t = observation_time
        else:
            return 0.0

        obs_hours = obs_t.hour + obs_t.minute / 60.0 + obs_t.second / 3600.0
        peak_hours = nominal_time.hour + nominal_time.minute / 60.0
        return float(max(0.0, peak_hours - obs_hours))

    def calculate_reachable_range(
        self,
        station_id: str,
        season: str,
        current_temp: float,
        target_type: str = "max",
        observation_time: Optional[Union[time, datetime, str]] = None,
        delta_hours: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute [T_min_possible, T_max_possible] reachability interval."""
        r_warm, r_cool = self.get_rate_limits(station_id, season)
        dt = max(0.0, float(delta_hours)) if delta_hours is not None else (self.calculate_delta_hours(target_type, observation_time) if observation_time is not None else 0.0)
        return (float(current_temp - r_cool * dt), float(current_temp + r_warm * dt))

    def enforce(
        self,
        distribution: Any,
        station_id: str,
        season: str,
        target_type: str,
        current_temp: Optional[float] = None,
        observation_time: Optional[Union[time, datetime, str]] = None,
        delta_hours: Optional[float] = None,
    ) -> ConstrainedDistribution:
        """Enforce physical warming/cooling constraints on the probability distribution."""
        t_type = target_type.lower()
        t_now = current_temp if current_temp is not None else getattr(distribution, "current_temperature", None)

        if t_now is None or np.isnan(t_now):
            return ConstrainedDistribution(underlying_distribution=distribution, target_type=t_type, is_constrained=False)

        t_min, t_max = self.calculate_reachable_range(
            station_id=station_id,
            season=season,
            current_temp=t_now,
            target_type=t_type,
            observation_time=observation_time,
            delta_hours=delta_hours,
        )

        return ConstrainedDistribution(
            underlying_distribution=distribution,
            target_type=t_type,
            t_min_possible=t_min,
            t_max_possible=t_max,
            current_temperature=t_now,
            is_constrained=True,
        )

    def apply_constraints(
        self,
        probability: float,
        current_temp: float,
        target_temp: float,
        time_remaining_hours: float,
        is_max_temp: bool,
        station_id: str = "ZSPD",
        season: str = "Summer",
    ) -> float:
        """Helper method to constrain a single tail probability."""
        t_type = "max" if is_max_temp else "min"
        t_min, t_max = self.calculate_reachable_range(
            station_id=station_id,
            season=season,
            current_temp=current_temp,
            target_type=t_type,
            delta_hours=time_remaining_hours,
        )

        if is_max_temp and target_temp > t_max:
            return 0.0
        if not is_max_temp and target_temp < t_min:
            return 0.0
        return max(0.0, min(1.0, float(probability)))
