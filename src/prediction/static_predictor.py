#!/usr/bin/env python3
"""
StaticPredictor: Base Gaussian EMOS prediction and model routing (Ticket 3.1-01 / Issue #26).

Implements (Phase 1C / v5.9.2):
    1. Direct routing to exact anchor EMOS models ({54, 30, 6}h for max, {48, 24}h for min).
    2. Seamless parameter linear interpolation across missing lead times via LeadTimeInterpolator.
    3. Minimum temperature short-lead (< 24h) physical variance shrinkage (sigma * sqrt(L / 24)).
    4. Exact square parameterization calculation:
       mu = a + b * ens_mean
       sigma^2 = c^2 + d^2 * ens_var + sigma_clim^2
    5. Automatic variance floor querying via ClimatologyCalculator if sigma_clim_squared is omitted.
    6. Batch vectorized DataFrame inference with confidence and prediction interval generation.
"""

from dataclasses import dataclass
from datetime import date, datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.interpolator import LeadTimeInterpolator
from src.modeling.partitioner import DatasetPartitioner
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class StaticPredictionResult:
    """Container holding static prediction parameters and probabilistic distribution."""

    station_id: str
    target_date: str
    target_type: str
    lead_time_hours: float
    season: str
    mu: float
    sigma: float
    distribution: GaussianEMOS
    is_interpolated: bool = False
    is_short_lead_decay: bool = False

    def confidence_interval(self, level: float = 0.90) -> Tuple[float, float]:
        """Compute symmetric confidence interval around mean."""
        alpha = 1.0 - level
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        margin = float(z * self.sigma)
        return (float(self.mu - margin), float(self.mu + margin))

    def prediction_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute prediction interval (alias with default 95%)."""
        return self.confidence_interval(level=level)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function value at x."""
        return self.distribution.cdf(x=x, mu=self.mu, sigma=self.sigma)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function value at x."""
        return self.distribution.pdf(x=x, mu=self.mu, sigma=self.sigma)

    def quantile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse CDF / quantile value for probability q."""
        return self.distribution.quantile(p=q, mu=self.mu, sigma=self.sigma)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        ci_90 = self.confidence_interval(0.90)
        ci_95 = self.confidence_interval(0.95)
        return {
            "station_id": self.station_id,
            "target_date": self.target_date,
            "target_type": self.target_type,
            "lead_time_hours": self.lead_time_hours,
            "season": self.season,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "is_interpolated": self.is_interpolated,
            "is_short_lead_decay": self.is_short_lead_decay,
            "ci_90_lower": ci_90[0],
            "ci_90_upper": ci_90[1],
            "ci_95_lower": ci_95[0],
            "ci_95_upper": ci_95[1],
        }


class StaticPredictor:
    """Base Gaussian EMOS inference engine and model routing orchestrator."""

    def __init__(
        self,
        model_registry: Optional[Union[ModelRegistry, Any]] = None,
        climatology_calculator: Optional[Union[ClimatologyCalculator, Any]] = None,
    ):
        self.registry = model_registry or ModelRegistry()
        self.clim_calc = climatology_calculator
        self.partitioner = DatasetPartitioner()
        self.interpolator = LeadTimeInterpolator()

    def _resolve_variance_floor(
        self,
        station: str,
        t_type: str,
        date_str: str,
        sigma_clim_sq: Optional[float],
    ) -> float:
        """Resolve climatological variance floor from explicit argument or calculator."""
        if sigma_clim_sq is not None:
            return float(sigma_clim_sq)
        if self.clim_calc is not None:
            return float(self.clim_calc.get_climatology_variance(
                station_id=station,
                target_type=t_type,
                target_date=date_str,
            ))
        return 0.0

    def _obtain_calibrated_distribution(
        self,
        station: str,
        date_str: str,
        t_type: str,
        lead: float,
        ens_mean: float,
        ens_var: float,
        clim_var: float,
    ) -> Tuple[GaussianEMOS, float, float]:
        """Obtain calibrated Gaussian distribution and evaluate its parameters."""
        if hasattr(self.registry, "predict"):
            dist = self.registry.predict(
                station_id=station,
                target_date=date_str,
                target_type=t_type,
                lead_hours=lead,
                ensemble_mean=ens_mean,
                ensemble_variance=ens_var,
                sigma_clim_squared=clim_var,
            )
            if dist.mu is not None and dist.sigma is not None:
                return dist, float(dist.mu), float(dist.sigma)
            mu, sigma = dist.compute_params(ens_mean, ens_var, clim_var)
            return dist, float(mu), float(sigma)

        model = self.registry.get_model(station, date_str, t_type, lead)
        mu, sigma = model.compute_params(ens_mean, ens_var, clim_var)
        return model, float(mu), float(sigma)

    def predict_single(
        self,
        station_id: str,
        target_date: Union[str, date, datetime, pd.Timestamp],
        target_type: str,
        lead_time_hours: Union[int, float],
        ensemble_mean: float,
        ensemble_variance: float,
        sigma_clim_squared: Optional[float] = None,
    ) -> StaticPredictionResult:
        """Generate calibrated Gaussian static prediction for a single timestamp and lead time."""
        t_type = target_type.lower()
        if t_type not in ["max", "min"]:
            raise ValueError(f"Invalid target_type '{target_type}'. Must be 'max' or 'min'.")

        station = station_id.upper()
        date_str = pd.to_datetime(target_date).strftime("%Y-%m-%d")
        season = self.partitioner.get_season(date_str)
        lead = float(lead_time_hours)

        clim_var = self._resolve_variance_floor(station, t_type, date_str, sigma_clim_squared)
        dist, mu, sigma = self._obtain_calibrated_distribution(
            station, date_str, t_type, lead, ensemble_mean, ensemble_variance, clim_var
        )

        anchor_leads = self.partitioner.get_lead_time_nodes(t_type)
        int_lead = int(round(lead))
        is_exact_anchor = (int_lead in anchor_leads) and np.isclose(lead, int_lead)
        is_short_lead = (t_type == "min") and (lead < 24.0)

        return StaticPredictionResult(
            station_id=station,
            target_date=date_str,
            target_type=t_type,
            lead_time_hours=lead,
            season=season,
            mu=mu,
            sigma=sigma,
            distribution=dist,
            is_interpolated=(not is_exact_anchor and not is_short_lead),
            is_short_lead_decay=is_short_lead,
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
        target_type: Optional[str] = None,
        lead_time_hours: Optional[float] = None,
    ) -> pd.DataFrame:
        """Batch predict Gaussian parameters for a DataFrame of ensemble features."""
        results = []
        for _, row in df.iterrows():
            row_station = station_id or row.get("station_id")
            row_type = target_type or row.get("target_type")
            row_date = row.get("target_date")
            row_lead = lead_time_hours if lead_time_hours is not None else row.get("lead_time_hours")
            row_ens_mean = float(row["ensemble_mean"])
            row_ens_var = float(row["ensemble_variance"])
            row_clim_var = float(row["sigma_clim_squared"]) if "sigma_clim_squared" in row and pd.notna(row["sigma_clim_squared"]) else None

            res = self.predict_single(
                station_id=str(row_station),
                target_date=row_date,
                target_type=str(row_type),
                lead_time_hours=float(row_lead),
                ensemble_mean=row_ens_mean,
                ensemble_variance=row_ens_var,
                sigma_clim_squared=row_clim_var,
            )
            results.append(res.to_dict())

        res_df = pd.DataFrame(results).rename(columns={"mu": "predicted_mu", "sigma": "predicted_sigma"})

        out_df = df.copy()
        for col in ["predicted_mu", "predicted_sigma", "season", "is_interpolated", "is_short_lead_decay",
                    "ci_90_lower", "ci_90_upper", "ci_95_lower", "ci_95_upper"]:
            if col in res_df.columns:
                out_df[col] = res_df[col].values

        return out_df
