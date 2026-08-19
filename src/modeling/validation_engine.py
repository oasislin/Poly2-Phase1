#!/usr/bin/env python3
"""
ValidationEngine: Out-of-Sample Validation Engine, Strict Time Wall Isolation, and Rolling-Origin Cross Validation (Ticket 2.3-01 / Issue #20).

Implements (v5.9.1 §5):
    - Strict Time Wall Isolation: Train (2000-2018) vs Holdout (2019) with 0 lookahead bias.
    - Out-of-sample skill scores: CRPS, CRPSS_raw, CRPSS_clim, MAE, 90% CI coverage.
    - Probability Integral Transform (PIT) value extraction for calibration checks.
    - Rolling-Origin Expanding Window Time Series Cross Validation splits.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.crps import gaussian_crps
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured container for out-of-sample validation metrics and daily predictions."""

    station_id: str
    target_type: str
    lead_hours: int
    sample_count: int
    mae_emos: float
    mae_raw: float
    mean_crps_emos: float
    mean_crps_raw: float
    mean_crps_clim: float
    crpss_vs_raw: float
    crpss_vs_clim: float
    coverage_90_ci: float
    pit_values: np.ndarray
    df_daily: pd.DataFrame = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize validation result summary to dictionary."""
        return {
            "station_id": self.station_id,
            "target_type": self.target_type,
            "lead_hours": self.lead_hours,
            "sample_count": self.sample_count,
            "mae_emos": self.mae_emos,
            "mae_raw": self.mae_raw,
            "mean_crps_emos": self.mean_crps_emos,
            "mean_crps_raw": self.mean_crps_raw,
            "mean_crps_clim": self.mean_crps_clim,
            "crpss_vs_raw": self.crpss_vs_raw,
            "crpss_vs_clim": self.crpss_vs_clim,
            "coverage_90_ci": self.coverage_90_ci,
        }


class ValidationEngine:
    """Out-of-sample validation runner and strict time wall isolation verifier."""

    def __init__(
        self,
        storage_manager: Any,
        climatology_calculator: Any,
        model_registry: ModelRegistry,
        train_start_year: int = 2000,
        train_end_year: int = 2018,
        val_start_year: int = 2019,
        val_end_year: int = 2019,
    ):
        self.storage_manager = storage_manager
        self.climatology_calculator = climatology_calculator
        self.model_registry = model_registry
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.val_start_year = val_start_year
        self.val_end_year = val_end_year

        if self.val_start_year <= self.train_end_year:
            raise ValueError(
                f"Time wall breach! val_start_year ({val_start_year}) must be > "
                f"train_end_year ({train_end_year})"
            )

    def load_train_data(self, station_id: str, target_type: str, lead_hours: int) -> pd.DataFrame:
        """Load in-sample training dataset with strict time wall enforcement."""
        df = self.storage_manager.load_training_dataset(
            station_id=station_id,
            target_type=target_type,
            lead_time_bucket=lead_hours,
            start_year=self.train_start_year,
            end_year=self.train_end_year,
        )
        years = pd.to_datetime(df["target_date"]).dt.year
        if not (years <= self.train_end_year).all():
            raise ValueError(
                f"Strict Time Wall Violated: Found training dates after {self.train_end_year}!"
            )
        return df

    def load_val_data(self, station_id: str, target_type: str, lead_hours: int) -> pd.DataFrame:
        """Load out-of-sample validation dataset."""
        df = self.storage_manager.load_training_dataset(
            station_id=station_id,
            target_type=target_type,
            lead_time_bucket=lead_hours,
            start_year=self.val_start_year,
            end_year=self.val_end_year,
        )
        years = pd.to_datetime(df["target_date"]).dt.year
        if not ((years >= self.val_start_year) & (years <= self.val_end_year)).all():
            raise ValueError(
                f"Validation dataset must strictly fall within [{self.val_start_year}, {self.val_end_year}]"
            )
        return df

    def evaluate_slice(
        self,
        station_id: str,
        target_type: str,
        lead_hours: int,
        df_val: Optional[pd.DataFrame] = None,
    ) -> ValidationResult:
        """Run out-of-sample evaluation on holdout dataset and compute statistical metrics."""
        if df_val is None:
            df_val = self.load_val_data(station_id, target_type, lead_hours)

        n_samples = len(df_val)
        if n_samples == 0:
            raise ValueError(f"No validation samples found for {station_id} {target_type} lead={lead_hours}")

        # Extract climatology baseline params for all test dates
        clim_vars = np.array([
            self.climatology_calculator.get_climatology_variance(station_id, target_type, d)
            for d in df_val["target_date"]
        ])
        clim_params = [
            self.climatology_calculator.get_climatology_params(station_id, target_type, d)
            for d in df_val["target_date"]
        ]
        mu_clim = np.array([p[0] for p in clim_params])
        sigma_clim = np.array([p[1] for p in clim_params])

        # Run model inference
        pred_dist = self.model_registry.predict(
            station_id=station_id,
            target_date=df_val["target_date"].iloc[0],  # Facade automatically handles season internally
            target_type=target_type,
            lead_hours=lead_hours,
            ensemble_mean=df_val["ensemble_mean"].values,
            ensemble_variance=df_val["ensemble_variance"].values,
            sigma_clim_squared=clim_vars,
        )

        obs = df_val["observed_temp"].values
        ens_mean = df_val["ensemble_mean"].values
        ens_var = df_val["ensemble_variance"].values
        raw_sigma = np.sqrt(np.maximum(1e-8, ens_var))

        # Vectorized metrics
        emos_mu = pred_dist.mu
        emos_sigma = pred_dist.sigma

        # Handle scalar or vector mu/sigma
        if np.ndim(emos_mu) == 0:
            emos_mu = np.full(n_samples, emos_mu)
            emos_sigma = np.full(n_samples, emos_sigma)

        crps_emos = gaussian_crps(obs, emos_mu, emos_sigma)
        crps_raw = gaussian_crps(obs, ens_mean, raw_sigma)
        crps_clim = gaussian_crps(obs, mu_clim, sigma_clim)

        mae_emos = float(np.mean(np.abs(emos_mu - obs)))
        mae_raw = float(np.mean(np.abs(ens_mean - obs)))

        mean_crps_emos = float(np.mean(crps_emos))
        mean_crps_raw = float(np.mean(crps_raw))
        mean_crps_clim = float(np.mean(crps_clim))

        crpss_raw = 1.0 - (mean_crps_emos / mean_crps_raw) if mean_crps_raw > 0 else 0.0
        crpss_clim = 1.0 - (mean_crps_emos / mean_crps_clim) if mean_crps_clim > 0 else 0.0

        # PIT values: Φ((y - μ) / σ)
        z_scores = (obs - emos_mu) / np.maximum(1e-8, emos_sigma)
        pit_values = stats.norm.cdf(z_scores)

        # 90% Confidence Interval Coverage: [Q_0.05, Q_0.95]
        q_low = emos_mu - 1.6448536269514722 * emos_sigma
        q_high = emos_mu + 1.6448536269514722 * emos_sigma
        in_90_ci = (obs >= q_low) & (obs <= q_high)
        coverage_90 = float(np.mean(in_90_ci))

        df_daily = pd.DataFrame({
            "target_date": df_val["target_date"].values,
            "observed_temp": obs,
            "ensemble_mean": ens_mean,
            "emos_mu": emos_mu,
            "emos_sigma": emos_sigma,
            "crps_emos": crps_emos,
            "crps_raw": crps_raw,
            "crps_clim": crps_clim,
            "pit_value": pit_values,
            "ci_90_low": q_low,
            "ci_90_high": q_high,
            "in_90_ci": in_90_ci,
        })

        return ValidationResult(
            station_id=station_id,
            target_type=target_type,
            lead_hours=lead_hours,
            sample_count=n_samples,
            mae_emos=mae_emos,
            mae_raw=mae_raw,
            mean_crps_emos=mean_crps_emos,
            mean_crps_raw=mean_crps_raw,
            mean_crps_clim=mean_crps_clim,
            crpss_vs_raw=crpss_raw,
            crpss_vs_clim=crpss_clim,
            coverage_90_ci=coverage_90,
            pit_values=pit_values,
            df_daily=df_daily,
        )

    @staticmethod
    def generate_rolling_origin_folds(
        start_year: int = 2000,
        end_year: int = 2018,
        min_train_years: int = 5,
    ) -> List[Tuple[Tuple[int, int], int]]:
        """Generate expanding window (rolling origin) CV folds."""
        folds = []
        for test_year in range(start_year + min_train_years, end_year + 1):
            train_span = (start_year, test_year - 1)
            folds.append((train_span, test_year))
        return folds
