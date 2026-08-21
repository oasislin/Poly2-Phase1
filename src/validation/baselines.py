#!/usr/bin/env python3
"""
Baselines: Reference Benchmark Predictors (Climatology, Raw GEFS, Persistence).
Part of Phase 1D Validation System (Ticket 4.2-01 / Issue #36).

Implements:
    - ClimatologyBaseline: Uses historical day-of-year mean and standard deviation floor.
    - RawGEFSBaseline: Uses uncalibrated physical ensemble mean and empirical spread.
    - PersistenceBaseline: Uses yesterday's observed temperature as mean and climatological standard deviation.
"""

from datetime import date, datetime, timedelta
import math
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ClimatologyBaseline:
    """Climatological reference benchmark based on OOS historical day-of-year statistics."""

    def __init__(self, climatology_calculator: Any):
        self.climatology_calculator = climatology_calculator

    def predict(
        self,
        station_id: str,
        target_date: Union[date, datetime, str],
        target_type: str,
    ) -> Tuple[float, float]:
        """Generate climatology (mu, sigma) prediction for given station, date, and target type."""
        return self.climatology_calculator.get_climatology(station_id, target_type, target_date)


class RawGEFSBaseline:
    """Uncalibrated numerical ensemble baseline directly taking raw GEFS statistics."""

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def predict(
        self,
        ensemble_mean: float,
        ensemble_variance: float,
    ) -> Tuple[float, float]:
        """Generate raw GEFS prediction (mean, max(sqrt(var), eps))."""
        mu = float(ensemble_mean)
        sigma = math.sqrt(max(float(ensemble_variance), 0.0))
        sigma_safe = max(sigma, self.eps)
        return mu, sigma_safe


class PersistenceBaseline:
    """Persistence benchmark using yesterday's observed temperature as forecast mean."""

    def __init__(
        self,
        storage_manager: Optional[Any] = None,
        climatology_calculator: Optional[Any] = None,
        default_sigma: float = 3.0,
    ):
        self.storage_manager = storage_manager
        self.climatology_calculator = climatology_calculator
        self.default_sigma = default_sigma

    def predict(
        self,
        station_id: str,
        target_date: Union[date, datetime, str],
        target_type: str,
        yesterday_truth: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Generate persistence forecast (mu=yesterday_obs, sigma=clim_sigma)."""
        if isinstance(target_date, str):
            dt = pd.to_datetime(target_date).date()
        elif isinstance(target_date, datetime):
            dt = target_date.date()
        else:
            dt = target_date

        if yesterday_truth is not None and not np.isnan(yesterday_truth):
            mu = float(yesterday_truth)
        elif self.storage_manager is not None:
            prev_date = dt - timedelta(days=1)
            mu = float(self.storage_manager.get_observed_temperature(station_id, prev_date, target_type))
        else:
            raise ValueError("Must provide yesterday_truth or storage_manager to evaluate PersistenceBaseline.")

        if self.climatology_calculator is not None:
            _, sigma = self.climatology_calculator.get_climatology(station_id, target_type, dt)
        else:
            sigma = self.default_sigma

        return mu, float(sigma)
