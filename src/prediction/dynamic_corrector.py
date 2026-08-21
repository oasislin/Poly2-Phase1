#!/usr/bin/env python3
"""
DynamicCorrector: Real-time observation integration and conditional probability truncation (Ticket 3.2-01 / Issue #27).

Implements (Phase 1C / v5.9.2):
    1. Maximum Temperature conditional truncation:
       - If L <= T_now: P(X >= L | X >= T_now) = 1.0 (F_post(L) = 0.0)
       - If L > T_now: P(X >= L | X >= T_now) = (1 - F(L)) / (1 - F(T_now))
         => F_post(L) = (F(L) - F(T_now)) / (1 - F(T_now))
    2. Minimum Temperature conditional truncation:
       - If L >= T_now: P(X <= L | X <= T_now) = 1.0 (F_post(L) = 1.0)
       - If L < T_now: P(X <= L | X <= T_now) = F(L) / F(T_now)
         => F_post(L) = F(L) / F(T_now)
    3. Strict invariant guarantees:
       - Posterior CDF F_post is monotonic non-decreasing on (-inf, +inf) and bounded in [0, 1].
       - Closed-form analytical quantile function for truncated distribution.
    4. Safety & numerical protections:
       - Safe epsilon clipping (eps=1e-7) against division by near-zero denominators.
       - Missing observation (None / NaN) transparently falls back to prior distribution.
"""

from datetime import datetime
import logging
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.gaussian_emos import GaussianEMOS

logger = logging.getLogger(__name__)


class TruncatedDistribution:
    """Posterior probability distribution conditioned on real-time temperature observation."""

    def __init__(
        self,
        base_distribution: GaussianEMOS,
        target_type: str,
        current_temperature: Optional[float] = None,
        observation_time: Optional[datetime] = None,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        self.base_distribution = base_distribution
        self.target_type = target_type.lower()
        if self.target_type not in ["max", "min"]:
            raise ValueError(f"Invalid target_type '{target_type}'. Must be 'max' or 'min'.")

        self.observation_time = observation_time

        # Validate current temperature
        if current_temperature is not None and not np.isnan(current_temperature):
            self.current_temperature: Optional[float] = float(current_temperature)
            self.is_truncated: bool = True
        else:
            self.current_temperature = None
            self.is_truncated = False

        # Extract or resolve distribution parameters
        if mu is not None and sigma is not None:
            self.mu = float(mu)
            self.sigma = float(sigma)
        elif base_distribution.mu is not None and base_distribution.sigma is not None:
            self.mu = float(base_distribution.mu)
            self.sigma = float(base_distribution.sigma)
        else:
            self.mu = float(base_distribution.a)
            self.sigma = float(np.sqrt(base_distribution.c ** 2 + 1e-6))

    def cdf(self, x: Union[float, np.ndarray, pd.Series, Sequence[float]]) -> Union[float, np.ndarray]:
        """Compute the conditionally truncated cumulative distribution function F_post(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = (np.ndim(x) == 0)

        # 1. Non-truncated fallback
        if not self.is_truncated or self.current_temperature is None:
            prior_cdf = self.base_distribution.cdf(x=x_arr, mu=self.mu, sigma=self.sigma)
            return float(prior_cdf) if is_scalar else prior_cdf

        t_now = self.current_temperature
        f_t_now = float(self.base_distribution.cdf(x=t_now, mu=self.mu, sigma=self.sigma))
        f_x = np.asarray(self.base_distribution.cdf(x=x_arr, mu=self.mu, sigma=self.sigma), dtype=np.float64)

        if self.target_type == "max":
            # Max Temp: Daily max >= T_now
            # If x <= T_now: F_post(x) = 0.0
            # If x > T_now: F_post(x) = (F(x) - F(T_now)) / (1.0 - F(T_now))
            denom = max(1.0 - f_t_now, 1e-7)
            f_post = np.where(
                x_arr <= t_now,
                0.0,
                (f_x - f_t_now) / denom,
            )
        else:
            # Min Temp: Daily min <= T_now
            # If x >= T_now: F_post(x) = 1.0
            # If x < T_now: F_post(x) = F(x) / F(T_now)
            denom = max(f_t_now, 1e-7)
            f_post = np.where(
                x_arr >= t_now,
                1.0,
                f_x / denom,
            )

        f_post = np.clip(f_post, 0.0, 1.0)
        return float(f_post) if is_scalar else f_post

    def probability_greater_than_or_equal(
        self,
        threshold: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray]:
        """Compute P(X >= threshold | observation)."""
        cdf_val = self.cdf(threshold)
        return 1.0 - cdf_val

    def probability_less_than_or_equal(
        self,
        threshold: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray]:
        """Compute P(X <= threshold | observation)."""
        return self.cdf(threshold)

    def probability_between(
        self,
        low: float,
        high: float,
    ) -> float:
        """Compute P(low <= X <= high | observation) = F_post(high) - F_post(low)."""
        if low > high:
            return 0.0
        p = float(self.cdf(high) - self.cdf(low))
        return max(0.0, min(1.0, p))

    def quantile(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Closed-form inverse CDF (quantile function) for the truncated Gaussian."""
        p_arr = np.asarray(p, dtype=np.float64)
        is_scalar = (np.ndim(p) == 0)

        if not self.is_truncated or self.current_temperature is None:
            q_prior = self.base_distribution.quantile(p=p_arr, mu=self.mu, sigma=self.sigma)
            return float(q_prior) if is_scalar else q_prior

        t_now = self.current_temperature
        f_t_now = float(self.base_distribution.cdf(x=t_now, mu=self.mu, sigma=self.sigma))

        if self.target_type == "max":
            # For Max: p = (F(x) - F(T_now)) / (1 - F(T_now)) => F(x) = F(T_now) + p * (1 - F(T_now))
            target_cdf = f_t_now + p_arr * (1.0 - f_t_now)
        else:
            # For Min: p = F(x) / F(T_now) => F(x) = p * F(T_now)
            target_cdf = p_arr * f_t_now

        target_cdf = np.clip(target_cdf, 1e-7, 1.0 - 1e-7)
        q_val = self.base_distribution.quantile(p=target_cdf, mu=self.mu, sigma=self.sigma)
        return float(q_val) if is_scalar else q_val

    def to_dict(self) -> Dict[str, Any]:
        """Serialize distribution metadata."""
        return {
            "target_type": self.target_type,
            "mu": self.mu,
            "sigma": self.sigma,
            "is_truncated": self.is_truncated,
            "current_temperature": self.current_temperature,
            "observation_time": self.observation_time.isoformat() if self.observation_time else None,
        }


class DynamicCorrector:
    """Real-time observation processor and conditional probability truncation engine."""

    def __init__(
        self,
        current_temperature: Optional[float] = None,
        observation_time: Optional[datetime] = None,
    ):
        self.current_temperature = current_temperature
        self.observation_time = observation_time

    def update_observation(
        self,
        new_temperature: Optional[float],
        observation_time: Optional[datetime] = None,
    ) -> None:
        """Update with a new real-time station temperature observation."""
        self.current_temperature = new_temperature
        self.observation_time = observation_time
        logger.debug(
            "Updated observation: T_now=%.2f at %s",
            new_temperature if new_temperature is not None else np.nan,
            observation_time,
        )

    def correct(
        self,
        base_distribution: Union[GaussianEMOS, Any],
        target_type: str,
        current_temperature: Optional[float] = None,
        observation_time: Optional[datetime] = None,
    ) -> TruncatedDistribution:
        """Apply conditional truncation to a base Gaussian distribution."""
        # Check if base_distribution is StaticPredictionResult or GaussianEMOS
        if hasattr(base_distribution, "distribution") and hasattr(base_distribution, "mu"):
            emos_model = base_distribution.distribution
            mu = base_distribution.mu
            sigma = base_distribution.sigma
        elif isinstance(base_distribution, GaussianEMOS):
            emos_model = base_distribution
            mu = base_distribution.mu
            sigma = base_distribution.sigma
        else:
            emos_model = base_distribution
            mu = getattr(base_distribution, "mu", None)
            sigma = getattr(base_distribution, "sigma", None)

        t_now = current_temperature if current_temperature is not None else self.current_temperature
        obs_time = observation_time if observation_time is not None else self.observation_time

        return TruncatedDistribution(
            base_distribution=emos_model,
            target_type=target_type,
            current_temperature=t_now,
            observation_time=obs_time,
            mu=mu,
            sigma=sigma,
        )

    def correct_max_temp_probability(
        self,
        base_distribution: Union[GaussianEMOS, Any],
        threshold: float,
        current_temp: Optional[float] = None,
    ) -> float:
        """Calculate P(X >= threshold | X >= T_now) for maximum temperature."""
        dist = self.correct(
            base_distribution=base_distribution,
            target_type="max",
            current_temperature=current_temp,
        )
        return float(dist.probability_greater_than_or_equal(threshold))

    def correct_min_temp_probability(
        self,
        base_distribution: Union[GaussianEMOS, Any],
        threshold: float,
        current_temp: Optional[float] = None,
    ) -> float:
        """Calculate P(X <= threshold | X <= T_now) for minimum temperature."""
        dist = self.correct(
            base_distribution=base_distribution,
            target_type="min",
            current_temperature=current_temp,
        )
        return float(dist.probability_less_than_or_equal(threshold))
