#!/usr/bin/env python3
"""
GaussianEMOS: Gaussian distribution class with squared-parameterization EMOS link functions (Ticket 2.1-02 / Issue #12).

Equations:
    μ = a + b · T̄_ens
    σ² = c² + d² · S²_ens + σ²_clim(d)
    σ = √(σ²)

Guarantees non-negative variance and strictly enforces the climatological variance floor σ² ≥ σ²_clim(d) > 0.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class GaussianEMOS:
    """Gaussian EMOS model and distribution wrapper."""

    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.0,
        d: float = 1.0,
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def from_params(cls, mu: Union[float, np.ndarray], sigma: Union[float, np.ndarray]) -> "GaussianEMOS":
        """Instantiate a bound Gaussian distribution with known μ and σ."""
        sigma_arr = np.asarray(sigma)
        if np.any(sigma_arr <= 0):
            raise ValueError("sigma must be strictly positive")
        return cls(mu=mu, sigma=sigma)

    def compute_params(
        self,
        ensemble_mean: Union[float, np.ndarray, pd.Series],
        ensemble_variance: Union[float, np.ndarray, pd.Series],
        sigma_clim_squared: Union[float, np.ndarray, pd.Series],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute Gaussian distribution parameters (μ, σ) from ensemble statistics and variance floor.

        Equations:
            μ = a + b · ensemble_mean
            σ² = c² + d² · ensemble_variance + sigma_clim_squared
            σ = √(σ²)
        """
        # Ensure non-negative input variance
        ens_var = np.maximum(0.0, np.asarray(ensemble_variance, dtype=np.float64))
        ens_mean = np.asarray(ensemble_mean, dtype=np.float64)
        clim_var = np.maximum(0.0, np.asarray(sigma_clim_squared, dtype=np.float64))

        mu = self.a + self.b * ens_mean
        variance = (self.c ** 2) + (self.d ** 2) * ens_var + clim_var

        # Numerical safety floor to avoid exact 0
        variance = np.maximum(1e-8, variance)
        sigma = np.sqrt(variance)

        if np.ndim(mu) == 0:
            return float(mu), float(sigma)
        return mu, sigma

    def _resolve_mu_sigma(
        self,
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Resolve mu and sigma from explicit arguments or instance attributes."""
        res_mu = mu if mu is not None else self.mu
        res_sigma = sigma if sigma is not None else self.sigma

        if res_mu is None or res_sigma is None:
            raise ValueError(
                "Distribution parameters (mu, sigma) are not set. "
                "Provide them explicitly or instantiate with GaussianEMOS.from_params(mu, sigma)."
            )
        return res_mu, res_sigma

    def pdf(
        self,
        x: Union[float, np.ndarray],
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Probability density function (PDF)."""
        loc, scale = self._resolve_mu_sigma(mu, sigma)
        return stats.norm.pdf(x, loc=loc, scale=scale)

    def cdf(
        self,
        x: Union[float, np.ndarray],
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Cumulative distribution function (CDF)."""
        loc, scale = self._resolve_mu_sigma(mu, sigma)
        return stats.norm.cdf(x, loc=loc, scale=scale)

    def quantile(
        self,
        p: Union[float, np.ndarray],
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Quantile function (Percent point function / Inverse CDF)."""
        loc, scale = self._resolve_mu_sigma(mu, sigma)
        return stats.norm.ppf(p, loc=loc, scale=scale)

    def ppf(
        self,
        p: Union[float, np.ndarray],
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Alias for quantile."""
        return self.quantile(p, mu=mu, sigma=sigma)

    def confidence_interval(
        self,
        level: float = 0.90,
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute equal-tailed confidence/prediction interval for a given level (e.g. 0.90)."""
        if not (0.0 < level < 1.0):
            raise ValueError(f"Confidence level must be between 0 and 1, got {level}")
        loc, scale = self._resolve_mu_sigma(mu, sigma)
        alpha = 1.0 - level
        lower = stats.norm.ppf(alpha / 2.0, loc=loc, scale=scale)
        upper = stats.norm.ppf(1.0 - alpha / 2.0, loc=loc, scale=scale)
        if np.ndim(lower) == 0:
            return float(lower), float(upper)
        return lower, upper

    def crps(
        self,
        observation: Union[float, np.ndarray, pd.Series],
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Compute closed-form Gaussian CRPS for given observation(s)."""
        from src.modeling.crps import gaussian_crps
        loc, scale = self._resolve_mu_sigma(mu, sigma)
        return gaussian_crps(observation, loc, scale)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters to a dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "mu": self.mu,
            "sigma": self.sigma,
        }
