#!/usr/bin/env python3
"""
Gneiting (2005) closed-form Gaussian CRPS analytical solution and optimization loss (Ticket 2.1-03 / Issue #13).

Formula:
    CRPS(y, μ, σ) = σ · [ z · (2Φ(z) - 1) + 2φ(z) - 1/√π ]
where:
    z = (y - μ) / σ
    Φ(z) = standard normal CDF
    φ(z) = standard normal PDF
"""

import math
from typing import Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import special

# Mathematical constant 1 / sqrt(pi)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SQRT_2 = math.sqrt(2.0)


def gaussian_crps(
    y: Union[float, np.ndarray, pd.Series],
    mu: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """Compute the closed-form Continuous Ranked Probability Score for a Gaussian distribution.

    Args:
        y: Observed truth temperature(s).
        mu: Predicted distribution mean(s).
        sigma: Predicted distribution standard deviation(s) (strictly positive).

    Returns:
        CRPS value(s) in the same unit as y (°C).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    is_scalar = (y_arr.ndim == 0) and (mu_arr.ndim == 0) and (sigma_arr.ndim == 0)

    # Broadcast to common shape
    y_b, mu_b, sigma_b = np.broadcast_arrays(y_arr, mu_arr, sigma_arr)

    # Numerical safety for zero / near-zero sigma: limit is MAE |y - mu|
    small_sigma_mask = sigma_b < 1e-8
    
    # Safe sigma for division
    safe_sigma = np.where(small_sigma_mask, 1.0, sigma_b)
    z = (y_b - mu_b) / safe_sigma

    # erf(z / sqrt(2)) == 2 * Phi(z) - 1
    erf_z = special.erf(z / SQRT_2)
    # pdf: phi(z) = 1/sqrt(2pi) * exp(-z^2/2) -> 2*phi(z) = sqrt(2/pi) * exp(-z^2/2)
    exp_term = SQRT_2_OVER_PI * np.exp(-0.5 * np.square(z))

    # Analytical CRPS formula
    crps_values = sigma_b * (z * erf_z + exp_term - INV_SQRT_PI)

    # For tiny sigma, replace with exact MAE limit
    if np.any(small_sigma_mask):
        mae = np.abs(y_b - mu_b)
        crps_values = np.where(small_sigma_mask, mae, crps_values)

    # Ensure non-negativity due to float rounding near 0
    crps_values = np.maximum(0.0, crps_values)

    if is_scalar:
        return float(crps_values.item())
    return crps_values


def emos_crps_loss(
    params: Sequence[float],
    ensemble_mean: Union[np.ndarray, pd.Series],
    ensemble_variance: Union[np.ndarray, pd.Series],
    sigma_clim_squared: Union[np.ndarray, pd.Series],
    observed_temp: Union[np.ndarray, pd.Series],
    l2_lambda_d: float = 0.0,
) -> float:
    """Vectorized sample average CRPS loss function with optional L2 regularization on parameter d.

    Args:
        params: (a, b, c, d) EMOS calibration parameters.
        ensemble_mean: Forecast ensemble mean array.
        ensemble_variance: Forecast ensemble variance array.
        sigma_clim_squared: Climatological variance floor array.
        observed_temp: True observed temperature array.
        l2_lambda_d: L2 regularization coefficient for parameter d.

    Returns:
        Scalar loss value to be minimized by scipy.optimize.
    """
    a, b, c, d = params[0], params[1], params[2], params[3]

    ens_mean = np.asarray(ensemble_mean, dtype=np.float64)
    ens_var = np.maximum(0.0, np.asarray(ensemble_variance, dtype=np.float64))
    clim_var = np.maximum(0.0, np.asarray(sigma_clim_squared, dtype=np.float64))
    y = np.asarray(observed_temp, dtype=np.float64)

    # Link functions: μ = a + b * T̄_ens, σ² = c² + d² * S²_ens + σ²_clim
    mu = a + b * ens_mean
    variance = (c ** 2) + (d ** 2) * ens_var + clim_var
    variance = np.maximum(1e-8, variance)
    sigma = np.sqrt(variance)

    # Compute batch CRPS
    crps_scores = gaussian_crps(y, mu, sigma)
    mean_crps = float(np.mean(crps_scores))

    # Apply L2 penalty only on d
    if l2_lambda_d > 0.0:
        mean_crps += float(l2_lambda_d * (d ** 2))

    return mean_crps
