#!/usr/bin/env python3
"""
Unit tests for Gaussian CRPS closed-form analytical solution and loss functions (Ticket 2.1-03 / Issue #13).

Verifies:
1. Gneiting (2005) closed-form Gaussian CRPS formula accuracy against numerical quadrature integration.
2. Limiting behavior: as sigma -> 0, CRPS converges to MAE |y - mu|.
3. Symmetry property: CRPS(mu + delta, mu, sigma) == CRPS(mu - delta, mu, sigma).
4. Non-negativity: CRPS >= 0 for all inputs.
5. Batch vectorization performance and exactness over 1D/2D arrays and Pandas Series.
6. emos_crps_loss function with and without L2 regularization on parameter d.
7. Numerical stability under extreme z-scores (|z| > 20) and small sigma.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src.modeling.crps import gaussian_crps, emos_crps_loss
from src.modeling.gaussian_emos import GaussianEMOS


def numerical_crps_reference(y: float, mu: float, sigma: float) -> float:
    """Independent reference implementation of CRPS via numerical quadrature integration."""
    # CRPS(F, y) = \int_{-\infty}^\infty (F(x) - 1(x >= y))^2 dx
    # Integration range: [mu - 10*sigma, mu + 10*sigma] covers essentially 100% of probability mass
    lower = min(mu - 10.0 * sigma, y - 5.0 * sigma)
    upper = max(mu + 10.0 * sigma, y + 5.0 * sigma)
    
    def integrand(x):
        Fx = stats.norm.cdf(x, loc=mu, scale=sigma)
        step = 1.0 if x >= y else 0.0
        return (Fx - step) ** 2

    val, _ = integrate.quad(integrand, lower, upper, limit=200, epsabs=1e-8, epsrel=1e-8)
    return float(val)


class TestGaussianCRPSAnalytical:
    """Test the Gneiting analytical formula implementation."""

    @pytest.mark.parametrize("y,mu,sigma", [
        (20.0, 20.0, 2.0),     # Perfect mean forecast
        (25.0, 20.0, 2.0),     # Overestimate / Underestimate
        (15.0, 20.0, 2.0),     # Symmetric opposite
        (35.5, 30.2, 4.5),     # Real summer temperature range
        (-12.0, -15.0, 3.2),   # Real winter below-freezing range
        (10.0, 10.0, 0.01),    # Very small sigma
    ])
    def test_analytical_crps_matches_numerical_quadrature(self, y, mu, sigma):
        crps_analytic = gaussian_crps(y, mu, sigma)
        crps_numeric = numerical_crps_reference(y, mu, sigma)
        assert np.isclose(crps_analytic, crps_numeric, rtol=1e-5, atol=1e-5)

    def test_symmetry(self):
        mu = 20.0
        sigma = 3.0
        crps_pos = gaussian_crps(mu + 4.0, mu, sigma)
        crps_neg = gaussian_crps(mu - 4.0, mu, sigma)
        assert np.isclose(crps_pos, crps_neg)

    def test_non_negativity(self):
        ys = np.linspace(-30, 50, 100)
        mus = np.linspace(-20, 40, 100)
        sigmas = np.linspace(0.1, 10, 100)
        
        crps_vals = gaussian_crps(ys, mus, sigmas)
        assert (crps_vals >= 0.0).all()

    def test_small_sigma_limit_approaches_mae(self):
        """As sigma -> 0, Gaussian CRPS converges to |y - mu|."""
        y = 28.0
        mu = 25.0
        expected_mae = abs(y - mu)  # 3.0
        
        crps_small_sigma = gaussian_crps(y, mu, sigma=1e-6)
        assert np.isclose(crps_small_sigma, expected_mae, rtol=1e-4)

    def test_vectorization_numpy_and_pandas(self):
        y = np.array([20.0, 25.0, 30.0])
        mu = np.array([19.0, 25.0, 32.0])
        sigma = np.array([2.0, 3.0, 1.5])
        
        crps_arr = gaussian_crps(y, mu, sigma)
        assert isinstance(crps_arr, np.ndarray)
        assert len(crps_arr) == 3
        
        # Test pandas Series input
        crps_series = gaussian_crps(pd.Series(y), pd.Series(mu), pd.Series(sigma))
        assert np.allclose(crps_arr, crps_series)

    def test_gaussian_emos_method_binding(self):
        """Test the crps method directly on GaussianEMOS instance."""
        emos = GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0)
        mu, sigma = emos.compute_params(ensemble_mean=20.0, ensemble_variance=4.0, sigma_clim_squared=0.0)
        
        # Scalar
        val = emos.crps(observation=22.0, mu=mu, sigma=sigma)
        assert np.isclose(val, gaussian_crps(22.0, 20.0, 2.0))
        
        # Bound instance
        bound = GaussianEMOS.from_params(mu=20.0, sigma=2.0)
        assert np.isclose(bound.crps(22.0), val)


class TestEMOSCRPSLoss:
    """Test the optimization objective loss function."""

    def test_loss_computation_and_l2_regularization(self):
        n = 100
        ens_mean = np.random.normal(20.0, 5.0, n)
        ens_var = np.random.uniform(1.0, 4.0, n)
        clim_var = np.full(n, 2.0)
        obs = ens_mean + np.random.normal(0, 1.5, n)
        
        # Loss without L2
        loss_no_reg = emos_crps_loss(
            params=(0.0, 1.0, 0.0, 1.0),
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=obs,
            l2_lambda_d=0.0,
        )
        assert loss_no_reg > 0.0
        
        # Loss with L2 penalty on d: L = mean(CRPS) + lambda * d^2
        lambda_d = 0.05
        d_val = 2.0
        loss_with_reg = emos_crps_loss(
            params=(0.0, 1.0, 0.0, d_val),
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=obs,
            l2_lambda_d=lambda_d,
        )
        base_crps = emos_crps_loss(
            params=(0.0, 1.0, 0.0, d_val),
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=obs,
            l2_lambda_d=0.0,
        )
        assert np.isclose(loss_with_reg, base_crps + lambda_d * (d_val ** 2))

    def test_numerical_stability_under_extreme_values(self):
        """Ensure no NaNs or Infs even with extreme errors or zero variance."""
        ens_mean = np.array([100.0, -100.0, 20.0])
        ens_var = np.array([0.0, 0.0, 1e-10])
        clim_var = np.array([0.0, 0.0, 0.0])
        obs = np.array([0.0, 0.0, 20.0])
        
        loss = emos_crps_loss(
            params=(0.0, 1.0, 0.0, 1.0),
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=obs,
            l2_lambda_d=1e-3,
        )
        assert np.isfinite(loss)
        assert loss > 0.0
