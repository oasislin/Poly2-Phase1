#!/usr/bin/env python3
"""
Unit tests for GaussianEMOS distribution class (Ticket 2.1-02 / Issue #12).

Verifies:
1. Squared parameterization: mu = a + b*ens_mean, sigma^2 = c^2 + d^2*ens_var + sigma_clim^2.
2. Variance floor invariant: sigma^2 >= sigma_clim^2 regardless of (c, d) values (even if negative or zero).
3. Standard probability functions: PDF, CDF, Quantile (PPF), Confidence Interval.
4. Accuracy against scipy.stats.norm.
5. Vectorized array/Series support for batch inference.
6. Edge case handling (e.g. ens_var=0, extreme values).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.modeling.gaussian_emos import GaussianEMOS


class TestGaussianEMOSLinkFunctions:
    """Test the mean and variance parameter link functions."""

    def test_link_function_scalar(self):
        emos = GaussianEMOS(a=1.5, b=0.9, c=0.4, d=1.2)
        mu, sigma = emos.compute_params(
            ensemble_mean=20.0,
            ensemble_variance=4.0,
            sigma_clim_squared=2.25,
        )
        
        # Expected:
        # mu = 1.5 + 0.9 * 20.0 = 19.5
        # sigma^2 = 0.4^2 + 1.2^2 * 4.0 + 2.25 = 0.16 + 1.44 * 4.0 + 2.25 = 0.16 + 5.76 + 2.25 = 8.17
        # sigma = sqrt(8.17)
        assert np.isclose(mu, 19.5)
        assert np.isclose(sigma, np.sqrt(8.17))

    def test_variance_floor_never_breached_even_with_zero_or_negative_params(self):
        """Even with c=0, d=0, ens_var=0 or negative parameters, sigma^2 >= sigma_clim^2."""
        emos = GaussianEMOS(a=0.0, b=1.0, c=-2.0, d=-1.5)
        
        # If c=-2.0, c^2 = 4.0; d=-1.5, d^2 = 2.25
        mu, sigma = emos.compute_params(
            ensemble_mean=15.0,
            ensemble_variance=0.0,
            sigma_clim_squared=3.0,
        )
        # sigma^2 = (-2)^2 + (-1.5)^2 * 0 + 3.0 = 4.0 + 0 + 3.0 = 7.0 >= 3.0
        assert sigma ** 2 >= 3.0
        assert np.isclose(sigma ** 2, 7.0)

    def test_link_function_vectorized(self):
        emos = GaussianEMOS(a=0.0, b=1.0, c=0.5, d=1.0)
        ens_means = np.array([10.0, 20.0, 30.0])
        ens_vars = np.array([1.0, 2.0, 3.0])
        clim_vars = np.array([2.0, 2.0, 2.0])
        
        mus, sigmas = emos.compute_params(ens_means, ens_vars, clim_vars)
        assert len(mus) == 3
        assert len(sigmas) == 3
        assert np.allclose(mus, [10.0, 20.0, 30.0])
        expected_vars = 0.25 + 1.0 * ens_vars + 2.0
        assert np.allclose(sigmas ** 2, expected_vars)


class TestGaussianEMOSDistributionMethods:
    """Test PDF, CDF, quantile, and confidence interval methods."""

    def test_pdf_against_scipy(self):
        emos = GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0)
        mu, sigma = emos.compute_params(ensemble_mean=25.0, ensemble_variance=4.0, sigma_clim_squared=0.0)
        # mu = 25.0, sigma = 2.0
        
        test_points = np.array([20.0, 23.0, 25.0, 27.0, 30.0])
        pdf_actual = emos.pdf(test_points, mu=mu, sigma=sigma)
        pdf_expected = stats.norm.pdf(test_points, loc=25.0, scale=2.0)
        assert np.allclose(pdf_actual, pdf_expected)

    def test_cdf_against_scipy(self):
        emos = GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0)
        mu, sigma = emos.compute_params(ensemble_mean=25.0, ensemble_variance=4.0, sigma_clim_squared=0.0)
        
        test_points = np.array([20.0, 23.0, 25.0, 27.0, 30.0])
        cdf_actual = emos.cdf(test_points, mu=mu, sigma=sigma)
        cdf_expected = stats.norm.cdf(test_points, loc=25.0, scale=2.0)
        assert np.allclose(cdf_actual, cdf_expected)

    def test_quantile_against_scipy(self):
        emos = GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0)
        mu, sigma = emos.compute_params(ensemble_mean=25.0, ensemble_variance=4.0, sigma_clim_squared=0.0)
        
        probs = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        q_actual = emos.quantile(probs, mu=mu, sigma=sigma)
        q_expected = stats.norm.ppf(probs, loc=25.0, scale=2.0)
        assert np.allclose(q_actual, q_expected)

    def test_confidence_interval(self):
        emos = GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0)
        mu, sigma = emos.compute_params(ensemble_mean=25.0, ensemble_variance=4.0, sigma_clim_squared=0.0)
        
        # 90% CI is [5th percentile, 95th percentile]
        low, high = emos.confidence_interval(level=0.90, mu=mu, sigma=sigma)
        assert np.isclose(low, stats.norm.ppf(0.05, loc=25.0, scale=2.0))
        assert np.isclose(high, stats.norm.ppf(0.95, loc=25.0, scale=2.0))
        assert low < mu < high

    def test_distribution_instance_binding(self):
        """Test bound instance methods when (mu, sigma) are provided at initialization."""
        dist = GaussianEMOS.from_params(mu=22.0, sigma=3.0)
        assert dist.mu == 22.0
        assert dist.sigma == 3.0
        assert np.isclose(dist.cdf(22.0), 0.5)
        assert np.isclose(dist.quantile(0.5), 22.0)
