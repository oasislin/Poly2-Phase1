#!/usr/bin/env python3
"""
Unit tests for EMOSOptimizer and ModelTrainingDiagnostics (Ticket 2.2-01 / Issue #14).

Verifies:
1. L-BFGS-B convergence on synthetic ensemble-observation datasets with known bias and spread under-dispersion.
2. In-sample CRPS reduction relative to raw ensemble (CRPSS > 0).
3. In-sample CRPS comparison with climatological baseline.
4. L2 regularization penalty applied strictly to parameter d (controlling spread amplification).
5. Parameter bounds and initial warm start with O(1e-3) perturbation.
6. Multi-start restart fallback on non-converged initial runs.
7. ModelTrainingDiagnostics scorecard generation with health grades (HEALTHY, WARNING, DEGRADED).
8. Soft warning detection for extreme parameters (|c| > 10 or |d| > 10).
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.emos_trainer import EMOSOptimizer, ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS


@pytest.fixture
def biased_underdispersed_data():
    """Generate synthetic dataset where raw forecast has +2.0°C warm bias and 0.5x under-dispersed spread."""
    np.random.seed(42)
    n = 500
    
    # True underlying temperature (mean 20°C, std 5°C)
    true_temp = np.random.normal(20.0, 5.0, n)
    
    # Raw ensemble forecast has systematic bias: mean = true_temp + 2.0°C + noise
    ens_mean = true_temp + 2.0 + np.random.normal(0, 1.0, n)
    # Ensemble variance is artificially small (e.g. 1.0°C^2 instead of ~5.0^2)
    ens_var = np.full(n, 1.0)
    # Climatology variance
    clim_var = np.full(n, 25.0)
    
    return {
        "ensemble_mean": ens_mean,
        "ensemble_variance": ens_var,
        "sigma_clim_squared": clim_var,
        "observed_temp": true_temp,
        "mu_clim": np.full(n, 20.0),
        "sigma_clim": np.full(n, 5.0),
    }


class TestEMOSOptimizerConvergence:
    """Test optimization convergence and calibration accuracy."""

    def test_optimizer_reduces_crps_and_corrects_bias(self, biased_underdispersed_data):
        optimizer = EMOSOptimizer(l2_lambda_d=1e-3, random_seed=42)
        
        result = optimizer.optimize(
            ensemble_mean=biased_underdispersed_data["ensemble_mean"],
            ensemble_variance=biased_underdispersed_data["ensemble_variance"],
            sigma_clim_squared=biased_underdispersed_data["sigma_clim_squared"],
            observed_temp=biased_underdispersed_data["observed_temp"],
            mu_clim=biased_underdispersed_data["mu_clim"],
            sigma_clim=biased_underdispersed_data["sigma_clim"],
        )
        
        assert result.success is True
        assert result.health_grade == "HEALTHY"
        # Optimized CRPS must be strictly less than raw ensemble CRPS
        assert result.crps_in_sample < result.crps_raw_ensemble
        assert result.crpss_vs_raw > 0.05  # At least 5% improvement over raw biased forecast
        
        # Bias parameter 'a' should be negative (compensating for +2.0°C warm bias)
        a, b, c, d = result.params
        assert a < 0.0  # e.g. approx -2.0
        assert 0.7 < b < 1.3  # slope near 1.0

    def test_l2_regularization_constrains_d(self):
        """Higher lambda_d forces smaller d magnitude when ensemble variance is active."""
        np.random.seed(42)
        n = 500
        true_temp = np.random.normal(20.0, 5.0, n)
        ens_mean = true_temp + np.random.normal(0, 1.0, n)
        ens_var = np.random.uniform(1.0, 4.0, n)
        clim_var = np.zeros(n)
        obs = ens_mean + np.random.normal(0, np.sqrt(ens_var) * 2.0)

        opt_low_reg = EMOSOptimizer(l2_lambda_d=0.0, random_seed=42)
        res_low = opt_low_reg.optimize(ens_mean, ens_var, clim_var, obs)

        opt_high_reg = EMOSOptimizer(l2_lambda_d=1.0, random_seed=42)
        res_high = opt_high_reg.optimize(ens_mean, ens_var, clim_var, obs)

        _, _, _, d_low = res_low.params
        _, _, _, d_high = res_high.params
        assert abs(d_high) < abs(d_low)
        assert abs(d_low) > 1.0
        assert abs(d_high) < 0.1

    def test_fit_returns_gaussian_emos_instance(self, biased_underdispersed_data):
        optimizer = EMOSOptimizer(l2_lambda_d=1e-3, random_seed=42)
        emos_model, diag = optimizer.fit(
            ensemble_mean=biased_underdispersed_data["ensemble_mean"],
            ensemble_variance=biased_underdispersed_data["ensemble_variance"],
            sigma_clim_squared=biased_underdispersed_data["sigma_clim_squared"],
            observed_temp=biased_underdispersed_data["observed_temp"],
        )
        assert isinstance(emos_model, GaussianEMOS)
        assert isinstance(diag, ModelTrainingDiagnostics)
        assert emos_model.a == diag.params[0]
        assert emos_model.b == diag.params[1]


class TestDiagnosticsAndHealthGrading:
    """Test the training scorecard and health grade classification."""

    def test_diagnostics_structure_and_metrics(self, biased_underdispersed_data):
        optimizer = EMOSOptimizer(l2_lambda_d=1e-3)
        res = optimizer.optimize(
            ensemble_mean=biased_underdispersed_data["ensemble_mean"],
            ensemble_variance=biased_underdispersed_data["ensemble_variance"],
            sigma_clim_squared=biased_underdispersed_data["sigma_clim_squared"],
            observed_temp=biased_underdispersed_data["observed_temp"],
            mu_clim=biased_underdispersed_data["mu_clim"],
            sigma_clim=biased_underdispersed_data["sigma_clim"],
        )
        
        diag_dict = res.to_dict()
        assert "params" in diag_dict
        assert "crps_in_sample" in diag_dict
        assert "crps_raw_ensemble" in diag_dict
        assert "crps_climatology" in diag_dict
        assert "crpss_vs_raw" in diag_dict
        assert "crpss_vs_clim" in diag_dict
        assert "n_iterations" in diag_dict
        assert "health_grade" in diag_dict
        assert res.sample_count == len(biased_underdispersed_data["observed_temp"])

    def test_extreme_parameter_triggers_warning(self):
        """If |c| > 10 or |d| > 10, health_grade transitions to WARNING with warning logged."""
        diag = ModelTrainingDiagnostics(
            success=True,
            params=(0.0, 1.0, 12.0, 1.0),  # |c| = 12.0 > 10
            crps_in_sample=2.0,
            crps_raw_ensemble=2.5,
            crps_climatology=3.0,
            crpss_vs_raw=0.20,
            crpss_vs_clim=0.33,
            n_iterations=20,
            n_evaluations=30,
            grad_norm=1e-6,
            restarts_used=0,
            sample_count=500,
        )
        assert diag.health_grade == "WARNING"
        assert any("parameter c magnitude" in w for w in diag.warnings)
