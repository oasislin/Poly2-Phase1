#!/usr/bin/env python3
"""
Unit tests for Two-Level Degradation and Soft Warnings (Ticket 2.2-02 / Issue #15).

Verifies:
1. Level 1: Normal Gaussian EMOS + variance floor active.
2. Hard Trigger: Unconverged optimizer, NaN/Inf parameters, or insufficient sample count forces Level 2 (Climatology).
3. Soft Trigger: When EMOS CRPS is statistically significantly worse than Climatology (p < 0.05), forces Level 2.
4. Soft Trigger: When EMOS CRPS is slightly worse but not statistically significant (p >= 0.05), remains Level 1.
5. Soft Warning: Extreme parameters |c| > 10 or |d| > 10 trigger WARNING log without forcing Level 2 degradation.
6. Distribution routing: DegradationHandler seamlessly produces correct GaussianEMOS instance for Level 1 vs Level 2.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.degradation import DegradationHandler, DegradationDecision
from src.modeling.emos_trainer import ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS


class TestHardDegradationTriggers:
    """Test deterministic hard triggers forcing Level 2 climatology fallback."""

    def test_unconverged_optimizer_triggers_hard_degradation(self):
        handler = DegradationHandler()
        diag = ModelTrainingDiagnostics(
            success=False,  # Optimization failed
            params=(0.0, 1.0, 0.0, 1.0),
            crps_in_sample=5.0,
            crps_raw_ensemble=5.0,
            crps_climatology=4.0,
            crpss_vs_raw=0.0,
            crpss_vs_clim=-0.25,
            n_iterations=1000,
            n_evaluations=2000,
            grad_norm=1.5,
            restarts_used=3,
            sample_count=100,
        )

        decision = handler.evaluate(diag)
        assert decision.level == 2
        assert decision.is_degraded is True
        assert "Optimizer failed to converge" in decision.reason

    def test_nan_or_inf_parameters_trigger_hard_degradation(self):
        handler = DegradationHandler()
        diag = ModelTrainingDiagnostics(
            success=True,
            params=(np.nan, 1.0, 0.0, np.inf),
            crps_in_sample=np.nan,
            crps_raw_ensemble=2.0,
            crps_climatology=2.5,
            crpss_vs_raw=0.0,
            crpss_vs_clim=0.0,
            n_iterations=50,
            n_evaluations=100,
            grad_norm=0.0,
            restarts_used=0,
            sample_count=100,
        )

        decision = handler.evaluate(diag)
        assert decision.level == 2
        assert decision.is_degraded is True
        assert "Non-finite parameter" in decision.reason

    def test_insufficient_sample_count_triggers_hard_degradation(self):
        handler = DegradationHandler(min_sample_count=10)
        diag = ModelTrainingDiagnostics(
            success=True,
            params=(0.0, 1.0, 0.0, 1.0),
            crps_in_sample=1.5,
            crps_raw_ensemble=2.0,
            crps_climatology=2.5,
            crpss_vs_raw=0.25,
            crpss_vs_clim=0.40,
            n_iterations=20,
            n_evaluations=30,
            grad_norm=1e-6,
            restarts_used=0,
            sample_count=5,  # Too few samples
        )

        decision = handler.evaluate(diag)
        assert decision.level == 2
        assert decision.is_degraded is True
        assert "Insufficient sample count" in decision.reason


class TestSoftDegradationTriggers:
    """Test statistical soft triggers using paired t-test between EMOS and Climatology CRPS."""

    def test_significantly_inferior_crps_triggers_soft_degradation(self):
        handler = DegradationHandler(alpha=0.05)
        np.random.seed(42)
        n = 100
        
        # EMOS CRPS is consistently higher (worse) than Climatology CRPS
        clim_crps = np.random.normal(2.0, 0.5, n)
        emos_crps = clim_crps + np.random.normal(0.4, 0.1, n)  # Statistically worse by +0.4°C

        diag = ModelTrainingDiagnostics(
            success=True,
            params=(0.0, 1.0, 0.0, 1.0),
            crps_in_sample=float(np.mean(emos_crps)),
            crps_raw_ensemble=3.0,
            crps_climatology=float(np.mean(clim_crps)),
            crpss_vs_raw=0.20,
            crpss_vs_clim=float(1.0 - np.mean(emos_crps) / np.mean(clim_crps)),
            n_iterations=25,
            n_evaluations=40,
            grad_norm=1e-6,
            restarts_used=0,
            sample_count=n,
        )

        decision = handler.evaluate(
            diag,
            emos_sample_crps=emos_crps,
            clim_sample_crps=clim_crps,
        )
        assert decision.level == 2
        assert decision.is_degraded is True
        assert "statistically significantly inferior to climatology" in decision.reason
        assert decision.p_value < 0.05

    def test_not_significantly_worse_remains_level_1(self):
        handler = DegradationHandler(alpha=0.05)
        np.random.seed(42)
        n = 100
        
        # EMOS and Climatology are almost identical with zero mean difference + noise
        clim_crps = np.random.normal(2.0, 0.5, n)
        emos_crps = clim_crps + np.random.normal(0.01, 0.5, n)  # Diff is +0.01 (not significant)

        diag = ModelTrainingDiagnostics(
            success=True,
            params=(0.0, 1.0, 0.0, 1.0),
            crps_in_sample=float(np.mean(emos_crps)),
            crps_raw_ensemble=2.5,
            crps_climatology=float(np.mean(clim_crps)),
            crpss_vs_raw=0.19,
            crpss_vs_clim=float(1.0 - np.mean(emos_crps) / np.mean(clim_crps)),
            n_iterations=25,
            n_evaluations=40,
            grad_norm=1e-6,
            restarts_used=0,
            sample_count=n,
        )

        decision = handler.evaluate(
            diag,
            emos_sample_crps=emos_crps,
            clim_sample_crps=clim_crps,
        )
        assert decision.level == 1
        assert decision.is_degraded is False


class TestSoftWarningAndRouting:
    """Test soft warning behavior and model prediction routing."""

    def test_extreme_parameter_produces_soft_warning_without_degradation(self):
        handler = DegradationHandler()
        diag = ModelTrainingDiagnostics(
            success=True,
            params=(0.0, 1.0, 12.5, 0.8),  # |c| = 12.5 > 10.0
            crps_in_sample=1.8,
            crps_raw_ensemble=2.5,
            crps_climatology=2.8,
            crpss_vs_raw=0.28,
            crpss_vs_clim=0.35,
            n_iterations=30,
            n_evaluations=45,
            grad_norm=1e-6,
            restarts_used=0,
            sample_count=200,
        )

        decision = handler.evaluate(diag)
        # Should stay at Level 1 (do not force degrade)
        assert decision.level == 1
        assert decision.is_degraded is False
        assert len(decision.warnings) > 0
        assert any("|c|=12.50 > 10" in w for w in decision.warnings)

    def test_route_prediction_level_1_vs_level_2(self):
        handler = DegradationHandler()
        emos_model = GaussianEMOS(a=0.5, b=0.9, c=0.2, d=1.0)
        
        # 1. Route Level 1 (EMOS + variance floor)
        dist_level_1 = handler.get_active_distribution(
            level=1,
            emos_model=emos_model,
            ens_mean=20.0,
            ens_var=4.0,
            mu_clim=18.0,
            sigma_clim=3.0,
        )
        # Expected mu = 0.5 + 0.9 * 20.0 = 18.5
        # Expected sigma^2 = 0.04 + 1.0 * 4.0 + 9.0 = 13.04 -> sigma = sqrt(13.04)
        assert np.isclose(dist_level_1.mu, 18.5)
        assert np.isclose(dist_level_1.sigma, np.sqrt(13.04))

        # 2. Route Level 2 (Climatology fallback)
        dist_level_2 = handler.get_active_distribution(
            level=2,
            emos_model=emos_model,
            ens_mean=20.0,
            ens_var=4.0,
            mu_clim=18.0,
            sigma_clim=3.0,
        )
        # Directly output N(mu_clim, sigma_clim^2)
        assert np.isclose(dist_level_2.mu, 18.0)
        assert np.isclose(dist_level_2.sigma, 3.0)
