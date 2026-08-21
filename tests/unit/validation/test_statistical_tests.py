#!/usr/bin/env python3
"""
Unit tests for StatisticalSignificance testing tools (Ticket 4.1-02 / Issue #34).

Verifies:
- Diebold-Mariano (DM) test (squared, absolute, and CRPS loss differentials with HLN 1997 modification)
- Wilcoxon Signed-Rank non-parametric paired test
- Paired Student's t-test with difference confidence intervals
- PIT Kolmogorov-Smirnov goodness-of-fit uniformity calibration test
"""

import numpy as np
import pytest
from scipy import stats

from src.validation.statistical_tests import (
    DieboldMarianoResult,
    StatisticalSignificance,
    diebold_mariano_test,
    paired_t_test,
    pit_ks_test,
    wilcoxon_signed_rank_test,
)


class TestDieboldMarianoTest:
    """Tests for Diebold-Mariano predictive accuracy comparison."""

    def test_identical_forecasts_dm_zero(self):
        np.random.seed(42)
        y = np.random.normal(20, 2, size=100)
        p1 = y + np.random.normal(0, 1, size=100)
        p2 = p1.copy()  # Identical

        res = diebold_mariano_test(y, p1, p2, loss_type="squared")
        assert isinstance(res, DieboldMarianoResult)
        assert res.dm_statistic == pytest.approx(0.0, abs=1e-6)
        assert res.p_value == pytest.approx(1.0, abs=1e-4)
        assert res.is_significant is False

    def test_clearly_superior_forecast_dm_significant(self):
        np.random.seed(42)
        n = 200
        y = np.random.normal(20, 2, size=n)
        p1 = y + np.random.normal(0, 0.5, size=n)  # Small error
        p2 = y + np.random.normal(0, 3.0, size=n)  # Large error

        res = diebold_mariano_test(y, p1, p2, loss_type="squared")
        # Model 1 has significantly smaller loss -> d = L1 - L2 < 0 -> negative DM stat
        assert res.dm_statistic < -3.0
        assert res.p_value < 0.001
        assert res.is_significant is True
        assert res.mean_loss_diff < 0.0

    def test_dm_with_different_loss_types(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10)
        p1 = y + 1.0
        p2 = y + 3.0

        res_sq = diebold_mariano_test(y, p1, p2, loss_type="squared")
        res_abs = diebold_mariano_test(y, p1, p2, loss_type="absolute")

        assert res_sq.mean_loss_diff == pytest.approx(1.0 - 9.0)  # 1^2 - 3^2 = -8
        assert res_abs.mean_loss_diff == pytest.approx(1.0 - 3.0)  # 1 - 3 = -2
        assert res_sq.is_significant is True
        assert res_abs.is_significant is True

    def test_dm_with_crps_loss_arrays(self):
        loss1 = np.array([0.5, 0.4, 0.6, 0.5] * 20)
        loss2 = np.array([1.2, 1.1, 1.3, 1.0] * 20)

        sig = StatisticalSignificance()
        res = sig.diebold_mariano_from_losses(loss1, loss2, h=1)
        assert res.dm_statistic < -5.0
        assert res.p_value < 1e-4
        assert res.is_significant is True


class TestPairedStatisticalTests:
    """Tests for Wilcoxon signed-rank and Paired t-test."""

    def test_wilcoxon_signed_rank_test(self):
        np.random.seed(42)
        n = 100
        loss1 = np.random.uniform(0.5, 1.5, size=n)
        loss2 = loss1 + np.random.uniform(0.1, 0.5, size=n)

        stat, p_val, is_sig = wilcoxon_signed_rank_test(loss1, loss2, alternative="less")
        assert p_val < 1e-4
        assert is_sig is True

    def test_paired_t_test_with_ci(self):
        np.random.seed(42)
        n = 100
        loss1 = np.random.normal(1.0, 0.2, size=n)
        loss2 = np.random.normal(1.5, 0.2, size=n)

        t_stat, p_val, ci_low, ci_high, is_sig = paired_t_test(loss1, loss2, alpha=0.05)
        assert t_stat < -10.0
        assert p_val < 1e-4
        assert is_sig is True
        assert ci_low < ci_high < 0.0  # Mean difference is strictly negative


class TestPITCalibrationTests:
    """Tests for PIT goodness-of-fit and calibration tests."""

    def test_pit_ks_test_uniform_passes(self):
        np.random.seed(42)
        pit_uniform = np.random.uniform(0.0, 1.0, size=500)
        ks_stat, p_val, is_calibrated = pit_ks_test(pit_uniform, alpha=0.05)

        assert is_calibrated is True
        assert p_val > 0.05
        assert ks_stat < 0.10

    def test_pit_ks_test_biased_fails(self):
        np.random.seed(42)
        pit_biased = np.random.beta(0.5, 0.5, size=500)  # U-shaped miscalibrated
        ks_stat, p_val, is_calibrated = pit_ks_test(pit_biased, alpha=0.05)

        assert is_calibrated is False
        assert p_val < 0.001
