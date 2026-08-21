#!/usr/bin/env python3
"""
Unit tests for MetricsCalculator (Ticket 4.1-01 / Issue #33).

Verifies:
- Continuous probabilistic metrics (CRPS, Mean CRPS, CRPSS, MAE, RMSE, Bias, 90% CI Coverage)
- Distribution calibration & reliability metrics (PIT values, PIT Histogram, Talagrand Rank Histogram, Spread-Error Ratio)
- Discrete Polymarket bin metrics (Binary & Multi-class Brier Score, BSS, Multi-class Log Loss, ECE, Reliability Diagram)
- Edge cases and numerical stability protections (zero variance floor, boundary probability log clipping)
"""

import math
import numpy as np
import pytest
from scipy import stats

from src.validation.metrics_calculator import MetricsCalculator, ReliabilityDiagramData


class TestContinuousProbabilisticMetrics:
    """Tests for continuous distribution verification metrics."""

    def test_crps_scalar_and_vectorized(self):
        calc = MetricsCalculator()

        # Zero error, near-zero sigma -> CRPS = 0
        crps_zero = calc.crps_gaussian(y=20.0, mu=20.0, sigma=1e-8)
        assert crps_zero == pytest.approx(0.0, abs=1e-6)

        # Standard test point: y = 20, mu = 20, sigma = 2
        # CRPS(20, 20, 2) = sigma * (sqrt(2) - 1) / sqrt(pi) = 2 * (sqrt(2) - 1) / sqrt(pi)
        crps_val = calc.crps_gaussian(y=20.0, mu=20.0, sigma=2.0)
        expected_crps = 2.0 * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
        assert crps_val == pytest.approx(expected_crps, rel=1e-4)

        # Vectorized input
        y_arr = np.array([20.0, 25.0, 15.0])
        mu_arr = np.array([20.0, 24.0, 16.0])
        sigma_arr = np.array([2.0, 1.5, 2.5])

        crps_arr = calc.crps_gaussian(y=y_arr, mu=mu_arr, sigma=sigma_arr)
        assert isinstance(crps_arr, np.ndarray)
        assert len(crps_arr) == 3
        assert np.all(crps_arr >= 0.0)

    def test_mean_crps_and_crpss(self):
        calc = MetricsCalculator()
        y_arr = np.array([10.0, 15.0, 20.0, 25.0])
        mu_model = np.array([10.5, 14.8, 19.5, 25.2])
        sigma_model = np.array([1.2, 1.0, 1.5, 1.1])

        # Climatology reference with larger spread and bias
        mu_ref = np.array([15.0, 15.0, 15.0, 15.0])
        sigma_ref = np.array([5.0, 5.0, 5.0, 5.0])

        mean_crps_model = calc.mean_crps(y_arr, mu_model, sigma_model)
        mean_crps_ref = calc.mean_crps(y_arr, mu_ref, sigma_ref)

        assert mean_crps_model < mean_crps_ref

        # Skill score: 1 - CRPS_model / CRPS_ref
        crpss = calc.crpss(mean_crps_model, mean_crps_ref)
        assert 0.0 < crpss < 1.0
        assert crpss == pytest.approx(1.0 - (mean_crps_model / mean_crps_ref), rel=1e-5)

        # Reference CRPS = 0 handling
        assert calc.crpss(0.0, 0.0) == 0.0
        assert calc.crpss(1.5, 0.0) == -math.inf

    def test_mae_rmse_bias_and_ci_coverage(self):
        calc = MetricsCalculator()
        y = np.array([10.0, 20.0, 30.0, 40.0])
        mu = np.array([12.0, 19.0, 31.0, 42.0])  # errors: +2, -1, +1, +2
        sigma = np.array([2.0, 2.0, 2.0, 2.0])

        assert calc.mae(y, mu) == pytest.approx(1.5)  # (2 + 1 + 1 + 2) / 4 = 1.5
        assert calc.rmse(y, mu) == pytest.approx(math.sqrt((4 + 1 + 1 + 4) / 4))  # sqrt(2.5) = 1.5811
        assert calc.bias(y, mu) == pytest.approx(1.0)  # (2 - 1 + 1 + 2) / 4 = 1.0

        # 90% CI: [mu - 1.64485*sigma, mu + 1.64485*sigma]
        # with sigma=2, half width = 3.2897. All errors within 3.2897 -> 100% coverage
        coverage = calc.coverage_confidence_interval(y, mu, sigma, confidence_level=0.90)
        assert coverage == pytest.approx(1.0)

        # One point far outside
        y_far = np.array([10.0, 20.0, 30.0, 100.0])  # error 58
        coverage_partial = calc.coverage_confidence_interval(y_far, mu, sigma, confidence_level=0.90)
        assert coverage_partial == pytest.approx(0.75)


class TestCalibrationMetrics:
    """Tests for distribution calibration and spread reliability."""

    def test_pit_values_and_histogram(self):
        calc = MetricsCalculator()
        np.random.seed(42)

        # Perfect calibration: y drawn from N(mu, sigma)
        n_samples = 1000
        mu = np.random.uniform(10, 30, size=n_samples)
        sigma = np.random.uniform(1, 3, size=n_samples)
        y = np.random.normal(mu, sigma)

        pit = calc.compute_pit_values(y, mu, sigma)
        assert len(pit) == n_samples
        assert np.all(pit >= 0.0) and np.all(pit <= 1.0)

        # Uniformity test: KS p-value should be > 0.01 for true normal samples
        ks_stat, ks_p = stats.kstest(pit, "uniform")
        assert ks_p > 0.01

        # PIT Histogram
        counts, rel_freqs, bin_edges = calc.pit_histogram(pit, num_bins=10)
        assert len(counts) == 10
        assert sum(counts) == n_samples
        assert pytest.approx(sum(rel_freqs), rel=1e-5) == 1.0
        assert len(bin_edges) == 11

    def test_talagrand_rank_histogram(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        n_samples = 500
        n_members = 5

        # 5 ensemble members + observation from same normal distribution
        ensemble = np.random.normal(loc=20.0, scale=3.0, size=(n_samples, n_members))
        y = np.random.normal(loc=20.0, scale=3.0, size=n_samples)

        ranks, rank_counts, rel_freqs = calc.talagrand_rank_histogram(y, ensemble)
        assert len(ranks) == n_samples
        assert np.all(ranks >= 0) and np.all(ranks <= n_members)
        assert len(rank_counts) == n_members + 1
        assert sum(rank_counts) == n_samples
        assert pytest.approx(sum(rel_freqs), rel=1e-5) == 1.0

    def test_spread_error_ratio(self):
        calc = MetricsCalculator()
        y = np.array([10.0, 20.0, 30.0, 40.0])
        mu = np.array([10.0, 20.0, 30.0, 40.0])
        sigma = np.array([2.0, 2.0, 2.0, 2.0])

        # When RMSE is 0
        assert calc.spread_error_ratio(y, mu, sigma) == math.inf

        # When RMSE = 2.0 and mean_spread = 2.0 -> ratio = 1.0
        mu_with_err = np.array([12.0, 18.0, 32.0, 38.0])  # errors: 2, -2, 2, -2 -> RMSE = 2.0
        ratio = calc.spread_error_ratio(y, mu_with_err, sigma)
        assert ratio == pytest.approx(1.0, rel=1e-4)


class TestDiscretePolymarketBinMetrics:
    """Tests for discrete Polymarket bin probability verification."""

    def test_brier_score_binary_and_skill_score(self):
        calc = MetricsCalculator()
        # Perfect forecasts
        obs_binary = np.array([1, 0, 1, 0])
        prob_perfect = np.array([1.0, 0.0, 1.0, 0.0])
        assert calc.brier_score(obs_binary, prob_perfect) == pytest.approx(0.0)

        # Complete wrong
        prob_wrong = np.array([0.0, 1.0, 0.0, 1.0])
        assert calc.brier_score(obs_binary, prob_wrong) == pytest.approx(1.0)

        # Realistic probabilities
        prob_model = np.array([0.8, 0.2, 0.7, 0.1])
        prob_clim = np.array([0.5, 0.5, 0.5, 0.5])

        bs_model = calc.brier_score(obs_binary, prob_model)
        bs_clim = calc.brier_score(obs_binary, prob_clim)
        assert bs_model < bs_clim

        bss = calc.brier_skill_score(bs_model, bs_clim)
        assert bss > 0.0
        assert bss == pytest.approx(1.0 - (bs_model / bs_clim), rel=1e-5)

    def test_multiclass_brier_score(self):
        calc = MetricsCalculator()
        # 3 classes / bins, 2 samples
        obs_one_hot = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ])
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
        ])
        # sample 1: (0.8-1)^2 + (0.1-0)^2 + (0.1-0)^2 = 0.04 + 0.01 + 0.01 = 0.06
        # sample 2: (0.2-0)^2 + (0.7-1)^2 + (0.1-0)^2 = 0.04 + 0.09 + 0.01 = 0.14
        # mean multi-class BS = (0.06 + 0.14) / 2 = 0.10
        bs_multi = calc.brier_score_multiclass(obs_one_hot, probs)
        assert bs_multi == pytest.approx(0.10, rel=1e-5)

    def test_multiclass_log_loss_cross_entropy(self):
        calc = MetricsCalculator()
        obs_one_hot = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ])
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
        ])
        loss = calc.multiclass_log_loss(obs_one_hot, probs)
        expected_loss = -(math.log(0.8) + math.log(0.7)) / 2.0
        assert loss == pytest.approx(expected_loss, rel=1e-4)

        # Boundary safety: prob 0.0 does not raise NaN/Inf
        probs_extreme = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        loss_extreme = calc.multiclass_log_loss(obs_one_hot, probs_extreme)
        assert not np.isnan(loss_extreme)
        assert not np.isinf(loss_extreme)
        assert loss_extreme > 10.0

    def test_ece_and_reliability_diagram(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        n = 1000
        probs = np.random.uniform(0.0, 1.0, size=n)
        outcomes = (np.random.uniform(0.0, 1.0, size=n) < probs).astype(int)

        rel_data = calc.reliability_diagram(outcomes, probs, num_bins=5)
        assert isinstance(rel_data, ReliabilityDiagramData)
        assert len(rel_data.bin_centers) == 5
        assert len(rel_data.bin_accuracies) == 5
        assert len(rel_data.bin_confidences) == 5
        assert sum(rel_data.bin_counts) == n

        ece = calc.expected_calibration_error(outcomes, probs, num_bins=5)
        assert ece >= 0.0
        assert ece < 0.10
