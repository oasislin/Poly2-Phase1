#!/usr/bin/env python3
"""
Unit tests for DynamicCorrector and TruncatedDistribution (Ticket 3.2-01 / Issue #27).

Verifies:
1. Mathematical exactness of conditional probability truncation formulas:
   - Max Temp: P(X >= L | X >= T_now) = (1 - F(L)) / (1 - F(T_now)) for L > T_now; 1.0 for L <= T_now.
   - Min Temp: P(X <= L | X <= T_now) = F(L) / F(T_now) for L < T_now; 1.0 for L >= T_now.
2. Posterior CDF mathematical invariants:
   - Strict monotonicity: F_post(L1) <= F_post(L2) for all L1 < L2.
   - Exact range: F_post(L) in [0.0, 1.0].
   - Boundary conditions: F_post(T_now) = 0 for max, F_post(T_now) = 1 for min.
3. Interval probability calculation: P(a <= X <= b) = F_post(b) - F_post(a) >= 0.
4. Quantile inversion consistency: CDF(quantile(p)) == p for p in (0, 1).
5. Missing observation safety: T_now=None/NaN cleanly returns exact Prior CDF.
6. Extreme boundary numerical protection:
   - Max temp with T_now = mu + 6*sigma (denominator -> 0).
   - Min temp with T_now = mu - 6*sigma (denominator -> 0).
7. Vectorized numpy evaluation over 1D arrays of thresholds.
8. State update via update_observation and observation time tracking.
"""

from datetime import datetime, timezone
import numpy as np
import pytest
from scipy import stats

from src.modeling.gaussian_emos import GaussianEMOS
from src.prediction.dynamic_corrector import DynamicCorrector, TruncatedDistribution
from src.prediction.static_predictor import StaticPredictionResult


@pytest.fixture
def base_emos():
    """Base Gaussian EMOS distribution: mu = 25.0°C, sigma = 2.0°C."""
    return GaussianEMOS.from_params(mu=25.0, sigma=2.0)


class TestDynamicCorrectorMaxTemp:
    """Mathematical verification for Maximum Temperature conditional truncation."""

    def test_max_temp_threshold_above_current_observation(self, base_emos):
        """When threshold L > T_now, P(X >= L | X >= T_now) = (1 - F(L)) / (1 - F(T_now))."""
        corrector = DynamicCorrector(current_temperature=24.0)
        
        # Prior F(24.0) and F(26.0)
        # mu=25.0, sigma=2.0
        # z(24.0) = -0.5 -> F(24.0) = stats.norm.cdf(-0.5) ≈ 0.3085375
        # z(26.0) = +0.5 -> F(26.0) = stats.norm.cdf(+0.5) ≈ 0.6914625
        # Expected P(X >= 26 | X >= 24) = (1 - 0.6914625) / (1 - 0.3085375) = 0.3085375 / 0.6914625 ≈ 0.44621
        prob_gte = corrector.correct_max_temp_probability(base_emos, threshold=26.0)
        expected_prob = (1.0 - stats.norm.cdf(26.0, 25.0, 2.0)) / (1.0 - stats.norm.cdf(24.0, 25.0, 2.0))
        assert np.isclose(prob_gte, expected_prob, atol=1e-6)

        # Truncated CDF at 26.0: F_post(26.0) = 1 - P(X >= 26 | X >= 24) ≈ 1 - 0.44621 = 0.55379
        dist = corrector.correct(base_emos, target_type="max")
        assert np.isclose(dist.cdf(26.0), 1.0 - expected_prob, atol=1e-6)

    def test_max_temp_threshold_below_current_observation(self, base_emos):
        """When threshold L <= T_now, today's max is guaranteed >= L (Probability = 1.0, CDF = 0.0)."""
        corrector = DynamicCorrector(current_temperature=24.0)
        
        # For L = 22.0 (< 24.0) and L = 24.0
        prob_gte_22 = corrector.correct_max_temp_probability(base_emos, threshold=22.0)
        prob_gte_24 = corrector.correct_max_temp_probability(base_emos, threshold=24.0)
        assert np.isclose(prob_gte_22, 1.0, atol=1e-6)
        assert np.isclose(prob_gte_24, 1.0, atol=1e-6)

        dist = corrector.correct(base_emos, target_type="max")
        assert np.isclose(dist.cdf(22.0), 0.0, atol=1e-6)
        assert np.isclose(dist.cdf(24.0), 0.0, atol=1e-6)


class TestDynamicCorrectorMinTemp:
    """Mathematical verification for Minimum Temperature conditional truncation."""

    def test_min_temp_threshold_below_current_observation(self, base_emos):
        """When threshold L < T_now, P(X <= L | X <= T_now) = F(L) / F(T_now)."""
        # Base: mu = 25.0, sigma = 2.0
        # Observation T_now = 26.0 (min temp cannot exceed 26.0)
        # Threshold L = 24.0
        # Expected P(X <= 24 | X <= 26) = F(24) / F(26)
        corrector = DynamicCorrector(current_temperature=26.0)
        
        prob_lte = corrector.correct_min_temp_probability(base_emos, threshold=24.0)
        expected_prob = stats.norm.cdf(24.0, 25.0, 2.0) / stats.norm.cdf(26.0, 25.0, 2.0)
        assert np.isclose(prob_lte, expected_prob, atol=1e-6)

        dist = corrector.correct(base_emos, target_type="min")
        assert np.isclose(dist.cdf(24.0), expected_prob, atol=1e-6)

    def test_min_temp_threshold_above_current_observation(self, base_emos):
        """When threshold L >= T_now, today's min is guaranteed <= L (Probability = 1.0, CDF = 1.0)."""
        corrector = DynamicCorrector(current_temperature=26.0)
        
        prob_lte_26 = corrector.correct_min_temp_probability(base_emos, threshold=26.0)
        prob_lte_28 = corrector.correct_min_temp_probability(base_emos, threshold=28.0)
        assert np.isclose(prob_lte_26, 1.0, atol=1e-6)
        assert np.isclose(prob_lte_28, 1.0, atol=1e-6)

        dist = corrector.correct(base_emos, target_type="min")
        assert np.isclose(dist.cdf(26.0), 1.0, atol=1e-6)
        assert np.isclose(dist.cdf(28.0), 1.0, atol=1e-6)


class TestTruncatedDistributionInvariants:
    """Statistical and invariant guarantees for the truncated posterior CDF."""

    def test_max_temp_cdf_strict_monotonicity_and_bounds(self, base_emos):
        """Truncated CDF must be strictly monotonic non-decreasing and bounded in [0, 1]."""
        corrector = DynamicCorrector(current_temperature=23.5)
        dist = corrector.correct(base_emos, target_type="max")

        thresholds = np.linspace(15.0, 35.0, 201)
        cdfs = dist.cdf(thresholds)

        # Bounds
        assert np.all(cdfs >= 0.0)
        assert np.all(cdfs <= 1.0)
        # Monotonicity
        diffs = np.diff(cdfs)
        assert np.all(diffs >= -1e-12)  # Non-decreasing within floating point precision

    def test_min_temp_cdf_strict_monotonicity_and_bounds(self, base_emos):
        """Min temp truncated CDF must be strictly monotonic non-decreasing and bounded in [0, 1]."""
        corrector = DynamicCorrector(current_temperature=27.0)
        dist = corrector.correct(base_emos, target_type="min")

        thresholds = np.linspace(15.0, 35.0, 201)
        cdfs = dist.cdf(thresholds)

        assert np.all(cdfs >= 0.0)
        assert np.all(cdfs <= 1.0)
        diffs = np.diff(cdfs)
        assert np.all(diffs >= -1e-12)

    def test_interval_probability_mass_positivity(self, base_emos):
        """Probability of any valid range [a, b] must be >= 0."""
        corrector = DynamicCorrector(current_temperature=24.0)
        dist = corrector.correct(base_emos, target_type="max")

        p_range = dist.probability_between(25.0, 28.0)
        expected_range = dist.cdf(28.0) - dist.cdf(25.0)
        assert np.isclose(p_range, expected_range, atol=1e-6)
        assert p_range >= 0.0

    def test_quantile_inversion_exactness(self, base_emos):
        """CDF(quantile(p)) == p for probabilities across the distribution."""
        corrector = DynamicCorrector(current_temperature=24.0)
        dist = corrector.correct(base_emos, target_type="max")

        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            q_val = dist.quantile(p)
            cdf_val = dist.cdf(q_val)
            assert np.isclose(cdf_val, p, atol=1e-4)


class TestBoundaryAndDegradationSafety:
    """Robustness against missing data and extreme numerical limits."""

    def test_missing_observation_returns_prior(self, base_emos):
        """When T_now is None or NaN, returns exact prior Gaussian CDF."""
        for missing_val in [None, np.nan]:
            corrector = DynamicCorrector(current_temperature=missing_val)
            dist = corrector.correct(base_emos, target_type="max")
            
            assert dist.is_truncated is False
            for x in [20.0, 25.0, 30.0]:
                assert np.isclose(dist.cdf(x), base_emos.cdf(x), atol=1e-7)

    def test_extreme_high_max_observation_numerical_safety(self, base_emos):
        """T_now far above mean (e.g. +6 sigma = 37.0°C) must not produce NaN or ZeroDivisionError."""
        corrector = DynamicCorrector(current_temperature=37.0)
        dist = corrector.correct(base_emos, target_type="max")

        # Thresholds below 37.0 should have CDF = 0.0
        assert np.isclose(dist.cdf(30.0), 0.0, atol=1e-6)
        assert np.isclose(dist.cdf(37.0), 0.0, atol=1e-6)
        # Threshold far above should approach 1.0 without NaN
        cdf_40 = dist.cdf(40.0)
        assert np.isfinite(cdf_40)
        assert 0.0 <= cdf_40 <= 1.0

    def test_extreme_low_min_observation_numerical_safety(self, base_emos):
        """T_now far below mean (e.g. -6 sigma = 13.0°C) must not produce NaN or ZeroDivisionError."""
        corrector = DynamicCorrector(current_temperature=13.0)
        dist = corrector.correct(base_emos, target_type="min")

        # Thresholds above 13.0 should have CDF = 1.0
        assert np.isclose(dist.cdf(20.0), 1.0, atol=1e-6)
        assert np.isclose(dist.cdf(13.0), 1.0, atol=1e-6)
        cdf_10 = dist.cdf(10.0)
        assert np.isfinite(cdf_10)
        assert 0.0 <= cdf_10 <= 1.0

    def test_update_observation_flow(self, base_emos):
        """Testing real-time observation update stream."""
        corrector = DynamicCorrector()
        dist0 = corrector.correct(base_emos, target_type="max")
        assert dist0.is_truncated is False

        # 08:00 observation: 20.0°C
        t0 = datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc)
        corrector.update_observation(20.0, observation_time=t0)
        dist1 = corrector.correct(base_emos, target_type="max")
        assert dist1.is_truncated is True
        assert dist1.current_temperature == 20.0

        # 12:00 observation: 24.5°C
        t1 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
        corrector.update_observation(24.5, observation_time=t1)
        dist2 = corrector.correct(base_emos, target_type="max")
        assert dist2.current_temperature == 24.5
        assert dist2.observation_time == t1
        # P(X >= 25) at 12:00 should be significantly higher than at 08:00
        assert dist2.probability_greater_than_or_equal(25.0) > dist1.probability_greater_than_or_equal(25.0)
