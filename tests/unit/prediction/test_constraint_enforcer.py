#!/usr/bin/env python3
"""
Unit tests for ConstraintEnforcer and ConstrainedDistribution (Ticket 3.3-01 / Issue #28).

Verifies:
1. Physical warming/cooling rate limits retrieval by station (ZSPD, KDEN) and season.
2. Reachable temperature range calculation:
   T_max_possible = T_now + r_warm * delta_t
   T_min_possible = T_now - r_cool * delta_t
3. Physical constraint override on Max Temp CDF:
   - For L > T_max_possible: P(X >= L) = 0.0 => F_phys(L) = 1.0
   - For L <= T_max_possible: preserves input distribution.
4. Physical constraint override on Min Temp CDF:
   - For L < T_min_possible: P(X <= L) = 0.0 => F_phys(L) = 0.0
   - For L >= T_min_possible: preserves input distribution.
5. Strict CDF invariants:
   - Monotonicity non-decreasing across full temperature spectrum.
   - Probability bounds in [0.0, 1.0].
6. Hierarchy and priority:
   - Physical constraints strictly override baseline and dynamic models.
7. Zero time remaining (at or past peak diurnal time):
   - delta_t = 0 => T_max_possible = T_now, T_min_possible = T_now.
8. Missing observation or unconstrained mode fallback:
   - When T_now is None, passes through underlying distribution untouched.
"""

from datetime import datetime, time, timezone
import numpy as np
import pytest

from src.modeling.gaussian_emos import GaussianEMOS
from src.prediction.constraint_enforcer import ConstrainedDistribution, ConstraintEnforcer
from src.prediction.dynamic_corrector import DynamicCorrector, TruncatedDistribution


@pytest.fixture
def base_emos():
    """Gaussian EMOS distribution: mu = 28.0°C, sigma = 3.0°C."""
    return GaussianEMOS.from_params(mu=28.0, sigma=3.0)


@pytest.fixture
def enforcer():
    """Default ConstraintEnforcer instance."""
    return ConstraintEnforcer()


class TestRateLimitsAndRangeCalculation:
    """Test physical rate lookup and reachable range formulas."""

    def test_default_rate_limits_shanghai_and_denver(self, enforcer):
        """Verify standard rate limits for Shanghai and Denver across seasons."""
        # Shanghai Summer: warm=3.5°C/h, cool=3.5°C/h
        r_warm_sh, r_cool_sh = enforcer.get_rate_limits(station_id="ZSPD", season="Summer")
        assert r_warm_sh == 3.5
        assert r_cool_sh == 3.5

        # Denver Winter (intense cold fronts): warm=5.0°C/h, cool=9.0°C/h
        r_warm_den, r_cool_den = enforcer.get_rate_limits(station_id="KDEN", season="Winter")
        assert r_warm_den == 5.0
        assert r_cool_den == 9.0

    def test_reachable_range_with_explicit_delta_hours(self, enforcer):
        """Test [T_min_possible, T_max_possible] with delta_hours = 2.0."""
        # ZSPD Summer: r_warm=3.5, r_cool=3.5. Current T_now = 30.0, delta_t = 2.0h
        # T_max_possible = 30.0 + 3.5 * 2.0 = 37.0°C
        # T_min_possible = 30.0 - 3.5 * 2.0 = 23.0°C
        t_min, t_max = enforcer.calculate_reachable_range(
            station_id="ZSPD",
            season="Summer",
            current_temp=30.0,
            delta_hours=2.0,
        )
        assert np.isclose(t_min, 23.0, atol=1e-5)
        assert np.isclose(t_max, 37.0, atol=1e-5)

    def test_reachable_range_from_observation_time(self, enforcer):
        """Observation at 12:00 local time with max temp peak at 15:00 local time (delta_t = 3h)."""
        # T_now = 28.0, 3 hours to 15:00 peak, r_warm = 3.5°C/h
        # T_max_possible = 28.0 + 3.5 * 3 = 38.5°C
        t_min, t_max = enforcer.calculate_reachable_range(
            station_id="ZSPD",
            season="Summer",
            current_temp=28.0,
            target_type="max",
            observation_time=time(12, 0),
        )
        assert np.isclose(t_max, 38.5, atol=1e-5)

    def test_past_peak_zero_delta_hours(self, enforcer):
        """Observation at 16:30 local time is past 15:00 peak -> delta_t = 0.0."""
        # T_now = 32.0. No further warming physically possible today.
        t_min, t_max = enforcer.calculate_reachable_range(
            station_id="ZSPD",
            season="Summer",
            current_temp=32.0,
            target_type="max",
            observation_time=time(16, 30),
        )
        assert np.isclose(t_max, 32.0, atol=1e-5)


class TestMaxTempConstraintEnforcement:
    """Test physical constraint overrides for Maximum Temperature."""

    def test_max_temp_above_reachable_limit_is_zero_prob(self, base_emos, enforcer):
        """For threshold L > T_max_possible, P(X >= L) = 0.0 (CDF = 1.0)."""
        # Base: mu = 28.0, sigma = 3.0
        # T_now = 26.0 at 13:00 (2h to 15:00 peak) -> T_max_possible = 26.0 + 3.5 * 2 = 33.0°C
        # Prior model may give P(X >= 35) > 0, but physics hard-clamps P(X >= 35) = 0 (CDF(35) = 1.0)
        constrained_dist = enforcer.enforce(
            distribution=base_emos,
            station_id="ZSPD",
            season="Summer",
            target_type="max",
            current_temp=26.0,
            delta_hours=2.0,
        )

        # Threshold 35.0 > 33.0 (Unreachable)
        assert np.isclose(constrained_dist.cdf(35.0), 1.0, atol=1e-6)
        assert np.isclose(constrained_dist.probability_greater_than_or_equal(35.0), 0.0, atol=1e-6)

        # Threshold 30.0 < 33.0 (Reachable): preserves underlying CDF
        assert np.isclose(constrained_dist.cdf(30.0), base_emos.cdf(30.0), atol=1e-6)

    def test_max_temp_cdf_strict_monotonicity(self, base_emos, enforcer):
        """Constrained CDF across [20, 45]°C must be strictly non-decreasing."""
        constrained_dist = enforcer.enforce(
            distribution=base_emos,
            station_id="ZSPD",
            season="Summer",
            target_type="max",
            current_temp=26.0,
            delta_hours=2.0,  # T_max_possible = 33.0
        )

        grid = np.linspace(20.0, 45.0, 251)
        cdfs = constrained_dist.cdf(grid)

        assert np.all(cdfs >= 0.0)
        assert np.all(cdfs <= 1.0)
        diffs = np.diff(cdfs)
        assert np.all(diffs >= -1e-12)


class TestMinTempConstraintEnforcement:
    """Test physical constraint overrides for Minimum Temperature."""

    def test_min_temp_below_reachable_limit_is_zero_prob(self, base_emos, enforcer):
        """For threshold L < T_min_possible, P(X <= L) = 0.0 (CDF = 0.0)."""
        # Base: mu = 28.0, sigma = 3.0
        # T_now = 24.0, delta_t = 2.0h, r_cool = 3.5 -> T_min_possible = 24.0 - 7.0 = 17.0°C
        # Threshold L = 14.0 (< 17.0) is physically unreachable -> CDF(14.0) = 0.0
        constrained_dist = enforcer.enforce(
            distribution=base_emos,
            station_id="ZSPD",
            season="Summer",
            target_type="min",
            current_temp=24.0,
            delta_hours=2.0,
        )

        assert np.isclose(constrained_dist.cdf(14.0), 0.0, atol=1e-6)
        assert np.isclose(constrained_dist.probability_less_than_or_equal(14.0), 0.0, atol=1e-6)

        # Threshold L = 22.0 (> 17.0) is reachable -> preserves base CDF
        assert np.isclose(constrained_dist.cdf(22.0), base_emos.cdf(22.0), atol=1e-6)

    def test_min_temp_cdf_strict_monotonicity(self, base_emos, enforcer):
        """Constrained Min Temp CDF must be strictly non-decreasing."""
        constrained_dist = enforcer.enforce(
            distribution=base_emos,
            station_id="ZSPD",
            season="Summer",
            target_type="min",
            current_temp=24.0,
            delta_hours=2.0,
        )

        grid = np.linspace(10.0, 35.0, 251)
        cdfs = constrained_dist.cdf(grid)

        assert np.all(cdfs >= 0.0)
        assert np.all(cdfs <= 1.0)
        diffs = np.diff(cdfs)
        assert np.all(diffs >= -1e-12)


class TestFullThreeLayerStackIntegration:
    """Test composition of Static Base -> Dynamic Corrector -> Physical Constraints."""

    def test_three_layer_stack_composition(self, base_emos, enforcer):
        """Dynamic truncation + Physical upper limit combination."""
        # Base: mu = 28.0, sigma = 3.0
        # Dynamic: T_now = 27.0 => max cannot be < 27.0 (CDF(L <= 27) = 0)
        corrector = DynamicCorrector(current_temperature=27.0)
        truncated_dist = corrector.correct(base_emos, target_type="max")

        # Physical: delta_t = 1.0h, r_warm = 3.5 => T_max_possible = 30.5°C
        # For L > 30.5, CDF(L) = 1.0 (P(X >= L) = 0)
        fully_constrained = enforcer.enforce(
            distribution=truncated_dist,
            station_id="ZSPD",
            season="Summer",
            target_type="max",
            current_temp=27.0,
            delta_hours=1.0,
        )

        # 1. Lower truncation at 27.0°C
        assert np.isclose(fully_constrained.cdf(25.0), 0.0, atol=1e-6)
        assert np.isclose(fully_constrained.cdf(27.0), 0.0, atol=1e-6)

        # 2. Upper physical hard constraint at 30.5°C
        assert np.isclose(fully_constrained.cdf(31.0), 1.0, atol=1e-6)
        assert np.isclose(fully_constrained.cdf(35.0), 1.0, atol=1e-6)

        # 3. Inside active range (27.0, 30.5): strictly increasing
        cdf_28 = fully_constrained.cdf(28.0)
        cdf_29 = fully_constrained.cdf(29.0)
        cdf_30 = fully_constrained.cdf(30.0)
        assert 0.0 < cdf_28 < cdf_29 < cdf_30 < 1.0

    def test_missing_observation_unconstrained_passthrough(self, base_emos, enforcer):
        """When T_now is None, returns identical CDF to input distribution."""
        constrained_dist = enforcer.enforce(
            distribution=base_emos,
            station_id="ZSPD",
            season="Summer",
            target_type="max",
            current_temp=None,
        )
        assert constrained_dist.is_constrained is False
        for x in [22.0, 28.0, 34.0]:
            assert np.isclose(constrained_dist.cdf(x), base_emos.cdf(x), atol=1e-7)

    def test_from_historical_data_factory(self):
        """Test ConstraintEnforcer.from_historical_data calculates empirical limits."""
        import pandas as pd
        # Create synthetic observations with varying diurnal ranges
        df = pd.DataFrame({
            "date": ["2019-07-01", "2019-07-02", "2019-07-03"] * 10,
            "temp_max": [35.0, 36.0, 37.0] * 10,
            "temp_min": [25.0, 26.0, 27.0] * 10,
        })
        custom_enforcer = ConstraintEnforcer.from_historical_data(df, station_id="ZSPD")
        r_warm, r_cool = custom_enforcer.get_rate_limits("ZSPD", "Summer")
        assert r_warm >= 1.0
        assert r_cool >= 1.0

