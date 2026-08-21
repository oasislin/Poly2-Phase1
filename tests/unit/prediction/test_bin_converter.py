#!/usr/bin/env python3
"""
Unit tests for BinConverter and MarketBin (Ticket 3.4-01 / Issue #29).

Verifies:
1. Single value bin ('=T') integration with half-degree continuity correction:
   P = F(T + 0.5) - F(T - 0.5).
2. Range bin ('T1-T2') integration:
   P = F(T2 + 0.5) - F(T1 - 0.5).
3. Lower/Upper boundary bins ('<=T' and '>=T') integration:
   P(<=T) = F(T + 0.5), P(>=T) = 1 - F(T - 0.5).
4. Unit conversion from Fahrenheit (Denver KDEN) to Celsius continuous CDF:
   T_C = (T_F +/- 0.5 - 32) * 5/9, seamless integration without discretization drift.
5. Probability Simplex guarantee:
   Sum of all mutually exclusive and collectively exhaustive bin probabilities == 1.000000.
6. Winning bin settlement resolution:
   Given Wunderground observed truth y, uniquely and accurately selects the winning bin (YES=1, others NO=0).
7. Adaptive bin generation across Shanghai (1°C steps) and Denver (1°F steps).
8. Serialization to dictionary and pandas DataFrame.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.modeling.gaussian_emos import GaussianEMOS
from src.prediction.bin_converter import BinConverter, MarketBin


@pytest.fixture
def base_emos():
    """Gaussian EMOS distribution: mu = 20.0°C, sigma = 2.0°C."""
    return GaussianEMOS.from_params(mu=20.0, sigma=2.0)


@pytest.fixture
def converter():
    return BinConverter()


class TestSingleBinProbabilityIntegration:
    """Test mathematical exactness of individual bin types in Celsius."""

    def test_exact_single_bin_continuity_correction(self, base_emos, converter):
        """Single bin '=20°C' integrates F(20.5) - F(19.5)."""
        # mu = 20.0, sigma = 2.0
        # F(20.5) = stats.norm.cdf(20.5, 20.0, 2.0)
        # F(19.5) = stats.norm.cdf(19.5, 20.0, 2.0)
        bin_20 = MarketBin(bin_id="b20", bin_type="exact", label="20°C", unit="C", value=20.0)
        res_bins = converter.calculate_bin_probabilities(base_emos, [bin_20], normalize=False)

        expected_p = stats.norm.cdf(20.5, 20.0, 2.0) - stats.norm.cdf(19.5, 20.0, 2.0)
        assert np.isclose(res_bins[0].probability, expected_p, atol=1e-6)

    def test_range_bin_integration(self, base_emos, converter):
        """Range bin '20-22°C' integrates F(22.5) - F(19.5)."""
        bin_range = MarketBin(bin_id="b_rng", bin_type="range", label="20-22°C", unit="C", low=20.0, high=22.0)
        res_bins = converter.calculate_bin_probabilities(base_emos, [bin_range], normalize=False)

        expected_p = stats.norm.cdf(22.5, 20.0, 2.0) - stats.norm.cdf(19.5, 20.0, 2.0)
        assert np.isclose(res_bins[0].probability, expected_p, atol=1e-6)

    def test_lower_boundary_bin(self, base_emos, converter):
        """Lower boundary bin '<=18°C' integrates from -inf to 18.5°C."""
        bin_lte = MarketBin(bin_id="b_lte", bin_type="lte", label="≤18°C", unit="C", value=18.0)
        res_bins = converter.calculate_bin_probabilities(base_emos, [bin_lte], normalize=False)

        expected_p = stats.norm.cdf(18.5, 20.0, 2.0)
        assert np.isclose(res_bins[0].probability, expected_p, atol=1e-6)

    def test_upper_boundary_bin(self, base_emos, converter):
        """Upper boundary bin '>=22°C' integrates from 21.5°C to +inf."""
        bin_gte = MarketBin(bin_id="b_gte", bin_type="gte", label="≥22°C", unit="C", value=22.0)
        res_bins = converter.calculate_bin_probabilities(base_emos, [bin_gte], normalize=False)

        expected_p = 1.0 - stats.norm.cdf(21.5, 20.0, 2.0)
        assert np.isclose(res_bins[0].probability, expected_p, atol=1e-6)


class TestFahrenheitUnitMapping:
    """Test Denver (KDEN) Fahrenheit market bin mapping to continuous Celsius CDF."""

    def test_fahrenheit_exact_bin_bounds_conversion(self, converter):
        """Fahrenheit bin '68°F' converts bounds [67.5°F, 68.5°F] to Celsius."""
        # 67.5°F = (67.5 - 32) * 5/9 = 35.5 * 5/9 ≈ 19.72222°C
        # 68.5°F = (68.5 - 32) * 5/9 = 36.5 * 5/9 ≈ 20.27778°C
        f_bin = MarketBin(bin_id="f68", bin_type="exact", label="68°F", unit="F", value=68.0)
        low_c, high_c = f_bin.get_celsius_bounds()

        assert np.isclose(low_c, (67.5 - 32.0) * 5.0 / 9.0, atol=1e-6)
        assert np.isclose(high_c, (68.5 - 32.0) * 5.0 / 9.0, atol=1e-6)

    def test_fahrenheit_market_integration(self, base_emos, converter):
        """Integrating a complete Fahrenheit market set sums to 1.0."""
        # Denver temperature market around 68°F (20°C)
        f_bins = [
            MarketBin(bin_id="f_le65", bin_type="lte", label="≤65°F", unit="F", value=65.0),
            MarketBin(bin_id="f66", bin_type="exact", label="66°F", unit="F", value=66.0),
            MarketBin(bin_id="f67", bin_type="exact", label="67°F", unit="F", value=67.0),
            MarketBin(bin_id="f68", bin_type="exact", label="68°F", unit="F", value=68.0),
            MarketBin(bin_id="f69", bin_type="exact", label="69°F", unit="F", value=69.0),
            MarketBin(bin_id="f70", bin_type="exact", label="70°F", unit="F", value=70.0),
            MarketBin(bin_id="f_ge71", bin_type="gte", label="≥71°F", unit="F", value=71.0),
        ]

        evaluated_bins = converter.calculate_bin_probabilities(base_emos, f_bins, normalize=True)
        total_p = sum(b.probability for b in evaluated_bins)

        assert np.isclose(total_p, 1.0, atol=1e-6)
        assert all(b.probability >= 0.0 for b in evaluated_bins)


class TestProbabilitySimplexAndAdaptiveGeneration:
    """Test full market sets and probability normalization."""

    def test_shanghai_standard_market_sums_to_one(self, base_emos, converter):
        """Shanghai standard 1°C bins [<=17, 18, 19, 20, 21, 22, >=23] sum to 1.0."""
        bins = converter.generate_bins(
            station_id="ZSPD",
            center_temp=20.0,
            spread=2.0,
            bin_width=1.0,
            num_bins=7,
        )

        assert len(bins) == 7
        assert bins[0].bin_type == "lte"
        assert bins[-1].bin_type == "gte"

        evaluated = converter.calculate_bin_probabilities(base_emos, bins, normalize=True)
        probs = [b.probability for b in evaluated]

        assert np.isclose(sum(probs), 1.0, atol=1e-7)
        # Peak at center bin (20°C)
        assert evaluated[3].label == "20°C"
        assert probs[3] == max(probs)

    def test_winning_bin_settlement_determination_celsius(self, converter):
        """Wunderground truth 20.3°C should resolve bin '20°C' as winner (YES=1)."""
        bins = [
            MarketBin(bin_id="b18", bin_type="lte", label="≤18°C", unit="C", value=18.0),
            MarketBin(bin_id="b19", bin_type="exact", label="19°C", unit="C", value=19.0),
            MarketBin(bin_id="b20", bin_type="exact", label="20°C", unit="C", value=20.0),
            MarketBin(bin_id="b21", bin_type="exact", label="21°C", unit="C", value=21.0),
            MarketBin(bin_id="b22", bin_type="gte", label="≥22°C", unit="C", value=22.0),
        ]

        # Case 1: 20.3°C in [19.5, 20.5) -> '20°C' wins
        win_idx, win_bin = converter.determine_winning_bin(bins, observed_temp=20.3, unit="C")
        assert win_idx == 2
        assert win_bin.label == "20°C"

        # Case 2: 17.5°C in (-inf, 18.5) -> '≤18°C' wins
        win_idx, win_bin = converter.determine_winning_bin(bins, observed_temp=17.5, unit="C")
        assert win_idx == 0
        assert win_bin.label == "≤18°C"

        # Case 3: 25.0°C in [21.5, +inf) -> '≥22°C' wins
        win_idx, win_bin = converter.determine_winning_bin(bins, observed_temp=25.0, unit="C")
        assert win_idx == 4
        assert win_bin.label == "≥22°C"

    def test_winning_bin_settlement_determination_fahrenheit(self, converter):
        """Wunderground Denver truth 68.2°F should resolve '68°F' bin as winner."""
        f_bins = [
            MarketBin(bin_id="f_le65", bin_type="lte", label="≤65°F", unit="F", value=65.0),
            MarketBin(bin_id="f66", bin_type="exact", label="66°F", unit="F", value=66.0),
            MarketBin(bin_id="f67", bin_type="exact", label="67°F", unit="F", value=67.0),
            MarketBin(bin_id="f68", bin_type="exact", label="68°F", unit="F", value=68.0),
            MarketBin(bin_id="f_ge69", bin_type="gte", label="≥69°F", unit="F", value=69.0),
        ]

        # Observed 68.2°F (or equivalent Celsius 20.11°C)
        win_idx, win_bin = converter.determine_winning_bin(f_bins, observed_temp=68.2, unit="F")
        assert win_idx == 3
        assert win_bin.label == "68°F"

    def test_to_dataframe_conversion(self, base_emos, converter):
        """Export market bins to structured pandas DataFrame."""
        bins = converter.generate_bins("ZSPD", center_temp=20.0)
        evaluated = converter.calculate_bin_probabilities(base_emos, bins)
        df = converter.to_dataframe(evaluated)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(bins)
        assert "bin_id" in df.columns
        assert "label" in df.columns
        assert "probability" in df.columns
        assert np.isclose(df["probability"].sum(), 1.0, atol=1e-6)
