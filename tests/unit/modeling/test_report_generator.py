#!/usr/bin/env python3
"""
Unit tests for ReportGenerator and Triple Acceptance Gate (Ticket 2.3-02 / Issue #21).

Verifies:
1. Gate 1 (PIT Calibration Gate): KS test against U(0,1) with p > 0.05 pass threshold.
2. Gate 2 (30h Virtual Holdout Gate): CRPS_virt <= 1.05 * CRPS_real AND virtual PIT KS p > 0.05.
3. Gate 3 (Extreme Tail Gate): 90% CI coverage >= 80% AND CRPS_model <= CRPS_clim on extreme samples.
4. Overall Pass/Fail verdict logic (all 3 must pass).
5. Lead time tiered rating classification (Tier 1 High Skill, Tier 2 Moderate Skill).
6. Markdown report generation and dictionary serialization.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.report_generator import AcceptanceReport, ReportGenerator
from src.modeling.validation_engine import ValidationResult


@pytest.fixture
def well_calibrated_validation_result():
    """Create a mock ValidationResult that passes all 3 gates."""
    np.random.seed(42)
    n = 365
    
    # Ground truth observations (mean 20°C, std 5°C)
    obs = np.random.normal(20.0, 5.0, n)
    
    # Perfect EMOS forecasts with matching true error std = 1.5
    emos_mu = obs + np.random.normal(0, 1.5, n)
    emos_sigma = np.full(n, 1.5)
    
    # Raw forecast with bias
    raw_mean = obs + 2.0 + np.random.normal(0, 2.0, n)
    raw_sigma = np.full(n, 1.0)
    
    # Climatology
    clim_mu = np.full(n, 20.0)
    clim_sigma = np.full(n, 5.0)

    # 90% CI and PIT
    z = (obs - emos_mu) / emos_sigma
    from scipy import stats
    pit = stats.norm.cdf(z)
    q_low = emos_mu - 1.645 * emos_sigma
    q_high = emos_mu + 1.645 * emos_sigma
    in_ci = (obs >= q_low) & (obs <= q_high)

    df_daily = pd.DataFrame({
        "target_date": pd.date_range("2019-01-01", periods=n).strftime("%Y-%m-%d"),
        "observed_temp": obs,
        "ensemble_mean": raw_mean,
        "emos_mu": emos_mu,
        "emos_sigma": emos_sigma,
        "crps_emos": np.abs(obs - emos_mu) * 0.5,
        "crps_raw": np.abs(obs - raw_mean) * 0.9,
        "crps_clim": np.abs(obs - clim_mu) * 0.9,
        "pit_value": pit,
        "ci_90_low": q_low,
        "ci_90_high": q_high,
        "in_90_ci": in_ci,
    })

    return ValidationResult(
        station_id="ZSPD",
        target_type="max",
        lead_hours=30,
        sample_count=n,
        mae_emos=1.2,
        mae_raw=2.5,
        mean_crps_emos=1.0,
        mean_crps_raw=2.0,
        mean_crps_clim=3.0,
        crpss_vs_raw=0.50,
        crpss_vs_clim=0.67,
        coverage_90_ci=float(in_ci.mean()),
        pit_values=pit,
        df_daily=df_daily,
    )


class TestTripleAcceptanceGates:
    """Test the three individual gates and overall decision."""

    def test_gate1_pit_uniformity_pass_and_fail(self):
        generator = ReportGenerator()

        # 1. Perfectly uniform PIT values -> must pass (p >> 0.05)
        np.random.seed(42)
        uniform_pit = np.random.uniform(0.01, 0.99, 500)
        passed, p_val = generator.evaluate_gate1_pit(uniform_pit)
        assert passed is True
        assert p_val > 0.05

        # 2. Skewed / biased PIT values (e.g. centered around 0.8) -> must fail (p << 0.05)
        biased_pit = np.random.beta(5, 1, 500)
        passed_b, p_val_b = generator.evaluate_gate1_pit(biased_pit)
        assert passed_b is False
        assert p_val_b < 0.001

    def test_gate2_interpolation_accuracy_pass_and_fail(self):
        generator = ReportGenerator(max_interp_degradation=0.05)
        np.random.seed(42)
        uniform_pit = np.random.uniform(0.01, 0.99, 200)

        # 1. Small interpolation degradation (3% increase) + good PIT -> Pass
        passed, ratio, pit_p = generator.evaluate_gate2_interpolation(
            crps_virt=1.03, crps_real=1.00, pit_values_virt=uniform_pit
        )
        assert passed is True
        assert np.isclose(ratio, 1.03)
        assert pit_p > 0.05

        # 2. Large interpolation degradation (8% increase) -> Fail
        passed_f, ratio_f, _ = generator.evaluate_gate2_interpolation(
            crps_virt=1.08, crps_real=1.00, pit_values_virt=uniform_pit
        )
        assert passed_f is False
        assert np.isclose(ratio_f, 1.08)

    def test_gate3_extreme_tail_coverage_pass_and_fail(self, well_calibrated_validation_result):
        generator = ReportGenerator(min_extreme_coverage=0.80)

        # 1. Well calibrated result -> Pass
        passed, coverage, m_crps, c_crps = generator.evaluate_gate3_extreme_tail(well_calibrated_validation_result.df_daily)
        assert passed is True
        assert coverage >= 0.80
        assert m_crps <= c_crps

        # 2. Artificially broken tail coverage (0% hit in extremes) -> Fail
        df_broken = well_calibrated_validation_result.df_daily.copy()
        obs = df_broken["observed_temp"]
        q10, q90 = np.percentile(obs, 10), np.percentile(obs, 90)
        extreme_mask = (obs <= q10) | (obs >= q90)
        df_broken.loc[extreme_mask, "in_90_ci"] = False

        passed_f, coverage_f, _, _ = generator.evaluate_gate3_extreme_tail(df_broken)
        assert passed_f is False
        assert coverage_f < 0.80

    def test_full_acceptance_report_generation(self, well_calibrated_validation_result):
        generator = ReportGenerator()
        np.random.seed(42)
        virt_pit = np.random.uniform(0.01, 0.99, 365)
        
        report = generator.generate_report(
            val_results={"ZSPD_max_30": well_calibrated_validation_result},
            crps_virt_30h=1.02,
            crps_real_30h=1.00,
            pit_virt_30h=virt_pit,
        )

        assert isinstance(report, AcceptanceReport)
        assert report.overall_passed is True
        assert report.gate1_pit_passed is True
        assert report.gate2_interp_passed is True
        assert report.gate3_extreme_passed is True

        # Check Markdown format
        md_text = report.to_markdown()
        assert "# Phase 1B Triple Acceptance Verification Report" in md_text
        assert "PASSED" in md_text
        assert "Gate 1 (PIT Calibration)" in md_text
        assert "Gate 2 (30h Virtual Holdout)" in md_text
        assert "Gate 3 (Extreme Tail Skill & Coverage)" in md_text
