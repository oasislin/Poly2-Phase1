#!/usr/bin/env python3
"""
Unit tests for TripleGateEvaluator (Ticket 4.1-03 / Issue #35).

Verifies v5.9.2 §5 Triple Acceptance Gates:
- Gate 1: Standard node PIT calibration uniformity (KS p > 0.05).
- Gate 2: 30h Holdout Virtual model interpolation (CRPS_virt <= 1.05 * CRPS_real & PIT p > 0.05).
- Gate 3: Extreme tail stress test (Strictly 2019 OOS: 90% CI coverage >= 80% & CRPS_model < CRPS_clim).
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.triple_gate import (
    GateEvaluationResult,
    TripleGateEvaluator,
    TripleGateReport,
)


class TestTripleGateEvaluator:
    """Tests for individual and overall Triple Acceptance Gates."""

    def test_gate1_pit_calibration_pass_and_fail(self):
        evaluator = TripleGateEvaluator()

        # Calibrated uniform PIT values -> PASS
        np.random.seed(42)
        pit_uniform = np.random.uniform(0.0, 1.0, size=300)
        res_pass = evaluator.evaluate_gate1_pit(pit_uniform)
        assert isinstance(res_pass, GateEvaluationResult)
        assert res_pass.passed is True
        assert res_pass.metrics["p_value"] > 0.05

        # Biased PIT values -> FAIL
        pit_biased = np.array([0.05] * 150 + [0.95] * 150)
        res_fail = evaluator.evaluate_gate1_pit(pit_biased)
        assert res_fail.passed is False
        assert res_fail.metrics["p_value"] < 0.05

    def test_gate2_virtual_interpolation_pass_and_fail(self):
        evaluator = TripleGateEvaluator()
        np.random.seed(42)
        pit_pass = np.random.uniform(0.0, 1.0, size=200)

        # Case 1: CRPS_virt = 1.20, CRPS_real = 1.18 -> Ratio = 1.017 <= 1.05 -> PASS
        res_pass = evaluator.evaluate_gate2_interpolation(
            crps_virt=1.20,
            crps_real=1.18,
            pit_values_virt=pit_pass,
        )
        assert res_pass.passed is True
        assert res_pass.metrics["ratio"] == pytest.approx(1.20 / 1.18, rel=1e-4)

        # Case 2: CRPS_virt = 1.35, CRPS_real = 1.18 -> Ratio = 1.144 > 1.05 -> FAIL
        res_fail_ratio = evaluator.evaluate_gate2_interpolation(
            crps_virt=1.35,
            crps_real=1.18,
            pit_values_virt=pit_pass,
        )
        assert res_fail_ratio.passed is False

        # Case 3: Ratio passes but PIT fails -> FAIL
        pit_fail = np.array([0.02] * 100 + [0.98] * 100)
        res_fail_pit = evaluator.evaluate_gate2_interpolation(
            crps_virt=1.20,
            crps_real=1.18,
            pit_values_virt=pit_fail,
        )
        assert res_fail_pit.passed is False

    def test_gate3_extreme_tail_pass_and_fail(self):
        evaluator = TripleGateEvaluator()

        # Case 1: Coverage = 85% (>= 80%), CRPS_model = 1.40 < CRPS_clim = 2.10 -> PASS
        res_pass = evaluator.evaluate_gate3_extremes(
            extreme_coverage_90=0.85,
            crps_model_extreme=1.40,
            crps_clim_extreme=2.10,
        )
        assert res_pass.passed is True
        assert res_pass.metrics["coverage_90"] == 0.85
        assert res_pass.metrics["skill_diff"] < 0.0

        # Case 2: Coverage = 75% (< 80%) -> FAIL
        res_fail_cov = evaluator.evaluate_gate3_extremes(
            extreme_coverage_90=0.75,
            crps_model_extreme=1.40,
            crps_clim_extreme=2.10,
        )
        assert res_fail_cov.passed is False

        # Case 3: CRPS_model = 2.30 > CRPS_clim = 2.10 -> FAIL
        res_fail_skill = evaluator.evaluate_gate3_extremes(
            extreme_coverage_90=0.88,
            crps_model_extreme=2.30,
            crps_clim_extreme=2.10,
        )
        assert res_fail_skill.passed is False

    def test_overall_triple_gate_report(self):
        evaluator = TripleGateEvaluator()
        np.random.seed(42)
        pit_uniform = np.random.uniform(0.0, 1.0, size=300)

        report = evaluator.generate_triple_gate_report(
            standard_pit_values=pit_uniform,
            crps_virt_30h=1.20,
            crps_real_30h=1.18,
            interp_pit_values=pit_uniform,
            extreme_coverage_90=0.86,
            crps_model_extreme=1.45,
            crps_clim_extreme=2.20,
            station_summaries={"ZSPD": {"mean_crps": 1.15}, "KDEN": {"mean_crps": 1.25}},
        )

        assert isinstance(report, TripleGateReport)
        assert report.overall_passed is True
        assert report.gate1.passed is True
        assert report.gate2.passed is True
        assert report.gate3.passed is True

        md_text = report.to_markdown()
        assert "# Phase 1D Triple Acceptance Verification Report" in md_text
        assert "PASSED" in md_text
        assert "ZSPD" in md_text

    def test_extract_extreme_samples_and_metrics(self):
        evaluator = TripleGateEvaluator()
        np.random.seed(42)

        # 2000-2018 Training: 500 samples with N(20, 5)
        df_train = pd.DataFrame({"truth": np.random.normal(20.0, 5.0, size=500)})

        # 2019 OOS: 100 samples with some normal and extreme points
        n_oos = 100
        truths_oos = np.random.normal(20.0, 5.0, size=n_oos)
        df_oos = pd.DataFrame({
            "truth": truths_oos,
            "mu_model": truths_oos + np.random.normal(0, 0.5, size=n_oos),
            "sigma_model": [1.5] * n_oos,
            "crps_clim": [2.5] * n_oos,
        })

        cov_90, crps_mod, crps_clim = evaluator.extract_extreme_samples_and_metrics(df_train, df_oos)

        assert 0.0 <= cov_90 <= 1.0
        assert crps_mod > 0.0
        assert crps_clim > 0.0
        assert crps_mod < crps_clim
