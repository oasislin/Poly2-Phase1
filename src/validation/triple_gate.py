#!/usr/bin/env python3
"""
TripleGateEvaluator: Formal implementation of v5.9.2 §5 Triple Acceptance Gates.
Part of Phase 1D Validation System (Ticket 4.1-03 / Issue #35).

Implements:
    - Gate 1 (Standard Node PIT Calibration): Kolmogorov-Smirnov test against U(0,1) with p > 0.05.
    - Gate 2 (30h Virtual Holdout Interpolation): Dual assertion:
        1. Accuracy conservation: CRPS_virt <= 1.05 * CRPS_real
        2. Calibration intact: PIT KS p > 0.05
    - Gate 3 (Extreme Tail Stress Test, Strictly 2019 OOS): Dual assertion:
        1. Relative skill: CRPS_model < CRPS_clim (outperforms climatology)
        2. Coverage: 90% CI coverage >= 80% (permits 10% loss to prevent missed alerts)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.validation.metrics_calculator import MetricsCalculator
from src.validation.statistical_tests import pit_ks_test

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastSlice:
    """Value object encapsulating station, target, and lead time coordinates."""

    station_id: str
    target_type: str
    lead_hours: int

    @property
    def key(self) -> str:
        return f"{self.station_id}_{self.target_type}_{self.lead_hours}h"


@dataclass
class GateEvaluationResult:
    """Detailed evaluation result for an individual acceptance gate."""

    gate_name: str
    passed: bool
    metrics: Dict[str, Any]
    thresholds: Dict[str, Any]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate evaluation to dictionary."""
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "reasons": self.reasons,
        }


@dataclass
class TripleGateReport:
    """Comprehensive Triple Acceptance Gates verdict report."""

    overall_passed: bool
    gate1: GateEvaluationResult
    gate2: GateEvaluationResult
    gate3: GateEvaluationResult
    station_summaries: Dict[str, Any] = field(default_factory=dict)
    lead_time_tiers: Dict[str, str] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete report to dictionary."""
        return {
            "overall_passed": self.overall_passed,
            "gate1_pit": self.gate1.to_dict(),
            "gate2_interpolation": self.gate2.to_dict(),
            "gate3_extreme_tails": self.gate3.to_dict(),
            "station_summaries": self.station_summaries,
            "lead_time_tiers": self.lead_time_tiers,
            "generated_at": self.generated_at,
        }

    def to_markdown(self) -> str:
        """Format human-readable GitHub-Flavored Markdown report."""
        verdict_str = "🟢 **PASSED (ACCEPTANCE CRITERIA MET)**" if self.overall_passed else "🔴 **FAILED (REJECTED)**"
        g1_str = "✅ PASS" if self.gate1.passed else "❌ FAIL"
        g2_str = "✅ PASS" if self.gate2.passed else "❌ FAIL"
        g3_str = "✅ PASS" if self.gate3.passed else "❌ FAIL"

        p_val_g1 = self.gate1.metrics.get("p_value", 0.0)
        ratio_g2 = self.gate2.metrics.get("ratio", 0.0)
        p_val_g2 = self.gate2.metrics.get("pit_p_value", 0.0)
        cov_g3 = self.gate3.metrics.get("coverage_90", 0.0)
        skill_diff_g3 = self.gate3.metrics.get("skill_diff", 0.0)

        lines = [
            "# Phase 1D Triple Acceptance Verification Report",
            f"**Generated At**: `{self.generated_at}`  ",
            f"**Final Verdict**: {verdict_str}",
            "",
            "## 1. Triple Acceptance Gates Breakdown",
            "| Acceptance Gate | Measured Value | Standard Threshold | Verdict |",
            "|---|:---:|:---:|:---:|",
            f"| **Gate 1 (PIT Calibration)** | KS $p={p_val_g1:.4f}$ | $p > 0.05$ | {g1_str} |",
            f"| **Gate 2 (30h Virtual Holdout)** | Ratio $= {ratio_g2:.3f}$ ($p_{{PIT}}={p_val_g2:.4f}$) | Ratio $\\le 1.05$ & $p_{{PIT}} > 0.05$ | {g2_str} |",
            f"| **Gate 3 (Extreme Tail Skill & Coverage)** | Cov $= {cov_g3:.1%}$, $\\Delta\\text{{CRPS}}={skill_diff_g3:+.3f}$ | Cov $\\ge 80\\%$ & $\\text{{CRPS}}_{{model}} < \\text{{CRPS}}_{{clim}}$ | {g3_str} |",
            "",
            "## 2. Gate Decision Details",
        ]

        for gate in [self.gate1, self.gate2, self.gate3]:
            status_icon = "🟢" if gate.passed else "🔴"
            lines.append(f"### {status_icon} {gate.gate_name}")
            if gate.reasons:
                for reason in gate.reasons:
                    lines.append(f"- {reason}")
            else:
                lines.append("- All acceptance criteria for this gate met.")
            lines.append("")

        if self.station_summaries:
            lines.append("## 3. Station Performance Summaries")
            lines.append("| Station | Summary Metrics |")
            lines.append("|---|---|")
            for station, data in self.station_summaries.items():
                lines.append(f"| **{station}** | `{data}` |")
            lines.append("")

        return "\n".join(lines)


class TripleGateEvaluator:
    """Independent validator executing v5.9.2 Triple Acceptance Gate logic."""

    def __init__(self, metrics_calc: Optional[MetricsCalculator] = None):
        self.metrics_calc = metrics_calc or MetricsCalculator()

    def evaluate_gate1_pit(
        self,
        pit_values: Union[np.ndarray, pd.Series, Sequence[float]],
        alpha: float = 0.05,
    ) -> GateEvaluationResult:
        """Gate 1: Standard Node PIT Calibration Goodness-of-Fit."""
        pits = np.asarray(pit_values, dtype=np.float64)
        ks_stat, p_val, is_calibrated = pit_ks_test(pits, alpha=alpha)

        reasons = []
        if not is_calibrated:
            reasons.append(
                f"PIT values non-uniform: KS statistic={ks_stat:.4f}, p-value={p_val:.4e} <= {alpha}"
            )
        else:
            reasons.append(
                f"PIT distribution uniform: KS statistic={ks_stat:.4f}, p-value={p_val:.4f} > {alpha}"
            )

        return GateEvaluationResult(
            gate_name="Gate 1 (PIT Calibration Uniformity)",
            passed=is_calibrated,
            metrics={"ks_statistic": float(ks_stat), "p_value": float(p_val)},
            thresholds={"alpha": alpha, "rule": "p_value > 0.05"},
            reasons=reasons,
        )

    def evaluate_gate2_interpolation(
        self,
        crps_virt: float,
        crps_real: float,
        pit_values_virt: Union[np.ndarray, pd.Series, Sequence[float]],
        ratio_threshold: float = 1.05,
        alpha: float = 0.05,
    ) -> GateEvaluationResult:
        """Gate 2: 30h Holdout Virtual Node Reconstructed Interpolation Dual Assertion."""
        safe_crps_real = max(1e-12, float(crps_real))
        ratio = float(crps_virt) / safe_crps_real

        pits_virt = np.asarray(pit_values_virt, dtype=np.float64)
        ks_stat, pit_p_val, is_calibrated = pit_ks_test(pits_virt, alpha=alpha)

        ratio_passed = bool(ratio <= ratio_threshold)
        overall_passed = bool(ratio_passed and is_calibrated)

        reasons = []
        if not ratio_passed:
            reasons.append(
                f"CRPS conservation ratio failed: CRPS_virt={crps_virt:.4f}, "
                f"CRPS_real={crps_real:.4f}, ratio={ratio:.4f} > {ratio_threshold}"
            )
        else:
            reasons.append(
                f"CRPS conservation ratio passed: ratio={ratio:.4f} <= {ratio_threshold}"
            )

        if not is_calibrated:
            reasons.append(
                f"Virtual model PIT non-uniform: p-value={pit_p_val:.4e} <= {alpha}"
            )
        else:
            reasons.append(
                f"Virtual model PIT calibrated: p-value={pit_p_val:.4f} > {alpha}"
            )

        return GateEvaluationResult(
            gate_name="Gate 2 (30h Virtual Holdout Interpolation)",
            passed=overall_passed,
            metrics={
                "crps_virt": float(crps_virt),
                "crps_real": float(crps_real),
                "ratio": float(ratio),
                "ks_statistic": float(ks_stat),
                "pit_p_value": float(pit_p_val),
            },
            thresholds={
                "ratio_threshold": ratio_threshold,
                "alpha": alpha,
                "rule": "ratio <= 1.05 AND pit_p_value > 0.05",
            },
            reasons=reasons,
        )

    def evaluate_gate3_extremes(
        self,
        extreme_coverage_90: float,
        crps_model_extreme: float,
        crps_clim_extreme: float,
        coverage_threshold: float = 0.80,
    ) -> GateEvaluationResult:
        """Gate 3: Extreme Tail Stress Test Dual Assertion (Strictly 2019 OOS)."""
        cov = float(extreme_coverage_90)
        c_mod = float(crps_model_extreme)
        c_clim = float(crps_clim_extreme)
        skill_diff = c_mod - c_clim

        coverage_passed = bool(cov >= coverage_threshold)
        skill_passed = bool(c_mod < c_clim)
        overall_passed = bool(coverage_passed and skill_passed)

        reasons = []
        if not coverage_passed:
            reasons.append(
                f"Extreme 90% CI coverage failed: measured={cov:.1%} < threshold={coverage_threshold:.1%}"
            )
        else:
            reasons.append(
                f"Extreme 90% CI coverage passed: measured={cov:.1%} >= {coverage_threshold:.1%}"
            )

        if not skill_passed:
            reasons.append(
                f"Extreme tail skill failed: CRPS_model={c_mod:.4f} >= CRPS_clim={c_clim:.4f} "
                f"(failed to beat climatology by {skill_diff:+.4f})"
            )
        else:
            reasons.append(
                f"Extreme tail skill passed: CRPS_model={c_mod:.4f} < CRPS_clim={c_clim:.4f} "
                f"(skill improvement {skill_diff:+.4f})"
            )

        return GateEvaluationResult(
            gate_name="Gate 3 (Extreme Tail Skill & Coverage)",
            passed=overall_passed,
            metrics={
                "coverage_90": cov,
                "crps_model_extreme": c_mod,
                "crps_clim_extreme": c_clim,
                "skill_diff": float(skill_diff),
            },
            thresholds={
                "coverage_threshold": coverage_threshold,
                "rule": "coverage_90 >= 0.80 AND crps_model < crps_clim",
            },
            reasons=reasons,
        )

    def extract_extreme_samples_and_metrics(
        self,
        df_history_train: pd.DataFrame,
        df_oos_2019: pd.DataFrame,
    ) -> Tuple[float, float, float]:
        """Extract extreme samples from 2019 OOS based on 2000-2018 thresholds and evaluate Gate 3 metrics.

        Args:
            df_history_train: 2000-2018 training DataFrame containing 'truth'.
            df_oos_2019: 2019 OOS evaluation DataFrame containing 'truth', 'mu_model', 'sigma_model', 'crps_model', 'crps_clim'.

        Returns:
            Tuple of (extreme_coverage_90, mean_crps_model_extreme, mean_crps_clim_extreme).
        """
        train_truth = df_history_train["truth"].to_numpy(dtype=np.float64)
        q90 = float(np.percentile(train_truth, 90.0))
        q10 = float(np.percentile(train_truth, 10.0))

        oos_truth = df_oos_2019["truth"].to_numpy(dtype=np.float64)
        extreme_mask = (oos_truth >= q90) | (oos_truth <= q10)

        df_ext = df_oos_2019[extreme_mask]
        if df_ext.empty:
            logger.warning("No extreme samples found in 2019 OOS dataset! Using full OOS dataset.")
            df_ext = df_oos_2019

        y_ext = df_ext["truth"].to_numpy(dtype=np.float64)
        mu_ext = df_ext["mu_model"].to_numpy(dtype=np.float64)
        sigma_ext = df_ext["sigma_model"].to_numpy(dtype=np.float64)

        cov_90 = self.metrics_calc.coverage_confidence_interval(y_ext, mu_ext, sigma_ext, 0.90)
        crps_mod_ext = self.metrics_calc.mean_crps(y_ext, mu_ext, sigma_ext)

        if "crps_clim" in df_ext.columns:
            crps_clim_ext = float(np.mean(df_ext["crps_clim"]))
        else:
            clim_mean = df_ext.get("clim_mean", np.full_like(y_ext, np.mean(train_truth)))
            clim_sigma = df_ext.get("clim_sigma", np.full_like(y_ext, np.std(train_truth, ddof=1)))
            crps_clim_ext = self.metrics_calc.mean_crps(y_ext, np.asarray(clim_mean), np.asarray(clim_sigma))

        return cov_90, crps_mod_ext, crps_clim_ext

    def generate_triple_gate_report(
        self,
        standard_pit_values: Union[np.ndarray, pd.Series, Sequence[float]],
        crps_virt_30h: float,
        crps_real_30h: float,
        interp_pit_values: Union[np.ndarray, pd.Series, Sequence[float]],
        extreme_coverage_90: float,
        crps_model_extreme: float,
        crps_clim_extreme: float,
        station_summaries: Optional[Dict[str, Any]] = None,
        lead_time_tiers: Optional[Dict[str, str]] = None,
    ) -> TripleGateReport:
        """Run all three acceptance gates and aggregate into a TripleGateReport."""
        g1 = self.evaluate_gate1_pit(standard_pit_values)
        g2 = self.evaluate_gate2_interpolation(crps_virt_30h, crps_real_30h, interp_pit_values)
        g3 = self.evaluate_gate3_extremes(extreme_coverage_90, crps_model_extreme, crps_clim_extreme)

        overall = bool(g1.passed and g2.passed and g3.passed)

        return TripleGateReport(
            overall_passed=overall,
            gate1=g1,
            gate2=g2,
            gate3=g3,
            station_summaries=station_summaries or {},
            lead_time_tiers=lead_time_tiers or {},
        )
