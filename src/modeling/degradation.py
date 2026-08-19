#!/usr/bin/env python3
"""
Two-Level Degradation and Soft Warning Mechanism (Ticket 2.2-02 / Issue #15).

Implements the two-tier operational hierarchy:
    Level 1: Gaussian EMOS with Climatological Variance Floor N(μ, σ²)
    Level 2: Historical Climatology Baseline N(μ_clim(d), σ_clim²(d))

Triggers (v5.9.1 §4):
    Hard Trigger (Forced Level 2):
        - Optimizer failed to converge
        - Non-finite parameters (NaN/Inf)
        - Insufficient sample count (N < 10)
    Soft Trigger (Statistical Level 2):
        - Validation/Sample CRPS statistically significantly worse than climatology (p < 0.05 via paired test)
    Soft Warning (Keep Level 1, Log Warning):
        - |c| > 10 or |d| > 10 (overfitting indicator)
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats

from src.modeling.emos_trainer import ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS

logger = logging.getLogger(__name__)


@dataclass
class DegradationDecision:
    """Encapsulates the degradation outcome and rationale."""

    level: int  # 1 for Gaussian EMOS + Floor, 2 for Climatology
    is_degraded: bool
    reason: str
    p_value: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize degradation decision to dictionary."""
        return {
            "level": self.level,
            "is_degraded": self.is_degraded,
            "reason": self.reason,
            "p_value": self.p_value,
            "warnings": self.warnings,
        }


class DegradationHandler:
    """Manages two-level fallback decisions and prediction routing."""

    def __init__(
        self,
        alpha: float = 0.05,
        min_sample_count: int = 10,
        warning_param_threshold: float = 10.0,
    ):
        self.alpha = float(alpha)
        self.min_sample_count = int(min_sample_count)
        self.warning_param_threshold = float(warning_param_threshold)

    def evaluate(
        self,
        diagnostics: ModelTrainingDiagnostics,
        emos_sample_crps: Optional[Union[np.ndarray, Sequence[float]]] = None,
        clim_sample_crps: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> DegradationDecision:
        """Evaluate training diagnostics and sample-level errors to determine Level 1 vs Level 2."""
        warnings: List[str] = list(diagnostics.warnings)

        # ---------------------------------------------------------------------
        # 1. Hard Trigger Checks (Deterministic Level 2 Fallback)
        # ---------------------------------------------------------------------
        if not diagnostics.success:
            logger.error("Hard degradation triggered: Optimizer failed to converge")
            return DegradationDecision(
                level=2,
                is_degraded=True,
                reason="Optimizer failed to converge (max iterations exceeded or loss non-finite)",
                warnings=warnings,
            )

        params_arr = np.asarray(diagnostics.params, dtype=np.float64)
        if not np.all(np.isfinite(params_arr)):
            logger.error("Hard degradation triggered: Non-finite parameter encountered %s", diagnostics.params)
            return DegradationDecision(
                level=2,
                is_degraded=True,
                reason=f"Non-finite parameter encountered in {diagnostics.params}",
                warnings=warnings,
            )

        if diagnostics.sample_count < self.min_sample_count:
            logger.error(
                "Hard degradation triggered: Insufficient sample count (%d < %d)",
                diagnostics.sample_count,
                self.min_sample_count,
            )
            return DegradationDecision(
                level=2,
                is_degraded=True,
                reason=f"Insufficient sample count: {diagnostics.sample_count} < {self.min_sample_count}",
                warnings=warnings,
            )

        # ---------------------------------------------------------------------
        # 2. Soft Trigger Checks (Statistical Paired Significance Test)
        # ---------------------------------------------------------------------
        p_val: Optional[float] = None
        if emos_sample_crps is not None and clim_sample_crps is not None:
            e_crps = np.asarray(emos_sample_crps, dtype=np.float64)
            c_crps = np.asarray(clim_sample_crps, dtype=np.float64)

            if len(e_crps) == len(c_crps) and len(e_crps) >= self.min_sample_count:
                mean_diff = float(np.mean(e_crps - c_crps))
                # If EMOS CRPS is on average higher (worse) than climatology
                if mean_diff > 0:
                    # One-sided paired t-test: H0: mean(e_crps - c_crps) <= 0 vs H1: > 0
                    t_res = stats.ttest_rel(e_crps, c_crps, alternative="greater")
                    p_val = float(t_res.pvalue)

                    if p_val < self.alpha:
                        reason = (
                            f"Soft degradation triggered: EMOS CRPS ({np.mean(e_crps):.3f}) is "
                            f"statistically significantly inferior to climatology ({np.mean(c_crps):.3f}) "
                            f"(p={p_val:.4f} < {self.alpha})"
                        )
                        logger.warning(reason)
                        return DegradationDecision(
                            level=2,
                            is_degraded=True,
                            reason=reason,
                            p_value=p_val,
                            warnings=warnings,
                        )

        # ---------------------------------------------------------------------
        # 3. Soft Warning Checks (Retain Level 1, Log Soft Overfitting Warning)
        # ---------------------------------------------------------------------
        a, b, c, d = diagnostics.params
        if abs(c) > self.warning_param_threshold:
            w_msg = f"Parameter c magnitude high: |c|={abs(c):.2f} > {self.warning_param_threshold}"
            if w_msg not in warnings:
                warnings.append(w_msg)
            logger.warning(w_msg)

        if abs(d) > self.warning_param_threshold:
            w_msg = f"Parameter d magnitude high: |d|={abs(d):.2f} > {self.warning_param_threshold}"
            if w_msg not in warnings:
                warnings.append(w_msg)
            logger.warning(w_msg)

        # All checks passed: Level 1 Normal
        return DegradationDecision(
            level=1,
            is_degraded=False,
            reason="Model healthy: passed hard and soft degradation thresholds",
            p_value=p_val,
            warnings=warnings,
        )

    def get_active_distribution(
        self,
        level: int,
        emos_model: GaussianEMOS,
        ens_mean: Union[float, np.ndarray],
        ens_var: Union[float, np.ndarray],
        mu_clim: Union[float, np.ndarray],
        sigma_clim: Union[float, np.ndarray],
    ) -> GaussianEMOS:
        """Route to appropriate probability distribution based on active degradation level."""
        if level == 1:
            # Level 1: Gaussian EMOS with Climatological Variance Floor
            sigma_clim_sq = np.square(np.asarray(sigma_clim, dtype=np.float64))
            mu, sigma = emos_model.compute_params(
                ensemble_mean=ens_mean,
                ensemble_variance=ens_var,
                sigma_clim_squared=sigma_clim_sq,
            )
            return GaussianEMOS.from_params(mu=mu, sigma=sigma)
        elif level == 2:
            # Level 2: Pure Climatological Baseline
            return GaussianEMOS.from_params(mu=float(mu_clim), sigma=float(sigma_clim))
        else:
            raise ValueError(f"Invalid degradation level: {level} (must be 1 or 2)")
