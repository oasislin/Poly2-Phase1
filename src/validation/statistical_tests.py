#!/usr/bin/env python3
"""
StatisticalSignificance: Rigorous hypothesis testing for forecast evaluation.
Part of Phase 1D Validation System (Ticket 4.1-02 / Issue #34).

Implements:
    - Diebold-Mariano (DM) test with Harvey-Leybourne-Newbold (HLN 1997) finite-sample correction.
    - Wilcoxon Signed-Rank non-parametric paired test.
    - Paired Student's t-test with difference confidence intervals.
    - Probability Integral Transform (PIT) Kolmogorov-Smirnov uniformity test.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DieboldMarianoResult:
    """Result container for Diebold-Mariano predictive accuracy test."""

    dm_statistic: float
    p_value: float
    mean_loss_diff: float
    hln_adjusted: bool
    is_significant: bool
    h: int
    loss_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "dm_statistic": float(self.dm_statistic),
            "p_value": float(self.p_value),
            "mean_loss_diff": float(self.mean_loss_diff),
            "hln_adjusted": self.hln_adjusted,
            "is_significant": self.is_significant,
            "h": self.h,
            "loss_type": self.loss_type,
        }


def diebold_mariano_test(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    pred1: Union[np.ndarray, pd.Series, Sequence[float]],
    pred2: Union[np.ndarray, pd.Series, Sequence[float]],
    loss_type: str = "squared",
    h: int = 1,
    alpha: float = 0.05,
    modified: bool = True,
) -> DieboldMarianoResult:
    """Run Diebold-Mariano test comparing two forecasts given truth observations.

    Args:
        y_true: Observed ground truth series.
        pred1: Forecast series 1 (Candidate model).
        pred2: Forecast series 2 (Benchmark/Reference model).
        loss_type: 'squared' (MSE), 'absolute' (MAE), or 'crps'.
        h: Forecast horizon (lead time steps), default 1.
        alpha: Significance level (default 0.05).
        modified: Whether to apply Harvey, Leybourne, Newbold (1997) adjustment for small samples.

    Returns:
        DieboldMarianoResult with statistic, p-value, and significance verdict.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p1 = np.asarray(pred1, dtype=np.float64)
    p2 = np.asarray(pred2, dtype=np.float64)

    e1 = p1 - y
    e2 = p2 - y

    if loss_type == "squared":
        loss1 = np.square(e1)
        loss2 = np.square(e2)
    elif loss_type == "absolute":
        loss1 = np.abs(e1)
        loss2 = np.abs(e2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'squared' or 'absolute'.")

    return StatisticalSignificance.diebold_mariano_from_losses(
        loss1=loss1,
        loss2=loss2,
        h=h,
        alpha=alpha,
        modified=modified,
        loss_type_name=loss_type,
    )


def wilcoxon_signed_rank_test(
    loss1: Union[np.ndarray, pd.Series, Sequence[float]],
    loss2: Union[np.ndarray, pd.Series, Sequence[float]],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """Run Wilcoxon signed-rank non-parametric test on paired forecast losses.

    Args:
        loss1: Array of loss values for candidate model.
        loss2: Array of loss values for reference model.
        alternative: 'two-sided', 'less' (loss1 < loss2), or 'greater'.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of (statistic, p_value, is_significant).
    """
    l1 = np.asarray(loss1, dtype=np.float64)
    l2 = np.asarray(loss2, dtype=np.float64)

    diff = l1 - l2
    if np.all(diff == 0.0):
        return 0.0, 1.0, False

    stat_res = stats.wilcoxon(l1, l2, alternative=alternative, zero_method="wilcox")
    is_sig = bool(stat_res.pvalue < alpha)
    return float(stat_res.statistic), float(stat_res.pvalue), is_sig


def paired_t_test(
    loss1: Union[np.ndarray, pd.Series, Sequence[float]],
    loss2: Union[np.ndarray, pd.Series, Sequence[float]],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tuple[float, float, float, float, bool]:
    """Run paired Student's t-test on forecast losses with confidence intervals.

    Returns:
        Tuple of (t_statistic, p_value, ci_lower, ci_upper, is_significant).
    """
    l1 = np.asarray(loss1, dtype=np.float64)
    l2 = np.asarray(loss2, dtype=np.float64)
    d = l1 - l2
    n = len(d)

    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n) if n > 0 else 0.0

    if se_d < 1e-15:
        return 0.0, 1.0, mean_d, mean_d, False

    t_res = stats.ttest_rel(l1, l2, alternative=alternative)
    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    is_sig = bool(p_val < alpha)

    return t_stat, p_val, ci_low, ci_high, is_sig


def pit_ks_test(
    pit_values: Union[np.ndarray, pd.Series, Sequence[float]],
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """Kolmogorov-Smirnov goodness-of-fit test of PIT values against Uniform(0, 1).

    Returns:
        Tuple of (ks_statistic, p_value, is_calibrated) where is_calibrated is True if p > alpha.
    """
    pits = np.asarray(pit_values, dtype=np.float64)
    res = stats.kstest(pits, "uniform")
    is_calibrated = bool(res.pvalue > alpha)
    return float(res.statistic), float(res.pvalue), is_calibrated


class StatisticalSignificance:
    """Class suite encapsulating statistical significance tools for forecast validation."""

    @staticmethod
    def diebold_mariano_from_losses(
        loss1: Union[np.ndarray, pd.Series, Sequence[float]],
        loss2: Union[np.ndarray, pd.Series, Sequence[float]],
        h: int = 1,
        alpha: float = 0.05,
        modified: bool = True,
        loss_type_name: str = "custom",
    ) -> DieboldMarianoResult:
        """Compute Diebold-Mariano test directly from precomputed loss vectors."""
        l1 = np.asarray(loss1, dtype=np.float64)
        l2 = np.asarray(loss2, dtype=np.float64)

        d = l1 - l2
        T = len(d)

        if T < 2:
            return DieboldMarianoResult(
                dm_statistic=0.0,
                p_value=1.0,
                mean_loss_diff=float(np.mean(d)) if T == 1 else 0.0,
                hln_adjusted=modified,
                is_significant=False,
                h=h,
                loss_type=loss_type_name,
            )

        mean_d = float(np.mean(d))
        d_centered = d - mean_d

        # Sample autocovariance gamma_k
        gamma_0 = float(np.mean(d_centered ** 2))
        long_run_var = gamma_0

        for k in range(1, h):
            gamma_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
            long_run_var += 2.0 * gamma_k

        # Numerical floor
        if long_run_var <= 1e-15:
            if abs(mean_d) < 1e-15:
                return DieboldMarianoResult(
                    dm_statistic=0.0,
                    p_value=1.0,
                    mean_loss_diff=mean_d,
                    hln_adjusted=modified,
                    is_significant=False,
                    h=h,
                    loss_type=loss_type_name,
                )
            long_run_var = max(gamma_0, 1e-12)

        v_d_bar = long_run_var / T
        dm_stat = mean_d / math.sqrt(v_d_bar)

        if modified:
            # Harvey, Leybourne, Newbold (1997) small-sample modification
            adj = math.sqrt((T + 1.0 - 2.0 * h + (h * (h - 1.0)) / T) / T)
            dm_stat = dm_stat * adj
            p_val = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=T - 1))
        else:
            p_val = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

        is_sig = bool(p_val < alpha)

        return DieboldMarianoResult(
            dm_statistic=float(dm_stat),
            p_value=float(p_val),
            mean_loss_diff=float(mean_d),
            hln_adjusted=modified,
            is_significant=is_sig,
            h=h,
            loss_type=loss_type_name,
        )

    @staticmethod
    def wilcoxon_test(loss1, loss2, alternative="two-sided", alpha=0.05):
        return wilcoxon_signed_rank_test(loss1, loss2, alternative=alternative, alpha=alpha)

    @staticmethod
    def paired_t_test(loss1, loss2, alpha=0.05, alternative="two-sided"):
        return paired_t_test(loss1, loss2, alpha=alpha, alternative=alternative)

    @staticmethod
    def pit_ks_test(pit_values, alpha=0.05):
        return pit_ks_test(pit_values, alpha=alpha)
