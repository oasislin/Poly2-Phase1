#!/usr/bin/env python3
"""
MetricsCalculator: Comprehensive Probabilistic, Point, Calibration, and Discrete Market Verification Metrics.
Part of Phase 1D Validation System (Ticket 4.1-01 / Issue #33).

Implements:
    - Continuous Probabilistic: Gaussian CRPS, Mean CRPS, CRPSS, MAE, RMSE, Bias, CI Coverage.
    - Calibration & Reliability: PIT values, PIT Histogram, Talagrand Rank Histogram, Spread-Error Ratio.
    - Discrete Market Bins (Polymarket): Brier Score (Binary & Multi-class), Brier Skill Score, Multi-class Log Loss (Cross-Entropy), Reliability Diagram, Expected Calibration Error (ECE).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.crps import gaussian_crps


@dataclass
class ReliabilityDiagramData:
    """Container for Reliability Diagram / Calibration Curve data."""

    bin_centers: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    ece: float
    num_bins: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reliability data to dictionary."""
        return {
            "bin_centers": self.bin_centers.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "ece": float(self.ece),
            "num_bins": self.num_bins,
        }


class MetricsCalculator:
    """Unified calculator for probabilistic forecast evaluation and discrete market settlement validation."""

    def __init__(self, eps: float = 1e-15):
        """Initialize calculator with numerical safety precision.

        Args:
            eps: Epsilon floor for divisions, log losses, and zero boundaries.
        """
        self.eps = eps

    # ==========================================
    # 1. Continuous Probabilistic Metrics
    # ==========================================

    def crps_gaussian(
        self,
        y: Union[float, np.ndarray, pd.Series],
        mu: Union[float, np.ndarray, pd.Series],
        sigma: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray]:
        """Compute closed-form Gaussian Continuous Ranked Probability Score (CRPS)."""
        return gaussian_crps(y=y, mu=mu, sigma=sigma)

    def mean_crps(
        self,
        y: Union[np.ndarray, pd.Series],
        mu: Union[np.ndarray, pd.Series],
        sigma: Union[np.ndarray, pd.Series],
    ) -> float:
        """Compute sample mean CRPS across a collection of forecasts."""
        crps_vals = self.crps_gaussian(y=y, mu=mu, sigma=sigma)
        return float(np.mean(crps_vals))

    def crpss(
        self,
        crps_model: float,
        crps_ref: float,
    ) -> float:
        """Compute Continuous Ranked Probability Skill Score (CRPSS) vs reference forecast.

        CRPSS = 1 - (CRPS_model / CRPS_ref)
        """
        if crps_ref <= self.eps:
            if crps_model <= self.eps:
                return 0.0
            return -math.inf
        return float(1.0 - (crps_model / crps_ref))

    def mae(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
    ) -> float:
        """Compute Mean Absolute Error (MAE)."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(np.abs(y_t - y_p)))

    def rmse(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
    ) -> float:
        """Compute Root Mean Squared Error (RMSE)."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(y_t - y_p))))

    def bias(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
    ) -> float:
        """Compute Mean Forecast Bias: mean(y_pred - y_true)."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(y_p - y_t))

    def coverage_confidence_interval(
        self,
        y_true: Union[np.ndarray, pd.Series],
        mu: Union[np.ndarray, pd.Series],
        sigma: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.90,
    ) -> float:
        """Compute empirical coverage proportion of Gaussian confidence intervals."""
        y_t = np.asarray(y_true, dtype=np.float64)
        m = np.asarray(mu, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)

        alpha = 1.0 - confidence_level
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

        lower_bound = m - z_crit * s
        upper_bound = m + z_crit * s

        covered = (y_t >= lower_bound) & (y_t <= upper_bound)
        return float(np.mean(covered))

    # ==========================================
    # 2. Calibration & Reliability Metrics
    # ==========================================

    def compute_pit_values(
        self,
        y_true: Union[np.ndarray, pd.Series],
        mu: Union[np.ndarray, pd.Series],
        sigma: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Compute Probability Integral Transform (PIT) values: p_i = Phi((y_i - mu_i) / sigma_i)."""
        y_t = np.asarray(y_true, dtype=np.float64)
        m = np.asarray(mu, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)

        safe_s = np.maximum(s, self.eps)
        z = (y_t - m) / safe_s
        pit_vals = stats.norm.cdf(z)
        return np.clip(pit_vals, 0.0, 1.0)

    def pit_histogram(
        self,
        pit_values: Union[np.ndarray, pd.Series, Sequence[float]],
        num_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute PIT histogram counts and relative frequencies."""
        pits = np.asarray(pit_values, dtype=np.float64)
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

        counts, _ = np.histogram(pits, bins=bin_edges)
        total = len(pits)
        rel_freqs = counts / max(1, total)

        return counts, rel_freqs, bin_edges

    def talagrand_rank_histogram(
        self,
        y_true: Union[np.ndarray, pd.Series],
        ensemble_members: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Talagrand Rank Histogram for ensemble forecasts.

        Args:
            y_true: 1D array of observed truths of shape (N,).
            ensemble_members: 2D array of ensemble forecasts of shape (N, M).

        Returns:
            Tuple of (individual ranks (0..M), rank_counts of length M+1, relative_frequencies of length M+1).
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        ens = np.asarray(ensemble_members, dtype=np.float64)

        if ens.ndim == 1:
            ens = ens.reshape(-1, 1)

        n_samples, n_members = ens.shape
        ranks = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            obs = y_t[i]
            members = ens[i, :]
            less_count = np.sum(members < obs)
            equal_count = np.sum(members == obs)
            if equal_count > 0:
                tie_offset = np.random.randint(0, equal_count + 1)
                ranks[i] = int(less_count + tie_offset)
            else:
                ranks[i] = int(less_count)

        rank_counts = np.bincount(ranks, minlength=n_members + 1)
        rel_freqs = rank_counts / max(1, n_samples)

        return ranks, rank_counts, rel_freqs

    def spread_error_ratio(
        self,
        y_true: Union[np.ndarray, pd.Series],
        mu: Union[np.ndarray, pd.Series],
        sigma: Union[np.ndarray, pd.Series],
    ) -> float:
        """Compute Spread-to-Error ratio: sqrt(mean(sigma^2)) / RMSE."""
        s = np.asarray(sigma, dtype=np.float64)
        root_mean_var = float(np.sqrt(np.mean(np.square(s))))
        rmse_val = self.rmse(y_true, mu)

        if rmse_val <= self.eps:
            if root_mean_var <= self.eps:
                return 1.0
            return math.inf

        return root_mean_var / rmse_val

    # ==========================================
    # 3. Discrete Market Bins (Polymarket)
    # ==========================================

    def brier_score(
        self,
        y_true_binary: Union[np.ndarray, pd.Series, Sequence[int]],
        predicted_probs: Union[np.ndarray, pd.Series, Sequence[float]],
    ) -> float:
        """Compute binary Brier Score: (1/N) * sum((p_i - y_i)^2)."""
        y_b = np.asarray(y_true_binary, dtype=np.float64)
        p = np.asarray(predicted_probs, dtype=np.float64)
        return float(np.mean(np.square(p - y_b)))

    def brier_skill_score(
        self,
        bs_model: float,
        bs_ref: float,
    ) -> float:
        """Compute Brier Skill Score (BSS) vs reference forecast.

        BSS = 1 - (BS_model / BS_ref)
        """
        if bs_ref <= self.eps:
            if bs_model <= self.eps:
                return 0.0
            return -math.inf
        return float(1.0 - (bs_model / bs_ref))

    def brier_score_multiclass(
        self,
        y_true_one_hot: Union[np.ndarray, pd.DataFrame],
        predicted_probs: Union[np.ndarray, pd.DataFrame],
    ) -> float:
        """Compute Multi-Class Brier Score across discrete category bins.

        BS_multi = (1/N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
        """
        y_oh = np.asarray(y_true_one_hot, dtype=np.float64)
        p = np.asarray(predicted_probs, dtype=np.float64)
        sample_brier = np.sum(np.square(p - y_oh), axis=-1)
        return float(np.mean(sample_brier))

    def multiclass_log_loss(
        self,
        y_true_one_hot: Union[np.ndarray, pd.DataFrame],
        predicted_probs: Union[np.ndarray, pd.DataFrame],
    ) -> float:
        """Compute Multi-Class Cross-Entropy / Log Loss with numerical safety clipping.

        Loss = - (1/N) * sum_{i=1}^N sum_{k=1}^K y_{ik} * log(clip(p_{ik}, eps, 1-eps))
        """
        y_oh = np.asarray(y_true_one_hot, dtype=np.float64)
        p = np.asarray(predicted_probs, dtype=np.float64)

        p_safe = np.clip(p, self.eps, 1.0 - self.eps)
        log_p = np.log(p_safe)

        cross_entropy_per_sample = -np.sum(y_oh * log_p, axis=-1)
        return float(np.mean(cross_entropy_per_sample))

    def reliability_diagram(
        self,
        y_true_binary: Union[np.ndarray, pd.Series, Sequence[int]],
        predicted_probs: Union[np.ndarray, pd.Series, Sequence[float]],
        num_bins: int = 10,
    ) -> ReliabilityDiagramData:
        """Compute Reliability Diagram calibration data and Expected Calibration Error (ECE)."""
        y_b = np.asarray(y_true_binary, dtype=np.float64)
        p = np.asarray(predicted_probs, dtype=np.float64)

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bin_accuracies = np.zeros(num_bins, dtype=np.float64)
        bin_confidences = np.zeros(num_bins, dtype=np.float64)
        bin_counts = np.zeros(num_bins, dtype=int)

        n_total = len(y_b)
        ece = 0.0

        for i in range(num_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            if i == num_bins - 1:
                mask = (p >= low) & (p <= high)
            else:
                mask = (p >= low) & (p < high)

            count = int(np.sum(mask))
            bin_counts[i] = count

            if count > 0:
                acc = float(np.mean(y_b[mask]))
                conf = float(np.mean(p[mask]))
                bin_accuracies[i] = acc
                bin_confidences[i] = conf
                ece += (count / max(1, n_total)) * abs(acc - conf)
            else:
                bin_accuracies[i] = bin_centers[i]
                bin_confidences[i] = bin_centers[i]

        return ReliabilityDiagramData(
            bin_centers=bin_centers,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            ece=float(ece),
            num_bins=num_bins,
        )

    def expected_calibration_error(
        self,
        y_true_binary: Union[np.ndarray, pd.Series, Sequence[int]],
        predicted_probs: Union[np.ndarray, pd.Series, Sequence[float]],
        num_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        rel_data = self.reliability_diagram(y_true_binary, predicted_probs, num_bins=num_bins)
        return float(rel_data.ece)
