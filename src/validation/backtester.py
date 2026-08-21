#!/usr/bin/env python3
"""
Backtester: Historical Backtesting Engine and Multi-Baseline Comparison.
Part of Phase 1D Validation System (Ticket 4.2-01 / Issue #36).

Implements:
    - Strictly ordered time series evaluation (0 lookahead bias).
    - Multi-baseline benchmarking: Candidate Model vs Climatology vs Raw GEFS vs Persistence.
    - End-to-end performance metric evaluation: CRPS, CRPSS, MAE, RMSE, Bias, 90% CI Coverage, PIT values.
    - Polymarket Discrete Bin conversion and Brier Score integration.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.prediction.bin_converter import BinConverter
from src.validation.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    RawGEFSBaseline,
)
from src.validation.metrics_calculator import MetricsCalculator
from src.validation.triple_gate import ForecastSlice

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Comprehensive container for historical backtest outcomes and benchmark comparisons."""

    slice_info: ForecastSlice
    sample_count: int
    mean_crps_model: float
    mean_crps_raw: float
    mean_crps_clim: float
    mean_crps_persistence: float
    crpss_vs_raw: float
    crpss_vs_clim: float
    crpss_vs_persistence: float
    mae_model: float
    mae_raw: float
    mae_clim: float
    mae_persistence: float
    coverage_90_ci_model: float
    pit_values: np.ndarray
    brier_score_discrete: float
    df_daily: pd.DataFrame = field(repr=False)

    @property
    def station_id(self) -> str:
        return self.slice_info.station_id

    @property
    def target_type(self) -> str:
        return self.slice_info.target_type

    @property
    def lead_hours(self) -> int:
        return self.slice_info.lead_hours

    def to_dict(self) -> Dict[str, Any]:
        """Serialize backtest result summary to dictionary."""
        return {
            "station_id": self.station_id,
            "target_type": self.target_type,
            "lead_hours": self.lead_hours,
            "sample_count": self.sample_count,
            "mean_crps_model": float(self.mean_crps_model),
            "mean_crps_raw": float(self.mean_crps_raw),
            "mean_crps_clim": float(self.mean_crps_clim),
            "mean_crps_persistence": float(self.mean_crps_persistence),
            "crpss_vs_raw": float(self.crpss_vs_raw),
            "crpss_vs_clim": float(self.crpss_vs_clim),
            "crpss_vs_persistence": float(self.crpss_vs_persistence),
            "mae_model": float(self.mae_model),
            "mae_raw": float(self.mae_raw),
            "mae_clim": float(self.mae_clim),
            "mae_persistence": float(self.mae_persistence),
            "coverage_90_ci_model": float(self.coverage_90_ci_model),
            "brier_score_discrete": float(self.brier_score_discrete),
        }


class BacktestEngine:
    """Backtesting runner orchestrating model predictions and multi-baseline comparisons."""

    def __init__(
        self,
        metrics_calc: Optional[MetricsCalculator] = None,
        climatology_calculator: Optional[Any] = None,
        storage_manager: Optional[Any] = None,
    ):
        self.metrics_calc = metrics_calc or MetricsCalculator()
        self.climatology_calculator = climatology_calculator
        self.storage_manager = storage_manager
        self.clim_baseline = ClimatologyBaseline(climatology_calculator) if climatology_calculator else None
        self.raw_gefs_baseline = RawGEFSBaseline()
        self.persistence_baseline = PersistenceBaseline(storage_manager, climatology_calculator)

    def _predict_model(
        self,
        df: pd.DataFrame,
        model_predictor: Any,
        slice_info: ForecastSlice,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate model predictions (mu, sigma) across DataFrame rows."""
        n_samples = len(df)
        mu_model = np.zeros(n_samples, dtype=np.float64)
        sigma_model = np.zeros(n_samples, dtype=np.float64)

        for i, row in df.iterrows():
            if hasattr(model_predictor, "predict"):
                pred_out = model_predictor.predict(
                    station_id=slice_info.station_id,
                    target_date=row["target_date"],
                    target_type=slice_info.target_type,
                    lead_hours=slice_info.lead_hours,
                    ensemble_mean=row["ensemble_mean"],
                    ensemble_variance=row["ensemble_variance"],
                )
                if isinstance(pred_out, tuple):
                    mu_model[i], sigma_model[i] = pred_out[0], pred_out[1]
                else:
                    mu_model[i], sigma_model[i] = pred_out.mu, pred_out.sigma
            elif callable(model_predictor):
                mu_model[i], sigma_model[i] = model_predictor(row)

        return mu_model, sigma_model

    def _resolve_baselines(
        self,
        df: pd.DataFrame,
        truth_arr: np.ndarray,
        slice_info: ForecastSlice,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resolve Raw GEFS, Climatology, and Persistence baseline prediction series."""
        n_samples = len(df)
        ens_mean_arr = df["ensemble_mean"].to_numpy(dtype=np.float64)
        ens_var_arr = df["ensemble_variance"].to_numpy(dtype=np.float64)

        # Raw GEFS
        mu_raw = ens_mean_arr.copy()
        sigma_raw = np.maximum(np.sqrt(np.maximum(ens_var_arr, 0.0)), 0.1)

        # Climatology
        if "clim_mean" in df.columns and "clim_sigma" in df.columns:
            mu_clim = df["clim_mean"].to_numpy(dtype=np.float64)
            sigma_clim = df["clim_sigma"].to_numpy(dtype=np.float64)
        elif self.clim_baseline is not None:
            mu_clim = np.zeros(n_samples, dtype=np.float64)
            sigma_clim = np.zeros(n_samples, dtype=np.float64)
            for i, row in df.iterrows():
                m, s = self.clim_baseline.predict(slice_info.station_id, row["target_date"], slice_info.target_type)
                mu_clim[i], sigma_clim[i] = m, s
        else:
            mu_clim = np.full(n_samples, float(np.mean(truth_arr)))
            sigma_clim = np.full(n_samples, float(np.std(truth_arr, ddof=1)) or 3.0)

        # Persistence
        if "yesterday_truth" in df.columns:
            mu_persist = df["yesterday_truth"].to_numpy(dtype=np.float64)
            sigma_persist = sigma_clim.copy()
        elif self.persistence_baseline is not None:
            mu_persist = np.zeros(n_samples, dtype=np.float64)
            sigma_persist = sigma_clim.copy()
            for i, row in df.iterrows():
                m, s = self.persistence_baseline.predict(
                    station_id=slice_info.station_id,
                    target_date=row["target_date"],
                    target_type=slice_info.target_type,
                    yesterday_truth=row.get("yesterday_truth", None),
                )
                mu_persist[i] = m
        else:
            mu_persist = np.roll(truth_arr, 1)
            mu_persist[0] = truth_arr[0]
            sigma_persist = sigma_clim.copy()

        return mu_raw, sigma_raw, mu_clim, sigma_clim, mu_persist, sigma_persist

    def _compute_discrete_brier(
        self,
        truth_arr: np.ndarray,
        mu_model: np.ndarray,
        sigma_model: np.ndarray,
        station_id: str,
    ) -> float:
        """Evaluate Polymarket discrete bin probabilities and multi-class Brier score."""
        n_samples = len(truth_arr)
        if n_samples == 0:
            return 0.0

        from scipy import stats
        brier_scores = []
        unit = "F" if (station_id.upper() == "KDEN") else "C"

        for i in range(n_samples):
            obs_temp = truth_arr[i]
            mu = mu_model[i]
            sigma = max(1e-4, sigma_model[i])

            bins = BinConverter.generate_bins(
                station_id=station_id,
                center_temp=float(obs_temp),
                num_bins=7,
            )
            dist = stats.norm(loc=mu, scale=sigma)
            eval_bins = BinConverter.calculate_bin_probabilities(dist, bins, normalize=True)
            pred_probs = np.array([b.probability for b in eval_bins], dtype=np.float64)

            obs_idx, _ = BinConverter.determine_winning_bin(eval_bins, observed_temp=obs_temp, unit="C")
            one_hot = np.zeros(len(eval_bins), dtype=np.float64)
            one_hot[obs_idx] = 1.0

            bs = self.metrics_calc.brier_score_multiclass(one_hot.reshape(1, -1), pred_probs.reshape(1, -1))
            brier_scores.append(bs)

        return float(np.mean(brier_scores))

    def run_backtest(
        self,
        station_id: str,
        target_type: str,
        lead_hours: int,
        dataset: pd.DataFrame,
        model_predictor: Any,
    ) -> BacktestResult:
        """Run historical backtest on evaluation dataset for a specific slice."""
        df = dataset.copy().sort_values(by="target_date").reset_index(drop=True)
        n_samples = len(df)
        if n_samples == 0:
            raise ValueError(f"Empty dataset for {station_id} {target_type} {lead_hours}h")

        slice_info = ForecastSlice(station_id=station_id, target_type=target_type, lead_hours=lead_hours)
        truth_arr = df["truth"].to_numpy(dtype=np.float64)

        # Predict Model & Baselines
        mu_mod, sig_mod = self._predict_model(df, model_predictor, slice_info)
        mu_r, sig_r, mu_c, sig_c, mu_p, sig_p = self._resolve_baselines(df, truth_arr, slice_info)

        # Compute CRPS
        crps_mod = self.metrics_calc.crps_gaussian(truth_arr, mu_mod, sig_mod)
        crps_r = self.metrics_calc.crps_gaussian(truth_arr, mu_r, sig_r)
        crps_c = self.metrics_calc.crps_gaussian(truth_arr, mu_c, sig_c)
        crps_p = self.metrics_calc.crps_gaussian(truth_arr, mu_p, sig_p)

        mean_crps_m = float(np.mean(crps_mod))
        mean_crps_raw = float(np.mean(crps_r))
        mean_crps_clim = float(np.mean(crps_c))
        mean_crps_persist = float(np.mean(crps_p))

        # Skill & Point Errors
        crpss_raw = self.metrics_calc.crpss(mean_crps_m, mean_crps_raw)
        crpss_clim = self.metrics_calc.crpss(mean_crps_m, mean_crps_clim)
        crpss_persist = self.metrics_calc.crpss(mean_crps_m, mean_crps_persist)

        mae_m = self.metrics_calc.mae(truth_arr, mu_mod)
        mae_r = self.metrics_calc.mae(truth_arr, mu_r)
        mae_c = self.metrics_calc.mae(truth_arr, mu_c)
        mae_p = self.metrics_calc.mae(truth_arr, mu_p)

        cov_90 = self.metrics_calc.coverage_confidence_interval(truth_arr, mu_mod, sig_mod, 0.90)
        pit_vals = self.metrics_calc.compute_pit_values(truth_arr, mu_mod, sig_mod)
        bs_discrete = self._compute_discrete_brier(truth_arr, mu_mod, sig_mod, station_id)

        # Assemble DataFrame
        df_daily = df.copy()
        df_daily["mu_model"] = mu_mod
        df_daily["sigma_model"] = sig_mod
        df_daily["mu_raw"] = mu_r
        df_daily["sigma_raw"] = sig_r
        df_daily["mu_clim"] = mu_c
        df_daily["sigma_clim"] = sig_c
        df_daily["mu_persistence"] = mu_p
        df_daily["sigma_persistence"] = sig_p
        df_daily["crps_model"] = crps_mod
        df_daily["crps_raw"] = crps_r
        df_daily["crps_clim"] = crps_c
        df_daily["crps_persistence"] = crps_p
        df_daily["error_model"] = mu_mod - truth_arr
        df_daily["error_raw"] = mu_r - truth_arr
        df_daily["pit_model"] = pit_vals

        return BacktestResult(
            slice_info=slice_info,
            sample_count=n_samples,
            mean_crps_model=mean_crps_m,
            mean_crps_raw=mean_crps_raw,
            mean_crps_clim=mean_crps_clim,
            mean_crps_persistence=mean_crps_persist,
            crpss_vs_raw=crpss_raw,
            crpss_vs_clim=crpss_clim,
            crpss_vs_persistence=crpss_persist,
            mae_model=mae_m,
            mae_raw=mae_r,
            mae_clim=mae_c,
            mae_persistence=mae_p,
            coverage_90_ci_model=cov_90,
            pit_values=pit_vals,
            brier_score_discrete=bs_discrete,
            df_daily=df_daily,
        )
