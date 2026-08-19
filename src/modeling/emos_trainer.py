#!/usr/bin/env python3
"""
EMOSOptimizer & ModelTrainingDiagnostics (Ticket 2.2-01 / Issue #14).

Optimizes EMOS calibration parameters (a, b, c, d) via L-BFGS-B on Gneiting closed-form CRPS loss.
Includes L2 regularization on d, warm-start perturbation, multi-start restarts, and in-sample health diagnostics.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import optimize

from src.modeling.crps import emos_crps_loss, gaussian_crps
from src.modeling.gaussian_emos import GaussianEMOS

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingDiagnostics:
    """Structured scorecard recording training optimization and calibration diagnostics."""

    success: bool
    params: Tuple[float, float, float, float]
    crps_in_sample: float
    crps_raw_ensemble: float
    crps_climatology: float
    crpss_vs_raw: float
    crpss_vs_clim: float
    n_iterations: int
    n_evaluations: int
    grad_norm: float
    restarts_used: int
    sample_count: int
    warnings: List[str] = field(default_factory=list)
    health_grade: str = "HEALTHY"

    def __post_init__(self):
        # Auto-compute warnings and determine health_grade
        a, b, c, d = self.params

        if abs(c) > 10.0:
            self.warnings.append(f"High parameter c magnitude: |c|={abs(c):.2f} > 10")
        if abs(d) > 10.0:
            self.warnings.append(f"High parameter d magnitude: |d|={abs(d):.2f} > 10")
        if b <= 0.2 or b >= 2.5:
            self.warnings.append(f"Unusual scale parameter b: b={b:.3f}")
        if self.crpss_vs_clim < 0.0:
            self.warnings.append(f"In-sample CRPS inferior to climatology: CRPSS={self.crpss_vs_clim:.2%}")

        if not self.success:
            self.health_grade = "DEGRADED"
        elif len(self.warnings) > 0:
            self.health_grade = "WARNING"
        else:
            self.health_grade = "HEALTHY"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize diagnostics to dictionary."""
        return {
            "success": self.success,
            "params": {
                "a": self.params[0],
                "b": self.params[1],
                "c": self.params[2],
                "d": self.params[3],
            },
            "crps_in_sample": self.crps_in_sample,
            "crps_raw_ensemble": self.crps_raw_ensemble,
            "crps_climatology": self.crps_climatology,
            "crpss_vs_raw": self.crpss_vs_raw,
            "crpss_vs_clim": self.crpss_vs_clim,
            "n_iterations": self.n_iterations,
            "n_evaluations": self.n_evaluations,
            "grad_norm": self.grad_norm,
            "restarts_used": self.restarts_used,
            "sample_count": self.sample_count,
            "warnings": self.warnings,
            "health_grade": self.health_grade,
        }


class EMOSOptimizer:
    """Parameter estimator for Gaussian EMOS using L-BFGS-B and closed-form CRPS loss."""

    def __init__(
        self,
        l2_lambda_d: float = 1e-3,
        max_iter: int = 1000,
        tolerance: float = 1e-7,
        random_seed: Optional[int] = 42,
        max_restarts: int = 3,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self.l2_lambda_d = float(l2_lambda_d)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.max_restarts = max_restarts
        self.bounds = bounds or (
            (-20.0, 20.0),  # a (bias)
            (0.1, 3.0),     # b (scale / slope)
            (-20.0, 20.0),  # c (residual spread)
            (-20.0, 20.0),  # d (spread gain)
        )
        self._rng = np.random.default_rng(random_seed)

    def optimize(
        self,
        ensemble_mean: Union[np.ndarray, pd.Series],
        ensemble_variance: Union[np.ndarray, pd.Series],
        sigma_clim_squared: Union[np.ndarray, pd.Series],
        observed_temp: Union[np.ndarray, pd.Series],
        mu_clim: Optional[Union[np.ndarray, pd.Series]] = None,
        sigma_clim: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> ModelTrainingDiagnostics:
        """Run L-BFGS-B optimization to fit parameters (a, b, c, d) minimizing CRPS.

        Includes multi-start restarts on optimization failure.
        """
        ens_mean = np.asarray(ensemble_mean, dtype=np.float64)
        ens_var = np.maximum(0.0, np.asarray(ensemble_variance, dtype=np.float64))
        clim_var = np.maximum(0.0, np.asarray(sigma_clim_squared, dtype=np.float64))
        obs = np.asarray(observed_temp, dtype=np.float64)
        n_samples = len(obs)

        if n_samples < 2:
            raise ValueError(f"Insufficient training samples ({n_samples}) for EMOS optimization")

        # Baseline 1: Raw ensemble (a=0, b=1, c=0, d=1, without variance floor)
        raw_sigma = np.sqrt(np.maximum(1e-8, ens_var))
        crps_raw = float(np.mean(gaussian_crps(obs, ens_mean, raw_sigma)))

        # Baseline 2: Pure climatology
        if mu_clim is not None and sigma_clim is not None:
            m_c = np.asarray(mu_clim, dtype=np.float64)
            s_c = np.asarray(sigma_clim, dtype=np.float64)
            crps_clim = float(np.mean(gaussian_crps(obs, m_c, s_c)))
        else:
            # Estimate empirical climatology from training observations
            m_c = float(np.mean(obs))
            s_c = float(np.std(obs, ddof=1))
            crps_clim = float(np.mean(gaussian_crps(obs, m_c, s_c)))

        # Objective function wrapper
        def objective(p):
            return emos_crps_loss(
                params=p,
                ensemble_mean=ens_mean,
                ensemble_variance=ens_var,
                sigma_clim_squared=clim_var,
                observed_temp=obs,
                l2_lambda_d=self.l2_lambda_d,
            )

        best_opt_res = None
        best_loss = float("inf")
        restarts_used = 0

        # Multi-start loop: Attempt 0 = standard warm start (0,1,0,1) + O(1e-3)
        # Attempt 1..max_restarts = random perturbations
        total_attempts = 1 + self.max_restarts
        for attempt in range(total_attempts):
            if attempt == 0:
                # Physics warm start with tiny 1e-3 perturbation
                init_p = np.array([0.0, 1.0, 0.0, 1.0]) + self._rng.normal(0, 1e-3, 4)
            else:
                # Restart with slightly larger perturbation
                restarts_used += 1
                init_p = np.array([
                    self._rng.uniform(-1.0, 1.0),
                    self._rng.uniform(0.8, 1.2),
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(0.5, 1.5),
                ])

            opt_res = optimize.minimize(
                fun=objective,
                x0=init_p,
                method="L-BFGS-B",
                bounds=self.bounds,
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tolerance,
                    "gtol": 1e-5,
                },
            )

            if opt_res.success and np.isfinite(opt_res.fun) and opt_res.fun < best_loss:
                best_loss = opt_res.fun
                best_opt_res = opt_res

        # If all L-BFGS-B runs failed, fallback to the last run or identity parameters
        if best_opt_res is None:
            best_opt_res = opt_res

        success = bool(best_opt_res.success and np.isfinite(best_opt_res.fun))
        optimal_params = tuple(float(x) for x in best_opt_res.x)

        # In-sample pure CRPS (without regularization term)
        in_sample_crps = emos_crps_loss(
            params=optimal_params,
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=obs,
            l2_lambda_d=0.0,
        )

        crpss_raw = 1.0 - (in_sample_crps / crps_raw) if crps_raw > 0 else 0.0
        crpss_clim = 1.0 - (in_sample_crps / crps_clim) if crps_clim > 0 else 0.0

        grad_norm = float(np.linalg.norm(best_opt_res.jac)) if hasattr(best_opt_res, "jac") and best_opt_res.jac is not None else 0.0
        n_iters = int(getattr(best_opt_res, "nit", 0))
        n_evals = int(getattr(best_opt_res, "nfev", 0))

        return ModelTrainingDiagnostics(
            success=success,
            params=optimal_params,
            crps_in_sample=in_sample_crps,
            crps_raw_ensemble=crps_raw,
            crps_climatology=crps_clim,
            crpss_vs_raw=crpss_raw,
            crpss_vs_clim=crpss_clim,
            n_iterations=n_iters,
            n_evaluations=n_evals,
            grad_norm=grad_norm,
            restarts_used=restarts_used,
            sample_count=n_samples,
        )

    def fit(
        self,
        ensemble_mean: Union[np.ndarray, pd.Series],
        ensemble_variance: Union[np.ndarray, pd.Series],
        sigma_clim_squared: Union[np.ndarray, pd.Series],
        observed_temp: Union[np.ndarray, pd.Series],
        mu_clim: Optional[Union[np.ndarray, pd.Series]] = None,
        sigma_clim: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[GaussianEMOS, ModelTrainingDiagnostics]:
        """Fit and return a parameterized GaussianEMOS model with its training diagnostics."""
        diag = self.optimize(
            ensemble_mean=ensemble_mean,
            ensemble_variance=ensemble_variance,
            sigma_clim_squared=sigma_clim_squared,
            observed_temp=observed_temp,
            mu_clim=mu_clim,
            sigma_clim=sigma_clim,
        )
        a, b, c, d = diag.params
        model = GaussianEMOS(a=a, b=b, c=c, d=d)
        return model, diag
