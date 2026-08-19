#!/usr/bin/env python3
"""
MatrixTrainer: 40-model matrix batch training orchestrator and MatrixScorecard aggregator (Ticket 2.2-04 / Issue #17).

Coordinates:
    - 2 stations: ZSPD (Shanghai), KDEN (Denver)
    - 4 seasons: Spring, Summer, Autumn, Winter
    - 5 discrete training nodes: Max {54h, 30h, 6h}, Min {48h, 24h}
    Total = 40 independent, localized EMOS models.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.crps import gaussian_crps
from src.modeling.degradation import DegradationDecision, DegradationHandler
from src.modeling.emos_trainer import EMOSOptimizer, ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.partitioner import DatasetPartitioner

logger = logging.getLogger(__name__)


@dataclass
class MatrixScorecard:
    """Aggregated scorecard and inventory for all 40 matrix-trained EMOS models."""

    models: Dict[Tuple[str, str, str, int], Tuple[GaussianEMOS, ModelTrainingDiagnostics, DegradationDecision]]
    total_trained: int = 0
    healthy_count: int = 0
    warning_count: int = 0
    degraded_count: int = 0
    mean_crpss_vs_raw: float = 0.0
    mean_crpss_vs_clim: float = 0.0

    def __post_init__(self):
        self.total_trained = len(self.models)
        if self.total_trained > 0:
            h_cnt, w_cnt, d_cnt = 0, 0, 0
            raw_skills, clim_skills = [], []

            for _, (_, diag, decision) in self.models.items():
                if decision.is_degraded:
                    d_cnt += 1
                elif diag.health_grade == "WARNING":
                    w_cnt += 1
                else:
                    h_cnt += 1

                raw_skills.append(diag.crpss_vs_raw)
                clim_skills.append(diag.crpss_vs_clim)

            self.healthy_count = h_cnt
            self.warning_count = w_cnt
            self.degraded_count = d_cnt
            self.mean_crpss_vs_raw = float(np.mean(raw_skills))
            self.mean_crpss_vs_clim = float(np.mean(clim_skills))

    def to_dataframe(self) -> pd.DataFrame:
        """Export all 40 model diagnostics and parameters to a tabular DataFrame."""
        records = []
        for (station, season, target_type, lead_bucket), (model, diag, decision) in sorted(self.models.items()):
            a, b, c, d = diag.params
            records.append({
                "station_id": station,
                "season": season,
                "target_type": target_type,
                "lead_bucket": lead_bucket,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "crps_in_sample": diag.crps_in_sample,
                "crps_raw_ensemble": diag.crps_raw_ensemble,
                "crps_climatology": diag.crps_climatology,
                "crpss_vs_raw": diag.crpss_vs_raw,
                "crpss_vs_clim": diag.crpss_vs_clim,
                "health_grade": diag.health_grade,
                "degradation_level": decision.level,
                "is_degraded": decision.is_degraded,
                "n_iterations": diag.n_iterations,
                "sample_count": diag.sample_count,
                "warnings": "; ".join(diag.warnings) if diag.warnings else "None",
            })
        return pd.DataFrame(records)

    def summary_report(self) -> str:
        """Generate a clean markdown summary report of the 40-model matrix training results."""
        df = self.to_dataframe()
        lines = [
            "# Matrix Training Scorecard (40 Models)",
            f"- **Total Models**: {self.total_trained}",
            f"- **Healthy (Level 1)**: {self.healthy_count} ({self.healthy_count / self.total_trained:.1%})",
            f"- **Warnings (Soft Alert)**: {self.warning_count} ({self.warning_count / self.total_trained:.1%})",
            f"- **Degraded (Level 2 Climatology)**: {self.degraded_count} ({self.degraded_count / self.total_trained:.1%})",
            f"- **Average In-Sample CRPSS vs Raw Ensemble**: {self.mean_crpss_vs_raw:+.2%}",
            f"- **Average In-Sample CRPSS vs Climatology**: {self.mean_crpss_vs_clim:+.2%}",
            "",
            "## Summary by Station",
        ]
        for station, st_df in df.groupby("station_id"):
            lines.append(f"### Station: {station}")
            lines.append(f"- Mean CRPSS vs Raw: {st_df['crpss_vs_raw'].mean():+.2%}")
            lines.append(f"- Mean CRPSS vs Climatology: {st_df['crpss_vs_clim'].mean():+.2%}")
            lines.append(f"- Healthy: {(st_df['health_grade'] == 'HEALTHY').sum()} / {len(st_df)}")
            lines.append("")

        return "\n".join(lines)


class MatrixTrainer:
    """Batch training engine orchestrating the 40-model parameter estimation matrix."""

    def __init__(
        self,
        storage_manager: Any,
        climatology_calculator: Any,
        stations: Optional[Sequence[str]] = None,
        train_start_year: int = 2000,
        train_end_year: int = 2018,
        l2_lambda_d: float = 1e-3,
        random_seed: Optional[int] = 42,
    ):
        self.storage_manager = storage_manager
        self.climatology_calculator = climatology_calculator
        self.stations = list(stations or ["ZSPD", "KDEN"])
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.l2_lambda_d = l2_lambda_d
        self.random_seed = random_seed

        self.optimizer = EMOSOptimizer(l2_lambda_d=l2_lambda_d, random_seed=random_seed)
        self.degradation_handler = DegradationHandler()
        self.partitioner = DatasetPartitioner()

    def train_slice(
        self,
        station_id: str,
        season: str,
        target_type: str,
        lead_bucket: int,
        df_slice: pd.DataFrame,
    ) -> Tuple[GaussianEMOS, ModelTrainingDiagnostics, DegradationDecision]:
        """Train a single EMOS model for one specific matrix partition slice."""
        n_samples = len(df_slice)

        if n_samples < self.degradation_handler.min_sample_count:
            # Insufficient samples -> construct identity model and degrade to Level 2
            fallback_model = GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0)
            diag = ModelTrainingDiagnostics(
                success=False,
                params=(0.0, 1.0, 0.0, 1.0),
                crps_in_sample=0.0,
                crps_raw_ensemble=0.0,
                crps_climatology=0.0,
                crpss_vs_raw=0.0,
                crpss_vs_clim=0.0,
                n_iterations=0,
                n_evaluations=0,
                grad_norm=0.0,
                restarts_used=0,
                sample_count=n_samples,
            )
            decision = self.degradation_handler.evaluate(diag)
            return fallback_model, diag, decision

        # Extract climatology variance floor and params for each sample
        clim_vars = np.array([
            self.climatology_calculator.get_climatology_variance(station_id, target_type, d)
            for d in df_slice["target_date"]
        ])
        clim_params = [
            self.climatology_calculator.get_climatology_params(station_id, target_type, d)
            for d in df_slice["target_date"]
        ]
        mu_clim = np.array([p[0] for p in clim_params])
        sigma_clim = np.array([p[1] for p in clim_params])

        # Run L-BFGS-B optimization
        model, diag = self.optimizer.fit(
            ensemble_mean=df_slice["ensemble_mean"],
            ensemble_variance=df_slice["ensemble_variance"],
            sigma_clim_squared=clim_vars,
            observed_temp=df_slice["observed_temp"],
            mu_clim=mu_clim,
            sigma_clim=sigma_clim,
        )

        # Compute sample-level CRPS for paired statistical testing in degradation handler
        pred_mu, pred_sigma = model.compute_params(
            ensemble_mean=df_slice["ensemble_mean"].values,
            ensemble_variance=df_slice["ensemble_variance"].values,
            sigma_clim_squared=clim_vars,
        )
        emos_sample_crps = gaussian_crps(df_slice["observed_temp"].values, pred_mu, pred_sigma)
        clim_sample_crps = gaussian_crps(df_slice["observed_temp"].values, mu_clim, sigma_clim)

        # Evaluate degradation
        decision = self.degradation_handler.evaluate(
            diagnostics=diag,
            emos_sample_crps=emos_sample_crps,
            clim_sample_crps=clim_sample_crps,
        )

        return model, diag, decision

    def train_all(self) -> MatrixScorecard:
        """Batch train all 40 matrix subsets across all configured stations."""
        models: Dict[Tuple[str, str, str, int], Tuple[GaussianEMOS, ModelTrainingDiagnostics, DegradationDecision]] = {}

        for station in self.stations:
            for target_type in ["max", "min"]:
                lead_nodes = self.partitioner.get_lead_time_nodes(target_type)
                for lead_bucket in lead_nodes:
                    # Load full training dataset for (station, target_type, lead_bucket)
                    df_full = self.storage_manager.load_training_dataset(
                        station_id=station,
                        target_type=target_type,
                        lead_time_bucket=lead_bucket,
                        start_year=self.train_start_year,
                        end_year=self.train_end_year,
                    )

                    # Split into 4 seasonal DataFrames
                    seasonal_splits = self.partitioner.split_by_season(df_full, date_col="target_date")

                    for season in ["Spring", "Summer", "Autumn", "Winter"]:
                        df_season = seasonal_splits[season]
                        model, diag, decision = self.train_slice(
                            station_id=station,
                            season=season,
                            target_type=target_type,
                            lead_bucket=lead_bucket,
                            df_slice=df_season,
                        )
                        models[(station, season, target_type, lead_bucket)] = (model, diag, decision)

        return MatrixScorecard(models=models)
