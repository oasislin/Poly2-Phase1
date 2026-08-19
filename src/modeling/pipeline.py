#!/usr/bin/env python3
"""
TrainingPipeline: End-to-end training and validation orchestrator for Phase 1B (Ticket 2.3-03 / Issue #22).

Coordinates:
    1. Climatology baseline fitting (2000-2018 OOS observations)
    2. 40-model matrix batch training (MatrixTrainer)
    3. Dense 6h-spaced grid interpolation (LeadTimeInterpolator)
    4. Standardized model persistence (ModelRegistry)
    5. Strict time-wall out-of-sample evaluation (ValidationEngine)
    6. Triple Acceptance Gate verification & Markdown report generation (ReportGenerator)
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.data_processing.storage_manager import StorageManager
from src.modeling.climatology import ClimatologyCalculator
from src.modeling.crps import gaussian_crps
from src.modeling.degradation import DegradationHandler
from src.modeling.emos_trainer import EMOSOptimizer
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.interpolator import LeadTimeInterpolator
from src.modeling.matrix_trainer import MatrixScorecard, MatrixTrainer
from src.modeling.partitioner import DatasetPartitioner
from src.modeling.registry import ModelRegistry
from src.modeling.report_generator import AcceptanceReport, ReportGenerator
from src.modeling.validation_engine import ValidationEngine, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Encapsulates all artifacts produced by an end-to-end training pipeline run."""

    scorecard: MatrixScorecard
    validation_results: Dict[Tuple[str, str, str, int], ValidationResult]
    acceptance_report: AcceptanceReport
    report_path: Optional[Path] = None


class TrainingPipeline:
    """Master orchestrator for Phase 1B model training, interpolation, persistence, and acceptance testing."""

    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        climatology_calculator: Optional[ClimatologyCalculator] = None,
        model_registry: Optional[ModelRegistry] = None,
        stations: Optional[Sequence[str]] = None,
        train_start_year: int = 2000,
        train_end_year: int = 2018,
        val_start_year: int = 2019,
        val_end_year: int = 2019,
        l2_lambda_d: float = 1e-3,
        report_dir: Union[str, Path] = "reports",
        verify_gates: bool = True,
        random_seed: Optional[int] = 42,
    ):
        self.storage_manager = storage_manager or StorageManager()
        self.climatology_calculator = climatology_calculator or ClimatologyCalculator(
            train_start_year=train_start_year,
            train_end_year=train_end_year,
        )
        self.model_registry = model_registry or ModelRegistry()
        self.stations = list(stations or ["ZSPD", "KDEN"])
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.val_start_year = val_start_year
        self.val_end_year = val_end_year
        self.l2_lambda_d = l2_lambda_d
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.verify_gates = verify_gates
        self.random_seed = random_seed

        self.partitioner = DatasetPartitioner()
        self.interpolator = LeadTimeInterpolator()
        self.report_generator = ReportGenerator()

    def run(self) -> PipelineResult:
        """Execute full training and validation lifecycle with modular step runners."""
        logger.info("=== Phase 1B Training Pipeline Started ===")
        self._ensure_climatology_fitted()

        # Step 1: Matrix batch training & persistence
        scorecard = self._train_matrix_models()
        self.model_registry.save_scorecard(scorecard, build_dense_grid=True)

        # Step 2: Out-of-sample validation on holdout period
        val_engine, val_results, val_dict = self._run_out_of_sample_validation()

        # Step 3: Gate 2 virtual holdout evaluation (30h)
        crps_virt, crps_real, pit_virt = self._evaluate_virtual_holdout_30h(val_engine)

        # Step 4: Triple acceptance gate report generation
        acceptance_report, report_path = self._generate_acceptance_report(
            val_results_dict=val_dict,
            crps_virt_30h=crps_virt,
            crps_real_30h=crps_real,
            pit_virt_30h=pit_virt,
        )

        return PipelineResult(
            scorecard=scorecard,
            validation_results=val_results,
            acceptance_report=acceptance_report,
            report_path=report_path,
        )

    def _ensure_climatology_fitted(self) -> None:
        """Fit climatology calculator if not already fit."""
        if hasattr(self.climatology_calculator, "is_fitted") and not self.climatology_calculator.is_fitted:
            logger.info("Fitting ClimatologyCalculator on historical database...")
            self.climatology_calculator.fit_from_db(station_ids=self.stations)

    def _train_matrix_models(self) -> MatrixScorecard:
        """Execute batch training of all 40 matrix subsets."""
        logger.info("Batch training 40-model matrix via MatrixTrainer...")
        trainer = MatrixTrainer(
            storage_manager=self.storage_manager,
            climatology_calculator=self.climatology_calculator,
            stations=self.stations,
            train_start_year=self.train_start_year,
            train_end_year=self.train_end_year,
            l2_lambda_d=self.l2_lambda_d,
            random_seed=self.random_seed,
        )
        return trainer.train_all()

    def _run_out_of_sample_validation(
        self,
    ) -> Tuple[ValidationEngine, Dict[Tuple[str, str, str, int], ValidationResult], Dict[str, ValidationResult]]:
        """Run validation engine across all matrix partitions."""
        logger.info("Running out-of-sample validation on [%d, %d]...", self.val_start_year, self.val_end_year)
        val_engine = ValidationEngine(
            storage_manager=self.storage_manager,
            climatology_calculator=self.climatology_calculator,
            model_registry=self.model_registry,
            train_start_year=self.train_start_year,
            train_end_year=self.train_end_year,
            val_start_year=self.val_start_year,
            val_end_year=self.val_end_year,
        )

        val_results: Dict[Tuple[str, str, str, int], ValidationResult] = {}
        val_dict: Dict[str, ValidationResult] = {}

        for (station, season, target_type, lead_bucket) in self.partitioner.get_all_matrix_keys():
            df_val_full = val_engine.load_val_data(station, target_type, lead_bucket)
            seasonal_splits = self.partitioner.split_by_season(df_val_full, date_col="target_date")
            df_val_season = seasonal_splits[season]

            val_res = val_engine.evaluate_slice(
                station_id=station,
                target_type=target_type,
                lead_hours=lead_bucket,
                df_val=df_val_season,
            )
            val_results[(station, season, target_type, lead_bucket)] = val_res
            val_dict[f"{station}_{season}_{target_type}_{lead_bucket}h"] = val_res

        return val_engine, val_results, val_dict

    def _evaluate_virtual_holdout_30h(
        self,
        val_engine: ValidationEngine,
    ) -> Tuple[float, float, np.ndarray]:
        """Evaluate 30h virtual holdout interpolation accuracy and PIT (Gate 2)."""
        st_ref = self.stations[0]
        df_30_val = val_engine.load_val_data(st_ref, "max", 30)

        # Retrieve anchor models and interpolate virtual 30h model via LeadTimeInterpolator
        m6 = self.model_registry.get_model(st_ref, target_date="2019-01-01", target_type="max", lead_hours=6)
        m54 = self.model_registry.get_model(st_ref, target_date="2019-01-01", target_type="max", lead_hours=54)
        m30_real = self.model_registry.get_model(st_ref, target_date="2019-01-01", target_type="max", lead_hours=30)
        m30_virt = self.interpolator.get_model_at_lead("max", 30, {6: m6, 54: m54})

        clim_vars_30 = np.array([
            self.climatology_calculator.get_climatology_variance(st_ref, "max", d)
            for d in df_30_val["target_date"]
        ])
        v_mu, v_sig = m30_virt.compute_params(
            df_30_val["ensemble_mean"].values,
            df_30_val["ensemble_variance"].values,
            clim_vars_30,
        )
        r_mu, r_sig = m30_real.compute_params(
            df_30_val["ensemble_mean"].values,
            df_30_val["ensemble_variance"].values,
            clim_vars_30,
        )

        obs = df_30_val["observed_temp"].values
        crps_virt = float(np.mean(gaussian_crps(obs, v_mu, v_sig)))
        crps_real = float(np.mean(gaussian_crps(obs, r_mu, r_sig)))

        z_virt = (obs - v_mu) / np.maximum(1e-8, v_sig)
        pit_virt = stats.norm.cdf(z_virt)

        return crps_virt, crps_real, pit_virt

    def _generate_acceptance_report(
        self,
        val_results_dict: Dict[str, ValidationResult],
        crps_virt_30h: float,
        crps_real_30h: float,
        pit_virt_30h: np.ndarray,
    ) -> Tuple[AcceptanceReport, Path]:
        """Generate and save Triple Acceptance Report markdown."""
        report = self.report_generator.generate_report(
            val_results=val_results_dict,
            crps_virt_30h=crps_virt_30h,
            crps_real_30h=crps_real_30h,
            pit_virt_30h=pit_virt_30h,
        )
        report_path = self.report_dir / "phase1b_acceptance_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        logger.info("Acceptance report saved to %s", report_path)
        return report, report_path
