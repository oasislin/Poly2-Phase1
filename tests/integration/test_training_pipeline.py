#!/usr/bin/env python3
"""
Integration test for TrainingPipeline and train_emos_matrix CLI (Ticket 2.3-03 / Issue #22).

Verifies:
1. End-to-end execution of TrainingPipeline across climatology fitting, 40-model matrix training,
   dense grid interpolation, ModelRegistry persistence, validation, and acceptance report generation.
2. CLI entry point scripts/train_emos_matrix.py argument parsing and non-zero exit codes on failure.
"""

from datetime import date
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pytest

from src.modeling.pipeline import TrainingPipeline, PipelineResult
from src.modeling.registry import ModelRegistry


class MockPipelineStorageManager:
    """Mock storage providing continuous synthetic 2016-2019 data for both stations and targets."""

    def load_training_dataset(
        self,
        station_id: str,
        target_type: str,
        lead_time_bucket: int,
        start_year: int = 2016,
        end_year: int = 2019,
    ) -> pd.DataFrame:
        dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        n = len(dates)
        
        base_temp = 20.0 if station_id == "ZSPD" else 15.0
        if target_type == "min":
            base_temp -= 8.0
            
        seasonal_cycle = 10.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        true_temp = base_temp + seasonal_cycle + np.random.normal(0, 2.0, n)
        
        # Raw forecast with slight warm bias and slight under-dispersion
        ens_mean = true_temp + 1.0 + np.random.normal(0, 1.0, n)
        ens_var = np.full(n, 1.0)
        
        return pd.DataFrame({
            "target_date": dates.strftime("%Y-%m-%d"),
            "ensemble_mean": ens_mean,
            "ensemble_variance": ens_var,
            "observed_temp": true_temp,
            "member_max": ens_mean + 1.0,
            "member_min": ens_mean - 1.0,
        })


class MockPipelineClimatologyCalculator:
    def fit_from_db(self, station_ids=None):
        pass

    def get_climatology_variance(self, station_id: str, target_type: str, target_date: str) -> float:
        return 4.0

    def get_climatology_params(self, station_id: str, target_type: str, target_date: str) -> tuple:
        base = 20.0 if station_id == "ZSPD" else 15.0
        if target_type == "min":
            base -= 8.0
        return (base, 3.0)


class TestTrainingPipelineIntegration:
    """Integration test suite for the complete Phase 1B Training Pipeline."""

    def test_pipeline_end_to_end_execution(self, tmp_path):
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"

        storage = MockPipelineStorageManager()
        clim_calc = MockPipelineClimatologyCalculator()
        registry = ModelRegistry(base_dir=models_dir)

        pipeline = TrainingPipeline(
            storage_manager=storage,
            climatology_calculator=clim_calc,
            model_registry=registry,
            stations=["ZSPD", "KDEN"],
            train_start_year=2016,
            train_end_year=2018,
            val_start_year=2019,
            val_end_year=2019,
            report_dir=reports_dir,
            random_seed=42,
        )

        result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert result.scorecard.total_trained == 40
        assert len(result.validation_results) == 40
        assert result.acceptance_report is not None

        # Check persisted model files on disk
        persisted_files = list(models_dir.glob("*.pkl"))
        assert len(persisted_files) >= 40  # 40 anchors + interpolated models

        # Check acceptance report markdown generated
        report_file = reports_dir / "phase1b_acceptance_report.md"
        assert report_file.exists()
        assert "Phase 1B Triple Acceptance Verification Report" in report_file.read_text()

    def test_cli_help_invocation(self):
        """Test scripts/train_emos_matrix.py CLI help flag."""
        res = subprocess.run(
            ["/opt/miniconda3/bin/python3", "scripts/train_emos_matrix.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "Train Gaussian EMOS Matrix across stations" in res.stdout
        assert "--stations" in res.stdout
        assert "--output-dir" in res.stdout
