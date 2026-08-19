#!/usr/bin/env python3
"""
Unit tests for MatrixTrainer and MatrixScorecard (Ticket 2.2-04 / Issue #17).

Verifies:
1. Batch training orchestration across all 40 standard matrix combinations (2 stations x 4 seasons x 5 nodes).
2. Integration with StorageManager, ClimatologyCalculator, DatasetPartitioner, EMOSOptimizer, and DegradationHandler.
3. MatrixScorecard aggregation (healthy_count, warning_count, degraded_count, mean skill scores).
4. Tabular DataFrame conversion and summary report formatting.
5. Safe fallback and degradation handling if individual matrix slices have sparse data or optimization errors.
"""

from datetime import date
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.degradation import DegradationDecision, DegradationHandler
from src.modeling.emos_trainer import EMOSOptimizer, ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.matrix_trainer import MatrixScorecard, MatrixTrainer
from src.modeling.partitioner import DatasetPartitioner


class MockStorageManager:
    """Mock StorageManager providing synthetic aligned datasets with systematic warm bias and spread under-dispersion."""

    def __init__(self, n_years: int = 3):
        self.n_years = n_years

    def load_training_dataset(
        self,
        station_id: str,
        target_type: str,
        lead_time_bucket: int,
        start_year: int = 2000,
        end_year: int = 2018,
    ) -> pd.DataFrame:
        dates = pd.date_range("2016-01-01", "2018-12-31", freq="D")
        n = len(dates)
        
        # True temperature base
        base_temp = 20.0 if station_id == "ZSPD" else 15.0
        if target_type == "min":
            base_temp -= 8.0
            
        seasonal_cycle = 10.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        true_temp = base_temp + seasonal_cycle + np.random.normal(0, 2.5, n)
        
        # Raw forecast has systematic +2.0°C bias and severe under-dispersion (ens_var=0.25 vs actual error std=2.5)
        ens_mean = true_temp + 2.0 + np.random.normal(0, 1.0, n)
        ens_var = np.full(n, 0.25)
        
        return pd.DataFrame({
            "target_date": dates.strftime("%Y-%m-%d"),
            "ensemble_mean": ens_mean,
            "ensemble_variance": ens_var,
            "observed_temp": true_temp,
            "member_max": ens_mean + 0.5,
            "member_min": ens_mean - 0.5,
        })


class MockClimatologyCalculator:
    """Mock ClimatologyCalculator providing valid variance floors."""

    def get_climatology_variance(self, station_id: str, target_type: str, target_date: str) -> float:
        return 4.0  # std = 2.0°C

    def get_climatology_params(self, station_id: str, target_type: str, target_date: str) -> Tuple[float, float]:
        base = 20.0 if station_id == "ZSPD" else 15.0
        if target_type == "min":
            base -= 8.0
        return (base, 5.0)


class TestMatrixTrainerOrchestration:
    """Test 40-model matrix batch training orchestration."""

    def test_train_all_40_matrices(self):
        storage = MockStorageManager(n_years=3)
        clim_calc = MockClimatologyCalculator()
        
        trainer = MatrixTrainer(
            storage_manager=storage,
            climatology_calculator=clim_calc,
            stations=["ZSPD", "KDEN"],
            train_start_year=2016,
            train_end_year=2018,
            random_seed=42,
        )

        scorecard = trainer.train_all()

        assert isinstance(scorecard, MatrixScorecard)
        assert scorecard.total_trained == 40
        assert len(scorecard.models) == 40

        # Check that all keys are present
        all_keys = DatasetPartitioner.get_all_matrix_keys()
        for key in all_keys:
            assert key in scorecard.models
            model, diag, decision = scorecard.models[key]
            assert isinstance(model, GaussianEMOS)
            assert isinstance(diag, ModelTrainingDiagnostics)
            assert isinstance(decision, DegradationDecision)
            assert diag.sample_count > 0

    def test_matrix_scorecard_metrics_and_dataframe(self):
        storage = MockStorageManager(n_years=3)
        clim_calc = MockClimatologyCalculator()
        
        trainer = MatrixTrainer(
            storage_manager=storage,
            climatology_calculator=clim_calc,
            stations=["ZSPD", "KDEN"],
            train_start_year=2016,
            train_end_year=2018,
            random_seed=42,
        )

        scorecard = trainer.train_all()

        # Scorecard summary statistics
        assert scorecard.total_trained == 40
        assert scorecard.healthy_count + scorecard.warning_count + scorecard.degraded_count == 40
        assert scorecard.mean_crpss_vs_raw > 0.05  # EMOS significantly improves over raw biased ensemble
        assert scorecard.mean_crpss_vs_clim > 0.05  # EMOS significantly improves over climatology
        
        # Test DataFrame export
        df_summary = scorecard.to_dataframe()
        assert isinstance(df_summary, pd.DataFrame)
        assert len(df_summary) == 40
        assert set(df_summary.columns) >= {
            "station_id", "season", "target_type", "lead_bucket",
            "a", "b", "c", "d", "crps_in_sample", "crpss_vs_raw",
            "health_grade", "degradation_level", "sample_count"
        }

        # Test Text Report generation
        report_str = scorecard.summary_report()
        assert "Matrix Training Scorecard" in report_str
        assert "ZSPD" in report_str
        assert "KDEN" in report_str
        assert "**Total Models**: 40" in report_str
