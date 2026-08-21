"""End-to-End system test for Polymarket temperature prediction pipeline (Ticket #45).

Validates end-to-end execution:
Ingest/Data Setup -> Feature Engineering -> 40 EMOS Matrix Training -> Multi-layer Inference -> Triple Gate Validation.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.pipeline.config import ConfigManager, PipelineConfig
from src.pipeline.main_pipeline import MainPipeline, PipelineStage
from src.pipeline.health import HealthChecker, HealthStatus
from src.modeling.registry import ModelRegistry
from src.modeling.gaussian_emos import GaussianEMOS
from src.prediction.prediction_pipeline import PredictionPipeline


class TestEndToEndPipeline:
    """System-level E2E integration test suite."""

    @pytest.fixture
    def e2e_env(self, tmp_path):
        """Build isolated test environment with config and mock datasets."""
        raw_dir = tmp_path / "data" / "raw"
        processed_dir = tmp_path / "data" / "processed"
        models_dir = tmp_path / "data" / "models"
        db_dir = tmp_path / "data" / "db"
        predictions_db = db_dir / "predictions.db"

        for p in [raw_dir, processed_dir, models_dir, db_dir]:
            p.mkdir(parents=True, exist_ok=True)

        config = PipelineConfig()
        config.env = "test"
        config.data.stations = ["ZSPD", "KDEN"]
        config.data.raw_dir = str(raw_dir)
        config.data.processed_dir = str(processed_dir)
        config.data.models_dir = str(models_dir)
        config.data.db_dir = str(db_dir)
        config.data.predictions_db_path = str(predictions_db)
        
        # Populate pre-trained dummy 40-model matrix to test prediction & validation lifecycle
        registry = ModelRegistry(base_dir=str(models_dir))
        seasons = ["Spring", "Summer", "Autumn", "Winter"]
        max_lts = [6, 30, 54]
        min_lts = [24, 48]

        for st in ["ZSPD", "KDEN"]:
            for sea in seasons:
                for lt in max_lts:
                    model = GaussianEMOS(a=0.5, b=0.98, c=0.8, d=0.2)
                    registry.save_model(model=model, station_id=st, season=sea, target_type="Max", lead_hours=lt)
                for lt in min_lts:
                    model = GaussianEMOS(a=0.3, b=0.99, c=0.7, d=0.15)
                    registry.save_model(model=model, station_id=st, season=sea, target_type="Min", lead_hours=lt)

        return config, registry

    def test_e2e_health_check(self, e2e_env):
        """Verify health check accurately detects all components as HEALTHY."""
        config, _ = e2e_env
        checker = HealthChecker(config)
        report = checker.run_all_checks()
        assert report["overall_status"] == HealthStatus.HEALTHY
        assert report["components"]["models"]["models_found"] == 40

    def test_e2e_prediction_pipeline_execution(self, e2e_env):
        """Verify multi-layer prediction pipeline produces valid database records."""
        config, registry = e2e_env
        pred_pipeline = PredictionPipeline(
            model_registry=registry,
            db_path=config.data.predictions_db_path,
        )

        rec = pred_pipeline.predict_single(
            station_id="ZSPD",
            target_date="2026-08-21",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=32.5,
            ensemble_variance=2.1,
            sigma_clim_squared=1.5,
            current_temp=30.0,
        )

        assert rec is not None
        probs = rec.get_bin_probabilities_dict()
        assert len(probs) > 0
        assert pytest.approx(sum(probs.values()), 1e-4) == 1.0

        # Verify DB persistence
        conn = sqlite3.connect(config.data.predictions_db_path)
        df = pd.read_sql_query("SELECT * FROM market_predictions", conn)
        conn.close()
        assert len(df) >= 1
        assert df.iloc[0]["station_id"] == "ZSPD"
        assert df.iloc[0]["target_type"] == "max"

    def test_e2e_main_pipeline_full_run(self, e2e_env):
        """Verify MainPipeline orchestrates full execution flow and outputs markdown profile."""
        config, _ = e2e_env
        pipeline = MainPipeline(config)
        exec_result = pipeline.run_all(date_str="2026-08-21")

        assert exec_result.success is True
        assert len(exec_result.errors) == 0
        assert "Pipeline Execution Profile" in exec_result.markdown_report
        assert "Total Elapsed Time" in exec_result.markdown_report
