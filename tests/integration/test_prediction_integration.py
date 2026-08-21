#!/usr/bin/env python3
"""
Integration tests for prediction CLI and multi-station end-to-end workflows (Ticket 3.5-02 / Issue #31).

Verifies:
1. CLI execution via main entrypoint for Shanghai (ZSPD) Max temperature prediction.
2. CLI execution for Denver (KDEN) Min temperature prediction with Fahrenheit output.
3. CLI `--json` output flag producing valid, parseable JSON payload to stdout.
4. Database persistence and queryability after CLI prediction runs.
5. Multi-station end-to-end integration with ModelRegistry and ClimatologyCalculator.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.registry import ModelRegistry
from src.prediction.prediction_pipeline import PredictionPipeline
from scripts.run_predictions import build_parser, run_cli


@pytest.fixture
def setup_models_and_db(tmp_path):
    """Setup temporary models and database directories with trained anchor models."""
    models_dir = tmp_path / "models"
    db_path = tmp_path / "predictions.db"
    registry = ModelRegistry(base_dir=models_dir)

    # Save representative anchor models for ZSPD and KDEN
    # 1. ZSPD Summer Max 30h
    registry.save_model(
        model=GaussianEMOS(a=1.0, b=0.95, c=0.4, d=0.8),
        station_id="ZSPD",
        season="Summer",
        target_type="max",
        lead_hours=30,
    )
    # 2. ZSPD Winter Min 24h
    registry.save_model(
        model=GaussianEMOS(a=-1.0, b=1.05, c=0.5, d=0.6),
        station_id="ZSPD",
        season="Winter",
        target_type="min",
        lead_hours=24,
    )
    # 3. KDEN Summer Max 30h
    registry.save_model(
        model=GaussianEMOS(a=0.5, b=0.98, c=0.3, d=0.7),
        station_id="KDEN",
        season="Summer",
        target_type="max",
        lead_hours=30,
    )
    # 4. KDEN Winter Min 24h
    registry.save_model(
        model=GaussianEMOS(a=-1.5, b=1.00, c=0.6, d=0.8),
        station_id="KDEN",
        season="Winter",
        target_type="min",
        lead_hours=24,
    )

    return {
        "models_dir": models_dir,
        "db_path": db_path,
        "registry": registry,
    }


class TestPredictionCLIIntegration:
    """Test run_predictions.py CLI execution and output formatting."""

    def test_cli_shanghai_max_prediction_run(self, setup_models_and_db, capsys):
        """Run CLI prediction for Shanghai Summer Max."""
        args = [
            "--station", "ZSPD",
            "--date", "2019-07-15",
            "--type", "max",
            "--lead-time", "30",
            "--ens-mean", "30.0",
            "--ens-var", "2.0",
            "--current-temp", "28.5",
            "--obs-time", "12:00",
            "--models-dir", str(setup_models_and_db["models_dir"]),
            "--db-path", str(setup_models_and_db["db_path"]),
            "--save-db",
        ]

        exit_code = run_cli(args)
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Polymarket Temperature Prediction Report" in captured.out
        assert "ZSPD" in captured.out
        assert "29.50°C" in captured.out  # Predicted mu = 1.0 + 0.95*30 = 29.5
        assert "Total Probability Sum: 100.0000%" in captured.out

        # Verify DB persistence
        pipeline = PredictionPipeline(
            model_registry=setup_models_and_db["registry"],
            db_path=setup_models_and_db["db_path"],
        )
        history = pipeline.get_history("ZSPD", target_date="2019-07-15")
        assert len(history) == 1
        assert np.isclose(history.iloc[0]["predicted_mu"], 29.5, atol=1e-5)

    def test_cli_denver_fahrenheit_json_output(self, setup_models_and_db, capsys):
        """Run CLI with --json flag for Denver Winter Min temperature."""
        args = [
            "--station", "KDEN",
            "--date", "2019-01-15",
            "--type", "min",
            "--lead-time", "24",
            "--ens-mean", "-5.0",  # -5°C ≈ 23°F
            "--ens-var", "2.0",
            "--current-temp", "-4.0",
            "--models-dir", str(setup_models_and_db["models_dir"]),
            "--db-path", str(setup_models_and_db["db_path"]),
            "--json",
        ]

        exit_code = run_cli(args)
        assert exit_code == 0

        captured = capsys.readouterr()
        json_output = json.loads(captured.out)

        assert json_output["station_id"] == "KDEN"
        assert json_output["target_type"] == "min"
        assert "predicted_mu" in json_output
        assert "bin_probabilities" in json_output
        # Denver bins are Fahrenheit
        labels = list(json_output["bin_probabilities"].keys())
        assert any("°F" in lbl for lbl in labels)
        total_p = sum(json_output["bin_probabilities"].values())
        assert np.isclose(total_p, 1.0, atol=1e-5)

    def test_multi_station_batch_integration(self, setup_models_and_db):
        """Test multi-station batch inference through PredictionPipeline."""
        pipeline = PredictionPipeline(
            model_registry=setup_models_and_db["registry"],
            db_path=setup_models_and_db["db_path"],
        )

        batch_df = pd.DataFrame({
            "station_id": ["ZSPD", "KDEN"],
            "target_date": ["2019-07-15", "2019-07-15"],
            "target_type": ["max", "max"],
            "lead_time_hours": [30, 30],
            "ensemble_mean": [30.0, 25.0],
            "ensemble_variance": [2.0, 1.5],
            "current_temp": [28.0, 22.0],
        })

        out_df = pipeline.predict_batch(batch_df, save_to_db=True)
        assert len(out_df) == 2
        assert out_df.iloc[0]["station_id"] == "ZSPD"
        assert out_df.iloc[1]["station_id"] == "KDEN"

        # Check records in SQLite
        history_zspd = pipeline.get_history("ZSPD", target_date="2019-07-15")
        history_kden = pipeline.get_history("KDEN", target_date="2019-07-15")
        assert len(history_zspd) >= 1
        assert len(history_kden) >= 1
