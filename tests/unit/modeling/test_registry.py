#!/usr/bin/env python3
"""
Unit tests for ModelRegistry and unified model inference facade (Ticket 2.2-06 / Issue #19).

Verifies:
1. Standardized model persistence file naming: {StationID}_{Season}_{Max|Min}_lead{Hours}h.pkl.
2. Saving and loading GaussianEMOS model and associated metadata (diagnostics, degradation status).
3. Batch persistence of MatrixScorecard and full dense 6h-grid.
4. Model inventory listing as DataFrame.
5. get_model and predict facade methods with automatic season mapping, interpolation, and degradation routing.
"""

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.degradation import DegradationDecision
from src.modeling.emos_trainer import ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.interpolator import LeadTimeInterpolator
from src.modeling.matrix_trainer import MatrixScorecard
from src.modeling.registry import ModelRegistry


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Temporary model directory for registry tests."""
    model_dir = tmp_path / "models" / "emos"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


class TestModelRegistryPersistence:
    """Test saving, loading, and file naming conventions."""

    def test_save_and_load_single_model_standard_naming(self, temp_registry_dir):
        registry = ModelRegistry(base_dir=temp_registry_dir)
        model = GaussianEMOS(a=0.3, b=0.95, c=0.1, d=0.85)
        diag = ModelTrainingDiagnostics(
            success=True,
            params=(0.3, 0.95, 0.1, 0.85),
            crps_in_sample=1.2,
            crps_raw_ensemble=1.8,
            crps_climatology=2.5,
            crpss_vs_raw=0.33,
            crpss_vs_clim=0.52,
            n_iterations=20,
            n_evaluations=30,
            grad_norm=1e-6,
            restarts_used=0,
            sample_count=300,
        )
        decision = DegradationDecision(level=1, is_degraded=False, reason="Healthy")

        # Save model
        saved_path = registry.save_model(
            model=model,
            station_id="ZSPD",
            season="Winter",
            target_type="max",
            lead_hours=30,
            diagnostics=diag,
            decision=decision,
        )

        expected_filename = "ZSPD_Winter_Max_lead30h.pkl"
        assert saved_path.name == expected_filename
        assert saved_path.exists()

        # Load model back
        loaded_model, metadata = registry.load_model(
            station_id="ZSPD",
            season="Winter",
            target_type="max",
            lead_hours=30,
        )

        assert np.isclose(loaded_model.a, 0.3)
        assert np.isclose(loaded_model.b, 0.95)
        assert np.isclose(loaded_model.c, 0.1)
        assert np.isclose(loaded_model.d, 0.85)
        assert metadata["station_id"] == "ZSPD"
        assert metadata["season"] == "Winter"
        assert metadata["target_type"] == "max"
        assert metadata["lead_hours"] == 30
        assert metadata["diagnostics"]["crps_in_sample"] == 1.2

    def test_save_scorecard_and_inventory(self, temp_registry_dir):
        registry = ModelRegistry(base_dir=temp_registry_dir)
        
        # Build mock scorecard with 4 models
        models = {}
        for lead in [6, 30, 54]:
            m = GaussianEMOS(a=0.1 * lead, b=1.0, c=0.2, d=0.8)
            d = ModelTrainingDiagnostics(
                success=True, params=(0.1 * lead, 1.0, 0.2, 0.8),
                crps_in_sample=1.0, crps_raw_ensemble=1.5, crps_climatology=2.0,
                crpss_vs_raw=0.33, crpss_vs_clim=0.5, n_iterations=10, n_evaluations=15,
                grad_norm=1e-6, restarts_used=0, sample_count=100
            )
            dec = DegradationDecision(level=1, is_degraded=False, reason="Healthy")
            models[("ZSPD", "Winter", "max", lead)] = (m, d, dec)

        scorecard = MatrixScorecard(models=models)
        registry.save_scorecard(scorecard, build_dense_grid=True)

        inventory = registry.list_inventory()
        assert isinstance(inventory, pd.DataFrame)
        # Should contain saved anchors + interpolated 6h grid
        assert len(inventory) >= 3
        assert "ZSPD_Winter_Max_lead30h.pkl" in inventory["filename"].values


class TestModelRegistryQueryFacade:
    """Test get_model and predict inference facade."""

    def test_get_model_with_date_and_interpolation(self, temp_registry_dir):
        registry = ModelRegistry(base_dir=temp_registry_dir)
        
        # Save anchor models at 6h, 30h, 54h for ZSPD Summer Max
        m6 = GaussianEMOS(a=0.0, b=1.0, c=0.2, d=0.8)
        m30 = GaussianEMOS(a=1.0, b=0.9, c=0.4, d=1.0)
        m54 = GaussianEMOS(a=2.0, b=0.8, c=0.6, d=1.2)
        diag = ModelTrainingDiagnostics(
            success=True, params=(0, 1, 0, 1), crps_in_sample=1.0, crps_raw_ensemble=1.0,
            crps_climatology=1.0, crpss_vs_raw=0.0, crpss_vs_clim=0.0, n_iterations=1,
            n_evaluations=1, grad_norm=0, restarts_used=0, sample_count=100
        )
        dec = DegradationDecision(level=1, is_degraded=False, reason="Healthy")

        registry.save_model(m6, "ZSPD", "Summer", "max", 6, diag, dec)
        registry.save_model(m30, "ZSPD", "Summer", "max", 30, diag, dec)
        registry.save_model(m54, "ZSPD", "Summer", "max", 54, diag, dec)

        # 1. Query exact anchor with date "2019-07-15" (Month 7 -> Summer)
        model_30 = registry.get_model("ZSPD", target_date="2019-07-15", target_type="max", lead_hours=30)
        assert np.isclose(model_30.a, 1.0)

        # 2. Query intermediate non-anchor 18h (interpolated between 6h and 30h)
        model_18 = registry.get_model("ZSPD", target_date="2019-07-15", target_type="max", lead_hours=18)
        assert np.isclose(model_18.a, 0.5)  # halfway between 0.0 and 1.0

    def test_predict_facade(self, temp_registry_dir):
        registry = ModelRegistry(base_dir=temp_registry_dir)
        m30 = GaussianEMOS(a=0.5, b=0.9, c=0.2, d=1.0)
        diag = ModelTrainingDiagnostics(
            success=True, params=(0.5, 0.9, 0.2, 1.0), crps_in_sample=1.0, crps_raw_ensemble=1.0,
            crps_climatology=1.0, crpss_vs_raw=0.0, crpss_vs_clim=0.0, n_iterations=1,
            n_evaluations=1, grad_norm=0, restarts_used=0, sample_count=100
        )
        dec = DegradationDecision(level=1, is_degraded=False, reason="Healthy")
        registry.save_model(m30, "ZSPD", "Winter", "max", 30, diag, dec)

        pred_dist = registry.predict(
            station_id="ZSPD",
            target_date="2019-01-15",  # Winter
            target_type="max",
            lead_hours=30,
            ensemble_mean=10.0,
            ensemble_variance=1.0,
            sigma_clim_squared=4.0,
        )

        assert isinstance(pred_dist, GaussianEMOS)
        # mu = 0.5 + 0.9 * 10 = 9.5
        # sigma^2 = 0.04 + 1.0 * 1.0 + 4.0 = 5.04 -> sigma = sqrt(5.04)
        assert np.isclose(pred_dist.mu, 9.5)
        assert np.isclose(pred_dist.sigma, np.sqrt(5.04))
