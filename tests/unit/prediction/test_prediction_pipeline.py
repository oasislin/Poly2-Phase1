#!/usr/bin/env python3
"""
Unit tests for PredictionPipeline (Ticket 3.5-01 / Issue #30).

Verifies:
1. End-to-end pipeline execution:
   Input features -> StaticPredictor -> DynamicCorrector -> ConstraintEnforcer -> BinConverter.
2. Complete PredictionRecord structure and metadata.
3. Database persistence: writing prediction records to SQLite and querying back.
4. Unobserved flow (no real-time temp) vs Observed flow (active dynamic truncation).
5. Batch DataFrame prediction and persistence.
6. Settlement winning bin determination via pipeline record.
"""

from datetime import datetime, timezone
import json
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.modeling.gaussian_emos import GaussianEMOS
from src.prediction.bin_converter import BinConverter, MarketBin
from src.prediction.constraint_enforcer import ConstraintEnforcer
from src.prediction.dynamic_corrector import DynamicCorrector
from src.prediction.prediction_pipeline import PredictionPipeline, PredictionRecord
from src.prediction.static_predictor import StaticPredictor, StaticPredictionResult


class MockRegistry:
    """Mock registry providing anchor models for ZSPD and KDEN."""

    def __init__(self):
        self.models = {
            ("ZSPD", "Jja", "max", 30): GaussianEMOS(a=1.0, b=0.95, c=0.4, d=0.8),
            ("ZSPD", "Djf", "min", 24): GaussianEMOS(a=-1.0, b=1.05, c=0.5, d=0.6),
            ("KDEN", "Jja", "max", 30): GaussianEMOS(a=0.0, b=1.0, c=0.3, d=0.7),
        }

    def get_model(self, station_id, target_date, target_type, lead_hours):
        month = pd.to_datetime(target_date).month
        season = "Jja" if month in [6, 7, 8] else ("Djf" if month in [12, 1, 2] else "Mam")
        int_lead = int(round(lead_hours))
        return self.models.get((station_id.upper(), season, target_type.lower(), int_lead), GaussianEMOS(a=0.0, b=1.0, c=0.0, d=1.0))

    def predict(self, station_id, target_date, target_type, lead_hours, ensemble_mean, ensemble_variance, sigma_clim_squared):
        month = pd.to_datetime(target_date).month
        season = "Jja" if month in [6, 7, 8] else ("Djf" if month in [12, 1, 2] else "Mam")
        anchors = {
            k[3]: m for k, m in self.models.items()
            if k[0] == station_id.upper() and k[1] == season and k[2] == target_type.lower()
        }
        from src.modeling.interpolator import LeadTimeInterpolator
        return LeadTimeInterpolator().predict_distribution(
            target_type=target_type,
            lead_hours=lead_hours,
            ensemble_mean=ensemble_mean,
            ensemble_variance=ensemble_variance,
            sigma_clim_squared=sigma_clim_squared,
            anchor_models=anchors,
        )


class MockClimatology:
    def get_climatology_variance(self, station_id, target_type, target_date):
        return 4.0


@pytest.fixture
def pipeline(tmp_path):
    db_file = tmp_path / "test_predictions.db"
    return PredictionPipeline(
        model_registry=MockRegistry(),
        climatology_calculator=MockClimatology(),
        db_path=db_file,
    )


class TestPredictionPipelineExecution:
    """Test full four-stage prediction execution and persistence."""

    def test_single_prediction_end_to_end_shanghai_max(self, pipeline):
        """Execute full end-to-end pipeline for Shanghai Summer Max temperature."""
        # GEFS features: ens_mean=30.0, ens_var=2.0
        # Observation at 12:00: T_now = 28.5°C
        record = pipeline.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            current_temp=28.5,
            observation_time="12:00",
            save_to_db=True,
        )

        assert isinstance(record, PredictionRecord)
        assert record.station_id == "ZSPD"
        assert record.target_date == "2019-07-15"
        assert record.target_type == "max"
        assert np.isclose(record.predicted_mu, 29.5, atol=1e-5)
        assert np.isclose(record.predicted_sigma, np.sqrt(5.44), atol=1e-5)
        assert record.is_truncated is True
        assert record.is_physically_constrained is True

        # Check market bins
        assert len(record.market_bins) == 7
        total_p = sum(b.probability for b in record.market_bins)
        assert np.isclose(total_p, 1.0, atol=1e-6)

        # Check persistence in SQLite
        history_df = pipeline.get_history("ZSPD", target_date="2019-07-15")
        assert len(history_df) == 1
        assert history_df.iloc[0]["station_id"] == "ZSPD"
        assert np.isclose(history_df.iloc[0]["predicted_mu"], 29.5, atol=1e-5)
        assert "bin_probabilities" in history_df.columns

    def test_single_prediction_denver_fahrenheit(self, pipeline):
        """Denver KDEN prediction should produce Fahrenheit discrete market bins."""
        record = pipeline.predict_single(
            station_id="KDEN",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=25.0,  # 25°C ≈ 77°F
            ensemble_variance=2.0,
            current_temp=22.0,
            save_to_db=True,
        )

        assert record.station_id == "KDEN"
        assert len(record.market_bins) > 0
        assert record.market_bins[0].unit == "F"
        total_p = sum(b.probability for b in record.market_bins)
        assert np.isclose(total_p, 1.0, atol=1e-6)

    def test_unobserved_static_flow(self, pipeline):
        """When current_temp is omitted, pipeline generates base forecast + discrete bins without truncation."""
        record = pipeline.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            current_temp=None,
            save_to_db=False,
        )

        assert record.is_truncated is False
        assert record.is_physically_constrained is False
        assert len(record.market_bins) == 7
        assert np.isclose(sum(b.probability for b in record.market_bins), 1.0, atol=1e-6)

    def test_batch_predictions_execution(self, pipeline):
        """Batch execution over DataFrame of dates."""
        df = pd.DataFrame({
            "station_id": ["ZSPD", "ZSPD"],
            "target_date": ["2019-07-15", "2019-07-16"],
            "target_type": ["max", "max"],
            "lead_time_hours": [30, 30],
            "ensemble_mean": [30.0, 31.0],
            "ensemble_variance": [2.0, 2.5],
            "current_temp": [28.0, 29.0],
            "sigma_clim_squared": [4.0, 4.0],
        })

        out_df = pipeline.predict_batch(df, save_to_db=True)

        assert len(out_df) == 2
        assert "predicted_mu" in out_df.columns
        assert "bin_probabilities" in out_df.columns

        # Verify 2 records in database
        db_records = pipeline.get_history("ZSPD")
        assert len(db_records) >= 2

    def test_winning_bin_determination_from_record(self, pipeline):
        """Determine settlement winning bin from PredictionRecord."""
        record = pipeline.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            current_temp=28.5,
        )

        # Ground truth observation is 30.2°C
        win_idx, win_bin = record.get_winning_bin(observed_temp=30.2, unit="C")
        assert win_bin.label == "30°C"
        assert win_bin.contains(30.2, "C")
