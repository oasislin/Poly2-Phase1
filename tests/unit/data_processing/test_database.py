"""Unit tests for SQLite TimeSeriesDatabase (Task 1.4 T1.4-03)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data_processing.database import TimeSeriesDatabase


class TestTimeSeriesDatabase:
    """Test SQLite schema initialization, observations, predictions, and metrics persistence."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "test_poly.db"
        db = TimeSeriesDatabase(db_path=db_path)
        db.init_schema()
        return db

    def test_init_schema_creates_tables(self, temp_db):
        table_names = temp_db.list_tables()
        assert "observations" in table_names
        assert "predictions" in table_names
        assert "validation_metrics" in table_names

    def test_save_and_query_observations(self, temp_db):
        obs_df = pd.DataFrame({
            "date": ["2019-07-01", "2019-07-02", "2019-07-03"],
            "station_id": ["ZSPD", "ZSPD", "ZSPD"],
            "temp_max": [32.5, 34.0, 31.0],
            "temp_min": [24.0, 25.5, 23.5],
            "humidity": [75.0, 80.0, 70.0],
            "wind_speed": [5.2, 4.8, 6.0],
            "pressure": [1008.0, 1006.5, 1009.0],
            "precipitation": [0.0, 2.5, 0.0],
        })

        saved_count = temp_db.save_observations(obs_df)
        assert saved_count == 3

        # Query back range
        res = temp_db.get_observations(
            station_id="ZSPD",
            start_date="2019-07-02",
            end_date="2019-07-03",
        )
        assert len(res) == 2
        assert list(res["date"]) == ["2019-07-02", "2019-07-03"]
        assert res.iloc[0]["temp_max"] == pytest.approx(34.0)

    def test_upsert_observations_updates_existing(self, temp_db):
        obs_initial = pd.DataFrame({
            "date": ["2019-07-01"],
            "station_id": ["ZSPD"],
            "temp_max": [30.0],
            "temp_min": [22.0],
        })
        temp_db.save_observations(obs_initial)

        # Upsert with corrected temp_max
        obs_updated = pd.DataFrame({
            "date": ["2019-07-01"],
            "station_id": ["ZSPD"],
            "temp_max": [32.0],  # Updated
            "temp_min": [22.0],
        })
        temp_db.save_observations(obs_updated)

        res = temp_db.get_observations(station_id="ZSPD", start_date="2019-07-01", end_date="2019-07-01")
        assert len(res) == 1
        assert res.iloc[0]["temp_max"] == pytest.approx(32.0)

    def test_save_and_query_predictions(self, temp_db):
        pred_df = pd.DataFrame({
            "station_id": ["ZSPD", "ZSPD"],
            "target_date": ["2019-07-02", "2019-07-02"],
            "target_type": ["max", "min"],
            "lead_time_bucket": [30, 24],
            "model_version": ["v5.9.1_gaussian_emos", "v5.9.1_gaussian_emos"],
            "mean": [33.2, 24.1],
            "variance": [2.4, 1.8],
            "floor_active": [0, 1],
            "degraded_to_climatology": [0, 0],
        })

        saved_count = temp_db.save_predictions(pred_df)
        assert saved_count == 2

        # Query max prediction
        res_max = temp_db.get_predictions(
            station_id="ZSPD",
            target_date="2019-07-02",
            target_type="max",
        )
        assert len(res_max) == 1
        assert res_max.iloc[0]["mean"] == pytest.approx(33.2)
        assert res_max.iloc[0]["lead_time_bucket"] == 30

    def test_save_and_query_validation_metrics(self, temp_db):
        metrics_df = pd.DataFrame({
            "station_id": ["ZSPD"],
            "season": ["JJA"],
            "target_type": ["max"],
            "lead_time_bucket": [30],
            "model_version": ["v5.9.1_gaussian_emos"],
            "crps_mean": [1.45],
            "pit_ks_pvalue": [0.28],
            "sample_count": [580],
            "extreme_warm_crps": [2.10],
            "extreme_cold_crps": [1.95],
        })

        temp_db.save_validation_metrics(metrics_df)

        res = temp_db.get_validation_metrics(
            station_id="ZSPD",
            season="JJA",
            target_type="max",
        )
        assert len(res) == 1
        assert res.iloc[0]["crps_mean"] == pytest.approx(1.45)
        assert res.iloc[0]["pit_ks_pvalue"] == pytest.approx(0.28)
