#!/usr/bin/env python3
"""
Unit tests for ValidationEngine and Time Wall Isolation (Ticket 2.3-01 / Issue #20).

Verifies:
1. Strict Time Wall isolation: training set never touches 2019+ data.
2. Out-of-sample evaluation: computing MAE, CRPS, CRPSS, 90% CI coverage, and PIT probability integral transform values.
3. PIT values strictly bounded in [0, 1].
4. Rolling-origin expanding window time series cross-validation folds generator.
5. ValidationResult serialization and daily detail DataFrame export.
"""

from datetime import date
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.registry import ModelRegistry
from src.modeling.validation_engine import ValidationEngine, ValidationResult


class MockStorageForValidation:
    """Mock storage providing realistic 2000-2019 continuous dataset."""

    def load_training_dataset(
        self,
        station_id: str,
        target_type: str,
        lead_time_bucket: int,
        start_year: int = 2000,
        end_year: int = 2019,
    ) -> pd.DataFrame:
        dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        n = len(dates)
        
        base_temp = 20.0 if station_id == "ZSPD" else 15.0
        if target_type == "min":
            base_temp -= 8.0
            
        seasonal_cycle = 10.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        true_temp = base_temp + seasonal_cycle + np.random.normal(0, 2.5, n)
        
        # Raw forecast with +1.5°C warm bias and moderate under-dispersion
        ens_mean = true_temp + 1.5 + np.random.normal(0, 1.0, n)
        ens_var = np.full(n, 1.0)
        
        return pd.DataFrame({
            "target_date": dates.strftime("%Y-%m-%d"),
            "ensemble_mean": ens_mean,
            "ensemble_variance": ens_var,
            "observed_temp": true_temp,
            "member_max": ens_mean + 1.0,
            "member_min": ens_mean - 1.0,
        })


class MockClimForValidation:
    def get_climatology_variance(self, station_id: str, target_type: str, target_date: str) -> float:
        return 4.0

    def get_climatology_params(self, station_id: str, target_type: str, target_date: str) -> Tuple[float, float]:
        base = 20.0 if station_id == "ZSPD" else 15.0
        if target_type == "min":
            base -= 8.0
        return (base, 4.0)


class TestTimeWallAndValidation:
    """Test time wall enforcement and evaluation metrics."""

    def test_strict_time_wall_enforcement(self, tmp_path):
        storage = MockStorageForValidation()
        clim_calc = MockClimForValidation()
        registry = ModelRegistry(base_dir=tmp_path / "models")

        engine = ValidationEngine(
            storage_manager=storage,
            climatology_calculator=clim_calc,
            model_registry=registry,
            train_end_year=2018,
            val_start_year=2019,
            val_end_year=2019,
        )

        # 1. Check training data filter strictly excludes 2019
        df_train = engine.load_train_data("ZSPD", "max", 30)
        train_years = pd.to_datetime(df_train["target_date"]).dt.year.unique()
        assert 2019 not in train_years
        assert max(train_years) <= 2018

        # 2. Check validation data filter strictly contains only 2019
        df_val = engine.load_val_data("ZSPD", "max", 30)
        val_years = pd.to_datetime(df_val["target_date"]).dt.year.unique()
        assert list(val_years) == [2019]

    def test_out_of_sample_evaluation_metrics(self, tmp_path):
        storage = MockStorageForValidation()
        clim_calc = MockClimForValidation()
        registry = ModelRegistry(base_dir=tmp_path / "models")

        # Save a calibrated model in registry for 4 seasons
        for season in ["Spring", "Summer", "Autumn", "Winter"]:
            m = GaussianEMOS(a=-1.5, b=1.0, c=0.5, d=1.0)
            registry.save_model(m, "ZSPD", season, "max", 30)

        engine = ValidationEngine(
            storage_manager=storage,
            climatology_calculator=clim_calc,
            model_registry=registry,
            train_end_year=2018,
            val_start_year=2019,
            val_end_year=2019,
        )

        res = engine.evaluate_slice(station_id="ZSPD", target_type="max", lead_hours=30)

        assert isinstance(res, ValidationResult)
        assert res.sample_count == 365  # 2019 is a non-leap year (365 days)
        assert res.station_id == "ZSPD"
        assert res.lead_hours == 30
        assert res.target_type == "max"

        # Check metrics are sensible
        assert res.mae_emos < res.mae_raw  # Bias correction reduces MAE
        assert res.mean_crps_emos < res.mean_crps_raw  # CRPS improves
        assert res.crpss_vs_raw > 0.05
        assert 0.80 <= res.coverage_90_ci <= 1.0  # 90% CI coverage is healthy

        # Check PIT values
        assert len(res.pit_values) == 365
        assert np.all(res.pit_values >= 0.0)
        assert np.all(res.pit_values <= 1.0)

        # Check daily DataFrame export
        df_daily = res.df_daily
        assert len(df_daily) == 365
        assert set(df_daily.columns) >= {
            "target_date", "observed_temp", "ensemble_mean", "emos_mu", "emos_sigma",
            "crps_emos", "crps_raw", "crps_clim", "pit_value", "in_90_ci"
        }

    def test_rolling_origin_folds_generator(self):
        """Test rolling origin expanding window CV fold splits."""
        folds = ValidationEngine.generate_rolling_origin_folds(
            start_year=2010,
            end_year=2018,
            min_train_years=5,
        )
        
        # 2010-2018 = 9 years.
        # min_train_years=5 -> Fold 1: train 2010-2014, test 2015
        # Fold 2: train 2010-2015, test 2016
        # Fold 3: train 2010-2016, test 2017
        # Fold 4: train 2010-2017, test 2018
        assert len(folds) == 4
        
        f1_train, f1_test = folds[0]
        assert f1_train == (2010, 2014)
        assert f1_test == 2015

        f4_train, f4_test = folds[3]
        assert f4_train == (2010, 2017)
        assert f4_test == 2018
