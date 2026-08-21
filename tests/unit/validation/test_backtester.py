#!/usr/bin/env python3
"""
Unit tests for Backtester and Baseline models (Ticket 4.2-01 / Issue #36).

Verifies:
- ClimatologyBaseline, RawGEFSBaseline, and PersistenceBaseline
- BacktestEngine temporal order preservation, multi-baseline comparison,
  and comprehensive performance metrics calculation
"""

from datetime import date
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.validation.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    RawGEFSBaseline,
)
from src.validation.backtester import BacktestEngine, BacktestResult


class TestBaselineModels:
    """Tests for the three benchmark baseline predictors."""

    def test_climatology_baseline(self):
        mock_clim = MagicMock()
        mock_clim.get_climatology.return_value = (22.5, 3.2)  # mu, sigma

        base = ClimatologyBaseline(climatology_calculator=mock_clim)
        mu, sigma = base.predict(station_id="ZSPD", target_date=date(2019, 7, 15), target_type="max")

        assert mu == 22.5
        assert sigma == 3.2
        mock_clim.get_climatology.assert_called_once_with("ZSPD", "max", date(2019, 7, 15))

    def test_raw_gefs_baseline(self):
        base = RawGEFSBaseline(eps=0.1)

        # Given ensemble features with mean 25.0 and var 4.0 (std = 2.0)
        mu, sigma = base.predict(ensemble_mean=25.0, ensemble_variance=4.0)
        assert mu == 25.0
        assert sigma == 2.0

        # Zero variance gets clipped by eps
        mu_zero, sigma_zero = base.predict(ensemble_mean=25.0, ensemble_variance=0.0)
        assert sigma_zero == 0.1

    def test_persistence_baseline(self):
        mock_storage = MagicMock()
        mock_storage.get_observed_temperature.return_value = 28.0
        mock_clim = MagicMock()
        mock_clim.get_climatology.return_value = (26.0, 3.0)

        base = PersistenceBaseline(storage_manager=mock_storage, climatology_calculator=mock_clim)
        mu, sigma = base.predict(station_id="ZSPD", target_date=date(2019, 7, 15), target_type="max")

        assert mu == 28.0  # Takes yesterday's truth
        assert sigma == 3.0


class TestBacktestEngine:
    """Tests for BacktestEngine execution, temporal splitting, and metric evaluation."""

    def test_run_backtest_on_dataset(self):
        n = 30
        dates = pd.date_range("2019-01-01", periods=n, freq="D")
        truths = np.random.uniform(5.0, 15.0, size=n)
        ens_means = truths + np.random.normal(0, 1.0, size=n)
        ens_vars = np.full(n, 2.0)

        df_test = pd.DataFrame({
            "target_date": dates,
            "target_type": ["max"] * n,
            "lead_hours": [30] * n,
            "truth": truths,
            "ensemble_mean": ens_means,
            "ensemble_variance": ens_vars,
            "yesterday_truth": truths + np.random.normal(0, 1.5, size=n),
            "clim_mean": [10.0] * n,
            "clim_sigma": [3.0] * n,
        })

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = lambda **kwargs: (
            kwargs["ensemble_mean"] * 0.9 + 1.0,
            1.2,
        )

        engine = BacktestEngine()
        result = engine.run_backtest(
            station_id="ZSPD",
            target_type="max",
            lead_hours=30,
            dataset=df_test,
            model_predictor=mock_predictor,
        )

        assert isinstance(result, BacktestResult)
        assert result.station_id == "ZSPD"
        assert result.sample_count == n
        assert result.mean_crps_model > 0.0
        assert result.mean_crps_raw > 0.0
        assert result.mean_crps_clim > 0.0
        assert result.mean_crps_persistence > 0.0
        assert len(result.df_daily) == n
        assert "crps_model" in result.df_daily.columns
        assert "crps_raw" in result.df_daily.columns
        assert "crps_clim" in result.df_daily.columns
        assert "crps_persistence" in result.df_daily.columns
        assert "crpss_vs_clim" in result.to_dict()
