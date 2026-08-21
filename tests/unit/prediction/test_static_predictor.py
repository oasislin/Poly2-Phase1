#!/usr/bin/env python3
"""
Unit tests for StaticPredictor (Ticket 3.1-01 / Issue #26).

Verifies:
1. Routing to exact anchor models via ModelRegistry (Max: {54, 30, 6}h, Min: {48, 24}h).
2. Missing node linear parameter interpolation (e.g., Max 18h, 42h; Min 36h).
3. Minimum temperature short-lead (< 24h) physical variance shrinkage:
   sigma_L = sigma_24h * sqrt(L / 24).
4. Square parameterization mathematical exactness:
   mu = a + b * ens_mean, sigma^2 = c^2 + d^2 * ens_var + sigma_clim^2.
5. Confidence & prediction intervals:
   90% CI = mu +/- 1.64485 * sigma, 95% CI = mu +/- 1.95996 * sigma.
6. Automatic ClimatologyCalculator integration for sigma_clim^2 lookup.
7. Batch inference over pandas DataFrame with vectorized outputs.
8. Error handling and boundary validation for invalid inputs.
"""

from datetime import date
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pytest

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.registry import ModelRegistry
from src.prediction.static_predictor import StaticPredictor, StaticPredictionResult


class MockRegistry:
    """Mock registry providing predefined anchor GaussianEMOS models for testing."""

    def __init__(self):
        # Anchor models with distinct identifiable parameters
        # ZSPD Summer Max anchors: 6h=(0.5, 1.0, 0.2, 0.5), 30h=(1.0, 0.95, 0.4, 0.8), 54h=(1.5, 0.90, 0.6, 1.0)
        # ZSPD Winter Min anchors: 24h=(-1.0, 1.05, 0.5, 0.6), 48h=(-1.5, 1.00, 0.8, 0.9)
        self.models: Dict[Tuple[str, str, str, int], GaussianEMOS] = {
            ("ZSPD", "Jja", "max", 6): GaussianEMOS(a=0.5, b=1.0, c=0.2, d=0.5),
            ("ZSPD", "Jja", "max", 30): GaussianEMOS(a=1.0, b=0.95, c=0.4, d=0.8),
            ("ZSPD", "Jja", "max", 54): GaussianEMOS(a=1.5, b=0.90, c=0.6, d=1.0),
            ("ZSPD", "Djf", "min", 24): GaussianEMOS(a=-1.0, b=1.05, c=0.5, d=0.6),
            ("ZSPD", "Djf", "min", 48): GaussianEMOS(a=-1.5, b=1.00, c=0.8, d=0.9),
            ("KDEN", "Jja", "max", 30): GaussianEMOS(a=0.0, b=1.0, c=0.3, d=0.7),
        }

    def get_model(self, station_id: str, target_date, target_type: str, lead_hours: float) -> GaussianEMOS:
        # Determine season
        month = pd.to_datetime(target_date).month
        if month in [12, 1, 2]:
            season = "Djf"
        elif month in [3, 4, 5]:
            season = "Mam"
        elif month in [6, 7, 8]:
            season = "Jja"
        else:
            season = "Son"

        int_lead = int(round(lead_hours))
        key = (station_id.upper(), season, target_type.lower(), int_lead)
        if key in self.models:
            return self.models[key]

        # On the fly interpolation if anchors exist
        anchors = {
            k[3]: m for k, m in self.models.items()
            if k[0] == station_id.upper() and k[1] == season and k[2] == target_type.lower()
        }
        if not anchors:
            raise FileNotFoundError(f"No model found for {key}")

        from src.modeling.interpolator import LeadTimeInterpolator
        return LeadTimeInterpolator().get_model_at_lead(target_type, lead_hours, anchors)

    def predict(
        self,
        station_id: str,
        target_date,
        target_type: str,
        lead_hours: float,
        ensemble_mean: float,
        ensemble_variance: float,
        sigma_clim_squared: float,
    ) -> GaussianEMOS:
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
    """Mock climatology calculator."""

    def get_climatology_variance(self, station_id: str, target_type: str, target_date: str) -> float:
        return 4.0 if station_id.upper() == "ZSPD" else 9.0


class TestStaticPredictorAlgorithms:
    """Mathematical and algorithmic tests for StaticPredictor."""

    @pytest.fixture
    def predictor(self):
        return StaticPredictor(
            model_registry=MockRegistry(),
            climatology_calculator=MockClimatology(),
        )

    def test_anchor_node_exact_parameter_evaluation(self, predictor):
        """Anchor lead time 30h for ZSPD Summer Max should evaluate exact square parameterization."""
        # Anchor (ZSPD, Jja, max, 30): a=1.0, b=0.95, c=0.4, d=0.8
        # ens_mean=30.0, ens_var=2.0, sigma_clim_sq=4.0
        # Expected mu = 1.0 + 0.95 * 30.0 = 29.5
        # Expected sigma^2 = 0.4^2 + 0.8^2 * 2.0 + 4.0 = 0.16 + 1.28 + 4.0 = 5.44
        # Expected sigma = sqrt(5.44) ≈ 2.33238
        res = predictor.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            sigma_clim_squared=4.0,
        )

        assert isinstance(res, StaticPredictionResult)
        assert np.isclose(res.mu, 29.5, atol=1e-5)
        assert np.isclose(res.sigma, np.sqrt(5.44), atol=1e-5)
        assert res.is_interpolated is False
        assert res.is_short_lead_decay is False
        assert res.season == "Summer"

    def test_missing_node_linear_parameter_interpolation(self, predictor):
        """Lead time 18h for ZSPD Summer Max linearly interpolates between 6h and 30h."""
        # 6h: a=0.5, b=1.0, c=0.2, d=0.5
        # 30h: a=1.0, b=0.95, c=0.4, d=0.8
        # 18h is exact midpoint (weight = 0.5):
        # a_interp = 0.75, b_interp = 0.975, c_interp = 0.3, d_interp = 0.65
        # ens_mean=20.0, ens_var=4.0, sigma_clim_sq=0.0
        # mu = 0.75 + 0.975 * 20.0 = 0.75 + 19.5 = 20.25
        # sigma^2 = 0.3^2 + 0.65^2 * 4.0 + 0 = 0.09 + 0.4225 * 4.0 = 0.09 + 1.69 = 1.78
        # sigma = sqrt(1.78) ≈ 1.33417
        res = predictor.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=18,
            ensemble_mean=20.0,
            ensemble_variance=4.0,
            sigma_clim_squared=0.0,
        )

        assert np.isclose(res.mu, 20.25, atol=1e-5)
        assert np.isclose(res.sigma, np.sqrt(1.78), atol=1e-5)
        assert res.is_interpolated is True
        assert res.is_short_lead_decay is False

    def test_min_temp_short_lead_physical_variance_shrinkage(self, predictor):
        """Min temp lead < 24h borrows 24h parameters and scales sigma by sqrt(L / 24)."""
        # ZSPD Winter Min 24h: a=-1.0, b=1.05, c=0.5, d=0.6
        # Test lead = 6h (< 24h):
        # 24h base sigma^2 with ens_mean=5.0, ens_var=2.0, sigma_clim_sq=0.0:
        # mu = -1.0 + 1.05 * 5.0 = 4.25
        # sigma_24h^2 = 0.5^2 + 0.6^2 * 2.0 = 0.25 + 0.72 = 0.97
        # sigma_24h = sqrt(0.97)
        # Scaled sigma_6h = sqrt(0.97) * sqrt(6 / 24) = sqrt(0.97 * 0.25) = sqrt(0.2425) ≈ 0.49244
        res = predictor.predict_single(
            station_id="ZSPD",
            target_date="2019-01-15",
            target_type="min",
            lead_time_hours=6,
            ensemble_mean=5.0,
            ensemble_variance=2.0,
            sigma_clim_squared=0.0,
        )

        assert np.isclose(res.mu, 4.25, atol=1e-5)
        expected_sigma = np.sqrt(0.97) * np.sqrt(6.0 / 24.0)
        assert np.isclose(res.sigma, expected_sigma, atol=1e-5)
        assert res.is_short_lead_decay is True

    def test_climatology_fallback_when_clim_var_omitted(self, predictor):
        """When sigma_clim_squared is None, predictor queries ClimatologyCalculator."""
        res = predictor.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            sigma_clim_squared=None,  # Should use MockClimatology -> 4.0
        )
        assert np.isclose(res.sigma, np.sqrt(5.44), atol=1e-5)

    def test_confidence_and_prediction_intervals(self, predictor):
        """Verify 90% and 95% confidence interval formulas: mu +/- z * sigma."""
        res = predictor.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            sigma_clim_squared=4.0,
        )
        # 90% CI: z = 1.6448536
        low_90, high_90 = res.confidence_interval(0.90)
        assert np.isclose(low_90, res.mu - 1.6448536 * res.sigma, atol=1e-4)
        assert np.isclose(high_90, res.mu + 1.6448536 * res.sigma, atol=1e-4)

        # 95% CI: z = 1.959964
        low_95, high_95 = res.prediction_interval(0.95)
        assert np.isclose(low_95, res.mu - 1.959964 * res.sigma, atol=1e-4)
        assert np.isclose(high_95, res.mu + 1.959964 * res.sigma, atol=1e-4)

    def test_distribution_methods_proxy_gaussian_emos(self, predictor):
        """Verify CDF, PDF, and Quantile work transparently on prediction result."""
        res = predictor.predict_single(
            station_id="ZSPD",
            target_date="2019-07-15",
            target_type="max",
            lead_time_hours=30,
            ensemble_mean=30.0,
            ensemble_variance=2.0,
            sigma_clim_squared=4.0,
        )
        # Median is mu
        assert np.isclose(res.cdf(res.mu), 0.5, atol=1e-5)
        assert np.isclose(res.quantile(0.5), res.mu, atol=1e-5)
        # PDF at mean is 1 / (sigma * sqrt(2*pi))
        expected_peak = 1.0 / (res.sigma * np.sqrt(2 * np.pi))
        assert np.isclose(res.pdf(res.mu), expected_peak, atol=1e-5)

    def test_batch_prediction_over_dataframe(self, predictor):
        """Verify predict_batch processes multiple rows and outputs all calibrated columns."""
        df = pd.DataFrame({
            "target_date": ["2019-07-15", "2019-07-16", "2019-07-17"],
            "station_id": ["ZSPD", "ZSPD", "ZSPD"],
            "target_type": ["max", "max", "max"],
            "lead_time_hours": [30, 18, 6],
            "ensemble_mean": [30.0, 20.0, 25.0],
            "ensemble_variance": [2.0, 4.0, 1.0],
            "sigma_clim_squared": [4.0, 0.0, 1.0],
        })

        pred_df = predictor.predict_batch(df)

        assert len(pred_df) == 3
        assert "predicted_mu" in pred_df.columns
        assert "predicted_sigma" in pred_df.columns
        assert "ci_90_lower" in pred_df.columns
        assert "ci_90_upper" in pred_df.columns
        assert "ci_95_lower" in pred_df.columns
        assert "ci_95_upper" in pred_df.columns

        # Verify row 0 matches single predict
        assert np.isclose(pred_df.iloc[0]["predicted_mu"], 29.5, atol=1e-5)
        assert np.isclose(pred_df.iloc[0]["predicted_sigma"], np.sqrt(5.44), atol=1e-5)

    def test_invalid_target_type_raises_error(self, predictor):
        """Invalid target_type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid target_type"):
            predictor.predict_single(
                station_id="ZSPD",
                target_date="2019-07-15",
                target_type="invalid_type",
                lead_time_hours=30,
                ensemble_mean=30.0,
                ensemble_variance=2.0,
            )
