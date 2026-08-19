#!/usr/bin/env python3
"""
Unit tests for LeadTimeInterpolator (Ticket 2.2-05 / Issue #18).

Verifies:
1. Linear interpolation of EMOS parameters (a, b, c, d) between anchor points.
2. Max Temp intermediate interpolation across {12, 18, 24, 36, 42, 48}h from anchors {6, 30, 54}h.
3. Min Temp intermediate interpolation across {30, 36, 42}h from anchors {24, 48}h.
4. Min Temp short-lead (L < 24h) physical variance decay: sigma_L^2 = sigma_24^2 * sqrt(L / 24).
5. Generation of complete 6h-interval model grids from 6h to 54h for all station-season pairs.
"""

import numpy as np
import pytest

from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.interpolator import LeadTimeInterpolator


@pytest.fixture
def anchor_models_max():
    """Anchor models for Max Temp: 6h, 30h, 54h."""
    return {
        6: GaussianEMOS(a=0.1, b=1.0, c=0.2, d=0.8),
        30: GaussianEMOS(a=0.5, b=0.9, c=0.6, d=1.0),
        54: GaussianEMOS(a=0.9, b=0.8, c=1.0, d=1.2),
    }


@pytest.fixture
def anchor_models_min():
    """Anchor models for Min Temp: 24h, 48h."""
    return {
        24: GaussianEMOS(a=0.2, b=0.95, c=0.4, d=0.9),
        48: GaussianEMOS(a=0.8, b=0.85, c=0.8, d=1.1),
    }


class TestLeadTimeLinearInterpolation:
    """Test linear parameter interpolation."""

    def test_max_temp_exact_anchor_points(self, anchor_models_max):
        interpolator = LeadTimeInterpolator()
        
        # Exact anchor points must return identical models
        for lead in [6, 30, 54]:
            model = interpolator.get_model_at_lead(
                target_type="max",
                lead_hours=lead,
                anchor_models=anchor_models_max,
            )
            assert np.isclose(model.a, anchor_models_max[lead].a)
            assert np.isclose(model.b, anchor_models_max[lead].b)
            assert np.isclose(model.c, anchor_models_max[lead].c)
            assert np.isclose(model.d, anchor_models_max[lead].d)

    def test_max_temp_intermediate_linear_interpolation(self, anchor_models_max):
        interpolator = LeadTimeInterpolator()
        
        # 18h is exactly halfway between 6h and 30h (fraction = 12 / 24 = 0.5)
        model_18h = interpolator.get_model_at_lead(
            target_type="max",
            lead_hours=18,
            anchor_models=anchor_models_max,
        )
        expected_a = 0.5 * (0.1 + 0.5)  # 0.3
        expected_b = 0.5 * (1.0 + 0.9)  # 0.95
        expected_c = 0.5 * (0.2 + 0.6)  # 0.4
        expected_d = 0.5 * (0.8 + 1.0)  # 0.9

        assert np.isclose(model_18h.a, expected_a)
        assert np.isclose(model_18h.b, expected_b)
        assert np.isclose(model_18h.c, expected_c)
        assert np.isclose(model_18h.d, expected_d)

        # 42h is halfway between 30h and 54h
        model_42h = interpolator.get_model_at_lead(
            target_type="max",
            lead_hours=42,
            anchor_models=anchor_models_max,
        )
        assert np.isclose(model_42h.a, 0.5 * (0.5 + 0.9))
        assert np.isclose(model_42h.b, 0.5 * (0.9 + 0.8))

    def test_min_temp_intermediate_linear_interpolation(self, anchor_models_min):
        interpolator = LeadTimeInterpolator()
        
        # 36h is halfway between 24h and 48h
        model_36h = interpolator.get_model_at_lead(
            target_type="min",
            lead_hours=36,
            anchor_models=anchor_models_min,
        )
        assert np.isclose(model_36h.a, 0.5 * (0.2 + 0.8))
        assert np.isclose(model_36h.b, 0.5 * (0.95 + 0.85))


class TestMinTempShortLeadDecay:
    """Test min temp physical variance decay for L < 24h."""

    def test_min_temp_short_lead_variance_decay(self, anchor_models_min):
        interpolator = LeadTimeInterpolator()
        
        ens_mean = 15.0
        ens_var = 2.0
        clim_var = 4.0
        
        # Base 24h model prediction
        model_24h = anchor_models_min[24]
        mu_24h, sigma_24h = model_24h.compute_params(ens_mean, ens_var, clim_var)

        # 1. Prediction at 12h: L=12 < 24 -> decay factor = sqrt(12 / 24) = sqrt(0.5)
        dist_12h = interpolator.predict_distribution(
            target_type="min",
            lead_hours=12,
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            anchor_models=anchor_models_min,
        )
        expected_decay_12h = np.sqrt(12.0 / 24.0)
        expected_sigma_12h = sigma_24h * expected_decay_12h
        
        assert np.isclose(dist_12h.mu, mu_24h)
        assert np.isclose(dist_12h.sigma, expected_sigma_12h)

        # 2. Prediction at 6h: L=6 < 24 -> decay factor = sqrt(6 / 24) = 0.5
        dist_6h = interpolator.predict_distribution(
            target_type="min",
            lead_hours=6,
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            anchor_models=anchor_models_min,
        )
        expected_sigma_6h = sigma_24h * np.sqrt(6.0 / 24.0)
        assert np.isclose(dist_6h.sigma, expected_sigma_6h)


class TestFullGridInterpolation:
    """Test generating full 6h to 54h grid of models."""

    def test_interpolate_full_grid_max_and_min(self, anchor_models_max, anchor_models_min):
        interpolator = LeadTimeInterpolator()
        
        grid_leads = [6, 12, 18, 24, 30, 36, 42, 48, 54]
        
        grid_max = interpolator.build_full_grid(
            target_type="max",
            anchor_models=anchor_models_max,
            grid_leads=grid_leads,
        )
        assert len(grid_max) == 9
        for lead in grid_leads:
            assert lead in grid_max
            assert isinstance(grid_max[lead], GaussianEMOS)

        grid_min = interpolator.build_full_grid(
            target_type="min",
            anchor_models=anchor_models_min,
            grid_leads=grid_leads,
        )
        assert len(grid_min) == 9
        for lead in grid_leads:
            assert lead in grid_min
            assert isinstance(grid_min[lead], GaussianEMOS)
