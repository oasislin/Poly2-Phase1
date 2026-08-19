#!/usr/bin/env python3
"""
LeadTimeInterpolator: Missing lead time linear interpolation and short-lead physical decay extrapolation (Ticket 2.2-05 / Issue #18).

Implements (v5.9.1 §4.3):
    1. Linear parameter interpolation across intermediate lead times:
       - Max Temp: Anchors {6, 30, 54}h -> Interpolates {12, 18, 24, 36, 42, 48}h
       - Min Temp: Anchors {24, 48}h -> Interpolates {30, 36, 42}h, Extrapolates {54}h
    2. Min Temp short-lead (L < 24h) physical variance decay:
       - σ_L² = σ_24h² · √(L / 24)
       - Ensures prediction uncertainty monotonically shrinks as lead time approaches 0.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.modeling.gaussian_emos import GaussianEMOS


class LeadTimeInterpolator:
    """Interpolates EMOS model parameters and applies short-lead physical variance decay."""

    @staticmethod
    def _linear_interp_params(
        l_target: float,
        l0: float,
        p0: Tuple[float, float, float, float],
        l1: float,
        p1: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """Linearly interpolate parameter vector p = (a, b, c, d) between (l0, p0) and (l1, p1)."""
        if np.isclose(l0, l1):
            return p0
        weight = (l_target - l0) / (l1 - l0)
        a = (1.0 - weight) * p0[0] + weight * p1[0]
        b = (1.0 - weight) * p0[1] + weight * p1[1]
        c = (1.0 - weight) * p0[2] + weight * p1[2]
        d = (1.0 - weight) * p0[3] + weight * p1[3]
        return (float(a), float(b), float(c), float(d))

    def get_model_at_lead(
        self,
        target_type: str,
        lead_hours: Union[int, float],
        anchor_models: Dict[int, GaussianEMOS],
    ) -> GaussianEMOS:
        """Retrieve or interpolate a GaussianEMOS parameter model for any discrete lead hour."""
        t_type = target_type.lower()
        lead = float(lead_hours)

        # 1. Exact anchor hit
        int_lead = int(round(lead))
        if int_lead in anchor_models and np.isclose(lead, int_lead):
            return anchor_models[int_lead]

        # 2. Extract sorted anchor keys
        sorted_anchors = sorted(anchor_models.keys())
        if not sorted_anchors:
            raise ValueError("No anchor models provided for interpolation")

        # Boundary clamping / extrapolation
        if lead <= sorted_anchors[0]:
            return anchor_models[sorted_anchors[0]]
        if lead >= sorted_anchors[-1]:
            return anchor_models[sorted_anchors[-1]]

        # Find enclosing anchor pair [L0, L1]
        l0, l1 = sorted_anchors[0], sorted_anchors[-1]
        for i in range(len(sorted_anchors) - 1):
            if sorted_anchors[i] <= lead <= sorted_anchors[i + 1]:
                l0 = sorted_anchors[i]
                l1 = sorted_anchors[i + 1]
                break

        m0 = anchor_models[l0]
        m1 = anchor_models[l1]
        p0 = (m0.a, m0.b, m0.c, m0.d)
        p1 = (m1.a, m1.b, m1.c, m1.d)

        a_interp, b_interp, c_interp, d_interp = self._linear_interp_params(lead, l0, p0, l1, p1)
        return GaussianEMOS(a=a_interp, b=b_interp, c=c_interp, d=d_interp)

    def predict_distribution(
        self,
        target_type: str,
        lead_hours: Union[int, float],
        ensemble_mean: Union[float, np.ndarray, pd.Series],
        ensemble_variance: Union[float, np.ndarray, pd.Series],
        sigma_clim_squared: Union[float, np.ndarray, pd.Series],
        anchor_models: Dict[int, GaussianEMOS],
    ) -> GaussianEMOS:
        """Compute the calibrated Gaussian prediction distribution at lead_hours.

        Applies sqrt(L / 24) physical variance decay for Min Temp when lead_hours < 24h.
        """
        t_type = target_type.lower()
        lead = float(lead_hours)

        # Min Temp short-lead physical decay for L < 24h
        if t_type == "min" and lead < 24.0:
            if 24 not in anchor_models:
                raise KeyError("Min Temp interpolation requires 24h anchor model")
            base_model = anchor_models[24]
            mu_24, sigma_24 = base_model.compute_params(
                ensemble_mean=ensemble_mean,
                ensemble_variance=ensemble_variance,
                sigma_clim_squared=sigma_clim_squared,
            )
            # Physical standard deviation decay per v5.9.1 §4: σ_final = σ_24h * sqrt(L / 24)
            decay_factor = np.sqrt(max(0.0, lead) / 24.0)
            decayed_sigma = np.maximum(1e-4, sigma_24 * decay_factor)

            if np.ndim(mu_24) == 0:
                return GaussianEMOS.from_params(mu=float(mu_24), sigma=float(decayed_sigma))
            return GaussianEMOS.from_params(mu=mu_24, sigma=decayed_sigma)

        # Standard linear-interpolated model forward pass
        model = self.get_model_at_lead(t_type, lead, anchor_models)
        mu, sigma = model.compute_params(
            ensemble_mean=ensemble_mean,
            ensemble_variance=ensemble_variance,
            sigma_clim_squared=sigma_clim_squared,
        )
        return GaussianEMOS.from_params(mu=mu, sigma=sigma)

    def build_full_grid(
        self,
        target_type: str,
        anchor_models: Dict[int, GaussianEMOS],
        grid_leads: Optional[Sequence[int]] = None,
    ) -> Dict[int, GaussianEMOS]:
        """Construct the complete dense 6h-spaced model dictionary across 6h to 54h."""
        leads = list(grid_leads or [6, 12, 18, 24, 30, 36, 42, 48, 54])
        return {
            lead: self.get_model_at_lead(target_type, lead, anchor_models)
            for lead in leads
        }
