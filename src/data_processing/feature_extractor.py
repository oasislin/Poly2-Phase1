#!/usr/bin/env python3
"""
Ensemble feature extraction module (Task 1.3 T1.3-04).

Extracts 5-member ensemble statistics for Gaussian EMOS modeling:
- Daily extreme collapsing over fully contained 6h windows (tmax -> max, tmin -> min)
- 4 core ensemble summary statistics: {ensemble_mean, ensemble_variance, member_max, member_min}
- Strict adherence to v5.9.1 spec: quantile features and artificial temporal features are omitted.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


def collapse_daily_extreme(
    data: Union[xr.DataArray, xr.Dataset],
    target_type: str,
    window_dim: str = "window",
) -> Union[xr.DataArray, xr.Dataset]:
    """Collapse 6h forecast windows into a single daily extreme per ensemble member.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data with window dimension.
    target_type : str
        'max' (for daily maximum temperature) or 'min' (for daily minimum temperature).
    window_dim : str, default 'window'
        Name of the window dimension.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Data with window dimension collapsed.
    """
    t_type = target_type.strip().lower()
    if t_type not in ("max", "min"):
        raise ValueError(f"target_type must be 'max' or 'min', got '{target_type}'")

    if window_dim not in data.dims:
        # Already collapsed or single window
        return data

    if t_type == "max":
        return data.max(dim=window_dim)
    else:
        return data.min(dim=window_dim)


def calculate_ensemble_statistics(
    member_values: Union[np.ndarray, List[float], xr.DataArray],
    ddof: int = 1,
) -> Dict[str, float]:
    """Calculate 4 standard ensemble statistics from ensemble member values.

    Parameters
    ----------
    member_values : array-like or xr.DataArray
        Array of member values (e.g. 5 members: c00, p01-p04).
    ddof : int, default 1
        Delta Degrees of Freedom for sample variance calculation.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ensemble_mean': sample mean
        - 'ensemble_variance': sample variance (ddof=1)
        - 'member_max': maximum across members
        - 'member_min': minimum across members
    """
    if isinstance(member_values, xr.DataArray):
        vals = member_values.values.flatten()
    else:
        vals = np.asarray(member_values, dtype=float).flatten()

    if len(vals) == 0:
        raise ValueError("member_values cannot be empty.")

    mean_val = float(np.mean(vals))
    var_val = float(np.var(vals, ddof=ddof)) if len(vals) > 1 else 0.0
    max_val = float(np.max(vals))
    min_val = float(np.min(vals))

    return {
        "ensemble_mean": mean_val,
        "ensemble_variance": var_val,
        "member_max": max_val,
        "member_min": min_val,
    }


class FeatureExtractor:
    """Service for extracting modeling features from interpolated station forecast datasets."""

    def extract_features(
        self,
        station_ds: xr.Dataset,
        target_type: str,
        target_date: Optional[Union[str, pd.Timestamp]] = None,
        lead_time_bucket: Optional[int] = None,
        var_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract daily ensemble features from an interpolated station Dataset.

        Parameters
        ----------
        station_ds : xr.Dataset
            Dataset with member dimension (and optional window dimension).
        target_type : str
            'max' or 'min'.
        target_date : str or pd.Timestamp, optional
            Target date for the feature record.
        lead_time_bucket : int, optional
            Lead time node (e.g. 54, 30, 6 for max; 48, 24 for min).
        var_name : str, optional
            Name of the temperature variable in station_ds ('tmax' or 'tmin').

        Returns
        -------
        dict
            Feature dictionary containing metadata and ensemble statistics.
        """
        t_type = target_type.strip().lower()
        if var_name is None:
            var_name = "tmax" if t_type == "max" else "tmin"

        if var_name not in station_ds:
            raise KeyError(f"Variable '{var_name}' not found in station_ds data variables.")

        da = station_ds[var_name]

        # 1. Collapse window dimension if present
        collapsed_da = collapse_daily_extreme(da, target_type=t_type)

        # 2. Calculate ensemble statistics over members
        stats = calculate_ensemble_statistics(collapsed_da)

        # 3. Assemble feature record
        station_id = station_ds.attrs.get("station_id", "UNKNOWN")
        feature_dict = {
            "target_date": str(target_date) if target_date is not None else None,
            "station_id": station_id,
            "target_type": t_type,
            "lead_time_bucket": lead_time_bucket,
            "ensemble_mean": stats["ensemble_mean"],
            "ensemble_variance": stats["ensemble_variance"],
            "member_max": stats["member_max"],
            "member_min": stats["member_min"],
        }
        return feature_dict

    def to_dataframe(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Convert one or multiple feature dictionaries into a pandas DataFrame."""
        if isinstance(features, dict):
            features = [features]
        df = pd.DataFrame(features)
        # Ensure column ordering
        cols = [
            "target_date",
            "station_id",
            "target_type",
            "lead_time_bucket",
            "ensemble_mean",
            "ensemble_variance",
            "member_max",
            "member_min",
        ]
        existing_cols = [c for c in cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in cols]
        return df[existing_cols + extra_cols]
