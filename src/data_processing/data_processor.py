#!/usr/bin/env python3
"""
Integrated DataProcessor service orchestrating end-to-end data processing (Task 1.3 T1.3-05).

Coordinates:
1. Time alignment & completely contained 6h window selection (subseteq local day)
2. Astronomical sunrise coverage verification for min temperature
3. 4-point bilinear spatial interpolation to station geographic coordinates
4. Temperature unit standardization to Celsius (°C)
5. Atmospheric lapse rate elevation correction (Γ = 0.0065 K/m)
6. Daily extreme collapsing and 5-member ensemble statistics extraction
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.data_processing.constants import STATION_COORDINATES
from src.data_processing.elevation_corrector import ElevationCorrector
from src.data_processing.feature_extractor import FeatureExtractor
from src.data_processing.spatial_interpolator import SpatialInterpolator
from src.data_processing.time_aligner import (
    TimeAligner,
    select_contained_6h_windows,
    select_contained_window_objects,
    verify_sunrise_coverage,
)
from src.data_processing.unit_converter import UnitConverter

logger = logging.getLogger(__name__)


class DataProcessor:
    """Unified pipeline processor transforming raw/cropped GEFS grids into calibrated station feature vectors."""

    def __init__(
        self,
        unit_converter: Optional[UnitConverter] = None,
        elevation_corrector: Optional[ElevationCorrector] = None,
        spatial_interpolator: Optional[SpatialInterpolator] = None,
        time_aligner: Optional[TimeAligner] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        self.unit_converter = unit_converter or UnitConverter()
        self.elevation_corrector = elevation_corrector or ElevationCorrector()
        self.spatial_interpolator = spatial_interpolator or SpatialInterpolator()
        self.time_aligner = time_aligner or TimeAligner()
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def _get_step_dim_name(self, ds: xr.Dataset) -> str:
        """Identify time step or forecast hour dimension name."""
        candidates = ["step", "window", "fxx", "lead_time", "forecast_hour", "time"]
        for cand in candidates:
            if cand in ds.dims:
                return cand
        raise ValueError(f"Could not identify forecast step dimension in dataset with dims: {ds.dims}")

    def _slice_contained_windows(
        self,
        dataset: xr.Dataset,
        station_id: str,
        target_date: date,
        init_time_utc: datetime,
        target_type: str,
        max_lead_hours: int,
    ) -> xr.Dataset:
        """Select and slice 6h forecast windows completely contained within the local day."""
        contained_fxx = select_contained_6h_windows(
            init_time_utc=init_time_utc,
            target_date=target_date,
            tz_or_station=station_id,
            max_lead_hours=max_lead_hours,
        )
        if not contained_fxx:
            raise ValueError(f"No contained 6h windows for {station_id} on {target_date} from {init_time_utc}")

        if target_type == "min":
            win_objs = select_contained_window_objects(init_time_utc, target_date, station_id, max_lead_hours)
            verify_sunrise_coverage(win_objs, target_date, station_id=station_id)

        step_dim = self._get_step_dim_name(dataset)
        avail_steps = [s for s in contained_fxx if s in dataset[step_dim].values]
        if not avail_steps:
            raise ValueError(f"Dataset missing required steps {contained_fxx}. Found: {list(dataset[step_dim].values)}")

        sliced_ds = dataset.sel({step_dim: avail_steps})
        return sliced_ds.rename({step_dim: "window"}) if step_dim != "window" else sliced_ds

    def _interpolate_and_standardize_units(
        self,
        sliced_ds: xr.Dataset,
        station_id: str,
        var_name: str,
    ) -> xr.Dataset:
        """Interpolate grid data to station coordinates and convert temperature units to Celsius."""
        station_ds = self.spatial_interpolator.interpolate_station(sliced_ds, station_id=station_id)
        if var_name not in station_ds:
            raise KeyError(f"Variable '{var_name}' not found in dataset. Found: {list(station_ds.data_vars.keys())}")

        units = station_ds[var_name].attrs.get("units", "").strip().lower()
        if units in ("k", "kelvin", "degk") or np.nanmean(station_ds[var_name].values) > 150.0:
            station_ds = self.unit_converter.convert_xarray(
                station_ds, from_unit="K", to_unit="C", var_names=[var_name]
            )
        return station_ds

    def _apply_elevation_and_extract_stats(
        self,
        station_ds: xr.Dataset,
        station_meta: Dict[str, Any],
        var_name: str,
        target_type: str,
        target_date: date,
        lead_time_bucket: Optional[int],
    ) -> pd.DataFrame:
        """Apply lapse rate elevation correction and extract 5-member ensemble statistics."""
        model_elev = 0.0
        for elev_var in ("orography", "elevation", "surface_geopotential_height", "gh"):
            if elev_var in station_ds:
                model_elev = float(np.nanmean(station_ds[elev_var].values))
                break

        corrected_ds = self.elevation_corrector.correct_xarray(
            station_ds,
            station_elevation=station_meta["elevation"],
            model_elevation=model_elev,
            var_names=[var_name],
        )

        feature_dict = self.feature_extractor.extract_features(
            station_ds=corrected_ds,
            target_type=target_type,
            target_date=target_date.isoformat(),
            lead_time_bucket=lead_time_bucket,
            var_name=var_name,
        )
        return self.feature_extractor.to_dataframe(feature_dict)

    def process_forecast_to_features(
        self,
        dataset: xr.Dataset,
        station_id: str,
        target_date: Union[date, str],
        init_time_utc: datetime,
        target_type: str,
        lead_time_bucket: Optional[int] = None,
        max_lead_hours: int = 120,
    ) -> pd.DataFrame:
        """Process a multi-grid GEFS forecast dataset into a calibrated station feature DataFrame."""
        t_date = date.fromisoformat(target_date) if isinstance(target_date, str) else target_date
        t_type = target_type.strip().lower()
        if t_type not in ("max", "min"):
            raise ValueError(f"target_type must be 'max' or 'min', got '{target_type}'")

        if station_id not in STATION_COORDINATES:
            raise ValueError(f"Unknown station_id: '{station_id}'. Configured: {list(STATION_COORDINATES.keys())}")

        station_meta = STATION_COORDINATES[station_id]
        var_name = "tmax" if t_type == "max" else "tmin"

        # Step 1: Slice completely contained 6h windows
        sliced_ds = self._slice_contained_windows(
            dataset, station_id, t_date, init_time_utc, t_type, max_lead_hours
        )

        # Step 2: Spatial bilinear interpolation and unit standardization
        station_ds = self._interpolate_and_standardize_units(sliced_ds, station_id, var_name)

        # Step 3: Elevation lapse rate correction and feature extraction
        return self._apply_elevation_and_extract_stats(
            station_ds, station_meta, var_name, t_type, t_date, lead_time_bucket
        )

    def batch_process(
        self,
        dataset: xr.Dataset,
        init_time_utc: datetime,
        requests: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Batch process multiple station/date/target requests from a single forecast dataset."""
        results = [
            self.process_forecast_to_features(
                dataset=dataset,
                station_id=req["station_id"],
                target_date=req["target_date"],
                init_time_utc=init_time_utc,
                target_type=req["target_type"],
                lead_time_bucket=req.get("lead_time_bucket"),
            )
            for req in requests
        ]
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
