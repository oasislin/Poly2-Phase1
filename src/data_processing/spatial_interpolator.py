#!/usr/bin/env python3
"""
Spatial bilinear interpolation module (Task 1.3 T1.3-02).

Performs 4-point bilinear interpolation from 0.25° GEFS grid datasets to
specific observation station coordinates. Strict requirement: nearest-neighbor
interpolation is forbidden (v5.9.1 / Phase 1 spec).
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

from src.data_processing.constants import STATION_COORDINATES


def normalize_longitude(lon: float, target_system: str = "-180_to_180") -> float:
    """Normalize longitude to either [-180, 180] or [0, 360] coordinate system."""
    if target_system in ("-180_to_180", "180"):
        return (lon + 180.0) % 360.0 - 180.0
    elif target_system in ("0_to_360", "360"):
        return lon % 360.0
    else:
        raise ValueError(f"Unknown target_system: {target_system}. Use '-180_to_180' or '0_to_360'.")


def _detect_grid_lon_system(grid_lons: np.ndarray) -> str:
    """Detect whether grid longitudes are in 0..360 or -180..180."""
    if np.any(grid_lons < 0):
        return "-180_to_180"
    if np.any(grid_lons > 180):
        return "0_to_360"
    return "0_to_360"


def _find_axis_bounding_indices(target_val: float, grid_vals: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """Find 1D bounding indices (idx0, idx1) and interpolation weight along an axis."""
    if grid_vals[0] < grid_vals[-1]:  # Ascending
        idx1 = int(np.searchsorted(grid_vals, target_val, side="right"))
        idx1 = min(max(idx1, 1), len(grid_vals) - 1)
        idx0 = idx1 - 1
        denom = grid_vals[idx1] - grid_vals[idx0]
        weight = float((target_val - grid_vals[idx0]) / denom) if denom != 0 else 0.0
    else:  # Descending
        idx0 = int(np.searchsorted(-grid_vals, -target_val, side="right")) - 1
        idx0 = min(max(idx0, 0), len(grid_vals) - 2)
        idx1 = idx0 + 1
        denom = grid_vals[idx0] - grid_vals[idx1]
        weight = float((grid_vals[idx0] - target_val) / denom) if denom != 0 else 0.0

    return (idx0, idx1), min(max(weight, 0.0), 1.0)


def find_surrounding_grid_indices(
    target_lat: float,
    target_lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int], float, float]:
    """Find the 2x2 bounding grid indices and relative weights (u, v) for a target point."""
    grid_lats = np.asarray(grid_lats)
    grid_lons = np.asarray(grid_lons)

    # 1. Coordinate check for Latitude
    lat_min, lat_max = float(np.min(grid_lats)), float(np.max(grid_lats))
    if target_lat < lat_min - 1e-6 or target_lat > lat_max + 1e-6:
        raise ValueError(
            f"Target latitude {target_lat} is out of grid bounds [{lat_min}, {lat_max}]"
        )

    # 2. Coordinate normalization and check for Longitude
    lon_system = _detect_grid_lon_system(grid_lons)
    norm_target_lon = normalize_longitude(target_lon, target_system=lon_system)

    lon_min, lon_max = float(np.min(grid_lons)), float(np.max(grid_lons))
    if norm_target_lon < lon_min - 1e-6 or norm_target_lon > lon_max + 1e-6:
        raise ValueError(
            f"Target longitude {target_lon} (norm: {norm_target_lon}) is out of grid bounds [{lon_min}, {lon_max}]"
        )

    # 3. Find bounding indices for both axes
    lat_indices, u = _find_axis_bounding_indices(target_lat, grid_lats)
    lon_indices, v = _find_axis_bounding_indices(norm_target_lon, grid_lons)

    return lat_indices, lon_indices, u, v


def bilinear_interp_2d(
    grid_2d: np.ndarray,
    target_lat: float,
    target_lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> float:
    """Perform 2D 4-point bilinear interpolation on a single 2D grid."""
    (i0, i1), (j0, j1), u, v = find_surrounding_grid_indices(
        target_lat, target_lon, grid_lats, grid_lons
    )

    q00 = grid_2d[i0, j0]
    q01 = grid_2d[i0, j1]
    q10 = grid_2d[i1, j0]
    q11 = grid_2d[i1, j1]

    val = (1.0 - u) * (1.0 - v) * q00 + (1.0 - u) * v * q01 + u * (1.0 - v) * q10 + u * v * q11
    return float(val)


class SpatialInterpolator:
    """Spatial interpolation service using 4-point bilinear interpolation."""

    @staticmethod
    def _find_spatial_dims(da: xr.DataArray) -> Tuple[str, str]:
        """Find latitude and longitude dimension names in a DataArray."""
        lat_names = ["latitude", "lat", "lat_0", "y"]
        lon_names = ["longitude", "lon", "lon_0", "x"]

        lat_dim = next((dim for dim in lat_names if dim in da.dims), None)
        lon_dim = next((dim for dim in lon_names if dim in da.dims), None)

        if lat_dim is None or lon_dim is None:
            raise ValueError(
                f"Could not identify spatial dimensions in DataArray with dims: {da.dims}"
            )
        return lat_dim, lon_dim

    def interpolate_dataarray(
        self,
        da: xr.DataArray,
        target_lat: float,
        target_lon: float,
    ) -> xr.DataArray:
        """Interpolate an xarray DataArray to a target (latitude, longitude) point."""
        lat_dim, lon_dim = self._find_spatial_dims(da)
        grid_lats = da[lat_dim].values
        grid_lons = da[lon_dim].values

        (i0, i1), (j0, j1), u, v = find_surrounding_grid_indices(
            target_lat, target_lon, grid_lats, grid_lons
        )

        q00 = da.isel({lat_dim: i0, lon_dim: j0})
        q01 = da.isel({lat_dim: i0, lon_dim: j1})
        q10 = da.isel({lat_dim: i1, lon_dim: j0})
        q11 = da.isel({lat_dim: i1, lon_dim: j1})

        result = (1.0 - u) * (1.0 - v) * q00 + (1.0 - u) * v * q01 + u * (1.0 - v) * q10 + u * v * q11

        res_da = result.drop_vars([lat_dim, lon_dim], errors="ignore")
        res_da.attrs = da.attrs.copy()
        res_da.attrs["interpolated_latitude"] = target_lat
        res_da.attrs["interpolated_longitude"] = target_lon
        return res_da

    def interpolate_dataset(
        self,
        ds: xr.Dataset,
        target_lat: float,
        target_lon: float,
    ) -> xr.Dataset:
        """Interpolate all spatial variables in an xarray Dataset to a target point."""
        interpolated_vars = {}
        for var_name, da in ds.data_vars.items():
            try:
                lat_dim, lon_dim = self._find_spatial_dims(da)
                if lat_dim in da.dims and lon_dim in da.dims:
                    interpolated_vars[var_name] = self.interpolate_dataarray(
                        da, target_lat, target_lon
                    )
                else:
                    interpolated_vars[var_name] = da.copy()
            except ValueError:
                interpolated_vars[var_name] = da.copy()

        res_ds = xr.Dataset(interpolated_vars, attrs=ds.attrs.copy())
        res_ds.attrs["interpolated_latitude"] = target_lat
        res_ds.attrs["interpolated_longitude"] = target_lon
        return res_ds

    def interpolate_station(
        self,
        ds_or_da: Union[xr.DataArray, xr.Dataset],
        station_id: str,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Interpolate dataset or dataarray to the exact geographic coordinates of a known station."""
        if station_id not in STATION_COORDINATES:
            raise ValueError(
                f"Unknown station_id: '{station_id}'. Known stations: {list(STATION_COORDINATES.keys())}"
            )

        coords = STATION_COORDINATES[station_id]
        target_lat = coords["latitude"]
        target_lon = coords["longitude"]

        if isinstance(ds_or_da, xr.DataArray):
            res = self.interpolate_dataarray(ds_or_da, target_lat, target_lon)
        elif isinstance(ds_or_da, xr.Dataset):
            res = self.interpolate_dataset(ds_or_da, target_lat, target_lon)
        else:
            raise TypeError(f"Expected xr.DataArray or xr.Dataset, got {type(ds_or_da)}")

        res.attrs["station_id"] = station_id
        res.attrs["target_latitude"] = target_lat
        res.attrs["target_longitude"] = target_lon
        res.attrs["station_elevation"] = coords["elevation"]
        return res
