#!/usr/bin/env python3
"""
Elevation lapse rate correction module (Task 1.3 T1.3-01).

Implements standard atmosphere lapse rate temperature corrections for stations
where the station physical altitude differs from the interpolated GEFS numerical
model or grid orography elevation.

Standard atmosphere lapse rate:
    Γ = 0.0065 K/m = 0.0065 °C/m (6.5 °C/km)

Correction formula:
    Δh = h_station - h_model
    T_corrected = T_interpolated - Γ * Δh
"""

from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

# Standard atmospheric lapse rate (International Standard Atmosphere): 0.0065 K/m
DEFAULT_LAPSE_RATE = 0.0065


def correct_elevation(
    temperature: Union[float, np.ndarray],
    station_elevation: Union[float, np.ndarray],
    model_elevation: Union[float, np.ndarray],
    lapse_rate: float = DEFAULT_LAPSE_RATE,
) -> Union[float, np.ndarray]:
    """Correct temperature using standard lapse rate based on elevation difference.

    Parameters
    ----------
    temperature : float or np.ndarray
        Uncorrected temperature in °C or K.
    station_elevation : float or np.ndarray
        Station elevation in meters above sea level.
    model_elevation : float or np.ndarray
        Model surface or geopotential elevation in meters above sea level.
    lapse_rate : float, default 0.0065
        Lapse rate in K/m (or °C/m). Must be positive.

    Returns
    -------
    float or np.ndarray
        Elevation-corrected temperature.
    """
    if lapse_rate <= 0:
        raise ValueError(f"Lapse rate must be positive, got {lapse_rate}")

    delta_h = station_elevation - model_elevation
    return temperature - lapse_rate * delta_h


class ElevationCorrector:
    """Class helper for elevation lapse rate corrections on various data structures."""

    def __init__(self, lapse_rate: float = DEFAULT_LAPSE_RATE):
        if lapse_rate <= 0:
            raise ValueError(f"Lapse rate must be positive, got {lapse_rate}")
        self.lapse_rate = lapse_rate

    def correct(
        self,
        temperature: Any,
        station_elevation: Any,
        model_elevation: Any,
    ) -> Any:
        """Apply lapse rate correction to scalar or numpy array."""
        return correct_elevation(
            temperature,
            station_elevation,
            model_elevation,
            lapse_rate=self.lapse_rate,
        )

    def correct_dataframe(
        self,
        df: pd.DataFrame,
        temp_columns: Iterable[str],
        station_elevation_col: Optional[str] = None,
        model_elevation_col: Optional[str] = None,
        station_elevation: Optional[float] = None,
        model_elevation: Optional[float] = None,
    ) -> pd.DataFrame:
        """Apply elevation correction to specified temperature columns in a DataFrame."""
        df_out = df.copy()

        if station_elevation_col is not None and station_elevation_col in df_out.columns:
            st_elev = df_out[station_elevation_col].values
        elif station_elevation is not None:
            st_elev = station_elevation
        else:
            raise ValueError("Must provide either station_elevation_col or station_elevation")

        if model_elevation_col is not None and model_elevation_col in df_out.columns:
            mod_elev = df_out[model_elevation_col].values
        elif model_elevation is not None:
            mod_elev = model_elevation
        else:
            raise ValueError("Must provide either model_elevation_col or model_elevation")

        for col in temp_columns:
            if col in df_out.columns:
                df_out[col] = correct_elevation(
                    df_out[col].values,
                    st_elev,
                    mod_elev,
                    lapse_rate=self.lapse_rate,
                )
        return df_out

    def correct_xarray(
        self,
        da_or_ds: Union[xr.DataArray, xr.Dataset],
        station_elevation: float,
        model_elevation: Union[float, xr.DataArray],
        var_names: Optional[Iterable[str]] = None,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Apply elevation correction to xarray DataArray or Dataset."""
        if isinstance(da_or_ds, xr.DataArray):
            corrected = correct_elevation(
                da_or_ds.values,
                station_elevation,
                model_elevation.values if isinstance(model_elevation, xr.DataArray) else model_elevation,
                lapse_rate=self.lapse_rate,
            )
            return da_or_ds.copy(data=corrected)
        elif isinstance(da_or_ds, xr.Dataset):
            ds_out = da_or_ds.copy()
            targets = var_names if var_names is not None else list(ds_out.data_vars.keys())
            mod_elev_val = (
                model_elevation.values
                if isinstance(model_elevation, xr.DataArray)
                else model_elevation
            )
            for var in targets:
                if var in ds_out:
                    corrected = correct_elevation(
                        ds_out[var].values,
                        station_elevation,
                        mod_elev_val,
                        lapse_rate=self.lapse_rate,
                    )
                    ds_out[var] = ds_out[var].copy(data=corrected)
            return ds_out
        else:
            raise TypeError(f"Expected xarray DataArray or Dataset, got {type(da_or_ds)}")
