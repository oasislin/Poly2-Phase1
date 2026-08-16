#!/usr/bin/env python3
"""
Unit conversion utilities for temperature standardization (Task 1.3 T1.3-01).

Supports scalar, numpy, pandas, and xarray conversions between Celsius (°C),
Fahrenheit (°F), and Kelvin (K). All internal modeling in this system uses
Celsius as the standard unit.
"""

from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.data_processing.constants import STATION_DEFAULT_UNITS

# Standard unit aliases mapping to canonical forms ('C', 'F', 'K')
UNIT_ALIASES = {
    "c": "C",
    "celsius": "C",
    "degc": "C",
    "°c": "C",
    "f": "F",
    "fahrenheit": "F",
    "degf": "F",
    "°f": "F",
    "k": "K",
    "kelvin": "K",
    "degk": "K",
}

SOURCE_DEFAULT_UNITS = {
    "gefs": "K",
    "wunderground": "C",  # Overridden by station defaults if station is given
    "reforecast": "K",
}


def _normalize_unit(unit: str) -> str:
    """Normalize unit string to canonical 'C', 'F', or 'K'."""
    normalized = UNIT_ALIASES.get(unit.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported temperature unit: {unit}. Supported units: C, F, K."
        )
    return normalized


def celsius_to_kelvin(c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Celsius (°C) to Kelvin (K)."""
    return c + 273.15


def kelvin_to_celsius(k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Kelvin (K) to Celsius (°C)."""
    return k - 273.15


def fahrenheit_to_celsius(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Fahrenheit (°F) to Celsius (°C)."""
    return (f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Celsius (°C) to Fahrenheit (°F)."""
    return c * 9.0 / 5.0 + 32.0


def fahrenheit_to_kelvin(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Fahrenheit (°F) to Kelvin (K)."""
    return celsius_to_kelvin(fahrenheit_to_celsius(f))


def kelvin_to_fahrenheit(k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Kelvin (K) to Fahrenheit (°F)."""
    return celsius_to_fahrenheit(kelvin_to_celsius(k))


def convert_temperature(
    val: Union[float, np.ndarray],
    from_unit: str,
    to_unit: str = "C",
) -> Union[float, np.ndarray]:
    """Convert temperature value(s) between supported units (C, F, K)."""
    src = _normalize_unit(from_unit)
    dst = _normalize_unit(to_unit)

    if src == dst:
        return val

    if src == "K" and dst == "C":
        return kelvin_to_celsius(val)
    elif src == "C" and dst == "K":
        return celsius_to_kelvin(val)
    elif src == "F" and dst == "C":
        return fahrenheit_to_celsius(val)
    elif src == "C" and dst == "F":
        return celsius_to_fahrenheit(val)
    elif src == "K" and dst == "F":
        return kelvin_to_fahrenheit(val)
    elif src == "F" and dst == "K":
        return fahrenheit_to_kelvin(val)
    else:
        raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")


class UnitConverter:
    """Class wrapper providing vectorized and object-aware conversions."""

    def convert(
        self,
        data: Any,
        from_unit: str,
        to_unit: str = "C",
    ) -> Any:
        """Convert data (scalar or array) from one unit to another."""
        return convert_temperature(data, from_unit, to_unit)

    def convert_dataframe(
        self,
        df: pd.DataFrame,
        columns: Iterable[str],
        from_unit: str,
        to_unit: str = "C",
    ) -> pd.DataFrame:
        """Convert specified temperature columns in a pandas DataFrame.

        Returns a shallow copy with modified columns.
        """
        df_out = df.copy()
        for col in columns:
            if col in df_out.columns:
                df_out[col] = convert_temperature(
                    df_out[col].values, from_unit=from_unit, to_unit=to_unit
                )
        return df_out

    def convert_xarray(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        from_unit: str,
        to_unit: str = "C",
        var_names: Optional[Iterable[str]] = None,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Convert temperature variables in an xarray DataArray or Dataset."""
        if isinstance(obj, xr.DataArray):
            converted_data = convert_temperature(obj.values, from_unit, to_unit)
            da_out = obj.copy(data=converted_data)
            da_out.attrs["units"] = f"deg{_normalize_unit(to_unit)}"
            return da_out
        elif isinstance(obj, xr.Dataset):
            ds_out = obj.copy()
            targets = var_names if var_names is not None else list(ds_out.data_vars.keys())
            for var in targets:
                if var in ds_out:
                    converted_data = convert_temperature(
                        ds_out[var].values, from_unit, to_unit
                    )
                    ds_out[var] = ds_out[var].copy(data=converted_data)
                    ds_out[var].attrs["units"] = f"deg{_normalize_unit(to_unit)}"
            return ds_out
        else:
            raise TypeError(f"Expected xarray DataArray or Dataset, got {type(obj)}")

    def to_standard_celsius(
        self,
        data: Any,
        station_id: Optional[str] = None,
        source: Optional[str] = None,
        from_unit: Optional[str] = None,
    ) -> Any:
        """Convert data to internal standard Celsius (°C) automatically based on metadata."""
        if from_unit is not None:
            unit = from_unit
        elif station_id is not None and station_id in STATION_DEFAULT_UNITS:
            unit = STATION_DEFAULT_UNITS[station_id]
        elif source is not None and source.lower() in SOURCE_DEFAULT_UNITS:
            unit = SOURCE_DEFAULT_UNITS[source.lower()]
        else:
            unit = "C"

        return convert_temperature(data, from_unit=unit, to_unit="C")
