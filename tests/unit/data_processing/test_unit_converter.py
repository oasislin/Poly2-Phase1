"""Unit tests for unit conversion module (Task 1.3 T1.3-01)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_processing.unit_converter import (
    UnitConverter,
    celsius_to_fahrenheit,
    celsius_to_kelvin,
    convert_temperature,
    fahrenheit_to_celsius,
    fahrenheit_to_kelvin,
    kelvin_to_celsius,
    kelvin_to_fahrenheit,
)


class TestScalarUnitConversions:
    """Test scalar temperature conversions across Celsius, Fahrenheit, Kelvin."""

    def test_celsius_to_kelvin_and_back(self):
        # 0 °C == 273.15 K
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15, rel=1e-5)
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0, abs=1e-5)

        # 100 °C == 373.15 K
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15, rel=1e-5)
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0, abs=1e-5)

        # -40 °C == 233.15 K
        assert celsius_to_kelvin(-40.0) == pytest.approx(233.15, rel=1e-5)
        assert kelvin_to_celsius(233.15) == pytest.approx(-40.0, abs=1e-5)

    def test_celsius_to_fahrenheit_and_back(self):
        # 0 °C == 32 °F
        assert celsius_to_fahrenheit(0.0) == pytest.approx(32.0, rel=1e-5)
        assert fahrenheit_to_celsius(32.0) == pytest.approx(0.0, abs=1e-5)

        # 100 °C == 212 °F
        assert celsius_to_fahrenheit(100.0) == pytest.approx(212.0, rel=1e-5)
        assert fahrenheit_to_celsius(212.0) == pytest.approx(100.0, abs=1e-5)

        # -40 °C == -40 °F
        assert celsius_to_fahrenheit(-40.0) == pytest.approx(-40.0, rel=1e-5)
        assert fahrenheit_to_celsius(-40.0) == pytest.approx(-40.0, abs=1e-5)

        # 37 °C == 98.6 °F
        assert celsius_to_fahrenheit(37.0) == pytest.approx(98.6, rel=1e-5)
        assert fahrenheit_to_celsius(98.6) == pytest.approx(37.0, abs=1e-5)

    def test_kelvin_to_fahrenheit_and_back(self):
        # 273.15 K == 32 °F
        assert kelvin_to_fahrenheit(273.15) == pytest.approx(32.0, rel=1e-5)
        assert fahrenheit_to_kelvin(32.0) == pytest.approx(273.15, rel=1e-5)

        # 373.15 K == 212 °F
        assert kelvin_to_fahrenheit(373.15) == pytest.approx(212.0, rel=1e-5)
        assert fahrenheit_to_kelvin(32.0) == pytest.approx(273.15, rel=1e-5)

    def test_generic_convert_temperature(self):
        assert convert_temperature(300.0, from_unit="K", to_unit="C") == pytest.approx(26.85, abs=1e-2)
        assert convert_temperature(86.0, from_unit="degF", to_unit="degC") == pytest.approx(30.0, abs=1e-2)
        assert convert_temperature(20.0, from_unit="C", to_unit="C") == 20.0

    def test_invalid_units_raise_error(self):
        with pytest.raises(ValueError, match="Unsupported temperature unit"):
            convert_temperature(100, from_unit="invalid", to_unit="C")
        with pytest.raises(ValueError, match="Unsupported temperature unit"):
            convert_temperature(100, from_unit="C", to_unit="invalid")


class TestVectorizedConversions:
    """Test conversions on numpy arrays, pandas series/dataframes, xarray objects."""

    def test_numpy_array_conversion(self):
        converter = UnitConverter()
        arr_k = np.array([273.15, 293.15, 313.15])
        arr_c = converter.convert(arr_k, from_unit="K", to_unit="C")
        np.testing.assert_allclose(arr_c, np.array([0.0, 20.0, 40.0]), atol=1e-5)

    def test_pandas_dataframe_conversion(self):
        converter = UnitConverter()
        df = pd.DataFrame({
            "station": ["KDEN", "KDEN"],
            "temp_max": [86.0, 68.0],
            "temp_min": [50.0, 32.0],
            "humidity": [45, 60],
        })

        converted_df = converter.convert_dataframe(
            df,
            columns=["temp_max", "temp_min"],
            from_unit="F",
            to_unit="C",
        )

        assert converted_df["temp_max"].iloc[0] == pytest.approx(30.0, abs=1e-4)
        assert converted_df["temp_max"].iloc[1] == pytest.approx(20.0, abs=1e-4)
        assert converted_df["temp_min"].iloc[0] == pytest.approx(10.0, abs=1e-4)
        assert converted_df["temp_min"].iloc[1] == pytest.approx(0.0, abs=1e-4)
        assert converted_df["humidity"].iloc[0] == 45  # Unchanged

    def test_xarray_dataset_conversion(self):
        converter = UnitConverter()
        data = np.array([[273.15, 283.15], [293.15, 303.15]])
        ds = xr.Dataset(
            {
                "tmax": (["lat", "lon"], data),
                "tmin": (["lat", "lon"], data - 10.0),
                "elevation": (["lat", "lon"], [[100.0, 200.0], [300.0, 400.0]]),
            },
            coords={"lat": [30.0, 31.0], "lon": [120.0, 121.0]},
        )

        # Convert only tmax and tmin from K to C
        ds_c = converter.convert_xarray(ds, from_unit="K", to_unit="C", var_names=["tmax", "tmin"])

        np.testing.assert_allclose(ds_c["tmax"].values, np.array([[0.0, 10.0], [20.0, 30.0]]), atol=1e-5)
        np.testing.assert_allclose(ds_c["tmin"].values, np.array([[-10.0, 0.0], [10.0, 20.0]]), atol=1e-5)
        # Elevation remains untouched
        np.testing.assert_allclose(ds_c["elevation"].values, ds["elevation"].values)


class TestStationAutoConversion:
    """Test auto-conversion to standard Celsius based on station or source convention."""

    def test_convert_by_station(self):
        converter = UnitConverter()
        # KDEN default Wunderground is Fahrenheit
        val_c = converter.to_standard_celsius(77.0, station_id="KDEN")
        assert val_c == pytest.approx(25.0, abs=1e-4)

        # ZSPD default Wunderground is Celsius
        val_c_shanghai = converter.to_standard_celsius(25.0, station_id="ZSPD")
        assert val_c_shanghai == 25.0

    def test_convert_by_source(self):
        converter = UnitConverter()
        # GEFS source is Kelvin
        val_c = converter.to_standard_celsius(298.15, source="gefs")
        assert val_c == pytest.approx(25.0, abs=1e-4)
