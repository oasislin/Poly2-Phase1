"""Unit tests for elevation corrector module (Task 1.3 T1.3-01)."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.elevation_corrector import (
    DEFAULT_LAPSE_RATE,
    ElevationCorrector,
    correct_elevation,
)


class TestElevationCorrectionFormulas:
    """Test standard atmosphere lapse rate elevation corrections."""

    def test_default_lapse_rate_constant(self):
        # Standard lapse rate is 0.0065 K/m (6.5 K/km)
        assert DEFAULT_LAPSE_RATE == pytest.approx(0.0065, rel=1e-5)

    def test_higher_station_elevation_cools_temperature(self):
        # Station at 1000m, Model at 0m (station is 1000m higher)
        # Delta_h = +1000m, Temp drop = 0.0065 * 1000 = 6.5 °C
        t_model = 20.0
        t_corr = correct_elevation(
            temperature=t_model,
            station_elevation=1000.0,
            model_elevation=0.0,
        )
        assert t_corr == pytest.approx(13.5, rel=1e-4)

    def test_lower_station_elevation_warms_temperature(self):
        # Station at 100m, Model at 500m (station is 400m lower)
        # Delta_h = -400m, Temp rise = 0.0065 * 400 = 2.6 °C
        t_model = 10.0
        t_corr = correct_elevation(
            temperature=t_model,
            station_elevation=100.0,
            model_elevation=500.0,
        )
        assert t_corr == pytest.approx(12.6, rel=1e-4)

    def test_zero_elevation_difference(self):
        t_model = 25.0
        t_corr = correct_elevation(
            temperature=t_model,
            station_elevation=50.0,
            model_elevation=50.0,
        )
        assert t_corr == pytest.approx(25.0, rel=1e-4)

    def test_denver_station_elevation_correction(self):
        # KDEN airport elevation ~1655m. Assume model grid elevation is 1600m.
        # Delta_h = +55m -> Temp adjustment = -0.0065 * 55 = -0.3575 °C
        t_model = 30.0
        t_corr = correct_elevation(
            temperature=t_model,
            station_elevation=1655.0,
            model_elevation=1600.0,
        )
        assert t_corr == pytest.approx(30.0 - 0.0065 * 55.0, rel=1e-4)

    def test_invalid_lapse_rate_raises(self):
        with pytest.raises(ValueError, match="Lapse rate must be positive"):
            correct_elevation(20.0, 100.0, 0.0, lapse_rate=-0.001)


class TestElevationCorrectorClass:
    """Test ElevationCorrector helper on arrays and dataframes."""

    def test_numpy_array_correction(self):
        corrector = ElevationCorrector(lapse_rate=0.0065)
        temps = np.array([10.0, 20.0, 30.0])
        # station_elev = 500, model_elev = 0 -> -3.25 °C
        corrected = corrector.correct(temps, station_elevation=500.0, model_elevation=0.0)
        np.testing.assert_allclose(corrected, np.array([6.75, 16.75, 26.75]), atol=1e-4)

    def test_dataframe_correction_on_tmax_and_tmin(self):
        corrector = ElevationCorrector()
        df = pd.DataFrame({
            "station": ["KDEN", "KDEN"],
            "tmax": [25.0, 30.0],
            "tmin": [10.0, 15.0],
            "model_elev": [1500.0, 1500.0],
            "station_elev": [1655.0, 1655.0],
        })

        delta_h = 155.0
        expected_diff = 0.0065 * delta_h  # ~1.0075

        res_df = corrector.correct_dataframe(
            df=df,
            temp_columns=["tmax", "tmin"],
            station_elevation_col="station_elev",
            model_elevation_col="model_elev",
        )

        assert res_df["tmax"].iloc[0] == pytest.approx(25.0 - expected_diff, abs=1e-4)
        assert res_df["tmin"].iloc[0] == pytest.approx(10.0 - expected_diff, abs=1e-4)
        assert res_df["tmax"].iloc[1] == pytest.approx(30.0 - expected_diff, abs=1e-4)
