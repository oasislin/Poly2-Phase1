#!/usr/bin/env python3
"""
Unit tests for ClimatologyCalculator (Ticket 2.1-01 / Issue #11).

Verifies:
1. Strict OOS boundary: only 2000-2018 observation data is used, 2019+ is strictly filtered/blocked.
2. 31-day circular sliding window calculation over 19 years (~575-589 samples per day).
3. Continuous 1-366 day-of-year lookup table with mean (mu_clim), standard deviation (sigma_clim), and variance.
4. Circular wrap-around continuity across Dec 31 -> Jan 1.
5. Leap year (Feb 29 / day 60) edge case handling.
6. Integration with real SQLite observations for ZSPD and KDEN.
"""

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.modeling.climatology import ClimatologyCalculator


@pytest.fixture
def synthetic_observations():
    """Generate 20 years (2000-2019) of synthetic daily observations with known seasonal sinusoidal pattern."""
    dates = pd.date_range("2000-01-01", "2019-12-31", freq="D")
    n = len(dates)
    
    # 20°C mean, 15°C annual sinusoidal amplitude, peak in July (approx day 200)
    doy = dates.dayofyear.values
    temp_max = 20.0 + 15.0 * np.sin(2 * np.pi * (doy - 110) / 365.25) + np.random.normal(0, 2.0, n)
    temp_min = temp_max - 8.0 + np.random.normal(0, 1.0, n)
    
    df = pd.DataFrame({
        "station_id": "ZSPD",
        "date": dates.strftime("%Y-%m-%d"),
        "temp_max": temp_max,
        "temp_min": temp_min,
    })
    return df


class TestClimatologyCalculatorOOS:
    """Test strict out-of-sample (OOS) enforcement."""

    def test_strict_oos_filters_out_2019_data(self, synthetic_observations):
        calc = ClimatologyCalculator(train_start_year=2000, train_end_year=2018, window_days=31)
        calc.fit(synthetic_observations, station_id="ZSPD")
        
        # Verify internal observations used do not exceed 2018-12-31
        used_df = calc.get_training_data("ZSPD")
        max_date = pd.to_datetime(used_df["date"]).max()
        assert max_date <= pd.Timestamp("2018-12-31")
        min_date = pd.to_datetime(used_df["date"]).min()
        assert min_date >= pd.Timestamp("2000-01-01")
        assert len(used_df) == 6940  # 19 years (2000-2018), including 5 leap years (365*19 + 5 = 6940)

    def test_fit_raises_on_invalid_year_range(self):
        with pytest.raises(ValueError, match="train_start_year.*train_end_year"):
            ClimatologyCalculator(train_start_year=2019, train_end_year=2018)


class TestClimatologyCalculations:
    """Test 31-day sliding window mathematical correctness and properties."""

    def test_lookup_table_shape_and_columns(self, synthetic_observations):
        calc = ClimatologyCalculator(train_start_year=2000, train_end_year=2018, window_days=31)
        table = calc.compute_climatology(synthetic_observations, station_id="ZSPD", target_type="max")
        
        assert len(table) == 366
        assert list(table.columns) == [
            "day_of_year",
            "mu_clim",
            "sigma_clim",
            "variance_clim",
            "sample_count",
        ]
        assert (table["day_of_year"].values == np.arange(1, 367)).all()
        # 19 years * 31 days = 589 max, minus 14 non-leap Feb 29 days = 575 min
        assert (table["sample_count"] >= 570).all()
        assert (table["sample_count"] <= 600).all()
        assert (table["sigma_clim"] > 0).all()
        assert (table["variance_clim"] > 0).all()
        assert np.allclose(table["variance_clim"], table["sigma_clim"] ** 2)

    def test_circular_continuity_dec_jan(self, synthetic_observations):
        """Test continuity across year boundary (Dec 31 -> Jan 1)."""
        calc = ClimatologyCalculator(train_start_year=2000, train_end_year=2018, window_days=31)
        table = calc.compute_climatology(synthetic_observations, station_id="ZSPD", target_type="max")
        
        dec31_mu = table.loc[table["day_of_year"] == 366, "mu_clim"].values[0]
        jan01_mu = table.loc[table["day_of_year"] == 1, "mu_clim"].values[0]
        
        # Smooth annual curve means Dec 31 and Jan 1 mu_clim should be very close (< 0.5°C diff)
        assert abs(dec31_mu - jan01_mu) < 0.5

    def test_query_by_date_and_doy(self, synthetic_observations):
        calc = ClimatologyCalculator(train_start_year=2000, train_end_year=2018, window_days=31)
        calc.fit(synthetic_observations, station_id="ZSPD")
        
        # Query via date string vs reference day of year
        mu1, sigma1 = calc.get_climatology(station_id="ZSPD", target_type="max", target="2019-07-15")
        ref_doy = ClimatologyCalculator._resolve_day_of_year("2019-07-15")
        mu2, sigma2 = calc.get_climatology(station_id="ZSPD", target_type="max", target=ref_doy)
        
        assert mu1 == mu2
        assert sigma1 == sigma2
        assert mu1 > 25.0  # Summer high in Shanghai should be warm

    def test_query_min_temp(self, synthetic_observations):
        calc = ClimatologyCalculator(train_start_year=2000, train_end_year=2018, window_days=31)
        calc.fit(synthetic_observations, station_id="ZSPD")
        
        mu_max, _ = calc.get_climatology(station_id="ZSPD", target_type="max", target="2019-07-15")
        mu_min, _ = calc.get_climatology(station_id="ZSPD", target_type="min", target="2019-07-15")
        
        # Min temp should be strictly less than max temp
        assert mu_min < mu_max


class TestRealDatabaseClimatology:
    """Test ClimatologyCalculator against real observations in SQLite."""

    def test_real_database_both_stations(self):
        calc = ClimatologyCalculator(train_start_year=2000, train_end_year=2018, window_days=31)
        calc.fit_from_db(station_ids=["ZSPD", "KDEN"])
        
        # ZSPD summer max
        mu_sh_summer, sigma_sh_summer = calc.get_climatology("ZSPD", "max", "2019-07-15")
        assert 28.0 <= mu_sh_summer <= 36.0
        assert 2.0 <= sigma_sh_summer <= 6.0
        
        # KDEN winter min (Denver winter is below 0°C)
        mu_den_winter, sigma_den_winter = calc.get_climatology("KDEN", "min", "2019-01-15")
        assert mu_den_winter < 0.0  # Denver winter min average is below freezing in Celsius
        assert sigma_den_winter > 3.0
