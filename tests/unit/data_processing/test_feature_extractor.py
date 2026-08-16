"""Unit tests for feature extractor module (Task 1.3 T1.3-04)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_processing.feature_extractor import (
    FeatureExtractor,
    calculate_ensemble_statistics,
    collapse_daily_extreme,
)


class TestDailyExtremeCollapse:
    """Test 6h window extreme collapsing to daily max / min extremes."""

    def test_collapse_tmax_takes_maximum_over_windows(self):
        # 5 members, 3 windows
        # Members across 3 windows (e.g. [24, 30, 36])
        data = np.array([
            [25.0, 32.0, 28.0],  # member 0 -> max is 32.0
            [26.0, 30.5, 29.0],  # member 1 -> max is 30.5
            [24.5, 33.0, 27.5],  # member 2 -> max is 33.0
            [27.0, 31.0, 30.0],  # member 3 -> max is 31.0
            [25.5, 31.5, 28.5],  # member 4 -> max is 31.5
        ])
        da = xr.DataArray(
            data,
            dims=["member", "window"],
            coords={"member": ["c00", "p01", "p02", "p03", "p04"], "window": [24, 30, 36]},
        )

        collapsed = collapse_daily_extreme(da, target_type="max")

        assert "window" not in collapsed.dims
        assert collapsed.dims == ("member",)
        np.testing.assert_allclose(
            collapsed.values,
            np.array([32.0, 30.5, 33.0, 31.0, 31.5]),
        )

    def test_collapse_tmin_takes_minimum_over_windows(self):
        data = np.array([
            [15.0, 12.0, 18.0],  # member 0 -> min is 12.0
            [16.0, 10.5, 19.0],  # member 1 -> min is 10.5
            [14.5, 13.0, 17.5],  # member 2 -> min is 13.0
            [17.0, 11.0, 20.0],  # member 3 -> min is 11.0
            [15.5, 11.5, 18.5],  # member 4 -> min is 11.5
        ])
        da = xr.DataArray(
            data,
            dims=["member", "window"],
            coords={"member": ["c00", "p01", "p02", "p03", "p04"], "window": [24, 30, 36]},
        )

        collapsed = collapse_daily_extreme(da, target_type="min")

        assert "window" not in collapsed.dims
        np.testing.assert_allclose(
            collapsed.values,
            np.array([12.0, 10.5, 13.0, 11.0, 11.5]),
        )

    def test_invalid_target_type_raises(self):
        da = xr.DataArray([[1.0]], dims=["member", "window"])
        with pytest.raises(ValueError, match="target_type must be 'max' or 'min'"):
            collapse_daily_extreme(da, target_type="mean")


class TestEnsembleStatisticsCalculation:
    """Test calculation of 5-member ensemble statistics."""

    def test_ensemble_statistics_exact_math(self):
        # 5 members: [30.0, 32.0, 28.0, 34.0, 36.0]
        # Mean = (30 + 32 + 28 + 34 + 36) / 5 = 160 / 5 = 32.0
        # Differences from mean: [-2, 0, -4, 2, 4]
        # Squared diffs: [4, 0, 16, 4, 16] -> Sum = 40
        # Sample variance (ddof=1) = 40 / 4 = 10.0
        # Member max = 36.0, Member min = 28.0
        members = np.array([30.0, 32.0, 28.0, 34.0, 36.0])

        stats = calculate_ensemble_statistics(members)

        assert stats["ensemble_mean"] == pytest.approx(32.0, abs=1e-5)
        assert stats["ensemble_variance"] == pytest.approx(10.0, abs=1e-5)
        assert stats["member_max"] == pytest.approx(36.0, abs=1e-5)
        assert stats["member_min"] == pytest.approx(28.0, abs=1e-5)

        # Confirm deprecated features are NOT present (v5.9.1 alignment)
        assert "ensemble_p10" not in stats
        assert "ensemble_p90" not in stats
        assert "day_of_year_sin" not in stats
        assert "day_of_year_cos" not in stats

    def test_calculate_from_xarray_dataarray(self):
        da = xr.DataArray(
            [20.0, 22.0, 24.0, 26.0, 28.0],
            dims=["member"],
            coords={"member": ["c00", "p01", "p02", "p03", "p04"]},
        )
        stats = calculate_ensemble_statistics(da)
        assert stats["ensemble_mean"] == pytest.approx(24.0)
        assert stats["member_max"] == pytest.approx(28.0)
        assert stats["member_min"] == pytest.approx(20.0)


class TestFeatureExtractorClass:
    """Test full FeatureExtractor pipeline on station datasets."""

    def test_extract_station_features_max(self):
        extractor = FeatureExtractor()

        # 5 members x 3 windows
        tmax_data = np.array([
            [25.0, 32.0, 28.0],  # max 32.0
            [26.0, 30.0, 29.0],  # max 30.0
            [24.0, 34.0, 27.0],  # max 34.0
            [27.0, 31.0, 30.0],  # max 31.0
            [25.0, 33.0, 28.0],  # max 33.0
        ])
        # Collapsed: [32, 30, 34, 31, 33] -> Mean = 32.0, Var = 2.5, Max = 34, Min = 30
        station_ds = xr.Dataset(
            {
                "tmax": (["member", "window"], tmax_data),
            },
            coords={
                "member": ["c00", "p01", "p02", "p03", "p04"],
                "window": [24, 30, 36],
            },
            attrs={"station_id": "ZSPD"},
        )

        features = extractor.extract_features(
            station_ds=station_ds,
            target_type="max",
            target_date="2019-07-02",
            lead_time_bucket=30,
        )

        assert isinstance(features, dict)
        assert features["station_id"] == "ZSPD"
        assert features["target_type"] == "max"
        assert features["lead_time_bucket"] == 30
        assert features["ensemble_mean"] == pytest.approx(32.0, abs=1e-4)
        assert features["ensemble_variance"] == pytest.approx(2.5, abs=1e-4)
        assert features["member_max"] == pytest.approx(34.0, abs=1e-4)
        assert features["member_min"] == pytest.approx(30.0, abs=1e-4)

    def test_to_dataframe(self):
        extractor = FeatureExtractor()
        feature_dict = {
            "target_date": "2019-07-02",
            "station_id": "ZSPD",
            "target_type": "max",
            "lead_time_bucket": 30,
            "ensemble_mean": 32.0,
            "ensemble_variance": 2.5,
            "member_max": 34.0,
            "member_min": 30.0,
        }
        df = extractor.to_dataframe([feature_dict])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == [
            "target_date",
            "station_id",
            "target_type",
            "lead_time_bucket",
            "ensemble_mean",
            "ensemble_variance",
            "member_max",
            "member_min",
        ]
