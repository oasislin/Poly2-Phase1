"""Unit tests for data validation and integrity checker (Task 1.4 T1.4-01)."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.data_validator import (
    DataValidator,
    ValidationError,
    ValidationResult,
)


class TestFeatureDataFrameValidation:
    """Test validation rules for processed modeling feature DataFrames."""

    @pytest.fixture
    def valid_feature_df(self):
        return pd.DataFrame({
            "target_date": ["2019-07-02", "2019-07-03"],
            "station_id": ["ZSPD", "ZSPD"],
            "target_type": ["max", "max"],
            "lead_time_bucket": [30, 54],
            "ensemble_mean": [28.5, 30.2],
            "ensemble_variance": [1.5, 2.1],
            "member_max": [31.0, 33.0],
            "member_min": [26.0, 28.0],
        })

    def test_valid_dataframe_passes(self, valid_feature_df):
        validator = DataValidator()
        result = validator.validate_features(valid_feature_df)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_column_fails(self, valid_feature_df):
        validator = DataValidator()
        invalid_df = valid_feature_df.drop(columns=["ensemble_variance"])
        result = validator.validate_features(invalid_df)
        assert result.is_valid is False
        assert any("Missing required column" in err for err in result.errors)

    def test_nan_or_inf_in_features_fails(self, valid_feature_df):
        validator = DataValidator()
        invalid_df = valid_feature_df.copy()
        invalid_df.loc[0, "ensemble_mean"] = np.nan
        result = validator.validate_features(invalid_df)
        assert result.is_valid is False
        assert any("NaN" in err for err in result.errors)

    def test_negative_variance_fails(self, valid_feature_df):
        validator = DataValidator()
        invalid_df = valid_feature_df.copy()
        invalid_df.loc[0, "ensemble_variance"] = -0.5
        result = validator.validate_features(invalid_df)
        assert result.is_valid is False
        assert any("variance" in err.lower() for err in result.errors)

    def test_member_min_greater_than_max_fails(self, valid_feature_df):
        validator = DataValidator()
        invalid_df = valid_feature_df.copy()
        invalid_df.loc[0, "member_min"] = 35.0  # min > max (31.0)
        result = validator.validate_features(invalid_df)
        assert result.is_valid is False
        assert any("member_min" in err for err in result.errors)

    def test_unphysical_temperature_fails(self, valid_feature_df):
        validator = DataValidator()
        invalid_df = valid_feature_df.copy()
        invalid_df.loc[0, "ensemble_mean"] = 85.0  # 85°C is physically impossible on Earth
        result = validator.validate_features(invalid_df)
        assert result.is_valid is False
        assert any("physical range" in err.lower() for err in result.errors)

    def test_strict_mode_raises_exception(self, valid_feature_df):
        validator = DataValidator(strict=True)
        invalid_df = valid_feature_df.drop(columns=["station_id"])
        with pytest.raises(ValidationError):
            validator.validate_features(invalid_df)


class TestObservationDataFrameValidation:
    """Test validation rules for Wunderground / station observation DataFrames."""

    @pytest.fixture
    def valid_obs_df(self):
        return pd.DataFrame({
            "date": ["2019-07-02", "2019-07-03"],
            "station_id": ["ZSPD", "ZSPD"],
            "temp_max": [33.0, 35.0],
            "temp_min": [25.0, 26.0],
        })

    def test_valid_observations_pass(self, valid_obs_df):
        validator = DataValidator()
        result = validator.validate_observations(valid_obs_df)
        assert result.is_valid is True

    def test_temp_min_greater_than_max_fails(self, valid_obs_df):
        validator = DataValidator()
        invalid_df = valid_obs_df.copy()
        invalid_df.loc[0, "temp_min"] = 38.0  # min (38) > max (33)
        result = validator.validate_observations(invalid_df)
        assert result.is_valid is False
        assert any("temp_min exceeds temp_max" in err for err in result.errors)
