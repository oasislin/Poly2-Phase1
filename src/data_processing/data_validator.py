#!/usr/bin/env python3
"""
Data integrity and schema validator for features and observations (Task 1.4 T1.4-01).

Validates:
- Schema column presence and types
- Absence of NaN/Inf in essential numeric features
- Physical domain limits (-60°C to +60°C for terrestrial temperatures)
- Internal consistency (variance >= 0, member_min <= mean <= member_max, temp_min <= temp_max)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Standard physical bounds for terrestrial surface temperature in Celsius
MIN_PHYSICAL_TEMP_C = -60.0
MAX_PHYSICAL_TEMP_C = 60.0

REQUIRED_FEATURE_COLUMNS = [
    "target_date",
    "station_id",
    "target_type",
    "lead_time_bucket",
    "ensemble_mean",
    "ensemble_variance",
    "member_max",
    "member_min",
]

REQUIRED_OBS_COLUMNS = [
    "date",
    "station_id",
    "temp_max",
    "temp_min",
]


class ValidationError(Exception):
    """Raised when data validation fails in strict mode."""


@dataclass
class ValidationResult:
    """Encapsulates data validation outcome and diagnostic messages."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Validates schema structure, numeric health, and physical consistency of data."""

    def __init__(
        self,
        strict: bool = False,
        min_temp: float = MIN_PHYSICAL_TEMP_C,
        max_temp: float = MAX_PHYSICAL_TEMP_C,
    ):
        self.strict = strict
        self.min_temp = min_temp
        self.max_temp = max_temp

    def _check_missing_columns(self, df: pd.DataFrame, required: List[str], errors: List[str]) -> None:
        """Check for presence of required columns."""
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")

    def _check_nulls_and_infs(self, df: pd.DataFrame, num_cols: List[str], errors: List[str]) -> None:
        """Check for NaN and infinite values in specified columns."""
        for col in num_cols:
            if col in df.columns:
                if df[col].isna().any():
                    errors.append(f"Column '{col}' contains NaN values.")
                if np.isinf(df[col].to_numpy(dtype=float, na_value=0.0)).any():
                    errors.append(f"Column '{col}' contains Inf values.")

    def _check_physical_temp_ranges(self, df: pd.DataFrame, temp_cols: List[str], errors: List[str]) -> None:
        """Check that temperatures stay within physically realistic limits."""
        for col in temp_cols:
            if col in df.columns:
                vals = df[col].dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    if np.any(vals < self.min_temp) or np.any(vals > self.max_temp):
                        errors.append(
                            f"Column '{col}' has values outside physical range [{self.min_temp}, {self.max_temp}] °C."
                        )

    def _check_feature_consistency(self, df: pd.DataFrame, errors: List[str]) -> None:
        """Check mathematical consistency across ensemble statistics."""
        if "ensemble_variance" in df.columns:
            if (df["ensemble_variance"] < 0).any():
                errors.append("Column 'ensemble_variance' contains negative variance values.")

        if "member_min" in df.columns and "member_max" in df.columns:
            if (df["member_min"] > df["member_max"]).any():
                errors.append("Found rows where member_min > member_max.")

        if "member_min" in df.columns and "ensemble_mean" in df.columns and "member_max" in df.columns:
            if (df["ensemble_mean"] < df["member_min"] - 1e-5).any() or (df["ensemble_mean"] > df["member_max"] + 1e-5).any():
                errors.append("Found rows where ensemble_mean is outside [member_min, member_max].")

    def validate_features(self, df: pd.DataFrame) -> ValidationResult:
        """Validate processed feature DataFrame."""
        errors: List[str] = []
        warnings: List[str] = []

        if df.empty:
            errors.append("Feature DataFrame is empty.")
            return self._finalize_result(errors, warnings)

        self._check_missing_columns(df, REQUIRED_FEATURE_COLUMNS, errors)
        num_cols = ["ensemble_mean", "ensemble_variance", "member_max", "member_min"]
        self._check_nulls_and_infs(df, num_cols, errors)
        self._check_physical_temp_ranges(df, ["ensemble_mean", "member_max", "member_min"], errors)
        self._check_feature_consistency(df, errors)

        return self._finalize_result(errors, warnings, {"row_count": len(df)})

    def validate_observations(self, df: pd.DataFrame) -> ValidationResult:
        """Validate historical weather station observations DataFrame."""
        errors: List[str] = []
        warnings: List[str] = []

        if df.empty:
            errors.append("Observation DataFrame is empty.")
            return self._finalize_result(errors, warnings)

        self._check_missing_columns(df, REQUIRED_OBS_COLUMNS, errors)
        self._check_nulls_and_infs(df, ["temp_max", "temp_min"], errors)
        self._check_physical_temp_ranges(df, ["temp_max", "temp_min"], errors)

        if "temp_min" in df.columns and "temp_max" in df.columns:
            if (df["temp_min"] > df["temp_max"]).any():
                errors.append("Found rows where temp_min exceeds temp_max.")

        return self._finalize_result(errors, warnings, {"row_count": len(df)})

    def _finalize_result(self, errors: List[str], warnings: List[str], details: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Construct ValidationResult and raise error if in strict mode."""
        is_valid = len(errors) == 0
        if not is_valid and self.strict:
            raise ValidationError(f"Data validation failed with {len(errors)} error(s): {errors}")
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, details=details or {})
