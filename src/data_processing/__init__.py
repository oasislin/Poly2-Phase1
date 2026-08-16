"""Data processing package for Polymarket Temperature Prediction System."""

from src.data_processing.constants import (
    STATION_COORDINATES,
    STATION_DEFAULT_UNITS,
    STATION_METADATA,
    STATION_TIMEZONES,
)
from src.data_processing.data_processor import DataProcessor
from src.data_processing.data_validator import (
    DataValidator,
    ValidationError,
    ValidationResult,
)
from src.data_processing.database import TimeSeriesDatabase
from src.data_processing.elevation_corrector import (
    DEFAULT_LAPSE_RATE,
    ElevationCorrector,
    correct_elevation,
)
from src.data_processing.feature_extractor import (
    FeatureExtractor,
    calculate_ensemble_statistics,
    collapse_daily_extreme,
)
from src.data_processing.parquet_store import ParquetFeatureStore
from src.data_processing.spatial_interpolator import (
    SpatialInterpolator,
    bilinear_interp_2d,
    find_surrounding_grid_indices,
    normalize_longitude,
)
from src.data_processing.storage_manager import StorageManager
from src.data_processing.time_aligner import (
    ForecastWindow,
    TimeAligner,
    TimeAlignmentError,
    calculate_sunrise_time,
    get_local_day_bounds_utc,
    select_contained_6h_windows,
    select_contained_window_objects,
    verify_sunrise_coverage,
)
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

__all__ = [
    "DataProcessor",
    "DataValidator",
    "ValidationResult",
    "ValidationError",
    "ParquetFeatureStore",
    "TimeSeriesDatabase",
    "StorageManager",
    "STATION_METADATA",
    "STATION_COORDINATES",
    "STATION_TIMEZONES",
    "STATION_DEFAULT_UNITS",
    "ForecastWindow",
    "TimeAligner",
    "TimeAlignmentError",
    "get_local_day_bounds_utc",
    "select_contained_6h_windows",
    "select_contained_window_objects",
    "calculate_sunrise_time",
    "verify_sunrise_coverage",
    "DEFAULT_LAPSE_RATE",
    "ElevationCorrector",
    "correct_elevation",
    "SpatialInterpolator",
    "bilinear_interp_2d",
    "find_surrounding_grid_indices",
    "normalize_longitude",
    "FeatureExtractor",
    "calculate_ensemble_statistics",
    "collapse_daily_extreme",
    "UnitConverter",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "fahrenheit_to_celsius",
    "celsius_to_fahrenheit",
    "fahrenheit_to_kelvin",
    "kelvin_to_fahrenheit",
    "convert_temperature",
]
