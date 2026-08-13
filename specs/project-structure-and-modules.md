# Project Structure and Module Design

## Directory Structure

```
polymarket-temperature-prediction/
├── data/                          # Data storage
│   ├── raw/                       # Raw downloaded data
│   │   ├── wunderground/          # Wunderground HTML/JSON
│   │   │   ├── ZSPD/              # Shanghai data
│   │   │   └── KDEN/              # Denver data
│   │   └── gefs/                  # GEFS GRIB2 files
│   │       ├── reforecast/        # Historical forecasts
│   │       └── realtime/          # Real-time forecasts
│   ├── processed/                 # Processed data
│   │   ├── features/              # Extracted features
│   │   ├── training/              # Training datasets
│   │   └── validation/            # Validation datasets
│   └── models/                    # Trained models
│       ├── max_temp/              # Maximum temperature models
│       │   ├── DJF/               # Winter models
│       │   ├── MAM/               # Spring models
│       │   ├── JJA/               # Summer models
│       │   └── SON/               # Fall models
│       └── min_temp/              # Minimum temperature models
│           ├── DJF/
│           ├── MAM/
│           ├── JJA/
│           └── SON/
├── src/                           # Source code
│   ├── data_acquisition/          # Data fetching modules
│   │   ├── __init__.py
│   │   ├── wunderground_scraper.py
│   │   ├── gefs_fetcher.py
│   │   └── realtime_observer.py
│   ├── data_processing/          # Data processing modules
│   │   ├── __init__.py
│   │   ├── time_aligner.py
│   │   ├── unit_converter.py
│   │   ├── spatial_interpolator.py
│   │   ├── elevation_corrector.py
│   │   ├── feature_extractor.py
│   │   └── quality_control.py
│   ├── modeling/                  # Model implementation
│   │   ├── __init__.py
│   │   ├── skewed_gaussian.py
│   │   ├── emos_trainer.py
│   │   ├── seasonal_trainer.py
│   │   └── model_registry.py
│   ├── prediction/               # Prediction generation
│   │   ├── __init__.py
│   │   ├── static_predictor.py
│   │   ├── dynamic_corrector.py
│   │   ├── constraint_enforcer.py
│   │   └── bin_converter.py
│   ├── validation/               # Validation and monitoring
│   │   ├── __init__.py
│   │   ├── metrics_calculator.py
│   │   ├── backtester.py
│   │   ├── alert_manager.py
│   │   └── report_generator.py
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── file_utils.py
│   └── pipeline/                 # Pipeline orchestration
│       ├── __init__.py
│       ├── training_pipeline.py
│       ├── prediction_pipeline.py
│       └── main_pipeline.py
├── tests/                        # Test suite
│   ├── unit/
│   │   ├── data_acquisition/
│   │   ├── data_processing/
│   │   ├── modeling/
│   │   └── prediction/
│   ├── integration/
│   ├── e2e/
│   └── conftest.py
├── configs/                      # Configuration files
│   ├── stations.yaml             # Station metadata
│   ├── model_params.yaml         # Model parameters
│   ├── validation_thresholds.yaml # Validation thresholds
│   └── alerts.yaml               # Alert configurations
├── scripts/                      # Utility scripts
│   ├── download_data.py          # Data download script
│   ├── train_model.py           # Model training script
│   ├── run_predictions.py       # Prediction script
│   └── validate_model.py        # Validation script
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   ├── model_development.ipynb
│   └── validation_results.ipynb
├── docs/                         # Documentation
│   ├── api/                      # API documentation
│   ├── user_guide/               # User guide
│   └── technical/                # Technical documentation
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Project configuration
├── .github/workflows/           # CI/CD pipelines
│   └── tests.yml
└── README.md                     # Project overview
```

## Module Specifications

### 1. Data Acquisition Modules

#### `wunderground_scraper.py`
```python
class WundergroundScraper:
    """Scrape historical temperature data from Wunderground"""
    
    def fetch_station_data(
        self, 
        station_id: str, 
        start_date: date, 
        end_date: date
    ) -> List[Dict]:
        """Fetch daily max/min temperatures for a station and date range"""
        
    def parse_daily_data(
        self, 
        html_content: str, 
        station_id: str
    ) -> List[Dict]:
        """Parse HTML to extract temperature data"""
        
    def validate_data_quality(
        self, 
        data: List[Dict]
    ) -> Dict[str, Any]:
        """Validate scraped data for completeness and consistency"""
```

#### `gefs_fetcher.py`
```python
class GEFSFetcher:
    """Download and process GEFS forecast data"""
    
    def download_reforecast(
        self,
        region_bounds: Dict[str, Tuple[float, float]],
        date_range: Tuple[date, date]
    ) -> xr.Dataset:
        """Download historical GEFS reforecast data for a region"""
        
    def download_realtime(
        self,
        region_bounds: Dict[str, Tuple[float, float]],
        forecast_time: datetime
    ) -> xr.Dataset:
        """Download real-time GEFS forecast data for a region"""
        
    def extract_region(
        self,
        dataset: xr.Dataset,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float]
    ) -> xr.Dataset:
        """Extract subset of data for target region"""
```

### 2. Data Processing Modules

#### `time_aligner.py`
```python
class TimeAligner:
    """Align forecast and observation times to local station time"""
    
    def utc_to_local(
        self, 
        utc_time: datetime, 
        timezone_str: str
    ) -> datetime:
        """Convert UTC time to station local time"""
        
    def extract_daily_window(
        self,
        time_series: xr.DataArray,
        target_date: date,
        timezone_str: str
    ) -> xr.DataArray:
        """Extract data for local daily window (00:00-23:59)"""
        
    def align_forecast_windows(
        self,
        max_temp_forecast: xr.Dataset,
        min_temp_forecast: xr.Dataset,
        station_meta: Dict
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """Align forecast windows for max/min temperature prediction"""
```

#### `feature_extractor.py`
```python
class FeatureExtractor:
    """Extract features from GEFS ensemble forecasts"""
    
    def calculate_ensemble_stats(
        self,
        ensemble_data: xr.DataArray
    ) -> Dict[str, float]:
        """Calculate ensemble statistics (mean, std, percentiles)"""
        
    def extract_temporal_features(
        self,
        forecast_time: datetime
    ) -> Dict[str, float]:
        """Extract temporal features (day of year, month, etc.)"""
        
    def create_feature_vector(
        self,
        ensemble_data: xr.DataArray,
        station_meta: Dict,
        forecast_time: datetime
    ) -> pd.DataFrame:
        """Create complete feature vector for model input"""
```

### 3. Modeling Modules

#### `skewed_gaussian.py`
```python
class SkewedGaussian:
    """Skewed Gaussian distribution with μ, σ, and skewness"""
    
    def __init__(
        self, 
        mu: float, 
        sigma: float, 
        skewness: float
    ):
        """Initialize distribution parameters"""
        
    def pdf(self, x: float) -> float:
        """Probability density function"""
        
    def cdf(self, x: float) -> float:
        """Cumulative distribution function"""
        
    def quantile(self, p: float) -> float:
        """Quantile function (inverse CDF)"""
        
    def crps(self, observation: float) -> float:
        """Continuous Ranked Probability Score"""
        
    @classmethod
    def fit_to_data(
        cls, 
        observations: np.ndarray
    ) -> 'SkewedGaussian':
        """Fit distribution parameters to data"""
```

#### `emos_trainer.py`
```python
class EMOSTrainer:
    """EMOS calibration for skewed Gaussian distributions"""
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        season: str
    ):
        """Initialize trainer for specific season and target"""
        
    def train(
        self,
        features: pd.DataFrame,
        observations: pd.Series
    ) -> Dict[str, Any]:
        """Train EMOS model to minimize CRPS"""
        
    def predict_parameters(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict distribution parameters for new data"""
        
    def calculate_crps(
        self,
        predictions: pd.DataFrame,
        observations: pd.Series
    ) -> float:
        """Calculate CRPS for predictions"""
```

### 4. Prediction Modules

#### `dynamic_corrector.py`
```python
class DynamicCorrector:
    """Apply conditional probability truncation based on current temperature"""
    
    def __init__(
        self,
        current_temperature: float,
        observation_time: datetime
    ):
        """Initialize with current temperature observation"""
        
    def correct_max_temp_probability(
        self,
        base_distribution: SkewedGaussian,
        threshold: float
    ) -> float:
        """Calculate P(X ≥ L | X > T_now) for maximum temperature"""
        
    def correct_min_temp_probability(
        self,
        base_distribution: SkewedGaussian,
        threshold: float
    ) -> float:
        """Calculate P(X ≤ L | X < T_now) for minimum temperature"""
        
    def update_observation(
        self,
        new_temperature: float,
        observation_time: datetime
    ):
        """Update with new temperature observation"""
```

#### `constraint_enforcer.py`
```python
class ConstraintEnforcer:
    """Enforce physical constraints based on historical warming/cooling rates"""
    
    def __init__(
        self,
        station_id: str,
        season: str,
        constraint_data: pd.DataFrame
    ):
        """Initialize with station-specific constraint data"""
        
    def get_max_warming_rate(
        self,
        current_temp: float,
        time_of_day: datetime
    ) -> float:
        """Get maximum possible warming rate for current conditions"""
        
    def get_max_cooling_rate(
        self,
        current_temp: float,
        time_of_day: datetime
    ) -> float:
        """Get maximum possible cooling rate for current conditions"""
        
    def apply_constraints(
        self,
        probability: float,
        current_temp: float,
        target_temp: float,
        time_of_day: datetime,
        is_max_temp: bool
    ) -> float:
        """Apply physical constraints to probability"""
```

### 5. Validation Modules

#### `metrics_calculator.py`
```python
class MetricsCalculator:
    """Calculate validation metrics for probabilistic forecasts"""
    
    def calculate_crps(
        self,
        predictions: List[SkewedGaussian],
        observations: List[float]
    ) -> float:
        """Calculate Continuous Ranked Probability Score"""
        
    def calculate_pit(
        self,
        predictions: List[SkewedGaussian],
        observations: List[float]
    ) -> np.ndarray:
        """Calculate Probability Integral Transform values"""
        
    def create_pit_histogram(
        self,
        pit_values: np.ndarray,
        bins: int = 10
    ) -> Dict[str, Any]:
        """Create PIT histogram and calculate uniformity test"""
        
    def calculate_talagrand(
        self,
        ensemble_predictions: List[np.ndarray],
        observations: List[float]
    ) -> Dict[str, Any]:
        """Calculate Talagrand diagram for ensemble forecasts"""
```

## Configuration Files

### `configs/stations.yaml`
```yaml
stations:
  ZSPD:
    name: "Shanghai Pudong International Airport"
    wunderground_id: "ZSPD"
    country: "cn"
    city: "shanghai"
    latitude: 31.15
    longitude: 121.80
    elevation: 4.0
    timezone: "Asia/Shanghai"
    temperature_unit: "C"
    polymarket_id: "shanghai"
    
  KDEN:
    name: "Denver International Airport"
    wunderground_id: "KDEN"
    country: "us"
    city: "denver"
    latitude: 39.86
    longitude: -104.67
    elevation: 1655.0
    timezone: "America/Denver"
    temperature_unit: "F"
    polymarket_id: "denver"

regions:
  east_asia:
    lat_range: [25.0, 35.0]
    lon_range: [115.0, 125.0]
    stations: ["ZSPD"]
    
  north_america:
    lat_range: [35.0, 45.0]
    lon_range: [-110.0, -100.0]
    stations: ["KDEN"]
```

### `configs/model_params.yaml`
```yaml
training:
  train_start_year: 2000
  train_end_year: 2012
  val_start_year: 2013
  val_end_year: 2017
  test_start_year: 2018
  test_end_year: 2019
  rolling_window_years: 5
  retrain_frequency: "quarterly"
  
seasons:
  DJF: [12, 1, 2]   # Winter
  MAM: [3, 4, 5]    # Spring
  JJA: [6, 7, 8]    # Summer
  SON: [9, 10, 11]  # Fall

emos:
  feature_columns:
    - "ensemble_mean"
    - "ensemble_std"
    - "ensemble_p10"
    - "ensemble_p90"
    - "day_of_year_sin"
    - "day_of_year_cos"
    - "month_sin"
    - "month_cos"
  target_columns:
    - "temp_max"
    - "temp_min"
  hyperparameters:
    learning_rate: 0.01
    max_iterations: 1000
    tolerance: 1e-6
```

## Test Structure

### Unit Tests
```python
# tests/unit/data_acquisition/test_gefs_fetcher.py
def test_download_reforecast_returns_xarray_dataset():
    """GEFS fetcher returns xarray Dataset with expected dimensions"""
    
def test_extract_region_crops_to_specified_bounds():
    """Region extraction returns data within specified lat/lon bounds"""
    
def test_download_handles_network_errors():
    """Fetcher handles network errors gracefully with retry logic"""

# tests/unit/modeling/test_skewed_gaussian.py
def test_skewed_gaussian_pdf_integrates_to_one():
    """PDF integrates to 1 over valid range"""
    
def test_crps_calculation_matches_reference():
    """CRPS calculation matches verified reference implementation"""
    
def test_parameter_estimation_converges():
    """Parameter estimation converges for synthetic data"""

# tests/unit/prediction/test_dynamic_corrector.py
def test_correction_increases_with_current_temp():
    """Probability of exceeding threshold increases with current temperature"""
    
def test_correction_respects_boundaries():
    """Corrected probabilities remain between 0 and 1"""
```

### Integration Tests
```python
# tests/integration/test_data_pipeline.py
def test_end_to_end_data_processing():
    """Raw GEFS data → features pipeline works correctly"""
    
def test_feature_extraction_pipeline():
    """Feature extraction produces expected columns and data types"""

# tests/integration/test_training_pipeline.py
def test_training_pipeline_completes():
    """Training pipeline runs from data to trained model without errors"""
    
def test_model_reproducibility():
    """Training produces identical models given same data and seed"""
```

### End-to-End Tests
```python
# tests/e2e/test_full_pipeline.py
def test_shanghai_summer_prediction():
    """Full pipeline produces predictions for Shanghai summer day"""
    
def test_denver_winter_prediction():
    """Full pipeline produces predictions for Denver winter day"""
    
def test_historical_backtest():
    """Historical backtest produces validation metrics"""
```

## Development Workflow

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### 2. Run Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src --cov-report=html
```

### 3. Development Cycle
```bash
# 1. Write failing test
# 2. Implement minimal code to pass test
# 3. Refactor while keeping tests green
# 4. Commit changes
# 5. Repeat for next behavior
```

### 4. Data Pipeline
```bash
# Download data
python scripts/download_data.py --station ZSPD --start 2000 --end 2019

# Process data
python scripts/process_data.py --station ZSPD --year 2023

# Train model
python scripts/train_model.py --station ZSPD --season JJA --temp-type max

# Run predictions
python scripts/run_predictions.py --station ZSPD --date 2023-07-15

# Validate model
python scripts/validate_model.py --station ZSPD --year 2019
```

## Dependencies

### Core Dependencies
```txt
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
xarray>=0.19.0
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
pyarrow>=6.0.0
requests>=2.26.0
beautifulsoup4>=4.10.0
herbie>=0.0.9
cfgrib>=0.9.9
pytz>=2021.3
python-dateutil>=2.8.2
pyyaml>=6.0
sqlite3>=3.35.0
```

### Development Dependencies
```txt
# requirements-dev.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
hypothesis>=6.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.910
pre-commit>=2.17.0
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

This structure provides a clean separation of concerns, follows Python best practices, and supports the TDD workflow outlined in the specification.