# Project Structure and Module Design

> **对齐 v5.9 执行规格（2026-08-15）**：模型矩阵为"季节 $\times$ 时效"（20 模型/站，高斯 EMOS + 气候学方差 Floor），特征为 6h 窗口 TMAX/TMIN 日极值 + 5 成员集合统计（均值/方差/成员极值）；缺失时效节点用参数 (a,b,c,d) 线性内插。

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
│   │   ├── climatology.py
│   │   ├── gaussian_emos.py
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
        """集合统计 = {mean, variance, member_max, member_min}（5 成员，弃分位数与时间特征）"""
        
    def create_feature_vector(
        self,
        ensemble_data: xr.DataArray,
        station_meta: Dict,
        forecast_time: datetime
    ) -> pd.DataFrame:
        """Create complete feature vector for model input"""
```

### 3. Modeling Modules

#### `climatology.py`
```python
class Climatology:
    """气候学基线：σ_clim(d)/μ_clim(d)（31 天滑动窗 × 2000-2018 实测，逐日平滑）"""
    
    def compute(self, observations, station, target_type) -> pd.Series:
        """按日历日输出平滑的 μ_clim(d)、σ_clim(d) 曲线（严格 OOS：不碰 2019）"""
    
    def sigma_clim(self, day_of_year: int) -> float:
        """查表返回目标日的 σ_clim(d)"""
```

#### `gaussian_emos.py`
```python
class GaussianEMOS:
    """高斯 EMOS 分布 N(μ, σ²)，σ² = c² + d²·S²_ens + σ²_clim(d)（无 skewness）"""
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        
    def cdf(self, x: float) -> float:
        """Cumulative distribution function"""
        
    def quantile(self, p: float) -> float:
        """Quantile function (inverse CDF)"""
        
    def crps(self, observation: float) -> float:
        """高斯 CRPS 闭式解（Gneiting 公式）"""
```

#### `emos_trainer.py`（v5.9 对齐）
```python
class EMOSTrainer:
    """EMOS calibration for Gaussian with variance floor（季节 × 时效矩阵，20 模型/站）"""
    
    def __init__(
        self,
        feature_columns: List[str],   # [ens_mean, ens_var, member_max, member_min]
        target_column: str,
        season: str,                  # DJF / MAM / JJA / SON
        lead_time_bucket: int,        # {54,30,6}（最高温）；{48,24}（最低温）
        target_type: str              # 'max' | 'min'
    ):
        """Initialize trainer for specific season × lead-time bucket and target"""
        
    def train(
        self,
        features: pd.DataFrame,
        observations: pd.Series
    ) -> Dict[str, Any]:
        """Train EMOS（高斯 CRPS 闭式解 + L-BFGS-B + L2(d) + 热启动）；两级降级：Level 1 高斯 EMOS+Floor → Level 2 气候学"""
        
    def predict_parameters(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict distribution parameters (μ, σ) for new data"""
        
    def calculate_crps(
        self,
        predictions: pd.DataFrame,
        observations: pd.Series
    ) -> float:
        """Calculate CRPS for predictions"""
```
> 模型矩阵规模：4 季节 × (3 + 2) 时效节点 = 20 模型/站点，2 站共 40 个。命名 `{Station}_{Season}_{Max|Min}_lead{H}h.pkl`。缺失时效节点用参数 (a,b,c,d) 线性内插。

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
        base_distribution: GaussianEMOS,
        threshold: float
    ) -> float:
        """Calculate P(X ≥ L | X > T_now) for maximum temperature"""
        
    def correct_min_temp_probability(
        self,
        base_distribution: GaussianEMOS,
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
        predictions: List[GaussianEMOS],
        observations: List[float]
    ) -> float:
        """Calculate Continuous Ranked Probability Score"""
        
    def calculate_pit(
        self,
        predictions: List[GaussianEMOS],
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

### `configs/model_params.yaml`（v5.9 对齐）
```yaml
training:
  train_start_year: 2000
  train_end_year: 2018
  val_start_year: 2019        # 单次留出主验收
  val_end_year: 2019
  rolling_origin: true        # 训练期滚动验证（Rolling-Origin）
  time_wall_isolation: true   # 训练严禁访问验证集
  
seasons:
  DJF: [12, 1, 2]   # Winter
  MAM: [3, 4, 5]    # Spring
  JJA: [6, 7, 8]    # Summer
  SON: [9, 10, 11]  # Fall

lead_time_nodes:              # 真实训练节点（00Z 起报可达）
  max: [54, 30, 6]            # 名义目标 15:00 LT
  min: [48, 24]               # 名义目标 06:00 LT
  interpolate:                # 缺失节点用 (a,b,c,d) 线性内插
    max: [12, 18, 24, 36, 42, 48]
    min: [30, 36, 42]

climatology:                  # σ_clim(d)/μ_clim(d) 计算
  window_days: 31             # 前后各 15 天
  years: [2000, 2018]         # 严格 OOS，不碰 2019

emos:
  feature_columns:
    - "ensemble_mean"
    - "ensemble_variance"
    - "member_max"
    - "member_min"
  target_columns:
    - "temp_max"
    - "temp_min"
  members: ["c00", "p01", "p02", "p03", "p04"]   # 5 成员集合对齐
  variance_floor: "sigma_clim_squared"           # σ²_clim(d)，不参与优化
  degradation:                # 两级降级（硬+软触发）
    level1: "gaussian_emos_with_floor"
    level2: "climatology"
  hyperparameters:
    max_iterations: 1000
    tolerance: 1e-6
    l2_lambda_d: 1e-3         # 仅对 d 的 L2 正则
    init: [0, 1, 0, 1]        # 热启动 (a,b,c,d) + O(1e-3) 扰动
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

# tests/unit/modeling/test_gaussian_emos.py
def test_gaussian_pdf_integrates_to_one():
    """PDF integrates to 1 over valid range"""
    
def test_crps_calculation_matches_reference():
    """CRPS calculation matches verified reference implementation"""
    
def test_variance_floor_never_below_climatology():
    """σ² 天然 ≥ σ²_clim(d)（Floor 生效）"""

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