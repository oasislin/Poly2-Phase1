# Testing Strategy: Phase 1 Implementation

## Testing Philosophy

Follow Test-Driven Development (TDD) principles:
1. **RED**: Write failing test for one observable behavior
2. **GREEN**: Write minimal code to pass the test
3. **REFACTOR**: Improve code while keeping tests green
4. **Repeat**: One behavior at a time, vertically through the stack

Tests verify **public behavior** not implementation details. Good tests survive refactoring.

## Test Architecture

### Test Pyramid
```
        E2E Tests (10%)
           │
    Integration Tests (20%)
           │
      Unit Tests (70%)
```

### Test Types
1. **Unit Tests**: Isolated function tests with mocked dependencies
2. **Integration Tests**: Component interaction tests with real dependencies
3. **End-to-End Tests**: Full pipeline tests with sample data
4. **Property Tests**: Verify invariants hold for all inputs
5. **Performance Tests**: Ensure time and memory constraints met

## Test Seams (Public Interfaces)

### Layer 1: Data Acquisition
```python
# Test seam: Data fetcher interface
def test_fetch_gefs_data_returns_expected_format():
    """Given valid station and date range, returns xarray Dataset"""
    pass

def test_fetch_wunderground_data_handles_missing_dates():
    """When Wunderground has gaps, returns partial data with quality flags"""
    pass

def test_fetch_current_temperature_falls_back_on_error():
    """When primary source fails, uses secondary source with degradation flag"""
    pass
```

### Layer 2: Data Processing
```python
# Test seam: Feature extraction interface
def test_extract_features_applies_elevation_correction():
    """Temperature corrected by (station_elev - model_elev) * 0.0065"""
    pass

def test_time_alignment_converts_to_local_time():
    """UTC times converted to station local time for daily windows"""
    pass

def test_resolution_downsampling_matches_training():
    """3-hour real-time data downsampled to 6-hour to match reforecast"""
    pass
```

### Layer 3: Model Training
```python
# Test seam: Model training interface
def test_emos_calibration_minimizes_crps():
    """Trained model has lower CRPS than initial parameters"""
    pass

def test_seasonal_bucketing_creates_four_models():
    """Separate models for DJF, MAM, JJA, SON seasons"""
    pass

def test_max_min_models_trained_separately():
    """Max temperature model parameters differ from min temperature"""
    pass
```

### Layer 4: Prediction Generation
```python
# Test seam: Prediction interface
def test_dynamic_correction_updates_probability():
    """Given current temperature, posterior probability ≠ prior probability"""
    pass

def test_physical_constraints_override_impossible_predictions():
    """When target exceeds max warming rate, probability forced to 0 or 1"""
    pass

def test_bin_probability_calculation_matches_spec():
    """Continuous distribution correctly converted to Polymarket bin probabilities"""
    pass
```

### Layer 5: Validation
```python
# Test seam: Validation interface
def test_crps_calculation_matches_reference_implementation():
    """CRPS calculation matches verified reference implementation"""
    pass

def test_pit_histogram_shows_uniform_distribution_for_calibrated_model():
    """Well-calibrated model produces uniform PIT histogram"""
    pass

def test_benchmark_comparison_detects_improvement():
    """Model CRPS significantly better than GEFS mean benchmark"""
    pass
```

## Test Data Strategy

### Golden Datasets
Small, representative datasets checked into version control:

1. **gefs_sample.grib2**: 1-day GEFS forecast for Shanghai
2. **wunderground_sample.csv**: 1-month Wunderground data for Shanghai
3. **station_metadata.yaml**: Station coordinates and elevations
4. **expected_features.parquet**: Pre-computed features for verification

### Synthetic Test Data
Generate edge cases programmatically:

```python
def generate_edge_case_data():
    """Create test data for boundary conditions"""
    return {
        'extreme_temperatures': [-50, 50],  # Physical limits
        'missing_ensemble_members': [0, 31],  # All or none
        'invalid_dates': ['2023-02-30', '2023-13-01'],  # Invalid
        'timezone_boundaries': ['2023-03-12 01:59', '2023-03-12 03:01'],  # DST
    }
```

### Historical Validation Data
Hold out 2018-2019 for final validation:
- Never used in training
- Used only for final model evaluation
- Simulates real deployment performance

## Test Implementation Patterns

### Fixture Pattern
```python
@pytest.fixture
def sample_gefs_data():
    """Provide sample GEFS data for multiple tests"""
    return xr.open_dataset('tests/data/gefs_sample.grib2')

@pytest.fixture  
def trained_max_temp_model():
    """Provide pre-trained model for prediction tests"""
    return load_model('tests/models/max_temp_djf.pkl')
```

### Parameterized Testing
```python
@pytest.mark.parametrize('station,season,expected_range', [
    ('ZSPD', 'JJA', (25, 35)),  # Shanghai summer
    ('ZSPD', 'DJF', (0, 10)),   # Shanghai winter
    ('KDEN', 'JJA', (20, 30)),  # Denver summer
    ('KDEN', 'DJF', (-10, 5)),  # Denver winter
])
def test_temperature_ranges_by_station_season(station, season, expected_range):
    """Temperature predictions respect station/season climatology"""
    pass
```

### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    current_temp=st.floats(-50, 50),
    target_temp=st.floats(-50, 50),
    max_warming_rate=st.floats(0, 20)
)
def test_physical_constraints_monotonic(current_temp, target_temp, max_warming_rate):
    """Probability of reaching target increases with current temperature"""
    if target_temp > current_temp + max_warming_rate:
        assert probability == 0
    elif target_temp < current_temp - max_warming_rate:
        assert probability == 1
    else:
        assert 0 <= probability <= 1
```

## Test Organization

### Directory Structure
```
tests/
├── unit/
│   ├── data_acquisition/
│   │   ├── test_gefs_fetcher.py
│   │   ├── test_wunderground_scraper.py
│   │   └── test_realtime_temp.py
│   ├── data_processing/
│   │   ├── test_time_aligner.py
│   │   ├── test_feature_extractor.py
│   │   └── test_quality_control.py
│   ├── modeling/
│   │   ├── test_emos.py
│   │   ├── test_seasonal_trainer.py
│   │   └── test_hyperparameter_tuner.py
│   └── prediction/
│       ├── test_dynamic_corrector.py
│       ├── test_constraint_enforcer.py
│       └── test_bin_converter.py
├── integration/
│   ├── test_data_pipeline.py
│   ├── test_training_pipeline.py
│   └── test_prediction_pipeline.py
├── e2e/
│   ├── test_full_pipeline.py
│   └── test_historical_validation.py
├── property/
│   ├── test_physical_invariants.py
│   └── test_mathematical_properties.py
└── conftest.py  # Shared fixtures
```

### Test Naming Convention
- **Unit tests**: `test_<function>_<scenario>_<expected>`
- **Integration tests**: `test_<component>_<component>_integration`
- **E2E tests**: `test_<input>_to_<output>_pipeline`

Examples:
- `test_fetch_gefs_data_returns_xarray_dataset`
- `test_time_aligner_converts_utc_to_local`
- `test_emos_training_minimizes_crps`
- `test_full_pipeline_zspd_summer_day`

## Continuous Integration

### Test Execution Order
1. **Fast unit tests** (< 100ms each) run first
2. **Integration tests** (< 1s each) run next  
3. **Slow E2E tests** (< 10s each) run last
4. **Property tests** can run in parallel

### CI Pipeline
```yaml
# GitHub Actions example
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --tb=short
      
      - name: Run integration tests
        run: pytest tests/integration/ -v --tb=short
      
      - name: Run E2E tests
        run: pytest tests/e2e/ -v --tb=short --timeout=30
      
      - name: Run property tests
        run: pytest tests/property/ -v --tb=short
```

## Performance Requirements

### Time Constraints
- **Hourly update**: Complete within 5 minutes of data availability
- **Model training**: Complete within 2 hours for 5-year dataset
- **Data download**: Complete within 30 minutes for 1-month GEFS data
- **Feature extraction**: Complete within 1 minute per station per day

### Memory Constraints
- **Training**: < 8GB RAM for 5-year dataset
- **Prediction**: < 1GB RAM for real-time operation
- **Data storage**: < 100GB for 20-year historical data

### Test Coverage Goals
- **Line coverage**: > 80%
- **Branch coverage**: > 70%
- **Critical path coverage**: 100%
- **Error handling coverage**: All documented exceptions tested

## Monitoring Test Health

### Test Metrics
- **Flakiness rate**: < 1% (no intermittent failures)
- **Execution time**: < 5 minutes total
- **Failure investigation**: Clear error messages with reproduction steps
- **Coverage trends**: Track coverage changes over time

### Test Maintenance
- **Weekly review**: Remove obsolete tests
- **Monthly audit**: Ensure tests match current behavior
- **Quarterly refresh**: Update golden datasets
- **Bi-annual review**: Update performance benchmarks

## Example Test Implementation

### Unit Test Example
```python
def test_apply_elevation_correction():
    """Temperature corrected by elevation difference * lapse rate"""
    # Arrange
    model_temp = 20.0  # °C at model elevation
    model_elev = 100.0  # meters
    station_elev = 4.0   # meters (Shanghai Pudong)
    lapse_rate = 0.0065  # K/m
    
    # Act
    corrected = apply_elevation_correction(
        model_temp, model_elev, station_elev, lapse_rate
    )
    
    # Assert
    elevation_diff = station_elev - model_elev
    expected = model_temp + elevation_diff * lapse_rate
    assert corrected == pytest.approx(expected, rel=1e-6)
```

### Integration Test Example
```python
def test_feature_extraction_pipeline():
    """Full feature extraction from raw GEFS to training features"""
    # Arrange
    gefs_data = load_sample_gefs()
    station = {'lat': 31.15, 'lon': 121.80, 'elevation': 4.0}
    
    # Act
    features = extract_features(gefs_data, station)
    
    # Assert
    assert 't2m_mean' in features.columns
    assert 't2m_std' in features.columns
    assert 't2m_p10' in features.columns
    assert 't2m_p90' in features.columns
    assert 'elevation_correction' in features.columns
    assert features.shape[0] > 0
```

### E2E Test Example
```python
def test_shanghai_summer_day_prediction():
    """End-to-end test for Shanghai summer day prediction"""
    # Arrange
    date = '2023-07-15'
    station = 'ZSPD'
    
    # Act
    predictions = run_pipeline(date, station)
    
    # Assert
    assert 'max_temp_distribution' in predictions
    assert 'min_temp_distribution' in predictions
    assert 'max_bin_probabilities' in predictions
    assert 'min_bin_probabilities' in predictions
    
    # Physical constraints
    max_probs = predictions['max_bin_probabilities']
    min_probs = predictions['min_bin_probabilities']
    assert sum(max_probs.values()) == pytest.approx(1.0, rel=1e-6)
    assert sum(min_probs.values()) == pytest.approx(1.0, rel=1e-6)
    assert all(0 <= p <= 1 for p in max_probs.values())
    assert all(0 <= p <= 1 for p in min_probs.values())
```