# Polymarket Temperature Prediction System - Phase 1 Complete Specification

## 1. Project Overview

### 1.1 Core Objective
Build a high-precision physical probability model for Polymarket temperature markets that predicts probability distributions for both maximum and minimum temperatures in Shanghai and Denver.

### 1.2 Phase 1 Scope
- **Data Sources**: Wunderground (ground truth), GEFS (forecasts), real-time observations
- **Target Cities**: Shanghai (ZSPD) and Denver (KDEN) only
- **Time Period**: Historical validation (2000-2019)
- **Model Type**: Skewed Gaussian EMOS with seasonal bucketing
- **Output**: Probability distributions for temperature thresholds
- **Excluded**: Market microstructure, liquidity analysis, trading execution

### 1.3 Success Criteria
- PIT histograms uniformly distributed (calibration)
- CRPS significantly better than naive benchmarks (accuracy)
- System produces predictions for >95% of requested times (reliability)
- Hourly updates complete within 5 minutes (performance)
- Clear documentation and validation reports (usability)

## 2. System Architecture

### 2.1 Three-Layer Prediction System
```
┌─────────────────────────────────────────────────────────────┐
│                    Physical Constraint Layer                 │
│  Enforce max warming/cooling rates from historical data     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Dynamic Correction Layer                  │
│  Conditional probability truncation based on current temp    │
│  P(X ≥ L | X > T_now) for max, P(X ≤ L | X < T_now) for min │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Static Base Model Layer                   │
│  Skewed Gaussian EMOS trained on GEFS ensemble forecasts     │
│  Separate models for max/min temps and DJF/MAM/JJA/SON      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow
```
Wunderground (historical) ───┐
                             ├─→ Time Alignment → Unit Conversion → Feature Engineering
GEFS (forecasts) ────────────┘
                             │
Real-time Observations ──────┘
                             │
                     Model Training (EMOS)
                             │
                    Static Predictions (μ, σ, skew)
                             │
                   Dynamic Correction (hourly updates)
                             │
                  Physical Constraints (warming limits)
                             │
                 Bin Probability Conversion (Polymarket)
                             │
                Validation & Monitoring (CRPS, PIT, Talagrand)
```

## 3. Data Requirements

### 3.1 Wunderground Data (Ground Truth)
- **Stations**: ZSPD (Shanghai), KDEN (Denver)
- **Period**: 2000-2019 for training/validation
- **Variables**: Daily maximum/minimum temperatures
- **Format**: HTML/JSON from wunderground.com/history/monthly/
- **Processing**: Web scraping with error handling and rate limiting
- **Unit Conversion**: Fahrenheit to Celsius for Denver, keep Celsius for Shanghai

### 3.2 GEFS Data (Forecast Features)
- **Source**: AWS Open Data via Herbie library
- **Type**: Both reforecast (historical) and real-time
- **Variables**: 2m temperature (t2m) ensemble members
- **Resolution**: 0.5° × 0.5° grid, 6-hour intervals
- **Region Selection**: Download only target regions, not global data
  - Shanghai region: 25°-35°N, 115°-125°E
  - Denver region: 35°-45°N, 100°-110°W
- **Processing**: Bilinear interpolation + elevation correction

### 3.3 Real-time Observations
- **Source**: METAR or weather API
- **Frequency**: Hourly updates
- **Purpose**: Dynamic probability correction
- **Fallback**: Use previous observation if current unavailable

## 4. Model Design

### 4.1 Skewed Gaussian Distribution
- **Parameters**: μ (mean), σ (standard deviation), skewness
- **Functions**: PDF, CDF, quantile function
- **Estimation**: Maximum likelihood or method of moments
- **CRPS**: Closed-form expression for skewed Gaussian

### 4.2 EMOS Calibration
- **Input Features**: GEFS ensemble mean, std, percentiles + temporal features
- **Training Objective**: Minimize CRPS for each season and temperature type
- **Seasonal Buckets**: DJF (Dec-Jan-Feb), MAM (Mar-Apr-May), JJA (Jun-Jul-Aug), SON (Sep-Oct-Nov)
- **Separate Models**: Maximum temperature and minimum temperature
- **Training Period**: 5-year rolling window, quarterly retraining

### 4.3 Dynamic Correction
- **Trigger**: New temperature observation or GEFS forecast
- **Formula for Max Temp**: P(X ≥ L | X > T_now) = (1 - F(L)) / (1 - F(T_now))
- **Formula for Min Temp**: P(X ≤ L | X < T_now) = F(L) / F(T_now)
- **Implementation**: Hourly updates, millisecond computation time

### 4.4 Physical Constraints
- **Data Source**: Calculate from Wunderground historical data
- **Metrics**: Maximum warming/cooling rates per station and season
- **Application**: If target temperature exceeds physically possible change, probability forced to 0 or 1
- **Hierarchy**: Constraints override model predictions

## 5. Implementation Tasks

### 5.1 Week 1-2: Data Infrastructure
1. **Wunderground Scraper**: Robust web scraping with error handling
2. **GEFS Fetcher**: Regional data download with Herbie
3. **Data Processing**: Time alignment, unit conversion, feature extraction
4. **Data Storage**: Parquet files, SQLite database, directory structure

### 5.2 Week 3-4: Model Implementation
1. **Skewed Gaussian**: Distribution class with CRPS calculation
2. **EMOS Training**: CRPS minimization with seasonal bucketing
3. **Training Pipeline**: 5-year rolling window, quarterly retraining
4. **Model Versioning**: DVC for reproducibility

### 5.3 Week 5-6: Prediction System
1. **Static Predictor**: Load models, generate base predictions
2. **Dynamic Corrector**: Hourly conditional probability updates
3. **Constraint Enforcer**: Apply physical warming/cooling limits
4. **Bin Converter**: Convert distributions to Polymarket bin probabilities

### 5.4 Week 7-8: Validation System
1. **Metrics Calculator**: CRPS, PIT histograms, Talagrand diagrams
2. **Historical Backtest**: Time-series validation on 2018-2019 data
3. **Benchmark Comparison**: GEFS mean, climatology, persistence
4. **Alert System**: CRPS degradation >20%, PIT non-uniformity

### 5.5 Week 9: System Integration
1. **Main Pipeline**: End-to-end orchestration
2. **Configuration System**: YAML configs with validation
3. **Logging & Monitoring**: Structured logs, performance metrics
4. **Documentation**: User guide, API reference, troubleshooting

## 6. Technical Decisions

### 6.1 Data Processing
- **Time Alignment**: All times to station local time, daily windows 00:00-23:59
- **Unit Standardization**: Celsius internally, Fahrenheit output for Denver only
- **Spatial Interpolation**: Bilinear (not nearest neighbor)
- **Elevation Correction**: Standard lapse rate 0.0065 K/m
- **Resolution Consistency**: Downsample real-time 3-hour GEFS to 6-hour to match training

### 6.2 Model Training
- **Training Split**: 2000-2012 train, 2013-2017 validation, 2018-2019 test
- **Feature Engineering**: Ensemble statistics + temporal features
- **Hyperparameter Tuning**: Grid search with time-series cross-validation
- **Model Persistence**: Pickle/joblib with metadata

### 6.3 System Design
- **Modular Architecture**: Clear separation between data, model, prediction layers
- **Error Handling**: Graceful degradation with fallback strategies
- **Performance**: Hourly updates within 5 minutes
- **Monitoring**: Comprehensive metrics and alerts

## 7. Testing Strategy

### 7.1 Test Pyramid
- **70% Unit Tests**: Individual functions with mocked dependencies
- **20% Integration Tests**: Component interactions with real dependencies
- **10% End-to-End Tests**: Full pipeline with sample data

### 7.2 Key Test Areas
1. **Data Acquisition**: Network error handling, data parsing, rate limiting
2. **Data Processing**: Time alignment, unit conversion, feature extraction
3. **Model Training**: CRPS minimization, parameter estimation, seasonal bucketing
4. **Prediction**: Dynamic correction, physical constraints, bin conversion
5. **Validation**: Metric calculations, benchmark comparisons

### 7.3 Test Data Strategy
- **Golden Datasets**: Small representative datasets in version control
- **Synthetic Edge Cases**: Generated data for boundary conditions
- **Historical Validation**: Hold-out 2018-2019 data for final validation
- **Property Tests**: Verify mathematical invariants hold for all inputs

## 8. Risk Mitigation

### 8.1 Technical Risks
- **GEFS Data Availability**: Implement caching and fallback to slightly stale data
- **Model Convergence**: Add regularization and fallback to simpler model
- **Performance Bottlenecks**: Profile and optimize critical paths
- **Numerical Stability**: Use double precision and stable algorithms

### 8.2 Project Risks
- **Scope Creep**: Strict Phase 1 boundaries (no trading logic)
- **Timeline Slippage**: Weekly progress reviews and adjustment
- **Data Quality Issues**: Robust validation and quality flags
- **Model Validation Failures**: Multiple fallback strategies

### 8.3 Mitigation Strategies
- **Weekly Demos**: Working components demonstrated every Friday
- **Incremental Delivery**: Usable components delivered weekly
- **Early Validation**: Validate against historical data early
- **Continuous Integration**: Automated testing catches issues immediately

## 9. Deliverables

### 9.1 Code Deliverables
- Complete Python codebase with all modules
- Configuration files for Shanghai and Denver
- Test suite with >80% coverage
- Documentation (README, API reference, examples)

### 9.2 Model Deliverables
- Trained models for all seasons and temperature types
- Model performance reports (CRPS, PIT, Talagrand)
- Benchmark comparison results
- Validation reports on 2018-2019 data

### 9.3 Operational Deliverables
- End-to-end pipeline script
- Monitoring dashboard setup
- Alert configuration
- Deployment instructions

## 10. Acceptance Criteria

### 10.1 Data Pipeline
- [ ] Wunderground data for 2000-2019 successfully scraped and validated
- [ ] GEFS data for target regions downloaded and processed
- [ ] All temperatures converted to Celsius internally
- [ ] Time alignment correct for Shanghai and Denver timezones
- [ ] Feature extraction produces expected statistics

### 10.2 Model Training
- [ ] Skewed Gaussian implementation mathematically correct
- [ ] EMOS training reduces CRPS compared to initial parameters
- [ ] Separate models for max/min temperatures and seasons
- [ ] Models can be saved, loaded, and reproduced
- [ ] Quarterly retraining pipeline works automatically

### 10.3 Prediction System
- [ ] Static predictions generated from GEFS features
- [ ] Dynamic correction updates probabilities correctly
- [ ] Physical constraints applied appropriately
- [ ] Bin probabilities calculated correctly and sum to 1
- [ ] Hourly updates complete within 5 minutes

### 10.4 Validation
- [ ] CRPS calculation matches reference implementation
- [ ] PIT histograms show uniform distribution for calibrated model
- [ ] Model outperforms GEFS mean and climatology benchmarks
- [ ] Historical backtest shows consistent performance
- [ ] Alerting triggers on performance degradation

### 10.5 System Integration
- [ ] End-to-end pipeline runs without errors
- [ ] Configuration system flexible and validated
- [ ] Logging provides sufficient debugging information
- [ ] Error handling is graceful and informative
- [ ] Documentation enables new users to run the system

## 11. Next Steps

### Immediate (Week 1)
1. Set up project structure and virtual environment
2. Implement Wunderground scraper with tests
3. Implement GEFS data fetcher with regional download
4. Create data processing utilities (time alignment, unit conversion)

### Short-term (Week 2-4)
1. Implement skewed Gaussian distribution with CRPS
2. Build EMOS training pipeline
3. Create feature extraction from GEFS ensembles
4. Set up model training and validation framework

### Medium-term (Week 5-8)
1. Implement dynamic correction and physical constraints
2. Build prediction pipeline with bin conversion
3. Create comprehensive validation system
4. Develop monitoring and alerting

### Long-term (Week 9)
1. Integrate all components into main pipeline
2. Create configuration system and documentation
3. Perform final validation and benchmarking
4. Prepare for Phase 2 (trading system integration)