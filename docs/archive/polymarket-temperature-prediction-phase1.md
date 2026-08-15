# Specification: Polymarket Temperature Prediction System - Phase 1

## Problem Statement

Polymarket operates binary options markets for temperature predictions in specific cities and dates. Traders need accurate probability predictions for temperature thresholds to make informed betting decisions. The current challenge is building a high-precision physical probability model that can predict both maximum and minimum temperature probability distributions using free public data sources, with validation against historical Wunderground data before moving to live trading.

## Solution

Build a three-layer prediction system:
1. **Static Base Model**: Skewed Gaussian EMOS model trained on GEFS ensemble forecasts and Wunderground historical data, with separate models for maximum and minimum temperatures
2. **Dynamic Correction Layer**: Hourly conditional probability truncation based on current temperature observations
3. **Physical Constraint Layer**: Hard constraints based on historical maximum warming/cooling rates per station and season

The system will output probability distributions for temperature thresholds, which can be converted to Polymarket bin probabilities.

## User Stories

### Data Scientist / Model Developer
1. As a data scientist, I want to train separate EMOS models for maximum and minimum temperatures, so that I can capture the different physical processes governing daily highs and lows
2. As a model developer, I want to train seasonal models (DJF, MAM, JJA, SON) separately, so that I can account for seasonal non-stationarity in temperature patterns
3. As a data engineer, I want automatic data acquisition from GEFS and Wunderground, so that I have consistent training data without manual intervention
4. As a researcher, I want to apply bilinear interpolation and elevation correction to GEFS data, so that model inputs match station locations accurately
5. As a model validator, I want comprehensive validation metrics (CRPS, PIT histograms, Talagrand diagrams), so that I can objectively assess model performance

### System Operator
6. As a system operator, I want the model to automatically retrain quarterly with a 5-year rolling window, so that the model adapts to climate trends without manual intervention
7. As an operator, I want fallback strategies (stale data → simplified model → climatology → null), so that the system remains available even when data sources fail
8. As a maintainer, I want versioned models with DVC, so that I can track model changes and roll back if performance degrades
9. As a monitor, I want alerts when CRPS degrades >20% vs benchmarks, so that I can investigate issues promptly

### Trader / End User
10. As a trader, I want probability predictions updated hourly with dynamic correction, so that I have the most current information for betting decisions
11. As a user, I want predictions for both maximum and minimum temperatures, so that I can bet on all Polymarket temperature markets
12. As a risk-conscious trader, I want physical constraints applied to predictions, so that I know predictions respect historical warming/cooling limits
13. As an API consumer, I want predictions in a format ready for Polymarket integration, so that I can automate betting strategies

### Quality Assurance
14. As a QA engineer, I want the system to validate that training and prediction use identical data resolution (6-hour GEFS data only), so that there's no data leakage or distribution shift
15. As a validator, I want the model to statistically outperform naive benchmarks (GEFS mean, climatology), so that I know the model adds value
16. As an auditor, I want clear separation between Phase 1 (model validation) and Phase 2 (trading), so that no live trading occurs before the model is validated

## Implementation Decisions

### Core Architecture
1. **Three-layer prediction system**: Static base model → Dynamic correction → Physical constraints
2. **Separate models for max/min temperatures**: Different physical processes govern daily highs and lows
3. **Seasonal bucketing**: Four seasonal models (DJF: Dec-Jan-Feb, MAM: Mar-Apr-May, JJA: Jun-Jul-Aug, SON: Sep-Oct-Nov)
4. **Quarterly retraining**: Models retrained every 3 months with 5-year rolling window

### Data Processing
5. **Time alignment protocol**: All times converted to station local time; daily windows defined as 00:00-23:59 local time
6. **Resolution consistency**: Force downsampling of real-time 3-hour GEFS to 6-hour intervals to match reforecast training data
7. **Spatial processing**: Bilinear interpolation (not nearest neighbor) from GEFS grid to station locations
8. **Elevation correction**: Apply standard temperature lapse rate (0.0065 K/m) to correct for elevation differences
9. **Unit standardization**: All temperatures converted to Celsius internally; converted to Fahrenheit for Denver output only

### Model Design
10. **Skewed Gaussian EMOS**: Extended EMOS with skewness parameter to capture asymmetric temperature distributions
11. **Feature engineering**: Use GEFS ensemble statistics (mean, std, percentiles) plus temporal features (day of year, month)
12. **Training objective**: Minimize Continuous Ranked Probability Score (CRPS) separately for max and min temperatures
13. **Validation framework**: PIT histograms as gold standard for calibration; CRPS for accuracy; Talagrand diagrams for spread assessment

### Dynamic Correction
14. **Conditional probability truncation**: Hourly updates using formula P(X ≥ L | X > T_now) for max temps, P(X ≤ L | X < T_now) for min temps
15. **Trigger mechanism**: Update on new temperature observation or GEFS forecast
16. **Fallback logic**: Use previous forecast if delay > 3 hours

### Physical Constraints
17. **Station-specific limits**: Calculate maximum warming/cooling rates from Wunderground historical data per station and season
18. **Constraint application**: If target temperature exceeds physically possible change from current, probability forced to 0 or 1
19. **Constraint hierarchy**: Physical constraints override model predictions

### System Integration
20. **Wunderground as ground truth**: Use Wunderground historical data for training and validation (2000-2019)
21. **GEFS as forecast source**: Use both reforecast (historical) and real-time GEFS data
22. **Real-time observations**: METAR or weather API for current temperatures
23. **Output format**: Probability distributions parameterized by μ, σ, skewness; convertible to Polymarket bin probabilities

### Validation & Deployment
24. **Time wall isolation**: Strict separation between training (2000-2017) and test (2018-2019) data
25. **Performance thresholds**: Must outperform naive benchmarks by statistical significance
26. **Phase gate**: No Phase 2 (trading) until Phase 1 validation passes all metrics
27. **Shadow testing**: New models tested against old before production deployment

## Testing Decisions

### What Makes a Good Test
- Tests verify **external behavior** through public interfaces, not implementation details
- Tests describe **observable behavior** ("system outputs probability distribution given forecast data") not implementation steps
- Tests use **realistic data scenarios** including edge cases (missing data, extreme values)
- Tests are **integration-style** where possible, exercising multiple components together
- Tests verify **physical constraints** are correctly applied (e.g., probabilities respect warming limits)

### Modules to Test
1. **Data acquisition modules**: Test GEFS download, Wunderground scraping, real-time data fetching
2. **Data processing pipeline**: Test time alignment, unit conversion, feature extraction, quality control
3. **Model training**: Test EMOS calibration, seasonal bucketing, hyperparameter tuning
4. **Prediction generation**: Test static prediction, dynamic correction, constraint application
5. **Validation metrics**: Test CRPS calculation, PIT histogram generation, benchmark comparison
6. **System integration**: Test end-to-end pipeline from raw data to predictions

### Test Seams (Public Interfaces)
1. **Data acquisition**: `fetch_gefs_data()`, `fetch_wunderground_data()`, `fetch_current_temperature()`
2. **Feature extraction**: `extract_features(gefs_data, station_meta)` → returns feature DataFrame
3. **Model prediction**: `predict_temperature_distribution(features, model)` → returns (μ, σ, skewness)
4. **Dynamic correction**: `apply_dynamic_correction(base_distribution, current_temp)` → returns corrected distribution
5. **Constraint application**: `apply_physical_constraints(distribution, station, season, current_temp)` → returns constrained distribution
6. **Validation**: `calculate_crps(predictions, observations)`, `calculate_pit(predictions, observations)`

### Testing Strategy
- **Unit tests**: Individual functions with mocked dependencies
- **Integration tests**: Complete data pipeline with sample data
- **Validation tests**: Historical backtests with Wunderground data
- **Property-based tests**: Verify physical constraints hold for all inputs
- **Performance tests**: Ensure hourly updates complete within time budget

### Test Data Strategy
- **Golden datasets**: Small, representative datasets checked into version control
- **Synthetic edge cases**: Generate data for testing boundary conditions
- **Historical validation**: Use held-out 2018-2019 data for final validation
- **Cross-validation**: Use 2000-2017 data with proper time-series splits

### Existing Test Patterns to Follow
- Wunderground integration tests show good pattern for data validation
- Need to extend to GEFS data processing and model components
- Use similar temporary directory patterns for test isolation

## Out of Scope for Phase 1

### Trading & Market Integration
- No market microstructure analysis
- No liquidity considerations
- No slippage modeling
- No money management strategies
- No position sizing algorithms
- No execution logic

### Real-time Production Concerns
- No high-availability deployment
- No load balancing
- No advanced monitoring beyond basic alerts
- No user authentication/authorization
- No payment processing

### Advanced Features
- No multi-model ensembles
- No machine learning beyond EMOS
- No atmospheric physics modeling beyond basic constraints
- No satellite or radar data integration
- No long-range (beyond 10-day) forecasting

### Geographical Expansion
- Only Shanghai (ZSPD) and Denver (KDEN) stations
- No additional cities or stations
- No regional or global modeling

## Further Notes

### Key Technical Challenges
1. **Data alignment**: Ensuring GEFS forecast windows match local daily windows (max: 00Z forecast, min: previous day 18Z forecast)
2. **Distribution fitting**: Skewed Gaussian parameter estimation can be numerically unstable
3. **Real-time performance**: Dynamic correction must complete within minutes of new data arrival
4. **Missing data handling**: Robust fallback strategies for delayed/missing GEFS data

### Success Criteria for Phase 1
1. **Calibration**: PIT histograms uniformly distributed for both max and min temperatures
2. **Accuracy**: CRPS significantly better than GEFS mean and climatology benchmarks
3. **Reliability**: System produces predictions for >95% of requested times
4. **Speed**: Hourly updates complete within 5 minutes of data availability
5. **Documentation**: Clear model documentation and validation reports

### Phase 1 → Phase 2 Transition
Phase 1 must pass all validation metrics before Phase 2 begins. The validation report will include:
- PIT histogram uniformity test results (χ² test)
- CRPS improvement over benchmarks (statistical significance test)
- Physical constraint satisfaction rate
- Missing data handling performance
- System latency measurements