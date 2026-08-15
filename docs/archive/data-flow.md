# Data Flow Specification

## Overview

This document details the flow of data through the Polymarket Temperature Prediction System, including data sources, transformations, and outputs at each stage.

## Data Sources

### 1. GEFS Forecast Data
**Format**: GRIB2 files
**Frequency**: Every 6 hours (00Z, 06Z, 12Z, 18Z)
**Variables**: 2m temperature (t2m), ensemble members (typically 31)
**Spatial Resolution**: 0.5° × 0.5° grid
**Temporal Resolution**: 3-hourly for real-time, 6-hourly for reforecast
**Access**: AWS Open Data via Herbie library

### 2. Wunderground Historical Data
**Format**: HTML/JSON from web pages
**Frequency**: Daily max/min temperatures
**Variables**: Maximum temperature, Minimum temperature
**Time Range**: 2000-2019 for training
**Access**: Web scraping (requests + BeautifulSoup)

### 3. Real-time Temperature Observations
**Format**: METAR or API JSON
**Frequency**: Hourly or more frequent
**Variables**: Current temperature
**Access**: Weather API or METAR parsing

## Data Processing Pipeline

### Stage 1: Raw Data Acquisition
```python
# Pseudo-code for data acquisition
def acquire_data(station, date_range):
    # GEFS data
    gefs_data = herbie.fetch(
        model='gefs',
        product='pgrb2a',
        fxx=range(0, 384, 6),  # 6-hour intervals
        date=date_range
    )
    
    # Wunderground data
    wu_data = scrape_wunderground(
        station_id=station.wu_id,
        start_date=date_range[0],
        end_date=date_range[1]
    )
    
    # Real-time data (if in prediction mode)
    if is_realtime:
        current_temp = get_current_temperature(station)
    
    return {
        'gefs': gefs_data,
        'wunderground': wu_data,
        'current_temp': current_temp if is_realtime else None
    }
```

### Stage 2: Time Alignment and Conversion
**Key Transformations**:
1. **Time Zone Conversion**: All timestamps → station local time
2. **Daily Window Definition**: 00:00-23:59 local time
3. **Forecast-Observation Pairing**:
   - Max temp: 00Z forecast → same day observation
   - Min temp: Previous day 18Z forecast → next day observation

**Code Logic**:
```python
def align_times(gefs_data, wu_data, station):
    # Convert to local time
    gefs_local = convert_to_local(gefs_data, station.timezone)
    wu_local = convert_to_local(wu_data, station.timezone)
    
    # Define daily windows
    daily_windows = create_daily_windows(wu_local['date'])
    
    # Pair forecasts with observations
    pairs = []
    for window in daily_windows:
        # For max temperature: use 00Z forecast
        max_forecast = extract_forecast(
            gefs_local, 
            init_time=window.date().replace(hour=0),
            lead_times=range(0, 24)  # Cover entire day
        )
        
        # For min temperature: use previous day 18Z forecast
        min_forecast = extract_forecast(
            gefs_local,
            init_time=(window.date() - timedelta(days=1)).replace(hour=18),
            lead_times=range(6, 30)  # Cover next day
        )
        
        pairs.append({
            'date': window.date(),
            'max_forecast': max_forecast,
            'min_forecast': min_forecast,
            'max_observed': wu_local[window]['max_temp'],
            'min_observed': wu_local[window]['min_temp']
        })
    
    return pairs
```

### Stage 3: Feature Extraction
**From GEFS Ensemble Forecasts**:
1. **Ensemble Statistics**:
   - Mean, standard deviation, skewness
   - 10th, 25th, 50th, 75th, 90th percentiles
   - Minimum, maximum
2. **Temporal Features**:
   - Diurnal cycle amplitude
   - Rate of change
3. **Spatial Features** (after interpolation):
   - Bilinear interpolated value at station
   - Elevation-corrected temperature

**Feature Engineering**:
```python
def extract_features(ensemble_forecast, station):
    # Spatial interpolation
    interpolated = bilinear_interpolate(
        ensemble_forecast, 
        lat=station.lat, 
        lon=station.lon
    )
    
    # Elevation correction
    model_elev = interpolate_elevation(ensemble_forecast, station)
    elevation_correction = (station.elevation - model_elev) * 0.0065
    corrected_temp = interpolated + elevation_correction
    
    # Ensemble statistics
    features = {
        'mean': np.mean(corrected_temp),
        'std': np.std(corrected_temp),
        'skew': skew(corrected_temp),
        'p10': np.percentile(corrected_temp, 10),
        'p25': np.percentile(corrected_temp, 25),
        'p50': np.percentile(corrected_temp, 50),
        'p75': np.percentile(corrected_temp, 75),
        'p90': np.percentile(corrected_temp, 90),
        'min': np.min(corrected_temp),
        'max': np.max(corrected_temp),
        # Seasonal features
        'day_of_year': window.date().timetuple().tm_yday,
        'month': window.date().month,
        'season': get_season(window.date())
    }
    
    return features
```

### Stage 4: Model Training
**Seasonal Bucketing**:
- DJF (Dec-Jan-Feb): Winter models
- MAM (Mar-Apr-May): Spring models  
- JJA (Jun-Jul-Aug): Summer models
- SON (Sep-Oct-Nov): Fall models

**Training Process**:
```python
def train_seasonal_models(training_data, season):
    # Filter data for season
    seasonal_data = filter_by_season(training_data, season)
    
    # Separate max and min temperature models
    max_features = extract_features(seasonal_data['max_forecast'])
    min_features = extract_features(seasonal_data['min_forecast'])
    
    # Train skewed Gaussian distribution
    max_model = train_skewed_gaussian(
        features=max_features,
        targets=seasonal_data['max_observed']
    )
    
    min_model = train_skewed_gaussian(
        features=min_features,
        targets=seasonal_data['min_observed']
    )
    
    # Validate
    max_metrics = validate_model(max_model, validation_data)
    min_metrics = validate_model(min_model, validation_data)
    
    return {
        'season': season,
        'max_model': max_model,
        'min_model': min_model,
        'max_metrics': max_metrics,
        'min_metrics': min_metrics
    }
```

### Stage 5: Real-time Prediction
**Prediction Workflow**:
1. **Load appropriate model** based on current date and season
2. **Extract features** from latest GEFS forecast
3. **Generate base distribution** (μ, σ, skewness)
4. **Apply dynamic correction** using current temperature
5. **Enforce physical constraints** (max warming/cooling rates)
6. **Convert to market bins** for Polymarket

```python
def generate_prediction(station, target_date, current_temp=None):
    # Determine season
    season = get_season(target_date)
    
    # Load model
    model = load_model(station, season)
    
    # Get latest GEFS forecast
    gefs_forecast = get_latest_gefs(station, target_date)
    
    # Extract features
    features = extract_features(gefs_forecast, station)
    
    # Generate base prediction
    mu, sigma, skew = model.predict(features)
    base_distribution = SkewedGaussian(mu, sigma, skew)
    
    # Apply dynamic correction if current temp available
    if current_temp is not None:
        if target_date == date.today():  # Only for today
            # Check if threshold already exceeded
            if is_max_temp and current_temp >= threshold:
                probability = 1.0
            elif is_min_temp and current_temp <= threshold:
                probability = 1.0
            else:
                # Conditional probability truncation
                probability = dynamic_correction(
                    base_distribution, 
                    current_temp, 
                    threshold
                )
        else:
            probability = base_distribution.cdf(threshold)
    else:
        probability = base_distribution.cdf(threshold)
    
    # Apply physical constraints
    probability = apply_physical_constraints(
        probability,
        current_temp,
        threshold,
        station,
        season,
        target_date
    )
    
    return probability
```

### Stage 6: Market Probability Conversion
**For Polymarket Temperature Bins**:
```python
def convert_to_market_bins(distribution, market_bins):
    """
    Convert continuous distribution to Polymarket bin probabilities
    
    Args:
        distribution: SkewedGaussian distribution
        market_bins: List of bin definitions, e.g.:
            [{'type': '<=', 'value': 18},
             {'type': '=', 'value': 19},
             {'type': '=', 'value': 20},
             {'type': '=', 'value': 21},
             {'type': '=', 'value': 22},
             {'type': '=', 'value': 23},
             {'type': '>=', 'value': 24}]
    
    Returns:
        Dictionary mapping bin to probability
    """
    probabilities = {}
    
    for bin_def in market_bins:
        if bin_def['type'] == '<=':
            prob = distribution.cdf(bin_def['value'])
        elif bin_def['type'] == '=':
            # Assuming 1°C bin width
            lower = bin_def['value'] - 0.5
            upper = bin_def['value'] + 0.5
            prob = distribution.cdf(upper) - distribution.cdf(lower)
        elif bin_def['type'] == '>=':
            prob = 1 - distribution.cdf(bin_def['value'] - 0.01)  # Small epsilon
        elif bin_def['type'] == 'range':
            lower = bin_def['min'] - 0.5
            upper = bin_def['max'] + 0.5
            prob = distribution.cdf(upper) - distribution.cdf(lower)
    
    # Normalize to ensure sum = 1 (handling rounding errors)
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v/total for k, v in probabilities.items()}
    
    return probabilities
```

## Data Quality Checks

### Validation Points
1. **Input Validation**:
   - GEFS data completeness (all ensemble members present)
   - Wunderground data availability (no missing days)
   - Time alignment consistency

2. **Processing Validation**:
   - Feature extraction sanity checks
   - Unit conversion accuracy
   - Spatial interpolation quality

3. **Output Validation**:
   - Probability distribution validity (0 ≤ p ≤ 1)
   - Physical constraint satisfaction
   - Market bin probabilities sum to 1 ± epsilon

### Monitoring Metrics
- **Data freshness**: Time since last GEFS update
- **Data completeness**: Percentage of expected data received
- **Processing latency**: Time from data receipt to prediction
- **Prediction quality**: CRPS, PIT histogram uniformity
- **System health**: Memory usage, CPU load, error rates