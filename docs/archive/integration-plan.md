# WunderGround Scraper Integration Plan

> ⚠️ **已废弃（早期 v2.x 设计文档，保留仅作历史参考）**：其中 Task 1.1（WunderGround 抓取器）部分为已完成冻结历史；模型层规划（偏态高斯 EMOS、分位数/时间特征等）与现行 v5.9 执行规格不一致。现行权威规格见 `specs/phase1-specification-complete.md` 与 `specs/implementation-tasks-phase1.md`。

## Overview

This document outlines how to integrate the existing WunderGround data scraping system into the Polymarket Temperature Prediction System. The existing system provides a robust foundation for historical temperature data acquisition that we can leverage for model training.

## Current WunderGround Scraper Capabilities

### ✅ **Already Implemented**

1. **Data Acquisition**
   - Web scraping of Weather Underground historical pages
   - Support for multiple stations (ZSPD, ZBAA, ZGGG, etc.)
   - Monthly data retrieval (2000-2026 range)
   - Async HTTP requests with rate limiting

2. **Data Extraction**
   - Daily observations table parsing
   - Chart data (JSON) extraction
   - Temperature parsing (high/low in °C)
   - Additional meteorological fields:
     - Dew point temperature
     - Humidity
     - Wind speed and gusts
     - Pressure
     - Precipitation
     - Weather conditions

3. **Data Processing**
   - HTML to structured data conversion
   - Unit parsing and normalization
   - Error handling and retry logic
   - Data validation

4. **Storage & Output**
   - JSON format with full metadata
   - CSV format for tabular data
   - Organized by station/city/year/month
   - Logging and error tracking

### 📋 **Current Data Structure**

```json
{
  "url": "https://www.wunderground.com/history/monthly/cn/shanghai/ZSPD/date/2026-7",
  "country": "cn",
  "city": "shanghai",
  "station": "ZSPD",
  "year": 2026,
  "month": 7,
  "daily_observations": [
    {
      "date": "7/31/2026",
      "temp_high": 37,
      "temp_low": 28,
      "dew_pt_high": 27,
      "dew_pt_low": 22,
      "humidity": 71,
      "wind": "",
      "max_wind": 22,
      "wind_gust": null,
      "pressure": 1007.58,
      "precipitation": 0.0,
      "condition": ""
    }
  ],
  "chart_data": [
    {
      "y": 2026,
      "m": 6,
      "d": 31,
      "high": 37,
      "low": 28
    }
  ]
}
```

## Integration Requirements for Temperature Prediction System

### 1. **Data Requirements for Model Training**

**Required Fields:**
- `date`: Date string (need to parse to datetime)
- `temp_high`: Maximum temperature (°C)
- `temp_low`: Minimum temperature (°C)
- `station`: Station ID (e.g., ZSPD)
- `city`: City name
- `country`: Country code

**Additional Useful Fields:**
- `dew_pt_high/low`: For humidity correction
- `pressure`: For atmospheric condition features
- `wind`: For weather pattern features

### 2. **Time Range Requirements**

**Phase 1 Training Data:**
- **Time Period**: 2000-2019 (as specified in domain model)
- **Cities**: Shanghai (ZSPD) and Denver (need to find Denver station code)
- **Frequency**: Daily maximum/minimum temperatures

**Current Scraper Status:**
- ✅ Shanghai (ZSPD) supported
- ❌ Denver station needs to be identified and added
- ✅ Time range configurable (2000-2019 achievable)

### 3. **Data Quality Requirements**

**Validation Needed:**
- Missing data handling (gaps in historical records)
- Data consistency across years
- Unit consistency (all temperatures in °C)
- Date format standardization

**Current Scraper Features:**
- ✅ Error handling and retry logic
- ✅ Data validation during parsing
- ✅ Logging for troubleshooting
- ❌ Automated data quality checks needed

## Integration Architecture

### **Module Integration Plan**

```
Existing WunderGround Scraper
    ↓
[Adapter Layer] - Convert to unified data format
    ↓
[Data Validation Layer] - Check quality, fill gaps
    ↓
[Feature Engineering Pipeline] - Extract training features
    ↓
Model Training System
```

### **Adapter Layer Design**

```python
class WunderGroundDataAdapter:
    """Adapts existing WunderGround scraper data to model training format"""
    
    def __init__(self, scraper_config_path="WunderGround Data Extraction/config.yaml"):
        self.scraper = WundergroundScraper()
        self.config = self.load_config(scraper_config_path)
    
    def fetch_training_data(self, stations, start_year, end_year):
        """Fetch historical data for model training"""
        training_data = []
        
        for station in stations:
            # Use existing scraper to get data
            monthly_data = self.scraper.get_multiple_months(
                country=station.country,
                city=station.city,
                station=station.code,
                start_year=start_year,
                start_month=1,
                end_year=end_year,
                end_month=12
            )
            
            # Convert to unified format
            unified_data = self._convert_to_unified_format(monthly_data)
            training_data.extend(unified_data)
        
        return training_data
    
    def _convert_to_unified_format(self, monthly_data):
        """Convert scraper format to model training format"""
        unified_records = []
        
        for month_data in monthly_data:
            for daily in month_data['daily_observations']:
                record = {
                    'station_id': month_data['station'],
                    'city': month_data['city'],
                    'country': month_data['country'],
                    'date': self._parse_date(daily['date'], month_data['year'], month_data['month']),
                    'temp_max': daily['temp_high'],
                    'temp_min': daily['temp_low'],
                    'dew_point_max': daily['dew_pt_high'],
                    'dew_point_min': daily['dew_pt_low'],
                    'humidity': daily['humidity'],
                    'wind_max': daily['max_wind'],
                    'pressure': daily['pressure'],
                    'precipitation': daily['precipitation'],
                    'source': 'wunderground',
                    'data_quality': self._assess_quality(daily)
                }
                unified_records.append(record)
        
        return unified_records
    
    def _parse_date(self, date_str, year, month):
        """Parse date string to datetime object"""
        # Handle formats like "7/31/2026" or "31"
        # Implementation depends on actual date format in data
        pass
    
    def _assess_quality(self, daily_record):
        """Assess data quality for each record"""
        quality_score = 1.0  # Default good quality
        
        # Check for missing critical fields
        if daily_record['temp_high'] is None or daily_record['temp_low'] is None:
            quality_score *= 0.5
        
        # Check for physically impossible values
        if daily_record['temp_high'] < -50 or daily_record['temp_high'] > 60:
            quality_score *= 0.3
        
        return quality_score
```

### **Data Validation Layer**

```python
class DataValidator:
    """Validates and cleans WunderGround data for model training"""
    
    def validate_temperature_data(self, data):
        """Validate temperature data quality"""
        validated_data = []
        
        for record in data:
            # Check for missing values
            if record['temp_max'] is None or record['temp_min'] is None:
                continue  # Skip records with missing temperatures
            
            # Check physical consistency
            if record['temp_max'] < record['temp_min']:
                # Swap if max < min (data error)
                record['temp_max'], record['temp_min'] = record['temp_min'], record['temp_max']
            
            # Check for extreme outliers
            if self._is_extreme_outlier(record['temp_max'], record['temp_min']):
                continue  # Skip extreme outliers
            
            # Check date validity
            if not self._is_valid_date(record['date']):
                continue
            
            validated_data.append(record)
        
        return validated_data
    
    def fill_missing_dates(self, data, station_id):
        """Fill missing dates with interpolation or NaN"""
        # Create complete date range
        dates = sorted(set([r['date'] for r in data]))
        full_range = pd.date_range(start=min(dates), end=max(dates), freq='D')
        
        # Fill missing dates
        filled_data = []
        # Implementation depends on pandas/numpy availability
        pass
    
    def _is_extreme_outlier(self, temp_max, temp_min):
        """Check if temperatures are physically impossible"""
        # Station-specific temperature ranges
        station_ranges = {
            'ZSPD': {'min': -20, 'max': 50},  # Shanghai
            'DENVER_STATION': {'min': -40, 'max': 45}  # Denver (to be determined)
        }
        # Implementation
        pass
```

## Implementation Steps

### **Phase 1: Data Pipeline Enhancement (Week 1-2)**

#### Step 1.1: Extend Existing Scraper
```python
# Add Denver station support
# config.yaml additions:
locations:
  - name: "Denver International"
    country: "us"
    city: "denver"
    station: "KDEN"  # Denver International Airport
    description: "丹佛国际机场"
```

#### Step 1.2: Create Data Adapter
- Implement `WunderGroundDataAdapter` class
- Add date parsing and validation
- Create unified data format

#### Step 1.3: Implement Data Validation
- Add quality assessment logic
- Implement outlier detection
- Add missing data handling

#### Step 1.4: Batch Data Collection
- Configure for 2000-2019 time range
- Implement progress tracking
- Add error recovery for long-running jobs

### **Phase 2: Integration with GEFS Data (Week 3-4)**

#### Step 2.1: Time Alignment Module
- Align WunderGround dates with GEFS forecast times
- Handle timezone conversions (Shanghai UTC+8, Denver UTC-7)
- Create aligned training dataset

#### Step 2.2: Feature Engineering Integration
- Combine WunderGround observations with GEFS forecasts
- Extract seasonal features (month, day of year)
- Calculate derived features (temperature ranges, trends)

#### Step 2.3: Training Dataset Creation
- Create labeled dataset for EMOS training
- Split by season (DJF/MAM/JJA/SON)
- Separate max/min temperature datasets

### **Phase 3: Production Data Pipeline (Week 5-6)**

#### Step 3.1: Automated Data Updates
- Schedule regular data collection
- Implement incremental updates
- Add data versioning

#### Step 3.2: Monitoring and Alerting
- Data quality monitoring
- Collection failure alerts
- Storage usage monitoring

#### Step 3.3: Documentation and Testing
- Update documentation with integrated system
- Create integration tests
- Performance testing for large datasets

## Configuration Updates

### **Updated config.yaml for Integrated System**

```yaml
# Temperature Prediction System Configuration

# Data Sources
data_sources:
  wunderground:
    enabled: true
    config_path: "WunderGround Data Extraction/config.yaml"
    stations:
      - id: "ZSPD"
        name: "Shanghai Pudong"
        country: "cn"
        city: "shanghai"
        timezone: "Asia/Shanghai"
        elevation: 4.0
        polymarket_id: "shanghai"
      
      - id: "KDEN"
        name: "Denver International"
        country: "us"
        city: "denver"
        timezone: "America/Denver"
        elevation: 1655.0  # meters
        polymarket_id: "denver"
    
    time_range:
      training_start: 2000
      training_end: 2019
      validation_start: 2020
      validation_end: 2023
  
  gefs:
    enabled: true
    source: "aws_open_data"
    reforecast: true
    realtime: true
    variables: ["t2m"]

# Model Training
model:
  seasonal_buckets: ["DJF", "MAM", "JJA", "SON"]
  target_variables: ["temp_max", "temp_min"]
  training_window_years: 5
  validation_split: 0.2
  test_split: 0.1

# Output
output:
  data_format: "parquet"
  model_format: "joblib"
  predictions_format: "csv"
  database:
    enabled: true
    type: "sqlite"
    path: "./data/predictions.db"
```

## Testing Strategy

### **Unit Tests**
1. **Data Adapter Tests**
   - Test date parsing
   - Test format conversion
   - Test quality assessment

2. **Validator Tests**
   - Test outlier detection
   - Test missing data handling
   - Test physical consistency checks

3. **Integration Tests**
   - Test full pipeline from scraping to training data
   - Test time alignment with GEFS data
   - Test seasonal bucket splitting

### **Data Quality Tests**
1. **Completeness Test**
   - Verify all dates in range have data
   - Check for large gaps (> 7 days)

2. **Consistency Test**
   - Verify temperature ranges are physically possible
   - Check for abrupt changes (> 20°C day-to-day)

3. **Cross-Validation Test**
   - Compare with alternative data sources (if available)
   - Verify station metadata consistency

## Risk Mitigation

### **Technical Risks**
1. **WunderGround Website Changes**
   - Mitigation: Regular monitoring, HTML parsing robustness tests
   - Fallback: Alternative data sources (NOAA, METAR archives)

2. **Data Quality Issues**
   - Mitigation: Multi-level validation, manual spot checks
   - Fallback: Interpolation, climate normals

3. **Scale Issues (2000-2019 data)**
   - Mitigation: Incremental loading, efficient storage (Parquet)
   - Fallback: Start with smaller time range, expand gradually

### **Operational Risks**
1. **Rate Limiting/Banning**
   - Mitigation: Conservative request delays, user agent rotation
   - Fallback: Proxy rotation, scheduled off-peak collection

2. **Storage Requirements**
   - Mitigation: Efficient compression, cloud storage
   - Fallback: Keep only processed features, not raw HTML

## Success Metrics

### **Phase 1 Completion Criteria**
- ✅ Shanghai data (2000-2019) successfully collected
- ✅ Denver data (2000-2019) successfully collected
- ✅ Data quality validation passes (> 95% completeness)
- ✅ Unified training dataset created
- ✅ Integration tests passing

### **Performance Targets**
- Data collection: < 24 hours for 20 years of data
- Data processing: < 1 hour for full dataset
- Storage: < 10GB for raw + processed data
- Reliability: > 99% successful data collection rate

## Next Steps

1. **Immediate (Next 2 days)**
   - Test Denver station data collection
   - Implement date parsing in adapter
   - Create unified data format specification

2. **Short-term (Week 1)**
   - Implement data validation layer
   - Test full 2000-2019 data collection
   - Create integration tests

3. **Medium-term (Week 2)**
   - Integrate with GEFS data pipeline
   - Create aligned training dataset
   - Begin model training with integrated data

This integration plan leverages the existing robust WunderGround scraper while adapting it to the specific needs of the temperature prediction system. The modular design allows for easy replacement of the data source if needed in the future.