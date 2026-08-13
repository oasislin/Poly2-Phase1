# Domain Context: Polymarket Temperature Market Prediction System

## Core Entities

### Temperature Market (温度市场)
A Polymarket binary options market for specific city, date, and temperature thresholds. Markets consist of mutually exclusive temperature bins (e.g., Shanghai max temperature: ≤18, 19, 20, 21, 22, 23, ≥24°C). Only one bin settles as "YES" (1) based on official weather station data.

### Station/Location (站点/位置)
Weather measurement location with attributes:
- `station_id`: Unique identifier (e.g., ZSPD for Shanghai Pudong)
- `city_name`: City name (e.g., Shanghai, Denver)
- `latitude`, `longitude`: Geographic coordinates
- `elevation`: Altitude in meters
- `timezone`: Local timezone with DST flag
- `polymarket_id`: Corresponding Polymarket market identifier
- `data_start_date`, `data_end_date`: Historical data availability
- `data_quality_flag`: Quality assessment indicator

### Temperature Types (温度类型)
- **Observed Temperature (观测温度)**: Actual measurements from Wunderground (historical, for training)
- **Forecast Temperature (预报温度)**: GEFS ensemble predictions (features for model)
- **Settlement Temperature (结算温度)**: Official temperature used by Polymarket for market resolution
- **Model Predicted Temperature (模型预测温度)**: Probability distribution output by our system

### Probability Distribution (概率分布)
Skewed Gaussian distribution parameterized by μ (mean), σ (standard deviation), and skewness parameter. Separate distributions for maximum and minimum temperatures. Can compute probability for any temperature threshold via cumulative distribution function.

### Time Concepts (时间概念)
- **Forecast Initialization Time (预报初始时间)**: GEFS forecast issuance time (00Z, 06Z, 12Z, 18Z)
- **Forecast Lead Time (预报提前时间)**: Forecast prediction horizon
- **Observation Time (观测时间)**: Actual temperature measurement time
- **Market Settlement Time (市场结算时间)**: Polymarket market resolution time (typically local time)
- **Prediction Issue Time (预测发布时间)**: When our system generates predictions
- **Local Day Window (本地日窗口)**: 00:00-23:59 local time for daily max/min temperatures

## Key Relationships

### Data Alignment (数据对齐)
- **Max Temperature Prediction**: Use 00Z GEFS forecast (local morning) covering entire natural day
- **Min Temperature Prediction**: Use previous day's 18Z GEFS forecast (local evening) covering next day's minimum
- **Time Conversion**: All times converted to station local time for alignment

### Unit Conversion (单位转换)
- Internal model: Celsius (°C)
- GEFS data: Convert from Kelvin (K - 273.15)
- Wunderground data: Convert from Fahrenheit if needed ((°F - 32) × 5/9)
- Polymarket output: Celsius for Shanghai, Fahrenheit for Denver

### Existing WunderGround Scraper Integration (现有WunderGround抓取器集成)
- **Data Source**: Weather Underground historical pages (https://www.wunderground.com/history/monthly/)
- **Data Format**: HTML tables with JSON chart data
- **Extracted Fields**: Daily high/low temperatures, humidity, wind, pressure, precipitation
- **Storage**: JSON and CSV formats, organized by station and month
- **Code Location**: `WunderGround Data Extraction/` directory with modular Python scripts

### Probability Calculation (概率计算)
For continuous distribution F(x):
- Single bin "=T": P(T) ≈ F(T+0.5) - F(T-0.5)
- Range bin "T1-T2": P = F(T2+0.5) - F(T1-0.5)
- Boundary bin "≤T": P = F(T)
- Boundary bin "≥T": P = 1 - F(T-ε)

### Dynamic Correction (动态修正)
Conditional probability truncation:
- If current temperature already exceeds threshold: probability = 100%
- Otherwise: P(final ≥ L | current = T_now) = (1 - F(L)) / (1 - F(T_now))
- Triggered with each new temperature observation

### Physical Constraints (物理约束)
- **Maximum warming/cooling rates**: Calculated from historical Wunderground data per station, season, time period
- **Constraint application**: If target temperature L differs from current T_now by more than historical maximum possible change, probability forced to 0 or 1

## System Boundaries

### Phase 1 Scope (第一阶段范围)
- Only Shanghai and Denver cities
- Only daily maximum/minimum temperature predictions
- Only physical probability modeling (no market microstructure)
- Only historical validation (not live trading)
- No liquidity analysis, slippage control, or money management

### Data Sources (数据源)
- **Truth labels**: Wunderground historical data (2000-2019)
- **Forecast features**: GEFS Reforecast (historical) and real-time GEFS
- **Real-time observations**: Current temperature for dynamic correction
- **Validation**: Wunderground data (Polymarket data unavailable pre-2019)

### Model Training (模型训练)
- **Time period**: 2000-2019 (2000-2012 train, 2013-2017 validation, 2018-2019 test)
- **Seasonal buckets**: DJF (Dec-Jan-Feb), MAM (Mar-Apr-May), JJA (Jun-Jul-Aug), SON (Sep-Oct-Nov)
- **Update frequency**: Quarterly retraining with 5-year rolling window
- **Missing data**: Skipped with missing rate recorded as quality metric

## Validation Framework

### Core Metrics (核心指标)
- **PIT Histogram**: Verify probability distribution uniformity (gold standard)
- **CRPS Score**: Measure overall probabilistic prediction accuracy
- **Talagrand Diagram**: Assess ensemble spread reliability

### Domain-Specific Validation (领域特定验证)
- Physical constraint satisfaction (temperatures within historical ranges)
- Station-specific error analysis
- Comparison against naive benchmarks (GEFS mean, climatology)
- Real-time monitoring with sliding windows (30-day CRPS, PIT)

### Alerting (警报机制)
- Trigger when CRPS degrades >20% vs benchmark
- Trigger when PIT histogram shows significant non-uniformity
- Mark predictions as "degraded" when using fallback strategies