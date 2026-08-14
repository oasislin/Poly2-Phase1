# Implementation Tasks: Phase 1

> **对齐 v5.7 执行规格（2026-08-14）**：Task 1.2 起按 v5.7 口径执行（时效分层 EMOS、6h 窗口 TMAX/TMIN 特征、5 成员集合、68 模型/站矩阵）。**Task 1.1 为已完成历史记录，冻结不改。**

## Phase 1A: Core Infrastructure (Weeks 1-2)

### Task 1.1: Wunderground Data Pipeline
**Priority**: High
**Estimate**: 3 days
**Dependencies**: None

**Description**: Implement Wunderground historical data scraping and preprocessing with complete field extraction
- [ ] Enhance existing `wunderground_scraper.py` to extract all 12 meteorological fields from HTML/JSON
- [ ] Support Shanghai (ZSPD) and Denver (KDEN) stations with automatic unit conversion
- [ ] Implement SQLite database for persistent storage with resume capability
- [ ] Handle web scraping errors and rate limiting (5-second delays between requests)
- [ ] Extract complete daily observations including:
  - Temperature (high/low)
  - Dew point (high/low) 
  - Humidity
  - Wind speed and direction
  - Pressure
  - Precipitation
  - Weather conditions
- [ ] Implement comprehensive data validation and quality assessment for all fields
- [ ] Create data caching to avoid redundant requests
- [ ] Write comprehensive tests for scraping and parsing all fields

**Acceptance Criteria**:
- Can fetch Wunderground data for 2000-2019 period for both stations
- Returns structured data with all 12 meteorological fields
- Handles network errors gracefully with retry logic and exponential backoff
- Converts units correctly (Fahrenheit to Celsius for Denver station)
- Validates data quality for all fields and flags issues
- Stores data in SQLite database with proper schema
- Supports resume capability for interrupted downloads
- Caches downloaded HTML pages locally (30-day TTL)
- Provides CSV export functionality for analysis
- Includes progress tracking and detailed error logging

### Task 1.2: GEFS Data Pipeline
**Priority**: High
**Estimate**: 3 days
**Dependencies**: None

**Description**: Implement GEFS data download and preprocessing（v5.7 口径）
- [ ] Create `gefs_fetcher.py` with Herbie integration（注意：PyPI 包名为 `herbie-data`，`herbie` 为无关包）
- [ ] 下载变量：`tmax_2m` / `tmin_2m`（6h 窗口 TMAX/TMIN，非 tmp_2m）
- [ ] 支持 reforecast（训练，2000-2019）与 realtime（预测）双模式
- [ ] **成员协议**：训练与预测均使用 c00 + p01-p04 共 5 成员（AWS reforecast 实证仅存 5 成员，2005/2015/2019 验证一致）
- [ ] **窗口下载**：每 init 下载覆盖目标本地日的 6h 窗口子集（每变量/成员 3-5 个窗口）
- [ ] **区域裁剪流程**：下载后立即裁剪上海/丹佛区域（21×21 格点）→ 输出"裁剪完成可移走 raw"信号 → 用户移走原始全球文件后继续下一分片
- [ ] 实现 GRIB2 → xarray 转换（含经度 0-360°/±180° 处理）
- [ ] 数据缓存 + 分片下载（按年/成员/时次）+ 断点续传 + MD5 校验
- [ ] Write unit tests for data fetching and parsing（mock 网络）

**Acceptance Criteria**:
- 可下载 reforecast（2000-2019）与 realtime 的 tmax_2m/tmin_2m 6h 窗口数据
- 返回含 latitude/longitude/time/member 坐标的 xarray Dataset
- 5 成员协议正确（训练/预测同成员集合）
- 区域裁剪正确，裁剪后可通知移走 raw，磁盘仅保留裁剪数据
- 网络错误重试 + 断点续传 + MD5 校验
- 缓存避免重复下载

### Task 1.3: Data Processing Foundation
**Priority**: High  
**Estimate**: 4 days
**Dependencies**: Task 1.1, 1.2

**Description**: Implement core data processing utilities（v5.7 口径）
- [ ] Create `time_aligner.py` for UTC to local time conversion（本地日 = 当地时钟 00:00-24:00，含丹佛 DST 平移）
- [ ] Implement `unit_converter.py` for temperature unit standardization
- [ ] Create `spatial_interpolator.py` with bilinear interpolation（禁止最近邻）
- [ ] Implement `elevation_corrector.py` with standard lapse rate（Γ = 0.0065 K/m，作用于 TMAX/TMIN 日极值特征）
- [ ] Add `feature_extractor.py`：
  - 日极值 = **完全包含（⊆ 本地日）**的 6h TMAX/TMIN 窗口极值（非相交窗口）
  - 集合统计量 = 集合均值 + 集合方差 + 成员极值范围（5 成员；不用分位数与时间特征）
  - **新增城市覆盖告警**：天文算法计算全年日出，验证覆盖跨度 ⊇ [日出−1h, 日出+0.5h]，不满足输出警告
- [ ] Write comprehensive tests for all transformations

**Acceptance Criteria**:
- 所有温度内部为 Celsius
- 本地日边界正确（含丹佛 DST：夏 12:00Z / 冬 13:00Z 平移）
- 窗口纳入规则 = 完全包含（上海 3 窗口 / 丹佛夏 4 窗口 / 丹佛冬 3 窗口，安全边际已验证）
- 双线性插值匹配参考实现
- 高程修正应用于 TMAX/TMIN 特征
- 特征提取输出 {mean, variance, member_max, member_min}，无分位数

### Task 1.4: Data Storage System
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 1.1, 1.2, 1.3

**Description**: Set up data storage infrastructure
- [ ] Create directory structure: `data/raw/`, `data/processed/`, `data/models/`
- [ ] Implement Parquet writer/reader for processed features
- [ ] Set up SQLite database for predictions and metrics
- [ ] Add data versioning with timestamps
- [ ] Implement data validation on load

**Acceptance Criteria**:
- Data organized by station and date
- Parquet files readable by pandas
- SQLite schema supports time-series queries
- Data integrity checks on load

## Phase 1B: Model Implementation (Weeks 3-4)

### Task 2.1: Skewed Gaussian Distribution
**Priority**: High
**Estimate**: 3 days
**Dependencies**: None

**Description**: Implement skewed Gaussian probability distribution
- [ ] Create `skewed_gaussian.py` with μ, σ, skewness parameters
- [ ] Implement PDF, CDF, and quantile functions
- [ ] Add parameter estimation from data
- [ ] Implement CRPS calculation for skewed Gaussian
- [ ] Write mathematical property tests

**Acceptance Criteria**:
- Distribution functions numerically stable
- CRPS calculation matches reference implementation
- Parameter estimation converges correctly
- Tests verify mathematical properties

### Task 2.2: EMOS Model Training
**Priority**: High
**Estimate**: 4 days
**Dependencies**: Task 2.1, 1.3

**Description**: Implement EMOS calibration for skewed Gaussian（v5.7 口径）
- [ ] Create `emos_trainer.py` with CRPS minimization
- [ ] **模型矩阵**：季节（DJF/MAM/JJA/SON）× 时效节点 → 68 模型/站点（最高温 9 节点 {54,48,42,36,30,24,18,12,6} + 最低温 8 节点 {48,42,36,30,24,18,12,6}），2 站共 136 个
- [ ] 命名规范 `{StationID}_{Season}_{Max|Min}_lead{Hours}h.pkl`
- [ ] **Lead Time 归桶**：名义目标时间（最高温 15:00 LT / 最低温 06:00 LT）− init，round_to_nearest_6h
- [ ] **三级降级**（v5.3）：Level 1 偏态 EMOS → Level 2 标准高斯 → Level 3 气候学；硬触发（优化失败/NaN/超迭代）+ 软触发（训练集 CRPS 无改进）
- [ ] 训练/预测成员对齐（c00+p01-p04 5 成员）
- [ ] Implement model persistence with pickle/joblib

**Acceptance Criteria**:
- 68 模型/站点矩阵正确生成，命名规范正确
- CRPS 随 Lead Time 衰减用**配对统计检验**验收（不显著倒挂视为持平通过）
- 三级降级正确触发（输入病态数据返回 Level 3 且不崩溃）
- 各时效分桶独立通过 PIT K-S 检验（p>0.05）
- MPIW Ratio < 0.9（锐度，相对气候学窄 ≥10%）
- Models can be saved and loaded

### Task 2.3: Training Pipeline
**Priority**: Medium
**Estimate**: 3 days
**Dependencies**: Task 2.2, 1.4

**Description**: Build complete training pipeline（v5.7 口径）
- [ ] Create `training_pipeline.py` orchestrating data→features→training
- [ ] 验证集双轨：单次留出（训练 2000-2018 / 验证 2019）+ 训练期滚动验证（Rolling-Origin，逐年滚动原点）
- [ ] 时间墙隔离：训练阶段严禁访问任何验证集数据
- [ ] Add model versioning with DVC
- [ ] Create training reports with metrics（CRPS 衰减曲线、PIT、MPIW）
- [ ] Add automated retraining scheduling

**Acceptance Criteria**:
- Pipeline runs end-to-end without errors
- Models versioned and reproducible
- Training reports include all validation metrics
- Quarterly retraining works correctly

## Phase 1C: Prediction System (Weeks 5-6)

### Task 3.1: Static Prediction
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 2.2, 1.2

**Description**: Implement base prediction from GEFS features
- [ ] Create `static_predictor.py` loading appropriate seasonal model
- [ ] Generate skewed Gaussian parameters (μ, σ, skewness)
- [ ] Add confidence intervals and prediction intervals
- [ ] Implement batch prediction for multiple stations/dates

**Acceptance Criteria**:
- Correct model selected by station, season, and temp type
- Predictions generated within time budget
- Confidence intervals mathematically correct
- Batch prediction efficient

### Task 3.2: Dynamic Correction
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 3.1

**Description**: Implement conditional probability truncation
- [ ] Create `dynamic_corrector.py` with hourly updates
- [ ] Implement P(X ≥ L | X > T_now) for max temps
- [ ] Implement P(X ≤ L | X < T_now) for min temps
- [ ] Add real-time temperature integration
- [ ] Handle missing current temperature gracefully

**Acceptance Criteria**:
- Posterior probability ≠ prior when current temp known
- Mathematical formulas implemented correctly
- Handles edge cases (current temp at boundaries)
- Efficient computation for hourly updates

### Task 3.3: Physical Constraints
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 1.2, 3.1

**Description**: Apply physical constraints based on historical rates
- [ ] Create `constraint_enforcer.py` with station-specific limits
- [ ] Calculate max warming/cooling rates from Wunderground data
- [ ] Implement constraint override logic
- [ ] Add seasonal variation in constraints

**Acceptance Criteria**:
- Constraints respect historical limits
- Override logic correct (0 or 1 probabilities)
- Seasonal constraints applied appropriately
- Constraints documented with source data

### Task 3.4: Bin Probability Conversion
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 3.1, 3.2, 3.3

**Description**: Convert continuous distributions to Polymarket bins
- [ ] Create `bin_converter.py` for probability calculation
- [ ] Implement bin types: "=T", "T1-T2", "≤T", "≥T"
- [ ] Add probability mass calculation for each bin
- [ ] Ensure probabilities sum to 1 across bins

**Acceptance Criteria**:
- Bin probabilities mathematically correct
- Probabilities sum to 1 within tolerance
- All bin types supported
- Efficient calculation for multiple thresholds

## Phase 1D: Validation System (Weeks 7-8)

### Task 4.1: Validation Metrics
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 2.1, 3.1

**Description**: Implement comprehensive validation metrics
- [ ] Create `metrics_calculator.py` with CRPS, PIT, Talagrand
- [ ] Implement PIT histogram calculation and uniformity test
- [ ] Add Talagrand diagram generation
- [ ] Create benchmark comparisons (GEFS mean, climatology)
- [ ] Implement statistical significance testing

**Acceptance Criteria**:
- CRPS calculation matches reference
- PIT histogram correctly computed
- Talagrand diagram shows spread reliability
- Statistical tests correctly implemented

### Task 4.2: Historical Backtesting
**Priority**: High
**Estimate**: 4 days
**Dependencies**: Task 4.1, 3.4

**Description**: Backtest system on historical data
- [ ] Create `backtester.py` for historical validation
- [ ] Implement time-series cross-validation respecting temporal order
- [ ] Generate validation reports with all metrics
- [ ] Compare against naive benchmarks
- [ ] Create visualizations of performance over time

**Acceptance Criteria**:
- Proper time-series splitting (no data leakage)
- Comprehensive performance reports
- Statistical superiority over benchmarks
- Clear visualizations of results

### Task 4.3: Alerting System
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 4.1

**Description**: Implement monitoring and alerting
- [ ] Create `alert_manager.py` with configurable thresholds
- [ ] Add CRPS degradation detection (>20% vs benchmark)
- [ ] Implement PIT non-uniformity detection
- [ ] Add data freshness monitoring
- [ ] Create alert channels (log, email, Slack)

**Acceptance Criteria**:
- Alerts trigger on performance degradation
- Configurable thresholds and channels
- Alert throttling to prevent spam
- Clear alert messages with context

## Phase 1E: System Integration (Week 9)

### Task 5.1: Main Pipeline Integration
**Priority**: High
**Estimate**: 3 days
**Dependencies**: All previous tasks

**Description**: Integrate all components into unified pipeline
- [ ] Create `main_pipeline.py` orchestrating data→features→prediction→validation
- [ ] Implement configuration system (YAML)
- [ ] Add logging and monitoring throughout
- [ ] Create command-line interface
- [ ] Add comprehensive error handling

**Acceptance Criteria**:
- Pipeline runs end-to-end from config
- All components integrated correctly
- Comprehensive logging at all stages
- Graceful error handling and recovery

### Task 5.2: Configuration System
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 5.1

**Description**: Create flexible configuration system
- [ ] Design YAML configuration schema
- [ ] Implement configuration validation
- [ ] Add environment-specific configs (dev, test, prod)
- [ ] Create configuration documentation
- [ ] Add secret management for API keys

**Acceptance Criteria**:
- All parameters configurable
- Configuration validation with clear errors
- Environment separation works correctly
- Secrets not exposed in logs

### Task 5.3: Documentation and Examples
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 5.1

**Description**: Create user documentation and examples
- [ ] Write comprehensive README
- [ ] Create example configuration files
- [ ] Add usage examples for common scenarios
- [ ] Document API/interfaces for each module
- [ ] Create troubleshooting guide

**Acceptance Criteria**:
- New user can run pipeline from documentation
- All configuration options documented
- Common issues and solutions documented
- API documentation complete

## Testing Tasks (Parallel)

### Test Infrastructure
**Priority**: High
**Estimate**: Ongoing

**Description**: Build and maintain test infrastructure
- [ ] Set up pytest with coverage reporting
- [ ] Create test data fixtures
- [ ] Implement CI/CD pipeline (GitHub Actions)
- [ ] Add property-based testing with hypothesis
- [ ] Create performance benchmarking

**Acceptance Criteria**:
- Test suite runs in CI
- Coverage > 80%
- Tests are fast and reliable
- Performance benchmarks established

### Test Implementation
**Priority**: High  
**Estimate**: Ongoing (parallel with development)

**Description**: Implement TDD tests for each component
- [ ] Data acquisition tests (mocked network)
- [ ] Data processing tests (unit and integration)
- [ ] Model training tests (mathematical correctness)
- [ ] Prediction tests (end-to-end scenarios)
- [ ] Validation tests (metric calculations)

**Acceptance Criteria**:
- All public interfaces tested
- Edge cases covered
- Tests are independent and repeatable
- Tests document expected behavior

## Success Criteria Checklist

### Data Pipeline
- [ ] GEFS data downloaded and processed correctly
- [ ] Wunderground data integrated and validated
- [ ] Time alignment works for all timezones
- [ ] Feature extraction produces expected outputs
- [ ] Data storage efficient and reliable

### Model Training
- [ ] Skewed Gaussian implementation mathematically correct
- [ ] EMOS training reduces CRPS
- [ ] Separate models for max/min and seasons
- [ ] Models can be saved and loaded
- [ ] Quarterly retraining works automatically

### Prediction System
- [ ] Static predictions generated from GEFS features
- [ ] Dynamic correction updates probabilities correctly
- [ ] Physical constraints applied appropriately
- [ ] Bin probabilities calculated correctly
- [ ] Hourly updates complete within 5 minutes

### Validation
- [ ] CRPS calculated correctly
- [ ] PIT histograms show calibration
- [ ] Model outperforms naive benchmarks statistically
- [ ] Backtesting shows consistent performance
- [ ] Alerting triggers on degradation

### System Integration
- [ ] End-to-end pipeline runs without errors
- [ ] Configuration system flexible and validated
- [ ] Logging provides debugging information
- [ ] Error handling graceful and informative
- [ ] Documentation complete and accurate

## Risk Mitigation

### Technical Risks
1. **GEFS data availability**: Implement fallback to cached data
2. **Model convergence issues**: Add regularization and fallback to simpler model
3. **Performance bottlenecks**: Profile and optimize critical paths
4. **Numerical instability**: Use double precision and stable algorithms

### Project Risks
1. **Scope creep**: Strict Phase 1 boundaries (no trading logic)
2. **Timeline slippage**: Weekly progress reviews and adjustment
3. **Data quality issues**: Robust validation and quality flags
4. **Model validation failures**: Multiple fallback strategies

### Mitigation Strategies
- **Weekly demos**: Show working components every Friday
- **Incremental delivery**: Deliver usable components weekly
- **Early validation**: Validate against historical data early
- **Continuous integration**: Catch issues immediately