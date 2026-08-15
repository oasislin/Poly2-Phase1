# Implementation Tasks: Phase 1

> **对齐 v5.9.1 执行规格（2026-08-15）**：Task 1.2 起按 v5.9.1 口径执行（v5.9 的更新版，无冲突：高斯 EMOS + 气候学方差 Floor、6h 窗口 TMAX/TMIN 特征、5 成员集合、20 模型/站矩阵 + 参数插值、Reforecast 仅 00Z 起报）。**Task 1.1 为已完成历史记录，冻结不改。**

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
**进度**：T01（GEFS reforecast 下载最小路径）已完成（2026-08-14）；T02（双变量 + 5 成员协议）已完成（2026-08-15）；T03-T07 待开始。

**Description**: Implement GEFS data download and preprocessing（v5.9.1 口径）
- [ ] Create `gefs_fetcher.py` with Herbie integration（注意：PyPI 包名为 `herbie-data`，`herbie` 为无关包）
- [ ] 下载变量：`tmax_2m` / `tmin_2m`（6h 窗口 TMAX/TMIN，非 tmp_2m）
- [ ] 支持 reforecast（训练，2000-2019）与 realtime（预测）双模式
- [ ] **起报时次**：reforecast 仅 00Z 单时次（AWS 实证，v5.9.1 §1）；realtime 多时次另计
- [ ] **成员协议**：训练与预测均使用 c00 + p01-p04 共 5 成员（AWS reforecast 实证仅存 5 成员，2005/2015/2019 验证一致）
- [ ] **窗口下载**：每 init 下载覆盖目标本地日的 6h 窗口子集（每变量/成员 3-5 个窗口）
- [ ] **区域裁剪流程**：下载后立即裁剪上海/丹佛区域（真实网格 0.25°，上海 25–35N/115–125E 裁剪 = 41×41 格点；是否规整到 21×21 见 T03）→ 输出"裁剪完成可移走 raw"信号 → 用户移走原始全球文件后继续下一分片
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

**Description**: Implement core data processing utilities（v5.9.1 口径）
- [ ] Create `time_aligner.py` for UTC to local time conversion（本地日 = 当地时钟 00:00-24:00，含丹佛 DST 平移）
- [ ] Implement `unit_converter.py` for temperature unit standardization
- [ ] Create `spatial_interpolator.py` with bilinear interpolation（禁止最近邻；从存储的 0.25° 网格裁剪区域（上海 41×41，最终尺寸见 T03 规整策略）提取站点周边 2×2 邻域插值）
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

### Task 2.1: Gaussian EMOS Distribution（带气候学方差 Floor）
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 1.1

**Description**: Implement Gaussian EMOS distribution with climatological variance floor（v5.9 口径；废弃偏态）
- [ ] Create `climatology.py` computing σ_clim(d)/μ_clim(d)（31 天滑动窗 × 2000-2018 实测，逐日平滑，严格 OOS）
- [ ] Implement Gaussian PDF, CDF, and quantile functions（无 skewness）
- [ ] Implement closed-form CRPS for Gaussian（Gneiting 公式）
- [ ] 连接函数：μ = a + b·T̄_ens；σ² = c² + d²·S²_ens + σ²_clim(d)（平方参数化）
- [ ] Write mathematical property tests

**Acceptance Criteria**:
- σ_clim(d) 计算仅用 2000-2018，不碰 2019 验证集
- Gaussian CRPS 匹配参考实现
- 连接函数 σ² 天然 ≥ σ²_clim(d)（Floor 生效）
- Tests verify mathematical properties

### Task 2.2: EMOS Model Training
**Priority**: High
**Estimate**: 4 days
**Dependencies**: Task 2.1, 1.3

**Description**: Implement EMOS calibration for Gaussian with variance floor（v5.9 口径）
- [ ] Create `emos_trainer.py` with closed-form CRPS minimization（L-BFGS-B）
- [ ] **模型矩阵**：季节 × 时效节点 → 20 模型/站点（最高温 3 节点 {54,30,6} + 最低温 2 节点 {48,24}），2 站共 40 个
- [ ] 命名规范 `{StationID}_{Season}_{Max|Min}_lead{Hours}h.pkl`
- [ ] **Lead Time 归桶**：名义目标时间（最高温 15:00 LT / 最低温 06:00 LT）− init，round_to_nearest_6h
- [ ] **两级降级**：Level 1 高斯 EMOS+Floor → Level 2 气候学；硬触发（超迭代/NaN/Inf）+ 软触发（验证集 CRPS 显著劣于气候学）；|c|/|d|>10 仅告警
- [ ] 优化：仅对 d 加 L2(λ=1e-3)；热启动 (a,b,c,d)=(0,1,0,1) + O(1e-3) 扰动
- [ ] 训练/预测成员对齐（c00+p01-p04 5 成员）
- [ ] Implement model persistence with pickle/joblib

**Acceptance Criteria**:
- 20 模型/站点矩阵正确生成，命名规范正确
- CRPS 随 Lead Time 衰减用**配对统计检验**验收（不显著倒挂视为持平通过）
- 两级降级正确触发（输入病态数据返回气候学且不崩溃）
- 各真实时效节点独立通过 PIT K-S 检验（p>0.05）
- 极端事件压力测试通过（见 Task 4.1）
- Models can be saved and loaded

### Task 2.3: Training Pipeline
**Priority**: Medium
**Estimate**: 3 days
**Dependencies**: Task 2.2, 1.4

**Description**: Build complete training pipeline（v5.9 口径）
- [ ] Create `training_pipeline.py` orchestrating data→features→training
- [ ] 验证集双轨：单次留出（训练 2000-2018 / 验证 2019）+ 训练期滚动验证（Rolling-Origin，逐年滚动原点）
- [ ] 时间墙隔离：训练阶段严禁访问任何验证集数据
- [ ] Add model versioning with DVC
- [ ] Create training reports with metrics（CRPS 衰减曲线、PIT、极端压力测试报告）
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

**Description**: Implement base prediction from GEFS features（v5.9 口径：高斯 + 插值）
- [ ] Create `static_predictor.py` loading appropriate seasonal model
- [ ] Generate Gaussian parameters (μ, σ)（无 skewness）
- [ ] **参数插值**：缺失时效节点在相邻真实节点间线性内插 (a,b,c,d)
- [ ] **短时效收缩**：最低温 <24h 借用 24h 参数，σ_final = σ_24h × √(L/24)
- [ ] Add confidence intervals and prediction intervals
- [ ] Implement batch prediction for multiple stations/dates

**Acceptance Criteria**:
- Correct model selected by station, season, temp type, and lead time（命中真实节点用真实模型，否则插值）
- 插值输出 = 端点参数线性组合
- 短时效收缩公式正确（√(L/24)）
- Predictions generated within time budget
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
**Dependencies**: Task 2.2, 3.1

**Description**: Implement comprehensive validation metrics（v5.9 三重验收）
- [ ] Create `metrics_calculator.py` with CRPS, PIT, Talagrand
- [ ] **标准节点验收**：PIT K-S（p>0.05）+ CRPS 衰减配对检验
- [ ] **插值节点验收**：留出最高温 30h，用 {6h,54h} 插值重建；双断言 CRPS_virtual ≤ 1.05×CRPS_real 且 PIT p>0.05
- [ ] **极端事件 OOS 压力测试**：2019 池（阈值 2000-2018 分位数），Warm/Cold Tail 对称；双断言 CRPS_model < CRPS_clim 且 90% CI 覆盖率 ≥ 80%
- [ ] Add Talagrand diagram generation
- [ ] Implement statistical significance testing

**Acceptance Criteria**:
- CRPS calculation matches reference
- PIT histogram correctly computed
- 三重验收全部通过（标准/插值/极端）
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
- [ ] Gaussian EMOS implementation mathematically correct (with variance floor)
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