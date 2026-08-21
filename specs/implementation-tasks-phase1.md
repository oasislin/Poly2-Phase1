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
**进度**：T01-T07 全部分拆 tickets 已完成（2026-08-15），Task 1.2 验收全部通过（100% 完成）。

**Description**: Implement GEFS data download and preprocessing（v5.9.1 口径）
- [x] Create `gefs_fetcher.py` with Herbie integration（注意：PyPI 包名为 `herbie-data`，`herbie` 为无关包）
- [x] 下载变量：`tmax_2m` / `tmin_2m`（6h 窗口 TMAX/TMIN，非 tmp_2m）
- [x] 支持 reforecast（训练，2000-2019）与 realtime（预测）双模式
- [x] **起报时次**：reforecast 仅 00Z 单时次（AWS 实证，v5.9.1 §1）；realtime 多时次另计
- [x] **成员协议**：训练与预测均使用 c00 + p01-p04 共 5 成员（AWS reforecast 实证仅存 5 成员，2005/2015/2019 验证一致）
- [x] **窗口下载**：每 init 下载覆盖目标本地日的 6h 窗口子集（每变量/成员 3-5 个窗口）
- [x] **区域裁剪流程**：下载后立即裁剪上海/丹佛区域（真实网格 0.25°，上海 25–35N/115–125E 裁剪 = 41×41 格点；**保持 41×41，不规整**）→ 输出"裁剪完成可移走 raw"信号 → 用户移走原始全球文件后继续下一分片
- [x] 实现 GRIB2 → xarray 转换（含经度 0-360°/±180° 处理）
- [x] 数据缓存 + 分片下载（按年/成员/时次）+ 断点续传 + MD5 校验
- [x] Write unit tests for data fetching and parsing（mock 网络）

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
**进度**：T1.3-01 至 T1.3-05 全部完成（2026-08-16），单元测试与集成测试（56 passed）及真实网络冒烟全绿，Task 1.3 验收全部通过（100% 完成）。

**Description**: Implement core data processing utilities（v5.9.1 口径）
- [x] Create `time_aligner.py` for UTC to local time conversion（本地日 = 当地时钟 00:00-24:00，含丹佛 DST 平移）
- [x] Implement `unit_converter.py` for temperature unit standardization
- [x] Create `spatial_interpolator.py` with bilinear interpolation（禁止最近邻；从存储的 0.25° 网格裁剪区域（上海 41×41）提取站点周边 2×2 邻域插值）
- [x] Implement `elevation_corrector.py` with standard lapse rate（Γ = 0.0065 K/m，作用于 TMAX/TMIN 日极值特征）
- [x] Add `feature_extractor.py`：
  - 日极值 = **完全包含（⊆ 本地日）**的 6h TMAX/TMIN 窗口极值（非相交窗口）
  - 集合统计量 = 集合均值 + 集合方差 + 成员极值范围（5 成员；不用分位数与时间特征）
  - **新增城市覆盖告警**：天文算法计算全年日出，验证覆盖跨度 ⊇ [日出−1h, 日出+0.5h]，不满足输出警告
- [x] Create `data_processor.py` for unified end-to-end processing pipeline orchestration
- [x] Write comprehensive tests for all transformations（unit + integration）

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
**进度**：T1.4-01 至 T1.4-04 全部完成（2026-08-16），包含 Schema 验证、Parquet 特征分区库、SQLite 时序数据库及统一 StorageManager，82 项测试全部通过（100% 完成）。

**Description**: Set up data storage infrastructure
- [x] Create directory structure: `data/raw/`, `data/processed/`, `data/models/`, `data/db/`
- [x] Implement Parquet writer/reader for processed features (`parquet_store.py`)
- [x] Set up SQLite database for predictions and metrics (`database.py`)
- [x] Add data versioning with timestamps (`updated_at`, `created_at`, `evaluated_at`)
- [x] Implement data validation on load (`data_validator.py`, `storage_manager.py`)

**Acceptance Criteria**:
- Data organized by station and date
- Parquet files readable by pandas
- SQLite schema supports time-series queries
- Data integrity checks on load

## Phase 1B: Model Implementation (Weeks 3-4)

> **对齐 v5.9.1 执行规格与 Spec #10（2026-08-19）**：Phase 1B 已细化拆解为 12 张垂直切片 Tickets（#11 ~ #22）。

### Task 2.1: Gaussian EMOS Distribution（带气候学方差 Floor）
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 1.1, Task 1.4
**Spec**: #10

- [x] **#11 (Ticket 2.1-01)**: `ClimatologyCalculator` 逐日气候学均值与方差底计算 (严格 OOS 2000-2018 实测, 31 天滑动窗, 1-366 日历日映射)
- [x] **#12 (Ticket 2.1-02)**: `GaussianEMOS` 分布类与数学基础方法 (平方参数化连接函数 $\mu = a + b\bar{T}_{ens}, \sigma^2 = c^2 + d^2 S^2 + \sigma_{clim}^2(d)$, `pdf`, `cdf`, `quantile`, `confidence_interval`)
- [x] **#13 (Ticket 2.1-03)**: Gneiting 闭式高斯 CRPS 向量化解析解与损失函数 ($\text{CRPS}(y, \mu, \sigma)$ 闭式解, 向量化 Batch 损失函数)

### Task 2.2: EMOS Model Training
**Priority**: High
**Estimate**: 4 days
**Dependencies**: Task 2.1, 1.3, 1.4
**Spec**: #10

- [x] **#14 (Ticket 2.2-01)**: `EMOSOptimizer` 单模型参数拟合器与体检评分卡 (L-BFGS-B 优化, $d$ 参数 L2 正则 $\lambda=10^{-3}$, 扰动热启动, 多起点重启, `ModelTrainingDiagnostics`)
- [x] **#15 (Ticket 2.2-02)**: 两级降级容灾与过拟合软告警机制 (Level 1 高斯 EMOS+Floor $\to$ Level 2 气候学; 未收敛/NaN/Inf 硬触发 + 显著劣于气候学软触发 + $|c|,|d|>10$ 软告警)
- [x] **#16 (Ticket 2.2-03)**: 时效归桶与季节分集器 (`DatasetPartitioner`: 名义目标 15:00/06:00 LT $- \text{init}$, `round_to_nearest_6h` 归入 {54,30,6}h/{48,24}h, 4 季节切分)
- [x] **#17 (Ticket 2.2-04)**: 40 组矩阵批量训练编排与矩阵评分看板 (`MatrixTrainer`: 2 站 × 4 季 × 5 节点批量训练, `Matrix Scorecard` 聚合健康看板)
- [x] **#18 (Ticket 2.2-05)**: 缺失时效节点参数插值与短时效外推器 (`LeadTimeInterpolator`: 线性内插 $\{12,18,24,36,42,48\}\text{h}$, 最低温 $<24\text{h}$ 的 $\sqrt{L/24}$ 方差物理衰减)
- [x] **#19 (Ticket 2.2-06)**: 模型持久化与统一注册中心 (`ModelRegistry`: `{StationID}_{Season}_{Max|Min}_lead{Hours}h.pkl` 规范, 统一查询门面 `get_model`)

### Task 2.3: Training & Validation Pipeline
**Priority**: Medium
**Estimate**: 3 days
**Dependencies**: Task 2.2, 1.4
**Spec**: #10

- [x] **#20 (Ticket 2.3-01)**: `ValidationEngine` 样本外验证引擎与时间墙隔离 (严格时间墙隔离 2000-2018 训练 / 2019 样本外验证, Rolling-Origin 时序交叉验证, 样本外 CRPS/MAE/Spread/PIT 计算)
- [x] **#21 (Ticket 2.3-02)**: 三重验收门禁与评估报告生成器 (`ReportGenerator`: 真实节点 PIT $p>0.05$, 30h 留出插值 $\text{CRPS}_{virt} \le 1.05\text{CRPS}_{real}$, 极端天气 90% CI 覆盖率 $\ge 80\%$, Pass/Fail 裁决与时效分级报告)
- [x] **#22 (Ticket 2.3-03)**: `TrainingPipeline` 端到端训练编排与 CLI 命令 (`scripts/train_emos_matrix.py` 一键全流程, 集成测试与端到端测试)

## Phase 1C: Prediction System (Weeks 5-6)

> **对齐 v5.9.2 执行规格与 Spec #25（2026-08-20）**：Phase 1C 细化拆解为 6 张垂直切片 Tickets（#26 ~ #31），覆盖静态基础预测、实时动态截断、物理变温率硬约束、Polymarket 规则离散化转换与端到端预测流水线。

### Task 3.1: Static Prediction
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 2.2, 1.2
**Spec**: #25

- [x] **#26 (Ticket 3.1-01)**: `StaticPredictor` 静态基础预测器与模型路由 (封装 `ModelRegistry` 与 `LeadTimeInterpolator`, 输出基准高斯分布参数 $(\mu, \sigma)$ 与 `GaussianEMOS`, 支持置信区间与 DataFrame 批量预测)

### Task 3.2: Dynamic Correction
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 3.1
**Spec**: #25

- [x] **#27 (Ticket 3.2-01)**: `DynamicCorrector` 实时实况条件概率截断修正器 ($P(X \ge L \mid X \ge T_{now})$ 与 $P(X \le L \mid X \le T_{now})$ 数学截断, 后验 CDF 单调性保证, 边界除零 $\epsilon=10^{-7}$ 保护与缺测无缝回退)

### Task 3.3: Physical Constraints
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 1.2, 3.1
**Spec**: #25

- [x] **#28 (Ticket 3.3-01)**: `ConstraintEnforcer` 历史变温率物理极限硬约束执行器 (基于各站点历史最大升/降温速率 $\times$ 极值窗口剩余时间 $\Delta t$ 计算机理不可达边界, 硬截断超限概率为 0.0/1.0, 物理硬约束覆盖优先级)

### Task 3.4: Bin Probability Conversion
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 3.1, 3.2, 3.3
**Spec**: #25

- [x] **#29 (Ticket 3.4-01)**: `BinConverter` Polymarket 盘口规则转换器与单位映射 (依据 Polymarket 规则模板与自适应区间生成互斥 Bins, $\pm 0.5^\circ$ 连续性修正, 丹佛华氏度/上海摄氏度转换, 概率和归一化, Wunderground 实测结算判定)

### Task 3.5: Prediction Pipeline & Orchestration
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 3.1, 3.2, 3.3, 3.4
**Spec**: #25

- [x] **#30 (Ticket 3.5-01)**: `PredictionPipeline` 端到端三层预测编排与 SQLite 持久化 (串联四层预测组件, 结果持久化写入 `predictions.db`, 降级状态与元数据记录)
- [x] **#31 (Ticket 3.5-02)**: `run_predictions.py` CLI 预测工具与全流程集成验证 (命令行交互参数, 预测看板生成, `tests/integration/test_prediction_integration.py` 两站端到端集成测试)

## Phase 1D: Validation System (Weeks 7-8)

> **对齐 v5.9.2 执行规格与 Spec #32（2026-08-20）**：Phase 1D 细化拆解为 7 张垂直切片 Tickets（#33 ~ #39），覆盖统计指标与显著性检验、v5.9.2 三重验收门禁、历史回测引擎与多基线比对、回测看板生成器、监控告警系统与 CLI 端到端集成。

### Task 4.1: Validation Metrics & Triple Acceptance System
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 2.2, 3.1, 3.4
**Spec**: #32

- [x] **#33 (Ticket 4.1-01)**: `MetricsCalculator` 高级概率与离散盘口验证指标库 (CRPS, CRPSS, Brier Score, BSS, Log Loss, PIT Histogram, Talagrand Rank Diagram, ECE 与 Reliability Curve)
- [x] **#34 (Ticket 4.1-02)**: `StatisticalSignificance` 预测技巧统计显著性检验器 (Diebold-Mariano Test, Wilcoxon Signed-Rank Test, Paired t-test, PIT KS 检验)
- [x] **#35 (Ticket 4.1-03)**: `TripleGateEvaluator` v5.9.2 三重验收门禁独立评估器 (标准节点 PIT+CRPS 检验、30h 留出插值 $\text{CRPS}_{virt} \le 1.05\text{CRPS}_{real}$ 双断言、2019 严格 OOS 极端事件 90% CI $\ge 80\%$ 覆盖率与气候学战胜双断言)

### Task 4.2: Historical Backtesting
**Priority**: High
**Estimate**: 4 days
**Dependencies**: Task 4.1, 3.4, 3.5
**Spec**: #32

- [x] **#36 (Ticket 4.2-01)**: `Backtester` 历史回测引擎与多基准 Baseline 对比 (Rolling-Origin 滚动时序回测、支持静态/动态/物理约束/离散化 Bins 端到端预测链路、Climatology / Raw GEFS / Persistence 基准对比)
- [x] **#37 (Ticket 4.2-02)**: `BacktestReporter` 回测综合评估看板与时序诊断生成器 (Lead Time 衰减分析、季节/站点分层矩阵、Polymarket 盘口校准可靠性评估、Markdown/JSON/CSV 报告导出)

### Task 4.3: Alerting System
**Priority**: Medium
**Estimate**: 2 days
**Dependencies**: Task 4.1, 4.2
**Spec**: #32

- [x] **#38 (Ticket 4.3-01)**: `AlertManager` & `AlertDispatcher` 性能劣化与数据异常监控告警系统 (CRPS 劣化 >20% 监控、PIT 偏离告警、数据时效 Freshness 检查、防抖节流 Throttling、多通道分发)

### Task 4.4: Validation CLI & System Integration
**Priority**: High
**Estimate**: 2 days
**Dependencies**: Task 4.1, 4.2, 4.3
**Spec**: #32

- [x] **#39 (Ticket 4.4-01)**: `run_backtest.py` CLI 命令行工具与 Phase 1D 全流程集成验证 (两站历史全流程回测验证、三重门禁自动化裁决、监控告警全链路集成测试)

## Phase 1E: System Integration (Week 9)

> **对齐 v5.9.2 执行规格与 Phase 1 验收标准（2026-08-21）**：Phase 1E 细化拆解为 7 张垂直切片 Tickets（#40 ~ #46），全量覆盖强类型配置系统、结构化日志与运行 Profiler、MainPipeline 状态机编排器、全局异常隔离与健康诊断、统一 CLI 交互中枢、E2E 系统级集成测试与运维排障手册。

### Task 5.1: Configuration & Logging Infrastructure
**Priority**: High
**Estimate**: 2 days
**Dependencies**: Phase 1A-1D

- [x] **#40 (Ticket 5.1-01)**: `ConfigManager` 强类型配置管理系统与多环境架构 (`src/pipeline/config.py`, `configs/default.yaml`, `configs/dev.yaml`, `configs/test.yaml`, Pydantic Schema 强校验, 环境变量 `POLY_*` 注入与敏感信息脱敏)
- [x] **#41 (Ticket 5.1-02)**: `StructuredLogger` 结构化日志与运行度量 (`src/utils/logger.py`, `src/utils/profiler.py`, 上下文绑定 `contextualize(station, stage, lead_time)`, `@profile_stage` 耗时统计与 Markdown 报告)

### Task 5.2: Pipeline Orchestration & Resilience
**Priority**: High
**Estimate**: 3 days
**Dependencies**: Task 5.1, All previous phases

- [x] **#42 (Ticket 5.2-01)**: `MainPipeline` 端到端全流程状态机编排器 (`src/pipeline/main_pipeline.py`, Ingest $\to$ Feature $\to$ Train $\to$ Predict $\to$ Validate 五阶段调度, 支持 `--resume-from` 断点恢复)
- [x] **#43 (Ticket 5.2-02)**: `PipelineResilience` & `HealthChecker` 全局异常隔离与健康自检 (`src/pipeline/resilience.py`, `src/pipeline/health.py`, 指数退避重试, 单点故障隔离, 存储/数据库/40模型就绪诊断)

### Task 5.3: Unified CLI Interface
**Priority**: High
**Estimate**: 1 day
**Dependencies**: Task 5.1, 5.2

- [x] **#44 (Ticket 5.3-01)**: `scripts/run_poly_pipeline.py` 统一 CLI 交互中枢 (支持 `all`, `health`, `ingest`, `feature`, `train`, `predict`, `backtest` 子命令与配置动态注入)

### Task 5.4: End-to-End System Testing
**Priority**: High
**Estimate**: 2 days
**Dependencies**: Task 5.1, 5.2, 5.3

- [x] **#45 (Ticket 5.4-01)**: `tests/e2e/test_e2e_pipeline.py` E2E 系统级全链路集成测试套件 (两站从输入到 40 模型推理与 Triple Gate 评测闭环验证)

### Task 5.5: Production Documentation & Assets
**Priority**: Medium
**Estimate**: 1 day
**Dependencies**: Task 5.1~5.4

- [x] **#46 (Ticket 5.5-01)**: 生产配置手册与实战排障运维指南 (`README.md`, `docs/configuration-guide.md`, `docs/troubleshooting.md`, 提炼自 ADRs 0001~0005 与真实踩坑实证)

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