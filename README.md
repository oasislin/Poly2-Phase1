# Polymarket 温度预测系统 - Phase 1 (高精度物理概率模型)

本项目旨在构建一个高精度的物理概率模型，用于预测 Polymarket 温度市场的日最高和最低气温概率分布。系统以高斯 EMOS（Ensemble Model Output Statistics）模型为核心，采用“离线气象数据特征提取 $\to$ 40 组模型矩阵训练 $\to$ 实时动态截断与物理约束 $\to$ Polymarket 离散盘口概率转换 $\to$ 三重验收门禁回测与监控告警 $\to$ 状态机流水线编排”的全链路量化架构。

当前项目执行规范严格遵循 **《项目执行文件 v5.9.2(细化版)》**，业务需求来源为 **《项目方案：Polymarket 温度市场量化投注系统 (v2.3)》**。

---

## 核心系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    物理约束层 (Phase 1C)                    │
│  基于站点历史极端变温率的物理边界硬拦截                      │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│                    动态修正层 (Phase 1C)                    │
│  实时观测温度条件概率截断: P(X ≥ L | X > T_now)             │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│                 静态高斯 EMOS 基础模型 (Phase 1B)           │
│  μ = a + b * ensemble_mean                                  │
│  σ² = c² + d² * ensemble_variance + σ_clim(d)² (平方参数化) │
│  (2 站 × 4 季 × 5 节点 = 40 组独立矩阵模型 + 时效插值)        │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│                 数据工程与特征存储基石 (Phase 1A)           │
│  Wunderground 实测 + GEFS 0.25° 网格 4 点双线性空间插值     │
│  本地日完全包含 6h 窗口切片 + 高程递减率修正 (Γ=0.0065 K/m) │
│  Parquet 特征库 ({station}/{year}.parquet) + SQLite 数据库  │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目阶段与完成状态 (Phase 1 100% 交付)

| 阶段 / 模块 | 包含任务 / Tickets | 完成度 | 状态与核心交付物 |
|---|---|:---:|---|
| **Phase 1A: 核心数据基础设施** | **Task 1.1 ~ Task 1.4** | **100%** | • Wunderground 历史实测抓取与持久化<br>• NOAA GEFS GRIB2 5 成员（c00+p01~p04）下载、41×41 区域裁剪与断点续传<br>• 本地日 6h 极值对齐、4 点双线性空间插值与高程修正（$\Gamma=0.0065\text{ K/m}$）<br>• Parquet 分区特征库（`data/processed/features/`）与 SQLite 存储管理器 |
| **Phase 1B: EMOS 概率模型** | **Task 2.1 ~ Task 2.3 (#11~#22)** | **100%** | • 31 天滑动窗严格 OOS 气候学方差 Floor $\sigma_{clim}^2(d)$（[ADR 0001](docs/adr/0001-gaussian-emos-with-variance-floor.md)）<br>• 平方参数化高斯 EMOS 与 Gneiting 闭式高斯 CRPS 损失<br>• L-BFGS-B 优化器（$d$ 正则化 + 扰动热启动）<br>• 40 组模型矩阵批量训练、两级降级容灾（[ADR 0003](docs/adr/0003-two-level-degradation.md)）与缺失时效参数插值/物理衰减（[ADR 0002](docs/adr/0002-five-lead-time-nodes-with-interpolation.md)） |
| **Phase 1C: 预测系统** | **Task 3.1 ~ Task 3.5 (#26~#31)** | **100%** | • 静态基准预测与模型路由 (`StaticPredictor`)<br>• 实时观测条件概率截断 (`DynamicCorrector`, $\epsilon=10^{-7}$)<br>• 历史极限变温率物理硬约束 (`ConstraintEnforcer`)<br>• Polymarket 离散盘口区间转换与归一化 (`BinConverter`)<br>• 端到端三层预测流水线与 SQLite 持久化 (`PredictionPipeline`) |
| **Phase 1D: 验证与回测系统** | **Task 4.1 ~ Task 4.4 (#33~#39)** | **100%** | • CRPS / CRPSS / Brier / LogLoss / PIT / ECE 验证指标库<br>• Diebold-Mariano 统计显著性检验器<br>• v5.9.2 三重验收门禁（真实节点 PIT、30h 留出插值、2019 极端天气双断言）<br>• Rolling-Origin 历史时序回测与多基准对比引擎 (`Backtester`)<br>• 回测综合报告生成器与 CRPS 劣化告警系统 (`AlertManager`) |
| **Phase 1E: 系统集成与工程化** | **Task 5.1 ~ Task 5.5 (#40~#46)** | **100%** | • 强类型 YAML 配置系统与环境隔离 (`ConfigManager`, Pydantic Schema)<br>• 结构化上下文日志与耗时分析器 (`StageProfiler`)<br>• 全流程状态机编排器 (`MainPipeline`, 支持 `--resume-from` 断点恢复)<br>• 生产级统一 CLI 交互中枢 (`scripts/run_poly_pipeline.py`)<br>• 系统级 E2E 全链路集成测试套件与排障运维手册 |

---

## 核心技术规格与设计决策

1. **集合协议与网格规范**：
   - **5 成员严格对齐**：仅使用 `c00` + `p01~p04` 共 5 个成员（[ADR 0004](docs/adr/0004-five-member-strict-alignment.md)），训练集与预测集成员集合严格同构；
   - **0.25° 网格裁剪**：全球 0.25° 网格下载后立即裁剪至站点区域（上海 41×41 格点，保持原始分辨率不规整），支持原始 raw 文件即刻移走以节省磁盘；
   - **4 点双线性插值**：严禁最近邻，基于站点周边 2×2 邻域插值，自动修正纬度方向与经度区间；
   - **高程物理修正**：按标准大气温度递减率 $\Gamma = 0.0065\text{ K/m}$（$6.5^\circ\text{C/km}$）进行地形差订正。

2. **数学模型与方差保护**：
   - **平方参数化方差**：$\sigma^2 = c^2 + d^2 S_{ens}^2 + \sigma_{clim}^2(d)$，天然保证非负性，消除负参数误判风险；
   - **气候学方差 Floor**：基于 2000-2018 严格 OOS 实测数据，采用 31 天滑动窗计算，彻底防止 5 成员小集合在低离散度天气下的过度自信与方差坍塌（[ADR 0001](docs/adr/0001-gaussian-emos-with-variance-floor.md)）；
   - **时效分桶与内插外推**：最高温真实节点 `{6h, 30h, 54h}`，最低温真实节点 `{24h, 48h}`。缺失时效线性内插，$<24\text{h}$ 最低温模型引入 $\sigma \cdot \sqrt{L/24}$ 物理衰减（[ADR 0002](docs/adr/0002-five-lead-time-nodes-with-interpolation.md)）。

3. **三层预测与盘口转化**：
   - **实时动态截断**：结合日内最新实况，对累积极值进行条件概率截断 $P(X \ge L \mid X \ge T_{now})$，后验单调性保障与 $\epsilon=10^{-7}$ 边界保护；
   - **变温率硬拦截**：基于站点历史最大升/降温速率 $\times$ 极值窗口剩余时间 $\Delta t$ 计算机理不可达边界，硬截断超限概率；
   - **Polymarket 盘口映射**：针对上海（℃）与丹佛（℉）市场规则，生成互斥 Bins，施加 $\pm 0.5^\circ$ 连续性修正，并保证概率全概率公式归一化（$\sum P = 1.0$）。

4. **v5.9.2 三重验收门禁**：
   - **标准节点校准**：真实时效节点 PIT 直方图 K-S 检验 $p > 0.05$；
   - **留出插值守恒**：30h 留出插值虚拟模型 $\text{CRPS}_{virt} \le 1.05 \times \text{CRPS}_{real}$ 且 PIT $p > 0.05$；
   - **极端天气压力测试**：2019 严格 OOS 极端事件 90% 置信区间覆盖率 $\ge 80\%$，且 $\text{CRPS}_{model} < \text{CRPS}_{clim}$（战胜气候学）。

---

## 目录结构

```
Poly Way2/
├── configs/                     # 系统配置文件
│   ├── default.yaml             # 生产级默认配置
│   ├── dev.yaml                 # 开发与快速调试配置
│   └── test.yaml                # 单元/集成测试配置
├── src/
│   ├── data_acquisition/        # 数据采集层
│   │   ├── gefs_fetcher.py              # NOAA GEFS GRIB2 下载器与区域裁剪
│   │   ├── gefs_batch_downloader.py     # CSV 状态机驱动的批量下载调度器
│   │   └── wunderground_scraper.py      # Wunderground 历史实测爬虫
│   ├── data_processing/         # 数据工程与存储层
│   │   ├── constants.py                 # 站点坐标、高程及中央常量
│   │   ├── unit_converter.py            # 温度单位向量化转换 (K/C/F)
│   │   ├── elevation_corrector.py       # 高程递减率物理订正
│   │   ├── spatial_interpolator.py      # 4 点双线性空间插值
│   │   ├── time_aligner.py              # 本地日时效对齐与 NOAA 天文日出校验
│   │   ├── feature_extractor.py         # 6h 极值折叠与 5 成员统计量提取
│   │   ├── data_processor.py            # 端到端数据处理统一编排
│   │   ├── data_validator.py            # Schema 与物理合理性校验器
│   │   ├── parquet_store.py             # Parquet 分区特征库 ({station}/{year}.parquet)
│   │   ├── database.py                  # SQLite 时序与指标数据库引擎
│   │   └── storage_manager.py           # 统一存储管理门面 (对齐训练集 X, y)
│   ├── modeling/                # EMOS 概率建模层
│   │   ├── climatology.py               # 31 天滑动窗 OOS 气候学方差 Floor
│   │   ├── gaussian_emos.py             # 平方参数化高斯 EMOS 分布类
│   │   ├── crps.py                      # Gneiting 闭式高斯 CRPS 向量化损失
│   │   ├── emos_trainer.py              # L-BFGS-B 参数优化器与体检评分卡
│   │   ├── degradation.py               # 两级降级容灾与过拟合软告警
│   │   ├── partitioner.py               # 季节分集与 6h 时效归桶器
│   │   ├── interpolator.py              # 缺失节点参数插值与短时效物理衰减
│   │   ├── matrix_trainer.py            # 40 组矩阵批量训练与评估看板
│   │   ├── registry.py                  # 模型持久化与注册中心门面
│   │   ├── validation_engine.py         # 样本外时序交叉验证引擎
│   │   ├── report_generator.py          # 三重门禁评估报告生成器
│   │   └── pipeline.py                  # 端到端模型训练编排器
│   ├── prediction/              # 三层预测与盘口转化层
│   │   ├── static_predictor.py          # 静态基础预测与模型路由
│   │   ├── dynamic_corrector.py         # 实时实况条件概率截断
│   │   ├── constraint_enforcer.py       # 历史变温率物理极限硬约束
│   │   ├── bin_converter.py             # Polymarket 盘口区间转换与结算判定
│   │   └── prediction_pipeline.py       # 端到端四层预测流水线与持久化
│   ├── validation/              # 回测验证与监控告警层
│   │   ├── metrics.py                   # CRPS / Brier / LogLoss / PIT / ECE 指标库
│   │   ├── significance.py              # Diebold-Mariano / Wilcoxon 统计显著性检验
│   │   ├── triple_gate.py               # v5.9.2 三重验收门禁独立评估器
│   │   ├── backtester.py                # Rolling-Origin 历史时序回测引擎
│   │   ├── backtest_reporter.py         # 回测综合评估看板与时序诊断生成器
│   │   ├── alert_manager.py             # 性能劣化与数据异常告警管理器
│   │   └── alert_dispatcher.py          # 告警防抖节流与多通道分发器
│   ├── pipeline/                # 全局流水线与健康中枢
│   │   ├── config.py                    # 强类型 Pydantic 配置管理系统
│   │   ├── health.py                    # 存储/数据库/模型就绪健康诊断
│   │   ├── resilience.py                # 全局重试与异常隔离
│   │   └── main_pipeline.py             # Ingest->Feature->Train->Predict->Validate 编排器
│   └── utils/                   # 通用工具层
│       ├── logger.py                    # 结构化上下文日志记录器
│       └── profiler.py                  # 阶段耗时统计与 Profiler
├── scripts/                     # 命令行工具与可执行脚本
│   ├── run_poly_pipeline.py             # 生产级统一 CLI 交互中枢
│   ├── train_emos_matrix.py             # 40 组矩阵模型一键训练脚本
│   ├── run_predictions.py               # 盘口概率预测 CLI 工具
│   ├── run_backtest.py                  # 历史时序回测与门禁评估 CLI 工具
│   ├── download_gefs_batch.py           # GEFS 历史数据批量下载
│   └── download_wunderground_batch.py   # Wunderground 历史实测批量抓取
├── tests/                       # 测试套件 (347+ 项测试全绿)
│   ├── unit/                            # 单元测试 (按模块分层隔离)
│   ├── integration/                     # 模块间集成测试
│   └── e2e/                             # 系统级全链路 E2E 验收测试
├── docs/                        # 项目文档体系
│   ├── adr/                             # 架构决策记录 (ADR 0001 ~ 0005)
│   ├── configuration-guide.md           # 生产配置手册
│   ├── troubleshooting.md               # 实战排障与运维指南
│   └── spec-update-process.md           # 规格变更流程
└── specs/                       # 实施任务与工程规格跟踪
    └── implementation-tasks-phase1.md   # Phase 1 细化任务跟踪表 (100% Complete)
```

---

## 快速上手与 CLI 命令

统一 CLI 入口为 [`scripts/run_poly_pipeline.py`](scripts/run_poly_pipeline.py)：

### 1. 系统健康自检
检查 SQLite 数据库、Parquet 特征库及 40 组模型就绪状态：
```bash
python scripts/run_poly_pipeline.py health --config configs/default.yaml
```

### 2. 端到端全流程运行
按顺序执行 `Ingest -> Feature -> Train -> Predict -> Validate` 全阶段：
```bash
# 开发环境运行
python scripts/run_poly_pipeline.py all --env dev

# 从训练阶段断点恢复运行后续所有阶段
python scripts/run_poly_pipeline.py all --resume-from train
```

### 3. 单阶段调用
```bash
# 训练 40 组 EMOS 矩阵模型
python scripts/run_poly_pipeline.py train --start-year 2000 --end-year 2018

# 单站单日盘口概率预测 (上海最高温)
python scripts/run_poly_pipeline.py predict --station ZSPD --date 2026-08-21 --target-type max

# 执行历史时序回测与三重门禁裁决
python scripts/run_poly_pipeline.py backtest --start-year 2018 --end-year 2019
```

---

## Python API 调用示例

### 1. 三层预测系统调用 (静态基准 + 动态截断 + 物理约束 + 盘口转换)
```python
from datetime import date
from src.prediction import PredictionPipeline
from src.pipeline.config import ConfigManager

# 1. 加载配置并初始化端到端预测流水线
config = ConfigManager.load_config("configs/default.yaml")
pipeline = PredictionPipeline(config=config)

# 2. 执行单站单日预测 (包含实时实况动态修正)
result = pipeline.predict_single_day(
    station_id="ZSPD",
    target_date=date(2026, 8, 21),
    target_type="max",
    current_obs_temp=32.5,  # 实时观测温度 (℃)
    lead_hours=18.0         # 距离目标有效时间提前量
)

# 3. 查看输出的高斯参数与 Polymarket 离散盘口概率
print(f"校准高斯分布: μ={result.static_mu:.2f}°C, σ={result.static_sigma:.2f}°C")
print("Polymarket 盘口区间概率分布:")
for bin_name, prob in result.bin_probabilities.items():
    print(f"  {bin_name}: {prob * 100:.1f}%")
```

### 2. 统一全流程编排器调用 (`MainPipeline`)
```python
from src.pipeline.main_pipeline import MainPipeline
from src.pipeline.config import ConfigManager

config = ConfigManager.load_config(env="dev")
pipeline = MainPipeline(config=config)

# 运行全链路流水线并获取各阶段执行摘要
summary = pipeline.run_all(resume_from="train")
print(f"流水线状态: {summary['status']}, 耗时报告: {summary['profile_report_path']}")
```

---

## 测试套件执行与验证

项目采用严格的 TDD（测试驱动开发）规范开发，全量覆盖单元测试、集成测试与系统级 E2E 测试：

```bash
# 1. 运行全量测试套件 (347+ 项测试全绿通过)
pytest tests/unit/ tests/integration/ tests/e2e/ -q

# 2. 运行系统级全链路 E2E 测试
pytest tests/e2e/test_e2e_pipeline.py -v

# 3. 运行真实 NOAA AWS 网络冒烟测试 (需外网访问)
RUN_NETWORK_TESTS=1 pytest tests/unit/data_acquisition/test_gefs_fetcher.py::test_network_reforecast_single_message -v
```

---

## 文档索引

- **系统配置指南**：[`docs/configuration-guide.md`](docs/configuration-guide.md)
- **运维排障手册**：[`docs/troubleshooting.md`](docs/troubleshooting.md)
- **架构决策记录 (ADR)**：[`docs/adr/`](docs/adr/)
  - `0001`: 高斯 EMOS 与 31 天滑动窗气候学方差 Floor
  - `0002`: 五个时效节点与参数插值/物理衰减
  - `0003`: 两级降级容灾架构
  - `0004`: 5 成员集合协议严格对齐
  - `0005`: GRIB 变量合并容差约束
- **任务实施规格**：[`specs/implementation-tasks-phase1.md`](specs/implementation-tasks-phase1.md)
- **核心执行规范**：[`项目执行文件 v5.9.2(细化版)`](项目执行文件%20v5.9.2(细化版).md)
- **业务方案蓝图**：[`Polymarket 温度市场量化投注系统 (v2.3)`](Polymarket%20温度市场量化投注系统%20(v2.3).md)