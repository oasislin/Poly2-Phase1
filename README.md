# Polymarket 温度预测系统 - Phase 1

本项目旨在构建一个高精度的物理概率模型，用于预测 Polymarket 温度市场的日最高和最低气温概率分布。系统以高斯 EMOS（Ensemble Model Output Statistics）模型为核心，采用“离线气象数据特征提取 $\to$ 统计后处理校准 $\to$ 概率分布建模与回测”的量化架构。

当前项目执行规范严格遵循 **《项目执行文件 v5.9.1(细化版)》**，业务需求来源为 **《项目方案：Polymarket 温度市场量化投注系统 (v2.3)》**。

---

## 核心系统架构

### 1. 业务与物理预测层级
```
┌─────────────────────────────────────────────────────────────┐
│                    物理约束层 (Phase 1C)                    │
│  基于站点历史极端变温率的物理边界硬拦截                      │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│                    动态修正层 (Phase 1C)                    │
│  实时观测温度动态概率截断: P(X ≥ L | X > T_now)             │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│                 静态高斯 EMOS 基础模型 (Phase 1B)           │
│  μ = a + b * ensemble_mean                                  │
│  σ² = max(c + d * ensemble_variance, σ_clim_floor²)         │
│  (分城市 2 站 × 分季节 4 季 × 分时效 5 桶 = 40 组独立模型)  │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│                 数据工程与特征存储基石 (Phase 1A)           │
│  Wunderground 实测 + GEFS 0.25° 网格 4 点双线性空间插值     │
│  本地日完全包含 6h 窗口切片 + 高程递减率修正 (Γ=0.0065 K/m) │
│  Parquet 特征库 ({station}/{year}.parquet) + SQLite 时序库  │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目当前进展与完成状态

| 阶段 / 模块 | 包含任务 | 完成度 | 状态与核心交付物 |
|---|---|:---:|---|
| **Phase 1A: 核心数据基础设施** | **Task 1.1 ~ Task 1.4** | **100%** | • **Task 1.1**: Wunderground 历史实测抓取与 SQLite 存储（2015-2024 年）<br>• **Task 1.2**: NOAA GEFS Reforecast/Realtime 下载器（0.25° 网格裁剪、5 成员、MD5 断点续传、CSV 状态机）<br>• **Task 1.3**: 核心数据加工（时效完全包含对齐、4 点双线性插值、摄氏度标准化、高程订正、5 成员极值折叠、NOAA 天文日出安全校验）<br>• **Task 1.4**: 数据存储与管理（Parquet 特征分区库、SQLite 时序库、Schema/物理校验器 `DataValidator`、一键对齐训练集门面 `StorageManager`） |
| **Phase 1B: EMOS 概率模型** | **Task 2.1 ~ Task 2.3** | 0% | • **Task 2.1**: 高斯 EMOS 分布类（含气候学方差 Floor）<br>• **Task 2.2**: CRPS 损失函数最小化优化器（BFGS/Nelder-Mead）<br>• **Task 2.3**: 滚动窗口时序回测引擎（40 个模型矩阵、CRPS/PIT 评估） |
| **Phase 1C: 修正层与系统验证** | **Task 3.1 ~ Task 3.3** | 0% | • **Task 3.1**: 实时观测动态概率截断修正<br>• **Task 3.2**: 历史变温率物理约束<br>• **Task 3.3**: 完整系统回测与压力测试 |

---

## 核心技术特性 (Phase 1A)

1. **时效对齐与 6h 窗口完全包含（$\subseteq$ 规则）**：
   - 严格按站点本地时钟 `00:00 - 24:00 LT`（支持丹佛 DST 夏冬令时平移）；
   - 仅纳入完全包含在目标自然日内的 6h 预报窗口（上海 3 窗口、丹佛夏 4 窗口、丹佛冬 3 窗口），杜绝未来信息穿越；
   - 内置 NOAA 天文算法，自动验证 6h 窗口对最低气温敏感时段 $[\text{日出}-1.0\text{h}, \text{日出}+0.5\text{h}]$ 的安全覆盖。

2. **空间插值与物理修正**：
   - **4 点双线性插值**（严格禁用最近邻）：精确匹配 GEFS 0.25° 网格 2×2 邻域，自动处理纬度降序及经度 $[0, 360]$ / $[-180, 180]$ 映射；
   - **国际标准大气递减率高程修正**：$\Gamma = 0.0065\text{ K/m}$（$6.5^\circ\text{C/km}$），按站点实际海拔与模式网格地形差对 TMAX/TMIN 进行物理订正。

3. **特征协议与模型规格**：
   - 严格输出 4 项 5 成员集合统计量：`{ensemble_mean, ensemble_variance, member_max, member_min}`；
   - 遵循 v5.9.1 规范，彻底剔除易引发过拟合的分位数（p10/p90）与人工时间特征（$\sin/\cos$）。

4. **存储与数据完整性保障**：
   - **特征存储**：按 `data/processed/features/{station_id}/{year}.parquet` 分层分区存储，支持原子写入（Atomic write）与主键去重 Upsert；
   - **完整性校验**：`DataValidator` 强制拦截缺失字段、NaN/Inf 以及非物理极限气温（$-60^\circ\text{C} \sim +60^\circ\text{C}$）。

---

## 目录结构

```
Poly2-Phase1/
├── src/
│   ├── data_acquisition/        # 数据采集层
│   │   ├── wunderground_scraper.py      # Wunderground 历史实测爬虫
│   │   ├── gefs_fetcher.py              # NOAA GEFS GRIB2 下载器与区域裁剪
│   │   └── gefs_batch_downloader.py     # CSV 状态机驱动的批量下载调度器
│   └── data_processing/         # 数据工程与存储层
│       ├── constants.py                 # 站点元数据与中央常量
│       ├── unit_converter.py            # 温度单位向量化转换 (K/C/F)
│       ├── elevation_corrector.py       # 高程递减率物理订正
│       ├── spatial_interpolator.py      # 4 点双线性空间插值
│       ├── time_aligner.py              # 本地日时效对齐与 NOAA 天文日出校验
│       ├── feature_extractor.py         # 6h 极值折叠与 5 成员统计量提取
│       ├── data_processor.py            # 端到端数据处理统一编排
│       ├── data_validator.py            # Schema 与物理合理性校验器
│       ├── parquet_store.py             # Parquet 分区特征库
│       ├── database.py                  # SQLite 时序与指标数据库引擎
│       └── storage_manager.py           # 统一存储管理门面 (对齐训练集 X, y)
├── scripts/                      # 工具与批处理脚本
│   ├── download_gefs_batch.py           # GEFS 2000-2019 批量下载与状态机
│   ├── download_wunderground_batch.py   # Wunderground 批量抓取
│   └── verify_offline_gefs.py           # 离线真实 GRIB2 子集快速验证
├── tests/                        # 测试套件 (82 项测试全绿)
│   ├── unit/                            # 单元测试 (采集、处理、校验、存储)
│   └── integration/                     # 端到端集成测试 (真实 GRIB2 管道实证)
├── data/                         # 数据目录
│   ├── raw/                             # 原始 GRIB2 / 下载缓存
│   ├── processed/                       # 处理后数据 (features/ 与 gefs/)
│   ├── models/                          # 模型权重与元数据
│   └── db/                              # SQLite 数据库 (predictions.db, wunderground.db)
├── specs/                        # 实施规格与任务跟踪
│   └── implementation-tasks-phase1.md   # Phase 1 细化任务跟踪表
├── 项目执行文件 v5.9.1(细化版).md   # 核心系统执行权威规范
└── 项目方案：Polymarket 温度市场量化投注系统 (v2.3).md # 总体需求方案
```

---

## 快速开始

### 1. 安装环境与依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整测试套件
```bash
# 运行全部 82 项单元测试与集成测试
pytest tests/ -v

# 运行真实 NOAA GEFS 网络冒烟测试 (需要外网)
RUN_NETWORK_TESTS=1 pytest tests/unit/data_acquisition/test_gefs_fetcher.py::test_network_reforecast_single_message -q
```

### 3. Python API 快速调用示例
```python
from datetime import date, datetime, timezone
import xarray as xr
from src.data_processing import DataProcessor, StorageManager

# 1. 初始化服务
processor = DataProcessor()
storage = StorageManager()

# 2. 假设已有裁剪后的 GEFS Dataset (ds)
# 将预报网格数据一键转化为校准特征 DataFrame
features_df = processor.process_forecast_to_features(
    dataset=ds,
    station_id="ZSPD",
    target_date=date(2019, 7, 2),
    init_time_utc=datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc),
    target_type="max",
    lead_time_bucket=30,
)

# 3. 持久化到 Parquet 特征库
storage.save_forecast_features(features_df)

# 4. Phase 1B 训练时一键加载特征与实测对齐的完整训练集
train_df = storage.load_training_dataset(
    station_id="ZSPD",
    target_type="max",
    lead_time_bucket=30,
    start_date="2019-01-01",
    end_date="2019-12-31",
)
print(train_df[["target_date", "ensemble_mean", "ensemble_variance", "observed_temp"]])
```

---

## 质量与测试保证

- **单元测试覆盖**：覆盖所有数学转换（开尔文/摄氏度/华氏度、递减率、双线性插值矩阵、时区 DST、日出计算、集合方差 $ddof=1$）。
- **真实数据实证**：通过本地缓存的真实 NOAA GRIB2 子集文件，验证了上海夏季（2019-07-01 00Z）与丹佛冬季（2019-01-01 00Z）在实际气象条件下的端到端处理与物理合理性。
- **契约测试与离线重放**：契约测试直接钉住 GRIB idx 正则与 0.25° 网格方向，防止外部依赖变更导致静默失败。