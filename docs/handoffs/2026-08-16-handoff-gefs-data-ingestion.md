# Handoff: GEFS 真实数据批量下载与特征库生产运行 (Data Ingestion Run)

**日期**: 2026-08-16  
**上下文**: Phase 1A（Task 1.1 ~ Task 1.4）已 100% 交付并通过 82 项测试与真实数据契约校验。本任务为独立的**数据实例化与生产运行任务**，为 Phase 1B 模型训练准备全量 20 年（2000-2019）的真实特征库（Parquet）与训练对齐数据。

---

## 1. 目标与产出物

1. **抓取目标**：NOAA GEFSv12 Reforecast 数据（2000-2019 年，上海 ZSPD 与丹佛 KDEN 两站点）。
2. **加工产出**：
   - 裁剪网格：`data/processed/gefs/{year}/{station}/*.nc`（0.25° 网格 41×41）
   - 特征库：`data/processed/features/{station}/{year}.parquet`（经双线性插值、高程修正、时效对齐与 5 成员极值折叠提取的标准特征 DataFrame）
3. **最终验收标准**：
   - 通过 `StorageManager.load_training_dataset(station_id, target_type, lead_time_bucket, ...)` 能够一键秒级加载特征 $X$ 与实测 $y$（来自 `data/wunderground.db`）对齐的训练数据集。

---

## 2. 核心可用组件与架构索引

新 Agent 可直接调用以下已测试完毕的核心服务模块（无需重复造轮子）：

| 模块类别 | 核心类 / 脚本 | 物理路径 | 作用 |
|---|---|---|---|
| **抓取引擎** | `GEFSFetcher` | [`src/data_acquisition/gefs_fetcher.py`](file:///Users/ericlin/SynologyDrive/Project/Poly%20Way2/src/data_acquisition/gefs_fetcher.py) | Herbie 下载 AWS S3 真实 GRIB2 子集、0.25° 网格 41×41 裁剪、5 成员（c00/p01-p04）、MD5 断点续传 |
| **批处理调度** | `GEFSBatchDownloader` | [`src/data_acquisition/gefs_batch_downloader.py`](file:///Users/ericlin/SynologyDrive/Project/Poly%20Way2/src/data_acquisition/gefs_batch_downloader.py) | CSV 状态机驱动、按年分块、空间自动释放 |
| **特征处理器** | `DataProcessor` | [`src/data_processing/data_processor.py`](file:///Users/ericlin/SynologyDrive/Project/Poly%20Way2/src/data_processing/data_processor.py) | 4 点双线性插值、开尔文转摄氏度、高程递减率订正、6h 极值折叠、5 成员统计量提取 |
| **特征存储器** | `ParquetFeatureStore` | [`src/data_processing/parquet_store.py`](file:///Users/ericlin/SynologyDrive/Project/Poly%20Way2/src/data_processing/parquet_store.py) | 按 `{station}/{year}.parquet` 分区落盘、时间戳版本管理、原子安全写入、Upsert 去重 |
| **统一管理门面** | `StorageManager` | [`src/data_processing/storage_manager.py`](file:///Users/ericlin/SynologyDrive/Project/Poly%20Way2/src/data_processing/storage_manager.py) | 串联特征 Parquet 与实测 SQLite，提供 `load_training_dataset` 一键合并训练集 |
| **历史实测库** | SQLite DB | `data/wunderground.db` | 已包含 2015-2024 年上海与丹佛的日最高温、最低温实测标签 $y$ |

---

## 3. 分阶段执行指南

### 阶段 1：2019 年单年小规模端到端联调（Smoke Gate）

**目的**：先跑 2019 单年，验证“下载 $\to$ 裁剪 $\to$ 提取特征 $\to$ Parquet 落盘 $\to$ 训练集对齐”全链路一键畅通。

**步骤**：
1. **组装胶水处理函数/脚本**（例如扩展 `GEFSBatchDownloader` 或提供独立脚本 `scripts/run_pipeline_year.py`）：
   ```python
   from datetime import datetime, timezone
   from pathlib import Path
   import xarray as xr
   from src.data_processing import DataProcessor, StorageManager

   processor = DataProcessor()
   storage = StorageManager()

   # 对裁剪后的每一年 NetCDF 文件提取特征并保存至 Parquet
   # 支持 max (lead_time_bucket=30) 与 min (lead_time_bucket=24)
   ```
2. **执行 2019 年下载与处理**：
   ```bash
   python scripts/download_gefs_batch.py --start-year 2019 --end-year 2019 --auto-continue
   ```
3. **验证生成的 Parquet 特征库**：
   - 检查 `data/processed/features/ZSPD/2019.parquet` 是否存在；
   - 检查 `data/processed/features/KDEN/2019.parquet` 是否存在；
   - 运行快速测试脚本，调用 `storage.load_training_dataset(station_id="ZSPD", target_type="max", lead_time_bucket=30)`，打印前 5 行确认特征与真实观测已完美对齐。

---

### 阶段 2：全量 2000-2018 年后台受控批量下载（Production Run）

在阶段 1 验证通过后，开启全量 20 年生产运行：

**执行命令**：
```bash
python scripts/download_gefs_batch.py --start-year 2000 --end-year 2018 --auto-continue
```

**运行管控机制**：
1. **状态跟踪**：所有年份下载与裁剪进度记录在 `data/gefs_download_state.csv`。
2. **断点续传**：支持随时中断；已下载并验证 MD5 的 `.nc` 文件会自动跳过，避免重复消耗网络。
3. **硬盘保护**：每一年裁剪完成后会自动清理原始临时文件，确保硬盘占用恒定可控。

---

## 4. 关键物理与工程约束（踩坑备忘）

1. **时次约束**：GEFS Reforecast 仅存在 **`00Z` 单时次**（06/12/18Z 不存在，已在 Issue #9 实证）。
2. **网格规格**：真实分辨率为 **0.25°**（上海 25-35°N, 115-125°E，丹佛 35-45°N, -110--100°E，均为 41×41 格点）。
3. **特征列规范**：严格为 4 项集合统计量 `{ensemble_mean, ensemble_variance, member_max, member_min}`，严禁引入分位数（p10/p90）或人工时间特征。
4. **运行环境**：macOS 本机运行 Python 或 pytest 时需保证使用对应虚拟环境或指定 `BypassSandbox: true`。

---

## 5. 验收交接指令

当新 Agent 完成 2019 年单年联调或全量下载后，执行以下命令验收：
```bash
pytest tests/unit/ tests/integration/ -v
```
并打印 `storage_manager.verify_storage_health()` 报告特征库存量即可！
