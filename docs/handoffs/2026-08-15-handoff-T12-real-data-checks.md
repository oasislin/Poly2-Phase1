# Handoff: Task 1.2 真实数据测试记录 + 离线校验

> 日期：2026-08-15
> 目的：记录 Task 1.2（GEFS 数据管道）完成后做的**真实联网测试**，把每个样例的
> 输入、逻辑、以及观测到的**黄金值**（golden values）固定下来。数据/逻辑出错时，
> **不联网**，用本地缓存的真实 GRIB 子集 + 离线校验脚本先排查。

## 一、资产位置

| 资产 | 路径 |
|---|---|
| 离线校验脚本 | `scripts/verify_offline_gefs.py` |
| 在线探针脚本 | `scripts/probe_gefs.py`（联网下载，单成员单日） |
| 本地真实数据缓存 | `data/raw/gefs_probe/`（约 14 MB，4 个 case 的 GRIB 子集 + idx） |

**离线校验用法**（无需网络）：

```bash
python scripts/verify_offline_gefs.py
# 期望输出 4/4 cases passed
```

## 二、数据规格结论（真实观测，非假设）

| 项 | 结论 |
|---|---|
| reforecast 网格 | 0.25°，上海/丹佛裁剪 = **41×41** |
| realtime 网格 | `atmos.25` = 0.25°，裁剪 = **41×41**（`atmos.5` 是 0.5° = 21×21，**不可用**） |
| dtype | `float32`（GRIB 解码原生） |
| 变量 | `tmax` / `tmin`（解码名；Herbie 文件名为 `tmax_2m` / `tmin_2m`） |
| 维度结构 | reforecast：`(time, step, latitude, longitude)`，`time`=reference，`step`=窗口；realtime 单 fxx：`(latitude, longitude)` + 标量 `time/step/valid_time` |
| `valid_time` | 6h 窗口**终点**（UTC），是本地日对齐的关键坐标 |
| 窗口选择 | 上海 3 窗口 `[24,30,36]`；丹佛夏 4 `[36,42,48,54]`；丹佛冬 3 `[42,48,54]` |

## 三、测试样例 + 黄金值

每个 case 都是**单成员（c00）+ 单 init 日**，输入通过 `GEFSFetcher.download_reforecast`
/ `download_realtime`，数据流 = `search 子集下载 → cfgrib 解码 → extract_region 裁剪 → merge`。

### Case 1 — 上海 reforecast 夏

- 输入：`region=shanghai(25–35N, 115–125E)`，`init=2019-07-01 00Z`，`members=[0]`，`forecast_hours=[24,30,36]`
- 黄金值：
  - `data_vars=['tmax','tmin']`，dtype `float32`
  - `step=3`（3 窗口），`latitude=41`，`longitude=41`
  - `valid_time = [2019-07-02T00:00, 2019-07-02T06:00, 2019-07-02T12:00]`（UTC）
  - lat 25.0..35.0，lon 115.0..125.0
- 本地缓存：
  - `gefs_reforecast/20190701/subset_19efc1fa__tmax_2m_2019070100_c00.grib2`
  - `gefs_reforecast/20190701/subset_19efc1fa__tmin_2m_2019070100_c00.grib2`

### Case 2 — 丹佛 reforecast 夏（DST）

- 输入：`region=denver(35–45N, −110..−100E)`，`init=2019-07-01 00Z`，`members=[0]`，`forecast_hours=[36,42,48,54]`
- 黄金值：
  - `step=4`，`latitude=41`，`longitude=41`
  - `valid_time = [2019-07-02T12:00, 2019-07-02T18:00, 2019-07-03T00:00, 2019-07-03T06:00]`
  - lat 35.0..45.0，lon −110..−100（经度已归一化到 ±180）
- 本地缓存：
  - `gefs_reforecast/20190701/subset_19efcf91__tmax_2m_2019070100_c00.grib2`
  - `gefs_reforecast/20190701/subset_19efcf91__tmin_2m_2019070100_c00.grib2`

### Case 3 — 丹佛 reforecast 冬

- 输入：`region=denver`，`init=2019-01-01 00Z`，`members=[0]`，`forecast_hours=[42,48,54]`
- 黄金值：
  - `step=3`，`latitude=41`，`longitude=41`
  - `valid_time = [2019-01-02T18:00, 2019-01-03T00:00, 2019-01-03T06:00]`
  - lat 35.0..45.0，lon −110..−100
- 本地缓存：
  - `gefs_reforecast/20190101/subset_97ef3872__tmax_2m_2019010100_c00.grib2`
  - `gefs_reforecast/20190101/subset_97ef3872__tmin_2m_2019010100_c00.grib2`

### Case 4 — 上海 realtime f006（atmos.25）

- 输入：`region=shanghai`，`forecast_time=2024-01-01 00Z`，`members=[0]`，`fxx_hours=[6]`
- 黄金值：
  - `data_vars=['tmax','tmin']`，dtype `float32`
  - `step=1`（单窗口），`latitude=41`，`longitude=41`
  - `valid_time = 2024-01-01T06:00`（**标量**，realtime 单消息解码）
  - lat 25.0..35.0，lon 115.0..125.0
- 本地缓存：
  - `gefs/20240101/subset_6bb21aeb__gec00.t00z.pgrb2s.0p25.f006`（tmax）
  - `gefs/20240101/subset_6bb2cf9d__gec00.t00z.pgrb2s.0p25.f006`（tmin）

## 四、离线校验脚本的校验逻辑

`scripts/verify_offline_gefs.py` 对每个 case：

1. `cfgrib.open_datasets(本地 GRIB, backend_kwargs={"indexpath": ""})` —— 只读本地文件，**不联网**（`indexpath=""` 让 cfgrib 忽略 Herbie 的 `.idx` 侧车文件）。
2. `GEFSFetcher.extract_region(ds, lat, lon)` —— 复用生产裁剪逻辑（含 0-360/±180 经度 wrapping）。
3. `xr.merge([tmax, tmin], compat="override")` —— 合并双变量。
4. 断言：变量存在 + `float32`、`valid_time` 长度 = 窗口数、`valid_time` 值 = 黄金值、网格 41×41、lat 边界 = 区域。

校验项设计说明：**`valid_time` 的长度**统一表达窗口数（reforecast 解码成 `step` 维度，
realtime 单消息 `step` 是标量坐标、不在 `sizes` 里），避免了对两种维度结构的硬编码分支。

## 五、测试过程中发现并修复的问题（重要）

1. **网格不一致**：realtime 原用 `product="atmos.5"`（0.5°，上海 21×21），与 reforecast
   0.25°（41×41）不对齐。已改 `atmos.25`（0.25°，41×41）。→ commit `5f99010`。

2. **R04 重试收窄回归**：R04 把重试从裸 `except Exception` 收窄成 `OSError`，但 Herbie
   把网络错误包装成 `RuntimeError`（`__cause__` 是 OSError），导致网络瞬时抖动**不再重试**。
   且下载中断会留下**半截子集文件**（实测抓到一个只含 1 条消息的 0.7MB 残文件，正常应
   3 条 2.2MB），被 Herbie `overwrite=False` 跳过。已修：`_is_retryable` 识别
   「cause 为 OSError 的 RuntimeError」+ `_download_with_retry` 下载失败删除半截文件。
   → commit `eab749e`。

3. **cfgrib 维度语义**：多窗口 → `step` 维度；单窗口 → `step`/`valid_time` 标量坐标。
   mock（`make_fake_forecast_ds` / `make_fake_realtime_ds`）已 mirror 这两种真实结构。

## 六、后续若需扩展离线校验

- 新增 case：跑 `python scripts/probe_gefs.py` 下载新日期 → 在 `verify_offline_gefs.py`
  的 `GOLDEN_CASES` 里补一条（含本地 GRIB 路径 + 观测到的黄金值）。
- 校验真实**批量下载器**（`GEFSBatchDownloader` 落盘的 `.nc`）：可复用
  `GEFSFetcher.calculate_md5` / `verify_file_md5` 对 `data/processed/...` 做 MD5 比对。
