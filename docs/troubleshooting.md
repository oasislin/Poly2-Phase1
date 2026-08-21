# 🛠️ Troubleshooting & Operations Guide (排障与运维手册)

本手册汇总了在开发与验证阶段已实证的真实技术决策（ADRs 0001~0005）、网络/数据管道踩坑记录（Handoffs）以及数值算法防御机制，提供标准故障排查与应急处置 SOP。

---

## 目录
1. [数据采集与网络管道故障](#1-数据采集与网络管道故障)
2. [GRIB 合并与网格维度异常](#2-grib-合并与网格维度异常)
3. [EMOS 模型优化与两级降级触发](#3-emos-模型优化与两级降级触发)
4. [实时动态截断与物理约束边界保护](#4-实时动态截断与物理约束边界保护)
5. [三重验收门禁失败定位](#5-三重验收门禁失败定位)

---

## 1. 数据采集与网络管道故障

### 1.1 Herbie 依赖与 AWS S3 下载 404
- **现象**：调用 GEFS 下载时提示 `ModuleNotFoundError` 或 AWS S3 路径 404 错误。
- **根因分析**：
  1. PyPI 上存在无关包 `herbie`，正确依赖包名为 `herbie-data`。
  2. NOAA 历史重预报在 AWS 上的分片路径格式随年份存在历史变迁，Mock 测试全绿不代表真实网络通畅。
- **处置 SOP**：
  1. 验证虚拟环境中安装的是 `herbie-data`：
     ```bash
     pip show herbie-data
     ```
  2. 运行真实网络冒烟测试验证连通性：
     ```bash
     RUN_NETWORK_TESTS=1 pytest tests/unit/data_acquisition/test_gefs_fetcher.py::test_network_reforecast_single_message -v
     ```
  3. 若离线开发或网络受限，确保启用本地 GRIB2 缓存目录（`data/raw/gefs/`）。

### 1.2 成员数量不匹配错误（Requesting 11 or 31 Members）
- **现象**：`DataConfig` 校验失败或 GEFS 下载报错缺少成员文件。
- **根因分析**：根据 **ADR 0004** 与 AWS 历史实证，2000-2019 年重预报仅存储 5 个成员（`c00` 控制预报 + `p01`~`p04` 扰动成员）。
- **处置 SOP**：
  - 检查配置文件中的 `data.members`，确认配置严格为 `[0, 1, 2, 3, 4]`，不得扩展。

---

## 2. GRIB 合并与网格维度异常

### 2.1 `xr.merge` / `xr.concat` 坐标微差报错
- **现象**：合并 `tmax_2m` 与 `tmin_2m` 时提示 `MergeError: conflicting values for coordinates 'valid_time'`。
- **根因分析**：根据 **ADR 0005**，NOAA 历史重预报不同批次或变量的 GRIB2 消息中，`valid_time` 属性存在微秒级或元数据标签差异。
- **处置 SOP**：
  - 在所有涉及 GRIB 数据合并的代码中，必须显式指定：
    ```python
    xr.merge([ds_tmax, ds_tmin], compat="override", coords="minimal")
    ```

### 2.2 空间插值网格维度不匹配
- **现象**：双线性插值时提示上海区域网格 shape 不为 `(41, 41)`。
- **根因分析**：上海区域定义为经度 `[115, 125]E`、纬度 `[25, 35]N`，在 0.25° 分辨率下恰好对应 $41 \times 41$ 格点。代码中严禁执行任何二次截断规整。
- **处置 SOP**：
  - 检查 `data/raw/` 裁剪逻辑，确认原始区域边界保留完整 41×41 网格后再传入 `SpatialInterpolator`。

---

## 3. EMOS 模型优化与两级降级触发

### 3.1 两级降级触发（Level 1 高斯 EMOS $\to$ Level 2 气候学）
- **现象**：模型注册中心返回降级标志，或日志记录 `DegradationDecision: Level 2 Climatology`。
- **触发类型与排查**：
  1. **硬触发（Hard Trigger）**：
     - *优化器超迭代未收敛*：检查训练集样本量是否不足或存在全 NaN 记录。
     - *参数出现 NaN/Inf*：检查集合方差或目标实测是否存在缺失值。
  2. **软触发（Soft Trigger）**：
     - *验证集 CRPS 显著劣于气候学（$p < 0.05$）*：说明 EMOS 发生严重过拟合，已自动回退到 31 天滑动窗历史气候学均值与方差。
  3. **软告警（Soft Alert）**：
     - *参数 $|c| > 10$ 或 $|d| > 10$*：记录日志告警，不强制中断预测。
- **处置 SOP**：
  1. 查询 `data/models/` 下对应模型元数据中的 `diagnostics` 与 `decision` 字段。
  2. 确认 31 天滑动窗 $\sigma_{clim}^2(d)$ 是否正常加载（严格限制 2000-2018 年数据）。

### 3.2 集合方差坍塌（Zero Ensemble Spread）
- **现象**：5 成员预报完全一致（$S_{ens}^2 \to 0$），导致传统 EMOS 方差极小、置信区间过窄。
- **保护机制**：根据 **ADR 0001**，平方参数化公式强制叠加气候学方差底：
  $$\sigma^2 = c^2 + d^2 S_{ens}^2 + \sigma_{clim}^2(d)$$
  天然保证了最小方差下限 $\sigma^2 \ge \sigma_{clim}^2(d) > 0$。

---

## 4. 实时动态截断与物理约束边界保护

### 4.1 动态截断除零与后验概率溢出
- **现象**：实况温度极高或极低时，条件概率分母 $1 - F(T_{now}) \approx 0$。
- **防御机制**：
  - `DynamicCorrector` 强制设置安全余量 $\epsilon = 10^{-7}$：
    $$P(X \ge L \mid X \ge T_{now}) = \frac{\max(0, 1 - F(\max(L, T_{now})))}{\max(\epsilon, 1 - F(T_{now}))}$$
  - 当分母 $< \epsilon$ 时，自动无缝回退至基准未截断分布并输出调试日志。

### 4.2 物理变温率硬约束截断
- **现象**：盘口概率中某些区间的概率被强制置为 `0.0` 或 `1.0`。
- **排查说明**：
  - `ConstraintEnforcer` 依据两站历史实测最大升/降温速率（℃/h）计算剩余时间 $\Delta t$ 内机理不可达边界。此为最高优先级物理硬截断，属于预期内正常机理保护。

---

## 5. 三重验收门禁失败定位

执行 `python scripts/run_poly_pipeline.py backtest` 时，若三重门禁未通过：

| 门禁项 | 失败判定标准 | 排查方向与处置 |
| :--- | :--- | :--- |
| **门禁 1: 真实节点校准** | 真实节点（Max 6/30/54h, Min 24/48h）PIT KS 检验 $p \le 0.05$ | 检查是否存在系统性均值偏差（Bias），确认高程递减率（$\Gamma=0.0065$ K/m）是否已正确作用。 |
| **门禁 2: 留出节点插值** | 30h 留出虚拟模型 $\text{CRPS}_{virt} > 1.05 \times \text{CRPS}_{real}$ | 检查 6h 与 54h 两端锚点模型的拟合质量，排查线性内插权重计算。 |
| **门禁 3: 极端天气压力** | 2019 样本外极端天气 90% CI 覆盖率 $< 80\%$ 或 未战胜气候学 | 确认是否混入了训练期样本（严禁时序污染）；检查方差底 $\sigma_{clim}(d)$ 是否足够平滑。 |
