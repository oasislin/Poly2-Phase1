# ⚙️ Configuration Guide (配置手册)

本项目采用分层 YAML 驱动的强类型配置系统（`src/pipeline/config.py` 与 `ConfigManager`），支持环境继承（`base` $\to$ `dev`/`prod`/`test`）、环境变量注入（`POLY_*`）及敏感密钥脱敏。

---

## 1. 配置文件层级与加载机制

配置加载按以下优先级依次覆盖（后者覆盖前者）：
1. **基础默认值**：`PipelineConfig` 数据结构内置默认值。
2. **配置文件**：`configs/default.yaml`（或通过 `--config` 指定的 YAML 路径）。
3. **环境覆盖文件**：根据 `--env <env>` 自动读取 `configs/<env>.yaml`。
4. **运行时显式参数**：CLI 传入的命令行参数（如 `--station`, `--date`）。
5. **系统环境变量**：以 `POLY_` 开头的环境变量（如 `POLY_ENV=prod`, `POLY_MODEL_L2_REG=0.005`）。

---

## 2. 核心配置字段详解

### `data` 数据管道配置

| 字段名 | 类型 | 默认值 | 说明与约束 |
| :--- | :--- | :--- | :--- |
| `stations` | `List[str]` | `["ZSPD", "KDEN"]` | 目标预测站点列表。Phase 1 仅支持上海浦东（`ZSPD`）与丹佛国际（`KDEN`）。 |
| `members` | `List[int]` | `[0, 1, 2, 3, 4]` | **5 成员协议（ADR 0004）**：必须为 `[0, 1, 2, 3, 4]`（对应 c00 与 p01~p04）。严禁改为 11 或 31 成员。 |
| `raw_dir` | `str` | `data/raw` | 原始 GRIB2 与 Wunderground 爬虫缓存存储目录。 |
| `processed_dir` | `str` | `data/processed` | 经过时间对齐与空间插值后的 Parquet 特征库目录。 |
| `models_dir` | `str` | `data/models` | 40 组 EMOS 模型权重 Pickle 序列化存储目录。 |
| `db_dir` | `str` | `data/db` | SQLite 数据库存储目录。 |
| `predictions_db_path` | `str` | `data/db/predictions.db` | 盘口概率与推理记录存储的 SQLite 文件路径。 |
| `train_years` | `Tuple[int, int]` | `[2000, 2018]` | 模型训练与历史气候学 Floor 计算年份区间（严格时间墙）。 |
| `val_years` | `Tuple[int, int]` | `[2019, 2019]` | 样本外评估与三重验收门禁年份区间（禁止参与训练）。 |

---

### `model` EMOS 模型与训练矩阵配置

| 字段名 | 类型 | 默认值 | 说明与约束 |
| :--- | :--- | :--- | :--- |
| `max_lead_times` | `List[int]` | `[6, 30, 54]` | 最高温离散真实训练节点（小时）。30h 为插值检验关键留出节点。 |
| `min_lead_times` | `List[int]` | `[24, 48]` | 最低温离散真实训练节点（小时）。$<24\text{h}$ 时效使用物理方差衰减。 |
| `l2_reg` | `float` | `0.001` | EMOS 优化中作用于集合方差系数 $d$ 的 L2 正则化惩罚项 $\lambda$。 |
| `climatology_window_days`| `int` | `31` | 气候学均值与方差底的滑动窗口天数（目标日 $\pm 15$ 天）。 |
| `optimizer` | `str` | `L-BFGS-B` | 优化算法（`scipy.optimize.minimize`）。 |
| `variance_floor_type` | `str` | `climatology` | 方差底机制：采用站点历史 2000-2018 实测方差曲线 $\sigma_{clim}^2(d)$。 |

---

### `prediction` 多层推理与盘口转换配置

| 字段名 | 类型 | 默认值 | 说明与约束 |
| :--- | :--- | :--- | :--- |
| `dynamic_correction_enabled` | `bool` | `true` | 是否启用实况观测条件概率截断（Stage 2）。 |
| `physical_constraints_enabled` | `bool` | `true` | 是否启用历史极限变温率不可达物理硬约束（Stage 3）。 |
| `bin_width` | `float` | `1.0` | 默认离散盘口区间宽度（摄氏度/华氏度）。 |
| `tolerance_epsilon` | `float` | `1.0e-7` | 动态截断中除零与边界溢出保护的浮点数安全余量 $\epsilon$。 |

---

### `validation` 三重门禁与回测配置

| 字段名 | 类型 | 默认值 | 说明与约束 |
| :--- | :--- | :--- | :--- |
| `triple_gate_enabled` | `bool` | `true` | 是否启用 Phase 1 三重验收门禁评测。 |
| `alpha_significance` | `float` | `0.05` | PIT KS 检验与双边显著性检验的统计显著性水平 $\alpha$。 |
| `virtual_crps_loss_threshold` | `float` | `1.05` | 30h 留出插值精度损失阈值：$\text{CRPS}_{virt} \le 1.05 \times \text{CRPS}_{real}$。 |
| `extreme_ci_target` | `float` | `0.90` | 极端天气评测的目标置信区间（90% CI）。 |
| `extreme_ci_coverage_min` | `float` | `0.80` | 2019 样本外极端天气 90% CI 的最低允许覆盖率（$\ge 80\%$）。 |

---

### `alert` 监控告警配置

| 字段名 | 类型 | 默认值 | 说明与约束 |
| :--- | :--- | :--- | :--- |
| `crps_degradation_threshold` | `float` | `0.20` | CRPS 相对历史基准劣化触发告警的百分比阈值（$>20\%$）。 |
| `enabled_channels` | `List[str]` | `["console"]` | 启用的告警通道（`console`, `file`, `webhook`）。 |
| `webhook_url` | `str | None` | `null` | 告警 Webhook 推送地址（在日志中自动执行掩码脱敏）。 |

---

## 3. 配置文件范例 (`configs/default.yaml`)

```yaml
env: default

data:
  stations:
    - ZSPD
    - KDEN
  members:
    - 0
    - 1
    - 2
    - 3
    - 4
  raw_dir: data/raw
  processed_dir: data/processed
  models_dir: data/models
  db_dir: data/db
  predictions_db_path: data/db/predictions.db
  train_years: [2000, 2018]
  val_years: [2019, 2019]

model:
  max_lead_times: [6, 30, 54]
  min_lead_times: [24, 48]
  l2_reg: 0.001
  climatology_window_days: 31
  optimizer: L-BFGS-B

prediction:
  dynamic_correction_enabled: true
  physical_constraints_enabled: true
  bin_width: 1.0
  tolerance_epsilon: 1.0e-7

validation:
  triple_gate_enabled: true
  alpha_significance: 0.05
  virtual_crps_loss_threshold: 1.05
  extreme_ci_target: 0.90
  extreme_ci_coverage_min: 0.80

alert:
  crps_degradation_threshold: 0.20
  enabled_channels: ["console"]
  webhook_url: null
```
