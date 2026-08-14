---
created: 2026-08-14 16:31:12
updated: 2026-08-14 16:32:50
---

# 项目执行文件（Phase 1：高精度物理概率模型构建）v5.6
## 版本记录
- **v5.6**：**[核心修正]** 引入 **“时效分层训练协议”**，废弃单一模型架构，改为“季节 $\times$ 时效”二维矩阵建模。修正 Lead Time 计算逻辑，确立全时效衰减验收标准。
- **v5.5**：引入数据量优化策略。
- **v5.4**：引入双重验收标准与时间墙隔离。
---
## 1. 核心算法修正：时效分层训练协议
### 1.1 物理与统计事实
GEFS 预报误差特征随 Lead Time 呈显著非线性变化：
- **长时效（如 48h）**：集合离散度 $\sigma_{ens}$ 大，均值存在系统性平滑，EMOS 参数需大幅修正方差。
- **短时效（如 12h）**：集合离散度小，确定性高，EMOS 参数更侧重均值校准。
**结论**：严禁使用单一静态模型处理不同时效特征。
### 1.2 建模架构：季节 $\times$ 时效二维矩阵
采用 **方案 A：时效分桶 EMOS（Lead-time Specific EMOS）**。
**模型矩阵定义**：
$$ M_{s, \Delta t} $$
其中 $s$ 为季节（DJF, MAM, JJA, SON），$\Delta t$ 为剩余时效（48h, 42h, ..., 6h）。
**模型文件命名规范**：
```text
{StationID}_{Season}_lead{Hours}h.pkl
例：ZSPD_JJA_lead48h.pkl, ZSPD_JJA_lead24h.pkl
```
共计：$4 \text{ (季节)} \times 8 \text{ (时效节点)} = 32$ 个独立模型文件/站点。
---
## 2. 数据提取与时间计算逻辑
### 2.1 Lead Time 标准计算公式
**严格遵循气象定义**：
$$ \text{Lead Time} = \text{Valid Time (目标极值时间)} - \text{Init Time (模式起报时间)} $$
### 2.2 特征提取修正（以上海 D-day 最高温为例）
**目标**：预测 D-day 14:00（当地时间）的最高温分布。
| 更新时间点 (UTC) | 起报时间 | 目标时间 (LT) | Lead Time ($\Delta t$) | 使用的模型 |
| :--- | :--- | :--- | :--- | :--- |
| D-2 00Z | D-2 00Z | D-day 06Z (14:00 LT) | **48h** | `lead48h.pkl` |
| D-2 06Z | D-2 06Z | D-day 06Z (14:00 LT) | **42h** | `lead42h.pkl` |
| D-1 00Z | D-1 00Z | D-day 06Z (14:00 LT) | **24h** | `lead24h.pkl` |
| D-1 12Z | D-1 12Z | D-day 06Z (14:00 LT) | **12h** | `lead12h.pkl` |
**特征提取代码逻辑修正**：
```python
def get_model_and_features(gefs_data, init_time_utc, target_time_local):
    # 1. 物理修正：插值 + 高程调整 (v5.2协议)
    features = extract_physical_features(gefs_data, init_time_utc)
    
    # 2. 计算精确 Lead Time
    lead_time_hours = (target_time_local - init_time_utc).total_seconds() / 3600
    
    # 3. 离散化到标准桶 (48, 42, 36...)
    lead_time_bucket = round_to_nearest_6h(lead_time_hours)
    
    # 4. 加载对应的专用模型
    season = get_season(target_time_local)
    model = load_model(f"ZSPD_{season}_lead{lead_time_bucket}h.pkl")
    
    return model.predict(features)
```
---
## 3. 验证引擎升级：全时效衰减验收
### 3.1 验收标准更新
废弃单一时效验收，改为**CRPS 衰减曲线验收**。
**验收逻辑**：
1. 分别计算 Lead Time 48h, 36h, 24h, 12h 的 CRPS 均值。
2. **断言检查**：CRPS 必须随 Lead Time 缩短而**单调下降**。
   - 若出现 `CRPS_36h < CRPS_24h`，则判定模型训练失败。
3. **PIT 均匀性**：每个时效分桶必须独立通过 PIT K-S 检验（P-value > 0.05）。
### 3.2 验证集划分
- **训练集**：2000-2018年（用于拟合各时效 EMOS 参数）。
- **验证集**：2019年（用于绘制 CRPS 衰减曲线）。
---
## 4. 量化交易策略联动
基于新架构，确立动态仓位管理策略：
| 预测阶段 | Lead Time | 模型特征 | 交易策略 |
| :--- | :--- | :--- | :--- |
| **早期埋伏** | 48h - 36h | 方差 $\sigma$ 大，分布宽 | **寻找赔率极高的尾部风险**：轻仓埋伏盘口定价错误的“极端天气”选项。 |
| **中期调整** | 24h - 18h | 方差收敛，趋势确立 | **动态对冲**：根据新旧模型输出分布的差异，调整仓位方向。 |
| **临盘重击** | 12h - 06h | 方差 $\sigma$ 极小，确定性高 | **Kelly 准则重仓**：确定性优势最大时，利用高置信度分布重仓出击。 |
---
## 5. 总结
v5.6 版本是对模型底层逻辑的彻底重构：
1. **从“一把尺子”改为“一套精密量具”**：针对不同时效的物理特性，独立训练校准参数。
2. **修复时间计算Bug**：杜绝 Lead Time 定义混淆，确保特征与模型严格对齐。
3. **验证驱动开发**：通过全时效衰减验收，强制模型在物理规律上符合“预报越准、方差越小”的基本假设。
这确保了我们的系统在 D-2 天具备防守能力（捕捉长尾机会），在 D-day 具备进攻能力（精准打击盘口偏差）。

