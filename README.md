# Polymarket 温度预测系统 - Phase 1

## 项目概述

构建一个高精度的物理概率模型，用于预测Polymarket温度市场的最高和最低温度概率分布。系统使用免费的公共数据源（Wunderground历史数据、GEFS预报数据、实时观测数据），并在进入实时交易前使用历史Wunderground数据进行验证。

## 核心目标

- **数据源**: Wunderground（地面真实数据）、GEFS（预报数据）、实时观测
- **目标城市**: 上海（ZSPD）和丹佛（KDEN）
- **时间范围**: 历史验证（2000-2019）
- **模型类型**: 偏态高斯EMOS模型，按季节分桶
- **输出**: 温度阈值的概率分布
- **排除**: 市场微观结构、流动性分析、交易执行

## 系统架构

### 三层预测系统
```
┌─────────────────────────────────────────────────────────────┐
│                    物理约束层                              │
│  基于历史数据的最大变温率约束                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    动态修正层                              │
│  基于当前温度的动态概率截断                                │
│  P(X ≥ L | X > T_now) for max, P(X ≤ L | X < T_now) for min │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    静态基础模型层                          │
│  基于GEFS集合预报训练的偏态高斯EMOS模型                    │
│  最高/最低温度和DJF/MAM/JJA/SON季节的独立模型              │
└─────────────────────────────────────────────────────────────┘
```

### 数据流
```
Wunderground（历史） ───┐
                       ├─→ 时间对齐 → 单位转换 → 特征工程
GEFS（预报） ───────────┘
                       │
实时观测 ──────────────┘
                       │
               模型训练（EMOS）
                       │
               静态预测（μ, σ, 偏度）
                       │
               动态修正（每小时更新）
                       │
               物理约束（变温率限制）
                       │
               分箱概率转换（Polymarket）
                       │
               验证与监控（CRPS, PIT, Talagrand）
```

## Phase 1 完成状态

### ✅ Task 1.1: Wunderground数据管道（已完成）
- **完整的12个气象字段提取**：温度、露点、湿度、风速风向、气压、降水量、天气状况
- **健壮的错误处理**：403 Forbidden错误处理（30秒重试，10次连续失败停止）
- **数据存储**：SQLite数据库持久化存储，CSV导出，断点续传
- **批量处理**：批量下载控制器，状态持久化，优雅中断处理

### 🔄 Task 1.2: GEFS数据管道（进行中）
- **数据源**: AWS Open Data via Herbie库
- **类型**: 回算预报（历史）和实时预报
- **变量**: 2米温度（t2m）集合成员
- **分辨率**: 0.5° × 0.5° 网格，6小时间隔

### 📋 Task 1.3: 数据处理基础（待开始）
- **时间对齐**: UTC到本地时间转换
- **单位转换**: 温度单位标准化（摄氏度）
- **空间插值**: 双线性插值
- **高程校正**: 标准递减率
- **特征提取**: 集合统计量

### 🗄️ Task 1.4: 数据存储系统（待开始）
- **目录结构**: `data/raw/`, `data/processed/`, `data/models/`
- **文件格式**: Parquet格式存储处理后的特征
- **数据库**: SQLite存储预测和指标
- **数据版本化**: 时间戳版本控制

## 当前进展

### 已完成
1. **Wunderground数据采集系统**（Task 1.1）
   - 完整的12字段提取
   - 增强的403错误处理
   - SQLite数据库存储
   - 批量下载控制器
   - 57个单元测试通过

2. **数据完整性验证**
   - 上海站（ZSPD）：7,540条记录（2000-2020）
   - 丹佛站（KDEN）：7,272条记录（1999-2019）
   - 缺失数据确认：网站本身数据不完整

### 技术特性

#### 增强的403处理
```python
scraper = WundergroundScraper(
    max_consecutive_403=10,      # 最大连续403错误次数
    forbidden_retry_delay=30,    # 403后重试延迟（秒）
    max_retries=5               # 最大重试次数
)
```

#### 智能缓存策略
- ✅ 成功响应（status_code=200）：缓存7天
- ⚠️ 404错误：缓存1天
- ❌ 其他错误：缓存1小时
- 🔄 避免缓存污染：不缓存错误响应

## 文件结构

```
Poly2-Phase1/
├── src/                          # 源代码
│   └── data_acquisition/        # 数据采集模块
│       ├── wunderground_scraper.py      # Wunderground爬虫（含403增强）
│       └── mock_wunderground.py         # 模拟数据生成器
├── scripts/                      # 工具脚本
│   ├── download_wunderground_batch_enhanced.py  # 批量下载控制器
│   ├── download_wunderground_batch.py           # 基础批量下载
│   └── download_wunderground_data.py            # 单次下载
├── specs/                        # 项目规范
│   ├── phase1-specification-complete.md         # Phase 1完整规范
│   ├── polymarket-temperature-prediction-phase1.md  # 温度预测系统规范
│   ├── project-structure-and-modules.md         # 项目结构和模块
│   ├── implementation-tasks-phase1.md           # 实施任务
│   ├── testing-strategy-phase1.md               # 测试策略
│   └── tdd-implementation-plan.md               # TDD实施计划
├── docs/                         # 文档
│   ├── system-architecture.md    # 系统架构
│   ├── data-flow.md              # 数据流图
│   ├── implementation-roadmap.md # 实施路线图
│   └── wunderground-integration-tasks.md  # Wunderground集成任务
├── tests/                        # 测试
│   └── unit/data_acquisition/    # 数据采集单元测试
├── data/                         # 数据目录
│   ├── raw/wunderground/         # 原始Wunderground数据
│   └── wunderground.db           # SQLite数据库
├── 403_ENHANCEMENTS_SUMMARY.md   # 403增强功能总结
├── README.md                     # 项目说明（本文件）
├── CONTEXT.md                    # 项目上下文
├── DEVELOPER_GUIDELINES.md       # 开发指南
├── README_TASK_1_1.md            # Task 1.1详细说明
├── TEST_SUMMARY.md               # 测试总结
└── requirements.txt              # Python依赖
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### Wunderground数据采集
```bash
# 下载所有站点数据（2000-2019）
python scripts/download_wunderground_batch_enhanced.py --station all --start-year 2000 --end-year 2019 --verbose

# 下载单个站点
python scripts/download_wunderground_batch_enhanced.py --station ZSPD --start-year 2000 --end-year 2020
```

### Python API
```python
from src.data_acquisition.wunderground_scraper import WundergroundScraper

# 初始化爬虫
scraper = WundergroundScraper(
    max_consecutive_403=10,
    forbidden_retry_delay=30,
    max_retries=5
)

# 下载单个月份数据
data = scraper.download_station_data("ZSPD", 2020, 1)

# 保存到数据库
scraper.save_to_database(data, "ZSPD")
```

## 成功标准

### Phase 1 验证指标
1. **校准性**: PIT直方图均匀分布
2. **准确性**: CRPS显著优于基准模型（GEFS均值、气候学）
3. **可靠性**: 系统为>95%的请求时间生成预测
4. **性能**: 每小时更新在5分钟内完成
5. **可用性**: 清晰的文档和验证报告

### 模型性能要求
- **CRPS改进**: 比GEFS集合均值提高至少20%
- **校准误差**: PIT直方图的χ²检验p值>0.05
- **覆盖概率**: 95%预测区间实际覆盖率为93-97%
- **季节性表现**: 所有季节（DJF、MAM、JJA、SON）表现一致

## 后续步骤

### Phase 1 剩余任务
1. **Task 1.2**: 实现GEFS数据管道
2. **Task 1.3**: 实现核心数据处理工具
3. **Task 1.4**: 建立数据存储系统
4. **Task 2.1**: 实现偏态高斯分布模型
5. **Task 2.2**: 实现EMOS校准
6. **Task 2.3**: 实现动态修正层
7. **Task 2.4**: 实现物理约束层
8. **Task 3.1**: 建立验证框架
9. **Task 3.2**: 建立监控系统

### Phase 2 规划
- 实时数据集成
- 交易信号生成
- 风险管理
- 回测框架

## 技术栈

- **数据采集**: Requests, BeautifulSoup4, SQLite
- **数据处理**: Pandas, NumPy, xarray
- **机器学习**: Scikit-learn, SciPy
- **可视化**: Matplotlib, Seaborn
- **测试**: Pytest, unittest
- **工作流**: DVC（数据版本控制）

## 许可证

MIT License

## 作者

Poly Way2 项目团队

## 详细规范

查看 [specs/](specs/) 目录获取完整项目规范文档。