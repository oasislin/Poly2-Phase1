# Wunderground 历史数据爬取管道

## 项目概述
用于爬取Weather Underground历史气象数据的Python工具，支持上海（ZSPD）和丹佛（KDEN）两个站点，时间范围2000-2019年。

## 主要功能

### ✅ Task 1.1 完成情况
- [x] **完整的12个气象字段提取**
  - 温度（最高/最低）
  - 露点温度（最高/最低）
  - 湿度
  - 风速和风向
  - 气压
  - 降水量
  - 天气状况

- [x] **健壮的错误处理**
  - 403 Forbidden错误处理（30秒重试，10次连续失败停止）
  - 指数退避重试机制
  - 头部轮换避免检测
  - 智能缓存策略

- [x] **数据存储**
  - SQLite数据库持久化存储
  - CSV文件导出
  - 断点续传支持

- [x] **批量处理**
  - 批量下载控制器
  - 状态持久化（JSON文件）
  - 优雅的中断处理（Ctrl+C）

## 技术特性

### 增强的403处理
```python
# 可配置参数
scraper = WundergroundScraper(
    max_consecutive_403=10,      # 最大连续403错误次数
    forbidden_retry_delay=30,    # 403后重试延迟（秒）
    max_retries=5               # 最大重试次数
)
```

### 智能缓存策略
- ✅ 成功响应（status_code=200）：缓存7天
- ⚠️ 404错误：缓存1天
- ❌ 其他错误：缓存1小时
- 🔄 避免缓存污染：不缓存错误响应

### 批量下载功能
```bash
# 下载所有站点数据（2000-2019）
python scripts/download_wunderground_batch_enhanced.py --station all --start-year 2000 --end-year 2019 --verbose

# 下载单个站点
python scripts/download_wunderground_batch_enhanced.py --station ZSPD --start-year 2000 --end-year 2020
```

## 数据完整性

### 上海站（ZSPD）
- **总记录数**: 7,540条
- **时间范围**: 2000-01-01 到 2020-12-31
- **缺失月份**: 2个（2000-03, 2000-04）- 网站本身没有数据
- **数据不完整月份**: 5个（网站数据不完整）

### 丹佛站（KDEN）
- **总记录数**: 7,272条
- **时间范围**: 1999-12-31 到 2019-12-31
- **缺失月份**: 无
- **数据不完整月份**: 6个（96-97%完整）

## 文件结构
```
├── src/data_acquisition/
│   ├── wunderground_scraper.py      # 主爬虫类（包含403增强）
│   └── mock_wunderground.py         # 模拟数据生成器
├── scripts/
│   ├── download_wunderground_batch_enhanced.py  # 批量下载控制器
│   ├── download_wunderground_batch.py           # 基础批量下载
│   └── download_wunderground_data.py            # 单次下载
├── data/
│   ├── wunderground.db              # SQLite数据库
│   └── raw/wunderground/            # 原始CSV文件
├── tests/                           # 单元测试
└── specs/                           # 项目规范文档
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用
```python
from src.data_acquisition.wunderground_scraper import WundergroundScraper

# 初始化爬虫
scraper = WundergroundScraper()

# 下载单个月份数据
data = scraper.download_station_data("ZSPD", 2020, 1)

# 保存到数据库
scraper.save_to_database(data, "ZSPD")
```

### 批量下载
```python
# 使用批量控制器
from scripts.download_wunderground_batch_enhanced import main

# 下载所有站点
main(station="all", start_year=2000, end_year=2019)
```

## 403增强功能详情

查看 [403_ENHANCEMENTS_SUMMARY.md](403_ENHANCEMENTS_SUMMARY.md) 了解完整的403处理增强功能。

## 测试覆盖
- 单元测试：57个测试通过
- 集成测试：完整的数据管道测试
- 错误处理测试：403、网络错误、解析错误

## 许可证
MIT License

## 作者
Poly Way2 项目团队