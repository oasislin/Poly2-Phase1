# TDD Implementation Plan - Clean Version

> **对齐 v5.9 执行规格（2026-08-15）**：GEFS 测试沿用 6h 窗口 TMAX/TMIN + 5 成员口径；特征测试 {mean, variance, member_max, member_min}（弃分位数与时间特征）；模型测试改为高斯 EMOS + 方差 Floor + 两级降级。

## Week 1-2: Data Infrastructure (TDD Focus)

### Day 1-2: Wunderground Scraper

#### Test 1.1: Basic Data Fetching
```python
def test_fetch_shanghai_data_returns_temperature_records():
    """Given Shanghai station and valid date, returns temperature data"""
    scraper = WundergroundScraper()
    data = scraper.fetch_station_data('ZSPD', date(2023, 7, 1), date(2023, 7, 31))
    assert len(data) > 0
    assert 'date' in data[0]
    assert 'temp_max' in data[0]
    assert 'temp_min' in data[0]
```

#### Test 1.2: Error Handling
```python
def test_fetch_data_handles_invalid_date_range():
    """When end date before start date, raises ValueError"""
    scraper = WundergroundScraper()
    with pytest.raises(ValueError):
        scraper.fetch_station_data('ZSPD', date(2023, 7, 31), date(2023, 7, 1))
```

#### Test 1.3: Data Parsing
```python
def test_parse_html_extracts_temperature_values():
    """HTML with temperature data parsed correctly"""
    scraper = WundergroundScraper()
    html = """<table><tr><td>31</td><td>35°</td><td>25°</td></tr></table>"""
    data = scraper.parse_daily_data(html, 'ZSPD')
    assert data[0]['day'] == 31
    assert data[0]['temp_max'] == 35.0
    assert data[0]['temp_min'] == 25.0
```

### Day 3-4: GEFS Data Fetcher

#### Test 2.1: Regional Data Download（v5.9 对齐）
```python
def test_download_gefs_returns_xarray_dataset():
    """GEFS fetcher returns xarray Dataset with expected variables（6h 窗口 TMAX/TMIN，5 成员）"""
    fetcher = GEFSFetcher()
    dataset = fetcher.download_reforecast(
        region_bounds={'lat': (25, 35), 'lon': (115, 125)},
        date_range=(date(2023, 1, 1), date(2023, 1, 2)),
        members=[0, 1, 2, 3, 4]   # c00 + p01-p04 共 5 成员
    )
    assert isinstance(dataset, xr.Dataset)
    assert 'tmax' in dataset.variables   # 6h 窗口 TMAX（非 t2m）
    assert 'tmin' in dataset.variables   # 6h 窗口 TMIN
    assert 'latitude' in dataset.coords
    assert 'longitude' in dataset.coords
    assert dataset.sizes['member'] == 5
```

#### Test 2.2: Region Extraction
```python
def test_extract_region_crops_to_bounds():
    """Extracted region contains only data within bounds"""
    fetcher = GEFSFetcher()
    full_dataset = create_mock_gefs_dataset(lat_range=(0, 50), lon_range=(100, 150))
    region_dataset = fetcher.extract_region(
        full_dataset,
        lat_range=(25, 35),
        lon_range=(115, 125)
    )
    assert region_dataset.latitude.min() >= 25
    assert region_dataset.latitude.max() <= 35
    assert region_dataset.longitude.min() >= 115
    assert region_dataset.longitude.max() <= 125
```

### Day 5: Time Alignment

#### Test 3.1: UTC to Local Conversion
```python
def test_utc_to_local_conversion():
    """UTC times converted to correct local time with DST"""
    aligner = TimeAligner()
    utc_time = datetime(2023, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    # Shanghai (UTC+8, no DST)
    shanghai_time = aligner.utc_to_local(utc_time, 'Asia/Shanghai')
    assert shanghai_time.hour == 20  # 12 UTC = 20 Shanghai
    
    # Denver (UTC-6 with DST in July)
    denver_time = aligner.utc_to_local(utc_time, 'America/Denver')
    assert denver_time.hour == 6  # 12 UTC = 6 Denver (MDT)
```

## Week 2: Data Processing & Feature Engineering

### Day 6-7: Unit Conversion and Quality Control

#### Test 4.1: Temperature Unit Conversion
```python
def test_fahrenheit_to_celsius_conversion():
    """Fahrenheit temperatures converted to Celsius correctly"""
    converter = UnitConverter()
    
    # Freezing point
    assert converter.fahrenheit_to_celsius(32.0) == 0.0
    
    # Boiling point
    assert converter.fahrenheit_to_celsius(212.0) == 100.0
    
    # Denver data (Fahrenheit) converted
    denver_temp_c = converter.convert_temperature(95.0, 'F', 'C')
    assert abs(denver_temp_c - 35.0) < 0.1
    
    # Shanghai data (Celsius) unchanged
    shanghai_temp_c = converter.convert_temperature(35.0, 'C', 'C')
    assert shanghai_temp_c == 35.0
```

#### Test 4.2: Data Quality Validation
```python
def test_quality_control_flags_impossible_temperatures():
    """Temperatures outside physical range flagged as invalid"""
    validator = QualityControl()
    data = pd.DataFrame({
        'temp_max': [50.0, 60.0, -30.0],  # 60°C is impossible
        'temp_min': [25.0, 20.0, -40.0]
    })
    result = validator.validate(data, station_id='ZSPD')
    assert not result['is_valid']
    assert 'impossible_temperature' in result['flags'][1]
```

### Day 8-9: Feature Extraction

#### Test 5.1: Ensemble Statistics Calculation（v5.9 对齐）
```python
def test_ensemble_statistics_calculation():
    """集合统计 = {mean, variance, member_max, member_min}（5 成员，弃分位数）"""
    extractor = FeatureExtractor()
    ensemble_data = xr.DataArray(
        np.random.randn(5, 10, 10),  # 5 ensemble members (c00+p01-p04)
        dims=['member', 'lat', 'lon']
    )
    stats = extractor.calculate_ensemble_stats(ensemble_data)
    
    assert 'ensemble_mean' in stats
    assert 'ensemble_variance' in stats
    assert 'member_max' in stats
    assert 'member_min' in stats
    assert 'ensemble_p10' not in stats      # 弃分位数
    assert 'day_of_year_sin' not in stats   # 弃时间特征
```

> **已废弃**：`extract_temporal_features`（Q18 = A，不引入时间特征；季节分桶已覆盖季节信号）。

#### Test 5.2: 日极值窗口特征（v5.7 新增，替代原时间特征测试）
```python
def test_daily_extreme_from_fully_contained_windows():
    """日极值 = 完全包含（⊆ 本地日）的 6h TMAX/TMIN 窗口极值"""
    extractor = FeatureExtractor()
    # 上海本地日 [(D-1)16Z, D 16Z]：完全包含窗口 = 3 个（[18Z,00Z],[00Z,06Z],[06Z,12Z]）
    daily_max = extractor.daily_extreme(tmax_windows, local_day, 'max')
    daily_min = extractor.daily_extreme(tmin_windows, local_day, 'min')
    assert 'window' not in daily_max.dims     # 窗口维度已折叠
    assert daily_max == tmax_windows.sel(windows ⊆ local_day).max('window')
```

## Current Status

我们已经完成了：
1. ✅ **需求分析** - 理解了完整的Polymarket温度预测系统需求
2. ✅ **规范创建** - 创建了完整的Phase 1规范文档
3. ✅ **架构设计** - 设计了模块化系统架构
4. ✅ **TDD计划** - 制定了详细的测试驱动开发计划
5. ✅ **任务分解** - 将项目分解为可管理的9周任务

## 下一步行动

根据TDD原则，我们应该从**第一个可测试的行为**开始：

### 立即开始的任务：
1. **创建项目结构**：按照`specs/project-structure-and-modules.md`创建目录
2. **编写第一个测试**：Test 1.1 - Wunderground数据获取
3. **实现最小代码**：让第一个测试通过
4. **重构**：改进代码结构
5. **重复**：继续下一个测试

### 具体步骤：
```bash
# 1. 创建项目结构
mkdir -p src/{data_acquisition,data_processing,modeling,prediction,validation,utils,pipeline}
mkdir -p tests/{unit,integration,e2e}
mkdir -p data/{raw,processed,models}
mkdir -p configs notebooks docs

# 2. 创建第一个测试文件
touch tests/unit/data_acquisition/test_wunderground_scraper.py

# 3. 编写第一个测试（Test 1.1）
# 4. 运行测试（应该失败 - RED阶段）
# 5. 实现最小代码让测试通过（GREEN阶段）
# 6. 重构代码（REFACTOR阶段）
```

你想让我开始实施第一个TDD循环吗？还是你想先讨论其他方面？