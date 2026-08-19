#!/usr/bin/env python3
"""
End-to-end Pipeline: GEFS Reforecast Data Ingestion, NetCDF Cropping,
Feature Extraction, and Parquet Persistence with Alignment Verification.

Usage:
  python scripts/run_pipeline_year.py --year 2019 --auto-continue
  python scripts/run_pipeline_year.py --start-year 2000 --end-year 2019 --auto-continue
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import xarray as xr

from src.data_acquisition.gefs_batch_downloader import GEFSBatchDownloader
from src.data_acquisition.gefs_fetcher import GEFSFetcher
from src.data_processing.data_processor import DataProcessor
from src.data_processing.storage_manager import StorageManager

logger = logging.getLogger(__name__)

STATION_MAP = {
    "shanghai": "ZSPD",
    "denver": "KDEN",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end GEFS Reforecast Data Ingestion and Feature Pipeline"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Single target year to process (e.g. 2019)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2019,
        help="Start year (default: 2019)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2019,
        help="End year (default: 2019)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default="shanghai,denver",
        help="Comma-separated station names (default: shanghai,denver)",
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default="data/raw/gefs",
        help="Raw cache directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/gefs",
        help="Cropped NetCDF processed directory",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed/features",
        help="Parquet feature store root directory",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        default=True,
        help="Auto-continue across download stages",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and only process existing cropped NetCDF files into features",
    )
    return parser.parse_args()


def process_year_features(
    year: int,
    stations: List[str],
    processed_dir: Path,
    processor: DataProcessor,
    storage: StorageManager,
) -> None:
    """Extract features from cropped NetCDF grids for a given year and persist to Parquet."""
    print(f"\n⚙️  [特征提取与落盘] 开始加工 {year} 年特征库...")
    t0 = time.perf_counter()

    for station in stations:
        station_id = STATION_MAP.get(station.lower(), station.upper())
        st_dir = processed_dir / str(year) / station
        nc_files = sorted(st_dir.glob("*.nc")) if st_dir.exists() else []

        if not nc_files:
            logger.warning(f"No NetCDF files found for {station_id} {year} under {st_dir}")
            continue

        feature_records = []
        for nc_file in nc_files:
            try:
                # File name format: YYYYMMDD.nc where YYYYMMDD is the init date
                init_date_str = nc_file.stem
                init_date = datetime.strptime(init_date_str, "%Y%m%d").date()
                target_date = init_date + timedelta(days=1)
                init_time_utc = datetime(init_date.year, init_date.month, init_date.day, 0, 0)

                with xr.open_dataset(nc_file, engine="scipy") as ds:
                    # 1. Max temperature feature (lead_time_bucket=30)
                    df_max = processor.process_forecast_to_features(
                        dataset=ds,
                        station_id=station_id,
                        target_date=target_date,
                        init_time_utc=init_time_utc,
                        target_type="max",
                        lead_time_bucket=30,
                    )
                    feature_records.append(df_max)

                    # 2. Min temperature feature (lead_time_bucket=24)
                    df_min = processor.process_forecast_to_features(
                        dataset=ds,
                        station_id=station_id,
                        target_date=target_date,
                        init_time_utc=init_time_utc,
                        target_type="min",
                        lead_time_bucket=24,
                    )
                    feature_records.append(df_min)
            except Exception as e:
                logger.error(f"Error extracting features from {nc_file}: {e}")

        if feature_records:
            all_feats = pd.concat(feature_records, ignore_index=True)
            saved_count = storage.save_forecast_features(all_feats, deduplicate=True)
            print(
                f"  ✨ [{station_id}] {year} 年特征写入完成: {len(nc_files)} 天网格 -> {saved_count} 条特征记录"
            )

    elapsed = time.perf_counter() - t0
    print(f"✅ {year} 年特征加工与落盘完成，耗时 {elapsed:.1f} 秒。")


def verify_year_alignment(year: int, stations: List[str], storage: StorageManager) -> None:
    """Verify that features and SQLite observation labels align properly."""
    print(f"\n🔍 [对齐检验] 校验 {year} 年训练集对齐情况...")
    for station in stations:
        station_id = STATION_MAP.get(station.lower(), station.upper())
        # Check max
        train_max = storage.load_training_dataset(
            station_id=station_id,
            target_type="max",
            lead_time_bucket=30,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
        )
        # Check min
        train_min = storage.load_training_dataset(
            station_id=station_id,
            target_type="min",
            lead_time_bucket=24,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
        )

        max_cnt = len(train_max) if not train_max.empty else 0
        min_cnt = len(train_min) if not train_min.empty else 0
        print(
            f"  📊 [{station_id}] 最高温对齐样本: {max_cnt} 条 | 最低温对齐样本: {min_cnt} 条"
        )
        if max_cnt > 0:
            sample = train_max.iloc[0]
            print(
                f"     样本示例 ({sample['target_date']}): ensemble_mean={sample['ensemble_mean']:.2f}°C, "
                f"ensemble_variance={sample['ensemble_variance']:.2f}, observed={sample['observed_temp']:.2f}°C"
            )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start_yr = args.year if args.year is not None else args.start_year
    end_yr = args.year if args.year is not None else args.end_year
    station_list = [s.strip() for s in args.stations.split(",") if s.strip()]

    print("=" * 75)
    print(f"🚀 NOAA GEFS Reforecast 自动化数据下载与特征加工流水线")
    print(f"   年份区间: {start_yr} ~ {end_yr}")
    print(f"   目标站点: {', '.join(station_list)}")
    print("=" * 75)

    fetcher = GEFSFetcher(cache_dir=args.raw_cache_dir, max_retries=5, verbose=False)
    downloader = GEFSBatchDownloader(
        state_file="data/gefs_download_state.csv",
        raw_cache_dir=args.raw_cache_dir,
        processed_dir=args.processed_dir,
        stations=station_list,
        fetcher=fetcher,
        auto_continue=args.auto_continue,
    )
    processor = DataProcessor()
    storage = StorageManager()

    # Step 1: Health check before start
    health = fetcher.check_link_health()
    print(f"🌐 [启动前链路检查] NOAA AWS S3: {health['message']}")
    if not health["healthy"]:
        print(f"❌ 数据链路连接异常，终止运行以防盲目重试。详情: {health}")
        sys.exit(1)

    for yr in range(start_yr, end_yr + 1):
        print(f"\n{'='*75}\n🌟 开始处理 {yr} 年度数据\n{'='*75}")
        if not args.skip_download:
            downloader.process_year(yr, downloader.load_or_init_state(yr, yr))

        # Extract features and save to Parquet
        process_year_features(
            year=yr,
            stations=station_list,
            processed_dir=Path(args.processed_dir),
            processor=processor,
            storage=storage,
        )

        # Verify alignment
        verify_year_alignment(yr, station_list, storage)

    print("\n" + "=" * 75)
    print("🎉 所有指定年份下载与特征库构建全部顺利完成！")
    health_status = storage.verify_storage_health()
    print(f"📦 特征库总记录数: {health_status['total_cached_records']} 条")
    print("=" * 75)


if __name__ == "__main__":
    main()
