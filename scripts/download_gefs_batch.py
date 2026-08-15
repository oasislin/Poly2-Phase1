#!/usr/bin/env python3
"""
CLI Batch Downloader for GEFS Reforecast Data (2000-2019).

Usage:
  python scripts/download_gefs_batch.py --start-year 2000 --end-year 2019
  python scripts/download_gefs_batch.py --start-year 2019 --end-year 2019 --auto-continue
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_acquisition.gefs_batch_downloader import GEFSBatchDownloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEFS Reforecast Batch Downloader with CSV State Machine"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="Start year to download (inclusive, default: 2000)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2019,
        help="End year to download (inclusive, default: 2019)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="data/gefs_download_state.csv",
        help="Path to CSV state machine tracking file",
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=str,
        default="data/raw/gefs",
        help="Directory to store raw GEFS files before moving",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/gefs",
        help="Directory to store cropped regional datasets",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default="shanghai,denver",
        help="Comma-separated station names (default: shanghai,denver)",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Auto-confirm raw file move without interactive pause",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=2.0,
        help="State file polling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    station_list = [s.strip() for s in args.stations.split(",") if s.strip()]

    downloader = GEFSBatchDownloader(
        state_file=args.state_file,
        raw_cache_dir=args.raw_cache_dir,
        processed_dir=args.processed_dir,
        stations=station_list,
        auto_continue=args.auto_continue,
        check_interval=args.check_interval,
        verbose=args.verbose,
    )

    downloader.run(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
