#!/usr/bin/env python3
"""
Minimal-cost real-data probe for GEFS (Task 1.2).

Verifies the per-day download logic across the key scenarios WITHOUT pulling a
full year: each case downloads a SINGLE member, a SINGLE init day (reforecast)
or a SINGLE fxx (realtime), prints the shape / windows / valid_time / grid.

Usage:
  python scripts/probe_gefs.py
"""

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_acquisition.gefs_fetcher import DEFAULT_REGIONS, GEFSFetcher


def _show(name, ds, windows=None):
    print(f"\n=== {name} ===")
    if windows is not None:
        print(f"  selected windows (fxx): {windows}")
    print(f"  data_vars: {list(ds.data_vars)}")
    print(f"  sizes: {dict(ds.sizes)}")
    if "valid_time" in ds.coords:
        vt = ds.valid_time.values
        print(f"  valid_time: {vt if vt.ndim == 0 else list(vt)}")
    if "latitude" in ds.coords:
        print(
            f"  lat {float(ds.latitude.min()):.2f}..{float(ds.latitude.max()):.2f}"
            f"  lon {float(ds.longitude.min()):.2f}..{float(ds.longitude.max()):.2f}"
        )


def main():
    fetcher = GEFSFetcher(cache_dir="data/raw/gefs_probe", verbose=False)

    cases = [
        ("shanghai reforecast (summer)", "shanghai", date(2019, 7, 1), date(2019, 7, 2)),
        ("denver reforecast (summer/DST)", "denver", date(2019, 7, 1), date(2019, 7, 2)),
        ("denver reforecast (winter)", "denver", date(2019, 1, 1), date(2019, 1, 2)),
    ]
    for name, station, init_day, target_day in cases:
        windows = fetcher.select_contained_windows(
            datetime(init_day.year, init_day.month, init_day.day, 0, 0),
            target_day,
            station,
        )
        ds = fetcher.download_reforecast(
            region_bounds=DEFAULT_REGIONS[station],
            date_range=(init_day, init_day),
            members=[0],  # single member -> 1/5 the download
            cycles=[0],
            forecast_hours=windows,
        )
        _show(name, ds, windows)

    ds = fetcher.download_realtime(
        region_bounds=DEFAULT_REGIONS["shanghai"],
        forecast_time=datetime(2024, 1, 1, 0, 0),
        members=[0],
        fxx_hours=[6],
    )
    _show("shanghai realtime f006 (atmos.25)", ds)


if __name__ == "__main__":
    main()
