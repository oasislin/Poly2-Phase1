#!/usr/bin/env python3
"""
GEFS Batch Download and State Machine Manager (Task 1.2 T07).

Coordinates yearly chunk downloads of GEFS reforecast data across 2000-2019,
applies regional cropping immediately, and tracks state using a CSV state machine:
pending -> downloading -> downloaded -> cropped -> raw_ready -> (user_check=moved) -> done.
"""

import csv
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import xarray as xr

from src.data_acquisition.gefs_fetcher import (
    DEFAULT_REGIONS,
    GEFSFetcher,
    VALID_MEMBERS,
)

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["year", "download", "crop", "user_check", "note"]


@dataclass
class YearState:
    year: int
    download: str  # pending, in_progress, done, failed
    crop: str  # pending, in_progress, done, failed
    user_check: str  # empty or 'moved'
    note: str = ""

    def is_fully_done(self) -> bool:
        return self.download == "done" and self.crop == "done" and self.user_check == "moved"


class GEFSBatchDownloader:
    """Orchestrates yearly GEFS downloads with a CSV state machine."""

    def __init__(
        self,
        state_file: str = "data/gefs_download_state.csv",
        raw_cache_dir: str = "data/raw/gefs",
        processed_dir: str = "data/processed/gefs",
        stations: Optional[List[str]] = None,
        fetcher: Optional[GEFSFetcher] = None,
        auto_continue: bool = False,
        check_interval: float = 2.0,
        verbose: bool = False,
    ):
        self.state_file = Path(state_file)
        self.raw_cache_dir = Path(raw_cache_dir)
        self.processed_dir = Path(processed_dir)
        self.stations = stations or ["shanghai", "denver"]
        self.fetcher = fetcher or GEFSFetcher(cache_dir=str(self.raw_cache_dir), verbose=verbose)
        self.auto_continue = auto_continue
        self.check_interval = check_interval
        self.verbose = verbose

        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load_or_init_state(self, start_year: int, end_year: int) -> Dict[int, YearState]:
        """Load state from CSV file or initialize with pending states."""
        states = {}
        if self.state_file.exists():
            with open(self.state_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row or not row.get("year"):
                        continue
                    yr = int(row["year"])
                    states[yr] = YearState(
                        year=yr,
                        download=row.get("download", "pending"),
                        crop=row.get("crop", "pending"),
                        user_check=row.get("user_check", ""),
                        note=row.get("note", ""),
                    )

        # Fill missing years in the requested range
        for yr in range(start_year, end_year + 1):
            if yr not in states:
                # Auto-detect if processed NetCDF files already exist for all stations
                all_processed = all(
                    bool(list((self.processed_dir / str(yr) / s).glob("*.nc")))
                    for s in self.stations
                )
                if all_processed:
                    states[yr] = YearState(
                        year=yr,
                        download="done",
                        crop="done",
                        user_check="moved",
                        note="Existing processed data detected",
                    )
                else:
                    states[yr] = YearState(
                        year=yr,
                        download="pending",
                        crop="pending",
                        user_check="",
                        note="",
                    )

        self.save_state(states)
        return states

    def save_state(self, states: Dict[int, YearState]) -> None:
        """Persist states dictionary to CSV."""
        with open(self.state_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for yr in sorted(states.keys()):
                st = states[yr]
                writer.writerow(
                    {
                        "year": st.year,
                        "download": st.download,
                        "crop": st.crop,
                        "user_check": st.user_check,
                        "note": st.note,
                    }
                )

    def process_year(
        self,
        year: int,
        states: Dict[int, YearState],
        download_func: Optional[Callable] = None,
        crop_func: Optional[Callable] = None,
    ) -> bool:
        """Execute state machine transitions for a single year chunk."""
        st = states[year]

        if st.is_fully_done():
            logger.info(f"Year {year} is already completed. Skipping.")
            return True

        # Step 1: Download
        if st.download != "done":
            st.download = "in_progress"
            self.save_state(states)
            try:
                if download_func:
                    download_func(year)
                else:
                    self._default_download_year(year)
                st.download = "done"
                st.note = f"Downloaded on {datetime.now().isoformat()}"
                self.save_state(states)
            except Exception as exc:
                st.download = "failed"
                st.note = f"Download failed: {exc}"
                self.save_state(states)
                raise

        # Step 2: Crop
        if st.crop != "done":
            st.crop = "in_progress"
            self.save_state(states)
            try:
                if crop_func:
                    crop_func(year)
                else:
                    self._default_crop_year(year)
                st.crop = "done"
                st.note = f"Cropped on {datetime.now().isoformat()}"
                self.save_state(states)
            except Exception as exc:
                st.crop = "failed"
                st.note = f"Crop failed: {exc}"
                self.save_state(states)
                raise

        # Step 3: Human-in-the-loop signal and pause
        if st.user_check != "moved":
            signal_msg = (
                f"\n{'='*70}\n"
                f"[SIGNAL] Year {year} cropping completed!\n"
                f"Raw global files in '{self.raw_cache_dir}' can now be safely moved/archived.\n"
                f"Please mark user_check='moved' in '{self.state_file}' to proceed.\n"
                f"{'='*70}\n"
            )
            print(signal_msg)

            if self.auto_continue:
                logger.info(f"--auto-continue enabled: setting user_check='moved' for {year}")
                st.user_check = "moved"
                self.save_state(states)
            else:
                self._wait_for_user_moved(year, states)

        return True

    def _wait_for_user_moved(self, year: int, states: Dict[int, YearState]) -> None:
        """Poll the CSV state file until user marks user_check='moved' for the year."""
        logger.info(f"Waiting for user to set user_check='moved' for year {year}...")
        while True:
            time.sleep(self.check_interval)
            updated_states = self.load_or_init_state(year, year)
            if updated_states[year].user_check.strip().lower() == "moved":
                states[year].user_check = "moved"
                self.save_state(states)
                logger.info(f"User check confirmed for year {year}. Resuming batch pipeline.")
                break

    def _default_download_year(self, year: int) -> None:
        """Download a full local year of reforecast per station, applying 6h
        window selection, and stage the cropped data under raw_cache_dir.

        For each target local day D the init is the PREVIOUS day 00Z
        (reforecast is 00Z-only and D's 6h windows live in D-1 00Z's lead
        hours). This +1 offset widens the init range to [year-1 Dec 31,
        year Dec 30] so the whole local year is covered (one extra init on each
        boundary).

        Resume granularity (do NOT re-implement byte-level resume here):
        - across runs: the CSV state machine skips years already `download=done`;
        - within a year: a per-init shard is skipped if its `.nc` + `.nc.md5`
          sidecar exist and the recorded MD5 still verifies (corrupt shards are
          re-downloaded); otherwise GEFSFetcher memoizes per-(init, member,
          variable) and Herbie's own `save_dir` cache skips files it already has.
        """
        init_start = max(date(year - 1, 12, 31), date(2000, 1, 1))
        init_end = date(year, 12, 30)
        staging_dir = self.raw_cache_dir / "cropped" / str(year)
        total_days = (init_end - init_start).days + 1

        for station in self.stations:
            bounds = DEFAULT_REGIONS.get(station.lower(), DEFAULT_REGIONS["shanghai"])
            out_dir = staging_dir / station
            out_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"==> 开始下载 {station.upper()} {year} 年数据 (共 {total_days} 个时次)...")

            init_day = init_start
            completed_in_year = 0
            month_count = 0
            month_elapsed = 0.0
            current_month = init_start.month
            year_start_time = time.perf_counter()

            while init_day <= init_end:
                target_date = init_day + timedelta(days=1)
                out_path = out_dir / f"{init_day:%Y%m%d}.nc"
                md5_path = out_dir / f"{init_day:%Y%m%d}.nc.md5"

                t_day_start = time.perf_counter()
                skipped = False

                # Resume: a shard already on disk that passes its recorded MD5
                # is skipped (no re-download / re-decode / re-write).
                if out_path.exists() and md5_path.exists():
                    if GEFSFetcher.verify_file_md5(
                        out_path, md5_path.read_text().strip()
                    ):
                        skipped = True
                    else:
                        # corrupted / half-written -> delete and re-download
                        out_path.unlink(missing_ok=True)
                        md5_path.unlink(missing_ok=True)

                if not skipped:
                    windows = self.fetcher.select_contained_windows(
                        datetime(init_day.year, init_day.month, init_day.day, 0, 0),
                        target_date,
                        station,
                    )
                    if not windows:
                        logger.warning(
                            f"{station} {target_date}: no contained 6h windows, skipping init {init_day}"
                        )
                        init_day += timedelta(days=1)
                        continue

                    ds = self.fetcher.download_reforecast(
                        region_bounds=bounds,
                        date_range=(init_day, init_day),
                        members=list(VALID_MEMBERS),
                        cycles=[0],
                        forecast_hours=windows,
                    )
                    ds.to_netcdf(out_path, engine="scipy")
                    md5_path.write_text(GEFSFetcher.calculate_md5(out_path))

                t_day_elapsed = time.perf_counter() - t_day_start
                completed_in_year += 1
                month_count += 1
                month_elapsed += t_day_elapsed

                # 延迟预警与主动链路健康检查 (阈值 > 15s)
                if not skipped and t_day_elapsed > 15.0:
                    health_fn = getattr(self.fetcher, "check_link_health", GEFSFetcher.check_link_health)
                    health = health_fn()
                    diag_box = (
                        f"\n{'-'*75}\n"
                        f"⚠️  [延迟预警] {station.upper()} {target_date} 下载耗时达 {t_day_elapsed:.1f}s\n"
                        f"🔍 [链路诊断] NOAA AWS S3: {health['message']}\n"
                        f"{'-'*75}\n"
                    )
                    print(diag_box)

                # 判断是否为该月最后一天或全年代际结束
                next_day = init_day + timedelta(days=1)
                is_month_end = (next_day > init_end) or (next_day.month != current_month)

                if is_month_end and month_count > 0:
                    pct = (completed_in_year / total_days) * 100
                    avg_speed = month_elapsed / max(month_count, 1)
                    rem_days = total_days - completed_in_year
                    rem_minutes = (rem_days * avg_speed) / 60.0
                    health_fn = getattr(self.fetcher, "check_link_health", None)
                    if health_fn:
                        health = health_fn()
                        status_text = "正常" if health["healthy"] else "异常"
                        rtt_str = f"RTT {health['rtt_ms']}ms" if health["rtt_ms"] is not None else "未知"
                    else:
                        status_text = "正常"
                        rtt_str = "未探测"

                    summary_line = (
                        f"[{init_day.year}-{current_month:02d} 完成] "
                        f"{month_count:2d} 天 ({pct:5.1f}%) | "
                        f"平均: {avg_speed:4.1f}s/天 | "
                        f"链路: {status_text} ({rtt_str}) | "
                        f"预估剩余: {rem_minutes:4.1f} 分钟"
                    )
                    print(summary_line)

                    # 重置月度统计
                    current_month = next_day.month
                    month_count = 0
                    month_elapsed = 0.0

                init_day += timedelta(days=1)

            total_year_elapsed = time.perf_counter() - year_start_time
            print(
                f"✅ {station.upper()} {year} 年下载裁剪完成: 共 {completed_in_year} 天, "
                f"总耗时 {total_year_elapsed/60:.1f} 分钟 (平均 {total_year_elapsed/max(completed_in_year, 1):.2f}s/天)"
            )

    def _default_crop_year(self, year: int) -> None:
        """Persist staged cropped data into the processed tree, verify it
        round-trips through xarray, then delete the staged copy to free raw-side
        space. crop=done is only written after the data is on disk, re-openable,
        and the staged copy is cleaned up."""
        staging_dir = self.raw_cache_dir / "cropped" / str(year)
        for station in self.stations:
            src_dir = staging_dir / station
            dst_dir = self.processed_dir / str(year) / station
            dst_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(src_dir.glob("*.nc")) if src_dir.exists() else []
            if not files:
                # idempotent resume: already cropped (staged cleaned), skip
                if list(dst_dir.glob("*.nc")):
                    continue
                raise FileNotFoundError(
                    f"no staged cropped data for {station} {year} under {src_dir}"
                )
            for src in files:
                ds = xr.open_dataset(src, engine="scipy")
                dst = dst_dir / src.name
                ds.to_netcdf(dst, engine="scipy")
                # verify round-trip before counting this file as cropped
                _ = xr.open_dataset(dst, engine="scipy")
            logger.info(f"Cropped {station} {year}: {len(files)} files -> {dst_dir}")
            # free raw-side space: staged copy is now redundant
            shutil.rmtree(src_dir)
        if staging_dir.exists() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()

    def run(
        self,
        start_year: int = 2000,
        end_year: int = 2019,
        download_func: Optional[Callable] = None,
        crop_func: Optional[Callable] = None,
    ) -> None:
        """Run the batch download scheduler from start_year to end_year."""
        states = self.load_or_init_state(start_year, end_year)
        for yr in range(start_year, end_year + 1):
            logger.info(f"--- Processing chunk: Year {yr} ---")
            self.process_year(
                yr,
                states,
                download_func=download_func,
                crop_func=crop_func,
            )
        logger.info("All requested years completed successfully.")
