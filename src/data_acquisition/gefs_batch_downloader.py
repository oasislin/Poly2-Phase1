#!/usr/bin/env python3
"""
GEFS Batch Download and State Machine Manager (Task 1.2 T07).

Coordinates yearly chunk downloads of GEFS reforecast data across 2000-2019,
applies regional cropping immediately, and tracks state using a CSV state machine:
pending -> downloading -> downloaded -> cropped -> raw_ready -> (user_check=moved) -> done.
"""

import csv
import logging
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
        - within a year: GEFSFetcher memoizes per-(init, member, variable) and
          Herbie's own `save_dir` cache skips files it already has.
        """
        init_start = date(year - 1, 12, 31)
        init_end = date(year, 12, 30)
        staging_dir = self.raw_cache_dir / "cropped" / str(year)

        for station in self.stations:
            bounds = DEFAULT_REGIONS.get(station.lower(), DEFAULT_REGIONS["shanghai"])
            out_dir = staging_dir / station
            out_dir.mkdir(parents=True, exist_ok=True)

            init_day = init_start
            while init_day <= init_end:
                target_date = init_day + timedelta(days=1)
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
                out_path = out_dir / f"{init_day:%Y%m%d}.nc"
                ds.to_netcdf(out_path, engine="scipy")
                init_day += timedelta(days=1)

    def _default_crop_year(self, year: int) -> None:
        """Persist staged cropped data into the processed tree and verify it
        round-trips through xarray. crop=done is only written after the data is
        actually on disk and re-openable."""
        staging_dir = self.raw_cache_dir / "cropped" / str(year)
        for station in self.stations:
            src_dir = staging_dir / station
            dst_dir = self.processed_dir / str(year) / station
            dst_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(src_dir.glob("*.nc"))
            if not files:
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
