"""
Unit tests for GEFS Batch Downloader and State Machine (Task 1.2 T07).
"""

import csv
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_acquisition.gefs_batch_downloader import (
    CSV_COLUMNS,
    GEFSBatchDownloader,
    YearState,
)


def _make_staged_ds(init_day, step_hours):
    """Small cropped reforecast-style Dataset mirroring the real decode shape
    (time=reference/init, step=forecast windows, valid_time=window end)."""
    init = pd.Timestamp(init_day)
    steps = [pd.Timedelta(hours=h) for h in step_hours]
    lats = np.arange(25, 25.75, 0.25)  # 3 points
    lons = np.arange(115, 115.75, 0.25)
    rng = np.random.default_rng(0)
    base = rng.random((1, len(steps), len(lats), len(lons)))
    ds = xr.Dataset(
        {
            "tmax": (("time", "step", "latitude", "longitude"), base),
            "tmin": (("time", "step", "latitude", "longitude"), base.copy()),
        },
        coords={
            "time": [init],
            "step": steps,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds = ds.assign_coords(valid_time=("step", [init + s for s in steps]))
    return ds.expand_dims(member=[0, 1, 2, 3, 4])


class RecordingFetcher:
    """Fake GEFSFetcher that records window-selection calls and only returns
    windows for one target day (keeps the full-year loop fast in tests)."""

    def __init__(self, only_target=None, windows=(24, 30, 36)):
        self.select_calls = []
        self.download_calls = []
        self.only_target = only_target
        self.windows = windows

    def select_contained_windows(self, init_time, target_date, station, **kw):
        self.select_calls.append((init_time, target_date, station))
        if self.only_target is not None and target_date == self.only_target:
            return list(self.windows)
        return []

    def download_reforecast(self, **kwargs):
        self.download_calls.append(kwargs)
        return _make_staged_ds(kwargs["date_range"][0], kwargs["forecast_hours"])


class TestGEFSBatchDownloader:
    def test_state_initialization(self, tmp_path):
        state_file = tmp_path / "test_state.csv"
        downloader = GEFSBatchDownloader(state_file=str(state_file))

        states = downloader.load_or_init_state(2000, 2002)

        assert state_file.exists()
        assert len(states) == 3
        assert states[2000].download == "pending"
        assert states[2000].crop == "pending"
        assert states[2000].user_check == ""

        # Verify CSV format
        with open(state_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_COLUMNS
            rows = list(reader)
            assert len(rows) == 3
            assert rows[0]["year"] == "2000"

    def test_resume_preserves_existing_state(self, tmp_path):
        state_file = tmp_path / "test_state.csv"
        downloader = GEFSBatchDownloader(state_file=str(state_file))

        # Initialize with 2000 done, 2001 in progress
        downloader.save_state(
            {
                2000: YearState(2000, "done", "done", "moved", "finished"),
                2001: YearState(2001, "done", "in_progress", "", "midway"),
            }
        )

        # Load expanding to 2002
        states = downloader.load_or_init_state(2000, 2002)
        assert states[2000].is_fully_done() is True
        assert states[2001].crop == "in_progress"
        assert states[2002].download == "pending"

    def test_full_pipeline_with_auto_continue(self, tmp_path):
        state_file = tmp_path / "test_state.csv"
        raw_dir = tmp_path / "raw"
        proc_dir = tmp_path / "processed"

        downloader = GEFSBatchDownloader(
            state_file=str(state_file),
            raw_cache_dir=str(raw_dir),
            processed_dir=str(proc_dir),
            auto_continue=True,
        )

        downloads_called = []
        crops_called = []

        def mock_download(year):
            downloads_called.append(year)

        def mock_crop(year):
            crops_called.append(year)

        downloader.run(
            start_year=2000,
            end_year=2001,
            download_func=mock_download,
            crop_func=mock_crop,
        )

        assert downloads_called == [2000, 2001]
        assert crops_called == [2000, 2001]

        states = downloader.load_or_init_state(2000, 2001)
        assert states[2000].is_fully_done() is True
        assert states[2001].is_fully_done() is True

    def test_skips_already_completed_year(self, tmp_path):
        state_file = tmp_path / "test_state.csv"
        downloader = GEFSBatchDownloader(state_file=str(state_file), auto_continue=True)

        downloader.save_state(
            {2000: YearState(2000, "done", "done", "moved", "complete")}
        )

        downloads = []
        downloader.run(
            start_year=2000,
            end_year=2000,
            download_func=lambda y: downloads.append(y),
        )
        assert downloads == []

    def test_handles_download_failure_and_persists_error(self, tmp_path):
        state_file = tmp_path / "test_state.csv"
        downloader = GEFSBatchDownloader(state_file=str(state_file), auto_continue=True)

        def failing_download(year):
            raise ConnectionError("Network down")

        with pytest.raises(ConnectionError):
            downloader.run(
                start_year=2000,
                end_year=2000,
                download_func=failing_download,
            )

        states = downloader.load_or_init_state(2000, 2000)
        assert states[2000].download == "failed"
        assert "Network down" in states[2000].note


class TestDefaultDownloadCrop:
    def _downloader(self, tmp_path, fetcher, stations=("shanghai",)):
        return GEFSBatchDownloader(
            state_file=str(tmp_path / "state.csv"),
            raw_cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            stations=list(stations),
            fetcher=fetcher,
        )

    def test_download_applies_window_selection_and_stages_data(self, tmp_path):
        fetcher = RecordingFetcher(only_target=date(2019, 7, 2))
        downloader = self._downloader(tmp_path, fetcher)

        downloader._default_download_year(2019)

        # window selection was invoked with the +1 init->target offset
        assert fetcher.select_calls
        first_init, first_target, first_station = fetcher.select_calls[0]
        assert first_target == date(2019, 1, 1)
        assert first_init == datetime(2018, 12, 31, 0, 0)
        assert first_station == "shanghai"
        last_init, last_target, _ = fetcher.select_calls[-1]
        assert last_target == date(2019, 12, 31)
        assert last_init == datetime(2019, 12, 30, 0, 0)

        # forecast_hours is the window-selection output, not None
        assert len(fetcher.download_calls) == 1
        assert fetcher.download_calls[0]["forecast_hours"] == [24, 30, 36]

        # cropped data staged under raw_cache_dir, not a touch marker
        staged = tmp_path / "raw" / "cropped" / "2019" / "shanghai"
        files = list(staged.glob("*.nc"))
        assert len(files) == 1
        ds = xr.open_dataset(files[0], engine="scipy")
        assert "tmax" in ds.data_vars
        assert "tmin" in ds.data_vars
        assert ds.sizes["member"] == 5
        assert ds.sizes["step"] == 3

    def test_crop_persists_to_processed_and_roundtrips(self, tmp_path):
        fetcher = RecordingFetcher(only_target=date(2019, 7, 2))
        downloader = self._downloader(tmp_path, fetcher)

        downloader._default_download_year(2019)
        downloader._default_crop_year(2019)

        out_dir = tmp_path / "processed" / "2019" / "shanghai"
        files = list(out_dir.glob("*.nc"))
        assert len(files) == 1
        ds = xr.open_dataset(files[0], engine="scipy")
        assert "tmax" in ds.data_vars
        assert "tmin" in ds.data_vars
        assert ds.sizes["member"] == 5
        assert ds.sizes["step"] == 3

    def test_crop_raises_without_staged_data(self, tmp_path):
        fetcher = RecordingFetcher()  # selects nothing -> no staged files
        downloader = self._downloader(tmp_path, fetcher)

        with pytest.raises(FileNotFoundError):
            downloader._default_crop_year(2019)

    def test_download_resume_skips_verified_shards(self, tmp_path):
        fetcher = RecordingFetcher(only_target=date(2019, 7, 2))
        downloader = self._downloader(tmp_path, fetcher)

        downloader._default_download_year(2019)
        assert len(fetcher.download_calls) == 1

        staged = tmp_path / "raw" / "cropped" / "2019" / "shanghai"
        assert len(list(staged.glob("*.nc"))) == 1
        assert len(list(staged.glob("*.nc.md5"))) == 1

        # re-run: verified shard is skipped, no re-download
        downloader._default_download_year(2019)
        assert len(fetcher.download_calls) == 1

    def test_download_redownloads_on_md5_mismatch(self, tmp_path):
        fetcher = RecordingFetcher(only_target=date(2019, 7, 2))
        downloader = self._downloader(tmp_path, fetcher)

        downloader._default_download_year(2019)
        staged = tmp_path / "raw" / "cropped" / "2019" / "shanghai"
        nc = list(staged.glob("*.nc"))[0]
        # corrupt the .nc but keep the .md5 sidecar
        nc.write_bytes(b"corrupted")

        downloader._default_download_year(2019)
        # corrupted shard re-downloaded and restored to a valid netCDF
        assert len(fetcher.download_calls) == 2
        xr.open_dataset(nc, engine="scipy")

    def test_crop_cleans_staged_data(self, tmp_path):
        fetcher = RecordingFetcher(only_target=date(2019, 7, 2))
        downloader = self._downloader(tmp_path, fetcher)

        downloader._default_download_year(2019)
        staged_station = tmp_path / "raw" / "cropped" / "2019" / "shanghai"
        assert staged_station.exists()

        downloader._default_crop_year(2019)
        # staged copy cleaned, processed preserved + readable
        assert not staged_station.exists()
        out_dir = tmp_path / "processed" / "2019" / "shanghai"
        files = list(out_dir.glob("*.nc"))
        assert len(files) == 1
        ds = xr.open_dataset(files[0], engine="scipy")
        assert "tmax" in ds.data_vars

    def test_crop_idempotent_rerun(self, tmp_path):
        fetcher = RecordingFetcher(only_target=date(2019, 7, 2))
        downloader = self._downloader(tmp_path, fetcher)

        downloader._default_download_year(2019)
        downloader._default_crop_year(2019)
        # second crop run: staged gone but processed present -> skip, no error
        downloader._default_crop_year(2019)

        out_dir = tmp_path / "processed" / "2019" / "shanghai"
        assert len(list(out_dir.glob("*.nc"))) == 1
