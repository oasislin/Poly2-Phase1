#!/usr/bin/env python3
"""
Unit tests for BacktestReporter (Ticket 4.2-02 / Issue #37).

Verifies:
- Multi-slice aggregation across stations, target types, and lead hours
- Lead time error decay analysis table generation
- Multi-format exports (Markdown scorecard, structured JSON, CSV details)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.validation.backtester import BacktestResult
from src.validation.backtest_reporter import BacktestReporter
from src.validation.triple_gate import ForecastSlice


def make_fake_backtest_result(station_id: str, target_type: str, lead_hours: int, crps_mod: float) -> BacktestResult:
    n = 20
    df = pd.DataFrame({
        "target_date": pd.date_range("2019-01-01", periods=n, freq="D"),
        "station_id": [station_id] * n,
        "target_type": [target_type] * n,
        "lead_hours": [lead_hours] * n,
        "truth": np.random.uniform(10, 20, size=n),
        "mu_model": np.random.uniform(10, 20, size=n),
        "sigma_model": [1.5] * n,
        "crps_model": [crps_mod] * n,
        "crps_clim": [crps_mod * 1.5] * n,
        "crps_raw": [crps_mod * 1.2] * n,
        "crps_persistence": [crps_mod * 1.6] * n,
        "error_model": np.random.normal(0, 1.0, size=n),
    })

    return BacktestResult(
        slice_info=ForecastSlice(station_id=station_id, target_type=target_type, lead_hours=lead_hours),
        sample_count=n,
        mean_crps_model=crps_mod,
        mean_crps_raw=crps_mod * 1.2,
        mean_crps_clim=crps_mod * 1.5,
        mean_crps_persistence=crps_mod * 1.6,
        crpss_vs_raw=1.0 - (1.0 / 1.2),
        crpss_vs_clim=1.0 - (1.0 / 1.5),
        crpss_vs_persistence=1.0 - (1.0 / 1.6),
        mae_model=crps_mod * 1.2,
        mae_raw=crps_mod * 1.5,
        mae_clim=crps_mod * 1.8,
        mae_persistence=crps_mod * 1.9,
        coverage_90_ci_model=0.90,
        pit_values=np.random.uniform(0, 1, size=n),
        brier_score_discrete=0.08,
        df_daily=df,
    )


class TestBacktestReporter:
    """Tests for BacktestReporter functionality."""

    def test_aggregation_and_decay_table(self):
        results = [
            make_fake_backtest_result("ZSPD", "max", 6, 1.05),
            make_fake_backtest_result("ZSPD", "max", 30, 1.25),
            make_fake_backtest_result("ZSPD", "max", 54, 1.45),
            make_fake_backtest_result("KDEN", "max", 6, 1.10),
            make_fake_backtest_result("KDEN", "max", 30, 1.30),
            make_fake_backtest_result("KDEN", "max", 54, 1.55),
        ]

        reporter = BacktestReporter(results=results)
        decay_df = reporter.generate_lead_time_decay_table()

        assert isinstance(decay_df, pd.DataFrame)
        assert len(decay_df) == 6
        assert set(decay_df["lead_hours"]) == {6, 30, 54}
        assert "mean_crps_model" in decay_df.columns
        assert "crpss_vs_clim" in decay_df.columns

    def test_markdown_report_generation(self):
        results = [
            make_fake_backtest_result("ZSPD", "max", 30, 1.20),
            make_fake_backtest_result("KDEN", "min", 24, 1.35),
        ]

        reporter = BacktestReporter(results=results)
        md = reporter.generate_markdown_report()

        assert "# Historical Backtest Evaluation Scorecard" in md
        assert "ZSPD" in md
        assert "KDEN" in md
        assert "Lead Time Decay Analysis" in md
        assert "Overall System Summary Table" in md

    def test_file_exports(self, tmp_path: Path):
        results = [
            make_fake_backtest_result("ZSPD", "max", 30, 1.20),
            make_fake_backtest_result("KDEN", "max", 30, 1.30),
        ]

        reporter = BacktestReporter(results=results)

        md_path = tmp_path / "backtest_scorecard.md"
        json_path = tmp_path / "backtest_summary.json"
        csv_path = tmp_path / "backtest_daily.csv"

        reporter.export_markdown(md_path)
        reporter.export_json(json_path)
        reporter.export_csv_details(csv_path)

        assert md_path.exists()
        assert json_path.exists()
        assert csv_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "slices" in data
            assert len(data["slices"]) == 2
            assert "overall_mean_crps_model" in data

        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 40  # 20 + 20
        assert "crps_model" in df_csv.columns
