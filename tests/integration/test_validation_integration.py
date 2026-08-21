#!/usr/bin/env python3
"""
Integration test for Phase 1D Validation System (Ticket 4.4-01 / Issue #39).

Verifies the entire validation workflow:
- Historical dataset slicing across multiple stations and lead hours
- Backtesting execution vs Climatology, Raw GEFS, and Persistence baselines
- Anomaly monitoring and alert dispatching
- v5.9.2 Triple Acceptance Gates evaluation
- Diagnostic Markdown, JSON, and CSV report exports
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.validation.alert_dispatcher import AlertDispatcher, FileAlertChannel
from src.validation.alert_manager import AlertManager, AlertType
from src.validation.backtest_reporter import BacktestReporter
from src.validation.backtester import BacktestEngine
from src.validation.triple_gate import TripleGateEvaluator


def make_synthetic_backtest_dataset(n_days: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")
    mu = 15.0 + 5.0 * np.sin(np.linspace(0, 3.14, n_days))
    sigma = 1.5
    truths = np.random.normal(mu, sigma)
    ens_means = mu + np.random.normal(0, 0.2, size=n_days)
    ens_vars = np.full(n_days, sigma ** 2)

    return pd.DataFrame({
        "target_date": dates,
        "truth": truths,
        "ensemble_mean": ens_means,
        "ensemble_variance": ens_vars,
        "clim_mean": np.full(n_days, 15.0),
        "clim_sigma": np.full(n_days, 4.0),
        "yesterday_truth": np.roll(truths, 1),
    })


def test_end_to_end_validation_integration(tmp_path: Path):
    # 1. Prepare datasets and mock predictor
    df_zspd = make_synthetic_backtest_dataset(200)
    df_kden = make_synthetic_backtest_dataset(200)

    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = lambda **kwargs: (
        kwargs["ensemble_mean"],
        1.5,
    )

    # 2. Run Backtest Engine across slices
    engine = BacktestEngine()
    r1 = engine.run_backtest("ZSPD", "max", 30, df_zspd, mock_predictor)
    r2 = engine.run_backtest("KDEN", "max", 30, df_kden, mock_predictor)

    assert r1.sample_count == 200
    assert r2.sample_count == 200
    assert r1.mean_crps_model < r1.mean_crps_clim

    # 3. Setup Alert System & Audit file
    alert_log_path = tmp_path / "alerts.jsonl"
    file_channel = FileAlertChannel(alert_log_path)
    dispatcher = AlertDispatcher(channels=[file_channel])
    alert_manager = AlertManager()

    # Check degradation
    alert = alert_manager.check_crps_degradation(
        "ZSPD", "max", 30, crps_current=r1.mean_crps_model * 1.5, crps_baseline=r1.mean_crps_model
    )
    if alert:
        dispatcher.dispatch(alert)

    assert alert_log_path.exists()
    alerts_content = alert_log_path.read_text(encoding="utf-8")
    assert "CRPS_DEGRADATION" in alerts_content

    # 4. Triple Acceptance Gates
    gate_evaluator = TripleGateEvaluator()
    gate_report = gate_evaluator.generate_triple_gate_report(
        standard_pit_values=r1.pit_values,
        crps_virt_30h=r1.mean_crps_model * 1.02,
        crps_real_30h=r1.mean_crps_model,
        interp_pit_values=r1.pit_values,
        extreme_coverage_90=0.88,
        crps_model_extreme=r1.mean_crps_model,
        crps_clim_extreme=r1.mean_crps_clim,
    )

    assert gate_report.overall_passed is True

    # 5. Reporter and Export Validation
    reporter = BacktestReporter(results=[r1, r2])
    md_file = tmp_path / "report.md"
    json_file = tmp_path / "report.json"
    csv_file = tmp_path / "details.csv"

    reporter.export_markdown(md_file)
    reporter.export_json(json_file)
    reporter.export_csv_details(csv_file)

    assert md_file.exists()
    assert json_file.exists()
    assert csv_file.exists()
    assert len(pd.read_csv(csv_file)) == 400
