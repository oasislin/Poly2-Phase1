#!/usr/bin/env python3
"""
run_predictions.py: Command-line prediction engine for Polymarket temperature markets (Ticket 3.5-02 / Issue #31).

Usage:
    python scripts/run_predictions.py --station ZSPD --date 2019-07-15 --type max --lead-time 30 --ens-mean 30.0 --ens-var 2.0 --current-temp 28.5 --obs-time 12:00
    python scripts/run_predictions.py --station KDEN --date 2019-01-15 --type min --lead-time 24 --ens-mean -5.0 --ens-var 2.0 --current-temp -4.0 --json
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from src.modeling.climatology import ClimatologyCalculator
from src.modeling.registry import ModelRegistry
from src.prediction.prediction_pipeline import PredictionPipeline, PredictionRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_predictions")

STATION_NAMES = {
    "ZSPD": "Shanghai Pudong International Airport",
    "KDEN": "Denver International Airport",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate calibrated probabilistic temperature forecasts for Polymarket markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--station", type=str, default="ZSPD", help="Weather station ID (e.g., ZSPD, KDEN)")
    parser.add_argument("--date", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="Target prediction date (YYYY-MM-DD)")
    parser.add_argument("--type", type=str, choices=["max", "min"], default="max", help="Target temperature type")
    parser.add_argument("--lead-time", type=float, default=None, help="Forecast lead time in hours (default: 30 for max, 24 for min)")
    parser.add_argument("--ens-mean", type=float, required=False, help="GEFS ensemble mean temperature (°C)")
    parser.add_argument("--ens-var", type=float, default=2.0, help="GEFS ensemble variance (°C²)")
    parser.add_argument("--current-temp", type=float, default=None, help="Real-time temperature observation (°C or °F depending on unit)")
    parser.add_argument("--obs-time", type=str, default=None, help="Observation time (e.g., 12:00 or HH:MM)")
    parser.add_argument("--delta-hours", type=float, default=None, help="Remaining hours until diurnal peak/valley")
    parser.add_argument("--models-dir", type=str, default="models/emos", help="Directory containing trained EMOS models")
    parser.add_argument("--db-path", type=str, default="data/db/predictions.db", help="SQLite database path for persistence")
    parser.add_argument("--save-db", dest="save_db", action="store_true", default=True, help="Save prediction record to SQLite database")
    parser.add_argument("--no-save-db", dest="save_db", action="store_false", help="Do not persist prediction to database")
    parser.add_argument("--json", dest="output_json", action="store_true", default=False, help="Output prediction result strictly as JSON to stdout")

    return parser


def _format_header(record: PredictionRecord) -> List[str]:
    """Format report header section."""
    st_name = STATION_NAMES.get(record.station_id.upper(), record.station_id)
    lead_status = "Interpolated" if record.is_interpolated else ("Decayed" if record.is_short_lead_decay else "Anchor Hit")
    return [
        "=" * 80,
        "📊 Polymarket Temperature Prediction Report (Phase 1C)",
        "=" * 80,
        f"Station:          {record.station_id} ({st_name})",
        f"Target Date:      {record.target_date} | Type: {record.target_type.upper()} | Season: {record.static_distribution.season}",
        f"Lead Time:        {record.lead_time_hours:.1f}h ({lead_status})",
        f"Issue Time:       {record.issue_time}",
        "-" * 80,
    ]


def _format_emos_and_bounds(record: PredictionRecord, unit: str) -> List[str]:
    """Format Gaussian EMOS forecast, dynamic observation, and physical bounds."""
    ci_90_low, ci_90_high = record.static_distribution.confidence_interval(0.90)
    lines = [
        "🌡️  Base Gaussian EMOS Forecast:",
        f"    μ (Mean):     {record.predicted_mu:.2f}°C",
        f"    σ (Std Dev):  {record.predicted_sigma:.2f}°C",
        f"    90% CI:       [{ci_90_low:.2f}°C, {ci_90_high:.2f}°C]",
        "-" * 80,
        "⏱️  Real-time Dynamic Truncation:",
    ]
    if record.current_temp is not None:
        obs_t = f" at {record.observation_time}" if record.observation_time else ""
        rule = f"Max temp guaranteed >= {record.current_temp:.2f}°{unit}" if record.target_type == "max" else f"Min temp guaranteed <= {record.current_temp:.2f}°{unit}"
        lines.append(f"    T_now:        {record.current_temp:.2f}°{unit}{obs_t}")
        lines.append(f"    Status:       Active Truncated ({rule})")
    else:
        lines.append("    Status:       No Observation (Prior Distribution Active)")

    lines.append("-" * 80)
    lines.append("🛡️  Physical Reachability Bounds:")
    if record.constrained_distribution.is_constrained:
        t_min = record.constrained_distribution.t_min_possible
        t_max = record.constrained_distribution.t_max_possible
        lines.append(f"    Reachable:    [{t_min:.2f}°C, {t_max:.2f}°C]")
        lines.append("    Status:       Hard Thermodynamic Envelope Enforced")
    else:
        lines.append("    Status:       Unconstrained")
    lines.append("-" * 80)
    return lines


def _format_bins_table(record: PredictionRecord) -> List[str]:
    """Format discrete Polymarket market bins probability table."""
    lines = [
        "🎯 Polymarket Discrete Bins Probability Table:",
        "+--------+------------+---------------+--------------------------------+",
        "| Bin ID | Type       | Label         | Probability                    |",
        "+--------+------------+---------------+--------------------------------+",
    ]
    for b in record.market_bins:
        pct = b.probability * 100.0
        bar_len = int(round(b.probability * 20))
        bar = "█" * bar_len + " " * (20 - bar_len)
        lines.append(f"| {b.bin_id:<6} | {b.bin_type:<10} | {b.label:<13} | {pct:>7.2f}%  [{bar}] |")

    lines.append("+--------+------------+---------------+--------------------------------+")
    total_p = sum(b.probability for b in record.market_bins) * 100.0
    lines.append(f"Total Probability Sum: {total_p:.4f}%")
    lines.append("=" * 80)
    return lines


def format_prediction_report(record: PredictionRecord) -> str:
    """Format rich terminal summary report for the prediction record."""
    unit = record.market_bins[0].unit if record.market_bins else "C"
    lines = _format_header(record)
    lines.extend(_format_emos_and_bounds(record, unit))
    lines.extend(_format_bins_table(record))
    return "\n".join(lines)


def run_cli(args: Optional[List[str]] = None) -> int:
    """Execute prediction CLI entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    lead_time = parsed.lead_time if parsed.lead_time is not None else (30.0 if parsed.type.lower() == "max" else 24.0)
    ens_mean = parsed.ens_mean if parsed.ens_mean is not None else (25.0 if parsed.station.upper() == "ZSPD" else 15.0)

    registry = ModelRegistry(base_dir=parsed.models_dir)
    pipeline = PredictionPipeline(
        model_registry=registry,
        db_path=parsed.db_path,
    )

    try:
        record = pipeline.predict_single(
            station_id=parsed.station,
            target_date=parsed.date,
            target_type=parsed.type,
            lead_time_hours=lead_time,
            ensemble_mean=ens_mean,
            ensemble_variance=parsed.ens_var,
            current_temp=parsed.current_temp,
            observation_time=parsed.obs_time,
            delta_hours=parsed.delta_hours,
            save_to_db=parsed.save_db,
        )

        if parsed.output_json:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            print(format_prediction_report(record))
        return 0

    except Exception as e:
        logger.error("Error executing prediction pipeline: %s", e, exc_info=True)
        return 1


def main() -> None:
    """Main CLI wrapper."""
    sys.exit(run_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
