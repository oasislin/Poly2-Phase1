#!/usr/bin/env python3
"""Unified CLI entrypoint for Polymarket Temperature Prediction Pipeline (Ticket #44)."""

import argparse
import sys
from typing import Any, Dict, List, Optional

from src.pipeline.config import ConfigManager, PipelineConfig
from src.pipeline.health import HealthChecker, HealthStatus
from src.pipeline.main_pipeline import MainPipeline, PipelineStage
from src.utils.logger import get_logger, setup_logger


def _create_common_parser() -> argparse.ArgumentParser:
    """Build parser for shared options."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    common.add_argument("--env", type=str, default=None, help="Target environment overlay (e.g. dev, prod, test)")
    common.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    common.add_argument("--json-logs", action="store_true", help="Emit structured logs in JSON format")
    return common


def _add_subcommands(subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Attach subcommands to parent parser."""
    subparsers.add_parser("health", parents=[common], help="Run system pre-flight and health diagnostics")

    p_all = subparsers.add_parser("all", parents=[common], help="Execute full end-to-end pipeline")
    p_all.add_argument("--resume-from", type=str, choices=[s.value for s in PipelineStage], default=None, help="Resume stage")
    p_all.add_argument("--date", type=str, default=None, help="Target date for prediction (YYYY-MM-DD)")

    subparsers.add_parser("ingest", parents=[common], help="Run Stage 1: Data Ingestion (Wunderground & GEFS)")
    subparsers.add_parser("feature", parents=[common], help="Run Stage 2: Feature Engineering")
    subparsers.add_parser("train", parents=[common], help="Run Stage 3: Train 40 EMOS Model Matrix")

    p_pred = subparsers.add_parser("predict", parents=[common], help="Run Stage 4: Multi-layer Prediction & Market Bins")
    p_pred.add_argument("--station", type=str, choices=["ZSPD", "KDEN"], default=None, help="Target station")
    p_pred.add_argument("--date", type=str, default=None, help="Prediction date (YYYY-MM-DD)")

    p_bt = subparsers.add_parser("backtest", parents=[common], help="Run Stage 5: Validation & Triple Gate Backtesting")
    p_bt.add_argument("--start-year", type=int, default=2018, help="Backtest start year")
    p_bt.add_argument("--end-year", type=int, default=2019, help="Backtest end year")


def build_parser() -> argparse.ArgumentParser:
    """Build unified argument parser with multi-stage subcommands and inherited common options."""
    common = _create_common_parser()
    parser = argparse.ArgumentParser(
        description="Polymarket Temperature Prediction System - Unified Pipeline CLI",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_subcommands(subparsers, common)
    return parser


def _handle_health(config: PipelineConfig) -> int:
    """Execute health command."""
    logger = get_logger("poly.cli")
    checker = HealthChecker(config)
    report = checker.run_all_checks()
    status = report["overall_status"]
    logger.info(f"System Health Status: [{status}]")
    for comp, res in report["components"].items():
        logger.info(f" - {comp}: {res.get('status')}")
    return 0 if status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED) else 1


def _handle_all(pipeline: MainPipeline, args: argparse.Namespace) -> int:
    """Execute all stages."""
    resume_stage = PipelineStage(args.resume_from) if args.resume_from else None
    res = pipeline.run_all(resume_from=resume_stage, date_str=getattr(args, "date", None))
    if res.markdown_report:
        print("\n" + res.markdown_report + "\n")
    return 0 if res.success else 1


def _handle_stage(pipeline: MainPipeline, args: argparse.Namespace) -> int:
    """Execute a single stage command."""
    logger = get_logger("poly.cli")
    stage_map = {
        "ingest": PipelineStage.INGEST,
        "feature": PipelineStage.FEATURE,
        "train": PipelineStage.TRAIN,
        "predict": PipelineStage.PREDICT,
        "backtest": PipelineStage.VALIDATE,
    }
    target_stage = stage_map.get(args.command)
    if not target_stage:
        logger.error(f"Unknown command: {args.command}")
        return 1

    kwargs: Dict[str, Any] = {}
    for attr in ["date", "target_date", "station", "start_year", "end_year"]:
        if hasattr(args, attr) and getattr(args, attr) is not None:
            key = "target_date" if attr == "date" else attr
            kwargs[key] = getattr(args, attr)

    res_dict = pipeline.run_stage(target_stage, **kwargs)
    logger.info(f"Stage [{args.command}] finished with status: {res_dict.get('status')}")
    return 0 if res_dict.get("status") == "SUCCESS" else 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI execution dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    setup_logger(level=args.log_level, json_format=args.json_logs)
    logger = get_logger("poly.cli")
    config = ConfigManager.load(config_path=args.config, env=args.env)
    logger.info(f"Loaded configuration for env=[{config.env}]")

    if args.command == "health":
        return _handle_health(config)

    pipeline = MainPipeline(config)
    if args.command == "all":
        return _handle_all(pipeline, args)

    return _handle_stage(pipeline, args)


if __name__ == "__main__":
    sys.exit(main())
