#!/usr/bin/env python3
"""
CLI entry point for Phase 1B Gaussian EMOS Matrix Training & Triple Acceptance Verification.

Usage:
    python scripts/train_emos_matrix.py --stations ZSPD KDEN --train-start-year 2000 --train-end-year 2018 --val-start-year 2019 --val-end-year 2019
"""

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_processing.storage_manager import StorageManager
from src.modeling.climatology import ClimatologyCalculator
from src.modeling.pipeline import TrainingPipeline
from src.modeling.registry import ModelRegistry


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Gaussian EMOS Matrix across stations, seasons, and lead times with Triple Acceptance Gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=["ZSPD", "KDEN"],
        help="List of weather station identifiers (e.g. ZSPD KDEN)",
    )
    parser.add_argument(
        "--train-start-year",
        type=int,
        default=2000,
        help="Start year of training data period",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2018,
        help="End year of training data period (strict time wall boundary)",
    )
    parser.add_argument(
        "--val-start-year",
        type=int,
        default=2019,
        help="Start year of validation holdout period",
    )
    parser.add_argument(
        "--val-end-year",
        type=int,
        default=2019,
        help="End year of validation holdout period",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/wunderground.db",
        help="Path to Wunderground SQLite observation database",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/processed/features",
        help="Directory containing processed Parquet forecast features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/emos",
        help="Directory to persist trained EMOS models",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save validation acceptance reports",
    )
    parser.add_argument(
        "--l2-lambda-d",
        type=float,
        default=1e-3,
        help="L2 regularization coefficient for parameter d",
    )
    parser.add_argument(
        "--verify-gates",
        action="store_true",
        default=True,
        help="Strictly verify Triple Acceptance Gates",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("train_emos_matrix")
    logger.info("Initializing StorageManager and Registry...")

    storage = StorageManager(db_path=args.db_path, feature_dir=args.features_dir)
    clim_calc = ClimatologyCalculator(
        database=storage.db,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
    )
    registry = ModelRegistry(base_dir=args.output_dir)

    pipeline = TrainingPipeline(
        storage_manager=storage,
        climatology_calculator=clim_calc,
        model_registry=registry,
        stations=args.stations,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        val_start_year=args.val_start_year,
        val_end_year=args.val_end_year,
        l2_lambda_d=args.l2_lambda_d,
        report_dir=args.report_dir,
        verify_gates=args.verify_gates,
    )

    try:
        result = pipeline.run()
        print("\n" + result.acceptance_report.to_markdown() + "\n")

        if args.verify_gates and not result.acceptance_report.overall_passed:
            logger.error("❌ Triple Acceptance Gates verification FAILED!")
            return 1

        logger.info("✅ Phase 1B Training Pipeline completed successfully.")
        return 0

    except Exception as e:
        logger.exception("Pipeline execution failed with unhandled error: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
