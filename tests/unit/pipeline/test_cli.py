"""Unit tests for Unified CLI interface scripts/run_poly_pipeline.py (Ticket #44)."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from scripts.run_poly_pipeline import build_parser, main


class TestPipelineCLI:
    """Test CLI argument parsing and subcommand dispatch."""

    def test_build_parser_subcommands(self):
        parser = build_parser()
        
        # Test 'all' subcommand
        args_all = parser.parse_args(["all", "--env", "dev", "--resume-from", "train"])
        assert args_all.command == "all"
        assert args_all.env == "dev"
        assert args_all.resume_from == "train"

        # Test 'health' subcommand
        args_health = parser.parse_args(["health"])
        assert args_health.command == "health"

        # Test 'predict' subcommand
        args_pred = parser.parse_args(["predict", "--station", "ZSPD", "--date", "2026-08-21"])
        assert args_pred.command == "predict"
        assert args_pred.station == "ZSPD"
        assert args_pred.date == "2026-08-21"

    @patch("scripts.run_poly_pipeline.HealthChecker")
    def test_main_health_command(self, mock_health_checker_cls, tmp_path):
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = {
            "overall_status": "HEALTHY",
            "components": {
                "storage": {"status": "HEALTHY"},
                "database": {"status": "HEALTHY"},
                "models": {"status": "HEALTHY"},
            },
        }
        mock_health_checker_cls.return_value = mock_checker

        exit_code = main(["health", "--config", "configs/default.yaml"])
        assert exit_code == 0
        assert mock_checker.run_all_checks.called

    @patch("scripts.run_poly_pipeline.MainPipeline")
    def test_main_all_command(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run_all.return_value = MagicMock(success=True, markdown_report="Profile Report")
        mock_pipeline_cls.return_value = mock_pipeline

        exit_code = main(["all", "--env", "dev"])
        assert exit_code == 0
        assert mock_pipeline.run_all.called

    @patch("scripts.run_poly_pipeline.MainPipeline")
    def test_main_stage_failure_returns_non_zero(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run_all.return_value = MagicMock(success=False, errors=["Train stage failed"])
        mock_pipeline_cls.return_value = mock_pipeline

        exit_code = main(["all"])
        assert exit_code == 1
