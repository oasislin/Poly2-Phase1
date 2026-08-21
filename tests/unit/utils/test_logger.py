"""Unit tests for StructuredLogger and StageProfiler (Ticket #41)."""

import json
import logging
import time
import pytest

from src.utils.logger import (
    get_logger,
    setup_logger,
    contextualize,
    get_current_context,
)
from src.utils.profiler import StageProfiler, profile_stage, get_global_profiler


class TestStructuredLogger:
    """Test logger configuration, context propagation, and formatting."""

    def test_contextualize_injects_metadata(self, caplog):
        setup_logger(level="INFO")
        logger = get_logger("test_ctx")
        
        with contextualize(station="ZSPD", stage="feature_extraction"):
            with caplog.at_level(logging.INFO):
                logger.info("Extracting daily extreme features")
                
        assert len(caplog.records) > 0
        record = caplog.records[-1]
        assert hasattr(record, "station")
        assert record.station == "ZSPD"
        assert getattr(record, "stage") == "feature_extraction"

    def test_nested_contextualize(self):
        with contextualize(station="KDEN"):
            assert get_current_context()["station"] == "KDEN"
            with contextualize(lead_time=30):
                ctx = get_current_context()
                assert ctx["station"] == "KDEN"
                assert ctx["lead_time"] == 30
            # Context restored
            assert "lead_time" not in get_current_context()
            assert get_current_context()["station"] == "KDEN"

    def test_json_formatter_output(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logger(level="INFO", log_file=str(log_file), json_format=True)
        logger = get_logger("test_json")

        with contextualize(station="ZSPD", lead_time=54):
            logger.info("JSON log test message")

        # Read back log file
        with open(log_file, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["message"] == "JSON log test message"
            assert data["station"] == "ZSPD"
            assert data["lead_time"] == 54
            assert data["level"] == "INFO"


class TestStageProfiler:
    """Test stage execution profiling and metric accumulation."""

    def test_stage_profiler_context_manager(self):
        profiler = StageProfiler()
        with profiler.profile("data_ingestion", sample_count=100):
            time.sleep(0.02)

        summary = profiler.get_summary()
        assert "data_ingestion" in summary["stages"]
        stage = summary["stages"]["data_ingestion"]
        assert stage["duration_sec"] >= 0.015
        assert stage["metrics"]["sample_count"] == 100
        assert summary["total_duration_sec"] >= 0.015

    def test_profile_stage_decorator(self):
        profiler = get_global_profiler()
        profiler.reset()

        @profile_stage("model_optimization", profiler=profiler)
        def dummy_opt(x: int):
            time.sleep(0.01)
            return x * 2

        res = dummy_opt(5)
        assert res == 10
        summary = profiler.get_summary()
        assert "model_optimization" in summary["stages"]
        assert summary["stages"]["model_optimization"]["duration_sec"] >= 0.009

    def test_format_markdown_report(self):
        profiler = StageProfiler()
        with profiler.profile("stage_1"):
            time.sleep(0.01)
        with profiler.profile("stage_2"):
            time.sleep(0.01)

        md_report = profiler.to_markdown()
        assert "| Stage | Duration (s) | Status |" in md_report
        assert "stage_1" in md_report
        assert "stage_2" in md_report
