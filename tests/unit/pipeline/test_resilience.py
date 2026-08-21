"""Unit tests for PipelineResilience and HealthChecker (Ticket #43)."""

import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from src.pipeline.config import PipelineConfig
from src.pipeline.health import HealthChecker, HealthStatus
from src.pipeline.resilience import PipelineResilience, retry_with_backoff


class TestHealthChecker:
    """Test system health diagnostics across directories, databases, and model matrix."""

    @pytest.fixture
    def test_env(self, tmp_path):
        config = PipelineConfig()
        config.data.raw_dir = str(tmp_path / "raw")
        config.data.processed_dir = str(tmp_path / "processed")
        config.data.models_dir = str(tmp_path / "models")
        config.data.db_dir = str(tmp_path / "db")
        config.data.predictions_db_path = str(tmp_path / "db" / "predictions.db")
        
        # Create directories
        for p in [config.data.raw_dir, config.data.processed_dir, config.data.models_dir, config.data.db_dir]:
            os.makedirs(p, exist_ok=True)
            
        # Create SQLite database
        conn = sqlite3.connect(config.data.predictions_db_path)
        conn.execute("CREATE TABLE test (id INT);")
        conn.close()
        
        return config

    def test_check_storage_healthy(self, test_env):
        checker = HealthChecker(test_env)
        res = checker.check_storage()
        assert res["status"] == HealthStatus.HEALTHY
        assert res["raw_accessible"] is True
        assert res["processed_accessible"] is True

    def test_check_database_healthy(self, test_env):
        checker = HealthChecker(test_env)
        res = checker.check_database()
        assert res["status"] == HealthStatus.HEALTHY
        assert res["can_connect"] is True

    def test_check_models_empty_reports_degraded_or_missing(self, test_env):
        checker = HealthChecker(test_env)
        res = checker.check_models()
        # Expecting 40 models, 0 found -> DEGRADED/NOT_READY
        assert res["status"] in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        assert res["models_found"] == 0
        assert res["expected_count"] == 40

    def test_overall_health_report(self, test_env):
        checker = HealthChecker(test_env)
        report = checker.run_all_checks()
        assert "storage" in report["components"]
        assert "database" in report["components"]
        assert "models" in report["components"]
        assert report["overall_status"] in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)


class TestPipelineResilience:
    """Test error isolation, retry mechanism, and alert dispatch on failure."""

    def test_retry_with_backoff_succeeds_after_retry(self):
        attempts = 0

        def flaky_function():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Transient network failure")
            return "SUCCESS"

        result = retry_with_backoff(flaky_function, max_retries=3, initial_delay=0.01)
        assert result == "SUCCESS"
        assert attempts == 3

    def test_retry_with_backoff_raises_when_exhausted(self):
        def always_failing():
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            retry_with_backoff(always_failing, max_retries=2, initial_delay=0.01)

    def test_isolate_and_record_failure(self):
        config = PipelineConfig()
        resilience = PipelineResilience(config)
        resilience.alert_dispatcher = MagicMock()

        err = RuntimeError("Unexpected feature computation error")
        isolated_res = resilience.handle_failure("feature", err, station="ZSPD")

        assert isolated_res["status"] == "FAILED"
        assert "Unexpected feature computation error" in isolated_res["error"]
        assert resilience.alert_dispatcher.dispatch.called
        alert_obj = resilience.alert_dispatcher.dispatch.call_args[0][0]
        assert alert_obj.station_id == "ZSPD"
        assert "feature" in alert_obj.message
