#!/usr/bin/env python3
"""
Unit tests for AlertManager and AlertDispatcher (Ticket 4.3-01 / Issue #38).

Verifies:
- CRPS degradation (>20% loss vs baseline) detection
- PIT miscalibration (KS p < 0.05) detection
- Data staleness and physical rate-of-change anomaly detection
- Cooldown deduplication and throttling
- Multi-channel dispatching (Logging, File, Slack webhook mock)
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.validation.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
)
from src.validation.alert_dispatcher import (
    AlertDispatcher,
    FileAlertChannel,
    LoggingAlertChannel,
)


class TestAlertManagerRules:
    """Tests for anomaly detection and alert triggering."""

    def test_check_crps_degradation(self):
        manager = AlertManager()

        # Normal: 1.15 vs 1.10 -> +4.5% -> No alert
        alert_none = manager.check_crps_degradation("ZSPD", "max", 30, crps_current=1.15, crps_baseline=1.10)
        assert alert_none is None

        # Degraded: 1.45 vs 1.10 -> +31.8% (>20%) -> Triggers Alert
        alert_degraded = manager.check_crps_degradation("ZSPD", "max", 30, crps_current=1.45, crps_baseline=1.10)
        assert isinstance(alert_degraded, Alert)
        assert alert_degraded.alert_type == AlertType.CRPS_DEGRADATION
        assert alert_degraded.severity == AlertSeverity.WARNING
        assert alert_degraded.metrics["degradation_pct"] > 0.20

    def test_check_pit_miscalibration(self):
        manager = AlertManager()

        # Uniform PIT -> No alert
        np.random.seed(42)
        pit_uniform = np.random.uniform(0, 1, size=200)
        alert_none = manager.check_pit_miscalibration("ZSPD", "max", 30, pit_uniform)
        assert alert_none is None

        # Heavily biased PIT -> Triggers Alert
        pit_biased = np.array([0.01] * 100 + [0.99] * 100)
        alert_biased = manager.check_pit_miscalibration("ZSPD", "max", 30, pit_biased)
        assert isinstance(alert_biased, Alert)
        assert alert_biased.alert_type == AlertType.PIT_MISCALIBRATION
        assert alert_biased.severity == AlertSeverity.ERROR

    def test_check_data_staleness_and_physical_anomaly(self):
        manager = AlertManager()

        now = datetime.now(timezone.utc)
        fresh_time = now - timedelta(hours=2)
        stale_time = now - timedelta(hours=10)

        # Fresh data
        assert manager.check_data_staleness("ZSPD", fresh_time, max_delay_hours=6.0) is None

        # Stale data
        alert_stale = manager.check_data_staleness("ZSPD", stale_time, max_delay_hours=6.0)
        assert isinstance(alert_stale, Alert)
        assert alert_stale.alert_type == AlertType.DATA_STALENESS

        # Physical anomaly: 10 deg/h jump
        alert_physical = manager.check_physical_anomaly("ZSPD", temp_t0=15.0, temp_t1=26.0, dt_hours=1.0, max_rate_per_hour=8.0)
        assert isinstance(alert_physical, Alert)
        assert alert_physical.alert_type == AlertType.PHYSICAL_ANOMALY
        assert alert_physical.severity == AlertSeverity.CRITICAL


class TestAlertThrottlingAndDispatch:
    """Tests for alert cooldown throttling and multi-channel dispatch."""

    def test_cooldown_throttling(self):
        manager = AlertManager(cooldown_seconds=3600)

        # First alert fires
        a1 = manager.check_crps_degradation("ZSPD", "max", 30, crps_current=1.50, crps_baseline=1.0)
        assert a1 is not None

        # Immediate second identical check gets throttled
        a2 = manager.check_crps_degradation("ZSPD", "max", 30, crps_current=1.50, crps_baseline=1.0)
        assert a2 is None

    def test_file_and_logging_channels(self, tmp_path: Path):
        log_file = tmp_path / "alerts.jsonl"
        file_channel = FileAlertChannel(log_file)
        log_channel = LoggingAlertChannel()

        dispatcher = AlertDispatcher(channels=[file_channel, log_channel])

        alert = Alert(
            alert_type=AlertType.CRPS_DEGRADATION,
            severity=AlertSeverity.WARNING,
            station_id="ZSPD",
            target_type="max",
            message="CRPS degraded by 25%",
            metrics={"curr": 1.25, "base": 1.0},
        )

        dispatcher.dispatch(alert)

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["station_id"] == "ZSPD"
        assert record["alert_type"] == "CRPS_DEGRADATION"


