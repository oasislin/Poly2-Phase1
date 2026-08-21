#!/usr/bin/env python3
"""
AlertManager: Performance Degradation & Data Anomaly Detection.
Part of Phase 1D Validation System (Ticket 4.3-01 / Issue #38).

Implements:
    - Alert dataclass and Enums (AlertType, AlertSeverity).
    - Detection rules: CRPS degradation (>20%), PIT miscalibration, Data staleness, Physical anomaly.
    - Anti-flood fingerprint cooldown and throttling.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import logging
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.validation.statistical_tests import pit_ks_test

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for system alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    """Categorized trigger types for alerts."""
    CRPS_DEGRADATION = "CRPS_DEGRADATION"
    PIT_MISCALIBRATION = "PIT_MISCALIBRATION"
    DATA_STALENESS = "DATA_STALENESS"
    PHYSICAL_ANOMALY = "PHYSICAL_ANOMALY"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"


@dataclass
class Alert:
    """Standardized alert data container."""

    alert_type: AlertType
    severity: AlertSeverity
    station_id: str
    target_type: Optional[str] = None
    lead_hours: Optional[int] = None
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def fingerprint(self) -> str:
        """Generate unique deterministic signature for cooldown deduplication."""
        raw_key = f"{self.alert_type.value}_{self.station_id}_{self.target_type}_{self.lead_hours}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "station_id": self.station_id,
            "target_type": self.target_type,
            "lead_hours": self.lead_hours,
            "message": self.message,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
        }


class AlertManager:
    """Monitor engine evaluating statistical metrics and data integrity for anomalies."""

    def __init__(self, cooldown_seconds: float = 3600.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_times: Dict[str, datetime] = {}

    def _is_throttled(self, alert: Alert) -> bool:
        """Check if an alert with identical signature is currently in cooldown window."""
        fp = alert.fingerprint
        now = datetime.now(timezone.utc)
        if fp in self._last_alert_times:
            elapsed = (now - self._last_alert_times[fp]).total_seconds()
            if elapsed < self.cooldown_seconds:
                logger.debug(f"Alert {fp} throttled (elapsed {elapsed:.1f}s < cooldown {self.cooldown_seconds}s)")
                return True

        self._last_alert_times[fp] = now
        return False

    def check_crps_degradation(
        self,
        station_id: str,
        target_type: str,
        lead_hours: int,
        crps_current: float,
        crps_baseline: float,
        threshold_pct: float = 0.20,
    ) -> Optional[Alert]:
        """Trigger warning if current rolling CRPS degrades by more than threshold_pct vs baseline."""
        safe_base = max(1e-12, float(crps_baseline))
        pct_increase = (float(crps_current) - safe_base) / safe_base

        if pct_increase > threshold_pct:
            alert = Alert(
                alert_type=AlertType.CRPS_DEGRADATION,
                severity=AlertSeverity.WARNING,
                station_id=station_id,
                target_type=target_type,
                lead_hours=lead_hours,
                message=(
                    f"Model performance degraded at {station_id} {target_type} ({lead_hours}h): "
                    f"CRPS increased by {pct_increase:+.1%} (current={crps_current:.4f}, base={crps_baseline:.4f})"
                ),
                metrics={
                    "crps_current": float(crps_current),
                    "crps_baseline": float(crps_baseline),
                    "degradation_pct": float(pct_increase),
                    "threshold_pct": threshold_pct,
                },
            )
            if not self._is_throttled(alert):
                return alert

        return None

    def check_pit_miscalibration(
        self,
        station_id: str,
        target_type: str,
        lead_hours: int,
        pit_values: Union[np.ndarray, pd.Series, Sequence[float]],
        alpha: float = 0.05,
    ) -> Optional[Alert]:
        """Trigger error if PIT uniformity hypothesis fails (p < alpha)."""
        pits = np.asarray(pit_values, dtype=np.float64)
        ks_stat, p_val, is_calibrated = pit_ks_test(pits, alpha=alpha)

        if not is_calibrated:
            alert = Alert(
                alert_type=AlertType.PIT_MISCALIBRATION,
                severity=AlertSeverity.ERROR,
                station_id=station_id,
                target_type=target_type,
                lead_hours=lead_hours,
                message=(
                    f"PIT calibration deviation detected at {station_id} {target_type} ({lead_hours}h): "
                    f"KS p-value={p_val:.4e} <= {alpha}"
                ),
                metrics={
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "alpha": alpha,
                },
            )
            if not self._is_throttled(alert):
                return alert

        return None

    def check_data_staleness(
        self,
        station_id: str,
        last_update_time: Union[datetime, str],
        max_delay_hours: float = 6.0,
    ) -> Optional[Alert]:
        """Trigger error if data arrival lag exceeds max_delay_hours."""
        if isinstance(last_update_time, str):
            dt = pd.to_datetime(last_update_time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = last_update_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        lag_hours = (now - dt).total_seconds() / 3600.0

        if lag_hours > max_delay_hours:
            alert = Alert(
                alert_type=AlertType.DATA_STALENESS,
                severity=AlertSeverity.ERROR,
                station_id=station_id,
                message=f"Data feed staleness at {station_id}: lag is {lag_hours:.1f}h (exceeds {max_delay_hours}h limit)",
                metrics={"lag_hours": lag_hours, "max_delay_hours": max_delay_hours, "last_seen": dt.isoformat()},
            )
            if not self._is_throttled(alert):
                return alert

        return None

    def check_physical_anomaly(
        self,
        station_id: str,
        temp_t0: float,
        temp_t1: float,
        dt_hours: float = 1.0,
        max_rate_per_hour: float = 8.0,
    ) -> Optional[Alert]:
        """Trigger critical alert if hourly temperature change violates physical bounds."""
        safe_dt = max(1e-3, float(dt_hours))
        rate = abs(float(temp_t1) - float(temp_t0)) / safe_dt

        if rate > max_rate_per_hour:
            alert = Alert(
                alert_type=AlertType.PHYSICAL_ANOMALY,
                severity=AlertSeverity.CRITICAL,
                station_id=station_id,
                message=(
                    f"Physical anomaly at {station_id}: Rate of change {rate:.2f}°C/h exceeds physical limit {max_rate_per_hour}°C/h "
                    f"(from {temp_t0:.1f}°C to {temp_t1:.1f}°C in {dt_hours}h)"
                ),
                metrics={
                    "temp_t0": temp_t0,
                    "temp_t1": temp_t1,
                    "dt_hours": dt_hours,
                    "rate_per_hour": rate,
                    "max_rate_per_hour": max_rate_per_hour,
                },
            )
            if not self._is_throttled(alert):
                return alert

        return None
