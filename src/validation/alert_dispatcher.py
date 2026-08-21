#!/usr/bin/env python3
"""
AlertDispatcher: Multi-Channel Distribution Adapters (Logging, File).
Part of Phase 1D Validation System (Ticket 4.3-01 / Issue #38).

Implements:
    - BaseAlertChannel abstract protocol.
    - LoggingAlertChannel: Structured logging output.
    - FileAlertChannel: Real-time append to JSONL audit files.
    - AlertDispatcher: Router delegating alerts to active channels.
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from src.validation.alert_manager import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class BaseAlertChannel(ABC):
    """Abstract interface for all notification delivery channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver alert to target channel. Returns True on success."""
        pass


class LoggingAlertChannel(BaseAlertChannel):
    """Channel logging alerts using standard Python logging."""

    def __init__(self, logger_name: str = "src.validation.alerts"):
        self._logger = logging.getLogger(logger_name)

    def send(self, alert: Alert) -> bool:
        msg = f"[{alert.severity.value}] [{alert.alert_type.value}] {alert.station_id}: {alert.message}"
        if alert.severity == AlertSeverity.CRITICAL:
            self._logger.critical(msg)
        elif alert.severity == AlertSeverity.ERROR:
            self._logger.error(msg)
        elif alert.severity == AlertSeverity.WARNING:
            self._logger.warning(msg)
        else:
            self._logger.info(msg)
        return True


class FileAlertChannel(BaseAlertChannel):
    """Channel persisting alerts to JSONL (JSON Lines) file."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        try:
            line = json.dumps(alert.to_dict(), ensure_ascii=False)
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to append alert to file {self.filepath}: {e}")
            return False


class AlertDispatcher:
    """Central router managing alert delivery to configured channels."""

    def __init__(self, channels: Optional[Sequence[BaseAlertChannel]] = None):
        self.channels: List[BaseAlertChannel] = list(channels) if channels else [LoggingAlertChannel()]

    def add_channel(self, channel: BaseAlertChannel) -> None:
        """Register a notification channel."""
        self.channels.append(channel)

    def dispatch(self, alert: Alert) -> Dict[str, bool]:
        """Send alert across all registered channels."""
        results = {}
        for ch in self.channels:
            ch_name = ch.__class__.__name__
            try:
                results[ch_name] = ch.send(alert)
            except Exception as e:
                logger.error(f"Error in channel {ch_name} while sending alert: {e}")
                results[ch_name] = False
        return results
