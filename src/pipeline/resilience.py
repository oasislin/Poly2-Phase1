"""Resilience, fault isolation, and retry utilities (Ticket #43)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, TypeVar

from src.pipeline.config import PipelineConfig
from src.utils.logger import contextualize, get_logger
from src.validation.alert_dispatcher import AlertDispatcher, LoggingAlertChannel
from src.validation.alert_manager import Alert, AlertManager, AlertSeverity, AlertType

logger = get_logger("poly.resilience")

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    retry_exceptions: tuple = (Exception,),
) -> T:
    """Execute a callable with exponential backoff on failure."""
    delay = initial_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except retry_exceptions as e:
            last_exc = e
            if attempt == max_retries:
                logger.error(f"Function failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
            delay *= backoff_factor

    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected retry loop termination")


class PipelineResilience:
    """Handles runtime exceptions, coordinates alerts, and enforces graceful degradation."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.alert_manager = AlertManager()
        self.alert_dispatcher = AlertDispatcher([LoggingAlertChannel()])

    def handle_failure(
        self,
        stage_name: str,
        error: Exception,
        station: Optional[str] = None,
        **extra_context: Any,
    ) -> Dict[str, Any]:
        """Record and isolate stage failure, dispatch alert notification."""
        ctx = {"stage": stage_name}
        if station:
            ctx["station"] = station
        ctx.update(extra_context)

        with contextualize(**ctx):
            logger.error(f"Stage '{stage_name}' encountered failure: {error}", exc_info=True)

        # Dispatch alert through AlertDispatcher
        try:
            alert = Alert(
                alert_type=AlertType.SYSTEM_FAILURE,
                severity=AlertSeverity.ERROR,
                station_id=station or "ALL",
                message=f"Stage '{stage_name}' failure: {error}",
                metrics={"stage": stage_name, "error": str(error)},
            )
            self.alert_dispatcher.dispatch(alert)
        except Exception as alert_err:
            logger.error(f"Failed to dispatch alert notification: {alert_err}")

        return {
            "status": "FAILED",
            "stage": stage_name,
            "error": str(error),
            "context": ctx,
        }
