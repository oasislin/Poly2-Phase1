"""Structured logging module with context propagation and JSON/Console formatting (Ticket #41)."""

from __future__ import annotations

import contextvars
import datetime
import json
import logging
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("log_context", default={})


def get_current_context() -> Dict[str, Any]:
    """Get the current active logging context."""
    return dict(_LOG_CONTEXT.get())


@contextmanager
def contextualize(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager to attach metadata keys/values to all logs emitted in this block."""
    current = dict(_LOG_CONTEXT.get())
    updated = {**current, **kwargs}
    token = _LOG_CONTEXT.set(updated)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


class ContextFilter(logging.Filter):
    """Logging filter that injects the active contextvars metadata into LogRecord attributes."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get()
        for k, v in ctx.items():
            setattr(record, k, v)
        return True


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON lines with context metadata."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = _LOG_CONTEXT.get()
        log_obj = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **ctx,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


class StandardFormatter(logging.Formatter):
    """Formats log records for human-readable console output."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = _LOG_CONTEXT.get()
        ctx_str = f" [{ ' '.join(f'{k}={v}' for k, v in ctx.items()) }]" if ctx else ""
        record.msg = f"{record.msg}{ctx_str}"
        return super().format(record)


def _build_standard_formatter() -> logging.Formatter:
    """Build standard human-readable log formatter."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure root / poly logger with appropriate handlers and formatters."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing poly handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        if getattr(handler, "_poly_handler", False):
            root_logger.removeHandler(handler)

    ctx_filter = ContextFilter()
    formatter = JsonFormatter() if json_format else _build_standard_formatter()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler._poly_handler = True  # type: ignore
    console_handler.addFilter(ctx_filter)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler._poly_handler = True  # type: ignore
        file_handler.addFilter(ctx_filter)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger("poly")


def get_logger(name: str = "poly") -> logging.Logger:
    """Get a logger instance with ContextFilter attached."""
    logger = logging.getLogger(name)
    has_filter = any(isinstance(f, ContextFilter) for f in logger.filters)
    if not has_filter:
        logger.addFilter(ContextFilter())
    return logger
