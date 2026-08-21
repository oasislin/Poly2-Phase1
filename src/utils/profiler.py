"""Stage performance profiling and metric tracking module (Ticket #41)."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional


class StageProfiler:
    """Tracks execution time, performance metrics, and success status across pipeline stages."""

    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Any]] = {}
        self._start_time: float = time.time()

    def reset(self) -> None:
        """Reset accumulated profile data."""
        self.stages.clear()
        self._start_time = time.time()

    @contextmanager
    def profile(self, stage_name: str, **metrics: Any) -> Generator[Dict[str, Any], None, None]:
        """Context manager to measure elapsed time and record metadata for a stage."""
        start_t = time.time()
        stage_record: Dict[str, Any] = {
            "start_time": start_t,
            "end_time": None,
            "duration_sec": None,
            "status": "RUNNING",
            "metrics": dict(metrics),
            "error": None,
        }
        self.stages[stage_name] = stage_record

        try:
            yield stage_record
            stage_record["status"] = "SUCCESS"
        except Exception as e:
            stage_record["status"] = "FAILED"
            stage_record["error"] = str(e)
            raise
        finally:
            end_t = time.time()
            stage_record["end_time"] = end_t
            stage_record["duration_sec"] = end_t - start_t

    def get_summary(self) -> Dict[str, Any]:
        """Return structured execution summary."""
        total_duration = time.time() - self._start_time
        return {
            "total_duration_sec": total_duration,
            "stage_count": len(self.stages),
            "stages": self.stages,
        }

    def to_markdown(self) -> str:
        """Generate a formatted Markdown summary table."""
        summary = self.get_summary()
        lines = [
            "### ⏱️ Pipeline Execution Profile",
            "",
            "| Stage | Duration (s) | Status | Key Metrics |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for name, data in summary["stages"].items():
            dur = f"{data['duration_sec']:.2f}" if data["duration_sec"] is not None else "-"
            status_icon = "✅ SUCCESS" if data["status"] == "SUCCESS" else f"❌ {data['status']}"
            metrics_str = ", ".join(f"{k}={v}" for k, v in data["metrics"].items()) or "-"
            lines.append(f"| `{name}` | {dur}s | {status_icon} | {metrics_str} |")

        lines.append("")
        lines.append(f"**Total Elapsed Time**: `{summary['total_duration_sec']:.2f}s`")
        return "\n".join(lines)


_GLOBAL_PROFILER: Optional[StageProfiler] = None


def get_global_profiler() -> StageProfiler:
    """Get or initialize the global StageProfiler singleton."""
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is None:
        _GLOBAL_PROFILER = StageProfiler()
    return _GLOBAL_PROFILER


def profile_stage(
    stage_name: str,
    profiler: Optional[StageProfiler] = None,
    **static_metrics: Any,
) -> Callable:
    """Decorator to profile a function as a named pipeline stage."""
    target_profiler = profiler or get_global_profiler()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with target_profiler.profile(stage_name, **static_metrics):
                return func(*args, **kwargs)
        return wrapper

    return decorator
