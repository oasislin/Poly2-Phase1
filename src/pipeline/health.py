"""System health checking and readiness diagnostics (Ticket #43)."""

from __future__ import annotations

import os
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from src.modeling.partitioner import SEASONS
from src.pipeline.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger("poly.health")


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class HealthChecker:
    """Performs pre-flight checks and runtime health diagnostics for pipeline components."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def check_storage(self) -> Dict[str, Any]:
        """Verify storage directories exist and have read/write access."""
        paths = {
            "raw": self.config.data.raw_dir,
            "processed": self.config.data.processed_dir,
            "models": self.config.data.models_dir,
            "db": self.config.data.db_dir,
        }
        status = HealthStatus.HEALTHY
        details: Dict[str, Any] = {}

        for name, path_str in paths.items():
            p = Path(path_str)
            accessible = False
            try:
                p.mkdir(parents=True, exist_ok=True)
                test_file = p / ".health_check_tmp"
                test_file.write_text("ok")
                test_file.unlink()
                accessible = True
            except Exception as e:
                logger.warning(f"Storage path {name} ({path_str}) not writable: {e}")
                status = HealthStatus.UNHEALTHY

            details[f"{name}_accessible"] = accessible
            details[f"{name}_path"] = str(p.resolve())

        details["status"] = status
        return details

    def check_database(self) -> Dict[str, Any]:
        """Verify SQLite predictions database is accessible."""
        db_path = Path(self.config.data.predictions_db_path)
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            conn.close()
            return {
                "status": HealthStatus.HEALTHY,
                "can_connect": True,
                "db_path": str(db_path.resolve()),
            }
        except Exception as e:
            logger.warning(f"Database connection failed at {db_path}: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "can_connect": False,
                "error": str(e),
            }

    def check_models(self) -> Dict[str, Any]:
        """Check if all 40 EMOS model matrix pickle files exist."""
        models_dir = Path(self.config.data.models_dir)
        stations = self.config.data.stations
        seasons = SEASONS  # ["Spring", "Summer", "Autumn", "Winter"]
        max_lts = self.config.model.max_lead_times  # [6, 30, 54]
        min_lts = self.config.model.min_lead_times  # [24, 48]

        expected_count = len(stations) * len(seasons) * (len(max_lts) + len(min_lts))  # 2 * 4 * (3 + 2) = 40
        found_count = 0
        missing: List[str] = []

        if models_dir.exists():
            for st in stations:
                for sea in seasons:
                    for lt in max_lts:
                        m_file = models_dir / f"{st}_{sea}_Max_lead{lt}h.pkl"
                        if m_file.exists():
                            found_count += 1
                        else:
                            missing.append(m_file.name)
                    for lt in min_lts:
                        m_file = models_dir / f"{st}_{sea}_Min_lead{lt}h.pkl"
                        if m_file.exists():
                            found_count += 1
                        else:
                            missing.append(m_file.name)

        if found_count == expected_count:
            status = HealthStatus.HEALTHY
        elif found_count > 0:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return {
            "status": status,
            "models_found": found_count,
            "expected_count": expected_count,
            "missing_sample": missing[:5],
        }

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all component diagnostics and summarize overall system health."""
        storage_res = self.check_storage()
        db_res = self.check_database()
        models_res = self.check_models()

        components = {
            "storage": storage_res,
            "database": db_res,
            "models": models_res,
        }

        # Overall Status Resolution
        if storage_res["status"] == HealthStatus.UNHEALTHY or db_res["status"] == HealthStatus.UNHEALTHY:
            overall = HealthStatus.UNHEALTHY
        elif models_res["status"] != HealthStatus.HEALTHY or storage_res["status"] == HealthStatus.DEGRADED:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "overall_status": overall,
            "components": components,
        }
