"""Strongly-typed configuration system and manager for Polymarket prediction pipeline (Ticket #40).

Supports hierarchical YAML loading, environment overrides, runtime dict overlays,
environment variable mapping (POLY_*), and sensitive credential redaction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


VALID_STATIONS = {"ZSPD", "KDEN"}
ALLOWED_5_MEMBERS = [0, 1, 2, 3, 4]


class DataConfig(BaseModel):
    """Configuration for data acquisition, processing and storage paths."""
    stations: List[str] = Field(default_factory=lambda: ["ZSPD", "KDEN"])
    members: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"
    db_dir: str = "data/db"
    predictions_db_path: str = "data/db/predictions.db"
    train_years: Tuple[int, int] = (2000, 2018)
    val_years: Tuple[int, int] = (2019, 2019)

    @field_validator("stations")
    @classmethod
    def validate_stations(cls, v: List[str]) -> List[str]:
        for s in v:
            if s not in VALID_STATIONS:
                raise ValueError(f"Invalid station '{s}'. Must be one of {VALID_STATIONS}")
        return v

    @field_validator("members")
    @classmethod
    def validate_members(cls, v: List[int]) -> List[int]:
        if sorted(v) != ALLOWED_5_MEMBERS:
            raise ValueError(
                f"Invalid ensemble members {v}. Phase 1 requires exact 5-member protocol [0, 1, 2, 3, 4] (c00 + p01-p04)."
            )
        return v


class ModelConfig(BaseModel):
    """Configuration for EMOS model training and matrix interpolation."""
    max_lead_times: List[int] = Field(default_factory=lambda: [6, 30, 54])
    min_lead_times: List[int] = Field(default_factory=lambda: [24, 48])
    l2_reg: float = 0.001
    climatology_window_days: int = 31
    optimizer: str = "L-BFGS-B"
    variance_floor_type: str = "climatology"

    @field_validator("max_lead_times", "min_lead_times")
    @classmethod
    def validate_positive_lead_times(cls, v: List[int]) -> List[int]:
        for lt in v:
            if lt <= 0:
                raise ValueError(f"Lead time must be positive integer, got {lt}")
        return v


class PredictionConfig(BaseModel):
    """Configuration for multi-layer inference and Polymarket market bins."""
    dynamic_correction_enabled: bool = True
    physical_constraints_enabled: bool = True
    bin_width: float = 1.0
    tolerance_epsilon: float = 1e-7


class ValidationConfig(BaseModel):
    """Configuration for triple acceptance gate and historical backtesting."""
    triple_gate_enabled: bool = True
    alpha_significance: float = 0.05
    virtual_crps_loss_threshold: float = 1.05
    extreme_ci_target: float = 0.90
    extreme_ci_coverage_min: float = 0.80


class AlertConfig(BaseModel):
    """Configuration for monitoring, alerting thresholds and notification channels."""
    crps_degradation_threshold: float = 0.20
    enabled_channels: List[str] = Field(default_factory=lambda: ["console"])
    webhook_url: Optional[str] = None


class PipelineConfig(BaseModel):
    """Top-level unified pipeline configuration."""
    env: str = "default"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary, with optional secret masking."""
        raw = self.model_dump()
        if mask_secrets and raw.get("alert", {}).get("webhook_url"):
            raw["alert"]["webhook_url"] = "********"
        return raw


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update a dictionary."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class ConfigManager:
    """Manager for loading, layering, and validating pipeline configurations."""

    @classmethod
    def _resolve_env_overlay(
        cls,
        target_env: str,
        config_path: Optional[str] = None,
        config_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search and load environment-specific overlay YAML."""
        if not target_env or target_env == "default":
            return {}

        search_dirs = []
        if config_dir:
            search_dirs.append(Path(config_dir))
        if config_path:
            search_dirs.append(Path(config_path).parent)
        search_dirs.append(Path("configs"))

        for s_dir in search_dirs:
            env_file = s_dir / f"{target_env}.yaml"
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        return {}

    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
        env: Optional[str] = None,
        config_dir: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PipelineConfig:
        """Load and resolve configuration hierarchy."""
        raw_dict: Dict[str, Any] = {}

        # 1. Load primary config file if specified
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                raw_dict = _deep_update(raw_dict, yaml.safe_load(f) or {})

        # 2. Determine env and load env overlay if available
        target_env = env or raw_dict.get("env") or os.getenv("POLY_ENV")
        if target_env:
            raw_dict["env"] = target_env
            env_overlay = cls._resolve_env_overlay(target_env, config_path, config_dir)
            raw_dict = _deep_update(raw_dict, env_overlay)

        # 3. Apply explicit dict overrides & env vars
        if overrides:
            raw_dict = _deep_update(raw_dict, overrides)

        cls._apply_env_vars(raw_dict)
        return PipelineConfig.model_validate(raw_dict)

    @classmethod
    def _apply_env_vars(cls, data: Dict[str, Any]) -> None:
        """Parse POLY_* environment variables into configuration dictionary."""
        for key, value in os.environ.items():
            if not key.startswith("POLY_"):
                continue

            parts = key[5:].lower().split("_")
            # Handle special top-level variables
            if len(parts) == 1 and parts[0] == "env":
                data["env"] = value
                continue

            # Nested sections
            section = parts[0]
            field_name = "_".join(parts[1:])

            if section not in data or not isinstance(data[section], dict):
                data[section] = {}

            # Attempt auto type conversion (int, float, bool)
            typed_val: Any = value
            if value.lower() in ("true", "1", "yes"):
                typed_val = True
            elif value.lower() in ("false", "0", "no"):
                typed_val = False
            else:
                try:
                    typed_val = int(value)
                except ValueError:
                    try:
                        typed_val = float(value)
                    except ValueError:
                        typed_val = value

            data[section][field_name] = typed_val
