"""Unit tests for ConfigManager and configuration models (Ticket #40)."""

import os
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.pipeline.config import (
    ConfigManager,
    PipelineConfig,
    DataConfig,
    ModelConfig,
    PredictionConfig,
    ValidationConfig,
    AlertConfig,
)


class TestConfigSchemas:
    """Test individual configuration sub-schemas and validation rules."""

    def test_default_pipeline_config_initialization(self):
        config = PipelineConfig()
        assert config.env == "default"
        assert config.data.stations == ["ZSPD", "KDEN"]
        assert config.data.members == [0, 1, 2, 3, 4]
        assert config.model.max_lead_times == [6, 30, 54]
        assert config.model.min_lead_times == [24, 48]
        assert config.model.l2_reg == 0.001
        assert config.prediction.dynamic_correction_enabled is True
        assert config.prediction.physical_constraints_enabled is True
        assert config.validation.triple_gate_enabled is True
        assert config.alert.crps_degradation_threshold == 0.20

    def test_data_config_invalid_members(self):
        # Only 5-member protocol [0, 1, 2, 3, 4] is allowed in Phase 1 (ADR 0004)
        with pytest.raises(ValidationError):
            DataConfig(members=[0, 1, 2])

    def test_data_config_invalid_stations(self):
        with pytest.raises(ValidationError):
            DataConfig(stations=["INVALID_STATION"])

    def test_model_config_invalid_lead_times(self):
        with pytest.raises(ValidationError):
            ModelConfig(max_lead_times=[-6, 30])


class TestConfigManager:
    """Test ConfigManager YAML loading, environment inheritance, and env var overrides."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        base_yaml = {
            "env": "default",
            "data": {
                "stations": ["ZSPD", "KDEN"],
                "raw_dir": str(tmp_path / "data" / "raw"),
                "processed_dir": str(tmp_path / "data" / "processed"),
            },
            "model": {
                "l2_reg": 0.001,
            },
            "alert": {
                "webhook_url": "https://hooks.slack.com/services/TEST/SECRET/TOKEN",
            },
        }
        dev_yaml = {
            "env": "dev",
            "model": {
                "l2_reg": 0.005,
            },
        }
        
        base_file = tmp_path / "default.yaml"
        dev_file = tmp_path / "dev.yaml"
        
        with open(base_file, "w", encoding="utf-8") as f:
            yaml.dump(base_yaml, f)
        with open(dev_file, "w", encoding="utf-8") as f:
            yaml.dump(dev_yaml, f)
            
        return tmp_path

    def test_load_default_yaml(self, temp_config_dir):
        config = ConfigManager.load(
            config_path=str(temp_config_dir / "default.yaml")
        )
        assert config.env == "default"
        assert config.model.l2_reg == 0.001
        assert "ZSPD" in config.data.stations

    def test_load_with_env_overlay(self, temp_config_dir):
        config = ConfigManager.load(
            config_path=str(temp_config_dir / "default.yaml"),
            env="dev",
            config_dir=str(temp_config_dir),
        )
        assert config.env == "dev"
        assert config.model.l2_reg == 0.005
        assert config.data.stations == ["ZSPD", "KDEN"]

    def test_load_with_dict_overrides(self, temp_config_dir):
        overrides = {
            "model": {"l2_reg": 0.01},
            "validation": {"triple_gate_enabled": False},
        }
        config = ConfigManager.load(
            config_path=str(temp_config_dir / "default.yaml"),
            overrides=overrides,
        )
        assert config.model.l2_reg == 0.01
        assert config.validation.triple_gate_enabled is False

    def test_env_var_override(self, temp_config_dir, monkeypatch):
        monkeypatch.setenv("POLY_ENV", "prod")
        monkeypatch.setenv("POLY_MODEL_L2_REG", "0.002")
        
        config = ConfigManager.load(
            config_path=str(temp_config_dir / "default.yaml")
        )
        assert config.env == "prod"
        assert config.model.l2_reg == 0.002

    def test_secret_redaction_in_to_dict(self, temp_config_dir):
        config = ConfigManager.load(
            config_path=str(temp_config_dir / "default.yaml")
        )
        safe_dict = config.to_dict(mask_secrets=True)
        assert safe_dict["alert"]["webhook_url"] == "********"
        
        raw_dict = config.to_dict(mask_secrets=False)
        assert "TEST/SECRET/TOKEN" in raw_dict["alert"]["webhook_url"]
