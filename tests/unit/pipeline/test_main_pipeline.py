"""Unit tests for MainPipeline end-to-end orchestrator (Ticket #42)."""

import pytest
from unittest.mock import MagicMock, patch

from src.pipeline.config import PipelineConfig
from src.pipeline.main_pipeline import MainPipeline, PipelineStage, PipelineExecutionResult


class TestMainPipeline:
    """Test MainPipeline stage routing, execution state, and artifact coordination."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = PipelineConfig()
        config.data.raw_dir = str(tmp_path / "raw")
        config.data.processed_dir = str(tmp_path / "processed")
        config.data.models_dir = str(tmp_path / "models")
        config.data.db_dir = str(tmp_path / "db")
        config.data.predictions_db_path = str(tmp_path / "db" / "predictions.db")
        return config

    def test_pipeline_initialization(self, mock_config):
        pipeline = MainPipeline(mock_config)
        assert pipeline.config == mock_config
        assert len(pipeline.stages) == 5
        assert PipelineStage.INGEST in pipeline.stages
        assert PipelineStage.FEATURE in pipeline.stages
        assert PipelineStage.TRAIN in pipeline.stages
        assert PipelineStage.PREDICT in pipeline.stages
        assert PipelineStage.VALIDATE in pipeline.stages

    @patch("src.pipeline.main_pipeline.MainPipeline._execute_ingest")
    @patch("src.pipeline.main_pipeline.MainPipeline._execute_feature")
    @patch("src.pipeline.main_pipeline.MainPipeline._execute_train")
    @patch("src.pipeline.main_pipeline.MainPipeline._execute_predict")
    @patch("src.pipeline.main_pipeline.MainPipeline._execute_validate")
    def test_run_all_executes_all_stages_sequentially(
        self, mock_val, mock_pred, mock_train, mock_feat, mock_ingest, mock_config
    ):
        mock_ingest.return_value = {"status": "SUCCESS", "files": 10}
        mock_feat.return_value = {"status": "SUCCESS", "processed_records": 100}
        mock_train.return_value = {"status": "SUCCESS", "models_trained": 40}
        mock_pred.return_value = {"status": "SUCCESS", "predictions_count": 50}
        mock_val.return_value = {"status": "SUCCESS", "triple_gate_passed": True}

        pipeline = MainPipeline(mock_config)
        result: PipelineExecutionResult = pipeline.run_all()

        assert result.success is True
        assert len(result.stage_results) == 5
        assert mock_ingest.called
        assert mock_feat.called
        assert mock_train.called
        assert mock_pred.called
        assert mock_val.called

    @patch("src.pipeline.main_pipeline.MainPipeline._execute_ingest")
    @patch("src.pipeline.main_pipeline.MainPipeline._execute_feature")
    @patch("src.pipeline.main_pipeline.MainPipeline._execute_train")
    def test_run_resume_from_stage(
        self, mock_train, mock_feat, mock_ingest, mock_config
    ):
        pipeline = MainPipeline(mock_config)
        result = pipeline.run_all(resume_from=PipelineStage.TRAIN, stages_to_run=[PipelineStage.TRAIN])

        assert not mock_ingest.called
        assert not mock_feat.called
        assert mock_train.called

    @patch("src.pipeline.main_pipeline.MainPipeline._execute_train")
    def test_run_single_stage(self, mock_train, mock_config):
        mock_train.return_value = {"status": "SUCCESS", "models": 40}
        pipeline = MainPipeline(mock_config)
        res = pipeline.run_stage(PipelineStage.TRAIN)

        assert res["status"] == "SUCCESS"
        assert mock_train.called
