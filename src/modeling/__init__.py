"""
Modeling module for Gaussian EMOS probability prediction and climatological baselines.
"""

from .climatology import ClimatologyCalculator
from .crps import emos_crps_loss, gaussian_crps
from .degradation import DegradationDecision, DegradationHandler
from .emos_trainer import EMOSOptimizer, ModelTrainingDiagnostics
from .gaussian_emos import GaussianEMOS
from .interpolator import LeadTimeInterpolator
from .matrix_trainer import MatrixScorecard, MatrixTrainer
from .partitioner import DatasetPartitioner
from .pipeline import PipelineResult, TrainingPipeline
from .registry import ModelRegistry
from .report_generator import AcceptanceReport, ReportGenerator
from .validation_engine import ValidationEngine, ValidationResult

__all__ = [
    "ClimatologyCalculator",
    "GaussianEMOS",
    "gaussian_crps",
    "emos_crps_loss",
    "EMOSOptimizer",
    "ModelTrainingDiagnostics",
    "DegradationDecision",
    "DegradationHandler",
    "DatasetPartitioner",
    "MatrixTrainer",
    "MatrixScorecard",
    "LeadTimeInterpolator",
    "ModelRegistry",
    "ValidationEngine",
    "ValidationResult",
    "AcceptanceReport",
    "ReportGenerator",
    "TrainingPipeline",
    "PipelineResult",
]
