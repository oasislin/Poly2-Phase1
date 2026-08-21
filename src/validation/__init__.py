"""
Validation package for Polymarket Temperature Prediction System.
Phase 1D: Evaluation metrics, statistical significance testing, triple acceptance gates, historical backtesting, and alerting.
"""

from src.validation.alert_dispatcher import (
    AlertDispatcher,
    BaseAlertChannel,
    FileAlertChannel,
    LoggingAlertChannel,
)
from src.validation.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
)
from src.validation.backtest_reporter import BacktestReporter
from src.validation.backtester import BacktestEngine, BacktestResult
from src.validation.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    RawGEFSBaseline,
)
from src.validation.metrics_calculator import MetricsCalculator, ReliabilityDiagramData
from src.validation.statistical_tests import (
    DieboldMarianoResult,
    StatisticalSignificance,
    diebold_mariano_test,
    paired_t_test,
    pit_ks_test,
    wilcoxon_signed_rank_test,
)
from src.validation.triple_gate import (
    ForecastSlice,
    GateEvaluationResult,
    TripleGateEvaluator,
    TripleGateReport,
)

__all__ = [
    "MetricsCalculator",
    "ReliabilityDiagramData",
    "DieboldMarianoResult",
    "StatisticalSignificance",
    "diebold_mariano_test",
    "wilcoxon_signed_rank_test",
    "paired_t_test",
    "pit_ks_test",
    "ForecastSlice",
    "TripleGateEvaluator",
    "TripleGateReport",
    "GateEvaluationResult",
    "BacktestEngine",
    "BacktestResult",
    "BacktestReporter",
    "ClimatologyBaseline",
    "RawGEFSBaseline",
    "PersistenceBaseline",
    "AlertManager",
    "AlertDispatcher",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "BaseAlertChannel",
    "LoggingAlertChannel",
    "FileAlertChannel",
]
