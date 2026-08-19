#!/usr/bin/env python3
"""
ModelRegistry: Standardized model persistence and unified inference query facade (Ticket 2.2-06 / Issue #19).

Implements (v5.9.1 §4.4):
    - Persistence naming: {StationID}_{Season}_{Max|Min}_lead{Hours}h.pkl
    - Storage payload: GaussianEMOS model object + rich diagnostic & degradation metadata
    - Unified inference facade:
        get_model(station_id, target_date, target_type, lead_hours)
        predict(station_id, target_date, target_type, lead_hours, ens_mean, ens_var, sigma_clim_sq)
    - Automatic season mapping, dense grid generation, and on-the-fly interpolation
"""

from datetime import date, datetime, timezone
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.modeling.degradation import DegradationDecision, DegradationHandler
from src.modeling.emos_trainer import ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS
from src.modeling.interpolator import LeadTimeInterpolator
from src.modeling.matrix_trainer import MatrixScorecard
from src.modeling.partitioner import DatasetPartitioner

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model storage repository and runtime inference facade for calibrated Gaussian EMOS models."""

    def __init__(self, base_dir: Union[str, Path] = "models/emos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.interpolator = LeadTimeInterpolator()
        self.partitioner = DatasetPartitioner()
        self._cache: Dict[str, Tuple[GaussianEMOS, Dict[str, Any]]] = {}

    @staticmethod
    def format_model_filename(station_id: str, season: str, target_type: str, lead_hours: int) -> str:
        """Format filename per v5.9.1 convention: {StationID}_{Season}_{Max|Min}_lead{Hours}h.pkl."""
        st = station_id.upper()
        seas = season.capitalize()
        t_name = "Max" if target_type.lower() == "max" else "Min"
        lead = int(round(lead_hours))
        return f"{st}_{seas}_{t_name}_lead{lead}h.pkl"

    def save_model(
        self,
        model: GaussianEMOS,
        station_id: str,
        season: str,
        target_type: str,
        lead_hours: int,
        diagnostics: Optional[ModelTrainingDiagnostics] = None,
        decision: Optional[DegradationDecision] = None,
        is_interpolated: bool = False,
    ) -> Path:
        """Persist a single GaussianEMOS model and its metadata to disk."""
        filename = self.format_model_filename(station_id, season, target_type, lead_hours)
        file_path = self.base_dir / filename

        metadata: Dict[str, Any] = {
            "station_id": station_id.upper(),
            "season": season.capitalize(),
            "target_type": target_type.lower(),
            "lead_hours": int(round(lead_hours)),
            "is_interpolated": is_interpolated,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "diagnostics": diagnostics.to_dict() if diagnostics is not None else None,
            "decision": decision.to_dict() if decision is not None else None,
        }

        payload = {
            "model": model,
            "metadata": metadata,
        }

        with open(file_path, "wb") as f:
            pickle.dump(payload, f)

        # Update cache
        self._cache[filename] = (model, metadata)
        logger.debug("Persisted EMOS model to %s", file_path)
        return file_path

    def load_model(
        self,
        station_id: str,
        season: str,
        target_type: str,
        lead_hours: int,
    ) -> Tuple[GaussianEMOS, Dict[str, Any]]:
        """Load a persisted model and its metadata from disk/cache."""
        filename = self.format_model_filename(station_id, season, target_type, lead_hours)
        
        if filename in self._cache:
            return self._cache[filename]

        file_path = self.base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, "rb") as f:
            payload = pickle.load(f)

        model: GaussianEMOS = payload["model"]
        metadata: Dict[str, Any] = payload.get("metadata", {})
        self._cache[filename] = (model, metadata)
        return model, metadata

    def save_scorecard(
        self,
        scorecard: MatrixScorecard,
        build_dense_grid: bool = True,
    ) -> List[Path]:
        """Persist all 40 models from MatrixScorecard and optionally interpolate full 6h-dense grid."""
        saved_paths: List[Path] = []

        # 1. Save all trained anchor models
        anchor_dict: Dict[Tuple[str, str, str], Dict[int, GaussianEMOS]] = {}
        for (station, season, target_type, lead), (model, diag, decision) in scorecard.models.items():
            path = self.save_model(
                model=model,
                station_id=station,
                season=season,
                target_type=target_type,
                lead_hours=lead,
                diagnostics=diag,
                decision=decision,
                is_interpolated=False,
            )
            saved_paths.append(path)

            key = (station, season, target_type)
            if key not in anchor_dict:
                anchor_dict[key] = {}
            anchor_dict[key][lead] = model

        # 2. Build and save dense 6h-grid models across 6h to 54h
        if build_dense_grid:
            all_leads = [6, 12, 18, 24, 30, 36, 42, 48, 54]
            for (station, season, target_type), anchors in anchor_dict.items():
                dense_grid = self.interpolator.build_full_grid(
                    target_type=target_type,
                    anchor_models=anchors,
                    grid_leads=all_leads,
                )
                for lead, interp_model in dense_grid.items():
                    if lead not in anchors:  # Intermediate interpolated model
                        path = self.save_model(
                            model=interp_model,
                            station_id=station,
                            season=season,
                            target_type=target_type,
                            lead_hours=lead,
                            is_interpolated=True,
                        )
                        saved_paths.append(path)

        return saved_paths

    def get_model(
        self,
        station_id: str,
        target_date: Union[date, str, pd.Timestamp],
        target_type: str,
        lead_hours: Union[int, float],
    ) -> GaussianEMOS:
        """Facade query: fetch or interpolate appropriate GaussianEMOS model for station, date, target, and lead time."""
        season = self.partitioner.get_season(target_date)
        int_lead = int(round(lead_hours))

        # Check if model exists directly on disk/cache
        try:
            model, _ = self.load_model(station_id, season, target_type, int_lead)
            return model
        except FileNotFoundError:
            # Fallback: dynamically load available anchor models and interpolate on the fly
            anchors = self._load_available_anchors(station_id, season, target_type)
            if not anchors:
                raise FileNotFoundError(
                    f"No models or anchors found for {station_id} {season} {target_type}"
                )
            return self.interpolator.get_model_at_lead(target_type, lead_hours, anchors)

    def predict(
        self,
        station_id: str,
        target_date: Union[date, str, pd.Timestamp],
        target_type: str,
        lead_hours: Union[int, float],
        ensemble_mean: Union[float, np.ndarray, pd.Series],
        ensemble_variance: Union[float, np.ndarray, pd.Series],
        sigma_clim_squared: Union[float, np.ndarray, pd.Series],
    ) -> GaussianEMOS:
        """Facade inference: predict calibrated Gaussian distribution with short-lead decay handling."""
        season = self.partitioner.get_season(target_date)
        anchors = self._load_available_anchors(station_id, season, target_type)

        if not anchors:
            # If no anchors loaded, try loading direct model
            model = self.get_model(station_id, target_date, target_type, lead_hours)
            mu, sigma = model.compute_params(ensemble_mean, ensemble_variance, sigma_clim_squared)
            return GaussianEMOS.from_params(mu=mu, sigma=sigma)

        return self.interpolator.predict_distribution(
            target_type=target_type,
            lead_hours=lead_hours,
            ensemble_mean=ensemble_mean,
            ensemble_variance=ensemble_variance,
            sigma_clim_squared=sigma_clim_squared,
            anchor_models=anchors,
        )

    def _load_available_anchors(
        self,
        station_id: str,
        season: str,
        target_type: str,
    ) -> Dict[int, GaussianEMOS]:
        """Load all anchor models available on disk for a station-season-target tuple."""
        anchor_leads = self.partitioner.get_lead_time_nodes(target_type)
        anchors: Dict[int, GaussianEMOS] = {}
        for lead in anchor_leads:
            try:
                model, _ = self.load_model(station_id, season, target_type, lead)
                anchors[lead] = model
            except FileNotFoundError:
                continue
        return anchors

    def list_inventory(self) -> pd.DataFrame:
        """List all models currently persisted in the registry directory."""
        records = []
        for file_path in sorted(self.base_dir.glob("*.pkl")):
            try:
                with open(file_path, "rb") as f:
                    payload = pickle.load(f)
                meta = payload.get("metadata", {})
                records.append({
                    "filename": file_path.name,
                    "station_id": meta.get("station_id"),
                    "season": meta.get("season"),
                    "target_type": meta.get("target_type"),
                    "lead_hours": meta.get("lead_hours"),
                    "is_interpolated": meta.get("is_interpolated", False),
                    "saved_at": meta.get("saved_at"),
                })
            except Exception as e:
                logger.warning("Error reading model payload from %s: %s", file_path, e)
        return pd.DataFrame(records)
