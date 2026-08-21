#!/usr/bin/env python3
"""
BacktestReporter: Comprehensive Evaluation Scorecard & Diagnostic Generator.
Part of Phase 1D Validation System (Ticket 4.2-02 / Issue #37).

Implements:
    - Multi-slice aggregation across stations, target types, and lead hours.
    - Lead time error decay analysis tables and skill degradation curves.
    - Export tools for Markdown scorecard, structured JSON metadata, and CSV daily tables.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.validation.backtester import BacktestResult


class BacktestReporter:
    """Reporter consolidating multiple backtest slices into scorecards and diagnostic tables."""

    def __init__(self, results: Sequence[BacktestResult]):
        self.results = list(results)

    def add_result(self, result: BacktestResult) -> None:
        """Append a new backtest result slice."""
        self.results.append(result)

    def generate_lead_time_decay_table(self) -> pd.DataFrame:
        """Construct a structured DataFrame showing performance metric decay as lead time grows."""
        records = []
        for r in self.results:
            records.append({
                "station_id": r.station_id,
                "target_type": r.target_type,
                "lead_hours": r.lead_hours,
                "sample_count": r.sample_count,
                "mean_crps_model": r.mean_crps_model,
                "mean_crps_raw": r.mean_crps_raw,
                "mean_crps_clim": r.mean_crps_clim,
                "mean_crps_persistence": r.mean_crps_persistence,
                "crpss_vs_raw": r.crpss_vs_raw,
                "crpss_vs_clim": r.crpss_vs_clim,
                "crpss_vs_persistence": r.crpss_vs_persistence,
                "mae_model": r.mae_model,
                "mae_raw": r.mae_raw,
                "coverage_90_ci": r.coverage_90_ci_model,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(by=["station_id", "target_type", "lead_hours"]).reset_index(drop=True)
        return df

    def generate_summary_dict(self) -> Dict[str, Any]:
        """Aggregate all slice metrics into a single dictionary summary."""
        if not self.results:
            return {"slices": [], "total_samples": 0}

        slices_data = [r.to_dict() for r in self.results]
        total_samples = sum(r.sample_count for r in self.results)

        # Weighted averages across all slices
        overall_crps_mod = float(np.average([r.mean_crps_model for r in self.results], weights=[r.sample_count for r in self.results]))
        overall_crps_raw = float(np.average([r.mean_crps_raw for r in self.results], weights=[r.sample_count for r in self.results]))
        overall_crps_clim = float(np.average([r.mean_crps_clim for r in self.results], weights=[r.sample_count for r in self.results]))
        overall_crps_persist = float(np.average([r.mean_crps_persistence for r in self.results], weights=[r.sample_count for r in self.results]))

        overall_mae_mod = float(np.average([r.mae_model for r in self.results], weights=[r.sample_count for r in self.results]))
        overall_mae_raw = float(np.average([r.mae_raw for r in self.results], weights=[r.sample_count for r in self.results]))
        overall_cov_90 = float(np.average([r.coverage_90_ci_model for r in self.results], weights=[r.sample_count for r in self.results]))

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_slices": len(self.results),
            "total_samples": total_samples,
            "overall_mean_crps_model": overall_crps_mod,
            "overall_mean_crps_raw": overall_crps_raw,
            "overall_mean_crps_clim": overall_crps_clim,
            "overall_mean_crps_persistence": overall_crps_persist,
            "overall_crpss_vs_raw": 1.0 - (overall_crps_mod / max(1e-12, overall_crps_raw)),
            "overall_crpss_vs_clim": 1.0 - (overall_crps_mod / max(1e-12, overall_crps_clim)),
            "overall_crpss_vs_persistence": 1.0 - (overall_crps_mod / max(1e-12, overall_crps_persist)),
            "overall_mae_model": overall_mae_mod,
            "overall_mae_raw": overall_mae_raw,
            "overall_coverage_90_ci": overall_cov_90,
            "slices": slices_data,
        }

    def generate_markdown_report(self) -> str:
        """Format human-readable GitHub-Flavored Markdown summary scorecard."""
        summary = self.generate_summary_dict()
        decay_df = self.generate_lead_time_decay_table()

        lines = [
            "# Historical Backtest Evaluation Scorecard",
            f"**Generated At**: `{summary.get('generated_at', '')}`  ",
            f"**Total Slices**: `{summary.get('total_slices', 0)}` | **Total Samples**: `{summary.get('total_samples', 0)}`",
            "",
            "## 1. Overall System Summary Table",
            "| Metric | Candidate EMOS | Raw GEFS Ensemble | Climatology Baseline | Persistence Baseline |",
            "|---|:---:|:---:|:---:|:---:|",
            f"| **Mean CRPS** | **`{summary.get('overall_mean_crps_model', 0.0):.4f}`** | `{summary.get('overall_mean_crps_raw', 0.0):.4f}` | `{summary.get('overall_mean_crps_clim', 0.0):.4f}` | `{summary.get('overall_mean_crps_persistence', 0.0):.4f}` |",
            f"| **CRPSS vs Baseline** | — | `{summary.get('overall_crpss_vs_raw', 0.0):+.2%}` | `{summary.get('overall_crpss_vs_clim', 0.0):+.2%}` | `{summary.get('overall_crpss_vs_persistence', 0.0):+.2%}` |",
            f"| **Mean MAE** | **`{summary.get('overall_mae_model', 0.0):.3f}°C`** | `{summary.get('overall_mae_raw', 0.0):.3f}°C` | — | — |",
            f"| **90% CI Coverage** | **`{summary.get('overall_coverage_90_ci', 0.0):.1%}`** | — | — | — |",
            "",
            "## 2. Lead Time Decay Analysis",
            "| Station | Target | Lead Hours | Samples | CRPS (EMOS) | CRPS (Raw) | CRPS (Clim) | CRPSS vs Clim | MAE (EMOS) | 90% CI Cov |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for _, row in decay_df.iterrows():
            lines.append(
                f"| `{row['station_id']}` | `{row['target_type']}` | **`{row['lead_hours']}h`** | "
                f"{row['sample_count']} | **`{row['mean_crps_model']:.4f}`** | `{row['mean_crps_raw']:.4f}` | "
                f"`{row['mean_crps_clim']:.4f}` | `{row['crpss_vs_clim']:+.2%}` | `{row['mae_model']:.3f}°C` | "
                f"`{row['coverage_90_ci']:.1%}` |"
            )

        lines.append("")
        return "\n".join(lines)

    def export_markdown(self, filepath: Union[str, Path]) -> None:
        """Export Markdown report to specified file path."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        md_text = self.generate_markdown_report()
        p.write_text(md_text, encoding="utf-8")

    def export_json(self, filepath: Union[str, Path]) -> None:
        """Export JSON summary to specified file path."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.generate_summary_dict()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_csv_details(self, filepath: Union[str, Path]) -> None:
        """Export all daily predictions and loss rows concatenated into a single CSV."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        all_dfs = []
        for r in self.results:
            if hasattr(r, "df_daily") and isinstance(r.df_daily, pd.DataFrame) and not r.df_daily.empty:
                df_sub = r.df_daily.copy()
                if "station_id" not in df_sub.columns:
                    df_sub["station_id"] = r.station_id
                if "target_type" not in df_sub.columns:
                    df_sub["target_type"] = r.target_type
                if "lead_hours" not in df_sub.columns:
                    df_sub["lead_hours"] = r.lead_hours
                all_dfs.append(df_sub)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(p, index=False)
        else:
            pd.DataFrame().to_csv(p, index=False)
