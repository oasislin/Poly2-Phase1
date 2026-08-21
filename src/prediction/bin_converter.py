#!/usr/bin/env python3
"""
BinConverter: Polymarket rule-based discrete market probability conversion and settlement resolution (Ticket 3.4-01 / Issue #29).

Implements (Phase 1C / v5.9.2):
    1. Comprehensive Polymarket Bin types:
       - Single value ('=T'): P = F(T + 0.5) - F(T - 0.5)
       - Range ('T1-T2'): P = F(T2 + 0.5) - F(T1 - 0.5)
       - Lower threshold ('<=T'): P = F(T + 0.5)
       - Upper threshold ('>=T'): P = 1.0 - F(T - 0.5)
    2. Unit conversion mapping:
       - Denver (KDEN): Integrates Fahrenheit integer bins via exact Celsius boundary transformation:
         T_C = (T_F +/- 0.5 - 32) * 5/9.
       - Shanghai (ZSPD): 1°C integer bins.
    3. Probability Simplex normalization:
       - Guarantees sum(P_i) == 1.000000 across exhaustive market partitions.
    4. Settlement resolution:
       - Evaluates Wunderground observed ground truth temperature y to uniquely determine the winning YES bin.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fahrenheit_to_celsius(f_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert temperature from Fahrenheit to Celsius."""
    return (np.asarray(f_val, dtype=np.float64) - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert temperature from Celsius to Fahrenheit."""
    return (np.asarray(c_val, dtype=np.float64) * 9.0 / 5.0) + 32.0


@dataclass
class MarketBin:
    """Represents a single Polymarket discrete temperature contract bin."""

    bin_id: str
    bin_type: str  # 'exact', 'range', 'lte', 'gte'
    label: str
    unit: str = "C"  # 'C' or 'F'
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    probability: float = 0.0

    def get_celsius_bounds(self) -> Tuple[float, float]:
        """Convert bin boundaries into continuous Celsius [lower_c, upper_c) with half-degree continuity correction."""
        b_type = self.bin_type.lower()
        is_f = (self.unit.upper() == "F")

        if b_type in ["exact", "=t", "eq"]:
            val = float(self.value)
            return (float(fahrenheit_to_celsius(val - 0.5)), float(fahrenheit_to_celsius(val + 0.5))) if is_f else (val - 0.5, val + 0.5)

        elif b_type in ["range", "interval"]:
            lo, hi = float(self.low), float(self.high)
            return (float(fahrenheit_to_celsius(lo - 0.5)), float(fahrenheit_to_celsius(hi + 0.5))) if is_f else (lo - 0.5, hi + 0.5)

        elif b_type in ["lte", "<=t", "le"]:
            val = float(self.value)
            return (-np.inf, float(fahrenheit_to_celsius(val + 0.5))) if is_f else (-np.inf, val + 0.5)

        elif b_type in ["gte", ">=t", "ge"]:
            val = float(self.value)
            return (float(fahrenheit_to_celsius(val - 0.5)), np.inf) if is_f else (val - 0.5, np.inf)

        raise ValueError(f"Unknown bin_type '{self.bin_type}'")

    def contains(self, observed_temp: float, unit: str = "C") -> bool:
        """Check if an observed temperature falls into this bin's continuous interval."""
        obs_c = float(fahrenheit_to_celsius(observed_temp)) if unit.upper() == "F" else float(observed_temp)
        low_c, high_c = self.get_celsius_bounds()

        if np.isneginf(low_c):
            return obs_c < high_c
        if np.isposinf(high_c):
            return obs_c >= low_c
        return (low_c <= obs_c < high_c)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bin to dictionary."""
        return {
            "bin_id": self.bin_id,
            "bin_type": self.bin_type,
            "label": self.label,
            "unit": self.unit,
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "probability": float(self.probability),
        }


class BinConverter:
    """Converts continuous predictive distributions into discrete Polymarket market probabilities."""

    @staticmethod
    def calculate_bin_probabilities(
        distribution: Any,
        bins: List[MarketBin],
        normalize: bool = True,
    ) -> List[MarketBin]:
        """Integrate continuous CDF across all bin boundaries and optionally normalize to simplex."""
        evaluated_bins: List[MarketBin] = []
        raw_probs: List[float] = []

        for b in bins:
            low_c, high_c = b.get_celsius_bounds()
            cdf_high = 1.0 if np.isposinf(high_c) else float(distribution.cdf(high_c))
            cdf_low = 0.0 if np.isneginf(low_c) else float(distribution.cdf(low_c))
            p = max(0.0, cdf_high - cdf_low)
            raw_probs.append(p)

            evaluated_bins.append(MarketBin(
                bin_id=b.bin_id,
                bin_type=b.bin_type,
                label=b.label,
                unit=b.unit,
                value=b.value,
                low=b.low,
                high=b.high,
                probability=p,
            ))

        if normalize and len(evaluated_bins) > 0:
            total_p = sum(raw_probs)
            if total_p > 1e-12:
                for b in evaluated_bins:
                    b.probability = float(b.probability / total_p)

        return evaluated_bins

    @staticmethod
    def _create_single_bin(idx: int, val: int, total_bins: int, unit: str) -> MarketBin:
        """Helper to create a single MarketBin in the standard layout."""
        if idx == 0:
            return MarketBin(bin_id=f"bin_{idx}", bin_type="lte", label=f"≤{val}°{unit}", unit=unit, value=float(val))
        elif idx == total_bins - 1:
            return MarketBin(bin_id=f"bin_{idx}", bin_type="gte", label=f"≥{val}°{unit}", unit=unit, value=float(val))
        return MarketBin(bin_id=f"bin_{idx}", bin_type="exact", label=f"{val}°{unit}", unit=unit, value=float(val))

    @classmethod
    def generate_bins(
        cls,
        station_id: str = "ZSPD",
        center_temp: float = 20.0,
        spread: float = 2.0,
        bin_width: float = 1.0,
        num_bins: int = 7,
    ) -> List[MarketBin]:
        """Generate a complete set of mutually exclusive, collectively exhaustive standard market bins."""
        is_f = (station_id.upper() == "KDEN")
        unit = "F" if is_f else "C"

        if is_f:
            center_val = round(float(celsius_to_fahrenheit(center_temp)) if center_temp < 45.0 else center_temp)
        else:
            center_val = round(center_temp)

        start_val = center_val - ((num_bins - 1) // 2)
        return [cls._create_single_bin(i, start_val + i, num_bins, unit) for i in range(num_bins)]

    @staticmethod
    def determine_winning_bin(
        bins: List[MarketBin],
        observed_temp: float,
        unit: str = "C",
    ) -> Tuple[int, MarketBin]:
        """Find the index and MarketBin that settles as YES (1) for the observed ground truth."""
        for idx, b in enumerate(bins):
            if b.contains(observed_temp=observed_temp, unit=unit):
                return idx, b

        if len(bins) > 0:
            low_c, _ = bins[0].get_celsius_bounds()
            obs_c = float(fahrenheit_to_celsius(observed_temp)) if unit.upper() == "F" else float(observed_temp)
            return (0, bins[0]) if obs_c < low_c else (len(bins) - 1, bins[-1])

        raise ValueError("No bins provided for settlement determination")

    @staticmethod
    def to_dataframe(bins: List[MarketBin]) -> pd.DataFrame:
        """Convert a list of MarketBins to a structured pandas DataFrame."""
        return pd.DataFrame([b.to_dict() for b in bins])
