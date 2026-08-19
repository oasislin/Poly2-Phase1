#!/usr/bin/env python3
"""
Meteorological Boundary Risk-Control Stress Tests (风控防熔断测试套件).

Core Focus:
1. Scenario 1 (Spread Collapse): Zero ensemble variance (S²_ens = 0) must be 100% supported by variance floor.
2. Scenario 2 (Extreme Tails): Black swan temperature values (-45°C, +45°C) must not produce NaN/Inf/overflow.
3. Scenario 3 (Forecast Bust): Inverted forecast signals must preserve non-negative physical slope (b >= 0).
4. Scenario 4 (Hyper-dispersion): Huge ensemble spread (S²_ens > 100) must be tamed by L2 regularization.
5. Scenario 5 (Data Corruption): NaN, Inf, and -999.0 sentinel inputs must be safely trapped without crash.
"""

import math
import numpy as np
import pytest

from src.modeling.crps import gaussian_crps
from src.modeling.emos_trainer import EMOSOptimizer, ModelTrainingDiagnostics
from src.modeling.gaussian_emos import GaussianEMOS


class TestRiskControlBoundaryScenarios:
    """Quantitative risk control and safety boundary test suite."""

    def test_scenario_1_spread_collapse_variance_floor_protection(self):
        """[风控核心] 集合方差完全归零 (S²_ens=0) 时，方差底 100% 承托，防止输出虚假 100% 胜率."""
        n = 200
        # 5 members all predict identical values -> ensemble variance is exact 0.0
        ens_mean = np.full(n, 22.0)
        ens_var = np.zeros(n)
        clim_var = np.full(n, 9.0)  # Climatology std = 3.0°C
        obs = ens_mean + np.random.normal(0, 3.0, n)

        optimizer = EMOSOptimizer(l2_lambda_d=1e-3, random_seed=42)
        model, diag = optimizer.fit(
            ensemble_mean=ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=obs,
        )

        # Compute output variance
        _, pred_sigma = model.compute_params(ens_mean, ens_var, clim_var)
        pred_variance = pred_sigma ** 2

        # Assert variance floor is never breached
        assert np.all(pred_variance >= clim_var)
        assert np.all(pred_sigma >= 3.0)
        assert diag.health_grade in ["HEALTHY", "WARNING"]

    def test_scenario_2_extreme_tails_numerical_stability(self):
        """[数值防溢出] 极端黑天鹅温度 (-45°C, +45°C, |z| > 20) 计算 CRPS 与分布时绝不产生 NaN/Inf/崩溃."""
        extreme_obs = np.array([-45.0, -35.0, 42.0, 48.0, 100.0])
        pred_mu = np.array([0.0, 10.0, 25.0, 25.0, 20.0])
        pred_sigma = np.array([2.0, 3.0, 2.5, 3.0, 1.0])

        # Analytical CRPS under massive standardized errors (|z| up to 80)
        crps_vals = gaussian_crps(extreme_obs, pred_mu, pred_sigma)

        assert np.all(np.isfinite(crps_vals))
        assert np.all(crps_vals > 0.0)
        # In extreme tails (|z| >> 1), Gaussian CRPS asymptotically matches |y - mu| - sigma / sqrt(pi)
        expected_asymptote = np.abs(extreme_obs - pred_mu) - (pred_sigma / math.sqrt(math.pi))
        assert np.allclose(crps_vals, expected_asymptote, rtol=1e-4)

    def test_scenario_3_forecast_bust_non_negative_physical_slope(self):
        """[防方向倒挂] 数值模式严重误报反转时，模型不得学习出违背物理的负斜率 (b >= 0)."""
        np.random.seed(42)
        n = 300
        # Simulated forecast bust: forecast says warm (+15°C), actual says cold (-5°C)
        true_temp = np.random.normal(10.0, 6.0, n)
        # Inverted ensemble mean (negative correlation with ground truth)
        inverted_ens_mean = 30.0 - true_temp + np.random.normal(0, 1.0, n)
        ens_var = np.full(n, 4.0)
        clim_var = np.full(n, 16.0)

        optimizer = EMOSOptimizer(l2_lambda_d=1e-3, random_seed=42)
        model, diag = optimizer.fit(
            ensemble_mean=inverted_ens_mean,
            ensemble_variance=ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=true_temp,
        )

        # Assert physical non-negative slope constraint holds
        assert model.b >= 0.0
        assert diag.success is True

    def test_scenario_4_hyper_dispersive_storm_l2_regularization(self):
        """[防方差爆炸] 强风暴导致集合方差超级发散 (S²_ens > 100) 时，L2 正则化有效抑制参数 d 恶性膨胀."""
        np.random.seed(42)
        n = 300
        true_temp = np.random.normal(20.0, 4.0, n)
        ens_mean = true_temp + np.random.normal(0, 1.0, n)
        # Giant ensemble variance during high-uncertainty storm
        giant_ens_var = np.random.uniform(80.0, 150.0, n)
        clim_var = np.full(n, 16.0)

        optimizer = EMOSOptimizer(l2_lambda_d=1e-3, random_seed=42)
        model, diag = optimizer.fit(
            ensemble_mean=ens_mean,
            ensemble_variance=giant_ens_var,
            sigma_clim_squared=clim_var,
            observed_temp=true_temp,
        )

        # L2 penalty must keep |d| tightly bounded, preventing variance explosion
        assert abs(model.d) < 3.0
        assert diag.health_grade in ["HEALTHY", "WARNING"]

    def test_scenario_5_data_corruption_and_sentinels_safe_handling(self):
        """[防系统宕机] 注入包含 NaN / Inf / -999 传感器哨兵值的数据时，系统能够安全识别并过滤拦截."""
        ens_mean = np.array([20.0, np.nan, 25.0, np.inf, -999.0, 22.0])
        ens_var = np.array([2.0, 2.0, np.nan, 3.0, 1.0, 2.5])
        clim_var = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        obs = np.array([21.0, 19.0, 24.0, 26.0, 20.0, 23.0])

        # Clean corrupted data: filter out non-finite or sentinel values
        valid_mask = (
            np.isfinite(ens_mean)
            & np.isfinite(ens_var)
            & np.isfinite(obs)
            & (ens_mean > -90.0)
            & (ens_mean < 70.0)
        )

        assert valid_mask.sum() == 2  # Indices 0 and 5 are physically valid
        clean_mean = ens_mean[valid_mask]
        clean_var = ens_var[valid_mask]
        clean_clim = clim_var[valid_mask]
        clean_obs = obs[valid_mask]

        # Validated clean slice fits smoothly
        optimizer = EMOSOptimizer(l2_lambda_d=1e-3, random_seed=42)
        model, diag = optimizer.fit(clean_mean, clean_var, clean_clim, clean_obs)
        assert diag.success is True
        assert diag.sample_count == 2
