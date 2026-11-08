"""
test_evaluation.py: Comprehensive test suite for metrics, off-policy evaluation, and noise stress tests.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_regression_metrics,
    evaluate_aami_standard,
    evaluate_bhs_grade,
    compute_hemodynamic_stability,
    compute_safety_violations,
)
from src.evaluation.off_policy_eval import OffPolicyEvaluator
from src.evaluation.noise_stress_test import NoiseStressTester


def test_regression_metrics():
    y_true = np.array([65.0, 70.0, 75.0, 68.0, 72.0])
    y_pred = np.array([66.0, 69.0, 74.0, 67.0, 73.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["rmse"] == 1.0
    assert metrics["mae"] == 1.0
    assert metrics["mape"] > 0.0
    assert metrics["pearson_r"] > 0.90


def test_aami_standard():
    # Compliant case (mean err <= 5, sd <= 8)
    y_true = np.ones(50) * 70.0
    y_pred = y_true + np.random.normal(1.5, 2.0, 50)
    res = evaluate_aami_standard(y_true, y_pred)
    assert res["aami_pass"] is True

    # Failing case
    y_bad = y_true + 12.0
    res_bad = evaluate_aami_standard(y_true, y_bad)
    assert res_bad["aami_pass"] is False


def test_bhs_grade():
    y_true = np.ones(100) * 70.0
    # Small errors within 5 mmHg -> Grade A
    y_pred = y_true + np.random.normal(0, 2.0, 100)
    res = evaluate_bhs_grade(y_true, y_pred)
    assert res["bhs_grade"] == "Grade A"
    assert res["pct_le_5mmHg"] >= 60.0


def test_hemodynamic_stability():
    # 10 steps, 8 in target [65, 75], 1 severe (<55), 1 overshoot (>90)
    traj = np.array([68.0, 70.0, 72.0, 66.0, 74.0, 69.0, 71.0, 67.0, 50.0, 95.0])
    res = compute_hemodynamic_stability(traj)
    assert res["titr_pct"] == 80.0
    assert res["severe_hypo_pct"] == 10.0
    assert res["overshoot_pct"] == 10.0
    assert res["aut_mmHg_min"] > 0.0


def test_safety_violations():
    # Rapid dose jump: 0.05 to 0.40 in 10s (rate = (0.35) / (1/6) = 2.1 ug/kg/min^2 >> 0.05)
    infusion = np.array([0.05, 0.40, 0.40])
    map_traj = np.array([70.0, 70.0, 70.0])
    res = compute_safety_violations(infusion, map_traj)
    assert res["unsafe_escalation_pct"] > 0.0


def test_off_policy_wis():
    evaluator = OffPolicyEvaluator(random_seed=42)
    trajectories = []
    for _ in range(20):
        T = 15
        trajectories.append({
            "rewards": np.ones(T) * 10.0,
            "target_probs": np.ones(T) * 0.8,
            "behavior_probs": np.ones(T) * 0.5,
        })
    res = evaluator.compute_wis(trajectories, n_bootstraps=50)
    assert res["wis_mean"] > 0.0
    assert res["ci_lower"] <= res["wis_mean"] <= res["ci_upper"]


def test_noise_stress_tester():
    tester = NoiseStressTester(fs=125.0)
    t = np.linspace(0, 4, 500)
    clean_abp = 80.0 + 20.0 * np.sin(2 * np.pi * 1.2 * t)
    clean_ppg = 0.5 + 0.3 * np.sin(2 * np.pi * 1.2 * t)

    # Test noise injection modes
    for mode in ["gaussian", "baseline_wander", "motion_burst", "catheter_flush", "pulse_dropout"]:
        corrupted = tester.inject_noise(clean_abp, mode=mode, severity=0.2)
        assert len(corrupted) == len(clean_abp)

    # Test stress benchmark
    res = tester.run_stress_benchmark(clean_abp, clean_ppg, noise_levels=[0.0, 0.25])
    assert 0.0 in res["fallback_triggered"]
    assert 0.25 in res["fallback_triggered"]
