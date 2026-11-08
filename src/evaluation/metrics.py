"""
metrics.py: Clinical, hemodynamic, and engineering evaluation metrics for PINODE-Vaso.

Implements standard benchmark criteria across:
- Regression Accuracy: RMSE, MAE, MAPE, Pearson r
- Clinical Blood Pressure Standards: AAMI (mean error <= 5 mmHg, SD <= 8 mmHg), BHS Grade (A/B/C)
- Hemodynamic Perfusion Targets: Time-in-Target Range (TITR 65–75 mmHg), Severe Hypotension (<55 mmHg), Area Under Threshold (AUT)
- Safety & Guardrail Compliance: Unsafe Dose Escalation Rate, Physiological Violations
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes trajectory regression errors: RMSE, MAE, MAPE, Pearson r.
    """
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / np.clip(y_true, 1e-3, None))) * 100.0)

    if len(y_true) > 1 and np.std(y_true) > 1e-5 and np.std(y_pred) > 1e-5:
        r_val, _ = stats.pearsonr(y_true, y_pred)
    else:
        r_val = 1.0

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 2),
        "pearson_r": round(float(r_val), 3),
    }


def evaluate_aami_standard(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Evaluates compliance with the Association for the Advancement of Medical Instrumentation (AAMI) standard:
    Mean Error <= 5.0 mmHg and Standard Deviation <= 8.0 mmHg.
    """
    errors = y_pred - y_true
    mean_err = float(np.mean(errors))
    sd_err = float(np.std(errors, ddof=1))
    passed = abs(mean_err) <= 5.0 and sd_err <= 8.0

    return {
        "mean_error": round(mean_err, 3),
        "sd_error": round(sd_err, 3),
        "aami_pass": bool(passed),
    }


def evaluate_bhs_grade(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Evaluates compliance with British Hypertension Society (BHS) Standard:
    Grade A: >= 60% <= 5 mmHg, >= 85% <= 10 mmHg, >= 95% <= 15 mmHg
    Grade B: >= 50% <= 5 mmHg, >= 75% <= 10 mmHg, >= 90% <= 15 mmHg
    Grade C: >= 40% <= 5 mmHg, >= 65% <= 10 mmHg, >= 85% <= 15 mmHg
    """
    abs_errors = np.abs(y_pred - y_true)
    pct_5 = float(np.mean(abs_errors <= 5.0) * 100.0)
    pct_10 = float(np.mean(abs_errors <= 10.0) * 100.0)
    pct_15 = float(np.mean(abs_errors <= 15.0) * 100.0)

    if pct_5 >= 60.0 and pct_10 >= 85.0 and pct_15 >= 95.0:
        grade = "Grade A"
    elif pct_5 >= 50.0 and pct_10 >= 75.0 and pct_15 >= 90.0:
        grade = "Grade B"
    elif pct_5 >= 40.0 and pct_10 >= 65.0 and pct_15 >= 85.0:
        grade = "Grade C"
    else:
        grade = "Grade D / Fail"

    return {
        "pct_le_5mmHg": round(pct_5, 1),
        "pct_le_10mmHg": round(pct_10, 1),
        "pct_le_15mmHg": round(pct_15, 1),
        "bhs_grade": grade,
    }


def compute_hemodynamic_stability(
    map_trajectory: np.ndarray,
    target_low: float = 65.0,
    target_high: float = 75.0,
    severe_thresh: float = 55.0,
    hypertensive_thresh: float = 90.0,
    dt_min: float = 1.0 / 6.0,  # 10 seconds in minutes
) -> Dict[str, float]:
    """
    Evaluates clinical resuscitation stability:
    - Time-in-Target Range (TITR: 65–75 mmHg) [%]
    - Severe Hypotension Exposure (<55 mmHg) [%]
    - Area Under Threshold (AUT: mmHg * min)
    - Hypertensive Overshoot (>90 mmHg) [%]
    """
    n_steps = len(map_trajectory)
    if n_steps == 0:
        return {"titr_pct": 0.0, "severe_hypo_pct": 0.0, "aut_mmHg_min": 0.0, "overshoot_pct": 0.0}

    in_target = (map_trajectory >= target_low) & (map_trajectory <= target_high)
    titr_pct = float(np.mean(in_target) * 100.0)

    severe_hypo = map_trajectory < severe_thresh
    severe_hypo_pct = float(np.mean(severe_hypo) * 100.0)

    overshoot = map_trajectory > hypertensive_thresh
    overshoot_pct = float(np.mean(overshoot) * 100.0)

    # Area Under Threshold (accumulated debt below 65 mmHg)
    hypo_deficit = np.maximum(0.0, target_low - map_trajectory)
    aut = float(np.sum(hypo_deficit) * dt_min)

    return {
        "titr_pct": round(titr_pct, 2),
        "severe_hypo_pct": round(severe_hypo_pct, 2),
        "aut_mmHg_min": round(aut, 2),
        "overshoot_pct": round(overshoot_pct, 2),
    }


def compute_safety_violations(
    infusion_trajectory: np.ndarray,
    map_trajectory: np.ndarray,
    max_slew_rate: float = 0.05,  # ug/kg/min^2
    dt_min: float = 1.0 / 6.0,    # 10s step
) -> Dict[str, float]:
    """
    Evaluates unsafe clinician or AI drug escalations and physiological plausibility violations.
    """
    if len(infusion_trajectory) < 2:
        return {"unsafe_escalation_pct": 0.0, "physio_violations_pct": 0.0}

    # Slew-rate check (rate of change per minute)
    dose_diffs = np.abs(np.diff(infusion_trajectory)) / dt_min
    unsafe_steps = dose_diffs > max_slew_rate
    unsafe_pct = float(np.mean(unsafe_steps) * 100.0)

    # Physiological plausibility checks:
    # 1. MAP out of absolute physiological range [20, 260] mmHg
    # 2. Impossible instantaneous MAP derivative |dMAP/dt| > 30 mmHg/min
    map_rate = np.abs(np.diff(map_trajectory)) / dt_min
    deriv_violations = map_rate > 30.0
    bound_violations = (map_trajectory < 20.0) | (map_trajectory > 260.0)
    total_violations = np.sum(deriv_violations) + np.sum(bound_violations[1:])
    violation_pct = float((total_violations / (len(map_trajectory) - 1)) * 100.0)

    return {
        "unsafe_escalation_pct": round(unsafe_pct, 2),
        "physio_violations_pct": round(violation_pct, 2),
    }
