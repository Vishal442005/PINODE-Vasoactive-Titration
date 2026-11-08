"""
safety_guardrails.py: Clinical safety constraints and guardrails for automated vasopressor infusion.

Enforces:
- Hard dosage floor (0.0 ug/kg/min) and ceiling (1.50 ug/kg/min)
- Slew-rate limiting (|du/dt| <= 0.05 ug/kg/min per minute)
- Conformal epistemic uncertainty bound checking (sigma_MC <= 6.0 mmHg)
- Signal Quality Index (SQI) verification (SQI >= 0.65)
"""

from typing import Dict, Tuple
import numpy as np


class SafetyGuardrails:
    """Enforces strict ICU patient safety boundaries on autonomous drug titration."""

    def __init__(
        self,
        dose_min_ug_kg_min: float = 0.0,
        dose_max_ug_kg_min: float = 1.50,
        max_slew_rate_per_min: float = 0.05,
        max_uncertainty_std_mmhg: float = 6.0,
        min_sqi_threshold: float = 0.65,
        default_basal_dose: float = 0.08,
    ) -> None:
        self.dose_min = dose_min_ug_kg_min
        self.dose_max = dose_max_ug_kg_min
        self.max_slew = max_slew_rate_per_min
        self.max_sigma = max_uncertainty_std_mmhg
        self.min_sqi = min_sqi_threshold
        self.default_basal = default_basal_dose

    def validate_and_clamp_dose(
        self,
        proposed_dose: float,
        previous_dose: float,
        dt_minutes: float,
        uncertainty_std: float,
        signal_quality: float,
    ) -> Tuple[float, bool, str]:
        """
        Validates proposed dose against all physiological safety guardrails.

        Args:
            proposed_dose: AI-recommended infusion rate (ug/kg/min).
            previous_dose: Currently active infusion rate (ug/kg/min).
            dt_minutes: Time elapsed since last dose adjustment (minutes).
            uncertainty_std: Epistemic standard deviation sigma_MC (mmHg).
            signal_quality: Composite SQI of current biosignal window [0, 1].

        Returns:
            Tuple of:
                - safe_dose: Verified, rate-limited, and clamped infusion rate.
                - is_safe: True if proposed dose passed without critical safety violation.
                - status_message: Diagnostic explanation of any intervention.
        """
        # Check 1: Signal Quality Index check
        if signal_quality < self.min_sqi:
            return (
                previous_dose,
                False,
                f"REJECTED: Low Signal Quality (SQI={signal_quality:.2f} < {self.min_sqi}). Retaining current rate.",
            )

        # Check 2: Epistemic uncertainty check
        if uncertainty_std > self.max_sigma:
            return (
                previous_dose,
                False,
                f"REJECTED: High Model Uncertainty (sigma={uncertainty_std:.2f} mmHg > {self.max_sigma}). Retaining current rate.",
            )

        # Check 3: Slew-rate limiting
        max_delta = self.max_slew * max(dt_minutes, 1.0 / 60.0)  # at least 1-second step
        delta = proposed_dose - previous_dose
        if abs(delta) > max_delta:
            clamped_delta = np.clip(delta, -max_delta, max_delta)
            rate_limited_dose = previous_dose + clamped_delta
            msg = f"SLEW_LIMITED: Delta {delta:.3f} exceeded max allowable {max_delta:.3f}. Clamped."
        else:
            rate_limited_dose = proposed_dose
            msg = "NOMINAL: Passed all safety bounds."

        # Check 4: Hard dosage floor and ceiling
        final_dose = float(np.clip(rate_limited_dose, self.dose_min, self.dose_max))

        return final_dose, True, msg
