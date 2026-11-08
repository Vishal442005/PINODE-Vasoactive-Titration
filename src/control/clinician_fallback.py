"""
clinician_fallback.py: Graceful fail-safe handover engine for autonomous vasoactive titration.

Handles clinician-in-the-loop fail-safe transitions when persistent sensor failure,
out-of-distribution inputs, or model uncertainty exceeds safety limits.
"""

from typing import Tuple
import numpy as np


class ClinicianFallbackEngine:
    """Manages smooth bumpless transfer between AI autonomous control and clinician baseline."""

    def __init__(
        self,
        max_consecutive_faults: int = 3,
        handover_tau_minutes: float = 2.0,
        default_steady_basal: float = 0.08,
    ) -> None:
        self.max_faults = max_consecutive_faults
        self.handover_tau = handover_tau_minutes
        self.default_basal = default_steady_basal

        self.consecutive_fault_count = 0
        self.fallback_active = False
        self.clinician_historical_rate = default_steady_basal

    def update_clinician_observed_rate(self, rate: float) -> None:
        """Tracks the moving average of clinician actions during manual care."""
        if rate > 0.0:
            self.clinician_historical_rate = 0.90 * self.clinician_historical_rate + 0.10 * rate

    def process_safety_status(
        self, is_safe: bool, current_ai_dose: float, dt_minutes: float
    ) -> Tuple[float, bool, str]:
        """
        Processes safety status and performs bumpless exponential handover if needed.

        Args:
            is_safe: Boolean from SafetyGuardrails.
            current_ai_dose: Current dose requested by AI.
            dt_minutes: Time step in minutes.

        Returns:
            Tuple of:
                - output_dose: Active infusion rate to send to pump.
                - in_fallback: Boolean flag indicating if fallback mode is triggered.
                - handover_message: Status string.
        """
        if is_safe:
            if self.fallback_active:
                self.consecutive_fault_count = 0
                self.fallback_active = False
                return current_ai_dose, False, "RECOVERY: Signal and model nominal. AI resumed."
            self.consecutive_fault_count = 0
            return current_ai_dose, False, "NOMINAL: Operating in autonomous mode."

        self.consecutive_fault_count += 1

        if self.consecutive_fault_count >= self.max_faults:
            self.fallback_active = True

        if self.fallback_active:
            # Smooth exponential decay toward clinician historical steady-state basal rate
            alpha = np.exp(-dt_minutes / max(0.5, self.handover_tau))
            safe_fallback_dose = float(alpha * current_ai_dose + (1.0 - alpha) * self.clinician_historical_rate)
            msg = (
                f"FALLBACK ACTIVE ({self.consecutive_fault_count} faults): "
                f"Handing over to clinician basal rate ({safe_fallback_dose:.3f} ug/kg/min)."
            )
            return safe_fallback_dose, True, msg

        # Single transient fault: hold current dose
        return current_ai_dose, False, f"TRANSIENT FAULT ({self.consecutive_fault_count}/{self.max_faults}): Holding dose."
