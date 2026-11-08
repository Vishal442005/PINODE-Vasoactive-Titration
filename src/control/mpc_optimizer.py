"""
mpc_optimizer.py: Model Predictive Control (MPC) with PINODE predictive core and safety guardrails.

Optimizes continuous-time vasoactive drug infusion rate over a 15-minute lookahead horizon
to maintain Mean Arterial Pressure strictly within the 65–75 mmHg target perfusion corridor.
"""

from typing import Dict, Optional, Tuple
import torch
import numpy as np
from scipy.optimize import minimize

from .safety_guardrails import SafetyGuardrails
from .clinician_fallback import ClinicianFallbackEngine
from ..models.pinode_core import PINODEVasoCore


class ModelPredictiveController:
    """Receding-horizon Model Predictive Controller for vasoactive infusion titration."""

    def __init__(
        self,
        pinode_model: PINODEVasoCore,
        target_map: float = 68.0,
        horizon_minutes: float = 15.0,
        control_step_seconds: float = 10.0,
        w_tracking: float = 1.0,
        w_dose: float = 0.05,
        w_slew: float = 0.30,
        w_uncertainty: float = 0.50,
        patient_weight_kg: float = 75.0,
    ) -> None:
        self.model = pinode_model
        self.target_map = target_map
        self.horizon_min = horizon_minutes
        self.dt_sec = control_step_seconds
        self.dt_min = control_step_seconds / 60.0
        self.weight = patient_weight_kg

        # Cost weights
        self.w_track = w_tracking
        self.w_dose = w_dose
        self.w_slew = w_slew
        self.w_unc = w_uncertainty

        # Safety modules
        self.guardrails = SafetyGuardrails()
        self.fallback = ClinicianFallbackEngine()

        self.last_applied_dose = 0.08  # Initial default basal rate

    def compute_optimal_infusion_step(
        self,
        current_state: torch.Tensor,
        z_vasc: torch.Tensor,
        covariates: torch.Tensor,
        map_base: torch.Tensor,
        signal_quality: float = 0.95,
        num_control_blocks: int = 3,
    ) -> Dict[str, float]:
        """
        Solves receding horizon optimization over the 15-minute horizon.

        Args:
            current_state: Tensor shape (1, 3) representing [Cp(t), Ce(t), MAP(t)].
            z_vasc: Tensor shape (1, 32) representing current vascular resistance embedding.
            covariates: Tensor shape (1, 6) patient covariates.
            map_base: Tensor shape (1, 1) baseline MAP.
            signal_quality: SQI score in [0, 1].
            num_control_blocks: Number of piecewise-constant dose segments over horizon.

        Returns:
            Dictionary with:
                - optimal_dose_ug_kg_min: Recommended dose for the next step.
                - predicted_map_horizon: Forecasted MAP at end of horizon.
                - uncertainty_std: Epistemic uncertainty std (mmHg).
                - is_safe: Safety status boolean.
                - status: Explanatory message.
        """
        device = current_state.device
        block_dur = self.horizon_min / num_control_blocks
        t_span = torch.linspace(0, self.horizon_min, 10, device=device)

        # Objective function for candidate block doses
        def cost_fn(block_doses: np.ndarray) -> float:
            total_cost = 0.0
            prev_u = self.last_applied_dose

            # Mean dose across blocks for fast ODE trajectory simulation
            mean_dose_ug_kg = float(np.mean(block_doses))
            u_total_ug_min = torch.tensor([[mean_dose_ug_kg * self.weight]], device=device, dtype=torch.float32)

            with torch.no_grad():
                traj = self.model.forward_trajectory(
                    current_state, t_span, u_total_ug_min, z_vasc, covariates, map_base
                )
                pred_maps = traj[:, 0, 2].cpu().numpy()

            # Tracking penalty: (MAP - 68)^2
            tracking_penalty = np.mean((pred_maps - self.target_map) ** 2)

            # Dose magnitude penalty
            dose_penalty = np.mean(block_doses ** 2)

            # Slew rate penalty
            slew_penalty = (block_doses[0] - prev_u) ** 2
            for k in range(1, len(block_doses)):
                slew_penalty += (block_doses[k] - block_doses[k - 1]) ** 2

            total_cost = (
                self.w_track * tracking_penalty
                + self.w_dose * dose_penalty
                + self.w_slew * slew_penalty
            )
            return float(total_cost)

        # Bounds for each block dose: [0.0, 1.50] ug/kg/min
        bounds = [(0.0, 1.50) for _ in range(num_control_blocks)]
        init_guess = np.full(num_control_blocks, self.last_applied_dose)

        # Fast L-BFGS-B bounded optimization
        res = minimize(cost_fn, init_guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": 25})
        raw_recommended_dose = float(res.x[0]) if res.success else self.last_applied_dose

        # Evaluate model uncertainty via Monte Carlo dropout
        u_rec_tensor = torch.tensor([[raw_recommended_dose * self.weight]], device=device, dtype=torch.float32)
        mean_traj, std_traj = self.model.predict_with_uncertainty(
            current_state, t_span, u_rec_tensor, z_vasc, covariates, map_base, num_mc_samples=8
        )
        pred_map_end = float(mean_traj[-1, 0, 2].item())
        uncertainty_std = float(std_traj[-1, 0, 2].item())

        # Validate with Safety Guardrails
        validated_dose, is_safe, safety_msg = self.guardrails.validate_and_clamp_dose(
            proposed_dose=raw_recommended_dose,
            previous_dose=self.last_applied_dose,
            dt_minutes=self.dt_min,
            uncertainty_std=uncertainty_std,
            signal_quality=signal_quality,
        )

        # Check clinician fallback
        final_pump_dose, in_fallback, fallback_msg = self.fallback.process_safety_status(
            is_safe=is_safe,
            current_ai_dose=validated_dose,
            dt_minutes=self.dt_min,
        )

        self.last_applied_dose = final_pump_dose

        return {
            "optimal_dose_ug_kg_min": round(final_pump_dose, 4),
            "predicted_map_end_mmhg": round(pred_map_end, 2),
            "uncertainty_std_mmhg": round(uncertainty_std, 2),
            "is_safe": 1.0 if is_safe else 0.0,
            "in_fallback": 1.0 if in_fallback else 0.0,
            "status_message": fallback_msg if in_fallback else safety_msg,
        }
