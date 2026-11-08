"""
Unit and integration tests for control subsystem:
- safety_guardrails.py
- clinician_fallback.py
- mpc_optimizer.py
"""

import pytest
import torch
import numpy as np

from src.control.safety_guardrails import SafetyGuardrails
from src.control.clinician_fallback import ClinicianFallbackEngine
from src.control.mpc_optimizer import ModelPredictiveController
from src.models.pinode_core import PINODEVasoCore


class TestControlSubsystem:
    """Test suite ensuring safety bounds, fail-safe handover, and MPC optimization."""

    def test_guardrails_ceiling_and_slew_rate(self):
        guard = SafetyGuardrails(
            dose_min_ug_kg_min=0.0,
            dose_max_ug_kg_min=1.50,
            max_slew_rate_per_min=0.05,
            max_uncertainty_std_mmhg=6.0,
            min_sqi_threshold=0.65,
        )

        prev_dose = 0.10
        # Extreme jump: propose 1.0 ug/kg/min in 1 minute
        safe_dose, is_safe, msg = guard.validate_and_clamp_dose(
            proposed_dose=1.0,
            previous_dose=prev_dose,
            dt_minutes=1.0,
            uncertainty_std=2.0,
            signal_quality=0.90,
        )
        assert is_safe is True
        # Must be clamped by max slew rate 0.05
        assert safe_dose == pytest.approx(0.15, abs=1e-3)
        assert "SLEW_LIMITED" in msg

        # Extreme high dose beyond ceiling
        safe_high, _, _ = guard.validate_and_clamp_dose(
            proposed_dose=5.0,
            previous_dose=1.48,
            dt_minutes=10.0,
            uncertainty_std=1.0,
            signal_quality=0.90,
        )
        assert safe_high <= 1.50

    def test_guardrails_rejects_uncertainty_and_sqi(self):
        guard = SafetyGuardrails()

        # Test high uncertainty
        dose_unc, safe_unc, msg_unc = guard.validate_and_clamp_dose(
            proposed_dose=0.30,
            previous_dose=0.10,
            dt_minutes=1.0,
            uncertainty_std=8.5,  # > 6.0
            signal_quality=0.90,
        )
        assert safe_unc is False
        assert dose_unc == 0.10  # Must hold previous dose
        assert "High Model Uncertainty" in msg_unc

        # Test poor signal quality
        dose_sqi, safe_sqi, msg_sqi = guard.validate_and_clamp_dose(
            proposed_dose=0.30,
            previous_dose=0.10,
            dt_minutes=1.0,
            uncertainty_std=2.0,
            signal_quality=0.40,  # < 0.65
        )
        assert safe_sqi is False
        assert dose_sqi == 0.10
        assert "Low Signal Quality" in msg_sqi

    def test_clinician_fallback_handover(self):
        engine = ClinicianFallbackEngine(max_consecutive_faults=3, default_steady_basal=0.08)

        # 1st fault -> transient hold
        d1, in_fb1, _ = engine.process_safety_status(is_safe=False, current_ai_dose=0.25, dt_minutes=0.5)
        assert in_fb1 is False

        # 2nd fault -> transient hold
        d2, in_fb2, _ = engine.process_safety_status(is_safe=False, current_ai_dose=0.25, dt_minutes=0.5)
        assert in_fb2 is False

        # 3rd fault -> triggers clinician fallback
        d3, in_fb3, msg3 = engine.process_safety_status(is_safe=False, current_ai_dose=0.25, dt_minutes=0.5)
        assert in_fb3 is True
        assert "FALLBACK ACTIVE" in msg3
        # Should smoothly decay toward 0.08 basal rate
        assert d3 < 0.25

        # Recovery test
        d_rec, in_fb_rec, msg_rec = engine.process_safety_status(is_safe=True, current_ai_dose=0.15, dt_minutes=0.5)
        assert in_fb_rec is False
        assert "RECOVERY" in msg_rec

    def test_mpc_optimizer_hypotension_response(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PINODEVasoCore(latent_dim=32, num_covariates=6, hidden_dim=32, solver="rk4", use_adjoint=False).to(device)

        mpc = ModelPredictiveController(
            pinode_model=model,
            target_map=68.0,
            horizon_minutes=10.0,
            control_step_seconds=10.0,
        )

        # Patient in acute hypotension (MAP = 55 mmHg)
        s_hypo = torch.tensor([[0.05, 0.02, 55.0]], device=device)
        z_vasc = torch.randn(1, 32, device=device)
        covs = torch.randn(1, 6, device=device)
        map_base = torch.tensor([[75.0]], device=device)

        step_result = mpc.compute_optimal_infusion_step(
            current_state=s_hypo,
            z_vasc=z_vasc,
            covariates=covs,
            map_base=map_base,
            signal_quality=0.90,
            num_control_blocks=2,
        )

        assert "optimal_dose_ug_kg_min" in step_result
        assert step_result["optimal_dose_ug_kg_min"] >= 0.0
        assert step_result["optimal_dose_ug_kg_min"] <= 1.50
        assert step_result["is_safe"] == 1.0
