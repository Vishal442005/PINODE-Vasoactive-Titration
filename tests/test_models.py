"""
Unit and gradient integration tests for deep learning and ODE models:
- morphomic_cnn.py
- pk_compartments.py
- pinode_core.py
- sensitivity.py
"""

import pytest
import torch
import numpy as np

from src.models.morphomic_cnn import MorphomicWaveformEncoder
from src.models.pk_compartments import TwoCompartmentHillPKPD
from src.models.pinode_core import PINODEVasoCore
from src.models.sensitivity import GlobalSensitivityAnalyzer


class TestModels:
    """Test suite ensuring deep learning and ODE mathematical integrity."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_morphomic_encoder_shapes_and_params(self, device):
        encoder = MorphomicWaveformEncoder(in_channels=2, latent_dim=32, dropout=0.15).to(device)

        # Count parameters
        num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert num_params < 85000, f"Encoder too large: {num_params} params"

        # Test forward pass with 30s window (3750 samples at 125 Hz)
        batch_size = 4
        dummy_input = torch.randn(batch_size, 2, 3750, device=device)
        z_vasc = encoder(dummy_input)

        assert z_vasc.shape == (batch_size, 32)
        assert not torch.isnan(z_vasc).any()

        # Test backpropagation
        loss = z_vasc.sum()
        loss.backward()
        assert encoder.init_conv[0].weight.grad is not None

    def test_pk_pd_compartments(self, device):
        pkpd = TwoCompartmentHillPKPD(default_ke0=0.35, num_covariates=6).to(device)

        batch_size = 4
        covariates = torch.tensor([
            [75.0, 62.0, 6.0, 1.4, 0.9, 2.8],
            [60.0, 45.0, 2.0, 0.8, 0.4, 1.2],
            [90.0, 78.0, 11.0, 2.6, 2.1, 5.4],
            [82.0, 55.0, 4.0, 1.1, 0.6, 1.9],
        ], device=device)

        cp = torch.full((batch_size, 1), 0.5, device=device)
        ce = torch.full((batch_size, 1), 0.3, device=device)
        u_ug_min = torch.full((batch_size, 1), 10.0, device=device)

        d_cp, d_ce = pkpd.compute_pk_derivatives(cp, ce, u_ug_min, covariates)
        assert d_cp.shape == (batch_size, 1)
        assert d_ce.shape == (batch_size, 1)
        assert not torch.isnan(d_cp).any()
        assert not torch.isnan(d_ce).any()

        # Test Hill PD effect monotonicity
        ce_low = torch.tensor([[0.05]], device=device)
        ce_high = torch.tensor([[1.50]], device=device)
        eff_low = pkpd.compute_pharmacodynamic_effect(ce_low)
        eff_high = pkpd.compute_pharmacodynamic_effect(ce_high)
        assert eff_high.item() > eff_low.item()
        assert 10.0 <= eff_high.item() <= 80.0

    def test_pinode_core_trajectory_and_adjoint(self, device):
        pinode = PINODEVasoCore(
            latent_dim=32,
            num_covariates=6,
            hidden_dim=32,
            solver="rk4",
            use_adjoint=True,
            dropout=0.20,
        ).to(device)

        batch_size = 2
        # Initial state [Cp0, Ce0, MAP0]
        s0 = torch.tensor([[0.1, 0.05, 62.0], [0.0, 0.0, 58.0]], device=device)
        t_span = torch.tensor([0.0, 1.0, 2.0], device=device)  # minutes
        infusion = torch.tensor([[8.0], [12.0]], device=device)
        z_vasc = torch.randn(batch_size, 32, device=device)
        covs = torch.randn(batch_size, 6, device=device)
        map_base = torch.tensor([[75.0], [70.0]], device=device)

        # Forward trajectory integration
        traj = pinode.forward_trajectory(s0, t_span, infusion, z_vasc, covs, map_base)
        assert traj.shape == (len(t_span), batch_size, 3)
        assert not torch.isnan(traj).any()

        # Test backward pass through adjoint solver
        loss = traj[-1, :, 2].sum()  # Loss on final MAP
        loss.backward()
        assert pinode.neural_drift.net[0].weight.grad is not None

    def test_pinode_monte_carlo_uncertainty(self, device):
        pinode = PINODEVasoCore(
            latent_dim=32,
            num_covariates=6,
            hidden_dim=32,
            use_adjoint=False,
            dropout=0.25,
        ).to(device)

        s0 = torch.tensor([[0.0, 0.0, 60.0]], device=device)
        t_span = torch.linspace(0, 5, 5, device=device)
        infusion = torch.tensor([[10.0]], device=device)
        z_vasc = torch.randn(1, 32, device=device)
        covs = torch.randn(1, 6, device=device)
        map_base = torch.tensor([[70.0]], device=device)

        mean_traj, std_traj = pinode.predict_with_uncertainty(
            s0, t_span, infusion, z_vasc, covs, map_base, num_mc_samples=5
        )

        assert mean_traj.shape == (len(t_span), 1, 3)
        assert std_traj.shape == (len(t_span), 1, 3)
        # Uncertainty std must be non-negative
        assert (std_traj >= 0.0).all()

    def test_global_sensitivity_analyzer(self):
        analyzer = GlobalSensitivityAnalyzer(sample_size=32, random_seed=42)
        results = analyzer.compute_sobol_indices(t_eval_min=15.0, infusion_rate=0.10)

        assert len(results) == 7
        for name, idxs in results.items():
            assert "S1" in idxs and "ST" in idxs
            assert 0.0 <= idxs["S1"] <= 1.5
            assert 0.0 <= idxs["ST"] <= 2.0
