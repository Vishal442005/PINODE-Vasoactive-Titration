"""
pinode_core.py: Physics-Informed Neural ODE (PINODE) for continuous-time hemodynamic prediction.

Couples 2-compartment Norepinephrine PK-PD equations with a neural autonomic drift field.
Supports Pontryagin's Adjoint Sensitivity Method (torchdiffeq.odeint_adjoint) for O(1) VRAM training.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .pk_compartments import TwoCompartmentHillPKPD


class HemodynamicNeuralDrift(nn.Module):
    """Residual neural network modeling unobserved physiological drift, sepsis vasodilation, and baroreflex."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_covariates: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        # Input: [norm_MAP, Ce, z_vasc (32), covariates (6)] -> 40 dimensions
        in_dim = 1 + 1 + latent_dim + num_covariates
        self.dropout_rate = dropout
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize last layer with small weights so physics dominates early training
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

        # Baroreceptor regulation gain (bounded positive)
        self.raw_lambda_baro = nn.Parameter(torch.tensor(0.05))

    @property
    def lambda_baro(self) -> torch.Tensor:
        """Baroreflex stabilization rate constant."""
        return 0.01 + 0.15 * torch.sigmoid(self.raw_lambda_baro)

    def forward(
        self,
        map_val: torch.Tensor,
        ce: torch.Tensor,
        z_vasc: torch.Tensor,
        covariates: torch.Tensor,
        map_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes neural drift d(MAP)/dt in mmHg/min.
        """
        # Normalize MAP around 68 mmHg setpoint
        norm_map = (map_val - 68.0) / 25.0
        x_in = torch.cat([norm_map, ce, z_vasc, covariates], dim=-1)
        f_nn = self.net(x_in)
        # Baroreflex restoration term: -lambda * (MAP - MAP_base)
        baro_term = -self.lambda_baro * (map_val - map_base)
        return f_nn + baro_term


class ODEVectorFieldWrapper(nn.Module):
    """Wraps the coupled vector field as an nn.Module for seamless adjoint sensitivity backprop."""

    def __init__(
        self,
        core: "PINODEVasoCore",
        infusion_rate_ug_min: torch.Tensor,
        z_vasc: torch.Tensor,
        covariates: torch.Tensor,
        map_base: torch.Tensor,
    ) -> None:
        super().__init__()
        self.core = core
        self.u = infusion_rate_ug_min
        self.z = z_vasc
        self.cov = covariates
        self.base = map_base

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.core.ode_func(t, state, self.u, self.z, self.cov, self.base)


class PINODEVasoCore(nn.Module):
    """
    Complete continuous-time Physics-Informed Neural ODE system for vasopressor titration.
    State s(t) = [Cp(t), Ce(t), MAP(t)] in R^3.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_covariates: int = 6,
        hidden_dim: int = 64,
        solver: str = "rk4",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = True,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        self.pk_pd = TwoCompartmentHillPKPD(num_covariates=num_covariates)
        self.neural_drift = HemodynamicNeuralDrift(
            latent_dim=latent_dim,
            num_covariates=num_covariates,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def ode_func(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        infusion_rate_ug_min: torch.Tensor,
        z_vasc: torch.Tensor,
        covariates: torch.Tensor,
        map_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the ODE system vector field ds/dt:
        dCp/dt = u/Vd - kel * Cp
        dCe/dt = ke0 * (Cp - Ce)
        dMAP/dt = Emax * Ce^gamma / (EC50^gamma + Ce^gamma) + f_drift(MAP, Ce, z_vasc, cov)
        """
        cp = state[:, 0:1]
        ce = state[:, 1:2]
        cur_map = state[:, 2:3]

        # 1. Pharmacokinetic derivatives
        d_cp, d_ce = self.pk_pd.compute_pk_derivatives(cp, ce, infusion_rate_ug_min, covariates)

        # 2. Pharmacodynamic Hill effect
        pd_effect = self.pk_pd.compute_pharmacodynamic_effect(ce)

        # 3. Neural drift and baroreflex
        drift = self.neural_drift(cur_map, ce, z_vasc, covariates, map_base)

        d_map = pd_effect + drift

        # Concatenate derivatives: shape (B, 3)
        return torch.cat([d_cp, d_ce, d_map], dim=-1)

    def forward_trajectory(
        self,
        initial_state: torch.Tensor,
        t_span_minutes: torch.Tensor,
        infusion_rate_ug_min: torch.Tensor,
        z_vasc: torch.Tensor,
        covariates: torch.Tensor,
        map_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrates the continuous-time trajectory from t0 to t_end.

        Args:
            initial_state: (Batch, 3) where [Cp0, Ce0, MAP0].
            t_span_minutes: 1D Tensor of integration timestamps in minutes.
            infusion_rate_ug_min: (Batch, 1) infusion rate.
            z_vasc: (Batch, 32) vascular resistance embedding.
            covariates: (Batch, 6) patient covariates.
            map_base: (Batch, 1) baseline MAP.

        Returns:
            Integrated trajectory of shape (Time, Batch, 3).
        """
        vector_field = ODEVectorFieldWrapper(
            self, infusion_rate_ug_min, z_vasc, covariates, map_base
        )

        integrate_fn = odeint_adjoint if self.use_adjoint else odeint

        trajectory = integrate_fn(
            vector_field,
            initial_state,
            t_span_minutes,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )
        return trajectory

    def predict_with_uncertainty(
        self,
        initial_state: torch.Tensor,
        t_span_minutes: torch.Tensor,
        infusion_rate_ug_min: torch.Tensor,
        z_vasc: torch.Tensor,
        covariates: torch.Tensor,
        map_base: torch.Tensor,
        num_mc_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs Monte Carlo Dropout forward passes during inference to quantify epistemic uncertainty.
        """
        self.neural_drift.train()

        samples = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                traj = self.forward_trajectory(
                    initial_state,
                    t_span_minutes,
                    infusion_rate_ug_min,
                    z_vasc,
                    covariates,
                    map_base,
                )
                samples.append(traj.unsqueeze(0))

        stacked = torch.cat(samples, dim=0)  # (N_mc, T, B, 3)
        mean_traj = stacked.mean(dim=0)
        std_traj = stacked.std(dim=0)

        self.neural_drift.eval()
        return mean_traj, std_traj
