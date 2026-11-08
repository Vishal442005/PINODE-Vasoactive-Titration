"""
pk_compartments.py: Two-compartment pharmacokinetic-pharmacodynamic (PK-PD) model for Norepinephrine.

Couples analytical differential equations for central plasma clearance Cp(t) and effect-site equilibration Ce(t)
with patient-specific covariate conditioning (renal function, liver clearance, SOFA score).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatientCovariateEncoder(nn.Module):
    """Maps static/slow clinical covariates to patient-specific PK parameters (k_el, V_d)."""

    def __init__(self, num_covariates: int = 6, hidden_dim: int = 32) -> None:
        """
        Covariates expected:
        [weight_kg, age, sofa_score, serum_creatinine, total_bilirubin, arterial_lactate]
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_covariates, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # Outputs (unconstrained k_el, V_d)
        )

    def forward(self, covariates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            covariates: Tensor of shape (Batch, num_covariates).

        Returns:
            Tuple of:
                - k_el: Elimination rate constant in [0.10, 0.55] min^-1.
                - V_d: Central distribution volume in [10.0, 30.0] Liters.
        """
        raw = self.net(covariates)
        # Constrain to strictly physiological ranges via Sigmoid scaling
        k_el = 0.10 + 0.45 * torch.sigmoid(raw[:, 0:1])
        V_d = 10.0 + 20.0 * torch.sigmoid(raw[:, 1:2])
        return k_el, V_d


class TwoCompartmentHillPKPD(nn.Module):
    """
    Two-compartment PK-PD differential system for vasoactive drugs (Norepinephrine):
    dCp/dt = u(t)/V_d - k_el * Cp(t)
    dCe/dt = k_e0 * (Cp(t) - Ce(t))
    Delta_MAP_PD = E_max * (Ce^gamma) / (EC50^gamma + Ce^gamma)
    """

    def __init__(
        self,
        default_ke0: float = 0.350,  # min^-1
        num_covariates: int = 6,
    ) -> None:
        super().__init__()
        self.covariate_encoder = PatientCovariateEncoder(num_covariates=num_covariates)

        # Effect-site equilibration rate constant (log parameterization for positivity)
        self.log_ke0 = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(default_ke0)))))

        # Learnable population PD Hill parameters
        self.raw_emax = nn.Parameter(torch.tensor(0.5))  # Scales to [10.0, 80.0] mmHg
        self.raw_ec50 = nn.Parameter(torch.tensor(0.35))  # Scales to [0.05, 1.20] ug/L
        self.raw_gamma = nn.Parameter(torch.tensor(1.8))  # Scales to [0.8, 3.5]

    @property
    def ke0(self) -> torch.Tensor:
        """Effect compartment equilibration rate constant min^-1."""
        return torch.exp(self.log_ke0).clamp(0.10, 1.0)

    @property
    def emax(self) -> torch.Tensor:
        """Maximum drug efficacy (mmHg)."""
        return 10.0 + 70.0 * torch.sigmoid(self.raw_emax)

    @property
    def ec50(self) -> torch.Tensor:
        """Half-maximal effect concentration (ug/L)."""
        return 0.05 + 1.15 * torch.sigmoid(self.raw_ec50)

    @property
    def gamma(self) -> torch.Tensor:
        """Hill steepness coefficient."""
        return 0.80 + 2.70 * torch.sigmoid(self.raw_gamma)

    def compute_pk_derivatives(
        self,
        cp: torch.Tensor,
        ce: torch.Tensor,
        infusion_rate_ug_min: torch.Tensor,
        covariates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes time derivatives [dCp/dt, dCe/dt] in minutes^-1.

        Args:
            cp: Central compartment concentration (ug/L), shape (B, 1).
            ce: Effect-site concentration (ug/L), shape (B, 1).
            infusion_rate_ug_min: Drug infusion rate (ug/min), shape (B, 1).
            covariates: Patient clinical covariates, shape (B, num_covariates).

        Returns:
            Tuple of (d_cp_dt, d_ce_dt).
        """
        k_el, V_d = self.covariate_encoder(covariates)
        d_cp = infusion_rate_ug_min / V_d - k_el * cp
        d_ce = self.ke0 * (cp - ce)
        return d_cp, d_ce

    def compute_pharmacodynamic_effect(self, ce: torch.Tensor) -> torch.Tensor:
        """
        Computes the sigmoidal Hill drug effect: Delta MAP = E_max * Ce^gamma / (EC50^gamma + Ce^gamma).
        """
        ce_clamped = torch.relu(ce) + 1e-7
        ce_pow = torch.pow(ce_clamped, self.gamma)
        ec50_pow = torch.pow(self.ec50, self.gamma)
        delta_map = self.emax * (ce_pow / (ec50_pow + ce_pow))
        return delta_map
