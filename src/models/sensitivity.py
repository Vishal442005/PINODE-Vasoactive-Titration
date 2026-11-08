"""
sensitivity.py: Global Sensitivity Analysis (GSA) and parameter identifiability for PINODE-Vaso.

Implements Sobol variance-based decomposition and parameter sensitivity screening
across the 7 core PK-PD and physiological parameters:
[V_d, k_el, k_e0, E_max, EC_50, gamma, lambda_baro].
"""

from typing import Dict, List, Tuple
import numpy as np


class GlobalSensitivityAnalyzer:
    """Computes first-order (S_i) and total-order (S_Ti) Sobol sensitivity indices for PINODE."""

    def __init__(self, sample_size: int = 256, random_seed: int = 42) -> None:
        self.N = sample_size
        self.rng = np.random.default_rng(random_seed)

        # Parameter names and physiological sampling bounds [min, max]
        self.param_bounds = {
            "V_d": (10.0, 30.0),        # L
            "k_el": (0.10, 0.55),       # min^-1
            "k_e0": (0.10, 0.80),       # min^-1
            "E_max": (15.0, 75.0),      # mmHg
            "EC_50": (0.05, 1.00),      # ug/L
            "gamma": (0.80, 3.20),      # unitless
            "lambda_baro": (0.01, 0.15) # min^-1
        }
        self.param_names = list(self.param_bounds.keys())
        self.D = len(self.param_names)

    def generate_sobol_matrices(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generates Saltelli sampling matrices A, B, and C_i for variance decomposition.
        Total evaluations = N * (D + 2).
        """
        # Base matrices in [0, 1]^D
        A = self.rng.random((self.N, self.D))
        B = self.rng.random((self.N, self.D))

        # Scale to physical bounds
        for j, (name, (low, high)) in enumerate(self.param_bounds.items()):
            A[:, j] = low + A[:, j] * (high - low)
            B[:, j] = low + B[:, j] * (high - low)

        # C_i matrices: Matrix B with column i taken from matrix A
        C_matrices = []
        for i in range(self.D):
            Ci = B.copy()
            Ci[:, i] = A[:, i]
            C_matrices.append(Ci)

        return A, B, C_matrices

    def evaluate_model_response(
        self, param_vector: np.ndarray, t_eval_min: float = 15.0, infusion_rate: float = 0.10
    ) -> float:
        """
        Evaluates the analytical response of MAP after t_eval_min minutes under constant infusion.
        """
        vd, kel, ke0, emax, ec50, gamma, lambda_baro = param_vector

        # Analytical 2-compartment solution under step infusion
        # Cp(t) = (u / (Vd * kel)) * (1 - exp(-kel * t))
        # Ce(t) reaches steady state delayed by ke0
        u_total = infusion_rate * 70.0  # ug/min for 70 kg
        cp_ss = u_total / (vd * kel)
        ce_t = cp_ss * (1.0 - np.exp(-ke0 * t_eval_min))

        # Hill equation
        ce_pow = np.power(max(ce_t, 1e-6), gamma)
        ec50_pow = np.power(ec50, gamma)
        pd_effect = emax * (ce_pow / (ec50_pow + ce_pow))

        # Baroreflex damped MAP response
        delta_map = pd_effect * (1.0 - np.exp(-lambda_baro * t_eval_min))
        return float(delta_map)

    def compute_sobol_indices(
        self, t_eval_min: float = 15.0, infusion_rate: float = 0.10
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes First-Order (S1) and Total-Order (ST) Sobol indices.

        Returns:
            Dictionary mapping parameter name -> {'S1': float, 'ST': float}.
        """
        A, B, C_matrices = self.generate_sobol_matrices()

        # Model evaluations for matrix A and B
        y_A = np.array([self.evaluate_model_response(p, t_eval_min, infusion_rate) for p in A])
        y_B = np.array([self.evaluate_model_response(p, t_eval_min, infusion_rate) for p in B])

        total_var = float(np.var(np.concatenate([y_A, y_B])))
        if total_var <= 1e-8:
            return {p: {"S1": 0.0, "ST": 0.0} for p in self.param_names}

        # Model evaluations for each Ci matrix
        y_C = []
        for Ci in C_matrices:
            y_Ci = np.array([self.evaluate_model_response(p, t_eval_min, infusion_rate) for p in Ci])
            y_C.append(y_Ci)

        results = {}
        for i, name in enumerate(self.param_names):
            # Jansen's formula for First-order index S1
            s1 = float(np.mean(y_B * (y_C[i] - y_A)) / total_var)

            # Total-order index ST
            st = float(0.5 * np.mean((y_A - y_C[i]) ** 2) / total_var)

            results[name] = {
                "S1": round(max(0.0, s1), 4),
                "ST": round(max(0.0, st), 4),
            }

        return results
