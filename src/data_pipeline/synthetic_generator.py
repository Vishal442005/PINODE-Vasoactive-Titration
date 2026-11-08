"""
synthetic_generator.py: High-fidelity synthetic physiological waveform generator for ICU septic shock.

Generates continuous 125 Hz Arterial Blood Pressure (ABP) and Photoplethysmography (PPG) waveforms
coupled with two-compartment Norepinephrine pharmacokinetics, autonomic baroreflex, and sepsis-induced vasodilation.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class SyntheticPatientGenerator:
    """Generates continuous-time synthetic cardiovascular waveforms and clinical trajectories."""

    def __init__(
        self,
        sampling_rate_hz: float = 125.0,
        patient_weight_kg: float = 75.0,
        baseline_map: float = 78.0,
        septic_shock_severity: float = 1.0,
        random_seed: Optional[int] = 42,
    ) -> None:
        """
        Args:
            sampling_rate_hz: Sampling frequency in Hertz (125 Hz).
            patient_weight_kg: Patient body weight in kilograms.
            baseline_map: Initial healthy/compensated Mean Arterial Pressure (mmHg).
            septic_shock_severity: Scaling factor for vasodilation rate (0.5 = mild, 1.5 = severe).
            random_seed: Random seed for reproducibility.
        """
        self.fs = sampling_rate_hz
        self.weight = patient_weight_kg
        self.baseline_map = baseline_map
        self.severity = septic_shock_severity
        if random_seed is not None:
            np.random.seed(random_seed)

        # Pharmacokinetic parameters (Norepinephrine)
        self.vd = 0.20 * self.weight  # Central volume (L)
        self.kel = 0.277  # min^-1 (half-life ~2.5 min)
        self.ke0 = 0.350  # min^-1 (effect-site equilibration)

        # Pharmacodynamic Hill equation parameters
        self.emax = 45.0  # Max increase in MAP (mmHg)
        self.ec50 = 0.35  # ug/L (half-maximal concentration)
        self.gamma = 1.8  # Hill steepness coefficient

    def simulate_hemodynamic_state(
        self,
        duration_minutes: float = 60.0,
        infusion_protocol: Optional[np.ndarray] = None,
        dt_seconds: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        Integrates differential equations for Cp, Ce, SVR, and MAP over continuous time.

        Args:
            duration_minutes: Total simulation duration in minutes.
            infusion_protocol: 1D array of Norepinephrine infusion rates (ug/kg/min) at each time step.
            dt_seconds: ODE numerical integration step size (seconds).

        Returns:
            Dictionary containing time arrays for t, Cp, Ce, MAP, HR, SVR, and InfusionRate.
        """
        total_steps = int((duration_minutes * 60.0) / dt_seconds)
        dt_min = dt_seconds / 60.0

        if infusion_protocol is None:
            # Default protocol: 15 min untreated sepsis -> nurse starts 0.08 ug/kg/min -> titrated
            infusion_protocol = np.zeros(total_steps)
            t_min = np.linspace(0, duration_minutes, total_steps)
            infusion_protocol[t_min >= 15.0] = 0.08
            infusion_protocol[t_min >= 30.0] = 0.18
            infusion_protocol[t_min >= 45.0] = 0.12

        cp = np.zeros(total_steps)
        ce = np.zeros(total_steps)
        map_trajectory = np.zeros(total_steps)
        hr_trajectory = np.zeros(total_steps)
        svr_trajectory = np.zeros(total_steps)

        curr_cp = 0.0
        curr_ce = 0.0
        curr_map = self.baseline_map
        curr_hr = 78.0

        for i in range(total_steps):
            u_ug_kg_min = infusion_protocol[i]
            # Convert ug/kg/min to ug/min
            u_total_ug_min = u_ug_kg_min * self.weight

            # 1. PK differential equations (dCp/dt, dCe/dt)
            d_cp = (u_total_ug_min / self.vd - self.kel * curr_cp) * dt_min
            d_ce = (self.ke0 * (curr_cp - curr_ce)) * dt_min
            curr_cp = max(0.0, curr_cp + d_cp)
            curr_ce = max(0.0, curr_ce + d_ce)

            # 2. Pharmacodynamic drug effect (Hill equation)
            drug_effect = self.emax * (curr_ce**self.gamma) / (self.ec50**self.gamma + curr_ce**self.gamma + 1e-6)

            # 3. Sepsis progressive vasodilation drift (decaying SVR)
            t_cur_min = i * dt_min
            septic_decay = self.severity * 18.0 * (1.0 - np.exp(-t_cur_min / 20.0))

            # 4. Baroreflex compensatory response
            baro_map = curr_map
            delta_map_baro = -0.05 * (baro_map - self.baseline_map) * dt_min

            # Combined MAP state
            target_map = self.baseline_map - septic_decay + drug_effect
            curr_map += (0.15 * (target_map - curr_map)) * dt_min * 60.0 + delta_map_baro

            # Physiological random walk noise
            curr_map += np.random.normal(0, 0.05)
            curr_map = np.clip(curr_map, 35.0, 160.0)

            # Heart rate: Compensatory tachycardia when MAP is low
            target_hr = 75.0 + max(0.0, (65.0 - curr_map) * 1.2) + min(25.0, curr_ce * 15.0)
            curr_hr += 0.1 * (target_hr - curr_hr) + np.random.normal(0, 0.1)

            # Systemic Vascular Resistance index (normalized 0.2 - 1.2)
            svr = (curr_map / curr_hr) * 1.1

            cp[i] = curr_cp
            ce[i] = curr_ce
            map_trajectory[i] = curr_map
            hr_trajectory[i] = curr_hr
            svr_trajectory[i] = svr

        time_axis_min = np.linspace(0, duration_minutes, total_steps)

        return {
            "time_minutes": time_axis_min,
            "cp_plasma": cp,
            "ce_effect": ce,
            "map": map_trajectory,
            "heart_rate": hr_trajectory,
            "svr": svr_trajectory,
            "infusion_rate": infusion_protocol,
        }

    def synthesize_waveforms(
        self,
        hemo_state: Dict[str, np.ndarray],
        noise_level: float = 0.03,
        inject_artifacts: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes high-frequency 125 Hz ABP and PPG waveforms driven by the continuous hemodynamic state.

        Args:
            hemo_state: Dictionary output from simulate_hemodynamic_state.
            noise_level: Gaussian noise amplitude.
            inject_artifacts: If True, injects synthetic line flushes and sensor noise.

        Returns:
            Tuple of (time_axis_sec, abp_125hz, ppg_125hz).
        """
        duration_sec = hemo_state["time_minutes"][-1] * 60.0
        total_samples = int(duration_sec * self.fs)
        t_sec = np.linspace(0, duration_sec, total_samples)

        # Interpolate MAP and HR onto 125 Hz time grid
        t_state_sec = hemo_state["time_minutes"] * 60.0
        map_interp = np.interp(t_sec, t_state_sec, hemo_state["map"])
        hr_interp = np.interp(t_sec, t_state_sec, hemo_state["heart_rate"])
        ce_interp = np.interp(t_sec, t_state_sec, hemo_state["ce_effect"])

        abp = np.zeros(total_samples)
        ppg = np.zeros(total_samples)

        # Beat-by-beat waveform generation
        sample_idx = 0
        phase = 0.0

        while sample_idx < total_samples:
            cur_hr = hr_interp[sample_idx]
            cur_map = map_interp[sample_idx]
            cur_ce = ce_interp[sample_idx]

            beat_period_sec = 60.0 / cur_hr
            beat_samples = int(beat_period_sec * self.fs)
            if sample_idx + beat_samples > total_samples:
                beat_samples = total_samples - sample_idx
            if beat_samples <= 5:
                break

            # Pulse pressure scales with stroke volume and vascular tone
            pulse_pressure = 0.45 * cur_map
            sbp = cur_map + 0.60 * pulse_pressure
            dbp = cur_map - 0.40 * pulse_pressure

            # Normalized beat time t_b in [0, 1]
            tb = np.linspace(0, 1.0, beat_samples)

            # Dicrotic notch height: In sepsis (low cur_map/ce), notch drops to 20%;
            # under Norepinephrine, notch rises toward 55%
            notch_height_ratio = 0.22 + 0.32 * min(1.0, cur_ce / (self.ec50 * 2.0))
            notch_time = 0.38

            # Beat contour model:
            # Phase 1: Systolic ejection (fast rise, peak at t=0.18)
            systole = np.exp(-((tb - 0.18) ** 2) / (2 * (0.07**2)))

            # Phase 2: Dicrotic notch reflection (local dip + bump at t=0.38)
            dicrotic_wave = notch_height_ratio * np.exp(-((tb - 0.42) ** 2) / (2 * (0.05**2)))

            # Phase 3: Diastolic runoff exponential decay
            diastolic_runoff = np.exp(-tb / 0.55)

            # Combine waves into pressure contour
            contour = 0.65 * systole + 0.35 * dicrotic_wave + 0.20 * diastolic_runoff
            contour = (contour - np.min(contour)) / (np.max(contour) - np.min(contour) + 1e-6)

            abp_beat = dbp + contour * pulse_pressure
            abp[sample_idx : sample_idx + beat_samples] = abp_beat

            # PPG: Peripheral optical pulse (smoother, dicrotic notch is delayed and rounded)
            ppg_contour = np.exp(-((tb - 0.26) ** 2) / (2 * (0.11**2))) + 0.20 * np.exp(-((tb - 0.52) ** 2) / (2 * (0.08**2)))
            ppg_contour = (ppg_contour - np.min(ppg_contour)) / (np.max(ppg_contour) - np.min(ppg_contour) + 1e-6)
            ppg[sample_idx : sample_idx + beat_samples] = ppg_contour

            sample_idx += beat_samples

        # Add physiological respiratory modulation (0.25 Hz baseline wander)
        resp_wander = 2.5 * np.sin(2 * np.pi * 0.25 * t_sec)
        abp += resp_wander
        ppg += 0.05 * np.sin(2 * np.pi * 0.25 * t_sec)

        # Add sensor noise
        abp += np.random.normal(0, noise_level * 50.0, total_samples)
        ppg += np.random.normal(0, noise_level, total_samples)

        # Optional artifact injection (e.g., catheter line flushes at t = 20 min)
        if inject_artifacts and duration_sec >= 1200:
            flush_start = int(1200 * self.fs)
            flush_len = int(3.5 * self.fs)
            if flush_start + flush_len < total_samples:
                # Extreme square wave spike characteristic of pressure line flush
                abp[flush_start : flush_start + flush_len] = 275.0 + np.random.normal(0, 3.0, flush_len)

        return t_sec, abp, ppg
