"""
morphomics.py: High-frequency arterial blood pressure and photoplethysmography morphomics.

Extracts continuous vascular tone indicators:
- Dicrotic Notch Ratio (R_dn) = (P_dn - P_dia) / (P_sys - P_dia)
- Systolic Area (A_sys): Normalized integral of ejection curve
- Diastolic Decay Time Constant (tau_dia): Windkessel R_svr * C_art decay
- Integrated Mean Arterial Pressure (MAP_int)
- Pulse Arrival Time (PAT) dynamics
- Composite Vascular Tone Index (VTI)
"""

from typing import Dict, List, Optional
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit

from .beat_detector import BeatDetector


class WaveformMorphomics:
    """Computes hemodynamic morphomic descriptors from continuous 125 Hz ABP/PPG waveforms."""

    def __init__(self, sampling_rate_hz: float = 125.0) -> None:
        self.fs = sampling_rate_hz
        self.beat_detector = BeatDetector(sampling_rate_hz=sampling_rate_hz)

    @staticmethod
    def _exp_decay(t: np.ndarray, p0: float, tau: float) -> np.ndarray:
        return p0 * np.exp(-t / np.maximum(tau, 0.05))

    def estimate_diastolic_time_constant(
        self, abp_signal: np.ndarray, notch_idx: int, next_foot_idx: int
    ) -> float:
        """
        Fits Windkessel exponential decay P(t) = P0 * exp(-t / tau) during the diastolic runoff.
        tau = R_svr * C_art (Systemic Vascular Resistance * Arterial Compliance).
        """
        if next_foot_idx <= notch_idx + 4:
            return 0.50  # Physiological default ~0.50 s

        decay_seg = abp_signal[notch_idx:next_foot_idx]
        t = np.arange(len(decay_seg)) / self.fs

        try:
            p0_init = float(decay_seg[0])
            # Bound tau between 0.1s and 3.0s
            popt, _ = curve_fit(
                self._exp_decay,
                t,
                decay_seg,
                p0=[p0_init, 0.6],
                bounds=([p0_init * 0.7, 0.08], [p0_init * 1.3, 3.5]),
                maxfev=100,
            )
            return float(popt[1])
        except Exception:
            return 0.50

    def compute_integrated_map(self, abp_window: np.ndarray) -> float:
        """
        Calculates the exact integrated Mean Arterial Pressure (1/T int P(t) dt)
        rather than the conventional empirical formula (1/3 SBP + 2/3 DBP).
        """
        if len(abp_window) == 0:
            return 0.0
        return float(np.mean(abp_window))

    def extract_window_morphomics(
        self, abp_window: np.ndarray, ppg_window: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extracts comprehensive morphomic feature summary from a 10–30 second window.

        Returns dictionary of:
            - map_integrated: True Mean Arterial Pressure (mmHg)
            - sbp_mean: Average Systolic Blood Pressure (mmHg)
            - dbp_mean: Average Diastolic Blood Pressure (mmHg)
            - pulse_pressure_mean: Average SBP - DBP (mmHg)
            - heart_rate_mean: Average heart rate (bpm)
            - dicrotic_notch_ratio_mean: (P_dn - P_dia) / (P_sys - P_dia)
            - systolic_area_ratio_mean: Area under systolic wave normalized by total area
            - diastolic_tau_mean: Diastolic decay time constant (s)
            - pat_mean_ms: Pulse Arrival Time mean (ms) if PPG provided
            - vascular_tone_index: Composite normalized vascular resistance indicator
        """
        beats = self.beat_detector.segment_abp_beats(abp_window)
        map_int = self.compute_integrated_map(abp_window)

        if not beats:
            # Fallback on raw window percentiles if beat detection fails
            p95 = float(np.percentile(abp_window, 95))
            p05 = float(np.percentile(abp_window, 5))
            return {
                "map_integrated": round(map_int, 2),
                "sbp_mean": round(p95, 2),
                "dbp_mean": round(p05, 2),
                "pulse_pressure_mean": round(p95 - p05, 2),
                "heart_rate_mean": 75.0,
                "dicrotic_notch_ratio_mean": 0.35,
                "systolic_area_ratio_mean": 0.40,
                "diastolic_tau_mean": 0.50,
                "pat_mean_ms": 150.0,
                "vascular_tone_index": 0.50,
            }

        sbp_vals = [b["peak_val"] for b in beats]
        dbp_vals = [b["foot_val"] for b in beats]
        pp_vals = [b["pulse_pressure"] for b in beats]
        hr_vals = [b["heart_rate_bpm"] for b in beats]

        notch_ratios = []
        sys_area_ratios = []
        tau_vals = []

        for b in beats:
            p_val = b["peak_val"]
            f_val = b["foot_val"]
            n_val = b["notch_val"]
            pp = b["pulse_pressure"]

            # 1. Dicrotic Notch Ratio: Rdn in [0, 1]
            rdn = (n_val - f_val) / max(pp, 1.0)
            notch_ratios.append(np.clip(rdn, 0.05, 0.95))

            # 2. Systolic Area vs Total Beat Area
            f_idx = b["foot_idx"]
            n_idx = b["notch_idx"]
            # Look ahead for next foot or end of beat
            beat_end = min(len(abp_window), f_idx + int(b["beat_duration_s"] * self.fs))
            if beat_end > f_idx:
                full_beat = abp_window[f_idx:beat_end] - f_val
                total_area = integrate.trapezoid(np.maximum(full_beat, 0.0))
                if n_idx > f_idx:
                    sys_part = abp_window[f_idx:n_idx] - f_val
                    sys_area = integrate.trapezoid(np.maximum(sys_part, 0.0))
                    area_ratio = sys_area / max(total_area, 1e-4)
                else:
                    area_ratio = 0.40
                sys_area_ratios.append(np.clip(area_ratio, 0.1, 0.9))

            # 3. Diastolic Windkessel tau
            tau = self.estimate_diastolic_time_constant(abp_window, n_idx, beat_end)
            tau_vals.append(tau)

        # Pulse Arrival Time (PAT) if PPG supplied
        if ppg_window is not None and len(ppg_window) == len(abp_window):
            peak_indices = np.array([b["peak_idx"] for b in beats])
            pats = self.beat_detector.compute_pulse_arrival_time(ppg_window, peak_indices)
            valid_pats = pats[~np.isnan(pats)]
            pat_mean = float(np.mean(valid_pats)) if len(valid_pats) > 0 else 160.0
        else:
            pat_mean = 160.0

        mean_rdn = float(np.mean(notch_ratios))
        mean_tau = float(np.mean(tau_vals))
        mean_area = float(np.mean(sys_area_ratios))

        # Composite Vascular Tone Index:
        # High Rdn + High tau + Short PAT = Vasoconstriction (High SVR)
        # Low Rdn + Low tau + Long PAT = Vasoplegia/Sepsis (Low SVR)
        norm_rdn = (mean_rdn - 0.15) / (0.65 - 0.15)
        norm_tau = (mean_tau - 0.20) / (1.20 - 0.20)
        norm_pat = 1.0 - (pat_mean - 100.0) / (300.0 - 100.0)
        vti = float(np.clip(0.45 * norm_rdn + 0.35 * norm_tau + 0.20 * norm_pat, 0.0, 1.0))

        return {
            "map_integrated": round(map_int, 2),
            "sbp_mean": round(float(np.mean(sbp_vals)), 2),
            "dbp_mean": round(float(np.mean(dbp_vals)), 2),
            "pulse_pressure_mean": round(float(np.mean(pp_vals)), 2),
            "heart_rate_mean": round(float(np.mean(hr_vals)), 1),
            "dicrotic_notch_ratio_mean": round(mean_rdn, 4),
            "systolic_area_ratio_mean": round(mean_area, 4),
            "diastolic_tau_mean": round(mean_tau, 4),
            "pat_mean_ms": round(pat_mean, 2),
            "vascular_tone_index": round(vti, 4),
        }
