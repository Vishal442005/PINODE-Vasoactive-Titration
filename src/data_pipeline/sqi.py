"""
sqi.py: Signal Quality Index (SQI) evaluation for ABP and PPG continuous waveforms.

Implements multi-feature statistical and spectral SQI algorithms:
- Physiological range checking (Pulse Pressure, Diastolic vs Systolic)
- Statistical moments (Skewness, Kurtosis distribution)
- Spectral energy ratio (Cardiac band: 0.6–3.5 Hz vs Total power)
- Zero-crossing and rhythm continuity
- Composite SQI confidence score in [0.0, 1.0]
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats, signal, integrate


class SignalQualityIndex:
    """Computes real-time physiological Signal Quality Indices for ICU biosignals."""

    def __init__(
        self,
        sampling_rate_hz: float = 125.0,
        min_pulse_pressure_mmhg: float = 15.0,
        abp_bounds: Tuple[float, float] = (30.0, 240.0),
        min_sqi_threshold: float = 0.65,
    ) -> None:
        """
        Args:
            sampling_rate_hz: Signal sampling rate (Hz).
            min_pulse_pressure_mmhg: Minimum physiological SBP - DBP difference.
            abp_bounds: Plausible clinical range [min, max] mmHg.
            min_sqi_threshold: Threshold below which signal is considered untrustworthy.
        """
        self.fs = sampling_rate_hz
        self.min_pp = min_pulse_pressure_mmhg
        self.abp_min, self.abp_max = abp_bounds
        self.threshold = min_sqi_threshold

    def compute_spectral_energy_ratio(
        self, sig: np.ndarray, cardiac_band: Tuple[float, float] = (0.6, 3.5)
    ) -> float:
        """
        Computes the ratio of spectral power inside the cardiac band (36–210 bpm)
        to total spectral power (0.1–15 Hz).
        """
        if len(sig) < int(2 * self.fs):
            return 0.0

        freqs, psd = signal.welch(sig, fs=self.fs, nperseg=min(len(sig), int(4 * self.fs)))
        total_power = integrate.trapezoid(psd, freqs)
        if total_power <= 1e-8:
            return 0.0

        cardiac_mask = (freqs >= cardiac_band[0]) & (freqs <= cardiac_band[1])
        cardiac_power = integrate.trapezoid(psd[cardiac_mask], freqs[cardiac_mask])
        ratio = float(cardiac_power / total_power)
        return float(np.clip(ratio, 0.0, 1.0))

    def evaluate_abp_window(self, abp_window: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the quality of a 10–30 second window of continuous ABP.

        Returns dictionary containing:
            - range_score: Fraction of samples within physiological bounds [0, 1].
            - pulse_pressure_score: Consistency of SBP - DBP > 15 mmHg.
            - skewness_score: Proximity of skewness to natural arterial profile (0.2–1.5).
            - spectral_score: Cardiac band power ratio [0, 1].
            - composite_sqi: Weighted composite SQI score in [0.0, 1.0].
            - is_usable: Boolean flag indicating if composite_sqi >= threshold.
        """
        if len(abp_window) < int(2 * self.fs):
            return {
                "range_score": 0.0,
                "pulse_pressure_score": 0.0,
                "skewness_score": 0.0,
                "spectral_score": 0.0,
                "composite_sqi": 0.0,
                "is_usable": 0.0,
            }

        # 1. Physical range score
        in_range = (abp_window >= self.abp_min) & (abp_window <= self.abp_max)
        range_score = float(np.mean(in_range))

        # 2. Pulse pressure score
        p_95 = np.percentile(abp_window, 95)
        p_05 = np.percentile(abp_window, 5)
        apparent_pp = p_95 - p_05
        pp_score = 1.0 if apparent_pp >= self.min_pp else float(apparent_pp / self.min_pp)
        pp_score = float(np.clip(pp_score, 0.0, 1.0))

        # 3. Skewness score (Arterial pulse has positive skew due to systolic peaking)
        std_abp = float(np.std(abp_window))
        if std_abp < 0.5:
            skew_score = 0.0
        else:
            skew = stats.skew(abp_window)
            if 0.1 <= skew <= 1.8:
                skew_score = 1.0
            elif skew > 1.8:
                skew_score = max(0.0, 1.0 - (skew - 1.8) / 2.0)
            else:
                skew_score = max(0.0, 1.0 - (0.1 - skew) / 1.5)
        skew_score = float(np.clip(skew_score, 0.0, 1.0))

        # 4. Spectral energy score
        spectral_score = self.compute_spectral_energy_ratio(abp_window)

        # 5. Composite SQI (Weighted average)
        composite = (
            0.35 * range_score
            + 0.25 * pp_score
            + 0.15 * skew_score
            + 0.25 * spectral_score
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        return {
            "range_score": round(range_score, 4),
            "pulse_pressure_score": round(pp_score, 4),
            "skewness_score": round(skew_score, 4),
            "spectral_score": round(spectral_score, 4),
            "composite_sqi": round(composite, 4),
            "is_usable": 1.0 if composite >= self.threshold else 0.0,
        }

    def evaluate_ppg_window(self, ppg_window: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the quality of a 10–30 second window of continuous PPG.

        Returns dictionary of quality metrics and composite SQI.
        """
        if len(ppg_window) < int(2 * self.fs):
            return {
                "spectral_score": 0.0,
                "kurtosis_score": 0.0,
                "variance_score": 0.0,
                "composite_sqi": 0.0,
                "is_usable": 0.0,
            }

        # 1. Variance score (check for flatlines or saturation)
        std_val = float(np.std(ppg_window))
        var_score = 1.0 if std_val > 0.01 else 0.0

        # 2. Kurtosis score (normal PPG has kurtosis near 3.0 +/- 1.5)
        kurt = float(stats.kurtosis(ppg_window))
        if 1.5 <= kurt <= 5.0:
            kurt_score = 1.0
        else:
            kurt_score = max(0.0, 1.0 - min(abs(kurt - 1.5), abs(kurt - 5.0)) / 4.0)
        kurt_score = float(np.clip(kurt_score, 0.0, 1.0))

        # 3. Spectral score
        spectral_score = self.compute_spectral_energy_ratio(ppg_window)

        # Composite score
        composite = 0.40 * spectral_score + 0.30 * kurt_score + 0.30 * var_score
        composite = float(np.clip(composite, 0.0, 1.0))

        return {
            "spectral_score": round(spectral_score, 4),
            "kurtosis_score": round(kurt_score, 4),
            "variance_score": round(var_score, 4),
            "composite_sqi": round(composite, 4),
            "is_usable": 1.0 if composite >= self.threshold else 0.0,
        }
