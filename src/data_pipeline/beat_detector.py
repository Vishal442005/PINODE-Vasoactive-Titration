"""
beat_detector.py: Beat-to-beat segmentation and landmark detection for 125 Hz ABP and PPG.

Extracts:
- Systolic peaks (P_sys)
- Diastolic troughs/feet (P_dia)
- Dicrotic notch (P_dn) using derivative zero-crossings and curvature
- Beat intervals and Pulse Arrival Time (PAT)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


class BeatDetector:
    """Detects physiological landmarks on individual cardiac cycles from continuous ABP and PPG."""

    def __init__(
        self,
        sampling_rate_hz: float = 125.0,
        min_heart_rate_bpm: float = 35.0,
        max_heart_rate_bpm: float = 210.0,
    ) -> None:
        """
        Args:
            sampling_rate_hz: Sampling frequency in Hertz (default 125 Hz).
            min_heart_rate_bpm: Lowest plausible heart rate (sets max beat window).
            max_heart_rate_bpm: Highest plausible heart rate (sets min peak distance).
        """
        self.fs = sampling_rate_hz
        self.min_distance_samples = int(self.fs * 60.0 / max_heart_rate_bpm)
        self.max_beat_samples = int(self.fs * 60.0 / min_heart_rate_bpm)

    def detect_systolic_peaks(self, abp_signal: np.ndarray) -> np.ndarray:
        """
        Identifies systolic peak indices in continuous ABP waveform using prominence and distance.
        """
        if len(abp_signal) < self.min_distance_samples:
            return np.array([], dtype=int)

        # Baseline prominence threshold = 15% of range
        sig_range = np.percentile(abp_signal, 95) - np.percentile(abp_signal, 5)
        prominence = max(5.0, 0.20 * sig_range)

        peaks, _ = signal.find_peaks(
            abp_signal,
            distance=self.min_distance_samples,
            prominence=prominence,
        )
        return peaks

    def detect_diastolic_feet(
        self, abp_signal: np.ndarray, systolic_peaks: np.ndarray
    ) -> np.ndarray:
        """
        Identifies diastolic foot (minimum pressure onset) immediately preceding each systolic peak.
        """
        feet = []
        for i in range(len(systolic_peaks)):
            curr_peak = systolic_peaks[i]
            if i == 0:
                search_start = max(0, curr_peak - self.max_beat_samples)
            else:
                search_start = systolic_peaks[i - 1]

            if curr_peak > search_start:
                foot_idx = search_start + np.argmin(abp_signal[search_start:curr_peak])
                feet.append(foot_idx)
            else:
                feet.append(max(0, curr_peak - 10))

        return np.array(feet, dtype=int)

    def detect_dicrotic_notch(
        self, abp_signal: np.ndarray, peak_idx: int, next_foot_idx: int
    ) -> Optional[int]:
        """
        Locates the dicrotic notch index between a systolic peak and subsequent diastolic foot.
        Uses first and second derivatives (zero-crossing of slope or peak in curvature).
        """
        if next_foot_idx <= peak_idx + 4:
            return None

        segment = abp_signal[peak_idx:next_foot_idx]
        seg_len = len(segment)

        # Notch typically occurs between 15% and 65% of the diastolic decay window
        start_search = int(0.12 * seg_len)
        end_search = int(0.70 * seg_len)

        if end_search <= start_search + 2:
            return None

        # Compute 1st and 2nd numerical derivatives
        d1 = np.gradient(segment)
        d2 = np.gradient(d1)

        # Search for relative minimum in |d1| (inflection point) or peak in d2 (aortic valve closure rebound)
        search_region_d2 = d2[start_search:end_search]
        search_region_d1 = np.abs(d1[start_search:end_search])

        # Candidate 1: Local maximum in second derivative (rebound upward acceleration)
        d2_peak_local = np.argmax(search_region_d2)
        notch_cand_d2 = start_search + d2_peak_local

        # Candidate 2: Local minimum in absolute velocity (momentary flattening)
        d1_min_local = np.argmin(search_region_d1)
        notch_cand_d1 = start_search + d1_min_local

        # Blend candidates (prefer d2 peak if prominent)
        best_local = notch_cand_d2 if search_region_d2[d2_peak_local] > 0 else notch_cand_d1
        global_notch_idx = peak_idx + best_local

        return int(global_notch_idx)

    def segment_abp_beats(self, abp_signal: np.ndarray) -> List[Dict[str, float]]:
        """
        Full beat-to-beat segmentation on continuous ABP.

        Returns list of segmented beat dictionaries:
            - beat_idx: Sequential beat counter
            - peak_idx, peak_val (Systolic Blood Pressure, SBP)
            - foot_idx, foot_val (Diastolic Blood Pressure, DBP)
            - notch_idx, notch_val (Dicrotic Notch)
            - pulse_pressure: SBP - DBP
            - beat_duration_s: Time duration in seconds
            - heart_rate_bpm: Local instantaneous heart rate
        """
        peaks = self.detect_systolic_peaks(abp_signal)
        if len(peaks) < 2:
            return []

        feet = self.detect_diastolic_feet(abp_signal, peaks)
        beats = []

        for i in range(len(peaks) - 1):
            p_idx = peaks[i]
            f_idx = feet[i]
            next_f_idx = feet[i + 1]

            notch_idx = self.detect_dicrotic_notch(abp_signal, p_idx, next_f_idx)

            p_val = float(abp_signal[p_idx])
            f_val = float(abp_signal[f_idx])
            next_f_val = float(abp_signal[next_f_idx])
            pp = float(p_val - f_val)

            if notch_idx is not None:
                notch_val = float(abp_signal[notch_idx])
            else:
                # Approximation: dicrotic notch at 40% of decay window if obscured
                approx_idx = p_idx + int(0.35 * (next_f_idx - p_idx))
                notch_idx = approx_idx
                notch_val = float(abp_signal[approx_idx])

            # Bound notch value between local diastolic floor and systolic peak
            notch_val = float(np.clip(notch_val, min(f_val, next_f_val), p_val))

            beat_dur = float(next_f_idx - f_idx) / self.fs
            hr_bpm = 60.0 / beat_dur if beat_dur > 0 else 0.0

            # Physiological filter (skip spurious detections)
            if 30.0 <= hr_bpm <= 220.0 and pp > 5.0:
                beats.append(
                    {
                        "beat_idx": len(beats),
                        "peak_idx": int(p_idx),
                        "peak_val": round(p_val, 2),
                        "foot_idx": int(f_idx),
                        "foot_val": round(f_val, 2),
                        "notch_idx": int(notch_idx),
                        "notch_val": round(notch_val, 2),
                        "pulse_pressure": round(pp, 2),
                        "beat_duration_s": round(beat_dur, 4),
                        "heart_rate_bpm": round(hr_bpm, 1),
                    }
                )

        return beats

    def compute_pulse_arrival_time(
        self, ppg_signal: np.ndarray, abp_peaks: np.ndarray
    ) -> np.ndarray:
        """
        Calculates Pulse Arrival Time (PAT) in milliseconds between PPG foot and ABP systolic peak.
        """
        if len(abp_peaks) == 0:
            return np.array([])

        pat_ms_list = []
        # PPG troughs/feet
        ppg_inv_peaks, _ = signal.find_peaks(-ppg_signal, distance=self.min_distance_samples)

        for p_idx in abp_peaks:
            # Find closest preceding PPG foot
            prec_ppg = ppg_inv_peaks[ppg_inv_peaks < p_idx]
            if len(prec_ppg) > 0:
                delay_samples = p_idx - prec_ppg[-1]
                delay_ms = (delay_samples / self.fs) * 1000.0
                if 20.0 <= delay_ms <= 450.0:
                    pat_ms_list.append(delay_ms)
                else:
                    pat_ms_list.append(np.nan)
            else:
                pat_ms_list.append(np.nan)

        return np.array(pat_ms_list)
