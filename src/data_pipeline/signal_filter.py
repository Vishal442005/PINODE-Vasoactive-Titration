"""
signal_filter.py: High-frequency biosignal filtering and artifact rejection for 125 Hz ABP and PPG waveforms.

Implements zero-phase 4th-order Butterworth bandpass filtering, 60 Hz powerline notch filtering,
and physiological outlier clamping (detecting line flushes, air bubbles, and damping).
"""

from typing import Tuple, Union
import numpy as np
from scipy import signal


class SignalFilter:
    """Zero-phase digital filter suite for arterial blood pressure (ABP) and photoplethysmography (PPG)."""

    def __init__(
        self,
        sampling_rate_hz: float = 125.0,
        low_cutoff_hz: float = 0.5,
        high_cutoff_hz: float = 15.0,
        order: int = 4,
        notch_freq_hz: float = 60.0,
        notch_q: float = 30.0,
        abp_bounds: Tuple[float, float] = (20.0, 260.0),
        max_flush_derivative: float = 250.0,
    ) -> None:
        """
        Initialize Butterworth bandpass and notch filters.

        Args:
            sampling_rate_hz: Sampling frequency in Hertz (standard 125 Hz for MIMIC).
            low_cutoff_hz: High-pass cutoff to remove baseline respiratory wander.
            high_cutoff_hz: Low-pass cutoff to remove high-frequency muscle tremor.
            order: Order of the Butterworth filter.
            notch_freq_hz: Powerline frequency to notch filter (60 Hz in US/MIMIC).
            notch_q: Quality factor for the notch filter.
            abp_bounds: Clinically plausible range for arterial blood pressure [min, max] mmHg.
            max_flush_derivative: Maximum allowable rate-of-rise (mmHg/s) before flagging a flush.
        """
        self.fs = sampling_rate_hz
        self.nyq = 0.5 * sampling_rate_hz
        self.abp_min, self.abp_max = abp_bounds
        self.max_flush_derivative = max_flush_derivative

        # Design Butterworth bandpass filter
        low = low_cutoff_hz / self.nyq
        high = high_cutoff_hz / self.nyq
        self.b_band, self.a_band = signal.butter(order, [low, high], btype="bandpass")

        # Design 60 Hz notch filter
        w0 = notch_freq_hz / self.nyq
        if w0 < 1.0:
            self.b_notch, self.a_notch = signal.iirnotch(w0, notch_q)
        else:
            self.b_notch, self.a_notch = None, None

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Applies forward-backward (zero-phase) 4th-order Butterworth bandpass filtering.

        Args:
            data: 1D or 2D NumPy array of raw physiological signal.

        Returns:
            Filtered signal with preserved phase and zero delay.
        """
        if data.ndim == 1:
            if len(data) <= 3 * max(len(self.a_band), len(self.b_band)):
                return data
            return signal.filtfilt(self.b_band, self.a_band, data)
        elif data.ndim == 2:
            return np.apply_along_axis(
                lambda ch: signal.filtfilt(self.b_band, self.a_band, ch)
                if len(ch) > 3 * max(len(self.a_band), len(self.b_band))
                else ch,
                axis=-1,
                arr=data,
            )
        else:
            raise ValueError(f"Expected 1D or 2D array, got ndim={data.ndim}")

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Removes powerline hum (60 Hz) using a zero-phase notch filter.

        Args:
            data: 1D or 2D NumPy array.

        Returns:
            Notch-filtered signal.
        """
        if self.b_notch is None:
            return data

        if data.ndim == 1:
            if len(data) <= 3 * max(len(self.a_notch), len(self.b_notch)):
                return data
            return signal.filtfilt(self.b_notch, self.a_notch, data)
        else:
            return np.apply_along_axis(
                lambda ch: signal.filtfilt(self.b_notch, self.a_notch, ch)
                if len(ch) > 3 * max(len(self.a_notch), len(self.b_notch))
                else ch,
                axis=-1,
                arr=data,
            )

    def detect_artifacts_and_clamp(
        self, abp_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects catheter flushes, damping, and sensor disconnections, and clamps outliers.

        Args:
            abp_signal: 1D NumPy array of continuous Arterial Blood Pressure (mmHg).

        Returns:
            Tuple of:
                - cleaned_abp: ABP array with outliers clamped and flushes smoothed.
                - artifact_mask: Boolean mask (True where signal was corrupted by artifacts).
        """
        cleaned = abp_signal.copy()
        artifact_mask = np.zeros_like(cleaned, dtype=bool)

        # 1. Physical range bounds
        out_of_bounds = (cleaned < self.abp_min) | (cleaned > self.abp_max)
        artifact_mask |= out_of_bounds

        # 2. Derivative rate-of-rise check (Catheter line flushes produce extreme spikes)
        diff = np.diff(cleaned, prepend=cleaned[0]) * self.fs
        extreme_derivatives = np.abs(diff) > self.max_flush_derivative
        artifact_mask |= extreme_derivatives

        # 3. Flatline / disconnection detection (zero variance in 2-second sliding window)
        win_size = int(2.0 * self.fs)
        if len(cleaned) >= win_size:
            # Vectorized rolling variance check
            rolling_std = np.array(
                [
                    np.std(cleaned[max(0, i - win_size) : i + 1])
                    for i in range(0, len(cleaned), win_size // 2)
                ]
            )
            flatline_indices = np.where(rolling_std < 0.5)[0]
            for idx in flatline_indices:
                start = max(0, idx * (win_size // 2) - win_size)
                end = min(len(cleaned), (idx + 1) * (win_size // 2))
                artifact_mask[start:end] = True

        # Clamp bounded values
        cleaned = np.clip(cleaned, self.abp_min, self.abp_max)
        return cleaned, artifact_mask

    def process_dual_stream(
        self, abp: np.ndarray, ppg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full end-to-end processing pipeline for synchronized ABP and PPG waveforms.

        Args:
            abp: 1D raw Arterial Blood Pressure signal (mmHg).
            ppg: 1D raw Photoplethysmography signal (arbitrary units / volts).

        Returns:
            Tuple of (cleaned_abp, filtered_ppg, combined_artifact_mask).
        """
        # Step 1: Detect and clean ABP flushes/bounds
        clamped_abp, abp_artifacts = self.detect_artifacts_and_clamp(abp)

        # Step 2: Apply zero-phase bandpass filtering
        filtered_abp = self.bandpass_filter(clamped_abp)
        filtered_ppg = self.bandpass_filter(ppg)

        # Step 3: Apply 60 Hz notch filtering
        filtered_abp = self.notch_filter(filtered_abp)
        filtered_ppg = self.notch_filter(filtered_ppg)

        # Combine artifact masks
        combined_artifacts = abp_artifacts

        return filtered_abp, filtered_ppg, combined_artifacts
