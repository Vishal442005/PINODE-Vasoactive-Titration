"""
noise_stress_test.py: Stress testing suite under 5 realistic ICU biosignal corruption modes.

Modes:
1. Gaussian Measurement Noise (sigma in 0% to 30%)
2. Respiratory Baseline Wander (0.1–0.3 Hz sinusoidal drift)
3. Motion Artifact Bursts (transient amplitude spiking)
4. Catheter Flush Spikes (square pressure spikes > 200 mmHg)
5. Peripheral Pulse Dropouts (sensor detachment or zero signal)

Evaluates whether the Signal Quality Index (SQI) and Conformal Safety Guardrail
correctly flag corrupted data and switch to clinician fallback.
"""

from typing import Dict, Tuple
import numpy as np
from src.data_pipeline.sqi import SignalQualityIndex
from src.control.safety_guardrails import SafetyGuardrails


class NoiseStressTester:
    """Stress tests the pipeline across synthetic artifacts and sensor faults."""

    def __init__(self, fs: float = 125.0) -> None:
        self.fs = fs
        self.sqi_index = SignalQualityIndex(sampling_rate_hz=fs)
        self.guardrails = SafetyGuardrails()

    def inject_noise(
        self,
        signal: np.ndarray,
        mode: str = "gaussian",
        severity: float = 0.15,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Injects controlled corruption into a physiological signal.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        corrupted = signal.copy()
        N = len(signal)
        sig_std = float(np.std(signal))

        if mode == "gaussian":
            noise = rng.normal(0, severity * sig_std, N)
            corrupted += noise

        elif mode == "baseline_wander":
            # Sinusoidal respiratory wander
            t = np.arange(N) / self.fs
            wander = severity * 20.0 * np.sin(2.0 * np.pi * 0.25 * t)
            corrupted += wander

        elif mode == "motion_burst":
            # 1 to 3 bursts of random high frequency spikes
            n_bursts = max(1, int(severity * 5))
            for _ in range(n_bursts):
                start = rng.integers(0, max(1, N - 100))
                duration = rng.integers(25, 125)
                end = min(N, start + duration)
                corrupted[start:end] += rng.uniform(-40.0, 40.0, end - start)

        elif mode == "catheter_flush":
            # High-pressure square wave spike (flush test)
            flush_start = int(N * 0.4)
            flush_end = min(N, flush_start + int(self.fs * 1.5))
            corrupted[flush_start:flush_end] = 220.0

        elif mode == "pulse_dropout":
            # Sensor disconnect
            drop_start = int(N * 0.5)
            drop_end = min(N, drop_start + int(self.fs * 2.0))
            corrupted[drop_start:drop_end] = 0.0

        return corrupted

    def run_stress_benchmark(
        self,
        clean_abp: np.ndarray,
        clean_ppg: np.ndarray,
        noise_levels: list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    ) -> Dict[str, Dict[float, float]]:
        """
        Tests SQI detection rate and guardrail fallback activation across noise levels.
        """
        results = {"abp_sqi": {}, "ppg_sqi": {}, "fallback_triggered": {}}
        rng = np.random.default_rng(42)

        for lvl in noise_levels:
            corrupt_abp = self.inject_noise(clean_abp, mode="gaussian", severity=lvl, rng=rng)
            corrupt_ppg = self.inject_noise(clean_ppg, mode="gaussian", severity=lvl, rng=rng)

            sqi_abp = self.sqi_index.evaluate_abp_window(corrupt_abp)["composite_sqi"]
            sqi_ppg = self.sqi_index.evaluate_ppg_window(corrupt_ppg)["composite_sqi"]

            # Check guardrail response
            safe_dose, is_safe, msg = self.guardrails.validate_and_clamp_dose(
                proposed_dose=0.15,
                previous_dose=0.12,
                dt_minutes=1.0 / 6.0,
                uncertainty_std=2.5 + lvl * 15.0,  # Uncertainty scales with noise
                signal_quality=min(sqi_abp, sqi_ppg),
            )

            results["abp_sqi"][lvl] = round(sqi_abp, 3)
            results["ppg_sqi"][lvl] = round(sqi_ppg, 3)
            results["fallback_triggered"][lvl] = not is_safe

        return results
