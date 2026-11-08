"""
Unit and integration tests for data pipeline modules:
- signal_filter.py
- sqi.py
- beat_detector.py
- morphomics.py
- synthetic_generator.py
"""

import pytest
import numpy as np

from src.data_pipeline.signal_filter import SignalFilter
from src.data_pipeline.sqi import SignalQualityIndex
from src.data_pipeline.beat_detector import BeatDetector
from src.data_pipeline.morphomics import WaveformMorphomics
from src.data_pipeline.synthetic_generator import SyntheticPatientGenerator


class TestDataPipeline:
    """Test suite ensuring 100% mathematical validity and edge case safety."""

    @pytest.fixture
    def synthetic_signals(self):
        """Generates 2 minutes of synthetic 125 Hz ABP and PPG for testing."""
        gen = SyntheticPatientGenerator(sampling_rate_hz=125.0, baseline_map=75.0, random_seed=42)
        state = gen.simulate_hemodynamic_state(duration_minutes=2.0)
        t, abp, ppg = gen.synthesize_waveforms(state, noise_level=0.01, inject_artifacts=False)
        return t, abp, ppg

    def test_synthetic_generator_outputs(self, synthetic_signals):
        t, abp, ppg = synthetic_signals
        assert len(t) == len(abp) == len(ppg)
        assert len(t) == int(2.0 * 60.0 * 125.0)
        # Check physiological bounds
        assert 35.0 < np.mean(abp) < 140.0
        assert np.max(abp) > np.min(abp) + 20.0  # Pulse pressure exists

    def test_signal_filter_bandpass_and_notch(self, synthetic_signals):
        _, abp, ppg = synthetic_signals
        filt = SignalFilter(sampling_rate_hz=125.0)

        # Inject 60 Hz hum
        t = np.arange(len(abp)) / 125.0
        abp_corrupted = abp + 15.0 * np.sin(2 * np.pi * 60.0 * t)

        cleaned_abp, cleaned_ppg, artifacts = filt.process_dual_stream(abp_corrupted, ppg)
        assert len(cleaned_abp) == len(abp)
        assert len(cleaned_ppg) == len(ppg)
        assert not np.isnan(cleaned_abp).any()
        assert not np.isnan(cleaned_ppg).any()

        # Check 60 Hz hum attenuation
        fft_raw = np.abs(np.fft.rfft(abp_corrupted))
        fft_clean = np.abs(np.fft.rfft(cleaned_abp))
        freqs = np.fft.rfftfreq(len(abp), 1.0 / 125.0)
        idx_60hz = np.argmin(np.abs(freqs - 60.0))
        assert fft_clean[idx_60hz] < 0.20 * fft_raw[idx_60hz]

    def test_artifact_detection_and_clamping(self):
        filt = SignalFilter(sampling_rate_hz=125.0)
        sig = np.full(500, 80.0)
        # Insert extreme flush spike
        sig[100:120] = 295.0
        # Insert out of bounds floor
        sig[250:270] = 5.0

        cleaned, mask = filt.detect_artifacts_and_clamp(sig)
        assert np.all(cleaned >= 20.0)
        assert np.all(cleaned <= 260.0)
        assert np.any(mask[100:120])
        assert np.any(mask[250:270])

    def test_signal_quality_index(self, synthetic_signals):
        _, abp, ppg = synthetic_signals
        sqi_eval = SignalQualityIndex(sampling_rate_hz=125.0)

        # Good signal test (first 30 seconds)
        clean_window = abp[: int(30 * 125)]
        metrics_clean = sqi_eval.evaluate_abp_window(clean_window)
        assert metrics_clean["composite_sqi"] >= 0.70
        assert metrics_clean["is_usable"] == 1.0

        # Corrupted / flatline test
        flat_window = np.full(int(30 * 125), 40.0)
        metrics_bad = sqi_eval.evaluate_abp_window(flat_window)
        assert metrics_bad["composite_sqi"] < 0.50
        assert metrics_bad["is_usable"] == 0.0

    def test_beat_detector_and_landmarks(self, synthetic_signals):
        _, abp, ppg = synthetic_signals
        detector = BeatDetector(sampling_rate_hz=125.0)
        window = abp[: int(30 * 125)]

        beats = detector.segment_abp_beats(window)
        assert len(beats) >= 25  # ~30-40 beats in 30 seconds for normal HR
        for b in beats:
            # Systolic peak must be greater than diastolic foot
            assert b["peak_val"] > b["foot_val"]
            assert b["pulse_pressure"] > 10.0
            # Dicrotic notch must lie between foot and peak (allowing for inter-beat drift)
            assert b["notch_val"] <= b["peak_val"]
            assert b["notch_val"] >= b["foot_val"] - 3.0
            assert 40.0 <= b["heart_rate_bpm"] <= 180.0

    def test_morphomics_extraction(self, synthetic_signals):
        _, abp, ppg = synthetic_signals
        morph = WaveformMorphomics(sampling_rate_hz=125.0)
        abp_win = abp[: int(30 * 125)]
        ppg_win = ppg[: int(30 * 125)]

        features = morph.extract_window_morphomics(abp_win, ppg_win)
        assert "map_integrated" in features
        assert "dicrotic_notch_ratio_mean" in features
        assert "vascular_tone_index" in features

        # Dicrotic notch ratio must be in physiological (0.10, 0.90)
        assert 0.10 <= features["dicrotic_notch_ratio_mean"] <= 0.90
        # Vascular tone index in [0, 1]
        assert 0.0 <= features["vascular_tone_index"] <= 1.0
        # MAP must be reasonable
        assert 40.0 <= features["map_integrated"] <= 140.0
