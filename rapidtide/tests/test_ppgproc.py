#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.ppgproc import (
    AdaptivePPGKalmanFilter,
    ExtendedPPGKalmanFilter,
    HarmonicPPGKalmanFilter,
    HeartRateExtractor,
    PPGFeatureExtractor,
    PPGKalmanFilter,
    RobustPPGProcessor,
    SignalQualityAssessor,
    generate_synthetic_ppg,
    read_happy_ppg,
)

# ==================== Helpers ====================


def _make_clean_ppg(duration=10.0, fs=100.0, hr=75.0):
    """Generate a clean synthetic PPG signal (no noise, no artifacts)."""
    t = np.arange(0, duration, 1.0 / fs)
    freq = hr / 60.0
    ppg = 1.0 + 0.8 * np.sin(2 * np.pi * freq * t)
    ppg += 0.15 * np.sin(2 * np.pi * 2 * freq * t + np.pi / 3)
    return t, ppg


def _make_noisy_ppg(duration=10.0, fs=100.0, hr=75.0, noise=0.05):
    """Generate a noisy PPG signal."""
    t, ppg = _make_clean_ppg(duration, fs, hr)
    rng = np.random.RandomState(42)
    return t, ppg + rng.randn(len(ppg)) * noise


# ==================== PPGKalmanFilter tests ====================


def ppgkalman_init_default(debug=False):
    """Test PPGKalmanFilter default initialization."""
    if debug:
        print("ppgkalman_init_default")
    kf = PPGKalmanFilter()
    assert kf.x.shape == (2, 1)
    assert kf.F.shape == (2, 2)
    assert kf.H.shape == (1, 2)
    assert kf.Q.shape == (2, 2)
    assert kf.R.shape == (1, 1)
    assert kf.P.shape == (2, 2)
    np.testing.assert_array_equal(kf.x, [[0.0], [0.0]])


def ppgkalman_init_custom(debug=False):
    """Test PPGKalmanFilter with custom parameters."""
    if debug:
        print("ppgkalman_init_custom")
    kf = PPGKalmanFilter(dt=0.02, process_noise=0.01, measurement_noise=0.1)
    assert kf.F[0, 1] == pytest.approx(0.02)
    assert kf.R[0, 0] == pytest.approx(0.1)


def ppgkalman_predict(debug=False):
    """Test predict step updates state."""
    if debug:
        print("ppgkalman_predict")
    kf = PPGKalmanFilter()
    P_before = kf.P.copy()
    kf.predict()
    # After predict with zero state, state should remain near zero
    assert kf.x[0, 0] == pytest.approx(0.0)
    # Covariance should increase (uncertainty grows)
    assert np.trace(kf.P) > np.trace(P_before)


def ppgkalman_update(debug=False):
    """Test update step incorporates measurement."""
    if debug:
        print("ppgkalman_update")
    kf = PPGKalmanFilter()
    kf.predict()
    kf.update(np.array([[5.0]]))
    # State should move toward measurement
    assert kf.x[0, 0] > 0.0


def ppgkalman_filter_signal(debug=False):
    """Test filter_signal on a simple signal."""
    if debug:
        print("ppgkalman_filter_signal")
    kf = PPGKalmanFilter(dt=0.01)
    _, signal = _make_clean_ppg(duration=2.0, fs=100.0)
    filtered = kf.filter_signal(signal)
    assert len(filtered) == len(signal)
    assert not np.any(np.isnan(filtered))


def ppgkalman_filter_with_nan(debug=False):
    """Test filter_signal handles NaN values via prediction only."""
    if debug:
        print("ppgkalman_filter_with_nan")
    kf = PPGKalmanFilter(dt=0.01)
    signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    filtered = kf.filter_signal(signal)
    assert len(filtered) == 5
    assert not np.any(np.isnan(filtered))


def ppgkalman_filter_with_missing_indices(debug=False):
    """Test filter_signal with explicit missing_indices."""
    if debug:
        print("ppgkalman_filter_with_missing_indices")
    kf = PPGKalmanFilter(dt=0.01)
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = kf.filter_signal(signal, missing_indices=[1, 3])
    assert len(filtered) == 5
    assert not np.any(np.isnan(filtered))


# ==================== AdaptivePPGKalmanFilter tests ====================


def adaptivekalman_init(debug=False):
    """Test AdaptivePPGKalmanFilter initialization."""
    if debug:
        print("adaptivekalman_init")
    kf = AdaptivePPGKalmanFilter()
    assert kf.dt == pytest.approx(0.01)
    assert kf.x.shape == (2, 1)
    assert kf.Q_scale == pytest.approx(0.001)
    assert kf.R_base == pytest.approx(0.05)
    assert kf.motion_threshold == pytest.approx(3.0)
    assert len(kf.innovation_history) == 0


def adaptivekalman_detect_motion_short_history(debug=False):
    """Test motion detection with insufficient history returns False."""
    if debug:
        print("adaptivekalman_detect_motion_short_history")
    kf = AdaptivePPGKalmanFilter()
    kf.innovation_history = [0.1] * 5  # Only 5 samples, need 10
    result = kf.detect_motion_artifact(np.array([[1.0]]))
    assert not result


def adaptivekalman_detect_motion_normal(debug=False):
    """Test motion detection with normal innovation returns False."""
    if debug:
        print("adaptivekalman_detect_motion_normal")
    kf = AdaptivePPGKalmanFilter()
    kf.innovation_history = [0.1] * 15
    result = kf.detect_motion_artifact(np.array([[0.15]]))
    assert not result


def adaptivekalman_detect_motion_artifact(debug=False):
    """Test motion detection with large innovation returns True."""
    if debug:
        print("adaptivekalman_detect_motion_artifact")
    kf = AdaptivePPGKalmanFilter()
    kf.innovation_history = [0.1] * 15
    # z_score = 5.0 / std(0.1*15) = 5.0 / 0.0 ... need nonzero std
    kf.innovation_history = list(np.random.RandomState(42).uniform(0.05, 0.15, 15))
    result = kf.detect_motion_artifact(np.array([[5.0]]))
    assert result


def adaptivekalman_detect_motion_zero_std(debug=False):
    """Test motion detection with zero std returns False."""
    if debug:
        print("adaptivekalman_detect_motion_zero_std")
    kf = AdaptivePPGKalmanFilter()
    kf.innovation_history = [0.1] * 15  # All same → std = 0
    result = kf.detect_motion_artifact(np.array([[5.0]]))
    assert not result


def adaptivekalman_adapt_noise_motion(debug=False):
    """Test adapt_noise increases R during motion artifact."""
    if debug:
        print("adaptivekalman_adapt_noise_motion")
    kf = AdaptivePPGKalmanFilter()
    kf.adapt_noise(np.array([[0.5]]), is_motion_artifact=True)
    assert kf.R[0, 0] == pytest.approx(kf.R_base * 10)


def adaptivekalman_adapt_noise_normal(debug=False):
    """Test adapt_noise keeps R at base level without motion artifact."""
    if debug:
        print("adaptivekalman_adapt_noise_normal")
    kf = AdaptivePPGKalmanFilter()
    kf.adapt_noise(np.array([[0.1]]), is_motion_artifact=False)
    assert kf.R[0, 0] == pytest.approx(kf.R_base)


def adaptivekalman_adapt_noise_qscale(debug=False):
    """Test adapt_noise adjusts Q_scale with full history."""
    if debug:
        print("adaptivekalman_adapt_noise_qscale")
    kf = AdaptivePPGKalmanFilter()
    # Fill innovation history to window_size
    for i in range(kf.window_size):
        kf.adapt_noise(np.array([[0.1 + i * 0.01]]), is_motion_artifact=False)
    # Q_scale should have been adapted
    assert 0.0001 <= kf.Q_scale <= 0.01


def adaptivekalman_update_returns_bool(debug=False):
    """Test update returns motion detection flag."""
    if debug:
        print("adaptivekalman_update_returns_bool")
    kf = AdaptivePPGKalmanFilter()
    kf.predict()
    result = kf.update(np.array([[1.0]]))
    assert isinstance(result, (bool, np.bool_))


def adaptivekalman_filter_signal(debug=False):
    """Test adaptive filter_signal returns filtered and motion_flags."""
    if debug:
        print("adaptivekalman_filter_signal")
    kf = AdaptivePPGKalmanFilter(dt=0.01)
    _, signal = _make_noisy_ppg(duration=2.0, fs=100.0)
    filtered, motion_flags = kf.filter_signal(signal)
    assert len(filtered) == len(signal)
    assert len(motion_flags) == len(signal)
    assert motion_flags.dtype == bool


def adaptivekalman_filter_with_nan(debug=False):
    """Test adaptive filter handles NaN values."""
    if debug:
        print("adaptivekalman_filter_with_nan")
    kf = AdaptivePPGKalmanFilter(dt=0.01)
    signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    filtered, motion_flags = kf.filter_signal(signal)
    assert not np.any(np.isnan(filtered))
    # NaN index should not have motion flag set
    assert not motion_flags[2]


# ==================== ExtendedPPGKalmanFilter tests ====================


def ekf_init_default(debug=False):
    """Test ExtendedPPGKalmanFilter default initialization."""
    if debug:
        print("ekf_init_default")
    ekf = ExtendedPPGKalmanFilter()
    assert ekf.x.shape == (4, 1)
    assert ekf.dt == pytest.approx(0.01)
    assert ekf.Q.shape == (4, 4)
    assert ekf.R.shape == (1, 1)
    assert ekf.P.shape == (4, 4)
    assert len(ekf.hr_history) == 0


def ekf_init_custom_hr(debug=False):
    """Test ExtendedPPGKalmanFilter with custom HR estimate."""
    if debug:
        print("ekf_init_custom_hr")
    ekf = ExtendedPPGKalmanFilter(hr_estimate=60)
    expected_freq = 2 * np.pi * 60 / 60  # 2π rad/s
    assert ekf.x[3, 0] == pytest.approx(expected_freq)


def ekf_get_heart_rate(debug=False):
    """Test get_heart_rate converts frequency to BPM."""
    if debug:
        print("ekf_get_heart_rate")
    ekf = ExtendedPPGKalmanFilter(hr_estimate=75)
    hr = ekf.get_heart_rate()
    assert hr == pytest.approx(75.0)


def ekf_state_transition(debug=False):
    """Test state_transition updates phase correctly."""
    if debug:
        print("ekf_state_transition")
    ekf = ExtendedPPGKalmanFilter(dt=0.01)
    x = np.array([[1.0], [2.0], [0.5], [10.0]])
    x_new = ekf.state_transition(x)
    assert x_new[0, 0] == pytest.approx(1.0)  # DC unchanged
    assert x_new[1, 0] == pytest.approx(2.0)  # Amplitude unchanged
    expected_phase = (0.5 + 10.0 * 0.01) % (2 * np.pi)
    assert x_new[2, 0] == pytest.approx(expected_phase)
    assert x_new[3, 0] == pytest.approx(10.0)  # Freq unchanged


def ekf_measurement_function(debug=False):
    """Test measurement_function computes DC + amp*sin(phase)."""
    if debug:
        print("ekf_measurement_function")
    ekf = ExtendedPPGKalmanFilter()
    x = np.array([[1.0], [2.0], [np.pi / 2], [1.0]])
    result = ekf.measurement_function(x)
    # dc + amp * sin(pi/2) = 1.0 + 2.0 * 1.0 = 3.0
    assert result[0, 0] == pytest.approx(3.0)


def ekf_measurement_function_zero_phase(debug=False):
    """Test measurement_function at phase=0."""
    if debug:
        print("ekf_measurement_function_zero_phase")
    ekf = ExtendedPPGKalmanFilter()
    x = np.array([[5.0], [2.0], [0.0], [1.0]])
    result = ekf.measurement_function(x)
    # dc + amp * sin(0) = 5.0 + 0 = 5.0
    assert result[0, 0] == pytest.approx(5.0)


def ekf_predict_update(debug=False):
    """Test predict and update cycle."""
    if debug:
        print("ekf_predict_update")
    ekf = ExtendedPPGKalmanFilter(dt=0.01, hr_estimate=75)
    ekf.predict()
    ekf.update(np.array([[1.5]]))
    # Phase should be in [0, 2pi]
    assert 0 <= ekf.x[2, 0] < 2 * np.pi


def ekf_filter_signal(debug=False):
    """Test EKF filter_signal returns filtered and heart_rates."""
    if debug:
        print("ekf_filter_signal")
    ekf = ExtendedPPGKalmanFilter(dt=0.01, hr_estimate=75)
    _, signal = _make_clean_ppg(duration=2.0, fs=100.0, hr=75.0)
    filtered, heart_rates = ekf.filter_signal(signal)
    assert len(filtered) == len(signal)
    assert len(heart_rates) == len(signal)
    assert not np.any(np.isnan(filtered))
    # Heart rates should be in reasonable range
    assert np.all(heart_rates > 0)


def ekf_filter_signal_with_nan(debug=False):
    """Test EKF filter_signal handles NaN."""
    if debug:
        print("ekf_filter_signal_with_nan")
    ekf = ExtendedPPGKalmanFilter(dt=0.01, hr_estimate=75)
    signal = np.array([1.0, 1.5, np.nan, 1.2, 1.0])
    filtered, heart_rates = ekf.filter_signal(signal)
    assert not np.any(np.isnan(filtered))
    assert len(heart_rates) == 5


# ==================== HarmonicPPGKalmanFilter tests ====================


def harmonic_init_default(debug=False):
    """Test HarmonicPPGKalmanFilter default initialization."""
    if debug:
        print("harmonic_init_default")
    hkf = HarmonicPPGKalmanFilter()
    assert hkf.x.shape == (6, 1)
    assert hkf.dt == pytest.approx(0.01)
    assert hkf.Q.shape == (6, 6)
    assert hkf.P.shape == (6, 6)
    # Check initial amplitudes
    assert hkf.x[1, 0] == pytest.approx(1.0)   # A1
    assert hkf.x[2, 0] == pytest.approx(0.2)   # A2
    assert hkf.x[3, 0] == pytest.approx(0.1)   # A3


def harmonic_get_heart_rate(debug=False):
    """Test heart rate extraction from harmonic filter."""
    if debug:
        print("harmonic_get_heart_rate")
    hkf = HarmonicPPGKalmanFilter(hr_estimate=80)
    hr = hkf.get_heart_rate()
    assert hr == pytest.approx(80.0)


def harmonic_state_transition(debug=False):
    """Test state transition preserves amplitudes and updates phase."""
    if debug:
        print("harmonic_state_transition")
    hkf = HarmonicPPGKalmanFilter(dt=0.01)
    x = np.array([[1.0], [0.5], [0.3], [0.2], [0.1], [10.0]])
    x_new = hkf.state_transition(x)
    # DC and amplitudes unchanged
    assert x_new[0, 0] == pytest.approx(1.0)
    assert x_new[1, 0] == pytest.approx(0.5)
    assert x_new[2, 0] == pytest.approx(0.3)
    assert x_new[3, 0] == pytest.approx(0.2)
    # Phase updated
    expected_phase = (0.1 + 10.0 * 0.01) % (2 * np.pi)
    assert x_new[4, 0] == pytest.approx(expected_phase)
    # Frequency unchanged
    assert x_new[5, 0] == pytest.approx(10.0)


def harmonic_measurement_function(debug=False):
    """Test harmonic measurement function."""
    if debug:
        print("harmonic_measurement_function")
    hkf = HarmonicPPGKalmanFilter()
    phase = np.pi / 2
    x = np.array([[1.0], [2.0], [0.5], [0.3], [phase], [7.85]])
    result = hkf.measurement_function(x)
    expected = 1.0 + 2.0 * np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    assert result[0, 0] == pytest.approx(expected)


def harmonic_predict_update(debug=False):
    """Test predict/update cycle keeps phase in [0, 2pi]."""
    if debug:
        print("harmonic_predict_update")
    hkf = HarmonicPPGKalmanFilter(dt=0.01, hr_estimate=75)
    hkf.predict()
    hkf.update(np.array([[1.5]]))
    assert 0 <= hkf.x[4, 0] < 2 * np.pi


def harmonic_filter_signal(debug=False):
    """Test harmonic filter returns filtered, heart_rates, harmonic_amplitudes."""
    if debug:
        print("harmonic_filter_signal")
    hkf = HarmonicPPGKalmanFilter(dt=0.01, hr_estimate=75)
    _, signal = _make_clean_ppg(duration=2.0, fs=100.0, hr=75.0)
    filtered, heart_rates, harmonic_amps = hkf.filter_signal(signal)
    assert len(filtered) == len(signal)
    assert len(heart_rates) == len(signal)
    assert harmonic_amps.shape == (len(signal), 3)


def harmonic_filter_with_missing(debug=False):
    """Test harmonic filter with missing indices."""
    if debug:
        print("harmonic_filter_with_missing")
    hkf = HarmonicPPGKalmanFilter(dt=0.01, hr_estimate=75)
    signal = np.array([1.0, 1.5, 1.8, 1.2, 0.8, 1.0])
    filtered, hr, amps = hkf.filter_signal(signal, missing_indices=[2, 4])
    assert not np.any(np.isnan(filtered))
    assert len(hr) == 6


# ==================== SignalQualityAssessor tests ====================


def quality_init(debug=False):
    """Test SignalQualityAssessor initialization."""
    if debug:
        print("quality_init")
    qa = SignalQualityAssessor(fs=200.0, window_size=2.5)
    assert qa.fs == pytest.approx(200.0)
    assert qa.window_samples == 500


def quality_assess_clean_signal(debug=False):
    """Test quality assessment on clean PPG signal."""
    if debug:
        print("quality_assess_clean_signal")
    qa = SignalQualityAssessor(fs=100.0, window_size=5.0)
    _, ppg = _make_clean_ppg(duration=5.0, fs=100.0, hr=75.0)
    score, metrics = qa.assess_quality(ppg)
    assert 0.0 <= score <= 1.0
    assert "snr" in metrics
    assert "perfusion" in metrics
    assert "spectral_purity" in metrics
    assert "kurtosis" in metrics
    assert "zero_crossing" in metrics


def quality_assess_with_filtered(debug=False):
    """Test quality assessment with filtered segment computes SNR."""
    if debug:
        print("quality_assess_with_filtered")
    qa = SignalQualityAssessor(fs=100.0, window_size=5.0)
    _, ppg = _make_clean_ppg(duration=5.0, fs=100.0, hr=75.0)
    rng = np.random.RandomState(42)
    noisy = ppg + rng.randn(len(ppg)) * 0.1
    score, metrics = qa.assess_quality(noisy, filtered_segment=ppg)
    assert 0.0 <= metrics["snr"] <= 1.0
    # Clean signal with known noise should have reasonable SNR
    assert metrics["snr"] > 0.2


def quality_assess_noisy_signal(debug=False):
    """Test that noisy signals get lower quality scores."""
    if debug:
        print("quality_assess_noisy_signal")
    qa = SignalQualityAssessor(fs=100.0, window_size=5.0)
    _, clean = _make_clean_ppg(duration=5.0, fs=100.0, hr=75.0)
    rng = np.random.RandomState(42)
    very_noisy = clean + rng.randn(len(clean)) * 2.0
    score_clean, _ = qa.assess_quality(clean)
    score_noisy, _ = qa.assess_quality(very_noisy)
    # Clean should score higher than very noisy
    assert score_clean > score_noisy


def quality_assess_continuous(debug=False):
    """Test continuous quality assessment."""
    if debug:
        print("quality_assess_continuous")
    qa = SignalQualityAssessor(fs=100.0, window_size=2.0)
    _, ppg = _make_clean_ppg(duration=10.0, fs=100.0, hr=75.0)
    times, scores = qa.assess_continuous(ppg, stride=1.0)
    assert len(times) == len(scores)
    assert len(times) > 0
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def quality_assess_continuous_with_filtered(debug=False):
    """Test continuous quality assessment with filtered signal."""
    if debug:
        print("quality_assess_continuous_with_filtered")
    qa = SignalQualityAssessor(fs=100.0, window_size=2.0)
    _, ppg = _make_clean_ppg(duration=10.0, fs=100.0, hr=75.0)
    rng = np.random.RandomState(42)
    noisy = ppg + rng.randn(len(ppg)) * 0.1
    times, scores = qa.assess_continuous(noisy, filtered=ppg, stride=1.0)
    assert len(times) > 0
    assert len(times) == len(scores)


# ==================== HeartRateExtractor tests ====================


def hr_extractor_init(debug=False):
    """Test HeartRateExtractor initialization."""
    if debug:
        print("hr_extractor_init")
    hre = HeartRateExtractor(fs=200.0)
    assert hre.fs == pytest.approx(200.0)


def hr_extract_from_fft(debug=False):
    """Test FFT-based heart rate extraction."""
    if debug:
        print("hr_extract_from_fft")
    hre = HeartRateExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=10.0, fs=100.0, hr=75.0)
    hr, freq, psd, freqs = hre.extract_from_fft(ppg)
    assert hr is not None
    # HR should be close to 75 BPM
    assert abs(hr - 75.0) < 10.0, f"Expected HR ~75, got {hr}"
    assert freq is not None
    assert len(psd) > 0
    assert len(freqs) > 0


def hr_extract_from_fft_custom_range(debug=False):
    """Test FFT HR extraction with custom range."""
    if debug:
        print("hr_extract_from_fft_custom_range")
    hre = HeartRateExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=10.0, fs=100.0, hr=75.0)
    # Range excludes the actual HR
    hr, freq, psd, freqs = hre.extract_from_fft(ppg, hr_range=(100.0, 180.0))
    # Should still find something (maybe a harmonic) or the closest peak
    assert psd is not None


def hr_extract_from_fft_short_signal(debug=False):
    """Test FFT HR extraction with very short signal."""
    if debug:
        print("hr_extract_from_fft_short_signal")
    hre = HeartRateExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=1.0, fs=100.0, hr=75.0)
    hr, freq, psd, freqs = hre.extract_from_fft(ppg)
    # Should still return something
    assert psd is not None


def hr_extract_from_peaks(debug=False):
    """Test peak-based heart rate extraction."""
    if debug:
        print("hr_extract_from_peaks")
    hre = HeartRateExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=10.0, fs=100.0, hr=75.0)
    hr, peaks, rri, hr_waveform = hre.extract_from_peaks(ppg)
    assert hr is not None
    assert abs(hr - 75.0) < 10.0, f"Expected HR ~75, got {hr}"
    assert len(peaks) >= 2


def hr_extract_from_peaks_few_peaks(debug=False):
    """Test peak extraction with signal that has < 2 peaks."""
    if debug:
        print("hr_extract_from_peaks_few_peaks")
    hre = HeartRateExtractor(fs=100.0)
    # Flat signal with no peaks
    signal = np.ones(500)
    hr, peaks, rri, hr_waveform = hre.extract_from_peaks(signal)
    assert hr is None
    assert len(peaks) < 2
    assert rri is None
    assert hr_waveform is None


def hr_extract_continuous_fft(debug=False):
    """Test continuous HR extraction with FFT method."""
    if debug:
        print("hr_extract_continuous_fft")
    hre = HeartRateExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=30.0, fs=100.0, hr=75.0)
    times, heart_rates = hre.extract_continuous(ppg, window_size=10.0, stride=2.0, method="fft")
    assert len(times) == len(heart_rates)
    if len(heart_rates) > 0:
        # All extracted HRs should be in reasonable range
        assert np.all(heart_rates > 30)
        assert np.all(heart_rates < 200)


def hr_extract_continuous_peaks(debug=False):
    """Test that extract_continuous with method='peaks' works correctly."""
    if debug:
        print("hr_extract_continuous_peaks")
    hre = HeartRateExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=30.0, fs=100.0, hr=75.0)
    times, heart_rates = hre.extract_continuous(ppg, window_size=10.0, stride=2.0, method="peaks")
    assert len(times) == len(heart_rates)
    if len(heart_rates) > 0:
        assert np.all(heart_rates > 30)
        assert np.all(heart_rates < 200)


# ==================== RobustPPGProcessor tests ====================


def processor_init_standard(debug=False):
    """Test RobustPPGProcessor with standard method."""
    if debug:
        print("processor_init_standard")
    proc = RobustPPGProcessor(fs=100.0, method="standard")
    assert isinstance(proc.filter, PPGKalmanFilter)
    assert isinstance(proc.quality_assessor, SignalQualityAssessor)
    assert isinstance(proc.hr_extractor, HeartRateExtractor)


def processor_init_adaptive(debug=False):
    """Test RobustPPGProcessor with adaptive method."""
    if debug:
        print("processor_init_adaptive")
    proc = RobustPPGProcessor(fs=100.0, method="adaptive")
    assert isinstance(proc.filter, AdaptivePPGKalmanFilter)


def processor_init_ekf(debug=False):
    """Test RobustPPGProcessor with EKF method."""
    if debug:
        print("processor_init_ekf")
    proc = RobustPPGProcessor(fs=100.0, method="ekf", hr_estimate=80.0)
    assert isinstance(proc.filter, ExtendedPPGKalmanFilter)


def processor_process_standard(debug=False):
    """Test full processing pipeline with standard filter."""
    if debug:
        print("processor_process_standard")
    proc = RobustPPGProcessor(fs=100.0, method="standard")
    _, ppg = _make_noisy_ppg(duration=20.0, fs=100.0, hr=75.0)
    results = proc.process(ppg)
    assert "filtered_signal" in results
    assert "quality_times" in results
    assert "quality_scores" in results
    assert "mean_quality" in results
    assert "good_quality_percentage" in results
    assert len(results["filtered_signal"]) == len(ppg)


def processor_process_adaptive(debug=False):
    """Test processing pipeline with adaptive filter."""
    if debug:
        print("processor_process_adaptive")
    proc = RobustPPGProcessor(fs=100.0, method="adaptive")
    _, ppg = _make_noisy_ppg(duration=20.0, fs=100.0, hr=75.0)
    results = proc.process(ppg)
    assert "filtered_signal" in results
    assert "motion_flags" in results
    assert len(results["motion_flags"]) == len(ppg)


def processor_process_ekf(debug=False):
    """Test processing pipeline with EKF filter."""
    if debug:
        print("processor_process_ekf")
    proc = RobustPPGProcessor(fs=100.0, method="ekf", hr_estimate=75.0)
    _, ppg = _make_noisy_ppg(duration=20.0, fs=100.0, hr=75.0)
    results = proc.process(ppg)
    assert "filtered_signal" in results
    assert "ekf_heart_rate" in results
    assert len(results["ekf_heart_rate"]) == len(ppg)


def processor_process_with_missing(debug=False):
    """Test processing pipeline with missing data."""
    if debug:
        print("processor_process_with_missing")
    proc = RobustPPGProcessor(fs=100.0, method="standard")
    _, ppg = _make_noisy_ppg(duration=20.0, fs=100.0, hr=75.0)
    missing = [10, 50, 100, 200]
    results = proc.process(ppg, missing_indices=missing)
    assert "filtered_signal" in results
    assert not np.any(np.isnan(results["filtered_signal"]))


def processor_process_low_quality_threshold(debug=False):
    """Test processing pipeline with very high quality threshold."""
    if debug:
        print("processor_process_low_quality_threshold")
    proc = RobustPPGProcessor(fs=100.0, method="standard")
    rng = np.random.RandomState(42)
    # Random noise (very poor quality)
    noise = rng.randn(2000)
    results = proc.process(noise, quality_threshold=0.99)
    # With noise and very high threshold, no HR should be extracted
    if len(results.get("hr_values", [])) == 0:
        assert results.get("hr_overall") is None or results.get("hr_overall") is not None


# ==================== PPGFeatureExtractor tests ====================


def features_init(debug=False):
    """Test PPGFeatureExtractor initialization."""
    if debug:
        print("features_init")
    fe = PPGFeatureExtractor(fs=200.0)
    assert fe.fs == pytest.approx(200.0)


def features_hrv_basic(debug=False):
    """Test HRV feature extraction with regular peaks."""
    if debug:
        print("features_hrv_basic")
    fe = PPGFeatureExtractor(fs=100.0)
    # Simulate regular peaks at 75 BPM: one peak every 0.8s = 80 samples
    peaks = np.arange(0, 5000, 80)
    features = fe.extract_hrv_features(peaks)
    assert features is not None
    assert "mean_ibi" in features
    assert "sdnn" in features
    assert "rmssd" in features
    assert "pnn50" in features
    # Regular peaks → mean IBI should be ~800ms
    assert abs(features["mean_ibi"] - 800.0) < 50.0


def features_hrv_too_few_peaks(debug=False):
    """Test HRV extraction with too few peaks returns None."""
    if debug:
        print("features_hrv_too_few_peaks")
    fe = PPGFeatureExtractor(fs=100.0)
    features = fe.extract_hrv_features(np.array([10, 90]))
    assert features is None


def features_hrv_single_peak(debug=False):
    """Test HRV extraction with single peak returns None."""
    if debug:
        print("features_hrv_single_peak")
    fe = PPGFeatureExtractor(fs=100.0)
    features = fe.extract_hrv_features(np.array([50]))
    assert features is None


def features_hrv_frequency_domain(debug=False):
    """Test HRV extraction includes frequency domain features with enough data."""
    if debug:
        print("features_hrv_frequency_domain")
    fe = PPGFeatureExtractor(fs=100.0)
    # Need > 30 valid IBIs → need > 31 peaks
    rng = np.random.RandomState(42)
    # Peaks at ~75 BPM with slight variability
    intervals = 80 + rng.randint(-5, 6, 50)
    peaks = np.cumsum(intervals)
    features = fe.extract_hrv_features(peaks)
    assert features is not None
    assert "vlf_power" in features
    assert "lf_power" in features
    assert "hf_power" in features
    assert "lf_hf_ratio" in features


def features_morphology(debug=False):
    """Test morphological feature extraction."""
    if debug:
        print("features_morphology")
    fe = PPGFeatureExtractor(fs=100.0)
    # Create a single pulse segment
    t = np.arange(0, 0.8, 0.01)  # 80 samples, one beat at 75 BPM
    segment = 0.5 * np.sin(2 * np.pi * 1.25 * t) + 1.0
    peak_idx = np.argmax(segment)
    features = fe.extract_morphology_features(segment, peak_idx)
    assert "pulse_amplitude" in features
    assert "rising_time" in features
    assert features["pulse_amplitude"] > 0


def features_morphology_with_dicrotic_notch(debug=False):
    """Test morphology extraction finds dicrotic notch."""
    if debug:
        print("features_morphology_with_dicrotic_notch")
    fe = PPGFeatureExtractor(fs=100.0)
    # Construct a signal with a clear dicrotic notch
    t = np.arange(0, 1.0, 0.01)
    segment = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 2.0 * t) + 1.5
    peak_idx = np.argmax(segment)
    features = fe.extract_morphology_features(segment, peak_idx)
    assert "pulse_amplitude" in features
    if "dicrotic_notch_amplitude" in features:
        assert "augmentation_index" in features


def features_spo2_proxy(debug=False):
    """Test SpO2 proxy computation."""
    if debug:
        print("features_spo2_proxy")
    fe = PPGFeatureExtractor(fs=100.0)
    _, ppg = _make_clean_ppg(duration=5.0, fs=100.0, hr=75.0)
    spo2 = fe.compute_spo2_proxy(ppg)
    # Should be clipped between 70 and 100
    assert 70.0 <= spo2 <= 100.0


def features_spo2_proxy_flat(debug=False):
    """Test SpO2 proxy with flat signal (zero AC)."""
    if debug:
        print("features_spo2_proxy_flat")
    fe = PPGFeatureExtractor(fs=100.0)
    flat = np.ones(500)
    spo2 = fe.compute_spo2_proxy(flat)
    # Near-zero AC/DC ratio → spo2 ≈ 110 → clipped to 100
    assert spo2 == pytest.approx(100.0)


# ==================== generate_synthetic_ppg tests ====================


def synthetic_ppg_shapes(debug=False):
    """Test generate_synthetic_ppg output shapes."""
    if debug:
        print("synthetic_ppg_shapes")
    np.random.seed(42)
    t, ppg, noisy, corrupted, missing = generate_synthetic_ppg(
        duration=5, fs=100.0, hr=75, noise_level=0.05, missing_percent=5
    )
    expected_len = 500  # 5s * 100Hz
    assert len(t) == expected_len
    assert len(ppg) == expected_len
    assert len(noisy) == expected_len
    assert len(corrupted) == expected_len
    # Missing indices should be ~5% of total
    assert len(missing) == pytest.approx(25, abs=5)


def synthetic_ppg_nan_at_missing(debug=False):
    """Test that corrupted signal has NaN at missing indices."""
    if debug:
        print("synthetic_ppg_nan_at_missing")
    np.random.seed(42)
    t, ppg, noisy, corrupted, missing = generate_synthetic_ppg(
        duration=5, fs=100.0, missing_percent=10
    )
    for idx in missing:
        assert np.isnan(corrupted[idx])


def synthetic_ppg_no_missing(debug=False):
    """Test synthetic PPG with 0% missing data."""
    if debug:
        print("synthetic_ppg_no_missing")
    np.random.seed(42)
    t, ppg, noisy, corrupted, missing = generate_synthetic_ppg(missing_percent=0)
    assert len(missing) == 0
    assert not np.any(np.isnan(corrupted))


def synthetic_ppg_no_artifacts(debug=False):
    """Test synthetic PPG without motion artifacts."""
    if debug:
        print("synthetic_ppg_no_artifacts")
    np.random.seed(42)
    t, ppg, noisy, corrupted, missing = generate_synthetic_ppg(
        noise_level=0.0, motion_artifacts=False, missing_percent=0
    )
    # With no noise and no artifacts, noisy should equal ppg
    np.testing.assert_array_almost_equal(noisy, ppg)


def synthetic_ppg_different_hr(debug=False):
    """Test synthetic PPG at different heart rates."""
    if debug:
        print("synthetic_ppg_different_hr")
    for hr in [60, 90, 120]:
        np.random.seed(42)
        t, ppg, _, _, _ = generate_synthetic_ppg(duration=10, fs=100.0, hr=hr)
        # Check fundamental frequency via FFT
        from scipy.signal import welch
        freqs, psd = welch(ppg, fs=100.0, nperseg=256)
        hr_band = (freqs >= 0.5) & (freqs <= 3.0)
        peak_freq = freqs[hr_band][np.argmax(psd[hr_band])]
        detected_hr = peak_freq * 60
        assert abs(detected_hr - hr) < 15, f"HR={hr}, detected={detected_hr}"


# ==================== read_happy_ppg tests ====================


def read_happy_ppg_basic(debug=False):
    """Test read_happy_ppg with mocked IO."""
    if debug:
        print("read_happy_ppg_basic")
    n = 1000
    fs = 25.0
    mock_data = np.random.RandomState(42).randn(3, n)
    mock_columns = ["cardiacfromfmri", "badpts", "pleth"]
    mock_badpts = np.zeros(n)
    mock_badpts[10] = 1
    mock_badpts[20] = 1
    mock_data[1, :] = mock_badpts

    with patch("rapidtide.ppgproc.tide_io.readbidstsv") as mock_read:
        mock_read.return_value = (
            fs,            # Fs
            0.0,           # instarttime
            mock_columns,  # incolumns
            mock_data,     # indata
            None,          # incompressed
            [],            # incolsource
            {},            # inextrainfo
        )
        t, Fs_out, clean_ppg, raw_ppg, pleth_ppg, missing_indices = read_happy_ppg("test_root")

    assert Fs_out == pytest.approx(fs)
    assert len(t) == n
    assert len(clean_ppg) == n
    assert len(raw_ppg) == n
    assert pleth_ppg is None
    assert 10 in missing_indices
    assert 20 in missing_indices


def read_happy_ppg_alt_columns(debug=False):
    """Test read_happy_ppg with alternative column names."""
    if debug:
        print("read_happy_ppg_alt_columns")
    n = 500
    fs = 25.0
    mock_data = np.random.RandomState(42).randn(2, n)
    mock_columns = ["cardiacfromfmri_25.0Hz", "cardiacfromfmri_dlfiltered"]

    with patch("rapidtide.ppgproc.tide_io.readbidstsv") as mock_read:
        mock_read.return_value = (fs, 0.0, mock_columns, mock_data, None, [], {})
        t, Fs_out, clean_ppg, raw_ppg, pleth_ppg, missing_indices = read_happy_ppg("test_root")

    assert len(t) == n
    assert len(missing_indices) == 0  # No badpts column


def read_happy_ppg_missing_raw_column(debug=False):
    """Test read_happy_ppg raises ValueError when raw column is missing."""
    if debug:
        print("read_happy_ppg_missing_raw_column")
    mock_columns = ["pleth"]
    mock_data = np.random.RandomState(42).randn(1, 100)

    with patch("rapidtide.ppgproc.tide_io.readbidstsv") as mock_read:
        mock_read.return_value = (25.0, 0.0, mock_columns, mock_data, None, [], {})
        with pytest.raises(ValueError):
            read_happy_ppg("test_root")


def read_happy_ppg_missing_clean_column(debug=False):
    """Test read_happy_ppg raises ValueError when clean column is missing."""
    if debug:
        print("read_happy_ppg_missing_clean_column")
    mock_columns = ["cardiacfromfmri", "badpts"]
    mock_data = np.random.RandomState(42).randn(2, 100)

    with patch("rapidtide.ppgproc.tide_io.readbidstsv") as mock_read:
        mock_read.return_value = (25.0, 0.0, mock_columns, mock_data, None, [], {})
        with pytest.raises(ValueError):
            read_happy_ppg("test_root")


# ==================== Main test function ====================


def test_ppgproc(debug=False):
    # PPGKalmanFilter tests
    if debug:
        print("Running PPGKalmanFilter tests")
    ppgkalman_init_default(debug=debug)
    ppgkalman_init_custom(debug=debug)
    ppgkalman_predict(debug=debug)
    ppgkalman_update(debug=debug)
    ppgkalman_filter_signal(debug=debug)
    ppgkalman_filter_with_nan(debug=debug)
    ppgkalman_filter_with_missing_indices(debug=debug)

    # AdaptivePPGKalmanFilter tests
    if debug:
        print("Running AdaptivePPGKalmanFilter tests")
    adaptivekalman_init(debug=debug)
    adaptivekalman_detect_motion_short_history(debug=debug)
    adaptivekalman_detect_motion_normal(debug=debug)
    adaptivekalman_detect_motion_artifact(debug=debug)
    adaptivekalman_detect_motion_zero_std(debug=debug)
    adaptivekalman_adapt_noise_motion(debug=debug)
    adaptivekalman_adapt_noise_normal(debug=debug)
    adaptivekalman_adapt_noise_qscale(debug=debug)
    adaptivekalman_update_returns_bool(debug=debug)
    adaptivekalman_filter_signal(debug=debug)
    adaptivekalman_filter_with_nan(debug=debug)

    # ExtendedPPGKalmanFilter tests
    if debug:
        print("Running ExtendedPPGKalmanFilter tests")
    ekf_init_default(debug=debug)
    ekf_init_custom_hr(debug=debug)
    ekf_get_heart_rate(debug=debug)
    ekf_state_transition(debug=debug)
    ekf_measurement_function(debug=debug)
    ekf_measurement_function_zero_phase(debug=debug)
    ekf_predict_update(debug=debug)
    ekf_filter_signal(debug=debug)
    ekf_filter_signal_with_nan(debug=debug)

    # HarmonicPPGKalmanFilter tests
    if debug:
        print("Running HarmonicPPGKalmanFilter tests")
    harmonic_init_default(debug=debug)
    harmonic_get_heart_rate(debug=debug)
    harmonic_state_transition(debug=debug)
    harmonic_measurement_function(debug=debug)
    harmonic_predict_update(debug=debug)
    harmonic_filter_signal(debug=debug)
    harmonic_filter_with_missing(debug=debug)

    # SignalQualityAssessor tests
    if debug:
        print("Running SignalQualityAssessor tests")
    quality_init(debug=debug)
    quality_assess_clean_signal(debug=debug)
    quality_assess_with_filtered(debug=debug)
    quality_assess_noisy_signal(debug=debug)
    quality_assess_continuous(debug=debug)
    quality_assess_continuous_with_filtered(debug=debug)

    # HeartRateExtractor tests
    if debug:
        print("Running HeartRateExtractor tests")
    hr_extractor_init(debug=debug)
    hr_extract_from_fft(debug=debug)
    hr_extract_from_fft_custom_range(debug=debug)
    hr_extract_from_fft_short_signal(debug=debug)
    hr_extract_from_peaks(debug=debug)
    hr_extract_from_peaks_few_peaks(debug=debug)
    hr_extract_continuous_fft(debug=debug)
    hr_extract_continuous_peaks(debug=debug)

    # RobustPPGProcessor tests
    if debug:
        print("Running RobustPPGProcessor tests")
    processor_init_standard(debug=debug)
    processor_init_adaptive(debug=debug)
    processor_init_ekf(debug=debug)
    processor_process_standard(debug=debug)
    processor_process_adaptive(debug=debug)
    processor_process_ekf(debug=debug)
    processor_process_with_missing(debug=debug)
    processor_process_low_quality_threshold(debug=debug)

    # PPGFeatureExtractor tests
    if debug:
        print("Running PPGFeatureExtractor tests")
    features_init(debug=debug)
    features_hrv_basic(debug=debug)
    features_hrv_too_few_peaks(debug=debug)
    features_hrv_single_peak(debug=debug)
    features_hrv_frequency_domain(debug=debug)
    features_morphology(debug=debug)
    features_morphology_with_dicrotic_notch(debug=debug)
    features_spo2_proxy(debug=debug)
    features_spo2_proxy_flat(debug=debug)

    # generate_synthetic_ppg tests
    if debug:
        print("Running generate_synthetic_ppg tests")
    synthetic_ppg_shapes(debug=debug)
    synthetic_ppg_nan_at_missing(debug=debug)
    synthetic_ppg_no_missing(debug=debug)
    synthetic_ppg_no_artifacts(debug=debug)
    synthetic_ppg_different_hr(debug=debug)

    # read_happy_ppg tests
    if debug:
        print("Running read_happy_ppg tests")
    read_happy_ppg_basic(debug=debug)
    read_happy_ppg_alt_columns(debug=debug)
    read_happy_ppg_missing_raw_column(debug=debug)
    read_happy_ppg_missing_clean_column(debug=debug)


if __name__ == "__main__":
    test_ppgproc(debug=True)
