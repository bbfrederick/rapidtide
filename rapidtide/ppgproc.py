#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


class PPGKalmanFilter:
    """
    Kalman filter optimized for PPG (photoplethysmogram) signals.
    PPG signals are lower frequency, more sinusoidal, and have different
    noise characteristics compared to ECG.
    """

    def __init__(self, dt=0.01, process_noise=0.001, measurement_noise=0.05):
        """
        Initialize Kalman filter for PPG signals.

        Parameters:
        -----------
        dt : float
            Sampling interval (default 0.01 for 100Hz sampling, typical for PPG)
        process_noise : float
            Process noise covariance (Q). PPG is smoother, so use lower values (0.0001-0.01)
        measurement_noise : float
            Measurement noise covariance (R). Represents sensor/motion artifact noise
        """
        # State vector: [position, velocity]
        self.x = np.array([[0.0], [0.0]])

        # State transition matrix (constant velocity model)
        self.F = np.array([[1, dt], [0, 1]])

        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0]])

        # Process noise covariance (lower for smoother PPG signals)
        self.Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * process_noise

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # Estimation error covariance
        self.P = np.eye(2)

    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Update step with measurement"""
        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def filter_signal(self, signal_data, missing_indices=None):
        """
        Filter entire signal and interpolate missing data.

        Parameters:
        -----------
        signal_data : array-like
            Input signal (use np.nan for missing values)
        missing_indices : array-like, optional
            Indices of missing data points

        Returns:
        --------
        filtered_signal : ndarray
            Filtered and interpolated signal
        """
        filtered = np.zeros(len(signal_data))

        for i, measurement in enumerate(signal_data):
            self.predict()

            # If data is missing or NaN, skip update step (prediction only)
            if missing_indices is not None and i in missing_indices:
                filtered[i] = self.x[0, 0]
            elif np.isnan(measurement):
                filtered[i] = self.x[0, 0]
            else:
                self.update(np.array([[measurement]]))
                filtered[i] = self.x[0, 0]

        return filtered


class AdaptivePPGKalmanFilter:
    """
    Adaptive Kalman filter for PPG signals with motion artifact detection.
    Adjusts parameters based on signal characteristics and detects motion artifacts.
    """

    def __init__(self, dt=0.01, initial_process_noise=0.001, initial_measurement_noise=0.05):
        self.dt = dt
        self.x = np.array([[0.0], [0.0]])
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.P = np.eye(2)

        # Adaptive noise parameters
        self.Q_scale = initial_process_noise
        self.R_base = initial_measurement_noise
        self.R = np.array([[initial_measurement_noise]])

        # Innovation history for adaptation and motion detection
        self.innovation_history = []
        self.window_size = 30  # Smaller window for faster adaptation

        # Motion artifact detection
        self.motion_threshold = 3.0  # Standard deviations

    def detect_motion_artifact(self, innovation):
        """Detect potential motion artifacts based on innovation magnitude"""
        if len(self.innovation_history) < 10:
            return False

        recent_std = np.std(self.innovation_history[-10:])
        if recent_std > 0:
            z_score = abs(innovation[0, 0]) / recent_std
            return z_score > self.motion_threshold
        return False

    def adapt_noise(self, innovation, is_motion_artifact):
        """Adapt noise parameters based on signal characteristics"""
        self.innovation_history.append(abs(innovation[0, 0]))

        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        # Adjust measurement noise if motion artifact detected
        if is_motion_artifact:
            self.R = np.array([[self.R_base * 10]])  # Increase measurement noise
        else:
            self.R = np.array([[self.R_base]])

        # Adjust process noise based on signal variability
        if len(self.innovation_history) >= self.window_size:
            innovation_std = np.std(self.innovation_history)
            # PPG needs lower process noise due to smoother signal
            self.Q_scale = max(0.0001, min(0.01, innovation_std * 0.05))

    def predict(self):
        Q = (
            np.array([[self.dt**4 / 4, self.dt**3 / 2], [self.dt**3 / 2, self.dt**2]])
            * self.Q_scale
        )
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, measurement):
        y = measurement - self.H @ self.x

        # Detect motion artifact before updating
        is_motion = self.detect_motion_artifact(y)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        self.adapt_noise(y, is_motion)

        return is_motion

    def filter_signal(self, signal_data, missing_indices=None):
        filtered = np.zeros(len(signal_data))
        motion_flags = np.zeros(len(signal_data), dtype=bool)

        for i, measurement in enumerate(signal_data):
            self.predict()

            if missing_indices is not None and i in missing_indices:
                filtered[i] = self.x[0, 0]
            elif np.isnan(measurement):
                filtered[i] = self.x[0, 0]
            else:
                is_motion = self.update(np.array([[measurement]]))
                filtered[i] = self.x[0, 0]
                motion_flags[i] = is_motion

        return filtered, motion_flags


class ExtendedPPGKalmanFilter:
    """
    Extended Kalman Filter for PPG with sinusoidal model.
    Models the PPG waveform as a sinusoid with varying amplitude and baseline.
    Better for capturing the periodic nature of PPG signals.
    """

    def __init__(self, dt=0.01, hr_estimate=75, process_noise=0.001, measurement_noise=0.05):
        """
        Parameters:
        -----------
        dt : float
            Sampling interval (default 0.01 for 100Hz sampling, typical for PPG)
        hr_estimate : float
            Initial heart rate estimate in BPM
        process_noise : float
            Process noise covariance (Q). PPG is smoother, so use lower values (0.0001-0.01)
        measurement_noise : float
            Measurement noise covariance (R). Represents sensor/motion artifact noise
        """
        # State: [DC offset, amplitude, phase, frequency]
        self.x = np.array([[0.0], [1.0], [0.0], [2 * np.pi * hr_estimate / 60]])
        self.dt = dt

        self.H = np.array([[1, 0, 0, 0]])  # We measure the overall signal

        self.Q = np.eye(4) * process_noise
        self.R = np.array([[measurement_noise]])
        self.P = np.eye(4) * 0.1

        # For heart rate extraction
        self.hr_history = []

    def get_heart_rate(self):
        """Extract current heart rate estimate from state"""
        frequency = self.x[3, 0]  # radians/second
        hr = frequency * 60 / (2 * np.pi)  # Convert to BPM
        return hr

    def state_transition(self, x):
        """Nonlinear state transition for sinusoidal model"""
        dc, amp, phase, freq = x.flatten()

        # Update phase based on frequency
        new_phase = (phase + freq * self.dt) % (2 * np.pi)

        return np.array([[dc], [amp], [new_phase], [freq]])

    def measurement_function(self, x):
        """Measurement model: DC + amplitude * sin(phase)"""
        dc, amp, phase, freq = x.flatten()
        return np.array([[dc + amp * np.sin(phase)]])

    def predict(self):
        """EKF prediction step"""
        # Propagate state
        self.x = self.state_transition(self.x)

        # Jacobian of state transition
        F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """EKF update step"""
        # Predicted measurement
        z_pred = self.measurement_function(self.x)

        # Innovation
        y = measurement - z_pred

        # Jacobian of measurement function
        dc, amp, phase, freq = self.x.flatten()
        H = np.array([[1, np.sin(phase), amp * np.cos(phase), 0]])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T / S

        # Update state
        self.x = self.x + K * y

        # Ensure phase stays in [0, 2π]
        self.x[2, 0] = self.x[2, 0] % (2 * np.pi)

        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def filter_signal(self, signal_data, missing_indices=None):
        filtered = np.zeros(len(signal_data))
        heart_rates = np.zeros(len(signal_data))

        for i, measurement in enumerate(signal_data):
            self.predict()

            if missing_indices is not None and i in missing_indices:
                filtered[i] = self.measurement_function(self.x)[0, 0]
            elif np.isnan(measurement):
                filtered[i] = self.measurement_function(self.x)[0, 0]
            else:
                self.update(np.array([[measurement]]))
                filtered[i] = self.measurement_function(self.x)[0, 0]

            # Track heart rate
            hr = self.get_heart_rate()
            heart_rates[i] = hr
            self.hr_history.append(hr)

        return filtered, heart_rates


class HarmonicPPGKalmanFilter:
    """
    Extended Kalman Filter for PPG with harmonic sinusoidal model.
    Models the PPG waveform as a fundamental sinusoid plus its first two harmonics.
    This provides a more sophisticated representation of the PPG signal, capturing
    the dicrotic notch and other morphological features.

    State vector: [DC offset, A1, A2, A3, phase, frequency]
    where:
        DC offset: baseline
        A1: amplitude of fundamental frequency
        A2: amplitude of second harmonic (2f)
        A3: amplitude of third harmonic (3f)
        phase: phase of fundamental
        frequency: angular frequency (rad/s)

    Measurement model: y = DC + A1*sin(phase) + A2*sin(2*phase) + A3*sin(3*phase)
    """

    def __init__(self, dt=0.01, hr_estimate=75, process_noise=0.001, measurement_noise=0.05):
        """
        Parameters:
        -----------
        dt : float
            Sampling interval (default 0.01 for 100Hz sampling)
        hr_estimate : float
            Initial heart rate estimate in BPM
        process_noise : float
            Process noise covariance (Q). Controls how much the state can change
        measurement_noise : float
            Measurement noise covariance (R). Represents sensor noise
        """
        # State: [DC offset, A1, A2, A3, phase, frequency]
        # Initialize with reasonable defaults for PPG signals
        self.x = np.array(
            [
                [0.0],  # DC offset (will be adjusted from data)
                [1.0],  # A1 - fundamental amplitude
                [0.2],  # A2 - second harmonic amplitude (typically ~20% of fundamental)
                [0.1],  # A3 - third harmonic amplitude (typically ~10% of fundamental)
                [0.0],  # phase
                [2 * np.pi * hr_estimate / 60],  # frequency in rad/s
            ]
        )
        self.dt = dt

        # Measurement matrix - we measure the overall signal
        self.H = np.array([[1, 0, 0, 0, 0, 0]])

        # Process noise - allow more flexibility in amplitudes and phase
        Q_diag = np.array(
            [
                process_noise * 0.1,  # DC changes slowly
                process_noise,  # A1 fundamental amplitude
                process_noise * 0.5,  # A2 second harmonic
                process_noise * 0.5,  # A3 third harmonic
                process_noise * 2.0,  # phase (changes fastest)
                process_noise * 0.1,  # frequency (changes slowly)
            ]
        )
        self.Q = np.diag(Q_diag)

        self.R = np.array([[measurement_noise]])
        self.P = np.eye(6) * 0.1

        # For heart rate extraction
        self.hr_history = []

    def get_heart_rate(self):
        """Extract current heart rate estimate from state"""
        frequency = self.x[5, 0]  # radians/second
        hr = frequency * 60 / (2 * np.pi)  # Convert to BPM
        return hr

    def state_transition(self, x):
        """Nonlinear state transition for harmonic sinusoidal model"""
        dc, a1, a2, a3, phase, freq = x.flatten()

        # Update phase based on frequency
        new_phase = (phase + freq * self.dt) % (2 * np.pi)

        # Other states remain constant in the model
        return np.array([[dc], [a1], [a2], [a3], [new_phase], [freq]])

    def measurement_function(self, x):
        """
        Measurement model: DC + A1*sin(phase) + A2*sin(2*phase) + A3*sin(3*phase)
        This models the fundamental frequency and first two harmonics.
        """
        dc, a1, a2, a3, phase, freq = x.flatten()
        y = dc + a1 * np.sin(phase) + a2 * np.sin(2 * phase) + a3 * np.sin(3 * phase)
        return np.array([[y]])

    def predict(self):
        """EKF prediction step"""
        # Propagate state
        self.x = self.state_transition(self.x)

        # Jacobian of state transition
        # dx_new/dx_old for each state variable
        F = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # dc
                [0, 1, 0, 0, 0, 0],  # a1
                [0, 0, 1, 0, 0, 0],  # a2
                [0, 0, 0, 1, 0, 0],  # a3
                [0, 0, 0, 0, 1, self.dt],  # phase depends on frequency
                [0, 0, 0, 0, 0, 1],  # freq
            ]
        )

        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """EKF update step"""
        # Predicted measurement
        z_pred = self.measurement_function(self.x)

        # Innovation
        y = measurement - z_pred

        # Jacobian of measurement function h(x) = dc + a1*sin(φ) + a2*sin(2φ) + a3*sin(3φ)
        dc, a1, a2, a3, phase, freq = self.x.flatten()

        # ∂h/∂dc = 1
        # ∂h/∂a1 = sin(φ)
        # ∂h/∂a2 = sin(2φ)
        # ∂h/∂a3 = sin(3φ)
        # ∂h/∂φ = a1*cos(φ) + 2*a2*cos(2φ) + 3*a3*cos(3φ)
        # ∂h/∂freq = 0
        H = np.array(
            [
                [
                    1,  # ∂h/∂dc
                    np.sin(phase),  # ∂h/∂a1
                    np.sin(2 * phase),  # ∂h/∂a2
                    np.sin(3 * phase),  # ∂h/∂a3
                    a1 * np.cos(phase)
                    + 2 * a2 * np.cos(2 * phase)
                    + 3 * a3 * np.cos(3 * phase),  # ∂h/∂φ
                    0,  # ∂h/∂freq
                ]
            ]
        )

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T / S

        # Update state
        self.x = self.x + K * y

        # Ensure phase stays in [0, 2π]
        self.x[4, 0] = self.x[4, 0] % (2 * np.pi)

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def filter_signal(self, signal_data, missing_indices=None):
        """
        Filter entire signal and track heart rate.

        Parameters:
        -----------
        signal_data : array-like
            Input signal (use np.nan for missing values)
        missing_indices : array-like, optional
            Indices of missing data points

        Returns:
        --------
        filtered : ndarray
            Filtered and interpolated signal
        heart_rates : ndarray
            Heart rate estimate at each time point
        harmonic_amplitudes : ndarray
            Array of shape (n_samples, 3) containing [A1, A2, A3] at each time
        """
        filtered = np.zeros(len(signal_data))
        heart_rates = np.zeros(len(signal_data))
        harmonic_amplitudes = np.zeros((len(signal_data), 3))

        for i, measurement in enumerate(signal_data):
            self.predict()

            if missing_indices is not None and i in missing_indices:
                filtered[i] = self.measurement_function(self.x)[0, 0]
            elif np.isnan(measurement):
                filtered[i] = self.measurement_function(self.x)[0, 0]
            else:
                self.update(np.array([[measurement]]))
                filtered[i] = self.measurement_function(self.x)[0, 0]

            # Track heart rate
            hr = self.get_heart_rate()
            heart_rates[i] = hr
            self.hr_history.append(hr)

            # Track harmonic amplitudes
            harmonic_amplitudes[i] = [self.x[1, 0], self.x[2, 0], self.x[3, 0]]

        return filtered, heart_rates, harmonic_amplitudes


class SignalQualityAssessor:
    """
    Assesses PPG signal quality based on multiple metrics.
    Provides a quality score from 0 (poor) to 1 (excellent).
    """

    def __init__(self, fs=100.0, window_size=5.0):
        """
        Parameters:
        -----------
        fs : float
            Sampling frequency
        window_size : float
            Window size in seconds for quality assessment
        """
        self.fs = fs
        self.window_samples = int(window_size * fs)

    def assess_quality(self, signal_segment, filtered_segment=None):
        """
        Assess signal quality for a segment.

        Returns:
        --------
        quality_score : float
            Overall quality score (0-1)
        metrics : dict
            Individual quality metrics
        """
        metrics = {}

        # 1. SNR estimate (signal-to-noise ratio)
        if filtered_segment is not None:
            noise = signal_segment - filtered_segment
            signal_power = np.var(filtered_segment)
            noise_power = np.var(noise)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            metrics["snr"] = max(0, min(snr / 20, 1))  # Normalize to 0-1
        else:
            metrics["snr"] = 0.5  # Unknown

        # 2. Perfusion (relative pulse amplitude)
        dc_component = np.mean(signal_segment)
        ac_component = np.std(signal_segment)
        perfusion = ac_component / (dc_component + 1e-10)
        metrics["perfusion"] = min(perfusion / 0.1, 1)  # Normalize, typical range 0.01-0.1

        # 3. Spectral purity (concentration of power in physiological range)
        freqs, psd = signal.welch(
            signal_segment, fs=self.fs, nperseg=min(256, len(signal_segment))
        )

        # Physiological range for heart rate: 0.5-3.0 Hz (30-180 BPM)
        hr_band = (freqs >= 0.5) & (freqs <= 3.0)
        total_power = np.sum(psd)
        hr_power = np.sum(psd[hr_band])
        metrics["spectral_purity"] = hr_power / (total_power + 1e-10)

        # 4. Kurtosis (measure of outliers/artifacts)
        from scipy.stats import kurtosis

        kurt = abs(kurtosis(signal_segment))
        metrics["kurtosis"] = max(0, 1 - kurt / 10)  # Lower kurtosis is better

        # 5. Zero crossing rate (should be regular for good PPG)
        zero_crossings = np.sum(np.diff(np.sign(signal_segment - np.mean(signal_segment))) != 0)
        expected_crossings = len(signal_segment) / self.fs * 2 * 1.5  # ~2 per beat at 75 BPM
        zcr_score = 1 - min(abs(zero_crossings - expected_crossings) / expected_crossings, 1)
        metrics["zero_crossing"] = zcr_score

        # Overall quality score (weighted average)
        weights = {
            "snr": 0.3,
            "perfusion": 0.2,
            "spectral_purity": 0.3,
            "kurtosis": 0.1,
            "zero_crossing": 0.1,
        }

        quality_score = sum(metrics[key] * weights[key] for key in weights.keys())

        return quality_score, metrics

    def assess_continuous(self, signal, filtered=None, stride=1.0):
        """
        Assess quality continuously along the signal.

        Parameters:
        -----------
        signal : array
            Input signal
        filtered : array, optional
            Filtered version of signal for SNR calculation
        stride : float
            Stride between windows in seconds

        Returns:
        --------
        times : array
            Time points for quality scores
        quality_scores : array
            Quality scores at each time point
        """
        stride_samples = int(stride * self.fs)
        n_windows = (len(signal) - self.window_samples) // stride_samples + 1

        times = np.zeros(n_windows)
        quality_scores = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * stride_samples
            end = start + self.window_samples

            if end > len(signal):
                break

            signal_segment = signal[start:end]
            filtered_segment = filtered[start:end] if filtered is not None else None

            quality_score, _ = self.assess_quality(signal_segment, filtered_segment)

            times[i] = (start + end) / 2 / self.fs
            quality_scores[i] = quality_score

        return times[: i + 1], quality_scores[: i + 1]


class HeartRateExtractor:
    """
    Extracts heart rate from PPG signals using multiple methods.
    """

    def __init__(self, fs=100.0):
        """
        Parameters:
        -----------
        fs : float
            Sampling frequency
        """
        self.fs = fs

    def extract_from_peaks(self, ppg_signal, min_distance=0.4):
        """
        Extract heart rate from peak detection.

        Parameters:
        -----------
        signal : array
            PPG signal
        min_distance : float
            Minimum time between peaks in seconds

        Returns:
        --------
        hr : float
            Heart rate in BPM
        peak_indices : array
            Indices of detected peaks
        """
        # Find peaks
        min_samples = int(min_distance * self.fs)
        peaks, properties = signal.find_peaks(ppg_signal, distance=min_samples, prominence=0.1)

        if len(peaks) < 2:
            return None, peaks

        # Calculate inter-beat intervals
        ibi = np.diff(peaks) / self.fs  # In seconds

        # Remove outliers (use median for robustness)
        median_ibi = np.median(ibi)
        valid_ibi = ibi[(ibi > median_ibi * 0.7) & (ibi < median_ibi * 1.3)]

        if len(valid_ibi) == 0:
            return None, peaks

        # make an RRI waveform
        rri = np.zeros(len(ppg_signal))
        for peakidx in range(len(peaks)-1):
            if (median_ibi * 0.7) <= ibi[peakidx] <= (median_ibi * 1.3):
                rri[peaks[peakidx]:peaks[peakidx+1]] = ibi[peakidx]
            else:
                rri[peaks[peakidx] : peaks[peakidx + 1]] = 0.0
        rri[0:peaks[0]] = rri[peaks[0]]
        rri[peaks[-1]:] = rri[peaks[-1]]

        # deal with the zeros
        badranges = []
        inarun = False
        first = -1
        for i in range(len(rri)):
            if rri[i] == 0.0:
                if not inarun:
                    first = i
                    inarun = True
            else:
                if inarun:
                    badranges.append((first, i - 1))
                    inarun = False
        if inarun:
            if first > 0:
                badranges.append((first, len(rri) - 1))
            else:
                rri = None
        print(f"badranges = {badranges}")

        if badranges is not None:
            for (first, last) in badranges:
                if first == 0:
                    rri[first : last + 1] = rri[last + 1]
                elif last == (len(rri) - 1):
                    rri[first : last+1] = rri[first - 1]
                else:
                    rri[first : last + 1] = (rri[first-1] + rri[last + 1]) / 2.0

        # Convert to heart rate
        hr = 60.0 / np.mean(valid_ibi)
        if rri is not None:
            hr_waveform = 60.0 / rri
        else:
            hr_waveform = None

        return hr, peaks, rri, hr_waveform

    def extract_from_fft(self, ppg_signal, hr_range=(40, 180)):
        """
        Extract heart rate using FFT (frequency domain).

        Parameters:
        -----------
        ppg_signal : array
            PPG signal
        hr_range : tuple
            Expected heart rate range in BPM

        Returns:
        --------
        hr : float
            Heart rate in BPM
        frequency : float
            Dominant frequency in Hz
        psd : array
            Power spectral density
        freqs : array
            Frequency array
        """
        # Compute power spectral density
        freqs, psd = signal.welch(ppg_signal, fs=self.fs, nperseg=min(256, len(ppg_signal)))

        # Convert HR range to frequency range
        freq_range = (hr_range[0] / 60, hr_range[1] / 60)

        # Find peak in physiological range
        valid_indices = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        valid_freqs = freqs[valid_indices]
        valid_psd = psd[valid_indices]

        if len(valid_psd) == 0:
            return None, None, psd, freqs

        peak_idx = np.argmax(valid_psd)
        dominant_freq = valid_freqs[peak_idx]
        hr = dominant_freq * 60

        return hr, dominant_freq, psd, freqs

    def extract_continuous(self, ppg_signal, window_size=10.0, stride=2.0, method="fft"):
        """
        Extract heart rate continuously along the PPG signal.

        Parameters:
        -----------
        ppg_signal : array
            PPG signal
        window_size : float
            Window size in seconds
        stride : float
            Stride between windows in seconds
        method : str
            'fft' or 'peaks'

        Returns:
        --------
        times : array
            Time points for HR estimates
        heart_rates : array
            Heart rates in BPM at each time point
        """
        window_samples = int(window_size * self.fs)
        stride_samples = int(stride * self.fs)
        n_windows = (len(ppg_signal) - window_samples) // stride_samples + 1

        times = []
        heart_rates = []

        for i in range(n_windows):
            start = i * stride_samples
            end = start + window_samples

            if end > len(ppg_signal):
                break

            segment = ppg_signal[start:end]

            if method == "fft":
                hr, _, _, _ = self.extract_from_fft(segment)
            else:  # peaks
                hr, _ = self.extract_from_peaks(segment)

            if hr is not None:
                times.append((start + end) / 2 / self.fs)
                heart_rates.append(hr)

        return np.array(times), np.array(heart_rates)


class RobustPPGProcessor:
    """
    Complete PPG processing pipeline combining filtering, quality assessment,
    and heart rate extraction with intelligent segment handling.
    """

    def __init__(self, fs=100.0, method="adaptive", hr_estimate=75.0, process_noise=0.0001):
        """
        Parameters:
        -----------
        fs : float
            Sampling frequency
        method : str
            Filter method: 'standard', 'adaptive', or 'ekf'
        """
        self.fs = fs
        self.dt = 1.0 / fs
        self.method = method
        self.process_noise = process_noise
        self.hr_estimate = hr_estimate

        # Initialize components
        if method == "standard":
            self.filter = PPGKalmanFilter(
                dt=self.dt, process_noise=self.process_noise, measurement_noise=0.05
            )
        elif method == "adaptive":
            self.filter = AdaptivePPGKalmanFilter(
                dt=self.dt,
                initial_process_noise=self.process_noise,
                initial_measurement_noise=0.05,
            )
        else:  # ekf
            self.filter = ExtendedPPGKalmanFilter(
                dt=self.dt,
                hr_estimate=self.hr_estimate,
                process_noise=self.process_noise,
                measurement_noise=0.05,
            )

        self.quality_assessor = SignalQualityAssessor(fs=fs, window_size=5.0)
        self.hr_extractor = HeartRateExtractor(fs=fs)

    def process(self, signal_data, missing_indices=None, quality_threshold=0.5):
        """
        Complete processing pipeline.

        Parameters:
        -----------
        signal_data : array
            Raw PPG signal
        missing_indices : array, optional
            Indices of missing data
        quality_threshold : float
            Minimum quality score (0-1) for accepting segments

        Returns:
        --------
        results : dict
            Dictionary containing all processing results
        """
        results = {}

        # Step 1: Filter signal
        if self.method == "adaptive":
            filtered, motion_flags = self.filter.filter_signal(signal_data, missing_indices)
            results["motion_flags"] = motion_flags
        elif self.method == "ekf":
            filtered, hr_continuous = self.filter.filter_signal(signal_data, missing_indices)
            results["ekf_heart_rate"] = hr_continuous
        elif self.method == "raw":
            filtered = signal_data
        else:
            filtered = self.filter.filter_signal(signal_data, missing_indices)

        results["filtered_signal"] = filtered

        # Step 2: Assess quality
        qual_times, qual_scores = self.quality_assessor.assess_continuous(
            signal_data, filtered, stride=1.0
        )
        results["quality_times"] = qual_times
        results["quality_scores"] = qual_scores

        # Step 3: Extract heart rate (only from good quality segments)
        good_quality_mask = qual_scores > quality_threshold

        if np.any(good_quality_mask):
            hr_times, hr_values = self.hr_extractor.extract_continuous(
                filtered, window_size=10.0, stride=2.0, method="fft"
            )
            results["hr_times"] = hr_times
            results["hr_values"] = hr_values

            # Overall heart rate from peaks
            hr_from_peaks, peak_indices, rri, hr_waveform_from_peaks = self.hr_extractor.extract_from_peaks(filtered)
            results["hr_overall"] = hr_from_peaks
            results["peak_indices"] = peak_indices
            results["rri"] = rri
            results["hr_waveform"] = hr_waveform_from_peaks
        else:
            results["hr_times"] = np.array([])
            results["hr_values"] = np.array([])
            results["hr_overall"] = None
            results["peak_indices"] = np.array([])

        # Step 4: Compute statistics
        results["mean_quality"] = np.mean(qual_scores)
        results["good_quality_percentage"] = (
            np.sum(good_quality_mask) / len(good_quality_mask) * 100
        )

        return results


class PPGFeatureExtractor:
    """
    Extract additional features from PPG signals useful for health monitoring.
    """

    def __init__(self, fs=100.0):
        self.fs = fs

    def extract_hrv_features(self, peak_indices):
        """
        Extract Heart Rate Variability (HRV) features.

        Parameters:
        -----------
        peak_indices : array
            Indices of detected peaks

        Returns:
        --------
        hrv_features : dict
            Dictionary of HRV metrics
        """
        if len(peak_indices) < 3:
            return None

        # Inter-beat intervals in milliseconds
        ibi = np.diff(peak_indices) / self.fs * 1000

        # Remove outliers
        median_ibi = np.median(ibi)
        valid_ibi = ibi[(ibi > median_ibi * 0.7) & (ibi < median_ibi * 1.3)]

        if len(valid_ibi) < 2:
            return None

        features = {}

        # Time domain features
        features["mean_ibi"] = np.mean(valid_ibi)
        features["sdnn"] = np.std(valid_ibi)  # Standard deviation of NN intervals
        features["rmssd"] = np.sqrt(
            np.mean(np.diff(valid_ibi) ** 2)
        )  # Root mean square of successive differences
        features["pnn50"] = (
            np.sum(np.abs(np.diff(valid_ibi)) > 50) / len(valid_ibi) * 100
        )  # % of intervals > 50ms different

        # Frequency domain features (requires longer recordings)
        if len(valid_ibi) > 30:
            # Resample to uniform time series
            time_points = np.cumsum(np.concatenate([[0], valid_ibi])) / 1000  # seconds
            f_interp = interp1d(
                time_points[:-1], valid_ibi, kind="cubic", fill_value="extrapolate"
            )

            # Create uniform time base at 4 Hz
            uniform_time = np.arange(0, time_points[-1], 0.25)
            uniform_ibi = f_interp(uniform_time)

            # Compute PSD
            freqs, psd = signal.welch(uniform_ibi, fs=4, nperseg=min(256, len(uniform_ibi)))

            # HRV frequency bands
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)  # Very low frequency
            lf_band = (freqs >= 0.04) & (freqs < 0.15)  # Low frequency
            hf_band = (freqs >= 0.15) & (freqs < 0.4)  # High frequency

            features["vlf_power"] = np.trapz(psd[vlf_band], freqs[vlf_band])
            features["lf_power"] = np.trapz(psd[lf_band], freqs[lf_band])
            features["hf_power"] = np.trapz(psd[hf_band], freqs[hf_band])
            features["lf_hf_ratio"] = features["lf_power"] / (features["hf_power"] + 1e-10)

        return features

    def extract_morphology_features(self, signal_segment, peak_idx):
        """
        Extract morphological features from a single PPG pulse.

        Parameters:
        -----------
        signal_segment : array
            PPG signal segment containing one pulse
        peak_idx : int
            Index of the systolic peak within the segment

        Returns:
        --------
        features : dict
            Morphological features
        """
        features = {}

        # Pulse amplitude
        baseline = np.min(signal_segment)
        features["pulse_amplitude"] = signal_segment[peak_idx] - baseline

        # Rising time (foot to peak)
        features["rising_time"] = peak_idx / self.fs

        # Find dicrotic notch (local minimum after peak)
        if peak_idx < len(signal_segment) - 10:
            search_window = signal_segment[
                peak_idx : min(peak_idx + int(0.3 * self.fs), len(signal_segment))
            ]
            if len(search_window) > 0:
                notch_idx = peak_idx + np.argmin(search_window)
                features["dicrotic_notch_amplitude"] = signal_segment[notch_idx] - baseline
                features["augmentation_index"] = (signal_segment[notch_idx] - baseline) / features[
                    "pulse_amplitude"
                ]

        # Pulse width at half maximum
        half_max = baseline + features["pulse_amplitude"] / 2
        above_half = signal_segment > half_max
        if np.any(above_half):
            transitions = np.diff(above_half.astype(int))
            rise_points = np.where(transitions == 1)[0]
            fall_points = np.where(transitions == -1)[0]
            if len(rise_points) > 0 and len(fall_points) > 0:
                features["pulse_width"] = (fall_points[0] - rise_points[0]) / self.fs

        return features

    def compute_spo2_proxy(self, filtered_signal):
        """
        Compute a proxy for SpO2 (oxygen saturation) based on AC/DC ratio.
        Note: This is a simplified proxy and not a real SpO2 measurement.
        Real SpO2 requires red and infrared PPG signals.

        Parameters:
        -----------
        filtered_signal : array
            Filtered PPG signal

        Returns:
        --------
        spo2_proxy : float
            Proxy value (not actual SpO2)
        """
        # AC component (pulsatile)
        ac = np.std(filtered_signal)

        # DC component (baseline)
        dc = np.mean(filtered_signal)

        # Compute ratio
        ratio = ac / (dc + 1e-10)

        # Empirical mapping (this is just a proxy!)
        # Real SpO2 uses calibration curves specific to the sensor
        spo2_proxy = 110 - 25 * ratio

        return np.clip(spo2_proxy, 70, 100)


def read_happy_ppg(filenameroot, debug=False):
    Fs, instarttime, incolumns, indata, incompressed, incolsource = tide_io.readbidstsv(
        f"{filenameroot}.json",
        neednotexist=True,
        debug=debug,
    )
    if debug:
        print(f"{indata.shape=}")

    t = np.linspace(0, (indata.shape[1] / Fs), num=indata.shape[1], endpoint=False)

    # set raw file
    try:
        rawindex = incolumns.index("cardiacfromfmri")
    except ValueError:
        try:
            rawindex = incolumns.index("cardiacfromfmri_25.0Hz")
        except ValueError:
            raise (ValueError("cardiacfromfmri column not found"))
    raw_ppg = indata[rawindex, :]

    # set badpts file
    try:
        badptsindex = incolumns.index("badpts")
        badpts = indata[badptsindex, :]
        print(badpts)
        missing_indices = np.where(badpts > 0)[0]
    except ValueError:
        missing_indices = []

    # use pleth, or dlfiltered if pleth is not available
    try:
        cleanindex = incolumns.index("pleth")
    except ValueError:
        try:
            cleanindex = incolumns.index("cardiacfromfmri_dlfiltered")
        except ValueError:
            try:
                cleanindex = incolumns.index("cardiacfromfmri_dlfiltered_25.0Hz")
            except ValueError:
                raise (ValueError("no clean ppg column found"))
    clean_ppg = indata[cleanindex, :]

    return t, Fs, clean_ppg, raw_ppg, missing_indices


def generate_synthetic_ppg(
    duration=10, fs=100.0, hr=75, noise_level=0.05, missing_percent=5, motion_artifacts=True
):
    """
    Generate synthetic PPG signal for testing.

    Parameters:
    -----------
    duration : float
        Duration in seconds
    fs : float
        Sampling frequency in Hz (typically 50-250 Hz for PPG)
    hr : int
        Heart rate in beats per minute
    noise_level : float
        Standard deviation of additive noise
    missing_percent : float
        Percentage of data points to randomly remove
    motion_artifacts : bool
        Whether to add motion artifacts
    """
    t = np.arange(0, duration, 1 / fs)

    # PPG signal: slower, more sinusoidal than ECG
    beat_period = 60 / hr
    ppg = np.zeros_like(t)

    # Main pulsatile component (smoother, more sinusoidal)
    fundamental_freq = hr / 60  # Hz
    ppg = 1.0 + 0.8 * np.sin(2 * np.pi * fundamental_freq * t)

    # Add harmonic for dicrotic notch (characteristic of PPG)
    ppg += 0.15 * np.sin(2 * np.pi * 2 * fundamental_freq * t + np.pi / 3)

    # Add slight baseline drift (common in PPG due to respiration)
    ppg += 0.2 * np.sin(2 * np.pi * 0.25 * t)  # ~15 breaths/min

    # Add Gaussian noise
    noisy_ppg = ppg + np.random.normal(0, noise_level, len(ppg))

    # Add motion artifacts if requested
    if motion_artifacts:
        n_artifacts = 3
        for _ in range(n_artifacts):
            artifact_start = np.random.randint(0, len(noisy_ppg) - int(fs))
            artifact_length = int(fs * np.random.uniform(0.5, 2))  # 0.5-2 second artifacts
            artifact_end = min(artifact_start + artifact_length, len(noisy_ppg))

            # Large amplitude motion artifact
            artifact = np.random.uniform(0.5, 2.0) * np.random.randn(artifact_end - artifact_start)
            noisy_ppg[artifact_start:artifact_end] += artifact

    # Randomly remove data points
    n_missing = int(len(noisy_ppg) * missing_percent / 100)
    missing_indices = np.random.choice(len(noisy_ppg), n_missing, replace=False)
    missing_indices = np.sort(missing_indices)

    corrupted_ppg = noisy_ppg.copy()
    corrupted_ppg[missing_indices] = np.nan

    return t, ppg, noisy_ppg, corrupted_ppg, missing_indices