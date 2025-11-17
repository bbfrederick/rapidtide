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
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
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

    def __init__(
        self, dt: float = 0.01, process_noise: float = 0.001, measurement_noise: float = 0.05
    ) -> None:
        """
        Initialize Kalman filter for PPG signals.

        Initialize Kalman filter with default parameters suitable for photoplethysmography (PPG)
        signal processing. Uses a constant velocity model with position and velocity states.

        Parameters
        ----------
        dt : float, optional
            Sampling interval in seconds, default is 0.01 (100Hz sampling rate typical for PPG)
        process_noise : float, optional
            Process noise covariance (Q) controlling state uncertainty, default is 0.001.
            Lower values (0.0001-0.01) appropriate for smoother PPG signals
        measurement_noise : float, optional
            Measurement noise covariance (R) representing sensor/motion artifact noise,
            default is 0.05

        Returns
        -------
        None
            Initializes internal filter parameters and state variables

        Notes
        -----
        The filter uses a constant velocity model with state vector [position, velocity].
        PPG signals are inherently smoother than other physiological signals, so lower
        process noise values are typically appropriate.

        Examples
        --------
        >>> filter = KalmanFilter(dt=0.02, process_noise=0.005, measurement_noise=0.1)
        >>> filter = KalmanFilter()  # Uses default parameters
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

    def predict(self) -> None:
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: NDArray) -> None:
        """
        Update step with measurement.

        Parameters
        ----------
        measurement : NDArray
            The measurement vector used to update the state estimate.

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return anything.

        Notes
        -----
        This function performs the update step of a Kalman filter. It computes the innovation,
        innovation covariance, Kalman gain, and then updates the state estimate and covariance
        matrix based on the measurement.

        Examples
        --------
        >>> kf = KalmanFilter()
        >>> measurement = np.array([1.0, 2.0])
        >>> kf.update(measurement)
        >>> print(kf.x)
        [0.95 1.98]
        """
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

    def filter_signal(self, signal_data: NDArray, missing_indices: list | None = None) -> NDArray:
        """
        Filter entire signal and interpolate missing data using a Kalman filter approach.

        This function applies a filtering process to signal data, handling missing values
        by either prediction-only steps or full update steps depending on the presence
        of missing data points. Missing values are represented as np.nan in the input
        signal_data.

        Parameters
        ----------
        signal_data : array-like
            Input signal data where missing values are represented as np.nan
        missing_indices : array-like, optional
            Indices of missing data points in the signal. If None, no indices are
            considered missing. Default is None.

        Returns
        -------
        filtered_signal : ndarray
            Filtered and interpolated signal with the same shape as input signal_data

        Notes
        -----
        The function uses a Kalman filter framework where:
        - For missing data points or NaN values, only prediction steps are performed
        - For valid measurements, both prediction and update steps are performed
        - The filtered result is stored in the state vector x[0, 0] after each step

        Examples
        --------
        >>> # Basic usage with missing data
        >>> signal = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        >>> missing_indices = [1, 4]
        >>> filtered = filter_signal(signal, missing_indices)

        >>> # Usage without missing data
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> filtered = filter_signal(signal)
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

    def __init__(
        self,
        dt: float = 0.01,
        initial_process_noise: float = 0.001,
        initial_measurement_noise: float = 0.05,
    ) -> None:
        """
        Initialize the Kalman filter with default parameters.

        Parameters
        ----------
        dt : float, optional
            Time step for the Kalman filter, by default 0.01
        initial_process_noise : float, optional
            Initial process noise covariance, by default 0.001
        initial_measurement_noise : float, optional
            Initial measurement noise covariance, by default 0.05

        Returns
        -------
        None
            This method initializes the instance variables and does not return anything.

        Notes
        -----
        This constructor sets up a 2D Kalman filter for position and velocity estimation.
        The filter uses a constant velocity model with adaptive noise scaling.

        Examples
        --------
        >>> filter = KalmanFilter()
        >>> filter = KalmanFilter(dt=0.02, initial_process_noise=0.005)
        """
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

    def detect_motion_artifact(self, innovation: NDArray) -> bool:
        """
        Detect potential motion artifacts based on innovation magnitude.

        This function analyzes the innovation signal to identify potential motion artifacts
        by computing a z-score against the recent history of innovation values.

        Parameters
        ----------
        innovation : NDArray
            The innovation signal to be analyzed, typically representing the difference
            between predicted and actual measurements. Expected to be a 2D array with
            shape (1, 1) for single sample analysis.

        Returns
        -------
        bool
            True if a motion artifact is detected (z-score exceeds threshold),
            False otherwise.

        Notes
        -----
        The function requires at least 10 samples in the innovation history to make
        a detection decision. If the recent standard deviation is zero, the function
        returns False to avoid division by zero errors.

        Examples
        --------
        >>> detector = MotionDetector()
        >>> detector.innovation_history = [0.1, 0.2, 0.15, 0.18, 0.22, 0.19, 0.21, 0.17, 0.23, 0.20]
        >>> detector.motion_threshold = 3.0
        >>> result = detector.detect_motion_artifact(np.array([[5.0]]))
        >>> print(result)
        True
        """
        if len(self.innovation_history) < 10:
            return False

        recent_std = np.std(self.innovation_history[-10:])
        if recent_std > 0:
            z_score = abs(innovation[0, 0]) / recent_std
            return z_score > self.motion_threshold
        return False

    def adapt_noise(self, innovation: NDArray, is_motion_artifact: bool) -> None:
        """
        Adapt noise parameters based on signal characteristics.

        This method adjusts the measurement and process noise covariance matrices
        based on the current innovation signal and motion artifact detection.

        Parameters
        ----------
        innovation : NDArray
            The innovation signal, typically the difference between predicted and
            actual measurements. Expected to be a 2D array with shape (1, 1) for
            single-dimensional measurements.
        is_motion_artifact : bool
            Flag indicating whether a motion artifact has been detected in the signal.
            When True, measurement noise is increased to handle the artifact.

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return
            any value.

        Notes
        -----
        The method maintains a sliding window history of innovation values to compute
        statistical measures for process noise adaptation. When motion artifacts are
        detected, the measurement noise covariance matrix R is increased by a factor
        of 10 to reduce the filter's trust in current measurements.

        Examples
        --------
        >>> # Assuming self is a Kalman filter instance
        >>> innovation = np.array([[0.5]])
        >>> adapt_noise(innovation, is_motion_artifact=True)
        >>> # Measurement noise R is increased, process noise Q_scale is adjusted
        """
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

    def predict(self) -> None:
        Q = (
            np.array([[self.dt**4 / 4, self.dt**3 / 2], [self.dt**3 / 2, self.dt**2]])
            * self.Q_scale
        )
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, measurement: NDArray) -> bool:
        """
        Update the state estimate using the Kalman filter update step.

        This method performs the measurement update step of the Kalman filter, incorporating
        new measurements into the state estimate while accounting for measurement noise
        and potential motion artifacts.

        Parameters
        ----------
        measurement : NDArray
            The new measurement vector used to update the state estimate.

        Returns
        -------
        bool
            True if motion artifact was detected, False otherwise.

        Notes
        -----
        The update process includes:
        1. Computing the innovation (measurement residual)
        2. Detecting motion artifacts using the detect_motion_artifact method
        3. Computing the Kalman gain
        4. Updating the state estimate and covariance matrix
        5. Adapting noise parameters based on the measurement residual and motion detection

        Examples
        --------
        >>> kf = KalmanFilter()
        >>> measurement = np.array([1.0, 2.0])
        >>> motion_detected = kf.update(measurement)
        >>> print(f"Motion artifact detected: {motion_detected}")
        """
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

    def filter_signal(self, signal_data: NDArray, missing_indices: list | None = None) -> NDArray:
        """
        Apply filtering to signal data using a Kalman filter approach.

        This function processes signal measurements using a Kalman filter framework,
        handling missing data and NaN values appropriately while tracking motion detection
        flags for each measurement.

        Parameters
        ----------
        signal_data : NDArray
            Array containing the signal measurements to be filtered.
        missing_indices : list of int, optional
            List of indices where measurements are missing. If None, no special
            handling is performed for missing data. Default is None.

        Returns
        -------
        tuple of (NDArray, NDArray)
            A tuple containing:
            - filtered : NDArray
              Array of filtered signal values
            - motion_flags : NDArray of bool
              Boolean array indicating motion detection for each measurement

        Notes
        -----
        The function uses a prediction-update cycle for each measurement:
        1. Prediction step is always performed
        2. For missing measurements or NaN values, the current state estimate is returned
        3. For valid measurements, the update step is performed and motion is detected

        Examples
        --------
        >>> # Basic usage
        >>> filtered_data, motion_flags = filter_signal(signal, missing_indices=[2, 5])
        >>>
        >>> # With no missing indices
        >>> filtered_data, motion_flags = filter_signal(signal)
        """
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

    def __init__(
        self,
        dt: float = 0.01,
        hr_estimate: float = 75,
        process_noise: float = 0.001,
        measurement_noise: float = 0.05,
    ) -> None:
        """
        Initialize the Kalman filter for heart rate estimation from PPG signals.

        This constructor initializes the state vector and covariance matrices for a
        Kalman filter designed to estimate heart rate from photoplethysmography (PPG)
        signals. The filter models the PPG signal as a sinusoidal waveform with
        time-varying parameters.

        Parameters
        ----------
        dt : float, optional
            Sampling interval in seconds (default 0.01 for 100Hz sampling, typical for PPG)
        hr_estimate : float, optional
            Initial heart rate estimate in beats per minute (BPM) (default 75)
        process_noise : float, optional
            Process noise covariance (Q). PPG is smoother, so use lower values (0.0001-0.01)
            (default 0.001)
        measurement_noise : float, optional
            Measurement noise covariance (R). Represents sensor/motion artifact noise
            (default 0.05)

        Returns
        -------
        None
            This method initializes the object's attributes in-place and does not return a value.

        Notes
        -----
        The state vector x contains [DC offset, amplitude, phase, frequency] representing
        the sinusoidal model of the PPG signal. The filter uses a constant velocity model
        for the frequency component to track heart rate variations.

        Examples
        --------
        >>> filter = KalmanFilter(dt=0.02, hr_estimate=80, process_noise=0.005)
        >>> print(filter.x)
        [[0.]
         [1.]
         [0.]
         [2.0943951023931953]]
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

    def get_heart_rate(self) -> float:
        """
        Extract current heart rate estimate from state.

        This function converts the angular frequency stored in the state vector
        to beats per minute (BPM), which represents the heart rate.

        Parameters
        ----------
        self : object
            The object instance containing the state vector. The state vector
            is expected to have at least 4 elements, with the third element
            (index 3) containing the angular frequency in radians/second.

        Returns
        -------
        float
            The estimated heart rate in beats per minute (BPM).

        Notes
        -----
        The conversion from angular frequency (radians/second) to heart rate (BPM)
        is calculated as: HR = ω × 60 / (2π), where ω is the angular frequency.

        Examples
        --------
        >>> heart_rate = obj.get_heart_rate()
        >>> print(f"Heart rate: {heart_rate:.1f} BPM")
        Heart rate: 72.0 BPM
        """
        frequency = self.x[3, 0]  # radians/second
        hr = frequency * 60 / (2 * np.pi)  # Convert to BPM
        return hr

    def state_transition(self, x: NDArray) -> NDArray:
        """
        Nonlinear state transition for sinusoidal model.

        This function performs the state transition for a sinusoidal model, updating
        the phase based on the frequency and time step, while keeping other state
        variables unchanged.

        Parameters
        ----------
        x : ndarray
            Input state vector of shape (4,) containing [dc_offset, amplitude, phase, frequency]

        Returns
        -------
        ndarray
            Transited state vector of shape (4, 1) containing [dc_offset, amplitude, new_phase, frequency]
            where new_phase = (phase + frequency * dt) % (2 * pi)

        Notes
        -----
        The phase is updated using the formula: new_phase = (phase + freq * dt) % (2 * pi)
        This ensures the phase remains within the range [0, 2π).

        Examples
        --------
        >>> import numpy as np
        >>> # Example usage
        >>> x = np.array([[1.0], [2.0], [0.5], [0.1]])
        >>> # Assuming self.dt = 0.01
        >>> result = state_transition(x)
        >>> print(result)
        [[1.0]
         [2.0]
         [0.501]
         [0.1]]
        """
        dc, amp, phase, freq = x.flatten()

        # Update phase based on frequency
        new_phase = (phase + freq * self.dt) % (2 * np.pi)

        return np.array([[dc], [amp], [new_phase], [freq]])

    def measurement_function(self, x: NDArray) -> NDArray:
        """
        Measurement model: DC + amplitude * sin(phase)

        This function implements a measurement model that combines a DC component with a sinusoidal
        signal. The model is defined as: y = dc + amp * sin(phase), where dc is the DC offset,
        amp is the amplitude, and phase is the phase angle.

        Parameters
        ----------
        x : ndarray
            Input array of shape (4,) containing the model parameters in order:
            [dc, amp, phase, freq]
            - dc : float
                DC offset component
            - amp : float
                Amplitude of the sinusoidal signal
            - phase : float
                Phase angle of the sinusoidal signal (in radians)
            - freq : float
                Frequency of the sinusoidal signal (not used in current implementation)

        Returns
        -------
        ndarray
            Measurement output array of shape (1, 1) containing the computed measurement:
            [[dc + amp * sin(phase)]]

        Notes
        -----
        The frequency parameter is included in the input array but not used in the current
        implementation of the measurement model.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, np.pi/4, 1.0])
        >>> result = measurement_function(None, x)
        >>> print(result)
        [[2.41421356]]
        """
        dc, amp, phase, freq = x.flatten()
        return np.array([[dc + amp * np.sin(phase)]])

    def predict(self) -> None:
        """
        EKF prediction step

        Performs the prediction step of the Extended Kalman Filter (EKF) algorithm.
        This step propagates the state estimate and covariance matrix forward in time
        using the state transition model and process noise.

        Parameters
        ----------
        self : object
            The EKF instance containing the following attributes:
            - x : array-like
                Current state vector
            - P : array-like
                Current covariance matrix
            - dt : float
                Time step
            - Q : array-like
                Process noise covariance matrix
            - state_transition : callable
                Function that computes the state transition

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return anything.

        Notes
        -----
        The prediction step follows the standard EKF equations:
        - State prediction: x = f(x)
        - Covariance prediction: P = F * P * F^T + Q

        The state transition matrix F is hardcoded for a 4-dimensional state vector
        with position and velocity components, where the third component (typically
        acceleration) is integrated over time.

        Examples
        --------
        >>> ekf = EKF()
        >>> ekf.predict()
        >>> # State and covariance matrices are updated in-place
        """
        # Propagate state
        self.x = self.state_transition(self.x)

        # Jacobian of state transition
        F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: NDArray) -> None:
        """
        EKF update step.

        Perform the measurement update step of the Extended Kalman Filter (EKF).

        Parameters
        ----------
        measurement : NDArray
            The actual measurement vector used to update the state estimate.

        Returns
        -------
        None
            This method modifies the state and covariance matrices in-place.

        Notes
        -----
        This implementation assumes a specific measurement function structure where the state vector
        contains [dc, amp, phase, freq] components. The phase is constrained to remain within [0, 2π]
        after each update.

        Examples
        --------
        >>> ekf = ExtendedKalmanFilter()
        >>> measurement = np.array([[1.0], [0.5], [0.2]])
        >>> ekf.update(measurement)
        >>> print(ekf.x)
        [[1.0]
         [0.5]
         [0.2]
         [0.0]]
        """
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

    def filter_signal(self, signal_data: NDArray, missing_indices: list | None = None) -> NDArray:
        """
        Apply filtering to signal data using a Kalman filter approach.

        This function processes signal data through a Kalman filter, handling missing measurements
        and NaN values appropriately. It returns both the filtered signal and corresponding heart rates.

        Parameters
        ----------
        signal_data : NDArray
            Array containing the raw signal measurements to be filtered.
        missing_indices : list of int, optional
            List of indices where measurements are missing. If None, no special handling
            is applied for missing data. Default is None.

        Returns
        -------
        tuple of (NDArray, NDArray)
            A tuple containing:
            - filtered : NDArray
                Array of filtered signal values corresponding to the input measurements
            - heart_rates : NDArray
                Array of heart rate values computed at each time step

        Notes
        -----
        The function uses the following logic for each measurement:
        - For missing indices: Uses current state prediction
        - For NaN values: Uses current state prediction
        - For valid measurements: Updates the filter with the measurement and returns the prediction

        Examples
        --------
        >>> filtered_signal, hr_values = filter_signal(signal_data, missing_indices=[2, 5])
        >>> print(f"Filtered signal shape: {filtered_signal.shape}")
        >>> print(f"Heart rates shape: {hr_values.shape}")
        """
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

    def __init__(
        self,
        dt: float = 0.01,
        hr_estimate: float = 75,
        process_noise: float = 0.001,
        measurement_noise: float = 0.05,
    ) -> None:
        """
        Initialize the Kalman filter for heart rate estimation from PPG signals.

        This constructor sets up the state vector, covariance matrices, and noise parameters
        for a Kalman filter designed to track the fundamental frequency and harmonics of
        a photoplethysmography (PPG) signal to extract heart rate information.

        Parameters
        ----------
        dt : float, optional
            Sampling interval in seconds (default is 0.01, corresponding to 100 Hz).
        hr_estimate : float, optional
            Initial heart rate estimate in beats per minute (BPM) (default is 75).
        process_noise : float, optional
            Process noise covariance (Q). Controls how much the state can change over time
            (default is 0.001).
        measurement_noise : float, optional
            Measurement noise covariance (R). Represents sensor noise level (default is 0.05).

        Returns
        -------
        None
            This method initializes instance attributes and does not return any value.

        Notes
        -----
        The state vector contains six elements:
        [DC offset, A1, A2, A3, phase, frequency]

        The Kalman filter uses a linear model to track the time-varying components of the
        PPG signal, with the frequency component directly related to heart rate.

        Examples
        --------
        >>> kf = KalmanFilter(dt=0.02, hr_estimate=80, process_noise=0.005)
        >>> print(kf.x)
        [[0. ]
         [1. ]
         [0.2]
         [0.1]
         [0. ]
         [2.0943951023931953]]
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

    def get_heart_rate(self) -> float:
        """
        Extract current heart rate estimate from state.

        This function converts the angular frequency stored in the state vector
        to beats per minute (BPM), which represents the current heart rate estimate.

        Parameters
        ----------
        self : object
            The instance containing the state vector with heart rate information.
            The state vector is expected to have at least 6 elements, with the 6th
            element (index 5) containing the angular frequency in radians/second.

        Returns
        -------
        float
            Current heart rate estimate in beats per minute (BPM).

        Notes
        -----
        The conversion from angular frequency (radians/second) to BPM is calculated as:
        BPM = frequency × 60 / (2 × π)

        Examples
        --------
        >>> hr = obj.get_heart_rate()
        >>> print(f"Current heart rate: {hr:.1f} BPM")
        Current heart rate: 72.0 BPM
        """
        frequency = self.x[5, 0]  # radians/second
        hr = frequency * 60 / (2 * np.pi)  # Convert to BPM
        return hr

    def state_transition(self, x: NDArray) -> NDArray:
        """
        Nonlinear state transition for harmonic sinusoidal model.

        This function performs the state transition for a harmonic sinusoidal model
        where the phase evolves according to the frequency and time step, while other
        state variables remain constant.

        Parameters
        ----------
        x : ndarray
            State vector of shape (6,) containing:
            - dc: DC offset component
            - a1: First harmonic amplitude
            - a2: Second harmonic amplitude
            - a3: Third harmonic amplitude
            - phase: Current phase value
            - freq: Frequency value

        Returns
        -------
        ndarray
            Transited state vector of shape (6, 1) with updated phase and unchanged
            other state components.

        Notes
        -----
        The phase is updated using the formula: new_phase = (phase + freq * dt) % (2 * π)
        where dt is the time step stored in self.dt. This ensures the phase remains
        within the [0, 2π) range.

        Examples
        --------
        >>> import numpy as np
        >>> model = HarmonicModel(dt=0.1)
        >>> x = np.array([[1.0], [0.5], [0.3], [0.2], [0.1], [0.05]])
        >>> x_new = model.state_transition(x)
        >>> print(x_new)
        [[1.0]
         [0.5]
         [0.3]
         [0.2]
         [0.105]
         [0.05]]
        """
        dc, a1, a2, a3, phase, freq = x.flatten()

        # Update phase based on frequency
        new_phase = (phase + freq * self.dt) % (2 * np.pi)

        # Other states remain constant in the model
        return np.array([[dc], [a1], [a2], [a3], [new_phase], [freq]])

    def measurement_function(self, x: NDArray) -> NDArray:
        """
        Measurement model: DC + A1*sin(phase) + A2*sin(2*phase) + A3*sin(3*phase)
        This models the fundamental frequency and first two harmonics.

        Parameters
        ----------
        x : ndarray
            Input array of shape (6,) containing parameters in order:
            [dc, a1, a2, a3, phase, freq]
            - dc: DC offset component
            - a1: Amplitude of fundamental frequency
            - a2: Amplitude of second harmonic
            - a3: Amplitude of third harmonic
            - phase: Phase angle in radians
            - freq: Frequency component (not directly used in calculation)

        Returns
        -------
        ndarray
            Measurement output array of shape (1, 1) containing the computed signal value.
            The output represents a sum of sinusoidal components with different frequencies
            and amplitudes, modeling a signal with fundamental and harmonic components.

        Notes
        -----
        This function implements a harmonic model that combines:
        - A DC component (dc)
        - Fundamental frequency component (a1 * sin(phase))
        - Second harmonic component (a2 * sin(2 * phase))
        - Third harmonic component (a3 * sin(3 * phase))

        The frequency parameter is included in the input array but not used in the computation,
        suggesting this might be part of a larger system where frequency is handled elsewhere.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([1.0, 0.5, 0.3, 0.2, np.pi/4, 1.0])
        >>> result = measurement_function(x)
        >>> print(result)
        [[1.82940168]]
        """
        dc, a1, a2, a3, phase, freq = x.flatten()
        y = dc + a1 * np.sin(phase) + a2 * np.sin(2 * phase) + a3 * np.sin(3 * phase)
        return np.array([[y]])

    def predict(self) -> None:
        """
        EKF prediction step.

        Performs the prediction step of the Extended Kalman Filter (EKF) algorithm,
        propagating the state estimate and error covariance forward in time.

        Parameters
        ----------
        self : object
            The EKF instance containing the following attributes:
            - x : array-like
                Current state vector
            - P : array-like
                Current error covariance matrix
            - Q : array-like
                Process noise covariance matrix
            - dt : float
                Time step

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return anything.

        Notes
        -----
        The prediction step involves:
        1. State propagation using the state transition function
        2. Jacobian matrix computation for the state transition
        3. Error covariance prediction using the matrix equation: P = F*P*F^T + Q

        The state transition matrix F is structured as:
        - First three rows represent constant state variables (dc, a1, a2)
        - Fourth row represents constant state variable (a3)
        - Fifth row represents phase that depends on frequency
        - Sixth row represents frequency with constant rate of change

        Examples
        --------
        >>> ekf = EKF()
        >>> ekf.predict()
        >>> # State and covariance matrices are updated in-place
        """
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

    def update(self, measurement: NDArray) -> None:
        """
        EKF update step.

        Perform the Kalman filter update step using the provided measurement.

        Parameters
        ----------
        measurement : NDArray
            The measured values used to update the state estimate. Shape should be
            compatible with the measurement function output.

        Returns
        -------
        None
            This method modifies the instance's state variables `self.x` and `self.P`
            in place.

        Notes
        -----
        The update step uses the measurement function to predict the measurement
        based on the current state, computes the innovation, and updates the state
        and covariance using the Kalman gain.

        The phase component of the state vector is constrained to the range [0, 2π]
        after the update.

        Examples
        --------
        >>> ekf = ExtendedKalmanFilter()
        >>> z = np.array([[1.0]])
        >>> ekf.update(z)
        """
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

    def filter_signal(self, signal_data: NDArray, missing_indices: list | None = None) -> NDArray:
        """
        Filter entire signal and track heart rate.

        This function applies a filtering process to the input signal data, handling missing values
        and tracking heart rate and harmonic amplitudes at each time point. It uses a Kalman filter
        framework for prediction and update steps, with special handling for missing data points.

        Parameters
        ----------
        signal_data : array-like
            Input signal data. Use `np.nan` to indicate missing values.
        missing_indices : array-like, optional
            Indices of missing data points in `signal_data`. If not provided, all data points
            are assumed to be valid.

        Returns
        -------
        filtered : ndarray
            Filtered and interpolated signal values.
        heart_rates : ndarray
            Estimated heart rate at each time point.
        harmonic_amplitudes : ndarray
            Array of shape (n_samples, 3) containing the amplitudes [A1, A2, A3] of the first
            three harmonics at each time point.

        Notes
        -----
        - The function assumes the existence of a Kalman filter class with methods:
          `predict()`, `update()`, `measurement_function()`, and `get_heart_rate()`.
        - Heart rate is tracked and stored in `self.hr_history` during processing.
        - Missing data points are handled by using the current state estimate from the filter
          without updating the filter with new measurements.

        Examples
        --------
        >>> filtered_signal, hr_estimates, harmonics = filter_signal(signal_data, missing_indices)
        >>> print(f"Filtered signal shape: {filtered_signal.shape}")
        >>> print(f"Heart rate estimates: {hr_estimates}")
        >>> print(f"Harmonic amplitudes: {harmonics}")
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

    def __init__(self, fs: float = 100.0, window_size: float = 5.0) -> None:
        """
        Initialize the quality assessment parameters.

        Parameters
        ----------
        fs : float, default=100.0
            Sampling frequency in Hz
        window_size : float, default=5.0
            Window size in seconds for quality assessment

        Returns
        -------
        None
            This method initializes instance attributes but does not return any value

        Notes
        -----
        The window size is converted to the number of samples based on the sampling frequency.
        The resulting number of samples is stored in ``self.window_samples``.

        Examples
        --------
        >>> qa = QualityAssessor(fs=200.0, window_size=2.5)
        >>> qa.window_samples
        500
        """
        self.fs = fs
        self.window_samples = int(window_size * fs)

    def assess_quality(
        self, signal_segment: NDArray, filtered_segment: NDArray | None = None
    ) -> tuple[float, dict]:
        """
        Assess the quality of a signal segment based on multiple physiological and signal-processing metrics.

        Parameters
        ----------
        signal_segment : ndarray
            The raw signal segment to be assessed.
        filtered_segment : ndarray, optional
            The filtered version of the signal segment. If provided, used to compute SNR.
            If not provided, SNR is set to a default value of 0.5.

        Returns
        -------
        quality_score : float
            Overall quality score normalized between 0 and 1, where 1 indicates the best quality.
        metrics : dict
            Dictionary containing individual quality metrics:
            - 'snr': Signal-to-noise ratio normalized to 0-1.
            - 'perfusion': Relative pulse amplitude normalized to 0-1.
            - 'spectral_purity': Proportion of power in the physiological heart rate band (0.5-3.0 Hz).
            - 'kurtosis': Measure of outliers/artifacts, normalized to 0-1 (lower is better).
            - 'zero_crossing': Regularity of zero crossings, normalized to 0-1.

        Notes
        -----
        This function computes a weighted average of several quality metrics to produce an overall score.
        The weights are chosen to reflect the relative importance of each metric in assessing PPG signal quality.

        Examples
        --------
        >>> quality_score, metrics = assess_quality(signal_segment, filtered_segment)
        >>> print(f"Quality Score: {quality_score:.2f}")
        >>> print(f"SNR: {metrics['snr']:.2f}")
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

    def assess_continuous(
        self, signal: NDArray, filtered: NDArray | None = None, stride: float = 1.0
    ) -> tuple[NDArray, NDArray]:
        """
        Assess quality continuously along the signal.

        This function evaluates signal quality at multiple time points by sliding a window
        across the input signal. For each window, a quality score is computed using the
        internal `assess_quality` method. The stride parameter controls the overlap between
        consecutive windows.

        Parameters
        ----------
        signal : array-like
            Input signal to be assessed for quality.
        filtered : array-like, optional
            Filtered version of the signal used for SNR calculation. If None, no filtering
            is applied in the quality assessment.
        stride : float, default=1.0
            Stride between windows in seconds. Controls the overlap between consecutive
            windows. A stride of 1.0 means no overlap, while smaller values create overlap.

        Returns
        -------
        times : ndarray
            Time points corresponding to quality scores, in seconds.
        quality_scores : ndarray
            Quality scores at each time point. The scores are computed using the internal
            `assess_quality` method for each signal segment.

        Notes
        -----
        The function uses a sliding window approach where the window size is defined by
        `self.window_samples` and the sampling frequency by `self.fs`. The quality assessment
        is performed on non-overlapping segments of the signal, with the stride determining
        the step size between consecutive segments.

        Examples
        --------
        >>> times, scores = obj.assess_continuous(signal, filtered, stride=0.5)
        >>> print(f"Quality scores: {scores}")
        >>> print(f"Time points: {times}")
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

    def __init__(self, fs: float = 100.0) -> None:
        """
        Initialize the object with a sampling frequency.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz. Default is 100.0 Hz.

        Returns
        -------
        None

        Notes
        -----
        This constructor sets the sampling frequency attribute for the object.
        The sampling frequency determines the rate at which signals are sampled
        and is crucial for digital signal processing applications.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.fs
        100.0

        >>> obj = MyClass(fs=200.0)
        >>> obj.fs
        200.0
        """
        self.fs = fs

    def extract_from_peaks(
        self, ppg_signal: NDArray, min_distance: float = 0.4
    ) -> tuple[float | None, NDArray, NDArray | None, NDArray | None]:
        """
        Extract heart rate from a PPG signal using peak detection.

        This function detects peaks in the PPG signal, computes inter-beat intervals (IBIs),
        removes outliers, and calculates the corresponding heart rate in beats per minute (BPM).
        It also generates a heart rate waveform (RRI) based on the detected peaks.

        Parameters
        ----------
        ppg_signal : array_like
            Input photoplethysmography (PPG) signal.
        min_distance : float, optional
            Minimum time between peaks in seconds. Default is 0.4 seconds.

        Returns
        -------
        tuple
            A tuple containing:
            - hr : float or None
                Heart rate in beats per minute (BPM). Returns None if not enough peaks are found.
            - peak_indices : ndarray
                Indices of detected peaks in the PPG signal.
            - rri : ndarray or None
                Inter-beat interval waveform (RRI) in seconds. Returns None if processing fails.
            - hr_waveform : ndarray or None
                Heart rate waveform derived from RRI. Returns None if processing fails.

        Notes
        -----
        - The function uses `scipy.signal.find_peaks` for peak detection.
        - Outliers in inter-beat intervals are filtered using a median-based approach.
        - The RRI waveform is constructed by interpolating valid IBIs between peaks.
        - If the signal has insufficient peaks or all IBIs are outliers, the function returns None for heart rate.

        Examples
        --------
        >>> hr, peaks, rri, hr_waveform = extractor.extract_from_peaks(ppg_signal, min_distance=0.4)
        >>> print(f"Heart Rate: {hr} BPM")
        Heart Rate: 72.5 BPM
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
        for peakidx in range(len(peaks) - 1):
            if (median_ibi * 0.7) <= ibi[peakidx] <= (median_ibi * 1.3):
                rri[peaks[peakidx] : peaks[peakidx + 1]] = ibi[peakidx]
            else:
                rri[peaks[peakidx] : peaks[peakidx + 1]] = 0.0
        rri[0 : peaks[0]] = rri[peaks[0]]
        rri[peaks[-1] :] = rri[peaks[-1]]

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
            for first, last in badranges:
                if first == 0:
                    rri[first : last + 1] = rri[last + 1]
                elif last == (len(rri) - 1):
                    rri[first : last + 1] = rri[first - 1]
                else:
                    rri[first : last + 1] = (rri[first - 1] + rri[last + 1]) / 2.0

        # Convert to heart rate
        hr = 60.0 / np.mean(valid_ibi)
        if rri is not None:
            hr_waveform = 60.0 / rri
        else:
            hr_waveform = None

        return hr, peaks, rri, hr_waveform

    def extract_from_fft(
        self, ppg_signal: NDArray, hr_range: tuple[float, float] = (40.0, 180.0)
    ) -> tuple[float | None, float | None, NDArray, NDArray]:
        """
        Extract heart rate using FFT (frequency domain).

        This function computes the power spectral density (PSD) of the input PPG signal
        using Welch's method and identifies the dominant frequency within a specified
        heart rate range. The dominant frequency is then converted to heart rate in BPM.

        Parameters
        ----------
        ppg_signal : NDArray
            Input photoplethysmography (PPG) signal.
        hr_range : tuple of float, optional
            Expected heart rate range in beats per minute (BPM). Default is (40.0, 180.0).

        Returns
        -------
        hr : float or None
            Dominant heart rate in BPM. Returns None if no valid peak is found in the
            specified range.
        frequency : float or None
            Dominant frequency in Hz. Returns None if no valid peak is found in the
            specified range.
        psd : NDArray
            Power spectral density values corresponding to the frequency array.
        freqs : NDArray
            Frequency values corresponding to the power spectral density.

        Notes
        -----
        The function uses Welch's method for PSD estimation, which is robust for
        noisy signals. The heart rate is derived from the peak frequency in the
        physiological range (default: 40-180 BPM).

        Examples
        --------
        >>> hr, freq, psd, freqs = extract_from_fft(ppg_signal, hr_range=(50.0, 120.0))
        >>> print(f"Heart rate: {hr} BPM")
        Heart rate: 72.5 BPM
        """
        # Compute power spectral density
        freqs, psd = signal.welch(ppg_signal, fs=self.fs, nperseg=min(256, len(ppg_signal)))

        # Convert HR range to frequency range
        freq_range = (hr_range[0] / 60.0, hr_range[1] / 60.0)

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

    def extract_continuous(
        self,
        ppg_signal: NDArray,
        window_size: float = 10.0,
        stride: float = 2.0,
        method: str = "fft",
    ) -> tuple[NDArray, NDArray]:
        """
        Extract heart rate continuously along the PPG signal.

        This function computes heart rate estimates over time by sliding a window
        across the PPG signal and applying either FFT-based or peak-based methods
        to each segment. The heart rate is estimated at the center of each window.

        Parameters
        ----------
        ppg_signal : array_like
            Input PPG signal as a 1D array of samples.
        window_size : float, optional
            Size of the sliding window in seconds. Default is 10.0 seconds.
        stride : float, optional
            Stride between consecutive windows in seconds. Default is 2.0 seconds.
        method : str, optional
            Estimation method to use. Either 'fft' for FFT-based heart rate estimation
            or 'peaks' for peak-based estimation. Default is 'fft'.

        Returns
        -------
        times : ndarray
            Time points (in seconds) corresponding to the heart rate estimates.
        heart_rates : ndarray
            Heart rate estimates in beats per minute (BPM) at each time point.

        Notes
        -----
        - The function uses the sampling frequency (`self.fs`) to convert time values
          into sample indices.
        - If the `method` is 'fft', the function calls `self.extract_from_fft()`.
        - If the `method` is 'peaks', the function calls `self.extract_from_peaks()`.
        - Heart rate estimates are only included if they are not `None`.

        Examples
        --------
        >>> times, hrs = extract_continuous(ppg_signal, window_size=5.0, stride=1.0)
        >>> print(times[:3])  # First three time points
        >>> print(hrs[:3])    # First three heart rate estimates
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

    def __init__(
        self,
        fs: float = 100.0,
        method: str = "adaptive",
        hr_estimate: float = 75.0,
        process_noise: float = 0.0001,
    ) -> None:
        """
        Initialize the PPG signal processing pipeline with specified parameters.

        Parameters
        ----------
        fs : float, optional
            Sampling frequency in Hz, default is 100.0
        method : str, optional
            Filter method to use, must be one of 'standard', 'adaptive', or 'ekf',
            default is 'adaptive'
        hr_estimate : float, optional
            Initial heart rate estimate in BPM, default is 75.0
        process_noise : float, optional
            Process noise covariance value, default is 0.0001

        Returns
        -------
        None
            This method initializes the instance attributes and sets up the
            appropriate filter and processing components based on the specified method.

        Notes
        -----
        The initialization creates different filter types based on the method parameter:
        - 'standard': Uses PPGKalmanFilter with fixed measurement noise
        - 'adaptive': Uses AdaptivePPGKalmanFilter with adaptive noise estimation
        - 'ekf': Uses ExtendedPPGKalmanFilter with extended Kalman filter approach

        Examples
        --------
        >>> processor = PPGProcessor(fs=125.0, method='ekf', hr_estimate=80.0)
        >>> processor = PPGProcessor()  # Uses default parameters
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

    def process(
        self,
        signal_data: NDArray,
        missing_indices: list | None = None,
        quality_threshold: float = 0.5,
    ) -> dict:
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
            hr_from_peaks, peak_indices, rri, hr_waveform_from_peaks = (
                self.hr_extractor.extract_from_peaks(filtered)
            )
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

    def __init__(self, fs: float = 100.0) -> None:
        """
        Initialize the object with sampling frequency.

        Parameters
        ----------
        fs : float, default=100.0
            Sampling frequency in Hz. This parameter determines the rate at which
            signals are sampled and is crucial for proper signal processing.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The sampling frequency is stored as an instance attribute and is used
        throughout the class for time-domain and frequency-domain calculations.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.fs
        100.0

        >>> obj = MyClass(fs=200.0)
        >>> obj.fs
        200.0
        """
        self.fs = fs

    def extract_hrv_features(self, peak_indices: NDArray) -> dict | None:
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

    def extract_morphology_features(self, signal_segment: NDArray, peak_idx: int) -> dict:
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

    def compute_spo2_proxy(self, filtered_signal: NDArray) -> float:
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


def read_happy_ppg(
    filenameroot: str, debug: bool = False
) -> tuple[NDArray, float, NDArray, NDArray, NDArray | None, list]:
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

    pleth_ppg = None

    return t, Fs, clean_ppg, raw_ppg, pleth_ppg, missing_indices


def generate_synthetic_ppg(
    duration: int = 10,
    fs: float = 100.0,
    hr: int = 75,
    noise_level: float = 0.05,
    missing_percent: int = 5,
    motion_artifacts: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
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
