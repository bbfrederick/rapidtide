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

import rapidtide.filter as tide_filt
import rapidtide.ppgproc as tide_ppg


def main():  # Example usage
    Fs = 25.0
    process_noise = 0.001
    hr_estimate = 65.0
    qual_thresh = 0.3

    # Generate synthetic PPG signal
    """
    t, clean_ppg, noisy_ppg, corrupted_ppg, missing_indices = generate_synthetic_ppg(
        duration=150, fs=Fs, hr=75, noise_level=0.08, missing_percent=8, motion_artifacts=True
    )"""

    # read in some real data
    filenameroot = "/Users/frederic/code/rapidtide/rapidtide/data/examples/dst/happy_desc-slicerescardfromfmri_timeseries"
    t, Fs, clean_ppg, corrupted_ppg, missing_indices = tide_ppg.read_happy_ppg(
        filenameroot, debug=True
    )
    print(f"{t.shape=}")
    print(f"{t=}")
    print(f"{Fs=}")
    print(f"{process_noise=}")
    print(f"{hr_estimate=}")
    print(f"{missing_indices=}")
    print(f"{clean_ppg.shape=}")
    print(f"{corrupted_ppg.shape=}")
    clean_ppg /= np.std(clean_ppg)
    rollofffilter = tide_filt.NoncausalFilter(filtertype="arb")
    rollofffilter.setfreqs(0.0, 0.0, 1.0, 4.0)
    corrupted_ppg = rollofffilter.apply(Fs, corrupted_ppg)
    corrupted_ppg /= np.std(corrupted_ppg)
    # corrupted_ppg = clean_ppg + 0.0

    """
    # Apply standard PPG Kalman filter
    kf = PPGKalmanFilter(dt=(1.0 / Fs), process_noise=process_noise, measurement_noise=0.10)
    filtered_standard = kf.filter_signal(corrupted_ppg, missing_indices)

    # Apply adaptive PPG Kalman filter
    akf = AdaptivePPGKalmanFilter(
        dt=(1.0 / Fs), initial_process_noise=process_noise, initial_measurement_noise=0.05
    )
    filtered_adaptive, motion_flags = akf.filter_signal(corrupted_ppg, missing_indices)
    """

    # Apply Extended Kalman filter with sinusoidal model
    ekf = tide_ppg.ExtendedPPGKalmanFilter(
        dt=(1.0 / Fs), hr_estimate=hr_estimate, process_noise=process_noise, measurement_noise=0.05
    )
    filtered_ekf, ekf_heart_rates = ekf.filter_signal(clean_ppg, missing_indices)

    """
    # Apply Harmonic Kalman filter with fundamental + 2 harmonics
    hkf = HarmonicPPGKalmanFilter(
        dt=(1.0 / Fs), hr_estimate=hr_estimate, process_noise=process_noise, measurement_noise=0.05
    )
    filtered_hkf, hkf_heart_rates, hkf_harmonic_amps = hkf.filter_signal(clean_ppg, missing_indices)
    """

    # Extract heart rate using frequency methods
    hr_extractor = tide_ppg.HeartRateExtractor(fs=Fs)
    hr_times, hr_values = hr_extractor.extract_continuous(
        filtered_ekf, window_size=10.0, stride=2.0, method="fft"
    )

    # Assess signal quality
    quality_assessor = tide_ppg.SignalQualityAssessor(fs=Fs, window_size=5.0)
    # qual_times, qual_scores = quality_assessor.assess_continuous(
    #    corrupted_ppg, filtered_ekf, stride=1.0
    # )
    qual_times, qual_scores = quality_assessor.assess_continuous(
        clean_ppg, filtered_ekf, stride=1.0
    )

    # Also get single HR estimate from peaks and beat to beat
    hr_from_peaks, peak_indices, rri, hr_waveform_from_peaks = hr_extractor.extract_from_peaks(
        filtered_ekf
    )
    print(f"HR from peaks: {hr_from_peaks}")
    print(f"Peak indices: {peak_indices}")
    print(f"RRIs: {rri}")
    print(f"hr_waveform_from_peaks: {hr_waveform_from_peaks}")

    # Plot results
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.3)

    thissubfig = 0
    # Plot 1: Original and corrupted
    ax1 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    ax1.plot(t, clean_ppg, "g-", label="Clean PPG", alpha=0.7, linewidth=1.5)
    ax1.plot(t, corrupted_ppg, "r.", label="Corrupted (noisy + missing)", markersize=3, alpha=0.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("PPG Signal: Original vs Corrupted")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    """
    # Plot 2: Standard Kalman filter
    ax2 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    ax2.plot(t, clean_ppg, "g-", label="Clean PPG", alpha=0.7, linewidth=1)
    ax2.plot(t, filtered_standard, "b-", label="Standard Kalman Filter", linewidth=1.5)
    ax2.plot(
        t[missing_indices],
        filtered_standard[missing_indices],
        "mo",
        label="Interpolated points",
        markersize=5,
        alpha=0.7,
    )
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Standard Kalman Filter Recovery")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Adaptive Kalman filter with motion detection
    ax3 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    ax3.plot(t, clean_ppg, "g-", label="Clean PPG", alpha=0.7, linewidth=1)
    ax3.plot(t, filtered_adaptive, "c-", label="Adaptive Kalman Filter", linewidth=1.5)
    ax3.plot(
        t[missing_indices],
        filtered_adaptive[missing_indices],
        "mo",
        label="Interpolated points",
        markersize=5,
        alpha=0.7,
    )
    if np.any(motion_flags):
        ax3.plot(
            t[motion_flags],
            filtered_adaptive[motion_flags],
            "r^",
            label="Detected motion artifacts",
            markersize=6,
            alpha=0.5,
        )
    # Mark detected peaks
    ax3.plot(
        t[peak_indices],
        filtered_adaptive[peak_indices],
        "kx",
        label="Detected peaks",
        markersize=8,
        markeredgewidth=2,
    )
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Adaptive Kalman Filter (with Motion Detection)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    """

    # Plot 4: Extended Kalman filter (sinusoidal model)
    ax4 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    ax4.plot(t, clean_ppg, "g-", label="Clean PPG", alpha=0.7, linewidth=1)
    ax4.plot(t, corrupted_ppg, "r.", label="Corrupted (noisy + missing)", markersize=3, alpha=0.5)
    ax4.plot(t, filtered_ekf, "m-", label="Extended Kalman Filter", linewidth=1.5)
    ax4.plot(
        t[missing_indices],
        filtered_ekf[missing_indices],
        "ro",
        label="Interpolated points",
        markersize=5,
        alpha=0.7,
    )
    # Mark detected peaks
    ax4.plot(
        t[peak_indices],
        filtered_ekf[peak_indices],
        "kx",
        label="Detected peaks",
        markersize=8,
        markeredgewidth=2,
    )
    ax4.set_ylabel("Amplitude")
    ax4.set_title("Extended Kalman Filter (Sinusoidal Model)")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    """
    # Plot 5: Harmonic Kalman filter (sinusoidal model)
    ax5 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    ax5.plot(t, clean_ppg, "g-", label="Clean PPG", alpha=0.7, linewidth=1)
    ax5.plot(t, corrupted_ppg, "r.", label="Corrupted (noisy + missing)", markersize=3, alpha=0.5)
    ax5.plot(t, filtered_hkf, "m-", label="Harmonic Kalman Filter", linewidth=1.5)
    ax5.plot(
        t[missing_indices],
        filtered_hkf[missing_indices],
        "ro",
        label="Interpolated points",
        markersize=5,
        alpha=0.7,
    )
    # Mark detected peaks
    ax5.plot(
        t[peak_indices],
        filtered_hkf[peak_indices],
        "kx",
        label="Detected peaks",
        markersize=8,
        markeredgewidth=2,
    )
    ax5.set_ylabel("Amplitude")
    ax5.set_title("Extended Kalman Filter (Harmonic Model)")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)
    """

    # Plot 6: Heart rate extraction
    ax6 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    ax6.plot(hr_times, hr_values, "b-", label="FFT-based HR", linewidth=2, marker="o")
    ax6.plot(t, ekf_heart_rates, "r-", label="EKF-based HR", linewidth=1.5, alpha=0.7)

    if hr_waveform_from_peaks is not None:
        ax6.plot(t, hr_waveform_from_peaks, "g-", label="Peak-based HR", linewidth=1.5, alpha=0.7)

    if hr_from_peaks is not None:
        ax6.axhline(
            y=hr_from_peaks,
            color="g",
            linestyle="--",
            label=f"Peak-based HR: {hr_from_peaks:.1f} BPM",
            linewidth=2,
        )
    ax6.set_ylabel("Heart Rate (BPM)")
    ax6.set_title("Heart Rate Extraction")
    ax6.legend(loc="upper right")
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([40, 110])

    # Plot 7: Signal quality assessment
    ax7 = fig.add_subplot(gs[thissubfig, 0])
    thissubfig += 1
    quality_colors = plt.cm.RdYlGn(qual_scores)  # Red=poor, Green=good
    for i in range(len(qual_times) - 1):
        ax7.axvspan(qual_times[i], qual_times[i + 1], alpha=0.3, color=quality_colors[i])
    ax7.plot(qual_times, qual_scores, "k-", linewidth=2, label="Quality Score")
    ax7.axhline(y=qual_thresh, color="orange", linestyle="--", label="Good quality threshold")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Quality Score")
    ax7.set_title("Signal Quality Assessment (0=Poor, 1=Excellent)")
    ax7.legend(loc="upper right")
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

    # Print comprehensive performance metrics
    """
    mse_standard = np.mean((clean_ppg - filtered_standard) ** 2)
    mse_adaptive = np.mean((clean_ppg - filtered_adaptive) ** 2)
    """
    mse_ekf = np.mean((clean_ppg - filtered_ekf) ** 2)
    """
    mse_hkf = np.mean((clean_ppg - filtered_hkf) ** 2)
    """

    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"\nFiltering Performance:")
    # print(f"  Standard Kalman MSE: {mse_standard:.6f}")
    # print(f"  Adaptive Kalman MSE: {mse_adaptive:.6f}")
    print(f"  Extended Kalman MSE: {mse_ekf:.6f}")
    """    
    print(f"  Harmonic Kalman MSE: {mse_hkf:.6f}")
    print(f"\nHarmonic Kalman Filter Details:")
    print(f"  Mean A1 (fundamental): {np.mean(hkf_harmonic_amps[:, 0]):.3f}")
    print(f"  Mean A2 (2nd harmonic): {np.mean(hkf_harmonic_amps[:, 1]):.3f}")
    print(f"  Mean A3 (3rd harmonic): {np.mean(hkf_harmonic_amps[:, 2]):.3f}")
    print(f"  A2/A1 ratio: {np.mean(hkf_harmonic_amps[:, 1])/np.mean(hkf_harmonic_amps[:, 0]):.3f}")
    print(f"  A3/A1 ratio: {np.mean(hkf_harmonic_amps[:, 2])/np.mean(hkf_harmonic_amps[:, 0]):.3f}")
    """
    print(f"\nData Recovery:")
    print(
        f"  Missing data points: {len(missing_indices)} ({len(missing_indices)/len(t)*100:.1f}%)"
    )
    # print(
    #    f"  Motion artifacts detected: {np.sum(motion_flags)} points ({np.sum(motion_flags)/len(t)*100:.1f}%)"
    # )
    print(f"\nHeart Rate Analysis:")
    print(
        f"  Peak-based HR: {hr_from_peaks:.1f} BPM"
        if hr_from_peaks
        else "  Peak-based HR: Unable to detect"
    )
    print(f"  FFT-based HR (mean): {np.mean(hr_values):.1f} ± {np.std(hr_values):.1f} BPM")
    print(
        f"  EKF-based HR (mean): {np.mean(ekf_heart_rates):.1f} ± {np.std(ekf_heart_rates):.1f} BPM"
    )
    print(f"  Number of detected peaks: {len(peak_indices)}")
    print(f"\nSignal Quality:")
    print(f"  Mean quality score: {np.mean(qual_scores):.3f}")
    print(
        f"  Percentage of good quality signal (>0.7): {np.sum(qual_scores > 0.7)/len(qual_scores)*100:.1f}%"
    )
    print(
        f"  Percentage of poor quality signal (<0.3): {np.sum(qual_scores < 0.3)/len(qual_scores)*100:.1f}%"
    )
    print(f"\n{'='*60}")

    # Demonstrate the complete processing pipeline
    print(f"\n{'='*60}")
    print(f"COMPLETE PROCESSING PIPELINE DEMO")
    print(f"{'='*60}")

    processor = tide_ppg.RobustPPGProcessor(
        fs=Fs, method="ekf", hr_estimate=75.0, process_noise=0.1
    )
    pipeline_results = processor.process(
        filtered_ekf, missing_indices, quality_threshold=qual_thresh
    )

    # processor = RobustPPGProcessor(fs=Fs, method="raw", hr_estimate=75.0, process_noise=0.1)
    # pipeline_results = processor.process(clean_ppg, missing_indices, quality_threshold=qual_thresh)

    print(f"\nPipeline Results:")
    print(f"  Mean quality score: {pipeline_results['mean_quality']:.3f}")
    print(f"  Good quality segments: {pipeline_results['good_quality_percentage']:.1f}%")
    if pipeline_results["hr_overall"] is not None:
        print(f"  Overall heart rate: {pipeline_results['hr_overall']:.1f} BPM")
        print(f"  Detected {len(pipeline_results['peak_indices'])} heartbeats")

    # Extract additional features
    feature_extractor = tide_ppg.PPGFeatureExtractor(fs=Fs)

    if len(peak_indices) > 5:
        hrv_features = feature_extractor.extract_hrv_features(peak_indices)

        if hrv_features is not None:
            print(f"\nHeart Rate Variability (HRV) Features:")
            print(f"  Mean IBI: {hrv_features['mean_ibi']:.1f} ms")
            print(f"  SDNN: {hrv_features['sdnn']:.1f} ms")
            print(f"  RMSSD: {hrv_features['rmssd']:.1f} ms")
            print(f"  pNN50: {hrv_features['pnn50']:.1f}%")

            if "lf_power" in hrv_features:
                print(f"\n  Frequency Domain:")
                print(f"    LF Power: {hrv_features['lf_power']:.2f}")
                print(f"    HF Power: {hrv_features['hf_power']:.2f}")
                print(f"    LF/HF Ratio: {hrv_features['lf_hf_ratio']:.2f}")

    # Extract morphology from a good segment
    if len(peak_indices) > 0:
        # Find a peak in the middle of the signal
        mid_peak = peak_indices[len(peak_indices) // 2]

        # Extract segment around this peak
        segment_start = max(0, mid_peak - int(0.4 * 100))
        segment_end = min(len(filtered_ekf), mid_peak + int(0.6 * 100))
        segment = filtered_ekf[segment_start:segment_end]
        peak_in_segment = mid_peak - segment_start

        morph_features = feature_extractor.extract_morphology_features(segment, peak_in_segment)

        print(f"\nPulse Morphology Features (single pulse):")
        print(f"  Pulse amplitude: {morph_features['pulse_amplitude']:.3f}")
        print(f"  Rising time: {morph_features['rising_time']:.3f} s")
        if "augmentation_index" in morph_features:
            print(f"  Augmentation index: {morph_features['augmentation_index']:.3f}")
        if "pulse_width" in morph_features:
            print(f"  Pulse width (FWHM): {morph_features['pulse_width']:.3f} s")

    # SpO2 proxy
    spo2_proxy = feature_extractor.compute_spo2_proxy(filtered_ekf)
    print(f"\nSpO2 Proxy: {spo2_proxy:.1f}% (Note: This is not a real SpO2 measurement!)")

    print(f"\n{'='*60}")
    print(f"USAGE TIPS")
    print(f"{'='*60}")
    print(
        f"""
For your own PPG data:

1. Choose the appropriate filter:
   - Standard: Clean signals, minimal motion
   - Adaptive: Real-world signals with motion artifacts (RECOMMENDED)
   - EKF: When you need continuous HR tracking

2. Typical parameters for wearable PPG (wrist/finger):
   processor = RobustPPGProcessor(fs=100, method='adaptive')
   results = processor.process(your_signal, quality_threshold=0.5)

3. Handle poor quality segments:
   - Use quality_threshold to filter unreliable data
   - Higher threshold (0.7+) for clinical applications
   - Lower threshold (0.3+) for general monitoring

4. Interpret HRV features:
   - SDNN: Overall HRV (higher = better autonomic function)
   - RMSSD: Short-term variability (parasympathetic activity)
   - LF/HF ratio: Sympatho-vagal balance (2-3 is normal)

5. For missing data:
   - Mark missing samples as np.nan or provide missing_indices
   - The Kalman filter will interpolate automatically
   - Check quality scores after interpolation
    """
    )


if __name__ == "__main__":
    main()
