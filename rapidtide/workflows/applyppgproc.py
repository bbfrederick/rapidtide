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
import argparse
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import rapidtide.filter as tide_filt
import rapidtide.ppgproc as tide_ppg
import rapidtide.workflows.parser_funcs as pf

DEFAULT_PROCESSNOISE = 0.001
DEFAULT_HRESTIMATE = 70.0
DEFAULT_MEASUREMENTNOISE = 0.05
DEFAULT_QUALTHRESH = 0.5


def _get_parser() -> Any:
    """
    Argument parser for applyppgproc.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    to parse command-line arguments for the `applyppgproc` tool. The tool calculates
    PPG (Photoplethysmography) metrics from a cardiacfromfmri output file.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the applyppgproc tool.

    Notes
    -----
    The parser includes both required and optional arguments. Required arguments are:
    - `infileroot`: Root name of the cardiacfromfmri input file (without extension).
    - `outfileroot`: Root name of the output files.

    Optional arguments include:
    - `--process_noise`: Process noise for the PPG filter (default: `DEFAULT_PROCESSNOISE`).
    - `--hr_estimate`: Starting guess for heart rate in BPM (default: `DEFAULT_HRESTIMATE`).
    - `--qual_thresh`: Quality threshold for PPG, between 0 and 1 (default: `DEFAULT_QUALTHRESH`).
    - `--measurement_noise`: Assumed measurement noise (default: `DEFAULT_MEASUREMENTNOISE`).
    - `--display`: Graph the processed waveforms.
    - `--debug`: Print debugging information.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        prog="applyppgproc",
        description=("Calculate PPG metrics from a happy cardiacfromfmri output file."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "infileroot",
        help="The root name of the cardiacfromfmri file (without the json or tsv.gz extension).",
    )
    parser.add_argument(
        "outfileroot",
        help="The root name of the output files.",
    )

    # optional arguments
    parser.add_argument(
        "--process_noise",
        dest="process_noise",
        metavar="NOISE",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help=f"Process noise for the PPG filter (default is {DEFAULT_PROCESSNOISE}). ",
        default=DEFAULT_PROCESSNOISE,
    )
    parser.add_argument(
        "--hr_estimate",
        dest="hr_estimate",
        metavar="BPM",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help=f"Starting guess for heart rate in BPM (default is {DEFAULT_HRESTIMATE}). ",
        default=DEFAULT_HRESTIMATE,
    )
    parser.add_argument(
        "--qual_thresh",
        dest="qual_thresh",
        metavar="THRESH",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help=f"Quality threshold for PPG, between 0 and 1 (default is {DEFAULT_QUALTHRESH}). ",
        default=DEFAULT_QUALTHRESH,
    )
    parser.add_argument(
        "--measurement_noise",
        dest="measurement_noise",
        metavar="NOISE",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help=f"Assumed measurement noise (default is {DEFAULT_MEASUREMENTNOISE}). ",
        default=DEFAULT_MEASUREMENTNOISE,
    )
    parser.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help=("Graph the processed waveforms."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print debugging information."),
        default=False,
    )
    return parser


def procppg(
    args: Any,
) -> Tuple[
    dict,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
]:
    """
    Process PPG (Photoplethysmography) signal using a combination of filtering,
    heart rate extraction, and signal quality assessment techniques.

    This function performs a complete PPG signal processing pipeline including:
    - Reading and preprocessing PPG data
    - Applying Extended Kalman Filter with sinusoidal model
    - Extracting heart rate using FFT, EKF, and peak detection methods
    - Assessing signal quality
    - Computing performance metrics and additional features like HRV and pulse morphology
    - Optionally displaying plots and printing detailed results

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - infileroot : str
            Root path to the input data file(s)
        - display : bool
            Whether to display plots
        - debug : bool
            Whether to print debug information
        - hr_estimate : float
            Initial heart rate estimate for EKF
        - process_noise : float
            Process noise for EKF
        - measurement_noise : float
            Measurement noise for EKF
        - qual_thresh : float
            Threshold for determining good quality signal segments

    Returns
    -------
    tuple
        A tuple containing:
        - ppginfo : dict
            Dictionary with various performance metrics and computed features
        - peak_indices : NDArray
            Indices of detected peaks in the filtered signal
        - rri : NDArray
            Inter-beat intervals (RRIs) derived from peak detection
        - hr_waveform_from_peaks : NDArray
            Heart rate waveform computed from peaks
        - hr_times : NDArray
            Time points for heart rate estimates
        - hr_values : NDArray
            Heart rate values (FFT-based)
        - filtered_ekf : NDArray
            Signal filtered using Extended Kalman Filter
        - ekf_heart_rates : NDArray
            Heart rate estimates from EKF
        - cardiacfromfmri_qual_times : NDArray
            Time points for quality scores from raw fMRI signal
        - cardiacfromfmri_qual_scores : NDArray
            Quality scores for raw fMRI signal
        - dlfiltered_qual_times : NDArray
            Time points for quality scores from DL-filtered signal
        - dlfiltered_qual_scores : NDArray
            Quality scores for DL-filtered signal

    Notes
    -----
    The function uses the `tide_ppg` module for signal processing and analysis.
    It supports visualization via matplotlib when `args.display` is True.
    Heart rate variability (HRV) and pulse morphology features are computed if sufficient peaks are detected.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     infileroot='data/ppg_data',
    ...     display=True,
    ...     debug=False,
    ...     hr_estimate=70.0,
    ...     process_noise=1e-4,
    ...     measurement_noise=1e-2,
    ...     qual_thresh=0.7
    ... )
    >>> procppg(args)
    """
    if args.display:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

    ppginfo = {}

    # read in a happy data file
    t, Fs, dlfiltered_ppg, cardiacfromfmri_ppg, pleth_ppg, missing_indices = (
        tide_ppg.read_happy_ppg(args.infileroot, debug=True)
    )
    dlfiltered_ppg /= np.std(dlfiltered_ppg)
    rollofffilter = tide_filt.NoncausalFilter(filtertype="arb")
    rollofffilter.setfreqs(0.0, 0.0, 1.0, 4.0)
    # cardiacfromfmri_ppg = rollofffilter.apply(Fs, cardiacfromfmri_ppg)
    cardiacfromfmri_ppg /= np.std(cardiacfromfmri_ppg)

    # Apply Extended Kalman filter with sinusoidal model to the dlfiltered timecourse
    ekf = tide_ppg.ExtendedPPGKalmanFilter(
        dt=(1.0 / Fs),
        hr_estimate=args.hr_estimate,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )
    filtered_ekf, ekf_heart_rates = ekf.filter_signal(dlfiltered_ppg, missing_indices)

    # Extract heart rate using frequency methods
    hr_extractor = tide_ppg.HeartRateExtractor(fs=Fs)
    hr_times, hr_values = hr_extractor.extract_continuous(
        filtered_ekf, window_size=10.0, stride=2.0, method="fft"
    )

    # Assess signal quality
    quality_assessor = tide_ppg.SignalQualityAssessor(fs=Fs, window_size=5.0)
    cardiacfromfmri_qual_times, cardiacfromfmri_qual_scores = quality_assessor.assess_continuous(
        cardiacfromfmri_ppg, filtered_ekf, stride=1.0
    )
    dlfiltered_qual_times, dlfiltered_qual_scores = quality_assessor.assess_continuous(
        dlfiltered_ppg, filtered_ekf, stride=1.0
    )

    # Also get single HR estimate from peaks and beat to beat
    ppginfo["hr_from_peaks"], peak_indices, rri, hr_waveform_from_peaks = (
        hr_extractor.extract_from_peaks(filtered_ekf)
    )

    if args.debug:
        print(f"HR from peaks: {ppginfo["hr_from_peaks"]}")
        print(f"Peak indices: {peak_indices}")
        print(f"RRIs: {rri}")
        print(f"hr_waveform_from_peaks: {hr_waveform_from_peaks}")

    # Plot results
    if args.display:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(4, 1, hspace=0.3)

        thissubfig = 0
        # Plot 1: Original and corrupted
        ax1 = fig.add_subplot(gs[thissubfig, 0])
        thissubfig += 1
        ax1.plot(t, dlfiltered_ppg, "g-", label="DL filtered PPG", alpha=0.7, linewidth=1.5)
        ax1.plot(
            t,
            cardiacfromfmri_ppg,
            "r.",
            label="Raw cardiac from fMRI with bad points",
            markersize=3,
            alpha=0.5,
        )
        ax1.set_ylabel("Amplitude")
        ax1.set_title("PPG Signal: Raw cardiac from fmri vs DL filtered")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot 4: Extended Kalman filter (sinusoidal model)
        ax4 = fig.add_subplot(gs[thissubfig, 0])
        thissubfig += 1
        ax4.plot(t, dlfiltered_ppg, "g-", label="DL filtered PPG", alpha=0.7, linewidth=1)
        ax4.plot(
            t,
            cardiacfromfmri_ppg,
            "r.",
            label="Raw cardiac from fMRI with bad points",
            markersize=3,
            alpha=0.5,
        )
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

        # Plot 6: Heart rate extraction
        ax6 = fig.add_subplot(gs[thissubfig, 0])
        thissubfig += 1
        ax6.plot(hr_times, hr_values, "b-", label="FFT-based HR", linewidth=2, marker="o")
        ax6.plot(t, ekf_heart_rates, "r-", label="EKF-based HR", linewidth=1.5, alpha=0.7)

        if hr_waveform_from_peaks is not None:
            ax6.plot(
                t, hr_waveform_from_peaks, "g-", label="Peak-based HR", linewidth=1.5, alpha=0.7
            )

        if ppginfo["hr_from_peaks"] is not None:
            ax6.axhline(
                y=ppginfo["hr_from_peaks"],
                color="g",
                linestyle="--",
                label=f"Peak-based HR: {ppginfo["hr_from_peaks"]:.1f} BPM",
                linewidth=2,
            )
        ax6.set_ylabel("Heart Rate (BPM)")
        ax6.set_title("Heart Rate Extraction")
        ax6.legend(loc="upper right")
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim((40.0, 110.0))

        # Plot 7: Signal quality assessment
        ax7 = fig.add_subplot(gs[thissubfig, 0])
        thissubfig += 1
        quality_colors = plt.cm.RdYlGn(dlfiltered_qual_scores)  # Red=poor, Green=good
        for i in range(len(dlfiltered_qual_times) - 1):
            ax7.axvspan(
                dlfiltered_qual_times[i],
                dlfiltered_qual_times[i + 1],
                alpha=0.3,
                color=quality_colors[i],
            )
        ax7.plot(
            dlfiltered_qual_times, dlfiltered_qual_scores, "k-", linewidth=2, label="Quality Score"
        )
        ax7.axhline(
            y=args.qual_thresh, color="orange", linestyle="--", label="Good quality threshold"
        )
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Quality Score")
        ax7.set_title("Signal Quality Assessment (0=Poor, 1=Excellent)")
        ax7.legend(loc="upper right")
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim((0.0, 1.0))

        plt.tight_layout()
        plt.show()

    # Print comprehensive performance metrics
    ppginfo["mse_ekf"] = np.mean((dlfiltered_ppg - filtered_ekf) ** 2)
    ppginfo["mean_fft_hr"] = np.mean(hr_values)
    ppginfo["std_fft_hr"] = np.std(hr_values)
    ppginfo["mean_ekf_hr"] = np.mean(ekf_heart_rates)
    ppginfo["std_ekf_hr"] = np.std(ekf_heart_rates)
    if hr_waveform_from_peaks is not None:
        ppginfo["mean_peaks_hr"] = np.mean(hr_waveform_from_peaks)
        ppginfo["std_peaks_hr"] = np.std(hr_waveform_from_peaks)
    ppginfo["num_detected_peaks"] = len(peak_indices)
    ppginfo["mean_dlfiltered_qual_scores"] = np.mean(dlfiltered_qual_scores)
    ppginfo["mean_cardiacfromfmri_qual_scores"] = np.mean(cardiacfromfmri_qual_scores)

    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"\nFiltering Performance:")
    print(f"  Extended Kalman MSE: {ppginfo['mse_ekf']:.6f}")
    print(f"\nData Recovery:")
    print(
        f"  Missing data points: {len(missing_indices)} ({len(missing_indices)/len(t)*100:.1f}%)"
    )
    print(f"\nHeart Rate Analysis:")
    print(
        f"  Peak-based HR: {ppginfo["hr_from_peaks"]:.1f} BPM"
        if ppginfo["hr_from_peaks"]
        else "  Peak-based HR: Unable to detect"
    )
    print(f"  FFT-based HR (mean): {ppginfo['mean_fft_hr']:.1f} ± {ppginfo['std_fft_hr']:.1f} BPM")
    print(f"  EKF-based HR (mean): {ppginfo['mean_ekf_hr']:.1f} ± {ppginfo['std_ekf_hr']:.1f} BPM")

    if hr_waveform_from_peaks is not None:
        print(
            f"  Peak-based HR (mean): {ppginfo['mean_peaks_hr']:.1f} ± {ppginfo['std_peaks_hr']:.1f} BPM"
        )
    print(f"  Number of detected peaks: {ppginfo['num_detected_peaks']}")
    print(f"\nSignal Quality:")
    print(f"  Mean quality score: {ppginfo['mean_dlfiltered_qual_scores']:.3f}")
    print(
        f"  Percentage of good quality signal (>0.7): {np.sum(dlfiltered_qual_scores > 0.7)/len(dlfiltered_qual_scores)*100:.1f}%"
    )
    print(
        f"  Percentage of poor quality signal (<0.3): {np.sum(dlfiltered_qual_scores < 0.3)/len(dlfiltered_qual_scores)*100:.1f}%"
    )
    print(f"\n{'='*60}")

    # Demonstrate the complete processing pipeline
    print(f"\n{'='*60}")
    print(f"COMPLETE PROCESSING PIPELINE DEMO")
    print(f"{'='*60}")

    processor = tide_ppg.RobustPPGProcessor(
        fs=Fs, method="ekf", hr_estimate=args.hr_estimate, process_noise=args.process_noise
    )
    pipeline_results = processor.process(
        filtered_ekf, missing_indices, quality_threshold=args.qual_thresh
    )
    if args.debug:
        print(pipeline_results)

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
        ppginfo.update(hrv_features)

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
        ppginfo.update(morph_features)

        print(f"\nPulse Morphology Features (single pulse):")
        print(f"  Pulse amplitude: {morph_features['pulse_amplitude']:.3f}")
        print(f"  Rising time: {morph_features['rising_time']:.3f} s")
        if "augmentation_index" in morph_features:
            print(f"  Augmentation index: {morph_features['augmentation_index']:.3f}")
        if "pulse_width" in morph_features:
            print(f"  Pulse width (FWHM): {morph_features['pulse_width']:.3f} s")

    # SpO2 proxy
    spo2_proxy = feature_extractor.compute_spo2_proxy(filtered_ekf)
    ppginfo["spo2_proxy"] = spo2_proxy
    print(f"\nSpO2 Proxy: {spo2_proxy:.1f}% (Note: This is not a real SpO2 measurement!)")

    print(f"\n{'='*60}")
    print(f"USAGE TIPS")
    print(f"{'='*60}")

    if args.display:
        plt.plot(
            cardiacfromfmri_qual_times, cardiacfromfmri_qual_scores, label="Cardiac from fMRI"
        )
        plt.plot(dlfiltered_qual_times, dlfiltered_qual_scores, label="DLfiltered")
        plt.show()

    if args.debug:
        print(f"{ppginfo=}")

    return (
        ppginfo,
        peak_indices,
        rri,
        hr_waveform_from_peaks,
        peak_indices,
        hr_times,
        hr_values,
        filtered_ekf,
        ekf_heart_rates,
        cardiacfromfmri_qual_times,
        cardiacfromfmri_qual_scores,
        dlfiltered_qual_times,
        dlfiltered_qual_scores,
    )
