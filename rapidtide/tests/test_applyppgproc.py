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
import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.applyppgproc import (
    DEFAULT_HRESTIMATE,
    DEFAULT_MEASUREMENTNOISE,
    DEFAULT_PROCESSNOISE,
    DEFAULT_QUALTHRESH,
    _get_parser,
    procppg,
)

# ---- helpers ----


def _make_args(**overrides):
    """Create a default argparse.Namespace for procppg."""
    defaults = dict(
        infileroot="testdata",
        outfileroot="testout",
        process_noise=DEFAULT_PROCESSNOISE,
        hr_estimate=DEFAULT_HRESTIMATE,
        measurement_noise=DEFAULT_MEASUREMENTNOISE,
        qual_thresh=DEFAULT_QUALTHRESH,
        display=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mock_ppg_data(npoints=500, fs=25.0, hr_bpm=72.0, rng=None):
    """Create synthetic PPG-like data for testing.

    Returns (t, Fs, dlfiltered, cardiacfromfmri, pleth, missing_indices).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    t = np.arange(npoints) / fs
    # Simulate a cardiac waveform: sum of harmonics
    hr_hz = hr_bpm / 60.0
    signal = np.sin(2 * np.pi * hr_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * hr_hz * t)
    dlfiltered = signal + rng.randn(npoints) * 0.05
    cardiacfromfmri = signal + rng.randn(npoints) * 0.2
    pleth = None
    missing_indices = [10, 50, 100]

    return t, fs, dlfiltered, cardiacfromfmri, pleth, missing_indices


def _make_mock_tide_ppg(npoints=500, fs=25.0, hr_bpm=72.0, num_peaks=10):
    """Set up all tide_ppg mock objects for a procppg call.

    Returns a dict of (patch_target -> configured mock) plus some
    reference arrays for assertions.
    """
    rng = np.random.RandomState(42)
    t, Fs, dlfiltered, cardiacfromfmri, pleth, missing_indices = _make_mock_ppg_data(
        npoints=npoints, fs=fs, hr_bpm=hr_bpm, rng=rng
    )

    # EKF outputs
    filtered_ekf = np.sin(2 * np.pi * (hr_bpm / 60.0) * t)  # clean sinusoidal
    ekf_heart_rates = np.full(npoints, hr_bpm) + rng.randn(npoints) * 0.5

    # HR extractor outputs
    hr_times = np.linspace(0, t[-1], 20)
    hr_values = np.full(20, hr_bpm) + rng.randn(20) * 1.0

    # Peak detection outputs
    if num_peaks > 0:
        peak_spacing = npoints // num_peaks
        peak_indices = np.arange(num_peaks) * peak_spacing + peak_spacing // 2
        rri = np.full(npoints, 60.0 / hr_bpm)
        hr_waveform_from_peaks = np.full(npoints, hr_bpm) + rng.randn(npoints) * 0.3
    else:
        peak_indices = np.array([], dtype=int)
        rri = np.array([])
        hr_waveform_from_peaks = None

    # Quality assessor outputs
    qual_times = np.linspace(0, t[-1], 30)
    qual_scores = rng.uniform(0.3, 0.95, 30)

    # Pipeline results
    pipeline_results = {
        "mean_quality": 0.75,
        "good_quality_percentage": 80.0,
        "hr_overall": hr_bpm,
        "peak_indices": peak_indices,
    }

    # HRV features
    hrv_features = {
        "mean_ibi": 833.3,
        "sdnn": 45.0,
        "rmssd": 32.0,
        "pnn50": 15.0,
        "lf_power": 0.45,
        "hf_power": 0.35,
        "lf_hf_ratio": 1.29,
    }

    # Morphology features
    morph_features = {
        "pulse_amplitude": 1.8,
        "rising_time": 0.12,
        "augmentation_index": 0.65,
        "pulse_width": 0.35,
        "dicrotic_notch_amplitude": 0.4,
    }

    # SpO2 proxy
    spo2_proxy = 97.5

    return {
        "read_data": (t, Fs, dlfiltered, cardiacfromfmri, pleth, missing_indices),
        "filtered_ekf": filtered_ekf,
        "ekf_heart_rates": ekf_heart_rates,
        "hr_times": hr_times,
        "hr_values": hr_values,
        "hr_from_peaks": hr_bpm,
        "peak_indices": peak_indices,
        "rri": rri,
        "hr_waveform_from_peaks": hr_waveform_from_peaks,
        "qual_times": qual_times,
        "qual_scores": qual_scores,
        "pipeline_results": pipeline_results,
        "hrv_features": hrv_features,
        "morph_features": morph_features,
        "spo2_proxy": spo2_proxy,
    }


def _run_procppg_with_mocks(args, mock_data):
    """Run procppg with fully mocked tide_ppg, tide_filt, and plt.

    Returns the result tuple from procppg.
    """
    # Build mock objects
    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )

    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )

    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]

    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]

    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        result = procppg(args)

    return result


# ---- _get_parser tests ----


def test_get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    if debug:
        print("Parser created successfully")


def test_get_parser_required_args(debug=False):
    """Test that parser requires both positional arguments."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_get_parser_with_valid_args(debug=False):
    """Test parser with valid positional arguments and check defaults."""
    parser = _get_parser()
    args = parser.parse_args(["input_root", "output_root"])
    assert args.infileroot == "input_root"
    assert args.outfileroot == "output_root"
    assert args.process_noise == DEFAULT_PROCESSNOISE
    assert args.hr_estimate == DEFAULT_HRESTIMATE
    assert args.measurement_noise == DEFAULT_MEASUREMENTNOISE
    assert args.qual_thresh == DEFAULT_QUALTHRESH
    assert args.display is False
    assert args.debug is False

    if debug:
        print(f"Parsed args: {args}")


def test_get_parser_optional_flags(debug=False):
    """Test parser with all optional flags set."""
    parser = _get_parser()
    args = parser.parse_args(
        [
            "infile",
            "outfile",
            "--process_noise",
            "0.01",
            "--hr_estimate",
            "80.0",
            "--qual_thresh",
            "0.7",
            "--measurement_noise",
            "0.1",
            "--display",
            "--debug",
        ]
    )
    assert args.process_noise == 0.01
    assert args.hr_estimate == 80.0
    assert args.qual_thresh == 0.7
    assert args.measurement_noise == 0.1
    assert args.display is True
    assert args.debug is True

    if debug:
        print(f"Parsed args: {args}")


def test_get_parser_invalid_float(debug=False):
    """Test that non-float values for float arguments cause an error."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infile", "outfile", "--process_noise", "notanumber"])


def test_get_parser_missing_outfileroot(debug=False):
    """Test that missing outfileroot causes an error."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infile_only"])


# ---- procppg tests ----


def test_procppg_basic_return_structure(debug=False):
    """Test that procppg returns the correct number and types of outputs."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg()

    result = _run_procppg_with_mocks(args, mock_data)

    # Should return a 13-element tuple
    assert isinstance(result, tuple)
    assert len(result) == 13

    ppginfo = result[0]
    assert isinstance(ppginfo, dict)

    # Remaining elements should be numpy arrays
    for i in range(1, 13):
        assert isinstance(result[i], np.ndarray), f"Element {i} is not ndarray: {type(result[i])}"

    if debug:
        print(f"Return tuple length: {len(result)}")
        print(f"ppginfo keys: {list(ppginfo.keys())}")


def test_procppg_ppginfo_keys(debug=False):
    """Test that ppginfo dict contains all expected metric keys."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg()

    result = _run_procppg_with_mocks(args, mock_data)
    ppginfo = result[0]

    # Core metrics always present
    expected_keys = [
        "hr_from_peaks",
        "mse_ekf",
        "mean_fft_hr",
        "std_fft_hr",
        "mean_ekf_hr",
        "std_ekf_hr",
        "num_detected_peaks",
        "mean_dlfiltered_qual_scores",
        "mean_cardiacfromfmri_qual_scores",
        "spo2_proxy",
    ]
    for key in expected_keys:
        assert key in ppginfo, f"Missing key: {key}"

    # When hr_waveform_from_peaks is not None, peak HR stats present
    assert "mean_peaks_hr" in ppginfo
    assert "std_peaks_hr" in ppginfo

    if debug:
        print(f"ppginfo has {len(ppginfo)} keys: {sorted(ppginfo.keys())}")


def test_procppg_ppginfo_values_reasonable(debug=False):
    """Test that ppginfo values are reasonable and finite."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(hr_bpm=72.0)

    result = _run_procppg_with_mocks(args, mock_data)
    ppginfo = result[0]

    assert ppginfo["hr_from_peaks"] == 72.0
    assert np.isfinite(ppginfo["mse_ekf"])
    assert ppginfo["mse_ekf"] >= 0.0
    assert np.isfinite(ppginfo["mean_fft_hr"])
    assert np.isfinite(ppginfo["mean_ekf_hr"])
    assert ppginfo["num_detected_peaks"] == 10
    assert 0.0 <= ppginfo["mean_dlfiltered_qual_scores"] <= 1.0
    assert ppginfo["spo2_proxy"] == 97.5

    if debug:
        for k, v in ppginfo.items():
            print(f"  {k}: {v}")


def test_procppg_return_arrays_correct(debug=False):
    """Test that returned arrays match mock data."""
    args = _make_args()
    npoints = 500
    mock_data = _make_mock_tide_ppg(npoints=npoints)

    result = _run_procppg_with_mocks(args, mock_data)

    (
        ppginfo,
        peak_indices_1,
        rri,
        hr_waveform,
        peak_indices_2,
        hr_times,
        hr_values,
        filtered_ekf,
        ekf_hr,
        cfmri_qual_times,
        cfmri_qual_scores,
        dl_qual_times,
        dl_qual_scores,
    ) = result

    # peak_indices returned twice (indices 1 and 4)
    np.testing.assert_array_equal(peak_indices_1, mock_data["peak_indices"])
    np.testing.assert_array_equal(peak_indices_2, mock_data["peak_indices"])
    np.testing.assert_array_equal(rri, mock_data["rri"])
    np.testing.assert_array_equal(hr_waveform, mock_data["hr_waveform_from_peaks"])
    np.testing.assert_array_equal(hr_times, mock_data["hr_times"])
    np.testing.assert_array_equal(hr_values, mock_data["hr_values"])
    np.testing.assert_array_equal(filtered_ekf, mock_data["filtered_ekf"])
    np.testing.assert_array_equal(ekf_hr, mock_data["ekf_heart_rates"])
    np.testing.assert_array_equal(cfmri_qual_times, mock_data["qual_times"])
    np.testing.assert_array_equal(dl_qual_times, mock_data["qual_times"])

    if debug:
        print(f"peak_indices shape: {peak_indices_1.shape}")
        print(f"filtered_ekf shape: {filtered_ekf.shape}")


def test_procppg_ekf_params_passed(debug=False):
    """Test that EKF is constructed with correct parameters from args."""
    args = _make_args(
        hr_estimate=80.0,
        process_noise=0.005,
        measurement_noise=0.1,
    )
    mock_data = _make_mock_tide_ppg(fs=25.0)

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        # Check EKF was constructed with correct params
        mock_ekf_cls.assert_called_once_with(
            dt=1.0 / 25.0,
            hr_estimate=80.0,
            process_noise=0.005,
            measurement_noise=0.1,
        )

    if debug:
        print("EKF params verified")


def test_procppg_hr_extractor_called_correctly(debug=False):
    """Test that HeartRateExtractor methods are called with correct params."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(fs=25.0)

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        # HeartRateExtractor created with correct fs
        mock_hr_cls.assert_called_once_with(fs=25.0)

        # extract_continuous called with correct windowing params
        mock_hr_instance.extract_continuous.assert_called_once()
        ec_args = mock_hr_instance.extract_continuous.call_args
        assert ec_args[1]["window_size"] == 10.0
        assert ec_args[1]["stride"] == 2.0
        assert ec_args[1]["method"] == "fft"

        # extract_from_peaks called with the EKF output
        mock_hr_instance.extract_from_peaks.assert_called_once()

    if debug:
        print("HR extractor calls verified")


def test_procppg_quality_assessor_called_twice(debug=False):
    """Test that quality assessor is called twice (cardiacfromfmri and dlfiltered)."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg()

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        # Quality assessor instantiated once with correct fs and window_size
        mock_qa_cls.assert_called_once_with(fs=25.0, window_size=5.0)
        # assess_continuous called twice: once for cardiacfromfmri, once for dlfiltered
        assert mock_qa_instance.assess_continuous.call_count == 2

    if debug:
        print("Quality assessor called twice as expected")


def test_procppg_pipeline_processor_called(debug=False):
    """Test that RobustPPGProcessor is constructed and called correctly."""
    args = _make_args(hr_estimate=80.0, process_noise=0.005, qual_thresh=0.7)
    mock_data = _make_mock_tide_ppg(fs=25.0)

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        # RobustPPGProcessor created with correct params
        mock_proc_cls.assert_called_once_with(
            fs=25.0, method="ekf", hr_estimate=80.0, process_noise=0.005
        )
        # process called with quality_threshold from args
        mock_processor_instance.process.assert_called_once()
        proc_kwargs = mock_processor_instance.process.call_args[1]
        assert proc_kwargs["quality_threshold"] == 0.7

    if debug:
        print("Pipeline processor calls verified")


def test_procppg_hrv_features_with_enough_peaks(debug=False):
    """Test that HRV features are extracted when > 5 peaks detected."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(num_peaks=10)  # > 5

    result = _run_procppg_with_mocks(args, mock_data)
    ppginfo = result[0]

    # HRV features should be merged into ppginfo
    assert "mean_ibi" in ppginfo
    assert "sdnn" in ppginfo
    assert "rmssd" in ppginfo
    assert "pnn50" in ppginfo

    if debug:
        print(f"HRV features present: mean_ibi={ppginfo['mean_ibi']}")


def test_procppg_no_hrv_with_few_peaks(debug=False):
    """Test that HRV features are NOT extracted when <= 5 peaks detected."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(num_peaks=3)  # <= 5

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        result = procppg(args)
        ppginfo = result[0]

        # HRV features should NOT be extracted
        mock_feature_instance.extract_hrv_features.assert_not_called()
        assert "mean_ibi" not in ppginfo

    if debug:
        print("HRV features correctly skipped for few peaks")


def test_procppg_morphology_with_peaks(debug=False):
    """Test that morphology features are extracted when peaks exist."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(num_peaks=10)

    result = _run_procppg_with_mocks(args, mock_data)
    ppginfo = result[0]

    # Morphology features should be merged into ppginfo
    assert "pulse_amplitude" in ppginfo
    assert "rising_time" in ppginfo

    if debug:
        print(f"Morphology: amplitude={ppginfo['pulse_amplitude']}")


def test_procppg_no_morphology_with_zero_peaks(debug=False):
    """Test that morphology is NOT extracted when no peaks detected."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(num_peaks=0)
    # Override peak_indices to be empty
    mock_data["peak_indices"] = np.array([], dtype=int)

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        None,  # hr_waveform_from_peaks is None when no peaks
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        result = procppg(args)
        ppginfo = result[0]

        # Morphology features should NOT be extracted
        mock_feature_instance.extract_morphology_features.assert_not_called()
        assert "pulse_amplitude" not in ppginfo

        # hr_waveform_from_peaks-based stats should NOT be in ppginfo
        assert "mean_peaks_hr" not in ppginfo

    if debug:
        print("Morphology correctly skipped for zero peaks")


def test_procppg_spo2_always_computed(debug=False):
    """Test that SpO2 proxy is always computed regardless of peaks."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg(num_peaks=0)
    mock_data["peak_indices"] = np.array([], dtype=int)

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        None,
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        result = procppg(args)
        ppginfo = result[0]

        mock_feature_instance.compute_spo2_proxy.assert_called_once()
        assert ppginfo["spo2_proxy"] == 97.5

    if debug:
        print("SpO2 proxy always computed")


def test_procppg_read_happy_ppg_called_with_infileroot(debug=False):
    """Test that read_happy_ppg is called with args.infileroot."""
    args = _make_args(infileroot="/data/my_happy_output")
    mock_data = _make_mock_tide_ppg()

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        mock_read.assert_called_once_with("/data/my_happy_output", debug=True)

    if debug:
        print("read_happy_ppg called with correct infileroot")


def test_procppg_signal_normalization(debug=False):
    """Test that dlfiltered and cardiacfromfmri signals are normalized to unit variance."""
    args = _make_args()
    rng = np.random.RandomState(42)
    npoints = 500
    fs = 25.0
    t = np.arange(npoints) / fs

    # Create signals with known std != 1
    dlfiltered = rng.randn(npoints) * 3.0  # std = 3.0
    cardiacfromfmri = rng.randn(npoints) * 5.0  # std = 5.0

    original_dl_std = np.std(dlfiltered)
    original_cf_std = np.std(cardiacfromfmri)
    read_data = (t, fs, dlfiltered.copy(), cardiacfromfmri.copy(), None, [10])

    mock_data = _make_mock_tide_ppg()
    mock_data["read_data"] = read_data

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = read_data
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        # After normalization, the EKF should receive the normalized signal.
        # We verify by checking the signal passed to ekf.filter_signal
        passed_signal = mock_ekf_instance.filter_signal.call_args[0][0]
        np.testing.assert_allclose(np.std(passed_signal), 1.0, atol=1e-10)

    if debug:
        print(f"Original std: dl={original_dl_std:.2f}, cf={original_cf_std:.2f}")
        print("Signals normalized to unit variance")


def test_procppg_debug_flag(debug=False):
    """Test that debug=True doesn't crash and prints extra info."""
    args = _make_args(debug=True)
    mock_data = _make_mock_tide_ppg()

    # Should not raise
    result = _run_procppg_with_mocks(args, mock_data)
    assert result is not None

    if debug:
        print("Debug flag test passed")


def test_procppg_mse_calculation(debug=False):
    """Test that MSE is computed correctly between dlfiltered and filtered_ekf."""
    args = _make_args()
    npoints = 500
    fs = 25.0
    rng = np.random.RandomState(42)
    t = np.arange(npoints) / fs

    # Make dlfiltered and filtered_ekf with known difference
    dlfiltered_raw = np.sin(2 * np.pi * 1.2 * t)
    dl_std = np.std(dlfiltered_raw)
    # After normalization: dlfiltered = dlfiltered_raw / dl_std
    dlfiltered_normalized = dlfiltered_raw / dl_std

    filtered_ekf = dlfiltered_normalized + 0.1  # known offset for MSE

    expected_mse = np.mean((dlfiltered_normalized - filtered_ekf) ** 2)

    mock_data = _make_mock_tide_ppg(npoints=npoints, fs=fs)
    mock_data["read_data"] = (t, fs, dlfiltered_raw.copy(), rng.randn(npoints), None, [])
    mock_data["filtered_ekf"] = filtered_ekf

    result = _run_procppg_with_mocks(args, mock_data)
    ppginfo = result[0]

    np.testing.assert_allclose(ppginfo["mse_ekf"], expected_mse, atol=1e-10)

    if debug:
        print(f"MSE: computed={ppginfo['mse_ekf']:.6f}, expected={expected_mse:.6f}")


def test_procppg_filter_setup(debug=False):
    """Test that NoncausalFilter is created with correct parameters."""
    args = _make_args()
    mock_data = _make_mock_tide_ppg()

    mock_ekf_instance = MagicMock()
    mock_ekf_instance.filter_signal.return_value = (
        mock_data["filtered_ekf"],
        mock_data["ekf_heart_rates"],
    )
    mock_hr_instance = MagicMock()
    mock_hr_instance.extract_continuous.return_value = (
        mock_data["hr_times"],
        mock_data["hr_values"],
    )
    mock_hr_instance.extract_from_peaks.return_value = (
        mock_data["hr_from_peaks"],
        mock_data["peak_indices"],
        mock_data["rri"],
        mock_data["hr_waveform_from_peaks"],
    )
    mock_qa_instance = MagicMock()
    mock_qa_instance.assess_continuous.side_effect = [
        (mock_data["qual_times"], mock_data["qual_scores"]),
        (mock_data["qual_times"], mock_data["qual_scores"]),
    ]
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = mock_data["pipeline_results"]
    mock_feature_instance = MagicMock()
    mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
    mock_feature_instance.extract_morphology_features.return_value = mock_data["morph_features"]
    mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

    with (
        patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter") as mock_ekf_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor") as mock_qa_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
        patch("rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor") as mock_feat_cls,
        patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter") as mock_filt,
        patch("rapidtide.workflows.applyppgproc.plt"),
    ):
        mock_read.return_value = mock_data["read_data"]
        mock_ekf_cls.return_value = mock_ekf_instance
        mock_hr_cls.return_value = mock_hr_instance
        mock_qa_cls.return_value = mock_qa_instance
        mock_proc_cls.return_value = mock_processor_instance
        mock_feat_cls.return_value = mock_feature_instance

        procppg(args)

        # NoncausalFilter should be created with filtertype="arb"
        mock_filt.assert_called_once_with(filtertype="arb")
        # setfreqs should be called with bandpass 0-4 Hz
        mock_filt.return_value.setfreqs.assert_called_once_with(0.0, 0.0, 1.0, 4.0)

    if debug:
        print("Filter setup verified")


def test_procppg_num_detected_peaks(debug=False):
    """Test that num_detected_peaks is correctly set from peak_indices length."""
    args = _make_args()
    for num_peaks in [0, 3, 10, 50]:
        mock_data = _make_mock_tide_ppg(npoints=1000, num_peaks=num_peaks)
        if num_peaks == 0:
            mock_data["peak_indices"] = np.array([], dtype=int)
            mock_data["hr_waveform_from_peaks"] = None

        mock_ekf_instance = MagicMock()
        mock_ekf_instance.filter_signal.return_value = (
            mock_data["filtered_ekf"],
            mock_data["ekf_heart_rates"],
        )
        mock_hr_instance = MagicMock()
        mock_hr_instance.extract_continuous.return_value = (
            mock_data["hr_times"],
            mock_data["hr_values"],
        )
        mock_hr_instance.extract_from_peaks.return_value = (
            mock_data["hr_from_peaks"],
            mock_data["peak_indices"],
            mock_data["rri"],
            mock_data["hr_waveform_from_peaks"],
        )
        mock_qa_instance = MagicMock()
        mock_qa_instance.assess_continuous.side_effect = [
            (mock_data["qual_times"], mock_data["qual_scores"]),
            (mock_data["qual_times"], mock_data["qual_scores"]),
        ]
        mock_processor_instance = MagicMock()
        mock_processor_instance.process.return_value = mock_data["pipeline_results"]
        mock_feature_instance = MagicMock()
        mock_feature_instance.extract_hrv_features.return_value = mock_data["hrv_features"]
        mock_feature_instance.extract_morphology_features.return_value = mock_data[
            "morph_features"
        ]
        mock_feature_instance.compute_spo2_proxy.return_value = mock_data["spo2_proxy"]

        with (
            patch("rapidtide.workflows.applyppgproc.tide_ppg.read_happy_ppg") as mock_read,
            patch(
                "rapidtide.workflows.applyppgproc.tide_ppg.ExtendedPPGKalmanFilter"
            ) as mock_ekf_cls,
            patch("rapidtide.workflows.applyppgproc.tide_ppg.HeartRateExtractor") as mock_hr_cls,
            patch(
                "rapidtide.workflows.applyppgproc.tide_ppg.SignalQualityAssessor"
            ) as mock_qa_cls,
            patch("rapidtide.workflows.applyppgproc.tide_ppg.RobustPPGProcessor") as mock_proc_cls,
            patch(
                "rapidtide.workflows.applyppgproc.tide_ppg.PPGFeatureExtractor"
            ) as mock_feat_cls,
            patch("rapidtide.workflows.applyppgproc.tide_filt.NoncausalFilter"),
            patch("rapidtide.workflows.applyppgproc.plt"),
        ):
            mock_read.return_value = mock_data["read_data"]
            mock_ekf_cls.return_value = mock_ekf_instance
            mock_hr_cls.return_value = mock_hr_instance
            mock_qa_cls.return_value = mock_qa_instance
            mock_proc_cls.return_value = mock_processor_instance
            mock_feat_cls.return_value = mock_feature_instance

            result = procppg(args)
            ppginfo = result[0]
            assert ppginfo["num_detected_peaks"] == num_peaks

    if debug:
        print("num_detected_peaks verified for all peak counts")


# ---- main test entry point ----


def test_applyppgproc(debug=False):
    """Run all applyppgproc sub-tests."""
    # _get_parser tests
    test_get_parser_returns_parser(debug=debug)
    test_get_parser_required_args(debug=debug)
    test_get_parser_with_valid_args(debug=debug)
    test_get_parser_optional_flags(debug=debug)
    test_get_parser_invalid_float(debug=debug)
    test_get_parser_missing_outfileroot(debug=debug)

    # procppg tests
    test_procppg_basic_return_structure(debug=debug)
    test_procppg_ppginfo_keys(debug=debug)
    test_procppg_ppginfo_values_reasonable(debug=debug)
    test_procppg_return_arrays_correct(debug=debug)
    test_procppg_ekf_params_passed(debug=debug)
    test_procppg_hr_extractor_called_correctly(debug=debug)
    test_procppg_quality_assessor_called_twice(debug=debug)
    test_procppg_pipeline_processor_called(debug=debug)
    test_procppg_hrv_features_with_enough_peaks(debug=debug)
    test_procppg_no_hrv_with_few_peaks(debug=debug)
    test_procppg_morphology_with_peaks(debug=debug)
    test_procppg_no_morphology_with_zero_peaks(debug=debug)
    test_procppg_spo2_always_computed(debug=debug)
    test_procppg_read_happy_ppg_called_with_infileroot(debug=debug)
    test_procppg_signal_normalization(debug=debug)
    test_procppg_debug_flag(debug=debug)
    test_procppg_mse_calculation(debug=debug)
    test_procppg_filter_setup(debug=debug)
    test_procppg_num_detected_peaks(debug=debug)


if __name__ == "__main__":
    test_applyppgproc(debug=True)
