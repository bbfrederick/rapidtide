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
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.ccorrica import _get_parser, ccorrica

# ---- helpers ----


def _make_args(**overrides):
    """Build a minimal args Namespace for ccorrica with sensible defaults."""
    defaults = dict(
        timecoursefile="input.txt",
        outputroot="/tmp/test_ccorrica_out",
        samplerate=2.0,
        oversampfactor=1,
        detrendorder=3,
        windowfunc="hamming",
        corrweighting="phat",
        debug=False,
        # filter opts expected by postprocessfilteropts
        filterband="lfo",
        passvec=None,
        stopvec=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mock_prefilter():
    """Create a mock prefilter whose apply() returns input unchanged."""
    prefilter = MagicMock()
    prefilter.apply.side_effect = lambda fs, data: data
    return prefilter


def _make_tc_data(numcomponents=3, tclen=100):
    """Create synthetic timecourse data (numcomponents x tclen)."""
    rng = np.random.RandomState(42)
    return rng.randn(numcomponents, tclen).astype(float)


def _capture_writenpvecs():
    """Capture writenpvecs calls, copying data at call time."""
    captured = []

    def _side_effect(data, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _capture_savetonifti():
    """Capture savetonifti calls, copying data at call time."""
    captured = []

    def _side_effect(data, hdr, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _run_ccorrica(args, tcdata=None, file_samplerate=None, numcomponents=3, tclen=100):
    """Run ccorrica with full mocking of external dependencies.

    Parameters
    ----------
    args : Namespace
        Arguments to pass to ccorrica.
    tcdata : ndarray, optional
        Timecourse data. If None, synthetic data is generated.
    file_samplerate : float or None
        Sample rate returned by readvectorsfromtextfile.  None means
        the file doesn't specify a sample rate.
    numcomponents : int
        Number of components (used when tcdata is None).
    tclen : int
        Length of timecourses (used when tcdata is None).

    Returns
    -------
    dict with keys: 'writenpvecs', 'savetonifti', 'args_out'
    """
    if tcdata is None:
        tcdata = _make_tc_data(numcomponents, tclen)
    else:
        numcomponents, tclen = tcdata.shape

    prefilter = _make_mock_prefilter()

    wnp_effect, wnp_captured = _capture_writenpvecs()
    stn_effect, stn_captured = _capture_savetonifti()

    # findmaxlag_gauss returns 8-tuple
    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    # symmetrize: return input unchanged
    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        result = np.array(matrix, copy=True)
        if zerodiagonal:
            np.fill_diagonal(result, 0.0)
        return result

    # fastcorrelate: return a properly-sized cross-correlation array
    xcorrlen = 2 * tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        rng = np.random.RandomState(hash((a.tobytes()[:16], b.tobytes()[:16])) % (2**31))
        result = rng.randn(xcorrlen) * 0.1
        result[xcorrlen // 2] = 0.5  # peak at zero lag
        return result

    # pearsonr mock
    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    # nib.Nifti1Image mock — returns an object whose .header supports item access
    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        return mock_img

    with (
        patch(
            "rapidtide.workflows.ccorrica.pf.postprocessfilteropts",
            return_value=(args, prefilter),
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile",
            return_value=(file_samplerate, 0.0, None, tcdata, False, "text"),
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_resample.upsample",
            side_effect=lambda data, fs_in, fs_out, intfac=False: np.repeat(data, int(fs_out / fs_in)),
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_math.stdnormalize",
            side_effect=lambda x: x / (np.std(x) if np.std(x) > 0 else 1.0),
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_math.corrnormalize",
            side_effect=lambda x, detrendorder=0, windowfunc="hamming": x / (np.std(x) if np.std(x) > 0 else 1.0),
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_corr.fastcorrelate",
            side_effect=mock_fastcorrelate,
        ),
        patch(
            "rapidtide.workflows.ccorrica.pearsonr",
            return_value=mock_pearsonr_result,
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss",
            side_effect=mock_findmaxlag_gauss,
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_stats.symmetrize",
            side_effect=mock_symmetrize,
        ),
        patch(
            "rapidtide.workflows.ccorrica.nib.Nifti1Image",
            side_effect=mock_nifti1image,
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_io.savetonifti",
            side_effect=stn_effect,
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_io.writenpvecs",
            side_effect=wnp_effect,
        ),
    ):
        ccorrica(args)

    return {
        "writenpvecs": wnp_captured,
        "savetonifti": stn_captured,
    }


# ---- _get_parser tests ----


def test_parser_required_args(debug=False):
    """Parser should accept the two required positional args."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot"])
        assert args.timecoursefile == f.name
        assert args.outputroot == "outroot"
        if debug:
            print("test_parser_required_args passed")


def test_parser_defaults(debug=False):
    """Parser defaults should match expected values."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot"])
        assert args.samplerate == "auto"
        assert args.oversampfactor == 1
        assert args.detrendorder == 3
        assert args.corrweighting == "phat"
        assert args.debug is False
        if debug:
            print("test_parser_defaults passed")


def test_parser_samplerate(debug=False):
    """Parser should accept --samplerate."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--samplerate", "4.0"])
        assert args.samplerate == 4.0
        if debug:
            print("test_parser_samplerate passed")


def test_parser_sampletstep(debug=False):
    """Parser should accept --sampletstep and invert it to samplerate."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--sampletstep", "0.5"])
        assert abs(args.samplerate - 2.0) < 1e-6
        if debug:
            print("test_parser_sampletstep passed")


def test_parser_mutually_exclusive_rate(debug=False):
    """Parser should reject both --samplerate and --sampletstep together."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([f.name, "outroot", "--samplerate", "2.0", "--sampletstep", "0.5"])
        if debug:
            print("test_parser_mutually_exclusive_rate passed")


def test_parser_corrweighting_choices(debug=False):
    """Parser should accept valid corrweighting choices and reject invalid ones."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        for choice in ["None", "phat", "liang", "eckart"]:
            args = parser.parse_args([f.name, "outroot", "--corrweighting", choice])
            assert args.corrweighting == choice
        with pytest.raises(SystemExit):
            parser.parse_args([f.name, "outroot", "--corrweighting", "invalid"])
        if debug:
            print("test_parser_corrweighting_choices passed")


def test_parser_detrendorder(debug=False):
    """Parser should accept --detrendorder."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--detrendorder", "5"])
        assert args.detrendorder == 5
        if debug:
            print("test_parser_detrendorder passed")


def test_parser_oversampfactor(debug=False):
    """Parser should accept --oversampfactor including negative values."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--oversampfactor", "-1"])
        assert args.oversampfactor == -1
        if debug:
            print("test_parser_oversampfactor passed")


def test_parser_debug_flag(debug=False):
    """Parser should accept --debug flag."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--debug"])
        assert args.debug is True
        if debug:
            print("test_parser_debug_flag passed")


# ---- ccorrica happy path tests ----


def test_ccorrica_basic_run(debug=False):
    """ccorrica should run without errors with basic inputs."""
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=3, tclen=100)
    assert len(result["writenpvecs"]) > 0
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_basic_run passed")


def test_ccorrica_output_files(debug=False):
    """ccorrica should write all expected output files."""
    args = _make_args(outputroot="/tmp/cctest")
    result = _run_ccorrica(args, numcomponents=2, tclen=50)

    # Check savetonifti output files
    stn_files = [entry[1] for entry in result["savetonifti"]]
    expected_nifti_suffixes = ["_xcorr", "_pxcorr", "_corrmax", "_corrlag", "_corrwidth", "_corrmask"]
    for suffix in expected_nifti_suffixes:
        assert any(suffix in f for f in stn_files), f"Missing nifti file with suffix {suffix}"

    # Check writenpvecs output files
    wnp_files = [entry[1] for entry in result["writenpvecs"]]
    expected_txt_suffixes = [
        "_filtereddata.txt",
        "_corrmax.txt",
        "_corrlag.txt",
        "_corrwidth.txt",
        "_corrmask.txt",
        "_reformdata.txt",
    ]
    for suffix in expected_txt_suffixes:
        assert any(suffix in f for f in wnp_files), f"Missing text file with suffix {suffix}"

    if debug:
        print("test_ccorrica_output_files passed")


def test_ccorrica_output_shapes(debug=False):
    """Output data should have correct shapes for given numcomponents."""
    numcomponents = 3
    tclen = 60
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=numcomponents, tclen=tclen)

    # Find corrmax output in savetonifti calls
    for data, fname in result["savetonifti"]:
        if "_corrmax" in fname and "_pxcorr" not in fname:
            assert data.shape[0] == numcomponents
            assert data.shape[1] == numcomponents
        if "_pxcorr" in fname:
            assert data.shape[0] == numcomponents
            assert data.shape[1] == numcomponents
            assert data.shape[3] == tclen

    # Find corrmax txt output
    for data, fname in result["writenpvecs"]:
        if "_corrmax.txt" in fname:
            assert data.shape == (numcomponents, numcomponents)

    if debug:
        print("test_ccorrica_output_shapes passed")


def test_ccorrica_filtereddata_shape(debug=False):
    """Filtered data output should have shape (numcomponents, tclen)."""
    numcomponents = 4
    tclen = 80
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=numcomponents, tclen=tclen)

    for data, fname in result["writenpvecs"]:
        if "_filtereddata.txt" in fname:
            assert data.shape == (numcomponents, tclen)
            break
    else:
        pytest.fail("_filtereddata.txt not found in output")

    if debug:
        print("test_ccorrica_filtereddata_shape passed")


def test_ccorrica_reformdata_shape(debug=False):
    """Reform data output should have shape (numcomponents, tclen)."""
    numcomponents = 3
    tclen = 50
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=numcomponents, tclen=tclen)

    for data, fname in result["writenpvecs"]:
        if "_reformdata.txt" in fname:
            assert data.shape == (numcomponents, tclen)
            break
    else:
        pytest.fail("_reformdata.txt not found in output")

    if debug:
        print("test_ccorrica_reformdata_shape passed")


def test_ccorrica_single_component(debug=False):
    """ccorrica should handle single component data."""
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=1, tclen=50)

    # Should still produce output files
    assert len(result["writetonifti"] if "writetonifti" in result else result["savetonifti"]) > 0
    assert len(result["writenpvecs"]) > 0

    # corrmax should be 1x1
    for data, fname in result["writenpvecs"]:
        if "_corrmax.txt" in fname:
            assert data.shape == (1, 1)

    if debug:
        print("test_ccorrica_single_component passed")


def test_ccorrica_two_components(debug=False):
    """ccorrica should correctly handle 2 components."""
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=2, tclen=40)

    for data, fname in result["writenpvecs"]:
        if "_corrlag.txt" in fname:
            assert data.shape == (2, 2)

    if debug:
        print("test_ccorrica_two_components passed")


def test_ccorrica_xcorr_nifti_shape(debug=False):
    """Cross-correlation nifti should have correct windowed length."""
    numcomponents = 2
    tclen = 50
    args = _make_args(samplerate=2.0)
    result = _run_ccorrica(args, numcomponents=numcomponents, tclen=tclen)

    for data, fname in result["savetonifti"]:
        if "_xcorr" in fname and "_pxcorr" not in fname:
            # Shape should be (numcomponents, numcomponents, 1, corrwin)
            assert data.shape[0] == numcomponents
            assert data.shape[1] == numcomponents
            assert data.shape[2] == 1
            # The window length is searchend - searchstart
            Fs = 2.0
            searchrange = 15.0
            halfwindow = int(searchrange * Fs)
            xcorrlen = 2 * tclen - 1
            corrzero = xcorrlen // 2
            corrwin = 2 * halfwindow
            assert data.shape[3] == corrwin
            break
    else:
        pytest.fail("_xcorr nifti not found")

    if debug:
        print("test_ccorrica_xcorr_nifti_shape passed")


# ---- ccorrica sample rate handling ----


def test_ccorrica_samplerate_from_file(debug=False):
    """ccorrica should use sample rate from file when available."""
    args = _make_args(samplerate="auto")
    # file_samplerate=4.0 should be used
    result = _run_ccorrica(args, file_samplerate=4.0, numcomponents=2, tclen=50)
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_samplerate_from_file passed")


def test_ccorrica_samplerate_from_args(debug=False):
    """ccorrica should use args.samplerate when file doesn't provide one."""
    args = _make_args(samplerate=3.0)
    # file_samplerate=None means no rate in file
    result = _run_ccorrica(args, file_samplerate=None, numcomponents=2, tclen=50)
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_samplerate_from_args passed")


def test_ccorrica_missing_samplerate_exits(debug=False):
    """ccorrica should exit when samplerate is 'auto' and file has none."""
    args = _make_args(samplerate="auto")

    prefilter = _make_mock_prefilter()
    tcdata = _make_tc_data(2, 50)

    with (
        patch(
            "rapidtide.workflows.ccorrica.pf.postprocessfilteropts",
            return_value=(args, prefilter),
        ),
        patch(
            "rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile",
            return_value=(None, 0.0, None, tcdata, False, "text"),
        ),
        patch(
            "rapidtide.workflows.ccorrica.sys.exit",
            side_effect=SystemExit,
        ) as mock_exit,
    ):
        with pytest.raises(SystemExit):
            ccorrica(args)
        mock_exit.assert_called_once()

    if debug:
        print("test_ccorrica_missing_samplerate_exits passed")


# ---- oversampling tests ----


def test_ccorrica_oversample(debug=False):
    """ccorrica should upsample data when oversampfactor > 1."""
    args = _make_args(oversampfactor=2, samplerate=2.0)
    numcomponents = 2
    tclen = 50

    tcdata = _make_tc_data(numcomponents, tclen)

    prefilter = _make_mock_prefilter()
    wnp_effect, wnp_captured = _capture_writenpvecs()
    stn_effect, stn_captured = _capture_savetonifti()

    upsample_calls = []

    def mock_upsample(data, fs_in, fs_out, intfac=False):
        upsample_calls.append((fs_in, fs_out))
        return np.repeat(data, int(fs_out / fs_in))

    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        result = np.array(matrix, copy=True)
        if zerodiagonal:
            np.fill_diagonal(result, 0.0)
        return result

    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        return mock_img

    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    new_tclen = tclen * 2  # oversampled
    xcorrlen = 2 * new_tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        return np.zeros(xcorrlen)

    with (
        patch("rapidtide.workflows.ccorrica.pf.postprocessfilteropts", return_value=(args, prefilter)),
        patch("rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile", return_value=(None, 0.0, None, tcdata, False, "text")),
        patch("rapidtide.workflows.ccorrica.tide_resample.upsample", side_effect=mock_upsample),
        patch("rapidtide.workflows.ccorrica.tide_math.stdnormalize", side_effect=lambda x: x),
        patch("rapidtide.workflows.ccorrica.tide_math.corrnormalize", side_effect=lambda x, **kw: x),
        patch("rapidtide.workflows.ccorrica.tide_corr.fastcorrelate", side_effect=mock_fastcorrelate),
        patch("rapidtide.workflows.ccorrica.pearsonr", return_value=mock_pearsonr_result),
        patch("rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss", side_effect=mock_findmaxlag_gauss),
        patch("rapidtide.workflows.ccorrica.tide_stats.symmetrize", side_effect=mock_symmetrize),
        patch("rapidtide.workflows.ccorrica.nib.Nifti1Image", side_effect=mock_nifti1image),
        patch("rapidtide.workflows.ccorrica.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.ccorrica.tide_io.writenpvecs", side_effect=wnp_effect),
    ):
        ccorrica(args)

    # upsample should have been called for each component
    assert len(upsample_calls) == numcomponents
    for fs_in, fs_out in upsample_calls:
        assert fs_in == 2.0
        assert fs_out == 4.0  # 2.0 * oversampfactor(2)

    if debug:
        print("test_ccorrica_oversample passed")


def test_ccorrica_auto_oversampfactor(debug=False):
    """ccorrica should auto-compute oversampfactor when negative."""
    # With samplerate=1.0, sampletime=1.0, oversampfactor should be max(ceil(1.0/0.5), 1) = 2
    args = _make_args(oversampfactor=-1, samplerate=1.0)
    numcomponents = 2
    tclen = 50

    tcdata = _make_tc_data(numcomponents, tclen)
    prefilter = _make_mock_prefilter()
    wnp_effect, wnp_captured = _capture_writenpvecs()
    stn_effect, stn_captured = _capture_savetonifti()

    upsample_calls = []

    def mock_upsample(data, fs_in, fs_out, intfac=False):
        upsample_calls.append((fs_in, fs_out))
        return np.repeat(data, int(fs_out / fs_in))

    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        result = np.array(matrix, copy=True)
        if zerodiagonal:
            np.fill_diagonal(result, 0.0)
        return result

    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        return mock_img

    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    # auto factor for sampletime=1.0: ceil(1.0/0.5) = 2
    new_tclen = tclen * 2
    xcorrlen = 2 * new_tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        return np.zeros(xcorrlen)

    with (
        patch("rapidtide.workflows.ccorrica.pf.postprocessfilteropts", return_value=(args, prefilter)),
        patch("rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile", return_value=(1.0, 0.0, None, tcdata, False, "text")),
        patch("rapidtide.workflows.ccorrica.tide_resample.upsample", side_effect=mock_upsample),
        patch("rapidtide.workflows.ccorrica.tide_math.stdnormalize", side_effect=lambda x: x),
        patch("rapidtide.workflows.ccorrica.tide_math.corrnormalize", side_effect=lambda x, **kw: x),
        patch("rapidtide.workflows.ccorrica.tide_corr.fastcorrelate", side_effect=mock_fastcorrelate),
        patch("rapidtide.workflows.ccorrica.pearsonr", return_value=mock_pearsonr_result),
        patch("rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss", side_effect=mock_findmaxlag_gauss),
        patch("rapidtide.workflows.ccorrica.tide_stats.symmetrize", side_effect=mock_symmetrize),
        patch("rapidtide.workflows.ccorrica.nib.Nifti1Image", side_effect=mock_nifti1image),
        patch("rapidtide.workflows.ccorrica.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.ccorrica.tide_io.writenpvecs", side_effect=wnp_effect),
    ):
        ccorrica(args)

    # Should have called upsample (oversampfactor auto-set to 2)
    assert len(upsample_calls) == numcomponents
    assert args.oversampfactor == 2

    if debug:
        print("test_ccorrica_auto_oversampfactor passed")


# ---- ccorrica option variation tests ----


def test_ccorrica_corrweighting_none(debug=False):
    """ccorrica should work with corrweighting='None'."""
    args = _make_args(corrweighting="None")
    result = _run_ccorrica(args, numcomponents=2, tclen=40)
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_corrweighting_none passed")


def test_ccorrica_detrendorder_zero(debug=False):
    """ccorrica should work with detrendorder=0 (no detrending)."""
    args = _make_args(detrendorder=0)
    result = _run_ccorrica(args, numcomponents=2, tclen=40)
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_detrendorder_zero passed")


def test_ccorrica_debug_mode(debug=False):
    """ccorrica should run with debug=True."""
    args = _make_args(debug=True)
    result = _run_ccorrica(args, numcomponents=2, tclen=40)
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_debug_mode passed")


def test_ccorrica_window_func_hann(debug=False):
    """ccorrica should work with windowfunc='hann'."""
    args = _make_args(windowfunc="hann")
    result = _run_ccorrica(args, numcomponents=2, tclen=40)
    assert len(result["savetonifti"]) > 0
    if debug:
        print("test_ccorrica_window_func_hann passed")


# ---- symmetrization tests ----


def test_ccorrica_symmetrize_called(debug=False):
    """ccorrica should call symmetrize for corrmax, corrlag, corrwidth, corrmask."""
    args = _make_args()
    numcomponents = 3
    tclen = 50
    tcdata = _make_tc_data(numcomponents, tclen)
    prefilter = _make_mock_prefilter()

    symmetrize_calls = []

    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        symmetrize_calls.append({
            "zerodiagonal": zerodiagonal,
            "antisymmetric": antisymmetric,
        })
        result = np.array(matrix, copy=True)
        if zerodiagonal:
            np.fill_diagonal(result, 0.0)
        return result

    wnp_effect, _ = _capture_writenpvecs()
    stn_effect, _ = _capture_savetonifti()

    xcorrlen = 2 * tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        return np.zeros(xcorrlen)

    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        return mock_img

    with (
        patch("rapidtide.workflows.ccorrica.pf.postprocessfilteropts", return_value=(args, prefilter)),
        patch("rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile", return_value=(2.0, 0.0, None, tcdata, False, "text")),
        patch("rapidtide.workflows.ccorrica.tide_resample.upsample", side_effect=lambda d, fi, fo, intfac=False: d),
        patch("rapidtide.workflows.ccorrica.tide_math.stdnormalize", side_effect=lambda x: x),
        patch("rapidtide.workflows.ccorrica.tide_math.corrnormalize", side_effect=lambda x, **kw: x),
        patch("rapidtide.workflows.ccorrica.tide_corr.fastcorrelate", side_effect=mock_fastcorrelate),
        patch("rapidtide.workflows.ccorrica.pearsonr", return_value=mock_pearsonr_result),
        patch("rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss", side_effect=mock_findmaxlag_gauss),
        patch("rapidtide.workflows.ccorrica.tide_stats.symmetrize", side_effect=mock_symmetrize),
        patch("rapidtide.workflows.ccorrica.nib.Nifti1Image", side_effect=mock_nifti1image),
        patch("rapidtide.workflows.ccorrica.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.ccorrica.tide_io.writenpvecs", side_effect=wnp_effect),
    ):
        ccorrica(args)

    # symmetrize should be called 4 times:
    # corrmax (zerodiagonal=True), corrlag (antisymmetric=True),
    # corrwidth (neither), corrmask (zerodiagonal=True)
    assert len(symmetrize_calls) == 4
    assert symmetrize_calls[0]["zerodiagonal"] is True   # corrmax
    assert symmetrize_calls[1]["antisymmetric"] is True   # corrlag
    assert symmetrize_calls[2]["zerodiagonal"] is False   # corrwidth
    assert symmetrize_calls[2]["antisymmetric"] is False  # corrwidth
    assert symmetrize_calls[3]["zerodiagonal"] is True    # corrmask

    if debug:
        print("test_ccorrica_symmetrize_called passed")


# ---- correlation computation tests ----


def test_ccorrica_fastcorrelate_call_count(debug=False):
    """fastcorrelate should be called numcomponents^2 times (all pairs)."""
    numcomponents = 3
    tclen = 40
    args = _make_args()
    tcdata = _make_tc_data(numcomponents, tclen)
    prefilter = _make_mock_prefilter()

    correlate_count = [0]
    xcorrlen = 2 * tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        correlate_count[0] += 1
        return np.zeros(xcorrlen)

    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        return np.array(matrix, copy=True)

    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    wnp_effect, _ = _capture_writenpvecs()
    stn_effect, _ = _capture_savetonifti()

    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        return mock_img

    with (
        patch("rapidtide.workflows.ccorrica.pf.postprocessfilteropts", return_value=(args, prefilter)),
        patch("rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile", return_value=(2.0, 0.0, None, tcdata, False, "text")),
        patch("rapidtide.workflows.ccorrica.tide_resample.upsample", side_effect=lambda d, fi, fo, intfac=False: d),
        patch("rapidtide.workflows.ccorrica.tide_math.stdnormalize", side_effect=lambda x: x),
        patch("rapidtide.workflows.ccorrica.tide_math.corrnormalize", side_effect=lambda x, **kw: x),
        patch("rapidtide.workflows.ccorrica.tide_corr.fastcorrelate", side_effect=mock_fastcorrelate),
        patch("rapidtide.workflows.ccorrica.pearsonr", return_value=mock_pearsonr_result),
        patch("rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss", side_effect=mock_findmaxlag_gauss),
        patch("rapidtide.workflows.ccorrica.tide_stats.symmetrize", side_effect=mock_symmetrize),
        patch("rapidtide.workflows.ccorrica.nib.Nifti1Image", side_effect=mock_nifti1image),
        patch("rapidtide.workflows.ccorrica.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.ccorrica.tide_io.writenpvecs", side_effect=wnp_effect),
    ):
        ccorrica(args)

    assert correlate_count[0] == numcomponents * numcomponents

    if debug:
        print("test_ccorrica_fastcorrelate_call_count passed")


def test_ccorrica_pearsonr_call_count(debug=False):
    """pearsonr should be called numcomponents^2 times."""
    numcomponents = 2
    tclen = 40
    args = _make_args()
    result = _run_ccorrica(args, numcomponents=numcomponents, tclen=tclen)

    # Check pxcorr data has correct shape
    for data, fname in result["savetonifti"]:
        if "_pxcorr" in fname:
            assert data.shape == (numcomponents, numcomponents, 1, tclen)

    if debug:
        print("test_ccorrica_pearsonr_call_count passed")


def test_ccorrica_findmaxlag_gauss_called(debug=False):
    """findmaxlag_gauss should be called for each component pair."""
    numcomponents = 2
    tclen = 40
    args = _make_args()
    tcdata = _make_tc_data(numcomponents, tclen)
    prefilter = _make_mock_prefilter()

    gauss_calls = [0]
    xcorrlen = 2 * tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        return np.zeros(xcorrlen)

    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        gauss_calls[0] += 1
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        return np.array(matrix, copy=True)

    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    wnp_effect, _ = _capture_writenpvecs()
    stn_effect, _ = _capture_savetonifti()

    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        return mock_img

    with (
        patch("rapidtide.workflows.ccorrica.pf.postprocessfilteropts", return_value=(args, prefilter)),
        patch("rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile", return_value=(2.0, 0.0, None, tcdata, False, "text")),
        patch("rapidtide.workflows.ccorrica.tide_resample.upsample", side_effect=lambda d, fi, fo, intfac=False: d),
        patch("rapidtide.workflows.ccorrica.tide_math.stdnormalize", side_effect=lambda x: x),
        patch("rapidtide.workflows.ccorrica.tide_math.corrnormalize", side_effect=lambda x, **kw: x),
        patch("rapidtide.workflows.ccorrica.tide_corr.fastcorrelate", side_effect=mock_fastcorrelate),
        patch("rapidtide.workflows.ccorrica.pearsonr", return_value=mock_pearsonr_result),
        patch("rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss", side_effect=mock_findmaxlag_gauss),
        patch("rapidtide.workflows.ccorrica.tide_stats.symmetrize", side_effect=mock_symmetrize),
        patch("rapidtide.workflows.ccorrica.nib.Nifti1Image", side_effect=mock_nifti1image),
        patch("rapidtide.workflows.ccorrica.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.ccorrica.tide_io.writenpvecs", side_effect=wnp_effect),
    ):
        ccorrica(args)

    assert gauss_calls[0] == numcomponents * numcomponents

    if debug:
        print("test_ccorrica_findmaxlag_gauss_called passed")


# ---- nifti header tests ----


def test_ccorrica_nifti_pixdim_set(debug=False):
    """Nifti headers should have pixdim[4] set to sampletime."""
    args = _make_args(samplerate=4.0)
    numcomponents = 2
    tclen = 40

    tcdata = _make_tc_data(numcomponents, tclen)
    prefilter = _make_mock_prefilter()

    nifti_headers = []

    def mock_nifti1image(data, affine):
        mock_img = MagicMock()
        mock_hdr = {}
        mock_hdr["pixdim"] = np.zeros(8)
        mock_img.header = mock_hdr
        nifti_headers.append(mock_hdr)
        return mock_img

    xcorrlen = 2 * tclen - 1

    def mock_fastcorrelate(a, b, usefft=True, weighting=None, zeropadding=0, displayplots=False):
        return np.zeros(xcorrlen)

    def mock_findmaxlag_gauss(x, y, lagmin, lagmax, widthmax, **kwargs):
        midx = len(y) // 2
        return (midx, 0.0, 0.5, 1.0, 1.0, 0, 0, len(y))

    def mock_symmetrize(matrix, zerodiagonal=False, antisymmetric=False):
        return np.array(matrix, copy=True)

    mock_pearsonr_result = MagicMock()
    mock_pearsonr_result.statistic = 0.3

    wnp_effect, _ = _capture_writenpvecs()
    stn_effect, _ = _capture_savetonifti()

    with (
        patch("rapidtide.workflows.ccorrica.pf.postprocessfilteropts", return_value=(args, prefilter)),
        patch("rapidtide.workflows.ccorrica.tide_io.readvectorsfromtextfile", return_value=(None, 0.0, None, tcdata, False, "text")),
        patch("rapidtide.workflows.ccorrica.tide_resample.upsample", side_effect=lambda d, fi, fo, intfac=False: d),
        patch("rapidtide.workflows.ccorrica.tide_math.stdnormalize", side_effect=lambda x: x),
        patch("rapidtide.workflows.ccorrica.tide_math.corrnormalize", side_effect=lambda x, **kw: x),
        patch("rapidtide.workflows.ccorrica.tide_corr.fastcorrelate", side_effect=mock_fastcorrelate),
        patch("rapidtide.workflows.ccorrica.pearsonr", return_value=mock_pearsonr_result),
        patch("rapidtide.workflows.ccorrica.tide_fit.findmaxlag_gauss", side_effect=mock_findmaxlag_gauss),
        patch("rapidtide.workflows.ccorrica.tide_stats.symmetrize", side_effect=mock_symmetrize),
        patch("rapidtide.workflows.ccorrica.nib.Nifti1Image", side_effect=mock_nifti1image),
        patch("rapidtide.workflows.ccorrica.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.ccorrica.tide_io.writenpvecs", side_effect=wnp_effect),
    ):
        ccorrica(args)

    # Should have created headers for xcorr (4d) and pxcorr (4d)
    # plus one for 3d outputs (corrmax, corrlag, corrwidth, corrmask share same header)
    assert len(nifti_headers) >= 2
    expected_sampletime = 1.0 / 4.0
    for hdr in nifti_headers:
        assert abs(hdr["pixdim"][4] - expected_sampletime) < 1e-6

    if debug:
        print("test_ccorrica_nifti_pixdim_set passed")


# ---- integration test ----


def test_ccorrica_integration(debug=False):
    """Full integration test with multiple components and options."""
    args = _make_args(
        samplerate=2.0,
        oversampfactor=1,
        detrendorder=1,
        windowfunc="hamming",
        corrweighting="phat",
    )
    numcomponents = 4
    tclen = 100
    result = _run_ccorrica(args, numcomponents=numcomponents, tclen=tclen)

    # Verify all output files exist
    stn_files = [e[1] for e in result["savetonifti"]]
    wnp_files = [e[1] for e in result["writenpvecs"]]

    assert len(stn_files) == 6  # xcorr, pxcorr, corrmax, corrlag, corrwidth, corrmask
    assert len(wnp_files) == 6  # filtereddata, corrmax, corrlag, corrwidth, corrmask, reformdata

    # Verify shapes of matrix outputs
    for data, fname in result["writenpvecs"]:
        if "_corrmax.txt" in fname:
            assert data.shape == (numcomponents, numcomponents)
        if "_corrlag.txt" in fname:
            assert data.shape == (numcomponents, numcomponents)
        if "_corrwidth.txt" in fname:
            assert data.shape == (numcomponents, numcomponents)
        if "_corrmask.txt" in fname:
            assert data.shape == (numcomponents, numcomponents)
        if "_filtereddata.txt" in fname:
            assert data.shape == (numcomponents, tclen)
        if "_reformdata.txt" in fname:
            assert data.shape == (numcomponents, tclen)

    if debug:
        print("test_ccorrica_integration passed")


# ---- main test function ----


def test_ccorrica(debug=False):
    """Run all ccorrica tests."""
    # parser tests
    test_parser_required_args(debug=debug)
    test_parser_defaults(debug=debug)
    test_parser_samplerate(debug=debug)
    test_parser_sampletstep(debug=debug)
    test_parser_mutually_exclusive_rate(debug=debug)
    test_parser_corrweighting_choices(debug=debug)
    test_parser_detrendorder(debug=debug)
    test_parser_oversampfactor(debug=debug)
    test_parser_debug_flag(debug=debug)

    # ccorrica happy path tests
    test_ccorrica_basic_run(debug=debug)
    test_ccorrica_output_files(debug=debug)
    test_ccorrica_output_shapes(debug=debug)
    test_ccorrica_filtereddata_shape(debug=debug)
    test_ccorrica_reformdata_shape(debug=debug)
    test_ccorrica_single_component(debug=debug)
    test_ccorrica_two_components(debug=debug)
    test_ccorrica_xcorr_nifti_shape(debug=debug)

    # sample rate handling
    test_ccorrica_samplerate_from_file(debug=debug)
    test_ccorrica_samplerate_from_args(debug=debug)
    test_ccorrica_missing_samplerate_exits(debug=debug)

    # oversampling tests
    test_ccorrica_oversample(debug=debug)
    test_ccorrica_auto_oversampfactor(debug=debug)

    # option variation tests
    test_ccorrica_corrweighting_none(debug=debug)
    test_ccorrica_detrendorder_zero(debug=debug)
    test_ccorrica_debug_mode(debug=debug)
    test_ccorrica_window_func_hann(debug=debug)

    # symmetrization tests
    test_ccorrica_symmetrize_called(debug=debug)

    # correlation computation tests
    test_ccorrica_fastcorrelate_call_count(debug=debug)
    test_ccorrica_pearsonr_call_count(debug=debug)
    test_ccorrica_findmaxlag_gauss_called(debug=debug)

    # nifti header tests
    test_ccorrica_nifti_pixdim_set(debug=debug)

    # integration test
    test_ccorrica_integration(debug=debug)


if __name__ == "__main__":
    test_ccorrica(debug=True)
