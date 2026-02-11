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
import os
import tempfile

import nibabel as nib
import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.workflows.localflow as lf
import rapidtide.workflows.parser_funcs as pf

# ==================== Helpers ====================


def _make_broadband_signal(timepoints, Fs, delay=0.0, seed=42):
    """Generate a broadband signal as a sum of sinusoids."""
    rng = np.random.RandomState(seed)
    t = np.arange(timepoints) / Fs
    signal = np.zeros(timepoints, dtype=float)
    for _ in range(30):
        freq = rng.uniform(0.01, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * (t - delay) + phase)
    return signal


def _make_4d_nifti(tmpdir, shape=(6, 6, 4, 50), tr=2.0, name="test4d"):
    """Create a 4D NIfTI file with correlated broadband signals and return path."""
    nx, ny, nz, nt = shape
    Fs = 1.0 / tr
    rng = np.random.RandomState(42)
    data = np.zeros(shape, dtype=np.float64)

    # Fill with broadband signals + noise so that mask and correlations are meaningful
    base_signal = _make_broadband_signal(nt, Fs, delay=0.0, seed=42)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                delay = 0.1 * (x + y + z)
                sig = _make_broadband_signal(nt, Fs, delay=delay, seed=x * 100 + y * 10 + z)
                data[x, y, z, :] = sig + 0.5 * rng.randn(nt)

    # Add a base offset so the mask works (makemask uses threshpct)
    data += 100.0

    img = nib.Nifti1Image(data, affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    hdr = img.header
    hdr.set_zooms((2.0, 2.0, 2.0, tr))
    hdr.set_xyzt_units("mm", "sec")
    filepath = os.path.join(tmpdir, name + ".nii.gz")
    nib.save(img, filepath)
    return filepath, data


# ==================== xyz2index tests ====================


def xyz2index_basic(debug=False):
    """Test xyz2index with basic coordinates."""
    if debug:
        print("xyz2index_basic")
    # For a (10, 10, 10) array, index = z + y*zsize + x*zsize*ysize
    result = lf.xyz2index(0, 0, 0, 10, 10, 10)
    assert result == 0

    result = lf.xyz2index(0, 0, 1, 10, 10, 10)
    assert result == 1

    result = lf.xyz2index(0, 1, 0, 10, 10, 10)
    assert result == 10

    result = lf.xyz2index(1, 0, 0, 10, 10, 10)
    assert result == 100


def xyz2index_corners(debug=False):
    """Test xyz2index with corner coordinates."""
    if debug:
        print("xyz2index_corners")
    xsize, ysize, zsize = 5, 7, 3
    # Last valid index
    result = lf.xyz2index(4, 6, 2, xsize, ysize, zsize)
    assert result == 2 + 6 * 3 + 4 * 3 * 7
    assert result == xsize * ysize * zsize - 1


def xyz2index_out_of_bounds(debug=False):
    """Test xyz2index returns -1 for out-of-bounds coordinates."""
    if debug:
        print("xyz2index_out_of_bounds")
    assert lf.xyz2index(-1, 0, 0, 10, 10, 10) == -1
    assert lf.xyz2index(0, -1, 0, 10, 10, 10) == -1
    assert lf.xyz2index(0, 0, -1, 10, 10, 10) == -1
    assert lf.xyz2index(10, 0, 0, 10, 10, 10) == -1
    assert lf.xyz2index(0, 10, 0, 10, 10, 10) == -1
    assert lf.xyz2index(0, 0, 10, 10, 10, 10) == -1


def xyz2index_boundary(debug=False):
    """Test xyz2index at boundary values."""
    if debug:
        print("xyz2index_boundary")
    # Last valid coordinate on each axis
    assert lf.xyz2index(9, 0, 0, 10, 10, 10) >= 0
    assert lf.xyz2index(0, 9, 0, 10, 10, 10) >= 0
    assert lf.xyz2index(0, 0, 9, 10, 10, 10) >= 0
    # One past the end
    assert lf.xyz2index(10, 0, 0, 10, 10, 10) == -1
    assert lf.xyz2index(0, 10, 0, 10, 10, 10) == -1
    assert lf.xyz2index(0, 0, 10, 10, 10, 10) == -1


def xyz2index_nonsquare(debug=False):
    """Test xyz2index with non-square dimensions."""
    if debug:
        print("xyz2index_nonsquare")
    xsize, ysize, zsize = 3, 5, 7
    result = lf.xyz2index(2, 4, 6, xsize, ysize, zsize)
    expected = 6 + 4 * 7 + 2 * 7 * 5
    assert result == expected


# ==================== index2xyz tests ====================


def index2xyz_basic(debug=False):
    """Test index2xyz with basic indices."""
    if debug:
        print("index2xyz_basic")
    x, y, z = lf.index2xyz(0, 10, 10)
    assert (x, y, z) == (0, 0, 0)

    x, y, z = lf.index2xyz(1, 10, 10)
    assert (x, y, z) == (0, 0, 1)

    x, y, z = lf.index2xyz(10, 10, 10)
    assert (x, y, z) == (0, 1, 0)

    x, y, z = lf.index2xyz(100, 10, 10)
    assert (x, y, z) == (1, 0, 0)


def index2xyz_nonsquare(debug=False):
    """Test index2xyz with non-square dimensions."""
    if debug:
        print("index2xyz_nonsquare")
    ysize, zsize = 5, 7
    # index = z + y*zsize + x*zsize*ysize
    idx = 6 + 4 * 7 + 2 * 7 * 5
    x, y, z = lf.index2xyz(idx, ysize, zsize)
    assert (x, y, z) == (2, 4, 6)


def index2xyz_roundtrip(debug=False):
    """Test that xyz2index and index2xyz are inverses."""
    if debug:
        print("index2xyz_roundtrip")
    xsize, ysize, zsize = 4, 5, 6
    for x in range(xsize):
        for y in range(ysize):
            for z in range(zsize):
                idx = lf.xyz2index(x, y, z, xsize, ysize, zsize)
                x2, y2, z2 = lf.index2xyz(idx, ysize, zsize)
                assert (x2, y2, z2) == (x, y, z), f"Roundtrip failed for ({x},{y},{z})"


def index2xyz_sequential(debug=False):
    """Test index2xyz produces sequential coordinates."""
    if debug:
        print("index2xyz_sequential")
    ysize, zsize = 3, 4
    # Check that indices 0..xsize*ysize*zsize-1 produce all valid coordinates
    xsize = 2
    coords = set()
    for idx in range(xsize * ysize * zsize):
        x, y, z = lf.index2xyz(idx, ysize, zsize)
        assert 0 <= x < xsize
        assert 0 <= y < ysize
        assert 0 <= z < zsize
        coords.add((x, y, z))
    assert len(coords) == xsize * ysize * zsize


# ==================== getcorrloc tests ====================


def getcorrloc_identical_signals(debug=False):
    """Test getcorrloc with identical signals returns high correlation at lag 0."""
    if debug:
        print("getcorrloc_identical_signals")
    Fs = 10.0
    npts = 500
    signal = _make_broadband_signal(npts, Fs, delay=0.0, seed=42)
    # Normalize for correlation
    import rapidtide.miscmath as tide_math

    sig_normed = tide_math.corrnormalize(signal, detrendorder=3, windowfunc="hamming")
    thedata = np.zeros((2, npts), dtype=float)
    thedata[0, :] = sig_normed
    thedata[1, :] = sig_normed

    maxcorr, maxtime, maskval, failreason = lf.getcorrloc(
        thedata, 0, 1, Fs, dofit=False, debug=debug,
    )
    if debug:
        print(f"  maxcorr={maxcorr:.4f}, maxtime={maxtime:.4f}")
    assert maxcorr > 0.9, f"Expected high correlation, got {maxcorr}"
    assert abs(maxtime) < 0.5, f"Expected lag near 0, got {maxtime}"
    assert maskval == 1
    assert failreason == 0


def getcorrloc_delayed_signals(debug=False):
    """Test getcorrloc with delayed signals returns correct lag."""
    if debug:
        print("getcorrloc_delayed_signals")
    Fs = 10.0
    npts = 500
    delay = 1.0  # 1 second delay
    sig1 = _make_broadband_signal(npts, Fs, delay=0.0, seed=42)
    sig2 = _make_broadband_signal(npts, Fs, delay=delay, seed=42)

    import rapidtide.miscmath as tide_math

    thedata = np.zeros((2, npts), dtype=float)
    thedata[0, :] = tide_math.corrnormalize(sig1, detrendorder=3, windowfunc="hamming")
    thedata[1, :] = tide_math.corrnormalize(sig2, detrendorder=3, windowfunc="hamming")

    maxcorr, maxtime, maskval, failreason = lf.getcorrloc(
        thedata, 0, 1, Fs, dofit=False, debug=debug,
    )
    if debug:
        print(f"  maxcorr={maxcorr:.4f}, maxtime={maxtime:.4f}")
    assert maxcorr > 0.5, f"Expected decent correlation, got {maxcorr}"
    # Cross-correlation convention: when sig2 is delayed by +delay relative to sig1,
    # the peak appears at -delay (sig1 leads sig2)
    assert abs(abs(maxtime) - delay) < 1.0 / Fs + 0.1, (
        f"Expected lag magnitude near {delay}, got {maxtime}"
    )


def getcorrloc_zero_signal(debug=False):
    """Test getcorrloc with zero signals returns zeros."""
    if debug:
        print("getcorrloc_zero_signal")
    Fs = 10.0
    npts = 100
    thedata = np.zeros((2, npts), dtype=float)
    maxcorr, maxtime, maskval, failreason = lf.getcorrloc(
        thedata, 0, 1, Fs, dofit=False,
    )
    assert maxcorr == 0.0
    assert maxtime == 0.0
    assert maskval == 0
    assert failreason == 0


def getcorrloc_with_fit(debug=False):
    """Test getcorrloc with dofit=True."""
    if debug:
        print("getcorrloc_with_fit")
    Fs = 10.0
    npts = 500
    sig1 = _make_broadband_signal(npts, Fs, delay=0.0, seed=42)
    sig2 = _make_broadband_signal(npts, Fs, delay=0.5, seed=42)

    import rapidtide.miscmath as tide_math

    thedata = np.zeros((2, npts), dtype=float)
    thedata[0, :] = tide_math.corrnormalize(sig1, detrendorder=3, windowfunc="hamming")
    thedata[1, :] = tide_math.corrnormalize(sig2, detrendorder=3, windowfunc="hamming")

    maxcorr, maxtime, maskval, failreason = lf.getcorrloc(
        thedata, 0, 1, Fs, dofit=True, debug=debug,
    )
    if debug:
        print(f"  maxcorr={maxcorr:.4f}, maxtime={maxtime:.4f}, maskval={maskval}, failreason={failreason}")
    # With fitting, we should get a result (even if fit fails, we get values back)
    # If the fit succeeds, maskval=1
    # If it fails, maskval=0 but we still get numbers


def getcorrloc_one_zero_signal(debug=False):
    """Test getcorrloc when one signal is all zeros."""
    if debug:
        print("getcorrloc_one_zero_signal")
    Fs = 10.0
    npts = 200
    thedata = np.zeros((2, npts), dtype=float)
    thedata[0, :] = _make_broadband_signal(npts, Fs, seed=42)
    # thedata[1, :] is all zeros

    maxcorr, maxtime, maskval, failreason = lf.getcorrloc(
        thedata, 0, 1, Fs, dofit=False,
    )
    assert maxcorr == 0.0
    assert maxtime == 0.0
    assert maskval == 0


def getcorrloc_search_range(debug=False):
    """Test getcorrloc with custom search ranges."""
    if debug:
        print("getcorrloc_search_range")
    Fs = 10.0
    npts = 500
    sig1 = _make_broadband_signal(npts, Fs, delay=0.0, seed=42)
    sig2 = _make_broadband_signal(npts, Fs, delay=0.0, seed=42)

    import rapidtide.miscmath as tide_math

    thedata = np.zeros((2, npts), dtype=float)
    thedata[0, :] = tide_math.corrnormalize(sig1, detrendorder=3, windowfunc="hamming")
    thedata[1, :] = tide_math.corrnormalize(sig2, detrendorder=3, windowfunc="hamming")

    maxcorr, maxtime, maskval, failreason = lf.getcorrloc(
        thedata, 0, 1, Fs, dofit=False,
        negsearch=5.0, possearch=5.0,
    )
    assert maskval == 1


# ==================== preprocdata tests ====================


def preprocdata_basic(debug=False):
    """Test preprocdata returns correct shapes."""
    if debug:
        print("preprocdata_basic")
    nx, ny, nz, nt = 3, 3, 2, 40
    Fs = 0.5
    tr = 2.0
    rng = np.random.RandomState(42)
    fmridata = rng.randn(nx, ny, nz, nt).astype(float) + 100.0
    themask = np.ones((nx, ny, nz), dtype=float)

    theprefilter = tide_filt.NoncausalFilter(filtertype="lfo")

    osfmridata, ostimepoints, oversamptr, numvoxels = lf.preprocdata(
        fmridata, themask, theprefilter, 2, Fs, tr,
        detrendorder=1, windowfunc="hamming", padseconds=0,
        showprogressbar=False,
    )
    numspatiallocs = nx * ny * nz
    assert osfmridata.shape[0] == numspatiallocs
    assert osfmridata.shape[1] == ostimepoints
    assert oversamptr == tr / 2
    assert numvoxels == numspatiallocs  # all voxels in mask


def preprocdata_with_mask(debug=False):
    """Test preprocdata only processes masked voxels."""
    if debug:
        print("preprocdata_with_mask")
    nx, ny, nz, nt = 4, 4, 2, 40
    Fs = 0.5
    tr = 2.0
    rng = np.random.RandomState(42)
    fmridata = rng.randn(nx, ny, nz, nt).astype(float) + 100.0
    themask = np.zeros((nx, ny, nz), dtype=float)
    # Only mask a few voxels
    themask[1, 1, 0] = 1.0
    themask[2, 2, 1] = 1.0
    themask[0, 3, 0] = 1.0

    theprefilter = tide_filt.NoncausalFilter(filtertype="lfo")

    osfmridata, ostimepoints, oversamptr, numvoxels = lf.preprocdata(
        fmridata, themask, theprefilter, 2, Fs, tr,
        detrendorder=1, windowfunc="hamming", padseconds=0,
        showprogressbar=False,
    )
    assert numvoxels == 3

    # Unmasked voxels should be all zeros
    numspatiallocs = nx * ny * nz
    mask_flat = themask.reshape(numspatiallocs)
    for i in range(numspatiallocs):
        if mask_flat[i] == 0:
            assert np.all(osfmridata[i, :] == 0.0)


def preprocdata_oversample_factor(debug=False):
    """Test preprocdata with different oversample factors."""
    if debug:
        print("preprocdata_oversample_factor")
    nx, ny, nz, nt = 2, 2, 2, 30
    Fs = 0.5
    tr = 2.0
    rng = np.random.RandomState(42)
    fmridata = rng.randn(nx, ny, nz, nt).astype(float) + 100.0
    themask = np.ones((nx, ny, nz), dtype=float)
    theprefilter = tide_filt.NoncausalFilter(filtertype="lfo")

    for factor in [1, 2, 4]:
        osfmridata, ostimepoints, oversamptr, numvoxels = lf.preprocdata(
            fmridata.copy(), themask, theprefilter, factor, Fs, tr,
            detrendorder=1, windowfunc="hamming", padseconds=0,
            showprogressbar=False,
        )
        expected_nt = nt * factor - (factor - 1)
        assert ostimepoints == expected_nt, (
            f"factor={factor}: expected {expected_nt} timepoints, got {ostimepoints}"
        )
        assert abs(oversamptr - tr / factor) < 1e-10


def preprocdata_detrend_order(debug=False):
    """Test preprocdata with different detrend orders."""
    if debug:
        print("preprocdata_detrend_order")
    nx, ny, nz, nt = 2, 2, 2, 40
    Fs = 0.5
    tr = 2.0
    rng = np.random.RandomState(42)
    fmridata = rng.randn(nx, ny, nz, nt).astype(float) + 100.0
    themask = np.ones((nx, ny, nz), dtype=float)
    theprefilter = tide_filt.NoncausalFilter(filtertype="lfo")

    for order in [0, 1, 3]:
        osfmridata, ostimepoints, oversamptr, numvoxels = lf.preprocdata(
            fmridata.copy(), themask, theprefilter, 2, Fs, tr,
            detrendorder=order, windowfunc="hamming", padseconds=0,
            showprogressbar=False,
        )
        assert osfmridata.shape[0] == nx * ny * nz


# ==================== _get_parser tests ====================


def get_parser_returns_parser(debug=False):
    """Test _get_parser returns an ArgumentParser."""
    if debug:
        print("get_parser_returns_parser")
    import argparse

    parser = lf._get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_defaults(debug=False):
    """Test _get_parser default values."""
    if debug:
        print("get_parser_defaults")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "output_root"])
    assert args.inputfilename == "input.nii.gz"
    assert args.outputroot == "output_root"
    assert args.npasses == lf.DEFAULT_NPASSES
    assert args.radius == lf.DEFAULT_RADIUS
    assert args.minlagdiff == lf.DEFAULT_MINLAGDIFF
    assert args.ampthresh == lf.DEFAULT_AMPTHRESH
    assert args.gausssigma == 0.0
    assert args.oversampfactor == -1
    assert not args.dofit
    assert args.detrendorder == lf.DEFAULT_DETREND_ORDER
    assert args.dosphere
    assert args.showprogressbar
    assert not args.debug


def get_parser_npasses(debug=False):
    """Test _get_parser accepts --npasses."""
    if debug:
        print("get_parser_npasses")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--npasses", "5"])
    assert args.npasses == 5


def get_parser_radius(debug=False):
    """Test _get_parser accepts --radius."""
    if debug:
        print("get_parser_radius")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--radius", "7.5"])
    assert args.radius == 7.5


def get_parser_ampthresh(debug=False):
    """Test _get_parser accepts --ampthresh."""
    if debug:
        print("get_parser_ampthresh")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--ampthresh", "0.5"])
    assert args.ampthresh == 0.5


def get_parser_gausssigma(debug=False):
    """Test _get_parser accepts --gausssigma."""
    if debug:
        print("get_parser_gausssigma")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--gausssigma", "3.0"])
    assert args.gausssigma == 3.0


def get_parser_oversampfac(debug=False):
    """Test _get_parser accepts --oversampfac."""
    if debug:
        print("get_parser_oversampfac")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--oversampfac", "4"])
    assert args.oversampfactor == 4


def get_parser_dofit(debug=False):
    """Test _get_parser accepts --dofit."""
    if debug:
        print("get_parser_dofit")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--dofit"])
    assert args.dofit


def get_parser_detrendorder(debug=False):
    """Test _get_parser accepts --detrendorder."""
    if debug:
        print("get_parser_detrendorder")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--detrendorder", "1"])
    assert args.detrendorder == 1


def get_parser_nosphere(debug=False):
    """Test _get_parser --nosphere disables sphere."""
    if debug:
        print("get_parser_nosphere")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--nosphere"])
    assert not args.dosphere


def get_parser_noprogressbar(debug=False):
    """Test _get_parser --noprogressbar disables progress bar."""
    if debug:
        print("get_parser_noprogressbar")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--noprogressbar"])
    assert not args.showprogressbar


def get_parser_debug(debug=False):
    """Test _get_parser accepts --debug."""
    if debug:
        print("get_parser_debug")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--debug"])
    assert args.debug


def get_parser_minlagdiff(debug=False):
    """Test _get_parser accepts --minlagdiff."""
    if debug:
        print("get_parser_minlagdiff")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--minlagdiff", "0.5"])
    assert args.minlagdiff == 0.5


def get_parser_windowfunc(debug=False):
    """Test _get_parser accepts --windowfunc."""
    if debug:
        print("get_parser_windowfunc")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--windowfunc", "hann"])
    assert args.windowfunc == "hann"


def get_parser_filterband(debug=False):
    """Test _get_parser accepts --filterband."""
    if debug:
        print("get_parser_filterband")
    parser = lf._get_parser()
    args = parser.parse_args(["input.nii.gz", "out", "--filterband", "lfo"])
    assert args.filterband == "lfo"


def get_parser_required_args(debug=False):
    """Test _get_parser requires inputfilename and outputroot."""
    if debug:
        print("get_parser_required_args")
    parser = lf._get_parser()
    try:
        parser.parse_args([])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


# ==================== localflow integration test ====================


def localflow_integration(debug=False):
    """Test localflow end-to-end with small synthetic data."""
    if debug:
        print("localflow_integration")
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (6, 6, 4, 50)
        tr = 2.0
        filepath, data = _make_4d_nifti(tmpdir, shape=shape, tr=tr)
        outputroot = os.path.join(tmpdir, "output")

        # Build args via parser and override
        parser = lf._get_parser()
        args = parser.parse_args([filepath, outputroot])
        # Use small parameters for speed
        args.npasses = 3
        args.radius = 3.0
        args.gausssigma = 0.0
        args.oversampfactor = 1
        args.detrendorder = 1
        args.showprogressbar = False
        args.dofit = False
        args.debug = debug
        args.ampthresh = 0.1

        # postprocess filter options (needed by localflow)
        _, theprefilter = pf.postprocessfilteropts(args)

        lf.localflow(args)

        # Check output files exist
        assert os.path.exists(outputroot + "_mask.nii.gz"), "mask not created"
        assert os.path.exists(outputroot + "_preprocdata.nii.gz"), "preprocdata not created"
        assert os.path.exists(outputroot + "_corrcoeffs.nii.gz"), "corrcoeffs not created"
        assert os.path.exists(outputroot + "_delays.nii.gz"), "delays not created"
        assert os.path.exists(outputroot + "_corrvalid.nii.gz"), "corrvalid not created"
        assert os.path.exists(outputroot + "_failreason.nii.gz"), "failreason not created"
        assert os.path.exists(outputroot + "_targetdelay.nii.gz"), "targetdelay not created"
        assert os.path.exists(outputroot + "_numneighbors.nii.gz"), "numneighbors not created"
        assert os.path.exists(outputroot + "_maxtime.nii.gz"), "maxtime not created"
        assert os.path.exists(outputroot + "_formattedruntimings.txt"), "timings not created"
        assert os.path.exists(outputroot + "_neighbors"), "neighbors not created"
        assert os.path.exists(outputroot + "_indexlist"), "indexlist not created"

        # Read back and verify shapes
        _, mask_data, _, maskdims, _ = tide_io_readfromnifti(outputroot + "_mask.nii.gz")
        assert mask_data.shape == shape[:3]

        _, targetdelay_data, _, _, _ = tide_io_readfromnifti(outputroot + "_targetdelay.nii.gz")
        assert targetdelay_data.shape == (shape[0], shape[1], shape[2], args.npasses)

        _, maxtime_data, _, _, _ = tide_io_readfromnifti(outputroot + "_maxtime.nii.gz")
        assert maxtime_data.shape == shape[:3]

        if debug:
            print("  Integration test passed - all output files verified")


def localflow_with_gausssigma(debug=False):
    """Test localflow with Gaussian spatial smoothing."""
    if debug:
        print("localflow_with_gausssigma")
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (6, 6, 4, 40)
        tr = 2.0
        filepath, data = _make_4d_nifti(tmpdir, shape=shape, tr=tr)
        outputroot = os.path.join(tmpdir, "output")

        parser = lf._get_parser()
        args = parser.parse_args([filepath, outputroot])
        args.npasses = 2
        args.radius = 3.0
        args.gausssigma = 2.0
        args.oversampfactor = 1
        args.detrendorder = 1
        args.showprogressbar = False
        args.dofit = False
        args.debug = debug
        args.ampthresh = 0.1

        _, theprefilter = pf.postprocessfilteropts(args)

        lf.localflow(args)

        assert os.path.exists(outputroot + "_mask.nii.gz")
        assert os.path.exists(outputroot + "_maxtime.nii.gz")
        if debug:
            print("  Gausssigma test passed")


# We need tide_io for reading back NIfTI files in integration tests
def tide_io_readfromnifti(filepath):
    """Helper to read NIfTI files for verification."""
    import rapidtide.io as tide_io
    return tide_io.readfromnifti(filepath)


# ==================== Main test function ====================


def test_localflow(debug=False):
    # xyz2index tests
    if debug:
        print("Running xyz2index tests")
    xyz2index_basic(debug=debug)
    xyz2index_corners(debug=debug)
    xyz2index_out_of_bounds(debug=debug)
    xyz2index_boundary(debug=debug)
    xyz2index_nonsquare(debug=debug)

    # index2xyz tests
    if debug:
        print("Running index2xyz tests")
    index2xyz_basic(debug=debug)
    index2xyz_nonsquare(debug=debug)
    index2xyz_roundtrip(debug=debug)
    index2xyz_sequential(debug=debug)

    # getcorrloc tests
    if debug:
        print("Running getcorrloc tests")
    getcorrloc_identical_signals(debug=debug)
    getcorrloc_delayed_signals(debug=debug)
    getcorrloc_zero_signal(debug=debug)
    getcorrloc_with_fit(debug=debug)
    getcorrloc_one_zero_signal(debug=debug)
    getcorrloc_search_range(debug=debug)

    # preprocdata tests
    if debug:
        print("Running preprocdata tests")
    preprocdata_basic(debug=debug)
    preprocdata_with_mask(debug=debug)
    preprocdata_oversample_factor(debug=debug)
    preprocdata_detrend_order(debug=debug)

    # _get_parser tests
    if debug:
        print("Running _get_parser tests")
    get_parser_returns_parser(debug=debug)
    get_parser_defaults(debug=debug)
    get_parser_npasses(debug=debug)
    get_parser_radius(debug=debug)
    get_parser_ampthresh(debug=debug)
    get_parser_gausssigma(debug=debug)
    get_parser_oversampfac(debug=debug)
    get_parser_dofit(debug=debug)
    get_parser_detrendorder(debug=debug)
    get_parser_nosphere(debug=debug)
    get_parser_noprogressbar(debug=debug)
    get_parser_debug(debug=debug)
    get_parser_minlagdiff(debug=debug)
    get_parser_windowfunc(debug=debug)
    get_parser_filterband(debug=debug)
    get_parser_required_args(debug=debug)

    # localflow integration tests
    if debug:
        print("Running localflow integration tests")
    localflow_integration(debug=debug)
    localflow_with_gausssigma(debug=debug)


if __name__ == "__main__":
    test_localflow(debug=True)
