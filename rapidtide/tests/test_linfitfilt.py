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
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.linfitfilt import _get_parser, linfitfilt, main

# ==================== Helpers ====================


def _make_mock_hdr(xsize, ysize, numslices, timepoints=1):
    """Create a mock NIfTI header that supports __getitem__ and __setitem__."""
    store = {
        "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
        "pixdim": [1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0],
    }
    hdr = MagicMock()
    hdr.__getitem__ = MagicMock(side_effect=lambda key: store[key])
    hdr.__setitem__ = MagicMock(side_effect=lambda key, val: store.__setitem__(key, val))

    def copy_fn():
        new_store = {
            "dim": list(store["dim"]),
            "pixdim": list(store["pixdim"]),
        }
        h = MagicMock()
        h.__getitem__ = MagicMock(side_effect=lambda key: new_store[key])
        h.__setitem__ = MagicMock(side_effect=lambda key, val: new_store.__setitem__(key, val))
        return h

    hdr.copy = copy_fn
    return hdr


def _make_4d_data(xsize=3, ysize=3, numslices=2, numtimepoints=20):
    """Create 4D data with signal + noise suitable for regression."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 4 * np.pi, numtimepoints)
    # Create data = baseline + signal * regressor + noise
    regressor = np.sin(t)
    data = np.zeros((xsize, ysize, numslices, numtimepoints), dtype=np.float64)
    for x in range(xsize):
        for y in range(ysize):
            for z in range(numslices):
                baseline = 100.0 + rng.rand() * 10
                amplitude = 5.0 + rng.rand() * 5
                noise = rng.randn(numtimepoints) * 0.5
                data[x, y, z, :] = baseline + amplitude * regressor + noise
    hdr = _make_mock_hdr(xsize, ysize, numslices, numtimepoints)
    dims = np.array([4, xsize, ysize, numslices, numtimepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes, regressor


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that parser creates successfully."""
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "linfitfilt"


def parser_required_args(debug=False):
    """Test parser has required arguments."""
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "inputfile" in actions
    assert "outputroot" in actions


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "outroot"])
    assert args.numskip == 0
    assert args.evfile is None
    assert args.datamaskname is None
    assert args.saveall is True


def parser_numskip(debug=False):
    """Test --numskip option."""
    if debug:
        print("parser_numskip")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "out", "--numskip", "5"])
    assert args.numskip == 5


def parser_limitoutput(debug=False):
    """Test --limitoutput flag sets saveall to False."""
    if debug:
        print("parser_limitoutput")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "out", "--limitoutput"])
    assert args.saveall is False


def parser_evfile(debug=False):
    """Test --evfile option with multiple files."""
    if debug:
        print("parser_evfile")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".txt") as f2, \
         tempfile.NamedTemporaryFile(suffix=".txt") as f3:
        args = parser.parse_args([f1.name, "out", "--evfile", f2.name, f3.name])
    assert len(args.evfile) == 2


def parser_dmask(debug=False):
    """Test --dmask option."""
    if debug:
        print("parser_dmask")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        args = parser.parse_args([f1.name, "out", "--dmask", f2.name])
    assert args.datamaskname is not None


# ==================== linfitfilt tests ====================


def _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints, numskip=0,
                                 ev_type="text", num_evs=1, with_mask=False):
    """Build mock functions and data for linfitfilt tests.

    Returns a dict with all the mock objects and data needed.
    """
    data, hdr, dims, sizes, regressor = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    effective_tp = numtimepoints - numskip

    # Build EV data
    ev_regressors = []
    for i in range(num_evs):
        rng = np.random.RandomState(100 + i)
        ev_regressors.append(np.sin(np.linspace(0, (2 + i) * np.pi, effective_tp)))

    # Mask data
    mask_data = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])

    saved_nifti = {}

    def mock_readfromnifti(fname, **kwargs):
        if "mask" in fname:
            return MagicMock(), mask_data, mask_hdr, mask_dims, sizes
        elif "ev_nifti" in fname:
            # 4D EV nifti
            ev_nifti_data = np.zeros((xsize, ysize, numslices, effective_tp), dtype=np.float64)
            for x in range(xsize):
                for y in range(ysize):
                    for z in range(numslices):
                        ev_nifti_data[x, y, z, :] = ev_regressors[0]
            ev_hdr = _make_mock_hdr(xsize, ysize, numslices, effective_tp)
            ev_dims = np.array([4, xsize, ysize, numslices, effective_tp, 1, 1, 1])
            return MagicMock(), ev_nifti_data, ev_hdr, ev_dims, sizes
        else:
            return MagicMock(), data, hdr, dims, sizes

    def mock_savetonifti(data_arr, header, fname, **kwargs):
        saved_nifti[fname] = data_arr.copy()

    def mock_checkifnifti(fname):
        return "ev_nifti" in fname

    def mock_checkifparfile(fname):
        return fname.endswith(".par")

    def mock_readvec(fname, **kwargs):
        return ev_regressors[0]

    def mock_readvecs(fname, **kwargs):
        # Return 6 rows for par file
        return np.array([ev_regressors[0]] * 6)

    return {
        "data": data,
        "hdr": hdr,
        "dims": dims,
        "sizes": sizes,
        "ev_regressors": ev_regressors,
        "mask_data": mask_data,
        "saved_nifti": saved_nifti,
        "mock_readfromnifti": mock_readfromnifti,
        "mock_savetonifti": mock_savetonifti,
        "mock_checkifnifti": mock_checkifnifti,
        "mock_checkifparfile": mock_checkifparfile,
        "mock_readvec": mock_readvec,
        "mock_readvecs": mock_readvecs,
    }


def linfitfilt_text_regressor(debug=False):
    """Test linfitfilt with a single text regressor, saveall=True."""
    if debug:
        print("linfitfilt_text_regressor")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints)

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", side_effect=m["mock_checkifnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", side_effect=m["mock_checkifparfile"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", side_effect=m["mock_readvec"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt",
            evfilename=["regressor.txt"],
            datamaskname=None,
            saveall=True,
        )

    saved = m["saved_nifti"]
    # With saveall=True, should save: mean, fit0, R2, totaltoremove, trimmed, filtered
    assert "/tmp/test_linfitfilt_mean" in saved
    assert "/tmp/test_linfitfilt_fit0" in saved
    assert "/tmp/test_linfitfilt_R2" in saved
    assert "/tmp/test_linfitfilt_totaltoremove" in saved
    assert "/tmp/test_linfitfilt_trimmed" in saved
    assert "/tmp/test_linfitfilt_filtered" in saved

    # Check shapes
    assert saved["/tmp/test_linfitfilt_mean"].shape == (xsize, ysize, numslices)
    assert saved["/tmp/test_linfitfilt_R2"].shape == (xsize, ysize, numslices)
    assert saved["/tmp/test_linfitfilt_filtered"].shape == (xsize, ysize, numslices, numtimepoints)


def linfitfilt_limitoutput(debug=False):
    """Test linfitfilt with saveall=False (limitoutput)."""
    if debug:
        print("linfitfilt_limitoutput")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints)

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", side_effect=m["mock_checkifnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", side_effect=m["mock_checkifparfile"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", side_effect=m["mock_readvec"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_lim",
            evfilename=["regressor.txt"],
            datamaskname=None,
            saveall=False,
        )

    saved = m["saved_nifti"]
    # With saveall=False, should only save: R2 and filtered
    assert "/tmp/test_linfitfilt_lim_R2" in saved
    assert "/tmp/test_linfitfilt_lim_filtered" in saved
    # Should NOT save: mean, fit0, totaltoremove, trimmed
    assert "/tmp/test_linfitfilt_lim_mean" not in saved
    assert "/tmp/test_linfitfilt_lim_fit0" not in saved
    assert "/tmp/test_linfitfilt_lim_totaltoremove" not in saved
    assert "/tmp/test_linfitfilt_lim_trimmed" not in saved


def linfitfilt_with_numskip(debug=False):
    """Test linfitfilt with numskip > 0 using a text regressor."""
    if debug:
        print("linfitfilt_with_numskip")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    numskip = 5
    effective_tp = numtimepoints - numskip

    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints, numskip=numskip)

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", side_effect=m["mock_checkifnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", side_effect=m["mock_checkifparfile"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", side_effect=m["mock_readvec"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=numskip,
            outputroot="/tmp/test_linfitfilt_skip",
            evfilename=["regressor.txt"],
            datamaskname=None,
            saveall=True,
        )

    saved = m["saved_nifti"]
    # Filtered data should have effective_tp timepoints
    assert saved["/tmp/test_linfitfilt_skip_filtered"].shape == (
        xsize, ysize, numslices, effective_tp
    )
    assert saved["/tmp/test_linfitfilt_skip_trimmed"].shape == (
        xsize, ysize, numslices, effective_tp
    )


def linfitfilt_with_mask(debug=False):
    """Test linfitfilt with a data mask."""
    if debug:
        print("linfitfilt_with_mask")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints, with_mask=True)
    # Zero out part of the mask
    m["mask_data"][0, 0, 0] = 0.0

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", side_effect=m["mock_checkifnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", side_effect=m["mock_checkifparfile"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", side_effect=m["mock_readvec"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_mask",
            evfilename=["regressor.txt"],
            datamaskname="dummy_mask.nii.gz",
            saveall=True,
        )

    saved = m["saved_nifti"]
    # Masked voxel should have zero mean and R2
    assert saved["/tmp/test_linfitfilt_mask_mean"][0, 0, 0] == 0.0
    assert saved["/tmp/test_linfitfilt_mask_R2"][0, 0, 0] == 0.0
    # Unmasked voxel should have non-zero values
    assert saved["/tmp/test_linfitfilt_mask_R2"][1, 1, 0] != 0.0


def linfitfilt_nifti_regressor(debug=False):
    """Test linfitfilt with a NIfTI (voxel-specific) regressor."""
    if debug:
        print("linfitfilt_nifti_regressor")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints, ev_type="nifti")

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", return_value=True), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_nifti",
            evfilename=["ev_nifti.nii.gz"],
            datamaskname=None,
            saveall=True,
        )

    saved = m["saved_nifti"]
    assert "/tmp/test_linfitfilt_nifti_filtered" in saved
    assert "/tmp/test_linfitfilt_nifti_R2" in saved


def linfitfilt_parfile_regressor(debug=False):
    """Test linfitfilt with an FSL par file (6 regressors)."""
    if debug:
        print("linfitfilt_parfile_regressor")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints)

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", return_value=True), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvecs", side_effect=m["mock_readvecs"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_par",
            evfilename=["motion.par"],
            datamaskname=None,
            saveall=True,
        )

    saved = m["saved_nifti"]
    # Par file produces 6 regressors => fit0 through fit5
    for j in range(6):
        assert f"/tmp/test_linfitfilt_par_fit{j}" in saved, \
            f"Expected fit{j} in saved files"
    assert "/tmp/test_linfitfilt_par_filtered" in saved


def linfitfilt_multiple_text_regressors(debug=False):
    """Test linfitfilt with multiple text regressors."""
    if debug:
        print("linfitfilt_multiple_text_regressors")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints, num_evs=3)

    ev_idx = {"i": 0}

    def mock_readvec(fname, **kwargs):
        idx = ev_idx["i"]
        ev_idx["i"] += 1
        return m["ev_regressors"][idx]

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", side_effect=mock_readvec), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_multi",
            evfilename=["reg1.txt", "reg2.txt", "reg3.txt"],
            datamaskname=None,
            saveall=True,
        )

    saved = m["saved_nifti"]
    # 3 regressors => fit0, fit1, fit2
    for j in range(3):
        assert f"/tmp/test_linfitfilt_multi_fit{j}" in saved
    assert "/tmp/test_linfitfilt_multi_filtered" in saved


def linfitfilt_mask_mismatch(debug=False):
    """Test that mask space mismatch causes exit."""
    if debug:
        print("linfitfilt_mask_mismatch")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints)

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=False):

        with pytest.raises(SystemExit):
            linfitfilt(
                inputfile="dummy_data.nii.gz",
                numskip=0,
                outputroot="/tmp/test_linfitfilt_mismatch",
                evfilename=["regressor.txt"],
                datamaskname="bad_mask.nii.gz",
                saveall=True,
            )


def linfitfilt_r2_values(debug=False):
    """Test that R2 values are reasonable for a known signal."""
    if debug:
        print("linfitfilt_r2_values")
    xsize, ysize, numslices, numtimepoints = 2, 2, 1, 50
    rng = np.random.RandomState(42)
    t = np.linspace(0, 4 * np.pi, numtimepoints)
    regressor = np.sin(t)

    # Create data that is a strong linear function of the regressor
    data = np.zeros((xsize, ysize, numslices, numtimepoints), dtype=np.float64)
    for x in range(xsize):
        for y in range(ysize):
            data[x, y, 0, :] = 100.0 + 10.0 * regressor + rng.randn(numtimepoints) * 0.01

    hdr = _make_mock_hdr(xsize, ysize, numslices, numtimepoints)
    dims = np.array([4, xsize, ysize, numslices, numtimepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])

    saved_nifti = {}

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), data, hdr, dims, sizes

    def mock_savetonifti(data_arr, header, fname, **kwargs):
        saved_nifti[fname] = data_arr.copy()

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", return_value=regressor), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=mock_savetonifti):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_r2",
            evfilename=["regressor.txt"],
            datamaskname=None,
            saveall=True,
        )

    r2 = saved_nifti["/tmp/test_linfitfilt_r2_R2"]
    # With very low noise, R2 should be close to 1
    for x in range(xsize):
        for y in range(ysize):
            assert r2[x, y, 0] > 0.99, f"R2 at ({x},{y},0) = {r2[x,y,0]}, expected > 0.99"


def linfitfilt_flat_voxel(debug=False):
    """Test that flat (zero-variance) voxels get zero fit values."""
    if debug:
        print("linfitfilt_flat_voxel")
    xsize, ysize, numslices, numtimepoints = 2, 2, 1, 20
    t = np.linspace(0, 4 * np.pi, numtimepoints)
    regressor = np.sin(t)

    data = np.zeros((xsize, ysize, numslices, numtimepoints), dtype=np.float64)
    # Make one voxel flat (constant)
    data[0, 0, 0, :] = 100.0
    # Make another voxel have signal
    data[1, 1, 0, :] = 100.0 + 5.0 * regressor

    hdr = _make_mock_hdr(xsize, ysize, numslices, numtimepoints)
    dims = np.array([4, xsize, ysize, numslices, numtimepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])

    saved_nifti = {}

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), data, hdr, dims, sizes

    def mock_savetonifti(data_arr, header, fname, **kwargs):
        saved_nifti[fname] = data_arr.copy()

    with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", return_value=False), \
         patch("rapidtide.workflows.linfitfilt.tide_io.readvec", return_value=regressor), \
         patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=mock_savetonifti):

        linfitfilt(
            inputfile="dummy_data.nii.gz",
            numskip=0,
            outputroot="/tmp/test_linfitfilt_flat",
            evfilename=["regressor.txt"],
            datamaskname=None,
            saveall=True,
        )

    saved = saved_nifti
    # Flat voxel should have zero R2 and zero fit
    assert saved["/tmp/test_linfitfilt_flat_R2"][0, 0, 0] == 0.0
    assert saved["/tmp/test_linfitfilt_flat_mean"][0, 0, 0] == 0.0
    assert saved["/tmp/test_linfitfilt_flat_fit0"][0, 0, 0] == 0.0
    # Signal voxel should have non-zero values
    assert saved["/tmp/test_linfitfilt_flat_R2"][1, 1, 0] > 0.0


# ==================== main tests ====================


def main_function(debug=False):
    """Test main function dispatches correctly."""
    if debug:
        print("main_function")
    xsize, ysize, numslices, numtimepoints = 3, 3, 2, 20
    m = _setup_mocks_for_linfitfilt(xsize, ysize, numslices, numtimepoints)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as input_f, \
         tempfile.NamedTemporaryFile(suffix=".txt") as ev_f:

        with patch("rapidtide.workflows.linfitfilt.tide_io.readfromnifti", side_effect=m["mock_readfromnifti"]), \
             patch("rapidtide.workflows.linfitfilt.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
             patch("rapidtide.workflows.linfitfilt.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
             patch("rapidtide.workflows.linfitfilt.tide_io.checkifnifti", return_value=False), \
             patch("rapidtide.workflows.linfitfilt.tide_io.checkifparfile", return_value=False), \
             patch("rapidtide.workflows.linfitfilt.tide_io.readvec", side_effect=m["mock_readvec"]), \
             patch("rapidtide.workflows.linfitfilt.tide_io.savetonifti", side_effect=m["mock_savetonifti"]), \
             patch("rapidtide.workflows.linfitfilt.tide_io.checkspacematch", return_value=True), \
             patch("sys.argv", ["linfitfilt", input_f.name, "/tmp/test_main_out",
                                "--evfile", ev_f.name]):

            main()

    saved = m["saved_nifti"]
    assert "/tmp/test_main_out_R2" in saved
    assert "/tmp/test_main_out_filtered" in saved


# ==================== Main test function ====================


def test_linfitfilt(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_numskip(debug=debug)
    parser_limitoutput(debug=debug)
    parser_evfile(debug=debug)
    parser_dmask(debug=debug)

    # linfitfilt tests
    if debug:
        print("Running linfitfilt tests")
    linfitfilt_text_regressor(debug=debug)
    linfitfilt_limitoutput(debug=debug)
    linfitfilt_with_numskip(debug=debug)
    linfitfilt_with_mask(debug=debug)
    linfitfilt_nifti_regressor(debug=debug)
    linfitfilt_parfile_regressor(debug=debug)
    linfitfilt_multiple_text_regressors(debug=debug)
    linfitfilt_mask_mismatch(debug=debug)
    linfitfilt_r2_values(debug=debug)
    linfitfilt_flat_voxel(debug=debug)

    # main tests
    if debug:
        print("Running main tests")
    main_function(debug=debug)


if __name__ == "__main__":
    test_linfitfilt(debug=True)
