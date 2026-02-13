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
import copy
import io
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from rapidtide.workflows.retrolagtcs import (
    DEFAULT_REGRESSIONFILTDERIVS,
    _get_parser,
    retrolagtcs,
)

# ---- helpers ----


def _make_mock_header():
    """Create a mock NIfTI header that supports getitem, setitem, and deepcopy."""
    hdr = MagicMock()
    hdr_data = {}

    def getitem(key):
        return hdr_data.get(key, None)

    def setitem(key, value):
        hdr_data[key] = value

    hdr.__getitem__ = MagicMock(side_effect=getitem)
    hdr.__setitem__ = MagicMock(side_effect=setitem)
    hdr.__deepcopy__ = MagicMock(side_effect=lambda memo: _make_mock_header())
    return hdr


def _make_default_args(**overrides):
    """Create default args Namespace for retrolagtcs."""
    defaults = dict(
        fmrifile="fmri.nii.gz",
        maskfile="mask.nii.gz",
        lagtimesfile="lagtimes.nii.gz",
        lagtcgeneratorfile="generator_timeseries",
        outputroot="/tmp/test_retrolagtcs_out",
        regressderivs=0,
        nprocs=1,
        numskip=0,
        showprogressbar=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mock_nifti_data(xsize=4, ysize=4, numslices=3, timepoints=100, fmritr=2.0):
    """Create consistent mock data for fMRI, mask, and lagtimes.

    Returns a dict with all the mock objects needed for retrolagtcs.
    """
    numspatiallocs = xsize * ysize * numslices

    # fMRI header data
    fmri_header = _make_mock_header()
    fmri_dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    fmri_sizes = np.array([4, 1.0, 1.0, 1.0, fmritr, 1.0, 1.0, 1.0])
    fmri_input = MagicMock()

    # mask: mark some voxels as active
    mask_header = _make_mock_header()
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mask_data = np.zeros(numspatiallocs, dtype=np.float64)
    # activate about half the voxels
    active_voxels = np.arange(0, numspatiallocs, 2)
    mask_data[active_voxels] = 1.0
    mask_input = MagicMock()

    # lag times: random lag values for each voxel
    lagtimes_header = _make_mock_header()
    lagtimes_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    lagtimes_sizes = np.array([3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rng = np.random.RandomState(42)
    lagtimes_data = rng.uniform(-3.0, 3.0, numspatiallocs)
    lagtimes_input = MagicMock()

    return {
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
        "fmritr": fmritr,
        "numspatiallocs": numspatiallocs,
        "active_voxels": active_voxels,
        "num_active": len(active_voxels),
        "fmri": (fmri_input, None, fmri_header, fmri_dims, fmri_sizes),
        "mask": (mask_input, mask_data, mask_header, mask_dims, mask_sizes),
        "lagtimes": (
            lagtimes_input,
            lagtimes_data,
            lagtimes_header,
            lagtimes_dims,
            lagtimes_sizes,
        ),
    }


def _run_retrolagtcs_with_mocks(args, mock_data, makelagged_side_effect=None):
    """Run retrolagtcs with fully mocked I/O.

    Returns a dict with captured calls and data.
    """
    captured = {"savemaplist_calls": [], "makelaggedtcs_call": None}

    def mock_readfromnifti(fname, headeronly=False):
        if "fmri" in fname or fname == args.fmrifile:
            return mock_data["fmri"]
        elif "mask" in fname or fname == args.maskfile:
            return mock_data["mask"]
        elif "lagtime" in fname or fname == args.lagtimesfile:
            return mock_data["lagtimes"]
        raise ValueError(f"Unexpected file: {fname}")

    def mock_parseniftidims(dims):
        return int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    def mock_parseniftisizes(sizes):
        return float(sizes[1]), float(sizes[2]), float(sizes[3]), float(sizes[4])

    def mock_checkspacematch(hdr1, hdr2):
        return True

    def mock_savemaplist(outputname, maplist, validvoxels, destshape, theheader,
                         bidsdict, **kwargs):
        captured["savemaplist_calls"].append({
            "outputname": outputname,
            "maplist": maplist,
            "validvoxels": validvoxels,
            "destshape": destshape,
        })

    num_active = mock_data["num_active"]
    timepoints = mock_data["timepoints"]

    def mock_makelaggedtcs(genlagtc, timeaxis, lagmask, lagtimes, lagtc, **kwargs):
        captured["makelaggedtcs_call"] = {
            "timeaxis_shape": timeaxis.shape,
            "lagmask_shape": lagmask.shape,
            "lagtimes_shape": lagtimes.shape,
            "lagtc_shape": lagtc.shape,
        }
        # fill lagtc with some synthetic data
        for i in range(lagtc.shape[0]):
            lagtc[i, :] = np.sin(
                2 * np.pi * 0.1 * np.arange(lagtc.shape[1]) + lagtimes[i]
            )
        return lagmask.shape[0]

    if makelagged_side_effect is not None:
        actual_makelagged = makelagged_side_effect
    else:
        actual_makelagged = mock_makelaggedtcs

    validtimepoints = timepoints - args.numskip

    def mock_allocarray(shape, dtype, shared=False):
        return np.zeros(shape, dtype=dtype), None

    mock_genlagtc = MagicMock()

    def mock_makevoxelspecificderivs(theevs, nderivs=1, debug=False):
        nvox, ntime = theevs.shape
        result = np.zeros((nvox, ntime, nderivs + 1), dtype=theevs.dtype)
        result[:, :, 0] = theevs
        for d in range(1, nderivs + 1):
            result[:, 1:, d] = np.diff(theevs, axis=1) / (d + 1.0)
        return result

    with (
        patch("rapidtide.workflows.retrolagtcs.tide_io.readfromnifti",
              side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftidims",
              side_effect=mock_parseniftidims),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftisizes",
              side_effect=mock_parseniftisizes),
        patch("rapidtide.workflows.retrolagtcs.tide_io.checkspacematch",
              side_effect=mock_checkspacematch),
        patch("rapidtide.workflows.retrolagtcs.tide_io.savemaplist",
              side_effect=mock_savemaplist),
        patch("rapidtide.workflows.retrolagtcs.tide_resample.FastResamplerFromFile",
              return_value=mock_genlagtc),
        patch("rapidtide.workflows.retrolagtcs.tide_makelagged.makelaggedtcs",
              side_effect=actual_makelagged),
        patch("rapidtide.workflows.retrolagtcs.tide_util.allocarray",
              side_effect=mock_allocarray),
        patch("rapidtide.workflows.retrolagtcs.tide_linfitfiltpass.makevoxelspecificderivs",
              side_effect=mock_makevoxelspecificderivs),
    ):
        retrolagtcs(args)

    return captured


# ---- _get_parser tests ----


def parser_basic(debug=False):
    """Test that _get_parser returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "retrolagtcs"

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires all five positional arguments."""
    parser = _get_parser()

    # no args
    with pytest.raises(SystemExit):
        parser.parse_args([])

    # only 1 arg
    with pytest.raises(SystemExit):
        parser.parse_args(["fmri.nii.gz"])

    # only 2 args
    with pytest.raises(SystemExit):
        parser.parse_args(["fmri.nii.gz", "mask.nii.gz"])

    # only 3 args
    with pytest.raises(SystemExit):
        parser.parse_args(["fmri.nii.gz", "mask.nii.gz", "lagtimes.nii.gz"])

    # only 4 args
    with pytest.raises(SystemExit):
        parser.parse_args(["fmri.nii.gz", "mask.nii.gz", "lagtimes.nii.gz", "gen.txt"])

    if debug:
        print("parser_required_args passed")


def parser_defaults(debug=False):
    """Test default values from the parser."""
    parser = _get_parser()
    # fmrifile uses is_valid_file validation, so we need a real file for that arg
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args([
            f.name, "mask.nii.gz", "lagtimes.nii.gz", "gen.txt", "output"
        ])

    assert args.maskfile == "mask.nii.gz"
    assert args.lagtimesfile == "lagtimes.nii.gz"
    assert args.lagtcgeneratorfile == "gen.txt"
    assert args.outputroot == "output"
    assert args.regressderivs == DEFAULT_REGRESSIONFILTDERIVS
    assert args.nprocs == 1
    assert args.numskip == 0
    assert args.showprogressbar is True
    assert args.debug is False

    if debug:
        print("parser_defaults passed")


def parser_regressderivs(debug=False):
    """Test --regressderivs option."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args([
            f.name, "mask.nii.gz", "lag.nii.gz", "gen.txt", "out",
            "--regressderivs", "3",
        ])
    assert args.regressderivs == 3

    if debug:
        print("parser_regressderivs passed")


def parser_nprocs(debug=False):
    """Test --nprocs option."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args([
            f.name, "mask.nii.gz", "lag.nii.gz", "gen.txt", "out",
            "--nprocs", "8",
        ])
    assert args.nprocs == 8

    if debug:
        print("parser_nprocs passed")


def parser_numskip(debug=False):
    """Test --numskip option."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args([
            f.name, "mask.nii.gz", "lag.nii.gz", "gen.txt", "out",
            "--numskip", "5",
        ])
    assert args.numskip == 5

    if debug:
        print("parser_numskip passed")


def parser_noprogressbar(debug=False):
    """Test --noprogressbar flag."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args([
            f.name, "mask.nii.gz", "lag.nii.gz", "gen.txt", "out",
            "--noprogressbar",
        ])
    assert args.showprogressbar is False

    if debug:
        print("parser_noprogressbar passed")


def parser_debug(debug=False):
    """Test --debug flag."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args([
            f.name, "mask.nii.gz", "lag.nii.gz", "gen.txt", "out",
            "--debug",
        ])
    assert args.debug is True

    if debug:
        print("parser_debug passed")


def parser_fmrifile_validation(debug=False):
    """Test that fmrifile must exist (is_valid_file check)."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "/nonexistent/file.nii.gz", "mask.nii.gz", "lag.nii.gz", "gen.txt", "out",
        ])

    if debug:
        print("parser_fmrifile_validation passed")


def parser_default_constant(debug=False):
    """Test that DEFAULT_REGRESSIONFILTDERIVS is 0."""
    assert DEFAULT_REGRESSIONFILTDERIVS == 0

    if debug:
        print("parser_default_constant passed")


# ---- retrolagtcs tests ----


def retrolagtcs_basic(debug=False):
    """Test basic retrolagtcs workflow with default settings."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args()

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    # makelaggedtcs should have been called
    assert captured["makelaggedtcs_call"] is not None

    # with regressderivs=0, one savemaplist call for 4D regressor maps
    assert len(captured["savemaplist_calls"]) == 1

    # the 4D map should have the correct shape
    call_4d = captured["savemaplist_calls"][0]
    assert call_4d["destshape"] == (4, 4, 3, 100)

    if debug:
        print("retrolagtcs_basic passed")


def retrolagtcs_makelaggedtcs_args(debug=False):
    """Test that makelaggedtcs is called with correct array shapes."""
    mock_data = _make_mock_nifti_data(xsize=4, ysize=4, numslices=3, timepoints=100)
    args = _make_default_args()

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    mc = captured["makelaggedtcs_call"]
    num_active = mock_data["num_active"]

    # timeaxis should have validtimepoints = timepoints - numskip = 100
    assert mc["timeaxis_shape"] == (100,)

    # lagmask and lagtimes should have shape (num_active,)
    assert mc["lagmask_shape"] == (num_active,)
    assert mc["lagtimes_shape"] == (num_active,)

    # lagtc shape: (num_active, validtimepoints)
    assert mc["lagtc_shape"] == (num_active, 100)

    if debug:
        print("retrolagtcs_makelaggedtcs_args passed")


def retrolagtcs_numskip(debug=False):
    """Test that numskip reduces validtimepoints correctly."""
    mock_data = _make_mock_nifti_data(timepoints=100)
    args = _make_default_args(numskip=10)

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    mc = captured["makelaggedtcs_call"]
    # validtimepoints = 100 - 10 = 90
    assert mc["timeaxis_shape"] == (90,)
    assert mc["lagtc_shape"][1] == 90

    # 4D output shape
    call_4d = captured["savemaplist_calls"][0]
    assert call_4d["destshape"] == (4, 4, 3, 90)

    if debug:
        print("retrolagtcs_numskip passed")


def retrolagtcs_voxel_selection(debug=False):
    """Test that only masked voxels are selected."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args()

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    mc = captured["makelaggedtcs_call"]
    num_active = mock_data["num_active"]

    # only voxels with mask > 0 should be included
    assert mc["lagmask_shape"] == (num_active,)

    if debug:
        print("retrolagtcs_voxel_selection passed")


def retrolagtcs_no_derivs(debug=False):
    """Test retrolagtcs with regressderivs=0 (single EV path)."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(regressderivs=0)

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    # only one savemaplist call for the 4D maps
    assert len(captured["savemaplist_calls"]) == 1

    # the maplist should contain exactly one entry (the lfofilterEV)
    maplist = captured["savemaplist_calls"][0]["maplist"]
    assert len(maplist) == 1
    assert maplist[0][1] == "lfofilterEV"

    if debug:
        print("retrolagtcs_no_derivs passed")


def retrolagtcs_with_derivs(debug=False):
    """Test retrolagtcs with regressderivs > 0 (multiple EV path)."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(regressderivs=2)

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    # one savemaplist call for the 4D maps
    assert len(captured["savemaplist_calls"]) == 1

    # maplist should have 3 entries: base EV + 2 derivative EVs
    maplist = captured["savemaplist_calls"][0]["maplist"]
    assert len(maplist) == 3
    assert maplist[0][1] == "lfofilterEV"
    assert maplist[1][1] == "lfofilterEVDeriv1"
    assert maplist[2][1] == "lfofilterEVDeriv2"

    if debug:
        print("retrolagtcs_with_derivs passed")


def retrolagtcs_with_one_deriv(debug=False):
    """Test retrolagtcs with regressderivs=1."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(regressderivs=1)

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    maplist = captured["savemaplist_calls"][0]["maplist"]
    assert len(maplist) == 2
    assert maplist[0][1] == "lfofilterEV"
    assert maplist[1][1] == "lfofilterEVDeriv1"

    if debug:
        print("retrolagtcs_with_one_deriv passed")


def retrolagtcs_debug_mode(debug=False):
    """Test retrolagtcs with debug=True produces extra savemaplist call."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(debug=True)

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    # in debug mode, there should be two savemaplist calls:
    # 1) 3D debug maps (lagtimes, mask)
    # 2) 4D regressor maps
    assert len(captured["savemaplist_calls"]) == 2

    # first call is the debug 3D maps
    debug_call = captured["savemaplist_calls"][0]
    debug_maplist = debug_call["maplist"]
    assert len(debug_maplist) == 2
    assert debug_maplist[0][1] == "maxtimeREAD"
    assert debug_maplist[1][1] == "maskREAD"
    assert debug_call["destshape"] == (4, 4, 3)

    # second call is the 4D maps
    call_4d = captured["savemaplist_calls"][1]
    assert call_4d["destshape"] == (4, 4, 3, 100)

    if debug:
        print("retrolagtcs_debug_mode passed")


def retrolagtcs_debug_output(debug=False):
    """Test that debug mode prints extra information."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(debug=True)

    captured_stdout = io.StringIO()
    with patch("sys.stdout", captured_stdout):
        _run_retrolagtcs_with_mocks(args, mock_data)

    output = captured_stdout.getvalue()
    assert "procmask_spacebytime.shape=" in output
    assert "lagtimes.shape=" in output
    assert "validvoxels.shape=" in output

    if debug:
        print("retrolagtcs_debug_output passed")


def retrolagtcs_space_mismatch_mask(debug=False):
    """Test that retrolagtcs raises ValueError on mask spatial mismatch."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args()

    call_count = [0]

    def mock_checkspacematch(hdr1, hdr2):
        call_count[0] += 1
        # fail on first call (mask check)
        return False

    def mock_readfromnifti(fname, headeronly=False):
        if fname == args.fmrifile:
            return mock_data["fmri"]
        elif fname == args.maskfile:
            return mock_data["mask"]
        elif fname == args.lagtimesfile:
            return mock_data["lagtimes"]
        raise ValueError(f"Unexpected file: {fname}")

    def mock_parseniftidims(dims):
        return int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    def mock_parseniftisizes(sizes):
        return float(sizes[1]), float(sizes[2]), float(sizes[3]), float(sizes[4])

    with pytest.raises(ValueError, match="procmask dimensions do not match"):
        with (
            patch("rapidtide.workflows.retrolagtcs.tide_io.readfromnifti",
                  side_effect=mock_readfromnifti),
            patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftidims",
                  side_effect=mock_parseniftidims),
            patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftisizes",
                  side_effect=mock_parseniftisizes),
            patch("rapidtide.workflows.retrolagtcs.tide_io.checkspacematch",
                  side_effect=mock_checkspacematch),
        ):
            retrolagtcs(args)

    if debug:
        print("retrolagtcs_space_mismatch_mask passed")


def retrolagtcs_space_mismatch_lagtimes(debug=False):
    """Test that retrolagtcs raises ValueError on lagtimes spatial mismatch."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args()

    call_count = [0]

    def mock_checkspacematch(hdr1, hdr2):
        call_count[0] += 1
        if call_count[0] == 1:
            return True  # mask check passes
        return False  # lagtimes check fails

    def mock_readfromnifti(fname, headeronly=False):
        if fname == args.fmrifile:
            return mock_data["fmri"]
        elif fname == args.maskfile:
            return mock_data["mask"]
        elif fname == args.lagtimesfile:
            return mock_data["lagtimes"]
        raise ValueError(f"Unexpected file: {fname}")

    def mock_parseniftidims(dims):
        return int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    def mock_parseniftisizes(sizes):
        return float(sizes[1]), float(sizes[2]), float(sizes[3]), float(sizes[4])

    with pytest.raises(ValueError, match="lagtimes dimensions do not match"):
        with (
            patch("rapidtide.workflows.retrolagtcs.tide_io.readfromnifti",
                  side_effect=mock_readfromnifti),
            patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftidims",
                  side_effect=mock_parseniftidims),
            patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftisizes",
                  side_effect=mock_parseniftisizes),
            patch("rapidtide.workflows.retrolagtcs.tide_io.checkspacematch",
                  side_effect=mock_checkspacematch),
        ):
            retrolagtcs(args)

    if debug:
        print("retrolagtcs_space_mismatch_lagtimes passed")


def retrolagtcs_nprocs_auto(debug=False):
    """Test that nprocs < 1 triggers maxcpus() and shared memory."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(nprocs=-1)

    maxcpus_called = [False]
    cleanup_calls = []

    original_run = _run_retrolagtcs_with_mocks

    def mock_readfromnifti(fname, headeronly=False):
        if fname == args.fmrifile:
            return mock_data["fmri"]
        elif fname == args.maskfile:
            return mock_data["mask"]
        elif fname == args.lagtimesfile:
            return mock_data["lagtimes"]
        raise ValueError(f"Unexpected file: {fname}")

    def mock_parseniftidims(dims):
        return int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    def mock_parseniftisizes(sizes):
        return float(sizes[1]), float(sizes[2]), float(sizes[3]), float(sizes[4])

    def mock_allocarray(shape, dtype, shared=False):
        return np.zeros(shape, dtype=dtype), MagicMock() if shared else None

    def mock_makelaggedtcs(genlagtc, timeaxis, lagmask, lagtimes, lagtc, **kwargs):
        for i in range(lagtc.shape[0]):
            lagtc[i, :] = np.sin(2 * np.pi * 0.1 * np.arange(lagtc.shape[1]))
        return lagmask.shape[0]

    with (
        patch("rapidtide.workflows.retrolagtcs.tide_io.readfromnifti",
              side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftidims",
              side_effect=mock_parseniftidims),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftisizes",
              side_effect=mock_parseniftisizes),
        patch("rapidtide.workflows.retrolagtcs.tide_io.checkspacematch",
              return_value=True),
        patch("rapidtide.workflows.retrolagtcs.tide_io.savemaplist"),
        patch("rapidtide.workflows.retrolagtcs.tide_resample.FastResamplerFromFile",
              return_value=MagicMock()),
        patch("rapidtide.workflows.retrolagtcs.tide_makelagged.makelaggedtcs",
              side_effect=mock_makelaggedtcs),
        patch("rapidtide.workflows.retrolagtcs.tide_util.allocarray",
              side_effect=mock_allocarray),
        patch("rapidtide.workflows.retrolagtcs.tide_multiproc.maxcpus",
              return_value=4) as mock_maxcpus,
        patch("rapidtide.workflows.retrolagtcs.tide_util.cleanup_shm") as mock_cleanup,
        patch("rapidtide.workflows.retrolagtcs.tide_linfitfiltpass.makevoxelspecificderivs"),
    ):
        retrolagtcs(args)
        maxcpus_called[0] = mock_maxcpus.called

        # nprocs should have been set to 4 (from maxcpus mock)
        assert args.nprocs == 4

        # cleanup_shm should have been called 3 times (fitNorm, fitcoeff, lagtc)
        assert mock_cleanup.call_count == 3

    assert maxcpus_called[0], "maxcpus should have been called for nprocs < 1"

    if debug:
        print("retrolagtcs_nprocs_auto passed")


def retrolagtcs_single_proc_no_shm(debug=False):
    """Test that nprocs=1 does not use shared memory or cleanup."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(nprocs=1)

    def mock_readfromnifti(fname, headeronly=False):
        if fname == args.fmrifile:
            return mock_data["fmri"]
        elif fname == args.maskfile:
            return mock_data["mask"]
        elif fname == args.lagtimesfile:
            return mock_data["lagtimes"]
        raise ValueError(f"Unexpected file: {fname}")

    def mock_parseniftidims(dims):
        return int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    def mock_parseniftisizes(sizes):
        return float(sizes[1]), float(sizes[2]), float(sizes[3]), float(sizes[4])

    def mock_allocarray(shape, dtype, shared=False):
        assert not shared, "Single-proc should not use shared memory"
        return np.zeros(shape, dtype=dtype), None

    def mock_makelaggedtcs(genlagtc, timeaxis, lagmask, lagtimes, lagtc, **kwargs):
        for i in range(lagtc.shape[0]):
            lagtc[i, :] = 0.0
        return lagmask.shape[0]

    with (
        patch("rapidtide.workflows.retrolagtcs.tide_io.readfromnifti",
              side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftidims",
              side_effect=mock_parseniftidims),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftisizes",
              side_effect=mock_parseniftisizes),
        patch("rapidtide.workflows.retrolagtcs.tide_io.checkspacematch",
              return_value=True),
        patch("rapidtide.workflows.retrolagtcs.tide_io.savemaplist"),
        patch("rapidtide.workflows.retrolagtcs.tide_resample.FastResamplerFromFile",
              return_value=MagicMock()),
        patch("rapidtide.workflows.retrolagtcs.tide_makelagged.makelaggedtcs",
              side_effect=mock_makelaggedtcs),
        patch("rapidtide.workflows.retrolagtcs.tide_util.allocarray",
              side_effect=mock_allocarray),
        patch("rapidtide.workflows.retrolagtcs.tide_util.cleanup_shm") as mock_cleanup,
        patch("rapidtide.workflows.retrolagtcs.tide_linfitfiltpass.makevoxelspecificderivs"),
    ):
        retrolagtcs(args)
        # cleanup_shm should NOT be called for single-proc
        assert mock_cleanup.call_count == 0

    if debug:
        print("retrolagtcs_single_proc_no_shm passed")


def retrolagtcs_output_root(debug=False):
    """Test that output files use the correct outputroot."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(outputroot="/tmp/myanalysis/results")

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    for call_data in captured["savemaplist_calls"]:
        assert call_data["outputname"] == "/tmp/myanalysis/results"

    if debug:
        print("retrolagtcs_output_root passed")


def retrolagtcs_bids_metadata(debug=False):
    """Test that BIDS metadata includes correct raw sources."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args(
        fmrifile="/data/fmri.nii.gz",
        maskfile="/data/mask.nii.gz",
        lagtimesfile="/data/lagtimes.nii.gz",
        lagtcgeneratorfile="/data/generator_ts",
        outputroot="/data/output",
    )

    # We need to capture the bidsdict - check via savemaplist
    saved_bidsdicts = []

    def mock_savemaplist(outputname, maplist, validvoxels, destshape, theheader,
                         bidsdict, **kwargs):
        saved_bidsdicts.append(bidsdict.copy())

    def mock_readfromnifti(fname, headeronly=False):
        if fname == args.fmrifile:
            return mock_data["fmri"]
        elif fname == args.maskfile:
            return mock_data["mask"]
        elif fname == args.lagtimesfile:
            return mock_data["lagtimes"]
        raise ValueError(f"Unexpected file: {fname}")

    def mock_parseniftidims(dims):
        return int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    def mock_parseniftisizes(sizes):
        return float(sizes[1]), float(sizes[2]), float(sizes[3]), float(sizes[4])

    def mock_allocarray(shape, dtype, shared=False):
        return np.zeros(shape, dtype=dtype), None

    def mock_makelaggedtcs(genlagtc, timeaxis, lagmask, lagtimes, lagtc, **kwargs):
        for i in range(lagtc.shape[0]):
            lagtc[i, :] = 0.0
        return lagmask.shape[0]

    with (
        patch("rapidtide.workflows.retrolagtcs.tide_io.readfromnifti",
              side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftidims",
              side_effect=mock_parseniftidims),
        patch("rapidtide.workflows.retrolagtcs.tide_io.parseniftisizes",
              side_effect=mock_parseniftisizes),
        patch("rapidtide.workflows.retrolagtcs.tide_io.checkspacematch",
              return_value=True),
        patch("rapidtide.workflows.retrolagtcs.tide_io.savemaplist",
              side_effect=mock_savemaplist),
        patch("rapidtide.workflows.retrolagtcs.tide_resample.FastResamplerFromFile",
              return_value=MagicMock()),
        patch("rapidtide.workflows.retrolagtcs.tide_makelagged.makelaggedtcs",
              side_effect=mock_makelaggedtcs),
        patch("rapidtide.workflows.retrolagtcs.tide_util.allocarray",
              side_effect=mock_allocarray),
        patch("rapidtide.workflows.retrolagtcs.tide_linfitfiltpass.makevoxelspecificderivs"),
    ):
        retrolagtcs(args)

    assert len(saved_bidsdicts) >= 1
    bids = saved_bidsdicts[0]
    assert "RawSources" in bids
    assert "Units" in bids
    assert bids["Units"] == "arbitrary"
    assert "CommandLineArgs" in bids

    if debug:
        print("retrolagtcs_bids_metadata passed")


def retrolagtcs_validvoxels_passed_to_savemaplist(debug=False):
    """Test that validvoxels array is correctly passed to savemaplist."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args()

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    for call_data in captured["savemaplist_calls"]:
        vv = call_data["validvoxels"]
        expected = mock_data["active_voxels"]
        np.testing.assert_array_equal(vv, expected)

    if debug:
        print("retrolagtcs_validvoxels_passed_to_savemaplist passed")


def retrolagtcs_prints_progress(debug=False):
    """Test that retrolagtcs prints progress messages."""
    mock_data = _make_mock_nifti_data()
    args = _make_default_args()

    captured_stdout = io.StringIO()
    with patch("sys.stdout", captured_stdout):
        _run_retrolagtcs_with_mocks(args, mock_data)

    output = captured_stdout.getvalue()
    assert "reading fmrifile header" in output
    assert "reading procfit maskfile" in output
    assert "reading lagtimes" in output
    assert "reading lagtc generator" in output
    assert "figuring out valid voxels" in output
    assert "selecting valid voxels" in output
    assert "calling makelaggedtcs" in output
    assert "generated regressors for" in output

    if debug:
        print("retrolagtcs_prints_progress passed")


def retrolagtcs_different_spatial_dims(debug=False):
    """Test retrolagtcs with non-square spatial dimensions."""
    mock_data = _make_mock_nifti_data(xsize=8, ysize=6, numslices=2, timepoints=50)
    args = _make_default_args()

    captured = _run_retrolagtcs_with_mocks(args, mock_data)

    call_4d = captured["savemaplist_calls"][0]
    assert call_4d["destshape"] == (8, 6, 2, 50)

    if debug:
        print("retrolagtcs_different_spatial_dims passed")


def retrolagtcs_fmri_time_axis(debug=False):
    """Test that the fMRI time axis is constructed correctly."""
    mock_data = _make_mock_nifti_data(timepoints=100, fmritr=2.0)
    args = _make_default_args(numskip=5)

    captured_timeaxis = {}

    def mock_makelaggedtcs(genlagtc, timeaxis, lagmask, lagtimes, lagtc, **kwargs):
        captured_timeaxis["timeaxis"] = timeaxis.copy()
        for i in range(lagtc.shape[0]):
            lagtc[i, :] = 0.0
        return lagmask.shape[0]

    captured = _run_retrolagtcs_with_mocks(args, mock_data,
                                            makelagged_side_effect=mock_makelaggedtcs)

    t = captured_timeaxis["timeaxis"]
    # validtimepoints = 100 - 5 = 95
    assert len(t) == 95
    # skiptime = 5 * 2.0 = 10.0
    assert np.isclose(t[0], 10.0)
    # endpoint=False so last point = skiptime + (validtimepoints-1) * tr
    expected_last = 10.0 + 94 * 2.0
    assert np.isclose(t[-1], expected_last)

    if debug:
        print("retrolagtcs_fmri_time_axis passed")


# ---- main test function ----


def test_retrolagtcs(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_regressderivs(debug=debug)
    parser_nprocs(debug=debug)
    parser_numskip(debug=debug)
    parser_noprogressbar(debug=debug)
    parser_debug(debug=debug)
    parser_fmrifile_validation(debug=debug)
    parser_default_constant(debug=debug)

    # retrolagtcs workflow tests
    retrolagtcs_basic(debug=debug)
    retrolagtcs_makelaggedtcs_args(debug=debug)
    retrolagtcs_numskip(debug=debug)
    retrolagtcs_voxel_selection(debug=debug)
    retrolagtcs_no_derivs(debug=debug)
    retrolagtcs_with_derivs(debug=debug)
    retrolagtcs_with_one_deriv(debug=debug)
    retrolagtcs_debug_mode(debug=debug)
    retrolagtcs_debug_output(debug=debug)
    retrolagtcs_space_mismatch_mask(debug=debug)
    retrolagtcs_space_mismatch_lagtimes(debug=debug)
    retrolagtcs_nprocs_auto(debug=debug)
    retrolagtcs_single_proc_no_shm(debug=debug)
    retrolagtcs_output_root(debug=debug)
    retrolagtcs_bids_metadata(debug=debug)
    retrolagtcs_validvoxels_passed_to_savemaplist(debug=debug)
    retrolagtcs_prints_progress(debug=debug)
    retrolagtcs_different_spatial_dims(debug=debug)
    retrolagtcs_fmri_time_axis(debug=debug)


if __name__ == "__main__":
    test_retrolagtcs(debug=True)
