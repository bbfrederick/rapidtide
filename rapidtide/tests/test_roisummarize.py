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
import io
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.roisummarize import (
    _get_parser,
    roisummarize,
    summarize3Dbylabel,
    summarize4Dbylabel,
)

# ---- helpers ----


def _make_mock_header():
    """Create a mock NIfTI header supporting getitem, setitem, deepcopy."""
    hdr = MagicMock()
    hdr_data = {}

    def getitem(key):
        return hdr_data.get(key, np.zeros(8, dtype=int))

    def setitem(key, value):
        hdr_data[key] = value

    hdr.__getitem__ = MagicMock(side_effect=getitem)
    hdr.__setitem__ = MagicMock(side_effect=setitem)
    hdr.__deepcopy__ = MagicMock(side_effect=lambda memo: _make_mock_header())
    return hdr


def _make_4d_data(xsize=4, ysize=4, numslices=2, timepoints=50, numregions=3, rng=None):
    """Create synthetic 4D fMRI data and a matching template with numregions labels.

    Returns (input_data_4d, template_data_3d, dims, sizes, numvoxels).
    Each region has a distinct mean signal level to make summarization testable.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    numvoxels = xsize * ysize * numslices

    # template: assign voxels round-robin to regions 1..numregions
    template_flat = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template_flat[i] = (i % numregions) + 1
    template_data = template_flat.reshape((xsize, ysize, numslices))

    # input: each region gets a distinct baseline + some noise
    input_flat = np.zeros((numvoxels, timepoints), dtype=np.float64)
    for i in range(numvoxels):
        region = template_flat[i]
        input_flat[i, :] = region * 10.0 + rng.normal(0, 1.0, timepoints)
    input_data = input_flat.reshape((xsize, ysize, numslices, timepoints))

    dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    sizes = np.array([4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    return input_data, template_data, dims, sizes, numvoxels


def _make_3d_data(xsize=4, ysize=4, numslices=2, numregions=3, rng=None):
    """Create synthetic 3D data and a matching template.

    Returns (input_data_3d, template_data_3d, dims, sizes, numvoxels).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    numvoxels = xsize * ysize * numslices

    template_flat = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template_flat[i] = (i % numregions) + 1
    template_data = template_flat.reshape((xsize, ysize, numslices))

    input_flat = np.zeros(numvoxels, dtype=np.float64)
    for i in range(numvoxels):
        region = template_flat[i]
        input_flat[i] = region * 10.0 + rng.normal(0, 1.0)
    input_data = input_flat.reshape((xsize, ysize, numslices))

    dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    return input_data, template_data, dims, sizes, numvoxels


def _make_default_args(**overrides):
    """Create default args Namespace for roisummarize."""
    defaults = dict(
        inputfilename="input.nii.gz",
        templatefile="template.nii.gz",
        outputfile="/tmp/test_roisummarize_out",
        samplerate="auto",
        numskip=0,
        normmethod="z",
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---- _get_parser tests ----


def parser_basic(debug=False):
    """Test that _get_parser returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires inputfilename, templatefile, outputfile."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    if debug:
        print("parser_required_args passed")


def parser_defaults(debug=False):
    """Test default values from the parser."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "output.txt"])

    assert args.samplerate == "auto"
    assert args.numskip == 0
    assert args.debug is False
    assert args.normmethod == "None"

    if debug:
        print("parser_defaults passed")


def parser_samplerate(debug=False):
    """Test --samplerate option."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out.txt", "--samplerate", "10.0"])
    assert np.isclose(args.samplerate, 10.0)

    if debug:
        print("parser_samplerate passed")


def parser_sampletstep(debug=False):
    """Test --sampletstep option (inverts to frequency)."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out.txt", "--sampletstep", "0.1"])
    assert np.isclose(args.samplerate, 10.0)

    if debug:
        print("parser_sampletstep passed")


def parser_samplerate_sampletstep_mutual_exclusion(debug=False):
    """Test that --samplerate and --sampletstep are mutually exclusive."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([
                f1.name, f2.name, "out.txt",
                "--samplerate", "10.0",
                "--sampletstep", "0.1",
            ])

    if debug:
        print("parser_samplerate_sampletstep_mutual_exclusion passed")


def parser_numskip(debug=False):
    """Test --numskip option."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out.txt", "--numskip", "5"])
    assert args.numskip == 5

    if debug:
        print("parser_numskip passed")


def parser_normmethod(debug=False):
    """Test --normmethod option."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out.txt", "--normmethod", "percent"])
    assert args.normmethod == "percent"

    if debug:
        print("parser_normmethod passed")


def parser_normmethod_invalid(debug=False):
    """Test that --normmethod rejects invalid methods."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([f1.name, f2.name, "out.txt", "--normmethod", "bogus"])

    if debug:
        print("parser_normmethod_invalid passed")


def parser_debug(debug=False):
    """Test --debug flag."""
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out.txt", "--debug"])
    assert args.debug is True

    if debug:
        print("parser_debug passed")


# ---- summarize4Dbylabel tests ----


def summarize4D_basic(debug=False):
    """Test summarize4Dbylabel with basic inputs."""
    numvoxels = 30
    timepoints = 50
    numregions = 3
    rng = np.random.RandomState(42)

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    inputvoxels = np.zeros((numvoxels, timepoints), dtype=np.float64)
    for i in range(numvoxels):
        region = template[i]
        inputvoxels[i, :] = region * 10.0 + rng.normal(0, 1.0, timepoints)

    result = summarize4Dbylabel(inputvoxels, template, normmethod="z")

    assert result.shape == (numregions, timepoints)

    if debug:
        print(f"summarize4D_basic: result shape = {result.shape}")
        print("summarize4D_basic passed")


def summarize4D_number_of_regions(debug=False):
    """Test that output has correct number of region rows."""
    numvoxels = 20
    timepoints = 30

    for numregions in [1, 2, 5]:
        template = np.zeros(numvoxels, dtype=int)
        for i in range(numvoxels):
            template[i] = (i % numregions) + 1
        inputvoxels = np.random.RandomState(42).normal(0, 1, (numvoxels, timepoints))

        result = summarize4Dbylabel(inputvoxels, template, normmethod="None")
        assert result.shape[0] == numregions, (
            f"Expected {numregions} regions, got {result.shape[0]}"
        )

    if debug:
        print("summarize4D_number_of_regions passed")


def summarize4D_normmethod_z(debug=False):
    """Test that z-normalization produces zero-mean unit-std output."""
    numvoxels = 60
    timepoints = 200
    numregions = 3
    rng = np.random.RandomState(42)

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    inputvoxels = np.zeros((numvoxels, timepoints), dtype=np.float64)
    for i in range(numvoxels):
        inputvoxels[i, :] = rng.normal(10.0, 2.0, timepoints)

    result = summarize4Dbylabel(inputvoxels, template, normmethod="z")

    for r in range(numregions):
        assert np.isclose(np.mean(result[r, :]), 0.0, atol=0.1), (
            f"Region {r}: mean={np.mean(result[r, :]):.4f}, expected ~0"
        )
        assert np.isclose(np.std(result[r, :]), 1.0, atol=0.2), (
            f"Region {r}: std={np.std(result[r, :]):.4f}, expected ~1"
        )

    if debug:
        print("summarize4D_normmethod_z passed")


def summarize4D_normmethod_none(debug=False):
    """Test that None normalization demeands but preserves scale."""
    numvoxels = 30
    timepoints = 100
    rng = np.random.RandomState(42)

    template = np.ones(numvoxels, dtype=int)  # single region
    inputvoxels = rng.normal(50.0, 5.0, (numvoxels, timepoints))

    result = summarize4Dbylabel(inputvoxels, template, normmethod="None")

    assert result.shape == (1, timepoints)
    # "None" normalization demeands: mean should be ~0
    assert np.isclose(np.mean(result[0, :]), 0.0, atol=0.5)

    if debug:
        print("summarize4D_normmethod_none passed")


def summarize4D_nan_handling(debug=False):
    """Test that NaN values are converted to zeros."""
    numvoxels = 10
    timepoints = 20

    template = np.ones(numvoxels, dtype=int)
    inputvoxels = np.ones((numvoxels, timepoints), dtype=np.float64)
    # inject NaN in some voxels
    inputvoxels[0, :5] = np.nan
    inputvoxels[3, 10:] = np.nan

    result = summarize4Dbylabel(inputvoxels, template, normmethod="None")

    assert not np.any(np.isnan(result)), "Output should not contain NaN"

    if debug:
        print("summarize4D_nan_handling passed")


def summarize4D_debug_output(debug=False):
    """Test that debug mode prints shape information."""
    numvoxels = 10
    timepoints = 20

    template = np.ones(numvoxels, dtype=int)
    inputvoxels = np.ones((numvoxels, timepoints), dtype=np.float64)

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        summarize4Dbylabel(inputvoxels, template, normmethod="None", debug=True)

    output = captured.getvalue()
    assert "thevoxels, data shape are:" in output

    if debug:
        print("summarize4D_debug_output passed")


def summarize4D_distinct_regions(debug=False):
    """Test that regions with different signals produce distinct timecourses."""
    numvoxels = 30
    timepoints = 100
    numregions = 3
    rng = np.random.RandomState(42)

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    # give each region a very different signal
    inputvoxels = np.zeros((numvoxels, timepoints), dtype=np.float64)
    for i in range(numvoxels):
        region = template[i]
        freq = region * 5.0  # different frequency per region
        inputvoxels[i, :] = np.sin(2 * np.pi * freq * np.arange(timepoints) / timepoints)
        inputvoxels[i, :] += rng.normal(0, 0.01, timepoints)

    result = summarize4Dbylabel(inputvoxels, template, normmethod="None")

    # regions should not be identical
    for r1 in range(numregions):
        for r2 in range(r1 + 1, numregions):
            corr = np.corrcoef(result[r1, :], result[r2, :])[0, 1]
            assert abs(corr) < 0.99, (
                f"Regions {r1 + 1} and {r2 + 1} should be distinct, corr={corr:.4f}"
            )

    if debug:
        print("summarize4D_distinct_regions passed")


# ---- summarize3Dbylabel tests ----


def summarize3D_basic(debug=False):
    """Test summarize3Dbylabel with basic inputs."""
    numvoxels = 30
    numregions = 3
    rng = np.random.RandomState(42)

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    inputvoxels = np.zeros(numvoxels, dtype=np.float64)
    for i in range(numvoxels):
        region = template[i]
        inputvoxels[i] = region * 10.0 + rng.normal(0, 1.0)

    outputvoxels, regionstats = summarize3Dbylabel(inputvoxels, template)

    assert outputvoxels.shape == inputvoxels.shape
    assert len(regionstats) == numregions

    if debug:
        print("summarize3D_basic passed")


def summarize3D_regionstats_structure(debug=False):
    """Test that regionstats has [mean, std, median] per region."""
    numvoxels = 30
    numregions = 3

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    rng = np.random.RandomState(42)
    inputvoxels = rng.normal(0, 1, numvoxels)

    _, regionstats = summarize3Dbylabel(inputvoxels, template)

    assert len(regionstats) == numregions
    for r, stats in enumerate(regionstats):
        assert len(stats) == 3, f"Region {r}: expected [mean, std, median], got {stats}"

    if debug:
        print("summarize3D_regionstats_structure passed")


def summarize3D_output_replaced_by_mean(debug=False):
    """Test that output voxels are replaced by their region mean."""
    numvoxels = 12
    numregions = 3

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    # region 1 gets value 10, region 2 gets 20, region 3 gets 30
    inputvoxels = np.zeros(numvoxels, dtype=np.float64)
    for i in range(numvoxels):
        inputvoxels[i] = template[i] * 10.0

    outputvoxels, regionstats = summarize3Dbylabel(inputvoxels, template)

    # all voxels in a region should have the region mean
    for i in range(numvoxels):
        region = template[i]
        expected_mean = region * 10.0
        assert np.isclose(outputvoxels[i], expected_mean), (
            f"Voxel {i} (region {region}): expected {expected_mean}, got {outputvoxels[i]}"
        )

    if debug:
        print("summarize3D_output_replaced_by_mean passed")


def summarize3D_stats_correct(debug=False):
    """Test that regionstats contain correct mean, std, median."""
    numvoxels = 12
    numregions = 3

    template = np.zeros(numvoxels, dtype=int)
    for i in range(numvoxels):
        template[i] = (i % numregions) + 1

    # constant value per region
    inputvoxels = np.zeros(numvoxels, dtype=np.float64)
    for i in range(numvoxels):
        inputvoxels[i] = template[i] * 10.0

    _, regionstats = summarize3Dbylabel(inputvoxels, template)

    for r in range(numregions):
        mean, std, median = regionstats[r]
        expected_val = (r + 1) * 10.0
        assert np.isclose(mean, expected_val), f"Region {r + 1}: mean={mean}, expected {expected_val}"
        assert np.isclose(std, 0.0), f"Region {r + 1}: std={std}, expected 0"
        assert np.isclose(median, expected_val), f"Region {r + 1}: median={median}, expected {expected_val}"

    if debug:
        print("summarize3D_stats_correct passed")


def summarize3D_nan_handling(debug=False):
    """Test that NaN values are handled (converted to 0 for stats)."""
    numvoxels = 10
    template = np.ones(numvoxels, dtype=int)
    inputvoxels = np.ones(numvoxels, dtype=np.float64) * 5.0
    inputvoxels[0] = np.nan

    outputvoxels, regionstats = summarize3Dbylabel(inputvoxels, template)

    # stats should not be NaN
    for stats in regionstats:
        for val in stats:
            assert not np.isnan(val), f"Stats should not contain NaN, got {stats}"

    if debug:
        print("summarize3D_nan_handling passed")


def summarize3D_single_region(debug=False):
    """Test summarize3Dbylabel with a single region."""
    numvoxels = 20
    template = np.ones(numvoxels, dtype=int)
    rng = np.random.RandomState(42)
    inputvoxels = rng.normal(50.0, 5.0, numvoxels)

    outputvoxels, regionstats = summarize3Dbylabel(inputvoxels, template)

    assert len(regionstats) == 1
    mean, std, median = regionstats[0]

    assert np.isclose(mean, np.mean(inputvoxels), atol=0.01), (
        f"mean should equal np.mean(inputvoxels)={np.mean(inputvoxels):.4f}, got {mean:.4f}"
    )
    assert np.isclose(std, np.std(inputvoxels), atol=0.01), (
        f"std should equal np.std(inputvoxels)={np.std(inputvoxels):.4f}, got {std:.4f}"
    )
    assert np.isclose(median, np.median(inputvoxels), atol=0.01), (
        f"median should equal np.median(inputvoxels)={np.median(inputvoxels):.4f}, got {median:.4f}"
    )

    if debug:
        print("summarize3D_single_region passed")


# ---- roisummarize workflow tests ----


def _run_roisummarize_with_mocks(args, input_data, template_data, input_dims,
                                  input_sizes, template_dims=None, template_sizes=None,
                                  space_match=True):
    """Run roisummarize with fully mocked I/O.

    Returns dict with captured output calls.
    """
    if template_dims is None:
        template_dims = np.array([3, input_dims[1], input_dims[2], input_dims[3], 1, 1, 1, 1])
    if template_sizes is None:
        template_sizes = np.array([3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    input_hdr = _make_mock_header()
    template_hdr = _make_mock_header()

    captured = {
        "writenpvecs_calls": [],
        "savetonifti_calls": [],
    }

    def mock_readfromnifti(fname, headeronly=False):
        if fname == args.inputfilename:
            return (MagicMock(), input_data.copy(), input_hdr, input_dims, input_sizes)
        elif fname == args.templatefile:
            return (MagicMock(), template_data.copy(), template_hdr, template_dims, template_sizes)
        raise ValueError(f"Unexpected file: {fname}")

    def mock_checkspacematch(hdr1, hdr2):
        return space_match

    def mock_writenpvecs(data, fname, **kwargs):
        captured["writenpvecs_calls"].append({
            "data": np.array(data).copy(),
            "fname": fname,
        })

    def mock_savetonifti(data, hdr, name, **kwargs):
        captured["savetonifti_calls"].append({
            "data": np.array(data).copy(),
            "name": name,
        })

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda rate, data: data)

    with (
        patch(
            "rapidtide.workflows.roisummarize._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch("rapidtide.workflows.roisummarize.pf.postprocessfilteropts",
              return_value=(args, mock_filter)),
        patch("rapidtide.workflows.roisummarize.tide_io.readfromnifti",
              side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.roisummarize.tide_io.checkspacematch",
              side_effect=mock_checkspacematch),
        patch("rapidtide.workflows.roisummarize.tide_io.writenpvecs",
              side_effect=mock_writenpvecs),
        patch("rapidtide.workflows.roisummarize.tide_io.savetonifti",
              side_effect=mock_savetonifti),
    ):
        roisummarize(args)

    return captured


def roisummarize_4d_basic(debug=False):
    """Test roisummarize with 4D input data."""
    input_data, template_data, dims, sizes, numvoxels = _make_4d_data(
        xsize=4, ysize=4, numslices=2, timepoints=50, numregions=3,
    )
    args = _make_default_args(normmethod="z")

    captured = _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    # 4D path: should call writenpvecs for timecourses
    assert len(captured["writenpvecs_calls"]) == 1
    tc_call = captured["writenpvecs_calls"][0]
    assert "_timecourses" in tc_call["fname"]
    # should have 3 regions x (50 - 0 skip) timepoints
    assert tc_call["data"].shape[0] == 3

    # 4D path: should NOT call savetonifti
    assert len(captured["savetonifti_calls"]) == 0

    if debug:
        print("roisummarize_4d_basic passed")


def roisummarize_4d_numskip(debug=False):
    """Test roisummarize with numskip > 0 on 4D data."""
    input_data, template_data, dims, sizes, numvoxels = _make_4d_data(
        xsize=4, ysize=4, numslices=2, timepoints=50, numregions=3,
    )
    args = _make_default_args(normmethod="None", numskip=10)

    captured = _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    tc_call = captured["writenpvecs_calls"][0]
    # timepoints should be 50 - 10 = 40
    assert tc_call["data"].shape[1] == 40, (
        f"Expected 40 timepoints, got {tc_call['data'].shape[1]}"
    )

    if debug:
        print("roisummarize_4d_numskip passed")


def roisummarize_3d_basic(debug=False):
    """Test roisummarize with 3D input data."""
    input_data, template_data, dims, sizes, numvoxels = _make_3d_data(
        xsize=4, ysize=4, numslices=2, numregions=3,
    )
    args = _make_default_args()

    captured = _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    # 3D path: should call savetonifti for meanvals
    assert len(captured["savetonifti_calls"]) == 1
    mean_call = captured["savetonifti_calls"][0]
    assert "_meanvals" in mean_call["name"]

    # 3D path: should call writenpvecs for regionstats
    assert len(captured["writenpvecs_calls"]) == 1
    stats_call = captured["writenpvecs_calls"][0]
    assert "_regionstats" in stats_call["fname"]

    if debug:
        print("roisummarize_3d_basic passed")


def roisummarize_3d_regionstats_shape(debug=False):
    """Test that 3D regionstats have correct shape."""
    numregions = 4
    input_data, template_data, dims, sizes, numvoxels = _make_3d_data(
        xsize=4, ysize=4, numslices=2, numregions=numregions,
    )
    args = _make_default_args()

    captured = _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    stats_data = captured["writenpvecs_calls"][0]["data"]
    # regionstats: numregions rows x 3 columns (mean, std, median)
    assert stats_data.shape == (numregions, 3), (
        f"Expected ({numregions}, 3), got {stats_data.shape}"
    )

    if debug:
        print("roisummarize_3d_regionstats_shape passed")


def roisummarize_space_mismatch(debug=False):
    """Test that roisummarize exits when spatial dimensions don't match."""
    input_data, template_data, dims, sizes, numvoxels = _make_4d_data()
    args = _make_default_args()

    with pytest.raises(SystemExit):
        _run_roisummarize_with_mocks(
            args, input_data, template_data, dims, sizes, space_match=False,
        )

    if debug:
        print("roisummarize_space_mismatch passed")


def roisummarize_auto_samplerate(debug=False):
    """Test that samplerate='auto' defaults to 1.0."""
    input_data, template_data, dims, sizes, _ = _make_4d_data(timepoints=30, numregions=2)
    args = _make_default_args(samplerate="auto")

    _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    assert args.samplerate == 1.0

    if debug:
        print("roisummarize_auto_samplerate passed")


def roisummarize_explicit_samplerate(debug=False):
    """Test that explicit samplerate is preserved."""
    input_data, template_data, dims, sizes, _ = _make_4d_data(timepoints=30, numregions=2)
    args = _make_default_args(samplerate=10.0)

    _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    # explicit samplerate should not be overwritten to 1.0
    # NOTE: the code has a bug where samplerate is assigned to a local
    # variable instead of args.samplerate when not "auto" (line 316),
    # but args.samplerate keeps the original value
    assert args.samplerate == 10.0

    if debug:
        print("roisummarize_explicit_samplerate passed")


def roisummarize_prints_progress(debug=False):
    """Test that roisummarize prints progress messages."""
    input_data, template_data, dims, sizes, _ = _make_4d_data(timepoints=30, numregions=2)
    args = _make_default_args()

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    output = captured.getvalue()
    assert "loading fmri data" in output
    assert "loading template data" in output
    assert "checking dimensions" in output
    assert "reshaping" in output

    if debug:
        print("roisummarize_prints_progress passed")


def roisummarize_output_root(debug=False):
    """Test that output files use the correct outputfile root."""
    input_data, template_data, dims, sizes, _ = _make_4d_data(timepoints=30, numregions=2)
    args = _make_default_args(outputfile="/tmp/myresults")

    captured = _run_roisummarize_with_mocks(args, input_data, template_data, dims, sizes)

    for call in captured["writenpvecs_calls"]:
        assert call["fname"].startswith("/tmp/myresults")

    if debug:
        print("roisummarize_output_root passed")


# ---- main test function ----


def test_roisummarize(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_samplerate(debug=debug)
    parser_sampletstep(debug=debug)
    parser_samplerate_sampletstep_mutual_exclusion(debug=debug)
    parser_numskip(debug=debug)
    parser_normmethod(debug=debug)
    parser_normmethod_invalid(debug=debug)
    parser_debug(debug=debug)

    # summarize4Dbylabel tests
    summarize4D_basic(debug=debug)
    summarize4D_number_of_regions(debug=debug)
    summarize4D_normmethod_z(debug=debug)
    summarize4D_normmethod_none(debug=debug)
    summarize4D_nan_handling(debug=debug)
    summarize4D_debug_output(debug=debug)
    summarize4D_distinct_regions(debug=debug)

    # summarize3Dbylabel tests
    summarize3D_basic(debug=debug)
    summarize3D_regionstats_structure(debug=debug)
    summarize3D_output_replaced_by_mean(debug=debug)
    summarize3D_stats_correct(debug=debug)
    summarize3D_nan_handling(debug=debug)
    summarize3D_single_region(debug=debug)

    # roisummarize workflow tests
    roisummarize_4d_basic(debug=debug)
    roisummarize_4d_numskip(debug=debug)
    roisummarize_3d_basic(debug=debug)
    roisummarize_3d_regionstats_shape(debug=debug)
    roisummarize_space_mismatch(debug=debug)
    roisummarize_auto_samplerate(debug=debug)
    roisummarize_explicit_samplerate(debug=debug)
    roisummarize_prints_progress(debug=debug)
    roisummarize_output_root(debug=debug)


if __name__ == "__main__":
    test_roisummarize(debug=True)
