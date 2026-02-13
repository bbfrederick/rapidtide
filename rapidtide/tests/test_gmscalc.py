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
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.gmscalc import (
    _get_parser,
    gmscalc_main,
    makecommandlinelist,
)


def _capture_writevec():
    """Return a side_effect function and its captured data list.

    gmscalc_main reuses the same writevec call pattern; capture copies
    of the data at call time so each entry is independent.
    """
    captured = []

    def _side_effect(data, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _make_mock_nifti_data(xsize=3, ysize=3, numslices=2, timepoints=50, rng=None):
    """Create synthetic 4D fMRI data and matching NIfTI metadata.

    Returns (img, data, hdr, dims, sizes) matching readfromnifti output.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    data = rng.randn(xsize, ysize, numslices, timepoints).astype(np.float64)
    # Add a global mean signal: slow oscillation present in all voxels
    t = np.arange(timepoints)
    gms_signal = np.sin(2.0 * np.pi * 0.05 * t)
    data += gms_signal[np.newaxis, np.newaxis, np.newaxis, :]

    img = MagicMock()
    hdr = MagicMock()
    dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    sizes = np.array([4, 2.0, 2.0, 3.0, 1.5, 1, 1, 1])  # voxel sizes: 2mm x 2mm x 3mm, TR=1.5s

    return img, data, hdr, dims, sizes


# ---- _get_parser tests ----


def test_get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    if debug:
        print("Parser created successfully")


def test_get_parser_required_args(debug=False):
    """Test that parser requires the two positional arguments."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_get_parser_with_valid_args(debug=False):
    """Test parser with valid positional arguments."""
    parser = _get_parser()
    args = parser.parse_args(["data.nii.gz", "output_root"])
    assert args.datafile == "data.nii.gz"
    assert args.outputroot == "output_root"
    assert args.datamaskname is None
    assert args.normfirst is False
    assert args.sigma == 0.0
    assert args.debug is False

    if debug:
        print(f"Parsed args: {args}")


def test_get_parser_optional_flags(debug=False):
    """Test parser with optional flags."""
    parser = _get_parser()
    args = parser.parse_args(
        [
            "data.nii.gz",
            "out",
            "--normfirst",
            "--smooth",
            "3.5",
            "--debug",
            "--normmethod",
            "percent",
        ]
    )
    assert args.normfirst is True
    assert args.sigma == 3.5
    assert args.debug is True
    assert args.normmethod == "percent"


def test_get_parser_default_normmethod(debug=False):
    """Test that default normmethod is 'None'."""
    parser = _get_parser()
    args = parser.parse_args(["data.nii", "out"])
    assert args.normmethod == "None"


# ---- makecommandlinelist tests ----


def test_makecommandlinelist_without_extra(debug=False):
    """Test command line list generation without extra info."""
    arglist = ["python", "gmscalc", "data.nii", "output"]
    starttime = time.time() - 5.0
    endtime = time.time()

    with patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version:
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        result = makecommandlinelist(arglist, starttime, endtime)

    assert len(result) == 4
    assert result[0].startswith("# Processed on")
    assert "Processing took" in result[1]
    assert "v2.9.0" in result[2]
    assert result[3] == "python gmscalc data.nii output"

    if debug:
        for line in result:
            print(line)


def test_makecommandlinelist_with_extra(debug=False):
    """Test command line list generation with extra info."""
    arglist = ["python", "gmscalc"]
    starttime = time.time() - 2.0
    endtime = time.time()

    with patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version:
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        result = makecommandlinelist(arglist, starttime, endtime, extra="smoothing: 3mm")

    assert len(result) == 5
    assert result[3] == "# smoothing: 3mm"
    assert result[4] == "python gmscalc"

    if debug:
        for line in result:
            print(line)


def test_makecommandlinelist_timing(debug=False):
    """Test that timing information is computed correctly."""
    arglist = ["test"]
    starttime = 1000.0
    endtime = 1010.5

    with patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version:
        mock_version.return_value = ("v1.0", "sha", "date", False)
        result = makecommandlinelist(arglist, starttime, endtime)

    assert "10.500" in result[1]


def test_makecommandlinelist_empty_arglist(debug=False):
    """Test with empty argument list."""
    arglist = []
    starttime = time.time()
    endtime = starttime + 1.0

    with patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version:
        mock_version.return_value = ("v1.0", "sha", "date", False)
        result = makecommandlinelist(arglist, starttime, endtime)

    assert result[-1] == ""


# ---- gmscalc_main tests ----


def test_gmscalc_main_basic(debug=False):
    """Test basic GMS calculation without mask or smoothing."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()
    xsize, ysize, numslices, timepoints = 3, 3, 2, 50

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "output_root"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)

        gmscalc_main()

        # Should write 4 files: gms, gmslfo, gmshf, commandline
        assert len(captured) == 4

        gms_data, gms_filename = captured[0]
        assert gms_filename == "output_root_gms.txt"
        assert len(gms_data) == timepoints

        lfo_data, lfo_filename = captured[1]
        assert lfo_filename == "output_root_gmslfo.txt"
        assert len(lfo_data) == timepoints

        hf_data, hf_filename = captured[2]
        assert hf_filename == "output_root_gmshf.txt"
        assert len(hf_data) == timepoints

        cmd_data, cmd_filename = captured[3]
        assert cmd_filename == "output_root_commandline.txt"

        if debug:
            print(f"GMS shape: {gms_data.shape}")
            print(f"GMS mean: {np.mean(gms_data):.4f}")
            print(f"LFO range: [{np.min(lfo_data):.4f}, {np.max(lfo_data):.4f}]")


def test_gmscalc_main_gms_is_voxel_mean(debug=False):
    """Test that GMS is the mean across all voxels at each timepoint."""
    rng = np.random.RandomState(99)
    xsize, ysize, numslices, timepoints = 2, 2, 1, 30
    img, data, hdr, dims, sizes = _make_mock_nifti_data(
        xsize=xsize, ysize=ysize, numslices=numslices, timepoints=timepoints, rng=rng
    )

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)

        gmscalc_main()

        gms_data = captured[0][0]

        # Manually compute expected GMS
        numvoxels = xsize * ysize * numslices
        reshaped = data.reshape((numvoxels, timepoints))
        expected_gms = np.mean(reshaped, axis=0)

        np.testing.assert_allclose(gms_data, expected_gms, atol=1e-10)

        if debug:
            print(f"GMS matches expected: max diff = {np.max(np.abs(gms_data - expected_gms))}")


def test_gmscalc_main_with_mask(debug=False):
    """Test GMS calculation with a data mask."""
    rng = np.random.RandomState(77)
    xsize, ysize, numslices, timepoints = 3, 3, 2, 40
    img, data, hdr, dims, sizes = _make_mock_nifti_data(
        xsize=xsize, ysize=ysize, numslices=numslices, timepoints=timepoints, rng=rng
    )

    # Create mask: only the first half of voxels are valid
    mask_data = np.zeros((xsize, ysize, numslices))
    mask_data[:2, :, :] = 1.0  # only first 2 x-slices

    mask_img = MagicMock()
    mask_hdr = MagicMock()
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([3, 2.0, 2.0, 3.0, 1, 1, 1, 1])

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out", "--dmask", "mask.nii.gz"]),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.checkspacematch", return_value=True),
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        # First readfromnifti call returns data, second returns mask
        mock_read.side_effect = [
            (img, data, hdr, dims, sizes),
            (mask_img, mask_data, mask_hdr, mask_dims, mask_sizes),
        ]

        gmscalc_main()

        gms_data = captured[0][0]
        assert len(gms_data) == timepoints

        # Verify GMS uses only masked voxels
        numvoxels = xsize * ysize * numslices
        reshaped = data.reshape((numvoxels, timepoints))
        mask_flat = mask_data.reshape(numvoxels)
        valid = np.where(mask_flat > 0)[0]
        expected_gms = np.mean(reshaped[valid, :], axis=0)

        np.testing.assert_allclose(gms_data, expected_gms, atol=1e-10)

        if debug:
            print(f"Masked GMS: {len(valid)} valid voxels out of {numvoxels}")


def test_gmscalc_main_mask_mismatch(debug=False):
    """Test that mismatched mask dimensions cause sys.exit."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()

    mask_img = MagicMock()
    mask_data = np.ones((3, 3, 2))
    mask_hdr = MagicMock()
    mask_dims = np.array([3, 3, 3, 2, 1, 1, 1, 1])
    mask_sizes = np.array([3, 2.0, 2.0, 3.0, 1, 1, 1, 1])

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out", "--dmask", "mask.nii.gz"]),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.checkspacematch", return_value=False),
        patch("rapidtide.workflows.gmscalc.tide_io.writevec"),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
        patch("rapidtide.workflows.gmscalc.sys.exit", side_effect=SystemExit),
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.side_effect = [
            (img, data, hdr, dims, sizes),
            (mask_img, mask_data, mask_hdr, mask_dims, mask_sizes),
        ]

        with pytest.raises(SystemExit):
            gmscalc_main()


def test_gmscalc_main_with_smoothing(debug=False):
    """Test GMS calculation with spatial smoothing enabled."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()
    timepoints = 50

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out", "--smooth", "4.0"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_filt.ssmooth") as mock_ssmooth,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)
        # ssmooth should return same-shaped array
        mock_ssmooth.side_effect = lambda xd, yd, sd, sigma, d: d

        gmscalc_main()

        # ssmooth should be called once per timepoint
        assert mock_ssmooth.call_count == timepoints
        # Verify sigma was passed correctly
        for call_args in mock_ssmooth.call_args_list:
            assert call_args[0][3] == 4.0  # sigma argument

        assert len(captured) == 4

        if debug:
            print(f"ssmooth called {mock_ssmooth.call_count} times")


def test_gmscalc_main_no_smoothing_when_zero(debug=False):
    """Test that smoothing is skipped when sigma=0."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_filt.ssmooth") as mock_ssmooth,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)

        gmscalc_main()

        # ssmooth should NOT be called when sigma=0
        mock_ssmooth.assert_not_called()


def test_gmscalc_main_with_normmethod(debug=False):
    """Test GMS calculation with a specific normalization method."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()
    timepoints = 50

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out", "--normmethod", "percent"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)

        gmscalc_main()

        assert len(captured) == 4
        # LFO and HF outputs should be normalized (finite values)
        lfo_data = captured[1][0]
        hf_data = captured[2][0]
        assert np.all(np.isfinite(lfo_data))
        assert np.all(np.isfinite(hf_data))

        if debug:
            print(f"LFO range with percent norm: [{np.min(lfo_data):.4f}, {np.max(lfo_data):.4f}]")


def test_gmscalc_main_output_filenames(debug=False):
    """Test that output files have correct names."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "/tmp/myoutput"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)

        gmscalc_main()

        filenames = [c[1] for c in captured]
        assert filenames[0] == "/tmp/myoutput_gms.txt"
        assert filenames[1] == "/tmp/myoutput_gmslfo.txt"
        assert filenames[2] == "/tmp/myoutput_gmshf.txt"
        assert filenames[3] == "/tmp/myoutput_commandline.txt"


def test_gmscalc_main_debug_flag(debug=False):
    """Test that debug flag doesn't cause errors."""
    img, data, hdr, dims, sizes = _make_mock_nifti_data()

    side_effect, captured = _capture_writevec()

    with (
        patch("sys.argv", ["gmscalc", "data.nii.gz", "out", "--debug"]),
        patch("rapidtide.workflows.gmscalc.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.gmscalc.tide_io.writevec", side_effect=side_effect),
        patch("rapidtide.workflows.gmscalc.tide_util.version") as mock_version,
    ):
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        mock_read.return_value = (img, data, hdr, dims, sizes)

        gmscalc_main()

        assert len(captured) == 4


# ---- main test entry point ----


def test_gmscalc(debug=False):
    """Run all gmscalc sub-tests."""
    # _get_parser tests
    test_get_parser_returns_parser(debug=debug)
    test_get_parser_required_args(debug=debug)
    test_get_parser_with_valid_args(debug=debug)
    test_get_parser_optional_flags(debug=debug)
    test_get_parser_default_normmethod(debug=debug)

    # makecommandlinelist tests
    test_makecommandlinelist_without_extra(debug=debug)
    test_makecommandlinelist_with_extra(debug=debug)
    test_makecommandlinelist_timing(debug=debug)
    test_makecommandlinelist_empty_arglist(debug=debug)

    # gmscalc_main tests
    test_gmscalc_main_basic(debug=debug)
    test_gmscalc_main_gms_is_voxel_mean(debug=debug)
    test_gmscalc_main_with_mask(debug=debug)
    test_gmscalc_main_mask_mismatch(debug=debug)
    test_gmscalc_main_with_smoothing(debug=debug)
    test_gmscalc_main_no_smoothing_when_zero(debug=debug)
    test_gmscalc_main_with_normmethod(debug=debug)
    test_gmscalc_main_output_filenames(debug=debug)
    test_gmscalc_main_debug_flag(debug=debug)


if __name__ == "__main__":
    test_gmscalc(debug=True)
