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
import os
from unittest.mock import MagicMock, call, patch

import numpy as np

from rapidtide.tests.utils import create_dir, get_test_temp_path, mse
from rapidtide.workflows.fdica import P2R, R2P, _get_parser, fdica, main

# ==================== Tests for P2R ====================


def P2R_zero_angle(debug=False):
    """Test P2R with zero angle returns real number."""
    result = P2R(1.0, 0.0)
    assert np.isclose(result, 1.0 + 0.0j)


def P2R_right_angle(debug=False):
    """Test P2R with pi/2 angle returns imaginary number."""
    result = P2R(2.0, np.pi / 2)
    assert np.isclose(result.imag, 2.0, atol=1e-10)
    assert np.isclose(result.real, 0.0, atol=1e-10)


def P2R_pi_angle(debug=False):
    """Test P2R with pi angle returns negative real number."""
    result = P2R(3.0, np.pi)
    assert np.isclose(result.real, -3.0, atol=1e-10)
    assert np.isclose(result.imag, 0.0, atol=1e-10)


def P2R_array_input(debug=False):
    """Test P2R with numpy array inputs."""
    radii = np.array([1.0, 2.0, 3.0])
    angles = np.array([0.0, np.pi / 2, np.pi])
    result = P2R(radii, angles)
    assert result.shape == (3,)
    assert np.isclose(result[0], 1.0 + 0.0j)
    assert np.isclose(result[1].imag, 2.0, atol=1e-10)
    assert np.isclose(result[2].real, -3.0, atol=1e-10)


def P2R_zero_radius(debug=False):
    """Test P2R with zero radius returns zero regardless of angle."""
    result = P2R(0.0, 1.23)
    assert np.isclose(result, 0.0 + 0.0j)


def P2R_2d_array(debug=False):
    """Test P2R with 2D array inputs."""
    radii = np.ones((3, 4))
    angles = np.zeros((3, 4))
    result = P2R(radii, angles)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(result.real, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.imag, 0.0, atol=1e-10)


# ==================== Tests for R2P ====================


def R2P_real_positive(debug=False):
    """Test R2P with positive real numbers."""
    mag, angle = R2P(np.array([3.0 + 0.0j]))
    assert np.isclose(mag[0], 3.0)
    assert np.isclose(angle[0], 0.0)


def R2P_real_negative(debug=False):
    """Test R2P with negative real numbers."""
    mag, angle = R2P(np.array([-2.0 + 0.0j]))
    assert np.isclose(mag[0], 2.0)
    assert np.isclose(np.abs(angle[0]), np.pi)


def R2P_pure_imaginary(debug=False):
    """Test R2P with pure imaginary numbers."""
    mag, angle = R2P(np.array([0.0 + 5.0j]))
    assert np.isclose(mag[0], 5.0)
    assert np.isclose(angle[0], np.pi / 2)


def R2P_complex(debug=False):
    """Test R2P with general complex number (3+4j has magnitude 5)."""
    mag, angle = R2P(np.array([3.0 + 4.0j]))
    assert np.isclose(mag[0], 5.0)
    assert np.isclose(angle[0], np.arctan2(4.0, 3.0))


def R2P_array(debug=False):
    """Test R2P with array of complex numbers."""
    x = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
    mag, angle = R2P(x)
    np.testing.assert_allclose(mag, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(angle, [0.0, np.pi / 2, np.pi])


def R2P_zero(debug=False):
    """Test R2P with zero."""
    mag, angle = R2P(np.array([0.0 + 0.0j]))
    assert np.isclose(mag[0], 0.0)
    assert np.isclose(angle[0], 0.0)


# ==================== Tests for P2R/R2P roundtrip ====================


def P2R_R2P_roundtrip(debug=False):
    """Test that P2R and R2P are inverse operations."""
    radii = np.array([1.0, 2.5, 0.5, 3.0])
    angles = np.array([0.3, -1.2, np.pi / 4, -np.pi / 3])
    complex_vals = P2R(radii, angles)
    recovered_radii, recovered_angles = R2P(complex_vals)
    np.testing.assert_allclose(recovered_radii, radii, atol=1e-10)
    np.testing.assert_allclose(recovered_angles, angles, atol=1e-10)


def R2P_P2R_roundtrip(debug=False):
    """Test that R2P then P2R recovers the original complex values."""
    original = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.5 - 0.5j])
    mag, angle = R2P(original)
    recovered = P2R(mag, angle)
    np.testing.assert_allclose(recovered.real, original.real, atol=1e-10)
    np.testing.assert_allclose(recovered.imag, original.imag, atol=1e-10)


# ==================== Tests for _get_parser ====================


def get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_prog_name(debug=False):
    """Test that the parser prog name is correct."""
    parser = _get_parser()
    assert parser.prog == "fdica"


def get_parser_defaults(debug=False):
    """Test that default values are correct."""
    parser = _get_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["gausssigma"] == 0.0
    assert defaults["pcacomponents"] == 0.9
    assert defaults["icacomponents"] is None
    assert defaults["debug"] is False


def get_parser_with_options(testtemproot, debug=False):
    """Test parser with all optional arguments."""
    # Create dummy input files
    datafile = os.path.join(testtemproot, "fdica_data.nii.gz")
    maskfile = os.path.join(testtemproot, "fdica_mask.nii.gz")
    with open(datafile, "w") as f:
        f.write("dummy")
    with open(maskfile, "w") as f:
        f.write("dummy")

    parser = _get_parser()
    args = parser.parse_args([
        datafile,
        maskfile,
        "output_root",
        "--spatialfilt", "2.5",
        "--pcacomponents", "0.95",
        "--icacomponents", "5",
        "--debug",
    ])
    assert args.gausssigma == 2.5
    assert args.pcacomponents == 0.95
    assert args.icacomponents == 5
    assert args.debug is True


def get_parser_required_args(testtemproot, debug=False):
    """Test parser with only required arguments."""
    datafile = os.path.join(testtemproot, "fdica_data2.nii.gz")
    maskfile = os.path.join(testtemproot, "fdica_mask2.nii.gz")
    with open(datafile, "w") as f:
        f.write("dummy")
    with open(maskfile, "w") as f:
        f.write("dummy")

    parser = _get_parser()
    args = parser.parse_args([datafile, maskfile, "myoutput"])
    assert args.datafile == datafile
    assert args.datamask == maskfile
    assert args.outputroot == "myoutput"
    assert args.gausssigma == 0.0


# ==================== Tests for fdica ====================


def _make_mock_nifti_data(xsize=4, ysize=4, numslices=2, timepoints=64):
    """Helper to create mock NIFTI data and headers for fdica tests."""
    data = np.random.randn(xsize, ysize, numslices, timepoints).astype(np.float64)
    mask = np.ones((xsize, ysize, numslices), dtype=np.float64)

    # Create a mock header with required methods
    hdr = MagicMock()
    hdr.get_xyzt_units.return_value = ("mm", "sec")
    hdr.__getitem__ = MagicMock(side_effect=lambda key: {
        "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
        "pixdim": [1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
    }[key])
    hdr.__setitem__ = MagicMock()
    hdr.__deepcopy__ = MagicMock(side_effect=lambda memo: _make_simple_header(
        xsize, ysize, numslices, timepoints
    ))

    dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

    return data, mask, hdr, dims, sizes


def _make_simple_header(xsize, ysize, numslices, timepoints):
    """Create a simple dict-like mock header for deepcopy results."""
    hdr = {}
    hdr["dim"] = [4, xsize, ysize, numslices, timepoints, 1, 1, 1]
    hdr["pixdim"] = [1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    return hdr


def fdica_basic(testtemproot, debug=False):
    """Test fdica with basic parameters and mocked IO."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_basic")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti") as mock_save, \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs") as mock_write, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=0.9,
            debug=debug,
        )

        assert mock_read.call_count == 2
        assert mock_save.call_count > 0
        assert mock_write.call_count > 0


def fdica_with_spatial_filter(testtemproot, debug=False):
    """Test fdica with Gaussian spatial filtering enabled."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_spatfilt")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti"), \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs"), \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check, \
         patch("rapidtide.workflows.fdica.tide_filt.ssmooth") as mock_ssmooth:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True
        mock_ssmooth.side_effect = lambda xd, yd, sd, gs, d: d

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            gausssigma=2.0,
            pcacomponents=0.9,
            debug=debug,
        )

        # ssmooth should be called once per timepoint
        assert mock_ssmooth.call_count == timepoints


def fdica_auto_gausssigma(testtemproot, debug=False):
    """Test fdica with negative gausssigma triggers automatic calculation."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_autogauss")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti"), \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs"), \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check, \
         patch("rapidtide.workflows.fdica.tide_filt.ssmooth") as mock_ssmooth:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True
        mock_ssmooth.side_effect = lambda xd, yd, sd, gs, d: d

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            gausssigma=-1.0,
            pcacomponents=0.9,
            debug=debug,
        )

        # ssmooth should still be called (auto sigma = mean(2,2,2)/2 = 1.0 > 0)
        assert mock_ssmooth.call_count == timepoints


def fdica_with_ica_components(testtemproot, debug=False):
    """Test fdica with explicit ICA component count."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_ica")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti") as mock_save, \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs") as mock_write, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=0.9,
            icacomponents=3,
            debug=debug,
        )

        assert mock_save.call_count > 0


def fdica_with_pca_integer(testtemproot, debug=False):
    """Test fdica with integer PCA components (>=1.0 branch).

    Note: pcacomponents must be passed as int (not float) when >= 1,
    because sklearn PCA requires int for explicit component counts.
    The CLI parser defines this as type=float, which would cause a
    failure when the user passes an integer value like 3 (it becomes 3.0).
    """
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_pcaint")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti"), \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs"), \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=3,
            debug=debug,
        )


def fdica_dim_mismatch_exits(testtemproot, debug=False):
    """Test fdica exits when spatial dimensions don't match."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_mismatch")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check, \
         patch("rapidtide.workflows.fdica.exit") as mock_exit:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = False
        mock_exit.side_effect = SystemExit(0)

        try:
            fdica("dummy_data.nii.gz", "dummy_mask.nii.gz", outputroot)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


def fdica_4d_mask_exits(testtemproot, debug=False):
    """Test fdica exits when mask is 4D instead of 3D."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    # mask with 4th dimension > 1
    mask_dims = np.array([4, xsize, ysize, numslices, 5, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_4dmask")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check, \
         patch("rapidtide.workflows.fdica.sys.exit") as mock_exit:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True
        mock_exit.side_effect = SystemExit(0)

        try:
            fdica("dummy_data.nii.gz", "dummy_mask.nii.gz", outputroot)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


def fdica_msec_tr(testtemproot, debug=False):
    """Test fdica handles TR in milliseconds correctly."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    # Set header to report msec
    hdr.get_xyzt_units.return_value = ("mm", "msec")

    outputroot = os.path.join(testtemproot, "fdica_msec")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti"), \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs"), \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        # TR = 2000 ms -> fmritr = 2.0 s
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2000.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=0.9,
            debug=debug,
        )


def fdica_negative_freq_bounds(testtemproot, debug=False):
    """Test fdica with negative frequency bounds (use all bins)."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_negfreq")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti"), \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs"), \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=0.9,
            lowerfreq=-1.0,
            upperfreq=-1.0,
            debug=debug,
        )


def fdica_output_files(testtemproot, debug=False):
    """Test that fdica saves the expected output files."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data, mask, hdr, dims, sizes = _make_mock_nifti_data(xsize, ysize, numslices, timepoints)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_outputs")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti") as mock_save, \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs") as mock_write, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=0.9,
            debug=debug,
        )

        # Collect the output name suffixes from savetonifti calls
        save_names = [c[0][2] for c in mock_save.call_args_list]
        expected_suffixes = [
            "_ifft",
            "_fullmagdata",
            "_fullphasedata",
            "_detrendedphase",
            "_phasemeans",
            "_phaseslopes",
            "_mag",
            "_phase",
            "_pcaphase",
            "_reconmag",
            "_reconphase",
            "_movingsignal",
        ]
        for suffix in expected_suffixes:
            assert any(name.endswith(suffix) for name in save_names), \
                f"Expected output with suffix '{suffix}' not found in {save_names}"

        # Collect the text output names from writenpvecs calls
        write_names = [c[0][1] for c in mock_write.call_args_list]
        assert any("_pcacomponents.txt" in name for name in write_names)
        assert any("_explained_variance_pct.txt" in name for name in write_names)
        assert any("_icacomponents.txt" in name for name in write_names)


def fdica_partial_mask(testtemproot, debug=False):
    """Test fdica with a partial mask (not all voxels selected)."""
    xsize, ysize, numslices, timepoints = 4, 4, 2, 64
    data = np.random.randn(xsize, ysize, numslices, timepoints).astype(np.float64)
    # Only select half the voxels
    mask = np.zeros((xsize, ysize, numslices), dtype=np.float64)
    mask[:2, :, :] = 1.0

    hdr = MagicMock()
    hdr.get_xyzt_units.return_value = ("mm", "sec")
    hdr.__getitem__ = MagicMock(side_effect=lambda key: {
        "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
        "pixdim": [1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
    }[key])
    hdr.__setitem__ = MagicMock()
    hdr.__deepcopy__ = MagicMock(side_effect=lambda memo: _make_simple_header(
        xsize, ysize, numslices, timepoints
    ))

    dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    outputroot = os.path.join(testtemproot, "fdica_partial")

    with patch("rapidtide.workflows.fdica.tide_io.readfromnifti") as mock_read, \
         patch("rapidtide.workflows.fdica.tide_io.savetonifti") as mock_save, \
         patch("rapidtide.workflows.fdica.tide_io.writenpvecs"), \
         patch("rapidtide.workflows.fdica.tide_io.parseniftisizes") as mock_sizes, \
         patch("rapidtide.workflows.fdica.tide_io.parseniftidims") as mock_dims, \
         patch("rapidtide.workflows.fdica.tide_io.checkspacedimmatch") as mock_check:

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask, hdr, mask_dims, mask_sizes),
        ]
        mock_sizes.return_value = (2.0, 2.0, 2.0, 2.0)
        mock_dims.return_value = (xsize, ysize, numslices, timepoints)
        mock_check.return_value = True

        fdica(
            "dummy_data.nii.gz",
            "dummy_mask.nii.gz",
            outputroot,
            pcacomponents=0.9,
            debug=debug,
        )

        assert mock_save.call_count > 0


# ==================== Tests for main ====================


def main_calls_fdica(debug=False):
    """Test that main parses args and calls fdica."""
    with patch("rapidtide.workflows.fdica._get_parser") as mock_parser, \
         patch("rapidtide.workflows.fdica.fdica") as mock_fdica:

        mock_args = argparse.Namespace(
            datafile="data.nii.gz",
            datamask="mask.nii.gz",
            outputroot="output",
            gausssigma=0.0,
            pcacomponents=0.9,
            icacomponents=None,
            debug=False,
        )
        mock_parser.return_value.parse_args.return_value = mock_args

        main()

        mock_fdica.assert_called_once_with(
            "data.nii.gz",
            "mask.nii.gz",
            "output",
            gausssigma=0.0,
            icacomponents=None,
            pcacomponents=0.9,
            debug=False,
        )


def main_parse_error(debug=False):
    """Test that main prints help on parse error."""
    with patch("rapidtide.workflows.fdica._get_parser") as mock_parser:
        mock_parser.return_value.parse_args.side_effect = SystemExit(2)

        try:
            main()
        except SystemExit:
            pass
        mock_parser.return_value.print_help.assert_called_once()


# ==================== Main test function ====================


def test_fdica(debug=False, local=False):
    # set up temp directory
    testtemproot = get_test_temp_path(local)
    create_dir(testtemproot)

    # P2R tests
    if debug:
        print("P2R_zero_angle()")
    P2R_zero_angle(debug=debug)

    if debug:
        print("P2R_right_angle()")
    P2R_right_angle(debug=debug)

    if debug:
        print("P2R_pi_angle()")
    P2R_pi_angle(debug=debug)

    if debug:
        print("P2R_array_input()")
    P2R_array_input(debug=debug)

    if debug:
        print("P2R_zero_radius()")
    P2R_zero_radius(debug=debug)

    if debug:
        print("P2R_2d_array()")
    P2R_2d_array(debug=debug)

    # R2P tests
    if debug:
        print("R2P_real_positive()")
    R2P_real_positive(debug=debug)

    if debug:
        print("R2P_real_negative()")
    R2P_real_negative(debug=debug)

    if debug:
        print("R2P_pure_imaginary()")
    R2P_pure_imaginary(debug=debug)

    if debug:
        print("R2P_complex()")
    R2P_complex(debug=debug)

    if debug:
        print("R2P_array()")
    R2P_array(debug=debug)

    if debug:
        print("R2P_zero()")
    R2P_zero(debug=debug)

    # Roundtrip tests
    if debug:
        print("P2R_R2P_roundtrip()")
    P2R_R2P_roundtrip(debug=debug)

    if debug:
        print("R2P_P2R_roundtrip()")
    R2P_P2R_roundtrip(debug=debug)

    # _get_parser tests
    if debug:
        print("get_parser_returns_parser()")
    get_parser_returns_parser(debug=debug)

    if debug:
        print("get_parser_prog_name()")
    get_parser_prog_name(debug=debug)

    if debug:
        print("get_parser_defaults()")
    get_parser_defaults(debug=debug)

    if debug:
        print("get_parser_with_options(testtemproot)")
    get_parser_with_options(testtemproot, debug=debug)

    if debug:
        print("get_parser_required_args(testtemproot)")
    get_parser_required_args(testtemproot, debug=debug)

    # fdica tests
    if debug:
        print("fdica_basic(testtemproot)")
    fdica_basic(testtemproot, debug=debug)

    if debug:
        print("fdica_with_spatial_filter(testtemproot)")
    fdica_with_spatial_filter(testtemproot, debug=debug)

    if debug:
        print("fdica_auto_gausssigma(testtemproot)")
    fdica_auto_gausssigma(testtemproot, debug=debug)

    if debug:
        print("fdica_with_ica_components(testtemproot)")
    fdica_with_ica_components(testtemproot, debug=debug)

    if debug:
        print("fdica_with_pca_integer(testtemproot)")
    fdica_with_pca_integer(testtemproot, debug=debug)

    if debug:
        print("fdica_dim_mismatch_exits(testtemproot)")
    fdica_dim_mismatch_exits(testtemproot, debug=debug)

    if debug:
        print("fdica_4d_mask_exits(testtemproot)")
    fdica_4d_mask_exits(testtemproot, debug=debug)

    if debug:
        print("fdica_msec_tr(testtemproot)")
    fdica_msec_tr(testtemproot, debug=debug)

    if debug:
        print("fdica_negative_freq_bounds(testtemproot)")
    fdica_negative_freq_bounds(testtemproot, debug=debug)

    if debug:
        print("fdica_output_files(testtemproot)")
    fdica_output_files(testtemproot, debug=debug)

    if debug:
        print("fdica_partial_mask(testtemproot)")
    fdica_partial_mask(testtemproot, debug=debug)

    # main tests
    if debug:
        print("main_calls_fdica()")
    main_calls_fdica(debug=debug)

    if debug:
        print("main_parse_error()")
    main_parse_error(debug=debug)


if __name__ == "__main__":
    test_fdica(debug=True, local=True)
