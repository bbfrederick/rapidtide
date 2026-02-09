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
"""
Tests for RapidtideDataset module - covers Timecourse, Overlay, RapidtideDataset classes
and the check_rt_spatialmatch function.
"""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.RapidtideDataset import (
    Overlay,
    RapidtideDataset,
    Timecourse,
    check_rt_spatialmatch,
)
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path

# ============================================================================
# Helper functions for creating test data
# ============================================================================


def create_synthetic_nifti(
    filepath, shape=(10, 10, 10), tr=2.0, toffset=0.0, data=None, affine=None
):
    """Create a synthetic NIfTI file for testing."""
    if data is None:
        data = np.random.rand(*shape).astype(np.float32)
    if affine is None:
        affine = np.eye(4)
        affine[0, 0] = -2.0  # negative = neurological orientation
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((2.0, 2.0, 2.0, tr)[:len(shape)])
    img.header["toffset"] = toffset
    img.header["sform_code"] = 1
    nib.save(img, filepath)
    return data


def create_synthetic_timecourse(filepath, length=100, samplerate=1.0):
    """Create a synthetic timecourse file for testing."""
    timedata = np.sin(np.linspace(0, 4 * np.pi, length))
    tide_io.writevec(timedata, filepath)
    return timedata


# ============================================================================
# Tests for check_rt_spatialmatch function
# ============================================================================


class MockDataset:
    """Mock dataset for testing check_rt_spatialmatch."""

    def __init__(self, xdim, ydim, zdim, xsize, ysize, zsize, space, affine):
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize
        self.space = space
        self.affine = affine


class TestCheckRtSpatialMatch:
    """Tests for the check_rt_spatialmatch function."""

    def test_all_match(self):
        """Test when all spatial properties match."""
        affine = np.eye(4)
        ds1 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine)
        ds2 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine)
        dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(ds1, ds2)
        assert dimmatch is True
        assert sizematch is True
        assert spacematch is True
        assert affinematch is True

    def test_dim_mismatch(self):
        """Test when dimensions don't match."""
        affine = np.eye(4)
        ds1 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine)
        ds2 = MockDataset(11, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine)
        dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(ds1, ds2)
        assert dimmatch is False
        assert sizematch is True

    def test_size_mismatch(self):
        """Test when voxel sizes don't match."""
        affine = np.eye(4)
        ds1 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine)
        ds2 = MockDataset(10, 20, 30, 1.5, 2.0, 3.0, "MNI152", affine)
        dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(ds1, ds2)
        assert dimmatch is True
        assert sizematch is False

    def test_space_mismatch(self):
        """Test when coordinate spaces don't match."""
        affine = np.eye(4)
        ds1 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine)
        ds2 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "native", affine)
        dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(ds1, ds2)
        assert spacematch is False

    def test_affine_mismatch(self):
        """Test when affine matrices don't match."""
        affine1 = np.eye(4)
        affine2 = np.eye(4) * 2
        ds1 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine1)
        ds2 = MockDataset(10, 20, 30, 1.0, 2.0, 3.0, "MNI152", affine2)
        dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(ds1, ds2)
        assert affinematch is False


# ============================================================================
# Tests for Timecourse class
# ============================================================================


class TestTimecourse:
    """Tests for the Timecourse class."""

    def test_init_basic(self, tmp_path):
        """Test basic Timecourse initialization."""
        # Create a test timecourse file
        filepath = str(tmp_path / "test_timecourse.txt")
        timedata = create_synthetic_timecourse(filepath, length=100)

        tc = Timecourse(
            name="test",
            filename=filepath,
            namebase="test_timecourse.txt",
            samplerate=10.0,
            displaysamplerate=10.0,
            starttime=0.0,
            verbose=0,
        )

        assert tc.name == "test"
        assert tc.filename == filepath
        assert tc.samplerate == 10.0
        assert tc.displaysamplerate == 10.0
        assert tc.starttime == 0.0
        assert tc.length == 100
        assert tc.timedata is not None
        assert len(tc.timedata) == 100

    def test_init_with_label(self, tmp_path):
        """Test Timecourse initialization with custom label."""
        filepath = str(tmp_path / "test_timecourse.txt")
        create_synthetic_timecourse(filepath)

        tc = Timecourse(
            name="test",
            filename=filepath,
            namebase="test_timecourse.txt",
            samplerate=10.0,
            displaysamplerate=10.0,
            label="Custom Label",
            verbose=0,
        )

        assert tc.label == "Custom Label"

    def test_init_label_defaults_to_name(self, tmp_path):
        """Test that label defaults to name when not provided."""
        filepath = str(tmp_path / "test_timecourse.txt")
        create_synthetic_timecourse(filepath)

        tc = Timecourse(
            name="myname",
            filename=filepath,
            namebase="test_timecourse.txt",
            samplerate=10.0,
            displaysamplerate=10.0,
            verbose=0,
        )

        assert tc.label == "myname"

    def test_readTimeData(self, tmp_path):
        """Test the readTimeData method."""
        filepath = str(tmp_path / "test_timecourse.txt")
        expected_data = create_synthetic_timecourse(filepath, length=50)

        tc = Timecourse(
            name="test",
            filename=filepath,
            namebase="test_timecourse.txt",
            samplerate=10.0,
            displaysamplerate=10.0,
            verbose=0,
        )

        # Verify attributes set by readTimeData
        assert tc.timedata is not None
        assert tc.length == 50
        assert tc.timeaxis is not None
        assert len(tc.timeaxis) == 50
        assert tc.specaxis is not None
        assert tc.specdata is not None
        assert hasattr(tc, "kurtosis")
        assert hasattr(tc, "skewness")

    def test_readTimeData_with_limits(self, tmp_path):
        """Test readTimeData with specified limits."""
        filepath = str(tmp_path / "test_timecourse.txt")
        create_synthetic_timecourse(filepath, length=100)

        tc = Timecourse(
            name="test",
            filename=filepath,
            namebase="test_timecourse.txt",
            samplerate=10.0,
            displaysamplerate=10.0,
            limits=(1.0, 5.0),  # time limits in seconds
            verbose=0,
        )

        assert tc.limits == (1.0, 5.0)
        assert tc.length == 100  # full length is still stored
        # spectral data computed on limited portion
        assert tc.specdata is not None

    def test_summarize(self, tmp_path, capsys):
        """Test the summarize method."""
        filepath = str(tmp_path / "test_timecourse.txt")
        create_synthetic_timecourse(filepath)

        tc = Timecourse(
            name="test",
            filename=filepath,
            namebase="test_timecourse.txt",
            samplerate=10.0,
            displaysamplerate=10.0,
            verbose=0,
        )

        tc.summarize()
        captured = capsys.readouterr()
        assert "Timecourse name:" in captured.out
        assert "test" in captured.out
        assert "samplerate:" in captured.out
        assert "length:" in captured.out
        assert "kurtosis:" in captured.out


# ============================================================================
# Tests for Overlay class
# ============================================================================


class TestOverlay:
    """Tests for the Overlay class."""

    def test_init_basic(self, tmp_path):
        """Test basic Overlay initialization."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        expected_data = create_synthetic_nifti(filepath, shape=(10, 10, 10))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        assert overlay.name == "test"
        assert overlay.filename == filepath
        assert overlay.namebase == "test_overlay"
        assert overlay.xdim == 10
        assert overlay.ydim == 10
        assert overlay.zdim == 10
        assert overlay.tdim == 1
        assert overlay.data is not None

    def test_init_with_label(self, tmp_path):
        """Test Overlay initialization with custom label."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            label="Custom Label",
            init_LUT=False,
            verbose=0,
        )

        assert overlay.label == "Custom Label"

    def test_init_label_defaults_to_name(self, tmp_path):
        """Test that label defaults to name when not provided."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="myname",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        assert overlay.label == "myname"

    def test_init_with_funcmask(self, tmp_path):
        """Test Overlay initialization with functional mask."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10))

        funcmask = np.ones((10, 10, 10))
        funcmask[5:, :, :] = 0

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            funcmask=funcmask,
            init_LUT=False,
            verbose=0,
        )

        assert overlay.funcmask is not None
        assert np.array_equal(overlay.funcmask, funcmask)

    def test_init_with_geommask(self, tmp_path):
        """Test Overlay initialization with geometric mask."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10))

        geommask = np.ones((10, 10, 10))
        geommask[:, :5, :] = 0

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            geommask=geommask,
            init_LUT=False,
            verbose=0,
        )

        assert overlay.geommask is not None
        assert np.array_equal(overlay.geommask, geommask)

    def test_init_as_mask(self, tmp_path):
        """Test Overlay initialization as a binary mask."""
        filepath = str(tmp_path / "test_mask.nii.gz")
        # Create data with values that should be binarized
        data = np.array([[[0.2, 0.8], [0.3, 0.9]], [[0.1, 0.7], [0.4, 0.6]]]).astype(
            np.float32
        )
        create_synthetic_nifti(filepath, shape=data.shape, data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_mask",
            isaMask=True,
            init_LUT=False,
            verbose=0,
        )

        # Values < 0.5 should be 0, values > 0.5 should be 1
        assert np.all(np.isin(overlay.data, [0.0, 1.0]))

    def test_init_invert_on_load(self, tmp_path):
        """Test Overlay initialization with inversion on load."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        original_data = np.ones((5, 5, 5)).astype(np.float32) * 2.0
        create_synthetic_nifti(filepath, shape=(5, 5, 5), data=original_data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            invertonload=True,
            init_LUT=False,
            verbose=0,
        )

        # Data should be inverted (multiplied by -1)
        assert np.allclose(overlay.data, -original_data)

    def test_duplicate(self, tmp_path):
        """Test the duplicate method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="original",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        duplicate = overlay.duplicate("copy", "Copy Label", init_LUT=False)

        assert duplicate.name == "copy"
        assert duplicate.label == "Copy Label"
        assert duplicate.filename == overlay.filename

    def test_updateStats(self, tmp_path):
        """Test the updateStats method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        data = np.random.rand(10, 10, 10).astype(np.float32)
        create_synthetic_nifti(filepath, shape=(10, 10, 10), data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        assert hasattr(overlay, "minval")
        assert hasattr(overlay, "maxval")
        assert hasattr(overlay, "robustmin")
        assert hasattr(overlay, "robustmax")
        assert hasattr(overlay, "quartiles")
        assert hasattr(overlay, "histx")
        assert hasattr(overlay, "histy")
        assert overlay.minval <= overlay.maxval

    def test_setData(self, tmp_path):
        """Test the setData method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(5, 5, 5))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        new_data = np.ones((5, 5, 5)) * 5.0
        overlay.setData(new_data)

        assert np.allclose(overlay.data, new_data)

    def test_setData_as_mask(self, tmp_path):
        """Test setData with isaMask=True."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Create data with shape that matches what we'll set
        data = np.array([[[0.3, 0.7], [0.2, 0.8]], [[0.1, 0.9], [0.4, 0.6]]]).astype(
            np.float32
        )
        create_synthetic_nifti(filepath, shape=data.shape, data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        # Set new data with same shape
        new_data = np.array([[[0.3, 0.7], [0.2, 0.8]], [[0.1, 0.9], [0.4, 0.6]]]).astype(
            np.float32
        )
        overlay.setData(new_data, isaMask=True)

        assert np.all(np.isin(overlay.data, [0.0, 1.0]))

    def test_readImageData(self, tmp_path):
        """Test the readImageData method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Use 4D data to properly test TR (3D NIfTI doesn't store TR in zooms)
        expected_data = create_synthetic_nifti(filepath, shape=(8, 9, 10, 5), tr=1.5)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        assert overlay.xdim == 8
        assert overlay.ydim == 9
        assert overlay.zdim == 10
        assert overlay.tdim == 5
        assert overlay.tr == 1.5
        assert overlay.data.shape == (8, 9, 10, 5)

    def test_setLabel(self, tmp_path):
        """Test the setLabel method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setLabel("New Label")
        assert overlay.label == "New Label"

    def test_real2tr(self, tmp_path):
        """Test the real2tr method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Use 4D data to properly set TR
        create_synthetic_nifti(filepath, shape=(5, 5, 5, 10), tr=2.0, toffset=1.0)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        # Verify TR and toffset were set correctly
        assert overlay.tr == 2.0
        assert overlay.toffset == 1.0
        # time = 5.0, toffset = 1.0, tr = 2.0
        # (5.0 - 1.0) / 2.0 = 2.0
        assert overlay.real2tr(5.0) == 2.0

    def test_tr2real(self, tmp_path):
        """Test the tr2real method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Use 4D data to properly set TR
        create_synthetic_nifti(filepath, shape=(5, 5, 5, 10), tr=2.0, toffset=1.0)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        # Verify TR and toffset were set correctly
        assert overlay.tr == 2.0
        assert overlay.toffset == 1.0
        # tpos = 3, toffset = 1.0, tr = 2.0
        # 1.0 + 2.0 * 3 = 7.0
        assert overlay.tr2real(3) == 7.0

    def test_setXYZpos(self, tmp_path):
        """Test the setXYZpos method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setXYZpos(5, 6, 7)
        assert overlay.xpos == 5
        assert overlay.ypos == 6
        assert overlay.zpos == 7

    def test_setTpos(self, tmp_path):
        """Test the setTpos method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10, 20))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setTpos(5)
        assert overlay.tpos == 5

        # Test bounds checking
        overlay.setTpos(100)  # Should be clamped to tdim - 1
        assert overlay.tpos == overlay.tdim - 1

    def test_getFocusVal_3d(self, tmp_path):
        """Test getFocusVal for 3D data."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        data = np.arange(1000).reshape((10, 10, 10)).astype(np.float32)
        create_synthetic_nifti(filepath, shape=(10, 10, 10), data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setXYZpos(2, 3, 4)
        expected_value = overlay.maskeddata[2, 3, 4]
        assert overlay.getFocusVal() == expected_value

    def test_getFocusVal_4d(self, tmp_path):
        """Test getFocusVal for 4D data."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        data = np.arange(2000).reshape((10, 10, 10, 2)).astype(np.float32)
        create_synthetic_nifti(filepath, shape=(10, 10, 10, 2), data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setXYZpos(2, 3, 4)
        overlay.setTpos(1)
        expected_value = overlay.maskeddata[2, 3, 4, 1]
        assert overlay.getFocusVal() == expected_value

    def test_setFuncMask(self, tmp_path):
        """Test the setFuncMask method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        funcmask = np.ones((10, 10, 10))
        funcmask[5:, :, :] = 0
        overlay.setFuncMask(funcmask)

        assert overlay.funcmask is not None
        assert np.array_equal(overlay.funcmask, funcmask)

    def test_setFuncMask_none(self, tmp_path):
        """Test setFuncMask with None creates default mask."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setFuncMask(None)
        assert overlay.funcmask is not None
        assert overlay.funcmask.shape == (10, 10, 10)

    def test_setGeomMask(self, tmp_path):
        """Test the setGeomMask method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        geommask = np.ones((10, 10, 10))
        geommask[:, 5:, :] = 0
        overlay.setGeomMask(geommask)

        assert overlay.geommask is not None
        assert np.array_equal(overlay.geommask, geommask)

    def test_setGeomMask_none(self, tmp_path):
        """Test setGeomMask with None creates default mask."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, shape=(10, 10, 10))

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setGeomMask(None)
        assert overlay.geommask is not None
        assert overlay.geommask.shape == (10, 10, 10)

    def test_maskData(self, tmp_path):
        """Test the maskData method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        data = np.ones((10, 10, 10)).astype(np.float32) * 5.0
        create_synthetic_nifti(filepath, shape=(10, 10, 10), data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        # Set a mask that zeros out half the data
        geommask = np.ones((10, 10, 10))
        geommask[5:, :, :] = 0
        overlay.setGeomMask(geommask)

        # Masked data should be zero where mask is 0
        assert overlay.maskeddata is not None
        assert np.all(overlay.maskeddata[5:, :, :] == 0)
        assert np.all(overlay.maskeddata[:5, :, :] == 5.0)

    def test_setReport(self, tmp_path):
        """Test the setReport method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setReport(True)
        assert overlay.report is True

        overlay.setReport(False)
        assert overlay.report is False

    def test_setTR(self, tmp_path):
        """Test the setTR method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath, tr=2.0)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setTR(3.5)
        assert overlay.tr == 3.5

    def test_settoffset(self, tmp_path):
        """Test the settoffset method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.settoffset(5.0)
        assert overlay.toffset == 5.0

    def test_setisdisplayed(self, tmp_path):
        """Test the setisdisplayed method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.setisdisplayed(True)
        assert overlay.display_state is True

        overlay.setisdisplayed(False)
        assert overlay.display_state is False

    def test_summarize(self, tmp_path, capsys):
        """Test the summarize method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        create_synthetic_nifti(filepath)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        overlay.summarize()
        captured = capsys.readouterr()
        assert "Overlay name:" in captured.out
        assert "test" in captured.out
        assert "xdim:" in captured.out
        assert "ydim:" in captured.out
        assert "zdim:" in captured.out

    def test_real2vox(self, tmp_path):
        """Test the real2vox method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Create with known affine and 4D data for proper TR handling
        affine = np.eye(4)
        affine[0, 0] = -2.0
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        create_synthetic_nifti(filepath, shape=(10, 10, 10, 5), tr=2.0, toffset=0.0, affine=affine)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        # Test coordinate conversion
        x, y, z, t = overlay.real2vox(0.0, 4.0, 6.0, 4.0)
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(z, int)
        assert isinstance(t, int)

    def test_vox2real(self, tmp_path):
        """Test the vox2real method."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Use 4D data for proper TR handling
        create_synthetic_nifti(filepath, shape=(10, 10, 10, 5), tr=2.0, toffset=0.0)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        # Test voxel to real coordinate conversion
        result = overlay.vox2real(0, 1, 2, 3)
        assert len(result) == 4
        assert isinstance(result, np.ndarray)

    def test_orientation_detection_neurological(self, tmp_path):
        """Test neurological orientation detection."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        affine = np.eye(4)
        affine[0, 0] = -2.0  # Negative = neurological
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        create_synthetic_nifti(filepath, affine=affine)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        assert overlay.RLfactor == -1.0

    def test_orientation_detection_radiological(self, tmp_path):
        """Test radiological orientation detection."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        affine = np.eye(4)
        affine[0, 0] = 2.0  # Positive = radiological
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        create_synthetic_nifti(filepath, affine=affine)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=0,
        )

        assert overlay.RLfactor == 1.0


# ============================================================================
# Tests for RapidtideDataset class
# ============================================================================


class TestRapidtideDataset:
    """Tests for the RapidtideDataset class.

    Note: These tests require rapidtide output files to be present in the test
    temp directory, which are created by the full rapidtide run tests.
    """

    def test_init_basic(self):
        """Test basic RapidtideDataset initialization with full rapidtide output."""
        # Use the output from the rapidtide test run
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        # Skip if the test data doesn't exist (requires fullrun test to be run first)
        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            anatname=None,
            geommaskname=None,
            userise=False,
            usecorrout=True,
            useatlas=False,
            forcetr=False,
            forceoffset=False,
            offsettime=0.0,
            init_LUT=False,
            verbose=0,
        )

        assert thesubject.name == "main"
        assert thesubject.fileroot == datafileroot
        assert thesubject.bidsformat is True

    def test_getoverlays(self):
        """Test the getoverlays method."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        overlays = thesubject.getoverlays()
        assert isinstance(overlays, dict)
        assert len(overlays) > 0
        assert "lagtimes" in overlays

    def test_getregressors(self):
        """Test the getregressors method."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        regressors = thesubject.getregressors()
        assert isinstance(regressors, dict)

    def test_setfocusregressor(self):
        """Test the setfocusregressor method."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        # Set to valid regressor
        initial_focus = thesubject.focusregressor
        thesubject.setfocusregressor("pass1")

        # If pass1 exists, it should be set
        if "pass1" in thesubject.regressors:
            assert thesubject.focusregressor == "pass1"

        # Set to invalid regressor - should fall back to prefilt
        thesubject.setfocusregressor("nonexistent_regressor")
        assert thesubject.focusregressor == "prefilt"

    def test_setfocusmap(self):
        """Test the setfocusmap method."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        # Set to valid map
        thesubject.setfocusmap("lagstrengths")
        if "lagstrengths" in thesubject.overlays:
            assert thesubject.focusmap == "lagstrengths"

        # Set to invalid map - should fall back to lagtimes
        thesubject.setfocusmap("nonexistent_map")
        assert thesubject.focusmap == "lagtimes"

    def test_setFuncMaskName(self):
        """Test the setFuncMaskName method."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        thesubject.setFuncMaskName("new_mask_name")
        assert thesubject.funcmaskname == "new_mask_name"

    def test_dataset_dimensions(self):
        """Test that dataset dimensions are properly set."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        assert thesubject.xdim > 0
        assert thesubject.ydim > 0
        assert thesubject.zdim > 0
        assert thesubject.xsize > 0
        assert thesubject.ysize > 0
        assert thesubject.zsize > 0

    def test_regressorfilterlimits(self):
        """Test that regressor filter limits are set."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=0,
        )

        assert thesubject.regressorfilterlimits is not None
        assert len(thesubject.regressorfilterlimits) == 2
        assert thesubject.regressorfilterlimits[0] >= 0


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests that use complete rapidtide output."""

    def test_full_workflow(self):
        """Test a complete workflow with RapidtideDataset."""
        testtemproot = get_test_temp_path()
        datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETEST1_")

        if not os.path.isfile(datafileroot + "desc-maxtime_map.nii.gz"):
            pytest.skip("Rapidtide output files not found - run fullrun tests first")

        # Create dataset
        thesubject = RapidtideDataset(
            "main",
            datafileroot,
            init_LUT=False,
            verbose=2,
        )

        # Get overlays and regressors
        overlays = thesubject.getoverlays()
        regressors = thesubject.getregressors()

        # Test changing focus
        assert thesubject.focusregressor == "prefilt"
        thesubject.setfocusregressor("pass3")
        if "pass3" in regressors:
            assert thesubject.focusregressor == "pass3"

        # Check that expected filter limits are set
        assert thesubject.regressorfilterlimits == (0.01, 0.15)


def main(runninglocally=False, debug=False):
    """Run tests manually for local development."""
    if runninglocally:
        datafileroot = "../data/examples/dst/sub-RAPIDTIDETEST_"
    else:
        print(f"get_test_temp_path={get_test_temp_path()}")
        datafileroot = os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST1_")

    anatname = None
    geommaskname = None
    userise = False
    usecorrout = True
    useatlas = False
    forcetr = False
    forceoffset = False
    offsettime = 0.0
    verbose = 2

    # read in the dataset
    thesubject = RapidtideDataset(
        "main",
        datafileroot,
        anatname=anatname,
        geommaskname=geommaskname,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        init_LUT=False,
        verbose=verbose,
    )

    print("getting overlays")
    theoverlays = thesubject.getoverlays()
    print("getting regressors")
    theregressors = thesubject.getregressors()

    assert thesubject.focusregressor == "prefilt"
    thesubject.setfocusregressor("pass3")
    assert thesubject.focusregressor == "pass3"

    if debug:
        print(thesubject.regressorfilterlimits)
    assert thesubject.regressorfilterlimits == (0.01, 0.15)


if __name__ == "__main__":
    main(runninglocally=True, debug=True)
