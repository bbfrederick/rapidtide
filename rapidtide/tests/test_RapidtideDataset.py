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

import json
import os

import nibabel as nib
import numpy as np
import pytest

import rapidtide.io as tide_io
import rapidtide.RapidtideDataset as rapidtide_dataset
import rapidtide.util as tide_util
import rapidtide.workflows.rapidtide as rapidtide_workflow
import rapidtide.workflows.rapidtide_parser as rapidtide_parser
from rapidtide.Colortables import gen_gray_state
from rapidtide.RapidtideDataset import (
    Overlay,
    RapidtideDataset,
    Timecourse,
    check_rt_spatialmatch,
)
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path

runninglocally = False

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
    img.header.set_zooms((2.0, 2.0, 2.0, tr)[: len(shape)])
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
# Fixture for synthetic rapidtide output
# ============================================================================


@pytest.fixture
def rapidtide_output(tmp_path):
    """Create a minimal set of synthetic BIDS-format rapidtide output files."""
    datafileroot = str(tmp_path / "sub-TESTDATA_")
    shape = (5, 5, 5)

    # --- Run options JSON ---
    runoptions = {
        "lowerpass": 0.01,
        "upperpass": 0.15,
        "fmrifreq": 0.5,
        "inputfreq": 0.5,
        "inputstarttime": 0.0,
        "oversampfactor": 2,
        "similaritymetric": "correlation",
        "validsimcalcstart": 0,
        "validsimcalcend": 100,
        "actual_passes": 2,
    }
    with open(datafileroot + "desc-runoptions_info.json", "w") as f:
        json.dump(runoptions, f)

    # --- Regressor BIDS TSV+JSON files ---
    # initialmovingregressor: columns prefilt, postfilt
    n_samples = 100
    regdata = np.column_stack([np.sin(np.linspace(0, 4 * np.pi, n_samples)) for _ in range(2)]).T
    tide_io.writebidstsv(
        datafileroot + "desc-initialmovingregressor_timeseries",
        regdata,
        samplerate=0.5,
        columns=["prefilt", "postfilt"],
        starttime=0.0,
    )

    # oversampledmovingregressor: columns pass1, pass2
    oversamp_data = np.column_stack(
        [np.sin(np.linspace(0, 4 * np.pi, n_samples)) for _ in range(2)]
    ).T
    tide_io.writebidstsv(
        datafileroot + "desc-oversampledmovingregressor_timeseries",
        oversamp_data,
        samplerate=1.0,
        columns=["pass1", "pass2"],
        starttime=0.0,
    )

    # --- Functional map NIfTI files (3D) ---
    func_maps = [
        "desc-maxtime_map",
        "desc-maxcorr_map",
        "desc-maxwidth_map",
        "desc-neglog10p_map",
    ]
    for mapname in func_maps:
        create_synthetic_nifti(datafileroot + mapname + ".nii.gz", shape=shape)

    # --- Functional mask NIfTI files (3D, binary) ---
    func_masks = [
        "desc-corrfit_mask",
        "desc-refine_mask",
        "desc-globalmean_mask",
    ]
    mask_data = np.ones(shape, dtype=np.float32)
    for maskname in func_masks:
        create_synthetic_nifti(datafileroot + maskname + ".nii.gz", shape=shape, data=mask_data)

    return datafileroot


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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
        )

        assert overlay.geommask is not None
        assert np.array_equal(overlay.geommask, geommask)

    def test_init_as_mask(self, tmp_path):
        """Test Overlay initialization as a binary mask."""
        filepath = str(tmp_path / "test_mask.nii.gz")
        # Create data with values that should be binarized
        data = np.array([[[0.2, 0.8], [0.3, 0.9]], [[0.1, 0.7], [0.4, 0.6]]]).astype(np.float32)
        create_synthetic_nifti(filepath, shape=data.shape, data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_mask",
            isaMask=True,
            init_LUT=False,
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
        )

        new_data = np.ones((5, 5, 5)) * 5.0
        overlay.setData(new_data)

        assert np.allclose(overlay.data, new_data)

    def test_setData_as_mask(self, tmp_path):
        """Test setData with isaMask=True."""
        filepath = str(tmp_path / "test_overlay.nii.gz")
        # Create data with shape that matches what we'll set
        data = np.array([[[0.3, 0.7], [0.2, 0.8]], [[0.1, 0.9], [0.4, 0.6]]]).astype(np.float32)
        create_synthetic_nifti(filepath, shape=data.shape, data=data)

        overlay = Overlay(
            name="test",
            filespec=filepath,
            namebase="test_overlay",
            init_LUT=False,
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
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
            verbose=2,
        )

        assert overlay.RLfactor == 1.0


# ============================================================================
# Tests for RapidtideDataset class
# ============================================================================


class TestRapidtideDataset:
    """Tests for the RapidtideDataset class using synthetic output files."""

    def test_init_basic(self, rapidtide_output):
        """Test basic RapidtideDataset initialization."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            anatname=None,
            geommaskname=None,
            userise=False,
            usecorrout=False,
            useatlas=False,
            forcetr=False,
            forceoffset=False,
            offsettime=0.0,
            init_LUT=False,
            verbose=2,
        )

        assert thesubject.name == "main"
        assert thesubject.fileroot == rapidtide_output
        assert thesubject.bidsformat is True

    def test_getoverlays(self, rapidtide_output):
        """Test the getoverlays method."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        overlays = thesubject.getoverlays()
        assert isinstance(overlays, dict)
        assert len(overlays) > 0
        assert "lagtimes" in overlays

    def test_getregressors(self, rapidtide_output):
        """Test the getregressors method."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        regressors = thesubject.getregressors()
        assert isinstance(regressors, dict)
        assert "prefilt" in regressors

    def test_setfocusregressor(self, rapidtide_output):
        """Test the setfocusregressor method."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        # Set to valid regressor
        thesubject.setfocusregressor("pass1")
        assert thesubject.focusregressor == "pass1"

        # Set to invalid regressor - should fall back to prefilt
        thesubject.setfocusregressor("nonexistent_regressor")
        assert thesubject.focusregressor == "prefilt"

    def test_setfocusmap(self, rapidtide_output):
        """Test the setfocusmap method."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        # Set to valid map
        thesubject.setfocusmap("lagstrengths")
        assert thesubject.focusmap == "lagstrengths"

        # Set to invalid map - should fall back to lagtimes
        thesubject.setfocusmap("nonexistent_map")
        assert thesubject.focusmap == "lagtimes"

    def test_setFuncMaskName(self, rapidtide_output):
        """Test the setFuncMaskName method."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        thesubject.setFuncMaskName("new_mask_name")
        assert thesubject.funcmaskname == "new_mask_name"

    def test_dataset_dimensions(self, rapidtide_output):
        """Test that dataset dimensions are properly set."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        assert thesubject.xdim > 0
        assert thesubject.ydim > 0
        assert thesubject.zdim > 0
        assert thesubject.xsize > 0
        assert thesubject.ysize > 0
        assert thesubject.zsize > 0

    def test_regressorfilterlimits(self, rapidtide_output):
        """Test that regressor filter limits are set."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        assert thesubject.regressorfilterlimits is not None
        assert len(thesubject.regressorfilterlimits) == 2
        assert thesubject.regressorfilterlimits == (0.01, 0.15)


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests using synthetic rapidtide output."""

    def test_full_workflow(self, rapidtide_output):
        """Test a complete workflow with RapidtideDataset."""
        # Create dataset
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            init_LUT=False,
            verbose=2,
        )

        # Get overlays and regressors
        overlays = thesubject.getoverlays()
        regressors = thesubject.getregressors()

        assert isinstance(overlays, dict)
        assert "lagtimes" in overlays
        assert isinstance(regressors, dict)
        assert "prefilt" in regressors

        # Test changing focus regressor
        assert thesubject.focusregressor == "prefilt"
        thesubject.setfocusregressor("pass1")
        assert thesubject.focusregressor == "pass1"

        # Test changing focus map
        thesubject.setfocusmap("lagstrengths")
        assert thesubject.focusmap == "lagstrengths"

        # Check that expected filter limits are set
        assert thesubject.regressorfilterlimits == (0.01, 0.15)


# ============================================================================
# Helpers and fixtures for the optional-overlay and legacy-naming tests
# ============================================================================


class FakeGradient:
    """Stand in for the pyqtgraph GradientWidget an Overlay builds its LUT from.

    Building a real one needs a live QApplication, so an Overlay constructed with
    init_LUT=True is untestable headless.  Everything the LUT code actually does -
    rewriting the alpha channel of the intermediate colour ticks - is plain data
    manipulation that happens before the widget is touched, so substituting this
    lets that logic be exercised without a GUI.

    Attributes
    ----------
    restoredstates : list of dict
        Every state handed to restoreState, in order.
    """

    def __init__(self):
        self.restoredstates = []

    def restoreState(self, thestate):
        """Record a gradient state instead of rendering it."""
        self.restoredstates.append(thestate)

    def getLookupTable(self, thesize, alpha=True):
        """Return a lookup table of the requested size, filled with zeros."""
        return np.zeros((thesize, 4 if alpha else 3), dtype=np.uint8)


@pytest.fixture
def fake_gradient(monkeypatch):
    """Replace getagradient in the RapidtideDataset namespace with a fake.

    Yields
    ------
    FakeGradient
        The instance every Overlay built during the test will share.
    """
    thegradient = FakeGradient()
    monkeypatch.setattr(rapidtide_dataset, "getagradient", lambda: thegradient)
    yield thegradient


def _writemnistylenifti(filepath, shape=(61, 73, 61), formcode=4, xscale=-3.0):
    """Write a nifti that claims to be in a standard space.

    Space detection keys off the form code plus the array dimensions, so both have
    to be set deliberately; a default header never reaches that branch.

    Parameters
    ----------
    filepath : str
        Where to write.
    shape : tuple, optional
        Array shape.  (61, 73, 61) and (91, 109, 91) are the two MNI152 grids the
        detection recognises.
    formcode : int, optional
        sform_code to declare.  4 means MNI152 in the nifti standard.
    xscale : float, optional
        The [0][0] entry of the affine.  Negative is neurological, positive
        radiological, zero indeterminate.

    Returns
    -------
    None
    """
    theaffine = np.eye(4)
    theaffine[0, 0] = xscale
    theaffine[1, 1] = 3.0
    theaffine[2, 2] = 3.0
    theimage = nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), theaffine)
    theimage.header.set_zooms((3.0, 3.0, 3.0)[: len(shape)])
    theimage.header["sform_code"] = formcode
    theimage.header["qform_code"] = 0
    nib.save(theimage, filepath)


@pytest.fixture
def legacy_output(tmp_path):
    """Create a rapidtide output set using the oldest, non-BIDS file names.

    The dataset class supports three naming generations, and picks between them by
    looking for particular files.  Only the BIDS one is otherwise exercised, so the
    legacy branches - a different map list, a different mask list and a different set
    of regressor filenames - are reached only through a fixture like this one.

    Returns
    -------
    str
        The file root the dataset should be opened with.
    """
    thefileroot = str(tmp_path / "legacy_")
    theshape = (5, 5, 5)

    with open(thefileroot + "desc-runoptions_info.json", "w") as thefile:
        json.dump(
            {
                "lowerpass": 0.01,
                "upperpass": 0.15,
                "fmrifreq": 0.5,
                "inputfreq": 0.5,
                "inputstarttime": 0.0,
                "oversampfactor": 2,
                "similaritymetric": "correlation",
                "validsimcalcstart": 0,
                "validsimcalcend": 100,
                "actual_passes": 2,
            },
            thefile,
        )

    # legacy regressors are plain text, one per pass
    for thename in [
        "reference_origres_prefilt.txt",
        "reference_origres.txt",
        "reference_resampres_pass1.txt",
        "reference_resampres_pass2.txt",
    ]:
        create_synthetic_timecourse(thefileroot + thename, length=100)

    for themap in ["lagtimes", "lagstrengths", "lagsigma", "MTT", "R2", "fitNorm", "fitcoff"]:
        create_synthetic_nifti(thefileroot + themap + ".nii.gz", shape=theshape)
    # note: no pass3 or pass4 regressor, so the optional-regressor skip is exercised
    for themask in ["lagmask", "refinemask", "meanmask"]:
        create_synthetic_nifti(
            thefileroot + themask + ".nii.gz",
            shape=theshape,
            data=np.ones(theshape, dtype=np.float32),
        )

    return thefileroot


# ============================================================================
# Overlay: LUT initialization
# ============================================================================


class TestOverlayLUT:
    """Tests for the lookup table setup an Overlay does when init_LUT is True."""

    def test_lut_is_built_on_init(self, tmp_path, fake_gradient):
        """With init_LUT on, the overlay builds a gradient and a lookup table."""
        thepath = str(tmp_path / "lutmap.nii.gz")
        create_synthetic_nifti(thepath, shape=(4, 4, 4))

        theoverlay = Overlay("lutmap", thepath, "base", init_LUT=True, verbose=0)

        assert theoverlay.theLUT is not None
        assert theoverlay.theLUT.shape == (512, 4)
        assert theoverlay.LUTname == gen_gray_state()["name"]
        assert len(fake_gradient.restoredstates) == 1

    def test_alpha_is_applied_to_intermediate_ticks_only(self, tmp_path, fake_gradient):
        """The alpha override rewrites the middle colour stops but leaves the two
        ends alone, which is what keeps a colour table transparent in the middle and
        opaque at its extremes."""
        thepath = str(tmp_path / "alphamap.nii.gz")
        create_synthetic_nifti(thepath, shape=(4, 4, 4))

        thelutstate = {
            "ticks": [
                (0.0, (0, 0, 0, 255)),
                (0.5, (128, 128, 128, 255)),
                (1.0, (255, 255, 255, 255)),
            ],
            "mode": "rgb",
            "name": "testlut",
        }
        theoverlay = Overlay(
            "alphamap",
            thepath,
            "base",
            lut_state=thelutstate,
            alpha=77,
            endalpha=11,
            init_LUT=True,
            verbose=2,
        )

        theticks = theoverlay.lut_state["ticks"]
        themiddle = [thetick for thetick in theticks if thetick[0] == 0.5]
        assert themiddle[0][1][3] == 77
        # the ends carry the end alpha, not the intermediate one
        assert theticks[0][1][3] == 11
        assert theticks[-1][1][3] == 11
        assert theoverlay.LUTname == "testlut"

    def test_alpha_none_leaves_the_ticks_alone(self, tmp_path, fake_gradient):
        """Passing no alpha means only the ends are touched."""
        thepath = str(tmp_path / "noalpha.nii.gz")
        create_synthetic_nifti(thepath, shape=(4, 4, 4))

        thelutstate = {
            "ticks": [
                (0.0, (0, 0, 0, 255)),
                (0.5, (128, 128, 128, 200)),
                (1.0, (255, 255, 255, 255)),
            ],
            "mode": "rgb",
            "name": "noalphalut",
        }
        theoverlay = Overlay("noalpha", thepath, "base", init_LUT=True, verbose=0)
        theoverlay.setLUT(thelutstate, alpha=None, endalpha=33)

        themiddle = [thetick for thetick in theoverlay.lut_state["ticks"] if thetick[0] == 0.5]
        assert themiddle[0][1][3] == 200


# ============================================================================
# Overlay: mask values, space detection and orientation
# ============================================================================


class TestOverlayMaskValues:
    """Tests for masks built from a list of label values rather than a threshold."""

    def test_filevals_select_specific_labels(self, tmp_path):
        """A filespec with a value list turns a label image into a binary mask
        containing exactly those labels.

        This is how a segmentation is turned into a gray or white matter mask, so
        the values not asked for have to come out as zero rather than as themselves.
        """
        thepath = str(tmp_path / "labels.nii.gz")
        thelabels = np.zeros((4, 4, 4), dtype=np.float32)
        thelabels[0, :, :] = 2.0
        thelabels[1, :, :] = 7.0
        thelabels[2, :, :] = 41.0
        create_synthetic_nifti(thepath, shape=(4, 4, 4), data=thelabels)

        theoverlay = Overlay(
            "labelmask", thepath + ":2,41", "base", isaMask=True, init_LUT=False, verbose=0
        )

        assert theoverlay.filevals == [2, 41]
        # the two requested labels survive
        assert np.all(theoverlay.data[0, :, :] == 1)
        assert np.all(theoverlay.data[2, :, :] == 1)
        # the one that was not requested does not, nor does the background
        assert np.all(theoverlay.data[1, :, :] == 0)
        assert np.all(theoverlay.data[3, :, :] == 0)


class TestOverlaySpaceDetection:
    """Tests for the standard-space and orientation detection done at load time."""

    def test_mni152_grid_is_recognised(self, tmp_path):
        """A form code of 4 on one of the two MNI152 grids means MNI152."""
        thepath = str(tmp_path / "mni3mm.nii.gz")
        _writemnistylenifti(thepath, shape=(61, 73, 61))

        theoverlay = Overlay("mni", thepath, "base", init_LUT=False, verbose=2)
        assert theoverlay.space == "MNI152"
        # a negative x scale is neurological
        assert theoverlay.RLfactor == -1.0

    def test_other_standard_grid_is_assumed_asymmetric(self, tmp_path):
        """A form code of 4 on any other grid is taken to be the asymmetric template."""
        thepath = str(tmp_path / "mniother.nii.gz")
        _writemnistylenifti(thepath, shape=(50, 50, 50))

        theoverlay = Overlay("mniother", thepath, "base", init_LUT=False, verbose=0)
        assert theoverlay.space == "MNI152NLin2009cAsym"

    def test_radiological_and_indeterminate_orientation(self, tmp_path, capsys):
        """The sign of the affine's first entry is the orientation, and a zero there
        means the file does not say."""
        theradpath = str(tmp_path / "radiological.nii.gz")
        _writemnistylenifti(theradpath, shape=(20, 20, 20), formcode=1, xscale=3.0)
        theradoverlay = Overlay("rad", theradpath, "base", init_LUT=False, verbose=2)
        assert theradoverlay.RLfactor == 1.0
        assert theradoverlay.space == "unspecified"
        assert "radiological orientation" in capsys.readouterr().out

        # an affine with a zero in the corner but still invertible: the x axis of the
        # array runs along y in world space, as it would after a 90 degree rotation
        theflatpath = str(tmp_path / "indeterminate.nii.gz")
        theobliqueaffine = np.array(
            [
                [0.0, 3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        theimage = nib.Nifti1Image(np.random.rand(20, 20, 20).astype(np.float32), theobliqueaffine)
        theimage.header["sform_code"] = 1
        theimage.header["qform_code"] = 0
        nib.save(theimage, theflatpath)
        theflatoverlay = Overlay("flat", theflatpath, "base", init_LUT=False, verbose=2)
        assert theflatoverlay.RLfactor == 0.0
        assert "indeterminate orientation" in capsys.readouterr().out

        # and the summary reports the same thing.  Both masks are filled in with
        # all-ones stand ins at construction, so a freshly loaded overlay always
        # reports them as set.
        theflatoverlay.summarize()
        thesummary = capsys.readouterr().out
        assert "orientation:       indeterminate" in thesummary
        assert "geometric mask is set" in thesummary
        assert "functional mask is set" in thesummary

        # clearing them is what the other half of the report is for
        theflatoverlay.geommask = None
        theflatoverlay.funcmask = None
        theflatoverlay.summarize()
        theclearedsummary = capsys.readouterr().out
        assert "geometric mask not set" in theclearedsummary
        assert "functional mask not set" in theclearedsummary


# ============================================================================
# RapidtideDataset: legacy naming
# ============================================================================


class TestLegacyNaming:
    """Tests for the two pre-BIDS naming conventions."""

    def test_oldstyle_dataset_loads(self, legacy_output):
        """The oldest layout has no desc- prefixes anywhere."""
        thesubject = RapidtideDataset("legacy", legacy_output, init_LUT=False, verbose=2)

        assert thesubject.bidsformat is False
        assert thesubject.newstylenames is False
        assert "lagtimes" in thesubject.getoverlays()
        assert "lagmask" in thesubject.getoverlays()
        # the legacy regressors are plain text files, read through the same machinery
        assert "prefilt" in thesubject.getregressors()

    def test_newstyle_dataset_is_detected_by_fitmask(self, legacy_output):
        """The intermediate layout is told apart by the presence of fitmask.nii.gz,
        and renames two things: the fit mask, and the linear fit coefficient map."""
        create_synthetic_nifti(
            legacy_output + "fitmask.nii.gz",
            shape=(5, 5, 5),
            data=np.ones((5, 5, 5), dtype=np.float32),
        )
        create_synthetic_nifti(legacy_output + "fitCoeff.nii.gz", shape=(5, 5, 5))

        thesubject = RapidtideDataset("newstyle", legacy_output, init_LUT=False, verbose=2)

        assert thesubject.bidsformat is False
        assert thesubject.newstylenames is True
        assert "fitcoff" in thesubject.getoverlays()
        assert "lagmask" in thesubject.getoverlays()

    def test_newstyle_usecorrout(self, legacy_output):
        """The intermediate layout has its own names for the similarity function
        and the fit failure reasons."""
        create_synthetic_nifti(
            legacy_output + "fitmask.nii.gz",
            shape=(5, 5, 5),
            data=np.ones((5, 5, 5), dtype=np.float32),
        )
        create_synthetic_nifti(legacy_output + "corrout.nii.gz", shape=(5, 5, 5, 7))
        create_synthetic_nifti(legacy_output + "corrfitfailreason.nii.gz", shape=(5, 5, 5, 7))

        thesubject = RapidtideDataset(
            "newstyle", legacy_output, usecorrout=True, init_LUT=False, verbose=2
        )

        assert thesubject.newstylenames is True
        assert "corrout" in thesubject.getoverlays()
        assert "failimage" in thesubject.getoverlays()

    def test_userise_swaps_in_the_epoch_maps(self, legacy_output):
        """A RISE analysis produces per-epoch maps instead of the width and norm
        maps, so the map list is different again."""
        for themap in ["risetime_epoch_0", "starttime_epoch_0", "maxamp_epoch_0"]:
            create_synthetic_nifti(legacy_output + themap + ".nii.gz", shape=(5, 5, 5))

        thesubject = RapidtideDataset(
            "rise", legacy_output, userise=True, init_LUT=False, verbose=2
        )

        theoverlays = thesubject.getoverlays()
        assert "risetime_epoch_0" in theoverlays
        assert "maxamp_epoch_0" in theoverlays
        # fitNorm is not in the RISE list, so it is not loaded even though it exists
        assert "fitNorm" not in theoverlays

    def test_usecorrout_adds_the_similarity_function(self, legacy_output):
        """The similarity function and the fit failure reasons are large and only
        loaded on request."""
        create_synthetic_nifti(legacy_output + "corrout.nii.gz", shape=(5, 5, 5, 7))
        create_synthetic_nifti(legacy_output + "failimage.nii.gz", shape=(5, 5, 5, 7))

        thesubject = RapidtideDataset(
            "corrout", legacy_output, usecorrout=True, init_LUT=False, verbose=2
        )

        theoverlays = thesubject.getoverlays()
        assert "corrout" in theoverlays
        assert "failimage" in theoverlays
        assert theoverlays["corrout"].tdim == 7

    def test_usecorrout_in_bids_format(self, rapidtide_output):
        """The BIDS layout names the same two files differently."""
        create_synthetic_nifti(rapidtide_output + "desc-corrout_info.nii.gz", shape=(5, 5, 5, 7))
        create_synthetic_nifti(
            rapidtide_output + "desc-corrfitfailreason_info.nii.gz", shape=(5, 5, 5, 7)
        )

        thesubject = RapidtideDataset(
            "corrout", rapidtide_output, usecorrout=True, init_LUT=False, verbose=2
        )

        assert "corrout" in thesubject.getoverlays()
        assert "failimage" in thesubject.getoverlays()


# ============================================================================
# RapidtideDataset: the optional overlays
# ============================================================================


class TestOptionalOverlays:
    """Tests for the anatomic, geometric and tissue overlays, all of which are
    loaded only if a matching file turns up."""

    def test_named_anatomic_is_used(self, rapidtide_output, tmp_path):
        """An explicitly named anatomic beats everything else."""
        theanatpath = str(tmp_path / "myanat.nii.gz")
        create_synthetic_nifti(theanatpath, shape=(5, 5, 5))

        thesubject = RapidtideDataset(
            "main", rapidtide_output, anatname=theanatpath, init_LUT=False, verbose=2
        )

        assert "anatomic" in thesubject.getoverlays()
        assert "anatomic" in thesubject.allloadedmaps

    def test_named_anatomic_that_does_not_exist_is_skipped(self, rapidtide_output, tmp_path):
        """A name that points nowhere leaves the dataset without an anatomic rather
        than falling through to one of the defaults."""
        create_synthetic_nifti(rapidtide_output + "mean.nii.gz", shape=(5, 5, 5))

        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            anatname=str(tmp_path / "nosuchanat.nii.gz"),
            init_LUT=False,
            verbose=2,
        )

        assert "anatomic" not in thesubject.getoverlays()

    def test_highres_head_is_preferred_over_highres(self, rapidtide_output):
        """With both present, the head image wins, since it shows more of the
        anatomy an overlay is being judged against."""
        create_synthetic_nifti(rapidtide_output + "highres_head.nii.gz", shape=(5, 5, 5))
        create_synthetic_nifti(rapidtide_output + "highres.nii.gz", shape=(5, 5, 5))

        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert thesubject.overlays["anatomic"].filename.endswith("highres_head.nii.gz")

    def test_highres_is_used_when_there_is_no_head(self, rapidtide_output):
        """Without the head image the brain-only one is next in line."""
        create_synthetic_nifti(rapidtide_output + "highres.nii.gz", shape=(5, 5, 5))

        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert thesubject.overlays["anatomic"].filename.endswith("highres.nii.gz")

    def test_mean_image_is_the_last_resort(self, rapidtide_output):
        """With no anatomic of any kind, the mean of the functional data stands in."""
        create_synthetic_nifti(rapidtide_output + "mean.nii.gz", shape=(5, 5, 5))

        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert thesubject.overlays["anatomic"].filename.endswith("mean.nii.gz")

    def test_no_anatomic_at_all(self, rapidtide_output):
        """None of the candidates existing is not an error."""
        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert "anatomic" not in thesubject.getoverlays()
        assert "anatomic" not in thesubject.allloadedmaps

    def test_named_geometric_mask_is_used(self, rapidtide_output, tmp_path):
        """A geometric mask trims the display to the brain."""
        themaskpath = str(tmp_path / "brainmask.nii.gz")
        create_synthetic_nifti(
            themaskpath, shape=(5, 5, 5), data=np.ones((5, 5, 5), dtype=np.float32)
        )

        thesubject = RapidtideDataset(
            "main", rapidtide_output, geommaskname=themaskpath, init_LUT=False, verbose=2
        )

        assert "geommask" in thesubject.getoverlays()
        assert "geommask" in thesubject.allloadedmaps

    def test_tissue_masks_are_loaded_from_their_specs(self, rapidtide_output, tmp_path):
        """Gray and white masks are given as file specs, so they can name the label
        values to pull out of a segmentation."""
        theseg = np.zeros((5, 5, 5), dtype=np.float32)
        theseg[0, :, :] = 2.0
        theseg[1, :, :] = 41.0
        thesegpath = str(tmp_path / "aseg.nii.gz")
        create_synthetic_nifti(thesegpath, shape=(5, 5, 5), data=theseg)

        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            graymaskspec=thesegpath + ":2",
            whitemaskspec=thesegpath + ":41",
            init_LUT=False,
            verbose=2,
        )

        assert "graymask" in thesubject.allloadedmaps
        assert "whitemask" in thesubject.allloadedmaps
        assert np.all(thesubject.overlays["graymask"].data[0, :, :] == 1)
        assert np.all(thesubject.overlays["graymask"].data[1, :, :] == 0)
        assert np.all(thesubject.overlays["whitemask"].data[1, :, :] == 1)

    def test_a_tissue_mask_that_does_not_exist_returns_none(self, rapidtide_output, tmp_path):
        """A spec naming a file that is not there returns None rather than False.

        Pinned, not fixed.  Both loaders are annotated ``-> bool``, and both return
        True or False on every path except this one: when the spec is set but the
        file is missing, control falls off the end of the ``if`` and the implicit
        None is returned.  The callers write ``if self._loadgraymask():``, so None
        behaves the same as False and nothing breaks today - but the annotation is
        wrong, and any caller comparing against False explicitly would be surprised.
        """
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            graymaskspec=str(tmp_path / "nosuchgray.nii.gz"),
            whitemaskspec=str(tmp_path / "nosuchwhite.nii.gz"),
            geommaskname=str(tmp_path / "nosuchgeom.nii.gz"),
            init_LUT=False,
            verbose=2,
        )

        assert thesubject._loadgraymask() is None
        assert thesubject._loadwhitemask() is None
        assert thesubject._loadgeommask() is None
        assert "graymask" not in thesubject.allloadedmaps
        assert "whitemask" not in thesubject.allloadedmaps
        assert "geommask" not in thesubject.allloadedmaps

    def test_forceoffset_overrides_the_header_toffset(self, rapidtide_output):
        """A dataset whose headers carry the wrong start time can be corrected at
        load, which is what this switch is for."""
        thesubject = RapidtideDataset(
            "main",
            rapidtide_output,
            forceoffset=True,
            offsettime=12.5,
            init_LUT=False,
            verbose=2,
        )

        assert thesubject.overlays["lagtimes"].toffset == 12.5

    def test_forcetr_raises_because_trval_is_never_set(self, rapidtide_output):
        """Asking for a forced TR fails with an AttributeError.

        Pinned, not fixed.  setupoverlays does ``self.overlays[themap].setTR(
        self.trval)`` when forcetr is on, but ``trval`` is neither a constructor
        parameter nor assigned anywhere in __init__ - the companion offsettime is
        both.  tidepool exposes this as ``--forcetr``, parses a value into
        ``args.trval``, and then constructs the dataset without passing it, so the
        option cannot work: it raises before any overlay is set up.  The fix is to
        thread trval through the constructor the way offsettime already is.
        """
        with pytest.raises(AttributeError, match="trval"):
            RapidtideDataset(
                "main",
                rapidtide_output,
                forcetr=True,
                init_LUT=False,
                verbose=2,
            )

        # and setTR itself works perfectly well when it is given a value
        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=0)
        thesubject.overlays["lagtimes"].setTR(3.7)
        assert thesubject.overlays["lagtimes"].tr == 3.7

    def test_no_geommask_without_a_coordinate_space(self, rapidtide_output):
        """With no mask named and no standard space, there is nothing to fall back
        on and the dataset simply has no geometric mask."""
        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert thesubject._loadgeommask() is False
        assert "geommask" not in thesubject.getoverlays()


# ============================================================================
# RapidtideDataset: run options handling
# ============================================================================


class TestRunOptionsDefaults:
    """Tests for what happens when a run options file is missing entries."""

    @pytest.fixture
    def sparse_output(self, tmp_path, monkeypatch):
        """A dataset whose run options carry nothing but a pass count.

        Older rapidtide versions wrote fewer keys, so setupregressors has a fallback
        for every one of them; a complete options file leaves all of those
        unexecuted.  The options are supplied by patching the reader rather than by
        writing a sparse file, because readoptionsfile does its own backfilling on
        the way past and would put most of the missing keys back.

        Returns
        -------
        str
            The file root.
        """
        thefileroot = str(tmp_path / "sparse_")
        theshape = (5, 5, 5)

        monkeypatch.setattr(tide_io, "readoptionsfile", lambda thefileroot: {"passes": 6})

        thedata = np.column_stack([np.sin(np.linspace(0, 4 * np.pi, 100)) for _ in range(2)]).T
        tide_io.writebidstsv(
            thefileroot + "desc-initialmovingregressor_timeseries",
            thedata,
            samplerate=1.0,
            columns=["prefilt", "postfilt"],
            starttime=0.0,
        )
        # pass1 is required too, and with six passes the last two specs ask for
        # pass5 and pass6 rather than the usual pass3 and pass4
        tide_io.writebidstsv(
            thefileroot + "desc-oversampledmovingregressor_timeseries",
            np.column_stack([np.sin(np.linspace(0, 4 * np.pi, 100)) for _ in range(6)]).T,
            samplerate=1.0,
            columns=[f"pass{i + 1}" for i in range(6)],
            starttime=0.0,
        )
        create_synthetic_nifti(thefileroot + "desc-maxtime_map.nii.gz", shape=theshape)
        return thefileroot

    def test_every_missing_option_gets_a_default(self, sparse_output):
        """Each absent key falls back to a documented default rather than raising."""
        thesubject = RapidtideDataset("sparse", sparse_output, init_LUT=False, verbose=2)

        assert thesubject.regressorfilterlimits == (0.0, 100.0)
        assert thesubject.fmrifreq == 1.0
        assert thesubject.inputfreq == 1.0
        assert thesubject.inputstarttime == 0.0
        assert thesubject.oversampfactor == 1
        assert thesubject.similaritymetric == "correlation"
        assert thesubject.regressorsimcalclimits == (0.0, 10000000.0)

    def test_pass_count_comes_from_passes_when_actual_passes_is_absent(self, sparse_output):
        """With more than four passes, the last two regressors loaded are the last
        two actually run, rather than a fixed pass 3 and pass 4."""
        thesubject = RapidtideDataset("sparse", sparse_output, init_LUT=False, verbose=2)

        assert thesubject.numberofpasses == 6
        thelabels = [thespec[1] for thespec in thesubject.regressorspecs]
        assert thelabels[-2:] == ["pass5", "pass6"]

    def test_a_required_regressor_that_is_missing_is_fatal(self, tmp_path):
        """Some regressors are optional and some are not; the initial moving
        regressor is the analysis input, so its absence cannot be shrugged off."""
        thefileroot = str(tmp_path / "noreg_")
        with open(thefileroot + "desc-runoptions_info.json", "w") as thefile:
            json.dump({"passes": 2, "lowerpass": 0.01, "upperpass": 0.15}, thefile)
        create_synthetic_nifti(thefileroot + "desc-maxtime_map.nii.gz", shape=(5, 5, 5))

        with pytest.raises(FileNotFoundError, match="regressor file"):
            RapidtideDataset("noreg", thefileroot, init_LUT=False, verbose=2)


# ============================================================================
# RapidtideDataset: standard space datasets
# ============================================================================


def _writestandardspacenifti(filepath, shape, voxelsize):
    """Write a nifti that declares itself to be in a standard space.

    Parameters
    ----------
    filepath : str
        Where to write.
    shape : tuple of int
        Array shape.
    voxelsize : float
        Isotropic voxel size, in mm.

    Returns
    -------
    None
    """
    theaffine = np.eye(4)
    theaffine[0, 0] = -voxelsize
    theaffine[1, 1] = voxelsize
    theaffine[2, 2] = voxelsize
    theimage = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), theaffine)
    theimage.header.set_zooms((voxelsize, voxelsize, voxelsize))
    theimage.header["sform_code"] = 4
    theimage.header["qform_code"] = 0
    nib.save(theimage, filepath)


def _makestandardspacedataset(tmp_path, thename, shape, voxelsize):
    """Build a minimal BIDS rapidtide output in a declared standard space.

    Standard-space detection is what unlocks the template, brain mask and atlas
    lookups, none of which a dataset in native space ever reaches.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to build in.
    thename : str
        Prefix for the file root.
    shape : tuple of int
        Array shape for every map.  (61, 73, 61) and (91, 109, 91) are the two grids
        recognised as MNI152; anything else is taken to be the asymmetric template.
    voxelsize : float
        Isotropic voxel size, in mm.

    Returns
    -------
    str
        The file root.
    """
    thefileroot = str(tmp_path / thename)
    with open(thefileroot + "desc-runoptions_info.json", "w") as thefile:
        json.dump({"passes": 2, "lowerpass": 0.01, "upperpass": 0.15}, thefile)
    tide_io.writebidstsv(
        thefileroot + "desc-initialmovingregressor_timeseries",
        np.column_stack([np.sin(np.linspace(0, 4 * np.pi, 100)) for _ in range(2)]).T,
        samplerate=1.0,
        columns=["prefilt", "postfilt"],
        starttime=0.0,
    )
    tide_io.writebidstsv(
        thefileroot + "desc-oversampledmovingregressor_timeseries",
        np.column_stack([np.sin(np.linspace(0, 4 * np.pi, 100)) for _ in range(2)]).T,
        samplerate=1.0,
        columns=["pass1", "pass2"],
        starttime=0.0,
    )
    _writestandardspacenifti(thefileroot + "desc-maxtime_map.nii.gz", shape, voxelsize)
    return thefileroot


@pytest.fixture
def fake_fsldir(tmp_path, monkeypatch):
    """Point FSLDIR at a temporary tree holding the 2mm MNI152 template and mask.

    The real FSL templates are found only on machines with FSL installed, so a test
    that depended on them would pass locally and be skipped in CI - exactly the
    branches most worth covering.  This makes them present everywhere.

    Yields
    ------
    str
        The directory FSLDIR was set to.
    """
    thefsldir = tmp_path / "fakefsl"
    thestandarddir = thefsldir / "data" / "standard"
    thestandarddir.mkdir(parents=True)
    _writestandardspacenifti(str(thestandarddir / "MNI152_T1_2mm.nii.gz"), (91, 109, 91), 2.0)
    _writestandardspacenifti(
        str(thestandarddir / "MNI152_T1_2mm_brain_mask.nii.gz"), (91, 109, 91), 2.0
    )
    monkeypatch.setenv("FSLDIR", str(thefsldir))
    yield str(thefsldir)


class TestStandardSpaceDatasets:
    """Tests for the template, brain mask and atlas lookups a standard-space
    dataset triggers."""

    def test_mni152_2mm_finds_its_template_and_mask(self, tmp_path, fake_fsldir):
        """At 2mm in MNI152 both the anatomic template and the brain mask come from
        the FSL installation."""
        thefileroot = _makestandardspacedataset(tmp_path, "mni2mm_", (91, 109, 91), 2.0)

        thesubject = RapidtideDataset("mni", thefileroot, init_LUT=False, verbose=2)

        assert thesubject.coordinatespace == "MNI152"
        assert thesubject.overlays["anatomic"].filename.endswith("MNI152_T1_2mm.nii.gz")
        assert thesubject.overlays["geommask"].filename.endswith("MNI152_T1_2mm_brain_mask.nii.gz")
        assert "anatomic" in thesubject.allloadedmaps
        assert "geommask" in thesubject.allloadedmaps

    def test_mni152_2mm_loads_the_atlas(self, tmp_path, fake_fsldir):
        """The arterial territory atlas ships with rapidtide at 2mm, and is what
        turns a lag map into a per-territory summary."""
        thefileroot = _makestandardspacedataset(tmp_path, "mniatlas_", (91, 109, 91), 2.0)

        thesubject = RapidtideDataset("mni", thefileroot, useatlas=True, init_LUT=False, verbose=2)

        assert "atlas" in thesubject.getoverlays()
        assert "atlasmask" in thesubject.getoverlays()
        assert "atlas" in thesubject.allloadedmaps
        assert "atlas" in thesubject.dispmaps
        assert len(thesubject.atlaslabels) > 0

    def test_mni152_3mm_has_no_atlas_to_load(self, tmp_path, fake_fsldir, capsys):
        """Only the 2mm atlas is shipped, so a 3mm dataset is told the template it
        wants is not there rather than silently going without."""
        thefileroot = _makestandardspacedataset(tmp_path, "mni3mm_", (61, 73, 61), 3.0)

        thesubject = RapidtideDataset(
            "mni3", thefileroot, useatlas=True, init_LUT=False, verbose=2
        )

        assert thesubject.coordinatespace == "MNI152"
        # the 3mm template does ship, so the anatomic is found
        assert thesubject.overlays["anatomic"].filename.endswith("MNI152_T1_3mm.nii.gz")
        assert "atlas" not in thesubject.getoverlays()
        theoutput = capsys.readouterr().out
        assert "does not exist!" in theoutput
        assert "there is not an atlas" in theoutput

    def test_mni152_3mm_brain_mask_is_missing(self, tmp_path, fake_fsldir):
        """The 3mm geometric mask points at a filename that is not shipped.

        Pinned, not fixed.  _loadgeommask builds the 3mm name as
        ``MNI152_T1_3mm_brain_mask_bin.nii.gz``, but what the reference directory
        actually contains is ``MNI152_T1_3mm_brain_mask.nii.gz`` - no ``_bin``.  So a
        3mm MNI dataset silently gets no geometric mask, while the 2mm one, which
        takes its mask from FSL under a different name, gets one.
        """
        thefileroot = _makestandardspacedataset(tmp_path, "mni3mask_", (61, 73, 61), 3.0)

        thesubject = RapidtideDataset("mni3", thefileroot, init_LUT=False, verbose=2)

        assert "geommask" not in thesubject.getoverlays()
        assert thesubject.geommaskname.endswith("MNI152_T1_3mm_brain_mask_bin.nii.gz")
        assert not os.path.isfile(thesubject.geommaskname)
        # while the file that does exist is the one without the suffix
        assert os.path.isfile(
            os.path.join(thesubject.referencedir, "MNI152_T1_3mm_brain_mask.nii.gz")
        )

    def test_asymmetric_template_is_looked_for_at_two_resolutions(self, tmp_path, fake_fsldir):
        """A standard-space file on any other grid is taken to be MNI152NLin2009cAsym,
        whose templates are looked up at 1mm and 2mm."""
        for theshape, thesize in [((50, 50, 50), 2.0), ((60, 60, 60), 1.0)]:
            thefileroot = _makestandardspacedataset(
                tmp_path, f"asym{int(thesize)}_", theshape, thesize
            )
            thesubject = RapidtideDataset("asym", thefileroot, init_LUT=False, verbose=2)

            assert thesubject.coordinatespace == "MNI152NLin2009cAsym"
            # the asymmetric templates are not shipped, so no anatomic is loaded
            assert "anatomic" not in thesubject.getoverlays()

    def test_no_geometric_mask_without_fsl(self, tmp_path, monkeypatch):
        """At 2mm the brain mask comes from FSL, so without FSL there is none."""
        monkeypatch.delenv("FSLDIR", raising=False)
        thefileroot = _makestandardspacedataset(tmp_path, "nofsl_", (91, 109, 91), 2.0)

        thesubject = RapidtideDataset("nofsl", thefileroot, init_LUT=False, verbose=2)

        assert thesubject.coordinatespace == "MNI152"
        assert "geommask" not in thesubject.getoverlays()
        assert "anatomic" not in thesubject.getoverlays()


# ============================================================================
# RapidtideDataset: the remaining anatomic fallbacks
# ============================================================================


class TestAnatomicFallbacks:
    """Tests for the tail of the anatomic search order."""

    @pytest.mark.parametrize(
        "thefilename", ["meanvalue.nii.gz", "desc-unfiltmean_map.nii.gz", "desc-mean_map.nii.gz"]
    )
    def test_each_mean_image_can_stand_in(self, rapidtide_output, thefilename):
        """Four different rapidtide versions named the mean image four different
        ways, and any of them will do as a backdrop."""
        create_synthetic_nifti(rapidtide_output + thefilename, shape=(5, 5, 5))

        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert thesubject.overlays["anatomic"].filename.endswith(thefilename)

    def test_mean_beats_the_later_candidates(self, rapidtide_output):
        """The search order is fixed, and mean.nii.gz comes before the rest."""
        for thefilename in [
            "mean.nii.gz",
            "meanvalue.nii.gz",
            "desc-unfiltmean_map.nii.gz",
            "desc-mean_map.nii.gz",
        ]:
            create_synthetic_nifti(rapidtide_output + thefilename, shape=(5, 5, 5))

        thesubject = RapidtideDataset("main", rapidtide_output, init_LUT=False, verbose=2)

        assert thesubject.overlays["anatomic"].filename.endswith("mean.nii.gz")


# ============================================================================
# Overlay: affine selection
# ============================================================================


class TestOverlayAffineSelection:
    """Tests for which of a nifti header's three affines an overlay uses."""

    def _writewithcodes(self, filepath, sformcode, qformcode):
        """Write a nifti declaring the given sform and qform codes.

        Parameters
        ----------
        filepath : str
            Where to write.
        sformcode : int
            sform_code to declare.
        qformcode : int
            qform_code to declare.

        Returns
        -------
        None
        """
        theaffine = np.eye(4)
        theaffine[0, 0] = -2.0
        theaffine[1, 1] = 2.0
        theaffine[2, 2] = 2.0
        theimage = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), theaffine)
        theimage.header.set_zooms((2.0, 2.0, 2.0))
        theimage.header.set_sform(theaffine, code=sformcode)
        theimage.header.set_qform(theaffine, code=qformcode)
        nib.save(theimage, filepath)

    def test_qform_is_used_when_there_is_no_sform(self, tmp_path):
        """The qform is the fallback, and is what an older scanner writes."""
        thepath = str(tmp_path / "qformonly.nii.gz")
        self._writewithcodes(thepath, 0, 1)

        theoverlay = Overlay("qform", thepath, "base", init_LUT=False, verbose=0)
        assert theoverlay.affine[0][0] == pytest.approx(-2.0)
        assert theoverlay.RLfactor == -1.0

    def test_base_affine_is_used_when_neither_is_set(self, tmp_path):
        """With both codes zero the header carries no orientation at all, and the
        voxel-size-only base affine is all there is to go on."""
        thepath = str(tmp_path / "nocodes.nii.gz")
        self._writewithcodes(thepath, 0, 0)

        theoverlay = Overlay("nocodes", thepath, "base", init_LUT=False, verbose=0)
        # the base affine is built from the zooms, and puts the origin at the centre
        assert theoverlay.affine.shape == (4, 4)
        assert abs(theoverlay.affine[0][0]) == pytest.approx(2.0)


# ============================================================================
# RapidtideDataset: map consistency
# ============================================================================


class TestFuncMapConsistency:
    """Tests for the checks that keep a mismatched map out of a dataset."""

    def test_a_map_with_the_wrong_dimensions_is_fatal(self, rapidtide_output):
        """Every map has to be on the same grid, since they are indexed together by
        a single set of cursor coordinates."""
        create_synthetic_nifti(rapidtide_output + "desc-MTT_map.nii.gz", shape=(6, 6, 6))

        with pytest.raises(SystemExit):
            RapidtideDataset("mismatch", rapidtide_output, init_LUT=False, verbose=2)

    def test_a_map_with_the_wrong_voxel_size_is_fatal(self, rapidtide_output):
        """Matching dimensions are not enough - the voxels have to be the same size
        too, or the maps cover different amounts of brain."""
        thedata = np.random.rand(5, 5, 5).astype(np.float32)
        theaffine = np.eye(4)
        theaffine[0, 0] = -2.0
        theaffine[1, 1] = 2.0
        theaffine[2, 2] = 2.0
        theimage = nib.Nifti1Image(thedata, theaffine)
        theimage.header.set_zooms((4.0, 4.0, 4.0))
        theimage.header["sform_code"] = 1
        nib.save(theimage, rapidtide_output + "desc-MTT_map.nii.gz")

        with pytest.raises(SystemExit):
            RapidtideDataset("mismatch", rapidtide_output, init_LUT=False, verbose=2)

    def test_varchange_is_inverted_on_load(self, rapidtide_output):
        """The in-band variance change map is stored one way round and displayed the
        other, so it is negated as it is read."""
        thedata = np.linspace(-1.0, 1.0, 125).reshape((5, 5, 5)).astype(np.float32)
        create_synthetic_nifti(
            rapidtide_output + "desc-lfofilterInbandVarianceChange_map.nii.gz",
            shape=(5, 5, 5),
            data=thedata.copy(),
        )

        thesubject = RapidtideDataset("inverted", rapidtide_output, init_LUT=False, verbose=2)

        np.testing.assert_allclose(thesubject.overlays["varChange"].data, -thedata, atol=1e-6)


def main(debug=False):
    """Run tests manually for local development."""
    exampleroot = get_examples_path(runninglocally)
    testtemproot = get_test_temp_path(runninglocally)
    print(f"get_examples_path={exampleroot}")
    print(f"get_test_temp_path={testtemproot}")
    datafileroot = os.path.join(testtemproot, "sub-RAPIDTIDETESTDATASET_")

    # run rapidtide
    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETESTDATASET"),
        "--spatialfilt",
        "2",
        "--simcalcrange",
        "4",
        "-1",
        "--nprocs",
        "-1",
        "--passes",
        "3",
        "--despecklepasses",
        "3",
        "--corrmask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_restrictedmask.nii.gz"),
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))

    compareresults = tide_util.comparerapidtideruns(
        os.path.join(testtemproot, "sub-RAPIDTIDETESTDATASET"),
        os.path.join(testtemproot, "sub-RAPIDTIDETESTDATASET"),
    )
    if debug:
        print(compareresults)

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
    runninglocally = True
    main(debug=True)
