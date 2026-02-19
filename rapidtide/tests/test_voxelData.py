#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

_THISFILE = os.path.abspath(__file__)
_REPOROOT = os.path.abspath(os.path.join(os.path.dirname(_THISFILE), "..", ".."))
if _REPOROOT not in sys.path:
    sys.path.insert(0, _REPOROOT)

import rapidtide.voxelData as tide_voxelData


class DummyHdr(dict):
    def get_xyzt_units(self):
        return ("mm", "sec")


def datavolume_tests(debug=False):
    if debug:
        print("datavolume_tests")
    d3 = tide_voxelData.dataVolume((2, 3, 4), shared=False, dtype=np.float64, thepid=0)
    assert d3.byvol().shape == (2, 3, 4)
    assert d3.byslice().shape == (6, 4)
    assert d3.byvoxel().shape == (24,)
    d3.destroy()

    d4 = tide_voxelData.dataVolume((2, 3, 4, 5), shared=False, dtype=np.float32, thepid=0)
    assert d4.byvol().shape == (2, 3, 4, 5)
    assert d4.byslice().shape == (6, 4, 5)
    assert d4.byvoxel().shape == (24, 5)
    d4.destroy()


def load_branch_tests(debug=False):
    if debug:
        print("load_branch_tests")

    # text load path
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=True),
        patch("rapidtide.voxelData.tide_io.readvecs", return_value=np.ones((3, 10))),
    ):
        v = tide_voxelData.VoxelData.__new__(tide_voxelData.VoxelData)
        v.filename = "dummy.txt"
        v.filetype = None
        v.load()
        assert v.nim is None
        assert v.nim_data.shape == (3, 10)
        assert v.resident is True

    # cifti load path
    cifti_hdr = {"dim": [5, 1, 1, 1, 10, 7]}
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=False),
        patch("rapidtide.voxelData.tide_io.checkifcifti", return_value=True),
        patch(
            "rapidtide.voxelData.tide_io.readfromcifti",
            return_value=(
                None,
                cifti_hdr,
                np.ones((7, 10)),
                cifti_hdr,
                [0, 1, 1, 1, 10, 7],
                [0, 1, 1, 1, 0.72],
                None,
            ),
        ),
    ):
        v = tide_voxelData.VoxelData.__new__(tide_voxelData.VoxelData)
        v.filename = "dummy.dtseries.nii"
        v.filetype = None
        v.load()
        assert v.filetype == "cifti"
        assert v.nim is None
        assert v.nim_data.shape == (7, 10)
        assert v.resident is True

    # nifti load path
    nim = SimpleNamespace(affine=np.eye(4))
    hdr = DummyHdr({"dim": [4, 2, 3, 4, 5], "pixdim": [0, 2.0, 2.0, 2.5, 1.5], "toffset": 0.0})
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=False),
        patch("rapidtide.voxelData.tide_io.checkifcifti", return_value=False),
        patch(
            "rapidtide.voxelData.tide_io.readfromnifti",
            return_value=(
                nim,
                np.ones((2, 3, 4, 5)),
                hdr,
                [4, 2, 3, 4, 5],
                [0, 2.0, 2.0, 2.5, 1.5],
            ),
        ),
    ):
        v = tide_voxelData.VoxelData.__new__(tide_voxelData.VoxelData)
        v.filename = "dummy.nii.gz"
        v.filetype = None
        v.load()
        assert v.nim is nim
        assert v.nim_data.shape == (2, 3, 4, 5)
        assert v.resident is True


def voxeldata_readdata_nifti_tests(debug=False):
    if debug:
        print("voxeldata_readdata_nifti_tests")

    nim = SimpleNamespace(affine=np.eye(4))
    hdr = DummyHdr({"dim": [4, 2, 3, 4, 5], "pixdim": [0, 2.0, 2.0, 2.5, 1.5], "toffset": 0.1})
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=False),
        patch("rapidtide.voxelData.tide_io.checkifcifti", return_value=False),
        patch(
            "rapidtide.voxelData.tide_io.readfromnifti",
            return_value=(
                nim,
                np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5),
                hdr,
                [4, 2, 3, 4, 5],
                [0, 2.0, 2.0, 2.5, 1.5],
            ),
        ),
        patch("rapidtide.voxelData.tide_io.parseniftidims", return_value=(2, 3, 4, 5)),
        patch("rapidtide.voxelData.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.5, 1.5)),
    ):
        v = tide_voxelData.VoxelData("dummy.nii.gz", timestep=0.0, validstart=1, validend=3)
        assert v.filetype == "nifti"
        assert v.getdims() == (2, 3, 4, 5)
        assert v.getsizes() == (2.0, 2.0, 2.5, 1.5)
        assert v.realtimepoints == 3
        assert v.nativefmrishape == (2, 3, 4, 3)
        assert v.byvol().shape == (2, 3, 4, 5)
        assert v.byvoltrimmed().shape == (2, 3, 4, 3)
        assert v.byvoxel().shape == (24, 3)
        assert v.byslice().shape == (6, 4, 3)
        v.setvalidvoxels(np.array([0, 2, 5], dtype=int))
        vd = v.validdata()
        assert vd.shape == (3, 3)
        h2 = v.copyheader(numtimepoints=1, tr=2.0, toffset=0.5)
        assert h2["dim"][0] == 3
        assert h2["dim"][4] == 1
        assert np.fabs(h2["pixdim"][4] - 2.0) < 1e-12
        assert np.fabs(h2["toffset"] - 0.5) < 1e-12
        h3 = v.copyheader(numtimepoints=4)
        assert h3["dim"][0] == 4
        assert h3["dim"][4] == 4
        v.summarize()
        v.unload()
        assert v.resident is False
        # triggers reload path via byvol
        assert v.byvol().shape == (2, 3, 4, 5)


def voxeldata_readdata_text_and_cifti_tests(debug=False):
    if debug:
        print("voxeldata_readdata_text_and_cifti_tests")

    # text path with timestep enforced
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=True),
        patch("rapidtide.voxelData.tide_io.readvecs", return_value=np.ones((3, 9))),
    ):
        with pytest.raises(ValueError):
            tide_voxelData.VoxelData("dummy.txt", timestep=0.0)
        vt = tide_voxelData.VoxelData("dummy.txt", timestep=2.5, validstart=1, validend=7)
        assert vt.filetype == "text"
        assert vt.timepoints == 9
        assert vt.realtimepoints == 7
        assert vt.nativespaceshape == 3
        assert vt.nativefmrishape == (3, 7)
        assert vt.byvoltrimmed().shape == (3, 7)
        assert vt.byvoxel().shape == (3, 7)
        assert vt.byslice().shape == (3, 1, 7)
        assert vt.copyheader() is None

    # cifti path
    cifti_hdr = {"dim": [5, 1, 1, 1, 8, 4], "pixdim": [0, 1, 1, 1, 0.72], "toffset": 0.0}
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=False),
        patch("rapidtide.voxelData.tide_io.checkifcifti", return_value=True),
        patch(
            "rapidtide.voxelData.tide_io.readfromcifti",
            return_value=(
                None,
                cifti_hdr,
                np.ones((4, 8)),
                cifti_hdr,
                [5, 1, 1, 1, 8, 4],
                [0, 1, 1, 1, 0.72],
                None,
            ),
        ),
        patch("rapidtide.voxelData.tide_io.parseniftisizes", return_value=(1.0, 1.0, 1.0, 0.72)),
    ):
        vc = tide_voxelData.VoxelData("dummy.dtseries.nii", timestep=0.0, validstart=0, validend=7)
        assert vc.filetype == "cifti"
        assert vc.timestep == 0.72
        assert vc.byvoltrimmed().shape == (4, 8)
        assert vc.byvoxel().shape == (4, 8)
        assert vc.byslice().shape == (4, 1, 8)
        h = vc.copyheader(numtimepoints=8)
        timeindex = h["dim"][0] - 1
        spaceindex = h["dim"][0]
        assert h["dim"][timeindex] == 8
        assert h["dim"][spaceindex] == vc.numspatiallocs


def smooth_tests(debug=False):
    if debug:
        print("smooth_tests")

    nim = SimpleNamespace(affine=np.eye(4))
    hdr = DummyHdr({"dim": [4, 2, 2, 2, 3], "pixdim": [0, 2.0, 2.0, 2.0, 1.0], "toffset": 0.0})
    with (
        patch("rapidtide.voxelData.tide_io.checkiftext", return_value=False),
        patch("rapidtide.voxelData.tide_io.checkifcifti", return_value=False),
        patch(
            "rapidtide.voxelData.tide_io.readfromnifti",
            return_value=(
                nim,
                np.ones((2, 2, 2, 3)),
                hdr,
                [4, 2, 2, 2, 3],
                [0, 2.0, 2.0, 2.0, 1.0],
            ),
        ),
        patch("rapidtide.voxelData.tide_io.parseniftidims", return_value=(2, 2, 2, 3)),
        patch("rapidtide.voxelData.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.0)),
        patch(
            "rapidtide.voxelData.tide_filt.ssmooth", side_effect=lambda x, y, z, s, arr: arr + 1.0
        ),
    ):
        v = tide_voxelData.VoxelData("dummy.nii.gz", timestep=1.0, validstart=0, validend=2)

        # automatic sigma path
        usedsigma = v.smooth(-1.0, premask=False, showprogressbar=False)
        assert usedsigma > 0.0

        # premask with tissue-only missing masks should fail
        with pytest.raises(ValueError):
            v.smooth(1.0, premask=True, premasktissueonly=True, graymask=None, whitemask=None)

        # premask with missing brainmask should fail
        with pytest.raises(ValueError):
            v.smooth(1.0, premask=True, premasktissueonly=False, brainmask=None)

        # premask tissue-only success
        g = np.ones((2, 2, 2))
        w = np.zeros((2, 2, 2))
        usedsigma2 = v.smooth(1.0, premask=True, premasktissueonly=True, graymask=g, whitemask=w)
        assert np.fabs(usedsigma2 - 1.0) < 1e-12

    # cifti/text smoothing disabled branch
    vc = tide_voxelData.VoxelData.__new__(tide_voxelData.VoxelData)
    vc.filetype = "cifti"
    vc.xdim = vc.ydim = vc.slicethickness = 1.0
    vc.validstart = 0
    vc.validend = 0
    vc.nim_data = np.ones((3, 2))
    vc.byvol = lambda: vc.nim_data
    assert np.fabs(vc.smooth(2.0) - 0.0) < 1e-12


def test_voxelData(debug=False, displayplots=False):
    np.random.seed(12345)
    datavolume_tests(debug=debug)
    load_branch_tests(debug=debug)
    voxeldata_readdata_nifti_tests(debug=debug)
    voxeldata_readdata_text_and_cifti_tests(debug=debug)
    smooth_tests(debug=debug)


if __name__ == "__main__":
    test_voxelData(debug=True, displayplots=False)
