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
import copy
import json
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

import rapidtide.io as tide_io

# ==================== Helpers ====================

EPSILON = 1e-5


def _make_4d_nifti(tmpdir, shape=(4, 5, 3, 10), name="test4d"):
    """Create a 4D NIfTI file with random data and return path, data, header."""
    data = np.random.RandomState(42).randn(*shape).astype(np.float64)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    filepath = os.path.join(tmpdir, name + ".nii.gz")
    nib.save(img, filepath)
    return filepath, data, img.header.copy()


def _make_3d_nifti(tmpdir, shape=(4, 5, 3), name="test3d"):
    """Create a 3D NIfTI file with random data and return path, data, header."""
    data = np.random.RandomState(42).randn(*shape).astype(np.float64)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    filepath = os.path.join(tmpdir, name + ".nii.gz")
    nib.save(img, filepath)
    return filepath, data, img.header.copy()


def _make_text_vec(tmpdir, data, name="testvec.txt"):
    """Write a 1D array to a text file, one value per line."""
    filepath = os.path.join(tmpdir, name)
    np.savetxt(filepath, data)
    return filepath


def _make_par_file(tmpdir, npoints=20, name="motion.par"):
    """Create a .par file with 6 columns of motion data."""
    rng = np.random.RandomState(42)
    data = rng.randn(npoints, 6)
    filepath = os.path.join(tmpdir, name)
    np.savetxt(filepath, data)
    return filepath, data


def _make_2d_array(nrows=6, ncols=100):
    """Create a 2D test array."""
    arr = np.zeros((nrows, ncols), dtype=float)
    t = np.linspace(0, 1.0, ncols, endpoint=False)
    arr[0, :] = t
    arr[1, :] = np.sin(t * 2 * np.pi)
    arr[2, :] = np.cos(t * 2 * np.pi)
    arr[3, :] = np.sin(2 * t * 2 * np.pi)
    arr[4, :] = np.cos(2 * t * 2 * np.pi)
    arr[5, :] = np.sin(3 * t * 2 * np.pi)
    return arr


# ==================== parseniftidims tests ====================


def parseniftidims_basic(debug=False):
    """Test parseniftidims extracts correct dimensions."""
    if debug:
        print("parseniftidims_basic")
    dims = np.array([4, 64, 80, 32, 100, 1, 1, 1])
    nx, ny, nz, nt = tide_io.parseniftidims(dims)
    assert nx == 64
    assert ny == 80
    assert nz == 32
    assert nt == 100


def parseniftidims_3d(debug=False):
    """Test parseniftidims with 3D data (nt=1)."""
    if debug:
        print("parseniftidims_3d")
    dims = np.array([3, 10, 20, 30, 1, 1, 1, 1])
    nx, ny, nz, nt = tide_io.parseniftidims(dims)
    assert nx == 10
    assert ny == 20
    assert nz == 30
    assert nt == 1


# ==================== parseniftisizes tests ====================


def parseniftisizes_basic(debug=False):
    """Test parseniftisizes extracts correct sizes."""
    if debug:
        print("parseniftisizes_basic")
    sizes = np.array([-1.0, 2.0, 2.0, 3.0, 1.5, 0.0, 0.0, 0.0])
    dx, dy, dz, dt = tide_io.parseniftisizes(sizes)
    assert dx == 2.0
    assert dy == 2.0
    assert dz == 3.0
    assert dt == 1.5


# ==================== niftifromarray tests ====================


def niftifromarray_basic(debug=False):
    """Test niftifromarray creates valid NIfTI image."""
    if debug:
        print("niftifromarray_basic")
    data = np.random.randn(10, 10, 10)
    img = tide_io.niftifromarray(data)
    assert isinstance(img, nib.Nifti1Image)
    assert img.shape == (10, 10, 10)
    np.testing.assert_array_almost_equal(img.affine, np.eye(4))


def niftifromarray_4d(debug=False):
    """Test niftifromarray with 4D data."""
    if debug:
        print("niftifromarray_4d")
    data = np.zeros((3, 4, 5, 6))
    img = tide_io.niftifromarray(data)
    assert img.shape == (3, 4, 5, 6)


# ==================== niftihdrfromarray tests ====================


def niftihdrfromarray_basic(debug=False):
    """Test niftihdrfromarray creates valid NIfTI header."""
    if debug:
        print("niftihdrfromarray_basic")
    data = np.random.randn(5, 5, 5)
    hdr = tide_io.niftihdrfromarray(data)
    assert isinstance(hdr, nib.Nifti1Header)
    dims = hdr["dim"]
    assert dims[1] == 5
    assert dims[2] == 5
    assert dims[3] == 5


# ==================== niftisplitext tests ====================


def niftisplitext_niigz(debug=False):
    """Test niftisplitext with .nii.gz extension."""
    if debug:
        print("niftisplitext_niigz")
    name, ext = tide_io.niftisplitext("myimage.nii.gz")
    assert name == "myimage"
    assert ext == ".nii.gz"


def niftisplitext_nii(debug=False):
    """Test niftisplitext with .nii extension."""
    if debug:
        print("niftisplitext_nii")
    name, ext = tide_io.niftisplitext("myimage.nii")
    assert name == "myimage"
    assert ext == ".nii"


def niftisplitext_path(debug=False):
    """Test niftisplitext with full path."""
    if debug:
        print("niftisplitext_path")
    name, ext = tide_io.niftisplitext("/data/sub-01/func/bold.nii.gz")
    assert name == "/data/sub-01/func/bold"
    assert ext == ".nii.gz"


# ==================== checkifcifti tests ====================


def checkifcifti_nifti(debug=False):
    """Test checkifcifti returns False for standard NIfTI file."""
    if debug:
        print("checkifcifti_nifti")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath, _, _ = _make_3d_nifti(tmpdir)
        assert not tide_io.checkifcifti(filepath)


# ==================== checkspaceresmatch tests ====================


def checkspaceresmatch_same(debug=False):
    """Test checkspaceresmatch with identical resolutions."""
    if debug:
        print("checkspaceresmatch_same")
    sizes = np.array([-1.0, 2.0, 2.0, 2.0, 1.0])
    assert tide_io.checkspaceresmatch(sizes, sizes)


def checkspaceresmatch_different(debug=False):
    """Test checkspaceresmatch with different resolutions."""
    if debug:
        print("checkspaceresmatch_different")
    sizes1 = np.array([-1.0, 2.0, 2.0, 2.0, 1.0])
    sizes2 = np.array([-1.0, 3.0, 2.0, 2.0, 1.0])
    assert not tide_io.checkspaceresmatch(sizes1, sizes2)


def checkspaceresmatch_within_tolerance(debug=False):
    """Test checkspaceresmatch with sizes within tolerance."""
    if debug:
        print("checkspaceresmatch_within_tolerance")
    sizes1 = np.array([-1.0, 2.0, 2.0, 2.0, 1.0])
    sizes2 = np.array([-1.0, 2.0001, 2.0001, 2.0001, 1.0])
    assert tide_io.checkspaceresmatch(sizes1, sizes2, tolerance=1e-3)


# ==================== checkspacedimmatch tests ====================


def checkspacedimmatch_same(debug=False):
    """Test checkspacedimmatch with identical dimensions."""
    if debug:
        print("checkspacedimmatch_same")
    dims = np.array([4, 64, 64, 32, 100, 1, 1, 1])
    assert tide_io.checkspacedimmatch(dims, dims)


def checkspacedimmatch_different(debug=False):
    """Test checkspacedimmatch with different dimensions."""
    if debug:
        print("checkspacedimmatch_different")
    dims1 = np.array([4, 64, 64, 32, 100, 1, 1, 1])
    dims2 = np.array([4, 64, 64, 33, 100, 1, 1, 1])
    assert not tide_io.checkspacedimmatch(dims1, dims2)


# ==================== checktimematch tests (with skip) ====================


def checktimematch_with_skip(debug=False):
    """Test checktimematch with skip parameters."""
    if debug:
        print("checktimematch_with_skip")
    dims1 = np.array([4, 64, 64, 32, 100, 1, 1, 1])
    dims2 = np.array([4, 64, 64, 32, 95, 1, 1, 1])
    assert tide_io.checktimematch(dims1, dims2, numskip1=5, numskip2=0)
    assert not tide_io.checktimematch(dims1, dims2, numskip1=0, numskip2=0)


# ==================== checkdatamatch tests ====================


def checkdatamatch_identical(debug=False):
    """Test checkdatamatch with identical data."""
    if debug:
        print("checkdatamatch_identical")
    data = np.array([1.0, 2.0, 3.0, 4.0])
    msematch, absmatch = tide_io.checkdatamatch(data, data)
    assert msematch
    assert absmatch


def checkdatamatch_different(debug=False):
    """Test checkdatamatch with very different data."""
    if debug:
        print("checkdatamatch_different")
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([10.0, 20.0, 30.0])
    msematch, absmatch = tide_io.checkdatamatch(data1, data2)
    assert not msematch
    assert not absmatch


def checkdatamatch_close(debug=False):
    """Test checkdatamatch with close but not identical data."""
    if debug:
        print("checkdatamatch_close")
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.0 + 1e-14, 2.0, 3.0])
    msematch, absmatch = tide_io.checkdatamatch(data1, data2)
    assert msematch
    assert absmatch


# ==================== checkniftifilematch tests ====================


def checkniftifilematch_same(debug=False):
    """Test checkniftifilematch with identical files."""
    if debug:
        print("checkniftifilematch_same")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath, data, hdr = _make_3d_nifti(tmpdir)
        assert tide_io.checkniftifilematch(filepath, filepath)


def checkniftifilematch_different(debug=False):
    """Test checkniftifilematch with different files."""
    if debug:
        print("checkniftifilematch_different")
    with tempfile.TemporaryDirectory() as tmpdir:
        f1, _, _ = _make_3d_nifti(tmpdir, shape=(4, 5, 3), name="a")
        f2, _, _ = _make_3d_nifti(tmpdir, shape=(6, 7, 3), name="b")
        assert not tide_io.checkniftifilematch(f1, f2)


# ==================== makedestarray tests ====================


def makedestarray_nifti_3d(debug=False):
    """Test makedestarray for 3D NIFTI output."""
    if debug:
        print("makedestarray_nifti_3d")
    arr, iss = tide_io.makedestarray((4, 5, 3), filetype="nifti")
    assert arr.shape == (60,)
    assert iss == 60


def makedestarray_nifti_4d(debug=False):
    """Test makedestarray for 4D NIFTI output."""
    if debug:
        print("makedestarray_nifti_4d")
    arr, iss = tide_io.makedestarray((4, 5, 3, 10), filetype="nifti")
    assert arr.shape == (60, 10)
    assert iss == 60


def makedestarray_text_1d(debug=False):
    """Test makedestarray for 1D text output."""
    if debug:
        print("makedestarray_text_1d")
    arr, iss = tide_io.makedestarray(100, filetype="text")
    assert arr.shape == (100,)
    assert iss == 100


def makedestarray_text_2d(debug=False):
    """Test makedestarray for 2D text output."""
    if debug:
        print("makedestarray_text_2d")
    arr, iss = tide_io.makedestarray((100, 5), filetype="text")
    assert arr.shape == (100, 5)
    assert iss == 100


def makedestarray_cifti(debug=False):
    """Test makedestarray for CIFTI output."""
    if debug:
        print("makedestarray_cifti")
    arr, iss = tide_io.makedestarray((10, 50), filetype="cifti")
    assert arr.shape == (50, 10)
    assert iss == 50


# ==================== populatemap tests ====================


def populatemap_1d_validvoxels(debug=False):
    """Test populatemap with 1D array and valid voxels."""
    if debug:
        print("populatemap_1d_validvoxels")
    themap = np.array([10.0, 20.0, 30.0])
    validvoxels = np.array([1, 3, 5])
    outarray = np.zeros(8)
    result = tide_io.populatemap(themap, 8, validvoxels, outarray)
    assert result[1] == 10.0
    assert result[3] == 20.0
    assert result[5] == 30.0
    assert result[0] == 0.0
    assert result[2] == 0.0


def populatemap_1d_no_validvoxels(debug=False):
    """Test populatemap with 1D array and no mask."""
    if debug:
        print("populatemap_1d_no_validvoxels")
    themap = np.array([1.0, 2.0, 3.0, 4.0])
    outarray = np.zeros(4)
    result = tide_io.populatemap(themap, 4, None, outarray)
    np.testing.assert_array_equal(result, themap)


def populatemap_2d_validvoxels(debug=False):
    """Test populatemap with 2D array and valid voxels."""
    if debug:
        print("populatemap_2d_validvoxels")
    themap = np.array([[1.0, 2.0], [3.0, 4.0]])
    validvoxels = np.array([0, 2])
    outarray = np.zeros((4, 2))
    result = tide_io.populatemap(themap, 4, validvoxels, outarray)
    np.testing.assert_array_equal(result[0], [1.0, 2.0])
    np.testing.assert_array_equal(result[2], [3.0, 4.0])
    np.testing.assert_array_equal(result[1], [0.0, 0.0])


# ==================== makeMNI tests ====================


def makeMNI_res2(debug=False):
    """Test makeMNI at 2mm resolution."""
    if debug:
        print("makeMNI_res2")
    data, hdr, affine = tide_io.makeMNI(2)
    assert data.shape == (91, 109, 91, 1)
    assert affine[0][0] == -2.0


def makeMNI_res1(debug=False):
    """Test makeMNI at 1mm resolution."""
    if debug:
        print("makeMNI_res1")
    data, hdr, affine = tide_io.makeMNI(1)
    assert data.shape == (182, 218, 182, 1)


def makeMNI_timepoints(debug=False):
    """Test makeMNI with multiple timepoints."""
    if debug:
        print("makeMNI_timepoints")
    data, hdr, affine = tide_io.makeMNI(2, timepoints=5)
    assert data.shape == (91, 109, 91, 5)


def makeMNI_invalid_res(debug=False):
    """Test makeMNI raises ValueError for invalid resolution."""
    if debug:
        print("makeMNI_invalid_res")
    try:
        tide_io.makeMNI(3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ==================== savemaplist tests ====================


def savemaplist_nifti(debug=False):
    """Test savemaplist writes NIFTI files with JSON sidecars."""
    if debug:
        print("savemaplist_nifti")
    with tempfile.TemporaryDirectory() as tmpdir:
        destshape = (4, 5, 3)
        data = np.random.randn(60).astype(np.float64)
        hdr = tide_io.niftihdrfromarray(np.zeros(destshape))
        maplist = [
            (data, "lagtime", "stat", "seconds", "Lag time map"),
        ]
        bidsdict = {"Dataset": "test"}
        outname = os.path.join(tmpdir, "testoutput")
        tide_io.savemaplist(
            outname,
            maplist,
            None,
            destshape,
            hdr,
            bidsdict,
            filetype="nifti",
            savejson=True,
        )
        assert os.path.exists(os.path.join(tmpdir, "testoutput_desc-lagtime_stat.nii.gz"))
        assert os.path.exists(os.path.join(tmpdir, "testoutput_desc-lagtime_stat.json"))
        with open(os.path.join(tmpdir, "testoutput_desc-lagtime_stat.json"), "r") as f:
            jdata = json.load(f)
        assert jdata["Units"] == "seconds"
        assert jdata["Description"] == "Lag time map"


def savemaplist_text(debug=False):
    """Test savemaplist writes text files."""
    if debug:
        print("savemaplist_text")
    with tempfile.TemporaryDirectory() as tmpdir:
        destshape = 10
        data = np.arange(10, dtype=np.float64)
        hdr = None  # not used for text
        maplist = [
            (data, "values", "stat", None, None),
        ]
        bidsdict = {}
        outname = os.path.join(tmpdir, "testtext")
        tide_io.savemaplist(
            outname,
            maplist,
            None,
            destshape,
            hdr,
            bidsdict,
            filetype="text",
            savejson=False,
        )
        assert os.path.exists(os.path.join(tmpdir, "testtext_values.txt"))


# ==================== writedicttojson / readdictfromjson tests ====================


def writedicttojson_basic(debug=False):
    """Test writedicttojson writes valid JSON."""
    if debug:
        print("writedicttojson_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        d = {"key1": "value1", "key2": 42, "key3": 3.14}
        tide_io.writedicttojson(d, filepath)
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            result = json.load(f)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert abs(result["key3"] - 3.14) < 1e-10


def writedicttojson_numpy_types(debug=False):
    """Test writedicttojson handles numpy types."""
    if debug:
        print("writedicttojson_numpy_types")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        d = {
            "int_val": np.int32(42),
            "float_val": np.float64(3.14),
            "arr_val": np.array([1, 2, 3]),
        }
        tide_io.writedicttojson(d, filepath)
        with open(filepath, "r") as f:
            result = json.load(f)
        assert result["int_val"] == 42
        assert isinstance(result["int_val"], int)
        assert abs(result["float_val"] - 3.14) < 1e-10
        assert result["arr_val"] == [1, 2, 3]


def readdictfromjson_basic(debug=False):
    """Test readdictfromjson reads valid JSON."""
    if debug:
        print("readdictfromjson_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        d = {"a": 1, "b": "hello"}
        with open(filepath, "w") as f:
            json.dump(d, f)
        result = tide_io.readdictfromjson(filepath)
        assert result["a"] == 1
        assert result["b"] == "hello"


def readdictfromjson_no_extension(debug=False):
    """Test readdictfromjson with filename without .json extension."""
    if debug:
        print("readdictfromjson_no_extension")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        d = {"x": 99}
        with open(filepath, "w") as f:
            json.dump(d, f)
        result = tide_io.readdictfromjson(os.path.join(tmpdir, "test"))
        assert result["x"] == 99


def readdictfromjson_missing(debug=False):
    """Test readdictfromjson returns empty dict for missing file."""
    if debug:
        print("readdictfromjson_missing")
    result = tide_io.readdictfromjson("/nonexistent/path/file.json")
    assert result == {}


# ==================== readbidssidecar tests ====================


def readbidssidecar_basic(debug=False):
    """Test readbidssidecar reads JSON sidecar."""
    if debug:
        print("readbidssidecar_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        # readbidssidecar uses os.path.splitext, so for "test.nii.gz" it strips ".gz"
        # and looks for "test.nii.json"
        jsonpath = os.path.join(tmpdir, "test.nii.json")
        d = {"RepetitionTime": 2.0, "TaskName": "rest"}
        with open(jsonpath, "w") as f:
            json.dump(d, f)
        result = tide_io.readbidssidecar(os.path.join(tmpdir, "test.nii.gz"))
        assert result["RepetitionTime"] == 2.0
        assert result["TaskName"] == "rest"


def readbidssidecar_missing(debug=False):
    """Test readbidssidecar returns empty dict when file missing."""
    if debug:
        print("readbidssidecar_missing")
    result = tide_io.readbidssidecar("/nonexistent/test.nii.gz")
    assert result == {}


# ==================== readlabelledtsv tests ====================


def readlabelledtsv_basic(debug=False):
    """Test readlabelledtsv reads labeled TSV file."""
    if debug:
        print("readlabelledtsv_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "confounds.tsv")
        import pandas as pd

        df = pd.DataFrame(
            {
                "trans_x": [1.0, 2.0, 3.0],
                "trans_y": [4.0, 5.0, 6.0],
                "rot_z": [0.1, 0.2, 0.3],
            }
        )
        df.to_csv(filepath, sep="\t", index=False)
        result = tide_io.readlabelledtsv(os.path.join(tmpdir, "confounds"))
        assert "trans_x" in result
        assert "trans_y" in result
        assert "rot_z" in result
        np.testing.assert_array_almost_equal(result["trans_x"], [1.0, 2.0, 3.0])


def readlabelledtsv_nan_handling(debug=False):
    """Test readlabelledtsv replaces NaN with 0."""
    if debug:
        print("readlabelledtsv_nan_handling")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.tsv")
        import pandas as pd

        df = pd.DataFrame({"col1": [1.0, np.nan, 3.0]})
        df.to_csv(filepath, sep="\t", index=False)
        result = tide_io.readlabelledtsv(os.path.join(tmpdir, "data"))
        assert result["col1"][1] == 0.0


# ==================== readcsv tests ====================


def readcsv_with_header(debug=False):
    """Test readcsv with header line."""
    if debug:
        print("readcsv_with_header")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.csv")
        import pandas as pd

        df = pd.DataFrame({"signal": [1.0, 2.0, 3.0], "noise": [0.1, 0.2, 0.3]})
        df.to_csv(filepath, index=False)
        result = tide_io.readcsv(os.path.join(tmpdir, "data"))
        assert "signal" in result
        assert "noise" in result
        np.testing.assert_array_almost_equal(result["signal"], [1.0, 2.0, 3.0])


def readcsv_without_header(debug=False):
    """Test readcsv without header line (numeric first row)."""
    if debug:
        print("readcsv_without_header")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.csv")
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.savetxt(filepath, data, delimiter=",")
        result = tide_io.readcsv(os.path.join(tmpdir, "data"))
        assert "col_00" in result
        assert "col_01" in result


# ==================== readconfounds tests ====================


def readconfounds_basic(debug=False):
    """Test readconfounds reads and returns dict of arrays."""
    if debug:
        print("readconfounds_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "confounds.tsv")
        import pandas as pd

        df = pd.DataFrame({"csf": [0.1, 0.2, 0.3], "wm": [0.4, 0.5, 0.6]})
        df.to_csv(filepath, sep="\t", index=False)
        # Need a json file to make it a bidscontinuous file, or it will be plaintsv
        result = tide_io.readconfounds(filepath)
        assert len(result) == 2


# ==================== sliceinfo tests ====================


def sliceinfo_sequential(debug=False):
    """Test sliceinfo with sequential slice times."""
    if debug:
        print("sliceinfo_sequential")
    slicetimes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    tr = 0.5
    numsteps, stepsize, offsets = tide_io.sliceinfo(slicetimes, tr)
    assert numsteps == 5
    assert abs(stepsize - 0.1) < 1e-10


def sliceinfo_interleaved(debug=False):
    """Test sliceinfo with interleaved slice times."""
    if debug:
        print("sliceinfo_interleaved")
    slicetimes = np.array([0.0, 0.2, 0.4, 0.1, 0.3])
    tr = 0.5
    numsteps, stepsize, offsets = tide_io.sliceinfo(slicetimes, tr)
    assert numsteps > 0


# ==================== getslicetimesfromfile tests ====================


def getslicetimesfromfile_json(debug=False):
    """Test getslicetimesfromfile with JSON file."""
    if debug:
        print("getslicetimesfromfile_json")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "slicetimes.json")
        d = {"SliceTiming": [0.0, 0.1, 0.2, 0.3]}
        with open(filepath, "w") as f:
            json.dump(d, f)
        times, normalized, isbids = tide_io.getslicetimesfromfile(filepath)
        assert len(times) == 4
        assert abs(times[0] - 0.0) < 1e-10
        assert abs(times[2] - 0.2) < 1e-10
        assert not normalized
        assert isbids


def getslicetimesfromfile_text(debug=False):
    """Test getslicetimesfromfile with text file."""
    if debug:
        print("getslicetimesfromfile_text")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "slicetimes.txt")
        np.savetxt(filepath, [0.0, 0.1, 0.2, 0.3])
        times, normalized, isbids = tide_io.getslicetimesfromfile(filepath)
        assert len(times) == 4
        assert normalized
        assert not isbids


# ==================== writedict / readdict tests ====================


def writedict_readdict_roundtrip(debug=False):
    """Test writedict and readdict roundtrip."""
    if debug:
        print("writedict_readdict_roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "testdict.txt")
        d = {"name": "test", "value": "42", "param": "3.14"}
        tide_io.writedict(d, filepath)
        result = tide_io.readdict(filepath)
        assert result["name"] == "test"
        assert result["value"] == "42"
        assert result["param"] == "3.14"


def writedict_machinereadable(debug=False):
    """Test writedict with machinereadable format."""
    if debug:
        print("writedict_machinereadable")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "testdict.txt")
        d = {"key1": "val1"}
        tide_io.writedict(d, filepath, machinereadable=True)
        with open(filepath, "r") as f:
            content = f.read()
        assert "{" in content
        assert "}" in content
        assert '"key1"' in content


def readdict_missing(debug=False):
    """Test readdict returns empty dict for missing file."""
    if debug:
        print("readdict_missing")
    result = tide_io.readdict("/nonexistent/file.txt")
    assert result == {}


# ==================== writevec / readvec tests ====================


def writevec_readvec_roundtrip(debug=False):
    """Test writevec and readvec roundtrip."""
    if debug:
        print("writevec_readvec_roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "vec.txt")
        data = np.array([1.0, 2.5, 3.7, 4.1, 5.9])
        tide_io.writevec(data, filepath)
        result = tide_io.readvec(filepath)
        np.testing.assert_array_almost_equal(result, data, decimal=5)


def writevec_lineendings(debug=False):
    """Test writevec with different line endings."""
    if debug:
        print("writevec_lineendings")
    with tempfile.TemporaryDirectory() as tmpdir:
        for le in ["mac", "win", "linux"]:
            filepath = os.path.join(tmpdir, f"vec_{le}.txt")
            data = np.array([1.0, 2.0, 3.0])
            tide_io.writevec(data, filepath, lineend=le)
            assert os.path.exists(filepath)


# ==================== writenpvecs tests ====================


def writenpvecs_2d(debug=False):
    """Test writenpvecs with 2D array."""
    if debug:
        print("writenpvecs_2d")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "vecs.txt")
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tide_io.writenpvecs(data, filepath)
        result = tide_io.readvecs(filepath)
        np.testing.assert_array_almost_equal(result, data, decimal=5)


def writenpvecs_1d(debug=False):
    """Test writenpvecs with 1D array."""
    if debug:
        print("writenpvecs_1d")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "vec1d.txt")
        data = np.array([1.0, 2.0, 3.0, 4.0])
        tide_io.writenpvecs(data, filepath)
        assert os.path.exists(filepath)


def writenpvecs_csv_headers(debug=False):
    """Test writenpvecs with CSV format and headers."""
    if debug:
        print("writenpvecs_csv_headers")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "vecs.csv")
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tide_io.writenpvecs(data, filepath, ascsv=True, headers=["A", "B"])
        with open(filepath, "r") as f:
            first_line = f.readline().strip()
        assert "A" in first_line
        assert "B" in first_line


def writenpvecs_altmethod_false(debug=False):
    """Test writenpvecs with altmethod=False."""
    if debug:
        print("writenpvecs_altmethod_false")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "vecs_alt.txt")
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tide_io.writenpvecs(data, filepath, altmethod=False)
        assert os.path.exists(filepath)


# ==================== readvecs tests ====================


def readvecs_basic(debug=False):
    """Test readvecs reads multi-column text file."""
    if debug:
        print("readvecs_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.txt")
        data = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.savetxt(filepath, data)
        result = tide_io.readvecs(filepath)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[1], [4.0, 5.0, 6.0])


def readvecs_colspec(debug=False):
    """Test readvecs with column specification."""
    if debug:
        print("readvecs_colspec")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.txt")
        data = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        np.savetxt(filepath, data)
        result = tide_io.readvecs(filepath, colspec="1")
        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], [4.0, 5.0, 6.0])


def readvecs_with_header(debug=False):
    """Test readvecs auto-detects header line."""
    if debug:
        print("readvecs_with_header")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.txt")
        with open(filepath, "w") as f:
            f.write("col1 col2\n")
            f.write("1.0 2.0\n")
            f.write("3.0 4.0\n")
        result = tide_io.readvecs(filepath)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result[0], [1.0, 3.0])


# ==================== readtc tests ====================


def readtc_text_singlecol(debug=False):
    """Test readtc reads single-column text file."""
    if debug:
        print("readtc_text_singlecol")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "tc.txt")
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Write as a single row (space-separated) so readvecs returns (1, N) and
        # transpose gives (N, 1), which is 2D. Use colnum=0 to select the column.
        np.savetxt(filepath, data)
        tc, freq, start = tide_io.readtc(filepath, colnum=0)
        np.testing.assert_array_almost_equal(tc, data)
        assert freq is None
        assert start is None


def readtc_text_multicol(debug=False):
    """Test readtc reads specific column from multi-column file."""
    if debug:
        print("readtc_text_multicol")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "tc.txt")
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        np.savetxt(filepath, data)
        tc, freq, start = tide_io.readtc(filepath, colnum=1)
        np.testing.assert_array_almost_equal(tc, [10.0, 20.0, 30.0])


# ==================== readlabels tests ====================


def readlabels_basic(debug=False):
    """Test readlabels reads lines from file."""
    if debug:
        print("readlabels_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "labels.txt")
        with open(filepath, "w") as f:
            f.write("label1\n")
            f.write("label2\n")
            f.write("label3\n")
        result = tide_io.readlabels(filepath)
        assert result == ["label1", "label2", "label3"]


# ==================== readcolfromtextfile tests ====================


def readcolfromtextfile_basic(debug=False):
    """Test readcolfromtextfile reads a single column."""
    if debug:
        print("readcolfromtextfile_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.txt")
        data = np.array([10.0, 20.0, 30.0])
        np.savetxt(filepath, data.reshape(-1, 1))
        result = tide_io.readcolfromtextfile(filepath)
        np.testing.assert_array_almost_equal(result, data)


# ==================== unique tests ====================


def unique_basic(debug=False):
    """Test unique removes duplicates preserving order."""
    if debug:
        print("unique_basic")
    result = tide_io.unique([3, 1, 2, 1, 3, 4])
    assert result == [3, 1, 2, 4]


def unique_no_duplicates(debug=False):
    """Test unique with no duplicates."""
    if debug:
        print("unique_no_duplicates")
    result = tide_io.unique([1, 2, 3])
    assert result == [1, 2, 3]


def unique_empty(debug=False):
    """Test unique with empty list."""
    if debug:
        print("unique_empty")
    result = tide_io.unique([])
    assert result == []


# ==================== makecolname tests ====================


def makecolname_basic(debug=False):
    """Test makecolname generates correct names."""
    if debug:
        print("makecolname_basic")
    assert tide_io.makecolname(0, 0) == "col_00"
    assert tide_io.makecolname(5, 10) == "col_15"
    assert tide_io.makecolname(1, 2) == "col_03"
    assert tide_io.makecolname(99, 1) == "col_100"


# ==================== processnamespec tests ====================


def processnamespec_with_colspec(debug=False):
    """Test processnamespec with column specification."""
    if debug:
        print("processnamespec_with_colspec")
    name, vals = tide_io.processnamespec("data.txt:1,3,5", "mask", "regions")
    assert name == "data.txt"
    assert vals == [1, 3, 5]


def processnamespec_without_colspec(debug=False):
    """Test processnamespec without column specification."""
    if debug:
        print("processnamespec_without_colspec")
    name, vals = tide_io.processnamespec("data.txt", "mask", "regions")
    assert name == "data.txt"
    assert vals is None


# ==================== readoptionsfile tests ====================


def readoptionsfile_json(debug=False):
    """Test readoptionsfile reads JSON options file."""
    if debug:
        print("readoptionsfile_json")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "opts.json")
        d = {"filtertype": "lfo", "lowerpass": 0.01, "upperpass": 0.15}
        with open(filepath, "w") as f:
            json.dump(d, f)
        result = tide_io.readoptionsfile(os.path.join(tmpdir, "opts"))
        assert result["filtertype"] == "lfo"
        assert result["lowerpass"] == 0.01


def readoptionsfile_txt(debug=False):
    """Test readoptionsfile reads TXT options file."""
    if debug:
        print("readoptionsfile_txt")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "opts.txt")
        d = {"filtertype": "lfo", "lowerpass": "0.01"}
        tide_io.writedict(d, filepath)
        result = tide_io.readoptionsfile(os.path.join(tmpdir, "opts"))
        assert result["filtertype"] == "lfo"


def readoptionsfile_backwards_compat(debug=False):
    """Test readoptionsfile fills default filter limits for old files."""
    if debug:
        print("readoptionsfile_backwards_compat")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "opts.json")
        d = {"filtertype": "lfo"}  # no lowerpass key
        with open(filepath, "w") as f:
            json.dump(d, f)
        result = tide_io.readoptionsfile(os.path.join(tmpdir, "opts"))
        assert result["lowerstop"] == 0.009
        assert result["lowerpass"] == 0.010
        assert result["upperpass"] == 0.15
        assert result["upperstop"] == 0.20


def readoptionsfile_missing(debug=False):
    """Test readoptionsfile raises FileNotFoundError."""
    if debug:
        print("readoptionsfile_missing")
    try:
        tide_io.readoptionsfile("/nonexistent/opts")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


# ==================== readbidstsv tests ====================


def readbidstsv_basic(debug=False):
    """Test readbidstsv reads BIDS TSV with JSON sidecar."""
    if debug:
        print("readbidstsv_basic")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test_physio")
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tide_io.writebidstsv(
            root, data, samplerate=10.0, starttime=0.5, columns=["sig1", "sig2"], compressed=True
        )
        sr, st, cols, rd, comp, colsrc, extra = tide_io.readbidstsv(root + ".json")
        assert sr == 10.0
        assert st == 0.5
        assert "sig1" in cols
        assert "sig2" in cols
        assert rd.shape == (2, 3)
        assert comp


def readbidstsv_neednotexist(debug=False):
    """Test readbidstsv with neednotexist=True for missing file."""
    if debug:
        print("readbidstsv_neednotexist")
    result = tide_io.readbidstsv("/nonexistent/file.json", neednotexist=True)
    assert result[0] is None
    assert result[3] is None


def readbidstsv_colspec(debug=False):
    """Test readbidstsv with column specification."""
    if debug:
        print("readbidstsv_colspec")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test_physio")
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        tide_io.writebidstsv(
            root, data, samplerate=10.0, columns=["a", "b", "c"], compressed=False
        )
        sr, st, cols, rd, comp, colsrc, extra = tide_io.readbidstsv(
            root + ".json", colspec="b", warn=False
        )
        assert cols == ["b"]
        assert rd.shape == (1, 3)
        np.testing.assert_array_almost_equal(rd[0], [4.0, 5.0, 6.0])


# ==================== readcolfrombidstsv tests ====================


def readcolfrombidstsv_by_name(debug=False):
    """Test readcolfrombidstsv by column name."""
    if debug:
        print("readcolfrombidstsv_by_name")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test_physio")
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tide_io.writebidstsv(
            root, data, samplerate=10.0, columns=["sig1", "sig2"], compressed=True
        )
        sr, st, col_data = tide_io.readcolfrombidstsv(root + ".json", columnname="sig2")
        assert sr == 10.0
        np.testing.assert_array_almost_equal(col_data, [4.0, 5.0, 6.0])


def readcolfrombidstsv_by_number(debug=False):
    """Test readcolfrombidstsv by column number."""
    if debug:
        print("readcolfrombidstsv_by_number")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test_physio")
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tide_io.writebidstsv(
            root, data, samplerate=10.0, columns=["sig1", "sig2"], compressed=True
        )
        sr, st, col_data = tide_io.readcolfrombidstsv(root + ".json", columnnum=1)
        np.testing.assert_array_almost_equal(col_data, [4.0, 5.0, 6.0])


# ==================== writebidstsv tests (extra coverage) ====================


def writebidstsv_1d_data(debug=False):
    """Test writebidstsv reshapes 1D data to 2D."""
    if debug:
        print("writebidstsv_1d_data")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test")
        data = np.array([1.0, 2.0, 3.0, 4.0])
        tide_io.writebidstsv(root, data, samplerate=10.0, columns=["signal"])
        sr, st, cols, rd, comp, colsrc, extra = tide_io.readbidstsv(root + ".json")
        assert rd.shape == (1, 4)
        np.testing.assert_array_almost_equal(rd[0], data)


def writebidstsv_uncompressed(debug=False):
    """Test writebidstsv with uncompressed output."""
    if debug:
        print("writebidstsv_uncompressed")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test")
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tide_io.writebidstsv(root, data, samplerate=5.0, compressed=False)
        assert os.path.exists(root + ".tsv")
        assert os.path.exists(root + ".json")


def writebidstsv_extraheaderinfo(debug=False):
    """Test writebidstsv with extra header info."""
    if debug:
        print("writebidstsv_extraheaderinfo")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test")
        data = np.array([[1.0, 2.0]])
        extra = {"TaskName": "rest", "Units": "mmHg"}
        tide_io.writebidstsv(root, data, samplerate=10.0, extraheaderinfo=extra)
        with open(root + ".json", "r") as f:
            d = json.load(f)
        assert d["TaskName"] == "rest"
        assert d["Units"] == "mmHg"


def writebidstsv_omitjson(debug=False):
    """Test writebidstsv with omitjson=True."""
    if debug:
        print("writebidstsv_omitjson")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "test")
        data = np.array([[1.0, 2.0]])
        tide_io.writebidstsv(root, data, samplerate=10.0, omitjson=True)
        assert os.path.exists(root + ".tsv.gz")
        assert not os.path.exists(root + ".json")


# ==================== savetonifti dtype tests ====================


def savetonifti_various_dtypes(debug=False):
    """Test savetonifti with various numpy dtypes."""
    if debug:
        print("savetonifti_various_dtypes")
    with tempfile.TemporaryDirectory() as tmpdir:
        base_data = np.random.RandomState(42).randn(4, 5, 3).astype(np.float64)
        hdr = tide_io.niftihdrfromarray(base_data)
        dtypes = [
            np.uint8,
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.int8,
            np.uint16,
            np.uint32,
            np.int64,
            np.uint64,
        ]
        for dtype in dtypes:
            outpath = os.path.join(tmpdir, f"test_{dtype.__name__}")
            typed_data = base_data.astype(dtype)
            tide_io.savetonifti(typed_data, copy.deepcopy(hdr), outpath)
            _, read_data, _, _, _ = tide_io.readfromnifti(outpath)
            assert read_data is not None


# ==================== readfromnifti headeronly test ====================


def readfromnifti_headeronly(debug=False):
    """Test readfromnifti with headeronly=True."""
    if debug:
        print("readfromnifti_headeronly")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath, _, _ = _make_3d_nifti(tmpdir)
        _, data, hdr, dims, sizes = tide_io.readfromnifti(filepath, headeronly=True)
        assert data is None
        assert hdr is not None


def readfromnifti_no_extension(debug=False):
    """Test readfromnifti finds file without extension."""
    if debug:
        print("readfromnifti_no_extension")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath, orig_data, _ = _make_3d_nifti(tmpdir, name="noext")
        # Remove the .nii.gz from the path
        noext_path = os.path.join(tmpdir, "noext")
        _, data, _, _, _ = tide_io.readfromnifti(noext_path)
        np.testing.assert_array_almost_equal(data, orig_data)


def readfromnifti_missing(debug=False):
    """Test readfromnifti raises FileNotFoundError for missing file."""
    if debug:
        print("readfromnifti_missing")
    try:
        tide_io.readfromnifti("/nonexistent/file")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


# ==================== colspectolist additional tests ====================


def colspectolist_simple_range(debug=False):
    """Test colspectolist with simple range."""
    if debug:
        print("colspectolist_simple_range")
    result = tide_io.colspectolist("1-5")
    assert result == [1, 2, 3, 4, 5]


def colspectolist_single(debug=False):
    """Test colspectolist with single value."""
    if debug:
        print("colspectolist_single")
    result = tide_io.colspectolist("7")
    assert result == [7]


def colspectolist_macro(debug=False):
    """Test colspectolist with SSEG_CSF macro."""
    if debug:
        print("colspectolist_macro")
    result = tide_io.colspectolist("SSEG_CSF")
    assert 4 in result
    assert 5 in result
    assert 14 in result
    assert 15 in result
    assert 24 in result
    assert 43 in result
    assert 44 in result


def colspectolist_none(debug=False):
    """Test colspectolist returns None for None input."""
    if debug:
        print("colspectolist_none")
    result = tide_io.colspectolist(None)
    assert result is None


# ==================== parsefilespec additional tests ====================


def parsefilespec_no_colspec(debug=False):
    """Test parsefilespec without column spec."""
    if debug:
        print("parsefilespec_no_colspec")
    name, spec = tide_io.parsefilespec("myfile.txt")
    assert name == "myfile.txt"
    assert spec is None


def parsefilespec_with_colspec(debug=False):
    """Test parsefilespec with column spec."""
    if debug:
        print("parsefilespec_with_colspec")
    name, spec = tide_io.parsefilespec("data.nii.gz:1,3,5-8")
    assert name == "data.nii.gz"
    assert spec == "1,3,5-8"


# ==================== Main test function ====================


def test_io2(debug=False):
    # parseniftidims
    if debug:
        print("Running parseniftidims tests")
    parseniftidims_basic(debug=debug)
    parseniftidims_3d(debug=debug)

    # parseniftisizes
    if debug:
        print("Running parseniftisizes tests")
    parseniftisizes_basic(debug=debug)

    # niftifromarray
    if debug:
        print("Running niftifromarray tests")
    niftifromarray_basic(debug=debug)
    niftifromarray_4d(debug=debug)

    # niftihdrfromarray
    if debug:
        print("Running niftihdrfromarray tests")
    niftihdrfromarray_basic(debug=debug)

    # niftisplitext
    if debug:
        print("Running niftisplitext tests")
    niftisplitext_niigz(debug=debug)
    niftisplitext_nii(debug=debug)
    niftisplitext_path(debug=debug)

    # checkifcifti
    if debug:
        print("Running checkifcifti tests")
    checkifcifti_nifti(debug=debug)

    # checkspaceresmatch
    if debug:
        print("Running checkspaceresmatch tests")
    checkspaceresmatch_same(debug=debug)
    checkspaceresmatch_different(debug=debug)
    checkspaceresmatch_within_tolerance(debug=debug)

    # checkspacedimmatch
    if debug:
        print("Running checkspacedimmatch tests")
    checkspacedimmatch_same(debug=debug)
    checkspacedimmatch_different(debug=debug)

    # checktimematch with skip
    if debug:
        print("Running checktimematch tests")
    checktimematch_with_skip(debug=debug)

    # checkdatamatch
    if debug:
        print("Running checkdatamatch tests")
    checkdatamatch_identical(debug=debug)
    checkdatamatch_different(debug=debug)
    checkdatamatch_close(debug=debug)

    # checkniftifilematch
    if debug:
        print("Running checkniftifilematch tests")
    checkniftifilematch_same(debug=debug)
    checkniftifilematch_different(debug=debug)

    # makedestarray
    if debug:
        print("Running makedestarray tests")
    makedestarray_nifti_3d(debug=debug)
    makedestarray_nifti_4d(debug=debug)
    makedestarray_text_1d(debug=debug)
    makedestarray_text_2d(debug=debug)
    makedestarray_cifti(debug=debug)

    # populatemap
    if debug:
        print("Running populatemap tests")
    populatemap_1d_validvoxels(debug=debug)
    populatemap_1d_no_validvoxels(debug=debug)
    populatemap_2d_validvoxels(debug=debug)

    # makeMNI
    if debug:
        print("Running makeMNI tests")
    makeMNI_res2(debug=debug)
    makeMNI_res1(debug=debug)
    makeMNI_timepoints(debug=debug)
    makeMNI_invalid_res(debug=debug)

    # savemaplist
    if debug:
        print("Running savemaplist tests")
    savemaplist_nifti(debug=debug)
    savemaplist_text(debug=debug)

    # writedicttojson / readdictfromjson
    if debug:
        print("Running JSON dict tests")
    writedicttojson_basic(debug=debug)
    writedicttojson_numpy_types(debug=debug)
    readdictfromjson_basic(debug=debug)
    readdictfromjson_no_extension(debug=debug)
    readdictfromjson_missing(debug=debug)

    # readbidssidecar
    if debug:
        print("Running readbidssidecar tests")
    readbidssidecar_basic(debug=debug)
    readbidssidecar_missing(debug=debug)

    # readlabelledtsv
    if debug:
        print("Running readlabelledtsv tests")
    readlabelledtsv_basic(debug=debug)
    readlabelledtsv_nan_handling(debug=debug)

    # readcsv
    if debug:
        print("Running readcsv tests")
    readcsv_with_header(debug=debug)
    readcsv_without_header(debug=debug)

    # readconfounds
    if debug:
        print("Running readconfounds tests")
    readconfounds_basic(debug=debug)

    # sliceinfo
    if debug:
        print("Running sliceinfo tests")
    sliceinfo_sequential(debug=debug)
    sliceinfo_interleaved(debug=debug)

    # getslicetimesfromfile
    if debug:
        print("Running getslicetimesfromfile tests")
    getslicetimesfromfile_json(debug=debug)
    getslicetimesfromfile_text(debug=debug)

    # writedict / readdict
    if debug:
        print("Running writedict/readdict tests")
    writedict_readdict_roundtrip(debug=debug)
    writedict_machinereadable(debug=debug)
    readdict_missing(debug=debug)

    # writevec / readvec
    if debug:
        print("Running writevec/readvec tests")
    writevec_readvec_roundtrip(debug=debug)
    writevec_lineendings(debug=debug)

    # writenpvecs
    if debug:
        print("Running writenpvecs tests")
    writenpvecs_2d(debug=debug)
    writenpvecs_1d(debug=debug)
    writenpvecs_csv_headers(debug=debug)
    writenpvecs_altmethod_false(debug=debug)

    # readvecs
    if debug:
        print("Running readvecs tests")
    readvecs_basic(debug=debug)
    readvecs_colspec(debug=debug)
    readvecs_with_header(debug=debug)

    # readtc
    if debug:
        print("Running readtc tests")
    readtc_text_singlecol(debug=debug)
    readtc_text_multicol(debug=debug)

    # readlabels
    if debug:
        print("Running readlabels tests")
    readlabels_basic(debug=debug)

    # readcolfromtextfile
    if debug:
        print("Running readcolfromtextfile tests")
    readcolfromtextfile_basic(debug=debug)

    # unique
    if debug:
        print("Running unique tests")
    unique_basic(debug=debug)
    unique_no_duplicates(debug=debug)
    unique_empty(debug=debug)

    # makecolname
    if debug:
        print("Running makecolname tests")
    makecolname_basic(debug=debug)

    # processnamespec
    if debug:
        print("Running processnamespec tests")
    processnamespec_with_colspec(debug=debug)
    processnamespec_without_colspec(debug=debug)

    # readoptionsfile
    if debug:
        print("Running readoptionsfile tests")
    readoptionsfile_json(debug=debug)
    readoptionsfile_txt(debug=debug)
    readoptionsfile_backwards_compat(debug=debug)
    readoptionsfile_missing(debug=debug)

    # readbidstsv
    if debug:
        print("Running readbidstsv tests")
    readbidstsv_basic(debug=debug)
    readbidstsv_neednotexist(debug=debug)
    readbidstsv_colspec(debug=debug)

    # readcolfrombidstsv
    if debug:
        print("Running readcolfrombidstsv tests")
    readcolfrombidstsv_by_name(debug=debug)
    readcolfrombidstsv_by_number(debug=debug)

    # writebidstsv extra
    if debug:
        print("Running writebidstsv extra tests")
    writebidstsv_1d_data(debug=debug)
    writebidstsv_uncompressed(debug=debug)
    writebidstsv_extraheaderinfo(debug=debug)
    writebidstsv_omitjson(debug=debug)

    # savetonifti dtypes
    if debug:
        print("Running savetonifti dtype tests")
    savetonifti_various_dtypes(debug=debug)

    # readfromnifti
    if debug:
        print("Running readfromnifti tests")
    readfromnifti_headeronly(debug=debug)
    readfromnifti_no_extension(debug=debug)
    readfromnifti_missing(debug=debug)

    # colspectolist
    if debug:
        print("Running colspectolist tests")
    colspectolist_simple_range(debug=debug)
    colspectolist_single(debug=debug)
    colspectolist_macro(debug=debug)
    colspectolist_none(debug=debug)

    # parsefilespec
    if debug:
        print("Running parsefilespec tests")
    parsefilespec_no_colspec(debug=debug)
    parsefilespec_with_colspec(debug=debug)


if __name__ == "__main__":
    test_io2(debug=True)
