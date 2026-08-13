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
"""Tests for rapidtide.workflows.slopefit.

slopefit fits a voxelwise polynomial relating the timecourses in two matched 4D
files.  The tests build data from a known polynomial and require the exact
coefficients back, which is the only check that distinguishes a working fit from
one that runs to completion and reports nonsense - the state this tool was in.
"""

import argparse
import os
import tempfile

import nibabel as nb
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.workflows.slopefit import _get_parser, main, slopefit

THEAFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


def _writenifti(thedir, thename, thedata):
    """Write an array out as a NIfTI file and return its path.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thename : str
        Base filename, without extension.
    thedata : NDArray
        The array to write.

    Returns
    -------
    str
        Full path of the file written.
    """
    thefilename = os.path.join(thedir, f"{thename}.nii.gz")
    nb.save(nb.Nifti1Image(np.asarray(thedata, dtype=np.float64), THEAFFINE), thefilename)
    return thefilename


def _runslopefit(thedir, thex, they, order=1, themask=None, debug=False):
    """Run slopefit on two arrays and read back the coefficients and R squared.

    Parameters
    ----------
    thedir : str
        Working directory.
    thex, they : NDArray
        The 4D explanatory and dependent volumes.
    order : int, optional
        Polynomial order.
    themask : NDArray or None, optional
        Optional 3D mask.
    debug : bool, optional
        Pass debug through.

    Returns
    -------
    thecoffs : NDArray
        Shape (x, y, z, order + 1).
    ther2 : NDArray
        Shape (x, y, z).
    """
    thexfile = _writenifti(thedir, "thex", thex)
    theyfile = _writenifti(thedir, "they", they)
    themaskfile = None if themask is None else _writenifti(thedir, "themask", themask)
    theroot = os.path.join(thedir, f"out{order}")

    slopefit(thexfile, theyfile, theroot, maskfile=themaskfile, order=order, debug=debug)

    dummy, thecoffs, dummy2, dummy3, dummy4 = tide_io.readfromnifti(f"{theroot}_coffs.nii.gz")
    dummy5, ther2, dummy6, dummy7, dummy8 = tide_io.readfromnifti(f"{theroot}_r2vals.nii.gz")
    return thecoffs, ther2


def test_get_parser_defaults():
    """order defaults to 1 (a plain slope) and the mask is optional."""
    theparser = _get_parser()
    assert theparser.prog == "slopefit"

    with tempfile.TemporaryDirectory() as thedir:
        thefiles = []
        for thename in ("a", "b"):
            thepath = os.path.join(thedir, f"{thename}.nii.gz")
            with open(thepath, "w") as thefile:
                thefile.write("placeholder")
            thefiles.append(thepath)
        theargs = theparser.parse_args(thefiles + ["outroot"])

    assert theargs.order == 1
    assert theargs.maskfile is None
    assert theargs.debug is False
    assert theargs.outputroot == "outroot"


def test_a_linear_relationship_is_recovered_exactly():
    """y = 2x + 3 must come back as an intercept of 3 and a slope of 2.

    This is the assertion the tool never had.  It used to crash outright on
    range(np.where(...)), and once that was fixed it still returned zero for every
    slope, because an explicit column of ones was being passed alongside mlregress's
    own intercept and pushed the real coefficient off the end of the output array.
    """
    theshape, thenumpoints = (4, 4, 3), 30
    therng = np.random.RandomState(0)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = 2.0 * thex + 3.0

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, ther2 = _runslopefit(thedir, thex, they, order=1)

    assert thecoffs.shape == theshape + (2,)
    np.testing.assert_allclose(thecoffs[..., 0], 3.0, atol=1e-6)  # intercept
    np.testing.assert_allclose(thecoffs[..., 1], 2.0, atol=1e-6)  # slope
    np.testing.assert_allclose(ther2, 1.0, atol=1e-9)


def test_a_quadratic_relationship_is_recovered_exactly():
    """Order 2 must give three coefficients in ascending power order."""
    theshape, thenumpoints = (3, 3, 2), 40
    therng = np.random.RandomState(1)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = 1.0 + 2.0 * thex + 3.0 * thex**2

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, ther2 = _runslopefit(thedir, thex, they, order=2)

    assert thecoffs.shape == theshape + (3,)
    np.testing.assert_allclose(thecoffs[..., 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(thecoffs[..., 1], 2.0, atol=1e-6)
    np.testing.assert_allclose(thecoffs[..., 2], 3.0, atol=1e-6)
    np.testing.assert_allclose(ther2, 1.0, atol=1e-9)


def test_each_voxel_is_fit_independently():
    """The fit is voxelwise, so voxels with different relationships must come back
    with different coefficients rather than one global answer."""
    theshape, thenumpoints = (2, 1, 1), 40
    therng = np.random.RandomState(2)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = np.zeros_like(thex)
    they[0, 0, 0, :] = 5.0 * thex[0, 0, 0, :] - 1.0
    they[1, 0, 0, :] = -2.0 * thex[1, 0, 0, :] + 7.0

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, ther2 = _runslopefit(thedir, thex, they, order=1)

    np.testing.assert_allclose(thecoffs[0, 0, 0], [-1.0, 5.0], atol=1e-6)
    np.testing.assert_allclose(thecoffs[1, 0, 0], [7.0, -2.0], atol=1e-6)


def test_an_imperfect_fit_reports_an_r2_below_one():
    """R squared has to be a real goodness of fit, not a constant.  Noise that the
    polynomial cannot explain must drive it below 1."""
    theshape, thenumpoints = (2, 2, 2), 60
    therng = np.random.RandomState(3)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = 2.0 * thex + therng.normal(scale=2.0, size=thex.shape)

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, ther2 = _runslopefit(thedir, thex, they, order=1)

    assert np.all(ther2 < 0.95), f"noisy data still reported R2 of {ther2.max()}"
    assert np.all(ther2 >= 0.0)


def test_a_mask_restricts_which_voxels_are_fit():
    """Voxels outside the mask are never fit, so their coefficients stay zero."""
    theshape, thenumpoints = (4, 4, 2), 30
    therng = np.random.RandomState(4)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = 2.0 * thex + 3.0

    themask = np.zeros(theshape)
    themask[:2] = 1.0

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, ther2 = _runslopefit(thedir, thex, they, order=1, themask=themask)

    theinside = themask > 0
    np.testing.assert_allclose(thecoffs[theinside][:, 1], 2.0, atol=1e-6)
    assert np.all(thecoffs[~theinside] == 0.0), "a voxel outside the mask was fit"
    assert np.all(ther2[~theinside] == 0.0)
    assert np.all(ther2[theinside] > 0.99)


def test_a_mask_threshold_of_point_nine_is_applied():
    """The mask is binarised at 0.9, not at zero, so a soft mask value below that is
    excluded even though it is nonzero."""
    theshape, thenumpoints = (3, 1, 1), 30
    therng = np.random.RandomState(5)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = 2.0 * thex + 3.0
    themask = np.array([[[1.0]], [[0.5]], [[0.0]]])

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, dummy = _runslopefit(thedir, thex, they, order=1, themask=themask)

    assert thecoffs[0, 0, 0, 1] != 0.0, "the fully in mask voxel was not fit"
    assert np.all(thecoffs[1, 0, 0] == 0.0), "a mask value of 0.5 was treated as in mask"
    assert np.all(thecoffs[2, 0, 0] == 0.0)


def test_order_below_one_is_rejected():
    """A zeroth order fit is just the mean and is not what this tool is for."""
    theshape, thenumpoints = (2, 2, 1), 10
    thex = np.zeros(theshape + (thenumpoints,))

    with tempfile.TemporaryDirectory() as thedir:
        thexfile = _writenifti(thedir, "thex", thex)
        with pytest.raises(SystemExit):
            slopefit(thexfile, thexfile, os.path.join(thedir, "out"), order=0)


def test_mismatched_inputs_are_rejected():
    """The two inputs are compared voxel by voxel and timepoint by timepoint, so a
    disagreement in either is fatal rather than silently broadcast."""
    therng = np.random.RandomState(6)
    thex = therng.normal(size=(4, 4, 2, 20))

    with tempfile.TemporaryDirectory() as thedir:
        thexfile = _writenifti(thedir, "thex", thex)
        theroot = os.path.join(thedir, "out")

        # different spatial dimensions
        theotherspace = _writenifti(thedir, "otherspace", therng.normal(size=(5, 4, 2, 20)))
        with pytest.raises(SystemExit):
            slopefit(thexfile, theotherspace, theroot)

        # same space, different number of timepoints
        theothertime = _writenifti(thedir, "othertime", therng.normal(size=(4, 4, 2, 25)))
        with pytest.raises(SystemExit):
            slopefit(thexfile, theothertime, theroot)

        # a mask that does not match the images
        thebadmask = _writenifti(thedir, "badmask", np.ones((5, 4, 2)))
        with pytest.raises(SystemExit):
            slopefit(thexfile, thexfile, theroot, maskfile=thebadmask)

        # a mask with a time dimension is not a mask
        thefourdmask = _writenifti(thedir, "fourdmask", np.ones((4, 4, 2, 3)))
        with pytest.raises(SystemExit):
            slopefit(thexfile, thexfile, theroot, maskfile=thefourdmask)


def test_debug_reporting_runs():
    """--debug prints shapes and per voxel fits from inside the voxel loop."""
    theshape, thenumpoints = (1, 1, 1), 20
    therng = np.random.RandomState(7)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = 2.0 * thex + 3.0

    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, dummy = _runslopefit(thedir, thex, they, order=1, debug=True)
    np.testing.assert_allclose(thecoffs[0, 0, 0], [3.0, 2.0], atol=1e-6)

    # the mask branch of the debug reporting only runs when there is a mask
    themask = np.ones(theshape)
    with tempfile.TemporaryDirectory() as thedir:
        thecoffs, dummy = _runslopefit(thedir, thex, they, order=1, themask=themask, debug=True)
    np.testing.assert_allclose(thecoffs[0, 0, 0], [3.0, 2.0], atol=1e-6)


def test_main_forwards_its_arguments():
    """main is the entry point the console script calls, and it has to hand every
    argument through to slopefit.

    Deliberately run at order 2 rather than the default: at order 1 a main that
    ignored --order entirely would produce exactly the same answer, so the test would
    pass while proving nothing.
    """
    theshape, thenumpoints = (2, 2, 1), 30
    therng = np.random.RandomState(8)
    thex = therng.normal(size=theshape + (thenumpoints,))
    they = -2.0 + 4.0 * thex + 6.0 * thex**2

    with tempfile.TemporaryDirectory() as thedir:
        thexfile = _writenifti(thedir, "thex", thex)
        theyfile = _writenifti(thedir, "they", they)
        theroot = os.path.join(thedir, "mainout")
        main(
            argparse.Namespace(
                inputfile1=thexfile,
                inputfile2=theyfile,
                outputroot=theroot,
                maskfile=None,
                order=2,
                debug=False,
            )
        )
        dummy, thecoffs, dummy2, dummy3, dummy4 = tide_io.readfromnifti(f"{theroot}_coffs.nii.gz")

    # three coefficients, which only happens if --order actually reached slopefit
    assert thecoffs.shape == theshape + (3,), thecoffs.shape
    np.testing.assert_allclose(thecoffs[..., 0], -2.0, atol=1e-6)
    np.testing.assert_allclose(thecoffs[..., 1], 4.0, atol=1e-6)
    np.testing.assert_allclose(thecoffs[..., 2], 6.0, atol=1e-6)


if __name__ == "__main__":
    import sys

    # Put the repo root ahead of anything else and use importlib import mode.  Running
    # this file directly otherwise puts rapidtide/tests/ on sys.path instead of the repo
    # root, so `import rapidtide` finds an installed copy and pytest then trips over two
    # conftest.py files with the same module name.  Same fix as test_util.py.
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
