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
"""Tests for rapidtide.workflows.randomatlas.

randomatlas is a thin CLI wrapper over regionops.partition_3d - it reads a mask,
optionally reads a tensor field, partitions, and writes the labels out under a name
encoding the region count and seed.  The partitioning itself is covered in
test_regionops.py, so these tests are about the wiring: what gets passed through,
what the output is called, and what happens when the inputs disagree.
"""

import argparse
import os
import tempfile

import nibabel as nb
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.workflows.randomatlas import _get_parser, randomatlas

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
    nb.save(nb.Nifti1Image(thedata.astype(np.float64), THEAFFINE), thefilename)
    return thefilename


def _makeargs(inputfilename, outputfileroot, numregions=4, **theoverrides):
    """Build the Namespace randomatlas expects, with the parser's own defaults.

    Parameters
    ----------
    inputfilename : str
        Path to the mask.
    outputfileroot : str
        Root for the output name.
    numregions : int, optional
        Number of regions to partition into.
    **theoverrides : Any
        Any argument to override.

    Returns
    -------
    argparse.Namespace
        Arguments ready to hand to randomatlas.
    """
    theargs = argparse.Namespace(
        inputfilename=inputfilename,
        outputfileroot=outputfileroot,
        numregions=numregions,
        seed=1234,
        alpha=0.5,
        anisotropyfile=None,
        anisotropystrength=0.0,
    )
    for thename, thevalue in theoverrides.items():
        setattr(theargs, thename, thevalue)
    return theargs


def test_get_parser_defaults():
    """The three positional arguments are required, and the tuning knobs default to
    the module constants the help text quotes."""
    import rapidtide.workflows.randomatlas as therandomatlas

    theparser = _get_parser()
    assert theparser.prog == "randomatlas"

    theargs = theparser.parse_args(["in.nii.gz", "outroot", "12"])
    assert theargs.inputfilename == "in.nii.gz"
    assert theargs.outputfileroot == "outroot"
    assert theargs.numregions == 12
    assert isinstance(theargs.numregions, int), "the region count must be parsed as an int"
    assert theargs.seed == therandomatlas.DEFAULT_RNGSEED
    assert theargs.alpha == therandomatlas.DEFAULT_ALPHA
    assert theargs.anisotropyfile is None
    assert theargs.anisotropystrength == therandomatlas.DEFAULT_ANISOTROPY_STRENGTH


def test_get_parser_rejects_a_negative_alpha():
    """alpha is a balance weight, and negative values are rejected at parse time
    rather than producing a partition that actively unbalances itself."""
    theparser = _get_parser()
    with pytest.raises(SystemExit):
        theparser.parse_args(["in.nii.gz", "outroot", "4", "--alpha", "-1.0"])
    with pytest.raises(SystemExit):
        theparser.parse_args(["in.nii.gz", "outroot", "4", "--anisotropystrength", "-2.0"])


def test_output_name_encodes_the_region_count_and_seed():
    """The output filename is how a sweep over seeds and region counts is kept
    straight, so the zero padded encoding is part of the interface."""
    themask = np.ones((8, 8, 6))
    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writenifti(thedir, "mask", themask)
        theroot = os.path.join(thedir, "atlas")
        randomatlas(_makeargs(theinput, theroot, numregions=7, seed=42))

        thewritten = sorted(
            thename for thename in os.listdir(thedir) if thename.startswith("atlas")
        )
        assert thewritten == ["atlas_r007_s0042.nii.gz"], thewritten


def test_labels_cover_the_mask_and_nothing_else():
    """Voxels inside the mask get a region; voxels outside keep partition_3d's -1."""
    themask = np.zeros((9, 9, 7))
    themask[1:8, 1:8, 1:6] = 1
    thenumregions = 5

    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writenifti(thedir, "mask", themask)
        theroot = os.path.join(thedir, "atlas")
        randomatlas(_makeargs(theinput, theroot, numregions=thenumregions, seed=7))
        dummy, thelabels, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theroot}_r{thenumregions:03d}_s0007.nii.gz"
        )

    theinside = themask > 0
    assert thelabels.shape == themask.shape
    assert np.all(thelabels[theinside] >= 0), "an in mask voxel was left unassigned"
    assert np.all(thelabels[~theinside] == -1), "labels leaked outside the mask"
    assert len(np.unique(thelabels[theinside])) == thenumregions


def test_the_seed_is_passed_through():
    """Two runs with the same seed must agree and two with different seeds must not,
    or the seed argument is not reaching the partitioner."""
    themask = np.ones((7, 7, 5))
    theresults = {}
    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writenifti(thedir, "mask", themask)
        for theseed in (11, 11, 22):
            theroot = os.path.join(thedir, f"atlas{theseed}")
            randomatlas(_makeargs(theinput, theroot, numregions=4, seed=theseed))
            dummy, thelabels, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
                f"{theroot}_r004_s{theseed:04d}.nii.gz"
            )
            theresults.setdefault(theseed, []).append(thelabels)

    np.testing.assert_array_equal(theresults[11][0], theresults[11][1])
    assert not np.array_equal(theresults[11][0], theresults[22][0]), "the seed had no effect"


def test_an_anisotropy_field_is_read_and_applied():
    """The tensor field is optional, but when given it has to reach the partitioner.
    A field strongly favouring x should give regions longer in x than an isotropic
    run does."""
    theshape = (14, 14, 5)
    themask = np.ones(theshape)
    thetensor = np.zeros(theshape + (6,))
    thetensor[..., 0] = 25.0  # xx
    thetensor[..., 1] = 1.0  # yy
    thetensor[..., 2] = 1.0  # zz

    theratios = {}
    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writenifti(thedir, "mask", themask)
        thetensorfile = _writenifti(thedir, "tensor", thetensor)
        for thelabel, thestrength in (("iso", 0.0), ("aniso", 12.0)):
            theextents = []
            for theseed in (1, 2, 3, 4):
                theroot = os.path.join(thedir, f"atlas{thelabel}{theseed}")
                randomatlas(
                    _makeargs(
                        theinput,
                        theroot,
                        numregions=4,
                        seed=theseed,
                        anisotropyfile=thetensorfile,
                        anisotropystrength=thestrength,
                    )
                )
                dummy, thelabels, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
                    f"{theroot}_r004_s{theseed:04d}.nii.gz"
                )
                for theregion in range(4):
                    thecoords = np.argwhere(thelabels == theregion)
                    if len(thecoords) < 10:
                        continue
                    theextents.append(
                        (np.ptp(thecoords[:, 0]) + 1) / (np.ptp(thecoords[:, 1]) + 1)
                    )
            theratios[thelabel] = float(np.mean(theextents))

    assert theratios["aniso"] > theratios["iso"], (
        f"the tensor field did not bias growth: x/y extent {theratios['iso']:.2f} "
        f"isotropic versus {theratios['aniso']:.2f} anisotropic"
    )


def test_a_mismatched_anisotropy_field_is_rejected():
    """A tensor field covering a different volume than the mask is a user error, and
    has to be reported rather than silently misaligned against the mask."""
    themask = np.ones((6, 6, 4))
    thetensor = np.zeros((5, 6, 4, 6))

    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writenifti(thedir, "mask", themask)
        thetensorfile = _writenifti(thedir, "tensor", thetensor)
        theargs = _makeargs(
            theinput,
            os.path.join(thedir, "atlas"),
            numregions=3,
            anisotropyfile=thetensorfile,
        )
        with pytest.raises(ValueError, match="spatial dimensions must match"):
            randomatlas(theargs)


def test_more_regions_than_voxels_is_rejected():
    """partition_3d cannot make more regions than there are voxels, and the error has
    to travel out of the workflow rather than being swallowed."""
    themask = np.zeros((4, 4, 4))
    themask[0, 0, 0] = 1
    themask[0, 0, 1] = 1

    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writenifti(thedir, "mask", themask)
        theargs = _makeargs(theinput, os.path.join(thedir, "atlas"), numregions=50)
        with pytest.raises(ValueError, match="fewer voxels than n_regions"):
            randomatlas(theargs)


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
