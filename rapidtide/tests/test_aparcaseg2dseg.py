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
"""Tests for rapidtide.workflows.aparcaseg2dseg.

aparcaseg2dseg collapses a FreeSurfer aparc+aseg, which carries a hundred or so
anatomical labels, down to a three tissue segmentation: 1 for gray, 2 for white, 3
for CSF.  The whole tool is that mapping, so the tests are about which label lands
in which class rather than about any numerics.
"""

import argparse
import os
import tempfile

import nibabel as nb
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.workflows.aparcaseg2dseg import _get_parser, aparcaseg2dseg

GRAYCODE, WHITECODE, CSFCODE = 1, 2, 3


def _writeaparcaseg(thedir, thevolume, thename="aparc+aseg"):
    """Write an integer labelled volume out as a NIfTI aparc+aseg.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thevolume : NDArray
        The 3D volume of FreeSurfer label numbers.
    thename : str, optional
        Base filename, without extension.

    Returns
    -------
    str
        Full path of the file written.
    """
    thefilename = os.path.join(thedir, f"{thename}.nii.gz")
    nb.save(
        nb.Nifti1Image(thevolume.astype(np.float64), np.diag([1.0, 1.0, 1.0, 1.0])),
        thefilename,
    )
    return thefilename


def _runaparcaseg2dseg(thevolume, debug=False):
    """Run the workflow on a volume and hand back the segmentation it produced.

    Parameters
    ----------
    thevolume : NDArray
        The 3D volume of FreeSurfer label numbers.
    debug : bool, optional
        Pass --debug through to the workflow.

    Returns
    -------
    NDArray
        The dseg volume, same shape as the input.
    """
    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writeaparcaseg(thedir, thevolume)
        theoutputroot = os.path.join(thedir, "thedseg")
        aparcaseg2dseg(
            argparse.Namespace(aparcasegname=theinput, dsegname=theoutputroot, debug=debug)
        )
        dummy, theresult, dummy2, dummy3, dummy4 = tide_io.readfromnifti(f"{theoutputroot}.nii.gz")
    return theresult


def test_get_parser_is_configured():
    """The parser takes an input aparc+aseg and an output root, and nothing else is
    required to run the tool."""
    theparser = _get_parser()
    assert theparser.prog == "aparcaseg2dseg"

    with tempfile.TemporaryDirectory() as thedir:
        theinput = os.path.join(thedir, "in.nii.gz")
        with open(theinput, "w") as thefile:
            thefile.write("placeholder")
        theargs = theparser.parse_args([theinput, os.path.join(thedir, "out.nii.gz")])
        assert theargs.aparcasegname == theinput
        assert theargs.debug is False
        theargs = theparser.parse_args([theinput, os.path.join(thedir, "out.nii.gz"), "--debug"])
        assert theargs.debug is True


def test_the_three_tissue_classes_are_disjoint():
    """Gray, white and CSF are encoded as 1, 2 and 3 by summing the masks with weights
    1, 2 and 3.  That only produces clean labels because the three FreeSurfer label
    sets do not overlap - if they ever did, a voxel in two classes would silently come
    out as some other class's number rather than as a conflict."""
    thegray = set(tide_io.colspectolist("APARC_GRAY"))
    thewhite = set(tide_io.colspectolist("APARC_WHITE"))
    thecsf = set(tide_io.colspectolist("APARC_CSF"))

    assert thegray and thewhite and thecsf, "a tissue class has no labels at all"
    assert not (thegray & thewhite), sorted(thegray & thewhite)
    assert not (thegray & thecsf), sorted(thegray & thecsf)
    assert not (thewhite & thecsf), sorted(thewhite & thecsf)


def test_each_tissue_gets_its_own_code():
    """A volume with one slab of each tissue must come back with that slab carrying
    the matching code, and unrecognised labels must come back as zero."""
    theshape = (4, 4, 4)
    thegraylabel = tide_io.colspectolist("APARC_GRAY")[0]
    thewhitelabel = tide_io.colspectolist("APARC_WHITE")[0]
    thecsflabel = tide_io.colspectolist("APARC_CSF")[0]

    thevolume = np.zeros(theshape, dtype=np.float64)
    thevolume[0] = thegraylabel
    thevolume[1] = thewhitelabel
    thevolume[2] = thecsflabel
    # slab 3 keeps label 0, which belongs to no tissue class

    theresult = _runaparcaseg2dseg(thevolume)

    assert theresult.shape == theshape
    assert np.all(theresult[0] == GRAYCODE)
    assert np.all(theresult[1] == WHITECODE)
    assert np.all(theresult[2] == CSFCODE)
    assert np.all(theresult[3] == 0), "an unlabelled voxel was assigned a tissue"
    assert set(np.unique(theresult)) == {0.0, 1.0, 2.0, 3.0}


def test_many_labels_from_one_class_all_collapse_together():
    """The point of the tool is that a hundred anatomical labels become three tissues,
    so several different gray labels must all end up as gray rather than only the
    first one being recognised."""
    thegraylabels = tide_io.colspectolist("APARC_GRAY")[:8]
    thewhitelabels = tide_io.colspectolist("APARC_WHITE")[:8]
    assert len(thegraylabels) == 8 and len(thewhitelabels) == 8

    thevolume = np.zeros((4, 4, 4), dtype=np.float64)
    for theindex, thelabel in enumerate(thegraylabels):
        thevolume[theindex // 4, theindex % 4, 0] = thelabel
    for theindex, thelabel in enumerate(thewhitelabels):
        thevolume[theindex // 4, theindex % 4, 1] = thelabel

    theresult = _runaparcaseg2dseg(thevolume)

    assert np.all(theresult[:2, :, 0] == GRAYCODE), theresult[:2, :, 0]
    assert np.all(theresult[:2, :, 1] == WHITECODE), theresult[:2, :, 1]


def test_an_all_background_volume_produces_an_empty_segmentation():
    """Nothing recognisable in, nothing out - and no crash on the empty masks."""
    theresult = _runaparcaseg2dseg(np.zeros((3, 3, 3), dtype=np.float64))
    assert np.all(theresult == 0)


def test_debug_reporting_runs():
    """--debug only adds printing, but it runs inside the workflow and has to work."""
    thevolume = np.zeros((3, 3, 3), dtype=np.float64)
    thevolume[0] = tide_io.colspectolist("APARC_GRAY")[0]
    theresult = _runaparcaseg2dseg(thevolume, debug=True)
    assert np.all(theresult[0] == GRAYCODE)


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
