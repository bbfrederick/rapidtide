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
"""Tests for rapidtide.workflows.fixtr.

fixtr rewrites the TR in a NIfTI header and copies the data through untouched.  The
whole tool is that one header field, plus the units conversion that has to happen
when the input declares its time axis in milliseconds rather than seconds.
"""

import argparse
import os
import sys
import tempfile

import nibabel as nb
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.workflows.fixtr import _get_parser, fixtr


def _writefmri(thedir, thename, thedata, thetr, theunits="sec"):
    """Write a 4D volume with a known TR and time unit.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thename : str
        Base filename, without extension.
    thedata : NDArray
        The 4D array to write.
    thetr : float
        The TR to record in pixdim[4], in whatever theunits says.
    theunits : str, optional
        Time unit for the header, 'sec' or 'msec'.

    Returns
    -------
    str
        Full path of the file written.
    """
    thefilename = os.path.join(thedir, f"{thename}.nii.gz")
    theimage = nb.Nifti1Image(thedata, np.diag([2.0, 2.0, 2.0, 1.0]))
    theimage.header.set_xyzt_units("mm", theunits)
    theimage.header["pixdim"][4] = thetr
    nb.save(theimage, thefilename)
    return thefilename


def _runfixtr(thedir, theinputfile, theoutputtr, debug=False):
    """Run fixtr and read back what it wrote.

    Parameters
    ----------
    thedir : str
        Directory for the output.
    theinputfile : str
        Path to the input NIfTI.
    theoutputtr : float
        The requested new TR, in seconds.
    debug : bool, optional
        Pass --debug through.

    Returns
    -------
    thedata : NDArray
        The output volume.
    thesizes : NDArray
        The output pixdim array, whose element 4 holds the TR.
    """
    theoutputroot = os.path.join(thedir, "fixed")
    fixtr(
        argparse.Namespace(
            inputfile=theinputfile,
            outputfile=theoutputroot,
            outputtr=theoutputtr,
            debug=debug,
        )
    )
    dummy, thedata, dummy2, dummy3, thesizes = tide_io.readfromnifti(f"{theoutputroot}.nii.gz")
    return thedata, thesizes


def test_get_parser_is_configured():
    """Three positional arguments, and outputtr has to arrive as a float."""
    theparser = _get_parser()
    assert theparser.prog == "fixtr"

    theargs = theparser.parse_args(["in.nii.gz", "outroot", "2.5"])
    assert theargs.inputfile == "in.nii.gz"
    assert theargs.outputfile == "outroot"
    assert theargs.outputtr == 2.5
    assert isinstance(theargs.outputtr, float), "the TR must be parsed as a number"
    assert theargs.debug is False

    assert theparser.parse_args(["in.nii.gz", "outroot", "2.5", "--debug"]).debug is True


def test_a_non_numeric_tr_is_rejected():
    """A TR that is not a number is a caller error caught at parse time."""
    theparser = _get_parser()
    with pytest.raises(SystemExit):
        theparser.parse_args(["in.nii.gz", "outroot", "notanumber"])


def test_the_tr_is_written_into_the_header():
    """The requested TR lands in pixdim[4], which is where every rapidtide tool
    reads it back from."""
    theshape, thenumpoints = (3, 3, 2), 8
    therng = np.random.RandomState(0)
    thedata = therng.normal(size=theshape + (thenumpoints,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writefmri(thedir, "input", thedata, thetr=1.0, theunits="sec")
        dummy, thesizes = _runfixtr(thedir, theinput, 2.5)

        # and fmritimeinfo, which is how the rest of rapidtide asks, agrees
        theoutputtr, thenumtrs = tide_io.fmritimeinfo(os.path.join(thedir, "fixed.nii.gz"))

    assert thesizes[4] == pytest.approx(2.5)
    assert theoutputtr == pytest.approx(2.5)
    assert thenumtrs == thenumpoints


def test_millisecond_units_are_converted():
    """A header declaring msec stores the TR in milliseconds, so the requested value
    in seconds has to be divided by 1000 before it is written.  Writing it unconverted
    would make the run appear a thousand times longer than it is.
    """
    theshape, thenumpoints = (2, 2, 2), 6
    thedata = np.random.RandomState(1).normal(size=theshape + (thenumpoints,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as thedir:
        # the input declares milliseconds, with a TR of 1000 ms
        theinput = _writefmri(thedir, "input", thedata, thetr=1000.0, theunits="msec")
        dummy, thesizes = _runfixtr(thedir, theinput, 2.5)

    # 2.5 seconds recorded in a millisecond header is 0.0025 in pixdim units
    assert thesizes[4] == pytest.approx(0.0025)


def test_second_units_are_not_converted():
    """The complement of the test above: a seconds header takes the value as given.
    Applying the conversion unconditionally would divide every ordinary file by 1000.
    """
    theshape, thenumpoints = (2, 2, 2), 6
    thedata = np.random.RandomState(2).normal(size=theshape + (thenumpoints,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writefmri(thedir, "input", thedata, thetr=1.0, theunits="sec")
        dummy, thesizes = _runfixtr(thedir, theinput, 2.5)

    assert thesizes[4] == pytest.approx(2.5)


def test_the_data_passes_through_untouched():
    """Only the header changes.  fixtr is used to correct a mislabelled TR, so
    altering the voxel values would be silently destructive."""
    theshape, thenumpoints = (4, 3, 2), 10
    therng = np.random.RandomState(3)
    thedata = therng.normal(size=theshape + (thenumpoints,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writefmri(thedir, "input", thedata, thetr=1.0)
        theresult, dummy = _runfixtr(thedir, theinput, 3.0)

    assert theresult.shape == thedata.shape
    np.testing.assert_allclose(theresult, thedata, rtol=1e-6)


def test_the_data_type_is_preserved():
    """The input dtype is read off the header and reapplied, so an integer volume
    does not silently become float."""
    theshape, thenumpoints = (2, 2, 2), 5
    thedata = (np.arange(np.prod(theshape) * thenumpoints) % 100).astype(np.int16)
    thedata = thedata.reshape(theshape + (thenumpoints,))

    with tempfile.TemporaryDirectory() as thedir:
        thefilename = os.path.join(thedir, "input.nii.gz")
        theimage = nb.Nifti1Image(thedata, np.diag([2.0, 2.0, 2.0, 1.0]))
        theimage.header.set_xyzt_units("mm", "sec")
        theimage.header["pixdim"][4] = 1.0
        theimage.header.set_data_dtype(np.int16)
        nb.save(theimage, thefilename)

        theoutputroot = os.path.join(thedir, "fixed")
        fixtr(
            argparse.Namespace(
                inputfile=thefilename, outputfile=theoutputroot, outputtr=2.0, debug=False
            )
        )
        theimage = nb.load(f"{theoutputroot}.nii.gz")
        thedtype = theimage.header.get_data_dtype()

    assert thedtype == np.dtype(np.int16), f"dtype came back as {thedtype}"


def test_debug_reporting_runs():
    """--debug reports the input timing before anything is written."""
    thedata = np.zeros((2, 2, 2, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as thedir:
        theinput = _writefmri(thedir, "input", thedata, thetr=1.5)
        dummy, thesizes = _runfixtr(thedir, theinput, 2.0, debug=True)
    assert thesizes[4] == pytest.approx(2.0)


if __name__ == "__main__":
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
