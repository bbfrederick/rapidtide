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
"""Tests for the training-data pipeline shared by dlfilter and dlfiltertorch.

getmatchedtcs, readindata and prep turn a directory of BIDS timecourse files into
the arrays the deep learning filters train on.  They are the largest untested part
of both modules, and they are where a silently dropped subject or a misaligned
input/target pair would come from - the kind of error that degrades a trained model
without ever raising.

Both modules carry their own copy of these functions.  The tests are parameterized
over the two so the copies cannot drift apart unnoticed.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

import rapidtide.io as tide_io

THEMODULENAMES = ["rapidtide.dlfilter", "rapidtide.dlfiltertorch"]


def _getmodule(themodulename):
    """Import one of the two filter modules by name.

    Parameters
    ----------
    themodulename : str
        Dotted module name.

    Returns
    -------
    module
        The imported module.
    """
    # dlfilter needs tensorflow and dlfiltertorch needs torch, neither of which is
    # guaranteed to be installed.  importorskip skips cleanly rather than failing
    # collection on an environment that has only one of them, or neither.
    return pytest.importorskip(themodulename)


def _writesubject(thedir, thesubject, thenumpoints=400, withinfo=True, thecorrelation=0.9):
    """Write one subject's timecourse pair, optionally with its _info companion.

    The filter trains a mapping from the fMRI derived cardiac estimate onto the
    measured plethysmogram, so the two columns are built with a controllable
    correlation between them.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thesubject : str
        Subject label, used in the filename.
    thenumpoints : int
        Length of the timecourses.
    withinfo : bool
        Write the _info file that marks the subject as complete.
    thecorrelation : float
        Roughly how strongly the two columns agree.

    Returns
    -------
    str
        Path of the timeseries json that getmatchedtcs will match.
    """
    theroot = os.path.join(thedir, f"sub-{thesubject}_desc-stdrescardfromfmri_timeseries")
    therng = np.random.RandomState(abs(hash(thesubject)) % 10000)
    thetimes = np.arange(thenumpoints) / 25.0
    theshared = np.sin(2.0 * np.pi * 1.1 * thetimes)
    theinput = theshared + (1.0 - thecorrelation) * therng.normal(size=thenumpoints)
    thetarget = theshared + (1.0 - thecorrelation) * therng.normal(size=thenumpoints)

    # A badpts column is written even though these tests do not use bad points.
    # dlfiltertorch.getmatchedtcs asks for colspec
    # "cardiacfromfmri_25.0Hz,normpleth,badpts" unconditionally, while
    # dlfilter.getmatchedtcs asks only for the first two - so a dataset without a
    # badpts column reads fine with the tensorflow filter and comes back as None with
    # the torch one, regardless of the usebadpts setting.
    thebadpts = np.zeros(thenumpoints)
    tide_io.writebidstsv(
        theroot,
        np.vstack((theinput, thetarget, thebadpts)),
        25.0,
        columns=["cardiacfromfmri_25.0Hz", "normpleth", "badpts"],
    )
    if withinfo:
        # readindata reads corrcoeff_raw2pleth out of this file and uses it as the
        # quality gate, so an empty dict is not enough - the subject would be rejected
        # (or raise) before its data was ever looked at
        thecorrcoeff = float(np.corrcoef(theinput, thetarget)[0, 1])
        tide_io.writedicttojson(
            {"corrcoeff_raw2pleth": thecorrcoeff},
            os.path.join(thedir, f"sub-{thesubject}_info.json"),
        )
    return f"{theroot}.json"


def _makedataset(thedir, thenumcomplete=3, thenumincomplete=1, thenumpoints=400):
    """Write a small dataset of complete and incomplete subjects.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thenumcomplete : int
        Subjects with an _info companion.
    thenumincomplete : int
        Subjects without one, which must be skipped.
    thenumpoints : int
        Length of each timecourse.

    Returns
    -------
    str
        The glob pattern getmatchedtcs should be given.
    """
    for thesubject in range(thenumcomplete):
        _writesubject(thedir, f"{thesubject:02d}", thenumpoints, withinfo=True)
    for thesubject in range(thenumincomplete):
        _writesubject(thedir, f"9{thesubject}", thenumpoints, withinfo=False)
    return os.path.join(thedir, "*_desc-stdrescardfromfmri_timeseries.json")


# ==================== getmatchedtcs ====================


@pytest.mark.parametrize("themodulename", THEMODULENAMES)
def test_getmatchedtcs_keeps_only_complete_subjects(themodulename):
    """A subject without its _info file is incomplete and must be dropped.

    Silently including one would feed the trainer a subject whose provenance cannot
    be checked; silently dropping a complete one would shrink the training set
    without saying so.
    """
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        thesearchstring = _makedataset(thedir, thenumcomplete=3, thenumincomplete=2)
        thematched, thetclen = themodule.getmatchedtcs(thesearchstring, debug=True)

    assert len(thematched) == 3, thematched
    # the incomplete ones are the 9x subjects
    assert all("sub-9" not in thename for thename in thematched)
    assert thetclen == 400


@pytest.mark.parametrize("themodulename", THEMODULENAMES)
def test_getmatchedtcs_reports_the_timecourse_length(themodulename):
    """tclen sets the window arithmetic downstream, so it has to come from the data
    rather than a default."""
    themodule = _getmodule(themodulename)
    for thenumpoints in (256, 512):
        with tempfile.TemporaryDirectory() as thedir:
            thesearchstring = _makedataset(
                thedir, thenumcomplete=2, thenumincomplete=0, thenumpoints=thenumpoints
            )
            dummy, thetclen = themodule.getmatchedtcs(thesearchstring)
        assert thetclen == thenumpoints


@pytest.mark.parametrize("themodulename", THEMODULENAMES)
def test_getmatchedtcs_with_no_matches_raises_indexerror(themodulename):
    """A search string that matches nothing indexes an empty list.

    The result is a bare IndexError rather than a message naming the search string
    that found nothing, which is the single most likely thing to go wrong when
    pointing the trainer at a new dataset.  Pinned rather than changed, since giving
    it a clearer error is a behaviour change for any caller catching IndexError.
    """
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        with pytest.raises(IndexError):
            themodule.getmatchedtcs(os.path.join(thedir, "nothing_matches_this*.json"))


# ==================== readindata ====================
#
# These run against dlfilter only.  dlfiltertorch.readindata applies additional
# acceptance criteria that this synthetic fixture does not satisfy - it rejects all
# three subjects and returns None where the tensorflow copy accepts them - so the
# two copies do not agree on what counts as usable training data.  That divergence
# is reported rather than papered over here; extending these tests to the torch copy
# needs a fixture built from its actual criteria.


@pytest.mark.parametrize("themodulename", ["rapidtide.dlfilter"])
def test_readindata_returns_aligned_input_and_target(themodulename):
    """readindata hands back one row per accepted subject, with the input and target
    arrays the same shape.  A mismatch there would train the filter against the wrong
    subject's plethysmogram."""
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        thesearchstring = _makedataset(thedir, thenumcomplete=4, thenumincomplete=0)
        thematched, thetclen = themodule.getmatchedtcs(thesearchstring)
        theresult = themodule.readindata(
            thematched, thetclen, startskip=20, endskip=20, debug=True
        )

    theinput, thetarget, thenames = theresult[0], theresult[1], theresult[2]
    # the arrays are (timepoints, subjects) - one column per accepted subject
    assert theinput.shape == thetarget.shape, (theinput.shape, thetarget.shape)
    assert theinput.shape[1] == len(thenames), (theinput.shape, len(thenames))
    assert theinput.shape[1] > 0, "every subject was rejected"
    assert theinput.shape[0] > 0, "the timecourses were trimmed away entirely"
    assert np.all(np.isfinite(theinput))
    assert np.all(np.isfinite(thetarget))


@pytest.mark.parametrize("themodulename", ["rapidtide.dlfilter"])
def test_readindata_skip_arguments_trim_the_timecourses(themodulename):
    """startskip and endskip drop samples from each end, which is how the transient
    at the start of a run is kept out of training."""
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        thesearchstring = _makedataset(thedir, thenumcomplete=3, thenumincomplete=0)
        thematched, thetclen = themodule.getmatchedtcs(thesearchstring)

        thenarrow = themodule.readindata(thematched, thetclen, startskip=20, endskip=20)[0]
        thewide = themodule.readindata(thematched, thetclen, startskip=60, endskip=40)[0]

    # 40 samples trimmed versus 100, so the wider skip leaves 60 fewer
    assert thewide.shape[0] == thenarrow.shape[0] - 60, (
        thenarrow.shape,
        thewide.shape,
    )


@pytest.mark.parametrize("themodulename", ["rapidtide.dlfilter"])
def test_readindata_readlim_limits_how_many_subjects_are_used(themodulename):
    """readlim caps the number of subjects read, for quick experiments on a subset."""
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        thesearchstring = _makedataset(thedir, thenumcomplete=5, thenumincomplete=0)
        thematched, thetclen = themodule.getmatchedtcs(thesearchstring)

        theall = themodule.readindata(thematched, thetclen, startskip=20, endskip=20)[2]
        thelimited = themodule.readindata(
            thematched, thetclen, startskip=20, endskip=20, readlim=2
        )[2]

    assert len(theall) > len(thelimited), (len(theall), len(thelimited))
    assert len(thelimited) <= 2


@pytest.mark.parametrize("themodulename", ["rapidtide.dlfilter"])
def test_readindata_rejects_a_poorly_correlated_subject(themodulename):
    """A subject whose fMRI estimate does not track its plethysmogram is not usable
    training data, and corrthresh_rp is what excludes it.  Raising the threshold above
    what any subject achieves has to empty the training set rather than quietly
    keeping everything."""
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        thesearchstring = _makedataset(thedir, thenumcomplete=3, thenumincomplete=0)
        thematched, thetclen = themodule.getmatchedtcs(thesearchstring)

        thekept = themodule.readindata(
            thematched, thetclen, startskip=20, endskip=20, corrthresh_rp=0.0
        )[2]
        # a threshold no real pair can reach
        thestrict = themodule.readindata(
            thematched, thetclen, startskip=20, endskip=20, corrthresh_rp=0.999999
        )[2]

    assert len(thekept) > 0
    assert len(thestrict) < len(thekept), (len(thekept), len(thestrict))


@pytest.mark.parametrize("themodulename", ["rapidtide.dlfilter"])
def test_readindata_with_a_zero_endskip_returns_nothing(themodulename):
    """readindata's own default of endskip=0 produces empty arrays.

    The trim is written x1[startskip:-endskip], and with endskip=0 that is
    x1[0:0] - the whole array is sliced away.  Normal use does not hit this because
    the DLFilter class defaults startskip and endskip to 200, but calling readindata
    directly with its documented defaults silently returns nothing at all.  Pinned
    rather than changed: the fix alters what a direct caller gets back.
    """
    themodule = _getmodule(themodulename)
    with tempfile.TemporaryDirectory() as thedir:
        thesearchstring = _makedataset(thedir, thenumcomplete=2, thenumincomplete=0)
        thematched, thetclen = themodule.getmatchedtcs(thesearchstring)
        theinput = themodule.readindata(thematched, thetclen)[0]

    assert theinput.shape[0] == 0, "the zero-endskip slice appears to have been fixed"


if __name__ == "__main__":
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
