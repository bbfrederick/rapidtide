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
"""Tests for the showxcorr_legacy console script.

The legacy tool is still a registered entry point, and its partial correlation
option (-z) was unusable: mlregress returns (coefficients, R2), and the second
value was being unpacked straight over the timecourse, so the correlation that
followed received a float and died.  These tests drive main() through sys.argv the
way the console script does.
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pytest

import rapidtide.scripts.showxcorr_legacy as legacy

SAMPLERATE = 10.0
NPOINTS = 2000


def _broadband(theseed, npoints=NPOINTS, samplerate=SAMPLERATE):
    """A broadband LFO band signal built from a sum of sinusoids.

    Parameters
    ----------
    theseed : int
        Seed for the frequencies' phases and amplitudes.
    npoints : int, optional
        Number of samples.
    samplerate : float, optional
        Samples per second.

    Returns
    -------
    NDArray
        The generated timecourse.
    """
    therng = np.random.RandomState(theseed)
    thetimes = np.arange(npoints) / samplerate
    theresult = np.zeros(npoints)
    for thefreq, thephase, theamp in zip(
        np.linspace(0.01, 0.2, 30),
        therng.uniform(0, 2 * np.pi, 30),
        therng.uniform(0.5, 1.5, 30),
    ):
        theresult += theamp * np.sin(2 * np.pi * thefreq * thetimes + thephase)
    return theresult


def _runlegacy(thedir, thesignal1, thesignal2, thecontrolvariable=None):
    """Drive showxcorr_legacy.main() through sys.argv and parse its summary line.

    Parameters
    ----------
    thedir : str
        Working directory for the input files.
    thesignal1, thesignal2 : NDArray
        The two input timecourses.
    thecontrolvariable : NDArray or None
        Passed with -z when not None.

    Returns
    -------
    dict
        The summary line, keyed by its column headings.
    """
    thefile1 = os.path.join(thedir, "signal1.txt")
    thefile2 = os.path.join(thedir, "signal2.txt")
    np.savetxt(thefile1, thesignal1)
    np.savetxt(thefile2, thesignal2)

    theargv = ["showxcorr_legacy", thefile1, thefile2, str(SAMPLERATE), "-d", "-A", "-a"]
    if thecontrolvariable is not None:
        thecontrolfile = os.path.join(thedir, "controlvars.txt")
        np.savetxt(thecontrolfile, thecontrolvariable)
        theargv += ["-z", thecontrolfile]

    thebuffer = io.StringIO()
    theoldargv = sys.argv
    try:
        sys.argv = theargv
        with contextlib.redirect_stdout(thebuffer):
            legacy.main()
    finally:
        sys.argv = theoldargv

    thelines = [theline for theline in thebuffer.getvalue().strip().splitlines() if theline]
    theheadings = thelines[-2].split("\t")
    thevalues = [thevalue.strip() for thevalue in thelines[-1].split("\t")]
    # the heading line has to describe exactly the values beneath it; it used to carry
    # an extra xcorr_t that nothing printed, silently shifting every later column
    assert len(theheadings) == len(thevalues), (
        f"{len(theheadings)} column headings for {len(thevalues)} values: "
        f"{theheadings} vs {thevalues}"
    )
    return dict(zip(theheadings, thevalues))


def _makeconfoundedsignals():
    """Two signals with a real shared component that a confound is masking.

    The confound enters with OPPOSITE sign in the two signals, so it drags the raw
    correlation negative and hides the relationship underneath.  Only removing it
    from BOTH timecourses recovers that relationship - with a confound shared in the
    same sense, cleaning either signal alone would be enough to kill the spurious
    correlation, and the test could not tell a complete fix from a half one.

    Returns
    -------
    theconfound, thesignal1, thesignal2 : NDArray
    """
    theshared = _broadband(3)
    theconfound = _broadband(9)
    thesignal1 = theshared + 3.0 * theconfound
    thesignal2 = theshared - 3.0 * theconfound
    return theconfound, thesignal1, thesignal2


def test_a_plain_run_reports_the_confounded_correlation():
    """Without -z the shared confound dominates, which is the situation the option
    exists to address.  Also confirms the fixture is really confounded, so the test
    below is not passing for some unrelated reason."""
    dummy, thesignal1, thesignal2 = _makeconfoundedsignals()
    with tempfile.TemporaryDirectory() as thedir:
        theresult = _runlegacy(thedir, thesignal1, thesignal2)

    # the confound masks the real relationship, so the raw correlation is negative
    assert float(theresult["pearson_r"]) < -0.5


def test_partial_correlation_runs_and_removes_the_control_variable():
    """-z used to raise TypeError before reporting anything, because the R2 return
    value was unpacked over the timecourse and the correlation then got a float.

    Both reported correlations have to collapse: unlike showxcorrx, the legacy tool
    correlates the filtered data, so partialling reaches the cross correlation too.
    """
    theconfound, thesignal1, thesignal2 = _makeconfoundedsignals()
    with tempfile.TemporaryDirectory() as thedir:
        theresult = _runlegacy(thedir, thesignal1, thesignal2, thecontrolvariable=theconfound)

    thepearson = float(theresult["pearson_r"])
    # Both signals cleaned recovers the relationship; cleaning only one lands near 0.3,
    # and subtracting the intercept rather than the slope leaves it negative.
    assert thepearson > 0.9, f"pearson_r was only {thepearson}"

    # xcorr_R is a correlation coefficient after partialling too.  The control
    # variables are removed from the raw trimmed data, so the normalization that
    # follows applies to the residual; removing them afterwards left the residual
    # unnormalized and the reported peak shrank with its amplitude instead.
    thexcorr = float(theresult["xcorr_R"])
    assert thexcorr > 0.9, f"xcorr_R was only {thexcorr}"
    # the masking confound is gone, so the two signals now agree at zero delay
    thedelay = float(theresult["xcorr_maxdelay"])
    assert abs(thedelay) < 0.5, f"partialled delay came out at {thedelay}"


def test_an_unrelated_control_variable_barely_changes_the_answer():
    """Partialling out something the signals do not contain must leave the result
    roughly alone - otherwise a fix that simply destroyed the data would look
    indistinguishable from a working one."""
    dummy, thesignal1, thesignal2 = _makeconfoundedsignals()
    theunrelated = np.random.RandomState(4321).normal(size=NPOINTS)

    with tempfile.TemporaryDirectory() as thedir:
        theplain = _runlegacy(thedir, thesignal1, thesignal2)
    with tempfile.TemporaryDirectory() as thedir:
        thepartial = _runlegacy(thedir, thesignal1, thesignal2, thecontrolvariable=theunrelated)

    thedifference = abs(float(thepartial["pearson_r"]) - float(theplain["pearson_r"]))
    assert thedifference < 0.1, f"an unrelated control moved pearson_r by {thedifference}"


def test_the_timecourses_survive_the_partial_step():
    """The regression must subtract the control variable off, not replace the data
    with the fit statistic.  A single sample would be enough to expose the old
    behaviour, since the array became a scalar."""
    theconfound, thesignal1, thesignal2 = _makeconfoundedsignals()
    with tempfile.TemporaryDirectory() as thedir:
        theresult = _runlegacy(thedir, thesignal1, thesignal2, thecontrolvariable=theconfound)

    # a real correlation came back at all, which it cannot if the arrays were replaced
    assert np.isfinite(float(theresult["pearson_r"]))
    assert np.isfinite(float(theresult["xcorr_maxdelay"]))


if __name__ == "__main__":
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
