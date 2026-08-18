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
"""Exercise the happy options that the other full run tests leave untouched.

v1 through v5 between them cover the regression types, the cardiac and motion file
paths, the deep learning filter and the aliased correlation.  This one covers the
remaining option gated feature blocks: global mean signal filtering, Wright's vessel
mask, and reading a respiration waveform from a file.

Deliberately not covered: ``--estimateflow``.  Measured against the full size dataset
it added about 18 minutes to a roughly 2 minute run, in exchange for around a dozen
lines - by far the worst ratio of any option here.  Both halves of that ratio shrink
with the cropped dataset this test now uses, but not the ratio itself, so the
conclusion is unchanged: if the optical flow code needs test coverage it wants a small
synthetic fixture aimed at ``calc_3d_optical_flow`` directly, not a full happy run.
"""

import os

import matplotlib as mpl
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.tests.utils import get_example_and_temp_roots, run_happy

pytestmark = pytest.mark.slow


def _makerespirationfile(thepath, numpoints=5000, samplerate=25.0, seed=20260815):
    """Write a synthetic respiration waveform for the respirationfile option.

    The waveform has to span the whole acquisition - happy checks that the physiological
    recording covers the slice time axis and refuses it otherwise - so the default length
    of 5000 samples at 25 Hz (200 s) comfortably covers the ~128 s test dataset.

    Parameters
    ----------
    thepath : str
        File root to write to (the BIDS writer appends the extension).
    numpoints : int, optional
        Number of samples.  Default is 5000.
    samplerate : float, optional
        Sample rate in Hz.  Default is 25.0.
    seed : int, optional
        Random seed for the additive noise.  Default is 20260815.

    Returns
    -------
    str
        The path to the written file.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(numpoints, dtype=float) / samplerate
    # a slow, roughly periodic breathing waveform
    thewaveform = np.sin(2.0 * np.pi * 0.25 * t) + 0.1 * rng.randn(numpoints)
    tide_io.writebidstsv(
        thepath,
        thewaveform,
        samplerate,
        starttime=0.0,
        columns=["respiration"],
    )
    thefile = thepath + ".tsv.gz"
    if not os.path.exists(thefile):
        thefile = thepath + ".tsv"
    return thefile


def test_fullrunhappy_v6(debug=False, local=False, displayplots=False):
    """Run happy with global mean signal filtering and Wright's vessel mask."""
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.json"),
        os.path.join(testtemproot, "happyout6"),
        "--mklthreads",
        "-1",
        "--gmsfilt",
        "--wrightiterations",
        "2",
    ]
    run_happy(inputargs)


def test_fullrunhappy_v6_respirationfile(debug=False, local=False, displayplots=False):
    """Run happy with a respiration waveform supplied from a file."""
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    respirationfile = _makerespirationfile(
        os.path.join(testtemproot, "happyout6_respiration"),
    )
    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.json"),
        os.path.join(testtemproot, "happyout6resp"),
        "--mklthreads",
        "-1",
        "--projmask",
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL_smallmask.nii.gz"),
        "--respirationfile",
        respirationfile,
    ]
    run_happy(inputargs)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunhappy_v6(debug=True, local=True, displayplots=True)
    test_fullrunhappy_v6_respirationfile(debug=True, local=True, displayplots=True)
