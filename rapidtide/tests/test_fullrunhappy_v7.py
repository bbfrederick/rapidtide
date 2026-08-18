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
"""Exercise reading a respiration waveform from a file.

This was the second test in v6 and moved here so the two can land on different CI
containers.  The slow shard splits with ``circleci tests split --split-by=timings``,
which balances whole *files*, so two tests in one file are always one indivisible lump
however long they take - v6 was the slowest file in the shard while holding both.

Splitting the remaining v6 run any further would be counterproductive: its two options
are one happy run between them, and separating them would buy a second base run - about
38 s on this dataset - to save around 13 s of option overhead.
"""

import os

import matplotlib as mpl
import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.tests.utils import get_example_and_temp_roots, run_happy

pytestmark = pytest.mark.slow


def _makerespirationfile(
    thepath: str,
    numpoints: int = 5000,
    samplerate: float = 25.0,
    seed: int = 20260815,
) -> str:
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


def test_fullrunhappy_v7(
    debug: bool = False, local: bool = False, displayplots: bool = False
) -> None:
    """Run happy with a respiration waveform supplied from a file.

    Parameters
    ----------
    debug : bool, optional
        Unused here, kept for the signature the other full run tests share.
    local : bool, optional
        Resolve the example and temp roots relative to the source tree, for running this
        file directly rather than under pytest.
    displayplots : bool, optional
        Unused here, kept for the signature the other full run tests share.

    Returns
    -------
    None
    """
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    respirationfile = _makerespirationfile(
        os.path.join(testtemproot, "happyout7_respiration"),
    )
    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.json"),
        os.path.join(testtemproot, "happyout7"),
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
    test_fullrunhappy_v7(debug=True, local=True, displayplots=True)
