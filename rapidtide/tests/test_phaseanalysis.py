#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2018-2026 Blaise Frederick
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
from rapidtide.tests.utils import mse


def eval_phaseanalysis(phasestep=0.01, amplitude=1.0, numpoints=100, displayplots=False):
    # read in some data
    """
    Evaluate phaseanalysis.

    Parameters
    ----------
    phasestep : float, optional
        Value supplied as ``phasestep`` by the test case.
    amplitude : float, optional
        Value supplied as ``amplitude`` by the test case.
    numpoints : int, optional
        Value supplied as ``numpoints`` by the test case.
    displayplots : bool, optional
        Display plots from the test.

    Returns
    -------
    tuple
        Values supplied together for the test case.
    """
    phases = np.linspace(0.0, numpoints * phasestep, num=numpoints, endpoint=False)
    testwaveform = amplitude * np.cos(phases)
    if displayplots:
        plt.figure()
        plt.plot(phases)
        plt.show()

        plt.figure()
        plt.plot(testwaveform)
        plt.show()

    # now calculate the phase waveform
    instantaneous_phase, amplitude_envelope, analytic_signal = tide_fit.phaseanalysis(testwaveform)
    filtered_phase = tide_math.trendfilt(instantaneous_phase, order=3, ndevs=2.0)

    if displayplots:
        plt.figure()
        plt.plot(instantaneous_phase)
        plt.plot(filtered_phase)
        plt.plot(phases)
        plt.show()

    return mse(phases, instantaneous_phase), mse(phases, filtered_phase)


def test_phaseanalysis(debug=False, displayplots=False):
    """
    Verify phaseanalysis.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    displayplots : bool, optional
        Display plots from the test.
    """
    msethresh = 1e-3
    instantaneous_mse, filtered_mse = eval_phaseanalysis(
        phasestep=0.1, amplitude=3.0, numpoints=1000, displayplots=displayplots
    )
    print(instantaneous_mse, filtered_mse)
    assert instantaneous_mse < msethresh
    assert filtered_mse < msethresh


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_phaseanalysis(debug=True, displayplots=True)
