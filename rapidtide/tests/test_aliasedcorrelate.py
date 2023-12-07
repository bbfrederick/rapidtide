#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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

import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
from rapidtide.correlate import AliasedCorrelator


def aliasedcorrelate(
    hiressignal,
    hires_Fs,
    lowressignal,
    lowres_Fs,
    timerange,
    hiresstarttime=0.0,
    lowresstarttime=0.0,
    padtime=30.0,
):
    """Perform an aliased correlation.

    This function is deprecated, and is retained here as a reference against
    which to test AliasedCorrelator.

    Parameters
    ----------
    hiressignal: 1D array
        The unaliased waveform to match
    hires_Fs: float
        The sample rate of the unaliased waveform
    lowressignal: 1D array
        The aliased waveform to match
    lowres_Fs: float
        The sample rate of the aliased waveform
    timerange: 1D array
        The delays for which to calculate the correlation function

    Returns
    -------
    corrfunc: 1D array
        The correlation function evaluated at timepoints of timerange
    """
    highresaxis = np.arange(0.0, len(hiressignal)) * (1.0 / hires_Fs) - hiresstarttime
    lowresaxis = np.arange(0.0, len(lowressignal)) * (1.0 / lowres_Fs) - lowresstarttime
    tcgenerator = tide_resample.FastResampler(highresaxis, hiressignal, padtime=padtime)
    targetsignal = tide_math.corrnormalize(lowressignal)
    corrfunc = timerange * 0.0
    for i in range(len(timerange)):
        aliasedhiressignal = tide_math.corrnormalize(tcgenerator.yfromx(lowresaxis + timerange[i]))
        corrfunc[i] = np.dot(aliasedhiressignal, targetsignal)
    return corrfunc


def test_aliasedcorrelate(displayplots=False):
    Fs_hi = 10.0
    Fs_lo = 1.0
    numsteps = 10
    siginfo = [[1.0, 1.36129345], [0.33, 2.0]]
    modamp = 0.01
    inlenhi = 1000
    inlenlo = 100
    offset = 0.5
    width = 2.5
    rangepts = 101
    timerange = np.linspace(0.0, width, num=101) - width / 2.0
    hiaxis = np.linspace(0.0, 2.0 * np.pi * inlenhi / Fs_hi, num=inlenhi, endpoint=False)
    loaxis = np.linspace(0.0, 2.0 * np.pi * inlenlo / Fs_lo, num=inlenlo, endpoint=False)
    sighi = hiaxis * 0.0
    siglo = loaxis * 0.0
    for theinfo in siginfo:
        sighi += theinfo[0] * np.sin(theinfo[1] * hiaxis)
        siglo += theinfo[0] * np.sin(theinfo[1] * loaxis)
    aliasedcorrelate_result = aliasedcorrelate(
        sighi, Fs_hi, siglo, Fs_lo, timerange, padtime=width
    )

    theAliasedCorrelator = AliasedCorrelator(sighi, Fs_hi, numsteps)
    aliasedcorrelate_result2 = theAliasedCorrelator.apply(siglo, 0)

    if displayplots:
        plt.figure()
        # plt.ylim([-1.0, 3.0])
        plt.plot(hiaxis, sighi, "k")
        plt.scatter(loaxis, siglo, c="r")
        plt.legend(["sighi", "siglo"])

        plt.figure()
        plt.plot(timerange, aliasedcorrelate_result, "k")
        plt.plot(timerange, aliasedcorrelate_result2, "r")
        print("maximum occurs at offset", timerange[np.argmax(aliasedcorrelate_result)])

        plt.show()

    # assert (fastcorrelate_result == stdcorrelate_result).all
    aethresh = 10
    # np.testing.assert_almost_equal(fastcorrelate_result, stdcorrelate_result, aethresh)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_aliasedcorrelate(displayplots=True)
