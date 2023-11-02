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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.miscmath as tide_math
from rapidtide.tests.utils import mse


def test_math(debug=False, displayplots=False):
    # test math functions
    EPSILON = 1e-5
    numpoints = 500
    xaxis = np.linspace(0, 1.0, num=numpoints, endpoint=False)
    the1darray = np.random.rand(numpoints)

    # complex_cepstrum smoke test
    thecepstrum = tide_math.complex_cepstrum(the1darray)

    # real_cepstrum smoke test
    thecepstrum = tide_math.real_cepstrum(the1darray)

    # derivs smoke test
    thederivs = tide_math.thederiv(the1darray)

    # primes test
    theprimefacs = tide_math.primes(24)
    assert theprimefacs == [2, 2, 2, 3]

    # largestfac test
    thelargestfac = tide_math.largestfac(24)
    assert thelargestfac == 3

    # normalize smoke test
    for themethod in ["None", "percent", "variance", "stddev", "z", "p2p", "mad"]:
        thenorm = tide_math.normalize(the1darray, method=themethod)
        thenorm = tide_math.normalize(0.0 * the1darray, method=themethod)

    # corrnormalize smoke test
    for window in ["None", "hamming"]:
        for detrendorder in range(2):
            thenorm = tide_math.corrnormalize(
                the1darray, detrendorder=detrendorder, windowfunc=window
            )
            thenorm = tide_math.corrnormalize(
                0.0 * the1darray, detrendorder=detrendorder, windowfunc=window
            )

    # rms test
    therms = tide_math.rms(np.sin(2.0 * np.pi * xaxis))
    assert np.fabs(therms - np.sqrt(2.0) / 2.0) < EPSILON

    # envdetect test
    hifreq = 100.0
    lowfreq = 3.0
    basefunc = np.sin(hifreq * 2.0 * np.pi * xaxis)
    modfunc = 0.5 + 0.1 * np.sin(lowfreq * 2.0 * np.pi * xaxis)
    theenvelope = tide_math.envdetect(1.0, basefunc * modfunc, cutoff=0.1, padlen=100)
    if displayplots:
        matplotlib.use("TkAgg")
        offset = 0.0
        plt.plot(xaxis, basefunc + offset)
        offset += 1.0
        plt.plot(xaxis, modfunc + offset)
        offset += 1.0
        plt.plot(xaxis, theenvelope + offset)
        plt.show()
        print(mse(theenvelope, modfunc))
    assert mse(theenvelope, modfunc) < 0.04

    # phasemod test
    # tested externally

    # trendfilt test
    # tested externally

    # ComplexPCA test
    the2darray = np.zeros((6, numpoints), dtype=float)


if __name__ == "__main__":
    test_math(debug=True, displayplots=True)
