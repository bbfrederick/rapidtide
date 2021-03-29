#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io


def test_proberegressor(debug=False, display=False):
    fmritr = 1.5
    offsettime = 0.0
    inputstarttime = 0.0
    oversampfactor = 2
    invertregressor = False
    detrendorder = 3
    preprocskip = 0

    filename = "testdata/lforegressor.txt"
    inputfreq = 1.0 / fmritr
    inputstarttime = 0.0
    inputperiod = 1.0 / inputfreq
    inputvec = tide_io.readvec(filename)
    numreference = len(inputvec)
    validstart = 0
    validend = numreference
    validtimepoints = validend - validstart + 1

    skiptime = fmritr * (preprocskip)
    reference_x = np.arange(0.0, numreference) * inputperiod - (inputstarttime + offsettime)
    initial_fmri_x = np.arange(0.0, validtimepoints) * fmritr + skiptime
    os_fmri_x = (
        np.arange(0.0, (validtimepoints) * oversampfactor - (oversampfactor - 1))
        * (fmritr / oversampfactor)
        + skiptime
    )

    # invert the regressor if necessary
    if invertregressor:
        invertfac = -1.0
    else:
        invertfac = 1.0

    # detrend the regressor if necessary
    if detrendorder > 0:
        reference_y = invertfac * tide_fit.detrend(
            inputvec[0:numreference], order=detrendorder, demean=True
        )
    else:
        reference_y = invertfac * (inputvec[0:numreference] - np.mean(inputvec[0:numreference]))

    if display:
        plt.figure()
        plt.plot(reference_x, reference_y)
        plt.show()

    assert True


if __name__ == "__main__":
    test_proberegressor(debug=True, display=True)
