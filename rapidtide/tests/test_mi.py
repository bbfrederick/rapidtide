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
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.io as tide_io
from rapidtide.correlate import calc_MI
from rapidtide.tests.utils import get_examples_path


def test_calc_MI(displayplots=False):
    inlen = 1000
    offset = 100
    filename1 = os.path.join(get_examples_path(), "lforegressor.txt")
    filename2 = os.path.join(get_examples_path(), "lforegressor.txt")
    sig1 = tide_io.readvec(filename1)
    sig2 = np.power(sig1, 2.0)
    sig3 = np.power(sig1, 3.0)

    kstart = 3
    kend = 100
    linmivals = []
    sqmivals = []
    cubemivals = []
    for clustersize in range(kstart, kend, 2):
        linmivals.append(calc_MI(sig1, sig1, bins=clustersize) / np.log(clustersize))
        sqmivals.append(calc_MI(sig2, sig1, bins=clustersize) / np.log(clustersize))
        cubemivals.append(calc_MI(sig3, sig1, bins=clustersize) / np.log(clustersize))

    if displayplots:
        plt.figure()
        # plt.ylim([-1.0, 3.0])
        plt.plot(np.array(range(kstart, kend, 2)), np.array(linmivals), "r")
        plt.plot(np.array(range(kstart, kend, 2)), np.array(sqmivals), "g")
        plt.plot(np.array(range(kstart, kend, 2)), np.array(cubemivals), "b")
        # print('maximum occurs at offset', np.argmax(stdcorrelate_result) - midpoint + 1)
        plt.legend(
            ["Mutual information", "Squared mutual information", "Cubed mutual information"]
        )
        plt.show()

    aethresh = 1e-5
    np.testing.assert_almost_equal(1.0, 1.0, aethresh)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_calc_MI(displayplots=True)
