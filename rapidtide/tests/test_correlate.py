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

from rapidtide.correlate import fastcorrelate


def test_fastcorrelate(displayplots=False):
    inlen = 1000
    offset = 100
    sig1 = np.zeros((inlen), dtype="float")
    sig2 = np.zeros((inlen), dtype="float")
    sig1[int(inlen // 2) + 1] = 1.0
    sig2[int(inlen // 2) + offset + 1] = 1.0
    fastcorrelate_result_pad0 = fastcorrelate(sig2, sig1, zeropadding=0)
    fastcorrelate_result_padneg1 = fastcorrelate(sig2, sig1, zeropadding=-1)
    fastcorrelate_result_pad1000 = fastcorrelate(sig2, sig1, zeropadding=1000)
    print(
        "lengths:",
        len(fastcorrelate_result_pad0),
        len(fastcorrelate_result_padneg1),
        len(fastcorrelate_result_pad1000),
    )
    stdcorrelate_result = np.correlate(sig2, sig1, mode="full")
    midpoint = int(len(stdcorrelate_result) // 2) + 1
    if displayplots:
        plt.figure()
        plt.ylim([-1.0, 4.0])
        plt.plot(fastcorrelate_result_pad1000 + 3.0)
        plt.plot(fastcorrelate_result_padneg1 + 2.0)
        plt.plot(fastcorrelate_result_pad0 + 1.0)
        plt.plot(stdcorrelate_result)
        print(stdcorrelate_result)
        print("maximum occurs at offset", np.argmax(stdcorrelate_result) - midpoint + 1)
        plt.legend(
            [
                "Fast correlate pad 1000",
                "Fast correlate pad -1",
                "Fast correlate nopad",
                "Standard correlate",
            ]
        )
        plt.show()

    aethresh = 10
    np.testing.assert_almost_equal(fastcorrelate_result_pad0, stdcorrelate_result, aethresh)

    # smoke test the weighted correlations
    for weighting in ["None", "liang", "eckart", "phat"]:
        weighted_result = fastcorrelate(sig2, sig1, weighting=weighting)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fastcorrelate(displayplots=True)
