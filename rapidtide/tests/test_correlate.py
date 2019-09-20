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
from __future__ import print_function, division

import numpy as np
import pylab as plt

from rapidtide.correlate import fastcorrelate


def test_fastcorrelate(display=False):
    inlen = 1000
    offset = 100
    sig1 = np.zeros((inlen), dtype='float')
    sig2 = np.zeros((inlen), dtype='float')
    sig1[int(inlen // 2) + 1] = 1.0
    sig2[int(inlen // 2) + offset + 1] = 1.0
    fastcorrelate_result = fastcorrelate(sig2, sig1)
    stdcorrelate_result = np.correlate(sig2, sig1, mode='full')
    midpoint = int(len(stdcorrelate_result) // 2) + 1
    if display:
        plt.figure()
        plt.ylim([-1.0, 3.0])
        plt.plot(fastcorrelate_result + 1.0)
        plt.plot(stdcorrelate_result)
        print('maximum occurs at offset', np.argmax(stdcorrelate_result) - midpoint + 1)
        plt.legend(['Fast correlate', 'Standard correlate'])
        plt.show()

    #assert (fastcorrelate_result == stdcorrelate_result).all
    aethresh = 10
    np.testing.assert_almost_equal(fastcorrelate_result, stdcorrelate_result, aethresh)


def main():
    test_fastcorrelate(display=True)


if __name__ == '__main__':
    main()
