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

from rapidtide.correlate import aliasedcorrelate, aliasedcorrelator


def test_aliasedcorrelate(display=False):
    Fs_hi = 10.0
    Fs_lo = 1.0
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
    aliasedcorrelate_result = aliasedcorrelate(sighi, Fs_hi, siglo, Fs_lo, timerange, padvalue=width)

    thecorrelator = aliasedcorrelator(sighi, Fs_hi, Fs_lo, timerange, padvalue=width)
    aliasedcorrelate_result2 = thecorrelator.apply(siglo, 0.0)
    
    if display:
        plt.figure()
        #plt.ylim([-1.0, 3.0])
        plt.plot(hiaxis, sighi, 'k')
        plt.scatter(loaxis, siglo, c='r')
        plt.legend(['sighi', 'siglo'])

        plt.figure()
        plt.plot(timerange, aliasedcorrelate_result, 'k')
        plt.plot(timerange, aliasedcorrelate_result2, 'r')
        print('maximum occurs at offset', timerange[np.argmax(aliasedcorrelate_result)])

        plt.show()

    #assert (fastcorrelate_result == stdcorrelate_result).all
    aethresh = 10
    #np.testing.assert_almost_equal(fastcorrelate_result, stdcorrelate_result, aethresh)


def main():
    test_aliasedcorrelate(display=True)


if __name__ == '__main__':
    main()
