#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function, division

import numpy as np
import scipy as sp

from rapidtide.resample import congrid
from rapidtide.filter import dolpfiltfilt
from rapidtide.tests.utils import mse

import matplotlib.pyplot as plt


def funcvalue2(x, frequency=0.01, phase=0.0, amplitude=1.5):
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def test_congrid(debug=False, display=False):
    # make the source waveform
    tr = 1.0
    sourcelen = 1000
    sourceaxis = sp.linspace(0.0, tr * sourcelen, num=sourcelen, endpoint=False)
    if debug:
        print('sourceaxis range:', sourceaxis[0], sourceaxis[-1])

    timecoursein = np.float64(sourceaxis * 0.0)
    for i in range(len(sourceaxis)):
        timecoursein[i] = funcvalue2(sourceaxis[i])

    # now make the destination
    gridlen = 150
    gridaxis = sp.linspace(0.0, tr * sourcelen, num=gridlen, endpoint=False)
    if debug:
        print('gridaxis range:', gridaxis[0], gridaxis[-1])
    weights = np.zeros((gridlen), dtype=float)
    griddeddata = np.zeros((gridlen), dtype=float)

    # define the gridding
    congridbins = 1.5

    for gridkernel in ['gauss']:
        print('about to grid')

        numsamples = 5000
        for i in range(numsamples):
            t = np.random.uniform() * tr * sourcelen
            thevals, theweights, theindices = congrid(gridaxis,
                                                    t,
                                                    funcvalue2(t),
                                                    congridbins,
                                                    kernel=gridkernel,
                                                    debug=True)
            for i in range(len(theindices)):
                weights[theindices[i]] += theweights[i]
                griddeddata[theindices[i]] += thevals[i]

        griddeddata = np.where(weights > 0.0, griddeddata / weights, 0.0)

        target = np.float64(gridaxis * 0.0)
        for i in range(len(gridaxis)):
            target[i] = funcvalue2(gridaxis[i])

        print('gridding done')
        print('debug:', debug)

        # plot if we are doing that
        if display:
            offset = 0.0
            legend = []
            plt.plot(sourceaxis, timecoursein)
            legend.append('Original')
            offset += 1.0
            plt.plot(gridaxis, target + offset)
            legend.append('Target')
            offset += 1.0
            plt.plot(gridaxis, griddeddata + offset)
            legend.append('Gridded')
            plt.plot(gridaxis, weights)
            legend.append('Weights')
            plt.legend(legend)
            plt.show()

        # do the tests
        msethresh = 1e-3
        themse = mse(target, griddeddata)
        if debug:
            print('mse:', themse)
        assert themse < msethresh

def main():
    test_congrid(debug=True, display=True)


if __name__ == '__main__':
    main()
