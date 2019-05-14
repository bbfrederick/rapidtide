#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function, division

import numpy as np
import scipy as sp

from rapidtide.resample import congrid
from rapidtide.filter import dolpfiltfilt
#from rapidtide.tests.utils import

import matplotlib.pyplot as plt


def funcvalue(x, midpoint=500.0, width=25.0):
    if midpoint - width <= x <= midpoint + width:
        return 0.5
    else:
        return -0.5

def funcvalue2(x, frequency=0.01, phase=0.0, amplitude=1.5):
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def test_congrid(debug=False):
    # make the source waveform
    tr = 1.0
    sourcelen = 1000
    sourceaxis = sp.linspace(0.0, tr * sourcelen, num=sourcelen, endpoint=False)
    if debug:
        print('sourceaxis range:', sourceaxis[0], sourceaxis[-1])

    timecoursein = np.float64(sourceaxis * 0.0)
    for i in range(len(sourceaxis)):
        timecoursein[i] = funcvalue2(sourceaxis[i])

    '''
    timecoursein = np.float64(sourceaxis * 0.0)
    midpoint = int(sourcelen // 2) + 1
    timecoursein[midpoint - 20:midpoint + 20] = np.float64(1.0)
    timecoursein -= 0.5
    '''

    # now make the destination
    gridlen = 150
    gridaxis = sp.linspace(0.0, tr * sourcelen, num=gridlen, endpoint=False)
    if debug:
        print('gridaxis range:', gridaxis[0], gridaxis[-1])
    weights = np.zeros((gridlen), dtype=float)
    griddeddata = np.zeros((gridlen), dtype=float)

    # define the gridding
    congridbins = 1.5
    gridkernel = 'gauss'

    #butterorder = 4
    #timecoursein = 0.5 * dolpfiltfilt(1.0, 0.25, timecoursein, butterorder) + 0.5

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

    print('gridding done')
    print('debug:', debug)

    # plot if we are doing that
    if debug:
        offset = 0.0
        legend = []
        offset += 1.0
        plt.plot(sourceaxis, timecoursein)
        legend.append('Original')
        offset += 1.0
        plt.plot(gridaxis, griddeddata)
        legend.append('Gridded')
        plt.plot(gridaxis, weights)
        legend.append('Weights')

    # do the tests
    #msethresh = 1e-6
    #aethresh = 2
    #assert mse(tcrolled, tcshifted) < msethresh
    #np.testing.assert_almost_equal(tcrolled, tcshifted, aethresh)

    if debug:
        plt.legend(legend)
        plt.show()


def main():
    test_congrid(debug=True)


if __name__ == '__main__':
    main()
