#!/usr/bin/env python
from __future__ import print_function, division
from rapidtide.tide_funcs import timeshift, dolpfiltfilt
import numpy as np
import pylab as plt

def mse(vec1, vec2):
    return np.mean(np.square(vec2 - vec1))


def testtimeshift(debug=False):
    tr = 1.0
    padvalue = 30.0
    testlen = 1000
    shiftdist = 30
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr
    #timecoursein = np.zeros((testlen), dtype='float64')
    timecoursein = np.float64(timeaxis * 0.0)
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint - 1] = np.float64(1.0)
    timecoursein[midpoint] = np.float64(1.0)
    timecoursein[midpoint + 1] = np.float64(1.0)
    timecoursein -= 0.5
    butterorder = 4
    timecoursein = 0.5 * dolpfiltfilt(1.0, 0.25, timecoursein, butterorder) + 0.5

    shiftlist = [-30, -20, -10, 0, 10, 20, 30]

    if debug:
        plt.figure()
        plt.ylim([-1.0, 2.0 * len(shiftlist) + 1.0])
        plt.plot(timecoursein)
        legend = ['Original']
        offset = 0.0

    for shiftdist in shiftlist:
        # generate the ground truth rolled regressor
        tcrolled = np.float64(np.roll(timecoursein, int(shiftdist)))

        # generate the fast resampled regressor
        #tcshifted = genlaggedtc.yfromx(timeaxis - shiftdist, debug=debug)
        tcshifted, weights, alltc, allweights = timeshift(timecoursein, 1.0 * shiftdist, int(padvalue // tr), doplot=False)

        # print out all elements
        print("len tcrolled=", len(tcrolled), "len tcshifted=", len(tcshifted))
        for i in range(0, len(tcrolled)):
            print(i, tcrolled[i], tcshifted[i], tcshifted[i] - tcrolled[i])

        # plot if we are doing that
        if debug:
            offset += 1.0
            plt.plot(tcrolled + offset)
            legend.append('Roll ' + str(shiftdist))
            offset += 1.0
            plt.plot(tcshifted + offset)
            legend.append('Timeshift ' + str(shiftdist))

        # do the tests
        msethresh = 1e-6
        aethresh = 2
        assert mse(tcrolled, tcshifted) < msethresh
        np.testing.assert_almost_equal(tcrolled, tcshifted, aethresh)

    if debug:
        plt.legend(legend)
        plt.show()
    
def main():
    testtimeshift(debug=True)

if __name__ == '__main__':
    main()
