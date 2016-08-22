#!/usr/bin/env python
from tide_funcs import fastresampler
import numpy as np
import pylab as plt

def mse(vec1, vec2):
    return np.mean(np.square(vec2 - vec1))


def testfastresampler(debug=False):
    tr = 1.0
    padvalue = 30.0
    testlen = 1000
    shiftdist = 30
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr
    timecoursein = np.zeros((testlen), dtype='float')
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint] = 1.0
    timecoursein -= 0.5

    # generate the ground truth rolled regressors
    tcrolled_forward = np.roll(timecoursein, shiftdist)
    tcrolled_backward = np.roll(timecoursein, -shiftdist)

    # generate the fast resampled regressor
    genlaggedtc = fastresampler(timeaxis, timecoursein, padvalue=padvalue)
    tcshifted_forward = genlaggedtc.yfromx(timeaxis - shiftdist, debug=debug)
    tcshifted_backward = genlaggedtc.yfromx(timeaxis + shiftdist, debug=debug)
    if debug:
        plt.figure()
        plt.ylim([-2,6])
        plt.hold(True)
        plt.plot(timecoursein)
        plt.plot(tcrolled_forward + 1.0)
        plt.plot(tcshifted_forward + 2.0)
        plt.plot(tcrolled_backward + 3.0)
        plt.plot(tcshifted_backward + 4.0)
        plt.legend(['Original', 'Straight shift forward', 'Fastresampler forward', 'Straight shift backward', 'Fastresampler backward'])
        plt.show()

    #assert mse(tcrolled_forward, tcshifted_forward) < 1e-10
    np.testing.assert_almost_equal(tcrolled_forward, tcshifted_forward)
    #np.testing.assert_almost_equal(tcrolled_backward, tcshifted_backward)
    
def main():
    testfastresampler(debug=True)

if __name__ == '__main__':
    main()
