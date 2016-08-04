#!/usr/bin/env python
from tide_funcs import fastresampler
import numpy as np
import pylab as plt

def mse(vec1, vec2):
    return np.mean(np.square(vec2 - vec1))


def testfastresampler(display=False):
    tr = 1.0
    testlen = 1000
    shiftdist = 60
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr
    timecoursein = np.zeros((testlen), dtype='float')
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint] = 1.0
    timecoursein -= 0.5

    # generate the ground truth rolled regressor
    tcrolled = np.roll(timecoursein, shiftdist)

    # generate the fast resampled regressor
    genlaggedtc = fastresampler(timeaxis, timecoursein)
    tcshifted = genlaggedtc.yfromx(timeaxis - shiftdist)
    if display:
        plt.figure()
        plt.ylim([-2,4])
        plt.hold(True)
        plt.plot(timecoursein)
        plt.plot(tcrolled + 1.0)
        plt.plot(tcshifted + 2.0)
        plt.legend(['Original', 'Straight shift', 'Fastresampler'])
        plt.show()

    assert mse(tcrolled, tcshifted) < 1e-10
    
def main():
    testfastresampler(display=True)

if __name__ == '__main__':
    main()
