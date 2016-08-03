#!/usr/bin/env python
from tide_funcs import fastresampler
import numpy as np
from pylab import plot, show

def mse(vec1, vec2):
    return np.mean(np.square(vec2 - vec1))


def testfastresampler():
    testlen = 1000
    shiftdist = 100
    timeaxis = np.arange(0.0, 1.0 * testlen)
    timecoursein = np.zeros((testlen), dtype='float')
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint] = 1.0
    tcrolled = np.roll(timecoursein, shiftdist)
    genlaggedtc = fastresampler(timeaxis, timecoursein)
    tcshifted = genlaggedtc.yfromx(timeaxis - shiftdist)
    assert mse(tcrolled, tcshifted) < 1e-10
    
def main():
    testfastresampler()

if __name__ == '__main__':
    main()
