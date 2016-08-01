#!/usr/bin/env python
from tide_funcs import fastresampler
import numpy as np
from pylab import plot, show

def mse(vec1, vec2):
    return np.mean(np.square(vec2 - vec1))


def testfastcorrelate(sig1, sig2):
    result = fastcorrelate(sig1,sig2)
    print(result)
    assert fastcorrelate(sig1,sig2) == result

def main():
    testlen = 100
    shiftdist = 10
    timeaxis = np.arange(0.0, 1.0 * testlen)
    timecoursein = np.zeros((testlen), dtype='float')
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint] = 1.0
    tcrolled = np.roll(timecoursein, shiftdist)
    genlaggedtc = fastresampler(timeaxis, timecoursein)
    tcshifted = genlaggedtc.yfromx(timeaxis - shiftdist)
    #plot(timeaxis, timecoursein, timeaxis, 1.1 + tcrolled, timeaxis, 2.2 + tcshifted)
    plot (timeaxis, tcshifted - tcrolled)
    print('mean squared error:', mse(tcrolled, tcshifted)) 
    show()
    
if __name__ == '__main__':
    main()
