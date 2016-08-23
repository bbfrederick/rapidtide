#!/usr/bin/env python
from __future__ import print_function, division

from tide_funcs import fastcorrelate
import numpy as np
import pylab as plt

def testfastcorrelate(display=False):
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
        plt.hold(True)
        plt.plot(stdcorrelate_result)
        print('maximum occurs at offset', np.argmax(stdcorrelate_result) - midpoint + 1)
        plt.legend(['Fast correlate', 'Standard correlate'])
        plt.show()

    #assert (fastcorrelate_result == stdcorrelate_result).all
    aethresh = 10
    np.testing.assert_almost_equal(fastcorrelate_result, stdcorrelate_result, aethresh)

def main():
    testfastcorrelate(display=True)

if __name__ == '__main__':
    main()
