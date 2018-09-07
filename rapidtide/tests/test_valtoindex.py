#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import pylab as plt

from rapidtide.util import valtoindex


def test_valtoindex(debug=False):
    tr = 1.0
    testtr = 0.7
    xaxislen = 100
    shiftdist = 30
    xaxis = np.arange(0.0, tr * xaxislen, tr)
    minx = np.min(xaxis)
    maxx = np.max(xaxis)
    testvec = np.arange(-1.0, 1.1 * maxx, testtr)
    for i in range(len(testvec)):
        testval = testvec[i]
        indclosest = valtoindex(xaxis, testval)
        print(testval, xaxis[indclosest])

def main():
    test_valtoindex(debug=True)


if __name__ == '__main__':
    main()
