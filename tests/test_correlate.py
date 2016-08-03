#!/usr/bin/env python
from __future__ import print_function

from tide_funcs import fastcorrelate
import numpy as np

def testfastcorrelate():
    sig1 = np.array([0.0,0.0,0,0,1.0,0.0,0,0,0.0])
    sig2 = np.array([0.0,0.0,0,0,1.0,0.0,0,0,0.0])
    assert (fastcorrelate(sig1,sig2) == np.correlate(sig1, sig2, mode='full')).all

def main():
    testfastcorrelate()

if __name__ == '__main__':
    main()
