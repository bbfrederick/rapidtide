#!/usr/bin/env python
from tide_funcs import fastcorrelate
import numpy as np

def testfastcorrelate(sig1, sig2):
    result = fastcorrelate(sig1,sig2)
    print(result)
    assert fastcorrelate(sig1,sig2) == result

def main():
    sig1 = np.array([0.0,0.0,0,0,1.0,0.0,0,0,0.0])
    sig2 = np.array([0.0,0.0,0,0,1.0,0.0,0,0,0.0])
    testfastcorrelate(sig1, sig2)

if __name__ == '__main__':
    main()
