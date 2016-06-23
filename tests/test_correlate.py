from tide_funcs import fastcorrelate
import numpy as np

sig1 = np.array([0.0,0.0,0,0,1.0,0.0,0,0,0.0])
sig2 = np.array([0.0,0.0,0,0,1.0,0.0,0,0,0.0])

def testfastcorrelate(sig1, sig2):
    result = fastcorrelate(sig1,sig2)
    print(result)
    assert fastcorrelate(sig1,sig2) == result
