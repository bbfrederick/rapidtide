#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from rapidtide.tests.utils import mse
import rapidtide.glmpass as tide_glmpass


def gen2d(xsize=150, xcycles=11, tsize=200, tcycles=13, mean=10.0):
    thearray = np.zeros((xsize, tsize), dtype=np.float)
    xwaves = np.zeros((xsize, tsize), dtype=np.float)
    twaves = np.zeros((xsize, tsize), dtype=np.float)
    xmax = 2.0 * np.pi * xcycles
    tmax = 2.0 * np.pi * tcycles
    xfreq = xmax / xsize
    tfreq = tmax / tsize
    for i in range(tsize):
        thearray[:, i] = np.sin(np.linspace(0.0, xmax, xsize, endpoint=False))
        xwaves[:, i] =   np.sin(np.linspace(0.0, xmax, xsize, endpoint=False))
    for i in range(xsize):
        thearray[i, :] *= np.sin(np.linspace(0.0, tmax, tsize, endpoint=False))
        twaves[i, :] =    np.sin(np.linspace(0.0, tmax, tsize, endpoint=False))
    return thearray, xwaves, twaves
    

def test_glmpass(debug=True, display=False):
    xsize = 150
    xcycles = 11
    tsize = 200
    tcycles = 13
    mean = 100.0
    noiselevel = 5.0

    targetarray, xwaveforms, twaveforms = gen2d(xsize=xsize, xcycles=xcycles, tsize=tsize, tcycles=tcycles)
    testarray = targetarray + np.random.random((xsize, tsize)) + mean
    if display:
        plt.figure()
        plt.imshow(targetarray)
        plt.show()


    filtereddata = (0.0 * testarray)
    datatoremove = (0.0 * testarray)
    threshval = 0.0
    meanvals_t = np.zeros(tsize, dtype=np.float64)
    rvals_t = np.zeros(tsize, dtype=np.float64)
    r2vals_t = np.zeros(tsize, dtype=np.float64)
    fitcoffs_t = np.zeros(tsize, dtype=np.float64)
    fitNorm_t = np.zeros(tsize, dtype=np.float64)

    meanvals_x = np.zeros(xsize, dtype=np.float64)
    rvals_x = np.zeros(xsize, dtype=np.float64)
    r2vals_x = np.zeros(xsize, dtype=np.float64)
    fitcoffs_x = np.zeros(xsize, dtype=np.float64)
    fitNorm_x = np.zeros(xsize, dtype=np.float64)

    # run along time direction
    # no multiproc
    tide_glmpass.glmpass(tsize, testarray, threshval, xwaveforms,
                         meanvals_t, rvals_t, r2vals_t, fitcoffs_t, fitNorm_t,
                         datatoremove,
                         filtereddata,
                         showprogressbar=False,
                         mp_chunksize=10,
                         procbyvoxel=False,
                         nprocs=1
                         )
    if display:
        plt.figure()
        plt.imshow(datatoremove)
        plt.show()
        plt.imshow(filtereddata)
        plt.show()
    if debug:
        print('proc by time, single proc:', mse(datatoremove, targetarray))
    assert mse(datatoremove, targetarray) < 1e-3
    
    # multiproc
    tide_glmpass.glmpass(tsize, testarray, threshval, xwaveforms,
                         meanvals_t, rvals_t, r2vals_t, fitcoffs_t, fitNorm_t,
                         datatoremove,
                         filtereddata,
                         showprogressbar=False,
                         mp_chunksize=10,
                         procbyvoxel=False,
                         nprocs=-1
                         )
    if display:
        plt.figure()
        plt.imshow(datatoremove)
        plt.show()
        plt.imshow(filtereddata)
        plt.show()
    if debug:
        print('proc by time, multi proc:', mse(datatoremove, targetarray))
    assert mse(datatoremove, targetarray) < 1e-3
    
    # run along spatial direction
    # no multiproc
    tide_glmpass.glmpass(xsize, testarray, threshval, twaveforms,
                         meanvals_x, rvals_x, r2vals_x, fitcoffs_x, fitNorm_x,
                         datatoremove,
                         filtereddata,
                         showprogressbar=False,
                         mp_chunksize=10,
                         procbyvoxel=True,
                         nprocs=1
                         )
    if display:
        plt.figure()
        plt.imshow(datatoremove)
        plt.show()
        plt.imshow(filtereddata)
        plt.show()
    if debug:
        print('proc by space, single proc:', mse(datatoremove, targetarray))
    assert mse(datatoremove, targetarray) < 1e-3
    
    # multiproc
    tide_glmpass.glmpass(xsize, testarray, threshval, twaveforms,
                         meanvals_x, rvals_x, r2vals_x, fitcoffs_x, fitNorm_x,
                         datatoremove,
                         filtereddata,
                         showprogressbar=False,
                         mp_chunksize=10,
                         procbyvoxel=True,
                         nprocs=-1
                         )
    if display:
        plt.figure()
        plt.imshow(datatoremove)
        plt.show()
        plt.imshow(filtereddata)
        plt.show()
    if debug:
        print('proc by space, multi proc:', mse(datatoremove, targetarray))
    assert mse(datatoremove, targetarray) < 1e-3
    

def main():
    test_glmpass(debug=True, display=True)


if __name__ == '__main__':
    main()
