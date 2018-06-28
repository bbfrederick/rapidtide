#!/usr/bin/env python
from __future__ import print_function
import rapidtide.tide_funcs as tide
import rapidtide.io_funcs as tide_io
import numpy as np
import pylab as plt

def mse(vec1, vec2):
    return np.mean(np.square(vec2 - vec1))

from scipy import arange


def testfindmaxlag(textfilename='../data/examples/src/lt_rt.txt', display=False, debug=False):
    # set default variable values
    searchfrac=0.75
    limitfit=False

    indata=tide_io.readvecs(textfilename)
    xvecs=indata[0,:]
    yvecs=indata[1,:]
    testmaxval = 0.8
    testmaxlag = 8.0
    testmaxsigma = 5.0
    yvecs=tide.gauss_eval(xvecs, np.array([testmaxval, testmaxlag, testmaxsigma]))
    lagmin = -20
    lagmax = 20
    widthlimit = 1000.0
    absmaxsigma = 1000.0

    maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide.findmaxlag_gauss(
        xvecs,
        yvecs,
        lagmin, lagmax, widthlimit,
        tweaklims=False,
        refine=True,
        debug=debug,
        searchfrac=searchfrac,
        zerooutbadfit=False)

    maxindexr, maxlagr, maxvalr, maxsigmar, maskvalr, failreasonr, peakstartr, peakendr = tide.findmaxlag_gauss_rev(
        xvecs,
        yvecs,
        lagmin, lagmax, widthlimit,
        absmaxsigma=absmaxsigma,
        tweaklims=False,
        refine=True,
        debug=debug,
        searchfrac=searchfrac,
        zerooutbadfit=False)

    print('final results:', maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend)
    print('final results:', maxindexr, maxlagr, maxvalr, maxsigmar, maskvalr, failreasonr, peakstartr, peakendr)
    oversampfactor=10
    gauss_xvecs=arange(xvecs[0],xvecs[-1],(xvecs[1]-xvecs[0])/oversampfactor,dtype='float')
    gauss_yvecs=tide.gauss_eval(gauss_xvecs, (maxval, maxlag, maxsigma))
    gauss_yvecsr=tide.gauss_eval(gauss_xvecs, (maxvalr, maxlagr, maxsigmar))
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xvecs,yvecs,'r')
        ax.plot(gauss_xvecs[(peakstart*oversampfactor):(peakend*oversampfactor+1)], 0.1 + gauss_yvecs[(peakstart*oversampfactor):(peakend*oversampfactor+1)],'g.')
        ax.plot(gauss_xvecs[(peakstartr*oversampfactor):(peakendr*oversampfactor+1)], 0.2 + gauss_yvecsr[(peakstartr*oversampfactor):(peakendr*oversampfactor+1)],'b.')
        #ax.set_xlim((lagmin, lagmax))
        plt.show()
    
def main():
    testfindmaxlag(display=True, debug=True)

if __name__ == '__main__':
    main()
