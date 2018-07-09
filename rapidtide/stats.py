#!/usr/bin/env python
#
#   Copyright 2016 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# $Author: frederic $
# $Date: 2016/07/12 13:50:29 $
# $Id: tide_funcs.py,v 1.4 2016/07/12 13:50:29 frederic Exp $
#
from __future__ import print_function, division

import numpy as np
import scipy as sp
from scipy import fftpack, ndimage, signal
from numpy.fft import rfftn, irfftn
import pylab as pl
import warnings
import time
import sys
import bisect
import os
import pandas as pd
import json
import resource

#from scipy import signal
from scipy.stats import johnsonsb

import rapidtide.io as tide_io

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False


try:
    from numba import jit

    numbaexists = True
except ImportError:
    numbaexists = False


try:
    import nibabel as nib

    nibabelexists = True
except ImportError:
    nibabelexists = False


donotusenumba = False


try:
    import pyfftw
    pyfftwexists = True
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()
except ImportError:
    pyfftwexists = False


def checkimports(optiondict):
    if pyfftwexists:
        print('monkey patched scipy.fftpack to use pyfftw')
    else:
        print('using standard scipy.fftpack')
    optiondict['pyfftwexists'] = pyfftwexists

    if numbaexists:
        print('numba exists')
    else:
        print('numba does not exist')
    optiondict['numbaexists'] = numbaexists

    if memprofilerexists:
        print('memprofiler exists')
    else:
        print('memprofiler does not exist')
    optiondict['memprofilerexists'] = memprofilerexists

    if nibabelexists:
        print('nibabel exists')
    else:
        print('nibabel does not exist')
    optiondict['nibabelexists'] = nibabelexists

    if donotbeaggressive:
        print('no aggressive optimization')
    else:
        print('aggressive optimization')
    optiondict['donotbeaggressive'] = donotbeaggressive

    global donotusenumba
    if donotusenumba:
        print('will not use numba even if present')
    else:
        print('using numba if present')
    optiondict['donotusenumba'] = donotusenumba


def conditionaljit():
    def resdec(f):
        if (not numbaexists) or donotusenumba:
            return f
        return jit(f)

    return resdec


def conditionaljit2():
    def resdec(f):
        if (not numbaexists) or donotusenumba or donotbeaggressive:
            return f
        return jit(f)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- probability functions -------------------------------------------------
def printthresholds(pcts, thepercentiles, labeltext):
    print(labeltext)
    for i in range(0, len(pcts)):
        print('\tp <', "{:.3f}".format(1.0 - thepercentiles[i]), ': ', pcts[i])


def fitjsbpdf(thehist, histlen, thedata, displayplots=False, nozero=False):
    thestore = np.zeros((2, histlen), dtype='float64')
    thestore[0, :] = thehist[1][:-1]
    thestore[1, :] = thehist[0][:] / (1.0 * len(thedata))

    # store the zero term for later
    zeroterm = thestore[1, 0]
    thestore[1, 0] = 0.0

    # fit the johnsonSB function
    params = johnsonsb.fit(thedata[np.where(thedata > 0.0)])
    #print('Johnson SB fit parameters for pdf:', params)

    # restore the zero term if needed
    # if nozero is True, assume that R=0 is not special (i.e. there is no spike in the
    # histogram at zero from failed fits)
    if nozero:
        zeroterm = 0.0
    else:
        thestore[1, 0] = zeroterm

    # generate the johnsonsb function
    johnsonsbvals = johnsonsb.pdf(thestore[0, :], params[0], params[1], params[2], params[3])
    corrfac = (1.0 - zeroterm) / (1.0 * histlen)
    johnsonsbvals *= corrfac
    johnsonsbvals[0] = zeroterm

    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('fitjsbpdf: histogram')
        pl.plot(thestore[0, :], thestore[1, :], 'b',
                thestore[0, :], johnsonsbvals, 'r')
        pl.legend(['histogram', 'fit to johnsonsb'])
        pl.show()
    return np.append(params, np.array([zeroterm]))


def getjohnsonppf(percentile, params, zeroterm):
    johnsonfunc = johnsonsb(params[0], params[1], params[2], params[3])
    corrfac = 1.0 - zeroterm


def sigFromDistributionData(vallist, histlen, thepercentiles, displayplots=False, twotail=False, nozero=False,
                            dosighistfit=True):
    thehistogram = makehistogram(np.abs(vallist), histlen, therange=[0.0, 1.0])
    if dosighistfit:
        histfit = fitjsbpdf(thehistogram, histlen, vallist, displayplots=displayplots, nozero=nozero)
    if twotail:
        thepercentiles = 1.0 - (1.0 - thepercentiles) / 2.0
        print('thepercentiles adapted for two tailed distribution:', thepercentiles)
    pcts_data = getfracvals(vallist, thepercentiles, numbins=int(np.sqrt(len(vallist)) * 5.0), nozero=nozero)
    if dosighistfit:
        pcts_fit = getfracvalsfromfit(histfit, thepercentiles, numbins=histlen, displayplots=displayplots)
        return pcts_data, pcts_fit, histfit
    else:
        return pcts_data, 0, 0


def rfromp(fitfile, thepercentiles, numbins=1000):
    thefit = np.array(tide_io.readvecs(fitfile)[0]).astype('float64')
    print('thefit = ', thefit)
    return getfracvalsfromfit(thefit, thepercentiles, numbins=1000, displayplots=True)


def tfromr(r, nsamps, dfcorrfac=1.0, oversampfactor=1.0, returnp=False):
    if r >= 1.0:
        tval = float("inf")
        pval = 0.0
    else:
        dof = int((dfcorrfac * nsamps) // oversampfactor)
        tval = r * np.sqrt(dof / (1 - r * r))
        pval = sp.stats.t.sf(abs(tval), dof) * 2.0
    if returnp:
        return tval, pval
    else:
        return tval


def zfromr(r, nsamps, dfcorrfac=1.0, oversampfactor=1.0, returnp=False):
    if r >= 1.0:
        zval = float("inf")
        pval = 0.0
    else:
        dof = int((dfcorrfac * nsamps) // oversampfactor)
        zval = r / np.sqrt(1.0 / (dof - 3))
        pval = 1.0 - sp.stats.norm.cdf(abs(zval))
    if returnp:
        return zval, pval
    else:
        return zval


def fisher(r):
    return 0.5 * np.log((1 + r) / (1 - r))


# --------------------------- histogram functions -------------------------------------------------
def gethistprops(indata, histlen, refine=False, therange=None):
    thestore = np.zeros((2, histlen), dtype='float64')
    if therange is None:
        thehist = np.histogram(indata, histlen)
    else:
        thehist = np.histogram(indata, histlen, therange)
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (thestore[1, peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = gaussfit(peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :])
    return peaklag, peakheight, peakwidth


def makehistogram(indata, histlen, therange=None):
    if therange is None:
        thehist = np.histogram(indata, histlen)
    else:
        thehist = np.histogram(indata, histlen, therange)
    return thehist


def makeandsavehistogram(indata, histlen, endtrim, outname,
                         displaytitle='histogram', displayplots=False,
                         refine=False, therange=None):
    thestore = np.zeros((2, histlen), dtype='float64')
    thehist = makehistogram(indata, histlen, therange)
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    # peakheight = max(thestore[1,:])
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (thestore[1, peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = gaussfit(peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :])
    tide_io.writenpvecs(np.array([peaklag]), outname + '_peak.txt')
    tide_io.writenpvecs(thestore, outname + '.txt')
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        pl.plot(thestore[0, :(-1 - endtrim)], thestore[1, :(-1 - endtrim)])


def symmetrize(a, antisymmetric=False, zerodiagonal=False):
    if antisymmetric:
        intermediate = (a - a.T) / 2.0
    else:
        intermediate = (a + a.T) / 2.0
    if zerodiagonal:
        return intermediate - np.diag(intermediate.diagonal())
    else:
        return intermediate


# Find the image intensity value which thefrac of the non-zero voxels in the image exceed
def getfracval(datamat, thefrac, numbins=200):
    themax = datamat.max()
    themin = datamat.min()
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    cummeanhist = np.cumsum(meanhist)
    target = cummeanhist[numbins - 1] * thefrac
    for i in range(0, numbins):
        if cummeanhist[i] >= target:
            return bins[i]
    return 0.0


def makepmask(rvals, pval, sighistfit, onesided=True):
    if onesided:
        return np.where(rvals > getfracvalsfromfit(sighistfit, 1.0 - pval), np.int16(1), np.int16(0))
    else:
        return np.where(np.abs(rvals) > getfracvalsfromfit(sighistfit, 1.0 - pval / 2.0), np.int16(1), np.int16(0))


def getfracvals(datamat, thefracs, numbins=200, displayplots=False, nozero=False):
    themax = datamat.max()
    themin = datamat.min()
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    cummeanhist = np.cumsum(meanhist)
    if nozero:
        cummeanhist = cummeanhist - cummeanhist[0]
    thevals = []
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('cumulative mean sum of histogram')
        plot(bins[-numbins:], cummeanhist[-numbins:])
        pl.show()
    for thisfrac in thefracs:
        target = cummeanhist[numbins - 1] * thisfrac
        thevals.append(0.0)
        for i in range(0, numbins):
            if cummeanhist[i] >= target:
                thevals[-1] = bins[i]
                break
    return thevals


def getfracvalsfromfit_old(histfit, thefracs, numbins=2000, displayplots=False):
    themax = 1.0
    themin = 0.0
    bins = np.arange(themin, themax, (themax - themin) / numbins)
    meanhist = johnsonsb.pdf(bins, histfit[0], histfit[1], histfit[2], histfit[3])
    corrfac = (1.0 - histfit[-1]) / (1.0 * numbins)
    meanhist *= corrfac
    meanhist[0] = histfit[-1]

    cummeanhist = histfit[-1] + (1.0 - histfit[-1]) * johnsonsb.cdf(bins, histfit[0], histfit[1], histfit[2],
                                                                    histfit[3])
    thevals = []
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(211)
        ax.set_title('probability histogram')
        pl.plot(bins[-numbins:], meanhist[-numbins:])
        ax = fig.add_subplot(212)
        ax.set_title('cumulative mean sum of histogram')
        pl.plot(bins[-numbins:], cummeanhist[-numbins:])
        pl.show()
    for thisfrac in thefracs:
        target = cummeanhist[numbins - 1] * thisfrac
        thevals.append(0.0)
        for i in range(0, numbins):
            if cummeanhist[i] >= target:
                thevals[-1] = bins[i]
                break
    return thevals


def getfracvalsfromfit(histfit, thefracs, numbins=2000, displayplots=True):
    # print('entering getfracvalsfromfit: histfit=',histfit, ' thefracs=', thefracs)
    thedist = johnsonsb(histfit[0], histfit[1], histfit[2], histfit[3])
    # print('froze the distribution')
    if displayplots:
        themin = 0.001
        themax = 0.999
        bins = np.arange(themin, themax, (themax - themin) / numbins)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('probability histogram')
        pl.plot(bins, johnsonsb.ppf(thefracs, histfit[0], histfit[1], histfit[2], histfit[3]))
        pl.show()
    # thevals = johnsonsb.ppf(thefracs, histfit[0], histfit[1], histfit[2], histfit[3])
    thevals = thedist.ppf(thefracs)
    return thevals


def makemask(image, threshpct=25.0, verbose=False):
    fracval = getfracval(image, 0.98)
    threshval = (threshpct / 100.0) * fracval
    if verbose:
        print('fracval:', fracval, ' threshpct:', threshpct, ' mask threshhold:', threshval)
    themask = np.where(image > threshval, np.int16(1), np.int16(0))
    return themask


