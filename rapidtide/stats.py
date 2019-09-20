#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
import pylab as pl

from scipy.stats import johnsonsb

import rapidtide.io as tide_io
import rapidtide.fit as tide_fit

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


def conditionaljit():
    def resdec(f):
        if (not numbaexists) or donotusenumba:
            return f
        return jit(f, nopython=False)

    return resdec


def conditionaljit2():
    def resdec(f):
        if (not numbaexists) or donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=False)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- probability functions -------------------------------------------------
def printthresholds(pcts, thepercentiles, labeltext):
    """

    Parameters
    ----------
    pcts
    thepercentiles
    labeltext

    Returns
    -------

    """
    print(labeltext)
    for i in range(0, len(pcts)):
        print('\tp <', "{:.3f}".format(1.0 - thepercentiles[i]), ': ', pcts[i])


def fitjsbpdf(thehist, histlen, thedata, displayplots=False, nozero=False):
    """

    Parameters
    ----------
    thehist
    histlen
    thedata
    displayplots
    nozero

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    percentile
    params
    zeroterm

    Returns
    -------

    """
    johnsonfunc = johnsonsb(params[0], params[1], params[2], params[3])
    corrfac = 1.0 - zeroterm


def sigFromDistributionData(vallist, histlen, thepercentiles, displayplots=False, twotail=False, nozero=False,
                            dosighistfit=True):
    """

    Parameters
    ----------
    vallist
    histlen
    thepercentiles
    displayplots
    twotail
    nozero
    dosighistfit

    Returns
    -------

    """
    # check to make sure there are nonzero values first
    if len(np.where(vallist != 0.0)[0]) == 0:
        print('no nonzero values - skipping percentile calculation')
        return None, 0, 0
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
    """

    Parameters
    ----------
    fitfile
    thepercentiles
    numbins

    Returns
    -------

    """
    thefit = np.array(tide_io.readvecs(fitfile)[0]).astype('float64')
    print('thefit = ', thefit)
    return getfracvalsfromfit(thefit, thepercentiles, numbins=1000, displayplots=True)


def tfromr(r, nsamps, dfcorrfac=1.0, oversampfactor=1.0, returnp=False):
    """

    Parameters
    ----------
    r
    nsamps
    dfcorrfac
    oversampfactor
    returnp

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    r
    nsamps
    dfcorrfac
    oversampfactor
    returnp

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    r

    Returns
    -------

    """
    return 0.5 * np.log((1 + r) / (1 - r))


# --------------------------- histogram functions -------------------------------------------------
def gethistprops(indata, histlen, refine=False, therange=None, pickleft=False):
    """

    Parameters
    ----------
    indata
    histlen
    refine
    therange
    pickleftpeak

    Returns
    -------

    """
    thestore = np.zeros((2, histlen), dtype='float64')
    if therange is None:
        thehist = np.histogram(indata, histlen)
    else:
        thehist = np.histogram(indata, histlen, therange)
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    if pickleft:
        overallmax = np.max(thestore[1, 1:-2])
        peakindex = 1
        i = 1
        started = False
        finished = False
        while i < len(thestore[1, :] - 2) and not finished:
            if thestore[1, i] > 0.33 * overallmax:
                started = True
            if thestore[1, i] > thestore[1, peakindex]:
                peakindex = i
            if started and (thestore[1, i] < 0.75 * thestore[1, peakindex]):
                finished = True
            i += 1
    else:
        peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (thestore[1, peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = tide_fit.gaussfit(peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :])
    return peaklag, peakheight, peakwidth


def makehistogram(indata, histlen, binsize=None, therange=None):
    """

    Parameters
    ----------
    indata
    histlen
    binsize
    therange

    Returns
    -------

    """
    if therange is None:
        therange = [indata.min(), indata.max()]
    if histlen is None and binsize is None:
        thebins = 10
    elif binsize is not None:
        thebins = sp.linspace(therange[0], therange[1], (therange[1] - therange[0]) / binsize + 1, endpoint=True)
    else:
        thebins = histlen
    thehist = np.histogram(indata, thebins, therange)
    return thehist


def makeandsavehistogram(indata, histlen, endtrim, outname,
                         binsize=None,
                         displaytitle='histogram',
                         displayplots=False,
                         refine=False,
                         therange=None):
    """

    Parameters
    ----------
    indata
    histlen
    endtrim
    outname
    displaytitle
    displayplots
    refine
    therange

    Returns
    -------

    """
    thehist = makehistogram(indata, histlen, binsize=binsize, therange=therange)
    thestore = np.zeros((2, len(thehist[0])), dtype='float64')
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
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
        peakheight, peaklag, peakwidth = tide_fit.gaussfit(peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :])
    centerofmass = np.sum(thestore[0, :] * thestore[1, :]) / np.sum(thestore[1, :])
    tide_io.writenpvecs(np.array([centerofmass]), outname + '_centerofmass.txt')
    tide_io.writenpvecs(np.array([peaklag]), outname + '_peak.txt')
    tide_io.writenpvecs(thestore, outname + '.txt')
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        pl.plot(thestore[0, :(-1 - endtrim)], thestore[1, :(-1 - endtrim)])


def symmetrize(a, antisymmetric=False, zerodiagonal=False):
    """

    Parameters
    ----------
    a
    antisymmetric
    zerodiagonal

    Returns
    -------

    """
    if antisymmetric:
        intermediate = (a - a.T) / 2.0
    else:
        intermediate = (a + a.T) / 2.0
    if zerodiagonal:
        return intermediate - np.diag(intermediate.diagonal())
    else:
        return intermediate


def makepmask(rvals, pval, sighistfit, onesided=True):
    """

    Parameters
    ----------
    rvals
    pval
    sighistfit
    onesided

    Returns
    -------

    """
    if onesided:
        return np.where(rvals > getfracvalsfromfit(sighistfit, 1.0 - pval), np.int16(1), np.int16(0))
    else:
        return np.where(np.abs(rvals) > getfracvalsfromfit(sighistfit, 1.0 - pval / 2.0), np.int16(1), np.int16(0))


# Find the image intensity value which thefrac of the non-zero voxels in the image exceed
def getfracval(datamat, thefrac, numbins=200, nozero=False):
    """

    Parameters
    ----------
    datamat
    thefrac
    numbins

    Returns
    -------

    """
    return getfracvals(datamat, [thefrac], numbins=numbins, nozero=nozero)[0]


def getfracvals(datamat, thefracs, numbins=200, displayplots=False, nozero=False):
    """

    Parameters
    ----------
    datamat
    thefracs
    numbins
    displayplots
    nozero

    Returns
    -------

    """
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


def getfracvalsfromfit_old(histfit, thefracs, numbins=2000, displayplots=False):
    """

    Parameters
    ----------
    histfit
    thefracs
    numbins
    displayplots

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    histfit
    thefracs
    numbins
    displayplots

    Returns
    -------

    """
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


def makemask(image, threshpct=25.0, verbose=False, nozero=False):
    """

    Parameters
    ----------
    image: array-like
        The image data to generate the mask for.
    threshpct: float
        Voxels with values greater then threshpct of the 98th percentile of voxel values are preserved.
    verbose: bool
        If true, print additional debugging information.

    Returns
    -------
    themask: array-like
        An int16 mask with dimensions matching the input. 1 for voxels to preserve, 0 elsewhere

    """
    pct2, pct98 = getfracvals(image, [0.02, 0.98], nozero=nozero)
    threshval = pct2 + (threshpct / 100.0) * (pct98 - pct2)
    if verbose:
        print('fracval:', fracval, ' threshpct:', threshpct, ' mask threshhold:', threshval)
    themask = np.where(image > threshval, np.int16(1), np.int16(0))
    return themask


def getmasksize(themask):
    """

    Parameters
    ----------
    image: array-like
        The mask data to check.

    Returns
    -------
    numvoxels: int
        The number of nonzero voxels in themask

    """
    return len(np.where(themask > 0)[0])


