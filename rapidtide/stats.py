#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import scipy as sp
from numba import jit
from scipy.stats import johnsonsb, kurtosis, kurtosistest

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()

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


donotusenumba = False


def disablenumba():
    global donotusenumba
    donotusenumba = True


def conditionaljit():
    def resdec(f):
        if donotusenumba:
            return f
        return jit(f, nopython=False)

    return resdec


def conditionaljit2():
    def resdec(f):
        if donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=False)

    return resdec


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
        print(
            "\tp <", "{:.3f}".format(1.0 - thepercentiles[i]), ": ", "{:.3f}".format(pcts[i]),
        )


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
    thestore = np.zeros((2, histlen), dtype="float64")
    thestore[0, :] = thehist[1][:-1]
    thestore[1, :] = thehist[0][:] / (1.0 * len(thedata))

    # store the zero term for later
    zeroterm = thestore[1, 0]
    thestore[1, 0] = 0.0

    # fit the johnsonSB function
    params = johnsonsb.fit(thedata[np.where(thedata > 0.0)])
    # print('Johnson SB fit parameters for pdf:', params)

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
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("fitjsbpdf: histogram")
        plt.plot(thestore[0, :], thestore[1, :], "b", thestore[0, :], johnsonsbvals, "r")
        plt.legend(["histogram", "fit to johnsonsb"])
        plt.show()
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


def sigFromDistributionData(
    vallist,
    histlen,
    thepercentiles,
    displayplots=False,
    twotail=False,
    nozero=False,
    dosighistfit=True,
):
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
        print("no nonzero values - skipping percentile calculation")
        return None, 0, 0
    thehistogram, peakheight, peakloc, peakwidth, centerofmass = makehistogram(
        np.abs(vallist), histlen, therange=[0.0, 1.0]
    )
    if dosighistfit:
        histfit = fitjsbpdf(
            thehistogram, histlen, vallist, displayplots=displayplots, nozero=nozero
        )
    if twotail:
        thepercentiles = 1.0 - (1.0 - thepercentiles) / 2.0
        print("thepercentiles adapted for two tailed distribution:", thepercentiles)
    pcts_data = getfracvals(vallist, thepercentiles, nozero=nozero)
    if dosighistfit:
        pcts_fit = getfracvalsfromfit(histfit, thepercentiles)
        return pcts_data, pcts_fit, histfit
    else:
        return pcts_data, 0, 0


def rfromp(fitfile, thepercentiles):
    """

    Parameters
    ----------
    fitfile
    thepercentiles

    Returns
    -------

    """
    thefit = np.array(tide_io.readvecs(fitfile)[0]).astype("float64")
    print("thefit = ", thefit)
    return getfracvalsfromfit(thefit, thepercentiles)


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


def kurtosisstats(timecourse):
    """

    Parameters
    ----------
    timecourse: array
        The timecourse to test

    :return:

    """
    testres = kurtosistest(timecourse)
    return kurtosis(timecourse), testres[0], testres[1]


# --------------------------- histogram functions -------------------------------------------------
def gethistprops(indata, histlen, refine=False, therange=None, pickleft=False, peakthresh=0.33):
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
    thestore = np.zeros((2, histlen), dtype="float64")
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
            if thestore[1, i] > peakthresh * overallmax:
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
    while (peakindex + numbins < histlen - 1) and (
        thestore[1, peakindex + numbins] > peakheight / 2.0
    ):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = tide_fit.gaussfit(
            peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :]
        )
    return peaklag, peakheight, peakwidth


def makehistogram(indata, histlen, binsize=None, therange=None, refine=False):
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
        thebins = np.linspace(
            therange[0], therange[1], (therange[1] - therange[0]) / binsize + 1, endpoint=True,
        )
    else:
        thebins = histlen
    thehist = np.histogram(indata, thebins, therange)

    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peakloc = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (
        thestore[1, peakindex + numbins] > peakheight / 2.0
    ):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peakloc, peakwidth = tide_fit.gaussfit(
            peakheight, peakloc, peakwidth, thestore[0, :], thestore[1, :]
        )
    centerofmass = np.sum(thestore[0, :] * thestore[1, :]) / np.sum(thestore[1, :])

    return thehist, peakheight, peakloc, peakwidth, centerofmass


def echoloc(indata, histlen, startoffset=5.0):
    thehist, peakheight, peakloc, peakwidth, centerofmass = makehistogram(
        indata, histlen, refine=True
    )
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]
    timestep = thestore[0, 1] - thestore[0, 0]
    startpt = np.argmax(thestore[1, :]) + int(startoffset // timestep)
    print("primary peak:", peakheight, peakloc, peakwidth)
    print("startpt, startloc, timestep:", startpt, thestore[1, startpt], timestep)
    while (thestore[1, startpt] > thestore[1, startpt + 1]) and (startpt < len(thehist[0]) - 2):
        startpt += 1
    echopeakindex = np.argmax(thestore[1, startpt:-2]) + startpt
    echopeakloc = thestore[0, echopeakindex + 1]
    echopeakheight = thestore[1, echopeakindex + 1]
    numbins = 1
    while (echopeakindex + numbins < histlen - 1) and (
        thestore[1, echopeakindex + numbins] > echopeakheight / 2.0
    ):
        numbins += 1
    echopeakwidth = (thestore[0, echopeakindex + numbins] - thestore[0, echopeakindex]) * 2.0
    echopeakheight, echopeakloc, echopeakwidth = tide_fit.gaussfit(
        echopeakheight, echopeakloc, echopeakwidth, thestore[0, :], thestore[1, :]
    )
    return echopeakloc - peakloc, (echopeakheight * echopeakwidth) / (peakheight * peakwidth)


def makeandsavehistogram(
    indata,
    histlen,
    endtrim,
    outname,
    binsize=None,
    displaytitle="histogram",
    displayplots=False,
    refine=False,
    therange=None,
    normalize=False,
    dictvarname=None,
    thedict=None,
    saveasbids=False,
    append=False,
    debug=False,
):
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
    normalize
    dictvarname
    thedict

    Returns
    -------

    """
    thehist, peakheight, peakloc, peakwidth, centerofmass = makehistogram(
        indata, histlen, binsize=binsize, therange=therange, refine=refine
    )
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]
    if normalize:
        totalval = np.sum(thestore[1, :])
        if totalval != 0.0:
            thestore[1, :] /= totalval
    if dictvarname is None:
        varroot = outname
    else:
        varroot = dictvarname
    if thedict is None:
        tide_io.writenpvecs(np.array([centerofmass]), outname + "_centerofmass.txt")
        tide_io.writenpvecs(np.array([peakloc]), outname + "_peak.txt")
    else:
        thedict[varroot + "_centerofmass.txt"] = centerofmass
        thedict[varroot + "_peak.txt"] = peakloc
    if saveasbids:
        tide_io.writebidstsv(
            outname,
            np.transpose(thestore[1, :]),
            1.0 / (thestore[0, 1] - thestore[0, 0]),
            starttime=thestore[0, 0],
            columns=[varroot],
            append=append,
            debug=debug,
        )
    else:
        tide_io.writenpvecs(thestore, outname + ".txt")
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        plt.plot(thestore[0, : (-1 - endtrim)], thestore[1, : (-1 - endtrim)])
        plt.show()


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
        return np.where(
            rvals > getfracvalsfromfit(sighistfit, 1.0 - pval), np.int16(1), np.int16(0)
        )
    else:
        return np.where(
            np.abs(rvals) > getfracvalsfromfit(sighistfit, 1.0 - pval / 2.0),
            np.int16(1),
            np.int16(0),
        )


# Find the image intensity value which thefrac of the non-zero voxels in the image exceed
def getfracval(datamat, thefrac, nozero=False):
    """

    Parameters
    ----------
    datamat
    thefrac

    Returns
    -------

    """
    return getfracvals(datamat, [thefrac], nozero=nozero)[0]


def getfracvals(datamat, thefracs, nozero=False, debug=False):
    """

    Parameters
    ----------
    datamat
    thefracs
    displayplots
    nozero
    debug

    Returns
    -------

    """
    thevals = []

    if nozero:
        maskmat = np.sort(datamat[np.where(datamat != 0.0)].flatten())
        if len(maskmat) == 0:
            for thisfrac in thefracs:
                thevals.append(0.0)
            return thevals
    else:
        maskmat = np.sort(datamat.flatten())
    maxindex = len(maskmat)

    for thisfrac in thefracs:
        theindex = np.min([int(np.round(thisfrac * maxindex, 0)), len(maskmat) - 1])
        thevals.append(maskmat[theindex])

    if debug:
        print("getfracvals: input datamat shape", datamat.shape)
        print("getfracvals: maskmat shape", maskmat.shape)
        print("getfracvals: thefracs", thefracs)
        print("getfracvals: maxindex", maxindex)
        print("getfracvals: thevals", thevals)

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

    cummeanhist = histfit[-1] + (1.0 - histfit[-1]) * johnsonsb.cdf(
        bins, histfit[0], histfit[1], histfit[2], histfit[3]
    )
    thevals = []
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_title("probability histogram")
        plt.plot(bins[-numbins:], meanhist[-numbins:])
        ax = fig.add_subplot(212)
        ax.set_title("cumulative mean sum of histogram")
        plt.plot(bins[-numbins:], cummeanhist[-numbins:])
        plt.show()
    for thisfrac in thefracs:
        target = cummeanhist[numbins - 1] * thisfrac
        thevals.append(0.0)
        for i in range(0, numbins):
            if cummeanhist[i] >= target:
                thevals[-1] = bins[i]
                break
    return thevals


def getfracvalsfromfit(histfit, thefracs):
    """

    Parameters
    ----------
    histfit
    thefracs
    displayplots

    Returns
    -------

    """
    # print('entering getfracvalsfromfit: histfit=',histfit, ' thefracs=', thefracs)
    thedist = johnsonsb(histfit[0], histfit[1], histfit[2], histfit[3])
    thevals = thedist.ppf(thefracs)
    return thevals


def makemask(image, threshpct=25.0, verbose=False, nozero=False, noneg=False):
    """

    Parameters
    ----------
    image: array-like
        The image data to generate the mask for.
    threshpct: float
        Voxels with values greater then threshpct of the 98th percentile of voxel values are preserved.
    verbose: bool
        If true, print additional debugging information.
    nozero: bool
        If true, exclude zero values when calculating percentiles
    noneg: bool
        If true, exclude negative values when calculating percentiles

    Returns
    -------
    themask: array-like
        An int16 mask with dimensions matching the input. 1 for voxels to preserve, 0 elsewhere

    """
    if noneg:
        pct2, pct98, pctthresh = getfracvals(
            np.where(image >= 0.0, image, 0.0), [0.02, 0.98, threshpct], nozero=nozero
        )
    else:
        pct2, pct98, pctthresh = getfracvals(image, [0.02, 0.98, threshpct], nozero=nozero)
    threshval = pct2 + (threshpct / 100.0) * (pct98 - pct2)
    print("old style threshval:", threshval, "new style threshval:", pctthresh)
    if verbose:
        print(
            "fracval:", pctthresh, " threshpct:", threshpct, " mask threshhold:", threshval,
        )
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
