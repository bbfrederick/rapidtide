#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
#
import warnings

import matplotlib.pyplot as plt
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

import scipy as sp
from scipy.stats import johnsonsb, kurtosis, kurtosistest, skew, skewtest

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()


# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from numba import jit
except ImportError:
    donotusenumba = True
else:
    donotusenumba = False


def disablenumba():
    global donotusenumba
    donotusenumba = True


def conditionaljit():
    def resdec(f):
        if donotusenumba:
            return f
        return jit(f, nopython=True)

    return resdec


def conditionaljit2():
    def resdec(f):
        if donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=True)

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
        print(f"\tp <{1.0 - thepercentiles[i]:.3f}: {pcts[i]:.3f}")


def fitgausspdf(thehist, histlen, thedata, displayplots=False, nozero=False, debug=False):
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

    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (
        thestore[1, peakindex + numbins] > peakheight / 2.0
    ):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if debug:
        print("Initial values:")
        print(f"\tPeak height: {peakheight}")
        print(f"\tPeak lag: {peaklag}")
        print(f"\tPeak width: {peakwidth}")
    peakheight, peaklag, peakwidth = tide_fit.gaussfit(
        peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :]
    )
    if debug:
        print("Refined values:")
        print(f"\tPeak height: {peakheight}")
        print(f"\tPeak lag: {peaklag}")
        print(f"\tPeak width: {peakwidth}")

    params = (peakheight, peaklag, peakwidth)

    # restore the zero term if needed
    # if nozero is True, assume that R=0 is not special (i.e. there is no spike in the
    # histogram at zero from failed fits)
    if nozero:
        zeroterm = 0.0
    else:
        thestore[1, 0] = zeroterm

    # generate the johnsonsb function
    gaussvals = tide_fit.gauss_eval(thestore[0, :], params)
    corrfac = (1.0 - zeroterm) / (1.0 * histlen)
    gaussvals *= corrfac
    gaussvals[0] = zeroterm

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("fitgausspdf: histogram")
        plt.plot(thestore[0, :], thestore[1, :], "b", thestore[0, :], gaussvals, "r")
        plt.legend(["histogram", "fit to gaussian"])
        plt.show()
    return np.append(params, np.array([zeroterm]))


def fitjsbpdf(thehist, histlen, thedata, displayplots=False, nozero=False, debug=False):
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
    if debug:
        print(f"Johnson SB fit parameters for pdf: {params}")

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
    similaritymetric="correlation",
    displayplots=False,
    twotail=False,
    nozero=False,
    dosighistfit=True,
    debug=False,
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
        return None, None, None
    thehistogram, peakheight, peakloc, peakwidth, centerofmass, peakpercentile = makehistogram(
        np.abs(vallist), histlen, therange=[0.0, 1.0]
    )
    if dosighistfit:
        if similaritymetric == "mutualinfo":
            histfit = fitgausspdf(
                thehistogram,
                histlen,
                vallist,
                displayplots=displayplots,
                nozero=nozero,
                debug=debug,
            )
        else:
            histfit = fitjsbpdf(
                thehistogram,
                histlen,
                vallist,
                displayplots=displayplots,
                nozero=nozero,
                debug=debug,
            )
    if twotail:
        thepercentiles = 1.0 - (1.0 - thepercentiles) / 2.0
        print(f"thepercentiles adapted for two tailed distribution: {thepercentiles}")
    pcts_data = getfracvals(vallist, thepercentiles, nozero=nozero)
    if dosighistfit:
        pcts_fit = getfracvalsfromfit(histfit, thepercentiles)
        return pcts_data, pcts_fit, histfit
    else:
        pcts_fit = []
        for i in len(pcts_data):
            pcts_fit.append(None)
        return pcts_data, pcts_fit, None


global neglogpfromr_interpolator, minrforneglogp, maxrforneglogp
neglogpfromr_interpolator = None
minrforneglogp = None
maxrforneglogp = None


def neglog10pfromr(
    rval, histfit, lutlen=3000, initialize=False, neglogpmin=0.0, neglogpmax=3.0, debug=False
):
    global neglogpfromr_interpolator, minrforneglogp, maxrforneglogp
    if neglogpfromr_interpolator is None or initialize:
        neglogparray = np.linspace(neglogpmin, neglogpmax, lutlen, endpoint=True)
        pvals = pow(10.0, -neglogparray)
        percentile_list = (1.0 - pvals).tolist()
        rforneglogp = np.asarray(getfracvalsfromfit(histfit, percentile_list), dtype=float)
        minrforneglogp = rforneglogp[0]
        maxrforneglogp = rforneglogp[-1]
        if debug:
            print("START NEGLOGPFROMR DEBUG")
            print("neglogp\tpval\tpct\trfornlp")
            for i in range(lutlen):
                print(f"{neglogparray[i]}\t{pvals[i]}\t{percentile_list[i]}\t{rforneglogp[i]}")
            print("END NEGLOGPFROMR DEBUG")
        neglogpfromr_interpolator = sp.interpolate.UnivariateSpline(
            rforneglogp, neglogparray, k=3, s=0
        )
    if rval > maxrforneglogp:
        return np.float64(neglogpmax)
    elif rval < minrforneglogp:
        return np.float64(neglogpmin)
    else:
        return np.float64(neglogpfromr_interpolator(np.asarray([rval], dtype=float))[0])


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
    print(f"thefit = {thefit}")
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


def pfromz(z, twotailed=True):
    # importing packages
    import scipy.stats

    # finding p-value
    if twotailed:
        return scipy.stats.norm.sf(abs(z)) * 2
    else:
        return scipy.stats.norm.sf(abs(z))


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


def zofcorrdiff(r1, r2, n1, n2):
    return (fisher(r1) - fisher(r2)) / stderrofdiff(n1, n2)


def stderrofdiff(n1, n2):
    return np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))


def fisher(r):
    """

    Parameters
    ----------
    r

    Returns
    -------

    """
    return 0.5 * np.log((1 + r) / (1 - r))


def permute_phase(time_series):
    # Compute the Fourier transform of the time series
    freq_domain = np.fft.rfft(time_series)

    # Randomly permute the phase of the Fourier coefficients
    phase = 2.0 * np.pi * (np.random.random(len(freq_domain)) - 0.5)
    freq_domain = np.abs(freq_domain) * np.exp(1j * phase)

    # Compute the inverse Fourier transform to get the permuted time series
    permuted_time_series = np.fft.irfft(freq_domain)

    return permuted_time_series


def skewnessstats(timecourse):
    """

    Parameters
    ----------
    timecourse: array
        The timecourse to test

    :return:

    """
    testres = skewtest(timecourse)
    return skew(timecourse), testres[0], testres[1]


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


def fast_ICC_rep_anova(Y, nocache=False, debug=False):
    """
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]

    This is a hacked up (but fully compatible) version of ICC_rep_anova
    from nipype that caches some very expensive operations that depend
    only on the input array shape - if you're going to run the routine
    multiple times (like, on every voxel of an image), this gives you a
    HUGE speed boost for large input arrays.  If you change the dimensions
    of Y, it will reinitialize automatically.  Set nocache to True to get
    the original, much slower behavior.  No, actually, don't do that. That would
    be silly.
    """
    global icc_inited
    global current_Y_shape
    global dfc, dfe, dfr
    global nb_subjects, nb_conditions
    global x, x0, X
    global centerbit

    try:
        current_Y_shape
        if nocache or (current_Y_shape != Y.shape):
            icc_inited = False
    except NameError:
        icc_inited = False

    if not icc_inited:
        [nb_subjects, nb_conditions] = Y.shape
        if debug:
            print(
                f"fast_ICC_rep_anova inited with nb_subjects = {nb_subjects}, nb_conditions = {nb_conditions}"
            )
        current_Y_shape = Y.shape
        dfc = nb_conditions - 1
        dfe = (nb_subjects - 1) * dfc
        dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    if not icc_inited:
        x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
        x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
        X = np.hstack([x, x0])
        centerbit = np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T)

    # Sum Square Error
    predicted_Y = np.dot(centerbit, Y.flatten("F"))
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals**2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between columns/sessions
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    ICC = np.nan_to_num((MSR - MSE) / (MSR + dfc * MSE))

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    icc_inited = True

    return ICC, r_var, e_var, session_effect_F, dfc, dfe


# --------------------------- histogram functions -------------------------------------------------
def gethistprops(
    indata,
    histlen,
    refine=False,
    therange=None,
    pickleft=False,
    peakthresh=0.33,
):
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
        while i < len(thestore[1, :] - 3) and not finished:
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


def prochistogram(
    thehist,
    refine=False,
    pickleft=False,
    peakthresh=0.33,
    ignorefirstpoint=False,
    debug=False,
):
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    histlen = len(thehist[1])
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]

    # get starting values for the peak, ignoring first and last point of histogram
    if ignorefirstpoint:
        xvals = thestore[0, 1:]
        yvals = thestore[1, 1:]
        histlen -= 1
    else:
        xvals = thestore[0, :]
        yvals = thestore[1, :]
    if pickleft:
        overallmax = np.max(yvals[1:-2])
        peakindex = 1
        i = 1
        started = False
        finished = False
        while i < len(yvals - 2) and not finished:
            if yvals[i] > peakthresh * overallmax:
                started = True
            if yvals[i] > yvals[peakindex]:
                peakindex = i
            if started and (yvals[i] < 0.75 * yvals[peakindex]):
                finished = True
            i += 1
    else:
        peakindex = np.argmax(yvals[1:-2]) + 1
    peakloc = xvals[peakindex]
    peakheight = yvals[peakindex]
    numbins = 1
    while (peakindex + numbins < histlen - 2) and (yvals[peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (xvals[peakindex + numbins] - xvals[peakindex]) * 2.0
    if debug:
        print(f"{xvals=}")
        print(f"{yvals=}")
        print("Before refine")
        print(f"{peakindex=}, {peakloc=}, {peakheight=}, {peakwidth=}")
    if refine:
        peakheight, peakloc, peakwidth = tide_fit.gaussfit(
            peakheight, peakloc, peakwidth, xvals, yvals
        )
    if debug:
        print("After refine")
        print(f"{peakindex=}, {peakloc=}, {peakheight=}, {peakwidth=}")
    centerofmass = np.sum(xvals * yvals) / np.sum(yvals)
    return peakheight, peakloc, peakwidth, centerofmass


def percentilefromloc(indata, peakloc, nozero=False):
    order = indata.argsort()
    orderedvalues = indata[order]
    if nozero:
        orderedvalues = orderedvalues[np.where(orderedvalues != 0.0)]
    peaklocindex = np.argmax(orderedvalues >= peakloc)
    thepercentile = 100.0 * peaklocindex / len(orderedvalues)
    return thepercentile


def makehistogram(
    indata,
    histlen,
    binsize=None,
    therange=None,
    pickleft=False,
    peakthresh=0.33,
    refine=False,
    normalize=False,
    ignorefirstpoint=False,
    debug=False,
):
    """

    Parameters
    ----------
    indata
    histlen
    binsize
    therange
    refine
    normalize

    Returns
    -------

    """
    if therange is None:
        therange = [indata.min(), indata.max()]
    if histlen is None and binsize is None:
        thebins = 10
    elif binsize is not None:
        thebins = np.linspace(
            therange[0],
            therange[1],
            (therange[1] - therange[0]) / binsize + 1,
            endpoint=True,
        )
    else:
        thebins = histlen

    thehist = np.histogram(indata, thebins, therange, density=normalize)

    peakheight, peakloc, peakwidth, centerofmass = prochistogram(
        thehist,
        refine=refine,
        pickleft=pickleft,
        peakthresh=peakthresh,
        ignorefirstpoint=ignorefirstpoint,
        debug=debug,
    )
    peakpercentile = percentilefromloc(indata, peakloc, nozero=ignorefirstpoint)
    return thehist, peakheight, peakloc, peakwidth, centerofmass, peakpercentile


def echoloc(indata, histlen, startoffset=5.0):
    thehist, peakheight, peakloc, peakwidth, centerofmass, peakpercentile = makehistogram(
        indata, histlen, refine=True
    )
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]
    timestep = thestore[0, 1] - thestore[0, 0]
    startpt = np.argmax(thestore[1, :]) + int(startoffset // timestep)
    print(f"primary peak: {peakheight:.2f}, {peakloc:.2f}, {peakwidth}")
    print("startpt, startloc, timestep: {startpt}, {thestore[1, startpt]}, {timestep}")
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
    saveimfile=False,
    displaytitle="histogram",
    displayplots=False,
    refine=False,
    therange=None,
    normalize=False,
    dictvarname=None,
    thedict=None,
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
    append
    debug

    Returns
    -------

    """
    thehist, peakheight, peakloc, peakwidth, centerofmass, peakpercentile = makehistogram(
        indata,
        histlen,
        binsize=binsize,
        therange=therange,
        refine=refine,
        normalize=normalize,
    )
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thebinsizes = np.diff(thehist[1][:])
    thestore[1, :] = thehist[0][-histlen:]
    if debug:
        print(f"histlen: {len(thestore[1, :])}, sizelen: {len(thebinsizes)}")
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
    extraheaderinfo = {}
    extraheaderinfo["Description"] = displaytitle
    extraheaderinfo["centerofmass"] = centerofmass
    extraheaderinfo["peakloc"] = peakloc
    extraheaderinfo["peakwidth"] = peakwidth
    extraheaderinfo["peakheight"] = peakheight
    extraheaderinfo["peakpercentile"] = peakpercentile
    (
        extraheaderinfo["pct02"],
        extraheaderinfo["pct25"],
        extraheaderinfo["pct50"],
        extraheaderinfo["pct75"],
        extraheaderinfo["pct98"],
    ) = getfracvals(indata, [0.02, 0.25, 0.5, 0.75, 0.98], debug=debug)
    tide_io.writebidstsv(
        outname,
        np.transpose(thestore[1, :]),
        1.0 / (thestore[0, 1] - thestore[0, 0]),
        starttime=thestore[0, 0],
        columns=[varroot],
        append=append,
        extraheaderinfo=extraheaderinfo,
        debug=debug,
    )
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        plt.plot(thestore[0, : (-1 - endtrim)], thestore[1, : (-1 - endtrim)])
        for thepct in ["pct02", "pct98"]:
            plt.axvline(
                extraheaderinfo[thepct],
                color="#99ff99",
                linewidth=1.0,
                linestyle="dotted",
                label=thepct,
            )
        for thepct in ["pct25", "pct75"]:
            plt.axvline(
                extraheaderinfo[thepct],
                color="#66ff66",
                linewidth=1.0,
                linestyle="dashed",
                label=thepct,
            )
        for thepct in ["pct50"]:
            plt.axvline(
                extraheaderinfo[thepct],
                color="#33ff33",
                linewidth=1.0,
                linestyle="solid",
                label=thepct,
            )
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
        thevals.append(float(maskmat[theindex]))

    if debug:
        print(f"getfracvals: {datamat.shape=}")
        print(f"getfracvals: {maskmat.shape=}")
        print(f"getfracvals: {thefracs=}")
        print(f"getfracvals: {maxindex=}")
        print(f"getfracvals: {thevals=}")

    return thevals


def getfracvalsfromfit(histfit, thefracs):
    """

    Parameters
    ----------
    histfit
    thefracs

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
    print(f"old style threshval: {threshval:.2f}, new style threshval: {pctthresh:.2f}")
    if verbose:
        print(
            f"fracval: {pctthresh:.2f}",
            f"threshpct: {threshpct:.2f}",
            f"mask threshold: {threshval:.2f}",
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
