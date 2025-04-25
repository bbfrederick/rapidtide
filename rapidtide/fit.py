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
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.special as sps
import statsmodels.api as sm
import tqdm
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert
from scipy.stats import entropy, moment
from sklearn.linear_model import LinearRegression
from statsmodels.robust import mad

import rapidtide.util as tide_util

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


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Fitting functions -------------------------------------------------
def gaussresidualssk(p, y, x):
    """

    Parameters
    ----------
    p
    y
    x

    Returns
    -------

    """
    err = y - gausssk_eval(x, p)
    return err


def gaussskresiduals(p, y, x):
    """

    Parameters
    ----------
    p
    y
    x

    Returns
    -------

    """
    return y - gausssk_eval(x, p)


@conditionaljit()
def gaussresiduals(p, y, x):
    """

    Parameters
    ----------
    p
    y
    x

    Returns
    -------

    """
    return y - p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * p[2] * p[2]))


def trapezoidresiduals(p, y, x, toplength):
    """

    Parameters
    ----------
    p
    y
    x
    toplength

    Returns
    -------

    """
    return y - trapezoid_eval_loop(x, toplength, p)


def risetimeresiduals(p, y, x):
    """

    Parameters
    ----------
    p
    y
    x

    Returns
    -------

    """
    return y - risetime_eval_loop(x, p)


def gausssk_eval(x, p):
    """

    Parameters
    ----------
    x
    p

    Returns
    -------

    """
    t = (x - p[1]) / p[2]
    return p[0] * sp.stats.norm.pdf(t) * sp.stats.norm.cdf(p[3] * t)


# @conditionaljit()
def kaiserbessel_eval(x, p):
    """

    Parameters
    ----------
    x: array-like
        arguments to the KB function
    p: array-like
        The Kaiser-Bessel window parameters [alpha, tau] (wikipedia) or [beta, W/2] (Jackson, J. I., Meyer, C. H.,
        Nishimura, D. G. & Macovski, A. Selection of a convolution function for Fourier inversion using gridding
        [computerised tomography application]. IEEE Trans. Med. Imaging 10, 473–478 (1991))

    Returns
    -------

    """
    normfac = sps.i0(p[0] * np.sqrt(1.0 - np.square((0.0 / p[1])))) / p[1]
    sqrtargs = 1.0 - np.square((x / p[1]))
    sqrtargs[np.where(sqrtargs < 0.0)] = 0.0
    return np.where(
        np.fabs(x) <= p[1],
        sps.i0(p[0] * np.sqrt(sqrtargs)) / p[1] / normfac,
        0.0,
    )


@conditionaljit()
def gauss_eval(x, p):
    """

    Parameters
    ----------
    x
    p

    Returns
    -------

    """
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * p[2] * p[2]))


def trapezoid_eval_loop(x, toplength, p):
    """

    Parameters
    ----------
    x
    toplength
    p

    Returns
    -------

    """
    r = np.zeros(len(x), dtype="float64")
    for i in range(0, len(x)):
        r[i] = trapezoid_eval(x[i], toplength, p)
    return r


def risetime_eval_loop(x, p):
    """

    Parameters
    ----------
    x
    p

    Returns
    -------

    """
    r = np.zeros(len(x), dtype="float64")
    for i in range(0, len(x)):
        r[i] = risetime_eval(x[i], p)
    return r


@conditionaljit()
def trapezoid_eval(x, toplength, p):
    """

    Parameters
    ----------
    x
    toplength
    p

    Returns
    -------

    """
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    elif 0.0 <= corrx < toplength:
        return p[1] * (1.0 - np.exp(-corrx / p[2]))
    else:
        return p[1] * (np.exp(-(corrx - toplength) / p[3]))


@conditionaljit()
def risetime_eval(x, p):
    """

    Parameters
    ----------
    x
    p

    Returns
    -------

    """
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    else:
        return p[1] * (1.0 - np.exp(-corrx / p[2]))


def gasboxcar(
    data,
    samplerate,
    firstpeakstart,
    firstpeakend,
    secondpeakstart,
    secondpeakend,
    risetime=3.0,
    falltime=3.0,
):
    return None


# generate the polynomial fit timecourse from the coefficients
@conditionaljit()
def trendgen(thexvals, thefitcoffs, demean):
    """

    Parameters
    ----------
    thexvals
    thefitcoffs
    demean

    Returns
    -------

    """
    theshape = thefitcoffs.shape
    order = theshape[0] - 1
    thepoly = thexvals
    thefit = 0.0 * thexvals
    if order > 0:
        for i in range(1, order + 1):
            thefit += thefitcoffs[order - i] * thepoly
            thepoly = np.multiply(thepoly, thexvals)
    if demean:
        thefit = thefit + thefitcoffs[order]
    return thefit


# @conditionaljit()
def detrend(inputdata, order=1, demean=False):
    """

    Parameters
    ----------
    inputdata
    order
    demean

    Returns
    -------

    """
    thetimepoints = np.arange(0.0, len(inputdata), 1.0) - len(inputdata) / 2.0
    try:
        thecoffs = Polynomial.fit(thetimepoints, inputdata, order).convert().coef[::-1]
    except np.lib.polynomial.RankWarning:
        thecoffs = [0.0, 0.0]
    thefittc = trendgen(thetimepoints, thecoffs, demean)
    return inputdata - thefittc


@conditionaljit()
def findfirstabove(theyvals, thevalue):
    """

    Parameters
    ----------
    theyvals
    thevalue

    Returns
    -------

    """
    for i in range(0, len(theyvals)):
        if theyvals[i] >= thevalue:
            return i
    return len(theyvals)


def findtrapezoidfunc(
    thexvals,
    theyvals,
    thetoplength,
    initguess=None,
    debug=False,
    minrise=0.0,
    maxrise=200.0,
    minfall=0.0,
    maxfall=200.0,
    minstart=-100.0,
    maxstart=100.0,
    refine=False,
    displayplots=False,
):
    """

    Parameters
    ----------
    thexvals
    theyvals
    thetoplength
    initguess
    debug
    minrise
    maxrise
    minfall
    maxfall
    minstart
    maxstart
    refine
    displayplots

    Returns
    -------

    """
    # guess at parameters: risestart, riseamplitude, risetime
    if initguess is None:
        initstart = 0.0
        initamp = np.mean(theyvals[-10:-1])
        initrisetime = 5.0
        initfalltime = 5.0
    else:
        initstart = initguess[0]
        initamp = initguess[1]
        initrisetime = initguess[2]
        initfalltime = initguess[3]

    p0 = np.array([initstart, initamp, initrisetime, initfalltime])
    if debug:
        for i in range(0, len(theyvals)):
            print(thexvals[i], theyvals[i])
    plsq, dummy = sp.optimize.leastsq(
        trapezoidresiduals, p0, args=(theyvals, thexvals, thetoplength), maxfev=5000
    )
    # except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (
        (minrise <= plsq[2] <= maxrise)
        and (minfall <= plsq[3] <= maxfall)
        and (minstart <= plsq[0] <= maxstart)
    ):
        return plsq[0], plsq[1], plsq[2], plsq[3], 1
    else:
        return 0.0, 0.0, 0.0, 0.0, 0


def findrisetimefunc(
    thexvals,
    theyvals,
    initguess=None,
    debug=False,
    minrise=0.0,
    maxrise=200.0,
    minstart=-100.0,
    maxstart=100.0,
    refine=False,
    displayplots=False,
):
    """

    Parameters
    ----------
    thexvals
    theyvals
    initguess
    debug
    minrise
    maxrise
    minstart
    maxstart
    refine
    displayplots

    Returns
    -------

    """
    # guess at parameters: risestart, riseamplitude, risetime
    if initguess is None:
        initstart = 0.0
        initamp = np.mean(theyvals[-10:-1])
        initrisetime = 5.0
    else:
        initstart = initguess[0]
        initamp = initguess[1]
        initrisetime = initguess[2]

    p0 = np.array([initstart, initamp, initrisetime])
    if debug:
        for i in range(0, len(theyvals)):
            print(thexvals[i], theyvals[i])
    plsq, dummy = sp.optimize.leastsq(
        risetimeresiduals, p0, args=(theyvals, thexvals), maxfev=5000
    )
    # except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (minrise <= plsq[2] <= maxrise) and (minstart <= plsq[0] <= maxstart):
        return plsq[0], plsq[1], plsq[2], 1
    else:
        return 0.0, 0.0, 0.0, 0


def territorydecomp(
    inputmap, template, atlas, inputmask=None, intercept=True, fitorder=1, debug=False
):
    """

    Parameters
    ----------
    inputmap
    atlas
    inputmask
    fitorder
    debug

    Returns
    -------

    """
    datadims = len(inputmap.shape)
    if datadims > 3:
        nummaps = inputmap.shape[3]
    else:
        nummaps = 1

    if nummaps > 1:
        if inputmask is None:
            inputmask = inputmap[:, :, :, 0] * 0.0 + 1.0
    else:
        if inputmask is None:
            inputmask = inputmap * 0.0 + 1.0

    tempmask = np.where(inputmask > 0.0, 1, 0)
    maskdims = len(tempmask.shape)
    if maskdims > 3:
        nummasks = tempmask.shape[3]
    else:
        nummasks = 1

    fitmap = inputmap * 0.0

    if intercept:
        thecoffs = np.zeros((nummaps, np.max(atlas), fitorder + 1))
    else:
        thecoffs = np.zeros((nummaps, np.max(atlas), fitorder))
    if debug:
        print(f"thecoffs.shape: {thecoffs.shape}")
        print(f"intercept: {intercept}, fitorder: {fitorder}")
    theR2s = np.zeros((nummaps, np.max(atlas)))
    for whichmap in range(nummaps):
        if nummaps == 1:
            thismap = inputmap
            thisfit = fitmap
        else:
            thismap = inputmap[:, :, :, whichmap]
            thisfit = fitmap[:, :, :, whichmap]
        if nummasks == 1:
            thismask = tempmask
        else:
            thismask = tempmask[:, :, :, whichmap]
        if nummaps > 1:
            print(f"decomposing map {whichmap + 1} of {nummaps}")
        for i in range(1, np.max(atlas) + 1):
            if debug:
                print("fitting territory", i)
            maskedvoxels = np.where(atlas * thismask == i)
            if len(maskedvoxels) > 0:
                if fitorder > 0:
                    evs = []
                    for order in range(1, fitorder + 1):
                        evs.append(np.power(template[maskedvoxels], order))
                    thefit, R2 = mlregress(evs, thismap[maskedvoxels], intercept=intercept)
                    thecoffs[whichmap, i - 1, :] = np.asarray(thefit[0]).reshape((-1))
                    theR2s[whichmap, i - 1] = 1.0 * R2
                    thisfit[maskedvoxels] = mlproject(thecoffs[whichmap, i - 1, :], evs, intercept)
                else:
                    thecoffs[whichmap, i - 1, 0] = np.mean(thismap[maskedvoxels])
                    theR2s[whichmap, i - 1] = 1.0
                    thisfit[maskedvoxels] = np.mean(thismap[maskedvoxels])

    return fitmap, thecoffs, theR2s


def territorystats(
    inputmap, atlas, inputmask=None, entropybins=101, entropyrange=None, debug=False
):
    """

    Parameters
    ----------
    inputmap
    atlas
    inputmask
    debug

    Returns
    -------

    """
    datadims = len(inputmap.shape)
    if datadims > 3:
        nummaps = inputmap.shape[3]
    else:
        nummaps = 1

    if nummaps > 1:
        if inputmask is None:
            inputmask = inputmap[:, :, :, 0] * 0.0 + 1.0
    else:
        if inputmask is None:
            inputmask = inputmap * 0.0 + 1.0

    tempmask = np.where(inputmask > 0.0, 1, 0)
    maskdims = len(tempmask.shape)
    if maskdims > 3:
        nummasks = tempmask.shape[3]
    else:
        nummasks = 1

    statsmap = inputmap * 0.0

    themeans = np.zeros((nummaps, np.max(atlas)))
    thestds = np.zeros((nummaps, np.max(atlas)))
    themedians = np.zeros((nummaps, np.max(atlas)))
    themads = np.zeros((nummaps, np.max(atlas)))
    thevariances = np.zeros((nummaps, np.max(atlas)))
    theskewnesses = np.zeros((nummaps, np.max(atlas)))
    thekurtoses = np.zeros((nummaps, np.max(atlas)))
    theentropies = np.zeros((nummaps, np.max(atlas)))
    if entropyrange is None:
        if inputmask is not None:
            thevoxels = inputmap[np.where(inputmask > 0.0)]
        else:
            thevoxels = inputmap
        entropyrange = [np.min(thevoxels), np.max(thevoxels)]
    if debug:
        print(f"entropy bins: {entropybins}")
        print(f"entropy range: {entropyrange}")
        print(f"themeans.shape: {themeans.shape}")
    for whichmap in range(nummaps):
        if nummaps == 1:
            thismap = inputmap
            thisstats = statsmap
        else:
            thismap = inputmap[:, :, :, whichmap]
            thisstats = statsmap[:, :, :, whichmap]
        if nummasks == 1:
            thismask = tempmask
        else:
            thismask = tempmask[:, :, :, whichmap]
        if nummaps > 1:
            print(f"calculating stats for map {whichmap + 1} of {nummaps}")
        for i in range(1, np.max(atlas) + 1):
            if debug:
                print("calculating stats for territory", i)
            maskedvoxels = np.where(atlas * thismask == i)
            if len(maskedvoxels) > 0:
                themeans[whichmap, i - 1] = np.mean(thismap[maskedvoxels])
                thestds[whichmap, i - 1] = np.std(thismap[maskedvoxels])
                themedians[whichmap, i - 1] = np.median(thismap[maskedvoxels])
                themads[whichmap, i - 1] = mad(thismap[maskedvoxels])
                thevariances[whichmap, i - 1] = moment(thismap[maskedvoxels], moment=2)
                theskewnesses[whichmap, i - 1] = moment(thismap[maskedvoxels], moment=3)
                thekurtoses[whichmap, i - 1] = moment(thismap[maskedvoxels], moment=4)
                theentropies[whichmap, i - 1] = entropy(
                    np.histogram(
                        thismap[maskedvoxels], bins=entropybins, range=entropyrange, density=True
                    )[0]
                )

    return (
        statsmap,
        themeans,
        thestds,
        themedians,
        themads,
        thevariances,
        theskewnesses,
        thekurtoses,
        theentropies,
    )


@conditionaljit()
def refinepeak_quad(x, y, peakindex, stride=1):
    # first make sure this actually is a peak
    ismax = None
    badfit = False
    if peakindex < stride - 1 or peakindex > len(x) - 1 - stride:
        print("cannot estimate peak location at end points")
        return 0.0, 0.0, 0.0, None, True
    if y[peakindex] >= y[peakindex - stride] and y[peakindex] >= y[peakindex + stride]:
        ismax = True
    elif y[peakindex] <= y[peakindex - stride] and y[peakindex] <= y[peakindex + stride]:
        ismax = False
    else:
        badfit = True
    if y[peakindex] == y[peakindex - stride] and y[peakindex] == y[peakindex + stride]:
        badfit = True

    # now find all the information about the peak
    alpha = y[peakindex - stride]
    beta = y[peakindex]
    gamma = y[peakindex + stride]
    binsize = x[peakindex + stride] - x[peakindex]
    offsetbins = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma)
    peakloc = x[peakindex] + offsetbins * binsize
    peakval = beta - 0.25 * (alpha - gamma) * offsetbins
    a = np.square(x[peakindex - stride] - peakloc) / (alpha - peakval)
    peakwidth = np.sqrt(np.fabs(a) / 2.0)
    return peakloc, peakval, peakwidth, ismax, badfit


@conditionaljit2()
def findmaxlag_gauss(
    thexcorr_x,
    thexcorr_y,
    lagmin,
    lagmax,
    widthmax,
    edgebufferfrac=0.0,
    threshval=0.0,
    uthreshval=30.0,
    debug=False,
    tweaklims=True,
    zerooutbadfit=True,
    refine=False,
    maxguess=0.0,
    useguess=False,
    searchfrac=0.5,
    fastgauss=False,
    lagmod=1000.0,
    enforcethresh=True,
    absmaxsigma=1000.0,
    absminsigma=0.1,
    displayplots=False,
):
    """

    Parameters
    ----------
    thexcorr_x
    thexcorr_y
    lagmin
    lagmax
    widthmax
    edgebufferfrac
    threshval
    uthreshval
    debug
    tweaklims
    zerooutbadfit
    refine
    maxguess
    useguess
    searchfrac
    fastgauss
    lagmod
    enforcethresh
    displayplots

    Returns
    -------

    """
    # set initial parameters
    # widthmax is in seconds
    # maxsigma is in Hz
    # maxlag is in seconds
    warnings.filterwarnings("ignore", "Number*")
    failreason = np.uint16(0)
    maxlag = np.float64(0.0)
    maxval = np.float64(0.0)
    maxsigma = np.float64(0.0)
    maskval = np.uint16(1)
    numlagbins = len(thexcorr_y)
    binwidth = thexcorr_x[1] - thexcorr_x[0]
    searchbins = int(widthmax // binwidth)
    lowerlim = int(numlagbins * edgebufferfrac)
    upperlim = numlagbins - lowerlim - 1
    if tweaklims:
        lowerlim = 0
        upperlim = numlagbins - 1
        while (thexcorr_y[lowerlim + 1] < thexcorr_y[lowerlim]) and (lowerlim + 1) < upperlim:
            lowerlim += 1
        while (thexcorr_y[upperlim - 1] < thexcorr_y[upperlim]) and (upperlim - 1) > lowerlim:
            upperlim -= 1
    FML_BADAMPLOW = np.uint16(0x01)
    FML_BADAMPHIGH = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value
    if useguess:
        maxindex = tide_util.valtoindex(thexcorr_x, maxguess)
        nlowerlim = int(maxindex - widthmax / 2.0)
        nupperlim = int(maxindex + widthmax / 2.0)
        if nlowerlim < lowerlim:
            nlowerlim = lowerlim
            nupperlim = lowerlim + int(widthmax)
        if nupperlim > upperlim:
            nupperlim = upperlim
            nlowerlim = upperlim - int(widthmax)
        maxval_init = thexcorr_y[maxindex].astype("float64")
    else:
        maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype("int32")
        maxval_init = thexcorr_y[maxindex].astype("float64")

    # now get a location for that value
    maxlag_init = (1.0 * thexcorr_x[maxindex]).astype("float64")

    # and calculate the width of the peak
    upperlimit = len(thexcorr_y) - 1
    lowerlimit = 0
    i = 0
    j = 0
    while (
        (maxindex + i <= upperlimit)
        and (thexcorr_y[maxindex + i] > searchfrac * maxval_init)
        and (i < searchbins)
    ):
        i += 1
    if (maxindex + i > upperlimit) or (i > searchbins):
        i -= 1
    while (
        (maxindex - j >= lowerlimit)
        and (thexcorr_y[maxindex - j] > searchfrac * maxval_init)
        and (j < searchbins)
    ):
        j += 1
    if (maxindex - j < lowerlimit) or (j > searchbins):
        j -= 1
    # This is calculated from first principles, but it's always big by a factor or ~1.4.
    #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
    maxsigma_init = np.float64(
        ((i + j + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0)
    )

    # now check the values for errors and refine if necessary
    fitend = min(maxindex + i + 1, upperlimit)
    fitstart = max(1, maxindex - j)

    if not (lagmin <= maxlag_init <= lagmax):
        failreason += FML_HITEDGE
        if maxlag_init <= lagmin:
            maxlag_init = lagmin
        else:
            maxlag_init = lagmax

    if i + j + 1 < 3:
        failreason += FML_BADSEARCHWINDOW
        maxsigma_init = np.float64(
            (3.0 * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0)
        )
    if maxsigma_init > widthmax:
        failreason += FML_BADWIDTH
        maxsigma_init = widthmax
    if (maxval_init < threshval) and enforcethresh:
        failreason += FML_BADAMPLOW
    if maxval_init < 0.0:
        failreason += FML_BADAMPLOW
        maxval_init = 0.0
    if maxval_init > 1.0:
        failreason |= FML_BADAMPHIGH
        maxval_init = 1.0
    if failreason > 0:
        maskval = np.uint16(0)
    if failreason > 0 and zerooutbadfit:
        maxval = np.float64(0.0)
        maxlag = np.float64(0.0)
        maxsigma = np.float64(0.0)
    else:
        if refine:
            data = thexcorr_y[fitstart:fitend]
            X = thexcorr_x[fitstart:fitend]
            if fastgauss:
                # do a non-iterative fit over the top of the peak
                # 6/12/2015  This is just broken.  Gives quantized maxima
                maxlag = np.float64(1.0 * sum(X * data) / sum(data))
                # maxsigma = np.sqrt(abs(np.square(sum((X - maxlag)) * data) / sum(data)))
                maxsigma = np.float64(
                    np.sqrt(np.fabs(np.sum((X - maxlag) ** 2 * data) / np.sum(data)))
                )
                maxval = np.float64(data.max())
            else:
                # do a least squares fit over the top of the peak
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype="float64")

                if fitend - fitstart >= 3:
                    plsq, dummy = sp.optimize.leastsq(
                        gaussresiduals, p0, args=(data, X), maxfev=5000
                    )
                    maxval = plsq[0]
                    maxlag = np.fmod((1.0 * plsq[1]), lagmod)
                    maxsigma = plsq[2]
                # if maxval > 1.0, fit failed catastrophically, zero out or reset to initial value
                #     corrected logic for 1.1.6
                if (np.fabs(maxval)) > 1.0 or (lagmin > maxlag) or (maxlag > lagmax):
                    if zerooutbadfit:
                        maxval = np.float64(0.0)
                        maxlag = np.float64(0.0)
                        maxsigma = np.float64(0.0)
                        maskval = np.int16(0)
                    else:
                        maxval = np.float64(maxval_init)
                        maxlag = np.float64(maxlag_init)
                        maxsigma = np.float64(maxsigma_init)
                if not absminsigma <= maxsigma <= absmaxsigma:
                    if zerooutbadfit:
                        maxval = np.float64(0.0)
                        maxlag = np.float64(0.0)
                        maxsigma = np.float64(0.0)
                        maskval = np.int16(0)
                    else:
                        if maxsigma > absmaxsigma:
                            maxsigma = absmaxsigma
                        else:
                            maxsigma = absminsigma

        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(np.fmod(maxlag_init, lagmod))
            maxsigma = np.float64(maxsigma_init)
        if maxval == 0.0:
            failreason += FML_FITFAIL
        if not (lagmin <= maxlag <= lagmax):
            failreason += FML_BADLAG
        if failreason > 0:
            maskval = np.uint16(0)
        if failreason > 0 and zerooutbadfit:
            maxval = np.float64(0.0)
            maxlag = np.float64(0.0)
            maxsigma = np.float64(0.0)
    if debug or displayplots:
        print(
            "init to final: maxval",
            maxval_init,
            maxval,
            ", maxlag:",
            maxlag_init,
            maxlag,
            ", width:",
            maxsigma_init,
            maxsigma,
        )
    if displayplots and refine and (maskval != 0.0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Data and fit")
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        plt.plot(
            X,
            data,
            "ro",
            hiresx,
            gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])),
            "b-",
        )
        plt.show()
    return maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend


@conditionaljit2()
def maxindex_noedge(thexcorr_x, thexcorr_y, bipolar=False):
    """

    Parameters
    ----------
    thexcorr_x
    thexcorr_y
    bipolar

    Returns
    -------

    """
    lowerlim = 0
    upperlim = len(thexcorr_x) - 1
    done = False
    while not done:
        flipfac = 1.0
        done = True
        maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype("int32")
        if bipolar:
            minindex = (np.argmax(np.fabs(thexcorr_y[lowerlim:upperlim])) + lowerlim).astype(
                "int32"
            )
            if np.fabs(thexcorr_y[minindex]) > np.fabs(thexcorr_y[maxindex]):
                maxindex = minindex
                flipfac = -1.0
        else:
            maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype("int32")
        if upperlim == lowerlim:
            done = True
        if maxindex == 0:
            lowerlim += 1
            done = False
        if maxindex == upperlim:
            upperlim -= 1
            done = False
    return maxindex, flipfac


def gaussfitsk(height, loc, width, skewness, xvals, yvals):
    """

    Parameters
    ----------
    height
    loc
    width
    skewness
    xvals
    yvals

    Returns
    -------

    """
    plsq, dummy = sp.optimize.leastsq(
        gaussresidualssk,
        np.array([height, loc, width, skewness]),
        args=(yvals, xvals),
        maxfev=5000,
    )
    return plsq


def gaussfunc(x, height, loc, FWHM):
    return height * np.exp(-((x - loc) ** 2) / (2 * (FWHM / 2.355) ** 2))


def gaussfit2(height, loc, width, xvals, yvals):
    popt, pcov = curve_fit(gaussfunc, xvals, yvals, p0=[height, loc, width])
    return popt[0], popt[1], popt[2]


def sincfunc(x, height, loc, FWHM, baseline):
    return height * np.sinc((3.79098852 / (FWHM * np.pi)) * (x - loc)) + baseline


# found this sinc fitting routine (and optimization) here:
# https://stackoverflow.com/questions/49676116/why-cant-scipy-optimize-curve-fit-fit-my-data-using-a-numpy-sinc-function
def sincfit(height, loc, width, baseline, xvals, yvals):
    popt, pcov = curve_fit(sincfunc, xvals, yvals, p0=[height, loc, width, baseline])
    return popt, pcov


def gaussfit(height, loc, width, xvals, yvals):
    """

    Parameters
    ----------
    height
    loc
    width
    xvals
    yvals

    Returns
    -------

    """
    plsq, dummy = sp.optimize.leastsq(
        gaussresiduals, np.array([height, loc, width]), args=(yvals, xvals), maxfev=5000
    )
    return plsq[0], plsq[1], plsq[2]


def gram_schmidt(theregressors, debug=False):
    if debug:
        print("gram_schmidt, input dimensions:", theregressors.shape)
    basis = []
    for i in range(theregressors.shape[0]):
        w = theregressors[i, :] - np.sum(np.dot(theregressors[i, :], b) * b for b in basis)
        if (np.fabs(w) > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    outputbasis = np.array(basis)
    if debug:
        print("gram_schmidt, output dimensions:", outputbasis.shape)
    return outputbasis


def mlproject(thefit, theevs, intercept):
    thedest = theevs[0] * 0.0
    if intercept:
        thedest[:] = thefit[0]
        startpt = 1
    else:
        startpt = 0
    for i in range(len(thefit) - 1):
        thedest += thefit[i + startpt] * theevs[i]
    return thedest


def olsregress(X, y, intercept=True, debug=False):
    """

    Parameters
    ----------
    X
    y
    intercept

    Returns
    -------

    """
    """Return the coefficients from a multiple linear regression, along with R, the coefficient of determination.

    X: The independent variables (nxp).
    y: The dependent variable (1xn or nx1).
    intercept: Specifies whether or not the slope intercept should be considered.

    The routine computes the coefficients (b_0, b_1, ..., b_p) from the data (x,y) under
    the assumption that y = b0 + b_1 * x_1 + b_2 * x_2 + ... + b_p * x_p.

    If intercept is False, the routine assumes that b0 = 0 and returns (b_1, b_2, ..., b_p).
    """
    if intercept:
        X = sm.add_constant(X, prepend=True)
    model = sm.OLS(y, exog=X)
    thefit = model.fit()
    return thefit.params, np.sqrt(thefit.rsquared)


def mlregress(X, y, intercept=True, debug=False):
    """

    Parameters
    ----------
    x
    y
    intercept

    Returns
    -------

    """
    """Return the coefficients from a multiple linear regression, along with R, the coefficient of determination.

    x: The independent variables (pxn or nxp).
    y: The dependent variable (1xn or nx1).
    intercept: Specifies whether or not the slope intercept should be considered.

    The routine computes the coefficients (b_0, b_1, ..., b_p) from the data (x,y) under
    the assumption that y = b0 + b_1 * x_1 + b_2 * x_2 + ... + b_p * x_p.

    If intercept is False, the routine assumes that b0 = 0 and returns (b_1, b_2, ..., b_p).
    """
    if debug:
        print(f"mlregress initial: {X.shape=}, {y.shape=}")
    y = np.atleast_1d(y)
    n = y.shape[0]

    X = np.atleast_2d(X)
    nx, p = X.shape

    if debug:
        print(f"mlregress: {n=}, {p=}, {nx=}")

    if nx != n:
        X = X.transpose()
        nx, p = X.shape
        if nx != n:
            raise AttributeError(
                "X and y must have have the same number of samples (%d and %d)" % (nx, n)
            )
    if debug:
        print(f"mlregress final: {X.shape=}, {y.shape=}")

    reg = LinearRegression(fit_intercept=intercept)
    reg.fit(X, y)
    coffs = reg.coef_
    theintercept = reg.intercept_
    R2 = reg.score(X, y)
    coffs = np.insert(coffs, 0, theintercept, axis=0)
    return np.asmatrix(coffs), R2


def calcexpandedregressors(
    confounddict, labels=None, start=0, end=-1, deriv=True, order=1, debug=False
):
    r"""Calculates various motion related timecourses from motion data dict, and returns an array

    Parameters
    ----------
    confounddict: dict
        A dictionary of the confound vectors

    Returns
    -------
    motionregressors: array
        All the derivative timecourses to use in a numpy array

    """
    if labels is None:
        localconfounddict = confounddict.copy()
        labels = list(localconfounddict.keys())
    else:
        localconfounddict = {}
        for label in labels:
            localconfounddict[label] = confounddict[label]
    if order > 1:
        for theorder in range(1, order):
            for thelabel in labels:
                localconfounddict[f"{thelabel}^{theorder+1}"] = (localconfounddict[thelabel]) ** (
                    theorder + 1
                )
        labels = list(localconfounddict.keys())
        if debug:
            print(f"{labels=}")

    numkeys = len(labels)
    numpoints = len(localconfounddict[labels[0]])
    if end == -1:
        end = numpoints - 1
    if (0 <= start <= numpoints - 1) and (start < end + 1):
        numoutputpoints = end - start + 1

    numoutputregressors = numkeys
    if deriv:
        numoutputregressors += numkeys
    if numoutputregressors > 0:
        outputregressors = np.zeros((numoutputregressors, numoutputpoints), dtype=float)
    else:
        print("no output types selected - exiting")
        sys.exit()
    activecolumn = 0
    outlabels = []
    for thelabel in labels:
        outputregressors[activecolumn, :] = localconfounddict[thelabel][start : end + 1]
        outlabels.append(thelabel)
        activecolumn += 1
    if deriv:
        for thelabel in labels:
            outputregressors[activecolumn, :] = np.gradient(
                localconfounddict[thelabel][start : end + 1]
            )
            outlabels.append(thelabel + "_deriv")
            activecolumn += 1
    return outputregressors, outlabels


def derivativelinfitfilt(thedata, theevs, nderivs=1, debug=False):
    r"""First perform multicomponent expansion on theevs (each ev replaced by itself,
    its square, its cube, etc.).  Then perform a linear fit of thedata using the vectors
    in thenewevs and return the result.

    Parameters
    ----------
    thedata : 1D numpy array
        Input data of length N to be filtered
        :param thedata:

    theevs : 2D numpy array
        NxP array of explanatory variables to be fit
        :param theevs:

    nderivs : integer
        Number of components to use for each ev.  Each successive component is a
        higher power of the initial ev (initial, square, cube, etc.)
        :param nderivs:

    debug: bool
        Flag to toggle debugging output
        :param debug:
    """
    if debug:
        print(f"{thedata.shape=}")
        print(f"{theevs.shape=}")
    if nderivs == 0:
        thenewevs = theevs
    else:
        if theevs.ndim > 1:
            thenewevs = np.zeros((theevs.shape[0], theevs.shape[1] * (nderivs + 1)), dtype=float)
            for ev in range(0, theevs.shape[1] - 1):
                thenewevs[:, nderivs * ev] = theevs[:, ev] * 1.0
                for i in range(1, nderivs + 1):
                    thenewevs[:, nderivs * (ev - 1) + i] = np.gradient(
                        thenewevs[:, nderivs * (ev - 1) + i - 1]
                    )
        else:
            thenewevs = np.zeros((theevs.shape[0], nderivs + 1), dtype=float)
            thenewevs[:, 0] = theevs * 1.0
            for i in range(1, nderivs + 1):
                thenewevs[:, i] = np.gradient(thenewevs[:, i - 1])
    if debug:
        print(f"{nderivs=}")
        print(f"{thenewevs.shape=}")
    filtered, datatoremove, R, coffs = linfitfilt(thedata, thenewevs, debug=debug)
    if debug:
        print(f"{R=}")

    return filtered, thenewevs, datatoremove, R, coffs


def expandedlinfitfilt(thedata, theevs, ncomps=1, debug=False):
    r"""First perform multicomponent expansion on theevs (each ev replaced by itself,
    its square, its cube, etc.).  Then perform a multiple regression fit of thedata using the vectors
    in thenewevs and return the result.

    Parameters
    ----------
    thedata : 1D numpy array
        Input data of length N to be filtered
        :param thedata:

    theevs : 2D numpy array
        NxP array of explanatory variables to be fit
        :param theevs:

    ncomps : integer
        Number of components to use for each ev.  Each successive component is a
        higher power of the initial ev (initial, square, cube, etc.)
        :param ncomps:

    debug: bool
        Flag to toggle debugging output
        :param debug:
    """
    if debug:
        print(f"{thedata.shape=}")
        print(f"{theevs.shape=}")
    if ncomps == 1:
        thenewevs = theevs
    else:
        if theevs.ndim > 1:
            thenewevs = np.zeros((theevs.shape[0], theevs.shape[1] * ncomps), dtype=float)
            for ev in range(1, theevs.shape[1]):
                thenewevs[:, ncomps * (ev - 1)] = theevs[:, ev - 1] * 1.0
                for i in range(1, ncomps):
                    thenewevs[:, ncomps * (ev - 1) + i] = (
                        thenewevs[:, ncomps * (ev - 1) + i - 1] * theevs[:, ev - 1]
                    )
        else:
            thenewevs = np.zeros((theevs.shape[0], ncomps), dtype=float)
            thenewevs[:, 0] = theevs * 1.0
            for i in range(1, ncomps):
                thenewevs[:, i] = thenewevs[:, i - 1] * theevs
    if debug:
        print(f"{ncomps=}")
        print(f"{thenewevs.shape=}")
    filtered, datatoremove, R, coffs = linfitfilt(thedata, thenewevs, debug=debug)
    if debug:
        print(f"{R=}")

    return filtered, thenewevs, datatoremove, R, coffs


def linfitfilt(thedata, theevs, returnintercept=False, debug=False):
    r"""Performs a multiple regression fit of thedata using the vectors in theevs
    and returns the result.

    Parameters
    ----------
    thedata : 1D numpy array
        Input data of length N to be filtered
        :param thedata:

    theevs : 2D numpy array
        NxP array of explanatory variables to be fit
        :param theevs:

    debug: bool
        Flag to toggle debugging output
        :param debug:
    """

    if debug:
        print(f"{thedata.shape=}")
        print(f"{theevs.shape=}")
    thefit, R2 = mlregress(theevs, thedata, debug=debug)
    retcoffs = np.zeros((thefit.shape[1] - 1), dtype=float)
    if debug:
        print(f"{thefit.shape=}")
        print(f"{thefit=}")
        print(f"{R2=}")
        print(f"{retcoffs.shape=}")
    datatoremove = thedata * 0.0

    if theevs.ndim > 1:
        for ev in range(1, thefit.shape[1]):
            if debug:
                print(f"{ev=}")
            theintercept = thefit[0, 0]
            retcoffs[ev - 1] = thefit[0, ev]
            datatoremove += thefit[0, ev] * theevs[:, ev - 1]
            if debug:
                print(f"{ev=}")
                print(f"\t{thefit[0, ev]=}")
                print(f"\tdatatoremove min max = {np.min(datatoremove)}, {np.max(datatoremove)}")
    else:
        datatoremove += thefit[0, 1] * theevs[:]
        retcoffs[0] = thefit[0, 1]
    filtered = thedata - datatoremove
    if debug:
        print(f"{retcoffs=}")
    if returnintercept:
        return filtered, datatoremove, R2, retcoffs, theintercept
    else:
        return filtered, datatoremove, R2, retcoffs


def confoundregress(
    data,
    regressors,
    debug=False,
    showprogressbar=True,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    r"""Filters multiple regressors out of an array of data

    Parameters
    ----------
    data : 2d numpy array
        A data array.  First index is the spatial dimension, second is the time (filtering) dimension.

    regressors: 2d numpy array
        The set of regressors to filter out of each timecourse.  The first dimension is the regressor number, second is the time (filtering) dimension:

    debug : boolean
        Print additional diagnostic information if True

    Returns
    -------
    """
    if debug:
        print("data shape:", data.shape)
        print("regressors shape:", regressors.shape)
    datatoremove = np.zeros(data.shape[1], dtype=rt_floattype)
    filtereddata = data * 0.0
    r2value = data[:, 0] * 0.0
    for i in tqdm(
        range(data.shape[0]),
        desc="Voxel",
        unit="voxels",
        disable=(not showprogressbar),
    ):
        datatoremove *= 0.0
        thefit, R2 = mlregress(regressors, data[i, :])
        if i == 0 and debug:
            print("fit shape:", thefit.shape)
        for j in range(regressors.shape[0]):
            datatoremove += rt_floatset(rt_floatset(thefit[0, 1 + j]) * regressors[j, :])
        filtereddata[i, :] = data[i, :] - datatoremove
        r2value[i] = R2
    return filtereddata, r2value


# --------------------------- Peak detection functions ----------------------------------------------
# The following three functions are taken from the peakdetect distribution by Sixten Bergman
# They were distributed under the DWTFYWTPL, so I'm relicensing them under Apache 2.0
# From his header:
# You can redistribute it and/or modify it under the terms of the Do What The
# Fuck You Want To Public License, Version 2, as published by Sam Hocevar. See
# http://www.wtfpl.net/ for more details.
def getpeaks(xvals, yvals, xrange=None, bipolar=False, displayplots=False):
    peaks, dummy = find_peaks(yvals, height=0)
    if bipolar:
        negpeaks, dummy = find_peaks(-yvals, height=0)
        peaks = np.concatenate((peaks, negpeaks))
    procpeaks = []
    if xrange is None:
        lagmin = xvals[0]
        lagmax = xvals[-1]
    else:
        lagmin = xrange[0]
        lagmax = xrange[1]
    originloc = tide_util.valtoindex(xvals, 0.0, discrete=False)
    for thepeak in peaks:
        if lagmin <= xvals[thepeak] <= lagmax:
            if bipolar:
                procpeaks.append(
                    [
                        xvals[thepeak],
                        yvals[thepeak],
                        tide_util.valtoindex(xvals, xvals[thepeak], discrete=False) - originloc,
                    ]
                )
            else:
                if yvals[thepeak] > 0.0:
                    procpeaks.append(
                        [
                            xvals[thepeak],
                            yvals[thepeak],
                            tide_util.valtoindex(xvals, xvals[thepeak], discrete=False)
                            - originloc,
                        ]
                    )
    if displayplots:
        plotx = []
        ploty = []
        offset = []
        for thepeak in procpeaks:
            plotx.append(thepeak[0])
            ploty.append(thepeak[1])
            offset.append(thepeak[2])
        plt.plot(xvals, yvals)
        plt.plot(plotx, ploty, "x")
        plt.plot(xvals, np.zeros_like(yvals), "--", color="gray")
        plt.show()
    return procpeaks


def parabfit(x_axis, y_axis, peakloc, points):
    """

    Parameters
    ----------
    x_axis
    y_axis
    peakloc
    peaksize

    Returns
    -------

    """
    func = lambda x, a, tau, c: a * ((x - tau) ** 2) + c
    distance = abs(x_axis[peakloc[1][0]] - x_axis[peakloc[0][0]]) / 4
    index = peakloc
    x_data = x_axis[index - points // 2 : index + points // 2 + 1]
    y_data = y_axis[index - points // 2 : index + points // 2 + 1]

    # get a first approximation of tau (peak position in time)
    tau = x_axis[index]

    # get a first approximation of peak amplitude
    c = y_axis[index]
    a = np.sign(c) * (-1) * (np.sqrt(abs(c)) / distance) ** 2
    """Derived from ABC formula to result in a solution where A=(rot(c)/t)**2"""

    # build list of approximations
    p0 = (a, tau, c)
    popt, pcov = curve_fit(func, x_data, y_data, p0)

    # retrieve tau and c i.e x and y value of peak
    x, y = popt[1:3]
    return x, y


def _datacheck_peakdetect(x_axis, y_axis):
    """

    Parameters
    ----------
    x_axis
    y_axis

    Returns
    -------

    """
    if x_axis is None:
        x_axis = range(len(y_axis))

    if np.shape(y_axis) != np.shape(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0.0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)

    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200)
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value

    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = np.shape(y_axis)[0]

    # perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.inf, -np.inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.inf
                mn = np.inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
                # else:  #slows shit down this does
                #    mx = ahead
                #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn + delta and mn != -np.inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.inf
                mx = -np.inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                    # else:  #slows shit down this does
                    #    mn = ahead
                    #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def ocscreetest(eigenvals, debug=False, displayplots=False):
    num = len(eigenvals)
    a = eigenvals * 0.0
    b = eigenvals * 0.0
    prediction = eigenvals * 0.0
    for i in range(num - 3, 1, -1):
        b[i] = (eigenvals[-1] - eigenvals[i + 1]) / (num - 1 - i - 1)
        a[i] = eigenvals[i + 1] - b[i + 1]
        if debug:
            print(f"{i=}, {a[i]=}, {b[i]=}")
    retained = eigenvals[np.where(eigenvals > 1.0)]
    retainednum = len(retained)
    for i in range(1, num - 1):
        prediction[i] = a[i + 1] + b[i + 1] * i
        if debug:
            print(f"{i=}, {eigenvals[i]=}, {prediction[i]=}")
        if eigenvals[i] < retained[i]:
            break
    if displayplots:
        print("making plots")
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax1.set_title("Original")
        plt.plot(eigenvals, color="k")
        ax2 = fig.add_subplot(412)
        ax2.set_title("a")
        plt.plot(a, color="g")
        ax3 = fig.add_subplot(413)
        ax3.set_title("b")
        plt.plot(b, color="r")
        ax4 = fig.add_subplot(414)
        ax4.set_title("prediction")
        plt.plot(prediction, color="b")
        plt.show()
    return i


def afscreetest(eigenvals, displayplots=False):
    num = len(eigenvals)
    firstderiv = np.gradient(eigenvals, edge_order=2)
    secondderiv = np.gradient(firstderiv, edge_order=2)
    maxaccloc = np.argmax(secondderiv)
    if displayplots:
        print("making plots")
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("Original")
        plt.plot(eigenvals, color="k")
        ax2 = fig.add_subplot(312)
        ax2.set_title("First derivative")
        plt.plot(firstderiv, color="r")
        ax3 = fig.add_subplot(313)
        ax3.set_title("Second derivative")
        plt.plot(secondderiv, color="g")
        plt.show()
    return maxaccloc - 1


def phaseanalysis(firstharmonic, displayplots=False):
    print("entering phaseanalysis")
    analytic_signal = hilbert(firstharmonic)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    if displayplots:
        print("making plots")
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("Analytic signal")
        X = np.linspace(0.0, 1.0, num=len(firstharmonic))
        plt.plot(X, analytic_signal.real, "k", X, analytic_signal.imag, "r")
        ax2 = fig.add_subplot(312)
        ax2.set_title("Phase")
        plt.plot(X, instantaneous_phase, "g")
        ax3 = fig.add_subplot(313)
        ax3.set_title("Amplitude")
        plt.plot(X, amplitude_envelope, "b")
        plt.show()
        plt.savefig("phaseanalysistest.jpg")
    instantaneous_phase = np.unwrap(instantaneous_phase)
    return instantaneous_phase, amplitude_envelope, analytic_signal


FML_NOERROR = np.uint32(0x0000)

FML_INITAMPLOW = np.uint32(0x0001)
FML_INITAMPHIGH = np.uint32(0x0002)
FML_INITWIDTHLOW = np.uint32(0x0004)
FML_INITWIDTHHIGH = np.uint32(0x0008)
FML_INITLAGLOW = np.uint32(0x0010)
FML_INITLAGHIGH = np.uint32(0x0020)
FML_INITFAIL = (
    FML_INITAMPLOW
    | FML_INITAMPHIGH
    | FML_INITWIDTHLOW
    | FML_INITWIDTHHIGH
    | FML_INITLAGLOW
    | FML_INITLAGHIGH
)

FML_FITAMPLOW = np.uint32(0x0100)
FML_FITAMPHIGH = np.uint32(0x0200)
FML_FITWIDTHLOW = np.uint32(0x0400)
FML_FITWIDTHHIGH = np.uint32(0x0800)
FML_FITLAGLOW = np.uint32(0x1000)
FML_FITLAGHIGH = np.uint32(0x2000)
FML_FITFAIL = (
    FML_FITAMPLOW
    | FML_FITAMPHIGH
    | FML_FITWIDTHLOW
    | FML_FITWIDTHHIGH
    | FML_FITLAGLOW
    | FML_FITLAGHIGH
)


def simfuncpeakfit(
    incorrfunc,
    corrtimeaxis,
    useguess=False,
    maxguess=0.0,
    displayplots=False,
    functype="correlation",
    peakfittype="gauss",
    searchfrac=0.5,
    lagmod=1000.0,
    enforcethresh=True,
    allowhighfitamps=False,
    lagmin=-30.0,
    lagmax=30.0,
    absmaxsigma=1000.0,
    absminsigma=0.25,
    hardlimit=True,
    bipolar=False,
    lthreshval=0.0,
    uthreshval=1.0,
    zerooutbadfit=True,
    debug=False,
):
    # check to make sure xcorr_x and xcorr_y match
    if corrtimeaxis is None:
        print("Correlation time axis is not defined - exiting")
        sys.exit()
    if len(corrtimeaxis) != len(incorrfunc):
        print(
            "Correlation time axis and values do not match in length (",
            len(corrtimeaxis),
            "!=",
            len(incorrfunc),
            "- exiting",
        )
        sys.exit()
    # set initial parameters
    # absmaxsigma is in seconds
    # maxsigma is in Hz
    # maxlag is in seconds
    warnings.filterwarnings("ignore", "Number*")
    failreason = FML_NOERROR
    maskval = np.uint16(1)  # start out assuming the fit will succeed
    binwidth = corrtimeaxis[1] - corrtimeaxis[0]

    # set the search range
    lowerlim = 0
    upperlim = len(corrtimeaxis) - 1
    if debug:
        print(
            "initial search indices are",
            lowerlim,
            "to",
            upperlim,
            "(",
            corrtimeaxis[lowerlim],
            corrtimeaxis[upperlim],
            ")",
        )

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value and its location
    flipfac = 1.0
    corrfunc = incorrfunc + 0.0
    if useguess:
        maxindex = tide_util.valtoindex(corrtimeaxis, maxguess)
        if (corrfunc[maxindex] < 0.0) and bipolar:
            flipfac = -1.0
    else:
        maxindex, flipfac = _maxindex_noedge(corrfunc, corrtimeaxis, bipolar=bipolar)
    corrfunc *= flipfac
    maxlag_init = (1.0 * corrtimeaxis[maxindex]).astype("float64")
    maxval_init = corrfunc[maxindex].astype("float64")
    if debug:
        print(
            "maxindex, maxlag_init, maxval_init:",
            maxindex,
            maxlag_init,
            maxval_init,
        )

    # set the baseline and baselinedev levels
    if (functype == "correlation") or (functype == "hybrid"):
        baseline = 0.0
        baselinedev = 0.0
    else:
        # for mutual information, there is a nonzero baseline, so we want the difference from that.
        baseline = np.median(corrfunc)
        baselinedev = mad(corrfunc)
    if debug:
        print("baseline, baselinedev:", baseline, baselinedev)

    # then calculate the width of the peak
    if peakfittype == "fastquad" or peakfittype == "COM":
        peakstart = np.max([1, maxindex - 2])
        peakend = np.min([len(corrtimeaxis) - 2, maxindex + 2])
    else:
        # come here for peakfittype of None, quad, gauss, fastgauss
        thegrad = np.gradient(corrfunc).astype(
            "float64"
        )  # the gradient of the correlation function
        if (functype == "correlation") or (functype == "hybrid"):
            if peakfittype == "quad":
                peakpoints = np.where(
                    corrfunc > maxval_init - 0.05, 1, 0
                )  # mask for places where correlation exceeds searchfrac*maxval_init
            else:
                peakpoints = np.where(
                    corrfunc > (baseline + searchfrac * (maxval_init - baseline)), 1, 0
                )  # mask for places where correlation exceeds searchfrac*maxval_init
        else:
            # for mutual information, there is a flattish, nonzero baseline, so we want the difference from that.
            peakpoints = np.where(
                corrfunc > (baseline + searchfrac * (maxval_init - baseline)),
                1,
                0,
            )

        peakpoints[0] = 0
        peakpoints[-1] = 0
        peakstart = np.max([1, maxindex - 1])
        peakend = np.min([len(corrtimeaxis) - 2, maxindex + 1])
        if debug:
            print("initial peakstart, peakend:", peakstart, peakend)
        if functype == "mutualinfo":
            while peakpoints[peakend + 1] == 1:
                peakend += 1
            while peakpoints[peakstart - 1] == 1:
                peakstart -= 1
        else:
            while thegrad[peakend + 1] <= 0.0 and peakpoints[peakend + 1] == 1:
                peakend += 1
            while thegrad[peakstart - 1] >= 0.0 and peakpoints[peakstart - 1] == 1:
                peakstart -= 1
        if debug:
            print("final peakstart, peakend:", peakstart, peakend)

        # deal with flat peak top
        while peakend < (len(corrtimeaxis) - 3) and corrfunc[peakend] == corrfunc[peakend - 1]:
            peakend += 1
        while peakstart > 2 and corrfunc[peakstart] == corrfunc[peakstart + 1]:
            peakstart -= 1
        if debug:
            print("peakstart, peakend after flattop correction:", peakstart, peakend)
            print("\n")
            for i in range(peakstart, peakend + 1):
                print(corrtimeaxis[i], corrfunc[i])
            print("\n")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("Peak sent to fitting routine")
            plt.plot(
                corrtimeaxis[peakstart : peakend + 1],
                corrfunc[peakstart : peakend + 1],
                "r",
            )
            plt.show()

        # This is calculated from first principles, but it's always big by a factor or ~1.4.
        #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
        maxsigma_init = np.float64(
            ((peakend - peakstart + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac))))
            / np.sqrt(2.0)
        )
        if debug:
            print("maxsigma_init:", maxsigma_init)

        # now check the values for errors
        if hardlimit:
            rangeextension = 0.0
        else:
            rangeextension = (lagmax - lagmin) * 0.75
        if not (
            (lagmin - rangeextension - binwidth)
            <= maxlag_init
            <= (lagmax + rangeextension + binwidth)
        ):
            if maxlag_init <= (lagmin - rangeextension - binwidth):
                failreason |= FML_INITLAGLOW
                maxlag_init = lagmin - rangeextension - binwidth
            else:
                failreason |= FML_INITLAGHIGH
                maxlag_init = lagmax + rangeextension + binwidth
            if debug:
                print("bad initial")
        if maxsigma_init > absmaxsigma:
            failreason |= FML_INITWIDTHHIGH
            maxsigma_init = absmaxsigma
            if debug:
                print("bad initial width - too high")
        if peakend - peakstart < 2:
            failreason |= FML_INITWIDTHLOW
            maxsigma_init = np.float64(
                ((2 + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0)
            )
            if debug:
                print("bad initial width - too low")
        if (functype == "correlation") or (functype == "hybrid"):
            if not (lthreshval <= maxval_init <= uthreshval) and enforcethresh:
                failreason |= FML_INITAMPLOW
                if debug:
                    print(
                        "bad initial amp:",
                        maxval_init,
                        "is less than",
                        lthreshval,
                    )
            if maxval_init < 0.0:
                failreason |= FML_INITAMPLOW
                maxval_init = 0.0
                if debug:
                    print("bad initial amp:", maxval_init, "is less than 0.0")
            if maxval_init > 1.0:
                failreason |= FML_INITAMPHIGH
                maxval_init = 1.0
                if debug:
                    print("bad initial amp:", maxval_init, "is greater than 1.0")
        else:
            # somewhat different rules for mutual information peaks
            if ((maxval_init - baseline) < lthreshval * baselinedev) or (maxval_init < baseline):
                failreason |= FML_INITAMPLOW
                maxval_init = 0.0
                if debug:
                    print("bad initial amp:", maxval_init, "is less than 0.0")
        if (failreason != FML_NOERROR) and zerooutbadfit:
            maxval = np.float64(0.0)
            maxlag = np.float64(0.0)
            maxsigma = np.float64(0.0)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(maxlag_init)
            maxsigma = np.float64(maxsigma_init)

    # refine if necessary
    if peakfittype != "None":
        if peakfittype == "COM":
            X = corrtimeaxis[peakstart : peakend + 1] - baseline
            data = corrfunc[peakstart : peakend + 1]
            maxval = maxval_init
            maxlag = np.sum(X * data) / np.sum(data)
            maxsigma = 10.0
        elif peakfittype == "gauss":
            X = corrtimeaxis[peakstart : peakend + 1] - baseline
            data = corrfunc[peakstart : peakend + 1]
            # do a least squares fit over the top of the peak
            # p0 = np.array([maxval_init, np.fmod(maxlag_init, lagmod), maxsigma_init], dtype='float64')
            p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype="float64")
            if debug:
                print("fit input array:", p0)
            try:
                plsq, dummy = sp.optimize.leastsq(gaussresiduals, p0, args=(data, X), maxfev=5000)
                maxval = plsq[0] + baseline
                maxlag = np.fmod((1.0 * plsq[1]), lagmod)
                maxsigma = plsq[2]
            except:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            if debug:
                print("fit output array:", [maxval, maxlag, maxsigma])
        elif peakfittype == "fastgauss":
            X = corrtimeaxis[peakstart : peakend + 1] - baseline
            data = corrfunc[peakstart : peakend + 1]
            # do a non-iterative fit over the top of the peak
            # 6/12/2015  This is just broken.  Gives quantized maxima
            maxlag = np.float64(1.0 * np.sum(X * data) / np.sum(data))
            maxsigma = np.float64(np.sqrt(np.abs(np.sum((X - maxlag) ** 2 * data) / np.sum(data))))
            maxval = np.float64(data.max()) + baseline
        elif peakfittype == "fastquad":
            maxlag, maxval, maxsigma, ismax, badfit = refinepeak_quad(
                corrtimeaxis, corrfunc, maxindex
            )
        elif peakfittype == "quad":
            X = corrtimeaxis[peakstart : peakend + 1]
            data = corrfunc[peakstart : peakend + 1]
            try:
                thecoffs = Polynomial.fit(X, data, 2).convert().coef[::-1]
                a = thecoffs[0]
                b = thecoffs[1]
                c = thecoffs[2]
                maxlag = -b / (2.0 * a)
                maxval = a * maxlag * maxlag + b * maxlag + c
                maxsigma = 1.0 / np.fabs(a)
                if debug:
                    print("poly coffs:", a, b, c)
                    print("maxlag, maxval, maxsigma:", maxlag, maxval, maxsigma)
            except np.lib.polynomial.RankWarning:
                maxlag = 0.0
                maxval = 0.0
                maxsigma = 0.0
            if debug:
                print("\n")
                for i in range(len(X)):
                    print(X[i], data[i])
                print("\n")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Peak and fit")
                plt.plot(X, data, "r")
                plt.plot(X, c + b * X + a * X * X, "b")
                plt.show()

        else:
            print("illegal peak refinement type")

        # check for errors in fit
        fitfail = False
        if bipolar:
            lowestcorrcoeff = -1.0
        else:
            lowestcorrcoeff = 0.0
        if (functype == "correlation") or (functype == "hybrid"):
            if maxval < lowestcorrcoeff:
                failreason |= FML_FITAMPLOW
                maxval = lowestcorrcoeff
                if debug:
                    print("bad fit amp: maxval is lower than lower limit")
                fitfail = True
            if np.abs(maxval) > 1.0:
                if not allowhighfitamps:
                    failreason |= FML_FITAMPHIGH
                    if debug:
                        print(
                            "bad fit amp: magnitude of",
                            maxval,
                            "is greater than 1.0",
                        )
                    fitfail = True
                maxval = 1.0 * np.sign(maxval)
        else:
            # different rules for mutual information peaks
            if ((maxval - baseline) < lthreshval * baselinedev) or (maxval < baseline):
                failreason |= FML_FITAMPLOW
                if debug:
                    if (maxval - baseline) < lthreshval * baselinedev:
                        print(
                            "FITAMPLOW: maxval - baseline:",
                            maxval - baseline,
                            " < lthreshval * baselinedev:",
                            lthreshval * baselinedev,
                        )
                    if maxval < baseline:
                        print("FITAMPLOW: maxval < baseline:", maxval, baseline)
                maxval_init = 0.0
                if debug:
                    print("bad fit amp: maxval is lower than lower limit")
        if (lagmin > maxlag) or (maxlag > lagmax):
            if debug:
                print("bad lag after refinement")
            if lagmin > maxlag:
                failreason |= FML_FITLAGLOW
                maxlag = lagmin
            else:
                failreason |= FML_FITLAGHIGH
                maxlag = lagmax
            fitfail = True
        if maxsigma > absmaxsigma:
            failreason |= FML_FITWIDTHHIGH
            if debug:
                print("bad width after refinement:", maxsigma, ">", absmaxsigma)
            maxsigma = absmaxsigma
            fitfail = True
        if maxsigma < absminsigma:
            failreason |= FML_FITWIDTHLOW
            if debug:
                print("bad width after refinement:", maxsigma, "<", absminsigma)
            maxsigma = absminsigma
            fitfail = True
        if fitfail:
            if debug:
                print("fit fail")
            if zerooutbadfit:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            maskval = np.uint16(0)
        # print(maxlag_init, maxlag, maxval_init, maxval, maxsigma_init, maxsigma, maskval, failreason, fitfail)
    else:
        maxval = np.float64(maxval_init)
        maxlag = np.float64(np.fmod(maxlag_init, lagmod))
        maxsigma = np.float64(maxsigma_init)
        if failreason != FML_NOERROR:
            maskval = np.uint16(0)

    if debug or displayplots:
        print(
            "init to final: maxval",
            maxval_init,
            maxval,
            ", maxlag:",
            maxlag_init,
            maxlag,
            ", width:",
            maxsigma_init,
            maxsigma,
        )
    if displayplots and (peakfittype != "None") and (maskval != 0.0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Data and fit")
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        plt.plot(
            X,
            data,
            "ro",
            hiresx,
            gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])),
            "b-",
        )
        plt.show()
    return (
        maxindex,
        maxlag,
        flipfac * maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    )


def _maxindex_noedge(corrfunc, corrtimeaxis, bipolar=False):
    """

    Parameters
    ----------
    corrfunc

    Returns
    -------

    """
    lowerlim = 0
    upperlim = len(corrtimeaxis) - 1
    done = False
    while not done:
        flipfac = 1.0
        done = True
        maxindex = (np.argmax(corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
        if bipolar:
            minindex = (np.argmax(-corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
            if np.fabs(corrfunc[minindex]) > np.fabs(corrfunc[maxindex]):
                maxindex = minindex
                flipfac = -1.0
        if upperlim == lowerlim:
            done = True
        if maxindex == 0:
            lowerlim += 1
            done = False
        if maxindex == upperlim:
            upperlim -= 1
            done = False
    return maxindex, flipfac
