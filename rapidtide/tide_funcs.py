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
import rapidtide.filter as tide_filt

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


def symmetrize(a, antisymmetric=False, zerodiagonal=False):
    if antisymmetric:
        intermediate = (a - a.T) / 2.0
    else:
        intermediate = (a + a.T) / 2.0
    if zerodiagonal:
        return intermediate - np.diag(intermediate.diagonal())
    else:
        return intermediate


### I don't remember where this came from.  Need to check license
def mlregress(x, y, intercept=True):
    """Return the coefficients from a multiple linear regression, along with R, the coefficient of determination.

    x: The independent variables (pxn or nxp).
    y: The dependent variable (1xn or nx1).
    intercept: Specifies whether or not the slope intercept should be considered.

    The routine computes the coefficients (b_0, b_1, ..., b_p) from the data (x,y) under
    the assumption that y = b0 + b_1 * x_1 + b_2 * x_2 + ... + b_p * x_p.

    If intercept is False, the routine assumes that b0 = 0 and returns (b_1, b_2, ..., b_p).
    """

    warnings.filterwarnings("ignore", "invalid*")
    y = np.atleast_1d(y)
    n = y.shape[0]

    x = np.atleast_2d(x)
    p, nx = x.shape

    if nx != n:
        x = x.transpose()
        p, nx = x.shape
        if nx != n:
            raise AttributeError('x and y must have have the same number of samples (%d and %d)' % (nx, n))

    if intercept is True:
        xc = np.vstack((np.ones(n), x))
        beta = np.ones(p + 1)
    else:
        xc = x
        beta = np.ones(p)

    solution = np.linalg.lstsq(np.mat(xc).T, np.mat(y).T, rcond=-1)

    # Computation of the coefficient of determination.
    Rx = np.atleast_2d(np.corrcoef(x, rowvar=1))
    c = np.corrcoef(x, y, rowvar=1)[-1, :p]
    R2 = np.dot(np.dot(c, np.linalg.inv(Rx)), c.T)
    R = np.sqrt(R2)

    return np.atleast_1d(solution[0].T), R


# --------------------------- miscellaneous math functions -------------------------------------------------
def primes(n):
    # found on stackoverflow: https://stackoverflow.com/questions/16996217/prime-factorization-list
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac


def largestfac(n):
    return primes(n)[-1]



# --------------------------- Peak detection functions ----------------------------------------------
# The following three functions are taken from the peakdetect distribution by Sixten Bergman
# They were distributed under the DWTFYWTPL, so I'm relicensing them under Apache 2.0
# From his header:
# You can redistribute it and/or modify it under the terms of the Do What The
# Fuck You Want To Public License, Version 2, as published by Sam Hocevar. See
# http://www.wtfpl.net/ for more details.

def parabfit(x_axis, y_axis, peakloc, peaksize):
    func = lambda x, a, tau, c: a * ((x - tau) ** 2) + c
    fitted_peaks = []
    distance = abs(x_axis[raw_peaks[1][0]] - x_axis[raw_peaks[0][0]]) / 4
    index = peakloc
    x_data = x_axis[index - points // 2: index + points // 2 + 1]
    y_data = y_axis[index - points // 2: index + points // 2 + 1]
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
    if x_axis is None:
        x_axis = range(len(y_axis))

    if np.shape(y_axis) != np.shape(x_axis):
        raise ValueError(
            "Input vectors y_axis and x_axis must have same length")

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
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
                # else:  #slows shit down this does
                #    mx = ahead
                #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
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


# --------------------------- Correlation functions -------------------------------------------------
def autocorrcheck(corrscale, thexcorr, delta=0.1, acampthresh=0.1, aclagthresh=10.0, displayplots=False, prewindow=True,
                  dodetrend=True):
    lookahead = 2
    peaks = peakdetect(thexcorr, x_axis=corrscale, delta=delta, lookahead=lookahead)
    maxpeaks = np.asarray(peaks[0], dtype='float64')
    minpeaks = np.asarray(peaks[1], dtype='float64')
    zeropkindex = np.argmin(abs(maxpeaks[:, 0]))
    for i in range(zeropkindex + 1, maxpeaks.shape[0]):
        if maxpeaks[i, 0] > aclagthresh:
            return None, None
        if maxpeaks[i, 1] > acampthresh:
            sidelobetime = maxpeaks[i, 0]
            sidelobeindex = valtoindex(corrscale, sidelobetime)
            sidelobeamp = thexcorr[sidelobeindex]
            numbins = 1
            while (sidelobeindex + numbins < np.shape(corrscale)[0] - 1) and (
                thexcorr[sidelobeindex + numbins] > sidelobeamp / 2.0):
                numbins += 1
            sidelobewidth = (corrscale[sidelobeindex + numbins] - corrscale[sidelobeindex]) * 2.0
            fitstart = sidelobeindex - numbins
            fitend = sidelobeindex + numbins
            sidelobeamp, sidelobetime, sidelobewidth = gaussfit(sidelobeamp, sidelobetime, sidelobewidth,
                                                                corrscale[fitstart:fitend + 1],
                                                                thexcorr[fitstart:fitend + 1])

            if displayplots:
                pl.plot(corrscale[fitstart:fitend + 1], thexcorr[fitstart:fitend + 1], 'k',
                        corrscale[fitstart:fitend + 1],
                        gauss_eval(corrscale[fitstart:fitend + 1], [sidelobeamp, sidelobetime, sidelobewidth]), 'r')
                pl.show()
            return sidelobetime, sidelobeamp
    return None, None


def quickcorr(data1, data2, windowfunc='hamming'):
    thepcorr = sp.stats.stats.pearsonr(corrnormalize(data1, True, True, windowfunc=windowfunc), corrnormalize(data2, True, True, windowfunc=windowfunc))
    return thepcorr


def shorttermcorr_1D(data1, data2, sampletime, windowtime, samplestep=1, prewindow=False, dodetrend=False, windowfunc='hamming'):
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)
    times = []
    corrpertime = []
    ppertime = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = corrnormalize(data1[i - halfwindow:i + halfwindow], prewindow, dodetrend, windowfunc=windowfunc)
        dataseg2 = corrnormalize(data2[i - halfwindow:i + halfwindow], prewindow, dodetrend, windowfunc=windowfunc)
        thepcorr = sp.stats.stats.pearsonr(dataseg1, dataseg2)
        times.append(i * sampletime)
        corrpertime.append(thepcorr[0])
        ppertime.append(thepcorr[1])
    return np.asarray(times, dtype='float64'), np.asarray(corrpertime, dtype='float64'), np.asarray(ppertime, dtype='float64')


def shorttermcorr_2D(data1, data2, sampletime, windowtime, samplestep=1, laglimit=None, weighting='none', prewindow=False, windowfunc='hamming', dodetrend=False, display=False):
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)

    if laglimit is None:
        laglimit = windowtime / 2.0

    dataseg1 = corrnormalize(data1[0:2 * halfwindow], prewindow, dodetrend, windowfunc=windowfunc)
    dataseg2 = corrnormalize(data2[0:2 * halfwindow], prewindow, dodetrend, windowfunc=windowfunc)
    thexcorr = fastcorrelate(dataseg1, dataseg2, weighting=weighting)
    xcorrlen = np.shape(thexcorr)[0]
    xcorr_x = np.arange(0.0, xcorrlen) * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    corrzero = int(xcorrlen // 2)
    xcorrpertime = []
    times = []
    Rvals = []
    delayvals = []
    valid = []
    for i in range(halfwindow,np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = corrnormalize(data1[i - halfwindow:i + halfwindow], prewindow, dodetrend, windowfunc=windowfunc)
        dataseg2 = corrnormalize(data2[i - halfwindow:i + halfwindow], prewindow, dodetrend, windowfunc=windowfunc)
        times.append(i * sampletime)
        xcorrpertime.append(fastcorrelate(dataseg1, dataseg2, weighting=weighting))
        maxindex, thedelayval, theRval, maxsigma, maskval, failreason, peakstart, peakend = findmaxlag_gauss(
            xcorr_x, xcorrpertime[-1], -laglimit, laglimit, 1000.0,
            refine=True,
            useguess=False,
            fastgauss=False,
            displayplots=False)
        delayvals.append(thedelayval)
        Rvals.append(theRval)
        if failreason == 0:
            valid.append(1)
        else:
            valid.append(0)
    if display:
        pl.imshow(xcorrpertime)
    return np.asarray(times, dtype='float64'), \
        np.asarray(xcorrpertime, dtype='float64'), \
        np.asarray(Rvals, dtype='float64'), \
        np.asarray(delayvals, dtype='float64'), \
        np.asarray(valid, dtype='float64')


def delayedcorr(data1, data2, delayval, timestep):
    return sp.stats.stats.pearsonr(data1, timeshift(data2, delayval/timestep, 30)[0])


def cepstraldelay(data1, data2, timestep, displayplots=True):
    # Choudhary, H., Bahl, R. & Kumar, A. 
    # Inter-sensor Time Delay Estimation using cepstrum of sum and difference signals in 
    #     underwater multipath environment. in 1–7 (IEEE, 2015). doi:10.1109/UT.2015.7108308
    ceps1, _ = complex_cepstrum(data1)
    ceps2, _ = complex_cepstrum(data2)
    additive_cepstrum, _ = complex_cepstrum(data1 + data2)
    difference_cepstrum, _ = complex_cepstrum(data1 - data2)
    residual_cepstrum = additive_cepstrum - difference_cepstrum
    if displayplots:
        tvec = timestep * np.arange(0.0, len(data1))
        fig = pl.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title('cepstrum 1')
        ax1.set_xlabel('quefrency in seconds')
        pl.plot(tvec, ceps1.real, tvec, ceps1.imag)
        ax2 = fig.add_subplot(212)
        ax2.set_title('cepstrum 2')
        ax2.set_xlabel('quefrency in seconds')
        pl.plot(tvec, ceps2.real, tvec, ceps2.imag)
        pl.show()

        fig = pl.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title('additive_cepstrum')
        ax1.set_xlabel('quefrency in seconds')
        pl.plot(tvec, additive_cepstrum.real)
        ax2 = fig.add_subplot(312)
        ax2.set_title('difference_cepstrum')
        ax2.set_xlabel('quefrency in seconds')
        pl.plot(tvec, difference_cepstrum)
        ax3 = fig.add_subplot(313)
        ax3.set_title('residual_cepstrum')
        ax3.set_xlabel('quefrency in seconds')
        pl.plot(tvec, residual_cepstrum.real)
        pl.show()
    return timestep * np.argmax(residual_cepstrum.real[0:len(residual_cepstrum) // 2])


# http://stackoverflow.com/questions/12323959/fast-cross-correlation-method-in-python
def fastcorrelate(input1, input2, usefft=True, weighting='none', displayplots=False):
    if usefft:
        # Do an array flipped convolution, which is a correlation.
        if weighting == 'none':
            return signal.fftconvolve(input1, input2[::-1], mode='full')
        else:
            return weightedfftconvolve(input1, input2[::-1], mode='full', weighting=weighting, displayplots=displayplots)
    else:
        return np.correlate(input1, input2, mode='full')


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _check_valid_mode_shapes(shape1, shape2):
    for d1, d2 in zip(shape1, shape2):
        if not d1 >= d2:
            raise ValueError(
                "in1 should have at least as many items as in2 in "
                "every dimension for 'valid' mode.")


def weightedfftconvolve(in1, in2, mode="full", weighting='none', displayplots=False):
    """Convolve two N-dimensional arrays using FFT.
    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.
    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if np.isscalar(in1) and np.isscalar(in2):  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same rank")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    size = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Always use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    if not complex_result:
        fft1 = rfftn(in1, fsize)
        fft2 = rfftn(in2, fsize)
        theorigmax = np.max(np.absolute(irfftn(gccproduct(fft1, fft2, 'none'), fsize)[fslice]))
        ret = irfftn(gccproduct(fft1, fft2, weighting, displayplots=displayplots), fsize)[fslice].copy()
        ret = irfftn(gccproduct(fft1, fft2, weighting, displayplots=displayplots), fsize)[fslice].copy()
        ret = ret.real
        ret *= theorigmax / np.max(np.absolute(ret))
    else:
        fft1 = fftpack.fftn(in1, fsize)
        fft2 = fftpack.fftn(in2, fsize)
        theorigmax = np.max(np.absolute(fftpack.ifftn(gccproduct(fft1, fft2, 'none'))[fslice]))
        ret = fftpack.ifftn(gccproduct(fft1, fft2, weighting, displayplots=displayplots))[fslice].copy()
        ret *= theorigmax / np.max(np.absolute(ret))

    # scale to preserve the maximum
   

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)


def gccproduct(fft1, fft2, weighting, threshfrac=0.1, displayplots=False):
    product = fft1 * fft2
    if weighting == 'none':
        return product

    # calculate the weighting function
    if weighting == 'Liang':
        denom = np.square(np.sqrt(np.absolute(fft1 * np.conjugate(fft1))) + np.sqrt(np.absolute(fft2 * np.conjugate(fft2))))
    elif weighting == 'Eckart':
        denom = np.sqrt(np.absolute(fft1 * np.conjugate(fft1))) * np.sqrt(np.absolute(fft2 * np.conjugate(fft2)))
    elif weighting == 'PHAT':
        denom = np.absolute(product)
    else:
        print('illegal weighting function specified in gccproduct')
        sys.exit()

    if displayplots:
        xvec = range(0, len(denom))
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('reciprocal weighting function')
        pl.plot(xvec, abs(denom))
        pl.show()

    # now apply it while preserving the max
    theorigmax = np.max(np.absolute(denom))
    thresh = theorigmax * threshfrac
    if thresh > 0.0: 
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.nan_to_num(np.where(np.absolute(denom) > thresh, product / denom, np.float64(0.0)))
    else:
        return 0.0 * product
    

def gaussresidualssk(p, y, x):
    err = y - gausssk_eval(x, p)
    return err


def gaussskresiduals(p, y, x):
    return y - gausssk_eval(x, p)


@conditionaljit()
def gaussresiduals(p, y, x):
    return y - p[0] * np.exp(-(x - p[1]) ** 2 / (2.0 * p[2] * p[2]))


def trapezoidresiduals(p, y, x, toplength):
    return y - trapezoid_eval_loop(x, toplength, p)


def risetimeresiduals(p, y, x):
    return y - risetime_eval_loop(x, p)


def thederiv(y):
    dyc = [0.0] * len(y)
    dyc[0] = (y[0] - y[1]) / 2.0
    for i in range(1, len(y) - 1):
        dyc[i] = (y[i + 1] - y[i - 1]) / 2.0
    dyc[-1] = (y[-1] - y[-2]) / 2.0
    return dyc


def gausssk_eval(x, p):
    t = (x - p[1]) / p[2]
    return p[0] * sp.stats.norm.pdf(t) * sp.stats.norm.cdf(p[3] * t)


@conditionaljit()
def gauss_eval(x, p):
    return p[0] * np.exp(-(x - p[1]) ** 2 / (2.0 * p[2] * p[2]))


def trapezoid_eval_loop(x, toplength, p):
    r = np.zeros(len(x), dtype='float64')
    for i in range(0, len(x)):
        r[i] = trapezoid_eval(x[i], toplength, p)
    return r


def risetime_eval_loop(x, p):
    r = np.zeros(len(x), dtype='float64')
    for i in range(0, len(x)):
        r[i] = risetime_eval(x[i], p)
    return r


@conditionaljit()
def trapezoid_eval(x, p):
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    elif 0.0 <= corrx < toplength:
        return p[1] * (1.0 - np.exp(-corrx / p[2]))
    else:
        return p[1] * (np.exp(-(corrx - toplength) / p[3]))


@conditionaljit()
def risetime_eval(x, p):
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    else:
        return p[1] * (1.0 - np.exp(-corrx / p[2]))


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


def makelaglist(lagstart, lagend, lagstep):
    numsteps = int((lagend - lagstart) // lagstep + 1)
    lagend = lagstart + lagstep * (numsteps - 1)
    print("creating list of ", numsteps, " lag steps (", lagstart, " to ", lagend, " in steps of ", lagstep, ")")
    #thelags = np.r_[0.0:1.0 * numsteps] * lagstep + lagstart
    thelags = np.arange(0.0, 1.0 * numsteps) * lagstep + lagstart
    return thelags


# --------------------------- Fitting functions -------------------------------------------------
def locpeak(data, samplerate, lastpeaktime, winsizeinsecs=5.0, thresh=0.75, hysteresissecs=0.4):
    # look at a limited time window
    winsizeinsecs = 5.0
    numpoints = int(winsizeinsecs * samplerate)
    startpoint = max((0, len(data) - numpoints))
    currenttime = (len(data) - 1) / samplerate

    # find normative limits
    recentmax = max(data[startpoint:])
    recentmin = min(data[startpoint:])
    recentrange = recentmax - recentmin

    # screen out obvious non-peaks
    if data[-1] < recentmin + recentrange * thresh:
        # print(currenttime,'below thresh')
        return -1.0
    if currenttime - lastpeaktime < hysteresissecs:
        # print(currenttime,'too soon')
        return -1.0

    # now check to see if we have just passed a peak
    if data[-1] < data[-2]:
        if data[-2] <= data[-3]:
            fitstart = -5
            fitdata = data[fitstart:]
            X = currenttime + (np.arange(0.0, len(fitdata)) - len(fitdata) + 1.0) / samplerate
            maxtime = sum(X * fitdata) / sum(fitdata)
            # maxsigma = np.sqrt(abs(np.square(sum((X - maxtime)) * fitdata) / sum(fitdata)))
            maxsigma = np.sqrt(abs(sum((X - maxtime) ** 2 * fitdata) / sum(fitdata)))
            maxval = fitdata.max()
            peakheight, peakloc, peakwidth = gaussfit(maxval, maxtime, maxsigma, X, fitdata)
            # print(currenttime,fitdata,X,peakloc)
            return peakloc
        else:
            # print(currenttime,'not less enough')
            return -1.0
    else:
        # print(currenttime,'not less')
        return -1.0


# generate the polynomial fit timecourse from the coefficients
@conditionaljit()
def trendgen(thexvals, thefitcoffs, demean):
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


@conditionaljit()
def detrend(inputdata, order=1, demean=False):
    thetimepoints = np.arange(0.0, len(inputdata), 1.0) - len(inputdata) / 2.0
    thecoffs = np.polyfit(thetimepoints, inputdata, order)
    thefittc = trendgen(thetimepoints, thecoffs, demean)
    return inputdata - thefittc


@conditionaljit()
def findfirstabove(theyvals, thevalue):
    for i in range(0, len(theyvals)):
        if theyvals[i] >= thevalue:
            return i
    return len(theyvals)


def findtrapezoidfunc(thexvals, theyvals, thetoplength, initguess=None, debug=False,
                      minrise=0.0, maxrise=200.0, minfall=0.0, maxfall=200.0, minstart=-100.0, maxstart=100.0,
                      refine=False, displayplots=False):
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
    plsq, dummy = sp.optimize.leastsq(trapezoidresiduals, p0, args=(theyvals, thexvals, thetoplength), maxfev=5000)
    # except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (minrise <= plsq[2] <= maxrise) and (minfall <= plsq[3] <= maxfall) and (minstart <= plsq[0] <= maxstart):
        return plsq[0], plsq[1], plsq[2], plsq[3], 1
    else:
        return 0.0, 0.0, 0.0, 0.0, 0


def findrisetimefunc(thexvals, theyvals, initguess=None, debug=False,
                     minrise=0.0, maxrise=200.0, minstart=-100.0, maxstart=100.0, refine=False, displayplots=False):
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
    plsq, dummy = sp.optimize.leastsq(risetimeresiduals, p0, args=(theyvals, thexvals), maxfev=5000)
    # except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (minrise <= plsq[2] <= maxrise) and (minstart <= plsq[0] <= maxstart):
        return plsq[0], plsq[1], plsq[2], 1
    else:
        return 0.0, 0.0, 0.0, 0


@conditionaljit2()
def findmaxlag_gauss(thexcorr_x, thexcorr_y, lagmin, lagmax, widthlimit,
               edgebufferfrac=0.0, threshval=0.0, uthreshval=30.0,
               debug=False, tweaklims=True, zerooutbadfit=True, refine=False, maxguess=0.0, useguess=False,
               searchfrac=0.5, fastgauss=False, lagmod=1000.0, enforcethresh=True, displayplots=False):
    # set initial parameters
    # widthlimit is in seconds
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
    searchbins = int(widthlimit // binwidth)
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
    FML_BADAMPNEG = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value
    if useguess:
        maxindex = valtoindex(thexcorr_x, maxguess)
        nlowerlim = int(maxindex - widthlimit / 2.0)
        nupperlim = int(maxindex + widthlimit / 2.0)
        if nlowerlim < lowerlim:
            nlowerlim = lowerlim
            nupperlim = lowerlim + int(widthlimit)
        if nupperlim > upperlim:
            nupperlim = upperlim
            nlowerlim = upperlim - int(widthlimit)
        maxval_init = thexcorr_y[maxindex].astype('float64')
    else:
        maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype('int32')
        maxval_init = thexcorr_y[maxindex].astype('float64')

    # now get a location for that value
    maxlag_init = (1.0 * thexcorr_x[maxindex]).astype('float64')

    # and calculate the width of the peak
    upperlimit = len(thexcorr_y) - 1
    lowerlimit = 0
    i = 0
    j = 0
    while (maxindex + i <= upperlimit) and (thexcorr_y[maxindex + i] > searchfrac * maxval_init) and (i < searchbins):
        i += 1
    i -= 1
    while (maxindex - j >= lowerlimit) and (thexcorr_y[maxindex - j] > searchfrac * maxval_init) and (j < searchbins):
        j += 1
    j -= 1
    # This is calculated from first principles, but it's always big by a factor or ~1.4. 
    #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
    maxsigma_init = np.float64(((i + j + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0))
    fitstart = lowerlimit
    fitend = upperlimit

    # now check the values for errors and refine if necessary
    fitend = min(maxindex + i + 1, upperlimit)
    fitstart = max(1, maxindex - j)
    if not ((lagmin + binwidth) <= maxlag_init <= (lagmax - binwidth)):
        failreason += FML_HITEDGE
    if i + j + 1 < 3:
        failreason += FML_BADSEARCHWINDOW
    if maxsigma_init > widthlimit:
        failreason += FML_BADWIDTH
    if (maxval_init < threshval) and enforcethresh:
        failreason += FML_BADAMPLOW
    if (maxval_init < 0.0) and enforcethresh:
        failreason += FML_BADAMPNEG
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
                maxsigma = np.float64(np.sqrt(np.fabs(np.sum((X - maxlag) ** 2 * data) / np.sum(data))))
                maxval = np.float64(data.max())
            else:
                # do a least squares fit over the top of the peak
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype='float64')

                if fitend - fitstart >= 3:
                    plsq, dummy = sp.optimize.leastsq(gaussresiduals, p0,
                                                      args=(data, X), maxfev=5000)
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
        print("init to final: maxval", maxval_init, maxval, ", maxlag:", maxlag_init, maxlag, ", width:", maxsigma_init,
              maxsigma)
    if displayplots and refine and (maskval != 0.0):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Data and fit')
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        pl.plot(X, data, 'ro', hiresx, gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])), 'b-')
        pl.show()
    return maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend


@conditionaljit2()
def maxindex_noedge(thexcorr_x, thexcorr_y, bipolar=False):
    lowerlim = 0
    upperlim = len(thexcorr_x) - 1
    done = False
    while not done:
        flipfac = 1.0
        done = True
        maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype('int32')
        if bipolar:
            minindex = (np.argmax(np.fabs(thexcorr_y[lowerlim:upperlim])) + lowerlim).astype('int32')
            if np.fabs(thexcorr_y[minindex]) > np.fabs(thexcorr_y[maxindex]):
                maxindex = minindex
                flipfac = -1.0
        else:
            maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype('int32')
        if upperlim == lowerlim:
            done = True
        if maxindex == 0:
            lowerlim += 1
            done = False
        if maxindex == upperlim:
            upperlim -= 1
            done = False
    return maxindex, flipfac
    

# disabled conditionaljit on 11/8/16.  This causes crashes on some machines (but not mine, strangely enough)
@conditionaljit2()
def findmaxlag_gauss_rev(thexcorr_x, thexcorr_y, lagmin, lagmax, widthlimit,
               absmaxsigma=1000.0,
               hardlimit=True,
               bipolar=False,
               edgebufferfrac=0.0, threshval=0.0, uthreshval=1.0,
               debug=False, tweaklims=True, zerooutbadfit=True, refine=False, maxguess=0.0, useguess=False,
               searchfrac=0.5, fastgauss=False, lagmod=1000.0, enforcethresh=True, displayplots=False):
    # set initial parameters 
    # widthlimit is in seconds
    # maxsigma is in Hz
    # maxlag is in seconds
    warnings.filterwarnings("ignore", "Number*")
    maxlag = np.float64(0.0)
    maxval = np.float64(0.0)
    maxsigma = np.float64(0.0)
    maskval = np.uint16(1)        # start out assuming the fit will succeed
    numlagbins = len(thexcorr_y)
    binwidth = thexcorr_x[1] - thexcorr_x[0]

    # define error values
    failreason = np.uint16(0)
    FML_BADAMPLOW = np.uint16(0x01)
    FML_BADAMPNEG = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)

    # set the search range
    lowerlim = 0
    upperlim = len(thexcorr_x) - 1
    #lowerlim = np.max([valtoindex(thexcorr_x, lagmin, toleft=True), 0])
    #upperlim = np.min([valtoindex(thexcorr_x, lagmax, toleft=False), len(thexcorr_x) - 1])
    if debug:
        print('initial search indices are', lowerlim, 'to', upperlim, '(', thexcorr_x[lowerlim], thexcorr_x[upperlim], ')')

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value and its location
    flipfac = 1.0
    if useguess:
        maxindex = valtoindex(thexcorr_x, maxguess)
    else:
        maxindex, flipfac = maxindex_noedge(thexcorr_x, thexcorr_y, bipolar=bipolar)
        thexcorr_y *= flipfac
    maxlag_init = (1.0 * thexcorr_x[maxindex]).astype('float64')
    maxval_init = thexcorr_y[maxindex].astype('float64')
    if debug:
            print('maxindex, maxlag_init, maxval_init:', maxindex, maxlag_init, maxval_init)

    # then calculate the width of the peak
    thegrad = np.gradient(thexcorr_y).astype('float64')                   # the gradient of the correlation function
    peakpoints = np.where(thexcorr_y > searchfrac * maxval_init, 1, 0)    # mask for places where correlaion exceeds serchfrac*maxval_init
    peakpoints[0] = 0
    peakpoints[-1] = 0
    peakstart = maxindex + 0
    peakend = maxindex + 0
    while thegrad[peakend + 1] < 0.0 and peakpoints[peakend + 1] == 1:
        peakend += 1
    while thegrad[peakstart - 1] > 0.0 and peakpoints[peakstart - 1] == 1:
        peakstart -= 1
    # This is calculated from first principles, but it's always big by a factor or ~1.4. 
    #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
    maxsigma_init = np.float64(((peakend - peakstart + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0))
    if debug:
            print('maxsigma_init:', maxsigma_init)

    # now check the values for errors
    if hardlimit:
        rangeextension = 0.0
    else:
        rangeextension = (lagmax - lagmin) * 0.75
    if not ((lagmin - rangeextension - binwidth) <= maxlag_init <= (lagmax + rangeextension + binwidth)):
        failreason |= (FML_INITFAIL | FML_BADLAG )
        if debug:
            print('bad initial')
    if maxsigma_init > absmaxsigma:
        failreason |= (FML_INITFAIL | FML_BADWIDTH )
        if debug:
            print('bad initial width - too high')
    if peakend - peakstart < 2:
        failreason |= (FML_INITFAIL | FML_BADSEARCHWINDOW )
        if debug:
            print('bad initial width - too low')
    if not (threshval <= maxval_init <= uthreshval) and enforcethresh:
        failreason |= (FML_INITFAIL | FML_BADAMPLOW )
        if debug:
            print('bad initial amp:', maxval_init, 'is less than', threshval)
    if (maxval_init < 0.0):
        failreason |= (FML_INITFAIL | FML_BADAMPNEG )
        if debug:
            print('bad initial amp:', maxval_init, 'is less than', threshval)
    if failreason > 0 and zerooutbadfit:
        maxval = np.float64(0.0)
        maxlag = np.float64(0.0)
        maxsigma = np.float64(0.0)
    else:
        maxval = np.float64(maxval_init)
        maxlag = np.float64(maxlag_init)
        maxsigma = np.float64(maxsigma_init)

    # refine if necessary
    if refine:
        data = thexcorr_y[peakstart:peakend]
        X = thexcorr_x[peakstart:peakend]
        if fastgauss:
            # do a non-iterative fit over the top of the peak
            # 6/12/2015  This is just broken.  Gives quantized maxima
            maxlag = np.float64(1.0 * sum(X * data) / sum(data))
            maxsigma = np.float64(np.sqrt(np.abs(np.sum((X - maxlag) ** 2 * data) / np.sum(data))))
            maxval = np.float64(data.max())
        else:
            # do a least squares fit over the top of the peak
            #p0 = np.array([maxval_init, np.fmod(maxlag_init, lagmod), maxsigma_init], dtype='float64')
            p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype='float64')
            if debug:
                print('fit input array:', p0)
            try:
                plsq, dummy = sp.optimize.leastsq(gaussresiduals, p0, args=(data, X), maxfev=5000)
                maxval = plsq[0]
                maxlag = np.fmod((1.0 * plsq[1]), lagmod)
                maxsigma = plsq[2]
            except:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            if debug:
                print('fit output array:', [maxval, maxlag, maxsigma])

        # check for errors in fit
        fitfail = False
        failreason = np.uint16(0)
        if not (0.0 <= np.fabs(maxval) <= 1.0):
            failreason |= (FML_FITFAIL + FML_BADAMPLOW)
            if debug:
                print('bad amp after refinement')
            fitfail = True
        if (lagmin > maxlag) or (maxlag > lagmax):
            failreason |= (FML_FITFAIL + FML_BADLAG)
            if debug:
                print('bad lag after refinement')
            fitfail = True
        if maxsigma > absmaxsigma:
            failreason |= (FML_FITFAIL + FML_BADWIDTH)
            if debug:
                print('bad width after refinement')
            fitfail = True
        if not (0.0 < maxsigma):
            failreason |= (FML_FITFAIL + FML_BADSEARCHWINDOW)
            if debug:
                print('bad width after refinement')
            fitfail = True
        if fitfail:
            if debug:
                print('fit fail')
            if zerooutbadfit:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            maskval = np.int16(0)
        #print(maxlag_init, maxlag, maxval_init, maxval, maxsigma_init, maxsigma, maskval, failreason, fitfail)
    else:
        maxval = np.float64(maxval_init)
        maxlag = np.float64(np.fmod(maxlag_init, lagmod))
        maxsigma = np.float64(maxsigma_init)
        if failreason > 0:
            maskval = np.uint16(0)

    if debug or displayplots:
        print("init to final: maxval", maxval_init, maxval, ", maxlag:", maxlag_init, maxlag, ", width:", maxsigma_init,
              maxsigma)
    if displayplots and refine and (maskval != 0.0):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Data and fit')
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        pl.plot(X, data, 'ro', hiresx, gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])), 'b-')
        pl.show()
    return maxindex, maxlag, flipfac * maxval, maxsigma, maskval, failreason, peakstart, peakend


@conditionaljit2()
def findmaxlag_quad(thexcorr_x, thexcorr_y, lagmin, lagmax, widthlimit,
               edgebufferfrac=0.0, threshval=0.0, uthreshval=30.0,
               debug=False, tweaklims=True, zerooutbadfit=True, refine=False, maxguess=0.0, useguess=False,
               fastgauss=False, lagmod=1000.0, enforcethresh=True, displayplots=False):
    # set initial parameters
    # widthlimit is in seconds
    # maxsigma is in Hz
    # maxlag is in seconds
    warnings.filterwarnings("ignore", "Number*")
    failreason = np.uint16(0)
    maskval = np.uint16(1)
    numlagbins = len(thexcorr_y)
    binwidth = thexcorr_x[1] - thexcorr_x[0]
    searchbins = int(widthlimit // binwidth)
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
    FML_BADAMPNEG = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value
    maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype('int16')
    maxval_init = thexcorr_y[maxindex].astype('float64')

    # now get a location for that value
    maxlag_init = (1.0 * thexcorr_x[maxindex]).astype('float64')

    # and calculate the width of the peak
    maxsigma_init = np.float64(0.0)
    upperlimit = len(thexcorr_y) - 1
    lowerlimit = 0
    i = 0
    j = 0
    searchfrac = 0.75
    while (maxindex + i <= upperlimit) and (thexcorr_y[maxindex + i] > searchfrac * maxval_init) and (i < searchbins):
        i += 1
    i -= 1
    while (maxindex - j >= lowerlimit) and (thexcorr_y[maxindex - j] > searchfrac * maxval_init) and (j < searchbins):
        j += 1
    j -= 1
    maxsigma_init = (2.0 * (binwidth * (i + j + 1) / 2.355)).astype('float64')

    fitend = min(maxindex + i + 1, upperlimit)
    fitstart = max(1, maxindex - j)
    yvals = thexcorr_y[fitstart:fitend]
    xvals = thexcorr_x[fitstart:fitend]
    if fitend - fitstart + 1 > 3:
        thecoffs = np.polyfit(xvals,yvals, 2)
        maxlag = -thecoffs[1]/(2.0 * thecoffs[0])
        maxval = thecoffs[0] * maxlag * maxlag + thecoffs[1] * maxlag + thecoffs[2]
        maxsigma = maxsigma_init
    else:
        maxlag = 0.0
        maxval = 0.0
        maxsigma = 0.0

    # if maxval > 1.0, fit failed catastrophically, zero out or reset to initial value
    #     corrected logic for 1.1.6
    if ((np.fabs(maxval)) > 1.0) or not (lagmin < maxlag < lagmax) or (maxsigma == 0.0):
        if zerooutbadfit:
            maxval = np.float64(0.0)
            maxlag = np.float64(0.0)
            maxsigma = np.float64(0.0)
            maskval = np.int16(0)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(maxlag_init)
            maxsigma = np.float64(maxsigma_init)
    else:
        maxval = np.float64(maxval)
        maxlag = np.float64(np.fmod(maxlag, lagmod))
        maxsigma = np.float64(np.sqrt(np.fabs(np.sum((xvals - maxlag) ** 2 * yvals) / np.sum(yvals))))
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
        print("init to final: maxval", maxval_init, maxval, ", maxlag:", maxlag_init, maxlag, ", width:", maxsigma_init,
              maxsigma)
    if displayplots and refine and (maskval != 0.0):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Data and fit')
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        pl.plot(X, data, 'ro', hiresx, gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])), 'b-')
        pl.show()
    return maxindex, maxlag, maxval, maxsigma, maskval, failreason, 0, 0


def gaussfitsk(height, loc, width, skewness, xvals, yvals):
    plsq, dummy = sp.optimize.leastsq(gaussresidualssk, np.array([height, loc, width, skewness]),
                                      args=(yvals, xvals), maxfev=5000)
    return plsq


def gaussfit(height, loc, width, xvals, yvals):
    plsq, dummy = sp.optimize.leastsq(gaussresiduals, np.array([height, loc, width]), args=(yvals, xvals), maxfev=5000)
    return plsq[0], plsq[1], plsq[2]


# --------------------------- Resampling and time shifting functions -------------------------------------------
'''
class congrid:
    def __init__(self, timeaxis, width, method='gauss', circular=True, upsampleratio=100, doplot=False, debug=False):
        self.upsampleratio = upsampleratio
        self.initstep = timeaxis[1] - timeaxis[0]
        self.initstart = timeaxis[0]
        self.initend = timeaxis[-1]
        self.hiresstep = self.initstep / np.float64(self.upsampleratio)
        if method == 'gauss':
            fullwidth = 2.355 * width
        fullwidthpts = int(np.round(fullwidth / self.hiresstep, 0))
        fullwidthpts += ((fullwidthpts % 2) - 1)
        self.hires_x = np.linspace(-fullwidth / 2.0, fullwidth / 2.0, numpts = fullwidthpts, endpoint=True)
        if method == 'gauss':
            self.hires_y = gauss_eval(self.hires_x, np.array([1.0, 0.0, width])
        if debug:
            print(self.hires_x)
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('congrid convolution function')
            pl.plot(self.hires_x, self.hires_y)
            pl.legend(('input', 'hires'))
            pl.show()

    def gridded(xvals, yvals):
        if len(xvals) != len(yvals):
            print('x and y vectors do not match - aborting')
            return None
        for i in range(len(xvals)):
        outindices = ((newtimeaxis - self.hiresstart) // self.hiresstep).astype(int)
'''

congridyvals = {}
def congrid(xaxis, loc, val, width, debug=False):
    xstep = xaxis[1] - xaxis[0]
    #weights = xaxis * 0.0
    widthinpts = int(np.round(width * 4.6 / xstep))
    widthinpts -= widthinpts % 2 - 1
    if loc < xaxis[0] or loc > xaxis[-1]:
        print('loc', loc, 'not in range', xaxis[0], xaxis[-1])
        
    center = valtoindex(xaxis, loc)
    offset = loc - xaxis[center]
    offsetkey = str(np.round(offset, 3))
    try:
        yvals = congridyvals[offsetkey]
    except:
        if debug:
            print('new key:', offsetkey)
        xvals = np.linspace(-xstep * (widthinpts // 2), xstep * (widthinpts // 2), num = widthinpts, endpoint=True) + offset
        congridyvals[offsetkey] = gauss_eval(xvals, np.array([1.0, 0.0, width]))
        yvals = congridyvals[offsetkey]
    startpt = int(center - widthinpts // 2)
    indices = range(startpt, startpt + widthinpts)
    indices = np.remainder(indices, len(xaxis))
    #weights[indices] = yvals[:]
    #return val * weights, weights, indices
    return val * yvals, yvals, indices
            

class fastresampler:
    def __init__(self, timeaxis, timecourse, padvalue=30.0, upsampleratio=100, doplot=False, debug=False, method='univariate'):
        self.upsampleratio = upsampleratio
        self.padvalue = padvalue
        self.initstep = timeaxis[1] - timeaxis[0]
        self.initstart = timeaxis[0]
        self.initend = timeaxis[-1]
        self.hiresstep = self.initstep / np.float64(self.upsampleratio)
        self.hires_x = np.arange(timeaxis[0] - self.padvalue, self.initstep * len(timeaxis) + self.padvalue, self.hiresstep)
        self.hiresstart = self.hires_x[0]
        self.hiresend = self.hires_x[-1]
        if method == 'poly':
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[int(self.padvalue // self.hiresstep) + 1:-(int(self.padvalue // self.hiresstep) + 1)] = \
                signal.resample_poly(timecourse, np.int(self.upsampleratio * 10), 10)
        elif method == 'fourier':
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[int(self.padvalue // self.hiresstep) + 1:-(int(self.padvalue // self.hiresstep) + 1)] = \
                signal.resample(timecourse, self.upsampleratio * len(timeaxis))
        else:
            self.hires_y = doresample(timeaxis, timecourse, self.hires_x, method=method)
        self.hires_y[:int(self.padvalue // self.hiresstep)] = self.hires_y[int(self.padvalue // self.hiresstep)]
        self.hires_y[-int(self.padvalue // self.hiresstep):] = self.hires_y[-int(self.padvalue // self.hiresstep)]
        if debug:
            print('fastresampler __init__:')
            print('    padvalue:, ', self.padvalue)
            print('    initstep, hiresstep:', self.initstep, self.hiresstep)
            print('    initial axis limits:', self.initstart, self.initend)
            print('    hires axis limits:', self.hiresstart, self.hiresend)

        # self.hires_y[:int(self.padvalue // self.hiresstep)] = 0.0
        # self.hires_y[-int(self.padvalue // self.hiresstep):] = 0.0
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('fastresampler initial timecourses')
            pl.plot(timeaxis, timecourse, self.hires_x, self.hires_y)
            pl.legend(('input', 'hires'))
            pl.show()

    def yfromx(self, newtimeaxis, doplot=False, debug=False):
        if debug:
            print('fastresampler: yfromx called with following parameters')
            print('    padvalue:, ', self.padvalue)
            print('    initstep, hiresstep:', self.initstep, self.hiresstep)
            print('    initial axis limits:', self.initstart, self.initend)
            print('    hires axis limits:', self.hiresstart, self.hiresend)
            print('    requested axis limits:', newtimeaxis[0], newtimeaxis[-1])
        outindices = ((newtimeaxis - self.hiresstart) // self.hiresstep).astype(int)
        if debug:
            print('len(self.hires_y):', len(self.hires_y))
        try:
            out_y = self.hires_y[outindices]
        except IndexError:
            print('')
            print('indexing out of bounds in fastresampler')
            print('    padvalue:, ', self.padvalue)
            print('    initstep, hiresstep:', self.initstep, self.hiresstep)
            print('    initial axis limits:', self.initstart, self.initend)
            print('    hires axis limits:', self.hiresstart, self.hiresend)
            print('    requested axis limits:', newtimeaxis[0], newtimeaxis[-1])
            sys.exit()
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('fastresampler timecourses')
            pl.plot(self.hires_x, self.hires_y, newtimeaxis, out_y)
            pl.legend(('hires', 'output'))
            pl.show()
        return out_y


def doresample(orig_x, orig_y, new_x, method='cubic', padlen=0):
    pad_y = tide_filt.padvec(orig_y, padlen=padlen)
    tstep = orig_x[1] - orig_x[0]
    if padlen > 0:
        pad_x = np.concatenate((np.arange(orig_x[0] - padlen * tstep, orig_x[0], tstep),
            orig_x,
            np.arange(orig_x[-1] + tstep, orig_x[-1] + tstep * (padlen + 1), tstep)))
    else:
        pad_x = orig_x
    if padlen > 0:
        print('padlen=',padlen)
        print('tstep=',tstep)
        print(pad_x)
    if method == 'cubic':
        cj = signal.cspline1d(pad_y)
        return tide_filt.unpadvec(np.float64(signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
    elif method == 'quadratic':
        qj = signal.qspline1d(pad_y)
        return tide_filt.unpadvec(np.float64(signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
    elif method == 'univariate':
        interpolator = sp.interpolate.UnivariateSpline(pad_x, pad_y, k=3, s=0)  # s=0 interpolates
        return tide_filt.unpadvec(np.float64(interpolator(new_x)), padlen=padlen)
    else:
        print('invalid interpolation method')
        return None


def arbresample(orig_y, init_freq, final_freq, intermed_freq=0.0, method='univariate', debug=False):
    if intermed_freq <= 0.0:
        intermed_freq = np.max([2.0 * init_freq, 2.0 * final_freq])
    orig_x = sp.linspace(0.0, 1.0 / init_freq * len(orig_y), num=len(orig_y), endpoint=False)
    if debug:
        print('arbresample:', len(orig_x), len(orig_y), init_freq, final_freq, intermed_freq)
    return dotwostepresample(orig_x, orig_y, intermed_freq, final_freq, method=method, debug=debug)

    
def dotwostepresample(orig_x, orig_y, intermed_freq, final_freq, method='univariate', debug=False):
    if intermed_freq <= final_freq:
        print('intermediate frequency must be higher than final frequency')
        sys.exit()

    # upsample
    endpoint = orig_x[-1] - orig_x[0]
    init_freq = len(orig_x) / endpoint
    intermed_ts = 1.0 / intermed_freq
    numresamppts = int(endpoint // intermed_ts + 1)
    intermed_x = np.arange(0.0, intermed_ts * numresamppts, intermed_ts)
    intermed_y = doresample(orig_x, orig_y, intermed_x, method=method)

    # antialias and ringstop filter
    aafilterfreq = np.min([final_freq, init_freq]) / 2.0 
    aafilter = tide_filt.noncausalfilter(filtertype='arb', usebutterworth=False, debug=debug)
    aafilter.setarb(0.0, 0.0, 0.95 * aafilterfreq, aafilterfreq)
    antialias_y = aafilter.apply(intermed_freq, intermed_y)

    # downsample
    final_ts = 1.0 / final_freq
    numresamppts = np.ceil(endpoint / final_ts) + 1
    final_x = np.arange(0.0, final_ts * numresamppts, final_ts)
    return doresample(intermed_x, antialias_y, final_x, method=method)


def calcsliceoffset(sotype, slicenum, numslices, tr, multiband=1):
    # Slice timing correction
    # 0 : None
    # 1 : Regular up (0, 1, 2, 3, ...)
    # 2 : Regular down
    # 3 : Use slice order file
    # 4 : Use slice timings file
    # 5 : Standard Interleaved (0, 2, 4 ... 1, 3, 5 ... )
    # 6 : Siemens Interleaved (0, 2, 4 ... 1, 3, 5 ... for odd number of slices)
    # (1, 3, 5 ... 0, 2, 4 ... for even number of slices)
    # 7 : Siemens Multiband Interleaved

    # default value of zero
    slicetime = 0.0

    # None
    if sotype == 0:
        slicetime = 0.0

    # Regular up
    if type == 1:
        slicetime = slicenum * (tr / numslices)

    # Regular down
    if sotype == 2:
        slicetime = (numslices - slicenum - 1) * (tr / numslices)

    # Slice order file not supported - do nothing
    if sotype == 3:
        slicetime = 0.0

    # Slice timing file not supported - do nothing
    if sotype == 4:
        slicetime = 0.0

    # Standard interleave
    if sotype == 5:
        if (slicenum % 2) == 0:
            # even slice number
            slicetime = (tr / numslices) * (slicenum / 2)
        else:
            # odd slice number
            slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens interleave format
    if sotype == 6:
        if (numslices % 2) == 0:
            # even number of slices - slices go 1,3,5,...,0,2,4,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (numslices / 2 + slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((slicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens multiband interleave format
    if sotype == 7:
        numberofshots = numslices / multiband
        modslicenum = slicenum % numberofshots
        if (numberofshots % 2) == 0:
            # even number of shots - slices go 1,3,5,...,0,2,4,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (numberofshots / 2 + modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((modslicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((numberofshots + 1) / 2 + (modslicenum - 1) / 2)
    return slicetime


# NB: a positive value of shifttrs delays the signal, a negative value advances it
# timeshift using fourier phase multiplication
def timeshift(inputtc, shifttrs, padtrs, doplot=False):
    # set up useful parameters
    thelen = np.shape(inputtc)[0]
    thepaddedlen = thelen + 2 * padtrs
    imag = 1.j

    # initialize variables
    preshifted_y = np.zeros(thepaddedlen, dtype='float')  # initialize the working buffer (with pad)
    weights = np.zeros(thepaddedlen, dtype='float')  # initialize the weight buffer (with pad)

    # now do the math
    preshifted_y[padtrs:padtrs + thelen] = inputtc[:]  # copy initial data into shift buffer
    weights[padtrs:padtrs + thelen] = 1.0  # put in the weight vector
    revtc = inputtc[::-1]  # reflect data around ends to
    preshifted_y[0:padtrs] = revtc[-padtrs:]  # eliminate discontinuities
    preshifted_y[padtrs + thelen:] = revtc[0:padtrs]

    # finish initializations
    fftlen = np.shape(preshifted_y)[0]

    # create the phase modulation timecourse
    initargvec = (np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(fftlen)) - np.pi)
    if len(initargvec) > fftlen:
        initargvec = initargvec[:fftlen]
    argvec = np.roll(initargvec * shifttrs, -int(fftlen // 2))
    modvec = np.cos(argvec) - imag * np.sin(argvec)

    # process the data (fft->modulate->ifft->filter)
    fftdata = fftpack.fft(preshifted_y)  # do the actual shifting
    shifted_y = fftpack.ifft(modvec * fftdata).real

    # process the weights
    w_fftdata = fftpack.fft(weights)  # do the actual shifting
    shifted_weights = fftpack.ifft(modvec * w_fftdata).real

    if doplot:
        xvec = range(0, thepaddedlen)  # make a ramp vector (with pad)
        print("shifttrs:", shifttrs)
        print("offset:", padtrs)
        print("thelen:", thelen)
        print("thepaddedlen:", thepaddedlen)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial vector')
        pl.plot(xvec, preshifted_y)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial and shifted vector')
        pl.plot(xvec, preshifted_y, xvec, shifted_y)

        pl.show()

    return ([shifted_y[padtrs:padtrs + thelen], shifted_weights[padtrs:padtrs + thelen], shifted_y,
             shifted_weights])


def envdetect(vector, filtwidth=3.0):
    demeaned = vector - np.mean(vector)
    sigabs = abs(demeaned)
    return tide_filt.dolptrapfftfilt(1.0, 1.0 / (2.0 * filtwidth), 1.1 / (2.0 * filtwidth), sigabs)


# --------------------------- Normalization functions -------------------------------------------------
def znormalize(vector):
    return stdnormalize(vector)


@conditionaljit()
def stdnormalize(vector):
    demeaned = vector - np.mean(vector)
    sigstd = np.std(demeaned)
    if sigstd > 0.0:
        return demeaned / sigstd
    else:
        return demeaned


def varnormalize(vector):
    demeaned = vector - np.mean(vector)
    sigvar = np.var(demeaned)
    if sigvar > 0.0:
        return demeaned / sigvar
    else:
        return demeaned


def pcnormalize(vector):
    sigmean = np.mean(vector)
    if sigmean > 0.0:
        return vector / sigmean - 1.0
    else:
        return vector


def ppnormalize(vector):
    demeaned = vector - np.mean(vector)
    sigpp = np.max(demeaned) - np.min(demeaned)
    if sigpp > 0.0:
        return demeaned / sigpp
    else:
        return demeaned


@conditionaljit()
def corrnormalize(thedata, prewindow, dodetrend, windowfunc='hamming'):
    # detrend first
    if dodetrend:
        intervec = stdnormalize(detrend(thedata, demean=True))
    else:
        intervec = stdnormalize(thedata)

    # then window
    if prewindow:
        return stdnormalize(tide_filt.windowfunction(np.shape(thedata)[0], type=windowfunc) * intervec) / np.sqrt(np.shape(thedata)[0])
    else:
        return stdnormalize(intervec) / np.sqrt(np.shape(thedata)[0])


def rms(vector):
    return np.sqrt(np.mean(np.square(vector)))


# --------------------------- Spectral analysis functions ---------------------------------------
def phase(mcv):
    return np.arctan2(mcv.imag, mcv.real)


def polarfft(invec, samplerate):
    if np.shape(invec)[0] % 2 == 1:
        thevec = invec[:-1]
    else:
        thevec = invec
    spec = fftpack.fft(tide_filt.hamming(np.shape(thevec)[0]) * thevec)[0:np.shape(thevec)[0] // 2]
    magspec = abs(spec)
    phspec = phase(spec)
    maxfreq = samplerate / 2.0
    freqs = np.arange(0.0, maxfreq, maxfreq / (np.shape(spec)[0]))
    return freqs, magspec, phspec


def complex_cepstrum(x):
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1: 
            center = 0  
        ndelay = np.array(np.round(unwrapped[...,center]/np.pi))
        unwrapped -= np.pi * ndelay[...,None] * np.arange(samples) / center
        return unwrapped, ndelay
        
    spectrum = fftpack.fft(x)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = fftpack.ifft(log_spectrum).real
    
    return ceps, ndelay


def real_cepstrum(x):
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    return fftpack.ifft(np.log(np.abs(fftpack.fft(x)))).real


# --------------------------- Utility functions -------------------------------------------------
def logmem(msg, file=None):
    global lastmaxrss_parent, lastmaxrss_child
    if msg is None:
        lastmaxrss_parent = 0
        lastmaxrss_child = 0
        logline = ','.join([ 
            '',
            'Self Max RSS',
            'Self Diff RSS',
            'Self Shared Mem',
            'Self Unshared Mem',
            'Self Unshared Stack',
            'Self Non IO Page Fault'
            'Self IO Page Fault'
            'Self Swap Out',
            'Children Max RSS',
            'Children Diff RSS',
            'Children Shared Mem',
            'Children Unshared Mem',
            'Children Unshared Stack',
            'Children Non IO Page Fault'
            'Children IO Page Fault'
            'Children Swap Out'])
    else:
        rcusage = resource.getrusage(resource.RUSAGE_SELF)
        outvals = [msg]
        outvals.append(str(rcusage.ru_maxrss))
        outvals.append(str(rcusage.ru_maxrss - lastmaxrss_parent))
        lastmaxrss_parent = rcusage.ru_maxrss
        outvals.append(str(rcusage.ru_ixrss))
        outvals.append(str(rcusage.ru_idrss))
        outvals.append(str(rcusage.ru_isrss))
        outvals.append(str(rcusage.ru_minflt))
        outvals.append(str(rcusage.ru_majflt))
        outvals.append(str(rcusage.ru_nswap))
        rcusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        outvals.append(str(rcusage.ru_maxrss))
        outvals.append(str(rcusage.ru_maxrss - lastmaxrss_child))
        lastmaxrss_child = rcusage.ru_maxrss
        outvals.append(str(rcusage.ru_ixrss))
        outvals.append(str(rcusage.ru_idrss))
        outvals.append(str(rcusage.ru_isrss))
        outvals.append(str(rcusage.ru_minflt))
        outvals.append(str(rcusage.ru_majflt))
        outvals.append(str(rcusage.ru_nswap))
        logline = ','.join(outvals)
    if file is None:
        print(logline)
    else:
        file.writelines(logline + "\n")


def findexecutable(command):
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        return shutil.which(command)
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            if os.access(os.path.join(path, command), os.X_OK):
                return os.path.join(path, command)
        return None


def isexecutable(command):
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        if shutil.which(command) is not None:
            return True
        else:
            return False
    else:
        return any(
            os.access(os.path.join(path, command), os.X_OK) 
            for path in os.environ["PATH"].split(os.pathsep)
        )


def savecommandline(theargs, thename):
    tide_io.writevec([' '.join(theargs)], thename + '_commandline.txt')


def valtoindex(thearray, thevalue, toleft=True):
    if toleft:
        return bisect.bisect_left(thearray, thevalue)
    else:
        return bisect.bisect_right(thearray, thevalue)


def progressbar(thisval, end_val, label='Percent', barsize=60):
    percent = float(thisval) / end_val
    hashes = '#' * int(round(percent * barsize))
    spaces = ' ' * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.3f}%".format(label, hashes + spaces, 100.0 * percent))
    sys.stdout.flush()


# ------------------------------------------ Version function ----------------------------------
def version():
    thispath, thisfile = os.path.split(__file__)
    print(thispath)
    if os.path.isfile(os.path.join(thispath, '_gittag.py')):
        with open(os.path.join(thispath, '_gittag.py')) as f:
            for line in f:
                if line.startswith('__gittag__'):
                    fulltag = (line.split()[2]).split('-')
                    break
        return fulltag[0][1:], '-'.join(fulltag[1:])[:-1]
    else:
        return 'UNKNOWN', 'UNKNOWN'


# --------------------------- timing functions -------------------------------------------------
def timefmt(thenumber):
    return "{:10.2f}".format(thenumber)


def proctiminginfo(thetimings, outputfile='', extraheader=None):
    theinfolist = []
    start = thetimings[0]
    starttime = float(start[1])
    lasteventtime = starttime
    if extraheader is not None:
        print(extraheader)
        theinfolist.append(extraheader)
    headerstring = 'Clock time\tProgram time\tDuration\tDescription'
    print(headerstring)
    theinfolist.append(headerstring)
    for theevent in thetimings:
        theduration = float(theevent[1] - lasteventtime)
        outstring = time.strftime("%Y%m%dT%H%M%S", time.localtime(theevent[1])) + \
                    timefmt(float(theevent[1]) - starttime) + \
                    '\t' + timefmt(theduration) + '\t' + theevent[0]
        if theevent[2] is not None:
            outstring += " ({0:.2f} {1}/second)".format(float(theevent[2]) / theduration, theevent[3])
        print(outstring)
        theinfolist.append(outstring)
        lasteventtime = float(theevent[1])
    if outputfile != '':
        tide_io.writevec(theinfolist, outputfile)


