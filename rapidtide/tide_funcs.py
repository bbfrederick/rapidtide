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
import rapidtide.resample as tide_resample
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
# --------------------------- miscellaneous math functions -------------------------------------------------
def thederiv(y):
    dyc = [0.0] * len(y)
    dyc[0] = (y[0] - y[1]) / 2.0
    for i in range(1, len(y) - 1):
        dyc[i] = (y[i + 1] - y[i - 1]) / 2.0
    dyc[-1] = (y[-1] - y[-2]) / 2.0
    return dyc


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
    return sp.stats.stats.pearsonr(data1, tide_resample.timeshift(data2, delayval/timestep, 30)[0])


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
        intervec = stdnormalize(tide_fit.detrend(thedata, demean=True))
    else:
        intervec = stdnormalize(thedata)

    # then window
    if prewindow:
        return stdnormalize(tide_filt.windowfunction(np.shape(thedata)[0], type=windowfunc) * intervec) / np.sqrt(np.shape(thedata)[0])
    else:
        return stdnormalize(intervec) / np.sqrt(np.shape(thedata)[0])


def rms(vector):
    return np.sqrt(np.mean(np.square(vector)))


def envdetect(vector, filtwidth=3.0):
    demeaned = vector - np.mean(vector)
    sigabs = abs(demeaned)
    return tide_filt.dolptrapfftfilt(1.0, 1.0 / (2.0 * filtwidth), 1.1 / (2.0 * filtwidth), sigabs)


