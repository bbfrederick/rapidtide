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
from scipy import fftpack, signal
from numpy.fft import rfftn, irfftn
import pylab as pl
import sys
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score

import rapidtide.util as tide_util
import rapidtide.resample as tide_resample
import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math

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


# --------------------------- Correlation functions -------------------------------------------------
def autocorrcheck(corrscale, thexcorr, delta=0.1, acampthresh=0.1, aclagthresh=10.0, displayplots=False, prewindow=True,
                  detrendorder=1, debug=False):
    """

    Parameters
    ----------
    corrscale
    thexcorr
    delta
    acampthresh
    aclagthresh
    displayplots
    prewindow
    detrendorder
    debug

    Returns
    -------

    """
    lookahead = 2
    peaks = tide_fit.peakdetect(thexcorr, x_axis=corrscale, delta=delta, lookahead=lookahead)
    maxpeaks = np.asarray(peaks[0], dtype='float64')
    minpeaks = np.asarray(peaks[1], dtype='float64')
    if len(peaks[0]) > 0:
        if debug:
            print(peaks)
        zeropkindex = np.argmin(abs(maxpeaks[:, 0]))
        for i in range(zeropkindex + 1, maxpeaks.shape[0]):
            if maxpeaks[i, 0] > aclagthresh:
                return None, None
            if maxpeaks[i, 1] > acampthresh:
                sidelobetime = maxpeaks[i, 0]
                sidelobeindex = tide_util.valtoindex(corrscale, sidelobetime)
                sidelobeamp = thexcorr[sidelobeindex]
                numbins = 1
                while (sidelobeindex + numbins < np.shape(corrscale)[0] - 1) and (
                        thexcorr[sidelobeindex + numbins] > sidelobeamp / 2.0):
                    numbins += 1
                sidelobewidth = (corrscale[sidelobeindex + numbins] - corrscale[sidelobeindex]) * 2.0
                fitstart = sidelobeindex - numbins
                fitend = sidelobeindex + numbins
                sidelobeamp, sidelobetime, sidelobewidth = tide_fit.gaussfit(sidelobeamp, sidelobetime, sidelobewidth,
                                                                             corrscale[fitstart:fitend + 1],
                                                                             thexcorr[fitstart:fitend + 1])

                if displayplots:
                    pl.plot(corrscale[fitstart:fitend + 1], thexcorr[fitstart:fitend + 1], 'k',
                            corrscale[fitstart:fitend + 1],
                            tide_fit.gauss_eval(corrscale[fitstart:fitend + 1], [sidelobeamp, sidelobetime, sidelobewidth]),
                            'r')
                    pl.show()
                return sidelobetime, sidelobeamp
    return None, None


def quickcorr(data1, data2, windowfunc='hamming'):
    """

    Parameters
    ----------
    data1
    data2
    windowfunc

    Returns
    -------

    """
    thepcorr = sp.stats.stats.pearsonr(tide_math.corrnormalize(data1,
                                                               prewindow=True,
                                                               detrendorder=1,
                                                               windowfunc=windowfunc),
                                       tide_math.corrnormalize(data2,
                                                               prewindow=True,
                                                               detrendorder=1,
                                                               windowfunc=windowfunc))
    return thepcorr


def shorttermcorr_1D(data1, data2, sampletime, windowtime, samplestep=1, prewindow=False, detrendorder=0,
                     windowfunc='hamming'):
    """

    Parameters
    ----------
    data1
    data2
    sampletime
    windowtime
    samplestep
    prewindow
    detrendorder
    windowfunc

    Returns
    -------

    """
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)
    times = []
    corrpertime = []
    ppertime = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = tide_math.corrnormalize(data1[i - halfwindow:i + halfwindow],
                                           prewindow=prewindow,
                                           detrendorder=detrendorder,
                                           windowfunc=windowfunc)
        dataseg2 = tide_math.corrnormalize(data2[i - halfwindow:i + halfwindow],
                                           prewindow=prewindow,
                                           detrendorder=detrendorder,
                                           windowfunc=windowfunc)
        thepcorr = sp.stats.stats.pearsonr(dataseg1, dataseg2)
        times.append(i * sampletime)
        corrpertime.append(thepcorr[0])
        ppertime.append(thepcorr[1])
    return np.asarray(times, dtype='float64'), np.asarray(corrpertime, dtype='float64'), np.asarray(ppertime,
                                                                                                    dtype='float64')


def shorttermcorr_2D(data1, data2, sampletime, windowtime, samplestep=1, laglimit=None, weighting='none',
                     prewindow=False, windowfunc='hamming', detrendorder=0, display=False):
    """

    Parameters
    ----------
    data1
    data2
    sampletime
    windowtime
    samplestep
    laglimit
    weighting
    prewindow
    windowfunc
    detrendorder
    display

    Returns
    -------

    """
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)

    if laglimit is None:
        laglimit = windowtime / 2.0

    dataseg1 = tide_math.corrnormalize(data1[0:2 * halfwindow], prewindow=prewindow, detrendorder=detrendorder, windowfunc=windowfunc)
    dataseg2 = tide_math.corrnormalize(data2[0:2 * halfwindow], prewindow=prewindow, detrendorder=detrendorder, windowfunc=windowfunc)
    thexcorr = fastcorrelate(dataseg1, dataseg2, weighting=weighting)
    xcorrlen = np.shape(thexcorr)[0]
    xcorr_x = np.arange(0.0, xcorrlen) * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    corrzero = int(xcorrlen // 2)
    xcorrpertime = []
    times = []
    Rvals = []
    delayvals = []
    valid = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = tide_math.corrnormalize(data1[i - halfwindow:i + halfwindow],
                                           prewindow=prewindow,
                                           detrendorder=detrendorder,
                                           windowfunc=windowfunc)
        dataseg2 = tide_math.corrnormalize(data2[i - halfwindow:i + halfwindow],
                                           prewindow=prewindow,
                                           detrendorder=detrendorder,
                                           windowfunc=windowfunc)
        times.append(i * sampletime)
        xcorrpertime.append(fastcorrelate(dataseg1, dataseg2, weighting=weighting))
        maxindex, thedelayval, theRval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_gauss(
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


# from https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy/20505476#20505476
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def delayedcorr(data1, data2, delayval, timestep):
    """

    Parameters
    ----------
    data1
    data2
    delayval
    timestep

    Returns
    -------

    """
    return sp.stats.stats.pearsonr(data1, tide_resample.timeshift(data2, delayval / timestep, 30)[0])


def cepstraldelay(data1, data2, timestep, displayplots=True):
    """

    Parameters
    ----------
    data1
    data2
    timestep
    displayplots

    Returns
    -------

    """
    # Choudhary, H., Bahl, R. & Kumar, A.
    # Inter-sensor Time Delay Estimation using cepstrum of sum and difference signals in
    #     underwater multipath environment. in 1-7 (IEEE, 2015). doi:10.1109/UT.2015.7108308
    ceps1, _ = tide_math.complex_cepstrum(data1)
    ceps2, _ = tide_math.complex_cepstrum(data2)
    additive_cepstrum, _ = tide_math.complex_cepstrum(data1 + data2)
    difference_cepstrum, _ = tide_math.complex_cepstrum(data1 - data2)
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


class aliasedcorrelator:

    def __init__(self, hiressignal, hires_Fs, lores_Fs, timerange, hiresstarttime=0.0, loresstarttime=0.0, padvalue=30.0):
        """

        Parameters
        ----------
        hiressignal: 1D array
            The unaliased waveform to match
        hires_Fs: float
            The sample rate of the unaliased waveform
        lores_Fs: float
            The sample rate of the aliased waveform
        timerange: 1D array
            The delays for which to calculate the correlation function

        """
        self.hiressignal = hiressignal
        self.hires_Fs = hires_Fs
        self.hiresstarttime = hiresstarttime
        self.lores_Fs = lores_Fs
        self.timerange = timerange
        self.loresstarttime = loresstarttime
        self.highresaxis = np.arange(0.0, len(self.hiressignal)) * (1.0 / self.hires_Fs) - self.hiresstarttime
        self.padvalue = padvalue
        self.tcgenerator = tide_resample.fastresampler(self.highresaxis, self.hiressignal, padvalue=self.padvalue)
        self.aliasedsignals = {}

    def apply(self, loressignal, extraoffset):
        """

        Parameters
        ----------
        loressignal: 1D array
            The aliased waveform to match
        extraoffset: float
            Additional offset to apply to hiressignal (e.g. for slice offset)

        Returns
        -------
        corrfunc: 1D array
            The correlation function evaluated at timepoints of timerange
        """
        loresaxis = np.arange(0.0, len(loressignal)) * (1.0 / self.lores_Fs) - self.loresstarttime
        targetsignal = tide_math.corrnormalize(loressignal)
        corrfunc = self.timerange * 0.0
        for i in range(len(self.timerange)):
            theoffset = self.timerange[i] + extraoffset
            offsetkey = "{:.3f}".format(theoffset)
            try:
                aliasedhiressignal = self.aliasedsignals[offsetkey]
                #print(offsetkey, ' - cache hit')
            except KeyError:
                #print(offsetkey, ' - cache miss')
                self.aliasedsignals[offsetkey] = tide_math.corrnormalize(self.tcgenerator.yfromx(loresaxis + theoffset))
                aliasedhiressignal = self.aliasedsignals[offsetkey]
            corrfunc[i] = np.dot(aliasedhiressignal, targetsignal)
        return corrfunc


def aliasedcorrelate(hiressignal, hires_Fs, lowressignal, lowres_Fs, timerange, hiresstarttime=0.0, lowresstarttime=0.0, padvalue=30.0):
    """

    Parameters
    ----------
    hiressignal: 1D array
        The unaliased waveform to match
    hires_Fs: float
        The sample rate of the unaliased waveform
    lowressignal: 1D array
        The aliased waveform to match
    lowres_Fs: float
        The sample rate of the aliased waveform
    timerange: 1D array
        The delays for which to calculate the correlation function

    Returns
    -------
    corrfunc: 1D array
        The correlation function evaluated at timepoints of timerange
    """
    highresaxis = np.arange(0.0, len(hiressignal)) * (1.0 / hires_Fs) - hiresstarttime
    lowresaxis = np.arange(0.0, len(lowressignal)) * (1.0 / lowres_Fs) - lowresstarttime
    tcgenerator = tide_resample.fastresampler(highresaxis, hiressignal, padvalue=padvalue)
    targetsignal = tide_math.corrnormalize(lowressignal)
    corrfunc = timerange * 0.0
    for i in range(len(timerange)):
        aliasedhiressignal = tide_math.corrnormalize(tcgenerator.yfromx(lowresaxis + timerange[i]))
        corrfunc[i] = np.dot(aliasedhiressignal, targetsignal)
    return corrfunc


# http://stackoverflow.com/questions/12323959/fast-cross-correlation-method-in-python
def fastcorrelate(input1, input2, usefft=True, weighting='none', displayplots=False):
    """

    Parameters
    ----------
    input1
    input2
    usefft
    weighting
    displayplots

    Returns
    -------

    """
    if usefft:
        # Do an array flipped convolution, which is a correlation.
        if weighting == 'none':
            return signal.fftconvolve(input1, input2[::-1], mode='full')
        else:
            return weightedfftconvolve(input1, input2[::-1], mode='full', weighting=weighting,
                                       displayplots=displayplots)
    else:
        return np.correlate(input1, input2, mode='full')


def _centered(arr, newsize):
    """

    Parameters
    ----------
    arr
    newsize

    Returns
    -------

    """
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _check_valid_mode_shapes(shape1, shape2):
    """

    Parameters
    ----------
    shape1
    shape2

    Returns
    -------

    """
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
    """Calculate product for generalized crosscorrelation

    Parameters
    ----------
    fft1
    fft2
    weighting
    threshfrac
    displayplots

    Returns
    -------

    """
    product = fft1 * fft2
    if weighting == 'none':
        return product

    # calculate the weighting function
    if weighting == 'Liang':
        denom = np.square(
            np.sqrt(np.absolute(fft1 * np.conjugate(fft1))) + np.sqrt(np.absolute(fft2 * np.conjugate(fft2))))
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
