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
from scipy import fftpack, ndimage, signal
import pylab as pl
#import time
import sys
import os

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
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


# --------------------------- Filtering functions -------------------------------------------------
# NB: No automatic padding for precalculated filters

def padvec(indata, padlen=20):
    if padlen > 0:
        return np.concatenate((indata[::-1][-padlen:], indata, indata[::-1][0:padlen]))
    else:
        return indata


def unpadvec(indata, padlen=20):
    if padlen > 0:
        return indata[padlen:-padlen]
    else:
        return indata


def ssmooth(xsize, ysize, zsize, sigma, thedata):
    return ndimage.gaussian_filter(thedata, [sigma / xsize, sigma / ysize, sigma / zsize])


# - direct filter with specified transfer function
def xfuncfilt(indata, xfunc, debug=False):
    return fftpack.ifft(xfunc * fftpack.fft(indata)).real


# - butterworth filters
def dolpfiltfilt(samplefreq, cutofffreq, indata, order, padlen=20, debug=False):
    if cutofffreq > samplefreq / 2.0:
        cutofffreq = samplefreq / 2.0
    if debug:
        print('dolpfiltfilt - samplefreq, cutofffreq, len(indata), order:', samplefreq, cutofffreq, len(indata), order)
    [b, a] = signal.butter(order, 2.0 * cutofffreq / samplefreq)
    return unpadvec(signal.filtfilt(b, a, padvec(indata, padlen=padlen)).real, padlen=padlen)


def dohpfiltfilt(samplefreq, cutofffreq, indata, order, padlen=20, debug=False):
    if cutofffreq < 0.0:
        cutofffreq = 0.0
    if debug:
        print('dohpfiltfilt - samplefreq, cutofffreq, len(indata), order:', samplefreq, cutofffreq, len(indata), order)
    [b, a] = signal.butter(order, 2.0 * cutofffreq / samplefreq, 'highpass')
    return unpadvec(signal.filtfilt(b, a, padvec(indata, padlen=padlen)).real, padlen=padlen)


def dobpfiltfilt(samplefreq, cutofffreq_low, cutofffreq_high, indata, order, padlen=20):
    if cutofffreq_high > samplefreq / 2.0:
        cutofffreq_high = samplefreq / 2.0
    if cutofffreq_log < 0.0:
        cutofffreq_low = 0.0
    [b, a] = signal.butter(order, [2.0 * cutofffreq_low / samplefreq, 2.0 * cutofffreq_high / samplefreq],
                              'bandpass')
    return unpadvec(signal.filtfilt(b, a, padvec(indata, padlen=padlen)).real, padlen=padlen)


def doprecalcfiltfilt(b, a, indata):
    return signal.filtfilt(b, a, indata).real


def dolpfastfiltfiltinit(samplefreq, cutofffreq, indata, order):
    [b, a] = signal.butter(order, cutofffreq / samplefreq)
    return fastfiltfiltinit(b, a, indata)


def dohpfastfiltfiltinit(samplefreq, cutofffreq, indata, order):
    [b, a] = signal.butter(order, cutofffreq / samplefreq, 'highpass')
    return fastfiltfiltinit(b, a, indata)


def dobpfastfiltfiltinit(samplefreq, cutofffreq_low, cutofffreq_high, indata, order):
    [b, a] = signal.butter(order, [cutofffreq_low / samplefreq, cutofffreq_high / samplefreq], 'bandpass')
    return fastfiltfiltinit(b, a, indata)


# - fft brickwall filters
def getlpfftfunc(samplefreq, cutofffreq, indata, debug=False):
    filterfunc = np.ones(np.shape(indata), dtype=np.float64)
    # cutoffbin = int((cutofffreq / samplefreq) * len(filterfunc) / 2.0)
    cutoffbin = int((cutofffreq / samplefreq) * np.shape(filterfunc)[0])
    if debug:
        print('getlpfftfunc - samplefreq, cutofffreq, len(indata):', samplefreq, cutofffreq, np.shpae(indata)[0])
    filterfunc[cutoffbin:-cutoffbin] = 0.0
    return filterfunc


def doprecalcfftfilt(filterfunc, indata):
    indata_trans = fftpack.fft(indata)
    indata_trans = indata_trans * filterfunc
    return fftpack.ifft(indata_trans).real


def dolpfftfilt(samplefreq, cutofffreq, indata, padlen=20, debug=False):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = getlpfftfunc(samplefreq, cutofffreq, padindata, debug=debug)
    indata_trans *= filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dohpfftfilt(samplefreq, cutofffreq, indata, padlen=20, debug=False):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = 1.0 - getlpfftfunc(samplefreq, cutofffreq, padindata, debug=debug)
    indata_trans *= filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dobpfftfilt(samplefreq, cutofffreq_low, cutofffreq_high, indata, padlen=20, debug=False):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = getlpfftfunc(samplefreq, cutofffreq_high, padindata, debug=debug) * (
        1.0 - getlpfftfunc(samplefreq, cutofffreq_low, padindata, debug=debug))
    indata_trans *= filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


# - fft trapezoidal filters
def getlptrapfftfunc(samplefreq, passfreq, stopfreq, indata, debug=False):
    filterfunc = np.ones(np.shape(indata), dtype='float64')
    passbin = int((passfreq / samplefreq) * np.shape(filterfunc)[0])
    cutoffbin = int((stopfreq / samplefreq) * np.shape(filterfunc)[0])
    translength = cutoffbin - passbin
    if debug:
        print('getlptrapfftfunc - samplefreq, passfreq, stopfreq:', samplefreq, passfreq, stopfreq)
        print('getlptrapfftfunc - passbin, translength, cutoffbin, len(indata):', passbin, translength, cutoffbin,
              len(indata))
    if translength > 0:
        transvector = np.arange(1.0 * translength) / translength
        filterfunc[passbin:cutoffbin] = 1.0 - transvector
        filterfunc[-cutoffbin:-passbin] = transvector
    if cutoffbin > 0:
        filterfunc[cutoffbin:-cutoffbin] = 0.0
    return filterfunc


def dolptrapfftfilt(samplefreq, passfreq, stopfreq, indata, padlen=20, debug=False):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = getlptrapfftfunc(samplefreq, passfreq, stopfreq, padindata, debug=debug)
    indata_trans *= filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dohptrapfftfilt(samplefreq, stopfreq, passfreq, indata, padlen=20, debug=False):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = 1.0 - getlptrapfftfunc(samplefreq, stopfreq, passfreq, padindata, debug=debug)
    indata_trans *= filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dobptrapfftfilt(samplefreq, stopfreq_low, passfreq_low, passfreq_high, stopfreq_high, indata, padlen=20,
                    debug=False):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    if False:
        print("samplefreq=", samplefreq, " Fstopl=", stopfreq_low, " Fpassl=", passfreq_low, " Fpassu=", passfreq_high,
              " Fstopu=", stopfreq_high)
    filterfunc = getlptrapfftfunc(samplefreq, passfreq_high, stopfreq_high, padindata, debug=debug) * (
        1.0 - getlptrapfftfunc(samplefreq, stopfreq_low, passfreq_low, padindata, debug=debug))
    if False:
        freqs = np.arange(0.0, samplefreq, samplefreq / np.shape(filterfunc)[0])
        pl.plot(freqs, filterfunc)
        pl.show()
        sys.exit()
    indata_trans *= filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


#### taken from filtfilt from scipy.org Cookbook http://www.scipy.org/Cookbook/FiltFilt
def lfilter_zi(b, a):
    # compute the zi state from the filter parameters. see [Gust96].

    # Based on:
    # [Gust96] Fredrik Gustafsson, Determining the initial states in forward-backward 
    # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996, 
    # Volume 44, Issue 4

    n = max(len(a), len(b))

    zin = (np.eye(n - 1) - np.hstack((-a[1:n, np.newaxis],
                                      np.vstack((np.eye(n - 2), np.zeros(n - 2, dtype='float64'))))))

    zid = b[1:n] - a[1:n] * b[0]

    zi_matrix = np.linalg.inv(zin) * (np.matrix(zid).transpose())
    zi_return = []

    # convert the result into a regular array (not a matrix)
    for i in range(len(zi_matrix)):
        zi_return.append(np.float64(zi_matrix[i][0]))

    return np.array(zi_return)


#### adapted from filtfilt from scipy.org Cookbook http://www.scipy.org/Cookbook/FiltFilt
def fastfiltfiltinit(b, a, x):
    # For now only accepting 1d arrays
    ntaps = max(len(a), len(b))
    edge = ntaps * 10

    if x.ndim != 1:
        raise ValueError("filtfilt is only accepting 1 dimension arrays.")

    # x must be bigger than edge
    if x.size < edge:
        raise ValueError("Input vector needs to be bigger than 3 * max(len(a),len(b).")

    if len(a) < ntaps:
        a = np.r_[a, np.zeros(len(b) - len(a), dtype='float64')]

    if len(b) < ntaps:
        b = np.r_[b, np.zeros(len(a) - len(b), dtype='float64')]

    zi = signal.lfilter_zi(b, a)

    return b, a, zi, edge


#### adapted from filtfilt from scipy.org Cookbook http://www.scipy.org/Cookbook/FiltFilt
def fastfiltfilt(b, a, zi, edge, x):
    # Grow the signal to have edges for stabilizing
    # the filter with inverted replicas of the signal
    s = np.r_[2 * x[0] - x[edge:1:-1], x, 2 * x[-1] - x[-1:-edge:-1]]
    # in the case of one go we only need one of the extrems
    # both are needed for filtfilt

    (y, zf) = signal.lfilter(b, a, s, -1, zi * s[0])

    (y, zf) = signal.lfilter(b, a, np.flipud(y), -1, zi * y[-1])

    return np.flipud(y[edge - 1:-edge + 1])


# Simple example of Wiener deconvolution in Python.
# We use a fixed SNR across all frequencies in this example.
#
# Written 2015 by Dan Stowell. Public domain.
def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR in the fourier domain"
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
    H = fftpack.fft(kernel)
    #deconvolved = np.real(fftpack.ifft(fftpack.fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
    deconvolved = np.roll(np.real(fftpack.ifft(fftpack.fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2))),
        int(len(signal) // 2))
    return deconvolved

def pspec(signal):
    S = fftpack.fft(signal)
    return(np.sqrt(S * np.conj(S)))


def csdfilter(obsdata, commondata, padlen=20, debug=False):
    padobsdata = padvec(obsdata, padlen=padlen)
    padcommondata = padvec(commondata, padlen=padlen)
    obsdata_trans = fftpack.fft(padobsdata)
    filterfunc = np.sqrt(np.abs(fftpack.fft(padobsdata)*np.conj(fftpack.fft(padcommondata))))
    obsdata_trans *= filterfunc
    return unpadvec(fftpack.ifft(obsdata_trans).real, padlen=padlen)
    

def specsplit(samplerate, inputdata, bandwidth, usebutterworth=False):
    lowestfreq = samplerate / (2.0 * np.shape(inputdata)[0])
    highestfreq = samplerate / 2.0
    if lowestfreq < 0.01:
        lowestfreq = 0.01
    if highestfreq > 5.0:
        highestfreq = 5.00
    freqfac = highestfreq / lowestfreq
    print("spectral range=", lowestfreq, " to ", highestfreq, ", factor of ", freqfac)
    lowerlim = lowestfreq
    upperlim = lowerlim * bandwidth
    numbands = 1
    while upperlim < highestfreq:
        lowerlim = lowerlim * bandwidth
        upperlim = upperlim * bandwidth
        numbands += 1
    print("dividing into ", numbands, " bands")
    lowerlim = lowestfreq
    upperlim = lowerlim * bandwidth
    alldata = np.zeros((np.shape(inputdata), numbands), dtype='float64')
    bandcenters = np.zeros(numbands, dtype='float')
    print(alldata.shape)
    for theband in range(0, numbands):
        print("filtering from ", lowerlim, " to ", upperlim)
        if usebutterworth:
            alldata[:, theband] = dobpfiltfilt(samplerate, lowerlim, upperlim, inputdata, 2)
        else:
            alldata[:, theband] = dobpfftfilt(samplerate, lowerlim, upperlim, inputdata)
        bandcenters[theband] = np.sqrt(upperlim * lowerlim)
        lowerlim = lowerlim * bandwidth
        upperlim = upperlim * bandwidth
    return bandcenters, lowestfreq, upperlim, alldata


@conditionaljit()
def arb_pass(samplerate, inputdata, arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop,
             usebutterworth=False, butterorder=defaultbutterorder,
             usetrapfftfilt=True, padlen=20, debug=False):
    # check filter limits to see if we should do a lowpass, bandpass, or highpass
    if arb_lowerpass <= 0.0:
        # set up for lowpass
        if usebutterworth:
            return dolpfiltfilt(samplerate, arb_upperpass, inputdata, butterorder, padlen=padlen, debug=debug)
        else:
            if usetrapfftfilt:
                return dolptrapfftfilt(samplerate, arb_upperpass, arb_upperstop, inputdata, padlen=padlen, debug=debug)
            else:
                return dolpfftfilt(samplerate, arb_upperpass, inputdata, padlen=padlen, debug=debug)
    elif (arb_upperpass >= samplerate / 2.0) or (arb_upperpass <= 0.0):
        # set up for highpass
        if usebutterworth:
            return dohpfiltfilt(samplerate, arb_lowerpass, inputdata, butterorder, padlen=padlen, debug=debug)
        else:
            if usetrapfftfilt:
                return dohptrapfftfilt(samplerate, arb_lowerstop, arb_lowerpass, inputdata, padlen=padlen, debug=debug)
            else:
                return dohpfftfilt(samplerate, arb_lowerpass, inputdata, padlen=padlen, debug=debug)
    else:
        # set up for bandpass
        if usebutterworth:
            return (dohpfiltfilt(samplerate, arb_lowerpass,
                                 dolpfiltfilt(samplerate, arb_upperpass, inputdata, butterorder, padlen=padlen, debug=debug),
                                     butterorder, padlen=padlen, debug=debug))
        else:
            if usetrapfftfilt:
                return (
                    dobptrapfftfilt(samplerate, arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop, inputdata,
                                    padlen=padlen, debug=debug))
            else:
                return dobpfftfilt(samplerate, arb_lowerpass, arb_upperpass, inputdata, padlen=padlen, debug=debug)


class noncausalfilter:
    def __init__(self, filtertype='none', usebutterworth=False, butterworthorder=3, usetrapfftfilt=True,
                 correctfreq=True, padtime=30.0, debug=False):
        self.filtertype = filtertype
        self.arb_lowerpass = 0.05
        self.arb_lowerstop = 0.9 * self.arb_lowerpass
        self.arb_upperpass = 0.20
        self.arb_upperstop = 1.1 * self.arb_upperpass
        self.lowerstop = 0.0
        self.lowerpass = 0.0
        self.upperpass = -1.0
        self.upperstop = -1.0
        self.usebutterworth = usebutterworth
        self.butterworthorder = butterworthorder
        self.usetrapfftfilt = usetrapfftfilt
        self.correctfreq = correctfreq
        self.padtime = padtime
        self.debug = debug
        self.VLF_UPPERPASS = 0.009
        self.VLF_UPPERSTOP = 0.010
        self.LF_LOWERSTOP = self.VLF_UPPERPASS
        self.LF_LOWERPASS = self.VLF_UPPERSTOP
        self.LF_UPPERPASS = 0.15
        self.LF_UPPERSTOP = 0.20
        self.RESP_LOWERSTOP = self.LF_UPPERPASS
        self.RESP_LOWERPASS = self.LF_UPPERSTOP
        self.RESP_UPPERPASS = 0.4
        self.RESP_UPPERSTOP = 0.5
        self.CARD_LOWERSTOP = self.RESP_UPPERPASS
        self.CARD_LOWERPASS = self.RESP_UPPERSTOP
        self.CARD_UPPERPASS = 2.5
        self.CARD_UPPERSTOP = 3.0
        self.settype(self.filtertype)

    def settype(self, thetype):
        self.filtertype = thetype
        if self.filtertype == 'vlf' or self.filtertype == 'vlf_stop':
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = 1.0 * self.VLF_UPPERPASS
            self.upperstop = 1.0 * self.VLF_UPPERSTOP
        elif self.filtertype == 'lfo' or self.filtertype == 'lfo_stop':
            self.lowerstop = 1.0 * self.LF_LOWERSTOP
            self.lowerpass = 1.0 * self.LF_LOWERPASS
            self.upperpass = 1.0 * self.LF_UPPERPASS
            self.upperstop = 1.0 * self.LF_UPPERSTOP
        elif self.filtertype == 'resp' or self.filtertype == 'resp_stop':
            self.lowerstop = 1.0 * self.RESP_LOWERSTOP
            self.lowerpass = 1.0 * self.RESP_LOWERPASS
            self.upperpass = 1.0 * self.RESP_UPPERPASS
            self.upperstop = 1.0 * self.RESP_UPPERSTOP
        elif self.filtertype == 'cardiac' or self.filtertype == 'cardiac_stop':
            self.lowerstop = 1.0 * self.CARD_LOWERSTOP
            self.lowerpass = 1.0 * self.CARD_LOWERPASS
            self.upperpass = 1.0 * self.CARD_UPPERPASS
            self.upperstop = 1.0 * self.CARD_UPPERSTOP
        elif self.filtertype == 'arb' or self.filtertype == 'arb_stop':
            self.lowerstop = 1.0 * self.arb_lowerstop
            self.lowerpass = 1.0 * self.arb_lowerpass
            self.upperpass = 1.0 * self.arb_upperpass
            self.upperstop = 1.0 * self.arb_upperstop
        else:
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = -1.0
            self.upperstop = -1.0

    def gettype(self):
        return self.filtertype

    def getfreqlimits(self):
        return self.lowerstop, self.lowerpass, self.upperpass, self.upperstop

    def setbutter(self, useit, order=3):
        self.usebutterworth = useit
        self.butterworthorder = order

    def setpadtime(self, padtime):
        self.padtime = padtime

    def setdebug(self, debug):
        self.debug = debug

    def getpadtime(self):
        return self.padtime

    def settrapfft(self, useit):
        self.usetrapfftfilt = useit

    def setarb(self, lowerstop, lowerpass, upperpass, upperstop):
        if not (lowerstop <= lowerpass < upperpass):
            print('noncausalfilter error: lowerpass must be between lowerstop and upperpass')
            sys.exit()
        if not (lowerpass < upperpass <= upperstop):
            print('noncausalfilter error: upperpass must be between lowerpass and upperstop')
            sys.exit()
        self.arb_lowerstop = 1.0 * lowerstop
        self.arb_lowerpass = 1.0 * lowerpass
        self.arb_upperpass = 1.0 * upperpass
        self.arb_upperstop = 1.0 * upperstop
        self.lowerstop = 1.0 * self.arb_lowerstop
        self.lowerpass = 1.0 * self.arb_lowerpass
        self.upperpass = 1.0 * self.arb_upperpass
        self.upperstop = 1.0 * self.arb_upperstop

    def apply(self, samplerate, data):
        # do some bounds checking
        nyquistlimit = 0.5 * samplerate
        lowestfreq = 2.0 * samplerate / np.shape(data)[0]

        # first see if entire range is out of bounds
        if self.lowerpass >= nyquistlimit:
            print('noncausalfilter error: filter lower pass ', self.lowerpass, ' exceeds nyquist frequency ',
                  nyquistlimit)
            sys.exit()
        if self.lowerstop >= nyquistlimit:
            print('noncausalfilter error: filter lower stop ', self.lowerstop, ' exceeds nyquist frequency ',
                  nyquistlimit)
            sys.exit()
        if -1.0 < self.upperpass <= lowestfreq:
            print('noncausalfilter error: filter upper pass ', self.upperpass, ' is below minimum frequency ',
                  lowestfreq)
            sys.exit()
        if -1.0 < self.upperstop <= lowestfreq:
            print('noncausalfilter error: filter upper stop ', self.upperstop, ' is below minimum frequency ',
                  lowestfreq)
            sys.exit()

        # now look for fixable errors
        if self.upperpass >= nyquistlimit:
            if self.correctfreq:
                self.upperpass = nyquistlimit
            else:
                print('noncausalfilter error: filter upper pass ', self.upperpass, ' exceeds nyquist frequency ',
                      nyquistlimit)
                sys.exit()
        if self.upperstop > nyquistlimit:
            if self.correctfreq:
                self.upperstop = nyquistlimit
            else:
                print('noncausalfilter error: filter upper stop ', self.upperstop, ' exceeds nyquist frequency ',
                      nyquistlimit)
                sys.exit()
        if self.lowerpass < lowestfreq:
            if self.correctfreq:
                self.lowerpass = lowestfreq
            else:
                print('noncausalfilter error: filter lower pass ', self.lowerpass, ' is below minimum frequency ',
                      lowestfreq)
                sys.exit()
        if self.lowerstop < lowestfreq:
            if self.correctfreq:
                self.lowerstop = lowestfreq
            else:
                print('noncausalfilter error: filter lower stop ', self.lowerstop, ' is below minimum frequency ',
                      lowestfreq)
                sys.exit()

        if self.padtime < 0.0:
            padlen = int(len(data) // 2)
        else:
            padlen = int(self.padtime * samplerate)
        if self.debug:
            print('samplerate=', samplerate)
            print('lowerstop=', self.lowerstop)
            print('lowerpass=', self.lowerpass)
            print('upperpass=', self.upperpass)
            print('upperstop=', self.upperstop)
            print('usebutterworth=', self.usebutterworth)
            print('butterworthorder=', self.butterworthorder)
            print('usetrapfftfilt=', self.usetrapfftfilt)
            print('padtime=', self.padtime)
            print('padlen=', padlen)

        # now do the actual filtering
        if self.filtertype == 'none':
            return data
        elif self.filtertype == 'ringstop':
            return (arb_pass(samplerate, data,
                             0.0, 0.0, samplerate / 4.0, 1.1 * samplerate / 4.0,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'vlf' or self.filtertype == 'lfo' \
                or self.filtertype == 'resp' or self.filtertype == 'cardiac':
            return (arb_pass(samplerate, data,
                             self.lowerstop, self.lowerpass, self.upperpass, self.upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'vlf_stop' or self.filtertype == 'lfo_stop' \
                or self.filtertype == 'resp_stop' or self.filtertype == 'cardiac_stop':
            return (data - arb_pass(samplerate, data,
                                    self.lowerstop, self.lowerpass, self.upperpass, self.upperstop,
                                    usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                                    usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'arb':
            return (arb_pass(samplerate, data,
                             self.arb_lowerstop, self.arb_lowerpass, self.arb_upperpass, self.arb_upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'arb_stop':
            return (data - arb_pass(samplerate, data,
                                    self.arb_lowerstop, self.arb_lowerpass, self.arb_upperpass, self.arb_upperstop,
                                    usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                                    usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        else:
            print("bad filter type")
            sys.exit()


# --------------------------- Window functions -------------------------------------------------
BHwindows = {}
def blackmanharris(length, debug=False):
    #return a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
    try:
        return BHwindows[str(length)]
    except:
        argvec = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(length))
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        BHwindows[str(length)] = a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
        if debug:
            print('initialized Blackman-Harris window for length', length)
        return BHwindows[str(length)]


hannwindows = {}
def hann(length, debug=False):
    #return 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
    try:
        return hannwindows[str(length)]
    except: 
        hannwindows[str(length)] = 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
        if debug:
            print('initialized hann window for length', length)
        return hannwindows[str(length)]


hammingwindows = {}
def hamming(length, debug=False):
#   return 0.54 - 0.46 * np.cos((np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
    try:
        return hammingwindows[str(length)]
    except:
        hammingwindows[str(length)] = 0.54 - 0.46 * np.cos((np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
        if debug:
            print('initialized hamming window for length', length)
        return hammingwindows[str(length)]


def windowfunction(length, type='hamming'):
    if type == 'hamming':
        return hamming(length)
    elif type == 'hann':
        return hann(length)
    elif type == 'blackmanharris':
        return blackmanharris(length)
    elif type == 'None':
        return np.ones(length)
    else:
        print('illegal window function')
        sys.exit()
