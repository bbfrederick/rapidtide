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
"""This module constains all the filtering operations for the rapidtide
package.

"""

from __future__ import print_function, division

import numpy as np
from scipy import fftpack, ndimage, signal
import pylab as pl
import sys

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


def conditionaljit():
    def resdec(f):
        if (not numbaexists) or donotusenumba:
            return f
        return jit(f)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Filtering functions -------------------------------------------------
# NB: No automatic padding for precalculated filters

def padvec(inputdata, padlen=20):
    r"""Returns a padded copy of the input data; padlen points of
    reflected data are prepended and appended to the input data to reduce
    end effects when the data is then filtered.

    Parameters
    ----------
    inputdata : 1D array
        An array of any numerical type.
        :param inputdata:
    padlen : int, optional
        The number of points to remove from each end.  Default is 20.
        :param padlen:

    Returns
    -------
    paddeddata : 1D array
        The input data, with padlen reflected points added to each end

    """
    if padlen > 0:
        return np.concatenate((inputdata[::-1][-padlen:], inputdata, inputdata[::-1][0:padlen]))
    else:
        return inputdata


def unpadvec(inputdata, padlen=20):
    r"""Returns a input data with the end pads removed (see padvec);
    padlen points of reflected data are removed from each end of the array.

    Parameters
    ----------
    inputdata : 1D array
        An array of any numerical type.
        :param inputdata:
    padlen : int, optional
        The number of points to remove from each end.  Default is 20.
        :param padlen:

    Returns
    -------
    unpaddeddata : 1D array
        The input data, with the padding data removed


    """
    if padlen > 0:
        return inputdata[padlen:-padlen]
    else:
        return inputdata


def ssmooth(xsize, ysize, zsize, sigma, inputdata):
    r"""Applies an isotropic gaussian spatial filter to a 3D array

    Parameters
    ----------
    xsize : float
        The array x step size in spatial units
        :param xsize:

    ysize : float
        The array y step size in spatial units
        :param ysize:

    zsize : float
        The array z step size in spatial units
        :param zsize:

    sigma : float
        The width of the gaussian filter kernel in spatial units
        :param sigma:

    inputdata : 3D numeric array
        The spatial data to filter
        :param inputdata:

    Returns
    -------
    filtereddata : 3D float array
        The filtered spatial data

    """
    return ndimage.gaussian_filter(inputdata, [sigma / xsize, sigma / ysize, sigma / zsize])


# - butterworth filters
def dolpfiltfilt(Fs, upperpass, inputdata, order, padlen=20, debug=False):
    r"""Performs a bidirectional (zero phase) Butterworth lowpass filter on an input vector
    and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:
    upperpass : float
        Upper end of passband in Hz
        :param upperpass:
    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:
    order : int
        Order of Butterworth filter.
        :param order:
    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:
    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data

    """
    if upperpass > Fs / 2.0:
        upperpass = Fs / 2.0
    if debug:
        print('dolpfiltfilt - Fs, upperpass, len(inputdata), order:', Fs, upperpass, len(inputdata), order)
    [b, a] = signal.butter(order, 2.0 * upperpass / Fs)
    return unpadvec(signal.filtfilt(b, a, padvec(inputdata, padlen=padlen)).real, padlen=padlen)


def dohpfiltfilt(Fs, lowerpass, inputdata, order, padlen=20, debug=False):
    r"""Performs a bidirectional (zero phase) Butterworth highpass filter on an input vector
    and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    order : int
        Order of Butterworth filter.
        :param order:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    if lowerpass < 0.0:
        lowerpass = 0.0
    if debug:
        print('dohpfiltfilt - Fs, lowerpass, len(inputdata), order:', Fs, lowerpass, len(inputdata), order)
    [b, a] = signal.butter(order, 2.0 * lowerpass / Fs, 'highpass')
    return unpadvec(signal.filtfilt(b, a, padvec(inputdata, padlen=padlen)).real, padlen=padlen)


def dobpfiltfilt(Fs, lowerpass, upperpass, inputdata, order, padlen=20, debug=False):
    r"""Performs a bidirectional (zero phase) Butterworth bandpass filter on an input vector
    and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    order : int
        Order of Butterworth filter.
        :param order:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    if upperpass > Fs / 2.0:
        upperpass = Fs / 2.0
    if lowerpass < 0.0:
        lowerpass = 0.0
    if debug:
        print('dobpfiltfilt - Fs, lowerpass, upperpass, len(inputdata), order:',
              Fs, lowerpass, upperpass, len(inputdata), order)
    [b, a] = signal.butter(order, [2.0 * lowerpass / Fs, 2.0 * upperpass / Fs],
                           'bandpass')
    return unpadvec(signal.filtfilt(b, a, padvec(inputdata, padlen=padlen)).real, padlen=padlen)


# - direct filter with specified transfer function
def transferfuncfilt(inputdata, transferfunc):
    r"""Filters input data using a previously calculated transfer function.

    Parameters
    ----------
    inputdata : 1D float array
        Input data to be filtered
        :param inputdata:

    transferfunc : 1D float array
        The transfer function
        :param transferfunc:

    Returns
    -------
    filtereddata : 1D float array
        Filtered input data
    """
    inputdata_trans = transferfunc * fftpack.fft(inputdata)
    return fftpack.ifft(inputdata_trans).real


# - fft brickwall filters
def getlpfftfunc(Fs, upperpass, inputdata, debug=False):
    r"""Generates a brickwall lowpass transfer function.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    transferfunc : 1D float array
        The transfer function
    """
    transferfunc = np.ones(np.shape(inputdata), dtype=np.float64)
    cutoffbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
    if debug:
        print('getlpfftfunc - Fs, upperpass, len(inputdata):', Fs, upperpass, np.shape(inputdata)[0])
    transferfunc[cutoffbin:-cutoffbin] = 0.0
    return transferfunc


def dolpfftfilt(Fs, upperpass, inputdata, padlen=20, debug=False):
    r"""Performs an FFT brickwall lowpass filter on an input vector
    and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlpfftfunc(Fs, upperpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


def dohpfftfilt(Fs, lowerpass, inputdata, padlen=20, debug=False):
    r"""Performs an FFT brickwall highpass filter on an input vector
    and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = 1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


def dobpfftfilt(Fs, lowerpass, upperpass, inputdata, padlen=20, debug=False):
    r"""Performs an FFT brickwall bandpass filter on an input vector
    and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlpfftfunc(Fs, upperpass, padinputdata, debug=debug) * (
            1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug))
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# - fft trapezoidal filters
def getlptrapfftfunc(Fs, upperpass, upperstop, inputdata, debug=False):
    r"""Generates a trapezoidal lowpass transfer function.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    upperstop : float
        Lower end of stopband in Hz
        :param upperstop:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    transferfunc : 1D float array
        The transfer function
    """
    transferfunc = np.ones(np.shape(inputdata), dtype='float64')
    passbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
    cutoffbin = int((upperstop / Fs) * np.shape(transferfunc)[0])
    transitionlength = cutoffbin - passbin
    if debug:
        print('getlptrapfftfunc - Fs, upperpass, upperstop:', Fs, upperpass, upperstop)
        print('getlptrapfftfunc - passbin, transitionlength, cutoffbin, len(inputdata):',
              passbin, transitionlength, cutoffbin, len(inputdata))
    if transitionlength > 0:
        transitionvector = np.arange(1.0 * transitionlength) / transitionlength
        transferfunc[passbin:cutoffbin] = 1.0 - transitionvector
        transferfunc[-cutoffbin:-passbin] = transitionvector
    if cutoffbin > 0:
        transferfunc[cutoffbin:-cutoffbin] = 0.0
    return transferfunc


def dolptrapfftfilt(Fs, upperpass, upperstop, inputdata, padlen=20, debug=False):
    r"""Performs an FFT filter with a trapezoidal lowpass transfer
    function on an input vector and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    upperstop : float
        Lower end of stopband in Hz
        :param upperstop:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


def dohptrapfftfilt(Fs, lowerstop, lowerpass, inputdata, padlen=20, debug=False):
    r"""Performs an FFT filter with a trapezoidal highpass transfer
    function on an input vector and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    lowerstop : float
        Upper end of stopband in Hz
        :param lowerstop:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = 1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


def dobptrapfftfilt(Fs, lowerstop, lowerpass, upperpass, upperstop, inputdata, padlen=20,
                    debug=False):
    r"""Performs an FFT filter with a trapezoidal bandpass transfer
    function on an input vector and returns the result.  Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    lowerstop : float
        Upper end of stopband in Hz
        :param lowerstop:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    upperstop : float
        Lower end of stopband in Hz
        :param upperstop:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen)
    inputdata_trans = fftpack.fft(padinputdata)
    if debug:
        print("Fs=", Fs, " Fstopl=", lowerstop, " Fpassl=", lowerpass, " Fpassu=", upperpass,
              " Fstopu=", upperstop)
    transferfunc = getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug) * (
            1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug))
    if debug:
        freqs = np.arange(0.0, Fs, Fs / np.shape(transferfunc)[0])
        pl.plot(freqs, transferfunc)
        pl.show()
        sys.exit()
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# Simple example of Wiener deconvolution in Python.
# We use a fixed SNR across all frequencies in this example.
#
# Written 2015 by Dan Stowell. Public domain.
def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR in the fourier domain"
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel))))  # zero pad the kernel to same length
    H = fftpack.fft(kernel)
    deconvolved = np.roll(np.real(fftpack.ifft(fftpack.fft(signal) * np.conj(H) / (H * np.conj(H) + lambd ** 2))),
                          int(len(signal) // 2))
    return deconvolved


def pspec(inputdata):
    r"""Calculate the power spectrum of an input signal
    Parameters
    ----------
    inputdata: 1D numpy array
        Input data

    Returns
    -------
    spectrum: 1D numpy array
        The power spectrum of the input signal.

    """
    S = fftpack.fft(inputdata)
    return np.sqrt(S * np.conj(S))


def spectrum(inputdata, Fs=1.0, mode='power'):
    r"""Performs an FFT of the input data, and returns the frequency axis and spectrum
    of the input signal.

    Parameters
    ----------
    inputdata : 1D numpy array
        Input data
        :param inputdata:

    Fs : float, optional
        Sample rate in Hz.  Defaults to 1.0
        :param Fs:

    mode : {'real', 'imag', 'mag', 'phase', 'power'}, optional
        The type of spectrum to return.  Default is 'power'.
        :param mode:

    Returns
    -------
    specaxis : 1D float array
        The frequency axis.

    specvals : 1D float array
        The spectral data.

    Other Parameters
    ----------------
    Fs : float
        Sample rate in Hz.  Defaults to 1.0
        :param Fs:

    mode : {'real', 'imag', 'mag', 'phase', 'power'}
        The type of spectrum to return.  Legal values are 'real', 'imag', 'mag', 'phase', and 'power' (default)
        :param mode:
    """
    specvals = fftpack.fft(inputdata)[0:len(inputdata) // 2]
    maxfreq = Fs / 2.0
    specaxis = np.linspace(0.0, maxfreq, len(specvals), endpoint=False)
    if mode == 'real':
        specvals = specvals.real
    elif mode == 'imag':
        specvals = specvals.imag
    elif mode == 'mag':
        specvals = np.absolute(specvals)
    elif mode == 'phase':
        specvals = np.angle(specvals)
    elif mode == 'power':
        specvals = np.sqrt(np.absolute(specvals))
    else:
        print('illegal spectrum mode')
        specvals = None
    return specaxis, specvals


def csdfilter(obsdata, commondata, padlen=20, debug=False):
    r"""Cross spectral density filter - makes a filter transfer function that preserves common frequencies.

    Parameters
    ----------
    obsdata: 1D numpy array
        Input data
    commondata: 1D numpy array
        Shared data
    padlen: int, optional
        Number of reflected points to add on each end of the input data.  Default is 20.
    debug: bool, optional
        Set to True for additiona information on function internals.  Default is False.

    Returns
    -------
    filtereddata: 1D numpy array
        The filtered data

    """
    padobsdata = padvec(obsdata, padlen=padlen)
    padcommondata = padvec(commondata, padlen=padlen)
    obsdata_trans = fftpack.fft(padobsdata)
    transferfunc = np.sqrt(np.abs(fftpack.fft(padobsdata) * np.conj(fftpack.fft(padcommondata))))
    obsdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(obsdata_trans).real, padlen=padlen)


@conditionaljit()
def arb_pass(Fs, inputdata, lowerstop, lowerpass, upperpass, upperstop,
             usebutterworth=False, butterorder=6,
             usetrapfftfilt=True, padlen=20, debug=False):
    r"""Filters an input waveform over a specified range.  By default it is a trapezoidal
    FFT filter, but brickwall and butterworth filters are also available.  Ends are padded to reduce
    transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    lowerstop : float
        Upper end of lower stopband in Hz
        :param lowerstop:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    upperstop : float
        Lower end of upper stopband in Hz
        :param upperstop:

    usebutterworth : boolean, optional
        Whether to use a Butterworth filter characteristic.  Default is False.
        :param usebutterworth:

    butterorder : int, optional
        Order of Butterworth filter.  Default is 6.
        :param butterorder:

    usetrapfftfilt : boolean, optional
        Whether to use trapezoidal transition band for FFT filter.  Default is True.
        :param usetrapfftfilt:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    # check filter limits to see if we should do a lowpass, bandpass, or highpass
    if lowerpass <= 0.0:
        # set up for lowpass
        if usebutterworth:
            return dolpfiltfilt(Fs, upperpass, inputdata, butterorder, padlen=padlen, debug=debug)
        else:
            if usetrapfftfilt:
                return dolptrapfftfilt(Fs, upperpass, upperstop, inputdata, padlen=padlen, debug=debug)
            else:
                return dolpfftfilt(Fs, upperpass, inputdata, padlen=padlen, debug=debug)
    elif (upperpass >= Fs / 2.0) or (upperpass <= 0.0):
        # set up for highpass
        if usebutterworth:
            return dohpfiltfilt(Fs, lowerpass, inputdata, butterorder, padlen=padlen, debug=debug)
        else:
            if usetrapfftfilt:
                return dohptrapfftfilt(Fs, lowerstop, lowerpass, inputdata, padlen=padlen, debug=debug)
            else:
                return dohpfftfilt(Fs, lowerpass, inputdata, padlen=padlen, debug=debug)
    else:
        # set up for bandpass
        if usebutterworth:
            return (dohpfiltfilt(Fs, lowerpass,
                                 dolpfiltfilt(Fs, upperpass, inputdata, butterorder, padlen=padlen,
                                              debug=debug),
                                 butterorder, padlen=padlen, debug=debug))
        else:
            if usetrapfftfilt:
                return (
                    dobptrapfftfilt(Fs, lowerstop, lowerpass, upperpass, upperstop, inputdata,
                                    padlen=padlen, debug=debug))
            else:
                return dobpfftfilt(Fs, lowerpass, upperpass, inputdata, padlen=padlen, debug=debug)


@conditionaljit()
def getarbpassfunc(Fs, inputdata, lowerstop, lowerpass, upperpass, upperstop,
                   usebutterworth=False, butterorder=6,
                   usetrapfftfilt=True, padlen=20, debug=False):
    r"""Generates the transfer function for an arb_pass filter for a given length of input waveform over a specified
    range.  By default it is a trapezoidal FFT filter, but brickwall and butterworth filters are also available.
    Ends are padded to reduce transients.

    Parameters
    ----------
    Fs : float
        Sample rate in Hz
        :param Fs:

    inputdata : 1D numpy array
        Input data to be filtered
        :param inputdata:

    lowerstop : float
        Upper end of lower stopband in Hz
        :param lowerstop:

    lowerpass : float
        Lower end of passband in Hz
        :param lowerpass:

    upperpass : float
        Upper end of passband in Hz
        :param upperpass:

    upperstop : float
        Lower end of upper stopband in Hz
        :param upperstop:

    usebutterworth : boolean, optional
        Whether to use a Butterworth filter characteristic.  Default is False.
        :param usebutterworth:

    butterorder : int, optional
        Order of Butterworth filter.  Default is 6.
        :param butterorder:

    usetrapfftfilt : boolean, optional
        Whether to use trapezoidal transition band for FFT filter.  Default is True.
        :param usetrapfftfilt:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data

    """
    padinputdata = padvec(inputdata, padlen=padlen)

    # check filter limits to see if we should do a lowpass, bandpass, or highpass
    if lowerpass <= 0.0:
        # set up for lowpass
        if usebutterworth:
            return dolpfiltfilt(Fs, upperpass, inputdata, butterorder, padlen=padlen, debug=debug)
        else:
            if usetrapfftfilt:
                return getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug)
            else:
                return getlptrapfftfunc(Fs, upperpass, padinputdata, debug=debug)
    elif (upperpass >= Fs / 2.0) or (upperpass <= 0.0):
        # set up for highpass
        if usebutterworth:
            return dohpfiltfilt(Fs, lowerpass, inputdata, butterorder, padlen=padlen, debug=debug)
        else:
            if usetrapfftfilt:
                return 1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)
            else:
                return 1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)
    else:
        # set up for bandpass
        if usebutterworth:
            return (dohpfiltfilt(Fs, lowerpass,
                                 dolpfiltfilt(Fs, upperpass, padinputdata,
                                              butterorder, padlen=padlen, debug=debug),
                                 butterorder, padlen=padlen, debug=debug))
        else:
            if usetrapfftfilt:
                return (
                        getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug) * \
                        (1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)))
            else:
                return (
                        getlpfftfunc(Fs, upperpass, padinputdata, debug=debug) * \
                        (1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)))


class noncausalfilter:
    """
    A zero time delay filter for one dimensional signals, especially physiological ones.

    ...

    Attributes
    ----------
    filtertype : {'none' 'vlf', 'lfo', 'resp', 'card', 'vlf_stop', 'lfo_stop', 'resp_stop', 'card_stop', 'arb', 'arb_stop', 'ringstop'}
        The type of filter.  Options are 'none' (default), 'vlf', 'lfo', 'resp', 'card', 'vlf_stop', 'lfo_stop',
        'resp_stop', 'card_stop', 'arb', 'arb_stop', 'ringstop'.

    species: {'human'}
        Species (for setting physiological ranges).  Options are 'human' (default)

    lowerpass: float
        Lower pass frequency for current filter.

    lowerstop: float
        Lower stop frequency for current filter.

    upperpass: float
        Upper pass frequency for current filter.

    upperstop: float
        Upper stop frequency for current filter.

    usebutterworth: boolean
        Use Butterworth filter.  Default is False.

    butterworthorder: int
        Butterworth filter order.  Default is 6.

    usetrapfftfilt: boolean
        Use trapezoidal pass band for FFT filter.  Default is True.

    correctfreq: boolean
        Fix pass frequencies that are impossible.  Default is True.

    padtime: float
        Amount of time to end pad to reduce edge effects.  Default is 30.0 seconds

    debug: boolean
        Enable extended debugging messages.  Default is False.

    Methods
    -------
    settype(thetype)
        Set the filter type. Options are 'none' (default), 'vlf', 'lfo', 'resp', 'card', 'vlf_stop', 'lfo_stop',
        'resp_stop', 'card_stop', 'arb', 'arb_stop', 'ringstop'.
    gettype()
        Return the current filter type.
    getfreqlimits()
        Return the current frequency limits.
    setbutter(useit, order=self.butterworthorder)
        Set options for Butterworth filter (set useit to True to make Butterworth the active filter type, set order
        with the order parameter)
    setpadtime(padtime)
        Set the end pad time in seconds.
    setdebug(debug)
        Turn debugging on and off with the debug flag.
    getpadtime()
        Return the current end pad time.
    settrapfft(useit)
        Set to use trapezoidal FFT filter.  If false use brickwall.
    setarb(lowerstop, lowerpass, upperpass, upperstop)
        Set the frequency parameters of the 'arb' and 'arb_stop' filter.
    apply(Fs, data)
        Apply the filter to a dataset.
    """

    def __init__(self, filtertype='none', usebutterworth=False, butterworthorder=6, usetrapfftfilt=True,
                 correctfreq=True, padtime=30.0, debug=False):
        r"""A zero time delay filter for one dimensional signals, especially physiological ones.

        Parameters
        ----------
        filtertype
        usebutterworth
        butterworthorder
        usetrapfftfilt
        correctfreq
        padtime
        debug
        """
        self.filtertype = filtertype
        self.species = 'human'
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

    def apply(self, Fs, data):
        # do some bounds checking
        nyquistlimit = 0.5 * Fs
        lowestfreq = 2.0 * Fs / np.shape(data)[0]

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
            padlen = int(self.padtime * Fs)
        if self.debug:
            print('Fs=', Fs)
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
            return (arb_pass(Fs, data,
                             0.0, 0.0, Fs / 4.0, 1.1 * Fs / 4.0,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'vlf' or self.filtertype == 'lfo' \
                or self.filtertype == 'resp' or self.filtertype == 'cardiac':
            return (arb_pass(Fs, data,
                             self.lowerstop, self.lowerpass, self.upperpass, self.upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'vlf_stop' or self.filtertype == 'lfo_stop' \
                or self.filtertype == 'resp_stop' or self.filtertype == 'cardiac_stop':
            return (data - arb_pass(Fs, data,
                                    self.lowerstop, self.lowerpass, self.upperpass, self.upperstop,
                                    usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                                    usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'arb':
            return (arb_pass(Fs, data,
                             self.arb_lowerstop, self.arb_lowerpass, self.arb_upperpass, self.arb_upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        elif self.filtertype == 'arb_stop':
            return (data - arb_pass(Fs, data,
                                    self.arb_lowerstop, self.arb_lowerpass, self.arb_upperpass, self.arb_upperstop,
                                    usebutterworth=self.usebutterworth, butterorder=self.butterworthorder,
                                    usetrapfftfilt=self.usetrapfftfilt, padlen=padlen, debug=self.debug))
        else:
            print("bad filter type")
            sys.exit()


# --------------------------- Window functions -------------------------------------------------
BHwindows = {}


def blackmanharris(length, debug=False):
    r"""Returns a Blackman Harris window function of the specified length.
    Once calculated, windows are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function
        :param length:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    windowfunc : 1D float array
        The window function
    """
    # return a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
    try:
        return BHwindows[str(length)]
    except KeyError:
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
    r"""Returns a Hann window function of the specified length.  Once calculated, windows
    are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function
        :param length:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    windowfunc : 1D float array
        The window function
    """
    # return 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
    try:
        return hannwindows[str(length)]
    except KeyError:
        hannwindows[str(length)] = 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
        if debug:
            print('initialized hann window for length', length)
        return hannwindows[str(length)]


hammingwindows = {}


def hamming(length, debug=False):
    #   return 0.54 - 0.46 * np.cos((np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
    r"""Returns a Hamming window function of the specified length.  Once calculated, windows
    are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function
        :param length:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    windowfunc : 1D float array
        The window function
    """
    try:
        return hammingwindows[str(length)]
    except KeyError:
        hammingwindows[str(length)] = 0.54 - 0.46 * np.cos(
            (np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
        if debug:
            print('initialized hamming window for length', length)
        return hammingwindows[str(length)]


def windowfunction(length, type='hamming', debug=False):
    r"""Returns a window function of the specified length and type.  Once calculated, windows
    are cached for speed.

    Parameters
    ----------
    length : int
        The length of the window function
         :param length:

    type : {'hamming', 'hann', 'blackmanharris'}, optional
        Window type.  Choices are 'hamming' (default), 'hann', and 'blackmanharris'.
       :param type:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    windowfunc : 1D float array
        The window function
    """
    if type == 'hamming':
        return hamming(length, debug=debug)
    elif type == 'hann':
        return hann(length, debug=debug)
    elif type == 'blackmanharris':
        return blackmanharris(length, debug=debug)
    elif type == 'None':
        return np.ones(length)
    else:
        print('illegal window function')
        sys.exit()


"""
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


def doprecalcfiltfilt(b, a, inputdata):
    return signal.filtfilt(b, a, inputdata).real


def dolpfastfiltfiltinit(Fs, cutofffreq, inputdata, order):
    [b, a] = signal.butter(order, cutofffreq / Fs)
    return fastfiltfiltinit(b, a, inputdata)


def dohpfastfiltfiltinit(Fs, cutofffreq, inputdata, order):
    [b, a] = signal.butter(order, cutofffreq / Fs, 'highpass')
    return fastfiltfiltinit(b, a, inputdata)


def dobpfastfiltfiltinit(Fs, cutofffreq_low, cutofffreq_high, inputdata, order):
    [b, a] = signal.butter(order, [cutofffreq_low / Fs, cutofffreq_high / Fs], 'bandpass')
    return fastfiltfiltinit(b, a, inputdata)

def specsplit(Fs, inputdata, bandwidth, usebutterworth=False):
    lowestfreq = Fs / (2.0 * np.shape(inputdata)[0])
    highestfreq = Fs / 2.0
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
            alldata[:, theband] = dobpfiltfilt(Fs, lowerlim, upperlim, inputdata, 2)
        else:
            alldata[:, theband] = dobpfftfilt(Fs, lowerlim, upperlim, inputdata)
        bandcenters[theband] = np.sqrt(upperlim * lowerlim)
        lowerlim = lowerlim * bandwidth
        upperlim = upperlim * bandwidth
    return bandcenters, lowestfreq, upperlim, alldata

"""
