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
"""This module contains all the filtering operations for the rapidtide
package.

"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from scipy import fftpack, ndimage, signal
from scipy.signal import savgol_filter

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False

try:
    from numba import jit
except ImportError:
    donotusenumba = True
else:
    donotusenumba = False

# hard disable numba, since it is currently broken on arm
donotusenumba = True


# ----------------------------------------- Conditional jit handling ----------------------------------
def conditionaljit():
    def resdec(f):
        global donotusenumba
        if donotusenumba:
            return f
        return jit(f, nopython=True)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Filtering functions -------------------------------------------------
# NB: No automatic padding for precalculated filters


def padvec(inputdata, padlen=20, cyclic=False, padtype="reflect"):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    Returns
    -------
    paddeddata : 1D array
        The input data, with padlen reflected points added to each end

    """
    if padlen > len(inputdata):
        print(
            "ERROR: padlen (",
            padlen,
            ") is greater than input data length (",
            len(inputdata),
            ")",
        )
        sys.exit()

    if padlen > 0:
        if cyclic:
            return np.concatenate((inputdata[-padlen:], inputdata, inputdata[0:padlen]))
        else:
            if padtype == "reflect":
                return np.concatenate(
                    (inputdata[::-1][-padlen:], inputdata, inputdata[::-1][0:padlen])
                )
            elif padtype == "zero":
                return np.concatenate(
                    (np.zeros((padlen), dtype=float), inputdata, np.zeros((padlen), dtype=float))
                )
            elif padtype == "constant":
                return np.concatenate(
                    (
                        inputdata[0] * np.ones((padlen), dtype=float),
                        inputdata,
                        inputdata[-1] * np.ones((padlen), dtype=float),
                    )
                )
            else:
                raise ValueError("Padtype must be one of 'reflect', 'zero', or 'constant'")
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
@conditionaljit()
def dolpfiltfilt(Fs, upperpass, inputdata, order, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

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
        print(
            "dolpfiltfilt - Fs, upperpass, len(inputdata), order:",
            Fs,
            upperpass,
            len(inputdata),
            order,
        )
    [b, a] = signal.butter(order, 2.0 * upperpass / Fs)
    return unpadvec(
        signal.filtfilt(b, a, padvec(inputdata, padlen=padlen, cyclic=cyclic)).real,
        padlen=padlen,
    ).astype(np.float64)


@conditionaljit()
def dohpfiltfilt(Fs, lowerpass, inputdata, order, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

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
        print(
            "dohpfiltfilt - Fs, lowerpass, len(inputdata), order:",
            Fs,
            lowerpass,
            len(inputdata),
            order,
        )
    [b, a] = signal.butter(order, 2.0 * lowerpass / Fs, "highpass")
    return unpadvec(
        signal.filtfilt(b, a, padvec(inputdata, padlen=padlen, cyclic=cyclic)).real,
        padlen=padlen,
    )


@conditionaljit()
def dobpfiltfilt(Fs, lowerpass, upperpass, inputdata, order, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

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
        print(
            "dobpfiltfilt - Fs, lowerpass, upperpass, len(inputdata), order:",
            Fs,
            lowerpass,
            upperpass,
            len(inputdata),
            order,
        )
    [b, a] = signal.butter(order, [2.0 * lowerpass / Fs, 2.0 * upperpass / Fs], "bandpass")
    return unpadvec(
        signal.filtfilt(b, a, padvec(inputdata, padlen=padlen, cyclic=cyclic)).real,
        padlen=padlen,
    )


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
        print(
            "getlpfftfunc - Fs, upperpass, len(inputdata):",
            Fs,
            upperpass,
            np.shape(inputdata)[0],
        )
    transferfunc[cutoffbin:-cutoffbin] = 0.0
    return transferfunc


@conditionaljit()
def dolpfftfilt(Fs, upperpass, inputdata, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlpfftfunc(Fs, upperpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dohpfftfilt(Fs, lowerpass, inputdata, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = 1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dobpfftfilt(Fs, lowerpass, upperpass, inputdata, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlpfftfunc(Fs, upperpass, padinputdata, debug=debug) * (
        1.0 - getlpfftfunc(Fs, lowerpass, padinputdata, debug=debug)
    )
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# - fft trapezoidal filters
@conditionaljit()
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
    transferfunc = np.ones(np.shape(inputdata), dtype="float64")
    passbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
    cutoffbin = int((upperstop / Fs) * np.shape(transferfunc)[0])
    transitionlength = cutoffbin - passbin
    if debug:
        print("getlptrapfftfunc - Fs, upperpass, upperstop:", Fs, upperpass, upperstop)
        print(
            "getlptrapfftfunc - passbin, transitionlength, cutoffbin, len(inputdata):",
            passbin,
            transitionlength,
            cutoffbin,
            inputdata.shape,
        )
    if transitionlength > 0:
        transitionvector = np.arange(1.0 * transitionlength) / transitionlength
        transferfunc[passbin:cutoffbin] = 1.0 - transitionvector
        transferfunc[-cutoffbin:-passbin] = transitionvector
    if cutoffbin > 0:
        transferfunc[cutoffbin:-cutoffbin] = 0.0
    return transferfunc


@conditionaljit()
def getlptransfunc(Fs, inputdata, upperpass=None, upperstop=None, type="brickwall", debug=False):
    if upperpass is None:
        print("getlptransfunc: upperpass must be specified")
        sys.exit()
    if debug:
        print("getlptransfunc:")
        print("\tFs:", Fs)
        print("\tnp.shape(inputdata)[0]:", np.shape(inputdata)[0])
        print("\tupperpass:", upperpass)
        print("\tupperstop:", upperstop)
        print("\ttype:", type)
    freqaxis = (
        np.linspace(0.0, 1.0, num=np.shape(inputdata)[0], endpoint=False, dtype="float64") / Fs
    )
    if type == "gaussian":
        halfloc = int(np.shape(inputdata)[0] // 2)
        sigma = upperpass / 2.35482
        transferfunc = np.zeros(np.shape(inputdata), dtype="float64")
        transferfunc[0:halfloc] = np.exp(-((freqaxis[0:halfloc]) ** 2) / (2.0 * sigma * sigma))
        transferfunc[halfloc + 1 :] = np.exp(
            -((freqaxis[halfloc + 1 :] - 1.0 / Fs) ** 2) / (2.0 * sigma * sigma)
        )
    elif type == "trapezoidal":
        if upperstop is None:
            upperstop = upperpass * 1.05
        transferfunc = np.ones(np.shape(inputdata), dtype="float64")
        passbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
        cutoffbin = int((upperstop / Fs) * np.shape(transferfunc)[0])
        transitionlength = cutoffbin - passbin
        if debug:
            print("getlptrapfftfunc - Fs, upperpass, upperstop:", Fs, upperpass, upperstop)
            print(
                "getlptrapfftfunc - passbin, transitionlength, cutoffbin, len(inputdata):",
                passbin,
                transitionlength,
                cutoffbin,
                inputdata.shape,
            )
        if transitionlength > 0:
            transitionvector = np.arange(1.0 * transitionlength) / transitionlength
            transferfunc[passbin:cutoffbin] = 1.0 - transitionvector
            transferfunc[-cutoffbin:-passbin] = transitionvector
        if cutoffbin > 0:
            transferfunc[cutoffbin:-cutoffbin] = 0.0
    elif type == "brickwall":
        transferfunc = np.ones(np.shape(inputdata), dtype=np.float64)
        cutoffbin = int((upperpass / Fs) * np.shape(transferfunc)[0])
        if debug:
            print(
                "getlpfftfunc - Fs, upperpass, len(inputdata):",
                Fs,
                upperpass,
                np.shape(inputdata)[0],
            )
        transferfunc[cutoffbin:-cutoffbin] = 0.0
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("LP Transfer function - " + type)
        plt.plot(freqaxis, transferfunc)
        plt.show()
    return transferfunc


def gethptransfunc(Fs, inputdata, lowerstop=None, lowerpass=None, type="brickwall", debug=False):
    if lowerpass is None:
        print("gethptransfunc: lowerpass must be specified")
        sys.exit()
    if type == "trapezoidal":
        transferfunc = 1.0 - getlptransfunc(
            Fs,
            inputdata,
            upperpass=lowerstop,
            upperstop=lowerpass,
            type=type,
            debug=debug,
        )
    else:
        transferfunc = 1.0 - getlptransfunc(
            Fs, inputdata, upperpass=lowerpass, type=type, debug=debug
        )
    return transferfunc


@conditionaljit()
def dolptransfuncfilt(
    Fs,
    inputdata,
    upperpass=None,
    upperstop=None,
    type="brickwall",
    padlen=20,
    cyclic=False,
    debug=False,
):
    r"""Performs an FFT filter with a gaussian lowpass transfer
    function on an input vector and returns the result.  Ends are padded to reduce transients.

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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptransfunc(
        Fs, padinputdata, upperpass=upperpass, upperstop=upperstop, type=type
    )
    if debug:
        freqaxis = (
            np.linspace(0.0, 1.0, num=np.shape(padinputdata)[0], endpoint=False, dtype="float64")
            / Fs
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("LP Transfer function - " + type + ", upperpass={:.2f}".format(upperpass))
        plt.plot(freqaxis, transferfunc)
        plt.show()
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dohptransfuncfilt(
    Fs,
    inputdata,
    lowerstop=None,
    lowerpass=None,
    type="brickwall",
    padlen=20,
    cyclic=False,
    debug=False,
):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    if lowerstop is None:
        lowerstop = lowerpass * (1.0 / 1.05)
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptransfunc(
        Fs, padinputdata, upperpass=lowerstop, upperstop=lowerpass, type=type
    )
    if debug:
        freqaxis = (
            np.linspace(0.0, 1.0, num=np.shape(padinputdata)[0], endpoint=False, dtype="float64")
            / Fs
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("HP Transfer function - " + type + ", lowerpass={:.2f}".format(lowerpass))
        plt.plot(freqaxis, transferfunc)
        plt.show()
    inputdata_trans *= 1.0 - transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dobptransfuncfilt(
    Fs,
    inputdata,
    lowerstop=None,
    lowerpass=None,
    upperpass=None,
    upperstop=None,
    type="brickwall",
    padlen=20,
    cyclic=False,
    debug=False,
):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    if lowerstop is None:
        lowerstop = lowerpass * (1.0 / 1.05)
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptransfunc(
        Fs,
        padinputdata,
        upperpass=upperpass,
        upperstop=upperstop,
        type=type,
        debug=False,
    ) * gethptransfunc(Fs, padinputdata, lowerstop=lowerstop, lowerpass=lowerpass, type=type)
    if debug:
        freqaxis = (
            np.linspace(0.0, 1.0, num=np.shape(padinputdata)[0], endpoint=False, dtype="float64")
            / Fs
        )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(
            "BP Transfer function - "
            + type
            + ", lowerpass={:.2f}, upperpass={:.2f}".format(lowerpass, upperpass)
        )
        plt.plot(freqaxis, transferfunc)
        plt.show()
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dolptrapfftfilt(Fs, upperpass, upperstop, inputdata, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dohptrapfftfilt(Fs, lowerstop, lowerpass, inputdata, padlen=20, cyclic=False, debug=False):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    transferfunc = 1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


@conditionaljit()
def dobptrapfftfilt(
    Fs,
    lowerstop,
    lowerpass,
    upperpass,
    upperstop,
    inputdata,
    padlen=20,
    cyclic=False,
    debug=False,
):
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

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

    debug : boolean, optional
        When True, internal states of the function will be printed to help debugging.
        :param debug:

    Returns
    -------
    filtereddata : 1D float array
        The filtered data
    """
    padinputdata = padvec(inputdata, padlen=padlen, cyclic=cyclic)
    inputdata_trans = fftpack.fft(padinputdata)
    if debug:
        print(
            "Fs=",
            Fs,
            " Fstopl=",
            lowerstop,
            " Fpassl=",
            lowerpass,
            " Fpassu=",
            upperpass,
            " Fstopu=",
            upperstop,
        )
    transferfunc = getlptrapfftfunc(Fs, upperpass, upperstop, padinputdata, debug=debug) * (
        1.0 - getlptrapfftfunc(Fs, lowerstop, lowerpass, padinputdata, debug=debug)
    )
    inputdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(inputdata_trans).real, padlen=padlen)


# Simple example of Wiener deconvolution in Python.
# We use a fixed SNR across all frequencies in this example.
#
# Written 2015 by Dan Stowell. Public domain.
def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR in the fourier domain"
    kernel = np.hstack(
        (kernel, np.zeros(len(signal) - len(kernel)))
    )  # zero pad the kernel to same length
    H = fftpack.fft(kernel)
    deconvolved = np.roll(
        np.real(fftpack.ifft(fftpack.fft(signal) * np.conj(H) / (H * np.conj(H) + lambd**2))),
        int(len(signal) // 2),
    )
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


def spectrum(inputdata, Fs=1.0, mode="power", trim=True):
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

    trim: bool
        If True (default) return only the positive frequency values

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

    mode : {'real', 'imag', 'complex', 'mag', 'phase', 'power'}
        The type of spectrum to return.  Legal values are 'real', 'imag', 'mag', 'phase', and 'power' (default)
        :param mode:
    """
    if trim:
        specvals = fftpack.fft(inputdata)[0 : len(inputdata) // 2]
        maxfreq = Fs / 2.0
        specaxis = np.linspace(0.0, maxfreq, len(specvals), endpoint=False)
    else:
        specvals = fftpack.fft(inputdata)
        maxfreq = Fs
        specaxis = np.linspace(0.0, maxfreq, len(specvals), endpoint=False)
    if mode == "real":
        specvals = specvals.real
    elif mode == "imag":
        specvals = specvals.imag
    elif mode == "complex":
        pass
    elif mode == "mag":
        specvals = np.absolute(specvals)
    elif mode == "phase":
        specvals = np.angle(specvals)
    elif mode == "power":
        specvals = np.sqrt(np.absolute(specvals))
    else:
        print("illegal spectrum mode")
        specvals = None
    return specaxis, specvals


def setnotchfilter(thefilter, thefreq, notchwidth=1.0):
    r"""Set notch filter - sets the filter parameters for the notch.

    Parameters
    ----------
    thefilter: NoncausalFilter function
        The filter function to use
    thefreq: float
        Frequency of the notch
    notchwidth: float
        width of the notch in Hz
    """
    thefilter.settype("arb_stop")
    thefilter.setfreqs(
        thefreq - notchwidth / 2.0,
        thefreq - notchwidth / 2.0,
        thefreq + notchwidth / 2.0,
        thefreq + notchwidth / 2.0,
    )


def harmonicnotchfilter(timecourse, Fs, Ffundamental, notchpct=1.0, debug=False):
    r"""Harmonic notch filter - removes a fundamental and its harmonics from a timecourse.

    Parameters
    ----------
    timecourse: 1D numpy array
        Input data
    Fs: float
        Sample rate
    Ffundamental: float
        Fundamental frequency to be removed from the data
    notchpct: float, optional
        Width of the notch relative to the filter frequency in percent.  Default is 1.0.
    debug: bool, optional
        Set to True for additiona information on function internals.  Default is False.

    Returns
    -------
    filteredtc: 1D numpy array
        The filtered data

    """
    # delete the fundamental and its harmonics
    filteredtc = timecourse + 0.0
    maxpass = Fs / 2.0
    if notchpct is not None:
        stopfreq = Ffundamental
        freqstep = 0.5 * Fs / len(filteredtc)
        maxharmonic = int(maxpass // stopfreq)
        if debug:
            print("highest harmonic is", maxharmonic, "(", maxharmonic * stopfreq, "Hz)")
        thenotchfilter = NoncausalFilter()
        for harmonic in range(1, maxharmonic + 1):
            notchfreq = harmonic * stopfreq
            if debug:
                print("removing harmonic at", notchfreq)
            notchwidth = np.max([notchpct * harmonic * stopfreq * 0.01, freqstep])
            if debug:
                print("\tFs:", Fs)
                print("\tstopfreq:", stopfreq)
                print("\tnotchpct:", notchpct)
                print("\tnotchwidth:", notchwidth)
                print("\tnotchfreq:", notchfreq)
                print("\tfreqstep:", freqstep)
                print("\tminfreqstep:", freqstep / notchfreq)
                print("\tbins:", int(notchwidth // freqstep))
                print()
            setnotchfilter(thenotchfilter, notchfreq, notchwidth=notchwidth)
            filteredtc = thenotchfilter.apply(Fs, filteredtc)
    return filteredtc


def savgolsmooth(data, smoothlen=101, polyorder=3):
    return savgol_filter(data, smoothlen, polyorder)


def csdfilter(obsdata, commondata, padlen=20, cyclic=False, debug=False):
    r"""Cross spectral density filter - makes a filter transfer function that preserves common frequencies.

    Parameters
    ----------
    obsdata: 1D numpy array
        Input data

    commondata: 1D numpy array
        Shared data

    padlen: int, optional
        Number of reflected points to add on each end of the input data.  Default is 20.

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends

    debug: bool, optional
        Set to True for additiona information on function internals.  Default is False.

    Returns
    -------
    filtereddata: 1D numpy array
        The filtered data

    """
    padobsdata = padvec(obsdata, padlen=padlen, cyclic=cyclic)
    padcommondata = padvec(commondata, padlen=padlen, cyclic=cyclic)
    obsdata_trans = fftpack.fft(padobsdata)
    transferfunc = np.sqrt(np.abs(fftpack.fft(padobsdata) * np.conj(fftpack.fft(padcommondata))))
    obsdata_trans *= transferfunc
    return unpadvec(fftpack.ifft(obsdata_trans).real, padlen=padlen)


@conditionaljit()
def arb_pass(
    Fs,
    inputdata,
    lowerstop,
    lowerpass,
    upperpass,
    upperstop,
    transferfunc="trapezoidal",
    butterorder=6,
    padlen=20,
    cyclic=False,
    debug=False,
):
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

    butterorder : int, optional
        Order of Butterworth filter, if used.  Default is 6.
        :param butterorder:

    padlen : int, optional
        Amount of points to reflect around each end of the input vector prior to filtering.  Default is 20.
        :param padlen:

    cyclic : bool, optional
        If True, pad by wrapping the data in a cyclic manner rather than reflecting at the ends
        :param cyclic:

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
        if transferfunc == "butterworth":
            retvec = dolpfiltfilt(
                Fs,
                upperpass,
                inputdata,
                butterorder,
                padlen=padlen,
                cyclic=False,
                debug=debug,
            )
            return retvec
        else:
            return dolptransfuncfilt(
                Fs,
                inputdata,
                upperpass=upperpass,
                upperstop=upperstop,
                type=transferfunc,
                padlen=padlen,
                cyclic=cyclic,
                debug=debug,
            )
    elif (upperpass >= Fs / 2.0) or (upperpass <= 0.0):
        # set up for highpass
        if transferfunc == "butterworth":
            return dohpfiltfilt(
                Fs,
                lowerpass,
                inputdata,
                butterorder,
                padlen=padlen,
                cyclic=False,
                debug=debug,
            )
        else:
            return dohptransfuncfilt(
                Fs,
                inputdata,
                lowerstop=lowerstop,
                lowerpass=lowerpass,
                type=transferfunc,
                padlen=padlen,
                cyclic=cyclic,
                debug=debug,
            )
    else:
        # set up for bandpass
        if transferfunc == "butterworth":
            return dohpfiltfilt(
                Fs,
                lowerpass,
                dolpfiltfilt(
                    Fs,
                    upperpass,
                    inputdata,
                    butterorder,
                    padlen=padlen,
                    cyclic=False,
                    debug=debug,
                ),
                butterorder,
                padlen=padlen,
                debug=debug,
            )
        else:
            return dobptransfuncfilt(
                Fs,
                inputdata,
                lowerstop=lowerstop,
                lowerpass=lowerpass,
                upperpass=upperpass,
                upperstop=upperstop,
                type=transferfunc,
                padlen=padlen,
                cyclic=cyclic,
                debug=debug,
            )


class Plethfilter:
    def __init_(self, Fs, Fl, Fh, order=4, attenuation=20):
        self.Fs = Fs
        self.Fh = Fh
        self.Fl = Fl
        self.attenuation = attenuation
        self.order = order
        retvec = signal.cheby2(
            self.order,
            self.attenuation,
            [self.Fl / self.Fn, self.Fh / self.Fn],
            btype="bandpass",
            analog=False,
            output="ba",
        )
        self.b = retvec[0]
        self.a = retvec[1]

    def apply(self, data):
        return signal.filtfilt(self.b, self.a, data, axis=-1, padtype="odd", padlen=None)


class NoncausalFilter:
    def __init__(
        self,
        filtertype="None",
        transitionfrac=0.05,
        transferfunc="trapezoidal",
        initlowerstop=None,
        initlowerpass=None,
        initupperpass=None,
        initupperstop=None,
        butterworthorder=6,
        correctfreq=True,
        padtime=30.0,
        cyclic=False,
        debug=False,
    ):
        r"""A zero time delay filter for one dimensional signals, especially physiological ones.

        Parameters
        ----------
        filtertype : {'None' 'vlf', 'lfo', 'resp', 'card', 'vlf_stop', 'lfo_stop', 'resp_stop', 'card_stop', 'hrv_ulf', 'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_vhf', 'hrv_ulf_stop', 'hrv_vlf_stop', 'hrv_lf_stop', 'hrv_hf_stop', 'hrv_vhf_stop', 'arb', 'arb_stop', 'ringstop'}, optional
            The type of filter.
        butterworthorder: int, optional
            Butterworth filter order.  Default is 6.
        correctfreq: boolean, optional
            Fix pass frequencies that are impossible.  Default is True.
        padtime: float, optional
            Amount of time to end pad to reduce edge effects.  Default is 30.0 seconds
        cyclic : boolean, optional
            If True, pad vectors cyclicly rather than reflecting data around the ends
        debug: boolean, optional
            Enable extended debugging messages.  Default is False.

        Methods
        -------
        settype(thetype)
            Set the filter type. Options are 'None' (default), 'vlf', 'lfo', 'resp', 'card',
            'vlf_stop', 'lfo_stop', 'resp_stop', 'card_stop',
            'hrv_ulf', 'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_vhf',
            'hrv_ulf_stop', 'hrv_vlf_stop', 'hrv_lf_stop', 'hrv_hf_stop', 'hrv_vhf_stop',
            'arb', 'arb_stop',
            'ringstop'.
        gettype()
            Return the current filter type.
        getfreqs()
            Return the current frequency limits.
        setbutterorder(order=self.butterworthorder)
            Set the order for Butterworth filter
        setpadtime(padtime)
            Set the end pad time in seconds.
        setdebug(debug)
            Turn debugging on and off with the debug flag.
        getpadtime()
            Return the current end pad time.
        setfreqs(lowerstop, lowerpass, upperpass, upperstop)
            Set the frequency parameters of the 'arb' and 'arb_stop' filter.
        """
        self.filtertype = filtertype
        self.species = "human"
        self.transitionfrac = transitionfrac
        self.transferfunc = transferfunc
        if initlowerpass is None:
            self.arb_lowerpass = 0.05
            self.arb_lowerstop = 0.9 * self.arb_lowerpass
        else:
            self.arb_lowerpass = initlowerpass
            self.arb_lowerstop = initlowerstop
        if initupperpass is None:
            self.arb_upperpass = 0.20
            self.arb_upperstop = 1.1 * self.arb_upperpass
        else:
            self.arb_upperpass = initupperpass
            self.arb_upperstop = initupperstop
        self.lowerstop = 0.0
        self.lowerpass = 0.0
        self.upperpass = -1.0
        self.upperstop = -1.0
        self.butterworthorder = butterworthorder
        self.correctfreq = correctfreq
        self.padtime = padtime
        self.cyclic = cyclic
        self.debug = debug

        self.VLF_UPPERPASS = 0.009
        self.VLF_UPPERSTOP = self.VLF_UPPERPASS * (1.0 + self.transitionfrac)

        self.LF_LOWERPASS = 0.01
        self.LF_UPPERPASS = 0.15
        self.LF_LOWERSTOP = self.LF_LOWERPASS * (1.0 - self.transitionfrac)
        self.LF_UPPERSTOP = self.LF_UPPERPASS * (1.0 + self.transitionfrac)

        self.LF_LEGACY_LOWERPASS = 0.01
        self.LF_LEGACY_UPPERPASS = 0.15
        self.LF_LEGACY_LOWERSTOP = 0.009
        self.LF_LEGACY_UPPERSTOP = 0.2

        self.RESP_LOWERPASS = 0.2
        self.RESP_UPPERPASS = 0.5
        self.RESP_LOWERSTOP = self.RESP_LOWERPASS * (1.0 - self.transitionfrac)
        self.RESP_UPPERSTOP = self.RESP_UPPERPASS * (1.0 + self.transitionfrac)

        self.CARD_LOWERPASS = 0.66
        self.CARD_UPPERPASS = 3.0
        self.CARD_LOWERSTOP = self.CARD_LOWERPASS * (1.0 - self.transitionfrac)
        self.CARD_UPPERSTOP = self.CARD_UPPERPASS * (1.0 + self.transitionfrac)

        self.HRVULF_UPPERPASS = 0.0033
        self.HRVULF_UPPERSTOP = self.HRVULF_UPPERPASS * (1.0 + self.transitionfrac)

        self.HRVVLF_LOWERPASS = 0.0033
        self.HRVVLF_UPPERPASS = 0.04
        self.HRVVLF_LOWERSTOP = self.HRVVLF_LOWERPASS * (1.0 - self.transitionfrac)
        self.HRVVLF_UPPERSTOP = self.HRVVLF_UPPERPASS * (1.0 + self.transitionfrac)

        self.HRVLF_LOWERPASS = 0.04
        self.HRVLF_UPPERPASS = 0.15
        self.HRVLF_LOWERSTOP = self.HRVLF_LOWERPASS * (1.0 - self.transitionfrac)
        self.HRVLF_UPPERSTOP = self.HRVLF_UPPERPASS * (1.0 + self.transitionfrac)

        self.HRVHF_LOWERPASS = 0.15
        self.HRVHF_UPPERPASS = 0.4
        self.HRVHF_LOWERSTOP = self.HRVHF_LOWERPASS * (1.0 - self.transitionfrac)
        self.HRVHF_UPPERSTOP = self.HRVHF_UPPERPASS * (1.0 + self.transitionfrac)

        self.HRVVHF_LOWERPASS = 0.4
        self.HRVVHF_UPPERPASS = 0.5
        self.HRVVHF_LOWERSTOP = self.HRVVHF_LOWERPASS * (1.0 - self.transitionfrac)
        self.HRVVHF_UPPERSTOP = self.HRVVHF_UPPERPASS * (1.0 + self.transitionfrac)

        self.settype(self.filtertype)

    def settype(self, thetype):
        self.filtertype = thetype
        if self.filtertype == "vlf" or self.filtertype == "vlf_stop":
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = 1.0 * self.VLF_UPPERPASS
            self.upperstop = 1.0 * self.VLF_UPPERSTOP
        elif self.filtertype == "lfo" or self.filtertype == "lfo_stop":
            self.lowerstop = 1.0 * self.LF_LOWERSTOP
            self.lowerpass = 1.0 * self.LF_LOWERPASS
            self.upperpass = 1.0 * self.LF_UPPERPASS
            self.upperstop = 1.0 * self.LF_UPPERSTOP
        elif self.filtertype == "lfo_legacy" or self.filtertype == "lfo_legacy_stop":
            self.lowerstop = 1.0 * self.LF_LEGACY_LOWERSTOP
            self.lowerpass = 1.0 * self.LF_LEGACY_LOWERPASS
            self.upperpass = 1.0 * self.LF_LEGACY_UPPERPASS
            self.upperstop = 1.0 * self.LF_LEGACY_UPPERSTOP
        elif self.filtertype == "resp" or self.filtertype == "resp_stop":
            self.lowerstop = 1.0 * self.RESP_LOWERSTOP
            self.lowerpass = 1.0 * self.RESP_LOWERPASS
            self.upperpass = 1.0 * self.RESP_UPPERPASS
            self.upperstop = 1.0 * self.RESP_UPPERSTOP
        elif self.filtertype == "cardiac" or self.filtertype == "cardiac_stop":
            self.lowerstop = 1.0 * self.CARD_LOWERSTOP
            self.lowerpass = 1.0 * self.CARD_LOWERPASS
            self.upperpass = 1.0 * self.CARD_UPPERPASS
            self.upperstop = 1.0 * self.CARD_UPPERSTOP
        elif self.filtertype == "hrv_ulf" or self.filtertype == "hrv_ulf_stop":
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = 1.0 * self.HRVULF_UPPERPASS
            self.upperstop = 1.0 * self.HRVULF_UPPERSTOP
        elif self.filtertype == "hrv_vlf" or self.filtertype == "hrv_vlf_stop":
            self.lowerstop = 1.0 * self.HRVVLF_LOWERSTOP
            self.lowerpass = 1.0 * self.HRVVLF_LOWERPASS
            self.upperpass = 1.0 * self.HRVVLF_UPPERPASS
            self.upperstop = 1.0 * self.HRVVLF_UPPERSTOP
        elif self.filtertype == "hrv_lf" or self.filtertype == "hrv_lf_stop":
            self.lowerstop = 1.0 * self.HRVLF_LOWERSTOP
            self.lowerpass = 1.0 * self.HRVLF_LOWERPASS
            self.upperpass = 1.0 * self.HRVLF_UPPERPASS
            self.upperstop = 1.0 * self.HRVLF_UPPERSTOP
        elif self.filtertype == "hrv_hf" or self.filtertype == "hrv_hf_stop":
            self.lowerstop = 1.0 * self.HRVHF_LOWERSTOP
            self.lowerpass = 1.0 * self.HRVHF_LOWERPASS
            self.upperpass = 1.0 * self.HRVHF_UPPERPASS
            self.upperstop = 1.0 * self.HRVHF_UPPERSTOP
        elif self.filtertype == "hrv_vhf" or self.filtertype == "hrv_vhf_stop":
            self.lowerstop = 1.0 * self.HRVVHF_LOWERSTOP
            self.lowerpass = 1.0 * self.HRVVHF_LOWERPASS
            self.upperpass = 1.0 * self.HRVVHF_UPPERPASS
            self.upperstop = 1.0 * self.HRVVHF_UPPERSTOP
        elif self.filtertype == "arb" or self.filtertype == "arb_stop":
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

    def setbutterorder(self, order=3):
        self.butterworthorder = order

    def setdebug(self, debug):
        self.debug = debug

    def setpadtime(self, padtime):
        self.padtime = padtime

    def getpadtime(self):
        return self.padtime

    def setcyclic(self, cyclic):
        self.cyclic = cyclic

    def getcyclic(self):
        return self.cyclic

    def settransferfunc(self, transferfunc):
        self.transferfunc = transferfunc

    def setfreqs(self, lowerstop, lowerpass, upperpass, upperstop):
        if lowerstop > lowerpass:
            print(
                "NoncausalFilter error: lowerstop (",
                lowerstop,
                ") must be <= lowerpass (",
                lowerpass,
                ")",
            )
            sys.exit()
        if upperpass > upperstop:
            print(
                "NoncausalFilter error: upperstop (",
                upperstop,
                ") must be >= upperpass (",
                upperpass,
                ")",
            )
            sys.exit()
        if (lowerpass > upperpass) and (upperpass >= 0.0):
            print(
                "NoncausalFilter error: lowerpass (",
                lowerpass,
                ") must be < upperpass (",
                upperpass,
                ")",
            )
            sys.exit()
        self.arb_lowerstop = 1.0 * lowerstop
        self.arb_lowerpass = 1.0 * lowerpass
        self.arb_upperpass = 1.0 * upperpass
        self.arb_upperstop = 1.0 * upperstop
        self.lowerstop = 1.0 * self.arb_lowerstop
        self.lowerpass = 1.0 * self.arb_lowerpass
        self.upperpass = 1.0 * self.arb_upperpass
        self.upperstop = 1.0 * self.arb_upperstop

    def getfreqs(self):
        return self.lowerstop, self.lowerpass, self.upperpass, self.upperstop

    def apply(self, Fs, data):
        r"""Apply the filter to a dataset.

        Parameters
        ----------
        Fs : float
            Sample frequency
        data : 1D float array
            The data to filter

        Returns
        -------
        filtereddata : 1D float array
            The filtered data
        """
        # if filterband is None, just return the data
        if self.filtertype == "None":
            return data

        # do some bounds checking
        nyquistlimit = 0.5 * Fs
        lowestfreq = 2.0 * Fs / np.shape(data)[0]

        # first see if entire range is out of bounds
        if self.lowerpass >= nyquistlimit:
            print(
                "NoncausalFilter error: filter lower pass ",
                self.lowerpass,
                " exceeds nyquist frequency ",
                nyquistlimit,
            )
            sys.exit()
        if self.lowerstop >= nyquistlimit:
            print(
                "NoncausalFilter error: filter lower stop ",
                self.lowerstop,
                " exceeds nyquist frequency ",
                nyquistlimit,
            )
            sys.exit()
        if -1.0 < self.upperpass <= lowestfreq:
            print(
                "NoncausalFilter error: filter upper pass ",
                self.upperpass,
                " is below minimum frequency ",
                lowestfreq,
            )
            sys.exit()
        if -1.0 < self.upperstop <= lowestfreq:
            print(
                "NoncausalFilter error: filter upper stop ",
                self.upperstop,
                " is below minimum frequency ",
                lowestfreq,
            )
            sys.exit()

        # now look for fixable errors
        if self.upperpass >= nyquistlimit:
            if self.correctfreq:
                self.upperpass = nyquistlimit
            else:
                print(
                    "NoncausalFilter error: filter upper pass ",
                    self.upperpass,
                    " exceeds nyquist frequency ",
                    nyquistlimit,
                )
                sys.exit()
        if self.upperstop > nyquistlimit:
            if self.correctfreq:
                self.upperstop = nyquistlimit
            else:
                print(
                    "NoncausalFilter error: filter upper stop ",
                    self.upperstop,
                    " exceeds nyquist frequency ",
                    nyquistlimit,
                )
                sys.exit()
        if self.lowerpass < lowestfreq:
            if self.correctfreq:
                self.lowerpass = lowestfreq
            else:
                print(
                    "NoncausalFilter error: filter lower pass ",
                    self.lowerpass,
                    " is below minimum frequency ",
                    lowestfreq,
                )
                sys.exit()
        if self.lowerstop < lowestfreq:
            if self.correctfreq:
                self.lowerstop = lowestfreq
            else:
                print(
                    "NoncausalFilter error: filter lower stop ",
                    self.lowerstop,
                    " is below minimum frequency ",
                    lowestfreq,
                )
                sys.exit()

        if self.padtime < 0.0:
            padlen = int(len(data) // 2)
        else:
            padlen = int(self.padtime * Fs)
        if self.debug:
            print("Fs=", Fs)
            print("lowerstop=", self.lowerstop)
            print("lowerpass=", self.lowerpass)
            print("upperpass=", self.upperpass)
            print("upperstop=", self.upperstop)
            print("butterworthorder=", self.butterworthorder)
            print("padtime=", self.padtime)
            print("padlen=", padlen)
            print("cyclic=", self.cyclic)

        # now do the actual filtering
        if self.filtertype == "None":
            return data
        elif self.filtertype == "ringstop":
            return arb_pass(
                Fs,
                data,
                0.0,
                0.0,
                Fs / 4.0,
                1.1 * Fs / 4.0,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                cyclic=self.cyclic,
                debug=self.debug,
            )
        elif (
            self.filtertype == "vlf"
            or self.filtertype == "lfo"
            or self.filtertype == "lfo_legacy"
            or self.filtertype == "resp"
            or self.filtertype == "cardiac"
            or self.filtertype == "hrv_ulf"
            or self.filtertype == "hrv_vlf"
            or self.filtertype == "hrv_lf"
            or self.filtertype == "hrv_hf"
            or self.filtertype == "hrv_vhf"
        ):
            return arb_pass(
                Fs,
                data,
                self.lowerstop,
                self.lowerpass,
                self.upperpass,
                self.upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                cyclic=self.cyclic,
                debug=self.debug,
            )
        elif (
            self.filtertype == "vlf_stop"
            or self.filtertype == "lfo_stop"
            or self.filtertype == "lfo_legacy_stop"
            or self.filtertype == "resp_stop"
            or self.filtertype == "cardiac_stop"
            or self.filtertype == "hrv_ulf_stop"
            or self.filtertype == "hrv_vlf_stop"
            or self.filtertype == "hrv_lf_stop"
            or self.filtertype == "hrv_hf_stop"
            or self.filtertype == "hrv_vhf_stop"
        ):
            return data - arb_pass(
                Fs,
                data,
                self.lowerstop,
                self.lowerpass,
                self.upperpass,
                self.upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                cyclic=self.cyclic,
                debug=self.debug,
            )
        elif self.filtertype == "arb":
            return arb_pass(
                Fs,
                data,
                self.arb_lowerstop,
                self.arb_lowerpass,
                self.arb_upperpass,
                self.arb_upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                cyclic=self.cyclic,
                debug=self.debug,
            )
        elif self.filtertype == "arb_stop":
            return data - arb_pass(
                Fs,
                data,
                self.arb_lowerstop,
                self.arb_lowerpass,
                self.arb_upperpass,
                self.arb_upperstop,
                transferfunc=self.transferfunc,
                butterorder=self.butterworthorder,
                padlen=padlen,
                cyclic=self.cyclic,
                debug=self.debug,
            )
        else:
            print(f"bad filter type: {self.filtertype}")
            sys.exit()


# --------------------------- FFT helper functions ---------------------------------------------
def polarfft(inputdata):
    complexxform = fftpack.fft(inputdata)
    return np.abs(complexxform), np.angle(complexxform)


def ifftfrompolar(r, theta):
    complexxform = r * np.exp(1j * theta)
    return fftpack.ifft(complexxform).real


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
        BHwindows[str(length)] = (
            a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
        )
        if debug:
            print("initialized Blackman-Harris window for length", length)
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
        hannwindows[str(length)] = 0.5 * (
            1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi)
        )
        if debug:
            print("initialized hann window for length", length)
        return hannwindows[str(length)]


hammingwindows = {}


def rect(length, L):
    thearray = np.abs(np.linspace(0, length, length, endpoint=False) - length / 2.0)
    return np.where(thearray <= L / 2.0, 1.0, 0.0)


def mRect(length, alpha=0.5, omegac=None, phi=0.0, debug=False):
    if omegac is None:
        omegac = 2.0 / length
    L = 1.0 / omegac
    indices = np.linspace(0, length, length, endpoint=False) - length / 2.0
    firstrect = rect(length, L)
    secondrect = alpha * rect(length, L * 2.0)
    costerm = np.cos(np.pi * omegac * indices + phi)
    thewindow = firstrect + secondrect * costerm
    if debug:
        plt.plot(firstrect)
        plt.plot(1.0 + secondrect * costerm)
        plt.show()
    return thewindow / np.max(thewindow)


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
            (np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi
        )
        if debug:
            print("initialized hamming window for length", length)
        return hammingwindows[str(length)]


def windowfunction(length, type="hamming", debug=False):
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
    if type == "hamming":
        return hamming(length, debug=debug)
    elif type == "hann":
        return hann(length, debug=debug)
    elif type == "blackmanharris":
        return blackmanharris(length, debug=debug)
    elif type == "None":
        return np.ones(length)
    else:
        print("illegal window function")
        sys.exit()
