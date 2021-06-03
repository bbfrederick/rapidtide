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
from numba import jit
from scipy import fftpack
from statsmodels.robust import mad

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotusenumba = True
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False


# ----------------------------------------- Conditional jit handling ----------------------------------
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
        return jit(f, nopython=True)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Spectral analysis functions ---------------------------------------
def phase(mcv):
    r"""Return phase of complex numbers.

    Parameters
    ----------
    mcv : complex array
        A complex vector

    Returns
    -------
    phase : float array
        The phase angle of the numbers, in radians

    """
    return np.arctan2(mcv.imag, mcv.real)


def polarfft(invec, samplerate):
    """

    Parameters
    ----------
    invec
    samplerate

    Returns
    -------

    """
    if np.shape(invec)[0] % 2 == 1:
        thevec = invec[:-1]
    else:
        thevec = invec
    spec = fftpack.fft(tide_filt.hamming(np.shape(thevec)[0]) * thevec)[
        0 : np.shape(thevec)[0] // 2
    ]
    magspec = abs(spec)
    phspec = phase(spec)
    maxfreq = samplerate / 2.0
    freqs = np.arange(0.0, maxfreq, maxfreq / (np.shape(spec)[0]))
    return freqs, magspec, phspec


def complex_cepstrum(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay

    spectrum = fftpack.fft(x)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = fftpack.ifft(log_spectrum).real

    return ceps, ndelay


def real_cepstrum(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    return fftpack.ifft(np.log(np.abs(fftpack.fft(x)))).real


# --------------------------- miscellaneous math functions -------------------------------------------------
def thederiv(y):
    """

    Parameters
    ----------
    y

    Returns
    -------

    """
    dyc = [0.0] * len(y)
    dyc[0] = (y[0] - y[1]) / 2.0
    for i in range(1, len(y) - 1):
        dyc[i] = (y[i + 1] - y[i - 1]) / 2.0
    dyc[-1] = (y[-1] - y[-2]) / 2.0
    return dyc


def primes(n):
    """

    Parameters
    ----------
    n

    Returns
    -------

    """
    # found on stackoverflow: https://stackoverflow.com/questions/16996217/prime-factorization-list
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def largestfac(n):
    """

    Parameters
    ----------
    n

    Returns
    -------

    """
    return primes(n)[-1]


# --------------------------- Normalization functions -------------------------------------------------
def normalize(vector, method="stddev"):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    if method == "None":
        return vector - np.mean(vector)
    elif method == "percent":
        return pcnormalize(vector)
    elif method == "variance":
        return varnormalize(vector)
    elif method == "stddev" or method == "z":
        return stdnormalize(vector)
    elif method == "p2p":
        return ppnormalize(vector)
    elif method == "mad":
        return madnormalize(vector)
    else:
        raise ValueError("Illegal normalization type")


def znormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    return stdnormalize(vector)


@conditionaljit()
def madnormalize(vector, returnnormfac=False):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demedianed = vector - np.median(vector)
    sigmad = mad(demedianed).astype(np.float64)
    if sigmad > 0.0:
        if returnnormfac:
            return demedianed / sigmad, sigmad
        else:
            return demedianed / sigmad
    else:
        if returnnormfac:
            return demedianed, sigmad
        else:
            return demedianed


@conditionaljit()
def stdnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demeaned = vector - np.mean(vector)
    sigstd = np.std(demeaned)
    if sigstd > 0.0:
        return demeaned / sigstd
    else:
        return demeaned


def varnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demeaned = vector - np.mean(vector)
    sigvar = np.var(demeaned)
    if sigvar > 0.0:
        return demeaned / sigvar
    else:
        return demeaned


def pcnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    sigmean = np.mean(vector)
    if sigmean > 0.0:
        return vector / sigmean - 1.0
    else:
        return vector


def ppnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demeaned = vector - np.mean(vector)
    sigpp = np.max(demeaned) - np.min(demeaned)
    if sigpp > 0.0:
        return demeaned / sigpp
    else:
        return demeaned


@conditionaljit()
def corrnormalize(thedata, detrendorder=1, windowfunc="hamming"):
    """

    Parameters
    ----------
    thedata
    detrendorder
    windowfunc

    Returns
    -------

    """
    # detrend first
    if detrendorder > 0:
        intervec = stdnormalize(tide_fit.detrend(thedata, order=detrendorder, demean=True))
    else:
        intervec = stdnormalize(thedata)

    # then window
    if windowfunc != "None":
        return stdnormalize(
            tide_filt.windowfunction(np.shape(thedata)[0], type=windowfunc) * intervec
        ) / np.sqrt(np.shape(thedata)[0])
    else:
        return stdnormalize(intervec) / np.sqrt(np.shape(thedata)[0])


def rms(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    return np.sqrt(np.mean(np.square(vector)))


def envdetect(Fs, inputdata, cutoff=0.25):
    """

    Parameters
    ----------
    Fs : float
        Sample frequency in Hz.
    inputdata : float array
        Data to be envelope detected
    cutoff : float
        Highest possible modulation frequency

    Returns
    -------
    envelope : float array
        The envelope function

    """
    demeaned = inputdata - np.mean(inputdata)
    sigabs = abs(demeaned)
    theenvbpf = tide_filt.NoncausalFilter(filtertype="arb")
    theenvbpf.setfreqs(0.0, 0.0, cutoff, 1.1 * cutoff)
    return theenvbpf.apply(Fs, sigabs)


def phasemod(phase, centric=True):
    """

    Parameters
    ----------
    phase : array-like
        An unwrapped phase vector
    centric: boolean, optional
        Determines whether to do modulo to centric (-np.pi to np.pi) or non-centric (0 to 2 * np.pi) range

    Returns
    -------
    wrapped : array-like
        The phase vector, remapped to the range of +/-np.pi
    """
    if centric:
        return ((-phase + np.pi) % (2.0 * np.pi) - np.pi) * -1.0
    else:
        return phase % (2.0 * np.pi)


def trendfilt(inputdata, order=3, ndevs=3.0, debug=False):
    """

    Parameters
    ----------
    inputdata : array-like
        A data vector with a polynomial trend and impulsive noise

    Returns
    -------
    patched : array-like
        The input data with the impulsive noise removed
    """
    thetimepoints = np.arange(0.0, len(inputdata), 1.0) - len(inputdata) / 2.0
    try:
        thecoffs = np.polyfit(thetimepoints, inputdata, order)
    except np.lib.RankWarning:
        thecoffs = [0.0, 0.0]
    thefittc = tide_fit.trendgen(thetimepoints, thecoffs, True)
    detrended = inputdata - thefittc
    if debug:
        plt.figure()
        plt.plot(detrended)
    detrended[np.where(np.fabs(madnormalize(detrended)) > ndevs)] = 0.0
    if debug:
        plt.plot(detrended)
        plt.show()
    return detrended + thefittc
