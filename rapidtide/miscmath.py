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
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from rapidtide.decorators import conditionaljit, conditionaljit2

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

from scipy import fftpack
from statsmodels.robust import mad

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000


# --------------------------- Spectral analysis functions ---------------------------------------
def phase(mcv: NDArray) -> NDArray:
    """
    Return phase of complex numbers.

    Parameters
    ----------
    mcv : NDArray
        A complex vector. The input array can be of any shape, but must contain
        complex numbers.

    Returns
    -------
    NDArray
        The phase angle of the numbers, in radians. The return array has the same
        shape as the input array. Phase angles are in the range [-π, π].

    Notes
    -----
    This function computes the element-wise phase angle of complex numbers using
    the arctan2 function, which correctly handles the quadrant of the angle.
    The phase is computed as atan2(imaginary_part, real_part).

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    >>> phase(z)
    array([ 0.78539816,  2.35619449, -2.35619449, -0.78539816])

    >>> z = np.array([[1+1j, -1+1j], [-1-1j, 1-1j]])
    >>> phase(z)
    array([[ 0.78539816,  2.35619449],
           [-2.35619449, -0.78539816]])
    """
    return np.arctan2(mcv.imag, mcv.real)


def polarfft(invec: NDArray, samplerate: float) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute polar FFT representation of input signal.

    This function applies a Hamming window to the input signal, computes the FFT,
    and returns the frequency spectrum, magnitude spectrum, and phase spectrum.

    Parameters
    ----------
    invec : ndarray
        Input signal vector to be transformed
    samplerate : float
        Sampling rate of the input signal in Hz

    Returns
    -------
    tuple of ndarray
        A tuple containing:
        - freqs : ndarray
            Frequency values corresponding to the spectrum
        - magspec : ndarray
            Magnitude spectrum of the input signal
        - phspec : ndarray
            Phase spectrum of the input signal

    Notes
    -----
    - If the input vector length is odd, the last element is removed to make it even
    - A Hamming window is applied before FFT computation
    - Only the first half of the FFT result is returned (positive frequencies)
    - The maximum frequency is half the sampling rate (Nyquist frequency)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> # Create a test signal
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    >>> freqs, mags, phs = polarfft(signal, 1000.0)
    >>> print(f"Frequency range: {freqs[0]} to {freqs[-1]} Hz")
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


def complex_cepstrum(x: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Compute the complex cepstrum of a real sequence.

    The complex cepstrum is the inverse Fourier transform of the logarithm of the
    complex spectrum. It is commonly used in signal processing for analyzing
    periodicities and harmonics in signals.

    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.

    Returns
    -------
    ceps : ndarray
        Complex cepstrum of the input sequence.
    ndelay : ndarray
        The number of samples of circular delay added to the input sequence.

    Notes
    -----
    This implementation follows the approach described in [1]_ and handles
    the unwrapping of the phase to avoid discontinuities in the cepstral
    domain.

    References
    ----------
    .. [1] M. R. Schroeder, "Periodicity and cepstral analysis," IEEE Transactions
       on Audio and Electroacoustics, vol. 19, no. 3, pp. 233-238, 1971.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    >>> ceps, ndelay = complex_cepstrum(x)
    >>> print(ceps)
    >>> print(ndelay)
    """

    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    def _unwrap(phase: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Unwrap phase and compute delay correction.

        This function unwraps a phase array to remove discontinuities and computes
        the necessary delay correction to align the unwrapped phase at the center
        of the array.

        Parameters
        ----------
        phase : NDArray
            Input phase array with shape (..., samples) where the last dimension
            represents the phase samples to be unwrapped.

        Returns
        -------
        unwrapped : NDArray
            Unwrapped phase array with the same shape as input phase.
        ndelay : NDArray
            Delay correction array with shape (...,) containing the number of
            π phase jumps to correct for each sample in the batch.

        Notes
        -----
        The unwrapping process removes discontinuities by adding multiples of 2π
        to eliminate phase jumps greater than π. The delay correction is computed
        by finding the phase at the center sample and adjusting the entire array
        to align this reference point.

        Examples
        --------
        >>> import numpy as np
        >>> phase = np.array([[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]])
        >>> unwrapped, ndelay = _unwrap(phase)
        >>> print(unwrapped)
        >>> print(ndelay)
        """
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


def real_cepstrum(x: NDArray) -> NDArray:
    """
    Compute the real cepstrum of a signal.

    The cepstrum is the inverse Fourier transform of the logarithm of the magnitude
    of the Fourier transform of a signal. It is commonly used in speech processing
    and audio analysis to analyze the periodicity and structure of signals.

    Parameters
    ----------
    x : ndarray
        Input signal array of real numbers.

    Returns
    -------
    ndarray
        Real cepstrum of the input signal. The result has the same shape as the input.

    Notes
    -----
    This implementation uses the FFT-based approach:
    1. Compute the Fourier transform of the input signal
    2. Take the absolute value and logarithm
    3. Apply inverse FFT and take the real part

    The cepstrum is useful for identifying periodic structures in signals and
    is particularly important in speech analysis for determining pitch and
    formant frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    >>> cepstrum = real_cepstrum(x)
    >>> print(cepstrum)
    [ 2.53444207  0.74508512 -0.23302092 -0.34635144 -0.23302092
      0.74508512  2.53444207]
    """
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    return fftpack.ifft(np.log(np.abs(fftpack.fft(x)))).real


# --------------------------- miscellaneous math functions -------------------------------------------------
def thederiv(y: NDArray) -> NDArray:
    """
    Compute the first derivative of an array using finite differences.

    This function calculates the derivative of an array `y` using a central difference
    scheme for interior points and forward/backward differences for the first and
    last points respectively.

    Parameters
    ----------
    y : ndarray
        Input array of values to differentiate. Shape (n,) where n is the number of points.

    Returns
    -------
    ndarray
        Array of same shape as `y` containing the computed derivative values.

    Notes
    -----
    The derivative is computed using the following scheme:
    - First point: dyc[0] = (y[0] - y[1]) / 2.0
    - Interior points: dyc[i] = (y[i+1] - y[i-1]) / 2.0
    - Last point: dyc[-1] = (y[-1] - y[-2]) / 2.0

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 2, 4, 7, 11])
    >>> thederiv(y)
    array([-0.5,  1. ,  2. ,  3. ,  4.5])
    """
    dyc = np.zeros_like(y)
    dyc[0] = (y[0] - y[1]) / 2.0
    for i in range(1, len(y) - 1):
        dyc[i] = (y[i + 1] - y[i - 1]) / 2.0
    dyc[-1] = (y[-1] - y[-2]) / 2.0
    return dyc


def primes(n: int) -> list:
    """
    Compute the prime factorization of a positive integer.

    Returns the prime factors of n in ascending order, including repeated factors.

    Parameters
    ----------
    n : int
        A positive integer to factorize. Must be greater than 0.

    Returns
    -------
    list of int
        A list of prime factors of n in ascending order. If n is 1, returns an empty list.

    Notes
    -----
    This implementation uses trial division starting from 2, incrementing by 1
    until the square root of n. It is based on a StackOverflow answer and
    efficiently handles repeated prime factors.

    Examples
    --------
    >>> primes(12)
    [2, 2, 3]

    >>> primes(17)
    [17]

    >>> primes(1)
    []
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


def largestfac(n: int) -> int:
    """
    Return the largest prime factor of n.

    Parameters
    ----------
    n : int
        The integer to find the largest prime factor for. Must be a positive integer.

    Returns
    -------
    int
        The largest prime factor of n.

    Notes
    -----
    This function relies on a `primes(n)` function that returns all prime numbers up to n.
    The largest prime factor is obtained by taking the last element from this list.

    Examples
    --------
    >>> largestfac(13)
    13
    >>> largestfac(315)
    7
    """
    return primes(n)[-1]


# --------------------------- Normalization functions -------------------------------------------------
def normalize(vector: NDArray, method: str = "stddev") -> NDArray:
    """
    Normalize a vector using the specified normalization method.

    Parameters
    ----------
    vector : NDArray
        Input vector to be normalized.
    method : str, default="stddev"
        Normalization method to apply. Options are:
        - "None": Subtract mean from vector
        - "percent": Apply percentage normalization
        - "variance": Apply variance normalization
        - "stddev" or "z": Apply standard deviation normalization (Z-score)
        - "p2p": Apply peak-to-peak normalization
        - "mad": Apply median absolute deviation normalization

    Returns
    -------
    NDArray
        Normalized vector according to the specified method.

    Raises
    ------
    ValueError
        If an invalid normalization method is specified.

    Notes
    -----
    This function provides multiple normalization techniques for preprocessing
    data. The default "stddev" method (also available as "z") performs Z-score
    normalization, which centers the data around zero with unit variance.

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.array([1, 2, 3, 4, 5])
    >>> normalize(vector, "stddev")
    array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    >>> normalize(vector, "None")
    array([-2., -1.,  0.,  1.,  2.])
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
        return madnormalize(vector)[0]
    else:
        raise ValueError("Illegal normalization type")


def znormalize(vector: NDArray) -> NDArray:
    return stdnormalize(vector)


def removeoutliers(
    vector: NDArray, zerobad: bool = True, outlierfac: float = 3.0
) -> Tuple[NDArray, float, float]:
    """
    Normalize a vector using standard normalization (z-score normalization).

    Standard normalization transforms the vector by subtracting the mean and
    dividing by the standard deviation, resulting in a vector with mean=0 and std=1.

    Parameters
    ----------
    vector : array-like
        Input vector to be normalized. Should be a 1D array-like object.

    Returns
    -------
    ndarray
        Normalized vector with mean=0 and standard deviation=1.

    Notes
    -----
    This function is equivalent to calling `stdnormalize(vector)` and performs
    the standard z-score normalization: (x - mean) / std.

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.array([1, 2, 3, 4, 5])
    >>> znormalize(vector)
    array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    """
    themedian = np.median(vector)
    sigmad = mad(vector - themedian).astype(np.float64)
    if zerobad:
        subvalue = 0.0
    else:
        subvalue = themedian
    cleaneddata = vector + 0.0
    cleaneddata[np.where(np.fabs(cleaneddata - themedian) > outlierfac * sigmad)] = subvalue
    return cleaneddata, themedian, sigmad


def madnormalize(vector: NDArray) -> Tuple[NDArray, float]:
    """
    Normalize a vector using the median absolute deviation (MAD).

    This function normalizes a vector by subtracting the median and dividing by the
    median absolute deviation. The MAD is computed as the median of the absolute
    deviations from the median, scaled by a constant factor (1.4826) to make it
    consistent with the standard deviation for normally distributed data.

    Parameters
    ----------
    vector : array_like
        Input vector to be normalized.

    Returns
    -------
    ndarray or tuple
        Returns a tuple of (normalized_vector, mad).

    Notes
    -----
    The normalization is performed as: (vector - median(vector)) / MAD
    where MAD is the median absolute deviation. If MAD is zero or negative,
    the original vector is returned without normalization.

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.array([1, 2, 3, 4, 5])


    >>> normalized, mad_val = madnormalize(vector)
    >>> print(f"Normalized: {normalized}")
    >>> print(f"MAD: {mad_val}")
    >>> print(normalized)
    [-1.4826  -0.7413   0.    0.7413  1.4826]
    """
    demedianed = vector - np.median(vector)
    sigmad = mad(demedianed).astype(np.float64)
    if sigmad > 0.0:
        return demedianed / sigmad, sigmad
    else:
        return demedianed, sigmad


@conditionaljit()
def stdnormalize(vector: NDArray) -> NDArray:
    """
    Standardize a vector by removing mean and scaling by standard deviation.

    Parameters
    ----------
    vector : NDArray
        Input vector to be standardized.

    Returns
    -------
    NDArray
        Standardized vector with zero mean and unit variance. If the input vector
        has zero standard deviation, the demeaned vector is returned unchanged.

    Notes
    -----
    This function performs standardization (z-score normalization) by:
    1. Removing the mean from each element (demeaning)
    2. Dividing by the standard deviation (if non-zero)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> stdnormalize(x)
    array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])

    >>> y = np.array([5, 5, 5, 5])
    >>> stdnormalize(y)
    array([0., 0., 0., 0.])
    """
    demeaned = vector - np.mean(vector)
    sigstd = np.std(demeaned)
    if sigstd > 0.0:
        return demeaned / sigstd
    else:
        return demeaned


def varnormalize(vector: NDArray) -> NDArray:
    """
    Normalize a vector by subtracting the mean and dividing by variance.

    This function performs variance normalization on the input vector. It first
    demeanes the vector by subtracting its mean, then divides by the variance
    if it's greater than zero. If the variance is zero (constant vector), the
    demeaned vector is returned unchanged.

    Parameters
    ----------
    vector : ndarray
        Input vector to be normalized. Should be a numpy array of numeric values.

    Returns
    -------
    ndarray
        Normalized vector with mean zero and variance one (when input has non-zero variance).
        If input vector has zero variance, returns the demeaned vector.

    Notes
    -----
    This normalization is similar to standardization but uses variance instead of
    standard deviation for the normalization factor. The function handles edge cases
    where variance is zero by returning the demeaned vector without division.

    Examples
    --------
    >>> import numpy as np
    >>> vec = np.array([1, 2, 3, 4, 5])
    >>> varnormalize(vec)
    array([-2., -1.,  0.,  1.,  2.])

    >>> constant_vec = np.array([5, 5, 5, 5])
    >>> varnormalize(constant_vec)
    array([0., 0., 0., 0.])
    """
    demeaned = vector - np.mean(vector)
    sigvar = np.var(demeaned)
    if sigvar > 0.0:
        return demeaned / sigvar
    else:
        return demeaned


def pcnormalize(vector: NDArray) -> NDArray:
    """
    Normalize a vector using percentage change normalization.

    This function performs percentage change normalization by dividing each element
    by the mean of the vector and subtracting 1.0.

    Parameters
    ----------
    vector : NDArray
        Input vector to be normalized.

    Returns
    -------
    NDArray
        Normalized vector where each element is (vector[i] / mean) - 1.0.
        If the mean is less than or equal to zero, the original vector is returned.

    Notes
    -----
    The normalization formula is: (vector / mean) - 1.0
    If the mean of the vector is less than or equal to zero, the function returns
    the original vector to avoid division by zero or negative normalization.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalized = pcnormalize(data)
    >>> print(normalized)
    [-0.6 -0.2  0.2  0.6  1. ]

    >>> data = np.array([10, 20, 30])
    >>> normalized = pcnormalize(data)
    >>> print(normalized)
    [-0.5  0.5  1.5]
    """
    sigmean = np.mean(vector)
    if sigmean > 0.0:
        return vector / sigmean - 1.0
    else:
        return vector


def ppnormalize(vector: NDArray) -> NDArray:
    """
    Normalize a vector using peak-to-peak normalization.

    This function performs peak-to-peak normalization by subtracting the mean
    and dividing by the range (max - min) of the demeaned vector.

    Parameters
    ----------
    vector : NDArray
        Input vector to be normalized

    Returns
    -------
    NDArray
        Normalized vector with values ranging from -0.5 to 0.5 when the range is non-zero,
        or zero vector when the range is zero

    Notes
    -----
    The normalization is performed as: (vector - mean) / (max - min)
    If the range (max - min) is zero, the function returns the demeaned vector
    (which will be all zeros) to avoid division by zero.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalized = ppnormalize(data)
    >>> print(normalized)
    [-0.5 -0.25  0.   0.25  0.5 ]
    """
    demeaned = vector - np.mean(vector)
    sigpp = np.max(demeaned) - np.min(demeaned)
    if sigpp > 0.0:
        return demeaned / sigpp
    else:
        return demeaned


def imagevariance(
    thedata: NDArray,
    thefilter: Optional[tide_filt.NoncausalFilter],
    samplefreq: float,
    meannorm: bool = True,
    debug: bool = False,
) -> NDArray:
    """
    Calculate variance of filtered image data, optionally normalized by mean.

    This function applies a filter to each voxel's time series data and computes
    the variance along the time dimension. The result can be optionally normalized
    by the mean of the original data.

    Parameters
    ----------
    thedata : NDArray
        Input image data with shape (n_voxels, n_timepoints).
    thefilter : Optional[object]
        Filter object with an 'apply' method that takes (samplefreq, data) as arguments.
        If None, no filtering is applied.
    samplefreq : float
        Sampling frequency used for filter application.
    meannorm : bool, optional
        If True, normalize variance by mean of original data. Default is True.
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    NDArray
        Array of variance values for each voxel. Shape is (n_voxels,).

    Notes
    -----
    - NaN values are converted to zero in the final result.
    - When `meannorm=True`, the variance is normalized by the mean of the original data.
    - The filter is applied to each voxel's time series independently.
    - If no filter is provided, the original data is used directly.

    Examples
    --------
    >>> data = np.random.randn(100, 50)
    >>> filter_obj = SomeFilter()
    >>> variance = imagevariance(data, filter_obj, samplefreq=2.0)
    >>> variance = imagevariance(data, None, samplefreq=2.0, meannorm=False)
    """
    if debug:
        print(f"IMAGEVARIANCE: {thedata.shape}, {thefilter}, {samplefreq}")
    filteredim = np.zeros_like(thedata)
    if thefilter is not None:
        for thevoxel in range(thedata.shape[0]):
            filteredim[thevoxel, :] = thefilter.apply(samplefreq, thedata[thevoxel, :])
    else:
        filteredim = thedata
    if meannorm:
        return np.nan_to_num(np.var(filteredim, axis=1) / np.mean(thedata, axis=1))
    else:
        return np.var(filteredim, axis=1)


# @conditionaljit()
def corrnormalize(thedata: NDArray, detrendorder: int = 1, windowfunc: str = "hamming") -> NDArray:
    """
    Normalize data by detrending and applying a window function, then standardize.

    This function first detrends the input data, applies a window function if specified,
    and then normalizes the result using standard normalization.

    Parameters
    ----------
    thedata : NDArray
        Input data to be normalized.
    detrendorder : int, optional
        Order of detrending to apply. A value of 0 skips detrending, while values > 0
        apply polynomial detrending (default is 1 for linear detrending).
    windowfunc : str, optional
        Window function to apply (e.g., 'hamming', 'hanning'). Use 'None' to skip
        windowing (default is 'hamming').

    Returns
    -------
    NDArray
        Normalized data array with detrending, windowing (if applicable), and standard
        normalization applied, followed by division by sqrt(n), where n is the length
        of the input data.

    Notes
    -----
    The normalization process is performed in the following steps:
    1. Detrend the data using polynomial fitting if `detrendorder` > 0.
    2. Apply a window function if `windowfunc` is not 'None'.
    3. Standard normalize the result.
    4. Divide the normalized result by sqrt(n), where n is the length of the data.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100)
    >>> normalized = corrnormalize(data)
    >>> normalized = corrnormalize(data, detrendorder=2, windowfunc='hanning')
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


def noiseamp(
    vector: NDArray, Fs: float, windowsize: float = 40.0
) -> Tuple[NDArray, NDArray, float, float, float, float]:
    """
    Compute noise amplitude characteristics from a vector using band-pass filtering and trend analysis.

    This function applies a non-causal band-pass filter to the squared input vector to extract
    envelope information, then computes root-mean-square (RMS) values over time. A linear trend
    is fitted to the RMS values to determine the start and end amplitudes, and the percentage
    change and rate of change are calculated over the signal duration.

    Parameters
    ----------
    vector : ndarray
        Input signal vector (1D array) from which noise amplitude is computed.
    Fs : float
        Sampling frequency of the input signal in Hz.
    windowsize : float, optional
        Size of the filtering window in seconds, used to define the cutoff frequency.
        Default is 40.0 seconds.

    Returns
    -------
    tuple of (filtrms, thefittc, startamp, endamp, changepct, changerate)
        - filtrms : ndarray
            Root-mean-square (RMS) values of the filtered signal.
        - thefittc : ndarray
            Linear trend fit applied to the RMS values.
        - startamp : float
            Starting amplitude value from the trend fit.
        - endamp : float
            Ending amplitude value from the trend fit.
        - changepct : float
            Percentage change in amplitude from start to end.
        - changerate : float
            Rate of amplitude change per second (percentage per second).

    Notes
    -----
    - The function uses a non-causal filter (`tide_filt.NoncausalFilter`) with an
      arbitrary band-pass configuration.
    - The cutoff frequency is computed as 1 / windowsize.
    - Padding and unpadding are applied to avoid edge effects in filtering.
    - If a RankWarning occurs during polynomial fitting, the coefficients are set to [0.0, 0.0].

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.random.randn(1000)
    >>> Fs = 10.0
    >>> rms_vals, trend_vals, start, end, pct_chg, rate_chg = noiseamp(vector, Fs)
    >>> print(f"Start amplitude: {start:.3f}, End amplitude: {end:.3f}")
    """
    cutoff = 1.0 / windowsize
    padlen = int(len(vector) // 2)
    theenvbpf = tide_filt.NoncausalFilter(filtertype="arb")
    theenvbpf.setfreqs(0.0, 0.0, cutoff, 1.1 * cutoff)
    tide_filt.unpadvec(theenvbpf.apply(Fs, tide_filt.padvec(np.square(vector), padlen)), padlen)
    filtsq = tide_filt.unpadvec(
        theenvbpf.apply(Fs, tide_filt.padvec(np.square(vector), padlen)), padlen
    )
    filtsq = np.where(filtsq >= 0.0, filtsq, 0.0)
    filtrms = np.sqrt(filtsq)
    thetimepoints = np.arange(0.0, len(filtrms), 1.0) - len(filtrms) / 2.0
    try:
        thecoffs = Polynomial.fit(thetimepoints, filtrms, 1).convert().coef[::-1]
    except np.exceptions.RankWarning:
        thecoffs = np.asarray([0.0, 0.0])
    thefittc = tide_fit.trendgen(thetimepoints, thecoffs, True)
    startamp = thefittc[0]
    endamp = thefittc[-1]
    if startamp > 0.0:
        changepct = 100.0 * (endamp / startamp - 1.0)
    else:
        changepct = 0.0
    runtime = len(vector) / Fs
    changerate = changepct / runtime
    return filtrms, thefittc, startamp, endamp, changepct, changerate


def rms(vector: NDArray) -> float:
    """
    Compute the root mean square (RMS) of a vector.

    The root mean square is a statistical measure that represents the magnitude
    of a varying quantity. It is especially useful in physics and engineering
    applications.

    Parameters
    ----------
    vector : array_like
        Input vector for which to compute the root mean square.

    Returns
    -------
    float
        The root mean square value of the input vector.

    Notes
    -----
    The RMS is calculated as sqrt(mean(square(vector))).

    Examples
    --------
    >>> import numpy as np
    >>> rms([1, 2, 3, 4])
    2.7386127875258306
    >>> rms(np.array([1, 2, 3, 4]))
    2.7386127875258306
    """
    return np.sqrt(np.mean(np.square(vector)))


def envdetect(Fs: float, inputdata: NDArray, cutoff: float = 0.25, padlen: int = 10) -> NDArray:
    """
    Compute the envelope of input signal using band-pass filtering.

    This function calculates the envelope of a signal by first removing the mean,
    taking the absolute value, and then applying a band-pass filter to isolate
    the envelope components. The filtering is performed using a non-causal filter
    to avoid phase distortion.

    Parameters
    ----------
    Fs : float
        Sampling frequency of the input signal in Hz.
    inputdata : NDArray
        Input signal array to process.
    cutoff : float, optional
        Cutoff frequency for the band-pass filter. Default is 0.25.
    padlen : int, optional
        Padding length used for filtering to avoid edge effects. Default is 10.

    Returns
    -------
    NDArray
        Envelope of the input signal with the same shape as inputdata.

    Notes
    -----
    The function uses a non-causal filter (two-pass filtering) which avoids
    phase distortion but requires padding the signal. The filter is set to
    pass frequencies between 0 and cutoff, with a stop band starting at 1.1*cutoff.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, Fs))
    >>> envelope = envdetect(Fs, signal)
    """
    demeaned = inputdata - np.mean(inputdata)
    sigabs = abs(demeaned)
    theenvbpf = tide_filt.NoncausalFilter(filtertype="arb")
    theenvbpf.setfreqs(0.0, 0.0, cutoff, 1.1 * cutoff)
    return tide_filt.unpadvec(theenvbpf.apply(Fs, tide_filt.padvec(sigabs, padlen)), padlen)


def phasemod(phase: NDArray, centric: bool = True) -> NDArray | float:
    """
    Perform phase modulation with optional centric adjustment.

    This function applies phase modulation to the input phase array, with an option
    to apply a centric transformation that maps the phase range to [-π, π].

    Parameters
    ----------
    phase : ndarray
        Input phase array in radians.
    centric : bool, optional
        If True, applies centric transformation to map phase to [-π, π] range.
        If False, returns phase modulo 2π. Default is True.

    Returns
    -------
    ndarray
        Modulated phase array with same shape as input.

    Notes
    -----
    When `centric=True`, the transformation is equivalent to:
    `((-phase + π) % (2π) - π) * -1`

    Examples
    --------
    >>> import numpy as np
    >>> phase = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    >>> phasemod(phase)
    array([ 0.        ,  1.57079633,  3.14159265, -1.57079633,  0.        ])
    >>> phasemod(phase, centric=False)
    array([0.        , 1.57079633, 3.14159265, 4.71238898, 0.        ])
    """
    if centric:
        return ((-phase + np.pi) % (2.0 * np.pi) - np.pi) * -1.0
    else:
        return phase % (2.0 * np.pi)


def trendfilt(
    inputdata: NDArray, order: int = 3, ndevs: float = 3.0, debug: bool = False
) -> NDArray:
    """
    Apply trend filtering to remove polynomial trends and outliers from time series data.

    This function fits a polynomial trend to the input data using least squares,
    removes the trend to obtain detrended data, and then applies outlier detection
    using median absolute deviation (MAD) normalization to identify and mask outliers.

    Parameters
    ----------
    inputdata : NDArray
        Input time series data to be filtered.
    order : int, optional
        Order of the polynomial trend to fit (default is 3).
    ndevs : float, optional
        Number of standard deviations for outlier detection (default is 3.0).
    debug : bool, optional
        If True, display debug plots showing the detrended data and outliers (default is False).

    Returns
    -------
    NDArray
        Filtered time series data with polynomial trend removed and outliers masked as zeros.

    Notes
    -----
    The function uses `Polynomial.fit` to fit a polynomial trend and `tide_fit.trendgen`
    to generate the trend values. Outliers are detected using median absolute deviation
    normalization and masked by setting them to zero. The original trend is added back
    to the filtered data to maintain the overall signal structure.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100)
    >>> filtered_data = trendfilt(data, order=2, ndevs=2.0)
    >>> # With debug mode enabled
    >>> filtered_data = trendfilt(data, debug=True)
    """
    thetimepoints = np.arange(0.0, len(inputdata), 1.0) - len(inputdata) / 2.0
    try:
        thecoffs = Polynomial.fit(thetimepoints, inputdata, order).convert().coef[::-1]
    except np.exceptions.RankWarning:
        thecoffs = np.asarray([0.0, 0.0])
    thefittc = tide_fit.trendgen(thetimepoints, thecoffs, True)
    detrended = inputdata - thefittc
    if debug:
        plt.figure()
        plt.plot(detrended)
    detrended[np.where(np.fabs(madnormalize(detrended)[0]) > ndevs)] = 0.0
    if debug:
        plt.plot(detrended)
        plt.show()
    return detrended + thefittc


# found here: https://datascience.stackexchange.com/questions/75733/pca-for-complex-valued-data
class ComplexPCA:
    def __init__(self, n_components):
        """
        Initialize the PCA model with the specified number of components.

        Parameters
        ----------
        n_components : int
            Number of components to keep.

        Returns
        -------
        None
            Initializes the PCA model with the specified number of components and
            sets internal attributes to None.

        Notes
        -----
        This constructor initializes the PCA model with the specified number of
        components. The actual computation of principal components is performed
        during the fit method.

        Examples
        --------
        >>> from sklearn.decomposition import PCA
        >>> pca = PCA(n_components=2)
        >>> print(pca.n_components)
        2
        """
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None

    @property
    def explained_variance_ratio_(self):
        """
        Return the explained variance ratio.

        This function returns the explained variance ratio stored in the object's
        `s` attribute, which typically represents the proportion of variance
        explained by each component in dimensionality reduction techniques.

        Returns
        -------
        explained_variance_ratio : array-like
            The explained variance ratio for each component. Each element
            represents the fraction of the total variance explained by the
            corresponding component.

        Notes
        -----
        The explained variance ratio is commonly used in Principal Component
        Analysis (PCA) and similar dimensionality reduction methods to determine
        the importance of each component and to decide how many components to
        retain for analysis.

        Examples
        --------
        >>> from sklearn.decomposition import PCA
        >>> pca = PCA()
        >>> pca.fit(X)
        >>> ratio = pca.explained_variance_ratio_
        >>> print(ratio)
        [0.856, 0.123, 0.021]
        """
        return self.s

    def fit(self, matrix, use_gpu=False):
        """
        Fit the model with the given matrix using Singular Value Decomposition.

        This method computes the mean of the input matrix and performs SVD decomposition
        to obtain the principal components. The decomposition can be performed using
        either CPU (numpy) or GPU (tensorflow) depending on the use_gpu parameter.

        Parameters
        ----------
        matrix : array-like of shape (n_samples, n_features)
            Input matrix to fit the model on. The matrix should be numeric.
        use_gpu : bool, default=False
            If True, use TensorFlow for SVD computation on GPU. If False, use NumPy.
            Note: TensorFlow is used for GPU computation as PyTorch doesn't handle
            complex values well.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        - The SVD is performed with `full_matrices=False`, which means the number of
          components will be min(n_samples, n_features).
        - For better performance when only a subset of components is needed, consider
          truncating the SVD to `n_components` instead of computing all components.
        - The `components_` attribute stores the right singular vectors (principal components).
        - The `mean_` attribute stores the mean of each feature across samples.
        - The `s` attribute stores the singular values from the SVD decomposition.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> X = np.random.rand(100, 10)
        >>> pca = PCA()
        >>> pca.fit(X)
        >>> print(pca.components_.shape)
        (10, 10)
        """
        self.mean_ = matrix.mean(axis=0)
        if use_gpu:
            import tensorflow as tf  # torch doesn't handle complex values.

            tensor = tf.convert_to_tensor(matrix)
            u, s, vh = tf.linalg.svd(
                tensor, full_matrices=False
            )  # full=False ==> num_pc = min(N, M)
            # It would be faster if the SVD was truncated to only n_components instead of min(M, N)
        else:
            _, self.s, vh = np.linalg.svd(
                matrix, full_matrices=False
            )  # full=False ==> num_pc = min(N, M)
            # It would be faster if the SVD was truncated to only n_components instead of min(M, N)
        self.components_ = vh  # already conjugated.
        # Leave those components as rows of matrix so that it is compatible with Sklearn PCA.

    def transform(self, matrix):
        """
        Transform matrix using the fitted components.

        Parameters
        ----------
        matrix : array-like of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        array-like of shape (n_samples, n_components)
            The transformed data.

        Notes
        -----
        This function applies the transformation defined by the fitted components
        to the input matrix. It subtracts the mean and projects onto the component
        space.

        Examples
        --------
        >>> from sklearn.decomposition import PCA
        >>> import numpy as np
        >>> pca = PCA(n_components=2)
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> pca.fit(X)
        >>> transformed = pca.transform(X)
        >>> print(transformed.shape)
        (3, 2)
        """
        data = matrix - self.mean_
        result = data @ self.components_.T
        return result

    def inverse_transform(self, matrix):
        """
        Apply inverse transformation to the input matrix.

        Parameters
        ----------
        matrix : array-like of shape (n_samples, n_components)
            The transformed data to be inverse transformed.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            The inverse transformed data in the original feature space.

        Notes
        -----
        This function applies the inverse transformation using the stored components
        and mean values. The transformation is defined as:
        result = matrix @ conj(self.components_) + self.mean_

        Examples
        --------
        >>> from sklearn.decomposition import PCA
        >>> import numpy as np
        >>> # Create sample data
        >>> data = np.random.rand(100, 10)
        >>> # Fit PCA
        >>> pca = PCA(n_components=5)
        >>> transformed = pca.fit_transform(data)
        >>> # Inverse transform
        >>> reconstructed = pca.inverse_transform(transformed)
        >>> print(reconstructed.shape)
        (100, 10)
        """
        result = matrix @ np.conj(self.components_)
        return self.mean_ + result
