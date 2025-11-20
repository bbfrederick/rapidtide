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
"""Functions for calculating correlations and similar metrics between arrays."""
import logging
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rapidtide.ffttools import optfftlen

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True


import scipy as sp
from numpy.fft import irfftn, rfftn
from scipy import fftpack, signal
from sklearn.metrics import mutual_info_score

import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.decorators import conditionaljit

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()
LGR = logging.getLogger("GENERAL")

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True


# --------------------------- Correlation functions -------------------------------------------------
def check_autocorrelation(
    corrscale: NDArray,
    thexcorr: NDArray,
    delta: float = 0.05,
    acampthresh: float = 0.1,
    aclagthresh: float = 10.0,
    displayplots: bool = False,
    detrendorder: int = 1,
    debug: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Check for autocorrelation peaks in a cross-correlation signal and fit a Gaussian to the sidelobe.

    This function identifies peaks in the cross-correlation signal and, if a significant
    sidelobe is detected (based on amplitude and lag thresholds), fits a Gaussian function
    to estimate the sidelobe's time and amplitude.

    Parameters
    ----------
    corrscale : NDArray
        Array of time lags corresponding to the cross-correlation values.
    thexcorr : NDArray
        Array of cross-correlation values.
    delta : float, optional
        Minimum distance between peaks, default is 0.05.
    acampthresh : float, optional
        Amplitude threshold for detecting sidelobes, default is 0.1.
    aclagthresh : float, optional
        Lag threshold beyond which sidelobes are ignored, default is 10.0.
    displayplots : bool, optional
        If True, display the cross-correlation plot with detected peaks, default is False.
    detrendorder : int, optional
        Order of detrending to apply to the signal, default is 1.
    debug : bool, optional
        If True, print debug information, default is False.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        A tuple containing the estimated sidelobe time and amplitude if a valid sidelobe is found,
        otherwise (None, None).

    Notes
    -----
    - The function uses `peakdetect` to find peaks in the cross-correlation.
    - A Gaussian fit is performed only if a peak is found beyond the zero-lag point and
      satisfies the amplitude and lag thresholds.
    - The fit is performed on a window around the detected sidelobe.

    Examples
    --------
    >>> corrscale = np.linspace(0, 20, 100)
    >>> thexcorr = np.exp(-0.5 * (corrscale - 5)**2 / 2) + 0.1 * np.random.rand(100)
    >>> time, amp = check_autocorrelation(corrscale, thexcorr, delta=0.1, acampthresh=0.05)
    >>> print(f"Sidelobe time: {time}, Amplitude: {amp}")
    """
    if debug:
        print("check_autocorrelation:")
        print(f"delta: {delta}")
        print(f"acampthresh: {acampthresh}")
        print(f"aclagthresh: {aclagthresh}")
        print(f"displayplots: {displayplots}")
    lookahead = 2
    if displayplots:
        print(f"check_autocorrelation: {displayplots=}")
        plt.plot(corrscale, thexcorr)
        plt.show()
    peaks = tide_fit.peakdetect(thexcorr, x_axis=corrscale, delta=delta, lookahead=lookahead)
    maxpeaks = np.asarray(peaks[0], dtype="float64")
    if len(peaks[0]) > 0:
        if debug:
            print(f"found {len(peaks[0])} peaks")
            print(peaks)
        LGR.debug(peaks)
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
                    thexcorr[sidelobeindex + numbins] > sidelobeamp / 2.0
                ):
                    numbins += 1
                sidelobewidth = (
                    corrscale[sidelobeindex + numbins] - corrscale[sidelobeindex]
                ) * 2.0
                fitstart = sidelobeindex - numbins
                fitend = sidelobeindex + numbins
                sidelobeamp, sidelobetime, sidelobewidth = tide_fit.gaussfit(
                    sidelobeamp,
                    sidelobetime,
                    sidelobewidth,
                    corrscale[fitstart : fitend + 1],
                    thexcorr[fitstart : fitend + 1],
                )

                if displayplots:
                    plt.plot(
                        corrscale[fitstart : fitend + 1],
                        thexcorr[fitstart : fitend + 1],
                        "k",
                        corrscale[fitstart : fitend + 1],
                        tide_fit.gauss_eval(
                            corrscale[fitstart : fitend + 1],
                            [sidelobeamp, sidelobetime, sidelobewidth],
                        ),
                        "r",
                    )
                    plt.show()
                return sidelobetime, sidelobeamp
    else:
        if debug:
            print("no peaks found")
    return None, None


def shorttermcorr_1D(
    data1: NDArray,
    data2: NDArray,
    sampletime: float,
    windowtime: float,
    samplestep: int = 1,
    detrendorder: int = 0,
    windowfunc: str = "hamming",
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute short-term cross-correlation between two 1D signals using sliding windows.

    This function calculates the Pearson correlation coefficient between two signals
    over short time windows, allowing for the analysis of time-varying correlations.
    The correlation is computed for overlapping windows across the input data,
    with optional detrending and windowing applied to each segment.

    Parameters
    ----------
    data1 : NDArray
        First input signal (1D array).
    data2 : NDArray
        Second input signal (1D array). Must have the same length as `data1`.
    sampletime : float
        Time interval between consecutive samples in seconds.
    windowtime : float
        Length of the sliding window in seconds.
    samplestep : int, optional
        Step size (in samples) between consecutive windows. Default is 1.
    detrendorder : int, optional
        Order of detrending to apply before correlation. 0 means no detrending.
        Default is 0.
    windowfunc : str, optional
        Window function to apply to each segment. Default is "hamming".

    Returns
    -------
    times : NDArray
        Array of time values corresponding to the center of each window.
    corrpertime : NDArray
        Array of Pearson correlation coefficients for each window.
    ppertime : NDArray
        Array of p-values associated with the correlation coefficients.

    Notes
    -----
    The function uses `tide_math.corrnormalize` for normalization and detrending
    of signal segments, and `scipy.stats.pearsonr` for computing the correlation.

    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.random.randn(1000)
    >>> data2 = np.random.randn(1000)
    >>> times, corr, pvals = shorttermcorr_1D(data1, data2, 0.1, 1.0)
    >>> print(f"Correlation at time {times[0]:.2f}: {corr[0]:.3f}")
    """
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)
    times = []
    corrpertime = []
    ppertime = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = tide_math.corrnormalize(
            data1[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        dataseg2 = tide_math.corrnormalize(
            data2[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        thepearsonresult = sp.stats.pearsonr(dataseg1, dataseg2)
        thepcorrR, thepcorrp = thepearsonresult.statistic, thepearsonresult.pvalue
        times.append(i * sampletime)
        corrpertime.append(thepcorrR)
        ppertime.append(thepcorrp)
    return (
        np.asarray(times, dtype="float64"),
        np.asarray(corrpertime, dtype="float64"),
        np.asarray(ppertime, dtype="float64"),
    )


def shorttermcorr_2D(
    data1: NDArray,
    data2: NDArray,
    sampletime: float,
    windowtime: float,
    samplestep: int = 1,
    laglimits: Optional[Tuple[float, float]] = None,
    weighting: str = "None",
    zeropadding: int = 0,
    windowfunc: str = "None",
    detrendorder: int = 0,
    compress: bool = False,
    displayplots: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Compute short-term cross-correlations between two 1D signals over sliding windows.

    This function computes the cross-correlation between two input signals (`data1` and `data2`)
    using a sliding window approach. For each window, the cross-correlation is computed and
    the peak lag and correlation coefficient are extracted. The function supports detrending,
    windowing, and various correlation weighting schemes.

    Parameters
    ----------
    data1 : NDArray
        First input signal (1D array).
    data2 : NDArray
        Second input signal (1D array). Must be of the same length as `data1`.
    sampletime : float
        Sampling interval of the input signals in seconds.
    windowtime : float
        Length of the sliding window in seconds.
    samplestep : int, optional
        Step size (in samples) for the sliding window. Default is 1.
    laglimits : Tuple[float, float], optional
        Minimum and maximum lag limits (in seconds) for peak detection.
        If None, defaults to ±windowtime/2.
    weighting : str, optional
        Type of weighting to apply during cross-correlation ('None', 'hamming', etc.).
        Default is 'None'.
    zeropadding : int, optional
        Zero-padding factor for the FFT-based correlation. Default is 0.
    windowfunc : str, optional
        Type of window function to apply ('None', 'hamming', etc.). Default is 'None'.
    detrendorder : int, optional
        Order of detrending to apply before correlation (0 = no detrend, 1 = linear, etc.).
        Default is 0.
    compress : bool, optional
        Whether to compress the correlation result. Default is False.
    displayplots : bool, optional
        Whether to display intermediate plots (e.g., correlation matrix). Default is False.

    Returns
    -------
    times : NDArray
        Array of time values corresponding to the center of each window.
    xcorrpertime : NDArray
        Array of cross-correlation functions for each window.
    Rvals : NDArray
        Correlation coefficients for each window.
    delayvals : NDArray
        Estimated time delays (lags) for each window.
    valid : NDArray
        Binary array indicating whether the peak detection was successful (1) or failed (0).

    Notes
    -----
    - The function uses `fastcorrelate` for efficient cross-correlation computation.
    - Peak detection is performed using `tide_fit.findmaxlag_gauss`.
    - If `displayplots` is True, an image of the cross-correlations is shown.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> signal1 = np.sin(2 * np.pi * 0.5 * t)
    >>> signal2 = np.sin(2 * np.pi * 0.5 * t + 0.1)
    >>> times, xcorrs, Rvals, delays, valid = shorttermcorr_2D(
    ...     signal1, signal2, sampletime=0.01, windowtime=1.0
    ... )
    """
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)

    if laglimits is not None:
        lagmin = laglimits[0]
        lagmax = laglimits[1]
    else:
        lagmin = -windowtime / 2.0
        lagmax = windowtime / 2.0

    LGR.debug(f"lag limits: {lagmin} {lagmax}")

    """dt = np.diff(time)[0]  # In days...
    fs = 1.0 / dt
    nfft = nperseg
    noverlap = (nperseg - 1)"""

    dataseg1 = tide_math.corrnormalize(
        data1[0 : 2 * halfwindow], detrendorder=detrendorder, windowfunc=windowfunc
    )
    dataseg2 = tide_math.corrnormalize(
        data2[0 : 2 * halfwindow], detrendorder=detrendorder, windowfunc=windowfunc
    )
    thexcorr = fastcorrelate(
        dataseg1, dataseg2, weighting=weighting, compress=compress, zeropadding=zeropadding
    )
    xcorrlen = np.shape(thexcorr)[0]
    xcorr_x = (
        np.arange(0.0, xcorrlen) * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    )
    xcorrpertime = []
    times = []
    Rvals = []
    delayvals = []
    valid = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = tide_math.corrnormalize(
            data1[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        dataseg2 = tide_math.corrnormalize(
            data2[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        times.append(i * sampletime)
        xcorrpertime.append(
            fastcorrelate(
                dataseg1, dataseg2, weighting=weighting, compress=compress, zeropadding=zeropadding
            )
        )
        (
            maxindex,
            thedelayval,
            theRval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        ) = tide_fit.findmaxlag_gauss(
            xcorr_x,
            xcorrpertime[-1],
            lagmin,
            lagmax,
            1000.0,
            refine=True,
            useguess=False,
            fastgauss=False,
            displayplots=False,
        )
        delayvals.append(thedelayval)
        Rvals.append(theRval)
        if failreason == 0:
            valid.append(1)
        else:
            valid.append(0)
    if displayplots:
        plt.imshow(xcorrpertime)
    return (
        np.asarray(times, dtype="float64"),
        np.asarray(xcorrpertime, dtype="float64"),
        np.asarray(Rvals, dtype="float64"),
        np.asarray(delayvals, dtype="float64"),
        np.asarray(valid, dtype="float64"),
    )


def calc_MI(x: NDArray, y: NDArray, bins: int = 50) -> float:
    """
    Calculate mutual information between two arrays.

    Parameters
    ----------
    x : array-like
        First array of data points
    y : array-like
        Second array of data points
    bins : int, optional
        Number of bins to use for histogram estimation, default is 50

    Returns
    -------
    float
        Mutual information between x and y

    Notes
    -----
    This implementation uses 2D histogram estimation followed by mutual information
    calculation. The method is based on the approach from:
    https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy/
    20505476#20505476

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> y = x + np.random.randn(1000) * 0.5
    >>> mi = calc_MI(x, y)
    >>> print(f"Mutual information: {mi:.3f}")
    """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


# @conditionaljit()
def mutual_info_2d_fast(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    bins: Tuple[NDArray, NDArray],
    sigma: float = 1,
    normalized: bool = True,
    EPS: float = 1.0e-6,
    debug: bool = False,
) -> float:
    """
    Compute (normalized) mutual information between two 1D variates from a joint histogram.

    Parameters
    ----------
    x : 1D NDArray[np.floating[Any]]
        First variable.
    y : 1D NDArray[np.floating[Any]]
        Second variable.
    bins : tuple of NDArray
        Bin edges for the histogram. The first element corresponds to `x` and the second to `y`.
    sigma : float, optional
        Sigma for Gaussian smoothing of the joint histogram. Default is 1.
    normalized : bool, optional
        If True, compute normalized mutual information as defined in [1]_. Default is True.
    EPS : float, optional
        Small constant to avoid numerical errors in logarithms. Default is 1e-6.
    debug : bool, optional
        If True, print intermediate values for debugging. Default is False.

    Returns
    -------
    float
        The computed mutual information (or normalized mutual information if `normalized=True`).

    Notes
    -----
    This function computes mutual information using a 2D histogram and Gaussian smoothing.
    The normalization follows the approach described in [1]_.

    References
    ----------
    .. [1] Colin Studholme, David John Hawkes, Derek L.G. Hill (1998).
           "Normalized entropy measure for multimodality image alignment".
           in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> y = np.random.randn(1000)
    >>> bins = (np.linspace(-3, 3, 64), np.linspace(-3, 3, 64))
    >>> mi = mutual_info_2d_fast(x, y, bins)
    >>> print(mi)
    """
    xstart = bins[0][0]
    xend = bins[0][-1]
    ystart = bins[1][0]
    yend = bins[1][-1]
    numxbins = int(len(bins[0]) - 1)
    numybins = int(len(bins[1]) - 1)
    cuts = (x >= xstart) & (x < xend) & (y >= ystart) & (y < yend)
    c = ((x[cuts] - xstart) / (xend - xstart) * numxbins).astype(np.int_)
    c += ((y[cuts] - ystart) / (yend - ystart) * numybins).astype(np.int_) * numxbins
    jh = np.bincount(c, minlength=numxbins * numybins).reshape(numxbins, numybins)

    return proc_MI_histogram(jh, sigma=sigma, normalized=normalized, EPS=EPS, debug=debug)


# @conditionaljit()
def mutual_info_2d(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    bins: Tuple[int, int],
    sigma: float = 1,
    normalized: bool = True,
    EPS: float = 1.0e-6,
    debug: bool = False,
) -> float:
    """
    Compute (normalized) mutual information between two 1D variates from a joint histogram.

    Parameters
    ----------
    x : 1D NDArray[np.floating[Any]]
        First variable.
    y : 1D NDArray[np.floating[Any]]
        Second variable.
    bins : tuple of int
        Number of bins for the histogram. The first element is the number of bins for `x`
        and the second for `y`.
    sigma : float, optional
        Sigma for Gaussian smoothing of the joint histogram. Default is 1.
    normalized : bool, optional
        If True, compute normalized mutual information as defined in [1]_. Default is True.
    EPS : float, optional
        Small constant to avoid numerical errors in logarithms. Default is 1e-6.
    debug : bool, optional
        If True, print intermediate values for debugging. Default is False.

    Returns
    -------
    float
        The computed mutual information (or normalized mutual information if `normalized=True`).

    Notes
    -----
    This function computes mutual information using a 2D histogram and Gaussian smoothing.
    The normalization follows the approach described in [1]_.

    References
    ----------
    .. [1] Colin Studholme, David John Hawkes, Derek L.G. Hill (1998).
           "Normalized entropy measure for multimodality image alignment".
           in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> y = np.random.randn(1000)
    >>> mi = mutual_info_2d(x, y)
    >>> print(mi)
    """
    jh, xbins, ybins = np.histogram2d(x, y, bins=bins)
    if debug:
        print(f"{xbins} {ybins}")

    return proc_MI_histogram(jh, sigma=sigma, normalized=normalized, EPS=EPS, debug=debug)


def proc_MI_histogram(
    jh: NDArray[np.floating[Any]],
    sigma: float = 1,
    normalized: bool = True,
    EPS: float = 1.0e-6,
    debug: bool = False,
) -> float:
    """
    Compute the mutual information (MI) between two variables from a joint histogram.

    This function calculates mutual information using the joint histogram of two variables,
    applying Gaussian smoothing and computing entropy-based MI. It supports both normalized
    and unnormalized versions of the mutual information.

    Parameters
    ----------
    jh : ndarray of shape (m, n)
        Joint histogram of two variables. Should be a 2D array of floating point values.
    sigma : float, optional
        Standard deviation for Gaussian smoothing of the joint histogram. Default is 1.0.
    normalized : bool, optional
        If True, returns normalized mutual information. If False, returns unnormalized
        mutual information. Default is True.
    EPS : float, optional
        Small constant added to the histogram to avoid numerical issues in log computation.
        Default is 1e-6.
    debug : bool, optional
        If True, prints intermediate values for debugging purposes. Default is False.

    Returns
    -------
    float
        The computed mutual information (MI) between the two variables. The value is
        positive and indicates the amount of information shared between the variables.

    Notes
    -----
    The function applies Gaussian smoothing to the joint histogram before computing
    marginal and joint entropies. The mutual information is computed as:

    .. math::
        MI = \\frac{H(X) + H(Y)}{H(X,Y)} - 1

    where :math:`H(X)`, :math:`H(Y)`, and :math:`H(X,Y)` are the marginal and joint entropies,
    respectively. If `normalized=False`, the unnormalized MI is returned instead.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> jh = np.random.rand(10, 10)
    >>> mi = proc_MI_histogram(jh, sigma=0.5, normalized=True)
    >>> print(mi)
    0.123456789
    """

    # smooth the jh with a gaussian filter of given sigma
    sp.ndimage.gaussian_filter(jh, sigma=sigma, mode="constant", output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    HX = -np.sum(s1 * np.log(s1))
    HY = -np.sum(s2 * np.log(s2))
    HXcommaY = -np.sum(jh * np.log(jh))
    # normfac = np.min([HX, HY])

    if normalized:
        mi = (HX + HY) / (HXcommaY) - 1.0
    else:
        mi = -(HXcommaY - HX - HY)
    pearson_r = (
        np.sqrt(1.0 - np.exp(-2 * mi))
        * np.sign(mi)
        * np.sqrt(2.0 * (1.0 - np.exp(-2 * (HX + HY - HXcommaY))))
        / np.sqrt(HX * HY)
    )

    if debug:
        print(f"{HX} {HY} {HXcommaY} {mi} {pearson_r}")

    return mi


# @conditionaljit
def cross_mutual_info(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    returnaxis: bool = False,
    negsteps: int = -1,
    possteps: int = -1,
    locs: Optional[NDArray] = None,
    Fs: float = 1.0,
    norm: bool = True,
    madnorm: bool = False,
    windowfunc: str = "None",
    bins: int = -1,
    prebin: bool = True,
    sigma: float = 0.25,
    fast: bool = True,
) -> Union[NDArray, Tuple[NDArray, NDArray, int]]:
    """
    Calculate cross-mutual information between two 1D arrays.

    This function computes the cross-mutual information (MI) between two signals
    `x` and `y` at various time lags or specified offsets. It supports normalization,
    windowing, and histogram smoothing for robust estimation.

    Parameters
    ----------
    x : NDArray[np.floating[Any]]
        First variable (signal).
    y : NDArray[np.floating[Any]]
        Second variable (signal). Must have length >= length of `x`.
    returnaxis : bool, optional
        If True, return the time axis along with the MI values. Default is False.
    negsteps : int, optional
        Number of negative time steps to compute MI for. If -1, uses default based on signal length.
        Default is -1.
    possteps : int, optional
        Number of positive time steps to compute MI for. If -1, uses default based on signal length.
        Default is -1.
    locs : ndarray of int, optional
        Specific time offsets at which to compute MI. If None, uses `negsteps` and `possteps`.
        Default is None.
    Fs : float, optional
        Sampling frequency. Used when `returnaxis` is True. Default is 1.0.
    norm : bool, optional
        If True, normalize the MI values. Default is True.
    madnorm : bool, optional
        If True, normalize the MI waveform by its median absolute deviation (MAD).
        Default is False.
    windowfunc : str, optional
        Name of the window function to apply to input signals before MI calculation.
        Default is "None".
    bins : int, optional
        Number of bins for the 2D histogram. If -1, automatically determined.
        Default is -1.
    prebin : bool, optional
        If True, precompute and cache the 2D histogram for all offsets.
        Default is True.
    sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel applied to the histogram.
        Default is 0.25.
    fast : bool, optional
        If True, apply speed optimizations. Default is True.

    Returns
    -------
    ndarray or tuple of ndarray
        If `returnaxis` is False:
            The set of cross-mutual information values.
        If `returnaxis` is True:
            Tuple of (time_axis, mi_values, num_values), where:
                - time_axis : ndarray of float
                    Time axis corresponding to the MI values.
                - mi_values : ndarray of float
                    Cross-mutual information values.
                - num_values : int
                    Number of MI values returned.

    Notes
    -----
    - The function normalizes input signals using detrending and optional windowing.
    - Cross-mutual information is computed using 2D histogram estimation and
      mutual information calculation.
    - If `prebin` is True, the 2D histogram is precomputed for efficiency.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(100)
    >>> y = np.random.randn(100)
    >>> mi = cross_mutual_info(x, y)
    >>> mi_axis, mi_vals, num = cross_mutual_info(x, y, returnaxis=True, Fs=10)
    """
    normx = tide_math.corrnormalize(x, detrendorder=1, windowfunc=windowfunc)
    normy = tide_math.corrnormalize(y, detrendorder=1, windowfunc=windowfunc)

    # see if we are using the default number of bins
    if bins < 1:
        bins = int(np.sqrt(len(x) / 5))
        LGR.debug(f"cross_mutual_info: bins set to {bins}")

    # find the bin locations
    if prebin:
        jh, bins0, bins1 = np.histogram2d(normx, normy, bins=(bins, bins))
        bins2d = (bins0, bins1)
    else:
        bins2d = (bins, bins)
        fast = False

    if (negsteps == -1) or (negsteps > len(normy) - 1):
        negsteps = -len(normy) + 1
    else:
        negsteps = -negsteps
    if (possteps == -1) or (possteps > len(normx) - 1):
        possteps = len(normx) - 1
    else:
        possteps = possteps
    if locs is None:
        thexmi_y = np.zeros((-negsteps + possteps + 1))
        LGR.debug(f"negsteps, possteps, len(thexmi_y): {negsteps} {possteps} {len(thexmi_y)}")
        irange = range(negsteps, possteps + 1)
    else:
        thexmi_y = np.zeros((len(locs)), dtype=np.float64)
        irange = np.asarray(locs)
    destloc = -1
    for i in irange:
        if locs is None:
            destloc = i - negsteps
        else:
            destloc += 1
        if i < 0:
            if fast:
                thexmi_y[destloc] = mutual_info_2d_fast(
                    normx[: i + len(normy)],
                    normy[-i:],
                    bins2d,
                    normalized=norm,
                    sigma=sigma,
                )
            else:
                thexmi_y[destloc] = mutual_info_2d(
                    normx[: i + len(normy)],
                    normy[-i:],
                    bins2d,
                    normalized=norm,
                    sigma=sigma,
                )
        elif i == 0:
            if fast:
                thexmi_y[destloc] = mutual_info_2d_fast(
                    normx,
                    normy,
                    bins2d,
                    normalized=norm,
                    sigma=sigma,
                )
            else:
                thexmi_y[destloc] = mutual_info_2d(
                    normx,
                    normy,
                    bins2d,
                    normalized=norm,
                    sigma=sigma,
                )
        else:
            if fast:
                thexmi_y[destloc] = mutual_info_2d_fast(
                    normx[i:],
                    normy[: len(normy) - i],
                    bins2d,
                    normalized=norm,
                    sigma=sigma,
                )
            else:
                thexmi_y[destloc] = mutual_info_2d(
                    normx[i:],
                    normy[: len(normy) - i],
                    bins2d,
                    normalized=norm,
                    sigma=sigma,
                )

    if madnorm:
        thexmi_y = tide_math.madnormalize(thexmi_y)[0]

    if returnaxis:
        if locs is None:
            thexmi_x = (
                np.linspace(0.0, len(thexmi_y) / Fs, num=len(thexmi_y), endpoint=False)
                + negsteps / Fs
            )
            return thexmi_x, thexmi_y, negsteps + 1
        else:
            thexmi_x = irange
            return thexmi_x, thexmi_y, len(thexmi_x)
    else:
        return thexmi_y


def mutual_info_to_r(themi: float, d: int = 1) -> float:
    """
    Convert mutual information to Pearson product-moment correlation.

    This function transforms mutual information values into Pearson correlation coefficients
    using the relationship derived from the assumption of joint Gaussian distributions.

    Parameters
    ----------
    themi : float
        Mutual information value (in nats) to be converted.
    d : int, default=1
        Dimensionality of the random variables. For single-dimensional variables, d=1.
        For multi-dimensional variables, d represents the number of dimensions.

    Returns
    -------
    float
        Pearson product-moment correlation coefficient corresponding to the input
        mutual information value. The result is in the range [0, 1].

    Notes
    -----
    The transformation is based on the formula:
    r = (1 - exp(-2*MI/d))^(-1/2)

    This approximation is valid under the assumption that the variables follow
    a joint Gaussian distribution. For non-Gaussian distributions, the relationship
    may not hold exactly.

    Examples
    --------
    >>> mutual_info_to_r(1.0)
    0.8416445342422313

    >>> mutual_info_to_r(2.0, d=2)
    0.9640275800758169
    """
    return np.power(1.0 - np.exp(-2.0 * themi / d), -0.5)


def delayedcorr(
    data1: NDArray, data2: NDArray, delayval: float, timestep: float
) -> Tuple[float, float]:
    return sp.stats.pearsonr(
        data1, tide_resample.timeshift(data2, delayval / timestep, 30).statistic
    )


def cepstraldelay(
    data1: NDArray, data2: NDArray, timestep: float, displayplots: bool = True
) -> float:
    """
    Calculate correlation between two datasets with a time delay applied to the second dataset.

    This function computes the Pearson correlation coefficient between two datasets,
    where the second dataset is time-shifted by a specified delay before correlation
    is calculated. The time shift is applied using the tide_resample.timeshift function.

    Parameters
    ----------
    data1 : NDArray
        First dataset for correlation calculation.
    data2 : NDArray
        Second dataset to be time-shifted and correlated with data1.
    delayval : float
        Time delay to apply to data2, specified in the same units as timestep.
    timestep : float
        Time step of the datasets, used to convert delayval to sample units.

    Returns
    -------
    Tuple[float, float]
        Pearson correlation coefficient and p-value from the correlation test.

    Notes
    -----
    The delayval is converted to sample units by dividing by timestep before
    applying the time shift. The tide_resample.timeshift function is used internally
    with a window parameter of 30.

    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.array([1, 2, 3, 4, 5])
    >>> data2 = np.array([2, 3, 4, 5, 6])
    >>> corr, p_value = delayedcorr(data1, data2, delay=1.0, timestep=0.1)
    >>> print(f"Correlation: {corr:.3f}")
    """
    ceps1, _ = tide_math.complex_cepstrum(data1)
    ceps2, _ = tide_math.complex_cepstrum(data2)
    additive_cepstrum, _ = tide_math.complex_cepstrum(data1 + data2)
    difference_cepstrum, _ = tide_math.complex_cepstrum(data1 - data2)
    residual_cepstrum = additive_cepstrum - difference_cepstrum
    if displayplots:
        tvec = timestep * np.arange(0.0, len(data1))
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title("cepstrum 1")
        ax1.set_xlabel("quefrency in seconds")
        plt.plot(tvec, ceps1.real, tvec, ceps1.imag)
        ax2 = fig.add_subplot(212)
        ax2.set_title("cepstrum 2")
        ax2.set_xlabel("quefrency in seconds")
        plt.plot(tvec, ceps2.real, tvec, ceps2.imag)
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("additive_cepstrum")
        ax1.set_xlabel("quefrency in seconds")
        plt.plot(tvec, additive_cepstrum.real)
        ax2 = fig.add_subplot(312)
        ax2.set_title("difference_cepstrum")
        ax2.set_xlabel("quefrency in seconds")
        plt.plot(tvec, difference_cepstrum)
        ax3 = fig.add_subplot(313)
        ax3.set_title("residual_cepstrum")
        ax3.set_xlabel("quefrency in seconds")
        plt.plot(tvec, residual_cepstrum.real)
        plt.show()
    return timestep * np.argmax(residual_cepstrum.real[0 : len(residual_cepstrum) // 2])


class AliasedCorrelator:
    def __init__(self, hiressignal, hires_Fs, numsteps):
        """
        Initialize the object with high-resolution signal parameters.

        Parameters
        ----------
        hiressignal : array-like
            High-resolution signal data to be processed.
        hires_Fs : float
            Sampling frequency of the high-resolution signal in Hz.
        numsteps : int
            Number of steps for signal processing.

        Returns
        -------
        None
            This method initializes the object attributes and does not return any value.

        Notes
        -----
        This constructor sets up the basic configuration for high-resolution signal processing
        by storing the sampling frequency and number of steps, then calls sethiressignal()
        to process the input signal.

        Examples
        --------
        >>> obj = MyClass(hiressignal, hires_Fs=44100, numsteps=100)
        >>> obj.hires_Fs
        44100
        >>> obj.numsteps
        100
        """
        self.hires_Fs = hires_Fs
        self.numsteps = numsteps
        self.sethiressignal(hiressignal)

    def sethiressignal(self, hiressignal):
        """
        Set high resolution signal and compute related parameters.

        This method processes the high resolution signal by normalizing it and computing
        correlation-related parameters including correlation length and correlation x-axis.

        Parameters
        ----------
        hiressignal : array-like
            High resolution signal data to be processed and normalized.

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return a value.

        Notes
        -----
        The method performs correlation normalization using `tide_math.corrnormalize` and
        computes the correlation length as `len(self.hiressignal) * 2 + 1`. The correlation
        x-axis is computed based on the sampling frequency (`self.hires_Fs`) and the length
        of the high resolution signal.

        Examples
        --------
        >>> obj.sethiressignal(hiressignal_data)
        >>> print(obj.corrlen)
        1001
        >>> print(obj.corrx.shape)
        (1001,)
        """
        self.hiressignal = tide_math.corrnormalize(hiressignal)
        self.corrlen = len(self.hiressignal) * 2 + 1
        self.corrx = (
            np.linspace(0.0, self.corrlen, num=self.corrlen) / self.hires_Fs
            - len(self.hiressignal) / self.hires_Fs
        )

    def getxaxis(self):
        """
        Return the x-axis correction value.

        This method retrieves the correction value applied to the x-axis.

        Returns
        -------
        float or int
            The correction value for the x-axis stored in `self.corrx`.

        Notes
        -----
        The returned value represents the x-axis correction that has been
        previously computed or set in the object's `corrx` attribute.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.corrx = 5.0
        >>> obj.getxaxis()
        5.0
        """
        return self.corrx

    def apply(self, loressignal, offset, debug=False):
        """
        Apply correlator to aliased waveform.

        NB: Assumes the highres frequency is an integral multiple of the lowres frequency

        Parameters
        ----------
        loressignal : 1D array
            The aliased waveform to match
        offset : int
            Integer offset to apply to the upsampled lowressignal (to account for slice time offset)
        debug : bool, optional
            Whether to print diagnostic information

        Returns
        -------
        corrfunc : 1D array
            The full correlation function

        Notes
        -----
        This function applies a correlator to an aliased waveform by:
        1. Creating an upsampled version of the high-resolution signal
        2. Inserting the low-resolution signal at the specified offset
        3. Computing the cross-correlation between the two signals
        4. Normalizing the result by the square root of the number of steps

        Examples
        --------
        >>> result = correlator.apply(signal, offset=5, debug=True)
        >>> print(result.shape)
        (len(highres_signal),)
        """
        if debug:
            print(offset, self.numsteps)
        osvec = np.zeros_like(self.hiressignal)
        osvec[offset :: self.numsteps] = loressignal[:]
        corrfunc = fastcorrelate(
            tide_math.corrnormalize(osvec), self.hiressignal
        ) * np.sqrt(self.numsteps)
        return corrfunc


def matchsamplerates(
    input1: NDArray,
    Fs1: float,
    input2: NDArray,
    Fs2: float,
    method: str = "univariate",
    debug: bool = False,
) -> Tuple[NDArray, NDArray, float]:
    """
    Match sampling rates of two input arrays by upsampling the lower sampling rate signal.

    This function takes two input arrays with potentially different sampling rates and
    ensures they have the same sampling rate by upsampling the signal with the lower
    sampling rate to match the higher one. The function preserves the original data
    while adjusting the sampling rate for compatibility.

    Parameters
    ----------
    input1 : NDArray
        First input array to be processed.
    Fs1 : float
        Sampling frequency of the first input array (Hz).
    input2 : NDArray
        Second input array to be processed.
    Fs2 : float
        Sampling frequency of the second input array (Hz).
    method : str, optional
        Resampling method to use, by default "univariate".
        See `tide_resample.upsample` for available methods.
    debug : bool, optional
        Enable debug output, by default False.

    Returns
    -------
    Tuple[NDArray, NDArray, float]
        Tuple containing:
        - matchedinput1: First input array upsampled to match the sampling rate
        - matchedinput2: Second input array upsampled to match the sampling rate
        - corrFs: The common sampling frequency used for both outputs

    Notes
    -----
    - If sampling rates are equal, no upsampling is performed
    - The function always upsamples to the higher sampling rate
    - The upsampling is performed using the `tide_resample.upsample` function
    - Both output arrays will have the same length and sampling rate

    Examples
    --------
    >>> import numpy as np
    >>> input1 = np.array([1, 2, 3, 4])
    >>> input2 = np.array([5, 6, 7])
    >>> Fs1 = 10.0
    >>> Fs2 = 5.0
    >>> matched1, matched2, common_fs = matchsamplerates(input1, Fs1, input2, Fs2)
    >>> print(common_fs)
    10.0
    """
    if Fs1 > Fs2:
        corrFs = Fs1
        matchedinput1 = input1
        matchedinput2 = tide_resample.upsample(input2, Fs2, corrFs, method=method, debug=debug)
    elif Fs2 > Fs1:
        corrFs = Fs2
        matchedinput1 = tide_resample.upsample(input1, Fs1, corrFs, method=method, debug=debug)
        matchedinput2 = input2
    else:
        corrFs = Fs2
        matchedinput1 = input1
        matchedinput2 = input2
    return matchedinput1, matchedinput2, corrFs


def arbcorr(
    input1: NDArray,
    Fs1: float,
    input2: NDArray,
    Fs2: float,
    start1: float = 0.0,
    start2: float = 0.0,
    windowfunc: str = "hamming",
    method: str = "univariate",
    debug: bool = False,
) -> Tuple[NDArray, NDArray, float, int]:
    """
    Compute the cross-correlation between two signals with arbitrary sampling rates.

    This function performs cross-correlation between two input signals after
    matching their sampling rates. It applies normalization and uses FFT-based
    convolution for efficient computation. The result includes the time lag axis,
    cross-correlation values, the matched sampling frequency, and the index of
    the zero-lag point.

    Parameters
    ----------
    input1 : NDArray
        First input signal array.
    Fs1 : float
        Sampling frequency of the first signal (Hz).
    input2 : NDArray
        Second input signal array.
    Fs2 : float
        Sampling frequency of the second signal (Hz).
    start1 : float, optional
        Start time of the first signal (default is 0.0).
    start2 : float, optional
        Start time of the second signal (default is 0.0).
    windowfunc : str, optional
        Window function used for normalization (default is "hamming").
    method : str, optional
        Method used for matching sampling rates (default is "univariate").
    debug : bool, optional
        If True, enables debug logging (default is False).

    Returns
    -------
    tuple
        A tuple containing:
        - thexcorr_x : NDArray
            Time lag axis for the cross-correlation (seconds).
        - thexcorr_y : NDArray
            Cross-correlation values.
        - corrFs : float
            Matched sampling frequency used for the computation (Hz).
        - zeroloc : int
            Index corresponding to the zero-lag point in the cross-correlation.

    Notes
    -----
    - The function upsamples the signals to the higher of the two sampling rates.
    - Normalization is applied using a detrend order of 1 and the specified window function.
    - The cross-correlation is computed using FFT convolution for efficiency.
    - The zero-lag point is determined as the index of the minimum absolute value in the time axis.

    Examples
    --------
    >>> import numpy as np
    >>> signal1 = np.random.randn(1000)
    >>> signal2 = np.random.randn(1000)
    >>> lags, corr_vals, fs, zero_idx = arbcorr(signal1, 10.0, signal2, 10.0)
    >>> print(f"Zero-lag index: {zero_idx}")
    """
    # upsample to the higher frequency of the two
    matchedinput1, matchedinput2, corrFs = matchsamplerates(
        input1,
        Fs1,
        input2,
        Fs2,
        method=method,
        debug=debug,
    )
    norm1 = tide_math.corrnormalize(matchedinput1, detrendorder=1, windowfunc=windowfunc)
    norm2 = tide_math.corrnormalize(matchedinput2, detrendorder=1, windowfunc=windowfunc)
    thexcorr_y = signal.fftconvolve(norm1, norm2[::-1], mode="full")
    thexcorr_x = (
        np.linspace(0.0, len(thexcorr_y) / corrFs, num=len(thexcorr_y), endpoint=False)
        - (len(norm1) // 2 + len(norm2) // 2) / corrFs
        + start1
        - start2
    )
    zeroloc = int(np.argmin(np.fabs(thexcorr_x)))
    LGR.debug(f"len(norm1) = {len(norm1)}")
    LGR.debug(f"len(norm2) = {len(norm2)}")
    LGR.debug(f"len(thexcorr_y) = {len(thexcorr_y)}")
    LGR.debug(f"zeroloc = {zeroloc}")
    return thexcorr_x, thexcorr_y, corrFs, zeroloc


def faststcorrelate(
    input1: NDArray,
    input2: NDArray,
    windowtype: str = "hann",
    nperseg: int = 32,
    weighting: str = "None",
    displayplots: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Perform correlation between short-time Fourier transformed arrays.

    This function computes the short-time cross-correlation between two input signals
    using their short-time Fourier transforms (STFTs). It applies a windowing function
    to each signal, computes the STFT, and then performs correlation in the frequency
    domain before inverse transforming back to the time domain. The result is normalized
    by the auto-correlation of each signal.

    Parameters
    ----------
    input1 : ndarray
        First input signal array.
    input2 : ndarray
        Second input signal array.
    windowtype : str, optional
        Type of window to apply. Default is 'hann'.
    nperseg : int, optional
        Length of each segment for STFT. Default is 32.
    weighting : str, optional
        Weighting method for the STFT. Default is 'None'.
    displayplots : bool, optional
        If True, display plots (not implemented in current version). Default is False.

    Returns
    -------
    corrtimes : ndarray
        Time shifts corresponding to the correlation results.
    times : ndarray
        Time indices of the STFT.
    stcorr : ndarray
        Short-time cross-correlation values.

    Notes
    -----
    The function uses `scipy.signal.stft` to compute the short-time Fourier transform
    of both input signals. The correlation is computed in the frequency domain and
    normalized by the square root of the auto-correlation of each signal.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> t = np.linspace(0, 1, 100)
    >>> x1 = np.sin(2 * np.pi * 5 * t)
    >>> x2 = np.sin(2 * np.pi * 5 * t + 0.1)
    >>> corrtimes, times, corr = faststcorrelate(x1, x2)
    >>> print(corr.shape)
    (32, 100)
    """
    nfft = nperseg
    noverlap = nperseg - 1
    onesided = False
    freqs, times, thestft1 = signal.stft(
        input1,
        fs=1.0,
        window=windowtype,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="linear",
        return_onesided=onesided,
        boundary="even",
        padded=True,
        axis=-1,
    )

    freqs, times, thestft2 = signal.stft(
        input2,
        fs=1.0,
        window=windowtype,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="linear",
        return_onesided=onesided,
        boundary="even",
        padded=True,
        axis=-1,
    )

    acorrfft1 = thestft1 * np.conj(thestft1)
    acorrfft2 = thestft2 * np.conj(thestft2)
    acorr1 = np.roll(fftpack.ifft(acorrfft1, axis=0).real, nperseg // 2, axis=0)[nperseg // 2, :]
    acorr2 = np.roll(fftpack.ifft(acorrfft2, axis=0).real, nperseg // 2, axis=0)[nperseg // 2, :]
    normfacs = np.sqrt(acorr1 * acorr2)
    product = thestft1 * np.conj(thestft2)
    stcorr = np.roll(fftpack.ifft(product, axis=0).real, nperseg // 2, axis=0)
    for i in range(len(normfacs)):
        stcorr[:, i] /= normfacs[i]

    timestep = times[1] - times[0]
    corrtimes = np.linspace(
        -timestep * (nperseg // 2),
        timestep * (nperseg // 2),
        num=nperseg,
        endpoint=False,
    )

    return corrtimes, times, stcorr


def primefacs(thelen: int) -> list:
    """
    Compute the prime factorization of a given integer.

    Parameters
    ----------
    thelen : int
        The positive integer to factorize. Must be greater than 0.

    Returns
    -------
    list
        A list of prime factors of `thelen`, sorted in ascending order.
        Each factor appears as many times as its multiplicity in the
        prime factorization.

    Notes
    -----
    This function implements trial division algorithm to find prime factors.
    The algorithm starts with the smallest prime (2) and continues with
    increasing integers until the square root of the remaining number.
    The final remaining number (if greater than 1) is also a prime factor.

    Examples
    --------
    >>> primefacs(12)
    [2, 2, 3]

    >>> primefacs(17)
    [17]

    >>> primefacs(100)
    [2, 2, 5, 5]
    """
    i = 2
    factors = []
    while i * i <= thelen:
        if thelen % i:
            i += 1
        else:
            factors.append(i)
            thelen //= i
    factors.append(thelen)
    return factors


def fastcorrelate(
    input1: NDArray,
    input2: NDArray,
    usefft: bool = True,
    zeropadding: int = 0,
    weighting: str = "None",
    compress: bool = False,
    displayplots: bool = False,
    debug: bool = False,
) -> NDArray:
    """
    Perform a fast correlation between two arrays.

    This function computes the cross-correlation of two input arrays, with options
    for using FFT-based convolution or direct correlation, as well as padding and
    weighting schemes.

    Parameters
    ----------
    input1 : ndarray
        First input array to correlate.
    input2 : ndarray
        Second input array to correlate.
    usefft : bool, optional
        If True, use FFT-based convolution for faster computation. Default is True.
    zeropadding : int, optional
        Zero-padding length. If 0, no padding is applied. If negative, automatic
        padding is applied. If positive, explicit padding is applied. Default is 0.
    weighting : str, optional
        Type of weighting to apply. If "None", no weighting is applied. Default is "None".
    compress : bool, optional
        If True and `weighting` is not "None", compress the result. Default is False.
    displayplots : bool, optional
        If True, display plots of padded inputs and correlation result. Default is False.
    debug : bool, optional
        If True, enable debug output. Default is False.

    Returns
    -------
    ndarray
        The cross-correlation of `input1` and `input2`. The length of the output is
        `len(input1) + len(input2) - 1`.

    Notes
    -----
    This implementation is based on the method described at:
    http://stackoverflow.com/questions/12323959/fast-cross-correlation-method-in-python

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([0, 1, 0])
    >>> result = fastcorrelate(a, b)
    >>> print(result)
    [0. 1. 2. 3. 0.]
    """
    len1 = len(input1)
    len2 = len(input2)
    outlen = len1 + len2 - 1
    if zeropadding < 0:
        # autopad
        newlen1 = optfftlen(len1 * 2)
        newlen2 = optfftlen(len2 * 2)
        zp1 = newlen1 - len1
        zp2 = newlen2 - len2
        paddedinput1 = np.zeros((newlen1), dtype=float)
        paddedinput2 = np.zeros((newlen2), dtype=float)
        paddedinput1[0:len1] = input1
        paddedinput2[0:len2] = input2
        startpt = (zp1 + zp2) // 2
    elif zeropadding > 0:
        # explicit pad
        newlen1 = len1 + zeropadding
        newlen2 = len2 + zeropadding
        paddedinput1 = np.zeros((newlen1), dtype=float)
        paddedinput2 = np.zeros((newlen2), dtype=float)
        # paddedinput1[0:len1] = input1
        paddedinput1[-len1:] = input1
        paddedinput2[0:len2] = input2
        startpt = zeropadding
    else:
        # no pad
        paddedinput1 = input1
        paddedinput2 = input2
        startpt = 0
    if displayplots:
        print(f"FASTCORRELATE - padding: {zeropadding}, startpt: {startpt}, outlen: {outlen}")
        plt.plot(paddedinput1 + 1.0)
        plt.plot(paddedinput2)
        plt.legend(
            [
                "Padded timecourse 1",
                "Padded timecourse 2",
            ]
        )

        plt.show()

    if usefft:
        # Do an array flipped convolution, which is a correlation.
        if weighting == "None":
            theweightedcorr = signal.fftconvolve(paddedinput1, paddedinput2[::-1], mode="full")
        else:
            theweightedcorr = convolve_weighted_fft(
                paddedinput1,
                paddedinput2[::-1],
                mode="full",
                weighting=weighting,
                compress=compress,
                displayplots=displayplots,
            )
        if displayplots:
            plt.plot(theweightedcorr)
            plt.legend(
                [
                    "Untrimmed correlation",
                ]
            )
            plt.show()

        return theweightedcorr[startpt : startpt + outlen]
    else:
        return np.correlate(paddedinput1, paddedinput2, mode="full")


def _centered(arr: NDArray, newsize: Union[int, NDArray]) -> NDArray:
    """
    Extract a centered subset of an array.

    Parameters
    ----------
    arr : array_like
        Input array from which to extract the centered subset.
    newsize : int or array_like
        The size of the output array. If int, the same size is used for all dimensions.
        If array_like, specifies the size for each dimension.

    Returns
    -------
    ndarray
        Centered subset of the input array with the specified size.

    Notes
    -----
    The function extracts a subset from the center of the input array. If the requested
    size is larger than the current array size in any dimension, the result will be
    padded with zeros (or the array will be truncated from the center).

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(24).reshape(4, 6)
    >>> _centered(arr, (2, 3))
    array([[ 7,  8,  9],
           [13, 14, 15]])
    """
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _check_valid_mode_shapes(shape1: Tuple, shape2: Tuple) -> None:
    """
    Check that shape1 is valid for 'valid' mode convolution with shape2.

    Parameters
    ----------
    shape1 : Tuple
        First shape tuple to compare
    shape2 : Tuple
        Second shape tuple to compare

    Returns
    -------
    None
        This function does not return anything but raises ValueError if condition is not met

    Notes
    -----
    This function is used to validate that the first shape has at least as many
    elements as the second shape in every dimension, which is required for
    'valid' mode convolution operations.

    Examples
    --------
    >>> _check_valid_mode_shapes((10, 10), (5, 5))
    >>> _check_valid_mode_shapes((10, 10), (10, 5))
    >>> _check_valid_mode_shapes((5, 5), (10, 5))
    Traceback (most recent call last):
        ...
    ValueError: in1 should have at least as many items as in2 in every dimension for 'valid' mode.
    """
    for d1, d2 in zip(shape1, shape2):
        if not d1 >= d2:
            raise ValueError(
                "in1 should have at least as many items as in2 in "
                "every dimension for 'valid' mode."
            )


def convolve_weighted_fft(
    in1: NDArray[np.floating[Any]],
    in2: NDArray[np.floating[Any]],
    mode: str = "full",
    weighting: str = "None",
    compress: bool = False,
    displayplots: bool = False,
) -> NDArray[np.floating[Any]]:
    """
    Convolve two N-dimensional arrays using FFT with optional weighting.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument. This is generally much
    faster than `convolve` for large arrays (n > ~500), but can be slower when
    only a few output values are needed. The function supports both real and
    complex inputs, and allows for optional weighting and compression of the
    FFT operations.

    Parameters
    ----------
    in1 : NDArray[np.floating[Any]]
        First input array.
    in2 : NDArray[np.floating[Any]]
        Second input array. Should have the same number of dimensions as `in1`;
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
    weighting : str, optional
        Type of weighting to apply during convolution. Default is "None".
        Other options may include "uniform", "gaussian", etc., depending on
        implementation of `gccproduct`.
    compress : bool, optional
        If True, compress the FFT data during computation. Default is False.
    displayplots : bool, optional
        If True, display intermediate plots during computation. Default is False.

    Returns
    -------
    out : NDArray[np.floating[Any]]
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`. The shape of the output depends on
        the `mode` parameter.

    Notes
    -----
    - This function uses real FFT (`rfftn`) for real inputs and standard FFT
      (`fftpack.fftn`) for complex inputs.
    - The convolution is computed in the frequency domain using the product
      of FFTs of the inputs.
    - For real inputs, the result is scaled to preserve the maximum amplitude.
    - The `gccproduct` function is used internally to compute the product
      of the FFTs with optional weighting.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[1, 0], [0, 1]])
    >>> result = convolve_weighted_fft(a, b)
    >>> print(result)
    [[1. 2.]
     [3. 4.]]
    """
    # if np.isscalar(in1) and np.isscalar(in2):  # scalar inputs
    #    return in1 * in2
    if not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same rank")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = np.issubdtype(in1.dtype, complex) or np.issubdtype(in2.dtype, complex)
    size = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Always use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    if not complex_result:
        fft1 = rfftn(in1, fsize)
        fft2 = rfftn(in2, fsize)
        theorigmax = np.max(
            np.absolute(irfftn(gccproduct(fft1, fft2, "None", compress=compress), fsize)[fslice])
        )
        ret = irfftn(
            gccproduct(fft1, fft2, weighting, compress=compress, displayplots=displayplots), fsize
        )[fslice].copy()
        ret = irfftn(
            gccproduct(fft1, fft2, weighting, compress=compress, displayplots=displayplots), fsize
        )[fslice].copy()
        ret = ret.real
        ret *= theorigmax / np.max(np.absolute(ret))
    else:
        fft1 = fftpack.fftn(in1, fsize)
        fft2 = fftpack.fftn(in2, fsize)
        theorigmax = np.max(
            np.absolute(fftpack.ifftn(gccproduct(fft1, fft2, "None", compress=compress))[fslice])
        )
        ret = fftpack.ifftn(
            gccproduct(fft1, fft2, weighting, compress=compress, displayplots=displayplots)
        )[fslice].copy()
        ret *= theorigmax / np.max(np.absolute(ret))

    # scale to preserve the maximum

    if mode == "full":
        retval = ret
    elif mode == "same":
        retval = _centered(ret, s1)
    elif mode == "valid":
        retval = _centered(ret, s1 - s2 + 1)

    return retval


def gccproduct(
    fft1: NDArray,
    fft2: NDArray,
    weighting: str,
    threshfrac: float = 0.1,
    compress: bool = False,
    displayplots: bool = False,
) -> NDArray:
    """
    Compute the generalized cross-correlation (GCC) product with optional weighting.

    This function computes the GCC product of two FFT arrays, applying a specified
    weighting scheme to enhance correlation performance. It supports several weighting
    methods including 'liang', 'eckart', 'phat', and 'regressor'. The result can be
    thresholded and optionally compressed to improve visualization and reduce noise.

    Parameters
    ----------
    fft1 : NDArray
        First FFT array (complex-valued).
    fft2 : NDArray
        Second FFT array (complex-valued).
    weighting : str
        Weighting method to apply. Options are:
        - 'liang': Liang weighting
        - 'eckart': Eckart weighting
        - 'phat': PHAT (Phase Transform) weighting
        - 'regressor': Regressor-based weighting (uses fft2 as reference)
        - 'None': No weighting applied.
    threshfrac : float, optional
        Threshold fraction used to determine the minimum value for output masking.
        Default is 0.1.
    compress : bool, optional
        If True, compress the weighting function using 10th and 90th percentiles.
        Default is False.
    displayplots : bool, optional
        If True, display the reciprocal weighting function as a plot.
        Default is False.

    Returns
    -------
    NDArray
        The weighted GCC product. The output is of the same shape as the input arrays.
        If `weighting` is 'None', the raw product is returned.
        If `threshfrac` is 0, a zero array of the same shape is returned.

    Notes
    -----
    The weighting functions are applied element-wise and are designed to suppress
    noise and enhance correlation peaks. The 'phat' weighting is commonly used in
    speech and signal processing due to its robustness.

    Examples
    --------
    >>> import numpy as np
    >>> fft1 = np.random.rand(100) + 1j * np.random.rand(100)
    >>> fft2 = np.random.rand(100) + 1j * np.random.rand(100)
    >>> result = gccproduct(fft1, fft2, weighting='phat', threshfrac=0.05)
    """
    product = fft1 * fft2
    if weighting == "None":
        return product

    # calculate the weighting function
    if weighting == "liang":
        denom = np.square(
            np.sqrt(np.absolute(fft1 * np.conjugate(fft1)))
            + np.sqrt(np.absolute(fft2 * np.conjugate(fft2)))
        )
    elif weighting == "eckart":
        denom = np.sqrt(np.absolute(fft1 * np.conjugate(fft1))) * np.sqrt(
            np.absolute(fft2 * np.conjugate(fft2))
        )
    elif weighting == "phat":
        denom = np.absolute(product)
    elif weighting == "regressor":
        # determine weighting entirely from regressor 2 (the reference regressor)
        denom = np.absolute(fft2 * np.conjugate(fft2))
    else:
        raise ValueError("illegal weighting function specified in gccproduct")

    if displayplots:
        xvec = range(0, len(denom))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("reciprocal weighting function")
        plt.plot(xvec, abs(denom))
        plt.show()

    # now apply it while preserving the max
    theorigmax = np.max(np.absolute(denom))
    thresh = theorigmax * threshfrac

    if thresh > 0.0:
        scalefac = np.absolute(denom)
        if compress:
            pct10, pct90 = tide_stats.getfracvals(np.absolute(denom), [0.10, 0.90], nozero=True)
            scalefac[np.where(scalefac > pct90)] = pct90
            scalefac[np.where(scalefac < pct10)] = pct10
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(np.where(scalefac > thresh, product / denom, np.float64(0.0)))
    else:
        return 0.0 * product


def aligntcwithref(
    fixedtc: NDArray,
    movingtc: NDArray,
    Fs: float,
    lagmin: float = -30,
    lagmax: float = 30,
    refine: bool = True,
    zerooutbadfit: bool = False,
    widthmax: float = 1000.0,
    display: bool = False,
    verbose: bool = False,
) -> Tuple[NDArray, float, float, int]:
    """
    Align a moving timecourse to a fixed reference timecourse using cross-correlation.

    This function computes the cross-correlation between two timecourses and finds the
    optimal time lag that maximizes their similarity. The moving timecourse is then
    aligned to the fixed one using this lag.

    Parameters
    ----------
    fixedtc : ndarray
        The reference timecourse to which the moving timecourse will be aligned.
    movingtc : ndarray
        The timecourse to be aligned to the fixed timecourse.
    Fs : float
        Sampling frequency of the timecourses in Hz.
    lagmin : float, optional
        Minimum lag to consider in seconds. Default is -30.
    lagmax : float, optional
        Maximum lag to consider in seconds. Default is 30.
    refine : bool, optional
        If True, refine the lag estimate using Gaussian fitting. Default is True.
    zerooutbadfit : bool, optional
        If True, zero out the cross-correlation values for bad fits. Default is False.
    widthmax : float, optional
        Maximum allowed width of the Gaussian fit in samples. Default is 1000.0.
    display : bool, optional
        If True, display plots of the cross-correlation and aligned timecourses. Default is False.
    verbose : bool, optional
        If True, print detailed information about the cross-correlation results. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - aligneddata : ndarray
            The moving timecourse aligned to the fixed timecourse.
        - maxdelay : float
            The estimated time lag (in seconds) that maximizes cross-correlation.
        - maxval : float
            The maximum cross-correlation value.
        - failreason : int
            Reason for failure (0 = success, other values indicate specific failure types).

    Notes
    -----
    This function uses `fastcorrelate` for efficient cross-correlation computation and
    `tide_fit.findmaxlag_gauss` to estimate the optimal lag with optional Gaussian refinement.
    The alignment is performed using `tide_resample.doresample`.

    Examples
    --------
    >>> import numpy as np
    >>> from typing import Tuple
    >>> fixed = np.random.rand(1000)
    >>> moving = np.roll(fixed, 10)  # shift by 10 samples
    >>> aligned, delay, corr, fail = aligntcwithref(fixed, moving, Fs=100)
    >>> print(f"Estimated delay: {delay}s")
    """
    # now fixedtc and 2 are on the same timescales
    thexcorr = fastcorrelate(tide_math.corrnormalize(fixedtc), tide_math.corrnormalize(movingtc))
    xcorrlen = len(thexcorr)
    timestep = 1.0 / Fs
    xcorr_x = (
        timestep * xcorrlen * np.linspace(0.0, 1.0, xcorrlen, endpoint=False)
        - (xcorrlen * timestep) / 2.0
        + timestep / 2.0
    )

    (
        maxindex,
        maxdelay,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = tide_fit.findmaxlag_gauss(
        xcorr_x,
        thexcorr,
        lagmin,
        lagmax,
        widthmax=widthmax,
        refine=refine,
        useguess=False,
        fastgauss=False,
        displayplots=False,
        zerooutbadfit=zerooutbadfit,
    )

    if verbose:
        print("Crosscorrelation_Rmax:\t", maxval)
        print("Crosscorrelation_maxdelay:\t", maxdelay)

    if display:
        plt.plot(xcorr_x, thexcorr, "r")
        plt.plot(maxdelay, maxval, "kv")
        plt.title("aligntcwithref correlation function")
        plt.show()

    # now align the second timecourse to the first
    timeaxis = np.linspace(0.0, 1.0 / Fs * len(fixedtc), num=len(fixedtc), endpoint=False)
    aligneddata = tide_resample.doresample(timeaxis, movingtc, timeaxis - maxdelay, padtype="zero")
    if display:
        plt.plot(timeaxis, tide_math.stdnormalize(fixedtc), "r")
        plt.plot(timeaxis, tide_math.stdnormalize(aligneddata) + 0.2, "b")
        plt.title("Timecourses")
        plt.legend(["fixedtc", "alignedtc"])
        plt.show()

    return aligneddata, maxdelay, maxval, failreason
