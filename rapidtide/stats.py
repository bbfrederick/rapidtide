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
from typing import Any, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rapidtide.decorators import conditionaljit, conditionaljit2

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

import scipy as sp
from scipy.stats import johnsonsb, kurtosis, kurtosistest, skew, skewtest
from statsmodels.robust import mad

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()


# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000


# --------------------------- probability functions -------------------------------------------------
def printthresholds(pcts: ArrayLike, thepercentiles: ArrayLike, labeltext: str) -> None:
    """Print significance thresholds with formatted output.

    Parameters
    ----------
    pcts : array-like
        Percentile threshold values
    thepercentiles : array-like
        Percentile levels (0-1)
    labeltext : str
        Label to print before thresholds
    """
    print(labeltext)
    for i in range(0, len(pcts)):
        print(f"\tp <{1.0 - thepercentiles[i]:.3f}: {pcts[i]:.3f}")


def fitgausspdf(
    thehist: Tuple,
    histlen: int,
    thedata: NDArray,
    displayplots: bool = False,
    nozero: bool = False,
    debug: bool = False,
) -> NDArray:
    """Fit a Gaussian probability density function to histogram data.

    Parameters
    ----------
    thehist : tuple
        Histogram tuple from np.histogram containing (counts, bin_edges)
    histlen : int
        Length of histogram
    thedata : array-like
        Original data used to create histogram
    displayplots : bool, optional
        If True, display fit visualization. Default: False
    nozero : bool, optional
        If True, ignore zero values. Default: False
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    array-like
        Array containing (peakheight, peakloc, peakwidth, zeroterm)
    """
    thestore = np.zeros((2, histlen), dtype="float64")
    thestore[0, :] = thehist[1][:-1]
    thestore[1, :] = thehist[0][:] / (1.0 * len(thedata))

    # store the zero term for later
    zeroterm = thestore[1, 0]
    thestore[1, 0] = 0.0

    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (
        thestore[1, peakindex + numbins] > peakheight / 2.0
    ):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if debug:
        print("Initial values:")
        print(f"\tPeak height: {peakheight}")
        print(f"\tPeak lag: {peaklag}")
        print(f"\tPeak width: {peakwidth}")
    peakheight, peaklag, peakwidth = tide_fit.gaussfit(
        peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :]
    )
    if debug:
        print("Refined values:")
        print(f"\tPeak height: {peakheight}")
        print(f"\tPeak lag: {peaklag}")
        print(f"\tPeak width: {peakwidth}")

    params = (peakheight, peaklag, peakwidth)

    # restore the zero term if needed
    # if nozero is True, assume that R=0 is not special (i.e. there is no spike in the
    # histogram at zero from failed fits)
    if nozero:
        zeroterm = 0.0
    else:
        thestore[1, 0] = zeroterm

    # generate the johnsonsb function
    gaussvals = tide_fit.gauss_eval(thestore[0, :], params)
    corrfac = (1.0 - zeroterm) / (1.0 * histlen)
    gaussvals *= corrfac
    gaussvals[0] = zeroterm

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("fitgausspdf: histogram")
        plt.plot(thestore[0, :], thestore[1, :], "b", thestore[0, :], gaussvals, "r")
        plt.legend(["histogram", "fit to gaussian"])
        plt.show()
    return np.append(params, np.array([zeroterm]))


def fitjsbpdf(
    thehist: Tuple,
    histlen: int,
    thedata: NDArray,
    displayplots: bool = False,
    nozero: bool = False,
    debug: bool = False,
) -> NDArray:
    """Fit a Johnson SB probability density function to histogram data.

    Parameters
    ----------
    thehist : tuple
        Histogram tuple from np.histogram containing (counts, bin_edges)
    histlen : int
        Length of histogram
    thedata : array-like
        Original data used to create histogram
    displayplots : bool, optional
        If True, display fit visualization. Default: False
    nozero : bool, optional
        If True, ignore zero values. Default: False
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    array-like
        Array containing (a, b, loc, scale, zeroterm) parameters of Johnson SB fit
    """
    thestore = np.zeros((2, histlen), dtype="float64")
    thestore[0, :] = thehist[1][:-1]
    thestore[1, :] = thehist[0][:] / (1.0 * len(thedata))

    # store the zero term for later
    zeroterm = thestore[1, 0]
    thestore[1, 0] = 0.0

    # fit the johnsonSB function
    params = johnsonsb.fit(thedata[np.where(thedata > 0.0)])
    if debug:
        print(f"Johnson SB fit parameters for pdf: {params}")

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
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("fitjsbpdf: histogram")
        plt.plot(thestore[0, :], thestore[1, :], "b", thestore[0, :], johnsonsbvals, "r")
        plt.legend(["histogram", "fit to johnsonsb"])
        plt.show()
    return np.append(params, np.array([zeroterm]))


def getjohnsonppf(percentile: float, params: ArrayLike, zeroterm: float) -> None:
    """Get percent point function (inverse CDF) for Johnson SB distribution.

    Note: This function is incomplete and only initializes variables.

    Parameters
    ----------
    percentile : float
        Percentile value (0-1)
    params : array-like
        Johnson SB distribution parameters (a, b, loc, scale)
    zeroterm : float
        Zero term correction factor
    """
    johnsonfunc = johnsonsb(params[0], params[1], params[2], params[3])
    corrfac = 1.0 - zeroterm


def sigFromDistributionData(
    vallist: NDArray,
    histlen: int,
    thepercentiles: ArrayLike,
    similaritymetric: str = "correlation",
    displayplots: bool = False,
    twotail: bool = False,
    nozero: bool = False,
    dosighistfit: bool = True,
    debug: bool = False,
) -> Tuple[Optional[list], Optional[list], Optional[NDArray]]:
    """Calculate significance thresholds from distribution data.

    Fits a probability distribution to data and calculates percentile thresholds
    for significance testing.

    Parameters
    ----------
    vallist : array-like
        List of similarity/correlation values
    histlen : int
        Length of histogram
    thepercentiles : array-like
        Percentile values to compute (0-1)
    similaritymetric : str, optional
        Type of similarity metric ("correlation" or "mutualinfo"). Default: "correlation"
    displayplots : bool, optional
        If True, display diagnostic plots. Default: False
    twotail : bool, optional
        If True, calculate two-tailed thresholds. Default: False
    nozero : bool, optional
        If True, exclude zero values. Default: False
    dosighistfit : bool, optional
        If True, fit distribution to data. Default: True
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    tuple
        (pcts_data, pcts_fit, histfit) - percentiles from data, fitted distribution, and fit parameters
    """
    # check to make sure there are nonzero values first
    if len(np.where(vallist != 0.0)[0]) == 0:
        print("no nonzero values - skipping percentile calculation")
        return None, None, None
    thehistogram, peakheight, peakloc, peakwidth, centerofmass, peakpercentile = makehistogram(
        np.abs(vallist), histlen, therange=[0.0, 1.0]
    )
    if dosighistfit:
        if similaritymetric == "mutualinfo":
            histfit = fitgausspdf(
                thehistogram,
                histlen,
                vallist,
                displayplots=displayplots,
                nozero=nozero,
                debug=debug,
            )
        else:
            histfit = fitjsbpdf(
                thehistogram,
                histlen,
                vallist,
                displayplots=displayplots,
                nozero=nozero,
                debug=debug,
            )
    if twotail:
        thepercentiles = 1.0 - (1.0 - thepercentiles) / 2.0
        print(f"thepercentiles adapted for two tailed distribution: {thepercentiles}")
    pcts_data = getfracvals(vallist, thepercentiles, nozero=nozero)
    if dosighistfit:
        pcts_fit = getfracvalsfromfit(histfit, thepercentiles)
        return pcts_data, pcts_fit, histfit
    else:
        pcts_fit = []
        for i in len(pcts_data):
            pcts_fit.append(None)
        return pcts_data, pcts_fit, None


global neglogpfromr_interpolator, minrforneglogp, maxrforneglogp
neglogpfromr_interpolator = None
minrforneglogp = None
maxrforneglogp = None


def neglog10pfromr(
    rval: float,
    histfit: ArrayLike,
    lutlen: int = 3000,
    initialize: bool = False,
    neglogpmin: float = 0.0,
    neglogpmax: float = 3.0,
    debug: bool = False,
) -> float:
    """Convert correlation value to negative log10 p-value using histogram fit.

    Parameters
    ----------
    rval : float
        Correlation value to convert
    histfit : array-like
        Histogram fit parameters from fitjsbpdf
    lutlen : int, optional
        Length of lookup table. Default: 3000
    initialize : bool, optional
        Force reinitialization of interpolator. Default: False
    neglogpmin : float, optional
        Minimum negative log10 p-value. Default: 0.0
    neglogpmax : float, optional
        Maximum negative log10 p-value. Default: 3.0
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    float
        Negative log10 p-value corresponding to the input correlation value
    """
    global neglogpfromr_interpolator, minrforneglogp, maxrforneglogp
    if neglogpfromr_interpolator is None or initialize:
        neglogparray = np.linspace(neglogpmin, neglogpmax, lutlen, endpoint=True)
        pvals = pow(10.0, -neglogparray)
        percentile_list = (1.0 - pvals).tolist()
        rforneglogp = np.asarray(getfracvalsfromfit(histfit, percentile_list), dtype=float)
        minrforneglogp = rforneglogp[0]
        maxrforneglogp = rforneglogp[-1]
        if debug:
            print("START NEGLOGPFROMR DEBUG")
            print("neglogp\tpval\tpct\trfornlp")
            for i in range(lutlen):
                print(f"{neglogparray[i]}\t{pvals[i]}\t{percentile_list[i]}\t{rforneglogp[i]}")
            print("END NEGLOGPFROMR DEBUG")
        neglogpfromr_interpolator = sp.interpolate.UnivariateSpline(
            rforneglogp, neglogparray, k=3, s=0
        )
    if rval > maxrforneglogp:
        return np.float64(neglogpmax)
    elif rval < minrforneglogp:
        return np.float64(neglogpmin)
    else:
        return np.float64(neglogpfromr_interpolator(np.asarray([rval], dtype=float))[0])


def rfromp(fitfile: str, thepercentiles: ArrayLike) -> NDArray:
    """Get correlation values from p-values using a saved distribution fit.

    Parameters
    ----------
    fitfile : str
        Path to file containing distribution fit parameters
    thepercentiles : array-like
        Percentile values to calculate (0-1)

    Returns
    -------
    array-like
        Correlation values corresponding to the percentiles
    """
    thefit = np.array(tide_io.readvecs(fitfile)[0]).astype("float64")
    print(f"thefit = {thefit}")
    return getfracvalsfromfit(thefit, thepercentiles)


def tfromr(
    r: float,
    nsamps: int,
    dfcorrfac: float = 1.0,
    oversampfactor: float = 1.0,
    returnp: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Convert correlation to t-statistic.

    Parameters
    ----------
    r : float
        Correlation coefficient
    nsamps : int
        Number of samples
    dfcorrfac : float, optional
        Degrees of freedom correction factor. Default: 1.0
    oversampfactor : float, optional
        Oversampling factor for DOF adjustment. Default: 1.0
    returnp : bool, optional
        If True, also return p-value. Default: False

    Returns
    -------
    float or tuple
        T-statistic, or (t-statistic, p-value) if returnp=True
    """
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


def pfromz(z: float, twotailed: bool = True) -> float:
    """Calculate p-value from z-score.

    Parameters
    ----------
    z : float
        Z-score value
    twotailed : bool, optional
        If True, calculate two-tailed p-value. Default: True

    Returns
    -------
    float
        P-value corresponding to the z-score
    """
    # importing packages
    import scipy.stats

    # finding p-value
    if twotailed:
        return scipy.stats.norm.sf(abs(z)) * 2
    else:
        return scipy.stats.norm.sf(abs(z))


def zfromr(
    r: float,
    nsamps: int,
    dfcorrfac: float = 1.0,
    oversampfactor: float = 1.0,
    returnp: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Convert correlation to z-statistic.

    Parameters
    ----------
    r : float
        Correlation coefficient
    nsamps : int
        Number of samples
    dfcorrfac : float, optional
        Degrees of freedom correction factor. Default: 1.0
    oversampfactor : float, optional
        Oversampling factor for DOF adjustment. Default: 1.0
    returnp : bool, optional
        If True, also return p-value. Default: False

    Returns
    -------
    float or tuple
        Z-statistic, or (z-statistic, p-value) if returnp=True
    """
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


def zofcorrdiff(r1: float, r2: float, n1: int, n2: int) -> float:
    """Calculate z-statistic for the difference between two correlations.

    Parameters
    ----------
    r1 : float
        First correlation coefficient
    r2 : float
        Second correlation coefficient
    n1 : int
        Sample size for first correlation
    n2 : int
        Sample size for second correlation

    Returns
    -------
    float
        Z-statistic for the difference between the two correlations
    """
    return (fisher(r1) - fisher(r2)) / stderrofdiff(n1, n2)


def stderrofdiff(n1: int, n2: int) -> float:
    """Calculate standard error of difference between two Fisher-transformed correlations.

    Parameters
    ----------
    n1 : int
        Sample size for first correlation
    n2 : int
        Sample size for second correlation

    Returns
    -------
    float
        Standard error of the difference
    """
    return np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))


def fisher(r: float) -> float:
    """Apply Fisher's r-to-z transformation to correlation coefficient.

    Parameters
    ----------
    r : float
        Correlation coefficient

    Returns
    -------
    float
        Fisher-transformed z-value
    """
    return 0.5 * np.log((1 + r) / (1 - r))


def permute_phase(time_series: NDArray) -> NDArray:
    """Generate phase-randomized surrogate time series.

    Creates a surrogate time series with the same power spectrum as the input
    but with randomized phases. Useful for generating null distributions in
    time series analysis.

    Parameters
    ----------
    time_series : array-like
        Input time series

    Returns
    -------
    array-like
        Phase-randomized surrogate time series with same length as input
    """
    # Compute the Fourier transform of the time series
    freq_domain = np.fft.rfft(time_series)

    # Randomly permute the phase of the Fourier coefficients
    phase = 2.0 * np.pi * (np.random.random(len(freq_domain)) - 0.5)
    freq_domain = np.abs(freq_domain) * np.exp(1j * phase)

    # Compute the inverse Fourier transform to get the permuted time series
    permuted_time_series = np.fft.irfft(freq_domain)

    return permuted_time_series


def skewnessstats(timecourse: NDArray) -> Tuple[float, float, float]:
    """Calculate skewness and statistical test for timecourse.

    Parameters
    ----------
    timecourse : array-like
        Input time series

    Returns
    -------
    tuple
        (skewness, z-statistic, p-value) from skewness test
    """
    testres = skewtest(timecourse)
    return skew(timecourse), testres[0], testres[1]


def kurtosisstats(timecourse: NDArray) -> Tuple[float, float, float]:
    """Calculate kurtosis and statistical test for timecourse.

    Parameters
    ----------
    timecourse : array-like
        Input time series

    Returns
    -------
    tuple
        (kurtosis, z-statistic, p-value) from kurtosis test
    """
    testres = kurtosistest(timecourse)
    return kurtosis(timecourse), testres[0], testres[1]


def fmristats(
    fmridata: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Calculate comprehensive statistics for fMRI data along time axis.

    Parameters
    ----------
    fmridata : ndarray
        2D array where rows are voxels and columns are timepoints

    Returns
    -------
    tuple
        (min, max, mean, std, median, mad, skew, kurtosis) - each as 1D array
        with length equal to number of voxels, calculated along timepoints
    """
    return (
        np.min(fmridata, axis=1),
        np.max(fmridata, axis=1),
        np.mean(fmridata, axis=1),
        np.std(fmridata, axis=1),
        np.median(fmridata, axis=1),
        mad(fmridata, axis=1),
        skew(fmridata, axis=1),
        kurtosis(fmridata, axis=1),
    )


def fast_ICC_rep_anova(
    Y: NDArray, nocache: bool = False, debug: bool = False
) -> Tuple[float, float, float, float, int, int]:
    """
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]

    This is a hacked up (but fully compatible) version of ICC_rep_anova
    from nipype that caches some very expensive operations that depend
    only on the input array shape - if you're going to run the routine
    multiple times (like, on every voxel of an image), this gives you a
    HUGE speed boost for large input arrays.  If you change the dimensions
    of Y, it will reinitialize automatically.  Set nocache to True to get
    the original, much slower behavior.  No, actually, don't do that. That would
    be silly.
    """
    global icc_inited
    global current_Y_shape
    global dfc, dfe, dfr
    global nb_subjects, nb_conditions
    global x, x0, X
    global centerbit

    try:
        current_Y_shape
        if nocache or (current_Y_shape != Y.shape):
            icc_inited = False
    except NameError:
        icc_inited = False

    if not icc_inited:
        [nb_subjects, nb_conditions] = Y.shape
        if debug:
            print(
                f"fast_ICC_rep_anova inited with nb_subjects = {nb_subjects}, nb_conditions = {nb_conditions}"
            )
        current_Y_shape = Y.shape
        dfc = nb_conditions - 1
        dfe = (nb_subjects - 1) * dfc
        dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    if not icc_inited:
        x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
        x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
        X = np.hstack([x, x0])
        centerbit = np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T)

    # Sum Square Error
    predicted_Y = np.dot(centerbit, Y.flatten("F"))
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals**2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between columns/sessions
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    ICC = np.nan_to_num((MSR - MSE) / (MSR + dfc * MSE))

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    icc_inited = True

    return ICC, r_var, e_var, session_effect_F, dfc, dfe


# --------------------------- histogram functions -------------------------------------------------
def gethistprops(
    indata: NDArray,
    histlen: int,
    refine: bool = False,
    therange: Optional[Tuple[float, float]] = None,
    pickleft: bool = False,
    peakthresh: float = 0.33,
) -> Tuple[float, float, float]:
    """Extract histogram peak properties from data.

    Parameters
    ----------
    indata : array-like
        Input data array
    histlen : int
        Number of histogram bins
    refine : bool, optional
        If True, refine peak estimates using Gaussian fit. Default: False
    therange : tuple, optional
        (min, max) range for histogram. If None, use data range. Default: None
    pickleft : bool, optional
        If True, pick leftmost peak above threshold. Default: False
    peakthresh : float, optional
        Threshold for peak detection (fraction of max). Default: 0.33

    Returns
    -------
    tuple
        (peakloc, peakheight, peakwidth) - peak location, height, and width
    """
    thestore = np.zeros((2, histlen), dtype="float64")
    if therange is None:
        thehist = np.histogram(indata, histlen)
    else:
        thehist = np.histogram(indata, histlen, therange)
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    if pickleft:
        overallmax = np.max(thestore[1, 1:-2])
        peakindex = 1
        i = 1
        started = False
        finished = False
        while i < len(thestore[1, :] - 3) and not finished:
            if thestore[1, i] > peakthresh * overallmax:
                started = True
            if thestore[1, i] > thestore[1, peakindex]:
                peakindex = i
            if started and (thestore[1, i] < 0.75 * thestore[1, peakindex]):
                finished = True
            i += 1
    else:
        peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (
        thestore[1, peakindex + numbins] > peakheight / 2.0
    ):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = tide_fit.gaussfit(
            peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :]
        )
    return peaklag, peakheight, peakwidth


def prochistogram(
    thehist: Tuple,
    refine: bool = False,
    pickleft: bool = False,
    peakthresh: float = 0.33,
    ignorefirstpoint: bool = False,
    debug: bool = False,
) -> Tuple[float, float, float, float]:
    """Process histogram data to extract peak properties.

    Parameters
    ----------
    thehist : tuple
        Histogram tuple from np.histogram containing (counts, bin_edges)
    refine : bool, optional
        If True, refine peak estimates using Gaussian fit. Default: False
    pickleft : bool, optional
        If True, pick leftmost peak above threshold. Default: False
    peakthresh : float, optional
        Threshold for peak detection (fraction of max). Default: 0.33
    ignorefirstpoint : bool, optional
        If True, ignore first histogram bin. Default: False
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    tuple
        (peakheight, peakloc, peakwidth, centerofmass) - peak properties
    """
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    histlen = len(thehist[1])
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]

    # get starting values for the peak, ignoring first and last point of histogram
    if ignorefirstpoint:
        xvals = thestore[0, 1:]
        yvals = thestore[1, 1:]
        histlen -= 1
    else:
        xvals = thestore[0, :]
        yvals = thestore[1, :]
    if pickleft:
        overallmax = np.max(yvals[1:-2])
        peakindex = 1
        i = 1
        started = False
        finished = False
        while i < len(yvals - 2) and not finished:
            if yvals[i] > peakthresh * overallmax:
                started = True
            if yvals[i] > yvals[peakindex]:
                peakindex = i
            if started and (yvals[i] < 0.75 * yvals[peakindex]):
                finished = True
            i += 1
    else:
        peakindex = np.argmax(yvals[1:-2]) + 1
    peakloc = xvals[peakindex]
    peakheight = yvals[peakindex]
    numbins = 1
    while (peakindex + numbins < histlen - 2) and (yvals[peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (xvals[peakindex + numbins] - xvals[peakindex]) * 2.0
    if debug:
        print(f"{xvals=}")
        print(f"{yvals=}")
        print("Before refine")
        print(f"{peakindex=}, {peakloc=}, {peakheight=}, {peakwidth=}")
    if refine:
        peakheight, peakloc, peakwidth = tide_fit.gaussfit(
            peakheight, peakloc, peakwidth, xvals, yvals
        )
    if debug:
        print("After refine")
        print(f"{peakindex=}, {peakloc=}, {peakheight=}, {peakwidth=}")
    centerofmass = np.sum(xvals * yvals) / np.sum(yvals)
    return peakheight, peakloc, peakwidth, centerofmass


def percentilefromloc(indata: NDArray, peakloc: float, nozero: bool = False) -> float:
    """Calculate the percentile corresponding to a given value location.

    Parameters
    ----------
    indata : array-like
        Input data array
    peakloc : float
        Value location to find percentile for
    nozero : bool, optional
        If True, exclude zero values from calculation. Default: False

    Returns
    -------
    float
        Percentile (0-100) corresponding to the given value location
    """
    order = indata.argsort()
    orderedvalues = indata[order]
    if nozero:
        orderedvalues = orderedvalues[np.where(orderedvalues != 0.0)]
    peaklocindex = np.argmax(orderedvalues >= peakloc)
    thepercentile = 100.0 * peaklocindex / len(orderedvalues)
    return thepercentile


def makehistogram(
    indata: NDArray,
    histlen: Optional[int],
    binsize: Optional[float] = None,
    therange: Optional[Tuple[float, float]] = None,
    pickleft: bool = False,
    peakthresh: float = 0.33,
    refine: bool = False,
    normalize: bool = False,
    ignorefirstpoint: bool = False,
    debug: bool = False,
) -> Tuple[Tuple, float, float, float, float, float]:
    """Create histogram and extract peak properties from data.

    Parameters
    ----------
    indata : array-like
        Input data array
    histlen : int or None
        Number of histogram bins. If None, binsize must be specified
    binsize : float, optional
        Bin size for histogram. If specified, overrides histlen. Default: None
    therange : tuple, optional
        (min, max) range for histogram. If None, use data range. Default: None
    pickleft : bool, optional
        If True, pick leftmost peak above threshold. Default: False
    peakthresh : float, optional
        Threshold for peak detection (fraction of max). Default: 0.33
    refine : bool, optional
        If True, refine peak estimates using Gaussian fit. Default: False
    normalize : bool, optional
        If True, normalize histogram to unit area. Default: False
    ignorefirstpoint : bool, optional
        If True, ignore first histogram bin. Default: False
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    tuple
        (histogram, peakheight, peakloc, peakwidth, centerofmass, peakpercentile)
    """
    if therange is None:
        therange = [indata.min(), indata.max()]
    if histlen is None and binsize is None:
        thebins = 10
    elif binsize is not None:
        thebins = np.linspace(
            therange[0],
            therange[1],
            (therange[1] - therange[0]) / binsize + 1,
            endpoint=True,
        )
    else:
        thebins = histlen

    thehist = np.histogram(indata, thebins, therange, density=normalize)

    peakheight, peakloc, peakwidth, centerofmass = prochistogram(
        thehist,
        refine=refine,
        pickleft=pickleft,
        peakthresh=peakthresh,
        ignorefirstpoint=ignorefirstpoint,
        debug=debug,
    )
    peakpercentile = percentilefromloc(indata, peakloc, nozero=ignorefirstpoint)
    return thehist, peakheight, peakloc, peakwidth, centerofmass, peakpercentile


def echoloc(indata: NDArray, histlen: int, startoffset: float = 5.0) -> Tuple[float, float]:
    """Detect and analyze echo peak in histogram data.

    Identifies a secondary (echo) peak in histogram data that occurs after
    the primary peak, useful for analyzing echo patterns in imaging data.

    Parameters
    ----------
    indata : array-like
        Input data array
    histlen : int
        Number of histogram bins
    startoffset : float, optional
        Offset from primary peak to start echo search. Default: 5.0

    Returns
    -------
    tuple
        (echo_lag, echo_ratio) where echo_lag is the distance between primary
        and echo peaks, and echo_ratio is the ratio of echo to primary peak areas
    """
    thehist, peakheight, peakloc, peakwidth, centerofmass, peakpercentile = makehistogram(
        indata, histlen, refine=True
    )
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thestore[1, :] = thehist[0][-histlen:]
    timestep = thestore[0, 1] - thestore[0, 0]
    startpt = np.argmax(thestore[1, :]) + int(startoffset // timestep)
    print(f"primary peak: {peakheight:.2f}, {peakloc:.2f}, {peakwidth}")
    print("startpt, startloc, timestep: {startpt}, {thestore[1, startpt]}, {timestep}")
    while (thestore[1, startpt] > thestore[1, startpt + 1]) and (startpt < len(thehist[0]) - 2):
        startpt += 1
    echopeakindex = np.argmax(thestore[1, startpt:-2]) + startpt
    echopeakloc = thestore[0, echopeakindex + 1]
    echopeakheight = thestore[1, echopeakindex + 1]
    numbins = 1
    while (echopeakindex + numbins < histlen - 1) and (
        thestore[1, echopeakindex + numbins] > echopeakheight / 2.0
    ):
        numbins += 1
    echopeakwidth = (thestore[0, echopeakindex + numbins] - thestore[0, echopeakindex]) * 2.0
    echopeakheight, echopeakloc, echopeakwidth = tide_fit.gaussfit(
        echopeakheight, echopeakloc, echopeakwidth, thestore[0, :], thestore[1, :]
    )
    return echopeakloc - peakloc, (echopeakheight * echopeakwidth) / (peakheight * peakwidth)


def makeandsavehistogram(
    indata: NDArray,
    histlen: int,
    endtrim: int,
    outname: str,
    binsize: Optional[float] = None,
    saveimfile: bool = False,
    displaytitle: str = "histogram",
    displayplots: bool = False,
    refine: bool = False,
    therange: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    dictvarname: Optional[str] = None,
    thedict: Optional[dict] = None,
    append: bool = False,
    debug: bool = False,
) -> None:
    """Create histogram, extract properties, and save results to file.

    Parameters
    ----------
    indata : array-like
        Input data array
    histlen : int
        Number of histogram bins
    endtrim : int
        Number of bins to trim from end when plotting
    outname : str
        Output file path (without extension)
    binsize : float, optional
        Bin size for histogram. If specified, overrides histlen. Default: None
    saveimfile : bool, optional
        Unused parameter. Default: False
    displaytitle : str, optional
        Title for display plots. Default: "histogram"
    displayplots : bool, optional
        If True, display histogram plot. Default: False
    refine : bool, optional
        If True, refine peak estimates using Gaussian fit. Default: False
    therange : tuple, optional
        (min, max) range for histogram. If None, use data range. Default: None
    normalize : bool, optional
        If True, normalize histogram to unit area. Default: False
    dictvarname : str, optional
        Variable name for dictionary storage. If None, use outname. Default: None
    thedict : dict, optional
        Dictionary to store results in. If None, write to file. Default: None
    append : bool, optional
        If True, append to existing file. Default: False
    debug : bool, optional
        Enable debug output. Default: False
    """
    thehist, peakheight, peakloc, peakwidth, centerofmass, peakpercentile = makehistogram(
        indata,
        histlen,
        binsize=binsize,
        therange=therange,
        refine=refine,
        normalize=normalize,
    )
    thestore = np.zeros((2, len(thehist[0])), dtype="float64")
    thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
    thebinsizes = np.diff(thehist[1][:])
    thestore[1, :] = thehist[0][-histlen:]
    if debug:
        print(f"histlen: {len(thestore[1, :])}, sizelen: {len(thebinsizes)}")
    if dictvarname is None:
        varroot = outname
    else:
        varroot = dictvarname
    if thedict is None:
        tide_io.writenpvecs(np.array([centerofmass]), outname + "_centerofmass.txt")
        tide_io.writenpvecs(np.array([peakloc]), outname + "_peak.txt")
    else:
        thedict[varroot + "_centerofmass.txt"] = centerofmass
        thedict[varroot + "_peak.txt"] = peakloc
    extraheaderinfo = {}
    extraheaderinfo["Description"] = displaytitle
    extraheaderinfo["centerofmass"] = centerofmass
    extraheaderinfo["peakloc"] = peakloc
    extraheaderinfo["peakwidth"] = peakwidth
    extraheaderinfo["peakheight"] = peakheight
    extraheaderinfo["peakpercentile"] = peakpercentile
    (
        extraheaderinfo["pct02"],
        extraheaderinfo["pct25"],
        extraheaderinfo["pct50"],
        extraheaderinfo["pct75"],
        extraheaderinfo["pct98"],
    ) = getfracvals(indata, [0.02, 0.25, 0.5, 0.75, 0.98], debug=debug)
    tide_io.writebidstsv(
        outname,
        np.transpose(thestore[1, :]),
        1.0 / (thestore[0, 1] - thestore[0, 0]),
        starttime=thestore[0, 0],
        columns=[varroot],
        append=append,
        extraheaderinfo=extraheaderinfo,
        debug=debug,
    )
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        plt.plot(thestore[0, : (-1 - endtrim)], thestore[1, : (-1 - endtrim)])
        for thepct in ["pct02", "pct98"]:
            plt.axvline(
                extraheaderinfo[thepct],
                color="#99ff99",
                linewidth=1.0,
                linestyle="dotted",
                label=thepct,
            )
        for thepct in ["pct25", "pct75"]:
            plt.axvline(
                extraheaderinfo[thepct],
                color="#66ff66",
                linewidth=1.0,
                linestyle="dashed",
                label=thepct,
            )
        for thepct in ["pct50"]:
            plt.axvline(
                extraheaderinfo[thepct],
                color="#33ff33",
                linewidth=1.0,
                linestyle="solid",
                label=thepct,
            )
        plt.show()


def symmetrize(a: NDArray, antisymmetric: bool = False, zerodiagonal: bool = False) -> NDArray:
    """Symmetrize a matrix.

    Parameters
    ----------
    a : ndarray
        Input matrix
    antisymmetric : bool, optional
        If True, create antisymmetric matrix (a - a.T) / 2. Default: False
    zerodiagonal : bool, optional
        If True, set diagonal elements to zero. Default: False

    Returns
    -------
    ndarray
        Symmetrized matrix
    """
    if antisymmetric:
        intermediate = (a - a.T) / 2.0
    else:
        intermediate = (a + a.T) / 2.0
    if zerodiagonal:
        return intermediate - np.diag(intermediate.diagonal())
    else:
        return intermediate


def makepmask(rvals: NDArray, pval: float, sighistfit: NDArray, onesided: bool = True) -> NDArray:
    """Create significance mask from p-value threshold and distribution fit.

    Parameters
    ----------
    rvals : array-like
        Array of correlation or similarity values
    pval : float
        P-value threshold (0-1)
    sighistfit : array-like
        Distribution fit parameters from fitjsbpdf
    onesided : bool, optional
        If True, use one-sided test. If False, use two-sided test. Default: True

    Returns
    -------
    ndarray
        Binary mask (int16) with 1 for significant values, 0 otherwise
    """
    if onesided:
        return np.where(
            rvals > getfracvalsfromfit(sighistfit, 1.0 - pval), np.int16(1), np.int16(0)
        )
    else:
        return np.where(
            np.abs(rvals) > getfracvalsfromfit(sighistfit, 1.0 - pval / 2.0),
            np.int16(1),
            np.int16(0),
        )


# Find the image intensity value which thefrac of the non-zero voxels in the image exceed
def getfracval(datamat: NDArray, thefrac: float, nozero: bool = False) -> float:
    """Get data value at a specific fractional position in sorted data.

    Parameters
    ----------
    datamat : array-like
        Input data array
    thefrac : float
        Fractional position (0-1) to find value at
    nozero : bool, optional
        If True, exclude zero values. Default: False

    Returns
    -------
    float
        Value at the specified fractional position
    """
    return getfracvals(datamat, [thefrac], nozero=nozero)[0]


def getfracvals(
    datamat: NDArray, thefracs: ArrayLike, nozero: bool = False, debug: bool = False
) -> list:
    """Get data values at multiple fractional positions in sorted data.

    Finds the intensity values that correspond to specified fractional positions
    when data is sorted in ascending order. Useful for percentile calculations.

    Parameters
    ----------
    datamat : array-like
        Input data array
    thefracs : array-like
        List of fractional positions (0-1) to find values at
    nozero : bool, optional
        If True, exclude zero values. Default: False
    debug : bool, optional
        Enable debug output. Default: False

    Returns
    -------
    list
        Values at the specified fractional positions
    """
    thevals = []

    if nozero:
        maskmat = np.sort(datamat[np.where(datamat != 0.0)].flatten())
        if len(maskmat) == 0:
            for thisfrac in thefracs:
                thevals.append(0.0)
            return thevals
    else:
        maskmat = np.sort(datamat.flatten())
    maxindex = len(maskmat)

    for thisfrac in thefracs:
        theindex = np.min([int(np.round(thisfrac * maxindex, 0)), len(maskmat) - 1])
        thevals.append(float(maskmat[theindex]))

    if debug:
        print(f"getfracvals: {datamat.shape=}")
        print(f"getfracvals: {maskmat.shape=}")
        print(f"getfracvals: {thefracs=}")
        print(f"getfracvals: {maxindex=}")
        print(f"getfracvals: {thevals=}")

    return thevals


def getfracvalsfromfit(histfit: ArrayLike, thefracs: ArrayLike) -> NDArray:
    """Get data values at fractional positions from a Johnson SB distribution fit.

    Uses the fitted Johnson SB distribution to calculate values corresponding
    to specified percentiles.

    Parameters
    ----------
    histfit : array-like
        Johnson SB distribution fit parameters (a, b, loc, scale, zeroterm) from fitjsbpdf
    thefracs : array-like
        List of fractional positions/percentiles (0-1) to calculate values for

    Returns
    -------
    array-like
        Values corresponding to the specified percentiles from the fitted distribution
    """
    # print('entering getfracvalsfromfit: histfit=',histfit, ' thefracs=', thefracs)
    thedist = johnsonsb(histfit[0], histfit[1], histfit[2], histfit[3])
    thevals = thedist.ppf(thefracs)
    return thevals


def makemask(
    image: NDArray,
    threshpct: float = 25.0,
    verbose: bool = False,
    nozero: bool = False,
    noneg: bool = False,
) -> NDArray:
    """

    Parameters
    ----------
    image: array-like
        The image data to generate the mask for.
    threshpct: float
        Voxels with values greater then threshpct of the 98th percentile of voxel values are preserved.
    verbose: bool
        If true, print additional debugging information.
    nozero: bool
        If true, exclude zero values when calculating percentiles
    noneg: bool
        If true, exclude negative values when calculating percentiles

    Returns
    -------
    themask: array-like
        An int16 mask with dimensions matching the input. 1 for voxels to preserve, 0 elsewhere

    """
    if noneg:
        pct2, pct98, pctthresh = getfracvals(
            np.where(image >= 0.0, image, 0.0), [0.02, 0.98, threshpct], nozero=nozero
        )
    else:
        pct2, pct98, pctthresh = getfracvals(image, [0.02, 0.98, threshpct], nozero=nozero)
    threshval = pct2 + (threshpct / 100.0) * (pct98 - pct2)
    print(f"old style threshval: {threshval:.2f}, new style threshval: {pctthresh:.2f}")
    if verbose:
        print(
            f"fracval: {pctthresh:.2f}",
            f"threshpct: {threshpct:.2f}",
            f"mask threshold: {threshval:.2f}",
        )
    themask = np.where(image > threshval, np.int16(1), np.int16(0))
    return themask


def getmasksize(themask: NDArray) -> int:
    """

    Parameters
    ----------
    image: array-like
        The mask data to check.

    Returns
    -------
    numvoxels: int
        The number of nonzero voxels in themask

    """
    return len(np.where(themask > 0)[0])
