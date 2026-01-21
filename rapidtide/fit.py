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
import sys
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.special as sps
import statsmodels.api as sm
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert
from scipy.stats import entropy, moment
from sklearn.linear_model import LinearRegression
from statsmodels.robust import mad
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from tqdm import tqdm

import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util
from rapidtide.decorators import conditionaljit, conditionaljit2

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000


# --------------------------- Fitting functions -------------------------------------------------
def gaussskresiduals(p: NDArray, y: NDArray, x: NDArray) -> NDArray:
    """
    Calculate residuals for skewed Gaussian fit.

    This function computes the residuals (observed values minus fitted values)
    for a skewed Gaussian model. The residuals are used to assess the quality
    of the fit and are commonly used in optimization routines.

    Parameters
    ----------
    p : NDArray
        Skewed Gaussian parameters [amplitude, center, width, skewness]
    y : NDArray
        Observed y values
    x : NDArray
        x values

    Returns
    -------
    residuals : NDArray
        Residuals (y - fitted values) for the skewed Gaussian model

    Notes
    -----
    The function relies on the `gausssk_eval` function to compute the fitted
    values of the skewed Gaussian model given the parameters and x values.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> p = np.array([1.0, 0.0, 1.0, 0.5])  # amplitude, center, width, skewness
    >>> y = gausssk_eval(x, p) + np.random.normal(0, 0.1, len(x))
    >>> residuals = gaussskresiduals(p, y, x)
    >>> print(f"Mean residual: {np.mean(residuals):.6f}")
    """
    return y - gausssk_eval(x, p)


@conditionaljit()
def gaussresiduals(p: NDArray, y: NDArray, x: NDArray) -> NDArray:
    """
    Calculate residuals for Gaussian fit.

    This function computes the residuals (observed values minus fitted values)
    for a Gaussian function with parameters [amplitude, center, width].

    Parameters
    ----------
    p : NDArray
        Gaussian parameters [amplitude, center, width]
    y : NDArray
        Observed y values
    x : NDArray
        x values

    Returns
    -------
    NDArray
        Residuals (y - fitted values) where fitted values are calculated as:
        y_fit = amplitude * exp(-((x - center) ** 2) / (2 * width ** 2))

    Notes
    -----
    The Gaussian function is defined as:
    f(x) = amplitude * exp(-((x - center) ** 2) / (2 * width ** 2))

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([1.0, 0.0, 0.5])  # amplitude=1.0, center=0.0, width=0.5
    >>> y = np.array([0.5, 0.8, 1.0, 0.8, 0.5])
    >>> x = np.linspace(-2, 2, 5)
    >>> residuals = gaussresiduals(p, y, x)
    >>> print(residuals)
    """
    return y - p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * p[2] * p[2]))


def trapezoidresiduals(p: NDArray, y: NDArray, x: NDArray, toplength: float) -> NDArray:
    """
    Calculate residuals for trapezoid fit.

    This function computes the residuals (observed values minus fitted values) for a trapezoid
    function fit. The trapezoid is defined by amplitude, center, and width parameters, with
    a specified flat top length.

    Parameters
    ----------
    p : NDArray
        Trapezoid parameters [amplitude, center, width]
    y : NDArray
        Observed y values
    x : NDArray
        x values
    toplength : float
        Length of the flat top of the trapezoid

    Returns
    -------
    residuals : NDArray
        Residuals (y - fitted values)

    Notes
    -----
    The function uses `trapezoid_eval_loop` to evaluate the trapezoid function with the
    given parameters and returns the difference between observed and predicted values.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = trapezoid_eval_loop(x, 2.0, [1.0, 5.0, 3.0]) + np.random.normal(0, 0.1, 100)
    >>> p = [1.0, 5.0, 3.0]
    >>> residuals = trapezoidresiduals(p, y, x, 2.0)
    """
    return y - trapezoid_eval_loop(x, toplength, p)


def risetimeresiduals(p: NDArray, y: NDArray, x: NDArray) -> NDArray:
    """
    Calculate residuals for rise time fit.

    This function computes the residuals between observed data and fitted rise time model
    by subtracting the evaluated model from the observed values.

    Parameters
    ----------
    p : NDArray
        Rise time parameters [amplitude, start, rise time] where:
        - amplitude: peak value of the rise time curve
        - start: starting time of the rise
        - rise time: time constant for the rise process
    y : NDArray
        Observed y values (dependent variable)
    x : NDArray
        x values (independent variable, typically time)

    Returns
    -------
    residuals : NDArray
        Residuals (y - fitted values) representing the difference between
        observed data and model predictions

    Notes
    -----
    This function assumes the existence of a `risetime_eval_loop` function that
    evaluates the rise time model at given x values with parameters p.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([1.0, 0.0, 0.5])
    >>> y = np.array([0.1, 0.3, 0.7, 0.9])
    >>> x = np.array([0.0, 0.2, 0.4, 0.6])
    >>> residuals = risetimeresiduals(p, y, x)
    >>> print(residuals)
    """
    return y - risetime_eval_loop(x, p)


def gausssk_eval(x: NDArray, p: NDArray) -> NDArray:
    """
    Evaluate a skewed Gaussian function.

    This function computes a skewed Gaussian distribution using the method described
    by Azzalini and Dacunha (1996) for generating skewed normal distributions.

    Parameters
    ----------
    x : NDArray
        x values at which to evaluate the function
    p : NDArray
        Skewed Gaussian parameters [amplitude, center, width, skewness]
        - amplitude: scaling factor for the peak height
        - center: location parameter (mean of the underlying normal distribution)
        - width: scale parameter (standard deviation of the underlying normal distribution)
        - skewness: skewness parameter (controls the asymmetry of the distribution)

    Returns
    -------
    y : NDArray
        Evaluated skewed Gaussian values

    Notes
    -----
    The skewed Gaussian is defined as:
    f(x) = amplitude * φ((x-center)/width) * Φ(skewness * (x-center)/width)
    where φ is the standard normal PDF and Φ is the standard normal CDF.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> params = [1.0, 0.0, 1.0, 2.0]  # amplitude, center, width, skewness
    >>> y = gausssk_eval(x, params)
    """
    t = (x - p[1]) / p[2]
    return p[0] * sp.stats.norm.pdf(t) * sp.stats.norm.cdf(p[3] * t)


# @conditionaljit()
def kaiserbessel_eval(x: NDArray, p: NDArray) -> NDArray:
    """

    Evaluate the Kaiser-Bessel window function.

    This function computes the Kaiser-Bessel window function, which is commonly used in
    signal processing and medical imaging applications for gridding and convolution operations.
    The window is defined by parameters alpha (or beta) and tau (or W/2).

    Parameters
    ----------
    x : NDArray
        Arguments to the KB function, typically representing spatial or frequency coordinates
    p : NDArray
        The Kaiser-Bessel window parameters [alpha, tau] (wikipedia) or [beta, W/2] (Jackson, J. I., Meyer, C. H.,
        Nishimura, D. G. & Macovski, A. Selection of a convolution function for Fourier inversion using gridding
        [computerised tomography application]. IEEE Trans. Med. Imaging 10, 473–478 (1991))

    Returns
    -------
    NDArray
        The evaluated Kaiser-Bessel window function values corresponding to input x

    Notes
    -----
    The Kaiser-Bessel window is defined as:
    KB(x) = I0(α√(1-(x/τ)²)) / (τ * I0(α)) for |x| ≤ τ
    KB(x) = 0 for |x| > τ

    where I0 is the zeroth-order modified Bessel function of the first kind.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-1, 1, 100)
    >>> p = np.array([4.0, 0.5])  # alpha=4.0, tau=0.5
    >>> result = kaiserbessel_eval(x, p)
    """
    normfac = sps.i0(p[0] * np.sqrt(1.0 - np.square((0.0 / p[1])))) / p[1]
    sqrtargs = 1.0 - np.square((x / p[1]))
    sqrtargs[np.where(sqrtargs < 0.0)] = 0.0
    return np.where(
        np.fabs(x) <= p[1],
        sps.i0(p[0] * np.sqrt(sqrtargs)) / p[1] / normfac,
        0.0,
    )


@conditionaljit()
def gauss_eval(
    x: NDArray[np.floating[Any]], p: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """
    Evaluate a Gaussian function.

    This function computes the values of a Gaussian (normal) distribution
    at given x points with specified parameters.

    Parameters
    ----------
    x : NDArray[np.floating[Any]]
        x values at which to evaluate the Gaussian function
    p : NDArray[np.floating[Any]]
        Gaussian parameters [amplitude, center, width] where:
        - amplitude: peak height of the Gaussian
        - center: x-value of the Gaussian center
        - width: standard deviation of the Gaussian

    Returns
    -------
    y : NDArray[np.floating[Any]]
        Evaluated Gaussian values with the same shape as x

    Notes
    -----
    The Gaussian function is defined as:
    f(x) = amplitude * exp(-((x - center)^2) / (2 * width^2))

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> params = np.array([1.0, 0.0, 1.0])  # amplitude=1, center=0, width=1
    >>> y = gauss_eval(x, params)
    >>> print(y.shape)
    (100,)
    """
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * p[2] * p[2]))


def trapezoid_eval_loop(x: NDArray, toplength: float, p: NDArray) -> NDArray:
    """
    Evaluate a trapezoid function at multiple points using a loop.

    This function evaluates a trapezoid-shaped function at given x values. The trapezoid
    is defined by its amplitude, center, and total width, with the flat top length
    specified separately.

    Parameters
    ----------
    x : NDArray
        x values at which to evaluate the function
    toplength : float
        Length of the flat top of the trapezoid
    p : NDArray
        Trapezoid parameters [amplitude, center, width]

    Returns
    -------
    y : NDArray
        Evaluated trapezoid values

    Notes
    -----
    The trapezoid function is defined as:
    - Zero outside the range [center - width/2, center + width/2]
    - Linearly increasing from 0 to amplitude in the range [center - width/2, center - width/2 + toplength/2]
    - Constant at amplitude in the range [center - width/2 + toplength/2, center + width/2 - toplength/2]
    - Linearly decreasing from amplitude to 0 in the range [center + width/2 - toplength/2, center + width/2]

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> p = [1.0, 5.0, 4.0]  # amplitude=1.0, center=5.0, width=4.0
    >>> result = trapezoid_eval_loop(x, 2.0, p)
    """
    r = np.zeros(len(x), dtype="float64")
    for i in range(0, len(x)):
        r[i] = trapezoid_eval(x[i], toplength, p)
    return r


def risetime_eval_loop(x: NDArray, p: NDArray) -> NDArray:
    """
    Evaluate a rise time function.

    This function evaluates a rise time function for a given set of x values and parameters.
    It iterates through each x value and applies the risetime_eval function to compute
    the corresponding y values.

    Parameters
    ----------
    x : NDArray
        x values at which to evaluate the function
    p : NDArray
        Rise time parameters [amplitude, start, rise time]

    Returns
    -------
    y : NDArray
        Evaluated rise time function values

    Notes
    -----
    This function uses a loop-based approach for evaluating the rise time function.
    For better performance with large arrays, consider using vectorized operations
    instead of this loop-based implementation.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> p = np.array([1.0, 0.0, 1.0])
    >>> result = risetime_eval_loop(x, p)
    >>> print(result)
    [0. 0.63212056 0.86466472 0.95021293 0.98168436]
    """
    r = np.zeros(len(x), dtype="float64")
    for i in range(0, len(x)):
        r[i] = risetime_eval(x[i], p)
    return r


@conditionaljit()
def trapezoid_eval(
    x: Union[float, NDArray], toplength: float, p: NDArray
) -> Union[float, NDArray]:
    """
    Evaluate the trapezoidal function at given points.

    The trapezoidal function is defined as:

    f(x) = A * (1 - exp(-x / tau))   if 0 <= x < L

    f(x) = A * exp(-(x - L) / gamma) if x >= L

    where A, tau, and gamma are parameters, and L is the length of the top plateau.

    Parameters
    ----------
    x : float or NDArray
        The point or vector at which to evaluate the trapezoidal function.
    toplength : float
        The length of the top plateau of the trapezoid.
    p : NDArray
        A list or tuple of four values [A, tau, gamma, L] where:
        - A is the amplitude,
        - tau is the time constant for the rising edge,
        - gamma is the time constant for the falling edge,
        - L is the length of the top plateau.

    Returns
    -------
    float or NDArray
        The value of the trapezoidal function at x. Returns a scalar if x is scalar,
        or an array if x is an array.

    Notes
    -----
    This function is vectorized and can handle arrays of input points.

    Examples
    --------
    >>> import numpy as np
    >>> p = [1.0, 2.0, 3.0, 4.0]  # A=1.0, tau=2.0, gamma=3.0, L=4.0
    >>> trapezoid_eval(2.0, 4.0, p)
    0.3934693402873665
    >>> trapezoid_eval(np.array([1.0, 2.0, 5.0]), 4.0, p)
    array([0.39346934, 0.63212056, 0.22313016])
    """
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    elif 0.0 <= corrx < toplength:
        return p[1] * (1.0 - np.exp(-corrx / p[2]))
    else:
        return p[1] * (np.exp(-(corrx - toplength) / p[3]))


@conditionaljit()
def risetime_eval(
    x: Union[float, NDArray[np.floating[Any]]], p: NDArray[np.floating[Any]]
) -> Union[float, NDArray[np.floating[Any]]]:
    """
    Evaluates the rise time function at a given point.

    The rise time function is defined as:

    f(x) = A * (1 - exp(-x / tau))

    where A and tau are parameters.

    Parameters
    ----------
    x : float or NDArray
        The point at which to evaluate the rise time function.
    p : NDArray
        An array of three values [x0, A, tau] where:
        - x0: offset parameter
        - A: amplitude parameter
        - tau: time constant parameter

    Returns
    -------
    float or NDArray
        The value of the rise time function at x. Returns 0.0 if x < x0.

    Notes
    -----
    This function is vectorized and can handle arrays of input points.
    The function implements a shifted exponential rise function commonly used
    in signal processing and physics applications.

    Examples
    --------
    >>> import numpy as np
    >>> p = [1.0, 2.0, 0.5]  # x0=1.0, A=2.0, tau=0.5
    >>> risetime_eval(2.0, p)
    1.2642411176571153
    >>> risetime_eval(np.array([0.5, 1.5, 2.5]), p)
    array([0.        , 0.63212056, 1.26424112])
    """
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    else:
        return p[1] * (1.0 - np.exp(-corrx / p[2]))


def gasboxcar(
    data: NDArray[np.floating[Any]],
    samplerate: float,
    firstpeakstart: float,
    firstpeakend: float,
    secondpeakstart: float,
    secondpeakend: float,
    risetime: float = 3.0,
    falltime: float = 3.0,
) -> None:
    """
    Apply gas boxcar filtering to the input data.

    This function applies a gas boxcar filtering operation to the provided data array,
    which is commonly used in gas detection and analysis applications to smooth and
    enhance specific signal features.

    Parameters
    ----------
    data : NDArray
        Input data array to be filtered
    samplerate : float
        Sampling rate of the input data in Hz
    firstpeakstart : float
        Start time of the first peak in seconds
    firstpeakend : float
        End time of the first peak in seconds
    secondpeakstart : float
        Start time of the second peak in seconds
    secondpeakend : float
        End time of the second peak in seconds
    risetime : float, optional
        Rise time parameter for the boxcar filter in seconds, default is 3.0
    falltime : float, optional
        Fall time parameter for the boxcar filter in seconds, default is 3.0

    Returns
    -------
    None
        This function modifies the input data in-place and returns None

    Notes
    -----
    The gas boxcar filtering operation is designed to enhance gas detection signals
    by applying specific filtering parameters based on the peak timing information.
    The function assumes that the input data is properly formatted and that the
    time parameters are within the valid range of the data.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(1000)
    >>> gasboxcar(data, samplerate=100.0, firstpeakstart=10.0,
    ...           firstpeakend=15.0, secondpeakstart=20.0,
    ...           secondpeakend=25.0, risetime=2.0, falltime=2.0)
    """
    return None


# generate the polynomial fit timecourse from the coefficients
@conditionaljit()
def trendgen(
    thexvals: NDArray[np.floating[Any]], thefitcoffs: NDArray[np.floating[Any]], demean: bool
) -> NDArray[np.floating[Any]]:
    """
    Generate a polynomial trend based on input x-values and coefficients.

    This function constructs a polynomial trend using the provided x-values and
    a set of polynomial coefficients. The order of the polynomial is determined
    from the shape of the `thefitcoffs` array. Optionally, a constant term
    (the highest order coefficient) can be included or excluded from the trend.

    Parameters
    ----------
    thexvals : NDArray[np.floating[Any]]
        The x-values (independent variable) at which to evaluate the polynomial trend.
        Expected to be a numpy array or similar.
    thefitcoffs : NDArray[np.floating[Any]]
        A 1D array of polynomial coefficients. The length of this array minus one
        determines the order of the polynomial. Coefficients are expected to be
        ordered from the highest power of x down to the constant term (e.g.,
        [a_n, a_n-1, ..., a_1, a_0] for a polynomial a_n*x^n + ... + a_0).
    demean : bool
        If True, the constant term (thefitcoffs[order]) is added to the generated
        trend. If False, the constant term is excluded, effectively generating
        a trend that is "demeaned" or centered around zero (assuming the constant
        term represents the mean or offset).

    Returns
    -------
    NDArray[np.floating[Any]]
        A numpy array containing the calculated polynomial trend, with the same
        shape as `thexvals`.

    Notes
    -----
    This function implicitly assumes that `thexvals` is a numpy array or
    behaves similarly for element-wise multiplication (`np.multiply`).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 5)
    >>> coeffs = np.array([1, 0, 1])  # x^2 + 1
    >>> trendgen(x, coeffs, demean=True)
    array([1.    , 1.0625, 1.25  , 1.5625, 2.    ])
    >>> trendgen(x, coeffs, demean=False)
    array([-0.    , -0.0625, -0.25  , -0.5625, -1.    ])
    """
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


# @conditionaljit()
def detrend(
    inputdata: NDArray[np.floating[Any]], order: int = 1, demean: bool = False
) -> NDArray[np.floating[Any]]:
    """
    Estimates and removes a polynomial trend timecourse.

    This routine calculates a polynomial defined by a set of coefficients
    at specified time points to create a trend timecourse, and subtracts it
    from the input signal. Optionally, it can remove the mean of the input
    data as well.

    Parameters
    ----------
    inputdata : NDArray[np.floating[Any]]
        A 1D NumPy array of input data from which the trend will be removed.
    order : int, optional
        The order of the polynomial to fit to the data. Default is 1 (linear).
    demean : bool, optional
        If True, the mean of the input data is subtracted before fitting the
        polynomial trend. Default is False.

    Returns
    -------
    NDArray[np.floating[Any]]
        A 1D NumPy array of the detrended data, with the polynomial trend removed.

    Notes
    -----
    - This function uses `numpy.polynomial.Polynomial.fit` to fit a polynomial
      to the input data and then evaluates it using `trendgen`.
    - If a `RankWarning` is raised during fitting (e.g., due to insufficient
      data or poor conditioning), the function defaults to a zero-order
      polynomial (constant trend).
    - The time points are centered around zero, ranging from -N/2 to N/2,
      where N is the length of the input data.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> detrended = detrend(data, order=1)
    >>> print(detrended)
    [0. 0. 0. 0. 0.]
    """
    thetimepoints = np.arange(0.0, len(inputdata), 1.0) - len(inputdata) / 2.0
    try:
        thecoffs = Polynomial.fit(thetimepoints, inputdata, order).convert().coef[::-1]
    except np.exceptions.RankWarning:
        thecoffs = np.array([0.0, 0.0])
    thefittc = trendgen(thetimepoints, thecoffs, demean)
    return inputdata - thefittc


def prewhiten(
    series: NDArray[np.floating[Any]], nlags: Optional[int] = None, debug: bool = False
) -> NDArray[np.floating[Any]]:
    """
    Prewhiten a time series using an AR model estimated via statsmodels.
    The resulting series has the same length as the input.

    Parameters
    ----------
    series : NDArray[np.floating[Any]]
        Input 1D time series data.
    nlags : int, optional
        Order of the autoregressive model. If None, automatically chosen via AIC.
        Default is None.
    debug : bool, optional
        If True, additional debug information may be printed. Default is False.

    Returns
    -------
    whitened : NDArray[np.floating[Any]]
        Prewhitened series of the same length as input. The prewhitening removes
        the autoregressive structure from the data, leaving only the residuals.

    Notes
    -----
    This function fits an AR(p) model to the input series using `statsmodels.tsa.ARIMA`
    and applies the inverse AR filter to prewhiten the data. If `nlags` is not provided,
    the function automatically selects the best model order based on the Akaike Information Criterion (AIC).

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tsa.arima.model import ARIMA
    >>> series = np.random.randn(100)
    >>> whitened = prewhiten(series)
    >>> print(whitened.shape)
    (100,)
    """
    series = np.asarray(series)

    # Fit AR(p) model using ARIMA
    if nlags is None:
        best_aic, best_model, best_p = np.inf, None, None
        for p in range(1, min(10, len(series) // 5)):
            try:
                model = sm.tsa.ARIMA(series, order=(p, 0, 0)).fit()
                if model.aic < best_aic:
                    best_aic, best_model, best_p = model.aic, model, p
            except Exception:
                continue
        model = best_model
        if model is None:
            raise RuntimeError("Failed to fit any AR model.")
    else:
        model = sm.tsa.ARIMA(series, order=(nlags, 0, 0)).fit()

    # Extract AR coefficients and apply filter
    ar_params = model.arparams
    b = np.array([1.0])  # numerator (no MA component)
    a = np.r_[1.0, -ar_params]  # denominator (AR polynomial)

    # Apply the inverse AR filter (prewhitening)
    whitened = signal.lfilter(b, a, series)

    # return whitened, model
    return whitened


def prewhiten2(
    timecourse: NDArray[np.floating[Any]], nlags: int, debug: bool = False, sel: bool = False
) -> NDArray[np.floating[Any]]:
    """
    Prewhiten a time course using autoregressive modeling.

    This function applies prewhitening to a time course by fitting an autoregressive
    model and then applying the corresponding filter to remove temporal autocorrelation.

    Parameters
    ----------
    timecourse : NDArray[np.floating[Any]]
        Input time course to be prewhitened, shape (n_times,)
    nlags : int
        Number of lags to use for the autoregressive model
    debug : bool, optional
        If True, print model summary and display diagnostic plots, by default False
    sel : bool, optional
        If True, use automatic lag selection, by default False

    Returns
    -------
    NDArray[np.floating[Any]]
        Prewhitened time course with standardized normalization applied

    Notes
    -----
    The prewhitening process involves:
    1. Fitting an autoregressive model to the input time course
    2. Computing filter coefficients from the model parameters
    3. Applying the filter using scipy.signal.lfilter
    4. Standardizing the result using tide_math.stdnormalize

    When `sel=True`, the function uses `ar_select_order` for automatic lag selection
    instead of using the fixed number of lags specified by `nlags`.

    Examples
    --------
    >>> import numpy as np
    >>> timecourse = np.random.randn(100)
    >>> whitened = prewhiten2(timecourse, nlags=3)
    >>> # With debugging enabled
    >>> whitened = prewhiten2(timecourse, nlags=3, debug=True)
    """
    if not sel:
        ar_model = AutoReg(timecourse, lags=nlags)
        ar_fit = ar_model.fit()
    else:
        ar_model = ar_select_order(timecourse, nlags)
        ar_model.ar_lags
        ar_fit = ar_model.model.fit()
    if debug:
        print(ar_fit.summary())
        fig = plt.figure(figsize=(16, 9))
        fig = ar_fit.plot_diagnostics(fig=fig, lags=nlags)
        plt.show()
    ar_params = ar_fit.params

    # The prewhitening filter coefficients are 1 for the numerator and
    # (1, -ar_params[1]) for the denominator
    b = [1]
    a = np.insert(-ar_params[1:], 0, 1)

    # Apply the filter to prewhiten the signal
    return tide_math.stdnormalize(signal.lfilter(b, a, timecourse))


def findtrapezoidfunc(
    thexvals: NDArray[np.floating[Any]],
    theyvals: NDArray[np.floating[Any]],
    thetoplength: float,
    initguess: NDArray[np.floating[Any]] | None = None,
    debug: bool = False,
    minrise: float = 0.0,
    maxrise: float = 200.0,
    minfall: float = 0.0,
    maxfall: float = 200.0,
    minstart: float = -100.0,
    maxstart: float = 100.0,
    refine: bool = False,
    displayplots: bool = False,
) -> Tuple[float, float, float, float, int]:
    """
    Find the best-fitting trapezoidal function parameters to a data set.

    This function uses least-squares optimization to fit a trapezoidal function
    defined by `trapezoid_eval` to the input data (`theyvals`), using `thexvals`
    as the independent variable. The shape of the trapezoid is fixed by `thetoplength`.

    Parameters
    ----------
    thexvals : NDArray[np.floating[Any]]
        Independent variable values (time points) for the data.
    theyvals : NDArray[np.floating[Any]]
        Dependent variable values (signal intensity) corresponding to `thexvals`.
    thetoplength : float
        The length of the top plateau of the trapezoid function.
    initguess : NDArray[np.floating[Any]], optional
        Initial guess for [start, amplitude, risetime, falltime].
        If None, uses defaults based on data statistics.
    debug : bool, optional
        If True, print intermediate values during computation (default: False).
    minrise : float, optional
        Minimum allowed rise time parameter (default: 0.0).
    maxrise : float, optional
        Maximum allowed rise time parameter (default: 200.0).
    minfall : float, optional
        Minimum allowed fall time parameter (default: 0.0).
    maxfall : float, optional
        Maximum allowed fall time parameter (default: 200.0).
    minstart : float, optional
        Minimum allowed start time parameter (default: -100.0).
    maxstart : float, optional
        Maximum allowed start time parameter (default: 100.0).
    refine : bool, optional
        If True, perform additional refinement steps (not implemented in this version).
    displayplots : bool, optional
        If True, display plots during computation (not implemented in this version).

    Returns
    -------
    tuple of floats
        The fitted parameters [start, amplitude, risetime, falltime] if successful,
        or [0.0, 0.0, 0.0, 0.0] if the solution is outside the valid parameter bounds.
        A fifth value (integer) indicating success (1) or failure (0).

    Notes
    -----
    The optimization is performed using `scipy.optimize.leastsq` with a residual
    function `trapezoidresiduals`. The function returns a tuple of five elements:
    (start, amplitude, risetime, falltime, success_flag), where success_flag is 1
    if all parameters are within the specified bounds, and 0 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = trapezoid_eval(x, start=2, amplitude=5, risetime=1, falltime=1, top_length=4)
    >>> y += np.random.normal(0, 0.1, len(y))  # Add noise
    >>> params = findtrapezoidfunc(x, y, thetoplength=4)
    >>> print(params)
    (2.05, 4.98, 1.02, 1.01, 1)
    """
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
    plsq, dummy = sp.optimize.leastsq(
        trapezoidresiduals, p0, args=(theyvals, thexvals, thetoplength), maxfev=5000
    )
    # except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (
        (minrise <= plsq[2] <= maxrise)
        and (minfall <= plsq[3] <= maxfall)
        and (minstart <= plsq[0] <= maxstart)
    ):
        return plsq[0], plsq[1], plsq[2], plsq[3], 1
    else:
        return 0.0, 0.0, 0.0, 0.0, 0


def findrisetimefunc(
    thexvals: NDArray[np.floating[Any]],
    theyvals: NDArray[np.floating[Any]],
    initguess: NDArray[np.floating[Any]] | None = None,
    debug: bool = False,
    minrise: float = 0.0,
    maxrise: float = 200.0,
    minstart: float = -100.0,
    maxstart: float = 100.0,
    refine: bool = False,
    displayplots: bool = False,
) -> Tuple[float, float, float, int]:
    """
    Find the rise time of a signal by fitting a model to the data.

    This function fits a rise time model to the provided signal data using least squares
    optimization. It returns the estimated start time, amplitude, and rise time of the signal,
    along with a success flag indicating whether the fit is within specified bounds.

    Parameters
    ----------
    thexvals : NDArray[np.floating[Any]]
        Array of x-axis values (time or independent variable).
    theyvals : NDArray[np.floating[Any]]
        Array of y-axis values (signal or dependent variable).
    initguess : NDArray[np.floating[Any]] | None, optional
        Initial guess for [start_time, amplitude, rise_time]. If None, defaults are used.
    debug : bool, optional
        If True, prints the x and y values during processing (default is False).
    minrise : float, optional
        Minimum allowed rise time (default is 0.0).
    maxrise : float, optional
        Maximum allowed rise time (default is 200.0).
    minstart : float, optional
        Minimum allowed start time (default is -100.0).
    maxstart : float, optional
        Maximum allowed start time (default is 100.0).
    refine : bool, optional
        Placeholder for future refinement logic (default is False).
    displayplots : bool, optional
        Placeholder for future plotting logic (default is False).

    Returns
    -------
    Tuple[float, float, float, int]
        A tuple containing:
        - start_time: Estimated start time of the rise.
        - amplitude: Estimated amplitude of the rise.
        - rise_time: Estimated rise time.
        - success: 1 if the fit is within bounds, 0 otherwise.

    Notes
    -----
    The function uses `scipy.optimize.leastsq` to perform the fitting. The model being fitted
    is defined in the `risetimeresiduals` function, which must be defined elsewhere in the code.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.exp(-x / 2) * np.sin(x)
    >>> start, amp, rise, success = findrisetimefunc(x, y)
    >>> print(f"Start: {start}, Amplitude: {amp}, Rise Time: {rise}, Success: {success}")
    """
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
    plsq, dummy = sp.optimize.leastsq(
        risetimeresiduals, p0, args=(theyvals, thexvals), maxfev=5000
    )
    # except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (minrise <= plsq[2] <= maxrise) and (minstart <= plsq[0] <= maxstart):
        return plsq[0], plsq[1], plsq[2], 1
    else:
        return 0.0, 0.0, 0.0, 0


def territorydecomp(
    inputmap: NDArray[np.floating[Any]],
    template: NDArray,
    atlas: NDArray,
    inputmask: Optional[NDArray] = None,
    intercept: bool = True,
    fitorder: int = 1,
    debug: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Decompose an input map into territories defined by an atlas using polynomial regression.

    This function performs a decomposition of an input map (e.g., a brain image) into
    distinct regions (territories) as defined by an atlas. For each territory, it fits
    a polynomial model to the template values and the corresponding data in that region.
    The resulting coefficients are used to project the model back onto the original map.

    Parameters
    ----------
    inputmap : NDArray[np.floating[Any]]
        Input data to be decomposed. Can be 3D or 4D (e.g., time series).
    template : NDArray
        Template values corresponding to the spatial locations in `inputmap`.
        Should have the same shape as `inputmap` (or be broadcastable).
    atlas : NDArray
        Atlas defining the territories. Each unique integer value represents a distinct region.
        Must have the same shape as `inputmap`.
    inputmask : NDArray, optional
        Mask to define valid voxels in `inputmap`. If None, all voxels are considered valid.
        Should have the same shape as `inputmap`.
    intercept : bool, optional
        If True, include an intercept term in the polynomial fit (default: True).
    fitorder : int, optional
        The order of the polynomial to fit for each territory (default: 1).
    debug : bool, optional
        If True, print debugging information during computation (default: False).

    Returns
    -------
    tuple of NDArray
        A tuple containing:
        - fitmap : NDArray
            The decomposed map with fitted values projected back onto the original spatial locations.
        - thecoffs : NDArray
            Array of polynomial coefficients for each territory and map. Shape is (nummaps, numterritories, fitorder+1)
            if `intercept` is True, or (nummaps, numterritories, fitorder) otherwise.
        - theR2s : NDArray
            R-squared values for the fits for each territory and map. Shape is (nummaps, numterritories).

    Notes
    -----
    - The function assumes that `inputmap` and `template` are aligned in space.
    - If `inputmask` is not provided, all voxels are considered valid.
    - The number of territories is determined by the maximum value in `atlas`.
    - For each territory, a polynomial regression is performed using the template values as predictors.

    Examples
    --------
    >>> import numpy as np
    >>> inputmap = np.random.rand(10, 10, 10)
    >>> template = np.random.rand(10, 10, 10)
    >>> atlas = np.ones((10, 10, 10), dtype=int)
    >>> fitmap, coeffs, r2s = territorydecomp(inputmap, template, atlas)
    """
    datadims = len(inputmap.shape)
    if datadims > 3:
        nummaps = inputmap.shape[3]
    else:
        nummaps = 1

    if nummaps > 1:
        if inputmask is None:
            inputmask = np.ones_like(inputmap[:, :, :, 0])
    else:
        if inputmask is None:
            inputmask = np.ones_like(inputmap)

    tempmask = np.where(inputmask > 0.0, 1, 0)
    maskdims = len(tempmask.shape)
    if maskdims > 3:
        nummasks = tempmask.shape[3]
    else:
        nummasks = 1

    fitmap = np.zeros_like(inputmap)

    if intercept:
        thecoffs = np.zeros((nummaps, np.max(atlas), fitorder + 1))
    else:
        thecoffs = np.zeros((nummaps, np.max(atlas), fitorder))
    if debug:
        print(f"thecoffs.shape: {thecoffs.shape}")
        print(f"intercept: {intercept}, fitorder: {fitorder}")
    theR2s = np.zeros((nummaps, np.max(atlas)))
    for whichmap in range(nummaps):
        if nummaps == 1:
            thismap = inputmap
            thisfit = fitmap
        else:
            thismap = inputmap[:, :, :, whichmap]
            thisfit = fitmap[:, :, :, whichmap]
        if nummasks == 1:
            thismask = tempmask
        else:
            thismask = tempmask[:, :, :, whichmap]
        if nummaps > 1:
            print(f"decomposing map {whichmap + 1} of {nummaps}")
        for i in range(1, np.max(atlas) + 1):
            if debug:
                print("fitting territory", i)
            maskedvoxels = np.where(atlas * thismask == i)
            if len(maskedvoxels) > 0:
                if fitorder > 0:
                    evs = []
                    for order in range(1, fitorder + 1):
                        evs.append(np.power(template[maskedvoxels], order))
                    thefit, R2 = mlregress(
                        np.asarray(evs), thismap[maskedvoxels], intercept=intercept
                    )
                    thecoffs[whichmap, i - 1, :] = np.asarray(thefit[0]).reshape((-1))
                    theR2s[whichmap, i - 1] = 1.0 * R2
                    thisfit[maskedvoxels] = mlproject(thecoffs[whichmap, i - 1, :], evs, intercept)
                else:
                    thecoffs[whichmap, i - 1, 0] = np.mean(thismap[maskedvoxels])
                    theR2s[whichmap, i - 1] = 1.0
                    thisfit[maskedvoxels] = np.mean(thismap[maskedvoxels])

    return fitmap, thecoffs, theR2s


def territorystats(
    inputmap: NDArray[np.floating[Any]],
    atlas: NDArray,
    inputmask: NDArray | None = None,
    entropybins: int = 101,
    entropyrange: Tuple[float, float] | None = None,
    debug: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Compute descriptive statistics for regions defined by an atlas within a multi-dimensional input map.

    This function calculates various statistical measures (mean, standard deviation, median, etc.)
    for each region (territory) defined in the `atlas` array, based on the data in `inputmap`.
    It supports both single and multi-map inputs, and optionally uses a mask to define valid regions.

    Parameters
    ----------
    inputmap : NDArray[np.floating[Any]]
        Input data array of shape (X, Y, Z) or (X, Y, Z, N), where N is the number of maps.
    atlas : ndarray
        Atlas array defining regions of interest, with each region labeled by an integer.
        Must be the same spatial dimensions as `inputmap`.
    inputmask : ndarray, optional
        Boolean or binary mask array of the same shape as `inputmap`. If None, all voxels are considered valid.
    entropybins : int, default=101
        Number of bins to use when computing entropy.
    entropyrange : tuple of float, optional
        Range (min, max) for histogram binning when computing entropy. If None, uses the full range of data.
    debug : bool, default=False
        If True, prints debug information during computation.

    Returns
    -------
    tuple of ndarray
        A tuple containing:
        - statsmap : ndarray
            Zero-initialized array of the same shape as `inputmap`, used for storing statistics.
        - themeans : ndarray
            Array of shape (N, max(atlas)) containing the mean values for each region in each map.
        - thestds : ndarray
            Array of shape (N, max(atlas)) containing the standard deviations for each region in each map.
        - themedians : ndarray
            Array of shape (N, max(atlas)) containing the median values for each region in each map.
        - themads : ndarray
            Array of shape (N, max(atlas)) containing the median absolute deviations for each region in each map.
        - thevariances : ndarray
            Array of shape (N, max(atlas)) containing the variance values for each region in each map.
        - theskewnesses : ndarray
            Array of shape (N, max(atlas)) containing the skewness values for each region in each map.
        - thekurtoses : ndarray
            Array of shape (N, max(atlas)) containing the kurtosis values for each region in each map.
        - theentropies : ndarray
            Array of shape (N, max(atlas)) containing the entropy values for each region in each map.

    Notes
    -----
    - The function supports both 3D and 4D input arrays. For 4D arrays, each map is processed separately.
    - Entropy is computed using the probability distribution from a histogram of voxel values.
    - If `inputmask` is not provided, all voxels are considered valid.
    - The `atlas` labels are expected to start from 1, and regions are indexed accordingly.

    Examples
    --------
    >>> import numpy as np
    >>> inputmap = np.random.rand(10, 10, 10)
    >>> atlas = np.ones((10, 10, 10), dtype=int)
    >>> statsmap, means, stds, medians, mads, variances, skewnesses, kurtoses, entropies = territorystats(inputmap, atlas)
    """
    datadims = len(inputmap.shape)
    if datadims > 3:
        nummaps = inputmap.shape[3]
    else:
        nummaps = 1

    if nummaps > 1:
        if inputmask is None:
            inputmask = np.ones_like(inputmap[:, :, :, 0])
    else:
        if inputmask is None:
            inputmask = np.ones_like(inputmap)

    tempmask = np.where(inputmask > 0.0, 1, 0)
    maskdims = len(tempmask.shape)
    if maskdims > 3:
        nummasks = tempmask.shape[3]
    else:
        nummasks = 1

    statsmap = np.zeros_like(inputmap)

    themeans = np.zeros((nummaps, np.max(atlas)))
    thestds = np.zeros((nummaps, np.max(atlas)))
    themedians = np.zeros((nummaps, np.max(atlas)))
    themads = np.zeros((nummaps, np.max(atlas)))
    thevariances = np.zeros((nummaps, np.max(atlas)))
    theskewnesses = np.zeros((nummaps, np.max(atlas)))
    thekurtoses = np.zeros((nummaps, np.max(atlas)))
    theentropies = np.zeros((nummaps, np.max(atlas)))
    if entropyrange is None:
        if inputmask is not None:
            thevoxels = inputmap[np.where(inputmask > 0.0)]
        else:
            thevoxels = inputmap
        entropyrange = (np.min(thevoxels), np.max(thevoxels))
    if debug:
        print(f"entropy bins: {entropybins}")
        print(f"entropy range: {entropyrange}")
        print(f"themeans.shape: {themeans.shape}")
    for whichmap in range(nummaps):
        if nummaps == 1:
            thismap = inputmap
            thisstats = statsmap
        else:
            thismap = inputmap[:, :, :, whichmap]
            thisstats = statsmap[:, :, :, whichmap]
        if nummasks == 1:
            thismask = tempmask
        else:
            thismask = tempmask[:, :, :, whichmap]
        if nummaps > 1:
            print(f"calculating stats for map {whichmap + 1} of {nummaps}")
        for i in range(1, np.max(atlas) + 1):
            if debug:
                print("calculating stats for territory", i)
            maskedvoxels = np.where(atlas * thismask == i)
            if len(maskedvoxels) > 0:
                themeans[whichmap, i - 1] = np.mean(thismap[maskedvoxels])
                thestds[whichmap, i - 1] = np.std(thismap[maskedvoxels])
                themedians[whichmap, i - 1] = np.median(thismap[maskedvoxels])
                themads[whichmap, i - 1] = mad(thismap[maskedvoxels])
                thevariances[whichmap, i - 1] = moment(thismap[maskedvoxels], order=2)
                theskewnesses[whichmap, i - 1] = moment(thismap[maskedvoxels], order=3)
                thekurtoses[whichmap, i - 1] = moment(thismap[maskedvoxels], order=4)
                theentropies[whichmap, i - 1] = entropy(
                    np.histogram(
                        thismap[maskedvoxels], bins=entropybins, range=entropyrange, density=True
                    )[0]
                )

    return (
        statsmap,
        themeans,
        thestds,
        themedians,
        themads,
        thevariances,
        theskewnesses,
        thekurtoses,
        theentropies,
    )


@conditionaljit()
def refinepeak_quad(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]], peakindex: int, stride: int = 1
) -> Tuple[float, float, float, Optional[bool], bool]:
    """
    Refine the location and properties of a peak using quadratic interpolation.

    This function takes a peak index and a set of data points to perform
    quadratic interpolation around the peak to estimate its precise location,
    value, and width. It also determines whether the point is a local maximum or minimum.

    Parameters
    ----------
    x : NDArray[np.floating[Any]]
        Independent variable values (e.g., time points).
    y : NDArray[np.floating[Any]]
        Dependent variable values (e.g., signal intensity) corresponding to `x`.
    peakindex : int
        Index of the peak in the arrays `x` and `y`.
    stride : int, optional
        Number of data points to use on either side of the peak for interpolation.
        Default is 1.

    Returns
    -------
    tuple
        A tuple containing:
        - peakloc : float
            The refined location of the peak.
        - peakval : float
            The refined value at the peak.
        - peakwidth : float
            The estimated width of the peak.
        - ismax : bool or None
            True if the point is a local maximum, False if it's a local minimum,
            and None if the point cannot be determined (e.g., at boundaries).
        - badfit : bool
            True if the fit could not be performed due to invalid conditions,
            such as being at the boundary or having equal values on both sides.

    Notes
    -----
    The function uses a quadratic fit to estimate peak properties. It checks for
    valid conditions before performing the fit, including ensuring that the peak
    is not at the edge of the data and that it's either a local maximum or minimum.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.exp(-0.5 * (x - 5)**2) + 0.1 * np.random.random(100)
    >>> peakloc, peakval, peakwidth, ismax, badfit = refinepeak_quad(x, y, 50, stride=2)
    >>> print(f"Peak location: {peakloc:.2f}, Peak value: {peakval:.2f}")
    """
    # first make sure this actually is a peak
    ismax = None
    badfit = False
    if peakindex < stride - 1 or peakindex > len(x) - 1 - stride:
        print("cannot estimate peak location at end points")
        return 0.0, 0.0, 0.0, None, True
    if y[peakindex] >= y[peakindex - stride] and y[peakindex] >= y[peakindex + stride]:
        ismax = True
    elif y[peakindex] <= y[peakindex - stride] and y[peakindex] <= y[peakindex + stride]:
        ismax = False
    else:
        badfit = True
    if y[peakindex] == y[peakindex - stride] and y[peakindex] == y[peakindex + stride]:
        badfit = True

    # now find all the information about the peak
    alpha = y[peakindex - stride]
    beta = y[peakindex]
    gamma = y[peakindex + stride]
    binsize = x[peakindex + stride] - x[peakindex]
    offsetbins = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma)
    peakloc = x[peakindex] + offsetbins * binsize
    peakval = beta - 0.25 * (alpha - gamma) * offsetbins
    a = np.square(x[peakindex - stride] - peakloc) / (alpha - peakval)
    peakwidth = np.sqrt(np.fabs(a) / 2.0)
    return peakloc, peakval, peakwidth, ismax, badfit


@conditionaljit2()
def findmaxlag_gauss(
    thexcorr_x: NDArray[np.floating[Any]],
    thexcorr_y: NDArray[np.floating[Any]],
    lagmin: float,
    lagmax: float,
    widthmax: float,
    edgebufferfrac: float = 0.0,
    threshval: float = 0.0,
    uthreshval: float = 30.0,
    debug: bool = False,
    tweaklims: bool = True,
    zerooutbadfit: bool = True,
    refine: bool = False,
    maxguess: float = 0.0,
    useguess: bool = False,
    searchfrac: float = 0.5,
    fastgauss: bool = False,
    lagmod: float = 1000.0,
    enforcethresh: bool = True,
    absmaxsigma: float = 1000.0,
    absminsigma: float = 0.1,
    displayplots: bool = False,
) -> Tuple[int, np.float64, np.float64, np.float64, np.uint16, np.uint16, int, int]:
    """
        Find the maximum lag in a cross-correlation function by fitting a Gaussian curve to the peak.

        This function locates the peak in a cross-correlation function and optionally fits a Gaussian
        curve to determine the precise lag time, amplitude, and width. It includes extensive error
        checking and validation to ensure robust results.

        Parameters
        ----------
        thexcorr_x : NDArray[np.floating[Any]]
            X-axis values (lag times) of the cross-correlation function.
        thexcorr_y : NDArray[np.floating[Any]]
            Y-axis values (correlation coefficients) of the cross-correlation function.
        lagmin : float
            Minimum allowable lag value in seconds.
        lagmax : float
            Maximum allowable lag value in seconds.
        widthmax : float
            Maximum allowable width of the Gaussian peak in seconds.
        edgebufferfrac : float, optional
            Fraction of array length to exclude from each edge during search. Default is 0.0.
        threshval : float, optional
            Minimum correlation threshold for a valid peak. Default is 0.0.
        uthreshval : float, optional
            Upper threshold value (currently unused). Default is 30.0.
        debug : bool, optional
            Enable debug output showing initial vs final parameter values. Default is False.
        tweaklims : bool, optional
            Automatically adjust search limits to avoid edge artifacts. Default is True.
        zerooutbadfit : bool, optional
            Set output to zero when fit fails rather than using initial guess. Default is True.
        refine : bool, optional
            Perform least-squares refinement of the Gaussian fit. Default is False.
        maxguess : float, optional
            Initial guess for maximum lag position. Used when useguess=True. Default is 0.0.
        useguess : bool, optional
            Use the provided maxguess instead of finding peak automatically. Default is False.
        searchfrac : float, optional
            Fraction of peak height used to determine initial width estimate. Default is 0.5.
        fastgauss : bool, optional
            Use fast non-iterative Gaussian fitting (less accurate). Default is False.
        lagmod : float, optional
            Modulus for lag values to handle wraparound. Default is 1000.0.
        enforcethresh : bool, optional
            Enforce minimum threshold requirements. Default is True.
        absmaxsigma : float, optional
            Absolute maximum allowed sigma (width) value. Default is 1000.0.
        absminsigma : float, optional
            Absolute minimum allowed sigma (width) value. Default is 0.1.
        displayplots : bool, optional
            Show matplotlib plots of data and fitted curve. Default is False.

        Returns
        -------
        maxindex : int
            Array index of the maximum correlation value.
        maxlag : numpy.float64
            Time lag at maximum correlation in seconds.
        maxval : numpy.float64
            Maximum correlation coefficient value.
        maxsigma : numpy.float64
            Width (sigma) of the fitted Gaussian peak.
        maskval : numpy.uint16
            Validity mask (1 = valid fit, 0 = invalid fit).
        failreason : numpy.uint16
            Bitwise failure reason code. Possible values:
            - 0x01: Correlation amplitude below threshold
            - 0x02: Correlation amplitude above maximum (>1.0)
            - 0x04: Search window too narrow (<3 points)
            - 0x08: Fitted width exceeds widthmax
            - 0x10: Fitted lag outside [lagmin, lagmax] range
            - 0x20: Peak found at edge of search range
            - 0x40: Fitting procedure failed
            - 0x80: Initial parameter estimation failed
        fitstart : int
            Starting index used for fitting.
        fitend : int
            Ending index used for fitting.

        Notes
        -----
        - The function assumes cross-correlation data where Y-values represent correlation
          coefficients (typically in range [-1, 1]).
        - When refine=False, uses simple peak-finding based on maximum value.
        - When refine=True, performs least-squares Gaussian fit for sub-bin precision.
        - All time-related parameters (lagmin, lagmax, widthmax) should be in the same
          units as thexcorr_x.
        - The fastgauss option provides faster but less accurate non-iterative fitting.

        Examples
        --------
        Basic usage without refinement:

        >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = \\
        ...     findmaxlag_gauss(lag_times, correlations, -10.0, 10.0, 5.0)
        >>> if maskval == 1:
        ...     print(f"Peak found at lag: {maxlag:.3f} s, correlation: {maxval:.3f}")

        Advanced usage with refinement:

        >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = \\
        ...     findmaxlag_gauss(lag_times, correlations, -5.0, 5.0, 2.0,
        ...                      refine=True, threshval=0.1, displayplots=True)

        Using an initial guess:

        >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = \\
        ...     findmaxlag_gauss(lag_times, correlations, -10.0, 10.0, 3.0,
        ...                      useguess=True, maxguess=2.5, refine=True)
        """
    """
    Find the maximum lag in a cross-correlation function by fitting a Gaussian curve to the peak.

    This function locates the peak in a cross-correlation function and optionally fits a Gaussian
    curve to determine the precise lag time, amplitude, and width. It includes extensive error
    checking and validation to ensure robust results.

    Parameters
    ----------
    thexcorr_x : NDArray
        X-axis values (lag times) of the cross-correlation function.
    thexcorr_y : NDArray
        Y-axis values (correlation coefficients) of the cross-correlation function.
    lagmin : float
        Minimum allowable lag value in seconds.
    lagmax : float
        Maximum allowable lag value in seconds.
    widthmax : float
        Maximum allowable width of the Gaussian peak in seconds.
    edgebufferfrac : float, optional
        Fraction of array length to exclude from each edge during search. Default is 0.0.
    threshval : float, optional
        Minimum correlation threshold for a valid peak. Default is 0.0.
    uthreshval : float, optional
        Upper threshold value (currently unused). Default is 30.0.
    debug : bool, optional
        Enable debug output showing initial vs final parameter values. Default is False.
    tweaklims : bool, optional
        Automatically adjust search limits to avoid edge artifacts. Default is True.
    zerooutbadfit : bool, optional
        Set output to zero when fit fails rather than using initial guess. Default is True.
    refine : bool, optional
        Perform least-squares refinement of the Gaussian fit. Default is False.
    maxguess : float, optional
        Initial guess for maximum lag position. Used when useguess=True. Default is 0.0.
    useguess : bool, optional
        Use the provided maxguess instead of finding peak automatically. Default is False.
    searchfrac : float, optional
        Fraction of peak height used to determine initial width estimate. Default is 0.5.
    fastgauss : bool, optional
        Use fast non-iterative Gaussian fitting (less accurate). Default is False.
    lagmod : float, optional
        Modulus for lag values to handle wraparound. Default is 1000.0.
    enforcethresh : bool, optional
        Enforce minimum threshold requirements. Default is True.
    absmaxsigma : float, optional
        Absolute maximum allowed sigma (width) value. Default is 1000.0.
    absminsigma : float, optional
        Absolute minimum allowed sigma (width) value. Default is 0.1.
    displayplots : bool, optional
        Show matplotlib plots of data and fitted curve. Default is False.

    Returns
    -------
    maxindex : int
        Array index of the maximum correlation value.
    maxlag : numpy.float64
        Time lag at maximum correlation in seconds.
    maxval : numpy.float64
        Maximum correlation coefficient value.
    maxsigma : numpy.float64
        Width (sigma) of the fitted Gaussian peak.
    maskval : numpy.uint16
        Validity mask (1 = valid fit, 0 = invalid fit).
    failreason : numpy.uint16
        Bitwise failure reason code. Possible values:
        - 0x01: Correlation amplitude below threshold
        - 0x02: Correlation amplitude above maximum (>1.0)
        - 0x04: Search window too narrow (<3 points)
        - 0x08: Fitted width exceeds widthmax
        - 0x10: Fitted lag outside [lagmin, lagmax] range
        - 0x20: Peak found at edge of search range
        - 0x40: Fitting procedure failed
        - 0x80: Initial parameter estimation failed
    fitstart : int
        Starting index used for fitting.
    fitend : int
        Ending index used for fitting.

    Notes
    -----
    - The function assumes cross-correlation data where Y-values represent correlation
      coefficients (typically in range [-1, 1]).
    - When refine=False, uses simple peak-finding based on maximum value.
    - When refine=True, performs least-squares Gaussian fit for sub-bin precision.
    - All time-related parameters (lagmin, lagmax, widthmax) should be in the same
      units as thexcorr_x.
    - The fastgauss option provides faster but less accurate non-iterative fitting.

    Examples
    --------
    Basic usage without refinement:

    >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = \\
    ...     findmaxlag_gauss(lag_times, correlations, -10.0, 10.0, 5.0)
    >>> if maskval == 1:
    ...     print(f"Peak found at lag: {maxlag:.3f} s, correlation: {maxval:.3f}")

    Advanced usage with refinement:

    >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = \\
    ...     findmaxlag_gauss(lag_times, correlations, -5.0, 5.0, 2.0,
    ...                      refine=True, threshval=0.1, displayplots=True)

    Using an initial guess:

    >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = \\
    ...     findmaxlag_gauss(lag_times, correlations, -10.0, 10.0, 3.0,
    ...                      useguess=True, maxguess=2.5, refine=True)
    """
    # set initial parameters
    # widthmax is in seconds
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
    searchbins = int(widthmax // binwidth)
    lowerlim = int(numlagbins * edgebufferfrac)
    upperlim = numlagbins - lowerlim - 1
    if tweaklims:
        lowerlim = 0
        upperlim = numlagbins - 1
        while (thexcorr_y[lowerlim + 1] < thexcorr_y[lowerlim]) and (lowerlim + 1) <= upperlim:
            lowerlim += 1
        while (thexcorr_y[upperlim - 1] < thexcorr_y[upperlim]) and (upperlim - 1) > lowerlim:
            upperlim -= 1
    FML_BADAMPLOW = np.uint16(0x01)
    FML_BADAMPHIGH = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value
    if useguess:
        maxindex = tide_util.valtoindex(thexcorr_x, maxguess)
        nlowerlim = int(maxindex - widthmax / 2.0)
        nupperlim = int(maxindex + widthmax / 2.0)
        if nlowerlim < lowerlim:
            nlowerlim = lowerlim
            nupperlim = lowerlim + int(widthmax)
        if nupperlim > upperlim:
            nupperlim = upperlim
            nlowerlim = upperlim - int(widthmax)
        maxval_init = thexcorr_y[maxindex].astype("float64")
    else:
        maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype("int32")
        maxval_init = thexcorr_y[maxindex].astype("float64")

    # now get a location for that value
    maxlag_init = (1.0 * thexcorr_x[maxindex]).astype("float64")

    # and calculate the width of the peak
    upperlimit = len(thexcorr_y) - 1
    lowerlimit = 0
    i = 0
    j = 0
    while (
        (maxindex + i <= upperlimit)
        and (thexcorr_y[maxindex + i] > searchfrac * maxval_init)
        and (i < searchbins)
    ):
        i += 1
    if (maxindex + i > upperlimit) or (i > searchbins):
        i -= 1
    while (
        (maxindex - j >= lowerlimit)
        and (thexcorr_y[maxindex - j] > searchfrac * maxval_init)
        and (j < searchbins)
    ):
        j += 1
    if (maxindex - j < lowerlimit) or (j > searchbins):
        j -= 1
    # This is calculated from first principles, but it's always big by a factor or ~1.4.
    #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division.
    if searchfrac <= 0 or searchfrac >= 1:
        raise ValueError("searchfrac must be between 0 and 1 (exclusive)")
    maxsigma_init = np.float64(
        ((i + j + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0)
    )

    # now check the values for errors and refine if necessary
    fitend = min(maxindex + i + 1, upperlimit)
    fitstart = max(1, maxindex - j)

    if not (lagmin <= maxlag_init <= lagmax):
        failreason += FML_HITEDGE
        if maxlag_init <= lagmin:
            maxlag_init = lagmin
        else:
            maxlag_init = lagmax

    if i + j + 1 < 3:
        failreason += FML_BADSEARCHWINDOW
        maxsigma_init = np.float64(
            (3.0 * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0)
        )
    if maxsigma_init > widthmax:
        failreason += FML_BADWIDTH
        maxsigma_init = np.float64(widthmax)
    if (maxval_init < threshval) and enforcethresh:
        failreason += FML_BADAMPLOW
    if maxval_init < 0.0:
        failreason += FML_BADAMPLOW
        maxval_init = np.float64(0.0)
    if maxval_init > 1.0:
        failreason |= FML_BADAMPHIGH
        maxval_init = np.float64(1.0)
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
                maxsigma = np.float64(
                    np.sqrt(np.fabs(np.sum((X - maxlag) ** 2 * data) / np.sum(data)))
                )
                maxval = np.float64(data.max())
            else:
                # do a least squares fit over the top of the peak
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype="float64")

                if fitend - fitstart >= 3:
                    try:
                        plsq, ier = sp.optimize.leastsq(
                            gaussresiduals, p0, args=(data, X), maxfev=5000
                        )
                        if ier not in [1, 2, 3, 4]:  # Check for successful convergence
                            maxval = np.float64(0.0)
                            maxlag = np.float64(0.0)
                            maxsigma = np.float64(0.0)
                        else:
                            maxval = plsq[0]
                            maxlag = np.fmod((1.0 * plsq[1]), lagmod)
                            maxsigma = plsq[2]
                    except:
                        maxval = np.float64(0.0)
                        maxlag = np.float64(0.0)
                        maxsigma = np.float64(0.0)
                # if maxval > 1.0, fit failed catastrophically, zero out or reset to initial value
                #     corrected logic for 1.1.6
                if (np.fabs(maxval)) > 1.0 or (lagmin > maxlag) or (maxlag > lagmax):
                    if zerooutbadfit:
                        maxval = np.float64(0.0)
                        maxlag = np.float64(0.0)
                        maxsigma = np.float64(0.0)
                        maskval = np.uint16(0)
                    else:
                        maxval = np.float64(maxval_init)
                        maxlag = np.float64(maxlag_init)
                        maxsigma = np.float64(maxsigma_init)
                if not absminsigma <= maxsigma <= absmaxsigma:
                    if zerooutbadfit:
                        maxval = np.float64(0.0)
                        maxlag = np.float64(0.0)
                        maxsigma = np.float64(0.0)
                        maskval = np.uint16(0)
                    else:
                        if maxsigma > absmaxsigma:
                            maxsigma = np.float64(absmaxsigma)
                        else:
                            maxsigma = np.float64(absminsigma)

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
        print(
            "init to final: maxval",
            maxval_init,
            maxval,
            ", maxlag:",
            maxlag_init,
            maxlag,
            ", width:",
            maxsigma_init,
            maxsigma,
        )
    if displayplots and refine and (maskval != 0.0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Data and fit")
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        plt.plot(
            X,
            data,
            "ro",
            hiresx,
            gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])),
            "b-",
        )
        plt.show()
    return maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend


@conditionaljit2()
def maxindex_noedge(
    thexcorr_x: NDArray, thexcorr_y: NDArray, bipolar: bool = False
) -> Tuple[int, float]:
    """
    Find the index of the maximum value in cross-correlation data, avoiding edge effects.

    This function searches for the maximum value in the cross-correlation data while
    ensuring that the result is not located at the edges of the data array. It handles
    both unipolar and bipolar cases, returning the index and a flip factor for bipolar
    cases where the minimum absolute value might be larger than the maximum.

    Parameters
    ----------
    thexcorr_x : NDArray
        Array containing the x-coordinates of the cross-correlation data
    thexcorr_y : NDArray
        Array containing the y-coordinates (cross-correlation values) of the data
    bipolar : bool, optional
        If True, considers both positive and negative values when finding the maximum.
        If False, only considers positive values. Default is False.

    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - int: The index of the maximum value in the cross-correlation data
        - float: Flip factor (-1.0 if bipolar case and minimum absolute value is larger,
          1.0 otherwise)

    Notes
    -----
    The function iteratively adjusts the search range to avoid edge effects by
    incrementing lowerlim when maxindex is 0, and decrementing upperlim when
    maxindex equals upperlim. This ensures the returned index is not at the boundaries
    of the input arrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = np.array([0.1, 0.5, 0.8, 0.3, 0.2])
    >>> index, flip = maxindex_noedge(x, y)
    >>> print(index)
    2
    >>> print(flip)
    1.0
    """
    lowerlim = 0
    upperlim = len(thexcorr_x) - 1
    done = False
    while not done:
        flipfac = 1.0
        done = True
        maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype("int32")
        if bipolar:
            minindex = (np.argmax(np.fabs(thexcorr_y[lowerlim:upperlim])) + lowerlim).astype(
                "int32"
            )
            if np.fabs(thexcorr_y[minindex]) > np.fabs(thexcorr_y[maxindex]):
                maxindex = minindex
                flipfac = -1.0
        else:
            maxindex = (np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim).astype("int32")
        if upperlim == lowerlim:
            done = True
        if maxindex == 0:
            lowerlim += 1
            done = False
        if maxindex == upperlim:
            upperlim -= 1
            done = False
    return maxindex, flipfac


def gaussfitsk(
    height: float, loc: float, width: float, skewness: float, xvals: ArrayLike, yvals: ArrayLike
) -> NDArray:
    """
    Fit a skewed Gaussian function to data using least squares optimization.

    This function performs least squares fitting of a skewed Gaussian model to the
    provided data points. The model includes parameters for height, location, width,
    and skewness of the Gaussian distribution.

    Parameters
    ----------
    height : float
        The amplitude or height of the Gaussian peak.
    loc : float
        The location (mean) of the Gaussian peak.
    width : float
        The width (standard deviation) of the Gaussian peak.
    skewness : float
        The skewness parameter that controls the asymmetry of the Gaussian.
    xvals : array-like
        The x-coordinates of the data points to be fitted.
    yvals : array-like
        The y-coordinates of the data points to be fitted.

    Returns
    -------
    ndarray
        Array containing the optimized parameters [height, loc, width, skewness] that
        best fit the data according to the least squares method.

    Notes
    -----
    This function uses `scipy.optimize.leastsq` internally for the optimization
    process. The fitting is performed using the `gaussskresiduals` residual function
    which should be defined elsewhere in the codebase.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = gaussfitsk(1.0, 0.0, 1.0, 0.0, x, y_data)
    >>> print(y)
    [height_opt, loc_opt, width_opt, skewness_opt]
    """
    plsq, dummy = sp.optimize.leastsq(
        gaussskresiduals,
        np.array([height, loc, width, skewness]),
        args=(yvals, xvals),
        maxfev=5000,
    )
    return plsq


def gaussfunc(x: NDArray, height: float, loc: float, FWHM: float) -> NDArray:
    """
    Calculate a Gaussian function.

    This function computes a Gaussian (normal) distribution with specified height,
    location, and Full Width at Half Maximum (FWHM).

    Parameters
    ----------
    x : NDArray
        Array of values at which to evaluate the Gaussian function.
    height : float
        The maximum height of the Gaussian curve.
    loc : float
        The location (mean) of the Gaussian curve.
    FWHM : float
        The Full Width at Half Maximum of the Gaussian curve.

    Returns
    -------
    NDArray
        Array of Gaussian function values evaluated at x.

    Notes
    -----
    The Gaussian function is defined as:
    f(x) = height * exp(-((x - loc) ** 2) / (2 * (FWHM / 2.355) ** 2))

    The conversion from FWHM to standard deviation (sigma) uses the relationship:
    sigma = FWHM / (2 * sqrt(2 * log(2))) ≈ FWHM / 2.355

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = gaussfunc(x, height=1.0, loc=0.0, FWHM=2.0)
    >>> print(y.shape)
    (100,)
    """
    return height * np.exp(-((x - loc) ** 2) / (2 * (FWHM / 2.355) ** 2))


def gaussfit2(
    height: float, loc: float, width: float, xvals: NDArray, yvals: NDArray
) -> Tuple[float, float, float]:
    """
    Calculate a Gaussian function.

    This function computes a Gaussian (normal) distribution with specified height,
    location, and Full Width at Half Maximum (FWHM).

    Parameters
    ----------
    x : array_like
        Input values for which to compute the Gaussian function
    height : float
        Height (amplitude) of the Gaussian peak
    loc : float
        Location (mean) of the Gaussian peak
    FWHM : float
        Full Width at Half Maximum of the Gaussian peak

    Returns
    -------
    ndarray
        Array of Gaussian function values computed at input x values

    Notes
    -----
    The Gaussian function is computed using the formula:
    f(x) = height * exp(-((x - loc)^2) / (2 * (FWHM / 2.355)^2))

    The conversion from FWHM to sigma (standard deviation) uses the relationship:
    sigma = FWHM / 2.355

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = gaussfunc(x, height=1.0, loc=0.0, FWHM=2.0)
    >>> print(y.shape)
    (100,)
    """
    popt, pcov = curve_fit(gaussfunc, xvals, yvals, p0=[height, loc, width])
    return popt[0], popt[1], popt[2]


def sincfunc(x: NDArray, height: float, loc: float, FWHM: float, baseline: float) -> NDArray:
    """
    Compute a scaled and shifted sinc function.

    This function evaluates a sinc function with specified height, location,
    full width at half maximum, and baseline offset. The sinc function is
    scaled by a factor that relates the FWHM to the sinc function's natural
    scaling.

    Parameters
    ----------
    x : NDArray
        Input array of values where the function is evaluated.
    height : float
        Height of the sinc function peak.
    loc : float
        Location (center) of the sinc function peak.
    FWHM : float
        Full width at half maximum of the sinc function.
    baseline : float
        Baseline offset added to the sinc function values.

    Returns
    -------
    NDArray
        Array of sinc function values with the same shape as input `x`.

    Notes
    -----
    The sinc function is defined as sin(πx)/(πx) with the convention that
    sinc(0) = 1. The scaling factor 3.79098852 is chosen to relate the FWHM
    to the natural sinc function properties.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = sincfunc(x, height=2.0, loc=0.0, FWHM=1.0, baseline=1.0)
    >>> print(y.shape)
    (100,)
    """
    return height * np.sinc((3.79098852 / (FWHM * np.pi)) * (x - loc)) + baseline


# found this sinc fitting routine (and optimization) here:
# https://stackoverflow.com/questions/49676116/why-cant-scipy-optimize-curve-fit-fit-my-data-using-a-numpy-sinc-function
def sincfit(
    height: float, loc: float, width: float, baseline: float, xvals: NDArray, yvals: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Sinc function for fitting and modeling.

    This function implements a scaled and shifted sinc function commonly used in
    signal processing and data fitting applications.

    Parameters
    ----------
    x : ndarray
        Array of x-values where the function is evaluated.
    height : float
        Height of the sinc function peak.
    loc : float
        Location (center) of the sinc function peak.
    FWHM : float
        Full Width at Half Maximum of the sinc function.
    baseline : float
        Baseline offset added to the sinc function.

    Returns
    -------
    ndarray
        Array of sinc function values evaluated at x.

    Notes
    -----
    The sinc function is defined as sin(πx)/(πx) with the convention that sinc(0) = 1.
    This implementation uses a scaled version with the specified FWHM parameter.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = sincfunc(x, height=1.0, loc=0.0, FWHM=2.0, baseline=0.0)
    >>> print(y.shape)
    (100,)
    """
    popt, pcov = curve_fit(sincfunc, xvals, yvals, p0=[height, loc, width, baseline])
    return popt, pcov


def gaussfit(
    height: float, loc: float, width: float, xvals: NDArray, yvals: NDArray
) -> Tuple[float, float, float]:
    """
    Performs a non-linear least squares fit of a Gaussian function to data.

    This routine uses `scipy.optimize.leastsq` to find the optimal parameters
    (height, location, and width) that best describe a Gaussian curve fitted
    to the provided `yvals` data against `xvals`. It requires an external
    `gaussresiduals` function to compute the residuals.

    Parameters
    ----------
    height : float
        Initial guess for the amplitude or peak height of the Gaussian.
    loc : float
        Initial guess for the mean (center) of the Gaussian.
    width : float
        Initial guess for the standard deviation (width) of the Gaussian.
    xvals : NDArray
        The independent variable data points.
    yvals : NDArray
        The dependent variable data points to which the Gaussian will be fitted.

    Returns
    -------
    tuple of float
        A tuple containing the fitted parameters:
        - height: The fitted height (amplitude) of the Gaussian.
        - loc: The fitted location (mean) of the Gaussian.
        - width: The fitted width (standard deviation) of the Gaussian.

    Notes
    -----
    - This function relies on an external function `gaussresiduals(params, y, x)`
      which should calculate the difference between the observed `y` values and
      the Gaussian function evaluated at `x` with the given `params` (height, loc, width).
    - `scipy.optimize.leastsq` is used for the optimization, which requires
      `scipy` and `numpy` to be imported (e.g., `import scipy.optimize as sp`
      and `import numpy as np`).
    - The optimization may fail if initial guesses are too far from the true values
      or if the data does not well-support a Gaussian fit.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = 2 * np.exp(-0.5 * ((x - 1) / 0.5)**2) + np.random.normal(0, 0.1, 100)
    >>> height, loc, width = gaussfit(1.0, 0.0, 1.0, x, y)
    >>> print(f"Fitted height: {height:.2f}, location: {loc:.2f}, width: {width:.2f}")
    """
    plsq, dummy = sp.optimize.leastsq(
        gaussresiduals, np.array([height, loc, width]), args=(yvals, xvals), maxfev=5000
    )
    return plsq[0], plsq[1], plsq[2]


def gram_schmidt(theregressors: NDArray, debug: bool = False) -> NDArray:
    """
    Performs Gram-Schmidt orthogonalization on a set of vectors.

    This routine takes a set of input vectors (rows of a 2D array) and
    transforms them into an orthonormal basis using the Gram-Schmidt process.
    It ensures that the resulting vectors are mutually orthogonal and
    have a unit norm. Linearly dependent vectors are effectively skipped
    if their orthogonal component is negligible.

    Parameters
    ----------
    theregressors : NDArray
        A 2D NumPy array where each row represents a vector to be orthogonalized.
    debug : bool, optional
        If True, prints debug information about input and output dimensions.
        Default is False.

    Returns
    -------
    NDArray
        A 2D NumPy array representing the orthonormal basis. Each row is an
        orthonormal vector. The number of rows may be less than the input if
        some vectors were linearly dependent.

    Notes
    -----
    - The function normalizes each orthogonalized vector to unit length.
    - A small tolerance (1e-10) is used to check if a vector's orthogonal
      component is effectively zero, indicating linear dependence.
    - Requires the `numpy` library for array operations and linear algebra.

    Examples
    --------
    >>> import numpy as np
    >>> vectors = np.array([[2, 1], [3, 4]])
    >>> basis = gram_schmidt(vectors)
    >>> print(basis)
    [[0.89442719 0.4472136 ]
     [-0.4472136  0.89442719]]
    """

    if debug:
        print("gram_schmidt, input dimensions:", theregressors.shape)
    basis: list[float] = []
    for i in range(theregressors.shape[0]):
        w = theregressors[i, :] - np.sum(
            np.array(np.dot(theregressors[i, :], b) * b) for b in basis
        )
        if (np.fabs(w) > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    outputbasis = np.array(basis)
    if debug:
        print("gram_schmidt, output dimensions:", outputbasis.shape)
    return outputbasis


def mlproject(thefit: NDArray, theevs: list, intercept: bool) -> NDArray:
    """
    Calculates a linear combination (weighted sum) of explanatory variables.

    This routine computes a predicted output by multiplying a set of
    explanatory variables by corresponding coefficients and summing the results.
    It can optionally include an intercept term. This is a common operation
    in linear regression and other statistical models.

    Parameters
    ----------
    thefit : NDArray
        A 1D array or list of coefficients (weights) to be applied to the
        explanatory variables. If `intercept` is True, the first element of
        `thefit` is treated as the intercept.
    theevs : list of NDArray
        A list where each element is a 1D NumPy array representing an
        explanatory variable (feature time series). The length of `theevs`
        should match the number of non-intercept coefficients in `thefit`.
    intercept : bool
        If True, the first element of `thefit` is used as an intercept term,
        and the remaining elements of `thefit` are applied to `theevs`. If False,
        no intercept is added, and all elements of `thefit` are applied to
        `theevs` starting from the first element.

    Returns
    -------
    NDArray
        A 1D NumPy array representing the calculated linear combination.
        Its length will be the same as the explanatory variables.

    Notes
    -----
    The calculation performed is conceptually equivalent to:
    `output = intercept_term + (coefficient_1 * ev_1) + (coefficient_2 * ev_2) + ...`
    where `intercept_term` is `thefit[0]` if `intercept` is True, otherwise 0.

    Examples
    --------
    >>> import numpy as np
    >>> thefit = np.array([1.0, 2.0, 3.0])
    >>> theevs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> result = mlproject(thefit, theevs, intercept=True)
    >>> print(result)
    [ 9. 14. 19.]
    """

    thedest = np.zeros_like(theevs[0])
    if intercept:
        thedest[:] = thefit[0]
        startpt = 1
    else:
        startpt = 0
    for i in range(len(thefit) - 1):
        thedest += thefit[i + startpt] * theevs[i]
    return thedest


def olsregress(
    X: ArrayLike, y: ArrayLike, intercept: bool = True, debug: bool = False
) -> Tuple[NDArray, float]:
    """
    Perform ordinary least squares regression.

    Parameters
    ----------
    X : array-like
        Independent variables (features) matrix of shape (n_samples, n_features).
    y : array-like
        Dependent variable (target) vector of shape (n_samples,).
    intercept : bool, optional
        Whether to add a constant term (intercept) to the model. Default is True.
    debug : bool, optional
        Whether to enable debug mode. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - params : ndarray
          Estimated regression coefficients (including intercept if specified)
        - rsquared : float
          Square root of the coefficient of determination (R-squared)

    Notes
    -----
    This function uses statsmodels OLS regression to fit a linear model.
    If intercept is True, a constant term is added to the design matrix.
    The function returns the regression parameters and the square root of R-squared.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 3])
    >>> params, r_squared = olsregress(X, y)
    >>> print(params)
    [0.1 0.4 0.2]
    """
    if intercept:
        X = sm.add_constant(X, prepend=True)
    model = sm.OLS(y, exog=X)
    thefit = model.fit()
    return thefit.params, np.sqrt(thefit.rsquared)


# @conditionaljit()
def mlregress(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    intercept: bool = True,
    debug: bool = False,
) -> Tuple[NDArray[np.floating[Any]], float]:
    """
    Perform multiple linear regression and return coefficients and R-squared value.

    This function fits a multiple linear regression model to the input data and
    returns the regression coefficients (including intercept if specified) along
    with the coefficient of determination (R-squared).

    Parameters
    ----------
    X : NDArray[np.floating[Any]]
        Input feature matrix of shape (n_samples, n_features) or (n_samples,)
        If 1D array is provided, it will be treated as a single feature.
    y : NDArray[np.floating[Any]]
        Target values of shape (n_samples,) or (n_samples, 1)
        If 1D array is provided, it will be treated as a single target.
    intercept : bool, optional
        Whether to calculate and include intercept term in the model.
        Default is True.
    debug : bool, optional
        If True, print debug information about the input shapes and processing steps.
        Default is False.

    Returns
    -------
    Tuple[NDArray[np.floating[Any]], float]
        A tuple containing:
        - coefficients : NDArray[np.floating[Any]] of shape (n_features + 1, 1) where the first
          element is the intercept (if intercept=True) and subsequent elements
          are the regression coefficients for each feature
        - R2 : float, the coefficient of determination (R-squared) of the fitted model

    Notes
    -----
    The function automatically handles shape adjustments for input arrays,
    ensuring that the number of samples in X matches the number of target values in y.
    If the input X is 1D, it will be converted to 2D. If the shapes don't match initially,
    the function will attempt to transpose X to match the number of samples in y.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([3, 7, 11])
    >>> coeffs, r2 = mlregress(X, y)
    >>> print(f"Coefficients: {coeffs.flatten()}")
    >>> print(f"R-squared: {r2}")
    """
    if debug:
        print(f"mlregress initial: {X.shape=}, {y.shape=}")
    y = np.atleast_1d(y)
    n = y.shape[0]

    X = np.atleast_2d(X)
    nx, p = X.shape

    if debug:
        print(f"mlregress: {n=}, {p=}, {nx=}")

    if nx != n:
        X = X.transpose()
        nx, p = X.shape
        if nx != n:
            raise AttributeError(
                "X and y must have have the same number of samples (%d and %d)" % (nx, n)
            )
    if debug:
        print(f"mlregress final: {X.shape=}, {y.shape=}")

    reg = LinearRegression(fit_intercept=intercept)
    reg.fit(X, y)
    coffs = reg.coef_
    theintercept = reg.intercept_
    R2 = reg.score(X, y)
    coffs = np.insert(coffs, 0, theintercept, axis=0)
    return np.asmatrix(coffs), R2


def calcexpandedregressors(
    confounddict: dict,
    labels: Optional[list] = None,
    start: int = 0,
    end: int = -1,
    deriv: bool = True,
    order: int = 1,
    debug: bool = False,
) -> Tuple[NDArray, list]:
    """
    Calculate expanded regressors from a dictionary of confound vectors.

    This routine generates a comprehensive set of motion-related regressors by
    including higher-order polynomial terms and derivatives of the original
    confound timecourses. It is commonly used in neuroimaging analysis to
    account for subject movement.

    Parameters
    ----------
    confounddict : dict
        A dictionary where keys are labels (e.g., 'rot_x', 'trans_y') and values
        are the corresponding 1D time series (NumPy arrays or lists).
    labels : list, optional
        A list of specific confound labels from `confounddict` to process. If None,
        all labels in `confounddict` will be used. Default is None.
    start : int, optional
        The starting index (inclusive) for slicing the timecourses. Default is 0.
    end : int, optional
        The ending index (exclusive) for slicing the timecourses. If -1, slicing
        continues to the end of the timecourse. Default is -1.
    deriv : bool, optional
        If True, the first derivative of each selected timecourse (and its
        polynomial expansions) is calculated and included as a regressor.
        Default is True.
    order : int, optional
        The polynomial order for expansion. If `order > 1`, terms like `label^2`,
        `label^3`, up to `label^order` will be included. Default is 1 (no
        polynomial expansion).
    debug : bool, optional
        If True, prints debug information during processing. Default is False.

    Returns
    -------
    tuple of (NDArray, list)
        A tuple containing:
        - outputregressors : NDArray
          A 2D NumPy array where each row represents a generated regressor
          (original, polynomial, or derivative) and columns represent time points.
        - outlabels : list of str
          A list of strings providing the labels for each row in `outputregressors`,
          indicating what each regressor represents (e.g., 'rot_x', 'rot_x^2',
          'rot_x_deriv').

    Notes
    -----
    - The derivatives are calculated using `numpy.gradient`.
    - The function handles slicing of the timecourses based on `start` and `end`
      parameters.
    - The output regressors are concatenated horizontally to form the final
      `outputregressors` array.

    Examples
    --------
    >>> confounddict = {
    ...     'rot_x': [0.1, 0.2, 0.3],
    ...     'trans_y': [0.05, 0.1, 0.15]
    ... }
    >>> regressors, labels = calcexpandedregressors(confounddict, order=2, deriv=True)
    >>> print(regressors.shape)
    (4, 3)
    >>> print(labels)
    ['rot_x', 'trans_y', 'rot_x^2', 'trans_y^2', 'rot_x_deriv', 'trans_y_deriv']
    """
    if labels is None:
        localconfounddict = confounddict.copy()
        labels = list(localconfounddict.keys())
    else:
        localconfounddict = {}
        for label in labels:
            localconfounddict[label] = confounddict[label]
    if order > 1:
        for theorder in range(1, order):
            for thelabel in labels:
                localconfounddict[f"{thelabel}^{theorder+1}"] = (localconfounddict[thelabel]) ** (
                    theorder + 1
                )
        labels = list(localconfounddict.keys())
        if debug:
            print(f"{labels=}")

    numkeys = len(labels)
    numpoints = len(localconfounddict[labels[0]])
    if end == -1:
        end = numpoints - 1
    if (0 <= start <= numpoints - 1) and (start < end + 1):
        numoutputpoints = end - start + 1

    numoutputregressors = numkeys
    if deriv:
        numoutputregressors += numkeys
    if numoutputregressors > 0:
        outputregressors = np.zeros((numoutputregressors, numoutputpoints), dtype=float)
    else:
        print("no output types selected - exiting")
        sys.exit()
    activecolumn = 0
    outlabels = []
    for thelabel in labels:
        outputregressors[activecolumn, :] = localconfounddict[thelabel][start : end + 1]
        outlabels.append(thelabel)
        activecolumn += 1
    if deriv:
        for thelabel in labels:
            outputregressors[activecolumn, :] = np.gradient(
                localconfounddict[thelabel][start : end + 1]
            )
            outlabels.append(thelabel + "_deriv")
            activecolumn += 1
    return outputregressors, outlabels


@conditionaljit()
def derivativelinfitfilt(
    thedata: NDArray, theevs: NDArray, nderivs: int = 1, debug: bool = False
) -> Tuple[NDArray, NDArray, NDArray, float, NDArray]:
    """
    Perform multicomponent expansion on explanatory variables and fit the data using linear regression.

    First, each explanatory variable is expanded into multiple components by taking
    successive derivatives (or powers, in the case of scalar inputs). Then, a linear
    fit is performed on the input data using the expanded set of explanatory variables.

    Parameters
    ----------
    thedata : NDArray
        Input data of length N to be filtered.
    theevs : NDArray
        NxP array of explanatory variables to be fit. If 1D, it is treated as a single
        explanatory variable.
    nderivs : int, optional
        Number of derivative components to compute for each explanatory variable.
        Default is 1. For each input variable, this creates a sequence of
        derivatives: original, first derivative, second derivative, etc.
    debug : bool, optional
        Flag to toggle debugging output. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - filtered : ndarray
            The filtered version of `thedata` after fitting.
        - thenewevs : ndarray
            The expanded set of explanatory variables (original + derivatives).
        - datatoremove : ndarray
            The part of the data that was removed during fitting.
        - R : float
            The coefficient of determination (R²) of the fit.
        - coffs : ndarray
            The coefficients of the linear fit.

    Notes
    -----
    This function is useful for filtering data when the underlying signal is expected
    to have smooth variations, and derivative information can improve the fit.
    The expansion of each variable into its derivatives allows for better modeling
    of local trends in the data.

    Examples
    --------
    >>> import numpy as np
    >>> from typing import Tuple
    >>> thedata = np.array([1, 2, 3, 4, 5])
    >>> theevs = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    >>> filtered, expanded_ev, removed, R, coeffs = derivativelinfitfilt(thedata, theevs, nderivs=2)
    """
    if debug:
        print(f"{thedata.shape=}")
        print(f"{theevs.shape=}")
    if nderivs == 0:
        thenewevs = theevs
    else:
        if theevs.ndim > 1:
            thenewevs = np.zeros((theevs.shape[0], theevs.shape[1] * (nderivs + 1)), dtype=float)
            for ev in range(0, theevs.shape[1] - 1):
                thenewevs[:, nderivs * ev] = theevs[:, ev] * 1.0
                for i in range(1, nderivs + 1):
                    thenewevs[:, nderivs * (ev - 1) + i] = np.gradient(
                        thenewevs[:, nderivs * (ev - 1) + i - 1]
                    )
        else:
            thenewevs = np.zeros((theevs.shape[0], nderivs + 1), dtype=float)
            thenewevs[:, 0] = theevs * 1.0
            for i in range(1, nderivs + 1):
                thenewevs[:, i] = np.gradient(thenewevs[:, i - 1])
    if debug:
        print(f"{nderivs=}")
        print(f"{thenewevs.shape=}")
    filtered, datatoremove, R, coffs, dummy = linfitfilt(thedata, thenewevs, debug=debug)
    if debug:
        print(f"{R=}")

    return filtered, thenewevs, datatoremove, R, coffs


@conditionaljit()
def expandedlinfitfilt(
    thedata: NDArray, theevs: NDArray, ncomps: int = 1, debug: bool = False
) -> Tuple[NDArray, NDArray, NDArray, float, NDArray]:
    """
    Perform multicomponent expansion on explanatory variables and fit a linear model.

    First, perform multicomponent expansion on the explanatory variables (`theevs`),
    where each variable is replaced by itself, its square, its cube, etc., up to `ncomps`
    components. Then, perform a multiple regression fit of `thedata` using the expanded
    explanatory variables and return the filtered data, the fitted model components,
    the residual sum of squares, and the coefficients.

    Parameters
    ----------
    thedata : NDArray
        Input data of length N to be filtered.
    theevs : array_like
        NxP array of explanatory variables to be fit.
    ncomps : int, optional
        Number of components to use for each ev. Each successive component is a
        higher power of the initial ev (initial, square, cube, etc.). Default is 1.
    debug : bool, optional
        Flag to toggle debugging output. Default is False.

    Returns
    -------
    filtered : ndarray
        The filtered version of `thedata` after fitting and removing the linear model.
    thenewevs : ndarray
        The expanded explanatory variables used in the fit.
    datatoremove : ndarray
        The portion of `thedata` that was removed during the fitting process.
    R : float
        Residual sum of squares from the linear fit.
    coffs : ndarray
        The coefficients of the linear fit.

    Notes
    -----
    If `ncomps` is 1, no expansion is performed and `theevs` is used directly.
    For each column in `theevs`, the expanded columns are created by taking powers
    of the original column (1st, 2nd, ..., ncomps-th power).

    Examples
    --------
    >>> import numpy as np
    >>> from typing import Tuple
    >>> thedata = np.array([1, 2, 3, 4, 5])
    >>> theevs = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    >>> filtered, expanded_ev, removed, R, coeffs = expandedlinfitfilt(thedata, theevs, ncomps=2)
    >>> print(filtered)
    [0. 0. 0. 0. 0.]
    """
    if debug:
        print(f"{thedata.shape=}")
        print(f"{theevs.shape=}")
    if ncomps == 1:
        thenewevs = theevs
    else:
        if theevs.ndim > 1:
            thenewevs = np.zeros((theevs.shape[0], theevs.shape[1] * ncomps), dtype=float)
            for ev in range(1, theevs.shape[1]):
                thenewevs[:, ncomps * (ev - 1)] = theevs[:, ev - 1] * 1.0
                for i in range(1, ncomps):
                    thenewevs[:, ncomps * (ev - 1) + i] = (
                        thenewevs[:, ncomps * (ev - 1) + i - 1] * theevs[:, ev - 1]
                    )
        else:
            thenewevs = np.zeros((theevs.shape[0], ncomps), dtype=float)
            thenewevs[:, 0] = theevs * 1.0
            for i in range(1, ncomps):
                thenewevs[:, i] = thenewevs[:, i - 1] * theevs
    if debug:
        print(f"{ncomps=}")
        print(f"{thenewevs.shape=}")
    filtered, datatoremove, R, coffs, dummy = linfitfilt(thedata, thenewevs, debug=debug)
    if debug:
        print(f"{R=}")

    return filtered, thenewevs, datatoremove, R, coffs


@conditionaljit()
def linfitfilt(
    thedata: NDArray, theevs: NDArray, debug: bool = False
) -> Tuple[NDArray, NDArray, float, NDArray, float]:
    """
    Performs a multiple regression fit of thedata using the vectors in theevs
    and returns the result.

    This function fits a linear model to the input data using the explanatory
    variables provided in `theevs`, then removes the fitted component from the
    original data to produce a filtered version.

    Parameters
    ----------
    thedata : NDArray
        Input data of length N to be filtered.
    theevs : NDArray
        NxP array of explanatory variables to be fit. If 1D, treated as a single
        explanatory variable.
    returnintercept : bool, optional
        If True, also return the intercept term from the regression. Default is False.
    debug : bool, optional
        If True, print debugging information during execution. Default is False.

    Returns
    -------
    filtered : ndarray
        The filtered data, i.e., the original data with the fitted component removed.
    datatoremove : ndarray
        The component of thedata that was removed during filtering.
    R2 : float
        The coefficient of determination (R-squared) of the regression.
    retcoffs : ndarray
        The regression coefficients (excluding intercept) for each explanatory variable.
    theintercept : float, optional
        The intercept term from the regression. Only returned if `returnintercept=True`.

    Notes
    -----
    This function uses `mlregress` internally to perform the linear regression.
    The intercept is always included in the model, but only returned if explicitly
    requested via `returnintercept`.

    Examples
    --------
    >>> import numpy as np
    >>> thedata = np.array([1, 2, 3, 4, 5])
    >>> theevs = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
    >>> filtered, datatoremove, R2, retcoffs, dummy = linfitfilt(thedata, theevs)
    >>> print(filtered)
    [0. 0. 0. 0. 0.]
    """

    if debug:
        print(f"{thedata.shape=}")
        print(f"{theevs.shape=}")
    thefit, R2 = mlregress(theevs, thedata, debug=debug)
    retcoffs = np.zeros((thefit.shape[1] - 1), dtype=float)
    if debug:
        print(f"{thefit.shape=}")
        print(f"{thefit=}")
        print(f"{R2=}")
        print(f"{retcoffs.shape=}")
    datatoremove = np.zeros_like(thedata)

    if theevs.ndim > 1:
        for ev in range(1, thefit.shape[1]):
            if debug:
                print(f"{ev=}")
            theintercept = thefit[0, 0]
            retcoffs[ev - 1] = thefit[0, ev]
            datatoremove += thefit[0, ev] * theevs[:, ev - 1]
            if debug:
                print(f"{ev=}")
                print(f"\t{thefit[0, ev]=}")
                print(f"\tdatatoremove min max = {np.min(datatoremove)}, {np.max(datatoremove)}")
    else:
        datatoremove += thefit[0, 1] * theevs[:]
        retcoffs[0] = thefit[0, 1]
    filtered = thedata - datatoremove
    if debug:
        print(f"{retcoffs=}")
    return filtered, datatoremove, R2, retcoffs, theintercept


@conditionaljit()
def confoundregress(
    data: NDArray,
    regressors: NDArray,
    debug: bool = False,
    showprogressbar: bool = True,
    rt_floattype: np.dtype = np.float64,
) -> Tuple[NDArray, NDArray]:
    """
    Filters multiple regressors out of an array of data using linear regression.

    This function removes the effect of nuisance regressors from each voxel's timecourse
    by fitting a linear model and subtracting the predicted signal.

    Parameters
    ----------
    data : 2d numpy array
        A data array where the first index is the spatial dimension (e.g., voxels),
        and the second index is the time (filtering) dimension.
    regressors : 2d numpy array
        The set of regressors to filter out of each timecourse. The first dimension
        is the regressor number, and the second is the time (filtering) dimension.
    debug : bool, optional
        Print additional diagnostic information if True. Default is False.
    showprogressbar : bool, optional
        Show progress bar during processing. Default is True.
    rt_floattype : np.dtype, optional
        The data type used for floating-point calculations. Default is np.float64.

    Returns
    -------
    filtereddata : 2d numpy array
        The data with regressors removed, same shape as input `data`.
    r2value : 1d numpy array
        The R-squared value for each voxel's regression fit, shape (data.shape[0],).

    Notes
    -----
    This function uses `mlregress` internally to perform the linear regression for each voxel.
    The regressors are applied in the order they appear in the input array.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 1000)
    >>> regressors = np.random.rand(3, 1000)
    >>> filtered_data, r2_values = confoundregress(data, regressors, debug=True)
    """
    if debug:
        print("data shape:", data.shape)
        print("regressors shape:", regressors.shape)
    datatoremove = np.zeros(data.shape[1], dtype=rt_floattype)
    filtereddata = np.zeros_like(data)
    r2value = np.zeros_like(data[:, 0])
    for i in tqdm(
        range(data.shape[0]),
        desc="Voxel",
        unit="voxels",
        disable=(not showprogressbar),
    ):
        datatoremove *= 0.0
        thefit, R2 = mlregress(regressors, data[i, :])
        if i == 0 and debug:
            print("fit shape:", thefit.shape)
        for j in range(regressors.shape[0]):
            datatoremove += (thefit[0, 1 + j] * regressors[j, :]).astype(rt_floattype)
        filtereddata[i, :] = data[i, :] - datatoremove
        r2value[i] = R2
    return filtereddata, r2value


# --------------------------- Peak detection functions ----------------------------------------------
# The following three functions are taken from the peakdetect distribution by Sixten Bergman
# They were distributed under the DWTFYWTPL, so I'm relicensing them under Apache 2.0
# From his header:
# You can redistribute it and/or modify it under the terms of the Do What The
# Fuck You Want To Public License, Version 2, as published by Sam Hocevar. See
# http://www.wtfpl.net/ for more details.
def getpeaks(
    xvals: NDArray,
    yvals: NDArray,
    xrange: Optional[Tuple[float, float]] = None,
    bipolar: bool = False,
    displayplots: bool = False,
) -> list:
    """
    Find peaks in y-values within a specified range and optionally display results.

    This function identifies local maxima (and optionally minima) in the input
    y-values and returns their coordinates along with an offset from the origin.
    It supports filtering by a range of x-values and can handle both unipolar and
    bipolar peak detection.

    Parameters
    ----------
    xvals : NDArray
        X-axis values corresponding to the y-values.
    yvals : NDArray
        Y-axis values where peaks are to be detected.
    xrange : tuple of float, optional
        A tuple (min, max) specifying the range of x-values to consider.
        If None, the full range is used.
    bipolar : bool, optional
        If True, detect both positive and negative peaks (minima and maxima).
        If False, only detect positive peaks.
    displayplots : bool, optional
        If True, display a plot showing the data and detected peaks.

    Returns
    -------
    list of lists
        A list of peaks, each represented as [x_value, y_value, offset_from_origin].
        The offset is calculated using `tide_util.valtoindex` relative to x=0.

    Notes
    -----
    - The function uses `scipy.signal.find_peaks` to detect peaks.
    - If `bipolar` is True, both positive and negative peaks are included.
    - The `displayplots` option requires `matplotlib.pyplot` to be imported as `plt`.

    Examples
    --------
    >>> x = np.linspace(-10, 10, 100)
    >>> y = np.sin(x)
    >>> peaks = getpeaks(x, y, xrange=(-5, 5), bipolar=True)
    >>> print(peaks)
    [[-1.5707963267948966, 1.0, -25], [1.5707963267948966, 1.0, 25]]
    """
    peaks, dummy = find_peaks(yvals, height=0)
    if bipolar:
        negpeaks, dummy = find_peaks(-yvals, height=0)
        peaks = np.concatenate((peaks, negpeaks))
    procpeaks = []
    if xrange is None:
        lagmin = xvals[0] + 0.0
        lagmax = xvals[-1] + 0.0
    else:
        lagmin = xrange[0] + 0.0
        lagmax = xrange[1] + 0.0
    originloc = tide_util.valtoindex(xvals, 0.0, discrete=False)
    for thepeak in peaks:
        if lagmin <= xvals[thepeak] <= lagmax:
            if bipolar:
                procpeaks.append(
                    [
                        xvals[thepeak],
                        yvals[thepeak],
                        tide_util.valtoindex(xvals, xvals[thepeak], discrete=False) - originloc,
                    ]
                )
            else:
                if yvals[thepeak] > 0.0:
                    procpeaks.append(
                        [
                            xvals[thepeak],
                            yvals[thepeak],
                            tide_util.valtoindex(xvals, xvals[thepeak], discrete=False)
                            - originloc,
                        ]
                    )
    if displayplots:
        plotx = []
        ploty = []
        offset = []
        for thepeak in procpeaks:
            plotx.append(thepeak[0])
            ploty.append(thepeak[1])
            offset.append(thepeak[2])
        plt.plot(xvals, yvals)
        plt.plot(plotx, ploty, "x")
        plt.plot(xvals, np.zeros_like(yvals), "--", color="gray")
        plt.show()
    return procpeaks


def parabfit(x_axis: NDArray, y_axis: NDArray, peakloc: int, points: int) -> Tuple[float, float]:
    """
    Fit a parabola to a localized region around a peak and return the peak coordinates.

    This function performs a quadratic curve fitting on a subset of data surrounding
    a specified peak location. It uses a parabolic model of the form a*(x-tau)^2 + c
    to estimate the precise peak position and amplitude.

    Parameters
    ----------
    x_axis : NDArray
        Array of x-axis values (typically time or frequency).
    y_axis : NDArray
        Array of y-axis values (typically signal amplitude).
    peakloc : int
        Index location of the peak in the data arrays.
    points : int
        Number of points to include in the local fit around the peak.

    Returns
    -------
    Tuple[float, float]
        A tuple containing (x_peak, y_peak) - the fitted peak coordinates.

    Notes
    -----
    The function uses a least-squares fitting approach with scipy.optimize.curve_fit.
    Initial parameter estimates are derived analytically based on the peak location
    and a distance calculation. The parabolic model assumes the peak has a symmetric
    quadratic shape.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2 * (x - 5)**2 + 1
    >>> peak_x, peak_y = parabfit(x, y, 50, 10)
    >>> print(f"Peak at x={peak_x:.2f}, y={peak_y:.2f}")
    """
    func = lambda x, a, tau, c: a * ((x - tau) ** 2) + c
    distance = abs(x_axis[peakloc[1][0]] - x_axis[peakloc[0][0]]) / 4
    index = peakloc
    x_data = x_axis[index - points // 2 : index + points // 2 + 1]
    y_data = y_axis[index - points // 2 : index + points // 2 + 1]

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


def _datacheck_peakdetect(x_axis: Optional[NDArray], y_axis: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Validate and convert input arrays for peak detection.

    Parameters
    ----------
    x_axis : NDArray, optional
        X-axis values. If None, range(len(y_axis)) is used.
    y_axis : NDArray
        Y-axis values to be processed.

    Returns
    -------
    tuple of ndarray
        Tuple containing (x_axis, y_axis) as numpy arrays.

    Raises
    ------
    ValueError
        If input vectors y_axis and x_axis have different lengths.

    Notes
    -----
    This function ensures that both input arrays are converted to numpy arrays
    and have matching shapes. If x_axis is None, it defaults to a range
    corresponding to the length of y_axis.

    Examples
    --------
    >>> import numpy as np
    >>> x, y = _datacheck_peakdetect([1, 2, 3], [4, 5, 6])
    >>> print(x)
    [1 2 3]
    >>> print(y)
    [4 5 6]

    >>> x, y = _datacheck_peakdetect(None, [4, 5, 6])
    >>> print(x)
    [0 1 2]
    """
    if x_axis is None:
        x_axis = np.arange(0, len(y_axis))

    if np.shape(y_axis) != np.shape(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(
    y_axis: NDArray[np.floating[Any]],
    x_axis: Optional[NDArray[np.floating[Any]]] = None,
    lookahead: int = 200,
    delta: float = 0.0,
) -> list:
    """
    Detect local maxima and minima in a signal.

    This function is based on a MATLAB script by Billauer, and identifies peaks
    by searching for values that are surrounded by lower (for maxima) or larger
    (for minima) values. It uses a lookahead window to confirm that a candidate
    is indeed a peak and not noise or jitter.

    Parameters
    ----------
    y_axis : NDArray[np.floating[Any]]
        A list or array containing the signal over which to find peaks.
    x_axis : NDArray[np.floating[Any]], optional
        An x-axis whose values correspond to the y_axis list. If omitted,
        an index of the y_axis is used. Default is None.
    lookahead : int, optional
        Distance to look ahead from a peak candidate to determine if it is
        the actual peak. Default is 200.
    delta : float, optional
        Minimum difference between a peak and the following points. If set,
        this helps avoid false peaks towards the end of the signal. Default is 0.0.

    Returns
    -------
    list of lists
        A list containing two sublists: ``[max_peaks, min_peaks]``.
        Each sublist contains tuples of the form ``(position, peak_value)``.
        For example, to unpack maxima into x and y coordinates:
        ``x, y = zip(*max_peaks)``.

    Notes
    -----
    - The function assumes that the input signal is sampled at regular intervals.
    - If ``delta`` is not provided, the function runs slower but may detect more
      peaks.
    - When ``delta`` is correctly specified (e.g., as 5 * RMS noise), it can
      significantly improve performance.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.5 * np.sin(3 * x)
    >>> max_peaks, min_peaks = peakdetect(y, x, lookahead=10, delta=0.1)
    >>> print("Max peaks:", max_peaks)
    >>> print("Min peaks:", min_peaks)
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
    if not (np.isscalar(delta) and (delta >= 0.0)):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.inf, -np.inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.inf
                mn = np.inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
                # else:  #slows shit down this does
                #    mx = ahead
                #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if (y > mn + delta) and (mn != -np.inf):
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.inf
                mx = -np.inf
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


def ocscreetest(eigenvals: NDArray, debug: bool = False, displayplots: bool = False) -> int:
    """
    Perform eigenvalue screening using the OCSCREE test to determine the number of retained components.

    This function implements a variant of the scree test for determining the number of significant
    eigenvalues in a dataset. It uses a linear regression approach to model the eigenvalue decay
    and identifies the point where the observed eigenvalues fall below the predicted values.

    Parameters
    ----------
    eigenvals : NDArray
        Array of eigenvalues, typically sorted in descending order.
    debug : bool, optional
        If True, print intermediate calculations for debugging purposes. Default is False.
    displayplots : bool, optional
        If True, display plots of the original eigenvalues, regression coefficients (a and b),
        and the predicted eigenvalue curve. Default is False.

    Returns
    -------
    int
        The index of the last retained component based on the OCSCREE criterion.

    Notes
    -----
    The function performs the following steps:
    1. Initialize arrays for regression coefficients 'a' and 'b'.
    2. Compute regression coefficients from the eigenvalues.
    3. Predict eigenvalues using the regression model.
    4. Identify the point where the actual eigenvalues drop below the predicted values.
    5. Optionally display diagnostic plots.

    Examples
    --------
    >>> import numpy as np
    >>> eigenvals = np.array([3.5, 2.1, 1.8, 1.2, 0.9, 0.5])
    >>> result = ocscreetest(eigenvals)
    >>> print(result)
    3
    """
    num = len(eigenvals)
    a = np.zeros_like(eigenvals)
    b = np.zeros_like(eigenvals)
    prediction = np.zeros_like(eigenvals)
    for i in range(num - 3, 1, -1):
        b[i] = (eigenvals[-1] - eigenvals[i + 1]) / (num - 1 - i - 1)
        a[i] = eigenvals[i + 1] - b[i + 1]
        if debug:
            print(f"{i=}, {a[i]=}, {b[i]=}")
    retained = eigenvals[np.where(eigenvals > 1.0)]
    retainednum = len(retained)
    for i in range(1, num - 1):
        prediction[i] = a[i + 1] + b[i + 1] * i
        if debug:
            print(f"{i=}, {eigenvals[i]=}, {prediction[i]=}")
        if eigenvals[i] < retained[i]:
            break
    if displayplots:
        print("making plots")
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax1.set_title("Original")
        plt.plot(eigenvals, color="k")
        ax2 = fig.add_subplot(412)
        ax2.set_title("a")
        plt.plot(a, color="g")
        ax3 = fig.add_subplot(413)
        ax3.set_title("b")
        plt.plot(b, color="r")
        ax4 = fig.add_subplot(414)
        ax4.set_title("prediction")
        plt.plot(prediction, color="b")
        plt.show()
    return i


def afscreetest(eigenvals: NDArray, displayplots: bool = False) -> int:
    """
    Detect the optimal number of components using the second derivative of eigenvalues.

    This function applies a second derivative analysis to the eigenvalues to identify
    the point where the rate of change of eigenvalues begins to decrease significantly,
    which typically indicates the optimal number of components to retain.

    Parameters
    ----------
    eigenvals : NDArray
        Array of eigenvalues, typically from a PCA or similar decomposition.
        Should be sorted in descending order.
    displayplots : bool, optional
        If True, display plots showing the original eigenvalues, first derivative,
        and second derivative (default is False).

    Returns
    -------
    int
        The index of the optimal number of components, adjusted by subtracting 1
        from the location of maximum second derivative.

    Notes
    -----
    The method works by:
    1. Computing the first derivative of eigenvalues
    2. Computing the second derivative of the first derivative
    3. Finding the maximum of the second derivative
    4. Returning the index of this maximum minus 1

    Examples
    --------
    >>> import numpy as np
    >>> eigenvals = np.array([5.0, 3.0, 1.5, 0.8, 0.2])
    >>> optimal_components = afscreetest(eigenvals)
    >>> print(optimal_components)
    1
    """
    num = len(eigenvals)
    firstderiv = np.gradient(eigenvals, edge_order=2)
    secondderiv = np.gradient(firstderiv, edge_order=2)
    maxaccloc = np.argmax(secondderiv)
    if displayplots:
        print("making plots")
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("Original")
        plt.plot(eigenvals, color="k")
        ax2 = fig.add_subplot(312)
        ax2.set_title("First derivative")
        plt.plot(firstderiv, color="r")
        ax3 = fig.add_subplot(313)
        ax3.set_title("Second derivative")
        plt.plot(secondderiv, color="g")
        plt.show()
    return int(maxaccloc - 1)


def phaseanalysis(
    firstharmonic: NDArray, displayplots: bool = False
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Perform phase analysis on a signal using analytic signal representation.

    This function computes the analytic signal of the input signal using the Hilbert transform,
    and extracts the instantaneous phase and amplitude envelope. Optionally displays plots
    of the analytic signal, phase, and amplitude.

    Parameters
    ----------
    firstharmonic : NDArray
        Input signal to analyze. Should be a 1D NDArray object.
    displayplots : bool, optional
        If True, displays plots of the analytic signal, phase, and amplitude.
        Default is False.

    Returns
    -------
    tuple of ndarray
        A tuple containing:
        - instantaneous_phase : ndarray
          The unwrapped instantaneous phase of the signal
        - amplitude_envelope : ndarray
          The amplitude envelope of the signal
        - analytic_signal : ndarray
          The analytic signal (complex-valued)

    Notes
    -----
    The function uses `scipy.signal.hilbert` to compute the analytic signal,
    which is defined as: :math:`x_a(t) = x(t) + j\\hat{x}(t)` where :math:`\\hat{x}(t)`
    is the Hilbert transform of :math:`x(t)`.

    The instantaneous phase is computed as the angle of the analytic signal and is
    unwrapped to remove discontinuities.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    >>> phase, amp, analytic = phaseanalysis(signal)
    >>> print(f"Phase shape: {phase.shape}")
    Phase shape: (100,)
    """
    print("entering phaseanalysis")
    analytic_signal = hilbert(firstharmonic)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    if displayplots:
        print("making plots")
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("Analytic signal")
        X = np.linspace(0.0, 1.0, num=len(firstharmonic))
        plt.plot(X, analytic_signal.real, "k", X, analytic_signal.imag, "r")
        ax2 = fig.add_subplot(312)
        ax2.set_title("Phase")
        plt.plot(X, instantaneous_phase, "g")
        ax3 = fig.add_subplot(313)
        ax3.set_title("Amplitude")
        plt.plot(X, amplitude_envelope, "b")
        plt.show()
        plt.savefig("phaseanalysistest.jpg")
    instantaneous_phase = np.unwrap(instantaneous_phase)
    return instantaneous_phase, amplitude_envelope, analytic_signal


FML_NOERROR = np.uint32(0x0000)

FML_INITAMPLOW = np.uint32(0x0001)
FML_INITAMPHIGH = np.uint32(0x0002)
FML_INITWIDTHLOW = np.uint32(0x0004)
FML_INITWIDTHHIGH = np.uint32(0x0008)
FML_INITLAGLOW = np.uint32(0x0010)
FML_INITLAGHIGH = np.uint32(0x0020)
FML_INITFAIL = (
    FML_INITAMPLOW
    | FML_INITAMPHIGH
    | FML_INITWIDTHLOW
    | FML_INITWIDTHHIGH
    | FML_INITLAGLOW
    | FML_INITLAGHIGH
)

FML_FITAMPLOW = np.uint32(0x0100)
FML_FITAMPHIGH = np.uint32(0x0200)
FML_FITWIDTHLOW = np.uint32(0x0400)
FML_FITWIDTHHIGH = np.uint32(0x0800)
FML_FITLAGLOW = np.uint32(0x1000)
FML_FITLAGHIGH = np.uint32(0x2000)
FML_FITFAIL = (
    FML_FITAMPLOW
    | FML_FITAMPHIGH
    | FML_FITWIDTHLOW
    | FML_FITWIDTHHIGH
    | FML_FITLAGLOW
    | FML_FITLAGHIGH
)


def simfuncpeakfit(
    incorrfunc: NDArray,
    corrtimeaxis: NDArray,
    useguess: bool = False,
    maxguess: float = 0.0,
    displayplots: bool = False,
    functype: str = "correlation",
    peakfittype: str = "gauss",
    searchfrac: float = 0.5,
    lagmod: float = 1000.0,
    enforcethresh: bool = True,
    allowhighfitamps: bool = False,
    lagmin: float = -30.0,
    lagmax: float = 30.0,
    absmaxsigma: float = 1000.0,
    absminsigma: float = 0.25,
    hardlimit: bool = True,
    bipolar: bool = False,
    lthreshval: float = 0.0,
    uthreshval: float = 1.0,
    zerooutbadfit: bool = True,
    debug: bool = False,
) -> Tuple[int, np.float64, np.float64, np.float64, np.uint16, np.uint32, int, int]:
    """
        Fit a peak in a correlation or mutual information function.

        This function performs peak fitting on a correlation or mutual information
        function to extract peak parameters such as location, amplitude, and width.
        It supports various fitting methods and includes error handling and
        validation for fit parameters.

        Parameters
        ----------
        incorrfunc : NDArray
            Input correlation or mutual information function values.
        corrtimeaxis : NDArray
            Time axis corresponding to the correlation function.
        useguess : bool, optional
            If True, use `maxguess` as an initial guess for the peak location.
            Default is False.
        maxguess : float, optional
            Initial guess for the peak location in seconds. Used only if `useguess` is True.
            Default is 0.0.
        displayplots : bool, optional
            If True, display plots of the peak and fit. Default is False.
        functype : str, optional
            Type of function to fit. Options are 'correlation', 'mutualinfo', or 'hybrid'.
            Default is 'correlation'.
        peakfittype : str, optional
            Type of peak fitting to perform. Options are 'gauss', 'fastgauss', 'quad',
            'fastquad', 'COM', or 'None'. Default is 'gauss'.
        searchfrac : float, optional
            Fraction of the peak maximum to define the search range for peak width.
            Default is 0.5.
        lagmod : float, optional
            Modulus for lag values, used to wrap around the lag values.
            Default is 1000.0.
        enforcethresh : bool, optional
            If True, enforce amplitude thresholds. Default is True.
        allowhighfitamps : bool, optional
            If True, allow fit amplitudes to exceed 1.0. Default is False.
        lagmin : float, optional
            Minimum allowed lag value in seconds. Default is -30.0.
        lagmax : float, optional
            Maximum allowed lag value in seconds. Default is 30.0.
        absmaxsigma : float, optional
            Maximum allowed sigma value in seconds. Default is 1000.0.
        absminsigma : float, optional
            Minimum allowed sigma value in seconds. Default is 0.25.
        hardlimit : bool, optional
            If True, enforce hard limits on lag values. Default is True.
        bipolar : bool, optional
            If True, allow negative correlation values. Default is False.
        lthreshval : float, optional
            Lower threshold for amplitude validation. Default is 0.0.
        uthreshval : float, optional
            Upper threshold for amplitude validation. Default is 1.0.
        zerooutbadfit : bool, optional
            If True, set fit results to zero if fit fails. Default is True.
        debug : bool, optional
            If True, print debug information. Default is False.

        Returns
        -------
        tuple of int, float, float, float, int, int, int, int
            A tuple containing:
            - maxindex: Index of the peak maximum.
            - maxlag: Fitted peak lag in seconds.
            - maxval: Fitted peak amplitude.
            - maxsigma: Fitted peak width (sigma) in seconds.
            - maskval: Mask indicating fit success (1 for success, 0 for failure).
            - failreason: Reason for fit failure (bitmask).
            - peakstart: Start index of the peak region used for fitting.
            - peakend: End index of the peak region used for fitting.

        Notes
        -----
        - The function automatically handles different types of correlation functions
          and mutual information functions with appropriate baseline corrections.
        - Various fitting methods are supported, each with its own strengths and
          trade-offs in terms of speed and accuracy.
        - Fit results are validated against physical constraints and thresholds.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import signal
        >>> # Create sample data
        >>> t = np.linspace(-50, 50, 1000)
        >>> corr = np.exp(-0.5 * (t / 2)**2) + 0.1 * np.random.randn(1000)
        >>> maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = \
        ...     simfuncpeakfit(corr, t, peakfittype='gauss')
        >>> print(f"Peak lag: {maxlag:.2f} s, Amplitude: {maxval:.2f}, Width: {maxsigma:.2f} s")
        """
    # check to make sure xcorr_x and xcorr_y match
    if corrtimeaxis is None:
        print("Correlation time axis is not defined - exiting")
        sys.exit()
    if len(corrtimeaxis) != len(incorrfunc):
        print(
            "Correlation time axis and values do not match in length (",
            len(corrtimeaxis),
            "!=",
            len(incorrfunc),
            "- exiting",
        )
        sys.exit()
    # set initial parameters
    # absmaxsigma is in seconds
    # maxsigma is in Hz
    # maxlag is in seconds
    warnings.filterwarnings("ignore", "Number*")
    failreason = FML_NOERROR
    maskval = np.uint16(1)  # start out assuming the fit will succeed
    binwidth = corrtimeaxis[1] - corrtimeaxis[0]

    # set the search range
    lowerlim = 0
    upperlim = len(corrtimeaxis) - 1
    if debug:
        print(
            "initial search indices are",
            lowerlim,
            "to",
            upperlim,
            "(",
            corrtimeaxis[lowerlim],
            corrtimeaxis[upperlim],
            ")",
        )

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value and its location
    flipfac = 1.0
    corrfunc = incorrfunc + 0.0
    if useguess:
        maxindex = tide_util.valtoindex(corrtimeaxis, maxguess)
        if (corrfunc[maxindex] < 0.0) and bipolar:
            flipfac = -1.0
    else:
        maxindex, flipfac = _maxindex_noedge(corrfunc, corrtimeaxis, bipolar=bipolar)
    corrfunc *= flipfac
    maxlag_init = (1.0 * corrtimeaxis[maxindex]).astype("float64")
    maxval_init = corrfunc[maxindex].astype("float64")
    if debug:
        print(
            "maxindex, maxlag_init, maxval_init:",
            maxindex,
            maxlag_init,
            maxval_init,
        )

    # set the baseline and baselinedev levels
    if (functype == "correlation") or (functype == "hybrid"):
        baseline = 0.0
        baselinedev = 0.0
    else:
        # for mutual information, there is a nonzero baseline, so we want the difference from that.
        baseline = float(np.median(corrfunc))
        baselinedev = mad(corrfunc)
    if debug:
        print("baseline, baselinedev:", baseline, baselinedev)

    # then calculate the width of the peak
    if peakfittype == "fastquad" or peakfittype == "COM":
        peakstart = np.max([1, maxindex - 2])
        peakend = np.min([len(corrtimeaxis) - 2, maxindex + 2])
    else:
        # come here for peakfittype of None, quad, gauss, fastgauss
        thegrad = np.gradient(corrfunc).astype(
            "float64"
        )  # the gradient of the correlation function
        if (functype == "correlation") or (functype == "hybrid"):
            if peakfittype == "quad":
                peakpoints = np.where(
                    corrfunc > maxval_init - 0.05, 1, 0
                )  # mask for places where correlation exceeds searchfrac*maxval_init
            else:
                peakpoints = np.where(
                    corrfunc > (baseline + searchfrac * (maxval_init - baseline)), 1, 0
                )  # mask for places where correlation exceeds searchfrac*maxval_init
        else:
            # for mutual information, there is a flattish, nonzero baseline, so we want the difference from that.
            peakpoints = np.where(
                corrfunc > (baseline + searchfrac * (maxval_init - baseline)),
                1,
                0,
            )

        peakpoints[0] = 0
        peakpoints[-1] = 0
        peakstart = int(np.max([1, maxindex - 1]))
        peakend = int(np.min([len(corrtimeaxis) - 2, maxindex + 1]))
        if debug:
            print("initial peakstart, peakend:", peakstart, peakend)
        if functype == "mutualinfo":
            while peakpoints[peakend + 1] == 1:
                peakend += 1
            while peakpoints[peakstart - 1] == 1:
                peakstart -= 1
        else:
            while thegrad[peakend + 1] <= 0.0 and peakpoints[peakend + 1] == 1:
                peakend += 1
            while thegrad[peakstart - 1] >= 0.0 and peakpoints[peakstart - 1] == 1:
                peakstart -= 1
        if debug:
            print("final peakstart, peakend:", peakstart, peakend)

        # deal with flat peak top
        while peakend < (len(corrtimeaxis) - 3) and corrfunc[peakend] == corrfunc[peakend - 1]:
            peakend += 1
        while peakstart > 2 and corrfunc[peakstart] == corrfunc[peakstart + 1]:
            peakstart -= 1
        if debug:
            print("peakstart, peakend after flattop correction:", peakstart, peakend)
            print("\n")
            for i in range(peakstart, peakend + 1):
                print(corrtimeaxis[i], corrfunc[i])
            print("\n")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("Peak sent to fitting routine")
            plt.plot(
                corrtimeaxis[peakstart : peakend + 1],
                corrfunc[peakstart : peakend + 1],
                "r",
            )
            plt.show()

        # This is calculated from first principles, but it's always big by a factor or ~1.4.
        #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
        maxsigma_init = np.float64(
            ((peakend - peakstart + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac))))
            / np.sqrt(2.0)
        )
        if debug:
            print("maxsigma_init:", maxsigma_init)

        # now check the values for errors
        if hardlimit:
            rangeextension = 0.0
        else:
            rangeextension = (lagmax - lagmin) * 0.75
        if not (
            (lagmin - rangeextension - binwidth)
            <= maxlag_init
            <= (lagmax + rangeextension + binwidth)
        ):
            if maxlag_init <= (lagmin - rangeextension - binwidth):
                failreason |= FML_INITLAGLOW
                maxlag_init = lagmin - rangeextension - binwidth
            else:
                failreason |= FML_INITLAGHIGH
                maxlag_init = lagmax + rangeextension + binwidth
            if debug:
                print("bad initial")
        if maxsigma_init > absmaxsigma:
            failreason |= FML_INITWIDTHHIGH
            maxsigma_init = np.float64(absmaxsigma)
            if debug:
                print("bad initial width - too high")
        if peakend - peakstart < 2:
            failreason |= FML_INITWIDTHLOW
            maxsigma_init = np.float64(
                ((2 + 1) * binwidth / (2.0 * np.sqrt(-np.log(searchfrac)))) / np.sqrt(2.0)
            )
            if debug:
                print("bad initial width - too low")
        if (functype == "correlation") or (functype == "hybrid"):
            if not (lthreshval <= maxval_init <= uthreshval) and enforcethresh:
                failreason |= FML_INITAMPLOW
                if debug:
                    print(
                        "bad initial amp:",
                        maxval_init,
                        "is less than",
                        lthreshval,
                    )
            if maxval_init < 0.0:
                failreason |= FML_INITAMPLOW
                maxval_init = 0.0
                if debug:
                    print("bad initial amp:", maxval_init, "is less than 0.0")
            if maxval_init > 1.0:
                failreason |= FML_INITAMPHIGH
                maxval_init = 1.0
                if debug:
                    print("bad initial amp:", maxval_init, "is greater than 1.0")
        else:
            # somewhat different rules for mutual information peaks
            if ((maxval_init - baseline) < lthreshval * baselinedev) or (maxval_init < baseline):
                failreason |= FML_INITAMPLOW
                maxval_init = 0.0
                if debug:
                    print("bad initial amp:", maxval_init, "is less than 0.0")
        if (failreason != FML_NOERROR) and zerooutbadfit:
            maxval = np.float64(0.0)
            maxlag = np.float64(0.0)
            maxsigma = np.float64(0.0)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(maxlag_init)
            maxsigma = np.float64(maxsigma_init)

    # refine if necessary
    if peakfittype != "None":
        if peakfittype == "COM":
            X = corrtimeaxis[peakstart : peakend + 1] - baseline
            data = corrfunc[peakstart : peakend + 1]
            maxval = maxval_init
            maxlag = np.sum(X * data) / np.sum(data)
            maxsigma = np.float64(10.0)
        elif peakfittype == "gauss":
            X = corrtimeaxis[peakstart : peakend + 1] - baseline
            data = corrfunc[peakstart : peakend + 1]
            # do a least squares fit over the top of the peak
            # p0 = np.array([maxval_init, np.fmod(maxlag_init, lagmod), maxsigma_init], dtype='float64')
            p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype="float64")
            if debug:
                print("fit input array:", p0)
            try:
                plsq, ier = sp.optimize.leastsq(gaussresiduals, p0, args=(data, X), maxfev=5000)
                if ier not in [1, 2, 3, 4]:  # Check for successful convergence
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                else:
                    maxval = plsq[0] + baseline
                    maxlag = np.fmod((1.0 * plsq[1]), lagmod)
                    maxsigma = plsq[2]
            except:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            if debug:
                print("fit output array:", [maxval, maxlag, maxsigma])
        elif peakfittype == "fastgauss":
            X = corrtimeaxis[peakstart : peakend + 1] - baseline
            data = corrfunc[peakstart : peakend + 1]
            # do a non-iterative fit over the top of the peak
            # 6/12/2015  This is just broken.  Gives quantized maxima
            maxlag = np.float64(1.0 * np.sum(X * data) / np.sum(data))
            maxsigma = np.float64(np.sqrt(np.abs(np.sum((X - maxlag) ** 2 * data) / np.sum(data))))
            maxval = np.float64(data.max()) + baseline
        elif peakfittype == "fastquad":
            maxlag, maxval, maxsigma, ismax, badfit = refinepeak_quad(
                corrtimeaxis, corrfunc, maxindex
            )
        elif peakfittype == "quad":
            X = corrtimeaxis[peakstart : peakend + 1]
            data = corrfunc[peakstart : peakend + 1]
            try:
                thecoffs = Polynomial.fit(X, data, 2).convert().coef[::-1]
                a = thecoffs[0]
                b = thecoffs[1]
                c = thecoffs[2]
                maxlag = -b / (2.0 * a)
                maxval = a * maxlag * maxlag + b * maxlag + c
                maxsigma = 1.0 / np.fabs(a)
                if debug:
                    print("poly coffs:", a, b, c)
                    print("maxlag, maxval, maxsigma:", maxlag, maxval, maxsigma)
            except np.exceptions.RankWarning:
                maxlag = np.float64(0.0)
                maxval = np.float64(0.0)
                maxsigma = np.float64(0.0)
            if debug:
                print("\n")
                for i in range(len(X)):
                    print(X[i], data[i])
                print("\n")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Peak and fit")
                plt.plot(X, data, "r")
                plt.plot(X, c + b * X + a * X * X, "b")
                plt.show()

        else:
            print("illegal peak refinement type")

        # check for errors in fit
        fitfail = False
        if bipolar:
            lowestcorrcoeff = -1.0
        else:
            lowestcorrcoeff = 0.0
        if (functype == "correlation") or (functype == "hybrid"):
            if maxval < lowestcorrcoeff:
                failreason |= FML_FITAMPLOW
                maxval = np.float64(lowestcorrcoeff)
                if debug:
                    print("bad fit amp: maxval is lower than lower limit")
                fitfail = True
            if np.abs(maxval) > 1.0:
                if not allowhighfitamps:
                    failreason |= FML_FITAMPHIGH
                    if debug:
                        print(
                            "bad fit amp: magnitude of",
                            maxval,
                            "is greater than 1.0",
                        )
                    fitfail = True
                maxval = 1.0 * np.sign(maxval)
        else:
            # different rules for mutual information peaks
            if ((maxval - baseline) < lthreshval * baselinedev) or (maxval < baseline):
                failreason |= FML_FITAMPLOW
                if debug:
                    if (maxval - baseline) < lthreshval * baselinedev:
                        print(
                            "FITAMPLOW: maxval - baseline:",
                            maxval - baseline,
                            " < lthreshval * baselinedev:",
                            lthreshval * baselinedev,
                        )
                    if maxval < baseline:
                        print("FITAMPLOW: maxval < baseline:", maxval, baseline)
                maxval_init = 0.0
                if debug:
                    print("bad fit amp: maxval is lower than lower limit")
        if (lagmin > maxlag) or (maxlag > lagmax):
            if debug:
                print("bad lag after refinement")
            if lagmin > maxlag:
                failreason |= FML_FITLAGLOW
                maxlag = np.float64(lagmin)
            else:
                failreason |= FML_FITLAGHIGH
                maxlag = np.float64(lagmax)
            fitfail = True
        if maxsigma > absmaxsigma:
            failreason |= FML_FITWIDTHHIGH
            if debug:
                print("bad width after refinement:", maxsigma, ">", absmaxsigma)
            maxsigma = np.float64(absmaxsigma)
            fitfail = True
        if maxsigma < absminsigma:
            failreason |= FML_FITWIDTHLOW
            if debug:
                print("bad width after refinement:", maxsigma, "<", absminsigma)
            maxsigma = np.float64(absminsigma)
            fitfail = True
        if fitfail:
            if debug:
                print("fit fail")
            if zerooutbadfit:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            maskval = np.uint16(0)
        # print(maxlag_init, maxlag, maxval_init, maxval, maxsigma_init, maxsigma, maskval, failreason, fitfail)
    else:
        maxval = np.float64(maxval_init)
        maxlag = np.float64(np.fmod(maxlag_init, lagmod))
        maxsigma = np.float64(maxsigma_init)
        if failreason != FML_NOERROR:
            maskval = np.uint16(0)

    if debug or displayplots:
        print(
            "init to final: maxval",
            maxval_init,
            maxval,
            ", maxlag:",
            maxlag_init,
            maxlag,
            ", width:",
            maxsigma_init,
            maxsigma,
        )
    if displayplots and (peakfittype != "None") and (maskval != 0.0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Data and fit")
        hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
        plt.plot(
            X,
            data,
            "ro",
            hiresx,
            gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])),
            "b-",
        )
        plt.show()
    return (
        maxindex,
        maxlag,
        flipfac * maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    )


def _maxindex_noedge(
    corrfunc: NDArray, corrtimeaxis: NDArray, bipolar: bool = False
) -> Tuple[int, float]:
    """
    Find the index of the maximum correlation value, avoiding edge effects.

    This function locates the maximum (or minimum, if bipolar=True) correlation value
    within the given time axis range, while avoiding edge effects by progressively
    narrowing the search window.

    Parameters
    ----------
    corrfunc : NDArray
        Correlation function values to search for maximum.
    corrtimeaxis : NDArray
        Time axis corresponding to the correlation function.
    bipolar : bool, optional
        If True, considers both positive and negative correlation values.
        Default is False.

    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - int: Index of the maximum correlation value
        - float: Flip factor (-1.0 if minimum was selected, 1.0 otherwise)

    Notes
    -----
    The function iteratively narrows the search range by excluding edges
    where the maximum was found. This helps avoid edge effects in correlation
    analysis. When bipolar=True, the function compares both maximum and minimum
    absolute values to determine the optimal selection.

    Examples
    --------
    >>> corrfunc = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    >>> corrtimeaxis = np.array([0, 1, 2, 3, 4])
    >>> index, flip = _maxindex_noedge(corrfunc, corrtimeaxis)
    >>> print(index)
    3
    """
    lowerlim = 0
    upperlim = len(corrtimeaxis) - 1
    done = False
    while not done:
        flipfac = 1.0
        done = True
        maxindex = (np.argmax(corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
        if bipolar:
            minindex = (np.argmax(-corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
            if np.fabs(corrfunc[minindex]) > np.fabs(corrfunc[maxindex]):
                maxindex = minindex
                flipfac = -1.0
        if upperlim == lowerlim:
            done = True
        if maxindex == 0:
            lowerlim += 1
            done = False
        if maxindex == upperlim:
            upperlim -= 1
            done = False
    return maxindex, flipfac
