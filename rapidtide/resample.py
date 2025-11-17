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
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

from typing import Any, Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import scipy as sp
from numpy.typing import ArrayLike, NDArray
from scipy import fftpack, signal

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.util as tide_util
from rapidtide.decorators import conditionaljit, conditionaljit2

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()


# this is here until numpy deals with their fft issue
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)

# --------------------------- Resampling and time shifting functions -------------------------------------------
congridyvals: dict = {}
congridyvals["kernel"] = "kaiser"
congridyvals["width"] = 3.0


def congrid(
    xaxis: NDArray[np.floating[Any]],
    loc: float,
    val: float,
    width: float,
    kernel: str = "kaiser",
    cache: bool = True,
    cyclic: bool = True,
    debug: bool = False,
    onlykeynotices: bool = True,
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Perform a convolution gridding operation with a Kaiser-Bessel or Gaussian kernel.

    This function convolves a given value with a specified gridding kernel and projects
    the result onto a target axis. It supports both cyclic and non-cyclic boundary
    conditions and caches kernel values for performance optimization.

    Parameters
    ----------
    xaxis : NDArray[np.floating[Any]]
        The target axis for resampling. Should be a 1D array of evenly spaced points.
    loc : float
        The location, in x-axis units, of the sample to be gridded.
    val : float
        The value to be gridded.
    width : float
        The width of the gridding kernel in target bins. Must be a half-integral value
        between 1.5 and 5.0 inclusive.
    kernel : {'old', 'gauss', 'kaiser'}, optional
        The type of convolution gridding kernel. Default is 'kaiser'.
        - 'old': Uses a Gaussian kernel with fixed width.
        - 'gauss': Uses a Gaussian kernel with optimized sigma.
        - 'kaiser': Uses a Kaiser-Bessel kernel with optimized beta.
    cache : bool, optional
        If True, caches kernel values for performance. Default is True.
    cyclic : bool, optional
        When True, gridding wraps around the endpoints of xaxis. Default is True.
    debug : bool, optional
        When True, outputs additional information about the gridding process.
        Default is False.
    onlykeynotices : bool, optional
        When True, suppresses certain debug messages. Default is True.

    Returns
    -------
    vals : NDArray[np.floating[Any]]
        The input value, convolved with the gridding kernel, projected onto x-axis points.
    weights : NDArray[np.floating[Any]]
        The values of the convolution kernel, projected onto x-axis points (used for normalization).
    indices : NDArray[int[Any]]
        The indices along the x-axis where the vals and weights fall.

    Notes
    -----
    This implementation is based on the method described in:
    IEEE TRANSACTIONS ON MEDICAL IMAGING. VOL. 10, NO. 3, SEPTEMBER 1991

    Examples
    --------
    >>> import numpy as np
    >>> xaxis = np.linspace(0, 10, 100)
    >>> vals, weights, indices = congrid(xaxis, 5.5, 1.0, 2.0)
    >>> print(vals.shape)
    (100,)
    """
    global congridyvals

    if (congridyvals["kernel"] != kernel) or (congridyvals["width"] != width):
        if congridyvals["kernel"] != kernel:
            if debug and not onlykeynotices:
                print(congridyvals["kernel"], "!=", kernel)
        if congridyvals["width"] != width:
            if debug and not onlykeynotices:
                print(congridyvals["width"], "!=", width)
        if debug:
            print("(re)initializing congridyvals")
        congridyvals = {}
        congridyvals["kernel"] = kernel
        congridyvals["width"] = width * 1.0
    optsigma = np.array([0.4241, 0.4927, 0.4839, 0.5063, 0.5516, 0.5695, 0.5682, 0.5974])
    optbeta = np.array([1.9980, 2.3934, 3.3800, 4.2054, 4.9107, 5.7567, 6.6291, 7.4302])
    xstep = xaxis[1] - xaxis[0]
    if (loc < xaxis[0] - xstep / 2.0 or loc > xaxis[-1] + xstep / 2.0) and not cyclic:
        print("loc", loc, "not in range", xaxis[0], xaxis[-1])

    # choose the smoothing kernel based on the width
    if kernel != "old":
        if not (1.5 <= width <= 5.0) or (np.fmod(width, 0.5) > 0.0):
            print("congrid: width is", width)
            print("congrid: width must be a half-integral value between 1.5 and 5.0 inclusive")
            sys.exit()
        else:
            kernelindex = int((width - 1.5) // 0.5)

    # find the closest grid point to the target location, calculate relative offsets from this point
    center = tide_util.valtoindex(xaxis, loc)
    offset = np.fmod(np.round((loc - xaxis[center]) / xstep, 4), 1.0)  # will vary from -0.5 to 0.5
    if cyclic:
        if center == len(xaxis) - 1 and offset > 0.5:
            center = 0
            offset -= 1.0
        if center == 0 and offset < -0.5:
            center = len(xaxis) - 1
            offset += 1.0
    if not (-0.5 <= offset <= 0.5):
        print("(loc, xstep, center, offset):", loc, xstep, center, offset)
        print("xaxis:", xaxis)
        sys.exit()
    offsetkey = str(offset)

    if kernel == "old":
        if debug:
            print("gridding with old kernel")
        widthinpts = int(np.round(width * 4.6 / xstep))
        widthinpts -= widthinpts % 2 - 1
        try:
            yvals = congridyvals[offsetkey]
        except KeyError:
            if debug:
                print("new key:", offsetkey)
            xvals = (
                np.linspace(
                    -xstep * (widthinpts // 2),
                    xstep * (widthinpts // 2),
                    num=widthinpts,
                    endpoint=True,
                )
                + offset
            )
            yvals = tide_fit.gauss_eval(xvals, np.array([1.0, 0.0, width]))
            if cache:
                congridyvals[offsetkey] = 1.0 * yvals
        startpt = int(center - widthinpts // 2)
        indices = range(startpt, startpt + widthinpts)
        indices = np.remainder(indices, len(xaxis))
        if debug and not onlykeynotices:
            print("center, offset, indices, yvals", center, offset, indices, yvals)
        return val * yvals, yvals, indices
    else:
        offsetinpts = center + offset
        startpt = int(np.ceil(offsetinpts - width / 2.0))
        endpt = int(np.floor(offsetinpts + width / 2.0))
        indices = np.remainder(np.array(list(range(startpt, endpt + 1))), len(xaxis))
        try:
            yvals = congridyvals[offsetkey]
        except KeyError:
            if debug:
                print("new key:", offsetkey)
            xvals = indices - center + offset
            if kernel == "gauss":
                sigma = optsigma[kernelindex]
                yvals = tide_fit.gauss_eval(xvals, np.array([1.0, 0.0, sigma]))
            elif kernel == "kaiser":
                beta = optbeta[kernelindex]
                yvals = tide_fit.kaiserbessel_eval(xvals, np.array([beta, width / 2.0]))
            else:
                print("illegal kernel value in congrid - exiting")
                sys.exit()
            if cache:
                congridyvals[offsetkey] = 1.0 * yvals
            if debug and not onlykeynotices:
                print("xvals, yvals", xvals, yvals)
        if debug and not onlykeynotices:
            print("center, offset, indices, yvals", center, offset, indices, yvals)
        return val * yvals, yvals, indices


class FastResampler:
    def __init__(
        self,
        timeaxis,
        timecourse,
        padtime=30.0,
        upsampleratio=100,
        doplot=False,
        debug=False,
        method="univariate",
    ):
        """
        Initialize the FastResampler with given time axis and time course data.

        This constructor prepares high-resolution time series data by resampling the input
        time course using one of several methods, with optional padding and plotting.

        Parameters
        ----------
        timeaxis : array-like
            The time axis of the input data. Should be a 1D array of time points.
        timecourse : array-like
            The time course data corresponding to `timeaxis`. Should be a 1D array of values.
        padtime : float, optional
            Padding time in seconds to extend the resampled time axis on both ends.
            Default is 30.0.
        upsampleratio : int, optional
            The upsampling ratio used for resampling. Default is 100.
        doplot : bool, optional
            If True, plot the original and high-resolution time courses. Default is False.
        debug : bool, optional
            If True, print debug information during initialization. Default is False.
        method : str, optional
            Resampling method to use. Options are:
            - "univariate": Uses custom resampling logic.
            - "poly": Uses `scipy.signal.resample_poly`.
            - "fourier": Uses `scipy.signal.resample`.
            Default is "univariate".

        Returns
        -------
        None
            This method initializes instance attributes and does not return a value.

        Notes
        -----
        The resampled time axis (`hires_x`) is generated by extending the original time axis
        by `padtime` on each side, using a step size that is `1 / upsampleratio` of the
        original step size. The `hires_y` array contains the resampled time course values.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import signal
        >>> timeaxis = np.linspace(0, 10, 100)
        >>> timecourse = np.sin(timeaxis)
        >>> resampler = FastResampler(timeaxis, timecourse, padtime=5.0, method="poly")
        """
        self.timeaxis = timeaxis
        self.timecourse = timecourse
        self.upsampleratio = upsampleratio
        self.padtime = padtime
        self.initstep = timeaxis[1] - timeaxis[0]
        self.initstart = timeaxis[0]
        self.initend = timeaxis[-1]
        self.hiresstep = self.initstep / np.float64(self.upsampleratio)
        self.hires_x = np.arange(
            timeaxis[0] - self.padtime,
            self.initstep * len(timeaxis) + self.padtime,
            self.hiresstep,
        )
        self.hiresstart = self.hires_x[0]
        self.hiresend = self.hires_x[-1]
        self.method = method
        if self.method == "poly":
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[
                int(self.padtime // self.hiresstep)
                + 1 : -(int(self.padtime // self.hiresstep) + 1)
            ] = signal.resample_poly(timecourse, int(self.upsampleratio * 10), 10)
        elif self.method == "fourier":
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[
                int(self.padtime // self.hiresstep)
                + 1 : -(int(self.padtime // self.hiresstep) + 1)
            ] = signal.resample(timecourse, self.upsampleratio * len(timeaxis))
        else:
            self.hires_y = doresample(timeaxis, timecourse, self.hires_x, method=method)
        self.hires_y[: int(self.padtime // self.hiresstep)] = self.hires_y[
            int(self.padtime // self.hiresstep)
        ]
        self.hires_y[-int(self.padtime // self.hiresstep) :] = self.hires_y[
            -int(self.padtime // self.hiresstep)
        ]
        if debug:
            print("FastResampler __init__:")
            print("    padtime:, ", self.padtime)
            print("    initstep, hiresstep:", self.initstep, self.hiresstep)
            print("    initial axis limits:", self.initstart, self.initend)
            print("    hires axis limits:", self.hiresstart, self.hiresend)

        # self.hires_y[:int(self.padtime // self.hiresstep)] = 0.0
        # self.hires_y[-int(self.padtime // self.hiresstep):] = 0.0
        if doplot:
            import matplolib.pyplot as pl

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("FastResampler initial timecourses")
            plt.plot(timeaxis, timecourse, self.hires_x, self.hires_y)
            plt.legend(("input", "hires"))
            plt.show()

    def getdata(self):
        """
        Retrieve time series data and related parameters.

        Returns
        -------
        tuple
            A tuple containing:
            - timeaxis : array_like
                Time axis values
            - timecourse : array_like
                Time course data values
            - hires_x : array_like
                High resolution x-axis data
            - hires_y : array_like
                High resolution y-axis data
            - inverse_initstep : float
                Reciprocal of the initial step size (1.0 / self.initstep)

        Notes
        -----
        This function provides access to all time series data and associated
        high resolution parameters stored in the object instance. The returned
        inverse_initstep value is commonly used for normalization or scaling
        operations in time series analysis.

        Examples
        --------
        >>> data = obj.getdata()
        >>> time_axis, time_course, hires_x, hires_y, inv_step = data
        >>> print(f"Time course shape: {time_course.shape}")
        """
        return self.timeaxis, self.timecourse, self.hires_x, self.hires_y, 1.0 / self.initstep

    def info(self, prefix=""):
        """
        Print information about the object's time and sampling parameters.

        This method displays various time-related attributes and sampling parameters
        of the object, with optional prefix for better formatting in output.

        Parameters
        ----------
        prefix : str, optional
            String to prepend to each printed line, useful for indentation or
            grouping related output. Default is empty string.

        Returns
        -------
        None
            This method prints information to stdout and does not return any value.

        Notes
        -----
        The method prints the following attributes:
        - timeaxis: Time axis values
        - timecourse: Time course data
        - upsampleratio: Upsampling ratio
        - padtime: Padding time
        - initstep: Initial step size
        - initstart: Initial start time
        - initend: Initial end time
        - hiresstep: High-resolution step size
        - hires_x[0]: First value of high-resolution x-axis
        - hires_x[-1]: Last value of high-resolution x-axis
        - hiresstart: High-resolution start time
        - hiresend: High-resolution end time
        - method: Interpolation/processing method used
        - hires_y[0]: First value of high-resolution y-axis
        - hires_y[-1]: Last value of high-resolution y-axis

        Examples
        --------
        >>> obj.info()
        timeaxis=100
        timecourse=200
        upsampleratio=4
        ...

        >>> obj.info(prefix="  ")
          timeaxis=100
          timecourse=200
          upsampleratio=4
          ...
        """
        print(f"{prefix}{self.timeaxis=}")
        print(f"{prefix}{self.timecourse=}")
        print(f"{prefix}{self.upsampleratio=}")
        print(f"{prefix}{self.padtime=}")
        print(f"{prefix}{self.initstep=}")
        print(f"{prefix}{self.initstart=}")
        print(f"{prefix}{self.initend=}")
        print(f"{prefix}{self.hiresstep=}")
        print(f"{prefix}{self.hires_x[0]=}")
        print(f"{prefix}{self.hires_x[-1]=}")
        print(f"{prefix}{self.hiresstart=}")
        print(f"{prefix}{self.hiresend=}")
        print(f"{prefix}{self.method=}")
        print(f"{prefix}{self.hires_y[0]=}")
        print(f"{prefix}{self.hires_y[-1]=}")

    def save(self, outputname):
        """
        Save the timecourse data to a TSV file.

        This method writes the internal timecourse data to a tab-separated values file
        with additional metadata in the header. The output includes the timecourse data
        along with timing information derived from the object's initialization parameters.

        Parameters
        ----------
        outputname : str
            The path and filename where the TSV output will be saved.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method uses the `tide_io.writebidstsv` function to perform the actual file writing.
        The time step is calculated as 1.0 divided by `self.initstep`, and the start time
        is taken from `self.initstart`. The output file includes a header with description
        metadata indicating this is a lagged timecourse generator.

        Examples
        --------
        >>> obj.save("output_timecourse.tsv")
        >>> # Creates a TSV file with the timecourse data and metadata
        """
        tide_io.writebidstsv(
            outputname,
            self.timecourse,
            1.0 / self.initstep,
            starttime=self.initstart,
            columns=["timecourse"],
            extraheaderinfo={"Description": "The lagged timecourse generator"},
            append=False,
        )

    def yfromx(self, newtimeaxis, doplot=False, debug=False):
        """
        Resample y-values from a high-resolution time axis to a new time axis.

        This method maps values from a high-resolution y-array (`self.hires_y`) to a
        new time axis (`newtimeaxis`) by linear interpolation based on the step size
        and start of the high-resolution axis.

        Parameters
        ----------
        newtimeaxis : array-like
            The new time axis to which the y-values will be resampled.
        doplot : bool, optional
            If True, plot the original high-resolution y-values and the resampled
            values for comparison. Default is False.
        debug : bool, optional
            If True, print debug information including internal parameters and
            bounds checking. Default is False.

        Returns
        -------
        out_y : ndarray
            The resampled y-values corresponding to `newtimeaxis`.

        Notes
        -----
        This function assumes that `self.hires_y` has been precomputed and that
        the internal parameters (`self.hiresstart`, `self.hiresstep`) are valid.
        An IndexError is raised if any index in `outindices` is out of bounds.

        Examples
        --------
        >>> resampler = FastResampler()
        >>> new_times = np.linspace(0, 10, 100)
        >>> y_vals = resampler.yfromx(new_times, doplot=True)
        """
        if debug:
            print("FastResampler: yfromx called with following parameters")
            print("    padtime:, ", self.padtime)
            print("    initstep, hiresstep:", self.initstep, self.hiresstep)
            print("    initial axis limits:", self.initstart, self.initend)
            print("    hires axis limits:", self.hiresstart, self.hiresend)
            print("    requested axis limits:", newtimeaxis[0], newtimeaxis[-1])
        outindices = ((newtimeaxis - self.hiresstart) // self.hiresstep).astype(int)
        if debug:
            print("len(self.hires_y):", len(self.hires_y))
        try:
            out_y = self.hires_y[outindices]
        except IndexError:
            print("")
            print("indexing out of bounds in FastResampler")
            print("    padtime:, ", self.padtime)
            print("    initstep, hiresstep:", self.initstep, self.hiresstep)
            print("    initial axis limits:", self.initstart, self.initend)
            print("    hires axis limits:", self.hiresstart, self.hiresend)
            print("    requested axis limits:", newtimeaxis[0], newtimeaxis[-1])
            sys.exit()
        if doplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("FastResampler timecourses")
            plt.plot(self.hires_x, self.hires_y, newtimeaxis, out_y)
            plt.legend(("hires", "output"))
            plt.show()
        return out_y


def FastResamplerFromFile(
    inputname: str, colspec: Optional[str] = None, debug: bool = False, **kwargs
) -> FastResampler:
    """
    Create a FastResampler from a BIDS TSV file.

    This function reads data from a BIDS TSV file and creates a FastResampler object
    for efficient time series resampling operations. The input file must contain
    exactly one column of data.

    Parameters
    ----------
    inputname : str
        Path to the input BIDS TSV file containing the time series data
    colspec : str, optional
        Column specification for selecting specific columns from the TSV file.
        If None, all columns are read (default: None)
    debug : bool, optional
        Enable debug output printing (default: False)
    **kwargs
        Additional keyword arguments passed to the FastResampler constructor

    Returns
    -------
    FastResampler
        A FastResampler object initialized with the time axis and data from the input file

    Raises
    ------
    ValueError
        If the input file contains multiple columns of data

    Notes
    -----
    The function internally calls `tide_io.readbidstsv` to read the input file and
    constructs a time axis using `np.linspace` based on the sampling rate and
    start time from the input file.

    Examples
    --------
    >>> resampler = FastResamplerFromFile('data.tsv')
    >>> resampler = FastResamplerFromFile('data.tsv', colspec='column1')
    >>> resampler = FastResamplerFromFile('data.tsv', debug=True)
    """
    (
        insamplerate,
        instarttime,
        incolumns,
        indata,
        incompressed,
        incolsource,
    ) = tide_io.readbidstsv(inputname, colspec=colspec, debug=debug)
    if incolumns is not None:
        if len(incolumns) > 1:
            raise ValueError("Multiple columns in input file")
    else:
        raise ValueError("No column names in file")
    intimecourse = indata[0, :]
    intimeaxis = np.linspace(
        instarttime,
        instarttime + len(intimecourse) / insamplerate,
        len(intimecourse),
        endpoint=False,
    )
    if debug:
        print(f"FastResamplerFromFile: {len(intimeaxis)=}, {intimecourse.shape=}")
    return FastResampler(intimeaxis, intimecourse, **kwargs)


def doresample(
    orig_x: NDArray,
    orig_y: NDArray,
    new_x: NDArray,
    method: str = "cubic",
    padlen: int = 0,
    padtype: str = "reflect",
    antialias: bool = False,
    debug: bool = False,
) -> NDArray:
    """
    Resample data from one spacing to another.

    By default, does not apply any antialiasing filter.

    Parameters
    ----------
    orig_x : NDArray
        Original x-coordinates of the data to be resampled.
    orig_y : NDArray
        Original y-values corresponding to `orig_x`.
    new_x : NDArray
        New x-coordinates at which to evaluate the resampled data.
    method : str, optional
        Interpolation method to use. Options are:
        - "cubic": cubic spline interpolation (default)
        - "quadratic": quadratic spline interpolation
        - "univariate": univariate spline interpolation using `scipy.interpolate.UnivariateSpline`
    padlen : int, optional
        Number of elements to pad the input data at both ends. Default is 0.
    padtype : str, optional
        Type of padding to use when `padlen > 0`. Default is "reflect".
        Passed to `tide_filt.padvec`.
    antialias : bool, optional
        If True, apply an antialiasing filter before resampling if the original
        sampling frequency is higher than the target frequency. Default is False.
    debug : bool, optional
        If True, print debug information and display a plot of the original and
        padded data. Default is False.

    Returns
    -------
    ndarray
        Resampled y-values at coordinates specified by `new_x`. If an invalid
        interpolation method is specified, returns None.

    Notes
    -----
    - The function uses padding to handle edge effects during interpolation.
    - When `antialias=True`, a non-causal filter is applied to reduce aliasing
      artifacts when downsampling.
    - The `tide_filt` module is used for padding and filtering operations.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> new_x = np.linspace(0, 10, 200)
    >>> y_resampled = doresample(x, y, new_x, method="cubic")
    """
    tstep = orig_x[1] - orig_x[0]
    if padlen > 0:
        rawxpad = np.linspace(0.0, padlen * tstep, num=padlen, endpoint=False)
        frontpad = rawxpad + orig_x[0] - padlen * tstep
        backpad = rawxpad + orig_x[-1] + tstep
        pad_x = np.concatenate((frontpad, orig_x, backpad))
        pad_y = tide_filt.padvec(orig_y, padlen=padlen, padtype=padtype)
    else:
        pad_x = orig_x
        pad_y = orig_y

    if debug:
        print("padlen=", padlen)
        print("tstep=", tstep)
        print("lens:", len(pad_x), len(pad_y))
        print(pad_x)
        print(pad_y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Original and padded vector")
        plt.plot(orig_x, orig_y + 1.0, pad_x, pad_y)
        plt.show()

    # antialias and ringstop filter
    init_freq = len(pad_x) / (pad_x[-1] - pad_x[0])
    final_freq = len(new_x) / (new_x[-1] - new_x[0])
    if antialias and (init_freq > final_freq):
        aafilterfreq = final_freq / 2.0
        aafilter = tide_filt.NoncausalFilter(filtertype="arb", transferfunc="trapezoidal")
        aafilter.setfreqs(0.0, 0.0, 0.95 * aafilterfreq, aafilterfreq)
        pad_y = aafilter.apply(init_freq, pad_y)

    if method == "cubic":
        cj = signal.cspline1d(pad_y)
        # return tide_filt.unpadvec(
        #   np.float64(signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
        return signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])
    elif method == "quadratic":
        qj = signal.qspline1d(pad_y)
        # return tide_filt.unpadvec(
        #    np.float64(signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
        return signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])
    elif method == "univariate":
        interpolator = sp.interpolate.UnivariateSpline(pad_x, pad_y, k=3, s=0)  # s=0 interpolates
        # return tide_filt.unpadvec(np.float64(interpolator(new_x)), padlen=padlen)
        return (interpolator(new_x)).astype(np.float64)
    else:
        print("invalid interpolation method")
        return None


def arbresample(
    inputdata: NDArray,
    init_freq: float,
    final_freq: float,
    intermed_freq: float = 0.0,
    method: str = "univariate",
    antialias: bool = True,
    decimate: bool = False,
    debug: bool = False,
) -> NDArray:
    """
    Resample input data from an initial frequency to a final frequency using either
    direct resampling or a two-step process with intermediate frequency.

    This function supports both upsampling and downsampling, with optional anti-aliasing
    and decimation. It can operate in debug mode to print intermediate steps and
    statistics.

    Parameters
    ----------
    inputdata : NDArray
        Input signal or data to be resampled. Should be a 1-D array-like object.
    init_freq : float
        Initial sampling frequency of the input data in Hz.
    final_freq : float
        Target sampling frequency of the output data in Hz.
    intermed_freq : float, optional
        Intermediate sampling frequency used in the two-step resampling process.
        If not specified (default: 0.0), it is automatically set to the maximum of
        2 * init_freq and 2 * final_freq.
    method : str, optional
        Interpolation method to use for resampling. Default is "univariate".
        Other options may be supported depending on the underlying implementation.
    antialias : bool, optional
        If True (default), apply anti-aliasing filter during downsampling using
        `scipy.signal.decimate`. If False, use simple interpolation for downsampling.
    decimate : bool, optional
        If True, perform upsampling followed by decimation for downsampling.
        If False, use a two-step resampling approach with `dotwostepresample`.
        Default is False.
    debug : bool, optional
        If True, print debug information including number of points before and after
        resampling, and intermediate steps. Default is False.

    Returns
    -------
    NDArray
        Resampled data as a NumPy array with length adjusted according to `final_freq`.

    Notes
    -----
    - When `decimate=True`, the function uses integer decimation for efficient downsampling.
    - For downsampling, if `antialias=False`, the function uses linear interpolation
      instead of filtering to reduce computational cost.
    - The `intermed_freq` is automatically calculated when `decimate=True` and
      `final_freq < init_freq`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100)
    >>> resampled = arbresample(data, init_freq=100, final_freq=50, decimate=True)
    >>> print(len(resampled))
    50

    See Also
    --------
    upsample : Function used for upsampling when `decimate=True`.
    dotwostepresample : Function used for two-step resampling when `decimate=False`.
    scipy.signal.decimate : Anti-aliasing filter used when `antialias=True`.
    """
    if debug:
        print("arbresample - initial points:", len(inputdata))
    if decimate:
        if final_freq > init_freq:
            # upsample only
            upsampled = upsample(inputdata, init_freq, final_freq, method=method, debug=debug)
            if debug:
                print("arbresample - upsampled points:", len(upsampled))
            return upsampled
        elif final_freq < init_freq:
            # downsampling, so upsample by an amount that allows integer decimation
            intermed_freq = final_freq * np.ceil(init_freq / final_freq)
            q = int(intermed_freq // final_freq)
            if debug:
                print(
                    "going from",
                    init_freq,
                    "to",
                    final_freq,
                    ": upsampling to",
                    intermed_freq,
                    "Hz, then decimating by,",
                    q,
                )
            if intermed_freq == init_freq:
                upsampled = inputdata
            else:
                upsampled = upsample(
                    inputdata, init_freq, intermed_freq, method=method, debug=debug
                )
            if debug:
                print("arbresample - upsampled points:", len(upsampled))
            if antialias:
                downsampled = signal.decimate(upsampled, q)
                if debug:
                    print("arbresample - downsampled points:", len(downsampled))
                return downsampled
            else:
                initaxis = np.linspace(0, len(upsampled), len(upsampled), endpoint=False)
                print(len(initaxis), len(upsampled))
                f = sp.interpolate.interp1d(initaxis, upsampled)
                downsampled = f(
                    q // 2
                    + q * np.linspace(0, len(upsampled) // q, len(upsampled) // q, endpoint=False)
                )
                return downsampled
        else:
            if debug:
                print("arbresample - final points:", len(inputdata))
            return inputdata
    else:
        if intermed_freq <= 0.0:
            intermed_freq = np.max([2.0 * init_freq, 2.0 * final_freq])
        orig_x = (1.0 / init_freq) * np.linspace(
            0.0, 1.0 * len(inputdata), len(inputdata), endpoint=False
        )
        resampled = dotwostepresample(
            orig_x,
            inputdata,
            intermed_freq,
            final_freq,
            method=method,
            antialias=antialias,
            debug=debug,
        )
        if debug:
            print("arbresample - resampled points:", len(resampled))
        return resampled


def upsample(
    inputdata: NDArray,
    Fs_init: float,
    Fs_higher: float,
    method: str = "univariate",
    intfac: bool = False,
    dofilt: bool = True,
    debug: bool = False,
) -> NDArray:
    """
    Upsample input data to a higher sampling frequency.

    This function increases the sampling rate of the input data using interpolation
    and optional filtering. It supports different interpolation methods and allows
    for control over the resampling factor and filtering behavior.

    Parameters
    ----------
    inputdata : NDArray
        Input time series data to be upsampled.
    Fs_init : float
        Initial sampling frequency of the input data (Hz).
    Fs_higher : float
        Target sampling frequency to upsample to (Hz). Must be greater than `Fs_init`.
    method : str, optional
        Interpolation method to use. Default is "univariate". Other options depend
        on the implementation of `doresample`.
    intfac : bool, optional
        If True, use integer resampling factor based on `Fs_higher // Fs_init`.
        If False, compute resampled points based on time duration. Default is False.
    dofilt : bool, optional
        If True, apply a non-causal filter to prevent aliasing. Default is True.
    debug : bool, optional
        If True, print timing information. Default is False.

    Returns
    -------
    NDArray
        Upsampled time series data with the new sampling frequency.

    Notes
    -----
    - The function uses linear interpolation by default, but the actual method
      depends on the `doresample` function implementation.
    - If `dofilt` is True, a trapezoidal non-causal filter is applied with a
      stop frequency set to the minimum of 1.1 * Fs_init / 2.0 and Fs_higher / 2.0.
    - The function will terminate if `Fs_higher` is not greater than `Fs_init`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> upsampled = upsample(data, Fs_init=10.0, Fs_higher=20.0)
    >>> print(len(upsampled))
    200
    """
    starttime = time.time()
    if Fs_higher <= Fs_init:
        print("upsample: target frequency must be higher than initial frequency")
        sys.exit()

    # upsample
    orig_x = np.linspace(0.0, (1.0 / Fs_init) * len(inputdata), num=len(inputdata), endpoint=False)
    endpoint = orig_x[-1] - orig_x[0]
    ts_higher = 1.0 / Fs_higher
    if intfac:
        numresamppts = int(Fs_higher // Fs_init) * len(inputdata)
    else:
        numresamppts = int(endpoint // ts_higher + 1)
    upsampled_x = np.arange(0.0, ts_higher * numresamppts, ts_higher)
    upsampled_y = doresample(orig_x, inputdata, upsampled_x, method=method)
    if dofilt:
        initfilter = tide_filt.NoncausalFilter(
            filtertype="arb", transferfunc="trapezoidal", debug=debug
        )
        stopfreq = np.min([1.1 * Fs_init / 2.0, Fs_higher / 2.0])
        initfilter.setfreqs(0.0, 0.0, Fs_init / 2.0, stopfreq)
        upsampled_y = initfilter.apply(Fs_higher, upsampled_y)
    if debug:
        print("upsampling took", time.time() - starttime, "seconds")
    return upsampled_y


def dotwostepresample(
    orig_x: NDArray,
    orig_y: NDArray,
    intermed_freq: float,
    final_freq: float,
    method: str = "univariate",
    antialias: bool = True,
    debug: bool = False,
) -> NDArray:
    """
    Resample a signal from original frequency to final frequency using a two-step process:
    first upsampling to an intermediate frequency, then downsampling to the final frequency.

    This function performs resampling by first interpolating the input signal to an
    intermediate frequency that is higher than the final desired frequency, followed by
    downsampling to the target frequency. Optional antialiasing filtering is applied
    during the upsampling step.

    Parameters
    ----------
    orig_x : NDArray
        Original time axis values (must be monotonically increasing).
    orig_y : NDArray
        Original signal values corresponding to `orig_x`.
    intermed_freq : float
        Intermediate frequency to which the signal is upsampled.
        Must be greater than `final_freq`.
    final_freq : float
        Target frequency to which the signal is downsampled.
    method : str, optional
        Interpolation method used for resampling. Default is "univariate".
        Should be compatible with the `doresample` function.
    antialias : bool, optional
        If True, apply an antialiasing filter during upsampling. Default is True.
    debug : bool, optional
        If True, print timing and intermediate information. Default is False.

    Returns
    -------
    ndarray
        Resampled signal values at the final frequency.

    Notes
    -----
    - The intermediate frequency must be strictly greater than the final frequency.
    - The function uses `doresample` for interpolation and `tide_filt.NoncausalFilter`
      for antialiasing if enabled.
    - Timing information is printed when `debug=True`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> resampled = dotwostepresample(x, y, intermed_freq=50.0, final_freq=10.0)
    """
    if intermed_freq <= final_freq:
        print("intermediate frequency must be higher than final frequency")
        sys.exit()

    # upsample
    starttime = time.time()
    endpoint = orig_x[-1] - orig_x[0]
    init_freq = len(orig_x) / endpoint
    intermed_ts = 1.0 / intermed_freq
    numresamppts = int(endpoint // intermed_ts + 1)
    intermed_x = intermed_ts * np.linspace(0.0, 1.0 * numresamppts, numresamppts, endpoint=False)
    intermed_y = doresample(orig_x, orig_y, intermed_x, method=method)
    if debug:
        print(
            "init_freq, intermed_freq, final_freq:",
            init_freq,
            intermed_freq,
            final_freq,
        )
        print("intermed_ts, numresamppts:", intermed_ts, numresamppts)
        print("upsampling took", time.time() - starttime, "seconds")

    # antialias and ringstop filter
    if antialias:
        starttime = time.time()
        aafilterfreq = np.min([final_freq, init_freq]) / 2.0
        aafilter = tide_filt.NoncausalFilter(
            filtertype="arb", transferfunc="trapezoidal", debug=debug
        )
        aafilter.setfreqs(0.0, 0.0, 0.95 * aafilterfreq, aafilterfreq)
        antialias_y = aafilter.apply(intermed_freq, intermed_y)
        if debug:
            print("antialiasing took", time.time() - starttime, "seconds")
    else:
        antialias_y = intermed_y

    # downsample
    starttime = time.time()
    final_ts = 1.0 / final_freq
    numresamppts = int(np.ceil(endpoint / final_ts))
    # final_x = np.arange(0.0, final_ts * numresamppts, final_ts)
    final_x = final_ts * np.linspace(0.0, 1.0 * numresamppts, numresamppts, endpoint=False)
    resampled_y = doresample(intermed_x, antialias_y, final_x, method=method)
    if debug:
        print("downsampling took", time.time() - starttime, "seconds")
    return resampled_y


def calcsliceoffset(
    sotype: int, slicenum: int, numslices: int, tr: float, multiband: int = 1
) -> float:
    """
    Calculate slice timing offset for slice timing correction.

    This function computes the timing offset (in seconds) for a given slice
    based on the slice timing correction method specified by `sotype`. The
    offset is used to align slice acquisition times for functional MRI data
    preprocessing.

    Parameters
    ----------
    sotype : int
        Slice timing correction method:
        - 0: None
        - 1: Regular up (0, 1, 2, 3, ...)
        - 2: Regular down
        - 3: Use slice order file (not supported)
        - 4: Use slice timings file (not supported)
        - 5: Standard interleaved (0, 2, 4, ..., 1, 3, 5, ...)
        - 6: Siemens interleaved
        - 7: Siemens multiband interleaved
    slicenum : int
        The index of the slice for which to compute the timing offset.
    numslices : int
        Total number of slices in the volume.
    tr : float
        Repetition time (TR) in seconds.
    multiband : int, optional
        Multiband factor (default is 1). Used only for sotype 7 (Siemens
        multiband interleaved).

    Returns
    -------
    float
        Slice timing offset in seconds.

    Notes
    -----
    For sotypes 3 and 4, the function returns 0.0 as these methods are not
    implemented.

    Examples
    --------
    >>> calcsliceoffset(1, 5, 32, 2.0)
    0.3125

    >>> calcsliceoffset(5, 3, 16, 1.5)
    0.28125
    """
    # Slice timing correction
    # 0 : None
    # 1 : Regular up (0, 1, 2, 3, ...)
    # 2 : Regular down
    # 3 : Use slice order file
    # 4 : Use slice timings file
    # 5 : Standard Interleaved (0, 2, 4 ... 1, 3, 5 ... )
    # 6 : Siemens Interleaved (0, 2, 4 ... 1, 3, 5 ... for odd number of slices)
    # (1, 3, 5 ... 0, 2, 4 ... for even number of slices)
    # 7 : Siemens Multiband Interleaved

    # default value of zero
    slicetime = 0.0

    # None
    if sotype == 0:
        slicetime = 0.0

    # Regular up
    if type == 1:
        slicetime = slicenum * (tr / numslices)

    # Regular down
    if sotype == 2:
        slicetime = (numslices - slicenum - 1) * (tr / numslices)

    # Slice order file not supported - do nothing
    if sotype == 3:
        slicetime = 0.0

    # Slice timing file not supported - do nothing
    if sotype == 4:
        slicetime = 0.0

    # Standard interleave
    if sotype == 5:
        if (slicenum % 2) == 0:
            # even slice number
            slicetime = (tr / numslices) * (slicenum / 2)
        else:
            # odd slice number
            slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens interleave format
    if sotype == 6:
        if (numslices % 2) == 0:
            # even number of slices - slices go 1,3,5,...,0,2,4,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (numslices / 2 + slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((slicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens multiband interleave format
    if sotype == 7:
        numberofshots = numslices / multiband
        modslicenum = slicenum % numberofshots
        if (numberofshots % 2) == 0:
            # even number of shots - slices go 1,3,5,...,0,2,4,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (numberofshots / 2 + modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((modslicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * (
                    (numberofshots + 1) / 2 + (modslicenum - 1) / 2
                )
    return slicetime


# NB: a positive value of shifttrs delays the signal, a negative value advances it
# timeshift using fourier phase multiplication
def timeshift(
    inputtc: ArrayLike, shifttrs: float, padtrs: int, doplot: bool = False, debug: bool = False
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Apply a time shift to a signal using FFT-based modulation and padding.

    This function performs a time shift on an input time course by applying
    phase modulation in the frequency domain, followed by inverse FFT. It uses
    padding and reflection to avoid edge discontinuities. The function also
    returns the corresponding shifted weights and the full padded results.

    Parameters
    ----------
    inputtc : array-like
        Input time course to be shifted. Should be a 1D array of real values.
    shifttrs : float
        Time shift in units of samples. Positive values shift the signal forward,
        negative values shift it backward.
    padtrs : int
        Number of samples to pad the input signal on each side before shifting.
        This helps reduce edge effects.
    doplot : bool, optional
        If True, plots the original and shifted signals. Default is False.
    debug : bool, optional
        If True, prints debug information during execution. Default is False.

    Returns
    -------
    tuple of ndarray
        A tuple containing:
        - shifted_y : ndarray
            The time-shifted signal, cropped to the original length.
        - shifted_weights : ndarray
            The corresponding shifted weights, cropped to the original length.
        - shifted_y_full : ndarray
            The full time-shifted signal including padding.
        - shifted_weights_full : ndarray
            The full shifted weights including padding.

    Notes
    -----
    The function uses reflection padding to minimize edge artifacts. The phase
    modulation is applied in the frequency domain using FFT and inverse FFT.
    The shift is implemented as a complex exponential modulation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> input_signal = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> shifted_sig, weights, full_shifted, full_weights = timeshift(
    ...     input_signal, shifttrs=5.0, padtrs=10, doplot=False
    ... )
    >>> print(shifted_sig.shape)
    (100,)
    """
    # set up useful parameters
    thelen = np.shape(inputtc)[0]
    thepaddedlen = thelen + 2 * padtrs
    if debug:
        print("timesshift: thelen, padtrs, thepaddedlen=", thelen, padtrs, thepaddedlen)
    imag = 1.0j

    # initialize variables
    preshifted_y = np.zeros(
        thepaddedlen, dtype="float"
    )  # initialize the working buffer (with pad)
    weights = np.zeros(thepaddedlen, dtype="float")  # initialize the weight buffer (with pad)

    # now do the math
    preshifted_y[padtrs : padtrs + thelen] = inputtc[:]  # copy initial data into shift buffer
    weights[padtrs : padtrs + thelen] = 1.0  # put in the weight vector
    revtc = inputtc[::-1]  # reflect data around ends to
    preshifted_y[0:padtrs] = revtc[-padtrs:]  # eliminate discontinuities
    preshifted_y[padtrs + thelen :] = revtc[0:padtrs]

    # finish initializations
    fftlen = np.shape(preshifted_y)[0]

    # create the phase modulation timecourse
    initargvec = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(fftlen)) - np.pi
    if len(initargvec) > fftlen:
        initargvec = initargvec[:fftlen]
    argvec = np.roll(initargvec * shifttrs, -int(fftlen // 2))
    modvec = np.cos(argvec) - imag * np.sin(argvec)

    # process the data (fft->modulate->ifft->filter)
    fftdata = fftpack.fft(preshifted_y)  # do the actual shifting
    shifted_y = fftpack.ifft(modvec * fftdata).real

    # process the weights
    w_fftdata = fftpack.fft(weights)  # do the actual shifting
    shifted_weights = fftpack.ifft(modvec * w_fftdata).real

    if doplot:
        xvec = range(0, thepaddedlen)  # make a ramp vector (with pad)
        print("shifttrs:", shifttrs)
        print("offset:", padtrs)
        print("thelen:", thelen)
        print("thepaddedlen:", thepaddedlen)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Initial vector")
        plt.plot(xvec, preshifted_y)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Initial and shifted vector")
        plt.plot(xvec, preshifted_y, xvec, shifted_y)

        plt.show()

    return (
        shifted_y[padtrs : padtrs + thelen],
        shifted_weights[padtrs : padtrs + thelen],
        shifted_y,
        shifted_weights,
    )


def timewarp(
    orig_x: NDArray,
    orig_y: NDArray,
    timeoffset: NDArray,
    demean: bool = True,
    method: str = "univariate",
    debug: bool = False,
) -> NDArray:
    """
    Apply time warping to align time series data based on time offsets.

    This function performs time warping by resampling input data according to
    provided time offsets. It can optionally remove the mean of the time offsets
    before resampling to center the data around zero.

    Parameters
    ----------
    orig_x : NDArray
        Original time axis values (x-coordinates) for the data to be warped.
    orig_y : NDArray
        Original signal values (y-coordinates) corresponding to orig_x.
    timeoffset : NDArray
        Time offsets to be applied to each point in the time axis. Positive values
        shift data forward in time, negative values shift backward.
    demean : bool, optional
        If True, remove the mean of timeoffset before resampling. Default is True.
    method : str, optional
        Resampling method to use. Options are 'univariate' or other methods
        supported by the underlying doresample function. Default is 'univariate'.
    debug : bool, optional
        If True, print debugging information about the warping process.
        Default is False.

    Returns
    -------
    NDArray
        Warped time series data after applying the time offsets and resampling.

    Notes
    -----
    The function calculates the maximum deviation in samples and uses half the
    length of the input data (or 30 seconds worth of samples, whichever is smaller)
    as padding length for the resampling operation.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> offsets = np.random.normal(0, 0.1, 100)
    >>> warped_y = timewarp(x, y, offsets)
    """
    if demean:
        demeanedoffset = timeoffset - np.mean(timeoffset)
        if debug:
            print("mean delay of ", np.mean(timeoffset), "seconds removed prior to resampling")
    else:
        demeanedoffset = timeoffset
    sampletime = float(orig_x[1] - orig_x[0])
    maxdevs = (float(np.min(demeanedoffset)), float(np.max(demeanedoffset)))
    maxsamps = (maxdevs[0] / sampletime, maxdevs[1] / sampletime)
    padlen = np.min([int(len(orig_x) // 2), int(30.0 / sampletime)])
    if debug:
        print("maximum deviation in samples:", maxsamps)
        print("padlen in samples:", padlen)
    return doresample(orig_x, orig_y, orig_x + demeanedoffset, method=method, padlen=padlen)
