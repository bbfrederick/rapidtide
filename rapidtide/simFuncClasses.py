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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from statsmodels.robust import mad

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util


class SimilarityFunctionator:
    reftc = None
    prepreftc = None
    testtc = None
    preptesttc = None
    timeaxis = None
    similarityfunclen = 0
    datavalid = False
    timeaxisvalid = False
    similarityfuncorigin = 0

    def __init__(
        self,
        Fs=0.0,
        similarityfuncorigin=0,
        lagmininpts=0,
        lagmaxinpts=0,
        ncprefilter=None,
        negativegradient=False,
        reftc=None,
        reftcstart=0.0,
        detrendorder=1,
        filterinputdata=True,
        debug=False,
    ):
        """
        Initialize the similarity function analysis object.

        Parameters
        ----------
        Fs : float, optional
            Sampling frequency in Hz. Default is 0.0.
        similarityfuncorigin : int, optional
            Origin point for similarity function calculation. Default is 0.
        lagmininpts : int, optional
            Minimum lag in samples. Default is 0.
        lagmaxinpts : int, optional
            Maximum lag in samples. Default is 0.
        ncprefilter : array-like, optional
            Pre-filter for cross-correlation calculation. Default is None.
        negativegradient : bool, optional
            Flag to indicate if negative gradient should be used. Default is False.
        reftc : array-like, optional
            Reference time course for cross-correlation. Default is None.
        reftcstart : float, optional
            Start time for reference time course. Default is 0.0.
        detrendorder : int, optional
            Order of detrending to apply to data. Default is 1.
        filterinputdata : bool, optional
            Flag to indicate if input data should be filtered. Default is True.
        debug : bool, optional
            Flag to enable debug mode. Default is False.

        Returns
        -------
        None
            This method initializes the object attributes but does not return any value.

        Notes
        -----
        The initialization sets up all necessary parameters for cross-correlation analysis.
        If a reference time course is provided, it is set using the setreftc method.
        All lag parameters are converted to integers by adding 0 to ensure proper type handling.

        Examples
        --------
        >>> obj = CrossCorrelationAnalyzer(Fs=100.0, lagmininpts=-10, lagmaxinpts=10)
        >>> obj = CrossCorrelationAnalyzer(reftc=reference_data, reftcstart=5.0)
        """
        self.setFs(Fs)
        self.similarityfuncorigin = similarityfuncorigin
        self.lagmininpts = lagmininpts + 0
        self.lagmaxinpts = lagmaxinpts + 0
        self.ncprefilter = ncprefilter
        self.negativegradient = negativegradient
        self.reftc = reftc
        self.detrendorder = detrendorder
        self.filterinputdata = filterinputdata
        self.debug = debug
        if self.reftc is not None:
            self.setreftc(self.reftc)
            self.reftcstart = reftcstart + 0.0

    def setFs(self, Fs: float) -> None:
        """Set the sampling frequency for the system.

        Parameters
        ----------
        Fs : float
            Sampling frequency in Hz. This parameter determines the rate at which
            samples are taken from the continuous signal.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The sampling frequency is a critical parameter that affects the
        resolution and accuracy of digital signal processing operations.
        It should be set appropriately based on the Nyquist criterion to
        avoid aliasing.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setFs(44100.0)
        >>> print(obj.Fs)
        44100.0
        """
        self.Fs = Fs

    def preptc(self, thetc: NDArray, isreftc: bool = False) -> NDArray:
        """
        Prepare timecourse by filtering, normalizing, detrending, and applying a window function.

        This function applies a series of preprocessing steps to the input timecourse, including
        optional filtering, normalization, detrending, and window function application. The specific
        processing steps depend on the input parameters and class configuration.

        Parameters
        ----------
        thetc : NDArray
            Input timecourse data to be prepared
        isreftc : bool, optional
            Flag indicating whether the input is a reference timecourse. If True, the timecourse
            is filtered using the class's prefilter and then normalized. Default is False.

        Returns
        -------
        NDArray
            Prepared and normalized timecourse data after filtering, normalization, detrending,
            and window function application

        Notes
        -----
        The preprocessing pipeline applies the following steps in order:
        1. Filtering (if applicable based on isreftc and filterinputdata flags)
        2. Gradient calculation (when negativegradient is True)
        3. Normalization with detrending and window function application

        When isreftc is True, the input is filtered using self.ncprefilter.apply() before
        normalization. When isreftc is False and negativegradient is True, the negative gradient
        of the filtered timecourse is used. Otherwise, the filtering behavior depends on the
        filterinputdata flag.

        Examples
        --------
        >>> # Prepare a timecourse with default settings
        >>> prepared_tc = processor.preptc(input_tc)
        >>>
        >>> # Prepare a reference timecourse
        >>> ref_tc = processor.preptc(input_tc, isreftc=True)
        """
        # prepare timecourse by filtering, normalizing, detrending, and applying a window function
        if isreftc:
            thenormtc = tide_math.corrnormalize(
                self.ncprefilter.apply(self.Fs, thetc),
                detrendorder=self.detrendorder,
                windowfunc=self.windowfunc,
            )
        else:
            if self.negativegradient:
                thenormtc = tide_math.corrnormalize(
                    -np.gradient(self.ncprefilter.apply(self.Fs, thetc)),
                    detrendorder=self.detrendorder,
                    windowfunc=self.windowfunc,
                )
            else:
                if self.filterinputdata:
                    thenormtc = tide_math.corrnormalize(
                        self.ncprefilter.apply(self.Fs, thetc),
                        detrendorder=self.detrendorder,
                        windowfunc=self.windowfunc,
                    )
                else:
                    thenormtc = tide_math.corrnormalize(
                        thetc,
                        detrendorder=self.detrendorder,
                        windowfunc=self.windowfunc,
                    )

        return thenormtc

    def trim(self, vector: NDArray) -> NDArray:
        """
        Trim vector based on similarity function origin and lag constraints.

        Parameters
        ----------
        vector : NDArray
            Input vector to be trimmed.

        Returns
        -------
        NDArray
            Trimmed vector containing elements from
            `self.similarityfuncorigin - self.lagmininpts` to
            `self.similarityfuncorigin + self.lagmaxinpts`.

        Notes
        -----
        This function extracts a subset of the input vector based on the origin point
        of the similarity function and the minimum/maximum lag constraints. The trimming
        ensures that only relevant portions of the vector are considered for similarity
        calculations.

        Examples
        --------
        >>> # Assuming self.similarityfuncorigin = 10, self.lagmininpts = 2, self.lagmaxinpts = 3
        >>> trimmed_vector = trim(vector)
        >>> # Returns vector[8:13] where 8 = 10 - 2 and 13 = 10 + 3
        """
        return vector[
            self.similarityfuncorigin
            - self.lagmininpts : self.similarityfuncorigin
            + self.lagmaxinpts
        ]

    def getfunction(self, trim: bool = True) -> tuple[NDArray | None, NDArray | None, int | None]:
        """
        Retrieve simulation function data with optional trimming.

        This method returns the simulation function data and time axis, with optional
        trimming based on the trim parameter. The method handles different validation
        states of the data and returns appropriate tuples of (function, time_axis, max_value)
        or None values depending on the data validity.

        Parameters
        ----------
        trim : bool, optional
            If True, trims the simulation function and time axis using the internal
            trim method. If False, returns the raw data without trimming. Default is True.

        Returns
        -------
        tuple
            A tuple containing:
            - NDArray or None: Trimmed or untrimmed simulation function data
            - NDArray or None: Trimmed or untrimmed time axis data
            - int or None: Global maximum value, or None if not available

        Notes
        -----
        The method checks the validity of data through `self.datavalid` and `self.timeaxisvalid` attributes.
        If `self.datavalid` is True, returns both function and time axis data.
        If `self.datavalid` is False but `self.timeaxisvalid` is True, returns only time axis data.
        If neither is valid, prints an error message and returns (None, None, None).

        Examples
        --------
        >>> result = obj.getfunction(trim=True)
        >>> func_data, time_data, max_val = result
        >>>
        >>> result = obj.getfunction(trim=False)
        >>> func_data, time_data, max_val = result
        """
        if self.datavalid:
            if trim:
                return (
                    self.trim(self.thesimfunc),
                    self.trim(self.timeaxis),
                    self.theglobalmax,
                )
            else:
                return self.thesimfunc, self.timeaxis, self.theglobalmax
        else:
            if self.timeaxisvalid:
                if trim:
                    return None, self.trim(self.timeaxis), None
                else:
                    return None, self.timeaxis, None
            else:
                print("must calculate similarity function before fetching data")
                return None, None, None


class MutualInformationator(SimilarityFunctionator):
    def __init__(
        self,
        windowfunc: str = "hamming",
        norm: bool = True,
        madnorm: bool = False,
        smoothingtime: float = -1.0,
        bins: int = 20,
        sigma: float = 0.25,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MutualInformationator object with specified parameters.

        Parameters
        ----------
        windowfunc : str, optional
            Window function to use for spectral analysis. Default is "hamming".
        norm : bool, optional
            Whether to normalize the data. Default is True.
        madnorm : bool, optional
            Whether to use median absolute deviation normalization. Default is False.
        smoothingtime : float, optional
            Time scale for smoothing filter. If > 0, a noncausal filter is set up.
            Default is -1.0 (no smoothing).
        bins : int, optional
            Number of bins for histogram-based calculations. Default is 20.
        sigma : float, optional
            Standard deviation for Gaussian smoothing. Default is 0.25.
        *args : Any
            Additional positional arguments passed to parent class.
        **kwargs : Any
            Additional keyword arguments passed to parent class.

        Returns
        -------
        None
            This method initializes the object in-place and does not return anything.

        Notes
        -----
        When `smoothingtime` is positive, a noncausal filter is initialized with
        frequency settings based on the specified smoothing time scale.

        Examples
        --------
        >>> mi = MutualInformationator(windowfunc="hanning", bins=30, smoothingtime=2.0)
        >>> mi = MutualInformationator(norm=False, madnorm=True, sigma=0.5)
        """
        self.windowfunc = windowfunc
        self.norm = norm
        self.madnorm = madnorm
        self.bins = bins
        self.sigma = sigma
        self.smoothingtime = smoothingtime
        self.smoothingfilter = tide_filt.NoncausalFilter(filtertype="arb")
        self.mi_norm = 1.0
        if self.smoothingtime > 0.0:
            self.smoothingfilter.setfreqs(
                0.0, 0.0, 1.0 / self.smoothingtime, 1.0 / self.smoothingtime
            )
        super(MutualInformationator, self).__init__(*args, **kwargs)

    def setlimits(self, lagmininpts: int, lagmaxinpts: int) -> None:
        """
        Set the minimum and maximum lag limits for the analysis.

        This function configures the lag limits based on the provided parameters and
        adjusts the smoothing filter padding time if necessary to ensure proper
        signal processing behavior.

        Parameters
        ----------
        lagmininpts : int
            The minimum lag value in terms of number of points.
        lagmaxinpts : int
            The maximum lag value in terms of number of points.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The function automatically adjusts the smoothing filter padding time to be
        no larger than the total time span of the data. If the adjustment is made,
        a message is printed to indicate the new padding time value.

        Examples
        --------
        >>> setlimits(10, 100)
        >>> # Sets minimum lag to 10 points and maximum lag to 100 points
        """
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts
        origpadtime = self.smoothingfilter.getpadtime()
        timespan = self.timeaxis[-1] - self.timeaxis[0]
        newpadtime = np.min([origpadtime, timespan])
        if newpadtime < origpadtime:
            print("lowering smoothing filter pad time to", newpadtime)
            self.smoothingfilter.setpadtime(newpadtime)

    def setbins(self, bins: int) -> None:
        """
        Set the number of bins for histogram calculation.

        Parameters
        ----------
        bins : int
            The number of bins to use for histogram calculation.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the specified number of bins to the instance variable
        `self.bins`. The bins parameter determines the granularity of the histogram
        distribution.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setbins(10)
        >>> print(obj.bins)
        10
        """
        self.bins = bins

    def setreftc(self, reftc: NDArray, offset: float = 0.0) -> None:
        """
        Set reference time course and compute cross-mutual information.

        This method initializes the reference time course and computes the cross-mutual
        information between the reference time course and itself to determine the
        optimal time alignment and similarity function.

        Parameters
        ----------
        reftc : NDArray
            Reference time course array to be set and processed.
        offset : float, optional
            Time offset to be applied to the time axis (default is 0.0).

        Returns
        -------
        None
            This method modifies the object's attributes in-place and does not return anything.

        Notes
        -----
        The method performs the following operations:
        1. Stores a copy of the reference time course
        2. Pre-processes the reference time course using preptc method
        3. Computes cross-mutual information using tide_corr.cross_mutual_info
        4. Adjusts the time axis by the specified offset
        5. Sets various internal attributes including similarity function normalization

        Examples
        --------
        >>> obj.setreftc(reference_data, offset=0.5)
        >>> print(obj.timeaxis)
        >>> print(obj.similarityfunclen)
        """
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc, isreftc=True)

        self.timeaxis, self.automi, self.similarityfuncorigin = tide_corr.cross_mutual_info(
            self.prepreftc,
            self.prepreftc,
            Fs=self.Fs,
            fast=True,
            negsteps=self.lagmininpts,
            possteps=self.lagmaxinpts,
            returnaxis=True,
        )

        self.timeaxis -= offset
        self.similarityfunclen = len(self.timeaxis)
        self.timeaxisvalid = True
        self.datavalid = False
        self.mi_norm = np.nan_to_num(1.0 / np.max(self.automi))
        if self.debug:
            print(f"MutualInformationator setreftc: {len(self.timeaxis)=}")
            print(f"MutualInformationator setreftc: {self.timeaxis}")
            print(f"MutualInformationator setreftc: {self.mi_norm=}")

    def getnormfac(self) -> float:
        """
        Return the normalization factor stored in the instance.

        This method provides access to the normalization factor that has been
        previously computed and stored in the instance variable `mi_norm`.

        Returns
        -------
        float
            The normalization factor value stored in `self.mi_norm`.

        Notes
        -----
        The normalization factor is typically used to scale or normalize
        data within the class. This value should be set before calling this
        method to ensure meaningful results.

        Examples
        --------
        >>> instance = MyClass()
        >>> instance.mi_norm = 2.5
        >>> norm_factor = instance.getnormfac()
        >>> print(norm_factor)
        2.5
        """
        return self.mi_norm

    def run(
        self,
        thetc: NDArray,
        locs: NDArray | None = None,
        trim: bool = True,
        gettimeaxis: bool = True,
    ) -> tuple[NDArray, NDArray, int] | NDArray:
        """
        Compute cross-mutual information between test and reference timecourses.

        This function calculates the cross-mutual information between a test timecourse
        and a reference timecourse, optionally applying preprocessing, trimming, and
        smoothing. It supports both trimmed and untrimmed outputs, and can return
        time axis information depending on the input parameters.

        Parameters
        ----------
        thetc : NDArray
            Test timecourse array of shape (n_times,).
        locs : NDArray | None, optional
            Locations to compute mutual information at; if provided, the function
            will return only the similarity function values at these locations.
            Default is None.
        trim : bool, optional
            If True, trim the output similarity function and time axis to the
            valid range defined by `lagmininpts` and `lagmaxinpts`. If False,
            the full similarity function is returned. Default is True.
        gettimeaxis : bool, optional
            If True, return the time axis along with the similarity function.
            If False, only the similarity function is returned. Default is True.

        Returns
        -------
        tuple[NDArray, NDArray, int] | NDArray
            If `locs` is not None, returns the similarity function values at the
            specified locations.
            If `trim` is True and `gettimeaxis` is True, returns a tuple of:
            - trimmed similarity function (NDArray)
            - trimmed time axis (NDArray)
            - index of the global maximum (int)
            If `trim` is False and `gettimeaxis` is True, returns a tuple of:
            - full similarity function (NDArray)
            - full time axis (NDArray)
            - index of the global maximum (int)
            If `gettimeaxis` is False, returns only the similarity function (NDArray).

        Notes
        -----
        This function uses `tide_corr.cross_mutual_info` for computing cross-mutual
        information, and applies normalization and optional smoothing based on
        instance attributes.

        Examples
        --------
        >>> result = obj.run(test_tc, locs=None, trim=True, gettimeaxis=True)
        >>> sim_func, time_axis, max_idx = result
        """
        if len(thetc) != len(self.reftc):
            print(
                "MutualInformationator: timecourses are of different sizes:",
                len(thetc),
                "!=",
                len(self.reftc),
                "- exiting",
            )
            sys.exit()

        self.testtc = thetc
        self.preptesttc = self.preptc(self.testtc)

        if locs is not None:
            gettimeaxis = True

        # now calculate the similarity function
        if trim:
            retvals = tide_corr.cross_mutual_info(
                self.preptesttc,
                self.prepreftc,
                norm=self.norm,
                negsteps=self.lagmininpts,
                possteps=self.lagmaxinpts,
                locs=locs,
                madnorm=self.madnorm,
                returnaxis=gettimeaxis,
                fast=True,
                Fs=self.Fs,
                sigma=self.sigma,
                bins=self.bins,
            )
        else:
            retvals = tide_corr.cross_mutual_info(
                self.preptesttc,
                self.prepreftc,
                norm=self.norm,
                negsteps=-1,
                possteps=-1,
                locs=locs,
                madnorm=self.madnorm,
                returnaxis=gettimeaxis,
                fast=True,
                Fs=self.Fs,
                sigma=self.sigma,
                bins=self.bins,
            )
        if gettimeaxis:
            self.timeaxis, self.thesimfunc, self.similarityfuncorigin = (
                retvals[0],
                retvals[1],
                retvals[2],
            )
            self.timeaxisvalid = True
        else:
            self.thesimfunc = retvals[0]

        # normalize
        self.thesimfunc *= self.mi_norm

        if locs is not None:
            return self.thesimfunc

        if self.smoothingtime > 0.0:
            self.thesimfunc = self.smoothingfilter.apply(self.Fs, self.thesimfunc)

        self.similarityfunclen = len(self.thesimfunc)
        if trim:
            self.similarityfuncorigin = self.lagmininpts + 1
        else:
            self.similarityfuncorigin = self.similarityfunclen // 2 + 1

        # find the global maximum value
        self.theglobalmax = np.argmax(self.thesimfunc)
        self.datavalid = True

        # make a dummy filtered baseline
        self.filteredbaseline = np.zeros_like(self.thesimfunc)

        if trim:
            return (
                self.trim(self.thesimfunc),
                self.trim(self.timeaxis),
                self.theglobalmax,
            )
        else:
            return self.thesimfunc, self.timeaxis, self.theglobalmax


class Correlator(SimilarityFunctionator):
    def __init__(
        self,
        windowfunc: str = "hamming",
        corrweighting: str = "None",
        corrpadding: int = 0,
        baselinefilter: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Correlator with specified parameters.

        Parameters
        ----------
        windowfunc : str, default="hamming"
            Window function to apply during correlation. Common options include
            'hamming', 'hanning', 'blackman', etc.
        corrweighting : str, default="None"
            Correlation weighting method. Can be 'None' or other weighting schemes
            depending on implementation.
        corrpadding : int, default=0
            Padding size to apply during correlation operations.
        baselinefilter : Any | None, default=None
            Baseline filtering method or object to apply. Can be None to skip filtering.
        *args : Any
            Additional positional arguments passed to parent class.
        **kwargs : Any
            Additional keyword arguments passed to parent class.

        Returns
        -------
        None
            This method initializes the instance and does not return any value.

        Notes
        -----
        The Correlator class inherits from a parent class, and this initialization
        method sets up the correlation parameters before calling the parent's
        initialization method.

        Examples
        --------
        >>> correlator = Correlator(windowfunc="hanning", corrpadding=10)
        >>> correlator = Correlator(baselinefilter=my_filter_object)
        """
        self.windowfunc = windowfunc
        self.corrweighting = corrweighting
        self.corrpadding = corrpadding
        self.baselinefilter = baselinefilter
        super(Correlator, self).__init__(*args, **kwargs)

    def setlimits(self, lagmininpts: int, lagmaxinpts: int) -> None:
        """
        Set the minimum and maximum lag limits for the analysis.

        Parameters
        ----------
        lagmininpts : int
            The minimum lag value in points for the analysis.
        lagmaxinpts : int
            The maximum lag value in points for the analysis.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided lag limits to instance variables
        `self.lagmininpts` and `self.lagmaxinpts`. The lag limits define the
        range of lags to be considered in the subsequent analysis operations.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setlimits(5, 20)
        >>> print(obj.lagmininpts)
        5
        >>> print(obj.lagmaxinpts)
        20
        """
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts

    def setreftc(self, reftc: NDArray, offset: float = 0.0) -> None:
        """
        Set reference time course and initialize related attributes.

        This function sets the reference time course, computes related parameters,
        and initializes the time axis for similarity function calculations.

        Parameters
        ----------
        reftc : NDArray
            Reference time course array used for similarity calculations.
        offset : float, optional
            Time offset to apply to the reference time axis (default is 0.0).

        Returns
        -------
        None
            This function modifies instance attributes in-place and does not return anything.

        Notes
        -----
        This function performs the following operations:
        1. Creates a copy of the reference time course
        2. Computes preprocessed reference time course using preptc method
        3. Calculates similarity function length and origin
        4. Constructs time axis based on sampling frequency and offset

        The time axis is centered around zero with the specified offset applied.

        Examples
        --------
        >>> setreftc(reftc_array, offset=0.5)
        >>> print(self.timeaxis)
        >>> print(self.similarityfunclen)
        """
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc, isreftc=True)
        self.similarityfunclen = len(self.reftc) * 2 - 1
        self.similarityfuncorigin = self.similarityfunclen // 2 + 1

        # make the reference time axis
        self.timeaxis = (
            np.arange(0.0, self.similarityfunclen) * (1.0 / self.Fs)
            - ((self.similarityfunclen - 1) * (1.0 / self.Fs)) / 2.0
        ) - offset
        self.timeaxisvalid = True
        self.datavalid = False

    def run(self, thetc: NDArray, trim: bool = True) -> tuple[NDArray, NDArray, int]:
        """
        Compute the correlation between test and reference timecourses.

        This function performs correlation analysis between a test timecourse and a reference
        timecourse, applying preprocessing and optional filtering. It returns the similarity
        function, time axis, and the index of the global maximum.

        Parameters
        ----------
        thetc : ndarray
            Test timecourse to be correlated with the reference timecourse.
        trim : bool, optional
            If True, trims the similarity function and time axis to remove zero-padding
            effects. Default is True.

        Returns
        -------
        tuple of (ndarray, ndarray, int)
            A tuple containing:
                - similarity function (ndarray)
                - time axis (ndarray)
                - index of the global maximum (int)

        Notes
        -----
        The function exits with status code 1 if the lengths of `thetc` and `self.reftc`
        do not match.

        Examples
        --------
        >>> result = correlator.run(test_timecourse, trim=True)
        >>> similarity_func, time_axis, max_index = result
        """
        if len(thetc) != len(self.reftc):
            print(
                "Correlator: timecourses are of different sizes:",
                len(thetc),
                "!=",
                len(self.reftc),
                "- exiting",
            )
            sys.exit()

        self.testtc = thetc
        self.preptesttc = self.preptc(self.testtc)

        # now actually do the correlation
        self.thesimfunc = tide_corr.fastcorrelate(
            self.preptesttc,
            self.prepreftc,
            usefft=True,
            weighting=self.corrweighting,
            zeropadding=self.corrpadding,
            debug=self.debug,
        )
        self.similarityfunclen = len(self.thesimfunc)
        self.similarityfuncorigin = self.similarityfunclen // 2 + 1

        if self.baselinefilter is not None:
            self.filteredbaseline = self.baselinefilter.apply(self.Fs, self.thesimfunc)
        else:
            self.filteredbaseline = np.zeros_like(self.thesimfunc)

        # find the global maximum value
        self.theglobalmax = np.argmax(self.thesimfunc)
        self.datavalid = True

        if trim:
            return (
                self.trim(self.thesimfunc),
                self.trim(self.timeaxis),
                self.theglobalmax,
            )
        else:
            return self.thesimfunc, self.timeaxis, self.theglobalmax


class SimilarityFunctionFitter:
    corrtimeaxis = None
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
    FML_FITALGOFAIL = np.uint32(0x0400)
    FML_FITFAIL = (
        FML_FITAMPLOW
        | FML_FITAMPHIGH
        | FML_FITWIDTHLOW
        | FML_FITWIDTHHIGH
        | FML_FITLAGLOW
        | FML_FITLAGHIGH
        | FML_FITALGOFAIL
    )

    def __init__(
        self,
        corrtimeaxis=None,
        lagmin=-30.0,
        lagmax=30.0,
        absmaxsigma=1000.0,
        absminsigma=0.25,
        hardlimit=True,
        bipolar=False,
        lthreshval=0.0,
        uthreshval=1.0,
        debug=False,
        zerooutbadfit=True,
        maxguess=0.0,
        useguess=False,
        searchfrac=0.5,
        lagmod=1000.0,
        enforcethresh=True,
        allowhighfitamps=False,
        displayplots=False,
        functype="correlation",
        peakfittype="gauss",
    ):
        """
        Initialize a correlation peak finder.

        This constructor sets up the parameters for fitting and searching correlation
        functions to find peak locations, amplitudes, and widths.

        Parameters
        ----------
        corrtimeaxis : 1D float array, optional
            The time axis of the correlation function. Default is None.
        lagmin : float, optional
            The minimum allowed lag time in seconds. Default is -30.0.
        lagmax : float, optional
            The maximum allowed lag time in seconds. Default is 30.0.
        absmaxsigma : float, optional
            The maximum allowed peak halfwidth in seconds. Default is 1000.0.
        absminsigma : float, optional
            The minimum allowed peak halfwidth in seconds. Default is 0.25.
        hardlimit : bool, optional
            If True, enforce hard limits on peak fitting. Default is True.
        bipolar : bool, optional
            If True, find the correlation peak with the maximum absolute value,
            regardless of sign. Default is False.
        lthreshval : float, optional
            Lower threshold value for correlation function. Default is 0.0.
        uthreshval : float, optional
            Upper threshold value for correlation function. Default is 1.0.
        debug : bool, optional
            If True, enable debug output. Default is False.
        zerooutbadfit : bool, optional
            If True, set bad fits to zero. Default is True.
        maxguess : float, optional
            Maximum guess for peak fitting. Default is 0.0.
        useguess : bool, optional
            If True, use initial guess for peak fitting. Default is False.
        searchfrac : float, optional
            Fraction of the search range to consider for peak fitting. Default is 0.5.
        lagmod : float, optional
            Modulus for lag values. Default is 1000.0.
        enforcethresh : bool, optional
            If True, enforce threshold constraints. Default is True.
        allowhighfitamps : bool, optional
            If True, allow high amplitude fits. Default is False.
        displayplots : bool, optional
            If True, display plots during fitting. Default is False.
        functype : str, optional
            Type of function to fit. Either "correlation" or "mutualinfo". Default is "correlation".
        peakfittype : str, optional
            Type of peak fit to use. Default is "gauss".

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Notes
        -----
        The `corrtimeaxis` must be provided before calling `fit()` method.
        The `functype` parameter determines whether to fit a correlation or mutual information function.

        Examples
        --------
        >>> peakfinder = PeakFinder(corrtimeaxis=time_axis, lagmin=-20, lagmax=20)
        >>> peak_location, peak_value, peak_width = peakfinder.fit(correlation_data)
        """
        self.setcorrtimeaxis(corrtimeaxis)
        self.lagmin = lagmin + 0.0
        self.lagmax = lagmax + 0.0
        self.absmaxsigma = absmaxsigma + 0.0
        self.absminsigma = absminsigma + 0.0
        self.hardlimit = hardlimit
        self.bipolar = bipolar
        self.lthreshval = lthreshval + 0.0
        self.uthreshval = uthreshval + 0.0
        self.debug = debug
        if functype == "correlation" or functype == "mutualinfo":
            self.functype = functype
        else:
            print("illegal functype")
            sys.exit()
        self.peakfittype = peakfittype
        self.zerooutbadfit = zerooutbadfit
        self.maxguess = maxguess + 0.0
        self.useguess = useguess
        self.searchfrac = searchfrac + 0.0
        self.lagmod = lagmod + 0.0
        self.enforcethresh = enforcethresh
        self.allowhighfitamps = allowhighfitamps
        self.displayplots = displayplots

    def _maxindex_noedge(self, corrfunc: NDArray) -> tuple[int, float]:
        """
        Find the index of the maximum value in correlation function, avoiding edge effects.

        This function searches for the maximum value in the correlation function while
        avoiding the edges of the data. It handles bipolar correlation functions by
        considering both positive and negative peaks, returning the one with the larger
        absolute value. The function also accounts for edge effects by adjusting the
        search boundaries when the maximum is found at the edge.

        Parameters
        ----------
        corrfunc : NDArray
            Correlation function array to search for maximum value

        Returns
        -------
        tuple[int, float]
            Tuple containing:
            - maxindex: Index of the maximum value in the correlation function
            - flipfac: Flipping factor (-1.0 if minimum was selected, 1.0 otherwise)

        Notes
        -----
        The function adjusts search boundaries to avoid edge effects:
        - If maximum is at index 0, lowerlim is incremented
        - If maximum is at upper limit, upperlim is decremented
        - For bipolar correlation functions, both positive and negative peaks are considered
        - The search continues until no edge effects are detected

        Examples
        --------
        >>> max_index, flip_factor = obj._maxindex_noedge(corrfunc)
        >>> print(f"Maximum at index {max_index} with flip factor {flip_factor}")
        """
        lowerlim = 0
        upperlim = len(self.corrtimeaxis) - 1
        done = False
        while not done:
            flipfac = 1.0
            done = True
            maxindex = (np.argmax(corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
            if self.bipolar:
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

    def setfunctype(self, functype: str) -> None:
        """
        Set the function type for the object.

        Parameters
        ----------
        functype : str
            The function type to be set. This should be a string identifier
            that defines the type of function this object represents.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided function type to the
        internal `functype` attribute of the object.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setfunctype('linear')
        >>> obj.functype
        'linear'
        """
        self.functype = functype

    def setpeakfittype(self, peakfittype: str) -> None:
        """
        Set the peak fitting type for the analysis.

        Parameters
        ----------
        peakfittype : str
            The type of peak fitting to be used. This parameter determines the
            mathematical model and fitting algorithm applied to the peak data.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided peak fitting type to the
        instance variable `self.peakfittype`. The valid values for peakfittype
        depend on the specific implementation of the peak fitting algorithms
        available in the class.

        Examples
        --------
        >>> analyzer = PeakAnalyzer()
        >>> analyzer.setpeakfittype('gaussian')
        >>> print(analyzer.peakfittype)
        'gaussian'
        """
        self.peakfittype = peakfittype

    def setrange(self, lagmin: float, lagmax: float) -> None:
        """
        Set the range of lags for the analysis.

        Parameters
        ----------
        lagmin : float
            The minimum lag value for the analysis range.
        lagmax : float
            The maximum lag value for the analysis range.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method updates the internal lag range parameters of the object.
        The lagmin value should be less than or equal to the lagmax value.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setrange(0.0, 10.0)
        >>> print(obj.lagmin)
        0.0
        >>> print(obj.lagmax)
        10.0
        """
        self.lagmin = lagmin
        self.lagmax = lagmax

    def setcorrtimeaxis(self, corrtimeaxis: NDArray | None) -> None:
        """
        Set the correlation time axis for the object.

        This method assigns the provided correlation time axis to the object's
        `corrtimeaxis` attribute. If the input is not None, a copy of the array is
        created to avoid modifying the original data.

        Parameters
        ----------
        corrtimeaxis : NDArray | None
            The correlation time axis array to be set. If None, the attribute will
            be set to None. If an array is provided, a copy will be created to
            prevent modification of the original array.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        When a numpy array is passed, the method creates a copy using `+ 0.0`
        to ensure that modifications to the original array do not affect the
        object's internal state.

        Examples
        --------
        >>> obj.setcorrtimeaxis(np.array([1, 2, 3, 4]))
        >>> obj.setcorrtimeaxis(None)
        """
        if corrtimeaxis is not None:
            self.corrtimeaxis = corrtimeaxis + 0.0
        else:
            self.corrtimeaxis = corrtimeaxis

    def setguess(self, useguess: bool, maxguess: float = 0.0) -> None:
        """
        Set the guess parameters for the optimization process.

        This method configures whether to use a guess value and sets the maximum
        guess value for optimization algorithms.

        Parameters
        ----------
        useguess : bool
            Flag indicating whether to use a guess value in the optimization process.
            If True, the algorithm will attempt to use the provided guess value.
            If False, no guess value will be used.
        maxguess : float, optional
            Maximum guess value to be used in the optimization process. Default is 0.0.
            This parameter is only relevant when useguess is True.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The maxguess parameter is typically used to constrain the search space
        during optimization. When useguess is False, the maxguess parameter has
        no effect on the optimization process.

        Examples
        --------
        >>> optimizer = Optimizer()
        >>> optimizer.setguess(True, 10.0)
        >>> optimizer.setguess(False)
        """
        self.useguess = useguess
        self.maxguess = maxguess

    def setlthresh(self, lthreshval: float) -> None:
        """
        Set the lower threshold value for the object.

        Parameters
        ----------
        lthreshval : float
            The lower threshold value to be set. This value will be assigned to
            the instance attribute `lthreshval`.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided threshold value to the instance attribute
        `lthreshval`. The threshold value is typically used for filtering or
        processing operations where values below this threshold are treated
        differently.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setlthresh(0.5)
        >>> print(obj.lthreshval)
        0.5
        """
        self.lthreshval = lthreshval

    def setuthresh(self, uthreshval: float) -> None:
        """
        Set the upper threshold value for the object.

        Parameters
        ----------
        uthreshval : float
            The upper threshold value to be set. This value will be assigned to
            the object's internal `uthreshval` attribute.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided threshold value to the object's
        internal attribute. No validation or processing is performed on the input value.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setuthresh(0.5)
        >>> print(obj.uthreshval)
        0.5
        """
        self.uthreshval = uthreshval

    def diagnosefail(self, failreason: Any) -> str:
        """
        Diagnose the cause of a failure based on bitwise flags.

        This function takes a failure reason encoded as a bitwise flag and returns
        a human-readable string describing the cause(s) of the failure. Each flag
        corresponds to a specific condition that may have led to the failure.

        Parameters
        ----------
        failreason : Any
            A value representing the failure reason, typically an integer or array
            of integers. It is cast to `np.uint32` for bitwise operations.

        Returns
        -------
        str
            A comma-separated string listing the reasons for the failure. If no
            reasons are found, returns "No error".

        Notes
        -----
        The function checks the following flags:
        - ``FML_INITAMPLOW``, ``FML_INITAMPHIGH``, ``FML_INITWIDTHLOW``,
          ``FML_INITWIDTHHIGH``, ``FML_INITLAGLOW``, ``FML_INITLAGHIGH``:
          Initial parameter values are out of bounds.
        - ``FML_FITAMPLOW``, ``FML_FITAMPHIGH``, ``FML_FITWIDTHLOW``,
          ``FML_FITWIDTHHIGH``, ``FML_FITLAGLOW``, ``FML_FITLAGHIGH``:
          Fit parameter values are out of bounds.
        - ``FML_FITALGOFAIL``: Nonlinear fitting algorithm failed.

        Examples
        --------
        >>> diagnosis = obj.diagnosefail(0x0001)
        >>> print(diagnosis)
        'Initial amplitude too low'
        """
        # define error values
        reasons = []
        if failreason.astype(np.uint32) & self.FML_INITAMPLOW:
            reasons.append("Initial amplitude too low")
        if failreason.astype(np.uint32) & self.FML_INITAMPHIGH:
            reasons.append("Initial amplitude too high")
        if failreason.astype(np.uint32) & self.FML_INITWIDTHLOW:
            reasons.append("Initial width too low")
        if failreason.astype(np.uint32) & self.FML_INITWIDTHHIGH:
            reasons.append("Initial width too high")
        if failreason.astype(np.uint32) & self.FML_INITLAGLOW:
            reasons.append("Initial Lag too low")
        if failreason.astype(np.uint32) & self.FML_INITLAGHIGH:
            reasons.append("Initial Lag too high")

        if failreason.astype(np.uint32) & self.FML_FITAMPLOW:
            reasons.append("Fit amplitude too low")
        if failreason.astype(np.uint32) & self.FML_FITAMPHIGH:
            reasons.append("Fit amplitude too high")
        if failreason.astype(np.uint32) & self.FML_FITWIDTHLOW:
            reasons.append("Fit width too low")
        if failreason.astype(np.uint32) & self.FML_FITWIDTHHIGH:
            reasons.append("Fit width too high")
        if failreason.astype(np.uint32) & self.FML_FITLAGLOW:
            reasons.append("Fit Lag too low")
        if failreason.astype(np.uint32) & self.FML_FITLAGHIGH:
            reasons.append("Fit Lag too high")
        if failreason.astype(np.uint32) & self.FML_FITALGOFAIL:
            reasons.append("Nonlinear fit failed")

        if len(reasons) > 0:
            return ", ".join(reasons)
        else:
            return "No error"

    def fit(self, incorrfunc: NDArray) -> tuple[int, float, float, float, int, Any, int, int]:
        """
        Fit a correlation function to determine peak parameters including lag, amplitude, and width.

        This function performs a fit on the provided correlation function to extract key parameters
        such as the peak lag, amplitude, and width. It supports multiple fitting methods and handles
        various edge cases including invalid inputs, out-of-bounds values, and fitting failures.

        Parameters
        ----------
        incorrfunc : ndarray
            The input correlation function to be fitted. Must match the length of `self.corrtimeaxis`.

        Returns
        -------
        tuple[int, float, float, float, int, Any, int, int]
            A tuple containing:

            - `maxindex` (int): Index of the maximum value in the correlation function.
            - `maxlag` (float): The lag corresponding to the peak, in seconds.
            - `maxval` (float): The amplitude of the peak, adjusted for flip factor.
            - `maxsigma` (float): The width of the peak, in seconds.
            - `maskval` (int): A flag indicating fit success (1) or failure (0).
            - `failreason` (Any): A bitmask indicating the reason for fit failure, if any.
            - `peakstart` (int): Start index of the peak region used in fitting.
            - `peakend` (int): End index of the peak region used in fitting.

        Notes
        -----
        The function performs several checks:

        - Ensures `self.corrtimeaxis` is defined and matches the input length.
        - Handles bipolar correlation functions and adjusts signs accordingly.
        - Applies initial parameter estimation based on the input data.
        - Supports multiple fitting algorithms including Gaussian, quadratic, and center-of-mass.
        - Applies bounds checking for lag, amplitude, and width to ensure physical validity.
        - Outputs debugging information if `self.debug` is set to True.

        Examples
        --------
        >>> # Assuming `fit_instance` is an instance of the class containing this method
        >>> corr_func = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        >>> result = fit_instance.fit(corr_func)
        >>> print(result)
        (2, 1.0, 1.0, 0.5, 1, 0, 1, 3)
        """
        # check to make sure xcorr_x and xcorr_y match
        if self.corrtimeaxis is None:
            print("Correlation time axis is not defined - exiting")
            sys.exit()
        if len(self.corrtimeaxis) != len(incorrfunc):
            print(
                "Correlation time axis and values do not match in length (",
                len(self.corrtimeaxis),
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
        failreason = self.FML_NOERROR
        maskval = np.uint16(1)  # start out assuming the fit will succeed
        binwidth = self.corrtimeaxis[1] - self.corrtimeaxis[0]

        # set the search range
        lowerlim = 0
        upperlim = len(self.corrtimeaxis) - 1
        if self.debug:
            print(
                "initial search indices are",
                lowerlim,
                "to",
                upperlim,
                "(",
                self.corrtimeaxis[lowerlim],
                self.corrtimeaxis[upperlim],
                ")",
            )

        # make an initial guess at the fit parameters for the gaussian
        # start with finding the maximum value and its location
        flipfac = 1.0
        corrfunc = incorrfunc + 0.0
        if self.useguess:
            maxindex = tide_util.valtoindex(self.corrtimeaxis, self.maxguess)
            if (corrfunc[maxindex] < 0.0) and self.bipolar:
                flipfac = -1.0
        else:
            maxindex, flipfac = self._maxindex_noedge(corrfunc)
        corrfunc *= flipfac
        maxlag_init = (1.0 * self.corrtimeaxis[maxindex]).astype("float64")
        maxval_init = corrfunc[maxindex].astype("float64")
        if self.debug:
            print(
                "maxindex, maxlag_init, maxval_init:",
                maxindex,
                maxlag_init,
                maxval_init,
            )

        # set the baseline and baselinedev levels
        if (self.functype == "correlation") or (self.functype == "hybrid"):
            baseline = 0.0
            baselinedev = 0.0
        else:
            # for mutual information, there is a nonzero baseline, so we want the difference from that.
            baseline = np.median(corrfunc)
            baselinedev = mad(corrfunc)
        if self.debug:
            print("baseline, baselinedev:", baseline, baselinedev)

        # then calculate the width of the peak
        if self.peakfittype == "fastquad" or self.peakfittype == "COM":
            peakstart = np.max([1, maxindex - 2])
            peakend = np.min([len(self.corrtimeaxis) - 2, maxindex + 2])
        else:
            # come here for peakfittype of None, quad, gauss, fastgauss
            thegrad = np.gradient(corrfunc).astype(
                "float64"
            )  # the gradient of the correlation function
            if (self.functype == "correlation") or (self.functype == "hybrid"):
                if self.peakfittype == "quad":
                    peakpoints = np.where(
                        corrfunc > maxval_init - 0.05, 1, 0
                    )  # mask for places where correlation exceeds searchfrac*maxval_init
                else:
                    peakpoints = np.where(
                        corrfunc > (baseline + self.searchfrac * (maxval_init - baseline)), 1, 0
                    )  # mask for places where correlation exceeds searchfrac*maxval_init
            else:
                # for mutual information, there is a flattish, nonzero baseline, so we want the difference from that.
                peakpoints = np.where(
                    corrfunc > (baseline + self.searchfrac * (maxval_init - baseline)),
                    1,
                    0,
                )

            peakpoints[0] = 0
            peakpoints[-1] = 0
            peakstart = np.max([1, maxindex - 1])
            peakend = np.min([len(self.corrtimeaxis) - 2, maxindex + 1])
            if self.debug:
                print("initial peakstart, peakend:", peakstart, peakend)
            if self.functype == "mutualinfo":
                while peakpoints[peakend + 1] == 1:
                    peakend += 1
                while peakpoints[peakstart - 1] == 1:
                    peakstart -= 1
            else:
                while (
                    thegrad[peakend + 1] <= 0.0
                    and peakpoints[peakend + 1] == 1
                    and peakend < len(self.corrtimeaxis) - 2
                ):
                    peakend += 1
                while (
                    thegrad[peakstart - 1] >= 0.0
                    and peakpoints[peakstart - 1] == 1
                    and peakstart >= 1
                ):
                    peakstart -= 1
            if self.debug:
                print("final peakstart, peakend:", peakstart, peakend)

            # deal with flat peak top
            while (
                peakend < (len(self.corrtimeaxis) - 3)
                and corrfunc[peakend] == corrfunc[peakend - 1]
            ):
                peakend += 1
            while peakstart > 2 and corrfunc[peakstart] == corrfunc[peakstart + 1]:
                peakstart -= 1
            if self.debug:
                print("peakstart, peakend after flattop correction:", peakstart, peakend)
                print("\n")
                for i in range(peakstart, peakend + 1):
                    print(self.corrtimeaxis[i], corrfunc[i])
                print("\n")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Peak sent to fitting routine")
                plt.plot(
                    self.corrtimeaxis[peakstart : peakend + 1],
                    corrfunc[peakstart : peakend + 1],
                    "r",
                )
                plt.show()

            # This is calculated from first principles, but it's always big by a factor or ~1.4.
            #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
            maxsigma_init = np.float64(
                ((peakend - peakstart + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac))))
                / np.sqrt(2.0)
            )
            if self.debug:
                print("maxsigma_init:", maxsigma_init)

            # now check the values for errors
            if self.hardlimit:
                rangeextension = 0.0
            else:
                rangeextension = (self.lagmax - self.lagmin) * 0.75
            if not (
                (self.lagmin - rangeextension - binwidth)
                <= maxlag_init
                <= (self.lagmax + rangeextension + binwidth)
            ):
                if maxlag_init <= (self.lagmin - rangeextension - binwidth):
                    failreason |= self.FML_INITLAGLOW
                    maxlag_init = self.lagmin - rangeextension - binwidth
                else:
                    failreason |= self.FML_INITLAGHIGH
                    maxlag_init = self.lagmax + rangeextension + binwidth
                if self.debug:
                    print("bad initial")
            if maxsigma_init > self.absmaxsigma:
                failreason |= self.FML_INITWIDTHHIGH
                maxsigma_init = self.absmaxsigma
                if self.debug:
                    print("bad initial width - too high")
            if peakend - peakstart < 2:
                failreason |= self.FML_INITWIDTHLOW
                maxsigma_init = np.float64(
                    ((2 + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac)))) / np.sqrt(2.0)
                )
                if self.debug:
                    print("bad initial width - too low")
            if (self.functype == "correlation") or (self.functype == "hybrid"):
                if not (self.lthreshval <= maxval_init <= self.uthreshval) and self.enforcethresh:
                    failreason |= self.FML_INITAMPLOW
                    if self.debug:
                        print(
                            "bad initial amp:",
                            maxval_init,
                            "is less than",
                            self.lthreshval,
                        )
                if maxval_init < 0.0:
                    failreason |= self.FML_INITAMPLOW
                    maxval_init = 0.0
                    if self.debug:
                        print("bad initial amp:", maxval_init, "is less than 0.0")
                if (maxval_init > 1.0) and self.enforcethresh:
                    failreason |= self.FML_INITAMPHIGH
                    maxval_init = 1.0
                    if self.debug:
                        print("bad initial amp:", maxval_init, "is greater than 1.0")
            else:
                # somewhat different rules for mutual information peaks
                if ((maxval_init - baseline) < self.lthreshval * baselinedev) or (
                    maxval_init < baseline
                ):
                    failreason |= self.FML_INITAMPLOW
                    maxval_init = 0.0
                    if self.debug:
                        print("bad initial amp:", maxval_init, "is less than 0.0")
            if (failreason != self.FML_NOERROR) and self.zerooutbadfit:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            else:
                maxval = np.float64(maxval_init)
                maxlag = np.float64(maxlag_init)
                maxsigma = np.float64(maxsigma_init)

        # refine if necessary
        if self.peakfittype != "None":
            if self.peakfittype == "COM":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                maxval = maxval_init
                maxlag = np.sum(X * data) / np.sum(data)
                maxsigma = 10.0
            elif self.peakfittype == "gauss":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                # do a least squares fit over the top of the peak
                # p0 = np.array([maxval_init, np.fmod(maxlag_init, lagmod), maxsigma_init], dtype='float64')
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype="float64")
                if self.debug:
                    print("fit input array:", p0)
                try:
                    plsq, ier = sp.optimize.leastsq(
                        tide_fit.gaussresiduals, p0, args=(data, X), maxfev=5000
                    )
                    if ier not in [1, 2, 3, 4]:  # Check for successful convergence
                        failreason |= self.FML_FITALGOFAIL
                        maxval = np.float64(0.0)
                        maxlag = np.float64(0.0)
                        maxsigma = np.float64(0.0)
                    else:
                        maxval = plsq[0] + baseline
                        maxlag = np.fmod((1.0 * plsq[1]), self.lagmod)
                        maxsigma = plsq[2]
                except:
                    failreason |= self.FML_FITALGOFAIL
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                if self.debug:
                    print("fit output array:", [maxval, maxlag, maxsigma])
            elif self.peakfittype == "gausscf":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                # do a least squares fit over the top of the peak
                try:
                    plsq, pcov = curve_fit(
                        tide_fit.gaussfunc,
                        X,
                        data,
                        p0=[maxval_init, maxlag_init, maxsigma_init],
                    )
                    maxval = plsq[0] + baseline
                    maxlag = np.fmod((1.0 * plsq[1]), self.lagmod)
                    maxsigma = plsq[2]
                except:
                    failreason |= self.FML_FITALGOFAIL
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                if self.debug:
                    print("fit output array:", [maxval, maxlag, maxsigma])
            elif self.peakfittype == "fastgauss":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                # do a non-iterative fit over the top of the peak
                # 6/12/2015  This is just broken.  Gives quantized maxima
                maxlag = np.float64(1.0 * np.sum(X * data) / np.sum(data))
                maxsigma = np.float64(
                    np.sqrt(np.abs(np.sum((X - maxlag) ** 2 * data) / np.sum(data)))
                )
                maxval = np.float64(data.max()) + baseline
            elif self.peakfittype == "fastquad":
                maxlag, maxval, maxsigma, ismax, badfit = tide_fit.refinepeak_quad(
                    self.corrtimeaxis, corrfunc, maxindex
                )
            elif self.peakfittype == "quad":
                X = self.corrtimeaxis[peakstart : peakend + 1]
                data = corrfunc[peakstart : peakend + 1]
                try:
                    thecoffs = Polynomial.fit(X, data, 2).convert().coef[::-1]
                    a = thecoffs[0]
                    b = thecoffs[1]
                    c = thecoffs[2]
                    maxlag = -b / (2.0 * a)
                    maxval = a * maxlag * maxlag + b * maxlag + c
                    maxsigma = 1.0 / np.fabs(a)
                    if self.debug:
                        print("poly coffs:", a, b, c)
                        print("maxlag, maxval, maxsigma:", maxlag, maxval, maxsigma)
                except np.exceptions.RankWarning:
                    failreason |= self.FML_FITALGOFAIL
                    maxlag = 0.0
                    maxval = 0.0
                    maxsigma = 0.0
                if self.debug:
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
            if self.bipolar:
                lowestcorrcoeff = -1.0
            else:
                lowestcorrcoeff = 0.0
            if (self.functype == "correlation") or (self.functype == "hybrid"):
                if maxval < lowestcorrcoeff:
                    failreason |= self.FML_FITAMPLOW
                    maxval = lowestcorrcoeff
                    if self.debug:
                        print("bad fit amp: maxval is lower than lower limit")
                    fitfail = True
                if np.abs(maxval) > 1.0:
                    if not self.allowhighfitamps:
                        failreason |= self.FML_FITAMPHIGH
                        if self.debug:
                            print(
                                "bad fit amp: magnitude of",
                                maxval,
                                "is greater than 1.0",
                            )
                        fitfail = True
                    maxval = 1.0 * np.sign(maxval)
            else:
                # different rules for mutual information peaks
                if ((maxval - baseline) < self.lthreshval * baselinedev) or (maxval < baseline):
                    failreason |= self.FML_FITAMPLOW
                    maxval = 0.0
                    if self.debug:
                        if (maxval - baseline) < self.lthreshval * baselinedev:
                            print(
                                "FITAMPLOW: maxval - baseline:",
                                maxval - baseline,
                                " < lthreshval * baselinedev:",
                                self.lthreshval * baselinedev,
                            )
                        if maxval < baseline:
                            print("FITAMPLOW: maxval < baseline:", maxval, baseline)
                    if self.debug:
                        print("bad fit amp: maxval is lower than lower limit")
            if (self.lagmin > maxlag) or (maxlag > self.lagmax):
                if self.debug:
                    print("bad lag after refinement")
                if self.lagmin > maxlag:
                    failreason |= self.FML_FITLAGLOW
                    maxlag = self.lagmin
                else:
                    failreason |= self.FML_FITLAGHIGH
                    maxlag = self.lagmax
                fitfail = True
            if maxsigma > self.absmaxsigma:
                failreason |= self.FML_FITWIDTHHIGH
                if self.debug:
                    print("bad width after refinement:", maxsigma, ">", self.absmaxsigma)
                maxsigma = self.absmaxsigma
                fitfail = True
            if maxsigma < self.absminsigma:
                failreason |= self.FML_FITWIDTHLOW
                if self.debug:
                    print("bad width after refinement:", maxsigma, "<", self.absminsigma)
                maxsigma = self.absminsigma
                fitfail = True
            if fitfail:
                if self.debug:
                    print("fit fail")
                if self.zerooutbadfit:
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                maskval = np.uint16(0)
            # print(maxlag_init, maxlag, maxval_init, maxval, maxsigma_init, maxsigma, maskval, failreason, fitfail)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(np.fmod(maxlag_init, self.lagmod))
            maxsigma = np.float64(maxsigma_init)
            if failreason != self.FML_NOERROR:
                maskval = np.uint16(0)
            else:
                maskval = np.uint16(1)

        if self.debug or self.displayplots:
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
        if self.displayplots and (self.peakfittype != "None") and (maskval != 0.0):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("Data and fit")
            hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
            plt.plot(
                X,
                data,
                "ro",
                hiresx,
                tide_fit.gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])),
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


class FrequencyTracker:
    freqs = None
    times = None

    def __init__(
        self,
        lowerlim: float = 0.1,
        upperlim: float = 0.6,
        nperseg: int = 32,
        Q: float = 10.0,
        debug: bool = False,
    ) -> None:
        """
        Initialize the object with spectral analysis parameters.

        Parameters
        ----------
        lowerlim : float, optional
            Lower frequency limit for spectral analysis, default is 0.1
        upperlim : float, optional
            Upper frequency limit for spectral analysis, default is 0.6
        nperseg : int, optional
            Number of samples per segment for spectral analysis, default is 32
        Q : float, optional
            Quality factor for spectral analysis, default is 10.0
        debug : bool, optional
            Debug flag for verbose output, default is False

        Returns
        -------
        None
            This method initializes the object attributes and does not return any value.

        Notes
        -----
        The ``nfft`` attribute is set equal to ``nperseg`` during initialization.

        Examples
        --------
        >>> obj = MyClass(lowerlim=0.2, upperlim=0.8, nperseg=64)
        >>> print(obj.lowerlim)
        0.2
        """
        self.lowerlim = lowerlim
        self.upperlim = upperlim
        self.nperseg = nperseg
        self.Q = Q
        self.debug = debug
        self.nfft = self.nperseg

    def track(self, x: NDArray, fs: float) -> tuple[NDArray, NDArray]:
        """
        Track peak frequencies in a signal using spectrogram analysis and peak fitting.

        This function computes the spectrogram of the input signal, then tracks the
        dominant frequency component over time by fitting peaks in each time segment.
        The result is a tuple of time indices and corresponding peak frequencies.

        Parameters
        ----------
        x : NDArray
            Input signal array to be analyzed.
        fs : float
            Sampling frequency of the input signal in Hz.

        Returns
        -------
        tuple[NDArray, NDArray]
            A tuple containing:
            - times : NDArray
                Time indices corresponding to the tracked peaks (excluding the last time bin).
            - peakfreqs : NDArray
                Array of peak frequencies corresponding to each time segment.
                If no valid peak is found within the specified frequency range,
                the value is set to -1.0.

        Notes
        -----
        - The input signal is padded with zeros at both ends to reduce edge effects.
        - The spectrogram is computed using a Hamming window and no overlap between segments.
        - Peak fitting is performed using a fast quadratic method.
        - Frequencies outside the range defined by `self.lowerlim` and `self.upperlim`
          are marked as invalid (set to -1.0).

        Examples
        --------
        >>> times, peakfreqs = obj.track(signal, fs)
        >>> print(f"Peak frequencies: {peakfreqs}")
        >>> print(f"Time indices: {times}")
        """
        self.freqs, self.times, thespectrogram = sp.signal.spectrogram(
            np.concatenate(
                [np.zeros(int(self.nperseg // 2)), x, np.zeros(int(self.nperseg // 2))],
                axis=0,
            ),
            fs=fs,
            detrend="constant",
            scaling="spectrum",
            nfft=None,
            window=np.hamming(self.nfft),
            noverlap=(self.nperseg - 1),
        )
        lowerliminpts = tide_util.valtoindex(self.freqs, self.lowerlim)
        upperliminpts = tide_util.valtoindex(self.freqs, self.upperlim)

        if self.debug:
            print(self.times.shape, self.freqs.shape, thespectrogram.shape)
            print(self.times)

        # initialize the peak fitter
        thefitter = SimilarityFunctionFitter(
            corrtimeaxis=self.freqs,
            lagmin=self.lowerlim,
            lagmax=self.upperlim,
            absmaxsigma=10.0,
            absminsigma=0.1,
            debug=self.debug,
            peakfittype="fastquad",
            zerooutbadfit=False,
            useguess=False,
        )

        peakfreqs = np.zeros((thespectrogram.shape[1] - 1), dtype=float)
        for i in range(0, thespectrogram.shape[1] - 1):
            (
                maxindex,
                peakfreqs[i],
                maxval,
                maxsigma,
                maskval,
                failreason,
                peakstart,
                peakend,
            ) = thefitter.fit(thespectrogram[:, i])
            if not (lowerliminpts <= maxindex <= upperliminpts):
                peakfreqs[i] = -1.0

        return self.times[:-1], peakfreqs

    def clean(
        self, x: NDArray, fs: float, times: NDArray, peakfreqs: NDArray, numharmonics: int = 2
    ) -> NDArray:
        """
        Apply harmonic cleaning to a signal based on detected peak frequencies and their harmonics.

        This function cleans a signal by applying bandpass filtering to specific time intervals
        centered at given peak frequencies. It supports filtering of harmonics up to a specified
        number and handles edge effects through padding.

        Parameters
        ----------
        x : ndarray
            Input signal to be cleaned.
        fs : float
            Sampling frequency of the signal in Hz.
        times : ndarray
            Array of time indices (in seconds) where cleaning is applied.
        peakfreqs : ndarray
            Array of peak frequencies (in Hz) corresponding to each time index.
        numharmonics : int, optional
            Maximum number of harmonics to filter (default is 2).

        Returns
        -------
        ndarray
            Cleaned signal with harmonics filtered out.

        Notes
        -----
        - The function uses Chebyshev type II filter design for each harmonic.
        - Harmonics are filtered using `scipy.signal.filtfilt` for zero-phase filtering.
        - Edge effects are mitigated by padding the input signal with zeros.

        Examples
        --------
        >>> cleaned_signal = obj.clean(x, fs=1000.0, times=[0.1, 0.5], peakfreqs=[50.0, 100.0])
        """
        nyquistfreq = 0.5 * fs
        y = np.zeros_like(x)
        halfwidth = int(self.nperseg // 2)
        padx = np.concatenate([np.zeros(halfwidth), x, np.zeros(halfwidth)], axis=0)
        pady = np.concatenate([np.zeros(halfwidth), y, np.zeros(halfwidth)], axis=0)
        padweight = np.zeros_like(padx)
        if self.debug:
            print(fs, len(times), len(peakfreqs))
        for i in range(0, len(times)):
            centerindex = int(times[i] * fs)
            xstart = centerindex - halfwidth
            xend = centerindex + halfwidth
            if peakfreqs[i] > 0.0:
                filtsignal = padx[xstart:xend]
                numharmonics = np.min([numharmonics, int((nyquistfreq // peakfreqs[i]) - 1)])
                if self.debug:
                    print("numharmonics:", numharmonics, nyquistfreq // peakfreqs[i])
                for j in range(numharmonics + 1):
                    workingfreq = (j + 1) * peakfreqs[i]
                    if self.debug:
                        print("workingfreq:", workingfreq)
                    ws = [workingfreq * 0.95, workingfreq * 1.05]
                    wp = [workingfreq * 0.9, workingfreq * 1.1]
                    gpass = 1.0
                    gstop = 40.0
                    b, a = sp.signal.iirdesign(wp, ws, gpass, gstop, ftype="cheby2", fs=fs)
                    if self.debug:
                        print(
                            i,
                            j,
                            times[i],
                            centerindex,
                            halfwidth,
                            xstart,
                            xend,
                            xend - xstart,
                            wp,
                            ws,
                            len(a),
                            len(b),
                        )
                    filtsignal = sp.signal.filtfilt(b, a, sp.signal.filtfilt(b, a, filtsignal))
                pady[xstart:xend] += filtsignal
            else:
                pady[xstart:xend] += padx[xstart:xend]
            padweight[xstart:xend] += 1.0
        return (pady / padweight)[halfwidth:-halfwidth]
