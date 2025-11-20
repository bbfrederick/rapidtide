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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.typing import NDArray

import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util


class fMRIDataset:
    thedata = None
    theshape = None
    xsize = None
    ysize = None
    numslices = None
    realtimepoints = None
    timepoints = None
    slicesize = None
    numvox = None
    numskip = 0
    validvoxels = None

    def __init__(
        self, thedata: NDArray, zerodata: bool = False, copydata: bool = False, numskip: int = 0
    ) -> None:
        """
        Initialize the fMRIDataset with data and configuration parameters.

        Parameters
        ----------
        thedata : NDArray
            The input data array to be stored in the object.
        zerodata : bool, optional
            If True, initializes the data with zeros instead of copying the input data.
            Default is False.
        copydata : bool, optional
            If True and zerodata is False, creates a copy of the input data.
            If False and zerodata is False, uses the input data directly.
            Default is False.
        numskip : int, optional
            Number of elements to skip during processing. Default is 0.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The initialization process involves:
        1. Setting the data based on the zerodata and copydata parameters
        2. Calling getsizes() to determine data dimensions
        3. Calling setnumskip() to configure the skip parameter

        Examples
        --------
        >>> obj = MyClass(data_array)
        >>> obj = MyClass(data_array, zerodata=True)
        >>> obj = MyClass(data_array, copydata=True, numskip=5)
        """
        if zerodata:
            self.thedata = np.zeros_like(thedata)
        else:
            if copydata:
                self.thedata = thedata + 0.0
            else:
                self.thedata = thedata
        self.getsizes()
        self.setnumskip(numskip)

    def getsizes(self) -> None:
        """
        Calculate and store various size parameters from data shape.

        This method extracts dimensional information from the data shape and computes
        derived quantities such as slice size and total number of voxels. It handles
        both 3D and 4D data arrays by checking for the presence of a fourth dimension.

        Parameters
        ----------
        self : object
            The instance containing the data array in `thedata` attribute.
            The `thedata` attribute should be a numpy array with shape (xsize, ysize, numslices, [realtimepoints])

        Returns
        -------
        None
            This method does not return any value but modifies the instance attributes.

        Notes
        -----
        The method assumes `thedata` is a numpy array with at least 2 dimensions.
        If the fourth dimension is not present, `realtimepoints` is set to 1.

        Attributes Modified
        -------------------
        theshape : tuple
            The shape of the data array
        xsize : int
            Size of the first dimension (x-axis)
        ysize : int
            Size of the second dimension (y-axis)
        numslices : int
            Number of slices (third dimension)
        realtimepoints : int
            Number of real-time points (fourth dimension, default 1)
        slicesize : int
            Product of xsize and ysize (number of pixels per slice)
        numvox : int
            Total number of voxels (slicesize * numslices)

        Examples
        --------
        >>> # Assuming self.thedata has shape (64, 64, 30, 100)
        >>> getsizes(self)
        >>> print(self.xsize, self.ysize, self.numslices, self.realtimepoints)
        64 64 30 100
        >>> print(self.slicesize, self.numvox)
        4096 122880
        """
        self.theshape = self.thedata.shape
        self.xsize = self.theshape[0]
        self.ysize = self.theshape[1]
        self.numslices = self.theshape[2]
        try:
            self.realtimepoints = self.theshape[3]
        except KeyError:
            self.realtimepoints = 1
        self.slicesize = self.xsize * self.ysize
        self.numvox = self.slicesize * self.numslices

    def setnumskip(self, numskip: int) -> None:
        """
        Set the number of timepoints to skip and update the timepoints accordingly.

        This method updates the internal `numskip` attribute and recalculates the
        `timepoints` by subtracting the number of skipped points from the real timepoints.

        Parameters
        ----------
        numskip : int
            The number of timepoints to skip. This value is stored in the `numskip`
            attribute and used to compute the effective timepoints.

        Returns
        -------
        None
            This method modifies the object's attributes in-place and does not return
            any value.

        Notes
        -----
        The `timepoints` attribute is automatically updated to reflect the difference
        between `realtimepoints` and `numskip`. This is typically used in time-series
        analysis where certain initial timepoints are excluded from calculations.

        Examples
        --------
        >>> obj.setnumskip(5)
        >>> print(obj.numskip)
        5
        >>> print(obj.timepoints)
        # Will show the difference between realtimepoints and 5
        """
        self.numskip = numskip
        self.timepoints = self.realtimepoints - self.numskip

    def setvalid(self, validvoxels: NDArray) -> None:
        """
        Set the valid voxels for the object.

        Parameters
        ----------
        validvoxels : NDArray
            Array containing the valid voxel indices or flags indicating
            which voxels are considered valid in the dataset.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided array to the internal `validvoxels` attribute
        of the object, which is typically used to filter or validate voxel data
        during processing operations.

        Examples
        --------
        >>> obj = MyClass()
        >>> valid_voxels = np.array([1, 2, 3, 5, 8])
        >>> obj.setvalid(valid_voxels)
        >>> print(obj.validvoxels)
        [1 2 3 5 8]
        """
        self.validvoxels = validvoxels

    def byslice(self) -> NDArray:
        """
        Return data sliced along the time dimension with specified skip.

        This method extracts data from the internal `thedata` array, skipping
        the first `numskip` time points and reshaping the result into a
        standardized 3D array format.

        Returns
        -------
        NDArray
            3D array with shape (slicesize, numslices, timepoints) containing
            the sliced data with skipped time points removed.

        Notes
        -----
        The returned array is reshaped from the original data structure to
        facilitate further processing and analysis. The slicing operation
        removes the first `numskip` time points from the original data.

        Examples
        --------
        >>> result = obj.byslice()
        >>> print(result.shape)
        (100, 50, 200)
        """
        return self.thedata[:, :, :, self.numskip :].reshape(
            (self.slicesize, self.numslices, self.timepoints)
        )

    def byvol(self) -> NDArray:
        """
        Reshape data to volume-time format.

        This method extracts a subset of data along the fourth dimension and reshapes
        it into a 2D array where rows represent voxels and columns represent timepoints.

        Returns
        -------
        NDArray
            2D array of shape (numvox, timepoints) containing the reshaped data.
            Each row corresponds to a voxel and each column to a timepoint.

        Notes
        -----
        The method slices the data array starting from index `numskip` along the
        fourth dimension and reshapes the remaining data into a 2D structure.

        Examples
        --------
        >>> result = obj.byvol()
        >>> print(result.shape)
        (numvox, timepoints)
        """
        return self.thedata[:, :, :, self.numskip :].reshape((self.numvox, self.timepoints))

    def byvox(self) -> NDArray:
        """
        Return voxel data with skip dimension sliced.

        This method extracts a subset of the fourth dimension from the internal
        data array, starting from the index specified by `numskip` to the end.

        Returns
        -------
        NDArray
            A numpy array containing the voxel data with the fourth dimension
            sliced from `numskip` index to the end. The shape will be
            (self.thedata.shape[0], self.thedata.shape[1], self.thedata.shape[2],
            self.thedata.shape[3] - self.numskip)

        Notes
        -----
        The function assumes that `self.thedata` is a 4-dimensional numpy array
        and `self.numskip` is a non-negative integer less than the size of the
        fourth dimension.

        Examples
        --------
        >>> # Assuming self.thedata has shape (10, 10, 10, 20) and self.numskip = 5
        >>> result = self.byvox()
        >>> result.shape
        (10, 10, 10, 15)
        """
        return self.thedata[:, :, :, self.numskip :]


class ProbeRegressor:
    inputtimeaxis = None
    inputvec = None
    inputfreq = None
    inputstart = 0.0
    inputoffset = 0.0
    targettimeaxis = None
    targetvec = None
    targetfreq = None
    targetstart = 0.0
    targetoffset = 0.0

    def __init__(
        self,
        inputvec: NDArray,
        inputfreq: float,
        targetperiod: float,
        targetpoints: int,
        targetstartpoint: int,
        targetoversample: int = 1,
        inputstart: float = 0.0,
        inputoffset: float = 0.0,
        targetstart: float = 0.0,
        targetoffset: float = 0.0,
    ) -> None:
        """
        Initialize the object with input and target parameters.

        Parameters
        ----------
        inputvec : NDArray
            Input vector data array
        inputfreq : float
            Input frequency in Hz
        targetperiod : float
            Target period in seconds
        targetpoints : int
            Number of target points
        targetstartpoint : int
            Starting point index for target
        targetoversample : int, optional
            Oversampling factor for target (default is 1)
        inputstart : float, optional
            Starting time for input (default is 0.0)
        inputoffset : float, optional
            Input offset value (default is 0.0)
        targetstart : float, optional
            Starting time for target (default is 0.0)
        targetoffset : float, optional
            Target offset value (default is 0.0)

        Returns
        -------
        None
            This method initializes the object attributes and does not return any value.

        Notes
        -----
        This constructor sets up the input vector with specified frequency and start time,
        and initializes target parameters for subsequent processing.

        Examples
        --------
        >>> obj = MyClass(inputvec=np.array([1, 2, 3]), inputfreq=100.0,
        ...               targetperiod=0.1, targetpoints=100, targetstartpoint=0)
        """
        self.inputoffset = inputoffset
        self.setinputvec(inputvec, inputfreq, inputstart=inputstart)
        self.targetperiod = targetperiod
        self.makeinputtimeaxis()
        self.targetoversample = targetoversample
        self.targetpoints = targetpoints
        self.targetstartpoint = targetstartpoint

    def setinputvec(self, inputvec: NDArray, inputfreq: float, inputstart: float = 0.0) -> None:
        """
        Set the input vector and associated parameters for the object.

        Parameters
        ----------
        inputvec : NDArray
            The input vector to be set. This is typically a numpy array containing
            the input signal or data values.
        inputfreq : float
            The input frequency value. This represents the sampling frequency or
            frequency parameter associated with the input vector.
        inputstart : float, optional
            The starting time or phase value for the input. Default is 0.0.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided input vector and its associated parameters
        to instance variables. The input vector is stored as ``self.inputvec``,
        the frequency as ``self.inputfreq``, and the start value as ``self.inputstart``.

        Examples
        --------
        >>> import numpy as np
        >>> obj = MyClass()
        >>> input_data = np.array([1, 2, 3, 4, 5])
        >>> obj.setinputvec(input_data, inputfreq=10.0, inputstart=0.5)
        >>> print(obj.inputvec)
        [1 2 3 4 5]
        >>> print(obj.inputfreq)
        10.0
        >>> print(obj.inputstart)
        0.5
        """
        self.inputvec = inputvec
        self.inputfreq = inputfreq
        self.inputstart = inputstart

    def makeinputtimeaxis(self) -> None:
        """
        Create input time axis based on input vector properties.

        This method generates a time axis for input data by linearly spacing
        from 0 to the length of the input vector, normalized by the input frequency,
        and adjusted by the input start time and offset.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method modifies the instance in-place by setting the `inputtimeaxis` attribute.

        Notes
        -----
        The time axis is calculated as:
        ``inputtimeaxis = np.linspace(0.0, len(inputvec)) / inputfreq - (inputstarttime + inputoffset)``

        Examples
        --------
        >>> obj.makeinputtimeaxis()
        >>> print(obj.inputtimeaxis)
        [ 0.          0.001       0.002 ...  0.998       0.999      ]
        """
        self.inputtimeaxis = np.linspace(0.0, len(self.inputvec)) / self.inputfreq - (
            self.inputstarttime + self.inputoffset
        )

    def maketargettimeaxis(self) -> None:
        """
        Create a target time axis for signal processing.

        This method generates a linearly spaced time axis based on the target period,
        start point, and number of points specified in the object's attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method modifies the object's `targettimeaxis` attribute in-place.

        Notes
        -----
        The time axis is generated using `numpy.linspace` with the following parameters:
        - Start point: `targetperiod * targetstartpoint`
        - End point: `targetperiod * targetstartpoint + targetperiod * targetpoints`
        - Number of points: `targetpoints`
        - Endpoint: True

        Examples
        --------
        >>> obj.maketargettimeaxis()
        >>> print(obj.targettimeaxis)
        [0.   0.1  0.2  0.3  0.4]
        """
        self.targettimeaxis = np.linspace(
            self.targetperiod * self.targetstartpoint,
            self.targetperiod * self.targetstartpoint + self.targetperiod * self.targetpoints,
            num=self.targetpoints,
            endpoint=True,
        )


class Coherer:
    reftc = None
    prepreftc = None
    testtc = None
    preptesttc = None
    freqaxis = None
    similarityfunclen = 0
    datavalid = False
    freqaxisvalid = False
    similarityfuncorigin = 0
    freqmin = None
    freqmax = None

    def __init__(
        self,
        Fs: float = 0.0,
        freqmin: float | None = None,
        freqmax: float | None = None,
        ncprefilter: Any | None = None,
        reftc: NDArray | None = None,
        detrendorder: int = 1,
        windowfunc: str = "hamming",
        debug: bool = False,
    ) -> None:
        """
        Initialize the Coherer object with configuration parameters.

        Parameters
        ----------
        Fs : float, default=0.0
            Sampling frequency in Hz.
        freqmin : float, optional
            Minimum frequency for filtering. If None, no minimum frequency filtering is applied.
        freqmax : float, optional
            Maximum frequency for filtering. If None, no maximum frequency filtering is applied.
        ncprefilter : Any, optional
            Pre-filtering configuration for non-coherent filtering.
        reftc : NDArray, optional
            Reference time course for coherence calculations.
        detrendorder : int, default=1
            Order of detrending to apply to the data. 0 for no detrending, 1 for linear detrending.
        windowfunc : str, default="hamming"
            Window function to apply during spectral analysis. Options include 'hamming', 'hanning', 'blackman', etc.
        debug : bool, default=False
            If True, print initialization debug information.

        Returns
        -------
        None
            This method initializes the object's attributes and performs setup operations.

        Notes
        -----
        The initialization process sets up all internal parameters and performs optional
        debug printing if requested. The freqmin and freqmax parameters are only stored
        if they are not None. The reftc parameter, if provided, triggers the setreftc method.

        Examples
        --------
        >>> coherer = Coherer(Fs=100.0, freqmin=1.0, freqmax=50.0, debug=True)
        Coherer init:
            Fs: 100.0
            windowfunc: hamming
            detrendorder: 1
            freqmin: 1.0
            freqmax: 50.0
        """
        self.Fs = Fs
        self.ncprefilter = ncprefilter
        self.reftc = reftc
        self.windowfunc = windowfunc
        self.detrendorder = detrendorder
        self.debug = debug
        if freqmin is not None:
            self.freqmin = freqmin
        if freqmax is not None:
            self.freqmax = freqmax
        if self.reftc is not None:
            self.setreftc(self.reftc)
        if self.debug:
            print("Coherer init:")
            print("\tFs:", self.Fs)
            print("\twindowfunc:", self.windowfunc)
            print("\tdetrendorder:", self.detrendorder)
            print("\tfreqmin:", self.freqmin)
            print("\tfreqmax:", self.freqmax)

    def preptc(self, thetc: NDArray) -> NDArray:
        """
        Prepare timecourse by filtering, normalizing, detrending, and applying a window function.

        This function applies a series of preprocessing steps to a timecourse signal including
        noise filtering, normalization, and detrending to prepare it for further analysis.

        Parameters
        ----------
        thetc : ndarray
            Input timecourse data to be prepared, typically a 1D array of signal values.

        Returns
        -------
        ndarray
            Preprocessed timecourse data after filtering, normalization, and detrending.

        Notes
        -----
        The preprocessing pipeline includes:
        1. Noise filtering using the class's ncprefilter
        2. Correlation-based normalization
        3. Detrending with specified order

        Examples
        --------
        >>> # Assuming 'obj' is an instance of the class containing this method
        >>> processed_tc = obj.preptc(timecourse_data)
        >>> print(processed_tc.shape)
        (n_timepoints,)
        """
        # prepare timecourse by filtering, normalizing, detrending, and applying a window function
        return tide_math.corrnormalize(
            self.ncprefilter.apply(self.Fs, thetc),
            detrendorder=self.detrendorder,
            windowfunc="None",
        )

    def setlimits(self, freqmin: float, freqmax: float) -> None:
        """
        Set frequency limits for the object and calculate corresponding indices.

        This method sets the minimum and maximum frequency values and calculates the
        corresponding indices in the frequency axis if the frequency axis is valid.

        Parameters
        ----------
        freqmin : float
            The minimum frequency value to set.
        freqmax : float
            The maximum frequency value to set.

        Returns
        -------
        None
            This method modifies the object's attributes in-place and does not return a value.

        Notes
        -----
        If `self.freqaxisvalid` is True, the method calculates `freqmininpts` and `freqmaxinpts`
        which represent the indices corresponding to the frequency limits in `self.freqaxis`.
        The calculation ensures that indices stay within valid bounds (0 to len(freqaxis)-1).

        Examples
        --------
        >>> obj.setlimits(0.1, 1.0)
        >>> print(obj.freqmin, obj.freqmax)
        0.1 1.0
        """
        self.freqmin = freqmin
        self.freqmax = freqmax
        if self.freqaxisvalid:
            self.freqmininpts = np.max([0, tide_util.valtoindex(self.freqaxis, self.freqmin)])
            self.freqmaxinpts = np.min(
                [
                    tide_util.valtoindex(self.freqaxis, self.freqmax),
                    len(self.freqaxis) - 1,
                ]
            )
        if self.debug:
            print("setlimits:")
            print("\tfreqmin,freqmax:", self.freqmin, self.freqmax)
            print("\tfreqmininpts,freqmaxinpts:", self.freqmininpts, self.freqmaxinpts)

    def getaxisinfo(self) -> tuple[float, float, float, int]:
        """
        Get frequency axis information for the object.

        Returns
        -------
        tuple[float, float, float, int]
            A tuple containing:
            - Minimum frequency value (float)
            - Maximum frequency value (float)
            - Frequency step size (float)
            - Number of frequency points (int)

        Notes
        -----
        This method extracts key frequency axis parameters from the object's frequency array.
        The frequency axis is assumed to be linearly spaced, and the returned values represent
        the range and spacing of the frequency data.

        Examples
        --------
        >>> info = obj.getaxisinfo()
        >>> print(f"Frequency range: {info[0]} to {info[1]} Hz")
        >>> print(f"Frequency step: {info[2]} Hz")
        >>> print(f"Number of points: {info[3]}")
        """
        return (
            self.freqaxis[self.freqmininpts],
            self.freqaxis[self.freqmaxinpts],
            self.freqaxis[1] - self.freqaxis[0],
            self.freqmaxinpts - self.freqmininpts,
        )

    def setreftc(self, reftc: NDArray) -> None:
        """
        Set reference time series and compute coherence statistics.

        This method assigns the reference time series, processes it through the preprocessing
        pipeline, and computes coherence statistics between the reference signal and itself.
        The method also sets up frequency axis parameters and validates the data.

        Parameters
        ----------
        reftc : NDArray
            Reference time series data to be processed. The array will be copied and converted
            to float type to ensure proper numerical operations.

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return any value.

        Notes
        -----
        The method performs the following operations:
        1. Assigns the input array to `self.reftc` with explicit float conversion
        2. Preprocesses the reference time series using `self.preptc()` method
        3. Computes coherence between the preprocessed reference signal and itself
        4. Sets up frequency axis and coherence data attributes
        5. Validates frequency limits and converts them to array indices
        6. Updates data validity flags

        Examples
        --------
        >>> # Assuming 'obj' is an instance of the class containing this method
        >>> reference_data = np.array([1.0, 2.0, 3.0, 4.0])
        >>> obj.setreftc(reference_data)
        >>> print(obj.freqaxis)
        >>> print(obj.thecoherence)
        """
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc)

        # get frequency axis, etc
        self.freqaxis, self.thecoherence = sp.signal.coherence(
            self.prepreftc, self.prepreftc, fs=self.Fs
        )
        #                                                       window=self.windowfunc)'''
        self.similarityfunclen = len(self.thecoherence)
        self.similarityfuncorigin = 0
        self.freqaxisvalid = True
        self.datavalid = False
        if self.freqmin is None or self.freqmax is None:
            self.setlimits(self.freqaxis[0], self.freqaxis[-1])
        self.freqmininpts = tide_util.valtoindex(
            self.freqaxis, self.freqmin, discretization="floor", debug=self.debug
        )
        self.freqmaxinpts = tide_util.valtoindex(
            self.freqaxis, self.freqmax, discretization="ceiling", debug=self.debug
        )

    def trim(self, vector: NDArray) -> NDArray:
        return vector[self.freqmininpts : self.freqmaxinpts]

    def run(
        self, thetc: NDArray, trim: bool = True, alt: bool = False
    ) -> tuple[NDArray, NDArray, int] | tuple[NDArray, NDArray, int, NDArray, NDArray, NDArray]:
        """
        Trim vector to specified frequency range indices.

        Parameters
        ----------
        vector : NDArray
            Input vector to be trimmed.

        Returns
        -------
        NDArray
            Trimmed vector containing elements from `self.freqmininpts` to `self.freqmaxinpts`.

        Notes
        -----
        This function uses array slicing to extract a portion of the input vector based on
        the frequency range boundaries defined by `self.freqmininpts` and `self.freqmaxinpts`.

        Examples
        --------
        >>> trimmed_data = obj.trim(data_vector)
        >>> print(trimmed_data.shape)
        (freqmaxinpts - freqmininpts,)
        """
        if len(thetc) != len(self.reftc):
            print(
                "Coherer: timecourses are of different sizes:",
                len(thetc),
                "!=",
                len(self.reftc),
                "- exiting",
            )
            sys.exit()

        self.testtc = thetc + 0.0
        self.preptesttc = self.preptc(self.testtc)

        # now actually do the coherence
        if self.debug:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(self.prepreftc, "r")
            plt.plot(self.preptesttc, "b")
            plt.legend(["reference", "test timecourse"])
            plt.show()

        if not alt:
            self.freqaxis, self.thecoherence = sp.signal.coherence(
                self.prepreftc, self.preptesttc, fs=self.Fs
            )
        else:
            self.freqaxis, self.thecsdxy = sp.signal.csd(
                10000.0 * self.prepreftc,
                10000.0 * self.preptesttc,
                fs=self.Fs,
                scaling="spectrum",
            )
            self.freqaxis, self.thecsdxx = sp.signal.csd(
                10000.0 * self.prepreftc,
                10000.0 * self.prepreftc,
                fs=self.Fs,
                scaling="spectrum",
            )
            self.freqaxis, self.thecsdyy = sp.signal.csd(
                10000.0 * self.preptesttc,
                10000.0 * self.preptesttc,
                fs=self.Fs,
                scaling="spectrum",
            )
            self.thecoherence = np.nan_to_num(
                abs(self.thecsdxy) ** 2 / (abs(self.thecsdxx) * abs(self.thecsdyy))
            )

        self.similarityfunclen = len(self.thecoherence)
        self.similarityfuncorigin = 0
        self.datavalid = True

        if trim:
            if alt:
                self.themax = np.argmax(self.thecoherence[self.freqmininpts : self.freqmaxinpts])
                return (
                    self.trim(self.thecoherence),
                    self.trim(self.freqaxis),
                    self.themax,
                    self.trim(self.thecsdxx),
                    self.trim(self.thecsdyy),
                    self.trim(self.thecsdxy),
                )
            else:
                self.themax = np.argmax(self.thecoherence[self.freqmininpts : self.freqmaxinpts])
                return (
                    self.trim(self.thecoherence),
                    self.trim(self.freqaxis),
                    self.themax,
                )
        else:
            if alt:
                self.themax = np.argmax(self.thecoherence)
                return (
                    self.thecoherence,
                    self.freqaxis,
                    self.themax,
                    self.thecsdxx,
                    self.thecsdyy,
                    self.thecsdxy,
                )

            else:
                self.themax = np.argmax(self.thecoherence)
                return self.thecoherence, self.freqaxis, self.themax
