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
import copy
import os
import sys
from typing import Any, Callable

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.Colortables import *
from rapidtide.stats import neglogpfromr_interpolator

atlases = {
    "ASPECTS": {"atlasname": "ASPECTS"},
    "ATT": {"atlasname": "ATTbasedFlowTerritories_split"},
    "JHU1": {"atlasname": "JHU-ArterialTerritoriesNoVent-LVL1"},
    "JHU2": {"atlasname": "JHU-ArterialTerritoriesNoVent-LVL2"},
}


def check_rt_spatialmatch(dataset1: Any, dataset2: Any) -> tuple[bool, bool, bool, bool]:
    """
    Check spatial matching between two datasets for dimension, size, space, and affine properties.

    This function compares four key spatial attributes between two datasets to determine
    if they are spatially compatible for further processing or analysis.

    Parameters
    ----------
    dataset1 : Any
        First dataset object containing spatial attributes xdim, ydim, zdim, xsize, ysize,
        zsize, space, and affine.
    dataset2 : Any
        Second dataset object containing the same spatial attributes as dataset1.

    Returns
    -------
    tuple[bool, bool, bool, bool]
        A tuple of four boolean values representing:
        - dimmatch: True if all spatial dimensions (xdim, ydim, zdim) match
        - sizematch: True if all spatial sizes (xsize, ysize, zsize) match
        - spacematch: True if spatial coordinate systems (space) match
        - affinematch: True if affine transformation matrices (affine) match

    Notes
    -----
    The function performs element-wise comparison of spatial attributes. For affine matrices,
    the comparison uses Python's default equality operator, which may be sensitive to
    floating-point precision differences.

    Examples
    --------
    >>> dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(dataset1, dataset2)
    >>> if dimmatch and sizematch:
    ...     print("Datasets have matching dimensions and sizes")
    """
    if (
        (dataset1.xdim == dataset2.xdim)
        and (dataset1.ydim == dataset2.ydim)
        and (dataset1.zdim == dataset2.zdim)
    ):
        dimmatch = True
    else:
        dimmatch = False
    if (
        (dataset1.xsize == dataset2.xsize)
        and (dataset1.ysize == dataset2.ysize)
        and (dataset1.zsize == dataset2.zsize)
    ):
        sizematch = True
    else:
        sizematch = False
    if dataset1.space == dataset2.space:
        spacematch = True
    else:
        spacematch = False
    if dataset1.affine == dataset2.affine:
        affinematch = True
    else:
        affinematch = False
    return dimmatch, sizematch, spacematch, affinematch


class Timecourse:
    "Store a timecourse and some information about it"

    def __init__(
        self,
        name: str,
        filename: str,
        namebase: str,
        samplerate: float,
        displaysamplerate: float,
        starttime: float = 0.0,
        label: str | None = None,
        report: bool = False,
        isbids: bool = False,
        limits: tuple[float, float] | None = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialize a Timecourse object for reading and managing time series data.

        This constructor sets up the basic properties of a timecourse and reads the
        time series data from the specified file.

        Parameters
        ----------
        name : str
            The name of the timecourse, used for identification
        filename : str
            The full path to the data file containing the time series
        namebase : str
            The base name of the file (without path)
        samplerate : float
            The sampling rate of the time series data in Hz
        displaysamplerate : float
            The sampling rate used for display purposes in Hz
        starttime : float, default=0.0
            The start time of the time series in seconds
        label : str, optional
            The label to use for display. If None, defaults to the name parameter
        report : bool, default=False
            Whether to generate a report during initialization
        isbids : bool, default=False
            Whether the data follows BIDS (Brain Imaging Data Structure) format
        limits : tuple of float, optional
            The (min, max) value limits for the time series data
        verbose : int, default=0
            Verbosity level for logging output (0 = quiet, higher values = more verbose)

        Returns
        -------
        None
            This method initializes the object and does not return any value

        Notes
        -----
        The method automatically reads the time series data using the readTimeData method
        after setting all the internal attributes. If verbose level is greater than 1,
        a message is printed indicating the file being read.

        Examples
        --------
        >>> tc = Timecourse(
        ...     name="ECG",
        ...     filename="/data/ecg_data.dat",
        ...     namebase="ecg_data",
        ...     samplerate=1000.0,
        ...     displaysamplerate=100.0,
        ...     starttime=0.0,
        ...     label="Electrocardiogram"
        ... )
        """
        self.verbose = verbose
        self.name = name
        self.filename = filename
        self.namebase = namebase
        self.samplerate = samplerate
        self.displaysamplerate = displaysamplerate
        self.starttime = starttime
        self.isbids = isbids
        self.limits = limits

        if label is None:
            self.label = name
        else:
            self.label = label
        self.report = report
        if self.verbose > 1:
            print("reading Timecourse ", self.name, " from ", self.filename, "...")
        self.readTimeData(self.label)

    def readTimeData(self, thename: str) -> None:
        """
        Read time series data from a file and compute associated statistics and spectral properties.

        This function reads time series data either from a BIDS-compatible TSV file or a vector file,
        depending on whether the object is configured to use BIDS format. It computes the time axis,
        spectral data, and statistical measures (kurtosis and skewness) for the selected column or
        the entire dataset.

        Parameters
        ----------
        thename : str
            The name of the column to extract from the BIDS TSV file. Ignored if not using BIDS format.

        Returns
        -------
        None
            This function does not return a value but updates the following attributes of the object:
            - `timedata`: The time series data as a numpy array.
            - `length`: The length of the time series.
            - `timeaxis`: The time axis corresponding to the data.
            - `specaxis`: The frequency axis for the power spectrum.
            - `specdata`: The power spectrum of the time series.
            - `kurtosis`, `kurtosis_z`, `kurtosis_p`: Kurtosis statistics.
            - `skewness`, `skewness_z`, `skewness_p`: Skewness statistics.

        Notes
        -----
        If the `thename` column is not found in a BIDS TSV file, the function sets `timedata` to `None`
        and returns early. The function uses `tide_io.readbidstsv` for BIDS files and `tide_io.readvec`
        for non-BIDS files. Spectral analysis and statistical computations are performed using
        `tide_filt.spectrum`, `tide_math.corrnormalize`, and `tide_stats.kurtosisstats`/`skewnessstats`.

        Examples
        --------
        >>> obj.readTimeData('signal')
        >>> print(obj.timedata)
        >>> print(obj.specdata)
        """
        if self.isbids:
            dummy, dummy, columns, indata, dummy, dummy = tide_io.readbidstsv(self.filename)
            try:
                self.timedata = indata[columns.index(thename), :]
            except ValueError:
                if self.verbose > 1:
                    print("no column named", thename, "in", columns)
                self.timedata = None
                return
        else:
            self.timedata = tide_io.readvec(self.filename)
        self.length = len(self.timedata)
        self.timeaxis = (
            np.linspace(0.0, self.length, num=self.length, endpoint=False) / self.samplerate
        ) - self.starttime
        if self.limits is not None:
            startpoint = np.max((int(np.round(self.limits[0] * self.samplerate, 0)), 0))
            endpoint = np.min((int(np.round(self.limits[1] * self.samplerate, 0)), self.length))
        else:
            startpoint = 0
            endpoint = self.length
        self.specaxis, self.specdata = tide_filt.spectrum(
            tide_math.corrnormalize(self.timedata[startpoint:endpoint]), self.samplerate
        )
        self.kurtosis, self.kurtosis_z, self.kurtosis_p = tide_stats.kurtosisstats(
            self.timedata[startpoint:endpoint]
        )
        self.skewness, self.skewness_z, self.skewness_p = tide_stats.skewnessstats(
            self.timedata[startpoint:endpoint]
        )

        if self.verbose > 1:
            print("Timecourse data range:", np.min(self.timedata), np.max(self.timedata))
            print("sample rate:", self.samplerate)
            print("Timecourse length:", self.length)
            print("timeaxis length:", len(self.timeaxis))
            print("kurtosis:", self.kurtosis)
            print("kurtosis_z:", self.kurtosis_z)
            print("kurtosis_p:", self.kurtosis_p)

            print()

    def summarize(self) -> None:
        """
        Print a summary of the timecourse properties.

        This method outputs a formatted summary of various properties associated with
        the timecourse object, including name, label, file information, sampling rate,
        length, and statistical measures.

        Parameters
        ----------
        self : object
            The timecourse object instance containing the properties to be summarized.
            Expected attributes include:
            - name: str, the name of the timecourse
            - label: str, the label associated with the timecourse
            - filename: str, the filename of the timecourse data
            - namebase: str, the base name of the file
            - samplerate: float, the sampling rate of the timecourse
            - length: int, the length of the timecourse
            - kurtosis: float, the kurtosis value of the timecourse
            - kurtosis_z: float, the z-score of the kurtosis
            - kurtosis_p: float, the p-value of the kurtosis

        Returns
        -------
        None
            This method prints the summary information to stdout and does not return any value.

        Notes
        -----
        The output is formatted for readability with consistent indentation and
        descriptive labels for each property. This method is typically used for
        debugging and quick inspection of timecourse properties.

        Examples
        --------
        >>> timecourse = Timecourse(name="test_signal", label="test", filename="test.csv")
        >>> timecourse.summarize()
        Timecourse name:       test_signal
            label:            test
            filename:         test.csv
            namebase:         test
            samplerate:       100.0
            length:           1000
            kurtosis:         0.5
            kurtosis_z:       1.2
            kurtosis_p:       0.23
        """
        print()
        print("Timecourse name:      ", self.name)
        print("    label:            ", self.label)
        print("    filename:         ", self.filename)
        print("    namebase:         ", self.namebase)
        print("    samplerate:       ", self.samplerate)
        print("    length:           ", self.length)
        print("    kurtosis:         ", self.kurtosis)
        print("    kurtosis_z:       ", self.kurtosis_z)
        print("    kurtosis_p:       ", self.kurtosis_p)


class Overlay:
    "Store a data overlay and some information about it"

    LUTname = None

    def __init__(
        self,
        name: str,
        filespec: str,
        namebase: str,
        funcmask: NDArray | None = None,
        geommask: NDArray | None = None,
        label: str | None = None,
        report: bool = False,
        lut_state: dict = gen_gray_state(),
        alpha: int = 128,
        endalpha: int = 0,
        display_state: bool = True,
        invertonload: bool = False,
        isaMask: bool = False,
        init_LUT: bool = True,
        verbose: int = 1,
    ) -> None:
        """
        Initialize an overlay object for rapidtide image data visualization.

        This constructor initializes an overlay by loading image data from a file,
        applying functional and geometric masks, and setting up display properties
        including lookup tables and affine transformations.

        Parameters
        ----------
        name : str
            Name of the overlay.
        filespec : str
            File specification string used to locate and load the image data.
        namebase : str
            Base name for the overlay, used in file naming and labeling.
        funcmask : NDArray | None, optional
            Functional mask to apply to the data. Default is None.
        geommask : NDArray | None, optional
            Geometric mask to apply to the data. Default is None.
        label : str | None, optional
            Label for the overlay. If None, defaults to the value of `name`. Default is None.
        report : bool, optional
            If True, enables reporting mode. Default is False.
        lut_state : dict, optional
            Lookup table state dictionary. Default is ``gen_gray_state()``.
        alpha : int, optional
            Initial alpha value for display. Default is 128.
        endalpha : int, optional
            End alpha value for display. Default is 0.
        display_state : bool, optional
            If True, initializes display state. Default is True.
        invertonload : bool, optional
            If True, inverts the data on load. Default is False.
        isaMask : bool, optional
            If True, treats the data as a mask. Default is False.
        init_LUT : bool, optional
            If True, initializes the lookup table. Default is True.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        None
            This method initializes the object in-place and does not return a value.

        Notes
        -----
        The constructor performs the following steps:
        1. Loads image data using `tide_io.processnamespec`.
        2. Applies functional and geometric masks.
        3. Sets up display parameters such as `dispmin` and `dispmax`.
        4. Initializes lookup table if `init_LUT` is True.
        5. Determines the spatial coordinate system and affine transformation matrix.
        6. Determines the orientation (neurological or radiological) based on the affine matrix.

        Examples
        --------
        >>> overlay = Overlay(
        ...     name="my_overlay",
        ...     filespec="/path/to/image.nii",
        ...     namebase="overlay_base",
        ...     verbose=2
        ... )
        """
        self.verbose = verbose
        self.name = name
        if label is None:
            self.label = name
        else:
            self.label = label
        self.report = report
        self.filename, self.filevals = tide_io.processnamespec(
            filespec, "Including voxels where ", "in mask"
        )
        self.namebase = namebase
        if self.verbose > 1:
            print("reading map ", self.name, " from ", self.filename, "...")
        self.maskhash = 0
        self.invertonload = invertonload
        self.readImageData(isaMask=isaMask)
        self.mask = None
        self.maskeddata = None
        self.setFuncMask(funcmask, maskdata=False)
        self.setGeomMask(geommask, maskdata=False)
        self.maskData()
        self.updateStats()
        self.dispmin = self.robustmin
        self.dispmax = self.robustmax
        if init_LUT:
            self.gradient = getagradient()
            self.lut_state = lut_state
            self.display_state = display_state
            self.theLUT = None
            self.alpha = alpha
            self.endalpha = endalpha
            self.setLUT(self.lut_state, alpha=self.alpha, endalpha=self.endalpha)
        self.space = "unspecified"
        if (self.header["sform_code"] == 4) or (self.header["qform_code"] == 4):
            if ((self.xdim == 61) and (self.ydim == 73) and (self.zdim == 61)) or (
                (self.xdim == 91) and (self.ydim == 109) and (self.zdim == 91)
            ):
                self.space = "MNI152"
            else:
                self.space = "MNI152NLin2009cAsym"
        if self.header["sform_code"] != 0:
            self.affine = self.header.get_sform()
        elif self.header["qform_code"] != 0:
            self.affine = self.header.get_qform()
        else:
            self.affine = self.header.get_base_affine()
        self.invaffine = np.linalg.inv(self.affine)
        if self.verbose > 1:
            print("affine matrix:")
            print(self.affine)
        if self.affine[0][0] < 0.0:
            self.RLfactor = -1.0
            if self.verbose > 1:
                print("Overlay appears to be in neurological orientation")
        elif self.affine[0][0] > 0.0:
            self.RLfactor = 1.0
            if self.verbose > 1:
                print("Overlay appears to be in radiological orientation")
        else:
            self.RLfactor = 0.0
            if self.verbose > 1:
                print("Overlay has indeterminate orientation")
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.tpos = 0
        self.xcoord = 0.0
        self.ycoord = 0.0
        self.zcoord = 0.0
        self.tcoord = 0.0

        if self.verbose > 1:
            print(
                "Overlay initialized:",
                self.name,
                self.filename,
                self.minval,
                self.dispmin,
                self.dispmax,
                self.maxval,
            )
        if self.verbose > 0:
            self.summarize()

    def duplicate(self, newname: str, newlabel: str, init_LUT: bool = True) -> Any:
        """
        Create a duplicate of the current overlay with new name and label.

        Parameters
        ----------
        newname : str
            The name for the new overlay instance.
        newlabel : str
            The label for the new overlay instance.
        init_LUT : bool, optional
            Whether to initialize the lookup table for the new overlay.
            Default is True.

        Returns
        -------
        Any
            A new Overlay instance with the specified name and label,
            inheriting all properties from the current overlay.

        Notes
        -----
        This method creates a shallow copy of the current overlay with
        updated name and label attributes. The new overlay maintains
        references to the original data files and masks.

        Examples
        --------
        >>> overlay = Overlay("original", "file.nii", "base")
        >>> new_overlay = overlay.duplicate("copy", "Copy Label")
        >>> print(new_overlay.name)
        'copy'
        >>> print(new_overlay.label)
        'Copy Label'
        """
        return Overlay(
            newname,
            self.filename,
            self.namebase,
            funcmask=self.funcmask,
            geommask=self.geommask,
            label=newlabel,
            report=self.report,
            init_LUT=init_LUT,
            verbose=self.verbose,
        )

    def updateStats(self) -> None:
        """
        Update statistical properties of the masked data.

        This method calculates various statistical measures from the masked data,
        including min/max values, quartiles, robust statistics, and histogram data.
        The results are stored as instance attributes for later use.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method does not return a value but updates instance attributes
            with statistical calculations.

        Notes
        -----
        The method uses the mask to filter data points where mask != 0.
        Statistical calculations include:
        - Minimum and maximum values
        - Robust statistics (0.02, 0.25, 0.5, 0.75, 0.98 percentiles)
        - Histogram data with 200 bins
        - Quartiles (25th, 50th, 75th percentiles)

        Examples
        --------
        >>> obj.updateStats()
        >>> print(obj.minval, obj.maxval)
        >>> print(obj.quartiles)
        """
        calcmaskeddata = self.data[np.where(self.mask != 0)]

        self.minval = calcmaskeddata.min()
        self.maxval = calcmaskeddata.max()
        (
            self.robustmin,
            self.pct25,
            self.pct50,
            self.pct75,
            self.robustmax,
        ) = tide_stats.getfracvals(calcmaskeddata, [0.02, 0.25, 0.5, 0.75, 0.98], nozero=False)
        self.histy, self.histx = np.histogram(
            calcmaskeddata, bins=np.linspace(self.minval, self.maxval, 200)
        )
        self.quartiles = [self.pct25, self.pct50, self.pct75]

        if self.verbose > 1:
            print(
                self.name,
                ":",
                self.minval,
                self.maxval,
                self.robustmin,
                self.robustmax,
                self.quartiles,
            )

    def setData(self, data: NDArray, isaMask: bool = False) -> None:
        """
        Set the data array and optionally convert it to a binary mask.

        This method assigns the provided data array to the internal data attribute
        and optionally converts it to a binary mask where values less than 0.5
        are set to 0.0 and values greater than 0.5 are set to 1.0.

        Parameters
        ----------
        data : NDArray
            The input data array to be set. A copy of this array is stored internally.
        isaMask : bool, optional
            If True, converts the data to a binary mask. Values less than 0.5 become 0.0,
            and values greater than 0.5 become 1.0. Default is False.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The data is stored as a copy to prevent external modifications from affecting
        the internal state. The mask conversion is performed using numpy's where function
        for efficient element-wise operations.

        Examples
        --------
        >>> obj.setData(np.array([0.2, 0.7, 0.3, 0.8]))
        >>> obj.setData(np.array([0.2, 0.7, 0.3, 0.8]), isaMask=True)
        """
        self.data = data.copy()
        if isaMask:
            self.data[np.where(self.data < 0.5)] = 0.0
            self.data[np.where(self.data > 0.5)] = 1.0
        self.updateStats()

    def readImageData(self, isaMask: bool = False) -> None:
        """
        Read image data from a NIfTI file and process it based on specified flags.

        This function loads image data from a NIfTI file using `tide_io.readfromnifti`,
        applies optional inversion and masking operations, and extracts dimension and
        spacing information from the header.

        Parameters
        ----------
        isaMask : bool, optional
            If True, process the data as a binary mask. For non-mask files, values
            less than 0.5 are set to 0, and values greater than 0.5 are set to 1.
            For mask files with `filevals` defined, the data is converted to a binary
            mask based on matching values in `filevals`. Default is False.

        Returns
        -------
        None
            This function does not return a value but updates the instance attributes
            `nim`, `data`, `header`, `dims`, `sizes`, `xdim`, `ydim`, `zdim`, `tdim`,
            `xsize`, `ysize`, `zsize`, `tr`, and `toffset`.

        Notes
        -----
        - If `invertonload` is True, the data is multiplied by -1.0.
        - The function prints data range and header information if `verbose > 1`.
        - Dimension and spacing information is parsed using `tide_io.parseniftidims` and
          `tide_io.parseniftisizes`.

        Examples
        --------
        >>> reader = ImageReader()
        >>> reader.filename = "example.nii"
        >>> reader.invertonload = True
        >>> reader.verbose = 2
        >>> reader.readImageData(isaMask=True)
        Overlay data range: -1.0 1.0
        header {'toffset': 0.0, ...}
        Overlay dims: 64 64 32 1
        Overlay sizes: 3.0 3.0 3.0 2.0
        Overlay toffset: 0.0
        """
        self.nim, self.data, self.header, self.dims, self.sizes = tide_io.readfromnifti(
            self.filename
        )
        if self.invertonload:
            self.data *= -1.0
        if isaMask:
            if self.filevals is None:
                self.data[np.where(self.data < 0.5)] = 0.0
                self.data[np.where(self.data > 0.5)] = 1.0
            else:
                tempmask = (0 * self.data).astype("uint16")
                for theval in self.filevals:
                    tempmask[np.where(self.data - theval == 0.0)] += 1
                self.data = np.where(tempmask > 0, 1, 0)
        if self.verbose > 1:
            print("Overlay data range:", np.min(self.data), np.max(self.data))
            print("header", self.header)
        self.xdim, self.ydim, self.zdim, self.tdim = tide_io.parseniftidims(self.dims)
        self.xsize, self.ysize, self.zsize, self.tr = tide_io.parseniftisizes(self.sizes)
        self.toffset = self.header["toffset"]
        if self.verbose > 1:
            print("Overlay dims:", self.xdim, self.ydim, self.zdim, self.tdim)
            print("Overlay sizes:", self.xsize, self.ysize, self.zsize, self.tr)
            print("Overlay toffset:", self.toffset)

    def setLabel(self, label: str) -> None:
        """
        Set the label for the object.

        Parameters
        ----------
        label : str
            The label to assign to the object.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided label to the object's label attribute.
        The label is typically used for identification or display purposes.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setLabel("New Label")
        >>> print(obj.label)
        'New Label'
        """
        self.label = label

    def real2tr(self, time: float) -> float:
        """
        Convert real time to trigger time.

        Convert a real time value to its corresponding trigger time value by applying
        the time offset and trigger period parameters.

        Parameters
        ----------
        time : float
            The real time value to be converted to trigger time.

        Returns
        -------
        float
            The converted trigger time value, rounded to the nearest integer.

        Notes
        -----
        The conversion is performed using the formula: ``round((time - toffset) / tr)``
        where ``toffset`` is the time offset and ``tr`` is the trigger period.

        Examples
        --------
        >>> obj.real2tr(10.5)
        2.0
        >>> obj.real2tr(5.0)
        0.0
        """
        return np.round((time - self.toffset) / self.tr, 0)

    def tr2real(self, tpos: int) -> float:
        return self.toffset + self.tr * tpos

    def real2vox(
        self, xcoord: float, ycoord: float, zcoord: float, time: float
    ) -> tuple[int, int, int, int]:
        """
        Convert a time position to a real time value.

        Parameters
        ----------
        tpos : int
            Time position index to convert to real time value.

        Returns
        -------
        float
            The corresponding real time value calculated as `toffset + tr * tpos`.

        Notes
        -----
        This function performs a linear transformation from discrete time positions
        to continuous time values using the instance's time offset and time rate
        parameters.

        Examples
        --------
        >>> obj.tr2real(0)
        10.0
        >>> obj.tr2real(5)
        15.0
        """
        x, y, z = nib.apply_affine(self.invaffine, [xcoord, ycoord, zcoord])
        t = self.real2tr(time)
        return (
            int(np.round(x, 0)),
            int(np.round(y, 0)),
            int(np.round(z, 0)),
            int(np.round(t, 0)),
        )

    def vox2real(self, xpos: int, ypos: int, zpos: int, tpos: int) -> NDArray:
        """
        Convert voxel coordinates to real-world coordinates.

        This function transforms voxel coordinates (x, y, z, t) to real-world coordinates
        using the affine transformation matrix and temporal transformation.

        Parameters
        ----------
        xpos : int
            X coordinate in voxel space
        ypos : int
            Y coordinate in voxel space
        zpos : int
            Z coordinate in voxel space
        tpos : int
            T coordinate in voxel space (temporal dimension)

        Returns
        -------
        NDArray
            Array containing real-world coordinates [x_real, y_real, z_real, t_real]

        Notes
        -----
        The conversion uses nibabel's apply_affine function for spatial transformation
        and self.tr2real for temporal transformation. The result includes both
        spatial and temporal coordinates in real-world units.

        Examples
        --------
        >>> vox2real(10, 20, 30, 5)
        array([12.5, 25.0, 37.5, 2.5])
        """
        return np.concatenate(
            (nib.apply_affine(self.affine, [xpos, ypos, zpos]), [self.tr2real(tpos)]),
            axis=0,
        )

    def setXYZpos(self, xpos: int, ypos: int, zpos: int) -> None:
        """
        Set the 3D position coordinates of the object.

        Parameters
        ----------
        xpos : int
            The x-coordinate position value.
        ypos : int
            The y-coordinate position value.
        zpos : int
            The z-coordinate position value.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        All position values are converted to integers before assignment.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setXYZpos(10, 20, 30)
        >>> print(obj.xpos, obj.ypos, obj.zpos)
        10 20 30
        """
        self.xpos = int(xpos)
        self.ypos = int(ypos)
        self.zpos = int(zpos)

    def setTpos(self, tpos: int) -> None:
        """
        Set the temporal position attribute with bounds checking.

        Parameters
        ----------
        tpos : int
            The temporal position to set. If greater than self.tdim - 1,
            it will be clamped to self.tdim - 1.

        Returns
        -------
        None
            This method modifies the instance in-place and does not return a value.

        Notes
        -----
        The temporal position is bounded by the dimensionality of the temporal
        space (self.tdim). If the input tpos exceeds this limit, it will be
        automatically adjusted to the maximum valid position (self.tdim - 1).

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.tdim = 5
        >>> obj.setTpos(3)
        >>> obj.tpos
        3
        >>> obj.setTpos(10)
        >>> obj.tpos
        4
        """
        if tpos > self.tdim - 1:
            self.tpos = int(self.tdim - 1)
        else:
            self.tpos = int(tpos)

    def getFocusVal(self) -> float:
        """
        Get the focus value at the current position.

        This method retrieves the data value from the masked data array at the
        current position coordinates. The method handles both 3D and 4D data arrays
        by checking the time dimension.

        Parameters
        ----------
        self : object
            The instance of the class containing the masked data and position
            coordinates.

        Returns
        -------
        float
            The data value at the current position. For 4D data (tdim > 1), returns
            the value at [xpos, ypos, zpos, tpos]. For 3D data (tdim <= 1), returns
            the value at [xpos, ypos, zpos].

        Notes
        -----
        The method assumes that the instance has the following attributes:
        - maskeddata: numpy array containing the data
        - xpos, ypos, zpos, tpos: integer coordinates
        - tdim: integer representing the time dimension

        Examples
        --------
        >>> value = obj.getFocusVal()
        >>> print(value)
        0.5
        """
        if self.tdim > 1:
            return self.maskeddata[self.xpos, self.ypos, self.zpos, self.tpos]
        else:
            return self.maskeddata[self.xpos, self.ypos, self.zpos]

    def setFuncMask(self, funcmask: NDArray | None, maskdata: bool = True) -> None:
        """
        Set the functional mask for the object.

        Parameters
        ----------
        funcmask : array-like, optional
            The functional mask to be set. If None, a default mask is created based on
            the dimensionality of the data. If provided, a copy of the mask is stored.
        maskdata : bool, default=True
            If True, calls maskData() method after setting the functional mask.

        Returns
        -------
        None
            This method modifies the object in-place and does not return any value.

        Notes
        -----
        When funcmask is None, the method creates a default mask:
        - For 1D data (tdim == 1): creates a mask with the same shape as self.data
        - For higher dimensional data: creates a mask with shape (self.data.shape[0], self.data.shape[1], self.data.shape[2], 1)

        Examples
        --------
        >>> obj.setFuncMask(None)
        >>> obj.setFuncMask(np.ones((10, 10)))
        >>> obj.setFuncMask(None, maskdata=False)
        """
        self.funcmask = funcmask
        if self.funcmask is None:
            if self.tdim == 1:
                self.funcmask = 1.0 + 0.0 * self.data
            else:
                self.funcmask = 1.0 + 0.0 * self.data[:, :, :, 0]
        else:
            self.funcmask = funcmask.copy()
        if maskdata:
            self.maskData()

    def setGeomMask(self, geommask: NDArray | None, maskdata: bool = True) -> None:
        """
        Set the geometric mask for the object and optionally mask the data.

        Parameters
        ----------
        geommask : ndarray or None
            Geometric mask array. If None, a default mask is created based on the
            object's dimensions. If not None, the provided mask is copied and used.
        maskdata : bool, default=True
            If True, applies the mask to the data by calling maskData() method.
            If False, only sets the geometric mask without masking the data.

        Returns
        -------
        None
            This method modifies the object in-place and does not return any value.

        Notes
        -----
        When geommask is None and tdim == 1, a mask of ones with the same shape as
        self.data is created. Otherwise, a mask of ones with the shape of
        self.data[:, :, :, 0] is created.

        Examples
        --------
        >>> obj.setGeomMask(None)
        >>> obj.setGeomMask(mask_array, maskdata=False)
        """
        self.geommask = geommask
        if self.geommask is None:
            if self.tdim == 1:
                self.geommask = 1.0 + 0.0 * self.data
            else:
                self.geommask = 1.0 + 0.0 * self.data[:, :, :, 0]
        else:
            self.geommask = geommask.copy()
        if maskdata:
            self.maskData()

    def maskData(self) -> None:
        """
        Apply mask to data and update statistics if mask has changed.

        This method combines geometric and functional masks to create a final mask,
        then checks if the mask has changed since the last update. If the mask has
        changed, it applies the mask to the data, sets masked values to zero, and
        updates the statistics.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method modifies the instance in-place and does not return any value.

        Notes
        -----
        The method uses a hash-based approach to efficiently detect when the mask
        has changed, avoiding expensive operations when the mask remains the same.
        The mask is applied by setting values to zero where the combined mask is
        less than 0.5.

        Examples
        --------
        >>> obj.maskData()
        # Applies mask and updates statistics if mask has changed
        """
        self.mask = self.geommask * self.funcmask
        maskhash = hash(self.mask.tobytes())
        # these operations are expensive, so only do them if the mask is changed
        if (maskhash == self.maskhash) and (self.verbose > 1):
            print("mask has not changed")
        else:
            if self.verbose > 1:
                print("mask changed - recalculating")
            self.maskeddata = self.data.copy()
            self.maskeddata[np.where(self.mask < 0.5)] = 0.0
            self.updateStats()
            self.maskhash = maskhash

    def setReport(self, report: bool) -> None:
        """
        Set the report flag for the object.

        Parameters
        ----------
        report : bool
            Flag indicating whether reporting is enabled or disabled.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided boolean value to the internal `report` attribute
        of the object. The attribute can be accessed later to check the current reporting state.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setReport(True)
        >>> obj.report
        True
        >>> obj.setReport(False)
        >>> obj.report
        False
        """
        self.report = report

    def setTR(self, trval: float) -> None:
        """
        Set the TR (repetition time) value for the object.

        Parameters
        ----------
        trval : float
            The repetition time value to be set. This parameter represents the
            time interval between successive MRI pulse sequences in seconds.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided value to the internal `tr` attribute
        of the object. The TR value is commonly used in MRI data processing and
        represents the time between the start of one pulse sequence and the start
        of the next.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setTR(2.0)
        >>> print(obj.tr)
        2.0
        """
        self.tr = trval

    def settoffset(self, toffset: float) -> None:
        """
        Set the time offset value.

        Parameters
        ----------
        toffset : float
            The time offset value to set.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided time offset value to the instance variable
        `self.toffset`. The time offset is typically used to adjust timing references
        in time-series data processing or temporal calculations.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.settoffset(5.0)
        >>> print(obj.toffset)
        5.0
        """
        self.toffset = toffset

    def setLUT(self, lut_state: dict, alpha: int = 255, endalpha: int = 128) -> None:
        """
        Set the lookup table (LUT) state with optional alpha blending adjustments.

        This function configures the lookup table state for gradient visualization,
        applying alpha transparency adjustments to the color ticks and restoring
        the gradient state with the updated LUT.

        Parameters
        ----------
        lut_state : dict
            Dictionary containing the lookup table state with keys:
            - "ticks": list of tuples representing color stops
            - "mode": color mapping mode
            - "name": name of the LUT
        alpha : int, optional
            Alpha value (0-255) to apply to intermediate color ticks, by default 255
        endalpha : int, optional
            Alpha value (0-255) to apply to the end color ticks, by default 128

        Returns
        -------
        None
            This function modifies the instance state in-place and does not return anything.

        Notes
        -----
        The function applies alpha blending to intermediate color ticks while preserving
        the original alpha values of the first and last ticks. When verbose mode is enabled
        (verbose > 1), the modified tick values are printed to the console.

        Examples
        --------
        >>> lut_state = {
        ...     "ticks": [(0, (0, 0, 0, 255)), (128, (128, 128, 128, 255)), (255, (255, 255, 255, 255))],
        ...     "mode": "RGB",
        ...     "name": "grayscale"
        ... }
        >>> setLUT(lut_state, alpha=128, endalpha=64)
        """
        if alpha is not None:
            theticks = [lut_state["ticks"][0]]
            for theelement in lut_state["ticks"][1:-1]:
                theticks.append(
                    (
                        theelement[0],
                        (theelement[1][0], theelement[1][1], theelement[1][2], alpha),
                    )
                )
            theticks.append(lut_state["ticks"][-1])
            if self.verbose > 1:
                print("setLUT alpha adjustment:\n", theticks)
            self.lut_state = setendalpha({"ticks": theticks, "mode": lut_state["mode"]}, endalpha)
        else:
            self.lut_state = setendalpha(lut_state, endalpha)
        self.gradient.restoreState(self.lut_state)
        self.theLUT = self.gradient.getLookupTable(512, alpha=True)
        self.LUTname = lut_state["name"]

    def setisdisplayed(self, display_state: bool) -> None:
        """
        Set the display state of the object.

        Parameters
        ----------
        display_state : bool
            The display state to set. True indicates the object should be displayed,
            False indicates it should be hidden.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method directly assigns the provided display state to the internal
        `display_state` attribute of the object. The display state controls whether
        the object is visible in the user interface or not.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setisdisplayed(True)
        >>> print(obj.display_state)
        True
        >>> obj.setisdisplayed(False)
        >>> print(obj.display_state)
        False
        """
        self.display_state = display_state

    def summarize(self) -> None:
        """
        Print a summary of the overlay's properties and metadata.

        This method outputs a formatted summary of the overlay's key attributes,
        including name, label, file information, dimensions, orientation, and
        data statistics. It also indicates whether geometric and functional masks
        are set.

        Notes
        -----
        The output is printed directly to the console and does not return any value.
        This method is intended for debugging or inspection purposes.

        Examples
        --------
        >>> overlay = Overlay(...)
        >>> overlay.summarize()
        Overlay name:          my_overlay
            label:            My Overlay
            filename:         /path/to/my_overlay.nii
            namebase:         my_overlay
            xdim:             64
            ydim:             64
            zdim:             32
            tdim:             100
            space:            MNI
            orientation:       radiological
            toffset:          0.0
            tr:               2.0
            min:              -1.5
            max:              2.0
            robustmin:        -1.0
            robustmax:        1.5
            dispmin:          -1.5
            dispmax:          2.0
            data shape:       (64, 64, 32, 100)
            masked data shape: (64, 64, 32, 100)
            geometric mask is set
            functional mask not set
        """
        print()
        print("Overlay name:         ", self.name)
        print("    label:            ", self.label)
        print("    filename:         ", self.filename)
        print("    namebase:         ", self.namebase)
        print("    xdim:             ", self.xdim)
        print("    ydim:             ", self.ydim)
        print("    zdim:             ", self.zdim)
        print("    tdim:             ", self.tdim)
        print("    space:            ", self.space)
        if self.RLfactor < 0.0:
            print("    orientation:       neurological")
        elif self.RLfactor > 0.0:
            print("    orientation:       radiological")
        else:
            print("    orientation:       indeterminate")
        print("    toffset:          ", self.toffset)
        print("    tr:               ", self.tr)
        print("    min:              ", self.minval)
        print("    max:              ", self.maxval)
        print("    robustmin:        ", self.robustmin)
        print("    robustmax:        ", self.robustmax)
        print("    dispmin:          ", self.dispmin)
        print("    dispmax:          ", self.dispmax)
        print("    data shape:       ", np.shape(self.data))
        print("    masked data shape:", np.shape(self.maskeddata))
        if self.geommask is None:
            print("    geometric mask not set")
        else:
            print("    geometric mask is set")
        if self.funcmask is None:
            print("    functional mask not set")
        else:
            print("    functional mask is set")


class RapidtideDataset:
    "Store all the data associated with a rapidtide dataset"

    fileroot = None
    focusregressor = None
    regressorfilterlimits = None
    regressorsimcalclimits = None
    focusmap = None
    dispmaps = None
    allloadedmaps = None
    loadedfuncmasks = None
    loadedfuncmaps = None
    atlaslabels = None
    atlasname = None
    useatlas = False
    xdim = 0
    ydim = 0
    zdim = 0
    tdim = 0
    xsize = 0.0
    ysize = 0.0
    zsize = 0.0
    tr = 0.0
    space = None
    affine = None

    def __init__(
        self,
        name: str,
        fileroot: str,
        anatname: str | None = None,
        geommaskname: str | None = None,
        funcmaskname: str | None = None,
        graymaskspec: str | None = None,
        whitemaskspec: str | None = None,
        userise: bool = False,
        usecorrout: bool = False,
        useatlas: bool = False,
        minimal: bool = False,
        forcetr: bool = False,
        forceoffset: bool = False,
        coordinatespace: str = "unspecified",
        offsettime: float = 0.0,
        init_LUT: bool = True,
        verbose: int = 0,
    ) -> None:
        """
        Initialize a RapidtideDataset object for processing neuroimaging data.

        This constructor sets up the dataset configuration based on provided parameters,
        determines the naming convention used by the dataset (BIDS or legacy), and
        initializes internal structures for regressor and overlay handling.

        Parameters
        ----------
        name : str
            Name of the dataset.
        fileroot : str
            Root path to the dataset files.
        anatname : str, optional
            Path to the anatomical image file. Default is None.
        geommaskname : str, optional
            Path to the geometric mask file. Default is None.
        funcmaskname : str, optional
            Path to the functional mask file. Default is None.
        graymaskspec : str, optional
            Specification for gray matter mask. Default is None.
        whitemaskspec : str, optional
            Specification for white matter mask. Default is None.
        userise : bool, optional
            Whether to use RISE (reconstruction of instantaneous signal estimates). Default is False.
        usecorrout : bool, optional
            Whether to use corrected output. Default is False.
        useatlas : bool, optional
            Whether to use atlas-based processing. Default is False.
        minimal : bool, optional
            Whether to run in minimal mode. Default is False.
        forcetr : bool, optional
            Whether to force TR (repetition time) correction. Default is False.
        forceoffset : bool, optional
            Whether to force offset correction. Default is False.
        coordinatespace : str, optional
            Coordinate space of the data. Default is "unspecified".
        offsettime : float, optional
            Time offset to apply. Default is 0.0.
        init_LUT : bool, optional
            Whether to initialize lookup tables. Default is True.
        verbose : int, optional
            Verbosity level. Default is 0.

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Notes
        -----
        The function automatically detects whether the dataset uses BIDS-style naming
        conventions by checking for the presence of specific files like
        ``<fileroot>desc-maxtime_map.nii.gz``. If not found, it checks for legacy naming
        patterns such as ``<fileroot>fitmask.nii.gz``.

        Examples
        --------
        >>> dataset = RapidtideDataset(
        ...     name="test_dataset",
        ...     fileroot="/path/to/data",
        ...     anatname="/path/to/anat.nii.gz",
        ...     verbose=1
        ... )
        """
        self.verbose = verbose
        self.name = name
        self.fileroot = fileroot
        self.anatname = anatname
        self.geommaskname = geommaskname
        self.funcmaskname = funcmaskname
        self.graymaskspec = graymaskspec
        self.whitemaskspec = whitemaskspec
        self.userise = userise
        self.usecorrout = usecorrout
        self.useatlas = useatlas
        self.forcetr = forcetr
        self.forceoffset = forceoffset
        self.coordinatespace = coordinatespace
        self.offsettime = offsettime
        self.init_LUT = init_LUT
        self.referencedir = tide_util.findreferencedir()

        # check which naming style the dataset has
        if os.path.isfile(self.fileroot + "desc-maxtime_map.nii.gz"):
            self.bidsformat = True
            self.newstylenames = True
        else:
            self.bidsformat = False
            if os.path.isfile(self.fileroot + "fitmask.nii.gz"):
                self.newstylenames = True
            else:
                self.newstylenames = False
        if self.verbose > 1:
            print(
                "RapidtideDataset init: self.bidsformat=",
                self.bidsformat,
                "self.newstylenames=",
                self.newstylenames,
            )

        self.setupregressors()
        self.setupoverlays()

    def _loadregressors(self) -> None:
        """
        Load regressor timecourses from specified files.

        This method iterates through the list of regressor specifications (`self.regressorspecs`)
        and attempts to load each regressor from the corresponding file. If a file exists, it is
        read into a `Timecourse` object and stored in `self.regressors`. If no regressor is
        successfully loaded, the first one in the list is set as the focus regressor.

        Parameters
        ----------
        self : object
            The instance of the class containing the method. Expected to have the following
            attributes:
            - `regressorspecs`: list of tuples specifying regressor files and parameters.
            - `fileroot`: string, base path for regressor files.
            - `regressors`: dict, to store loaded regressors.
            - `focusregressor`: str, name of the currently focused regressor.
            - `verbose`: int, level of verbosity for logging.
            - `bidsformat`: bool, flag indicating BIDS format usage.
            - `regressorsimcalclimits`: tuple, limits for regressor calculation.
            - `isbids`: bool, flag indicating BIDS format usage (likely a typo for `bidsformat`).

        Returns
        -------
        None
            This method modifies the instance's attributes in place and does not return any value.

        Notes
        -----
        - If a regressor file does not exist and the corresponding flag in `regressorspecs` is True,
          a `FileNotFoundError` is raised.
        - If a regressor file does not exist and the flag is False, the file is skipped with a message.
        - The first successfully loaded regressor is set as the `focusregressor` if none is already set.

        Examples
        --------
        >>> loader = MyLoader()
        >>> loader._loadregressors()
        # Loads regressors specified in `loader.regressorspecs` into `loader.regressors`.
        """
        self.focusregressor = None
        for thisregressor in self.regressorspecs:
            if os.path.isfile(self.fileroot + thisregressor[2]):
                if self.verbose > 1:
                    print("file: ", self.fileroot + thisregressor[2], " exists - reading...")
                thepath, thebase = os.path.split(self.fileroot + thisregressor[2])
                theregressor = Timecourse(
                    thisregressor[0],
                    self.fileroot + thisregressor[2],
                    thebase,
                    thisregressor[3],
                    thisregressor[4],
                    label=thisregressor[1],
                    starttime=thisregressor[5],
                    isbids=self.bidsformat,
                    limits=self.regressorsimcalclimits,
                    verbose=self.verbose,
                )
                if theregressor.timedata is not None:
                    self.regressors[thisregressor[0]] = copy.deepcopy(theregressor)
                    if self.verbose > 0:
                        theregressor.summarize()
                if self.focusregressor is None:
                    self.focusregressor = thisregressor[0]
            else:
                if thisregressor[6]:
                    raise FileNotFoundError(
                        f"regressor file {self.fileroot + thisregressor[2]} does not exist"
                    )
                else:
                    if self.verbose > 1:
                        print(
                            "file: ",
                            self.fileroot + thisregressor[2],
                            " does not exist - skipping...",
                        )

    def _loadfuncmaps(self) -> None:
        """
        Load functional maps from NIfTI files and initialize overlays.

        This function iterates through the list of functional maps specified in
        `self.funcmaps`, loads each map from a NIfTI file (if it exists), and
        initializes an `Overlay` object for each. It ensures that all loaded
        maps have consistent dimensions and voxel sizes. If a map is listed in
        `mapstoinvert`, it will be inverted upon loading.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return a value but updates the following
            instance attributes:
            - `self.overlays`: Dictionary mapping map names to `Overlay` objects.
            - `self.loadedfuncmaps`: List of successfully loaded map names.

        Notes
        -----
        - The function checks for the existence of `.nii.gz` files before attempting
          to load them.
        - If dimensions or voxel sizes of the loaded maps do not match, the program
          will exit with an error message.
        - Map names in `mapstoinvert` (currently only "varChange") will be inverted
          during loading.

        Examples
        --------
        Assuming `self.funcmaps` contains entries like:
        [("varChange", "varchange_map"), ("tstat", "tstat_map")]

        And the corresponding files exist, this function will load these maps and
        store them in `self.overlays` with appropriate inversion applied where
        needed.
        """
        mapstoinvert = ["varChange"]
        self.loadedfuncmaps = []
        xdim = 0
        ydim = 0
        zdim = 0
        for mapname, mapfilename in self.funcmaps:
            if self.verbose > 1:
                print(f"loading {mapname} from {mapfilename}")
            if os.path.isfile(self.fileroot + mapfilename + ".nii.gz"):
                if self.verbose > 1:
                    print(
                        "file: ",
                        self.fileroot + mapfilename + ".nii.gz",
                        " exists - reading...",
                    )
                thepath, thebase = os.path.split(self.fileroot)
                if mapname in mapstoinvert:
                    invertthismap = True
                else:
                    invertthismap = False
                self.overlays[mapname] = Overlay(
                    mapname,
                    self.fileroot + mapfilename + ".nii.gz",
                    thebase,
                    init_LUT=self.init_LUT,
                    report=True,
                    invertonload=invertthismap,
                    verbose=self.verbose,
                )
                if xdim == 0:
                    xdim = self.overlays[mapname].xdim
                    ydim = self.overlays[mapname].ydim
                    zdim = self.overlays[mapname].zdim
                    tdim = self.overlays[mapname].tdim
                    xsize = self.overlays[mapname].xsize
                    ysize = self.overlays[mapname].ysize
                    zsize = self.overlays[mapname].zsize
                    tr = self.overlays[mapname].tr
                else:
                    if (
                        xdim != self.overlays[mapname].xdim
                        or ydim != self.overlays[mapname].ydim
                        or zdim != self.overlays[mapname].zdim
                    ):
                        print("overlay dimensions do not match!")
                        sys.exit()
                    if (
                        xsize != self.overlays[mapname].xsize
                        or ysize != self.overlays[mapname].ysize
                        or zsize != self.overlays[mapname].zsize
                    ):
                        print("overlay voxel sizes do not match!")
                        sys.exit()
                self.loadedfuncmaps.append(mapname)
            else:
                if self.verbose > 1:
                    print("map: ", self.fileroot + mapfilename + ".nii.gz", " does not exist!")
        if self.verbose > 1:
            print("functional maps loaded:", self.loadedfuncmaps)

    def _loadfuncmasks(self) -> None:
        """
        Load functional masks from specified files and create overlay objects.

        This method iterates through the functional masks defined in `self.funcmasks`
        and attempts to load each mask file. If a mask file exists, it creates an
        Overlay object and stores it in `self.overlays` with the mask name as key.

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - funcmasks : list of tuples
                List of (maskname, maskfilename) pairs to load
            - fileroot : str
                Root directory path for mask files
            - overlays : dict
                Dictionary to store loaded overlay objects
            - verbose : int
                Verbosity level for printing status messages
            - init_LUT : bool, optional
                Flag to initialize lookup table for overlays
            - loadedfuncmasks : list
                List to store names of successfully loaded masks

        Returns
        -------
        None
            This method modifies instance attributes in-place and does not return a value.

        Notes
        -----
        - Mask files are expected to have .nii.gz extension
        - Only masks that exist at the constructed file path are loaded
        - Progress information is printed based on verbosity level
        - Successfully loaded mask names are stored in `self.loadedfuncmasks`

        Examples
        --------
        >>> # Assuming self.funcmasks = [('mask1', 'mask1_file'), ('mask2', 'mask2_file')]
        >>> # and mask files exist at self.fileroot + maskfilename + ".nii.gz"
        >>> _loadfuncmasks()
        >>> print(self.loadedfuncmasks)
        ['mask1', 'mask2']
        >>> print(self.overlays['mask1'])
        <Overlay object at 0x...>
        """
        self.loadedfuncmasks = []
        for maskname, maskfilename in self.funcmasks:
            if self.verbose > 1:
                print(f"loading {maskname} from {maskfilename}")
            if os.path.isfile(self.fileroot + maskfilename + ".nii.gz"):
                thepath, thebase = os.path.split(self.fileroot)
                self.overlays[maskname] = Overlay(
                    maskname,
                    self.fileroot + maskfilename + ".nii.gz",
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                self.loadedfuncmasks.append(maskname)
            else:
                if self.verbose > 1:
                    print(
                        "mask: ",
                        self.fileroot + maskfilename + ".nii.gz",
                        " does not exist!",
                    )
        if self.verbose > 1:
            print(self.loadedfuncmasks)

    def _genpmasks(self, pvals: list[float] = [0.05, 0.01, 0.005, 0.001]) -> None:
        """
        Generate binary masks for specified p-value thresholds from negative log10 p-values.

        This function creates binary masks based on negative log10 p-values stored in
        self.overlays["neglog10p"]. Each mask represents regions where the negative
        log10 p-values exceed the specified threshold.

        Parameters
        ----------
        pvals : list of float, optional
            List of p-value thresholds for mask generation. Default is [0.05, 0.01, 0.005, 0.001].
            Each threshold is converted to a mask name format "p_lt_{threshold}_mask".

        Returns
        -------
        None
            This function modifies the instance's overlays and loadedfuncmasks attributes
            in-place and does not return any value.

        Notes
        -----
        - Mask names are formatted to replace "0.0" with "0p0" (e.g., "0.05" becomes "0p05")
        - The function uses the last loaded functional mask as the base for duplication
        - Generated masks are stored in self.overlays dictionary with corresponding names
        - The function updates self.loadedfuncmasks with the names of newly created masks

        Examples
        --------
        >>> _genpmasks([0.05, 0.01])
        # Generates masks for p-values 0.05 and 0.01 based on neglog10p data

        >>> _genpmasks()
        # Generates masks for default p-values [0.05, 0.01, 0.005, 0.001]
        """
        for thepval in pvals:
            maskname = f"p_lt_{thepval:.3f}_mask".replace("0.0", "0p0")
            nlpthresh = -np.log10(thepval)
            if self.verbose > 1:
                print(f"generating {maskname} from neglog10p")
            self.overlays[maskname] = self.overlays[self.loadedfuncmasks[-1]].duplicate(
                maskname, None, self.init_LUT
            )
            self.overlays[maskname].setData(
                np.where(self.overlays["neglog10p"].data > nlpthresh, 1.0, 0.0), isaMask=True
            )
            self.loadedfuncmasks.append(maskname)
        if self.verbose > 1:
            print(self.loadedfuncmasks)

    def _loadgeommask(self) -> bool:
        """
        Load a geometric mask based on configuration settings and available files.

        This function attempts to load a geometric mask either from a user-specified
        file or from a default location based on the coordinate space and voxel size.
        The mask is stored in `self.overlays["geommask"]` if successfully loaded.

        Returns
        -------
        bool
            True if a geometric mask was successfully loaded, False otherwise.

        Notes
        -----
        - If `self.geommaskname` is set, the function attempts to load the mask from that file.
        - If `self.coordinatespace` is "MNI152", the function searches for a default mask
          based on the voxel size (`xsize`, `ysize`, `zsize`).
        - The function uses the FSL directory to locate default masks when available.
        - Verbose output is printed if `self.verbose` is greater than 1.

        Examples
        --------
        >>> loader = SomeClass()
        >>> loader.geommaskname = "/path/to/custom_mask.nii.gz"
        >>> loader._loadgeommask()
        True

        >>> loader.coordinatespace = "MNI152"
        >>> loader.xsize = 2.0
        >>> loader.ysize = 2.0
        >>> loader.zsize = 2.0
        >>> loader._loadgeommask()
        True  # if default mask is found
        """
        if self.geommaskname is not None:
            if os.path.isfile(self.geommaskname):
                thepath, thebase = os.path.split(self.geommaskname)
                self.overlays["geommask"] = Overlay(
                    "geommask",
                    self.geommaskname,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.geommaskname, " as geometric mask")
                # allloadedmaps.append('geommask')
                return True
        elif self.coordinatespace == "MNI152":
            try:
                fsldir = os.environ["FSLDIR"]
            except KeyError:
                fsldir = None
            if self.verbose > 1:
                print("fsldir set to ", fsldir)
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if fsldir is not None:
                    self.geommaskname = os.path.join(
                        fsldir, "data", "standard", "MNI152_T1_2mm_brain_mask.nii.gz"
                    )
            elif self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                self.geommaskname = os.path.join(
                    self.referencedir, "MNI152_T1_3mm_brain_mask_bin.nii.gz"
                )
            if os.path.isfile(self.geommaskname):
                thepath, thebase = os.path.split(self.geommaskname)
                self.overlays["geommask"] = Overlay(
                    "geommask",
                    self.geommaskname,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.geommaskname, " as background")
                # allloadedmaps.append('geommask')
                return True
            else:
                if self.verbose > 1:
                    print("no geometric mask loaded")
                return False
        else:
            if self.verbose > 1:
                print("no geometric mask loaded")
            return False

    def _loadanatomics(self) -> bool:
        """
        Load anatomic image data based on available files and coordinate space settings.

        This method attempts to load anatomic images from various possible sources,
        prioritizing user-specified files, high-resolution templates, MNI templates,
        and mean-based images. The loaded image is stored in `self.overlays["anatomic"]`.

        Returns
        -------
        bool
            True if anatomic image was successfully loaded, False otherwise.

        Notes
        -----
        The method checks for the following files in order:
        1. User-specified anatomic file (`self.anatname`)
        2. High-resolution head image: `highres_head.nii.gz`
        3. High-resolution image: `highres.nii.gz`
        4. MNI152 template based on resolution (`xsize`, `ysize`, `zsize`)
        5. MNI152NLin2009cAsym template based on resolution
        6. Mean image: `mean.nii.gz`
        7. Mean value image: `meanvalue.nii.gz`
        8. Described mean image: `desc-unfiltmean_map.nii.gz`
        9. Described mean image: `desc-mean_map.nii.gz`

        If `FSLDIR` environment variable is set, it is used to locate MNI152 templates
        with 2mm resolution.

        Examples
        --------
        >>> loader = MyLoader()
        >>> loader.fileroot = "/path/to/data/"
        >>> loader.coordinatespace = "MNI152"
        >>> loader.xsize = 2.0
        >>> loader.ysize = 2.0
        >>> loader.zsize = 2.0
        >>> loader._loadanatomics()
        True
        """
        try:
            fsldir = os.environ["FSLDIR"]
        except KeyError:
            fsldir = None

        if self.anatname is not None:
            if self.verbose > 1:
                print("using user input anatomic name")
            if os.path.isfile(self.anatname):
                thepath, thebase = os.path.split(self.anatname)
                self.overlays["anatomic"] = Overlay(
                    "anatomic",
                    self.anatname,
                    thebase,
                    init_LUT=self.init_LUT,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.anatname, " as background")
                # allloadedmaps.append('anatomic')
                return True
            else:
                if self.verbose > 1:
                    print("specified file does not exist!")
                return False
        elif os.path.isfile(self.fileroot + "highres_head.nii.gz"):
            if self.verbose > 1:
                print("using hires_head anatomic name")
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "highres_head.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "highres_head.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "highres.nii.gz"):
            if self.verbose > 1:
                print("using hires anatomic name")
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "highres.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "highres.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif self.coordinatespace == "MNI152":
            mniname = ""
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if self.verbose > 1:
                    print("using 2mm MNI anatomic name")
                if fsldir is not None:
                    mniname = os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm.nii.gz")
            elif self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                if self.verbose > 1:
                    print("using 3mm MNI anatomic name")
                mniname = os.path.join(self.referencedir, "MNI152_T1_3mm.nii.gz")
            if os.path.isfile(mniname):
                self.overlays["anatomic"] = Overlay(
                    "anatomic",
                    mniname,
                    "MNI152",
                    init_LUT=self.init_LUT,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", mniname, " as background")
                # allloadedmaps.append('anatomic')
                return True
            else:
                if self.verbose > 1:
                    print("xsize, ysize, zsize=", self.xsize, self.ysize, self.zsize)
                    print("MNI template brain ", mniname, " not loaded")
        elif self.coordinatespace == "MNI152NLin2009cAsym":
            mniname = ""
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if self.verbose > 1:
                    print("using 2mm MNI anatomic name")
                if fsldir is not None:
                    mniname = os.path.join(
                        self.referencedir, "mni_icbm152_nlin_asym_09c_2mm.nii.gz"
                    )
            elif self.xsize == 1.0 and self.ysize == 1.0 and self.zsize == 1.0:
                if self.verbose > 1:
                    print("using 1mm MNI anatomic name")
                mniname = os.path.join(self.referencedir, "mni_icbm152_nlin_asym_09c_1mm.nii.gz")
            if os.path.isfile(mniname):
                self.overlays["anatomic"] = Overlay(
                    "anatomic",
                    mniname,
                    "MNI152NLin2009cAsym",
                    init_LUT=self.init_LUT,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", mniname, " as background")
                # allloadedmaps.append('anatomic')
                return True
            else:
                if self.verbose > 1:
                    print("xsize, ysize, zsize=", self.xsize, self.ysize, self.zsize)
                    print("MNI template brain ", mniname, " not loaded")
        elif os.path.isfile(self.fileroot + "mean.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "mean.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "mean.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "meanvalue.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "meanvalue.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "meanvalue.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "desc-unfiltmean_map.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "desc-unfiltmean_map.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print(
                    "using ",
                    self.fileroot + "desc-unfiltmean_map.nii.gz",
                    " as background",
                )
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "desc-mean_map.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "desc-mean_map.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print(
                    "using ",
                    self.fileroot + "desc-mean_map.nii.gz",
                    " as background",
                )
            # allloadedmaps.append('anatomic')
            return True
        else:
            if self.verbose > 1:
                print("no anatomic image loaded")
            return False

    def _loadgraymask(self) -> bool:
        """
        Load gray matter mask from specification.

        Load a gray matter mask from the file specification stored in `self.graymaskspec`.
        If successful, the mask is stored in `self.overlays["graymask"]` and the function
        returns True. If no mask specification is provided or the file doesn't exist,
        the function returns False.

        Returns
        -------
        bool
            True if gray matter mask was successfully loaded, False otherwise.

        Notes
        -----
        This function checks if `self.graymaskspec` is not None and if the specified
        file exists. If both conditions are met, it creates an Overlay object for the
        gray mask and stores it in `self.overlays["graymask"]`. The function also
        prints verbose messages when loading or skipping the mask.

        Examples
        --------
        >>> loaded = self._loadgraymask()
        >>> print(loaded)
        True
        """
        if self.graymaskspec is not None:
            filename, dummy = tide_io.parsefilespec(self.graymaskspec)
            if os.path.isfile(filename):
                thepath, thebase = os.path.split(self.graymaskspec)
                self.overlays["graymask"] = Overlay(
                    "graymask",
                    self.graymaskspec,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.graymaskspec, " as gray matter mask")
                # allloadedmaps.append('geommask')
                return True
        else:
            if self.verbose > 1:
                print("no gray mask loaded")
            return False

    def _loadwhitemask(self) -> bool:
        """
        Load white matter mask from specification if available.

        This method attempts to load a white matter mask from the specification
        stored in `self.whitemaskspec`. If the specification is valid and the
        corresponding file exists, it creates an Overlay object for the white
        matter mask and stores it in `self.overlays["whitemask"]`.

        Parameters
        ----------
        self : object
            The instance containing the white matter mask specification and
            overlay storage.

        Returns
        -------
        bool
            True if white matter mask was successfully loaded, False otherwise.

        Notes
        -----
        The method checks if `self.whitemaskspec` is not None and if the
        specified file exists before attempting to load it. If successful,
        the mask is stored in `self.overlays["whitemask"]` and a verbose
        message is printed if `self.verbose` is greater than 1.

        Examples
        --------
        >>> loaded = self._loadwhitemask()
        >>> print(loaded)
        True
        """
        if self.whitemaskspec is not None:
            filename, dummy = tide_io.parsefilespec(self.whitemaskspec)
            if os.path.isfile(filename):
                thepath, thebase = os.path.split(self.whitemaskspec)
                self.overlays["whitemask"] = Overlay(
                    "whitemask",
                    self.whitemaskspec,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.whitemaskspec, " as white matter mask")
                # allloadedmaps.append('geommask')
                return True
        else:
            if self.verbose > 1:
                print("no white mask loaded")
            return False

    def setupregressors(self) -> None:
        """
        Set up regressor specifications and load regressor data.

        This method initializes the regressor specifications based on the BIDS format
        and the run options, and loads the corresponding regressor data. It handles
        various configuration parameters such as filter limits, sampling frequencies,
        and similarity metrics, and prepares a list of regressor specifications for
        use in subsequent processing steps.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method does not return any value but updates the instance attributes
            `regressors`, `regressorfilterlimits`, `fmrifreq`, `inputfreq`,
            `inputstarttime`, `oversampfactor`, `similaritymetric`, `regressorsimcalclimits`,
            `numberofpasses`, and `regressorspecs`.

        Notes
        -----
        - The method reads run options from a file specified by `self.fileroot + "desc-runoptions_info"`.
        - If `self.bidsformat` is True, the regressor files are named according to BIDS conventions.
        - The method determines the number of passes and sets up the regressor specifications accordingly.
        - The `regressorspecs` list contains information for loading regressors at different stages:
          pre-filtered, post-filtered, and multiple passes, with associated file names, frequencies,
          and time offsets.

        Examples
        --------
        >>> setupregressors()
        # Updates instance attributes with regressor configurations and loads data.
        """
        # load the regressors
        self.regressors = {}
        self.therunoptions = tide_io.readoptionsfile(self.fileroot + "desc-runoptions_info")
        if self.verbose > 1:
            print("regressor similarity calculation limits:", self.regressorsimcalclimits)
        try:
            self.regressorfilterlimits = (
                float(self.therunoptions["lowerpass"]),
                float(self.therunoptions["upperpass"]),
            )
        except KeyError:
            self.regressorfilterlimits = (0.0, 100.0)
        if self.verbose > 1:
            print("regressor filter limits:", self.regressorfilterlimits)
        try:
            self.fmrifreq = float(self.therunoptions["fmrifreq"])
        except KeyError:
            self.fmrifreq = 1.0
        try:
            self.inputfreq = float(self.therunoptions["inputfreq"])
        except KeyError:
            self.inputfreq = 1.0
        try:
            self.inputstarttime = float(self.therunoptions["inputstarttime"])
        except KeyError:
            self.inputstarttime = 0.0
        try:
            self.oversampfactor = int(self.therunoptions["oversampfactor"])
        except KeyError:
            self.oversampfactor = 1
        try:
            self.similaritymetric = self.therunoptions["similaritymetric"]
        except KeyError:
            self.similaritymetric = "correlation"
        try:
            self.regressorsimcalclimits = (
                float((1.0 / self.fmrifreq) * self.therunoptions["validsimcalcstart"]),
                float((1.0 / self.fmrifreq) * self.therunoptions["validsimcalcend"]),
            )
        except KeyError:
            self.regressorsimcalclimits = (0.0, 10000000.0)
        try:
            self.numberofpasses = int(self.therunoptions["actual_passes"])
        except KeyError:
            self.numberofpasses = int(self.therunoptions["passes"])
        if self.numberofpasses > 4:
            secondtolast = self.numberofpasses - 1
            last = self.numberofpasses
        else:
            secondtolast = 3
            last = 4
        if self.bidsformat:
            self.regressorspecs = [
                [
                    "prefilt",
                    "prefilt",
                    "desc-initialmovingregressor_timeseries.json",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "postfilt",
                    "postfilt",
                    "desc-initialmovingregressor_timeseries.json",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "pass1",
                    "pass1",
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    True,
                ],
                [
                    "pass2",
                    "pass2",
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass3",
                    "pass{:d}".format(secondtolast),
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass4",
                    "pass{:d}".format(last),
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
            ]
        else:
            self.regressorspecs = [
                [
                    "prefilt",
                    "prefilt",
                    "reference_origres_prefilt.txt",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "postfilt",
                    "postfilt",
                    "reference_origres.txt",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "pass1",
                    "pass1",
                    "reference_resampres_pass1.txt",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    True,
                ],
                [
                    "pass2",
                    "pass2",
                    "reference_resampres_pass2.txt",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass3",
                    "pass{:d}".format(secondtolast),
                    "reference_resampres_pass{:d}.txt".format(secondtolast),
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass4",
                    "pass{:d}".format(last),
                    "reference_resampres_pass{:d}.txt".format(last),
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
            ]
        self._loadregressors()

    def getregressors(self) -> dict:
        """
        Return the regressors stored in the object.

        Returns
        -------
        dict
            A dictionary containing the regressors. The keys are typically
            regressor names and the values are the corresponding regressor objects.

        Notes
        -----
        This method provides access to the internal regressors dictionary
        that stores all regression models used by the object.

        Examples
        --------
        >>> model = MyRegressionModel()
        >>> regressors = model.getregressors()
        >>> print(regressors)
        {'linear_reg': LinearRegression(), 'ridge_reg': Ridge()}
        """
        return self.regressors

    def setfocusregressor(self, whichregressor: str) -> None:
        """
        Set the focus regressor for the current instance.

        This method sets the focus regressor to the specified regressor name if it exists
        in the regressors dictionary. If the specified regressor does not exist, it defaults
        to "prefilt".

        Parameters
        ----------
        whichregressor : str
            The name of the regressor to set as the focus regressor. This should be a key
            present in the instance's regressors dictionary.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        If the specified regressor name is not found in self.regressors, the method will
        automatically fall back to setting the focus regressor to "prefilt".

        Examples
        --------
        >>> instance.setfocusregressor("regressor1")
        >>> print(instance.focusregressor)
        'regressor1'

        >>> instance.setfocusregressor("nonexistent")
        >>> print(instance.focusregressor)
        'prefilt'
        """
        try:
            testregressor = self.regressors[whichregressor]
            self.focusregressor = whichregressor
        except KeyError:
            self.focusregressor = "prefilt"

    def setupoverlays(self) -> None:
        """
        Set up and load all overlays for functional and anatomical data.

        This function initializes the overlays dictionary and loads various functional
        maps, masks, and anatomical data based on the configuration parameters such as
        BIDS format, use of correlation outputs, and coordinate space. It also handles
        setting TR and time offset values for the loaded maps, and loads additional
        data such as atlases and tissue masks if applicable.

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Expected to have the
            following attributes:

            - bidsformat : bool
                Indicates whether the data is in BIDS format.
            - usecorrout : bool
                Whether to include correlation output maps.
            - newstylenames : bool
                Whether to use new-style naming conventions for maps.
            - userise : bool
                Whether to include rise time-related maps.
            - forcetr : bool
                Whether to force TR value for loaded maps.
            - forceoffset : bool
                Whether to force time offset for loaded maps.
            - trval : float
                The TR value to be set if `forcetr` is True.
            - offsettime : float
                The time offset to be set if `forceoffset` is True.
            - verbose : int
                Verbosity level for output messages.
            - referencedir : str
                Directory containing reference data such as atlases.
            - init_LUT : bool
                Whether to initialize lookup tables for overlays.
            - useatlas : bool
                Whether to load atlas data.
            - coordinatespace : str
                The coordinate space of the data (e.g., "MNI152", "MNI152NLin2009cAsym").

        Returns
        -------
        None
            This function does not return any value. It modifies the instance's
            attributes in place.

        Notes
        -----
        - Functional maps are loaded based on the BIDS format and naming conventions.
        - The function dynamically builds lists of functional maps and masks depending
          on the configuration.
        - If an atlas is to be used and the coordinate space is compatible, it will be
          loaded and added to the overlays.
        - The function sets up several instance variables such as `xdim`, `ydim`, `zdim`,
          `tdim`, `xsize`, `ysize`, `zsize`, and `tr` from the focus map.

        Examples
        --------
        >>> setupoverlays()
        # Loads all overlays and sets up the instance for further processing.
        """
        # load the overlays
        self.overlays = {}

        # first the functional maps
        if self.bidsformat:
            self.funcmaps = [
                ["lagtimes", "desc-maxtime_map"],
                ["lagtimesrefined", "desc-maxtimerefined_map"],
                ["timepercentile", "desc-timepercentile_map"],
                ["lagstrengths", "desc-maxcorr_map"],
                ["lagstrengthsrefined", "desc-maxcorrrefined_map"],
                ["lagsigma", "desc-maxwidth_map"],
                ["MTT", "desc-MTT_map"],
                ["R2", "desc-lfofilterR2_map"],
                ["CoV", "desc-CoV_map"],
                ["confoundR2", "desc-confoundfilterR2_map"],
                ["varBefore", "desc-lfofilterInbandVarianceBefore_map"],
                ["varAfter", "desc-lfofilterInbandVarianceAfter_map"],
                ["varChange", "desc-lfofilterInbandVarianceChange_map"],
                ["fitNorm", "desc-lfofilterNorm_map"],
                ["fitcoff", "desc-lfofilterCoeff_map"],
                ["neglog10p", "desc-neglog10p_map"],
                ["delayoffset", "desc-delayoffset_map"],
            ]
            if self.usecorrout:
                self.funcmaps += [["corrout", "desc-corrout_info"]]
                # self.funcmaps += [['gaussout', 'desc-gaussout_info']]
                self.funcmaps += [["failimage", "desc-corrfitfailreason_info"]]

        else:
            if self.newstylenames:
                self.funcmaps = [
                    ["lagtimes", "lagtimes"],
                    ["lagstrengths", "lagstrengths"],
                    ["lagsigma", "lagsigma"],
                    ["MTT", "MTT"],
                    ["R2", "R2"],
                    ["fitNorm", "fitNorm"],
                    ["fitcoff", "fitCoeff"],
                ]
                if self.usecorrout:
                    self.funcmaps += [["corrout", "corrout"]]
                    # self.funcmaps += [['gaussout', 'gaussout']]
                    self.funcmaps += [["failimage", "corrfitfailreason"]]

            else:
                self.funcmaps = [
                    ["lagtimes", "lagtimes"],
                    ["lagstrengths", "lagstrengths"],
                    ["lagsigma", "lagsigma"],
                    ["MTT", "MTT"],
                    ["R2", "R2"],
                    ["fitNorm", "fitNorm"],
                    ["fitcoff", "fitcoff"],
                ]
                if self.userise:
                    self.funcmaps = [
                        ["lagtimes", "lagtimes"],
                        ["lagstrengths", "lagstrengths"],
                        ["lagsigma", "lagsigma"],
                        ["MTT", "MTT"],
                        ["R2", "R2"],
                        ["risetime_epoch_0", "risetime_epoch_0"],
                        ["starttime_epoch_0", "starttime_epoch_0"],
                        ["maxamp_epoch_0", "maxamp_epoch_0"],
                    ]
                if self.usecorrout:
                    self.funcmaps += [["corrout", "corrout"]]
                    # self.funcmaps += [['gaussout', 'gaussout']]
                    self.funcmaps += [["failimage", "failimage"]]

        self._loadfuncmaps()
        for themap in self.loadedfuncmaps:
            if self.forcetr:
                self.overlays[themap].setTR(self.trval)
            if self.forceoffset:
                self.overlays[themap].settoffset(self.offsettime)
            if self.overlays[themap].space == "MNI152":
                self.coordinatespace = "MNI152"
            elif self.overlays[themap].space == "MNI152NLin2009cAsym":
                self.coordinatespace = "MNI152NLin2009cAsym"

        # report results of load
        if self.verbose > 1:
            print("loaded functional maps: ", self.loadedfuncmaps)

        self.allloadedmaps = list(self.loadedfuncmaps)
        self.dispmaps = list(self.loadedfuncmaps)

        # extract some useful information about this dataset from the focusmap
        self.focusmap = "lagtimes"

        self.xdim = self.overlays[self.focusmap].xdim
        self.ydim = self.overlays[self.focusmap].ydim
        self.zdim = self.overlays[self.focusmap].zdim
        self.tdim = self.overlays[self.focusmap].tdim
        self.xsize = self.overlays[self.focusmap].xsize
        self.ysize = self.overlays[self.focusmap].ysize
        self.zsize = self.overlays[self.focusmap].zsize
        self.tr = self.overlays[self.focusmap].tr

        # then load the anatomics
        if self._loadanatomics():
            self.allloadedmaps.append("anatomic")

        # then the functional masks
        if self.bidsformat:
            self.funcmasks = [
                ["lagmask", "desc-corrfit_mask"],
                ["refinemask", "desc-refine_mask"],
                ["meanmask", "desc-globalmean_mask"],
                ["brainmask", "desc-brainmask_mask"],
                ["preselectmask", "desc-globalmeanpreselect_mask"],
            ]
            if not ("neglog10p" in self.loadedfuncmaps):
                # load p maps manually
                self.funcmasks += [
                    ["p_lt_0p050_mask", "desc-plt0p050_mask"],
                    ["p_lt_0p010_mask", "desc-plt0p010_mask"],
                    ["p_lt_0p005_mask", "desc-plt0p005_mask"],
                    ["p_lt_0p001_mask", "desc-plt0p001_mask"],
                ]
        else:
            if self.newstylenames:
                self.funcmasks = [
                    ["lagmask", "fitmask"],
                    ["refinemask", "refinemask"],
                    ["meanmask", "meanmask"],
                    ["p_lt_0p050_mask", "p_lt_0p050_mask"],
                    ["p_lt_0p010_mask", "p_lt_0p010_mask"],
                    ["p_lt_0p005_mask", "p_lt_0p005_mask"],
                    ["p_lt_0p001_mask", "p_lt_0p001_mask"],
                ]
            else:
                self.funcmasks = [
                    ["lagmask", "lagmask"],
                    ["refinemask", "refinemask"],
                    ["meanmask", "meanmask"],
                    ["p_lt_0p050_mask", "p_lt_0p050_mask"],
                    ["p_lt_0p010_mask", "p_lt_0p010_mask"],
                    ["p_lt_0p005_mask", "p_lt_0p005_mask"],
                    ["p_lt_0p001_mask", "p_lt_0p001_mask"],
                ]
        self._loadfuncmasks()
        if "neglog10p" in self.loadedfuncmaps:
            # generate p maps on the fly
            self._genpmasks()

        # then the geometric masks
        if self._loadgeommask():
            self.allloadedmaps.append("geommask")

        # then the tissue masks
        if self._loadgraymask():
            self.allloadedmaps.append("graymask")
        if self._loadwhitemask():
            self.allloadedmaps.append("whitemask")

        if self.useatlas and (
            (self.coordinatespace == "MNI152")
            or (self.coordinatespace == "MNI152NLin6")
            or (self.coordinatespace == "MNI152NLin2009cAsym")
        ):
            self.atlasshortname = "JHU1"
            self.atlasname = atlases[self.atlasshortname]["atlasname"]
            self.atlaslabels = tide_io.readlabels(
                os.path.join(self.referencedir, self.atlasname + "_regions.txt")
            )
            if self.verbose > 1:
                print(self.atlaslabels)
            self.atlasniftiname = None
            if self.coordinatespace == "MNI152":
                spacename = "_space-MNI152NLin6Asym"
                if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                    self.atlasniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_2mm.nii.gz"
                    )
                    self.atlasmaskniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_2mm_mask.nii.gz"
                    )
                if self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                    self.atlasniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_3mm.nii.gz"
                    )
                    self.atlasmaskniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_3mm_mask.nii.gz"
                    )
            else:
                pass
                """if xsize == 2.0 and ysize == 2.0 and zsize == 2.0:
                    atlasniftiname = os.path.join(referencedir, atlasname + '_nlin_asym_09c_2mm.nii.gz')
                    atlasmaskniftiname = os.path.join(referencedir, atlasname + '_nlin_asym_09c_2mm_mask.nii.gz')"""
            if self.atlasniftiname is not None:
                if os.path.isfile(self.atlasniftiname):
                    self.overlays["atlas"] = Overlay(
                        "atlas",
                        self.atlasniftiname,
                        self.atlasname,
                        report=True,
                        init_LUT=self.init_LUT,
                        verbose=self.verbose,
                    )
                    self.overlays["atlasmask"] = Overlay(
                        "atlasmask",
                        self.atlasmaskniftiname,
                        self.atlasname,
                        init_LUT=self.init_LUT,
                        report=True,
                        verbose=self.verbose,
                    )
                    self.allloadedmaps.append("atlas")
                    self.dispmaps.append("atlas")
                else:
                    print(
                        self.atlasname + " template: ",
                        self.atlasniftiname,
                        " does not exist!",
                    )

        try:
            test = self.overlays["atlas"]
            if self.verbose > 1:
                print("there is an atlas")
        except KeyError:
            if self.verbose > 1:
                print("there is not an atlas")
        if self.verbose > 1:
            print("done")

    def getoverlays(self) -> dict:
        """
        Return the overlays dictionary.

        Returns
        -------
        dict
            A dictionary containing the overlays data.

        Notes
        -----
        This method provides access to the internal overlays attribute.
        The returned dictionary contains all overlay data managed by this instance.

        Examples
        --------
        >>> overlays = obj.getoverlays()
        >>> print(overlays)
        {'overlay1': <OverlayObject>, 'overlay2': <OverlayObject>}
        """
        return self.overlays

    def setfocusmap(self, whichmap: str) -> None:
        """
        Set the focus map to the specified map name.

        This method sets the current focus map to the specified map name if it exists
        in the overlays dictionary. If the specified map does not exist, it defaults
        to setting the focus map to "lagtimes".

        Parameters
        ----------
        whichmap : str
            The name of the map to set as the focus map. This should correspond to
            a key in the self.overlays dictionary.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        If the specified `whichmap` is not found in `self.overlays`, the method
        will automatically fall back to setting the focus map to "lagtimes".

        Examples
        --------
        >>> obj.setfocusmap("temperature")
        >>> obj.focusmap
        'temperature'

        >>> obj.setfocusmap("nonexistent")
        >>> obj.focusmap
        'lagtimes'
        """
        try:
            testmap = self.overlays[whichmap]
            self.focusmap = whichmap
        except KeyError:
            self.focusmap = "lagtimes"

    def setFuncMaskName(self, maskname: str) -> None:
        """
        Set the function mask name attribute.

        Parameters
        ----------
        maskname : str
            The name to assign to the function mask attribute.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided mask name to the internal `funcmaskname` attribute
        of the object instance.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setFuncMaskName("my_mask")
        >>> print(obj.funcmaskname)
        'my_mask'
        """
        self.funcmaskname = maskname
