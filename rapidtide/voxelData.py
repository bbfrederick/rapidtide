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
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.util as tide_util


class dataVolume:
    xsize = None
    ysize = None
    numslices = None
    numspatiallocs = None
    timepoints = None
    dtype = None
    dimensions = None
    data = None
    data_shm = None
    thepid = None

    def __init__(
        self,
        shape: tuple,
        shared: bool = False,
        dtype: type = np.float64,
        thepid: int = 0,
    ) -> None:
        """
        Initialize a data container with specified shape and properties.

        Parameters
        ----------
        shape : tuple
            The shape of the data array. Must be either 3D (x, y, z) or 4D (x, y, z, t).
        shared : bool, optional
            Whether to create a shared memory array, by default False
        dtype : type, optional
            Data type of the array, by default np.float64
        thepid : int, optional
            Process ID for naming shared memory, by default 0

        Returns
        -------
        None
            This method initializes the object in-place and does not return a value.

        Notes
        -----
        The function automatically determines the number of dimensions based on the shape tuple length.
        For 3D shapes, timepoints is set to 1. For 4D shapes, timepoints is set to the fourth dimension.
        Invalid shapes will trigger a print statement with the error message.

        Examples
        --------
        >>> data_container = DataContainer((64, 64, 32))
        >>> data_container = DataContainer((64, 64, 32, 10), shared=True, dtype=np.float32)
        """
        if len(shape) == 3:
            self.xsize = int(shape[0])
            self.ysize = int(shape[1])
            self.numslices = int(shape[2])
            self.timepoints = 1
            self.dimensions = 3
        elif len(shape) == 4:
            self.xsize = int(shape[0])
            self.ysize = int(shape[1])
            self.numslices = int(shape[2])
            self.timepoints = int(shape[3])
            self.dimensions = 4
        else:
            print(f"illegal shape: {shape}")
        self.numspatiallocs = self.xsize * self.ysize * self.numslices
        self.dtype = dtype
        self.data, self.data_shm = tide_util.allocarray(
            shape, self.dtype, shared=shared, name=f"filtereddata_{thepid}"
        )

    def byvol(self) -> NDArray:
        """
        Return the data array.

        Returns
        -------
        NDArray
            The underlying data array stored in the object.

        Notes
        -----
        This method provides direct access to the internal data array.
        The returned array is a view of the original data and modifications
        to it will affect the original object.

        Examples
        --------
        >>> obj = MyClass()
        >>> data = obj.byvol()
        >>> print(data.shape)
        (100, 50)
        """
        return self.data

    def byslice(self) -> NDArray:
        """
        Reshape data by slices for 2D processing.

        Reshapes the internal data array to facilitate 2D processing operations
        by combining the x and y dimensions while preserving slice information.

        Parameters
        ----------
        self : object
            The instance containing the data to be reshaped. Expected to have
            attributes: dimensions (int), data (array-like), xsize (int),
            ysize (int), and numslices (int).

        Returns
        -------
        NDArray
            Reshaped array with dimensions (xsize * ysize, -1). For 3D data,
            the shape becomes (xsize * ysize, -1). For 4D data, the shape becomes
            (xsize * ysize, numslices, -1).

        Notes
        -----
        This function is useful for preparing data for 2D processing operations
        where the spatial dimensions need to be flattened while maintaining
        temporal or spectral slice information.

        Examples
        --------
        >>> # For 3D data with shape (100, 100, 50)
        >>> result = obj.byslice()
        >>> # Result shape: (10000, 50)
        >>>
        >>> # For 4D data with shape (50, 50, 10, 20)
        >>> result = obj.byslice()
        >>> # Result shape: (2500, 10, 20)
        """
        if self.dimensions == 3:
            return self.data.reshape(self.xsize * self.ysize, -1)
        else:
            return self.data.reshape(self.xsize * self.ysize, self.numslices, -1)

    def byvoxel(self) -> NDArray:
        """
        Reshape data to voxel format based on dimensions.

        Returns
        -------
        NDArray
            Reshaped array where each row represents a voxel. For 3D data, returns
            a 1D array of shape (numspatiallocs,). For non-3D data, returns a 2D array
            of shape (numspatiallocs, -1).

        Notes
        -----
        This method reshapes the internal ``data`` attribute to a voxel-based
        structure. The ``numspatiallocs`` attribute determines the first dimension
        of the output array, while the second dimension is determined by the
        remaining data dimensions.

        Examples
        --------
        >>> # For 3D data
        >>> result = obj.byvoxel()
        >>> print(result.shape)
        (1000,)  # where 1000 = numspatiallocs

        >>> # For 2D data
        >>> result = obj.byvoxel()
        >>> print(result.shape)
        (1000, 5)  # where 1000 = numspatiallocs, 5 = remaining dimensions
        """
        if self.dimensions == 3:
            return self.data.reshape(self.numspatiallocs)
        else:
            return self.data.reshape(self.numspatiallocs, -1)

    def destroy(self) -> None:
        """
        Clean up and destroy the object's resources.

        This method releases the internal data storage and performs cleanup of
        shared memory resources if they exist.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method first deletes the internal `data` attribute, then checks if
        `data_shm` (shared memory) exists and performs cleanup if it does.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.destroy()
        >>> # Object resources have been cleaned up
        """
        del self.data
        if self.data_shm is not None:
            tide_util.cleanup_shm(self.data_shm)


class VoxelData:
    nim = None
    nim_data = None
    nim_hdr = None
    nim_affine = None
    theshape = None
    xsize = None
    ysize = None
    numslices = None
    timepoints = None
    dimensions = None
    realtimepoints = None
    xdim = None
    ydim = None
    slicethickness = None
    timestep = None
    thesizes = None
    thedims = None
    numslicelocs = None
    numspatiallocs = None
    nativespaceshape = None
    nativefmrishape = None
    validvoxels = None
    cifti_hdr = None
    filetype = None
    resident = False

    def __init__(
        self,
        filename: str,
        timestep: float = 0.0,
        validstart: int | None = None,
        validend: int | None = None,
    ) -> None:
        """
        Initialize the object with filename and optional data reading parameters.

        Parameters
        ----------
        filename : str
            Path to the data file to be read.
        timestep : float, optional
            Time step for data processing, default is 0.0.
        validstart : int, optional
            Starting index for valid data range, default is None (all data).
        validend : int, optional
            Ending index for valid data range, default is None (all data).

        Returns
        -------
        None
            This method initializes the object and reads data but does not return any value.

        Notes
        -----
        This method calls `readdata()` internally with the provided parameters to load
        and process the data from the specified file.

        Examples
        --------
        >>> obj = MyClass("data.txt")
        >>> obj = MyClass("data.txt", timestep=0.1, validstart=10, validend=100)
        """

        self.filename = filename
        self.readdata(timestep, validstart, validend)

    def readdata(self, timestep: float, validstart: int | None, validend: int | None) -> None:
        """
        Load and process data from a file based on its type (NIfTI, CIFTI, or text).

        This function loads data using `self.load()` and determines the file type
        (NIfTI, CIFTI, or text) to set appropriate attributes such as dimensions,
        timepoints, and spatial locations. It also handles time-related parameters
        like `timestep` and `toffset`, and sets valid time ranges.

        Parameters
        ----------
        timestep : float
            The time step size (in seconds) for the data. If <= 0.0, the function
            will attempt to infer it from the file metadata (except for text files,
            which require this to be explicitly set).
        validstart : int, optional
            The starting index of the valid time range. If None, defaults to the
            beginning of the data.
        validend : int, optional
            The ending index of the valid time range. If None, defaults to the end
            of the data.

        Returns
        -------
        None
            This function does not return a value but updates the instance attributes
            of `self` based on the loaded data and parameters.

        Notes
        -----
        - For text files, `timestep` must be provided explicitly; otherwise, a
          `ValueError` is raised.
        - For CIFTI files, the `timestep` is hardcoded to 0.72 seconds as a temporary
          workaround until full XML parsing is implemented.
        - The function sets various internal attributes such as `xsize`, `ysize`,
          `numslices`, `timepoints`, `numspatiallocs`, and `nativespaceshape`
          depending on the file type.

        Examples
        --------
        >>> readdata(timestep=1.0, validstart=0, validend=100)
        # Loads data with a 1-second timestep and valid time range from 0 to 100.

        >>> readdata(timestep=0.0, validstart=None, validend=None)
        # Loads data and infers timestep from file metadata (if not text).
        """
        # load the data
        self.load()

        if tide_io.checkiftext(self.filename):
            self.filetype = "text"
            self.nim_hdr = None
            self.nim_affine = None
            self.theshape = np.shape(self.nim_data)
            self.xsize = self.theshape[0]
            self.ysize = 1
            self.numslices = 1
            self.numslicelocs = None
            self.timepoints = int(self.theshape[1])
            self.thesizes = [0, int(self.xsize), 1, 1, int(self.timepoints)]
            self.toffset = 0.0
            self.numspatiallocs = int(self.xsize)
            self.nativespaceshape = self.xsize
            self.cifti_hdr = None
        else:
            if tide_io.checkifcifti(self.filename):
                self.filetype = "cifti"
                self.nim_affine = None
                self.numslicelocs = None
                self.timepoints = int(self.nim_data.shape[1])
                self.numspatiallocs = self.nim_data.shape[0]
                self.nativespaceshape = (1, 1, 1, 1, self.numspatiallocs)
            else:
                self.filetype = "nifti"
                self.nim_affine = self.nim.affine
                self.xsize, self.ysize, self.numslices, self.timepoints = tide_io.parseniftidims(
                    self.thedims
                )
                if self.timepoints == 1:
                    self.dimensions = 3
                else:
                    self.dimensions = 4
                self.numslicelocs = int(self.xsize) * int(self.ysize)
                self.numspatiallocs = int(self.xsize) * int(self.ysize) * int(self.numslices)
                self.cifti_hdr = None
                self.nativespaceshape = (self.xsize, self.ysize, self.numslices)
            self.xdim, self.ydim, self.slicethickness, dummy = tide_io.parseniftisizes(
                self.thesizes
            )

        # correct some fields if necessary
        if self.filetype == "cifti":
            self.timestep = 0.72  # this is wrong and is a hack until I can parse CIFTI XML
            self.toffset = 0.0
        else:
            if self.filetype == "text":
                if timestep <= 0.0:
                    raise ValueError(
                        "for text file data input, you must use the -t option to set the timestep"
                    )
            else:
                if self.nim_hdr.get_xyzt_units()[1] == "msec":
                    self.timestep = self.thesizes[4] / 1000.0
                    self.toffset = self.nim_hdr["toffset"] / 1000.0
                else:
                    self.timestep = self.thesizes[4]
                    self.toffset = self.nim_hdr["toffset"]
        if timestep > 0.0:
            self.timestep = timestep

        self.setvalidtimes(validstart, validend)
        self.resident = True

    def copyheader(
        self,
        numtimepoints: int | None = None,
        tr: float | None = None,
        toffset: float | None = None,
    ) -> Any | None:
        """
        Copy and modify header information for neuroimaging files.

        This method creates a deep copy of the current header and modifies specific
        dimensions and parameters based on the file type (CIFTI or other formats).
        For text files, returns None immediately. For CIFTI files, modifies time and
        space dimensions. For other file types, updates time dimensions and related
        parameters.

        Parameters
        ----------
        numtimepoints : int, optional
            Number of time points to set in the header. If None, time dimension
            remains unchanged for non-CIFTI files.
        tr : float, optional
            Repetition time (TR) to set in the header. If None, TR remains unchanged.
        toffset : float, optional
            Time offset to set in the header. If None, time offset remains unchanged.

        Returns
        -------
        dict or None
            Modified header dictionary for non-text files, or None for text files.
            Returns None if the file type is "text".

        Notes
        -----
        For CIFTI files:
            - Time dimension is updated to numtimepoints
            - Space dimension is set to self.numspatiallocs
        For other file types:
            - Time dimension is updated to numtimepoints (index 4)
            - Dimension index 0 is updated based on numtimepoints (4 for >1, 3 for 1)
            - TR is set in pixdim[4] if provided
            - Time offset is set in toffset if provided

        Examples
        --------
        >>> header = obj.copyheader(numtimepoints=100, tr=2.0)
        >>> header = obj.copyheader(toffset=-5.0)
        >>> header = obj.copyheader()
        """
        if self.filetype == "text":
            return None
        else:
            thisheader = copy.deepcopy(self.nim_hdr)
            if self.filetype == "cifti":
                timeindex = thisheader["dim"][0] - 1
                spaceindex = thisheader["dim"][0]
                thisheader["dim"][timeindex] = numtimepoints
                thisheader["dim"][spaceindex] = self.numspatiallocs
            else:
                if numtimepoints is not None:
                    thisheader["dim"][4] = numtimepoints
                    if numtimepoints > 1:
                        thisheader["dim"][0] = 4
                    else:
                        thisheader["dim"][0] = 3
                        thisheader["pixdim"][4] = 1.0
                if toffset is not None:
                    thisheader["toffset"] = toffset
                if tr is not None:
                    thisheader["pixdim"][4] = tr
            return thisheader

    def getsizes(self) -> tuple[float, float, float, float]:
        """
        Return the dimensions and spacing parameters of the data structure.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing four float values in order:
            - xdim: x-dimension size
            - ydim: y-dimension size
            - slicethickness: thickness of each slice
            - timestep: time step between measurements

        Notes
        -----
        This method provides access to the fundamental spatial and temporal
        parameters of the data structure. The returned values represent the
        physical dimensions and spacing characteristics that define the
        coordinate system of the data.

        Examples
        --------
        >>> sizes = obj.getsizes()
        >>> print(sizes)
        (100.0, 100.0, 1.0, 0.1)
        >>> x_size, y_size, slice_thick, time_step = obj.getsizes()
        """
        return self.xdim, self.ydim, self.slicethickness, self.timestep

    def getdims(self) -> tuple[int, int, int, int]:
        """
        Return the dimensions of the data structure.

        Returns
        -------
        tuple[int, int, int, int]
            A tuple containing four integers representing:
            - xsize: width dimension
            - ysize: height dimension
            - numslices: number of slices
            - timepoints: number of time points

        Notes
        -----
        This method provides access to the fundamental spatial and temporal dimensions
        of the data structure. The returned tuple follows the order (x, y, slices, time).

        Examples
        --------
        >>> dims = obj.getdims()
        >>> print(dims)
        (640, 480, 32, 100)
        >>> x, y, slices, time = obj.getdims()
        >>> print(f"Data shape: {x} x {y} x {slices} x {time}")
        Data shape: 640 x 480 x 32 x 100
        """
        return self.xsize, self.ysize, self.numslices, self.timepoints

    def unload(self) -> None:
        """
        Unload Nim data and clean up resources.

        This method removes the Nim data and Nim object references from the instance
        and marks the instance as not resident in memory.

        Notes
        -----
        This method should be called to properly clean up resources when the Nim
        data is no longer needed. The method deletes the internal references to
        ``nim_data`` and ``nim`` objects and sets the ``resident`` flag to ``False``.

        Examples
        --------
        >>> instance = MyClass()
        >>> instance.load()  # Load some data
        >>> instance.unload()  # Clean up resources
        >>> instance.resident
        False
        """
        del self.nim_data
        del self.nim
        self.resident = False

    def load(self) -> None:
        """
        Load data from file based on file type detection.

        This method loads data from the specified filename, automatically detecting
        whether the file is text, CIFTI, or NIFTI format. The loaded data is stored
        in instance variables for subsequent processing.

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Expected to have
            attributes: filename (str), filetype (str or None), and various data
            storage attributes (nim_data, nim, cifti_hdr, nim_hdr, thedims, thesizes).

        Returns
        -------
        None
            This method does not return any value but updates instance attributes
            with loaded data.

        Notes
        -----
        - If filetype is not None, the method prints "reloading non-resident data"
        - For text files, data is read using tide_io.readvecs()
        - For CIFTI files, data is read using tide_io.readfromcifti() and stored
          in multiple attributes including cifti_hdr and nim_hdr
        - For NIFTI files, data is read using tide_io.readfromnifti() and stored
          in nim, nim_data, nim_hdr, thedims, and thesizes attributes
        - The method sets self.resident = True upon successful completion

        Examples
        --------
        >>> loader = DataContainer()
        >>> loader.filename = "data.nii.gz"
        >>> loader.load()
        loading data from data.nii.gz
        """
        if self.filetype is not None:
            print("reloading non-resident data")
        else:
            print(f"loading data from {self.filename}")
        if tide_io.checkiftext(self.filename):
            self.nim_data = tide_io.readvecs(self.filename)
            self.nim = None
        else:
            if tide_io.checkifcifti(self.filename):
                self.filetype = "cifti"
                (
                    dummy,
                    self.cifti_hdr,
                    self.nim_data,
                    self.nim_hdr,
                    self.thedims,
                    self.thesizes,
                    dummy,
                ) = tide_io.readfromcifti(self.filename)
                self.nim = None
            else:
                self.nim, self.nim_data, self.nim_hdr, self.thedims, self.thesizes = (
                    tide_io.readfromnifti(self.filename)
                )
        self.resident = True

    def setvalidtimes(self, validstart: int | None, validend: int | None) -> None:
        """
        Set valid time points for the object based on start and end indices.

        This method configures the valid time range for the object by setting
        `validstart` and `validend` attributes. It also calculates the number of
        real time points and updates the native fMRI shape based on the file type.

        Parameters
        ----------
        validstart : int, optional
            The starting index of valid time points. If None, defaults to 0.
        validend : int, optional
            The ending index of valid time points. If None, defaults to
            `self.timepoints - 1`.

        Returns
        -------
        None
            This method modifies the object's attributes in-place and does not return anything.

        Notes
        -----
        The method calculates `realtimepoints` as `validend - validstart + 1`.
        The `nativefmrishape` is updated based on the `filetype` attribute:
        - "nifti": (xsize, ysize, numslices, realtimepoints)
        - "cifti": (1, 1, 1, realtimepoints, numspatiallocs)
        - else: (xsize, realtimepoints)

        Examples
        --------
        >>> obj.setvalidtimes(5, 15)
        >>> obj.setvalidtimes(None, 10)
        >>> obj.setvalidtimes(0, None)
        """
        if validstart is None:
            self.validstart = 0
        else:
            self.validstart = validstart
        if validend is None:
            self.validend = self.timepoints - 1
        else:
            self.validend = validend
        self.realtimepoints = self.validend - self.validstart + 1
        if self.filetype == "nifti":
            self.nativefmrishape = (self.xsize, self.ysize, self.numslices, self.realtimepoints)
        elif self.filetype == "cifti":
            self.nativefmrishape = (1, 1, 1, self.realtimepoints, self.numspatiallocs)
        else:
            self.nativefmrishape = (self.xsize, self.realtimepoints)

    def setvalidvoxels(self, validvoxels: NDArray) -> None:
        """
        Set the valid voxels for the object.

        Parameters
        ----------
        validvoxels : NDArray
            Array containing the valid voxel coordinates. The first dimension
            represents the number of valid spatial locations.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method updates two attributes:
        - `self.validvoxels`: Stores the provided valid voxel array
        - `self.numvalidspatiallocs`: Stores the number of valid spatial locations
          (derived from the shape of the validvoxels array)

        Examples
        --------
        >>> obj = MyClass()
        >>> voxels = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> obj.setvalidvoxels(voxels)
        >>> print(obj.numvalidspatiallocs)
        3
        """
        self.validvoxels = validvoxels
        self.numvalidspatiallocs = np.shape(self.validvoxels)[0]

    def byvol(self) -> NDArray:
        """
        Return the nim_data array from the object.

        If the object is not resident, it will be loaded first.

        Returns
        -------
        NDArray
            The nim_data array stored in the object.

        Notes
        -----
        This method checks if the object is resident and loads it if necessary
        before returning the nim_data array.

        Examples
        --------
        >>> obj = MyClass()
        >>> data = obj.byvol()
        >>> print(data.shape)
        (100, 100)
        """
        if not self.resident:
            self.load()
        return self.nim_data

    def byvoltrimmed(self) -> NDArray:
        """
        Return data with volume trimming applied based on valid time range.

        This method returns the neuroimaging data with temporal dimensions trimmed
        according to the valid start and end indices. The behavior varies depending
        on the file type and dimensions of the data.

        Returns
        -------
        NDArray
            Trimmed neuroimaging data array. For NIfTI files with 4D data or
            CIFTI/text files, returns data with shape (X, Y, Z, T) where T is
            the trimmed temporal dimension. For other file types, returns data
            with shape (X, Y, Z) or (X, T) depending on the data structure.

        Notes
        -----
        - If the data is not resident in memory, it will be loaded automatically
        - For NIfTI files with 4D data, CIFTI files, or text files, the temporal
          dimension is trimmed using self.validstart and self.validend
        - For other file types, only the temporal dimension is trimmed
        - The validend index is inclusive in the returned data

        Examples
        --------
        >>> data = obj.byvoltrimmed()
        >>> print(data.shape)
        (64, 64, 32, 100)  # For 4D NIfTI data with 100 valid time points
        """
        if not self.resident:
            self.load()
        if self.filetype == "nifti":
            if self.dimensions == 4 or self.filetype == "cifti" or self.filetype == "text":
                return self.nim_data[:, :, :, self.validstart : self.validend + 1]
            else:
                return self.nim_data[:, :, :]
        else:
            return self.nim_data[:, self.validstart : self.validend + 1]

    def byvoxel(self) -> NDArray:
        """
        Reshape data by voxel across spatial locations.

        This method reshapes the trimmed volume data to organize it by voxel
        across all spatial locations. The output format depends on the
        file type and dimensions of the data.

        Returns
        -------
        NDArray
            Reshaped array where each row represents a voxel and columns
            represent time points or other dimensions. For 4D data or
            CIFTI/text files, the shape is (numspatiallocs, -1), otherwise
            (numspatiallocs,).

        Notes
        -----
        - For 4D data, CIFTI files, or text files, the result is reshaped to
          have an additional dimension for time points or other temporal
          dimensions
        - For other file types, the result is reshaped to a 2D array with
          shape (numspatiallocs,)

        Examples
        --------
        >>> data = MyDataClass()
        >>> voxel_data = data.byvoxel()
        >>> print(voxel_data.shape)
        (numspatiallocs, num_timepoints)
        """
        if self.dimensions == 4 or self.filetype == "cifti" or self.filetype == "text":
            return self.byvoltrimmed().reshape(self.numspatiallocs, -1)
        else:
            return self.byvoltrimmed().reshape(self.numspatiallocs)

    def byslice(self) -> NDArray:
        """
        Reshape data by slice dimensions.

        Reshapes the data returned by `byvoltrimmed()` to organize data by slice
        locations and slices. The output format depends on the file type and
        dimensions of the data.

        Returns
        -------
        NDArray
            Reshaped array with dimensions (numslicelocs, numslices, -1) for
            CIFTI or text files, or (numslicelocs, numslices) for other file types.

        Notes
        -----
        This method is particularly useful for organizing volumetric data by
        slice locations and slices. For CIFTI and text file types, the last
        dimension is automatically expanded to accommodate additional data
        dimensions.

        Examples
        --------
        >>> data = MyClass()
        >>> result = data.byslice()
        >>> print(result.shape)
        (10, 20, 100)  # for CIFTI/text files
        >>> print(result.shape)
        (10, 20)       # for other file types
        """
        if self.dimensions == 4 or self.filetype == "cifti" or self.filetype == "text":
            return self.byvoltrimmed().reshape(self.numslicelocs, self.numslices, -1)
        else:
            return self.byvoltrimmed().reshape(self.numslicelocs, self.numslices)

    def validdata(self) -> NDArray:
        """
        Return valid voxel data based on validvoxels mask.

        If validvoxels is None, returns all voxel data. Otherwise, returns
        only the subset of voxel data indicated by the validvoxels mask.

        Returns
        -------
        NDArray
            Array containing voxel data. If validvoxels is None, returns
            all voxel data from byvoxel() method. If validvoxels is not None,
            returns subset of voxel data filtered by validvoxels mask.

        Notes
        -----
        This method relies on the byvoxel() method to generate the base
        voxel data array and applies filtering based on the validvoxels
        attribute if it exists.

        Examples
        --------
        >>> # Get all voxel data
        >>> data = obj.validdata()
        >>>
        >>> # Get filtered voxel data
        >>> obj.validvoxels = [0, 1, 2, 5, 10]
        >>> data = obj.validdata()
        """
        if self.validvoxels is None:
            return self.byvoxel()
        else:
            return self.byvoxel()[self.validvoxels, :]

    # def validdatabyslice(self):
    #    if self.validvoxels is None:
    #       return self.byslice()
    #    else:
    #        return self.byvoxel()[self.validvoxels, :].reshape(self.numslicelocs, self.numslices, -1)

    def smooth(
        self,
        gausssigma: float,
        brainmask: NDArray | None = None,
        graymask: NDArray | None = None,
        whitemask: NDArray | None = None,
        premask: bool = False,
        premasktissueonly: bool = False,
        showprogressbar: bool = False,
    ) -> float:
        """
        Apply Gaussian spatial smoothing to the data.

        This function applies a Gaussian spatial filter to the data, with optional
        pre-masking of brain or tissue regions. For CIFTI and text file types, the
        smoothing is skipped by setting `gausssigma` to 0.0. If `gausssigma` is less
        than 0.0, it is automatically set to the mean of the image dimensions divided
        by 2.0.

        Parameters
        ----------
        gausssigma : float
            Standard deviation for the Gaussian kernel. If less than 0.0, it is
            automatically calculated as the mean of image dimensions divided by 2.0.
        brainmask : NDArray | None, optional
            Binary mask for the brain region. Required if `premask` is True and
            `premasktissueonly` is False.
        graymask : NDArray | None, optional
            Binary mask for gray matter. Required if `premask` is True and
            `premasktissueonly` is True.
        whitemask : NDArray | None, optional
            Binary mask for white matter. Required if `premask` is True and
            `premasktissueonly` is True.
        premask : bool, optional
            If True, applies the mask before smoothing. Default is False.
        premasktissueonly : bool, optional
            If True, applies the mask only to gray and white matter. Requires
            `graymask` and `whitemask` to be provided. Default is False.
        showprogressbar : bool, optional
            If True, displays a progress bar during processing. Default is False.

        Returns
        -------
        float
            The actual `gausssigma` value used for smoothing.

        Notes
        -----
        - For CIFTI and text file types, smoothing is skipped.
        - The function modifies `self.nim_data` in-place.
        - If `premask` is True, the mask is applied to the timepoints specified by
          `self.validstart` to `self.validend`.

        Examples
        --------
        >>> # Apply smoothing with automatic sigma
        >>> smooth(-1.0)
        1.5

        >>> # Apply smoothing with a custom sigma and pre-mask
        >>> smooth(2.0, brainmask=mask, premask=True)
        2.0
        """
        # do spatial filtering if requested
        if self.filetype == "cifti" or self.filetype == "text":
            gausssigma = 0.0
        if gausssigma < 0.0:
            # set gausssigma automatically
            gausssigma = np.mean([self.xdim, self.ydim, self.slicethickness]) / 2.0
        if gausssigma > 0.0:
            # premask data if requested
            if premask:
                if premasktissueonly:
                    if (graymask is not None) and (whitemask is not None):
                        multmask = graymask + whitemask
                    else:
                        raise ValueError(
                            "ERROR: graymask and whitemask must be defined to use premasktissueonly - exiting"
                        )
                else:
                    if brainmask is not None:
                        multmask = brainmask
                    else:
                        raise ValueError(
                            "ERROR: brainmask must be defined to use premask - exiting"
                        )
                print(f"premasking timepoints {self.validstart} to {self.validend}")
                for i in tqdm(
                    range(self.validstart, self.validend + 1),
                    desc="Timepoint",
                    unit="timepoints",
                    disable=(not showprogressbar),
                ):
                    self.nim_data[:, :, :, i] *= multmask

            # now apply the filter
            print(
                f"applying gaussian spatial filter to timepoints {self.validstart} "
                f"to {self.validend} with sigma={gausssigma}"
            )
            sourcedata = self.byvol()
            for i in tqdm(
                range(self.validstart, self.validend + 1),
                desc="Timepoint",
                unit="timepoints",
                disable=(not showprogressbar),
            ):
                self.nim_data[:, :, :, i] = tide_filt.ssmooth(
                    self.xdim,
                    self.ydim,
                    self.slicethickness,
                    gausssigma,
                    sourcedata[:, :, :, i],
                )
        return gausssigma

    def summarize(self) -> None:
        """
        Print a comprehensive summary of voxel data properties.

        This method outputs detailed information about the voxel data structure,
        including image dimensions, spatial properties, temporal characteristics,
        and file metadata. The summary includes both geometric and temporal
        parameters that define the voxel space and data organization.

        Notes
        -----
        The method prints to standard output and does not return any value.
        All attributes are accessed from the instance and displayed in a
        formatted manner for easy inspection of the data structure.

        Examples
        --------
        >>> obj.summarize()
        Voxel data summary:
            self.nim=...
            self.nim_data.shape=...
            self.nim_hdr=...
            self.nim_affine=...
            ...
        """
        print("Voxel data summary:")
        print(f"\t{self.nim=}")
        print(f"\t{self.nim_data.shape=}")
        print(f"\t{self.nim_hdr=}")
        print(f"\t{self.nim_affine=}")
        print(f"\t{self.theshape=}")
        print(f"\t{self.xsize=}")
        print(f"\t{self.ysize=}")
        print(f"\t{self.numslices=}")
        print(f"\t{self.timepoints=}")
        print(f"\t{self.timestep=}")
        print(f"\t{self.thesizes=}")
        print(f"\t{self.thedims=}")
        print(f"\t{self.numslicelocs=}")
        print(f"\t{self.numspatiallocs=}")
        print(f"\t{self.nativespaceshape=}")
        print(f"\t{self.cifti_hdr=}")
        print(f"\t{self.filetype=}")
        print(f"\t{self.resident=}")
