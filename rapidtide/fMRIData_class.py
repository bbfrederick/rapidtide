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
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io


class fMRIData:
    header = None
    data = None
    data_byvoxel = None
    data_valid = None
    nativeshape = None
    mask = None
    validvoxels = None
    filename = None
    filetype = None

    def __init__(
        self,
        filename: str | None = None,
        header: Any | None = None,
        data: NDArray | None = None,
        mask: NDArray | None = None,
        validvoxels: NDArray | None = None,
        dims: tuple | None = None,
        sizes: tuple | None = None,
        description: str | None = None,
    ) -> None:
        """
        Initialize a new instance.

        Parameters
        ----------
        filename : str, optional
            Path to the data file. Default is None.
        header : Any, optional
            Header information associated with the data. Default is None.
        data : NDArray, optional
            Main data array. Default is None.
        mask : NDArray, optional
            Mask array for data filtering. Default is None.
        validvoxels : NDArray, optional
            Array indicating valid voxels. Default is None.
        dims : tuple, optional
            Dimensions of the data. Default is None.
        sizes : tuple, optional
            Size information for the data. Default is None.
        description : str, optional
            Description of the data. Default is None.

        Returns
        -------
        None
            This method initializes the instance attributes but does not return anything.

        Notes
        -----
        This constructor sets up the basic attributes for data handling. All parameters
        are optional and can be None, allowing for flexible initialization of instances.

        Examples
        --------
        >>> instance = MyClass()
        >>> instance = MyClass(filename="data.nii", data=np.array([1, 2, 3]))
        """
        self.filename = filename
        self.header = header
        self.data = data
        self.mask = mask
        self.validvoxels = validvoxels
        self.dims = dims
        self.sizes = sizes
        self.description = description

    def load(self, filename: str) -> None:
        """
        Load fMRI data from a file.

        Load fMRI data from a file, supporting text, CIFTI, and NIFTI formats.
        The function determines the file type and reads the data accordingly,
        setting appropriate attributes such as data shape, timepoints, and spatial dimensions.

        Parameters
        ----------
        filename : str
            Path to the input file. Supported formats are text, CIFTI, and NIFTI.

        Returns
        -------
        None
            This function does not return a value but updates the instance attributes
            with loaded data and metadata.

        Notes
        -----
        This function modifies the instance state by setting the following attributes:
        - `filename`: The path to the input file.
        - `filetype`: The type of the file ('text', 'cifti', or 'nifti').
        - `data`: The loaded data array.
        - `header`: The header information (if available).
        - `xsize`, `ysize`, `numslices`, `timepoints`: Spatial and temporal dimensions.
        - `thesizes`: A list containing the dimensions of the data.
        - `numspatiallocs`: Total number of spatial locations.
        - `nativespaceshape`: Shape of the native space.
        - `cifti_hdr`: CIFTI header (if applicable).

        Examples
        --------
        >>> loader = DataReader()
        >>> loader.load('fmri_data.nii.gz')
        >>> print(loader.timepoints)
        100
        """
        self.filename = filename
        ####################################################
        #  Read data
        ####################################################
        # open the fmri datafile
        if tide_io.checkiftext(self.filename):
            self.filetype = "text"
            self.data = tide_io.readvecs(self.filename)
            self.header = None
            theshape = np.shape(nim_data)
            self.xsize = theshape[0]
            self.ysize = 1
            self.numslices = 1
            self.timepoints = theshape[1]
            self.thesizes = [0, int(self.xsize), 1, 1, int(self.timepoints)]
            self.numspatiallocs = int(self.xsize)
            self.nativespaceshape = self.xsize
            self.cifti_hdr = None
        elif tide_io.checkifcifti(self.filename):
            self.filetype = "cifti"
            (
                cifti,
                cifti_hdr,
                self.data,
                self.header,
                thedims,
                thesizes,
                dummy,
            ) = tide_io.readfromcifti(self.filename)
            self.isgrayordinate = True
            self.timepoints = nim_data.shape[1]
            numspatiallocs = nim_data.shape[0]
            LGR.debug(f"cifti file has {timepoints} timepoints, {numspatiallocs} numspatiallocs")
            slicesize = numspatiallocs
            nativespaceshape = (1, 1, 1, 1, numspatiallocs)
        else:
            self.filetype = "nifti"
            LGR.debug("input file is NIFTI")
            nim, self.data, self.header, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
            optiondict["isgrayordinate"] = False
            xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
            numspatiallocs = int(xsize) * int(ysize) * int(numslices)
            cifti_hdr = None
            nativespaceshape = (xsize, ysize, numslices)
            xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
