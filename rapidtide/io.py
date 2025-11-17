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
import json
import operator as op
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rapidtide.tests.utils import mse


# ---------------------------------------- NIFTI file manipulation ---------------------------
def readfromnifti(
    inputfile: str, headeronly: bool = False
) -> Tuple[Any, Optional[NDArray], Any, NDArray, NDArray]:
    """
    Open a nifti file and read in the various important parts

    Parameters
    ----------
    inputfile : str
        The name of the nifti file. Can be provided with or without file extension
        (.nii or .nii.gz).
    headeronly : bool, optional
        If True, only read the header without loading data. Default is False.

    Returns
    -------
    tuple
        A tuple containing:

        - nim : nifti image structure
        - nim_data : array-like or None
          The image data if headeronly=False, None otherwise
        - nim_hdr : nifti header
          The header information copied from the nifti image
        - thedims : int array
          The dimensions from the nifti header
        - thesizes : float array
          The pixel dimensions from the nifti header

    Notes
    -----
    This function automatically detects the file extension (.nii or .nii.gz) if
    not provided in the inputfile parameter. If neither .nii nor .nii.gz extension
    is found, it will look for the file with these extensions in order.

    Examples
    --------
    >>> nim, data, hdr, dims, sizes = readfromnifti('my_image')
    >>> nim, data, hdr, dims, sizes = readfromnifti('my_image.nii.gz', headeronly=True)
    """
    if os.path.isfile(inputfile):
        inputfilename = inputfile
    elif os.path.isfile(f"{inputfile}.nii.gz"):
        inputfilename = f"{inputfile}.nii.gz"
    elif os.path.isfile(f"{inputfile}.nii"):
        inputfilename = f"{inputfile}.nii"
    else:
        raise FileNotFoundError(f"nifti file {inputfile} does not exist")
    nim = nib.load(inputfilename)
    if headeronly:
        nim_data = None
    else:
        nim_data = nim.get_fdata()
    nim_hdr = nim.header.copy()
    thedims = nim_hdr["dim"].copy()
    thesizes = nim_hdr["pixdim"].copy()
    return nim, nim_data, nim_hdr, thedims, thesizes


def readfromcifti(
    inputfile: str, debug: bool = False
) -> Tuple[Any, Any, NDArray, Any, NDArray, NDArray, Optional[float]]:
    """
    Open a cifti file and read in the various important parts

    Parameters
    ----------
    inputfile : str
        The name of the cifti file.
    debug : bool, optional
        Enable debug output. Default is False

    Returns
    -------
    nim : nifti image structure
    nim_data : array-like
    nim_hdr : nifti header
    thedims : int array
    thesizes : float array

    """
    if os.path.isfile(inputfile):
        inputfilename = inputfile
    elif os.path.isfile(f"{inputfile}.nii"):
        inputfilename = f"{inputfile}.nii"
    else:
        raise FileNotFoundError(f"cifti file {inputfile} does not exist")

    cifti = nib.load(inputfilename)
    nifti_data = np.transpose(cifti.get_fdata(dtype=np.float32))
    cifti_hdr = cifti.header
    nifti_hdr = cifti.nifti_header

    if nifti_hdr["intent_code"] == 3002:
        timestep, starttime = getciftitr(cifti_hdr)
    else:
        timestep, starttime = None, None
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    if debug:
        for theaxis in axes:
            print(theaxis)

    thedims = nifti_hdr["dim"].copy()
    thesizes = nifti_hdr["pixdim"].copy()
    return cifti, cifti_hdr, nifti_data, nifti_hdr, thedims, thesizes, timestep


def getciftitr(cifti_hdr: Any) -> Tuple[float, float]:
    """
    Extract the TR (repetition time) from a CIFTI header.

    This function extracts timing information from a CIFTI header, specifically
    the time between timepoints (TR) and the start time of the first timepoint.
    It searches for a SeriesAxis in the CIFTI header matrix to extract this
    information.

    Parameters
    ----------
    cifti_hdr : Any
        The CIFTI header object containing timing information. This should be
        a valid CIFTI header that supports the matrix.mapped_indices and
        matrix.get_axis methods.

    Returns
    -------
    tuple of (float, float)
        A tuple containing:
        - timestep : float
          The TR (time between timepoints) in seconds
        - starttime : float
          The start time of the first timepoint in seconds

    Raises
    ------
    SystemExit
        If no SeriesAxis is found in the CIFTI header, the function will
        print an error message and exit the program.

    Notes
    -----
    The function specifically looks for a SeriesAxis in the CIFTI header's
    matrix. If multiple SeriesAxes exist, only the first one encountered
    will be used. The timing information is extracted using the get_element()
    method on the SeriesAxis object.

    Examples
    --------
    >>> import nibabel as nib
    >>> cifti_hdr = nib.load('file.cifti').header
    >>> tr, start_time = getciftitr(cifti_hdr)
    >>> print(f"TR: {tr} seconds, Start time: {start_time} seconds")
    TR: 0.8 seconds, Start time: 0.0 seconds
    """
    seriesaxis = None
    for theaxis in cifti_hdr.matrix.mapped_indices:
        if isinstance(cifti_hdr.matrix.get_axis(theaxis), nib.cifti2.SeriesAxis):
            seriesaxis = theaxis
    if seriesaxis is not None:
        timepoint1 = cifti_hdr.matrix.get_axis(seriesaxis).get_element(1)
        starttime = cifti_hdr.matrix.get_axis(seriesaxis).get_element(0)
        timestep = timepoint1 - starttime
    else:
        print("No series axis found!  Exiting")
        sys.exit()
    return timestep, starttime


# dims are the array dimensions along each axis
def parseniftidims(thedims: NDArray) -> Tuple[int, int, int, int]:
    """
    Split the dims array into individual elements

    This function extracts the dimension sizes from a NIfTI dimensions array,
    returning the number of points along each spatial and temporal dimension.

    Parameters
    ----------
    thedims : NDArray of int
        The NIfTI dimensions structure, where:
        - thedims[0] contains the data type
        - thedims[1] contains the number of points along x-axis (nx)
        - thedims[2] contains the number of points along y-axis (ny)
        - thedims[3] contains the number of points along z-axis (nz)
        - thedims[4] contains the number of points along t-axis (nt)

    Returns
    -------
    nx : int
        Number of points along the x-axis
    ny : int
        Number of points along the y-axis
    nz : int
        Number of points along the z-axis
    nt : int
        Number of points along the t-axis (time)

    Notes
    -----
    The input array is expected to be a NIfTI dimensions array with at least 5 elements.
    This function assumes the standard NIfTI dimension ordering where dimensions 1-4
    correspond to spatial x, y, z, and temporal t dimensions respectively.

    Examples
    --------
    >>> import numpy as np
    >>> dims = np.array([0, 64, 64, 32, 100, 1, 1, 1])
    >>> nx, ny, nz, nt = parseniftidims(dims)
    >>> print(f"Dimensions: {nx} x {ny} x {nz} x {nt}")
    Dimensions: 64 x 64 x 32 x 100
    """
    return int(thedims[1]), int(thedims[2]), int(thedims[3]), int(thedims[4])


# sizes are the mapping between voxels and physical coordinates
def parseniftisizes(thesizes: NDArray) -> Tuple[float, float, float, float]:
    """
    Split the size array into individual elements

    This function extracts voxel size information from a NIfTI header structure
    and returns the scaling factors for spatial dimensions (x, y, z) and time (t).

    Parameters
    ----------
    thesizes : NDArray of float
        The NIfTI voxel size structure containing scaling information.
        Expected to be an array where indices 1-4 correspond to
        x, y, z, and t scaling factors respectively.

    Returns
    -------
    dimx : float
        Scaling factor from voxel number to physical coordinates in x dimension
    dimy : float
        Scaling factor from voxel number to physical coordinates in y dimension
    dimz : float
        Scaling factor from voxel number to physical coordinates in z dimension
    dimt : float
        Scaling factor from voxel number to physical coordinates in t dimension

    Notes
    -----
    The function assumes the input array follows the NIfTI standard where:
    - Index 0: unused or padding
    - Index 1: x-dimension scaling
    - Index 2: y-dimension scaling
    - Index 3: z-dimension scaling
    - Index 4: t-dimension scaling

    Examples
    --------
    >>> import numpy as np
    >>> sizes = np.array([0.0, 2.0, 2.0, 2.0, 1.0])
    >>> x, y, z, t = parseniftisizes(sizes)
    >>> print(x, y, z, t)
    2.0 2.0 2.0 1.0
    """
    return thesizes[1], thesizes[2], thesizes[3], thesizes[4]


def dumparraytonifti(thearray: NDArray, filename: str) -> None:
    """
    Save a numpy array to a NIFTI file with an identity affine transform.

    This function saves a numpy array to a NIFTI file format with an identity
    affine transformation matrix. The resulting NIFTI file will have unit
    spacing and no rotation or translation.

    Parameters
    ----------
    thearray : NDArray
        The data array to save. Can be 2D, 3D, or 4D array representing
        medical imaging data or other volumetric data.
    filename : str
        The output filename (without extension). The function will append
        '.nii' or '.nii.gz' extension based on the nibabel library's
        default behavior.

    Returns
    -------
    None
        This function does not return any value. It saves the array to disk
        as a NIFTI file.

    Notes
    -----
    - The function uses an identity affine matrix with dimensions 4x4
    - The affine matrix represents unit spacing with no rotation or translation
    - This is useful for simple data storage without spatial information
    - The function relies on the `savetonifti` helper function for the actual
      NIFTI file writing operation

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 64)
    >>> dumparraytonifti(data, 'my_data')
    >>> # Creates 'my_data.nii' file with identity affine transform
    """
    outputaffine = np.zeros((4, 4), dtype=float)
    for i in range(4):
        outputaffine[i, i] = 1.0
    outputheader = nib.nifti1Header
    outputheader.set_affine(outputaffine)
    savetonifti(thearray, outputheader, filename)


def savetonifti(thearray: NDArray, theheader: Any, thename: str, debug: bool = False) -> None:
    """
    Save a data array out to a nifti file

    Parameters
    ----------
    thearray : array-like
        The data array to save.
    theheader : nifti header
        A valid nifti header
    thename : str
        The name of the nifti file to save
    debug : bool, optional
        Enable debug output. Default is False

    Returns
    -------
    None
    """
    outputaffine = theheader.get_best_affine()
    qaffine, qcode = theheader.get_qform(coded=True)
    saffine, scode = theheader.get_sform(coded=True)
    thedtype = thearray.dtype
    if thedtype == np.uint8:
        thedatatypecode = 2
        thebitpix = 8
    elif thedtype == np.int16:
        thedatatypecode = 4
        thebitpix = 16
    elif thedtype == np.int32:
        thedatatypecode = 8
        thebitpix = 32
    elif thedtype == np.float32:
        thedatatypecode = 16
        thebitpix = 32
    elif thedtype == np.complex64:
        thedatatypecode = 32
        thebitpix = 64
    elif thedtype == np.float64:
        thedatatypecode = 64
        thebitpix = 64
    elif thedtype == np.int8:
        thedatatypecode = 256
        thebitpix = 8
    elif thedtype == np.uint16:
        thedatatypecode = 512
        thebitpix = 16
    elif thedtype == np.uint32:
        thedatatypecode = 768
        thebitpix = 32
    elif thedtype == np.int64:
        thedatatypecode = 1024
        thebitpix = 64
    elif thedtype == np.uint64:
        thedatatypecode = 1280
        thebitpix = 64
    # elif thedtype == np.float128:
    #        thedatatypecode = 1536
    #        thebitpix = 128
    elif thedtype == np.complex128:
        thedatatypecode = 1792
        thebitpix = 128
    # elif thedtype == np.complex256:
    #    thedatatypecode = 2048
    #    thebitpix = 256
    else:
        raise TypeError("type", thedtype, "is not legal")
    theheader["datatype"] = thedatatypecode
    theheader["bitpix"] = thebitpix
    if debug:
        print(f"savetonifti:")
        print(f"\tinput data array is type {thedtype}")
        targetdatatype = theheader["datatype"]
        print(f"\ttargetdatatype={targetdatatype}")

    if theheader["magic"] == "n+2":
        output_nifti = nib.Nifti2Image(thearray.astype(thedtype), outputaffine, header=theheader)
        suffix = ".nii"
    else:
        output_nifti = nib.Nifti1Image(thearray.astype(thedtype), outputaffine, header=theheader)
        suffix = ".nii.gz"
    output_nifti.set_qform(qaffine, code=int(qcode))
    output_nifti.set_sform(saffine, code=int(scode))

    output_nifti.to_filename(thename + suffix)
    output_nifti = None


def niftifromarray(data: NDArray) -> Any:
    """
    Create a NIFTI image object from a numpy array with identity affine.

    This function converts a numpy array into a NIFTI image object using an identity
    affine transformation matrix. The resulting image has no spatial transformation
    applied, meaning the voxel coordinates directly correspond to the array indices.

    Parameters
    ----------
    data : NDArray
        The data array to convert to NIFTI format. Can be 2D, 3D, or 4D array
        representing image data with arbitrary data types.

    Returns
    -------
    nibabel.Nifti1Image
        The NIFTI image object with identity affine matrix. The returned object
        can be saved to disk using nibabel's save functionality.

    Notes
    -----
    - The affine matrix is set to identity (4x4), which means no spatial
      transformation is applied
    - This function is useful for creating NIFTI images from processed data
      that doesn't require spatial registration
    - The data array is copied into the NIFTI image object

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 32)
    >>> img = niftifromarray(data)
    >>> print(img.shape)
    (64, 64, 32)
    >>> print(img.affine)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """
    return nib.Nifti1Image(data, affine=np.eye(4))


def niftihdrfromarray(data: NDArray) -> Any:
    """
    Create a NIFTI header from a numpy array with identity affine.

    This function creates a NIFTI header object from a numpy array by constructing
    a minimal NIFTI image with an identity affine matrix and extracting its header.
    The resulting header contains basic NIFTI metadata but no spatial transformation
    information beyond the identity matrix.

    Parameters
    ----------
    data : NDArray
        The data array to create a header for. The array can be of any shape and
        data type, but should typically represent medical imaging data.

    Returns
    -------
    nibabel.Nifti1Header
        The NIFTI header object containing metadata for the input data array.

    Notes
    -----
    The returned header is a copy of the header from a NIFTI image with identity
    affine matrix. This is useful for creating NIFTI headers without requiring
    full NIFTI image files or spatial transformation information.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 64)
    >>> header = niftihdrfromarray(data)
    >>> print(header)
    <nibabel.nifti1.Nifti1Header object at 0x...>
    """
    return nib.Nifti1Image(data, affine=np.eye(4)).header.copy()


def makedestarray(
    destshape: Union[Tuple, NDArray],
    filetype: str = "nifti",
    rt_floattype: np.dtype = np.dtype(np.float64),
) -> Tuple[NDArray, int]:
    """
    Create a destination array for output data based on file type and shape.

    Parameters
    ----------
    destshape : tuple or numpy array
        Shape specification for the output array. For 'nifti' files, this is expected
        to be a 3D or 4D shape; for 'cifti', it is expected to be a 2D or 3D shape
        where the last dimension corresponds to spatial data and the second-to-last
        to time; for 'text', it is expected to be a 1D or 2D shape.
    filetype : str, optional
        Type of output file. Must be one of 'nifti', 'cifti', or 'text'. Default is 'nifti'.
    rt_floattype : np.dtype, optional
        Data type for the output array. Default is 'np.float64'.

    Returns
    -------
    outmaparray : numpy array
        Pre-allocated output array with appropriate shape and dtype. The shape depends
        on the `filetype` and `destshape`:
        - For 'nifti': 1D array if 3D input, 2D array if 4D input.
        - For 'cifti': 1D or 2D array depending on time dimension.
        - For 'text': 1D or 2D array depending on time dimension.
    internalspaceshape : int
        The flattened spatial dimension size used to determine the shape of the output array.

    Notes
    -----
    This function handles different file types by interpreting the input `destshape`
    differently:
    - For 'nifti', the spatial dimensions are multiplied together to form the
      `internalspaceshape`, and the time dimension is inferred from the fourth
      axis if present.
    - For 'cifti', the last dimension is treated as spatial, and the second-to-last
      as temporal if it exceeds 1.
    - For 'text', the first dimension is treated as spatial, and the second as time.

    Examples
    --------
    >>> import numpy as np
    >>> from typing import Tuple, Union
    >>> makedestarray((64, 64, 32), filetype="nifti")
    (array([0., 0., ..., 0.]), 2097152)

    >>> makedestarray((100, 50), filetype="text")
    (array([0., 0., ..., 0.]), 100)

    >>> makedestarray((100, 50, 20), filetype="cifti")
    (array([[0., 0., ..., 0.], ..., [0., 0., ..., 0.]]), 20)
    """
    if filetype == "text":
        try:
            internalspaceshape = destshape[0]
            timedim = destshape[1]
        except TypeError:
            internalspaceshape = destshape
            timedim = None
    elif filetype == "cifti":
        spaceindex = len(destshape) - 1
        timeindex = spaceindex - 1
        internalspaceshape = destshape[spaceindex]
        if destshape[timeindex] > 1:
            timedim = destshape[timeindex]
        else:
            timedim = None
    else:
        internalspaceshape = int(destshape[0]) * int(destshape[1]) * int(destshape[2])
        if len(destshape) == 3:
            timedim = None
        else:
            timedim = destshape[3]
    if timedim is None:
        outmaparray = np.zeros(internalspaceshape, dtype=rt_floattype)
    else:
        outmaparray = np.zeros((internalspaceshape, timedim), dtype=rt_floattype)
    return outmaparray, internalspaceshape


def populatemap(
    themap: NDArray,
    internalspaceshape: int,
    validvoxels: Optional[NDArray],
    outmaparray: NDArray,
    debug: bool = False,
) -> NDArray:
    """
    Populate an output array with data from a map, handling valid voxel masking.

    This function populates an output array with data from a source map, optionally
    masking invalid voxels. It supports both 1D and 2D output arrays.

    Parameters
    ----------
    themap : NDArray
        The source data to populate into the output array. Shape is either
        ``(internalspaceshape,)`` for 1D or ``(internalspaceshape, N)`` for 2D.
    internalspaceshape : int
        The total spatial dimension size, used to determine the expected shape
        of the input map and the output array.
    validvoxels : NDArray or None
        Indices of valid voxels to populate. If None, all voxels are populated.
        Shape should be ``(M,)`` where M is the number of valid voxels.
    outmaparray : NDArray
        The destination array to populate. Shape should be either ``(internalspaceshape,)``
        for 1D or ``(internalspaceshape, N)`` for 2D.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    NDArray
        The populated output array with the same shape as `outmaparray`.

    Notes
    -----
    - If `validvoxels` is provided, only the specified voxels are updated.
    - The function modifies `outmaparray` in-place and returns it.
    - For 2D arrays, the second dimension is preserved in the output.

    Examples
    --------
    >>> import numpy as np
    >>> themap = np.array([1, 2, 3, 4])
    >>> outmaparray = np.zeros(4)
    >>> validvoxels = np.array([0, 2])
    >>> result = populatemap(themap, 4, validvoxels, outmaparray)
    >>> print(result)
    [1. 0. 3. 0.]

    >>> outmaparray = np.zeros((4, 2))
    >>> result = populatemap(themap.reshape((4, 1)), 4, None, outmaparray)
    >>> print(result)
    [[1.]
     [2.]
     [3.]
     [4.]]
    """
    if len(outmaparray.shape) == 1:
        outmaparray[:] = 0.0
        if validvoxels is not None:
            outmaparray[validvoxels] = themap[:].reshape((np.shape(validvoxels)[0]))
        else:
            outmaparray = themap[:].reshape((internalspaceshape))
    else:
        outmaparray[:, :] = 0.0
        if validvoxels is not None:
            outmaparray[validvoxels, :] = themap[:, :].reshape(
                (np.shape(validvoxels)[0], outmaparray.shape[1])
            )
        else:
            outmaparray = themap[:, :].reshape((internalspaceshape, outmaparray.shape[1]))
    if debug:
        print(f"populatemap: output array shape is {outmaparray.shape}")
    return outmaparray


def savemaplist(
    outputname: str,
    maplist: List[Tuple],
    validvoxels: Optional[NDArray],
    destshape: Union[Tuple, NDArray],
    theheader: Any,
    bidsbasedict: Dict[str, Any],
    filetype: str = "nifti",
    rt_floattype: np.dtype = np.dtype(np.float64),
    cifti_hdr: Optional[Any] = None,
    savejson: bool = True,
    debug: bool = False,
) -> None:
    """
    Save a list of data maps to files with appropriate BIDS metadata.

    This function saves a list of data maps to output files (NIfTI, CIFTI, or text)
    using the specified file type and includes BIDS-compliant metadata in JSON sidecars.
    It supports mapping data into a destination array, handling valid voxels, and
    writing out the final files with appropriate naming and headers.

    Parameters
    ----------
    outputname : str
        Base name for output files (without extension).
    maplist : list of tuples
        List of (data, suffix, maptype, unit, description) tuples to save.
        Each tuple corresponds to one map to be saved.
    validvoxels : numpy array or None
        Indices of valid voxels in the data. If None, all voxels are considered valid.
    destshape : tuple or numpy array
        Shape of the destination array into which data will be mapped.
    theheader : nifti/cifti header
        Header object for the output files (NIfTI or CIFTI).
    bidsbasedict : dict
        Base BIDS metadata to include in JSON sidecars.
    filetype : str, optional
        Output file type ('nifti', 'cifti', or 'text'). Default is 'nifti'.
    rt_floattype : str, optional
        Data type for output arrays. Default is 'float64'.
    cifti_hdr : cifti header or None, optional
        CIFTI header if filetype is 'cifti'. Default is None.
    savejson : bool, optional
        Whether to save JSON sidecar files. Default is True.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    None
        This function does not return any value; it writes files to disk.

    Notes
    -----
    - For CIFTI files, if the data is a series (multi-dimensional), it is saved with
      the provided names; otherwise, it uses temporal offset and step information.
    - The function uses `makedestarray` to prepare the output array and `populatemap`
      to copy data into the array based on valid voxels.
    - If `savejson` is True, a JSON file is created for each map with metadata
      including unit and description.

    Examples
    --------
    >>> savemaplist(
    ...     outputname="sub-01_task-rest",
    ...     maplist=[
    ...         (data1, "stat", "stat", "z", "Statistical map"),
    ...         (data2, "mask", "mask", None, "Binary mask"),
    ...     ],
    ...     validvoxels=valid_indices,
    ...     destshape=(100, 100, 100),
    ...     theheader=nifti_header,
    ...     bidsbasedict={"Dataset": "MyDataset"},
    ...     filetype="nifti",
    ...     savejson=True,
    ... )
    """
    outmaparray, internalspaceshape = makedestarray(
        destshape,
        filetype=filetype,
        rt_floattype=rt_floattype,
    )
    if debug:
        print("maplist:")
        print(maplist)
    for themap, mapsuffix, maptype, theunit, thedescription in maplist:
        # copy the data into the output array, remapping if warranted
        if debug:
            print(f"processing map {mapsuffix}")
            if validvoxels is None:
                print(f"savemaplist: saving {mapsuffix} of shape {themap.shape} to {destshape}")
            else:
                print(
                    f"savemaplist: saving {mapsuffix} of shape {themap.shape} to {destshape} from {np.shape(validvoxels)[0]} valid voxels"
                )
        outmaparray = populatemap(
            themap,
            internalspaceshape,
            validvoxels,
            outmaparray.astype(themap.dtype),
            debug=False,
        )

        # actually write out the data
        bidsdict = bidsbasedict.copy()
        if theunit is not None:
            bidsdict["Units"] = theunit
        if thedescription is not None:
            bidsdict["Description"] = thedescription
        if filetype == "text":
            writenpvecs(
                outmaparray.reshape(destshape),
                f"{outputname}_{mapsuffix}.txt",
            )
        else:
            savename = f"{outputname}_desc-{mapsuffix}_{maptype}"
            if savejson:
                writedicttojson(bidsdict, savename + ".json")
            if filetype == "nifti":
                savetonifti(outmaparray.reshape(destshape), theheader, savename)
            else:
                isseries = len(outmaparray.shape) != 1
                if isseries:
                    savetocifti(
                        outmaparray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=isseries,
                        names=[mapsuffix],
                    )
                else:
                    savetocifti(
                        outmaparray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=isseries,
                        start=theheader["toffset"],
                        step=theheader["pixdim"][4],
                    )


def savetocifti(
    thearray: NDArray,
    theciftiheader: Any,
    theniftiheader: Any,
    thename: str,
    isseries: bool = False,
    names: List[str] = ["placeholder"],
    start: float = 0.0,
    step: float = 1.0,
    debug: bool = False,
) -> None:
    """
    Save a data array out to a CIFTI file.

    This function saves a given data array to a CIFTI file (either dense or parcellated,
    scalar or series) based on the provided headers and parameters.

    Parameters
    ----------
    thearray : array-like
        The data array to be saved. The shape is expected to be (n_timepoints, n_vertices)
        or (n_vertices,) for scalar data.
    theciftiheader : cifti header
        A valid CIFTI header object containing axis information, including BrainModelAxis
        or ParcelsAxis.
    theniftiheader : nifti header
        A valid NIfTI header object to be used for setting the intent of the output file.
    thename : str
        The base name of the output CIFTI file (without extension).
    isseries : bool, optional
        If True, the output will be a time series file (dtseries or ptseries).
        If False, it will be a scalar file (dscalar or pscalar). Default is False.
    names : list of str, optional
        Names for scalar maps when `isseries` is False. Default is ['placeholder'].
    start : float, optional
        Start time in seconds for the time series. Default is 0.0.
    step : float, optional
        Time step in seconds for the time series. Default is 1.0.
    debug : bool, optional
        If True, print debugging information. Default is False.

    Returns
    -------
    None
        This function does not return anything; it saves the file to disk.

    Notes
    -----
    The function automatically detects whether the input CIFTI header contains a
    BrainModelAxis or a ParcelsAxis and builds the appropriate output structure.
    The correct CIFTI file extension (e.g., .dtseries.nii, .dscalar.nii) is appended
    to the output filename based on the `isseries` and parcellation flags.

    Examples
    --------
    >>> import numpy as np
    >>> import nibabel as nib
    >>> data = np.random.rand(100, 50)
    >>> cifti_header = nib.load('input.cifti').header
    >>> nifti_header = nib.load('input.nii').header
    >>> savetocifti(data, cifti_header, nifti_header, 'output', isseries=True)
    """
    if debug:
        print("savetocifti:", thename)
    workingarray = np.transpose(thearray)
    if len(workingarray.shape) == 1:
        workingarray = workingarray.reshape((1, -1))

    # find the ModelAxis from the input file
    modelaxis = None
    for theaxis in theciftiheader.matrix.mapped_indices:
        if isinstance(theciftiheader.matrix.get_axis(theaxis), nib.cifti2.BrainModelAxis):
            modelaxis = theaxis
            parcellated = False
            if debug:
                print("axis", theaxis, "is the BrainModelAxis")
        elif isinstance(theciftiheader.matrix.get_axis(theaxis), nib.cifti2.ParcelsAxis):
            modelaxis = theaxis
            parcellated = True
            if debug:
                print("axis", theaxis, "is the ParcelsAxis")

    # process things differently for dscalar and dtseries files
    if isseries:
        # make a proper series header
        if parcellated:
            if debug:
                print("ptseries path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES"
            theintentname = "ConnParcelSries"
        else:
            if debug:
                print("dtseries path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES"
            theintentname = "ConnDenseSeries"
        if modelaxis is not None:
            seriesaxis = nib.cifti2.cifti2_axes.SeriesAxis(start, step, workingarray.shape[0])
            axislist = [seriesaxis, theciftiheader.matrix.get_axis(modelaxis)]
        else:
            raise KeyError("no ModelAxis found in source file - exiting")
    else:
        # make a proper scalar header
        if parcellated:
            if debug:
                print("pscalar path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SCALAR"
            theintentname = "ConnParcelScalr"
        else:
            if debug:
                print("dscalar path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS"
            theintentname = "ConnDenseScalar"
        if len(names) != workingarray.shape[0]:
            print("savetocifti - number of supplied names does not match array size - exiting.")
            sys.exit()
        if modelaxis is not None:
            scalaraxis = nib.cifti2.cifti2_axes.ScalarAxis(names)
            axislist = [scalaraxis, theciftiheader.matrix.get_axis(modelaxis)]
        else:
            raise KeyError("no ModelAxis found in source file - exiting")
    # now create the output file structure
    if debug:
        print("about to create cifti image - nifti header is:", theniftiheader)

    img = nib.cifti2.Cifti2Image(dataobj=workingarray, header=axislist)

    # make the header right
    img.nifti_header.set_intent(theintent, name=theintentname)
    img.update_headers()

    if isseries:
        if parcellated:
            suffix = ".ptseries.nii"
            if debug:
                print("\tPARCELLATED_SERIES")
        else:
            suffix = ".dtseries.nii"
            if debug:
                print("\tDENSE_SERIES")
    else:
        if parcellated:
            suffix = ".pscalar.nii"
            if debug:
                print("\tPARCELLATED_SCALARS")
        else:
            suffix = ".dscalar.nii"
            if debug:
                print("\tDENSE_SCALARS")

    if debug:
        print("after update_headers() - nifti header is:", theniftiheader)

    # save the data
    nib.cifti2.save(img, thename + suffix)


def checkifnifti(filename: str) -> bool:
    """
    Check to see if a file name is a valid nifti name.

    This function determines whether a given filename has a valid NIfTI file extension.
    NIfTI files typically have extensions ".nii" or ".nii.gz" for compressed files.

    Parameters
    ----------
    filename : str
        The file name to check for valid NIfTI extension.

    Returns
    -------
    bool
        True if the filename ends with ".nii" or ".nii.gz", False otherwise.

    Notes
    -----
    This function only checks the file extension and does not verify if the file actually exists
    or contains valid NIfTI data. It performs a simple string matching operation.

    Examples
    --------
    >>> checkifnifti("image.nii")
    True
    >>> checkifnifti("data.nii.gz")
    True
    >>> checkifnifti("scan.json")
    False
    >>> checkifnifti("volume.nii.gz")
    True
    """
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return True
    else:
        return False


def niftisplitext(filename: str) -> Tuple[str, str]:
    """
    Split nifti filename into name base and extension.

    This function splits a NIfTI filename into its base name and extension components.
    It handles NIfTI files that may have double extensions (e.g., '.nii.gz') by properly
    combining the extensions.

    Parameters
    ----------
    filename : str
        The NIfTI file name to split, which may contain double extensions like '.nii.gz'

    Returns
    -------
    tuple[str, str]
        A tuple containing:
        - name : str
          Base name of the NIfTI file (without extension)
        - ext : str
          Extension of the NIfTI file (including any additional extensions)

    Notes
    -----
    This function is specifically designed for NIfTI files which commonly have
    double extensions (e.g., '.nii.gz', '.nii.bz2'). It properly handles these
    cases by combining the two extension components.

    Examples
    --------
    >>> niftisplitext('image.nii.gz')
    ('image', '.nii.gz')

    >>> niftisplitext('data.nii')
    ('data', '.nii')

    >>> niftisplitext('volume.nii.bz2')
    ('volume', '.nii.bz2')
    """
    firstsplit = os.path.splitext(filename)
    secondsplit = os.path.splitext(firstsplit[0])
    if secondsplit[1] is not None:
        return secondsplit[0], secondsplit[1] + firstsplit[1]
    else:
        return firstsplit[0], firstsplit[1]


def niftisplit(inputfile: str, outputroot: str, axis: int = 3) -> None:
    """
    Split a NIFTI file along a specified axis into separate files.

    This function splits a NIFTI image along a given axis into multiple
    individual NIFTI files, each corresponding to a slice along that axis.
    The output files are named using the provided root name with zero-padded
    slice indices.

    Parameters
    ----------
    inputfile : str
        Path to the input NIFTI file to be split.
    outputroot : str
        Base name for the output files. Each output file will be named
        ``outputroot + str(i).zfill(4)`` where ``i`` is the slice index.
    axis : int, optional
        Axis along which to split the NIFTI file. Valid values are 0-4,
        corresponding to the dimensions of the NIFTI file. Default is 3,
        which corresponds to the time axis in 4D or 5D NIFTI files.

    Returns
    -------
    None
        This function does not return any value. It writes the split slices
        as separate NIFTI files to disk.

    Notes
    -----
    - The function supports both 4D and 5D NIFTI files.
    - The header information is preserved for each output slice, with the
      dimension along the split axis set to 1.
    - Slice indices in the output file names are zero-padded to four digits
      (e.g., ``0000``, ``0001``, etc.).

    Examples
    --------
    >>> niftisplit('input.nii.gz', 'slice_', axis=2)
    Splits the input NIFTI file along the third axis (axis=2) and saves
    the resulting slices as ``slice_0000.nii.gz``, ``slice_0001.nii.gz``, etc.
    """
    infile, infile_data, infile_hdr, infiledims, infilesizes = readfromnifti(inputfile)
    theheader = copy.deepcopy(infile_hdr)
    numpoints = infiledims[axis + 1]
    print(infiledims)
    theheader["dim"][axis + 1] = 1
    if infile_data is not None:
        for i in range(numpoints):
            if infiledims[0] == 5:
                if axis == 0:
                    thisslice = infile_data[i : i + 1, :, :, :, :]
                elif axis == 1:
                    thisslice = infile_data[:, i : i + 1, :, :, :]
                elif axis == 2:
                    thisslice = infile_data[:, :, i : i + 1, :, :]
                elif axis == 3:
                    thisslice = infile_data[:, :, :, i : i + 1, :]
                elif axis == 4:
                    thisslice = infile_data[:, :, :, :, i : i + 1]
                else:
                    raise ValueError("illegal axis")
            elif infiledims[0] == 4:
                if axis == 0:
                    thisslice = infile_data[i : i + 1, :, :, :]
                elif axis == 1:
                    thisslice = infile_data[:, i : i + 1, :, :]
                elif axis == 2:
                    thisslice = infile_data[:, :, i : i + 1, :]
                elif axis == 3:
                    thisslice = infile_data[:, :, :, i : i + 1]
                else:
                    raise ValueError("illegal axis")
            savetonifti(thisslice, theheader, outputroot + str(i).zfill(4))
    else:
        raise ValueError("file contains no data!")


def niftimerge(
    inputlist: List[str],
    outputname: str,
    writetodisk: bool = True,
    axis: int = 3,
    returndata: bool = False,
    debug: bool = False,
) -> Optional[Tuple[NDArray, Any]]:
    """
    Merge multiple NIFTI files along a specified axis.

    This function reads a list of NIFTI files, concatenates their data along a
    specified axis, and optionally writes the result to a new NIFTI file. It can
    also return the merged data and header for further processing.

    Parameters
    ----------
    inputlist : list of str
        List of input NIFTI file paths to merge.
    outputname : str
        Path for the merged output NIFTI file.
    writetodisk : bool, optional
        If True, write the merged data to disk. Default is True.
    axis : int, optional
        Axis along which to concatenate the data (0-4). Default is 3, which
        corresponds to the time axis. The dimension of the output along this
        axis will be the number of input files.
    returndata : bool, optional
        If True, return the merged data array and header. Default is False.
    debug : bool, optional
        If True, print debug information during execution. Default is False.

    Returns
    -------
    tuple of (NDArray, Any) or None
        If `returndata` is True, returns a tuple of:
            - `output_data`: The merged NIFTI data as a numpy array.
            - `infile_hdr`: The header from the last input file.
        If `returndata` is False, returns None.

    Notes
    -----
    - The function assumes all input files have compatible dimensions except
      along the concatenation axis.
    - If the input file has 3D dimensions, it is reshaped to 4D before concatenation.
    - The output NIFTI header is updated to reflect the new dimension along the
      concatenation axis.

    Examples
    --------
    >>> input_files = ['file1.nii', 'file2.nii', 'file3.nii']
    >>> niftimerge(input_files, 'merged.nii', axis=3, writetodisk=True)
    >>> data, header = niftimerge(input_files, 'merged.nii', returndata=True)
    """
    inputdata = []
    for thefile in inputlist:
        if debug:
            print("reading", thefile)
        infile, infile_data, infile_hdr, infiledims, infilesizes = readfromnifti(thefile)
        if infiledims[0] == 3:
            inputdata.append(
                infile_data.reshape((infiledims[1], infiledims[2], infiledims[3], 1)) + 0.0
            )
        else:
            inputdata.append(infile_data + 0.0)
    theheader = copy.deepcopy(infile_hdr)
    theheader["dim"][axis + 1] = len(inputdata)
    output_data = np.concatenate(inputdata, axis=axis)
    if writetodisk:
        savetonifti(output_data, theheader, outputname)
    if returndata:
        return output_data, infile_hdr
    else:
        return None


def niftiroi(inputfile: str, outputfile: str, startpt: int, numpoints: int) -> None:
    """
    Extract a region of interest (ROI) from a NIFTI file along the time axis.

    This function extracts a specified number of timepoints from a NIFTI file starting
    at a given timepoint index. The extracted data is saved to a new NIFTI file.

    Parameters
    ----------
    inputfile : str
        Path to the input NIFTI file
    outputfile : str
        Path for the output ROI file
    startpt : int
        Starting timepoint index (0-based)
    numpoints : int
        Number of timepoints to extract

    Returns
    -------
    None
        This function does not return any value but saves the extracted ROI to the specified output file.

    Notes
    -----
    The function handles both 4D and 5D NIFTI files. For 5D files, the function preserves
    the fifth dimension in the output. The time dimension is reduced according to the
    specified number of points.

    Examples
    --------
    >>> niftiroi('input.nii', 'output.nii', 10, 50)
    Extracts timepoints 10-59 from input.nii and saves to output.nii
    """
    print(inputfile, outputfile, startpt, numpoints)
    infile, infile_data, infile_hdr, infiledims, infilesizes = readfromnifti(inputfile)
    theheader = copy.deepcopy(infile_hdr)
    theheader["dim"][4] = numpoints
    if infiledims[0] == 5:
        output_data = infile_data[:, :, :, startpt : startpt + numpoints, :]
    else:
        output_data = infile_data[:, :, :, startpt : startpt + numpoints]
    savetonifti(output_data, theheader, outputfile)


def checkifcifti(filename: str, debug: bool = False) -> bool:
    """
    Check to see if the specified file is CIFTI format

    This function determines whether a given neuroimaging file is in CIFTI (Connectivity Information Format)
    by examining the file's header information. CIFTI files have specific intent codes that distinguish them
    from other neuroimaging formats like NIFTI.

    Parameters
    ----------
    filename : str
        The path to the file to be checked for CIFTI format
    debug : bool, optional
        Enable debug output to see intermediate processing information. Default is False

    Returns
    -------
    bool
        True if the file header indicates this is a CIFTI file (intent code between 3000 and 3099),
        False otherwise

    Notes
    -----
    CIFTI files are identified by their intent code, which should be in the range [3000, 3100) for valid
    CIFTI format files. This function uses nibabel to load the file and examine its NIfTI header properties.

    Examples
    --------
    >>> is_cifti = checkifcifti('my_data.nii.gz')
    >>> print(is_cifti)
    True

    >>> is_cifti = checkifcifti('my_data.nii.gz', debug=True)
    >>> print(is_cifti)
    True
    """
    theimg = nib.load(filename)
    thedict = vars(theimg)
    if debug:
        print("thedict:", thedict)
    try:
        intent = thedict["_nifti_header"]["intent_code"]
        if debug:
            print("intent found")
        return intent >= 3000 and intent < 3100
    except KeyError:
        if debug:
            print("intent not found")
        return False


def checkiftext(filename: str) -> bool:
    """
    Check to see if the specified filename ends in '.txt'

    This function determines whether a given filename has a '.txt' extension
    by checking if the string ends with the specified suffix.

    Parameters
    ----------
    filename : str
        The file name to check for '.txt' extension

    Returns
    -------
    bool
        True if filename ends with '.txt', False otherwise

    Notes
    -----
    This function performs a case-sensitive check. For case-insensitive
    checking, convert the filename to lowercase before calling this function.

    Examples
    --------
    >>> checkiftext("document.txt")
    True
    >>> checkiftext("image.jpg")
    False
    >>> checkiftext("notes.TXT")
    False
    """
    if filename.endswith(".txt"):
        return True
    else:
        return False


def getniftiroot(filename: str) -> str:
    """
    Strip a nifti filename down to the root with no extensions.

    This function removes NIfTI file extensions (.nii or .nii.gz) from a filename,
    returning only the root name without any extensions.

    Parameters
    ----------
    filename : str
        The NIfTI filename to strip of extensions

    Returns
    -------
    str
        The filename without NIfTI extensions (.nii or .nii.gz)

    Notes
    -----
    This function only removes the standard NIfTI extensions (.nii and .nii.gz).
    For filenames without these extensions, the original filename is returned unchanged.

    Examples
    --------
    >>> getniftiroot("sub-01_task-rest_bold.nii")
    'sub-01_task-rest_bold'

    >>> getniftiroot("anatomical.nii.gz")
    'anatomical'

    >>> getniftiroot("image.nii.gz")
    'image'

    >>> getniftiroot("data.txt")
    'data.txt'
    """
    if filename.endswith(".nii"):
        return filename[:-4]
    elif filename.endswith(".nii.gz"):
        return filename[:-7]
    else:
        return filename


def fmriheaderinfo(niftifilename: str) -> Tuple[NDArray, NDArray]:
    """
    Retrieve the header information from a nifti file.

    This function extracts repetition time and timepoints information from a NIfTI file header.
    The repetition time is returned in seconds, and the number of timepoints is extracted
    from the header dimensions.

    Parameters
    ----------
    niftifilename : str
        The name of the NIfTI file to read header information from.

    Returns
    -------
    tuple of (NDArray, NDArray)
        A tuple containing:
        - tr : float
          The repetition time, in seconds
        - timepoints : int
          The number of points along the time axis

    Notes
    -----
    The function uses nibabel to load the NIfTI file and extracts header information
    from the 'dim' and 'pixdim' fields. If the time unit is specified as milliseconds,
    the repetition time is converted to seconds.

    Examples
    --------
    >>> tr, timepoints = fmriheaderinfo('subject_01.nii.gz')
    >>> print(f"Repetition time: {tr} seconds")
    >>> print(f"Number of timepoints: {timepoints}")
    """
    nim = nib.load(niftifilename)
    hdr = nim.header.copy()
    thedims = hdr["dim"].copy()
    thesizes = hdr["pixdim"].copy()
    if hdr.get_xyzt_units()[1] == "msec":
        thesizes[4] /= 1000.0
    return thesizes, thedims


def fmritimeinfo(niftifilename: str) -> Tuple[float, int]:
    """
    Retrieve the repetition time and number of timepoints from a nifti file

    Parameters
    ----------
    niftifilename : str
        The name of the nifti file

    Returns
    -------
    tr : float
        The repetition time, in seconds
    timepoints : int
        The number of points along the time axis

    Notes
    -----
    This function extracts the repetition time (TR) and number of timepoints from
    the NIfTI file header. The repetition time is extracted from the pixdim[4] field
    and converted to seconds if necessary. The number of timepoints is extracted
    from the dim[4] field.

    Examples
    --------
    >>> tr, timepoints = fmritimeinfo('sub-01_task-rest_bold.nii.gz')
    >>> print(f"Repetition time: {tr}s, Timepoints: {timepoints}")
    Repetition time: 2.0s, Timepoints: 240
    """
    nim = nib.load(niftifilename)
    hdr = nim.header.copy()
    thedims = hdr["dim"].copy()
    thesizes = hdr["pixdim"].copy()
    if hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    timepoints = thedims[4]
    return tr, timepoints


def checkspacematch(hdr1: Any, hdr2: Any, tolerance: float = 1.0e-3) -> bool:
    """
    Check the headers of two nifti files to determine if they cover the same volume at the same resolution (within tolerance)

    Parameters
    ----------
    hdr1 : nifti header structure
        The header of the first file
    hdr2 : nifti header structure
        The header of the second file
    tolerance : float, optional
        Tolerance for comparison. Default is 1.0e-3

    Returns
    -------
    bool
        True if the spatial dimensions and resolutions of the two files match.

    Notes
    -----
    This function performs two checks:
    1. Dimension matching using `checkspaceresmatch` on pixel dimensions (`pixdim`)
    2. Spatial dimension matching using `checkspacedimmatch` on array dimensions (`dim`)

    Examples
    --------
    >>> import nibabel as nib
    >>> img1 = nib.load('file1.nii.gz')
    >>> img2 = nib.load('file2.nii.gz')
    >>> checkspacematch(img1.header, img2.header)
    True
    """
    dimmatch = checkspaceresmatch(hdr1["pixdim"], hdr2["pixdim"], tolerance=tolerance)
    resmatch = checkspacedimmatch(hdr1["dim"], hdr2["dim"])
    return dimmatch and resmatch


def checkspaceresmatch(sizes1: NDArray, sizes2: NDArray, tolerance: float = 1.0e-3) -> bool:
    """
    Check the spatial pixdims of two nifti files to determine if they have the same resolution (within tolerance)

    Parameters
    ----------
    sizes1 : array_like
        The size array from the first nifti file, typically containing spatial dimensions and pixel sizes
    sizes2 : array_like
        The size array from the second nifti file, typically containing spatial dimensions and pixel sizes
    tolerance : float, optional
        The fractional difference that is permissible between the two sizes that will still match,
        default is 1.0e-3 (0.1%)

    Returns
    -------
    bool
        True if the spatial resolutions of the two files match within the specified tolerance,
        False otherwise

    Notes
    -----
    This function compares the spatial dimensions (indices 1-3) of two nifti file size arrays.
    The comparison is performed using fractional difference: |sizes1[i] - sizes2[i]| / sizes1[i].
    Only dimensions 1-3 are compared (typically x, y, z spatial dimensions).
    The function returns False immediately upon finding any dimension that exceeds the tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> sizes1 = np.array([1.0, 2.0, 2.0, 2.0])
    >>> sizes2 = np.array([1.0, 2.0005, 2.0005, 2.0005])
    >>> checkspaceresmatch(sizes1, sizes2, tolerance=1e-3)
    True

    >>> sizes1 = np.array([1.0, 2.0, 2.0, 2.0])
    >>> sizes2 = np.array([1.0, 2.5, 2.5, 2.5])
    >>> checkspaceresmatch(sizes1, sizes2, tolerance=1e-3)
    File spatial resolutions do not match within tolerance of 0.001
        size of dimension 1: 2.0 != 2.5 (0.25 difference)
    False
    """
    for i in range(1, 4):
        fracdiff = np.fabs(sizes1[i] - sizes2[i]) / sizes1[i]
        if fracdiff > tolerance:
            print(f"File spatial resolutions do not match within tolerance of {tolerance}")
            print(f"\tsize of dimension {i}: {sizes1[i]} != {sizes2[i]} ({fracdiff} difference)")
            return False
    return True



def checkspacedimmatch(dims1: NDArray, dims2: NDArray, verbose: bool = False) -> bool:
    """
    Check the dimension arrays of two nifti files to determine if they cover the same number of voxels in each dimension.

    Parameters
    ----------
    dims1 : NDArray
        The dimension array from the first nifti file. Should contain spatial dimensions
        (typically the first dimension is the number of time points, and dimensions 1-3
        represent x, y, z spatial dimensions).
    dims2 : NDArray
        The dimension array from the second nifti file. Should contain spatial dimensions
        (typically the first dimension is the number of time points, and dimensions 1-3
        represent x, y, z spatial dimensions).
    verbose : bool, optional
        Enable verbose output. Default is False. When True, prints detailed information
        about dimension mismatches.

    Returns
    -------
    bool
        True if the spatial dimensions (dimensions 1-3) of the two files match.
        False if any of the spatial dimensions differ between the files.

    Notes
    -----
    This function compares dimensions 1 through 3 (inclusive) of the two dimension arrays,
    which typically represent the spatial dimensions (x, y, z) of the nifti files.
    The first dimension is usually the number of time points and is not compared.

    Examples
    --------
    >>> import numpy as np
    >>> dims1 = np.array([10, 64, 64, 32])
    >>> dims2 = np.array([10, 64, 64, 32])
    >>> checkspacedimmatch(dims1, dims2)
    True

    >>> dims3 = np.array([10, 64, 64, 33])
    >>> checkspacedimmatch(dims1, dims3)
    False
    """
    for i in range(1, 4):
        if dims1[i] != dims2[i]:
            if verbose:
                print("File spatial voxels do not match")
                print("dimension ", i, ":", dims1[i], "!=", dims2[i])
            return False
    return True


def checktimematch(
    dims1: NDArray,
    dims2: NDArray,
    numskip1: int = 0,
    numskip2: int = 0,
    verbose: bool = False,
) -> bool:
    """
    Check the dimensions of two nifti files to determine if they cover the same number of timepoints.

    This function compares the time dimensions of two NIfTI files after accounting for skipped timepoints
    at the beginning of each file. It is commonly used to verify temporal consistency between paired
    NIfTI datasets.

    Parameters
    ----------
    dims1 : NDArray
        The dimension array from the first NIfTI file. The time dimension is expected to be at index 4.
    dims2 : NDArray
        The dimension array from the second NIfTI file. The time dimension is expected to be at index 4.
    numskip1 : int, optional
        Number of timepoints skipped at the beginning of file 1. Default is 0.
    numskip2 : int, optional
        Number of timepoints skipped at the beginning of file 2. Default is 0.
    verbose : bool, optional
        Enable verbose output. If True, prints detailed information about the comparison.
        Default is False.

    Returns
    -------
    bool
        True if the effective time dimensions of the two files match after accounting for skipped
        timepoints, False otherwise.

    Notes
    -----
    The function assumes that the time dimension is stored at index 4 of the dimension arrays.
    This is typical for NIfTI files where dimensions are ordered as [x, y, z, t, ...].

    Examples
    --------
    >>> import numpy as np
    >>> dims1 = np.array([64, 64, 32, 1, 100, 1])
    >>> dims2 = np.array([64, 64, 32, 1, 95, 1])
    >>> checktimematch(dims1, dims2, numskip1=0, numskip2=5)
    True
    >>> checktimematch(dims1, dims2, numskip1=0, numskip2=3)
    False
    """
    if (dims1[4] - numskip1) != (dims2[4] - numskip2):
        if verbose:
            print("File numbers of timepoints do not match")
            print(
                "dimension ",
                4,
                ":",
                dims1[4],
                "(skip ",
                numskip1,
                ") !=",
                dims2[4],
                " (skip ",
                numskip2,
                ")",
            )
        return False
    else:
        return True


def checkdatamatch(
    data1: NDArray,
    data2: NDArray,
    absthresh: float = 1e-12,
    msethresh: float = 1e-12,
    debug: bool = False,
) -> Tuple[bool, bool]:
    """
    Check if two data arrays match within specified tolerances.

    This function compares two numpy arrays using both mean squared error (MSE) and
    maximum absolute difference metrics to determine if they match within given thresholds.

    Parameters
    ----------
    data1 : NDArray
        First data array to compare
    data2 : NDArray
        Second data array to compare
    absthresh : float, optional
        Absolute difference threshold. Default is 1e-12
    msethresh : float, optional
        Mean squared error threshold. Default is 1e-12
    debug : bool, optional
        Enable debug output. Default is False

    Returns
    -------
    tuple of (bool, bool)
        msematch : bool
            True if mean squared error is below msethresh threshold
        absmatch : bool
            True if maximum absolute difference is below absthresh threshold

    Notes
    -----
    The function uses numpy's `mse` function for mean squared error calculation
    and `np.max(np.fabs(data1 - data2))` for maximum absolute difference.

    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.array([1.0, 2.0, 3.0])
    >>> data2 = np.array([1.000000000001, 2.000000000001, 3.000000000001])
    >>> checkdatamatch(data1, data2)
    (True, True)

    >>> checkdatamatch(data1, data2, absthresh=1e-15)
    (True, False)
    """
    msediff = mse(data1, data2)
    absdiff = np.max(np.fabs(data1 - data2))
    if debug:
        print(f"msediff {msediff}, absdiff {absdiff}")
    return msediff < msethresh, absdiff < absthresh


def checkniftifilematch(
    filename1: str,
    filename2: str,
    absthresh: float = 1e-12,
    msethresh: float = 1e-12,
    spacetolerance: float = 1e-3,
    debug: bool = False,
) -> bool:
    """
    Check if two NIFTI files match in dimensions, resolution, and data values.

    This function compares two NIFTI files for spatial compatibility and data
    equivalence. It verifies that the files have matching spatial dimensions,
    resolution, time dimensions, and that their voxel data values are within
    specified tolerances.

    Parameters
    ----------
    filename1 : str
        Path to the first NIFTI file to be compared.
    filename2 : str
        Path to the second NIFTI file to be compared.
    absthresh : float, optional
        Absolute difference threshold for voxel-wise data comparison.
        If any voxel differs by more than this value, the files are considered
        not to match. Default is 1e-12.
    msethresh : float, optional
        Mean squared error threshold for data comparison. If the MSE between
        the data arrays exceeds this value, the files are considered not to match.
        Default is 1e-12.
    spacetolerance : float, optional
        Tolerance for comparing spatial dimensions and resolution between files.
        Default is 1e-3.
    debug : bool, optional
        If True, enables debug output to assist in troubleshooting.
        Default is False.

    Returns
    -------
    bool
        True if all checks (spatial, temporal, and data) pass within the specified
        tolerances; False otherwise.

    Notes
    -----
    The function internally calls several helper functions:
    - `readfromnifti`: Reads NIFTI file metadata and data.
    - `checkspacematch`: Compares spatial dimensions and resolution.
    - `checktimematch`: Compares time dimensions.
    - `checkdatamatch`: Compares data values using MSE and absolute difference.

    Examples
    --------
    >>> match = checkniftifilematch('file1.nii', 'file2.nii')
    >>> print(match)
    True

    >>> match = checkniftifilematch('file1.nii', 'file2.nii', absthresh=1e-10)
    >>> print(match)
    False
    """
    im1, im1_data, im1_hdr, im1_dims, im1_sizes = readfromnifti(filename1)
    im2, im2_data, im2_hdr, im2_dims, im2_sizes = readfromnifti(filename2)
    spacematch = checkspacematch(im1_hdr, im2_hdr, tolerance=spacetolerance)
    if not spacematch:
        print(
            "file spatial dimensions or resolution do not match within tolerance {spacetolerance}"
        )
        return False
    timematch = checktimematch(im1_dims, im2_dims)
    if not timematch:
        print(f"file time dimensions do not match")
        return False
    msedatamatch, absdatamatch = checkdatamatch(
        im1_data,
        im2_data,
        absthresh=absthresh,
        msethresh=msethresh,
        debug=debug,
    )
    if not msedatamatch:
        print(f"file data mse does not match within tolerance {msethresh}")
        return False
    if not absdatamatch:
        print(f"files differ by at least {absthresh} in at least one voxel")
        return False
    return True


# --------------------------- non-NIFTI file I/O functions ------------------------------------------
def checkifparfile(filename: str) -> bool:
    """
    Checks to see if a file is an FSL style motion parameter file

    This function determines whether a given filename corresponds to an FSL-style
    motion parameter file by checking if it ends with the '.par' extension.

    Parameters
    ----------
    filename : str
        The name of the file in question, including the file extension.

    Returns
    -------
    bool
        True if the filename ends with '.par', False otherwise.

    Notes
    -----
    FSL (FMRIB Software Library) motion parameter files typically have the '.par'
    extension and contain motion correction parameters for neuroimaging data.

    Examples
    --------
    >>> checkifparfile("subject1.par")
    True
    >>> checkifparfile("subject1.txt")
    False
    >>> checkifparfile("motion.par")
    True
    """
    if filename.endswith(".par"):
        return True
    else:
        return False


def readconfounds(filename: str, debug: bool = False) -> Dict[str, NDArray]:
    """
    Read confound regressors from a text file.

    This function reads confound regressors from a text file and returns them as a dictionary
    mapping confound names to timecourse arrays. The function handles both structured column
    names and automatically generated names for cases where column information is missing.

    Parameters
    ----------
    filename : str
        Path to the confounds file
    debug : bool, optional
        Enable debug output. Default is False

    Returns
    -------
    dict of str to NDArray
        Dictionary mapping confound names to timecourse arrays. Each key is a confound name
        and each value is a 1D numpy array containing the timecourse data for that confound.

    Notes
    -----
    The function internally calls `readvectorsfromtextfile` to parse the input file, which
    returns metadata including sample rate, start time, column names, and the actual data.
    If column names are not present in the file, automatically generated names are created
    in the format 'confound_000', 'confound_001', etc.

    Examples
    --------
    >>> confounds = readconfounds('confounds.txt')
    >>> print(confounds.keys())
    dict_keys(['motion_000', 'motion_001', 'motion_002', 'scrubbing'])
    >>> print(confounds['motion_000'].shape)
    (1000,)
    """
    (
        thesamplerate,
        thestarttime,
        thecolumns,
        thedata,
        compressed,
        filetype,
    ) = readvectorsfromtextfile(filename, debug=debug)
    if thecolumns is None:
        thecolumns = []
        for i in range(thedata.shape[0]):
            thecolumns.append(f"confound_{str(i).zfill(3)}")
    theconfounddict = {}
    for i in range(thedata.shape[0]):
        theconfounddict[thecolumns[i]] = thedata[i]
    return theconfounddict


def readparfile(filename: str) -> Dict[str, NDArray]:
    """
    Read motion parameters from an FSL-style .par file.

    This function reads motion parameters from FSL-style .par files and returns
    them as a dictionary with timecourses keyed by parameter names.

    Parameters
    ----------
    filename : str
        The name of the FSL-style .par file to read. This file should contain
        motion parameters in the standard FSL format with 6 columns representing
        translation (X, Y, Z) and rotation (RotX, RotY, RotZ) parameters.

    Returns
    -------
    dict of NDArray
        Dictionary containing the motion parameters as timecourses. Keys are:
        - 'X': translation along x-axis
        - 'Y': translation along y-axis
        - 'Z': translation along z-axis
        - 'RotX': rotation around x-axis
        - 'RotY': rotation around y-axis
        - 'RotZ': rotation around z-axis
        Each value is a 1D numpy array containing the timecourse for that parameter.

    Notes
    -----
    The .par file format expected by this function is the standard FSL format
    where each row represents a timepoint and each column represents a motion
    parameter. The function assumes the file contains exactly 6 columns in the
    order: X, Y, Z, RotX, RotY, RotZ.

    Examples
    --------
    >>> motion_data = readparfile('motion.par')
    >>> print(motion_data.keys())
    dict_keys(['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'])
    >>> print(motion_data['X'].shape)
    (100,)  # assuming 100 timepoints
    """
    labels = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
    motiontimeseries = readvecs(filename)
    motiondict = {}
    for j in range(0, 6):
        motiondict[labels[j]] = 1.0 * motiontimeseries[j, :]
    return motiondict


def readmotion(filename: str, tr: float = 1.0, colspec: Optional[str] = None) -> Dict[str, Any]:
    """
    Read motion regressors from a file (.par, .tsv, or other text format).

    Parameters
    ----------
    filename : str
        The name of the file in question.
    colspec: str, optional
        The column numbers from the input file to use for the 6 motion regressors

    Returns
    -------
    motiondict: dict
        All the timecourses in the file, keyed by name

    """
    # read in the motion timecourses
    print("reading motion timecourses...")
    filebase, extension = os.path.splitext(filename)
    if extension == ".par":
        allmotion = readvecs(filename)
        motiondict = {}
        motiondict["xtrans"] = allmotion[3, :] * 1.0
        motiondict["ytrans"] = allmotion[4, :] * 1.0
        motiondict["ztrans"] = allmotion[5, :] * 1.0
        motiondict["maxtrans"] = np.max(
            [
                np.max(motiondict["xtrans"]),
                np.max(motiondict["ytrans"]),
                np.max(motiondict["ztrans"]),
            ]
        )
        motiondict["mintrans"] = np.min(
            [
                np.min(motiondict["xtrans"]),
                np.min(motiondict["ytrans"]),
                np.min(motiondict["ztrans"]),
            ]
        )
        motiondict["xrot"] = allmotion[0, :] * 1.0
        motiondict["yrot"] = allmotion[1, :] * 1.0
        motiondict["zrot"] = allmotion[2, :] * 1.0
        motiondict["maxrot"] = np.max(
            [
                np.max(motiondict["xrot"]),
                np.max(motiondict["yrot"]),
                np.max(motiondict["zrot"]),
            ]
        )
        motiondict["minrot"] = np.min(
            [
                np.min(motiondict["xrot"]),
                np.min(motiondict["yrot"]),
                np.min(motiondict["zrot"]),
            ]
        )
    elif extension == ".tsv":
        allmotion = readlabelledtsv(filebase, compressed=False)
        motiondict = {}
        motiondict["xtrans"] = allmotion["trans_x"] * 1.0
        motiondict["ytrans"] = allmotion["trans_y"] * 1.0
        motiondict["ztrans"] = allmotion["trans_z"] * 1.0
        motiondict["maxtrans"] = np.max(
            [
                np.max(motiondict["xtrans"]),
                np.max(motiondict["ytrans"]),
                np.max(motiondict["ztrans"]),
            ]
        )
        motiondict["mintrans"] = np.min(
            [
                np.min(motiondict["xtrans"]),
                np.min(motiondict["ytrans"]),
                np.min(motiondict["ztrans"]),
            ]
        )
        motiondict["xrot"] = allmotion["rot_x"] * 1.0
        motiondict["yrot"] = allmotion["rot_y"] * 1.0
        motiondict["zrot"] = allmotion["rot_z"] * 1.0
        motiondict["maxrot"] = np.max(
            [
                np.max(motiondict["xrot"]),
                np.max(motiondict["yrot"]),
                np.max(motiondict["zrot"]),
            ]
        )
        motiondict["minrot"] = np.min(
            [
                np.min(motiondict["xrot"]),
                np.min(motiondict["yrot"]),
                np.min(motiondict["zrot"]),
            ]
        )
    else:
        # handle weird files gracefully
        allmotion = readvecs(filename, colspec=colspec)
        if allmotion.shape[0] != 6:
            print(
                "motion files without a .par or .tsv extension must either have 6 columns or have 6 columns specified"
            )
            sys.exit()
        # we are going to assume the columns are in FSL order, not that it really matters
        motiondict = {}
        motiondict["xtrans"] = allmotion[3, :] * 1.0
        motiondict["ytrans"] = allmotion[4, :] * 1.0
        motiondict["ztrans"] = allmotion[5, :] * 1.0
        motiondict["maxtrans"] = np.max(
            [
                np.max(motiondict["xtrans"]),
                np.max(motiondict["ytrans"]),
                np.max(motiondict["ztrans"]),
            ]
        )
        motiondict["mintrans"] = np.min(
            [
                np.min(motiondict["xtrans"]),
                np.min(motiondict["ytrans"]),
                np.min(motiondict["ztrans"]),
            ]
        )
        motiondict["xrot"] = allmotion[0, :] * 1.0
        motiondict["yrot"] = allmotion[1, :] * 1.0
        motiondict["zrot"] = allmotion[2, :] * 1.0
        motiondict["maxrot"] = np.max(
            [
                np.max(motiondict["xrot"]),
                np.max(motiondict["yrot"]),
                np.max(motiondict["zrot"]),
            ]
        )
        motiondict["minrot"] = np.min(
            [
                np.min(motiondict["xrot"]),
                np.min(motiondict["yrot"]),
                np.min(motiondict["zrot"]),
            ]
        )
    motiondict["tr"] = tr
    return motiondict


def sliceinfo(slicetimes: NDArray, tr: float) -> Tuple[int, float, NDArray]:
    """
    Find out what slicetimes we have, their spacing, and which timepoint each slice occurs at.  This assumes
    uniform slice time spacing, but supports any slice acquisition order and multiband acquisitions.

    Parameters
    ----------
    slicetimes : 1d float array
        List of all the slicetimes relative to the start of the TR
    tr : float
        The TR of the acquisition

    Returns
    -------
    numsteps : int
        The number of unique slicetimes in the list
    stepsize : float
        The stepsize in seconds between subsequent slice acquisitions
    sliceoffsets : 1d int array
        Which acquisition time each slice was acquired at

    Notes
    -----
    This function assumes uniform slice time spacing and works with any slice acquisition order
    and multiband acquisitions. The function determines the minimum time step between slices
    and maps each slice to its corresponding timepoint within the TR.

    Examples
    --------
    >>> import numpy as np
    >>> slicetimes = np.array([0.0, 0.1, 0.2, 0.3])
    >>> tr = 1.0
    >>> numsteps, stepsize, sliceoffsets = sliceinfo(slicetimes, tr)
    >>> print(numsteps, stepsize, sliceoffsets)
    (4, 0.1, [0 1 2 3])
    """
    sortedtimes = np.sort(slicetimes)
    diffs = sortedtimes[1:] - sortedtimes[0:-1]
    minstep = np.max(diffs)
    numsteps = int(np.round(tr / minstep, 0))
    sliceoffsets = np.around(slicetimes / minstep).astype(int) % numsteps
    return numsteps, minstep, sliceoffsets


def getslicetimesfromfile(slicetimename: str) -> Tuple[NDArray, bool, bool]:
    """
    Read slice timing information from a file.

    This function reads slice timing data from either a JSON file (BIDS sidecar format)
    or a text file containing slice timing values. It returns the slice times along
    with metadata indicating how the data was processed.

    Parameters
    ----------
    slicetimename : str
        Path to the slice timing file. Can be either a JSON file (BIDS sidecar format)
        or a text file containing slice timing values.

    Returns
    -------
    tuple of (NDArray, bool, bool)
        A tuple containing:
        - slicetimes : NDArray
          Array of slice timing values as floats
        - normalizedtotr : bool
          True if the slice times were normalized to TR (time resolution),
          False if they were read directly from a JSON file
        - fileisbidsjson : bool
          True if the input file was a BIDS JSON sidecar file,
          False if it was a text file

    Notes
    -----
    - For JSON files, the function expects a "SliceTiming" key in the JSON dictionary
    - For text files, the function uses readvec() to parse the slice timing values
    - If a JSON file doesn't contain the required "SliceTiming" key, the function
      prints an error message and exits the program
    - Slice timing values are converted to float64 dtype for precision

    Examples
    --------
    >>> slicetimes, normalized, is_bids = getslicetimesfromfile("sub-01_task-rest_bold.json")
    >>> print(slicetimes)
    [0.0, 0.1, 0.2, 0.3, 0.4]
    >>> print(normalized, is_bids)
    (False, True)
    """
    filebase, extension = os.path.splitext(slicetimename)
    if extension == ".json":
        jsoninfodict = readdictfromjson(slicetimename)
        try:
            slicetimelist = jsoninfodict["SliceTiming"]
            slicetimes = np.zeros((len(slicetimelist)), dtype=np.float64)
            for idx, thetime in enumerate(slicetimelist):
                slicetimes[idx] = float(thetime)
            normalizedtotr = False
            fileisbidsjson = True
        except KeyError:
            print(slicetimename, "is not a valid BIDS sidecar file")
            sys.exit()
    else:
        slicetimes = readvec(slicetimename)
        normalizedtotr = True
        fileisbidsjson = False
    return slicetimes, normalizedtotr, fileisbidsjson


def readbidssidecar(inputfilename: str) -> Dict[str, Any]:
    """
    Read key value pairs out of a BIDS sidecar file

    This function reads JSON sidecar files commonly used in BIDS (Brain Imaging Data Structure)
    datasets and returns the key-value pairs as a dictionary.

    Parameters
    ----------
    inputfilename : str
        The name of the sidecar file (with extension). The function will automatically
        look for a corresponding .json file with the same base name.

    Returns
    -------
    dict
        A dictionary containing the key-value pairs from the JSON sidecar file.
        Returns an empty dictionary if the sidecar file does not exist.

    Notes
    -----
    The function expects the sidecar file to have the same base name as the input file
    but with a .json extension. For example, if inputfilename is "sub-01_task-rest_bold.nii.gz",
    the function will look for "sub-01_task-rest_bold.json".

    Examples
    --------
    >>> sidecar_data = readbidssidecar("sub-01_task-rest_bold.nii.gz")
    >>> print(sidecar_data['RepetitionTime'])
    2.0

    >>> sidecar_data = readbidssidecar("nonexistent_file.nii.gz")
    sidecar file does not exist
    >>> print(sidecar_data)
    {}
    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if os.path.exists(thefileroot + ".json"):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            return d
    else:
        print("sidecar file does not exist")
        return {}


def writedicttojson(thedict: Dict[str, Any], thefilename: str) -> None:
    """
    Write key-value pairs to a JSON file with proper numpy type handling.

    This function writes a dictionary to a JSON file, automatically converting
    numpy data types to their Python equivalents to ensure proper JSON serialization.

    Parameters
    ----------
    thedict : dict[str, Any]
        Dictionary containing key-value pairs to be written to JSON file
    thefilename : str
        Path and name of the output JSON file (including extension)

    Returns
    -------
    None
        This function does not return any value

    Notes
    -----
    The function automatically converts numpy data types:
    - numpy.integer → Python int
    - numpy.floating → Python float
    - NDArray → Python list

    The output JSON file will be formatted with:
    - Sorted keys
    - 4-space indentation
    - Comma-separated values without spaces

    Examples
    --------
    >>> import numpy as np
    >>> data = {
    ...     'name': 'John',
    ...     'age': np.int32(30),
    ...     'score': np.float64(95.5),
    ...     'values': np.array([1, 2, 3, 4])
    ... }
    >>> writedicttojson(data, 'output.json')
    >>> # Creates output.json with properly formatted data
    """
    thisdict = {}
    for key in thedict:
        if isinstance(thedict[key], np.integer):
            thisdict[key] = int(thedict[key])
        elif isinstance(thedict[key], np.floating):
            thisdict[key] = float(thedict[key])
        elif isinstance(thedict[key], np.ndarray):
            thisdict[key] = thedict[key].tolist()
        else:
            thisdict[key] = thedict[key]
    with open(thefilename, "wb") as fp:
        fp.write(
            json.dumps(thisdict, sort_keys=True, indent=4, separators=(",", ":")).encode("utf-8")
        )


def readdictfromjson(inputfilename: str) -> Dict[str, Any]:
    """
    Read key value pairs out of a json file.

    This function reads a JSON file and returns its contents as a dictionary.
    The function automatically appends the ".json" extension to the input filename
    if it's not already present.

    Parameters
    ----------
    inputfilename : str
        The name of the json file (with or without extension). If the extension
        is not provided, ".json" will be appended automatically.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the key-value pairs from the JSON file. Returns
        an empty dictionary if the specified file does not exist.

    Notes
    -----
    - The function checks for the existence of the file before attempting to read it
    - If the input filename doesn't have a ".json" extension, it will be automatically added
    - If the file doesn't exist, a message will be printed and an empty dictionary returned

    Examples
    --------
    >>> data = readdictfromjson("config")
    >>> print(data)
    {'key1': 'value1', 'key2': 'value2'}

    >>> data = readdictfromjson("data.json")
    >>> print(data)
    {'name': 'John', 'age': 30}
    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if os.path.exists(thefileroot + ".json"):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            return d
    else:
        print("specified json file does not exist")
        return {}


def readlabelledtsv(inputfilename: str, compressed: bool = False) -> Dict[str, NDArray]:
    """
    Read time series out of an fmriprep confounds tsv file

    Parameters
    ----------
    inputfilename : str
        The root name of the tsv file (without extension)
    compressed : bool, optional
        If True, reads from a gzipped tsv file (.tsv.gz), otherwise reads from
        a regular tsv file (.tsv). Default is False.

    Returns
    -------
    dict of str to NDArray
        Dictionary containing all the timecourses in the file, keyed by the
        column names from the first row of the tsv file. Each value is a
        numpy array containing the time series data for that column.

    Raises
    ------
    FileNotFoundError
        If the specified tsv file (with appropriate extension) does not exist.

    Notes
    -----
    - NaN values in the input file are replaced with 0.0
    - If the file does not exist or is not valid, an empty dictionary is returned
    - The function supports both compressed (.tsv.gz) and uncompressed (.tsv) files

    Examples
    --------
    >>> confounds = readlabelledtsv("sub-01_task-rest_bold_confounds")
    >>> print(confounds.keys())
    dict_keys(['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
    >>> print(confounds['trans_x'].shape)
    (100,)
    """
    confounddict = {}
    if compressed:
        theext = ".tsv.gz"
    else:
        theext = ".tsv"

    if not os.path.isfile(inputfilename + theext):
        raise FileNotFoundError(f"Labelled tsv file {inputfilename + theext} does not exist")

    df = pd.read_csv(inputfilename + theext, sep="\t", quotechar='"')

    # replace nans with 0
    df = df.fillna(0.0)

    for thecolname, theseries in df.items():
        confounddict[thecolname] = theseries.values
    return confounddict


def readcsv(inputfilename: str, debug: bool = False) -> Dict[str, NDArray]:
    """
    Read time series out of an unlabelled csv file.

    This function reads a CSV file and returns a dictionary of time series,
    where keys are column names (or generated names if no header is present)
    and values are NumPy arrays of the corresponding time series data.

    Parameters
    ----------
    inputfilename : str
        The root name of the CSV file (without the '.csv' extension).
    debug : bool, optional
        If True, prints debug information about whether a header line is detected,
        by default False.

    Returns
    -------
    dict of str to NDArray
        A dictionary where keys are column names (or generated names like "col0", "col1", etc.)
        and values are NumPy arrays containing the time series data. If the file does not exist
        or is invalid, an empty dictionary is returned.

    Notes
    -----
    - If the first column of the CSV contains non-numeric values, it is assumed to be a header.
    - If the first column is numeric, it is treated as part of the data, and columns are
      named "col0", "col1", etc.
    - NaN values in the CSV are replaced with 0.0.
    - If the file does not exist or cannot be read, a FileNotFoundError is raised.

    Examples
    --------
    >>> data = readcsv("timeseries_data")
    >>> print(data.keys())
    ['col0', 'col1', 'col2']
    >>> print(data['col0'])
    [1.0, 2.0, 3.0, 4.0]

    >>> data = readcsv("labeled_data", debug=True)
    there is a header line
    >>> print(data.keys())
    ['time', 'signal1', 'signal2']
    """
    if not os.path.isfile(inputfilename + ".csv"):
        raise FileNotFoundError(f"csv file {inputfilename}.csv does not exist")

    timeseriesdict = {}

    # Read the data in initially with no header
    df = pd.read_csv(inputfilename + ".csv", sep=",", quotechar='"', header=0)

    # replace nans with 0
    df = df.fillna(0.0)

    # Check to see if the first element is a float
    try:
        dummy = float(df.columns[0])
    except ValueError:
        if debug:
            print("there is a header line")
        for thecolname, theseries in df.items():
            timeseriesdict[thecolname] = theseries.values
    else:
        # if we got to here, reread the data, but assume there is no header line
        df = pd.read_csv(inputfilename + ".csv", sep=",", quotechar='"', header=None)

        # replace nans with 0
        df = df.fillna(0.0)

        if debug:
            print("there is no header line")
        colnum = 0
        for dummy, theseries in df.items():
            timeseriesdict[makecolname(colnum, 0)] = theseries.values
            colnum += 1
        # timeseriesdict["columnsource"] = "synthetic"

    return timeseriesdict


def readfslmat(inputfilename: str, debug: bool = False) -> Dict[str, NDArray]:
    """
    Read time series out of an FSL design.mat file

    Parameters
    ----------
    inputfilename : str
        The root name of the .mat file (no extension)
    debug : bool, optional
        If True, print the DataFrame contents for debugging purposes. Default is False

    Returns
    -------
    dict of NDArray
        Dictionary containing all the timecourses in the file, keyed by column names.
        If the first row exists, it is used as keys; otherwise, keys are generated as
        "col1, col2...colN". Returns an empty dictionary if file does not exist or is not valid.

    Raises
    ------
    FileNotFoundError
        If the specified FSL mat file does not exist

    Notes
    -----
    This function reads FSL design.mat files and extracts time series data. The function
    skips the first 5 rows of the file (assumed to be header information) and treats
    subsequent rows as time series data. The column names are generated using the
    `makecolname` helper function.

    Examples
    --------
    >>> timeseries = readfslmat("design")
    >>> print(timeseries.keys())
    dict_keys(['col0', 'col1', 'col2'])
    >>> print(timeseries['col0'])
    [0.1, 0.2, 0.3, 0.4]
    """
    if not os.path.isfile(inputfilename + ".mat"):
        raise FileNotFoundError(f"FSL mat file {inputfilename}.mat does not exist")

    timeseriesdict = {}

    # Read the data in with no header
    df = pd.read_csv(inputfilename + ".mat", delim_whitespace=True, header=None, skiprows=5)

    if debug:
        print(df)
    colnum = 0
    for dummy, theseries in df.items():
        timeseriesdict[makecolname(colnum, 0)] = theseries.values
        colnum += 1

    return timeseriesdict


def readoptionsfile(inputfileroot: str) -> Dict[str, Any]:
    """
    Read a run options from a JSON or TXT configuration file.

    This function attempts to read rapidtide run options from a file with the given root name,
    checking for `.json` and `.txt` extensions in that order. If neither file exists,
    a `FileNotFoundError` is raised. The function also handles backward compatibility
    for older options files by filling in default filter limits based on the `filtertype`.

    Parameters
    ----------
    inputfileroot : str
        The base name of the options file (without extension). The function will
        first look for `inputfileroot.json`, then `inputfileroot.txt`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the run options. The dictionary includes keys such as
        `filtertype`, `lowerstop`, `lowerpass`, `upperpass`, and `upperstop`, depending
        on the file content and filter type.

    Raises
    ------
    FileNotFoundError
        If neither `inputfileroot.json` nor `inputfileroot.txt` exists.

    Notes
    -----
    For backward compatibility, older options files without `lowerpass` key are updated
    with default values based on the `filtertype`:

    - "None": All limits set to 0.0 or -1.0
    - "vlf": 0.0, 0.0, 0.009, 0.010
    - "lfo": 0.009, 0.010, 0.15, 0.20
    - "resp": 0.15, 0.20, 0.4, 0.5
    - "card": 0.4, 0.5, 2.5, 3.0
    - "arb": Uses values from `arb_lowerstop`, `arb_lower`, `arb_upper`, `arb_upperstop`

    Examples
    --------
    >>> options = readoptionsfile("myfilter")
    >>> print(options["filtertype"])
    'vlf'
    """
    if os.path.isfile(inputfileroot + ".json"):
        # options saved as json
        thedict = readdictfromjson(inputfileroot + ".json")
    elif os.path.isfile(inputfileroot + ".txt"):
        # options saved as text
        thedict = readdict(inputfileroot + ".txt")
    else:
        raise FileNotFoundError(f"options file {inputfileroot}(.json/.txt) does not exist")

    # correct behavior for older options files
    try:
        test = thedict["lowerpass"]
    except KeyError:
        print("no filter limits found in options file - filling in defaults")
        if thedict["filtertype"] == "None":
            thedict["lowerstop"] = 0.0
            thedict["lowerpass"] = 0.0
            thedict["upperpass"] = -1.0
            thedict["upperstop"] = -1.0
        elif thedict["filtertype"] == "vlf":
            thedict["lowerstop"] = 0.0
            thedict["lowerpass"] = 0.0
            thedict["upperpass"] = 0.009
            thedict["upperstop"] = 0.010
        elif thedict["filtertype"] == "lfo":
            thedict["lowerstop"] = 0.009
            thedict["lowerpass"] = 0.010
            thedict["upperpass"] = 0.15
            thedict["upperstop"] = 0.20
        elif thedict["filtertype"] == "resp":
            thedict["lowerstop"] = 0.15
            thedict["lowerpass"] = 0.20
            thedict["upperpass"] = 0.4
            thedict["upperstop"] = 0.5
        elif thedict["filtertype"] == "card":
            thedict["lowerstop"] = 0.4
            thedict["lowerpass"] = 0.5
            thedict["upperpass"] = 2.5
            thedict["upperstop"] = 3.0
        elif thedict["filtertype"] == "arb":
            thedict["lowerstop"] = thedict["arb_lowerstop"]
            thedict["lowerpass"] = thedict["arb_lower"]
            thedict["upperpass"] = thedict["arb_upper"]
            thedict["upperstop"] = thedict["arb_upperstop"]
        else:
            print("cannot determine filtering")
            thedict["lowerstop"] = 0.0
            thedict["lowerpass"] = 0.0
            thedict["upperpass"] = -1.0
            thedict["upperstop"] = -1.0
    return thedict


def makecolname(colnum: int, startcol: int) -> str:
    """
    Generate a column name in the format 'col_##' where ## is a zero-padded number.

    This function creates standardized column names by adding a starting offset to
    a column number and formatting it with zero-padding to ensure consistent
    two-digit representation.

    Parameters
    ----------
    colnum : int
        The base column number to be used in the name generation.
    startcol : int
        The starting column offset to be added to colnum.

    Returns
    -------
    str
        A column name in the format 'col_##' where ## represents the zero-padded
        sum of colnum and startcol.

    Notes
    -----
    The resulting number is zero-padded to always have at least two digits.
    For example, if colnum=5 and startcol=10, the result will be 'col_15'.
    If colnum=1 and startcol=2, the result will be 'col_03'.

    Examples
    --------
    >>> makecolname(0, 0)
    'col_00'

    >>> makecolname(5, 10)
    'col_15'

    >>> makecolname(1, 2)
    'col_03'
    """
    return f"col_{str(colnum + startcol).zfill(2)}"


def writebidstsv(
    outputfileroot: str,
    data: NDArray,
    samplerate: float,
    extraheaderinfo: Optional[Dict[str, Any]] = None,
    compressed: bool = True,
    columns: Optional[List[str]] = None,
    xaxislabel: str = "time",
    yaxislabel: str = "arbitrary value",
    starttime: float = 0.0,
    append: bool = False,
    samplerate_tolerance: float = 1e-6,
    starttime_tolerance: float = 1e-6,
    colsinjson: bool = True,
    colsintsv: bool = False,
    omitjson: bool = False,
    debug: bool = False,
) -> None:
    """
    Write physiological or stimulation data to a BIDS-compatible TSV file with optional JSON sidecar.

    This function writes time series data to a TSV file following BIDS conventions for physiological
    (``_physio``) and stimulation (``_stim``) data. It supports optional compression, appending to
    existing files, and includes metadata in a corresponding JSON file.

    Parameters
    ----------
    outputfileroot : str
        Root name of the output files (without extension). The function will write
        ``<outputfileroot>.tsv`` or ``<outputfileroot>.tsv.gz`` and ``<outputfileroot>.json``.
    data : NDArray
        Time series data to be written. If 1D, it will be reshaped to (1, n_timesteps).
        Shape should be (n_channels, n_timesteps).
    samplerate : float
        Sampling frequency of the data in Hz.
    extraheaderinfo : dict, optional
        Additional key-value pairs to include in the JSON sidecar file.
    compressed : bool, default=True
        If True, compress the TSV file using gzip (.tsv.gz). If False, write uncompressed (.tsv).
    columns : list of str, optional
        Column names for the TSV file. If None, default names are generated using
        ``makecolname``.
    xaxislabel : str, default="time"
        Label for the x-axis in the JSON sidecar.
    yaxislabel : str, default="arbitrary value"
        Label for the y-axis in the JSON sidecar.
    starttime : float, default=0.0
        Start time of the recording in seconds.
    append : bool, default=False
        If True, append data to an existing file. The function checks compatibility of
        sampling rate, start time, and number of columns.
    samplerate_tolerance : float, default=1e-6
        Tolerance for comparing sampling rates when appending data.
    starttime_tolerance : float, default=1e-6
        Tolerance for comparing start times when appending data.
    colsinjson : bool, default=True
        If True, include the column names in the JSON file under the "Columns" key.
    colsintsv : bool, default=False
        If True, write column headers in the TSV file. BIDS convention requires no headers.
    omitjson : bool, default=False
        If True, do not write the JSON sidecar file.
    debug : bool, default=False
        If True, print debug information during execution.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - BIDS-compliant TSV files require:
        1. Compression (.tsv.gz)
        2. Presence of "SamplingFrequency", "StartTime", and "Columns" in the JSON file
        3. No column headers in the TSV file
        4. File name ending in "_physio" or "_stim"
    - If ``append=True``, the function will validate compatibility of sampling rate, start time,
      and number of columns with the existing file.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(2, 1000)
    >>> writebidstsv("sub-01_task-rest_physio", data, samplerate=100.0)
    >>> # Writes:
    >>> #   sub-01_task-rest_physio.tsv.gz
    >>> #   sub-01_task-rest_physio.json

    See Also
    --------
    readbidstsv : Read BIDS physiological or stimulation data from TSV and JSON files.
    """
    if debug:
        print("entering writebidstsv:")
        print("\toutputfileroot:", outputfileroot)
        print("\tdata.shape:", data.shape)
        print("\tsamplerate:", samplerate)
        print("\tcompressed:", compressed)
        print("\tcolumns:", columns)
        print("\txaxislabel:", xaxislabel)
        print("\tyaxislabel:", yaxislabel)
        print("\tstarttime:", starttime)
        print("\tappend:", append)
    if len(data.shape) == 1:
        reshapeddata = data.reshape((1, -1))
        if debug:
            print("input data reshaped from", data.shape, "to", reshapeddata.shape)
    else:
        reshapeddata = data
    if append:
        insamplerate, instarttime, incolumns, indata, incompressed, incolsource = readbidstsv(
            outputfileroot + ".json",
            neednotexist=True,
            debug=debug,
        )
        if debug:
            print("appending")
            print(insamplerate, instarttime, incolumns, indata, incompressed, incolsource)
        if insamplerate is None:
            # file does not already exist
            if debug:
                print("creating file:", data.shape, columns, samplerate)
            startcol = 0
        else:
            # file does already exist
            if debug:
                print(
                    "appending:",
                    insamplerate,
                    instarttime,
                    incolumns,
                    incolsource,
                    indata.shape,
                    reshapeddata.shape,
                )
            compressed = incompressed
            if (
                np.fabs(insamplerate - samplerate) < samplerate_tolerance
                and np.fabs(instarttime - starttime) < starttime_tolerance
                and reshapeddata.shape[1] == indata.shape[1]
            ):
                startcol = len(incolumns)
            else:
                print("data dimensions not compatible with existing dimensions")
                print(samplerate, insamplerate)
                print(starttime, instarttime)
                print(columns, incolumns, incolsource)
                print(indata.shape, reshapeddata.shape)
                sys.exit()
    else:
        startcol = 0

    if columns is None:
        columns = []
        for i in range(reshapeddata.shape[0]):
            columns.append(makecolname(i, startcol))
    else:
        if len(columns) != reshapeddata.shape[0]:
            raise ValueError(
                f"number of column names ({len(columns)}) ",
                f"does not match number of columns ({reshapeddata.shape[0]}) in data",
            )
    if startcol > 0:
        df = pd.DataFrame(data=np.transpose(indata), columns=incolumns)
        for i in range(len(columns)):
            df[columns[i]] = reshapeddata[i, :]
    else:
        df = pd.DataFrame(data=np.transpose(reshapeddata), columns=columns)
    if compressed:
        df.to_csv(
            outputfileroot + ".tsv.gz", sep="\t", compression="gzip", header=colsintsv, index=False
        )
    else:
        df.to_csv(
            outputfileroot + ".tsv", sep="\t", compression=None, header=colsintsv, index=False
        )
    headerdict = {}
    headerdict["SamplingFrequency"] = float(samplerate)
    headerdict["StartTime"] = float(starttime)
    headerdict["XAxisLabel"] = xaxislabel
    headerdict["YAxisLabel"] = yaxislabel
    if colsinjson:
        if startcol == 0:
            headerdict["Columns"] = columns
        else:
            headerdict["Columns"] = incolumns + columns
    if extraheaderinfo is not None:
        for key in extraheaderinfo:
            headerdict[key] = extraheaderinfo[key]

    if not omitjson:
        with open(outputfileroot + ".json", "wb") as fp:
            fp.write(
                json.dumps(headerdict, sort_keys=True, indent=4, separators=(",", ":")).encode(
                    "utf-8"
                )
            )


def readvectorsfromtextfile(
    fullfilespec: str, onecol: bool = False, debug: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[List[str]], NDArray, Optional[bool], str]:
    """
    Read time series data from a text-based file (TSV, CSV, MAT, or BIDS-style TSV).

    This function reads timecourse data from various file formats, including plain TSV,
    gzipped TSV (.tsv.gz), CSV, and BIDS-style continuous data files (.tsv with associated .json).
    It automatically detects the file type and parses the data accordingly.

    Parameters
    ----------
    fullfilespec : str
        Path to the input file. May include a column specification (e.g., ``"file.tsv[0:5]"``).
    colspec : str, optional
        Column specification for selecting specific columns. For TSV/CSV files, this can be a
        comma-separated list of column names or integer indices. For BIDS-style TSV files, it
        should be a comma-separated list of column names.
    onecol : bool, optional
        If True, returns only the first column of data. Default is False.
    debug : bool, optional
        If True, prints additional debugging information. Default is False.

    Returns
    -------
    samplerate : float
        Sample rate in Hz. None if not knowable.
    starttime : float
        Time of first point, in seconds. None if not knowable.
    columns : str array
        Names of the timecourses contained in the file. None if not knowable.
    data : 2D numpy array
        Timecourses from the file.
    compressed : bool
        True if time data is gzipped (as in a .tsv.gz file).
    filetype : str
        One of "text", "csv", "plaintsv", "bidscontinuous".

    Notes
    -----
    - If the file does not exist or is not valid, all return values are None.
    - For BIDS-style TSV files, the associated .json sidecar file is used to determine
      sample rate and start time.
    - For plain TSV files, column names are read from the header row.
    - If ``onecol`` is True, only the first column is returned.

    Examples
    --------
    >>> samplerate, starttime, columns, data, compressed, filetype = readvectorsfromtextfile("data.tsv")
    >>> samplerate, starttime, columns, data, compressed, filetype = readvectorsfromtextfile("data.tsv[0:3]")
    >>> samplerate, starttime, columns, data, compressed, filetype = readvectorsfromtextfile("data.tsv", onecol=True)
    """

    thefilename, colspec = parsefilespec(fullfilespec)
    thefileroot, theext = os.path.splitext(thefilename)

    if theext == ".gz":
        thefileroot, thenextext = os.path.splitext(thefileroot)
        if thenextext is not None:
            theext = thenextext + theext
    if debug:
        print("thefileroot:", thefileroot)
        print("theext:", theext)
        print("colspec:", colspec)
    jsonexists = os.path.exists(thefileroot + ".json")
    tsvexists = os.path.exists(thefileroot + ".tsv.gz") or os.path.exists(thefileroot + ".tsv")
    compressed = os.path.exists(thefileroot + ".tsv.gz")
    csvexists = os.path.exists(thefileroot + ".csv")
    matexists = os.path.exists(thefileroot + ".mat")
    if debug:
        print("jsonexists=", jsonexists)
        print("tsvexists=", tsvexists)
        print("compressed=", compressed)
    if tsvexists:
        if jsonexists:
            filetype = "bidscontinuous"
        else:
            filetype = "plaintsv"
    elif csvexists:
        filetype = "csv"
    elif matexists:
        filetype = "mat"
    else:
        filetype = "text"
    if debug:
        print(f"detected file type is {filetype}")
    if debug:
        print(f"filetype of {fullfilespec} determined to be", filetype)
    if filetype == "text":
        # colspec can only be None or a list of integer ranges
        thedata = readvecs(thefilename, colspec=colspec, debug=debug)
        if onecol and thedata.shape[0] > 1:
            print("specify a single column from", thefilename)
            sys.exit()
        thesamplerate = None
        thestarttime = None
        thecolumns = None
        compressed = None
    elif filetype == "bidscontinuous":
        # colspec can be None or a list of comma separated column names
        colspectouse = None
        if colspec is not None:
            try:
                colspectouse = makecolname(int(colspec), 0)
            except ValueError:
                colspectouse = colspec
        thesamplerate, thestarttime, thecolumns, thedata, compressed, colsource = readbidstsv(
            thefilename, colspec=colspectouse, debug=debug
        )
        if thedata is None:
            raise ValueError(f"specified column {colspectouse} does not exist")
        if onecol and thedata.shape[0] > 1:
            raise ValueError("specify a single column from", thefilename)
    elif filetype == "plaintsv":
        thedatadict = readlabelledtsv(thefileroot, compressed=compressed)
        if colspec is None:
            thecolumns = list(thedatadict.keys())
        else:
            try:
                thecolumns = [makecolname(int(colspec), 0)]
            except ValueError:
                thecolumns = colspec.split(",")
                colsource = "data"
            else:
                colsource = "synthetic"
        if onecol and len(thecolumns) > 1:
            print("specify a single column from", thefilename)
            sys.exit()
        thedatacols = []
        for thekey in thecolumns:
            try:
                thedatacols.append(thedatadict[thekey])
            except KeyError:
                print(thefilename, "does not contain column", thekey)
                sys.exit()
        thedata = np.array(thedatacols)
        thesamplerate = None
        thestarttime = None
        compressed = None
    elif (filetype == "csv") or (filetype == "mat"):
        if filetype == "csv":
            thedatadict = readcsv(thefileroot, debug=debug)
        else:
            thedatadict = readfslmat(thefileroot, debug=debug)
        if colspec is None:
            thecolumns = list(thedatadict.keys())
        else:
            thecolumns = colspec.split(",")
        if onecol and len(thecolumns) > 1:
            print("specify a single column from", thefilename)
            sys.exit()
        thedatacols = []
        for thekey in thecolumns:
            try:
                thedatacols.append(thedatadict[thekey])
            except KeyError:
                # try expanding the numbers
                try:
                    thedatacols.append(thedatadict[makecolname(int(thekey), 0)])
                except:
                    print(thefilename, "does not contain column", thekey)
                    sys.exit()
        thedata = np.array(thedatacols)
        thesamplerate = None
        thestarttime = None
        compressed = None
    else:
        print("illegal file type:", filetype)

    if onecol:
        thedata = thedata[0, :]

    if debug:
        print("\tthesamplerate:", thesamplerate)
        print("\tthestarttime:", thestarttime)
        print("\tthecolumns:", thecolumns)
        print("\tthedata.shape:", thedata.shape)
        print("\tcompressed:", compressed)
        print("\tfiletype:", filetype)

    return thesamplerate, thestarttime, thecolumns, thedata, compressed, filetype


def readbidstsv(
    inputfilename: str,
    colspec: Optional[str] = None,
    warn: bool = True,
    neednotexist: bool = False,
    debug: bool = False,
) -> Tuple[
    float,
    float,
    Optional[List[str]],
    Optional[NDArray],
    Optional[bool],
    Optional[str],
]:
    """
    Read BIDS-compatible TSV data file with associated JSON metadata.

    This function reads a TSV file (optionally gzipped) and its corresponding JSON
    metadata file to extract timecourse data, sample rate, start time, and column names.
    It supports both compressed (.tsv.gz) and uncompressed (.tsv) TSV files.

    Parameters
    ----------
    inputfilename : str
        The root name of the TSV and accompanying JSON file (without extension).
    colspec : str, optional
        A comma-separated list of column names to return. If None, all columns are returned.
    debug : bool, optional
        If True, print additional debugging information. Default is False.
    warn : bool, optional
        If True, print warnings for missing metadata fields. Default is True.
    neednotexist : bool, optional
        If True, return None values instead of raising an exception if files do not exist.
        Default is False.

    Returns
    -------
    tuple of (samplerate, starttime, columns, data, is_compressed, columnsource)
        samplerate : float
            Sample rate in Hz.
        starttime : float
            Time of first point in seconds.
        columns : list of str
            Names of the timecourses contained in the file.
        data : NDArray, optional
            2D array of timecourses from the file. Returns None if file does not exist or is invalid.
        is_compressed : bool
            Indicates whether the TSV file was gzipped.
        columnsource : str
            Source of column names: either 'json' or 'tsv'.

    Notes
    -----
    - If the TSV file does not exist or is not valid, all return values are None.
    - If the JSON metadata file is missing required fields (SamplingFrequency, StartTime, Columns),
      default values are used and warnings are issued if `warn=True`.
    - The function handles both gzipped and uncompressed TSV files.
    - If a header line is found in the TSV file, it is skipped and a warning is issued.

    Examples
    --------
    >>> samplerate, starttime, columns, data, is_compressed, source = readbidstsv('sub-01_task-rest')
    >>> print(f"Sample rate: {samplerate} Hz")
    Sample rate: 10.0 Hz

    >>> samplerate, starttime, columns, data, is_compressed, source = readbidstsv(
    ...     'sub-01_task-rest', colspec='column1,column2'
    ... )
    >>> print(f"Selected columns: {columns}")
    Selected columns: ['column1', 'column2']
    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if theext == ".gz":
        thefileroot, thenextext = os.path.splitext(thefileroot)
        if thenextext is not None:
            theext = thenextext + theext

    if debug:
        print("thefileroot:", thefileroot)
        print("theext:", theext)
    if os.path.exists(thefileroot + ".json") and (
        os.path.exists(thefileroot + ".tsv.gz") or os.path.exists(thefileroot + ".tsv")
    ):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            try:
                samplerate = float(d["SamplingFrequency"])
            except:
                print("no samplerate found in json, setting to 1.0")
                samplerate = 1.0
                if warn:
                    print(
                        "Warning - SamplingFrequency not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            try:
                starttime = float(d["StartTime"])
            except:
                print("no starttime found in json, setting to 0.0")
                starttime = 0.0
                if warn:
                    print(
                        "Warning - StartTime not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            try:
                columns = d["Columns"]
            except:
                if debug:
                    print("no columns found in json, will take labels from the tsv file")
                columns = None
                if warn:
                    print(
                        "Warning - Columns not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            else:
                columnsource = "json"
        if os.path.exists(thefileroot + ".tsv.gz"):
            compression = "gzip"
            theextension = ".tsv.gz"
        else:
            compression = None
            theextension = ".tsv"
            if warn:
                print(
                    "Warning - "
                    + thefileroot
                    + ".tsv is uncompressed.  This is not BIDS compliant."
                )

        df = pd.read_csv(
            thefileroot + theextension,
            compression=compression,
            names=columns,
            header=None,
            sep="\t",
            quotechar='"',
        )

        # replace nans with 0
        df = df.fillna(0.0)

        # check for header line
        if any(df.iloc[0].apply(lambda x: isinstance(x, str))):
            headerlinefound = True
            # reread the data, skipping the first row
            df = pd.read_csv(
                thefileroot + theextension,
                compression=compression,
                names=columns,
                header=0,
                sep="\t",
                quotechar='"',
            )

            # replace nans with 0
            df = df.fillna(0.0)

            if warn:
                print(
                    "Warning - Column header line found in "
                    + thefileroot
                    + ".tsv.  This is not BIDS compliant."
                )
        else:
            headerlinefound = False

        if columns is None:
            columns = list(df.columns.values)
            columnsource = "tsv"
        if debug:
            print(
                samplerate,
                starttime,
                columns,
                np.transpose(df.to_numpy()).shape,
                (compression == "gzip"),
                warn,
                headerlinefound,
            )

        # select a subset of columns if they were specified
        if colspec is None:
            return (
                samplerate,
                starttime,
                columns,
                np.transpose(df.to_numpy()),
                (compression == "gzip"),
                columnsource,
            )
        else:
            collist = colspec.split(",")
            try:
                selectedcols = df[collist]
            except KeyError:
                print("specified column list cannot be found in", inputfilename)
                return [None, None, None, None, None, None]
            columns = list(selectedcols.columns.values)
            return (
                samplerate,
                starttime,
                columns,
                np.transpose(selectedcols.to_numpy()),
                (compression == "gzip"),
                columnsource,
            )
    else:
        if neednotexist:
            return [None, None, None, None, None, None]
        else:
            raise FileNotFoundError(f"file pair {thefileroot}(.json/.tsv[.gz]) does not exist")


def readcolfrombidstsv(
    inputfilename: str,
    columnnum: Optional[int] = 0,
    columnname: Optional[str] = None,
    neednotexist: bool = False,
    debug: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[NDArray]]:
    """
    Read a specific column from a BIDS TSV file.

    Extracts a single column of data from a BIDS TSV file, either by column name
    or by column index. The function handles both compressed and uncompressed files.

    Parameters
    ----------
    inputfilename : str
        Path to the input BIDS TSV file (can be .tsv or .tsv.gz)
    columnname : str, optional
        Name of the column to extract. If specified, ``columnnum`` is ignored.
        Default is None.
    columnnum : int, optional
        Index of the column to extract (0-based). Ignored if ``columnname`` is specified.
        Default is 0.
    neednotexist : bool, optional
        If True, the function will not raise an error if the file does not exist.
        Default is False.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    tuple
        A tuple containing:

        - samplerate : float or None
          Sampling rate extracted from the file, or None if no valid data found
        - starttime : float or None
          Start time extracted from the file, or None if no valid data found
        - data : NDArray or None
          The extracted column data as a 1D array, or None if no valid data found

    Notes
    -----
    - If both ``columnname`` and ``columnnum`` are specified, ``columnname`` takes precedence
    - Column indices are 0-based
    - The function handles both compressed (.tsv.gz) and uncompressed (.tsv) files
    - Returns None for all values if no valid data is found

    Examples
    --------
    >>> # Read first column by index
    >>> samplerate, starttime, data = readcolfrombidstsv('data.tsv', columnnum=0)

    >>> # Read column by name
    >>> samplerate, starttime, data = readcolfrombidstsv('data.tsv', columnname='reaction_time')

    >>> # Read column with debug output
    >>> samplerate, starttime, data = readcolfrombidstsv('data.tsv', columnname='rt', debug=True)
    """
    samplerate, starttime, columns, data, compressed, colsource = readbidstsv(
        inputfilename, neednotexist=neednotexist, debug=debug
    )
    if data is None:
        print("no valid datafile found")
        return None, None, None
    else:
        if columnname is not None:
            # looking for a named column
            try:
                thecolnum = columns.index(columnname)
                return samplerate, starttime, data[thecolnum, :]
            except:
                print("no column named", columnname, "in", inputfilename)
                return None, None, None
        # we can only get here if columnname is undefined
        if not (0 < columnnum < len(columns)):
            print(
                "specified column number",
                columnnum,
                "is out of range in",
                inputfilename,
            )
            return None, None, None
        else:
            return samplerate, starttime, data[columnnum, :]


def parsefilespec(filespec: str, debug: bool = False) -> Tuple[str, Optional[str]]:
    """
    Parse a file specification string into filename and column specification.

    This function splits a file specification string using ':' as the delimiter.
    On Windows platforms, it handles special cases where the second character
    is ':' (e.g., "C:file.txt") by treating the first two parts as the filename.

    Parameters
    ----------
    filespec : str
        The file specification string to parse. Expected format is
        "filename[:column_specification]".
    debug : bool, optional
        If True, print debug information during execution. Default is False.

    Returns
    -------
    tuple[str, str or None]
        A tuple containing:
        - thefilename : str
            The parsed filename part of the specification
        - thecolspec : str or None
            The parsed column specification, or None if not provided

    Raises
    ------
    ValueError
        If the file specification is malformed (e.g., too many parts when
        special case handling is not applicable).

    Notes
    -----
    On Windows systems, this function correctly handles drive letter specifications
    such as "C:file.txt" by treating the first two elements ("C:" and "file.txt")
    as the filename part.

    Examples
    --------
    >>> parsefilespec("data.csv")
    ('data.csv', None)

    >>> parsefilespec("data.csv:1,3,5")
    ('data.csv', '1,3,5')

    >>> parsefilespec("C:file.txt:col1")
    ('C:file.txt', 'col1')
    """
    inputlist = filespec.split(":")
    if debug:
        print(f"PARSEFILESPEC: input string >>>{filespec}<<<")
        print(f"PARSEFILESPEC: platform is {platform.system()}")

    specialcase = False
    if len(inputlist) > 1:
        if filespec[1] == ":" and platform.system() == "Windows":
            specialcase = True
    if specialcase:
        thefilename = ":".join([inputlist[0], inputlist[1]])
        if len(inputlist) == 3:
            thecolspec = inputlist[2]
        elif len(inputlist) == 2:
            thecolspec = None
        else:
            raise ValueError(
                f"PARSEFILESPEC: Badly formed file specification {filespec} - exiting"
            )
    else:
        thefilename = inputlist[0]
        if len(inputlist) == 2:
            thecolspec = inputlist[1]
        elif len(inputlist) == 1:
            thecolspec = None
        else:
            raise ValueError(
                f"PARSEFILESPEC: Badly formed file specification {filespec} - exiting"
            )
    if debug:
        print(f"PARSEFILESPEC: thefilename is >>>{filespec}<<<, thecolspec is >>>{thecolspec}<<<")
    return thefilename, thecolspec


def unique(list1: List[Any]) -> List[Any]:
    """
    Convert a column specification string to a list of column indices.

    This function parses a column specification string and converts it into a list of
    zero-based column indices. The specification can include ranges (e.g., "0-5") and
    individual column numbers (e.g., "7") separated by commas.

    Parameters
    ----------
    colspec : str or None
        Column specification string in format like "0-5,7,10-12" or predefined macro.
        If None, returns None.
    debug : bool, optional
        Enable debug output. Default is False

    Returns
    -------
    list of int or None
        List of column indices corresponding to the specification, or None if input is None.
        Returns empty list if specification is empty or invalid.

    Notes
    -----
    - Column indices are zero-based
    - Ranges are inclusive on both ends
    - Individual columns can be specified as single numbers
    - Multiple specifications can be combined with commas
    - Invalid ranges or columns will be skipped

    Examples
    --------
    >>> colspectolist("0-2,5,7-9")
    [0, 1, 2, 5, 7, 8, 9]

    >>> colspectolist("3,1-4,6")
    [3, 1, 2, 3, 4, 6]

    >>> colspectolist(None)
    None
    """
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if op.countOf(unique_list, x) == 0:
            unique_list.append(x)
    return unique_list


def colspectolist(colspec: Optional[str], debug: bool = False) -> Optional[List[int]]:
    """
    Convert a column specification string into a sorted list of integers.

    This function parses a column specification string that may contain
    individual integers, ranges (e.g., "1-5"), or predefined macros (e.g.,
    "APARC_GRAY"). It expands macros into their corresponding ranges and
    returns a sorted list of unique integers.

    Parameters
    ----------
    colspec : str or None
        A column specification string. Can include:
        - Individual integers (e.g., "1", "10")
        - Ranges (e.g., "1-5")
        - Predefined macros (e.g., "APARC_GRAY")
        If None, the function prints an error and returns None.
    debug : bool, optional
        If True, enables debug output showing processing steps. Default is False.

    Returns
    -------
    list of int or None
        A sorted list of unique integers corresponding to the column
        specification. Returns None if an error occurs during processing.

    Notes
    -----
    Predefined macros:
        - APARC_SUBCORTGRAY: 8-13,17-20,26-28,47-56,58-60,96,97
        - APARC_CORTGRAY: 1000-1035,2000-2035
        - APARC_GRAY: 8-13,17-20,26-28,47-56,58-60,96,97,1000-1035,2000-2035
        - APARC_WHITE: 2,7,41,46,177,219,3000-3035,4000-4035,5001,5002
        - APARC_CSF: 4,5,14,15,24,31,43,44,63,72
        - APARC_ALLBUTCSF: 2,7-13,17-20,26-28,41,46-56,58-60,96,97,177,219,1000-1035,2000-2035,3000-3035,4000-4035,5001,5002
        - SSEG_GRAY: 3,8,10-13,16-18,26,42,47,49-54,58
        - SSEG_WHITE: 2,7,41,46
        - SSEG_CSF: 4,5,14,15,24,43,44

    Examples
    --------
    >>> colspectolist("1-3,5,7-9")
    [1, 2, 3, 5, 7, 8, 9]

    >>> colspectolist("APARC_GRAY")
    [8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 96, 97, 1000, 1001, ..., 2035]

    >>> colspectolist(None)
    COLSPECTOLIST: no range specification - exiting
    None
    """
    if colspec is None:
        print("COLSPECTOLIST: no range specification - exiting")
        return None
    collist = []
    theranges = colspec.split(",")

    def safeint(s):
        """
        Convert a value to integer safely, handling various input types.

        This function attempts to convert the input value to an integer. It handles
        strings, floats, and other numeric types gracefully, with special handling
        for string representations that may contain commas or ranges.

        Parameters
        ----------
        value : str, int, float
            The value to convert to integer. If string, may contain comma-separated
            values or range notation (e.g., "2-5", "1,3,5").

        Returns
        -------
        int or list of int
            Integer value or list of integers if input contains multiple values
            or ranges. Returns single integer for simple numeric inputs.

        Notes
        -----
        - For string inputs containing commas, values are split and converted
        - For string inputs containing hyphens, ranges are expanded into individual integers
        - Non-numeric strings will raise ValueError
        - Float inputs are truncated to integers

        Examples
        --------
        >>> safeint("42")
        42

        >>> safeint("2,7-13,17-20")
        [2, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20]

        >>> safeint(3.14)
        3

        >>> safeint("10-15")
        [10, 11, 12, 13, 14, 15]
        """
        try:
            int(s)
            return int(s)
        except ValueError:
            print("COLSPECTOLIST:", s, "is not a legal integer - exiting")
            return None

    # convert value macros, if present
    macros = (
        ("APARC_SUBCORTGRAY", "8-13,17-20,26-28,47-56,58-60,96,97"),
        ("APARC_CORTGRAY", "1000-1035,2000-2035"),
        ("APARC_GRAY", "8-13,17-20,26-28,47-56,58-60,96,97,1000-1035,2000-2035"),
        ("APARC_WHITE", "2,7,41,46,177,219,3000-3035,4000-4035,5001,5002"),
        ("APARC_CSF", "4,5,14,15,24,31,43,44,63,72"),
        (
            "APARC_ALLBUTCSF",
            "2,7-13,17-20,26-28,41,46-56,58-60,96,97,177,219,1000-1035,2000-2035,3000-3035,4000-4035,5001,5002",
        ),
        ("SSEG_GRAY", "3,8,10-13,16-18,26,42,47,49-54,58"),
        ("SSEG_WHITE", "2,7,41,46"),
        ("SSEG_CSF", "4,5,14,15,24,43,44"),
    )
    preprocessedranges = []
    for thisrange in theranges:
        converted = False
        for themacro in macros:
            if thisrange == themacro[0]:
                if debug:
                    print(f"COLSPECTOLIST: macro {thisrange} detected.")
                preprocessedranges += themacro[1].split(",")
                converted = True
        if not converted:
            preprocessedranges.append(thisrange)
    for thisrange in preprocessedranges:
        if debug:
            print("processing range", thisrange)
        theendpoints = thisrange.split("-")
        if len(theendpoints) == 1:
            collist.append(safeint(theendpoints[0]))
        elif len(theendpoints) == 2:
            start = safeint(theendpoints[0])
            end = safeint(theendpoints[1])
            if start < 0:
                print("COLSPECTOLIST:", start, "must be greater than zero")
                return None
            if end < start:
                print("COLSPECTOLIST:", end, "must be greater than or equal to", start)
                return None
            for i in range(start, end + 1):
                collist.append(i)
        else:
            print("COLSPECTOLIST: bad range specification - exiting")
            return None

    return unique(sorted(collist))


def processnamespec(
    maskspec: str, spectext1: str, spectext2: str, debug: bool = False
) -> Tuple[str, Optional[List[int]]]:
    """
    Parse a file specification and extract filename and column specifications.

    This function takes a file specification string and parses it to separate the filename
    from any column specification. The column specification is converted into a list of
    column indices for further processing.

    Parameters
    ----------
    maskspec : str
        Input file specification string containing filename and optional column specification
    debug : bool, optional
        Enable debug output. Default is False

    Returns
    -------
    filename : str
        Parsed filename
    collist : list of int or None
        List of column indices, or None if no column spec provided

    Notes
    -----
    The function uses `parsefilespec` to split the input string and `colspectolist` to
    convert column specifications into lists of integers.

    Examples
    --------
    >>> processnamespec("data.txt:1,3,5")
    ('data.txt', [1, 3, 5])

    >>> processnamespec("data.txt")
    ('data.txt', None)
    """
    thename, colspec = parsefilespec(maskspec)
    if colspec is not None:
        thevals = colspectolist(colspec)
    else:
        thevals = None
    if (thevals is not None) and debug:
        print(spectext1, thename, " = ", thevals, spectext2)
    return thename, thevals


def readcolfromtextfile(inputfilespec: str) -> NDArray:
    """
    Read columns from a text file and return as numpy array.

    This function reads data from a text file, optionally skipping header lines
    and specifying which columns to read. It supports various column specification
    formats and allows for debugging output.

    Parameters
    ----------
    inputfilename : str
        Path to the input text file to read.
    colspec : str, optional
        Column specification string. Can be:
        - None: read all columns
        - Comma-separated column numbers (e.g., "1,3,5")
        - Column ranges (e.g., "1-3,5-7")
        - Single column number (e.g., "3")
    numskip : int, default: 0
        Number of header lines to skip before reading data.
    debug : bool, default: False
        If True, print debug information during execution.
    thedtype : type, default: float
        Data type to convert the read data to.

    Returns
    -------
    NDArray
        Numpy array containing the read data. Shape depends on the number of
        columns specified and the number of rows in the input file.

    Notes
    -----
    - The function uses numpy's genfromtxt internally for reading the file
    - Column indexing starts from 1 (not 0)
    - If colspec is not provided, all columns are read
    - The function handles various text file formats including space and comma delimited data

    Examples
    --------
    >>> # Read all columns from a file
    >>> data = readvecs('data.txt')

    >>> # Read only columns 1, 3, and 5
    >>> data = readvecs('data.txt', colspec='1,3,5')

    >>> # Read columns 2 through 4
    >>> data = readvecs('data.txt', colspec='2-4')

    >>> # Skip first 5 lines and read columns 1 and 3
    >>> data = readvecs('data.txt', colspec='1,3', numskip=5)
    """
    inputfilename, colspec = parsefilespec(inputfilespec)
    if inputfilename is None:
        print("Badly formed file specification", inputfilespec, "- exiting")
        sys.exit()

    inputdata = np.transpose(readvecs(inputfilename, colspec=colspec))
    if np.shape(inputdata)[1] > 1:
        print("specify only one column for input file 1")
        sys.exit()
    else:
        return inputdata[:, 0]


def readvecs(
    inputfilename: str,
    colspec: Optional[str] = None,
    numskip: int = 0,
    debug: bool = False,
    thedtype: np.dtype = np.dtype(np.float64),
) -> NDArray:
    """
    Read vectors from a text file and return them as a transposed numpy array.

    Parameters
    ----------
    inputfilename : str
        The name of the text file to read data from.
    colspec : str, optional
        A string specifying which columns to read. If None, all columns in the first
        line are read. Default is None.
    numskip : int, optional
        Number of lines to skip at the beginning of the file. If 0, the function
        attempts to auto-detect if the first line contains headers. Default is 0.
    thedtype : type, optional
        The data type to convert the read values to. Default is float.
    debug : bool, optional
        If True, print debug information including input parameters and processing
        details. Default is False.

    Returns
    -------
    NDArray
        A 2D numpy array where each row corresponds to a vector read from the file.
        The array is transposed such that each column represents a vector.

    Notes
    -----
    - The function assumes that the input file contains numeric data separated by
      whitespace.
    - If `colspec` is not provided, all columns from the first line are read.
    - If `numskip` is 0, the function attempts to detect whether the first line
      contains headers by trying to convert the first element to a float.
    - The function raises a `ValueError` if any requested column index is out of
      bounds.

    Examples
    --------
    >>> data = readvecs('data.txt')
    >>> data = readvecs('data.txt', colspec='1:3', numskip=1)
    >>> data = readvecs('data.txt', colspec='0,2,4', thedtype=int)
    """
    if debug:
        print(f"inputfilename: {inputfilename}")
        print(f"colspec: {colspec}")
        print(f"numskip: {numskip}")
    with open(inputfilename, "r") as thefile:
        lines = thefile.readlines()
    if colspec is None:
        numvecs = len(lines[0].split())
        collist = range(0, numvecs)
    else:
        collist = colspectolist(colspec)
        if collist[-1] > len(lines[0].split()):
            print("READVECS: too many columns requested - exiting")
            sys.exit()
        if max(collist) > len(lines[0].split()) - 1:
            raise ValueError("READVECS: requested column", max(collist), "too large - exiting")
    inputvec = []
    if numskip == 0:
        try:
            test = float((lines[0].split())[0])
        except ValueError:
            numskip = 1
    for line in lines[numskip:]:
        if len(line) > 1:
            thetokens = line.split()
            thisvec = []
            for vecnum in collist:
                thisvec.append(thedtype.type(thetokens[vecnum]))
            inputvec.append(thisvec)
    theoutarray = np.transpose(np.asarray(inputvec, dtype=thedtype))
    return theoutarray


def readvec(inputfilename: str, numskip: int = 0) -> NDArray:
    """
    Read a timecourse from a text or BIDS TSV file.

    This function reads numerical data from a text file and returns it as a numpy array.
    It can handle both plain text files and BIDS TSV files, with optional column selection
    and debugging output.

    Parameters
    ----------
    inputfilename : str
        Path to the input file
    colnum : int, optional
        Column number to read (0-indexed). If None, reads all columns.
    colname : str, optional
        Column name to read. If None, reads all columns.
    debug : bool, optional
        If True, enables debug output. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - NDArray: The read timecourse data
        - float, optional: Minimum value in the data
        - float, optional: Maximum value in the data

    Notes
    -----
    - The function handles both text files and BIDS TSV files
    - Empty lines are skipped during reading
    - Data is converted to float64 type
    - If both colnum and colname are provided, colnum takes precedence
    - The function returns the minimum and maximum values only when the data is read successfully

    Examples
    --------
    >>> data, min_val, max_val = readtc('timecourse.txt')
    >>> data, min_val, max_val = readtc('bids_file.tsv', colnum=2)
    >>> data, min_val, max_val = readtc('data.txt', colname='signal', debug=True)
    """
    inputvec = []
    with open(inputfilename, "r") as thefile:
        lines = thefile.readlines()
        for line in lines[numskip:]:
            if len(line) > 1:
                inputvec.append(np.float64(line))
    return np.asarray(inputvec, dtype=float)


def readtc(
    inputfilename: str,
    colnum: Optional[int] = None,
    colname: Optional[str] = None,
    debug: bool = False,
) -> Tuple[NDArray, Optional[float], Optional[float]]:
    """
    Read timecourse data from a file, supporting BIDS TSV and other formats.

    This function reads timecourse data from a file, with support for BIDS TSV files
    and generic multi-column text files. For BIDS TSV files, a column name or number
    must be specified. For other file types, column selection is limited to numeric indices.

    Parameters
    ----------
    inputfilename : str
        Path to the input file to read. Can be a BIDS TSV file (`.tsv`) or a generic
        text file with multiple columns.
    colname : str or None, optional
        Column name to read from a BIDS TSV file. Required if the file is a BIDS TSV
        and `colnum` is not specified. Default is None.
    colnum : int or None, optional
        Column number to read from a BIDS TSV file or a generic multi-column file.
        Required for generic files when `colname` is not specified. Default is None.
    debug : bool, optional
        Enable debug output to print intermediate information. Default is False.

    Returns
    -------
    timecourse : NDArray
        The timecourse data as a 1D numpy array.
    inputfreq : float or None
        Sampling frequency (Hz) if available from the file metadata. Default is None.
    inputstart : float or None
        Start time (seconds) if available from the file metadata. Default is None.

    Notes
    -----
    - For BIDS TSV files (`.tsv`), the function reads the specified column using
      `readcolfrombidstsv`, which extracts metadata such as sampling frequency and
      start time.
    - For generic text files, the function transposes the data and selects the
      specified column if `colnum` is provided.
    - If the input file is a `.json` file, it is assumed to contain metadata for
      a BIDS TSV file and is processed accordingly.

    Examples
    --------
    >>> timecourse, freq, start = readtc('data.tsv', colname='signal')
    >>> timecourse, freq, start = readtc('data.txt', colnum=0, debug=True)
    """
    # check file type
    filebase, extension = os.path.splitext(inputfilename)
    inputfreq = None
    inputstart = None
    if debug:
        print("filebase:", filebase)
        print("extension:", extension)
    if extension == ".json":
        if (colnum is None) and (colname is None):
            raise ValueError("You must specify a column name or number to read a bidstsv file")
        if (colnum is not None) and (colname is not None):
            raise ValueError(
                "You must specify a column name or number, but not both, to read a bidstsv file"
            )
        inputfreq, inputstart, timecourse = readcolfrombidstsv(
            inputfilename, columnname=colname, columnnum=colnum, debug=debug
        )
    else:
        timecourse = np.transpose(readvecs(inputfilename))
        if debug:
            print(timecourse.shape)
        if len(timecourse.shape) != 1:
            if (colnum is None) or (colname is not None):
                raise TypeError(
                    "You must specify a column number (not a name) to read a column from a multicolumn file"
                )
            timecourse = timecourse[:, colnum]

    return timecourse, inputfreq, inputstart


def readlabels(inputfilename: str) -> List[str]:
    """
    Write all the key value pairs from a dictionary to a text file.

    Parameters
    ----------
    thedict : dict
        A dictionary containing key-value pairs to be written to file.
    outputfile : str
        The name of the output file where dictionary contents will be saved.
    lineend : {'mac', 'win', 'linux'}, optional
        Line ending style to use. Default is 'linux'.
        - 'mac': Uses carriage return ('\r')
        - 'win': Uses carriage return + line feed ('\r\n')
        - 'linux': Uses line feed ('\n')
    machinereadable : bool, optional
        If True, outputs in a machine-readable format (default is False).
        When False, outputs in a human-readable format with key-value pairs on separate lines.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - The function will overwrite the output file if it already exists.
    - Keys and values are converted to strings before writing.
    - If `machinereadable` is True, the output format may differ from the default human-readable format.

    Examples
    --------
    >>> my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
    >>> writedict(my_dict, 'output.txt')
    # Writes dictionary to output.txt in human-readable format

    >>> writedict(my_dict, 'output.txt', lineend='win')
    # Writes dictionary with Windows-style line endings

    >>> writedict(my_dict, 'output.txt', machinereadable=True)
    # Writes dictionary in machine-readable format
    """
    inputvec = []
    with open(inputfilename, "r") as thefile:
        lines = thefile.readlines()
        for line in lines:
            inputvec.append(line.rstrip())
    return inputvec


def writedict(
    thedict: Dict[str, Any], outputfile: str, lineend: str = "", machinereadable: bool = False
) -> None:
    """
    Write a dictionary to a text file with customizable line endings and formatting.

    Parameters
    ----------
    thedict : dict
        Dictionary containing key-value pairs to be written to file
    outputfile : str
        Path to the output file where dictionary will be written
    lineend : str, optional
        Line ending style to use ('mac', 'win', 'linux'), default is 'linux'
    machinereadable : bool, optional
        If True, write in machine-readable JSON-like format with quotes around keys,
        default is False

    Returns
    -------
    None
        Function writes to file but does not return any value

    Notes
    -----
    - For 'mac' line endings, uses carriage return (`\\r`)
    - For 'win' line endings, uses carriage return + line feed (`\\r\\n`)
    - For 'linux' line endings, uses line feed (`\\n`)
    - When `machinereadable=True`, keys are quoted and formatted with tab separators
    - When `machinereadable=False`, keys are written without quotes

    Examples
    --------
    >>> my_dict = {'name': 'John', 'age': 30}
    >>> writedict(my_dict, 'output.txt', lineend='linux', machinereadable=False)
    >>> writedict(my_dict, 'output.json', lineend='win', machinereadable=True)
    """
    if lineend == "mac":
        thelineending = "\r"
        openmode = "wb"
    elif lineend == "win":
        thelineending = "\r\n"
        openmode = "wb"
    elif lineend == "linux":
        thelineending = "\n"
        openmode = "wb"
    else:
        thelineending = "\n"
        openmode = "w"
    with open(outputfile, openmode) as FILE:
        if machinereadable:
            FILE.writelines("{" + thelineending)
        for key, value in sorted(thedict.items()):
            if machinereadable:
                FILE.writelines('"' + str(key) + '"' + ":\t" + str(value) + thelineending)
            else:
                FILE.writelines(str(key) + ":\t" + str(value) + thelineending)
        if machinereadable:
            FILE.writelines("}" + thelineending)


def readdict(inputfilename: str) -> Dict[str, Any]:
    """
    Read a dictionary from a text file.

    Read a dictionary from a text file where each line contains a key followed by one or more values.
    The key is the first element of each line (with the trailing character removed), and the values
    are the remaining elements on that line.

    Parameters
    ----------
    inputfilename : str
        The name of the input file to read the dictionary from.

    Returns
    -------
    dict
        A dictionary where keys are the first element of each line (with last character removed)
        and values are the remaining elements. If a line contains only one value, that value is
        returned as a string rather than a list. If the file does not exist, an empty dictionary
        is returned.

    Notes
    -----
    - The function assumes that the input file exists and is properly formatted
    - Keys are processed by removing the last character from the first field
    - Values are stored as lists unless there's only one value, in which case it's stored as a string
    - If the file does not exist, a message is printed and an empty dictionary is returned

    Examples
    --------
    >>> # Assuming a file 'data.txt' with content:
    >>> # key1 val1 val2 val3
    >>> # key2 val4
    >>> result = readdict('data.txt')
    >>> print(result)
    {'key': ['val1', 'val2', 'val3'], 'key2': 'val4'}
    """
    if os.path.exists(inputfilename):
        thedict = {}
        with open(inputfilename, "r") as f:
            for line in f:
                values = line.split()
                key = values[0][:-1]
                thevalues = values[1:]
                if len(thevalues) == 1:
                    thevalues = thevalues[0]
                thedict[key] = thevalues
        return thedict
    else:
        print("specified file does not exist")
        return {}


def writevec(thevec: NDArray, outputfile: str, lineend: str = "") -> None:
    """
    Write a vector to a text file, one value per line.

    Parameters
    ----------
    thevec : 1D numpy or python array
        The array to write. Must be a 1D array-like object.
    outputfile : str
        The name of the output file to write to.
    lineend : {'mac', 'win', 'linux'}, optional
        Line ending style to use. Default is 'linux'.
        - 'mac': Use Mac line endings (\r)
        - 'win': Use Windows line endings (\r\n)
        - 'linux': Use Linux line endings (\n)

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function opens the output file in binary mode for all line ending types except
    when an invalid lineend value is provided, in which case it opens in text mode
    with default line endings.

    Examples
    --------
    >>> import numpy as np
    >>> vec = np.array([1, 2, 3, 4, 5])
    >>> writevec(vec, 'output.txt')
    >>> writevec(vec, 'output_win.txt', lineend='win')
    """
    if lineend == "mac":
        thelineending = "\r"
        openmode = "wb"
    elif lineend == "win":
        thelineending = "\r\n"
        openmode = "wb"
    elif lineend == "linux":
        thelineending = "\n"
        openmode = "wb"
    else:
        thelineending = "\n"
        openmode = "w"
    with open(outputfile, openmode) as FILE:
        for i in thevec:
            FILE.writelines(str(i) + thelineending)


def writevectorstotextfile(
    thevecs: NDArray,
    outputfile: str,
    samplerate: float = 1.0,
    starttime: float = 0.0,
    columns: Optional[List[str]] = None,
    compressed: bool = True,
    filetype: str = "text",
    lineend: str = "",
    debug: bool = False,
) -> None:
    """
    Write vectors to a text file in various formats.

    This function writes data vectors to a text file, supporting multiple output formats
    including plain text, CSV, BIDS continuous data, and plain TSV. The format is determined
    by the `filetype` parameter. It supports optional headers, line ending styles, and
    compression for BIDS formats.

    Parameters
    ----------
    thevecs : NDArray
        Data vectors to write. Should be a 2D array where each row is a vector.
    outputfile : str
        Output file path. The extension determines the file format if not explicitly specified.
    samplerate : float, optional
        Sampling rate in Hz. Default is 1.0. Used in BIDS formats.
    starttime : float, optional
        Start time in seconds. Default is 0.0. Used in BIDS formats.
    columns : list of str, optional
        Column names for the output file. If None, no headers are written.
    compressed : bool, optional
        Whether to compress the output file (for BIDS formats). Default is True.
    filetype : str, optional
        Output format. Options are:
        - 'text': Plain text with space-separated values
        - 'csv': Comma-separated values
        - 'bidscontinuous': BIDS continuous data format (TSV with JSON sidecar)
        - 'plaintsv': Plain TSV format without JSON sidecar
        Default is 'text'.
    lineend : str, optional
        Line ending style. Options are:
        - 'mac' (``\r``)
        - 'win' (``\r\n``)
        - 'linux' (``\n``)
        - '' (system default)
        Default is ''.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - For BIDS formats, the function uses `writebidstsv` internally and splits the
      output filename using `niftisplitext`.
    - The `columns` parameter is only used when writing headers.
    - The `lineend` parameter controls how newlines are written to the file.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> writevectorstotextfile(data, "output.txt", filetype="text")
    >>> writevectorstotextfile(data, "output.csv", filetype="csv", columns=["A", "B", "C"])
    >>> writevectorstotextfile(data, "output.tsv", filetype="bidscontinuous", samplerate=100.0)
    """
    if filetype == "text":
        writenpvecs(thevecs, outputfile, headers=columns, lineend=lineend)
    elif filetype == "csv":
        writenpvecs(thevecs, outputfile, headers=columns, ascsv=True, lineend=lineend)
    elif filetype == "bidscontinuous":
        writebidstsv(
            niftisplitext(outputfile)[0],
            thevecs,
            samplerate,
            compressed=compressed,
            columns=columns,
            starttime=starttime,
            append=False,
            colsinjson=True,
            colsintsv=False,
            debug=debug,
        )
    elif filetype == "plaintsv":
        writebidstsv(
            niftisplitext(outputfile)[0],
            thevecs,
            samplerate,
            compressed=compressed,
            columns=columns,
            starttime=starttime,
            append=False,
            colsinjson=False,
            colsintsv=True,
            omitjson=True,
            debug=debug,
        )

    else:
        raise ValueError("illegal file type")


# rewritten to guarantee file closure, combines writenpvec and writenpvecs
def writenpvecs(
    thevecs: NDArray,
    outputfile: str,
    ascsv: bool = False,
    headers: Optional[List[str]] = None,
    altmethod: bool = True,
    lineend: str = "",
) -> None:
    """
    Write out a two dimensional numpy array to a text file.

    This function writes a numpy array to a text file, with options for
    CSV-style output, custom headers, and line ending styles.

    Parameters
    ----------
    thevecs : NDArray
        A 1D or 2D numpy array containing the data to be written. If 1D,
        the array is written as a single column. If 2D, each column is
        written as a separate line in the output file.
    outputfile : str
        The path to the output file where the data will be written.
    ascsv : bool, optional
        If True, use comma as the separator; otherwise, use tab. Default is False.
    headers : list of str, optional
        A list of header strings to write at the beginning of the file.
        If provided, the number of headers must match the number of columns
        in the data (for 2D arrays) or 1 (for 1D arrays).
    altmethod : bool, optional
        If True, use an optimized method for writing 2D data. If False,
        use a nested loop approach. Default is True.
    lineend : str, optional
        Line ending style to use. Options are 'mac' (\r), 'win' (\r\n),
        'linux' (\n), or empty string (uses system default). Default is 'linux'.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - For 2D arrays, data is written column-wise.
    - When `altmethod` is True, the function uses vectorized operations
      for better performance.
    - If `headers` are provided, they are written as the first line
      in the file, separated by the chosen delimiter.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> writenpvecs(data, 'output.txt')
    # Writes data as tab-separated columns to 'output.txt'

    >>> headers = ['Col1', 'Col2', 'Col3']
    >>> writenpvecs(data, 'output.csv', ascsv=True, headers=headers)
    # Writes CSV-formatted data with headers to 'output.csv'
    """
    theshape = np.shape(thevecs)
    if lineend == "mac":
        thelineending = "\r"
        openmode = "wb"
    elif lineend == "win":
        thelineending = "\r\n"
        openmode = "wb"
    elif lineend == "linux":
        thelineending = "\n"
        openmode = "wb"
    else:
        thelineending = "\n"
        openmode = "w"
    if ascsv:
        theseparator = ","
    else:
        theseparator = "\t"
    if headers is not None:
        if thevecs.ndim == 2:
            if len(headers) != theshape[0]:
                raise ValueError("number of header lines must equal the number of data columns")
        else:
            if len(headers) != 1:
                raise ValueError("number of header lines must equal the number of data columns")
    with open(outputfile, openmode) as FILE:
        if headers is not None:
            FILE.writelines(theseparator.join(headers) + thelineending)
        if thevecs.ndim == 2:
            for i in range(0, theshape[1]):
                if altmethod:
                    outline = theseparator.join(thevecs[:, i].astype(str).tolist()) + thelineending
                    FILE.writelines(outline)
                else:
                    for j in range(0, theshape[0]):
                        FILE.writelines(str(thevecs[j, i]) + "\t")
                    FILE.writelines(thelineending)
        else:
            for i in range(0, theshape[0]):
                FILE.writelines(str(thevecs[i]) + thelineending)
