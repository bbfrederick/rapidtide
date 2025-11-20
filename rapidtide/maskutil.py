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
import bisect
import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from nilearn import masking
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.stats as tide_stats

LGR = logging.getLogger("GENERAL")


def resampmask(themask: ArrayLike, thetargetres: float) -> NDArray:
    """Resample a mask to a target resolution.

    Parameters
    ----------
    themask : array_like
        Input mask array to be resampled.
    thetargetres : float
        Target resolution for the resampled mask.

    Returns
    -------
    NDArray
        Resampled mask array with the specified target resolution.

    Notes
    -----
    This function currently returns the input mask unchanged. A full implementation
    would perform actual resampling operations to adjust the mask to the target
    resolution.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> resampled = resampmask(mask, 0.5)
    >>> print(resampled)
    [[0 1 0]
     [1 1 1]
     [0 1 0]]
    """
    resampmask = themask
    return themask


def makeepimask(nim: Any) -> Any:
    """Compute EPI mask from neuroimaging data.

    This function computes an EPI (Echo Planar Imaging) mask from neuroimaging data
    using the masking.compute_epi_mask function from nilearn.

    Parameters
    ----------
    nim : Any
        Neuroimaging data object (typically Nifti1Image or similar) from which
        to compute the EPI mask. This can be a nibabel image object or array-like
        data representing neuroimaging volumes.

    Returns
    -------
    Any
        EPI mask computed from the input neuroimaging data. The return type
        depends on the underlying masking.compute_epi_mask implementation and
        typically represents a binary mask image.

    Notes
    -----
    This function is a wrapper around nilearn's masking.compute_epi_mask function
    and is commonly used in neuroimaging preprocessing pipelines to automatically
    generate brain masks from EPI functional MRI data.

    Examples
    --------
    >>> import nibabel as nib
    >>> from nilearn.masking import compute_epi_mask
    >>> # Assuming 'img' is a nibabel image object
    >>> mask = makeepimask(img)
    >>> print(mask.shape)
    """
    return masking.compute_epi_mask(nim)


def maketmask(
    filename: str, timeaxis: ArrayLike, maskvector: NDArray, debug: bool = False
) -> NDArray:
    """Create a temporal mask from time interval data.

    This function reads time interval data from a file and generates a binary mask
    vector indicating which time points should be included in analysis. The mask
    can be generated from either a simple vector of nonzero values or from time
    intervals specified by start and duration values.

    Parameters
    ----------
    filename : str
        Path to the file containing time interval data. The file should contain
        either a single vector of values or two rows of data (start times and durations).
    timeaxis : ArrayLike
        Array of time points corresponding to the fMRI time series. Used to map
        time intervals to specific time indices.
    maskvector : NDArray
        Pre-allocated array to store the resulting temporal mask. Should be the
        same length as the fMRI time series.
    debug : bool, optional
        If True, enables debug logging output. Default is False.

    Returns
    -------
    NDArray
        Binary mask vector where 1.0 indicates time points to include and 0.0
        indicates time points to exclude.

    Notes
    -----
    The function handles two input formats:
    1. Single row: Each nonzero value indicates inclusion of the corresponding time point
    2. Two rows: First row contains start times, second row contains durations

    Examples
    --------
    >>> import numpy as np
    >>> timeaxis = np.arange(0, 100, 2)  # 50 time points
    >>> maskvector = np.zeros(50)
    >>> mask = maketmask('time_intervals.txt', timeaxis, maskvector)
    """
    inputdata = tide_io.readvecs(filename)
    theshape = np.shape(inputdata)
    if theshape[0] == 1:
        # this is simply a vector, one per TR.  If the value is nonzero, include the point, otherwise don't
        if theshape[1] == len(timeaxis):
            maskvector = np.where(inputdata[0, :] > 0.0, 1.0, 0.0)
        else:
            raise ValueError("tmask length does not match fmri data")
    else:
        maskvector *= 0.0
        for idx in range(0, theshape[1]):
            starttime = inputdata[0, idx]
            endtime = starttime + inputdata[1, idx]
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(maskvector) - 1))
            maskvector[startindex:endindex] = 1.0
            LGR.info(f"{starttime}, {startindex}, {endtime}, {endindex}")
    return maskvector


def readamask(
    maskfilename: str,
    nim_hdr: Any,
    xsize: int,
    istext: bool = False,
    valslist: Optional[list] = None,
    thresh: Optional[float] = None,
    maskname: str = "the",
    tolerance: float = 1.0e-3,
    debug: bool = False,
) -> NDArray:
    """
    Read and process a mask file, returning a binary mask array.

    This function reads a mask from either a text file or NIfTI format, applies
    optional thresholding or value selection, and returns a binary mask array
    compatible with the input data dimensions.

    Parameters
    ----------
    maskfilename : str
        Path to the mask file. Can be in text format (if `istext=True`) or NIfTI format.
    nim_hdr : Any
        Header information from the NIfTI file of the input data. Used for spatial
        dimension matching.
    xsize : int
        Expected size of the first dimension of the mask array.
    istext : bool, optional
        If True, the mask is read as a text file. Default is False.
    valslist : list of int, optional
        List of values to include in the mask. If provided, only voxels matching
        these values are set to 1. Default is None.
    thresh : float, optional
        Threshold value for binarizing the mask. If provided, voxels greater than
        `thresh` are set to 1, others to 0. Default is None.
    maskname : str, optional
        Name of the mask for logging and error messages. Default is "the".
    tolerance : float, optional
        Tolerance for spatial dimension matching between the mask and input data.
        Default is 1e-3.
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    NDArray
        A binary mask array of type `uint16`, where 1 indicates included voxels
        and 0 indicates excluded voxels.

    Notes
    -----
    - If `istext=True`, the mask file is expected to contain numeric values
      arranged in a single column or row.
    - If `thresh` is provided, the mask is binarized based on the threshold.
    - If `valslist` is provided, only voxels matching values in the list are set to 1.
    - The function raises a `ValueError` if spatial dimensions of the mask and
      input data do not match within the specified tolerance.

    Examples
    --------
    >>> mask = readamask(
    ...     maskfilename="mask.nii.gz",
    ...     nim_hdr=hdr,
    ...     xsize=64,
    ...     thresh=0.5,
    ...     maskname="brain"
    ... )
    >>> print(mask.shape)
    (64, 64, 64)
    """
    LGR.debug(f"readamask called with filename: {maskfilename} vals: {valslist}")
    if debug:
        print("getmaskset:")
        print(f"{maskname=}")
        print(f"\tincludefilename={maskfilename}")
        print(f"\tincludevals={valslist}")
        print(f"\t{istext=}")
        print(f"\t{tolerance=}")
    if istext:
        maskarray = tide_io.readvecs(maskfilename).astype("uint16")
        theshape = np.shape(maskarray)
        theincludexsize = theshape[0]
        if not theincludexsize == xsize:
            raise ValueError(
                f"Dimensions of {maskname} mask do not match the input data - exiting"
            )
    else:
        themask, maskarray, mask_hdr, maskdims, masksizes = tide_io.readfromnifti(maskfilename)
        if not tide_io.checkspacematch(mask_hdr, nim_hdr, tolerance=tolerance):
            raise ValueError(f"Dimensions of {maskname} mask do not match the fmri data - exiting")
        if thresh is None:
            maskarray = np.round(maskarray, 0).astype("uint16")
        else:
            maskarray = np.where(maskarray > thresh, 1, 0).astype("uint16")

    if valslist is not None:
        tempmask = (0 * maskarray).astype("uint16")
        for theval in valslist:
            LGR.debug(f"looking for voxels matching {theval}")
            tempmask[np.where(maskarray - theval == 0)] += 1
        maskarray = np.where(tempmask > 0, 1, 0)

    maskarray = np.where(maskarray > 0, 1, 0).astype("uint16")
    return maskarray


def getmaskset(
    maskname: str,
    includename: Optional[str],
    includevals: Optional[list],
    excludename: Optional[str],
    excludevals: Optional[list],
    datahdr: Any,
    numspatiallocs: int,
    extramask: Optional[str] = None,
    extramaskthresh: float = 0.1,
    istext: bool = False,
    tolerance: float = 1.0e-3,
    debug: bool = False,
) -> Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
    """
    Construct and return masks for inclusion, exclusion, and an additional mask.

    This function builds masks based on provided parameters, including optional
    inclusion and exclusion criteria, as well as an extra mask. It performs
    validation to ensure that the resulting masks are not empty or overly restrictive.

    Parameters
    ----------
    maskname : str
        Name of the mask being constructed, used for logging and labeling.
    includename : str, optional
        File name or identifier for the mask to be used for inclusion.
    includevals : list of float, optional
        List of values to include in the inclusion mask. If ``None``, all values
        are included.
    excludename : str, optional
        File name or identifier for the mask to be used for exclusion.
    excludevals : list of float, optional
        List of values to exclude from the exclusion mask. If ``None``, no values
        are excluded.
    datahdr : Any
        Header information for the data, passed to mask reading functions.
    numspatiallocs : int
        Number of spatial locations in the data.
    extramask : str, optional
        File name or identifier for an additional mask to be applied.
    extramaskthresh : float, default=0.1
        Threshold value for the extra mask, used when reading the mask.
    istext : bool, default=False
        If ``True``, treat input files as text-based.
    tolerance : float, default=1e-03
        Tolerance for floating-point comparisons when reading masks.
    debug : bool, default=False
        If ``True``, print debug information during execution.

    Returns
    -------
    tuple of (Optional[NDArray], Optional[NDArray], Optional[NDArray])
        A tuple containing:
        - ``internalincludemask``: The inclusion mask, reshaped to ``numspatiallocs``.
        - ``internalexcludemask``: The exclusion mask, reshaped to ``numspatiallocs``.
        - ``internalextramask``: The extra mask, reshaped to ``numspatiallocs``.

    Notes
    -----
    - If both inclusion and exclusion masks are specified, the function ensures
      that at least one voxel remains after applying both masks.
    - If an extra mask is specified, it is applied in combination with the inclusion
      and exclusion masks.
    - The function raises a ``ValueError`` if any of the resulting masks are invalid:
      e.g., empty inclusion mask, or masks that leave no voxels.

    Examples
    --------
    >>> maskname = "brain"
    >>> includename = "brain_include.nii"
    >>> includevals = [1]
    >>> excludename = "ventricles.nii"
    >>> excludevals = [1]
    >>> datahdr = header
    >>> numspatiallocs = 10000
    >>> includemask, excludemask, extramask = getmaskset(
    ...     maskname, includename, includevals, excludename, excludevals,
    ...     datahdr, numspatiallocs
    ... )
    """
    internalincludemask = None
    internalexcludemask = None
    internalextramask = None

    if debug:
        print("getmaskset:")
        print(f"{maskname=}")
        print(f"\t{includename=}")
        print(f"\t{includevals=}")
        print(f"\t{excludename=}")
        print(f"\t{excludevals=}")
        print(f"\t{istext=}")
        print(f"\t{tolerance=}")
        print(f"\t{extramask=}")
        print(f"\t{extramaskthresh=}")

    if includename is not None:
        LGR.info(f"constructing {maskname} include mask")
        theincludemask = readamask(
            includename,
            datahdr,
            numspatiallocs,
            istext=istext,
            valslist=includevals,
            maskname=f"{maskname} include",
            tolerance=tolerance,
        )
        internalincludemask = theincludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalincludemask) == 0:
            raise ValueError(
                f"ERROR: there are no voxels in the {maskname} include mask - exiting"
            )

    if excludename is not None:
        LGR.info(f"constructing {maskname} exclude mask")
        theexcludemask = readamask(
            excludename,
            datahdr,
            numspatiallocs,
            istext=istext,
            valslist=excludevals,
            maskname=f"{maskname} exclude",
            tolerance=tolerance,
        )
        internalexcludemask = theexcludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalexcludemask) == numspatiallocs:
            raise ValueError(
                f"ERROR: the {maskname} exclude mask does not leave any voxels - exiting"
            )

    if extramask is not None:
        LGR.info(f"reading {maskname} extra mask")
        internalextramask = readamask(
            extramask,
            datahdr,
            numspatiallocs,
            istext=istext,
            valslist=None,
            thresh=extramaskthresh,
            maskname=f"{maskname} extra",
            tolerance=tolerance,
        )

    if (internalincludemask is not None) and (internalexcludemask is not None):
        if tide_stats.getmasksize(internalincludemask * (1 - internalexcludemask)) == 0:
            raise ValueError(
                f"ERROR: the {maskname} include and exclude masks do not leave any voxels between them - exiting"
            )
        if internalextramask is not None:
            if (
                tide_stats.getmasksize(
                    internalincludemask * (1 - internalexcludemask) * internalextramask
                )
                == 0
            ):
                raise ValueError(
                    f"ERROR: the {maskname} include, exclude, and extra masks do not leave any voxels between them - exiting"
                )

    return internalincludemask, internalexcludemask, internalextramask


def getregionsignal(
    indata: NDArray,
    filter: Optional[Any] = None,
    Fs: float = 1.0,
    includemask: Optional[NDArray] = None,
    excludemask: Optional[NDArray] = None,
    signalgenmethod: str = "sum",
    pcacomponents: Union[float, str] = 0.8,
    signame: str = "global mean",
    rt_floattype: type = np.float64,
    debug: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Compute a global signal from a 2D array of voxel data using specified methods.

    This function computes a global signal from input data by applying optional masking,
    and then combining voxel signals using one of several methods: sum, meanscale, PCA,
    or random. The resulting signal can be filtered and normalized.

    Parameters
    ----------
    indata : ndarray
        Input 2D array of shape (n_voxels, n_timepoints) containing voxel time series.
    filter : optional
        A filter object with an `apply` method to apply to the computed signal.
        Default is None.
    Fs : float, optional
        Sampling frequency (Hz) used for filtering. Default is 1.0.
    includemask : ndarray, optional
        Binary mask to include specific voxels. Voxels not included will be ignored.
        Default is None.
    excludemask : ndarray, optional
        Binary mask to exclude specific voxels. Voxels marked as 1 will be excluded.
        Default is None.
    signalgenmethod : str, optional
        Method used to generate the global signal. Options are:
        - "sum": Mean of selected voxels (default).
        - "meanscale": Scale each voxel by its mean before averaging.
        - "pca": Use PCA to reduce dimensionality and compute signal.
        - "random": Generate a random signal.
        Default is "sum".
    pcacomponents : float or str, optional
        Number of PCA components to use. If float, specifies number of components;
        if "mle", uses maximum likelihood estimation. Default is 0.8.
    signame : str, optional
        Name of the signal for logging purposes. Default is "global mean".
    rt_floattype : type, optional
        Data type for internal computations. Default is np.float64.
    debug : bool, optional
        If True, print debugging information. Default is False.

    Returns
    -------
    tuple of ndarray
        A tuple containing:
        - normalized_global_signal : ndarray
          The normalized global signal of shape (n_timepoints,).
        - final_mask : ndarray
          The final voxel mask used in computation, shape (n_voxels,).

    Notes
    -----
    - The function applies `includemask` and `excludemask` sequentially to define
      which voxels are used in signal computation.
    - For "pca" method, PCA is applied to the transposed scaled voxel data.
    - If filtering is applied, the signal is filtered in-place using the provided filter.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> indata = np.random.rand(100, 50)
    >>> signal, mask = getregionsignal(indata, signalgenmethod="sum")
    >>> print(signal.shape)
    (50,)
    """
    # Start with all voxels
    themask = np.ones_like(indata[:, 0])

    # modify the mask if needed
    if includemask is not None:
        themask = themask * includemask
    if excludemask is not None:
        themask = themask * (1 - excludemask)

    # combine all the voxels using one of the three methods
    globalmean = (indata[0, :]).astype(rt_floattype)
    thesize = np.shape(themask)
    numvoxelsused = int(np.sum(np.where(themask > 0.0, 1, 0)))
    selectedvoxels = indata[np.where(themask > 0.0), :][0]
    if debug:
        print(f"getregionsignal: {selectedvoxels.shape=}")
    LGR.info(f"constructing global mean signal using {signalgenmethod}")
    if signalgenmethod == "sum":
        globalmean = np.mean(selectedvoxels, axis=0)
        globalmean -= np.mean(globalmean)
        if debug:
            print("Sum method")
            print(f"getregionsignal: {globalmean.shape=}")
    elif signalgenmethod == "meanscale":
        themean = np.mean(indata, axis=1)
        for vox in range(0, thesize[0]):
            if themask[vox] > 0.0:
                if themean[vox] != 0.0:
                    globalmean += indata[vox, :] / themean[vox] - 1.0
        if debug:
            print("Meanscale method")
            print(f"getregionsignal: {globalmean.shape=}")
    elif signalgenmethod == "pca":
        themean = np.mean(indata, axis=1)
        thevar = np.var(indata, axis=1)
        scaledvoxels = np.zeros_like(selectedvoxels)
        for vox in range(0, selectedvoxels.shape[0]):
            scaledvoxels[vox, :] = selectedvoxels[vox, :] - themean[vox]
            if thevar[vox] > 0.0:
                scaledvoxels[vox, :] = selectedvoxels[vox, :] / thevar[vox]
        try:
            thefit = PCA(n_components=pcacomponents).fit(scaledvoxels)
        except ValueError:
            if pcacomponents == "mle":
                LGR.warning("mle estimation failed - falling back to pcacomponents=0.8")
                thefit = PCA(n_components=0.8).fit(scaledvoxels)
            else:
                raise ValueError("unhandled math exception in PCA refinement - exiting")

        varex = 100.0 * np.cumsum(thefit.explained_variance_ratio_)[len(thefit.components_) - 1]
        # thetransform = thefit.transform(np.transpose(scaledvoxels))
        thetransform = thefit.transform(scaledvoxels)
        cleanedvoxels = thefit.inverse_transform(thetransform) * thevar[:, None]
        globalmean = np.mean(cleanedvoxels, axis=0)
        globalmean -= np.mean(globalmean)
        if debug:
            print("PCA method")
            print(
                f"getregionsignal: {cleanedvoxels.shape=}, {thetransform.shape=}, {scaledvoxels.shape=}, {globalmean.shape=}"
            )
            print(
                f"getregionsignal: {(thefit.components_).shape=}, {thefit.n_samples_=}, {thefit.n_features_in_=}"
            )
            print(f"getregionsignal: {varex=}")
        LGR.info(
            f"Using {len(thefit.components_)} component(s), accounting for "
            f"{varex:.2f}% of the variance"
        )
    elif signalgenmethod == "random":
        globalmean = np.random.standard_normal(size=len(globalmean))
        if debug:
            print("Random method")
            print(f"getregionsignal: {globalmean.shape=}")
    else:
        raise ValueError(f"illegal signal generation method: {signalgenmethod}")
    LGR.info(f"used {numvoxelsused} voxels to calculate {signame} signal")
    if filter is not None:
        globalmean = filter.apply(Fs, globalmean)
    if debug:
        print(f"getregionsignal: {globalmean=}")
    return tide_math.stdnormalize(globalmean), themask


def saveregionaltimeseries(
    tcdesc: str,
    tcname: str,
    fmridata: NDArray,
    includemask: NDArray,
    fmrifreq: float,
    outputname: str,
    filter: Optional[Any] = None,
    initfile: bool = False,
    excludemask: Optional[NDArray] = None,
    filedesc: str = "regional",
    suffix: str = "",
    signalgenmethod: str = "sum",
    pcacomponents: Union[float, str] = 0.8,
    rt_floattype: type = np.float64,
    debug: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Save regional time series data from fMRI data to a BIDS-compatible TSV file.

    This function extracts regional signal time courses from fMRI data using the
    specified masking and filtering parameters, then writes the results to a
    BIDS-style TSV file. The function supports various signal generation methods
    and can handle both inclusive and exclusive masking.

    Parameters
    ----------
    tcdesc : str
        Description of the time course for the output file header
    tcname : str
        Name of the time course to be used in the output file column header
    fmridata : NDArray
        4D fMRI data array (time x x x y z)
    includemask : NDArray
        Binary mask defining regions to include in the analysis
    fmrifreq : float
        Sampling frequency of the fMRI data (Hz)
    outputname : str
        Base name for the output file (without extension)
    filter : Optional[Any], default=None
        Filter to apply to the time series data
    initfile : bool, default=False
        If True, initializes a new file; if False, appends to existing file
    excludemask : Optional[NDArray], default=None
        Binary mask defining regions to exclude from the analysis
    filedesc : str, default="regional"
        Description string for the output file name
    suffix : str, default=""
        Suffix to append to the column name in the output file
    signalgenmethod : str, default="sum"
        Method for generating the signal ('sum', 'mean', 'pca', etc.)
    pcacomponents : Union[float, str], default=0.8
        Number of PCA components to use (or fraction of variance explained)
    rt_floattype : np.dtype, default=np.float64
        Data type for floating point operations
    debug : bool, default=False
        If True, enables debug mode for additional logging

    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple containing:
        - thetimecourse : NDArray
          The extracted time course data
        - themask : NDArray
          The mask used for extraction

    Notes
    -----
    The function uses `getregionsignal` to compute the regional signal and
    `tide_io.writebidstsv` to write the output file in BIDS TSV format.
    The output file name follows the pattern:
    {outputname}_desc-{filedesc}_timeseries.tsv

    Examples
    --------
    >>> import numpy as np
    >>> fmri_data = np.random.rand(100, 10, 10, 10)
    >>> mask = np.ones((10, 10, 10))
    >>> timecourse, mask_used = saveregionaltimeseries(
    ...     tcdesc="mean_signal",
    ...     tcname="signal",
    ...     fmridata=fmri_data,
    ...     includemask=mask,
    ...     fmrifreq=2.0,
    ...     outputname="sub-01_task-rest",
    ...     filter=None,
    ...     initfile=True
    ... )
    """
    thetimecourse, themask = getregionsignal(
        fmridata,
        filter=filter,
        Fs=fmrifreq,
        includemask=includemask,
        excludemask=excludemask,
        signalgenmethod=signalgenmethod,
        pcacomponents=pcacomponents,
        signame=tcdesc,
        rt_floattype=rt_floattype,
        debug=debug,
    )
    tide_io.writebidstsv(
        f"{outputname}_desc-{filedesc}_timeseries",
        thetimecourse,
        fmrifreq,
        columns=[f"{tcname}{suffix}"],
        extraheaderinfo={
            "Description": "Regional timecourse averages",
        },
        append=(not initfile),
    )
    return thetimecourse, themask
