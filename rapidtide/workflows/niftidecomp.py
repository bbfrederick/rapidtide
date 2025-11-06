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
import argparse
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA, FastICA, SparsePCA

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_float, is_valid_file


def _get_parser(decompaxis: Any) -> Any:
    """
    Argument parser for spatialdecomp and temporaldecomp.

    This function constructs and returns an `argparse.ArgumentParser` object
    configured for either spatial or temporal decomposition tasks. The parser
    is tailored to handle common arguments required for PCA or ICA decomposition
    of neuroimaging data (e.g., NIfTI files).

    Parameters
    ----------
    decompaxis : Any
        Specifies the axis along which decomposition is performed. Must be
        either "temporal" or "spatial". Determines the program name and
        description of the parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the specified decomposition type.

    Raises
    ------
    ValueError
        If `decompaxis` is not "temporal" or "spatial".

    Notes
    -----
    The returned parser includes support for:
    - Input data file validation
    - Output root naming
    - Data masking
    - Number of components to extract
    - Spatial smoothing
    - Decomposition type (PCA, ICA, sparse)
    - Preprocessing options such as demeaning and variance normalization

    Examples
    --------
    >>> parser = _get_parser("temporal")
    >>> args = parser.parse_args(['data.nii', 'output_root'])
    """
    if decompaxis == "temporal":
        parser = argparse.ArgumentParser(
            prog="temporaldecomp",
            description="Perform PCA or ICA decomposition on a data file in the time dimension.",
            allow_abbrev=False,
        )
    elif decompaxis == "spatial":
        parser = argparse.ArgumentParser(
            prog="spatialdecomp",
            description="Perform PCA or ICA decomposition on a data file in the spatial dimension.",
            allow_abbrev=False,
        )
    else:
        raise ValueError(f"Illegal decomposition type: {type}")

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit",
    )
    parser.add_argument("outputroot", help="The root name for the output nifti files")

    # Optional arguments
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    parser.add_argument(
        "--ncomp",
        dest="ncomp",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="NCOMPS",
        help=("The number of PCA/ICA components to return (default is to estimate the number)."),
        default=-1.0,
    )
    parser.add_argument(
        "--smooth",
        dest="sigma",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="SIGMA",
        help=("Spatially smooth the input data with a SIGMA mm kernel."),
        default=0.0,
    )
    parser.add_argument(
        "--type",
        dest="decomptype",
        action="store",
        type=str,
        choices=["pca", "sparse", "ica"],
        help=("Type of decomposition to perform. Default is pca."),
        default="pca",
    )
    parser.add_argument(
        "--nodemean",
        dest="demean",
        action="store_false",
        help=("Do not demean data prior to decomposition."),
        default=True,
    )
    parser.add_argument(
        "--novarnorm",
        dest="varnorm",
        action="store_false",
        help=("Do not variance normalize data prior to decomposition."),
        default=True,
    )

    return parser


def _get_parser_temporal() -> Any:
    """
    Get parser for the temporal variant of the program.

    This function is a convenience wrapper that calls the internal `_get_parser`
    function with the argument "temporal" to obtain the temporal parser object.

    Returns
    -------
    Any
        The temporal parser instance. The specific type depends on the implementation
        of the underlying `_get_parser` function and the temporal parser configuration.

    Notes
    -----
    This function is intended for internal use only and should not be called directly
    by end users. The returned parser is typically used for parsing temporal data
    such as dates, times, and time intervals.

    Examples
    --------
    >>> parser = _get_parser_temporal()
    >>> # Use parser for temporal data processing
    """
    return _get_parser("temporal")


def _get_parser_spatial() -> Any:
    """
    Get parser for the spatial variant of the program.

    Returns
    -------
    Any
        The spatial parser object returned by `_get_parser` function.

    Notes
    -----
    This function is a convenience wrapper that calls `_get_parser` with the
    argument "spatial". It is used internally to retrieve spatial parsing
    capabilities for the application.

    Examples
    --------
    >>> parser = _get_parser_spatial()
    >>> isinstance(parser, SomeParserClass)
    True
    """
    return _get_parser("spatial")


def transposeifspatial(data: Any, decompaxis: str = "temporal") -> None:
    """
    Transpose data if decomposition axis is spatial.

    Parameters
    ----------
    data : Any
        Input data to be transposed if necessary
    decompaxis : str, default="temporal"
        Decomposition axis specification. If "spatial", the data will be transposed
        using numpy.transpose() function. Otherwise, the original data is returned
        unchanged.

    Returns
    -------
    Any
        Transposed data if decompaxis is "spatial", otherwise returns the original data

    Notes
    -----
    This function provides a conditional transpose operation based on the decomposition
    axis specification. It's useful in scenarios where data processing needs to
    adapt based on the spatial or temporal nature of the decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> transposed = transposeifspatial(data, decompaxis="spatial")
    >>> print(transposed)
    [[1 4]
     [2 5]
     [3 6]]

    >>> original = transposeifspatial(data, decompaxis="temporal")
    >>> print(original)
    [[1 2 3]
     [4 5 6]]
    """
    if decompaxis == "spatial":
        return np.transpose(data)
    else:
        return data


def niftidecomp_workflow(
    decompaxis: Any,
    datafilelist: Any,
    datamaskname: Optional[Any] = None,
    decomptype: str = "pca",
    pcacomponents: float = 0.5,
    icacomponents: Optional[Any] = None,
    varnorm: bool = True,
    demean: bool = True,
    sigma: float = 0.0,
) -> None:
    """
    Perform PCA or ICA decomposition on 4D NIfTI data along a specified axis.

    This function reads a list of NIfTI files, applies optional smoothing and masking,
    and performs either Principal Component Analysis (PCA) or Independent Component Analysis (ICA)
    on the data. The decomposition is performed along either the temporal or spatial axis,
    depending on the `decompaxis` parameter. The results include component images, coefficients,
    and inverse-transformed data.

    Parameters
    ----------
    decompaxis : Any
        Axis along which to perform decomposition. Either "temporal" or "spatial".
    datafilelist : Any
        List of paths to NIfTI data files to be processed.
    datamaskname : Any, optional
        Path to a NIfTI mask file. If provided, only voxels within the mask will be processed.
        Default is None.
    decomptype : str, optional
        Type of decomposition to perform. Either "pca" or "ica". Default is "pca".
    pcacomponents : float, optional
        Number of components to retain for PCA. If less than 1, specifies the fraction of
        variance to retain. Default is 0.5.
    icacomponents : Any, optional
        Number of components to retain for ICA. If None, all components are returned.
        Default is None.
    varnorm : bool, optional
        Whether to perform variance normalization. Default is True.
    demean : bool, optional
        Whether to demean the data. Default is True.
    sigma : float, optional
        Standard deviation for Gaussian smoothing. If 0.0, no smoothing is applied.
        Default is 0.0.

    Returns
    -------
    tuple
        A tuple containing:
        - outputcomponents : ndarray
            Decomposed components reshaped to original spatial dimensions.
        - outputcoefficients : ndarray
            Coefficients of the decomposition.
        - outinvtrans : ndarray
            Inverse-transformed data.
        - exp_var : ndarray
            Explained variance for each component.
        - exp_var_pct : ndarray
            Explained variance percentage for each component.
        - datafile_hdr : dict
            Header of the input data file.
        - datafiledims : list
            Dimensions of the input data file.
        - datafilesizes : list
            Size parameters of the input data file.

    Notes
    -----
    - The function assumes all input NIfTI files have matching spatial and temporal dimensions.
    - For ICA, the `FastICA` algorithm is used.
    - For PCA, both regular PCA and Sparse PCA are supported.
    - If `datamaskname` is provided, the mask is applied before decomposition.
    - The output data is reshaped to match the original NIfTI dimensions.

    Examples
    --------
    >>> niftidecomp_workflow(
    ...     decompaxis="temporal",
    ...     datafilelist=["data1.nii", "data2.nii"],
    ...     decomptype="pca",
    ...     pcacomponents=0.95,
    ...     varnorm=True,
    ...     demean=True
    ... )
    """
    print(f"Will perform {decomptype} analysis along the {decompaxis} axis")

    if decompaxis == "temporal":
        decompaxisnum = 1
    else:
        decompaxisnum = 0

    # read in data
    print("reading in data arrays")
    if datamaskname is not None:
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(datamaskname)

    numfiles = len(datafilelist)
    for idx, datafile in enumerate(datafilelist):
        (
            datafile_img,
            datafile_data,
            datafile_hdr,
            datafiledims,
            datafilesizes,
        ) = tide_io.readfromnifti(datafile)

        if idx == 0:
            xsize, ysize, numslices, timepoints = tide_io.parseniftidims(datafiledims)
            xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(datafilesizes)
            totaltimepoints = timepoints * numfiles
            originaldatafiledims = datafiledims.copy()

            fulldataarray = np.zeros((xsize, ysize, numslices, timepoints * numfiles), dtype=float)
        else:
            if (not tide_io.checkspacedimmatch(datafiledims, originaldatafiledims)) or (
                not tide_io.checktimematch(datafiledims, originaldatafiledims)
            ):
                print("all input data files must have the same dimensions")
                exit()

        # smooth the data
        if sigma > 0.0:
            print("smoothing data")
            for i in range(timepoints):
                datafile_data[:, :, :, i] = tide_filt.ssmooth(
                    xdim, ydim, slicethickness, sigma, datafile_data[:, :, :, i]
                )
        fulldataarray[:, :, :, idx * timepoints : (idx + 1) * timepoints] = datafile_data[
            :, :, :, :
        ]

    # check dimensions
    if datamaskname is not None:
        print("checking mask dimensions")
        if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
            print("input mask spatial dimensions do not match image")
            exit()
        if not (tide_io.checktimematch(datafiledims, datamaskdims) or datamaskdims[4] == 1):
            print("input mask time dimension does not match image")
            exit()

    # allocating arrays
    print("reshaping arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = fulldataarray.reshape((numspatiallocs, totaltimepoints))

    print("masking arrays")
    maskthresh = 0.25
    if datamaskname is not None:
        if datamaskdims[4] == 1:
            proclocs = np.where(datamask_data.reshape(numspatiallocs) > maskthresh)
        else:
            proclocs = np.where(
                np.mean(datamask_data.reshape((numspatiallocs, totaltimepoints)), axis=1)
                > maskthresh
            )
            rs_mask = datamask_data.reshape((numspatiallocs, totaltimepoints))[proclocs, :]
            rs_mask = np.where(rs_mask > maskthresh, 1.0, 0.0)[0]
    else:
        datamaskdims = [1, xsize, ysize, numslices, 1]
        themaxes = np.max(rs_datafile, axis=1)
        themins = np.min(rs_datafile, axis=1)
        thediffs = (themaxes - themins).reshape(numspatiallocs)
        proclocs = np.where(thediffs > 0.0)
    procdata = rs_datafile[proclocs, :][0]
    print(rs_datafile.shape, procdata.shape)

    # normalize the individual images
    if demean:
        print("demeaning array")
        themean = np.mean(procdata, axis=decompaxisnum)
        print("shape of mean", themean.shape)
        for i in range(procdata.shape[1 - decompaxisnum]):
            if decompaxisnum == 1:
                procdata[i, :] -= themean[i]
            else:
                procdata[:, i] -= themean[i]
    else:
        themean = np.ones(procdata.shape[1 - decompaxisnum])

    if varnorm:
        print("variance normalizing array")
        thevar = np.var(procdata, axis=decompaxisnum)
        print("shape of var", thevar.shape)
        for i in range(procdata.shape[1 - decompaxisnum]):
            if decompaxisnum == 1:
                procdata[i, :] /= thevar[i]
            else:
                procdata[:, i] /= thevar[i]
        procdata = np.nan_to_num(procdata)
    else:
        thevar = np.ones(procdata.shape[1 - decompaxisnum])

    # applying mask
    if datamaskdims[4] > 1:
        procdata *= rs_mask

    # now perform the decomposition
    if decomptype == "ica":
        print("performing ica decomposition")
        if icacomponents is None:
            print("will return all significant components")
        else:
            print("will return", icacomponents, "components")
        thefit = FastICA(n_components=icacomponents).fit(
            transposeifspatial(procdata, decompaxis=decompaxis)
        )  # Reconstruct signals
        if icacomponents is None:
            thecomponents = transposeifspatial(thefit.components_[:], decompaxis=decompaxis)
            print(thecomponents.shape[1], "components found")
        else:
            thecomponents = transposeifspatial(
                thefit.components_[0:icacomponents], decompaxis=decompaxis
            )
            print("returning first", thecomponents.shape[1], "components found")
    else:
        print("performing pca decomposition")
        if pcacomponents < 1.0:
            print(
                "will return the components accounting for",
                pcacomponents * 100.0,
                "% of the variance",
            )
        else:
            print("will return", pcacomponents, "components")
        if decomptype == "pca":
            thepca = PCA(n_components=pcacomponents)
        else:
            thepca = SparsePCA(n_components=pcacomponents)
        thefit = thepca.fit(transposeifspatial(procdata, decompaxis=decompaxis))
        thetransform = thepca.transform(transposeifspatial(procdata, decompaxis=decompaxis))
        theinvtrans = transposeifspatial(
            thepca.inverse_transform(thetransform), decompaxis=decompaxis
        )
        if pcacomponents < 1.0:
            thecomponents = transposeifspatial(thefit.components_[:], decompaxis=decompaxis)
            print("returning", thecomponents.shape[1], "components")
        else:
            thecomponents = transposeifspatial(
                thefit.components_[0:pcacomponents], decompaxis=decompaxis
            )

        # stash the eigenvalues
        exp_var = thefit.explained_variance_
        exp_var_pct = 100.0 * thefit.explained_variance_ratio_

        if decompaxis == "temporal":
            # save the components
            outputcomponents = thecomponents

            """# save the singular values
            print("writing singular values")
            tide_io.writenpvecs(np.transpose(thesingvals), outputroot + "_singvals.txt")"""

            # save the coefficients
            coefficients = thetransform
            print("coefficients shape:", coefficients.shape)
            outputcoefficients = np.zeros((numspatiallocs, coefficients.shape[1]), dtype="float")
            outputcoefficients[proclocs, :] = coefficients[:, :]
            outputcoefficients = outputcoefficients.reshape(
                (xsize, ysize, numslices, coefficients.shape[1])
            )

            # unnormalize the dimensionality reduced data
            for i in range(procdata.shape[1 - decompaxisnum]):
                theinvtrans[i, :] = thevar[i] * theinvtrans[i, :] + themean[i]

        else:
            # save the component images
            outputcomponents = np.zeros((numspatiallocs, thecomponents.shape[1]), dtype="float")
            outputcomponents[proclocs, :] = thecomponents[:, :]
            outputcomponents = outputcomponents.reshape(
                (xsize, ysize, numslices, thecomponents.shape[1])
            )

            # save the coefficients
            outputcoefficients = np.transpose(thetransform)
            # tide_io.writenpvecs(
            #    outputcoefficients * thevar[i], outputroot + "_denormcoefficients.txt"
            # )

            # unnormalize the dimensionality reduced data
            for i in range(totaltimepoints):
                theinvtrans[:, i] = thevar[i] * theinvtrans[:, i] + themean[i]

        print("writing fit data")
        theheader = datafile_hdr
        theheader["dim"][4] = theinvtrans.shape[1]
        outinvtrans = np.zeros((numspatiallocs, theinvtrans.shape[1]), dtype="float")
        outinvtrans[proclocs, :] = theinvtrans[:, :]
        outinvtrans = outinvtrans.reshape((xsize, ysize, numslices, theinvtrans.shape[1]))
    return (
        outputcomponents,
        outputcoefficients,
        outinvtrans,
        exp_var,
        exp_var_pct,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    )


def main(decompaxis: Any, args: Any) -> None:
    """
    Main function for performing decomposition analysis on neuroimaging data.

    This function handles the configuration of decomposition parameters based on
    the number of components specified (`ncomp`), and then executes the decomposition
    workflow using `niftidecomp_workflow`. It saves the results including components,
    coefficients, and explained variance to disk.

    Parameters
    ----------
    decompaxis : Any
        The axis along which decomposition is performed (e.g., 'temporal' or 'spatial').
    args : Any
        A dictionary containing various configuration parameters such as:
        - `ncomp`: Number of components to extract.
        - `datafile`: Path to the input data file.
        - `datamaskname`: Path to the data mask file.
        - `decomptype`: Type of decomposition (e.g., PCA, ICA).
        - `varnorm`: Whether to normalize variance.
        - `demean`: Whether to demean the data.
        - `sigma`: Sigma value for smoothing.
        - `outputroot`: Root name for output files.

    Returns
    -------
    None
        This function does not return a value but writes multiple output files to disk.

    Notes
    -----
    - If `ncomp` is less than 0.0, `pcacomponents` is set to 0.5 and `icacomponents` is set to None.
    - If `ncomp` is between 0.0 and 1.0, `pcacomponents` is set to `ncomp` and `icacomponents` is set to None.
    - Otherwise, both `pcacomponents` and `icacomponents` are set to the integer value of `ncomp`.

    Examples
    --------
    >>> args = {
    ...     "ncomp": 5,
    ...     "datafile": "data.nii.gz",
    ...     "datamaskname": "mask.nii.gz",
    ...     "decomptype": "pca",
    ...     "varnorm": True,
    ...     "demean": True,
    ...     "sigma": 2.0,
    ...     "outputroot": "output"
    ... }
    >>> main("temporal", args)
    """
    if args["ncomp"] < 0.0:
        args["pcacomponents"] = 0.5
        args["icacomponents"] = None
    elif args["ncomp"] < 1.0:
        args["pcacomponents"] = args["ncomp"]
        args["icacomponents"] = None
    else:
        args["pcacomponents"] = int(args["ncomp"])
        args["icacomponents"] = int(args["ncomp"])

    # args = getparameters(decompaxis)

    # save the command line
    tide_io.writevec([" ".join(sys.argv)], args["outputroot"] + "_commandline.txt")

    (
        outputcomponents,
        outputcoefficients,
        outinvtrans,
        exp_var,
        exp_var_pct,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = niftidecomp_workflow(
        decompaxis,
        [args["datafile"]],
        datamaskname=args["datamaskname"],
        decomptype=args["decomptype"],
        pcacomponents=args["pcacomponents"],
        icacomponents=args["icacomponents"],
        varnorm=args["varnorm"],
        demean=args["demean"],
        sigma=args["sigma"],
    )

    # save the eigenvalues
    print("variance explained by component:", exp_var)
    tide_io.writenpvecs(
        exp_var,
        args["outputroot"] + "_explained_variance.txt",
    )
    print("percentage variance explained by component:", exp_var_pct)
    tide_io.writenpvecs(
        exp_var_pct,
        args["outputroot"] + "_explained_variance_pct.txt",
    )

    if decompaxis == "temporal":
        # save the components
        print("writing component timecourses")
        tide_io.writenpvecs(outputcomponents, args["outputroot"] + "_components.txt")

        # save the coefficients
        print("writing out the coefficients")
        theheader = datafile_hdr.copy()
        theheader["dim"][4] = outputcoefficients.shape[3]
        tide_io.savetonifti(
            outputcoefficients,
            theheader,
            args["outputroot"] + "_coefficients",
        )
    else:
        # save the component images
        print("writing component images")
        theheader = datafile_hdr.copy()
        theheader["dim"][4] = outputcomponents.shape[3]
        tide_io.savetonifti(
            outputcomponents,
            theheader,
            args["outputroot"] + "_components",
        )

        # save the coefficients
        print("writing out the coefficients")
        tide_io.writenpvecs(outputcoefficients, args["outputroot"] + "_coefficients.txt")
        # tide_io.writenpvecs(
        #    outputcoefficients * thevar[i], args["outputroot"] + "_denormcoefficients.txt"
        # )
    print("writing fit data")
    tide_io.savetonifti(
        outinvtrans,
        datafile_hdr,
        args["outputroot"] + "_fit",
    )


def main_temporal(args: Any) -> None:
    """
    Execute main function for temporal processing.

    This function serves as a wrapper that calls the main execution function
    with "temporal" as the processor type and converts the arguments to a dictionary.

    Parameters
    ----------
    args : Any
        Command line arguments or configuration arguments containing
        parameters needed for temporal processing. Typically contains
        various temporal processing options and settings.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function delegates to a main execution function with "temporal" processor
    type, making it a specialized entry point for temporal data processing workflows.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--input', type=str, help='Input file path')
    >>> args = parser.parse_args(['--input', 'data.csv'])
    >>> main_temporal(args)
    """
    main("temporal", vars(args))


def main_spatial(args: Any) -> None:
    """
    Main function for spatial processing pipeline.

    This function serves as the entry point for spatial data processing operations,
    delegating to the main processing function with "spatial" as the operation type.

    Parameters
    ----------
    args : Any
        Command line arguments or configuration parameters containing spatial
        processing options. Expected to be an argparse.Namespace or similar
        structure with relevant attributes for spatial operations.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function internally calls `main("spatial", vars(args))` which routes
    the spatial processing workflow to the main execution engine.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(input_file="data.csv", output_dir="output/")
    >>> main_spatial(args)
    """
    main("spatial", vars(args))
