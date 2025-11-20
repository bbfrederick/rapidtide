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

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math


def _get_parser() -> Any:
    """
    Argument parser for spatialmi.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    to parse command-line arguments for the `spatialmi` tool, which calculates localized
    spatial mutual information between two NIfTI images.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the spatial mutual information calculation tool.

    Notes
    -----
    The parser requires four mandatory positional arguments: two input NIfTI image filenames,
    two corresponding mask filenames, and an output root name. Optional arguments allow
    customization of the calculation, including neighborhood shape, filtering, normalization,
    and debugging output.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    >>> print(args.inputfilename1)
    'input1.nii.gz'
    """
    parser = argparse.ArgumentParser(
        prog="spatialmi",
        description=("Calculate the localized spatial mutual information between two images"),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputfilename1", type=str, help="The name of the first input image nifti file."
    )
    parser.add_argument(
        "maskfilename1", type=str, help="The name of the first input mask nifti file."
    )
    parser.add_argument(
        "inputfilename2", type=str, help="The name of the second input image nifti file."
    )
    parser.add_argument(
        "maskfilename2", type=str, help="The name of the second input mask nifti file."
    )
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")

    # add optional arguments
    parser.add_argument(
        "--noprebin",
        dest="prebin",
        action="store_false",
        help=("Dynamically calculate histogram bins for each voxel (slower)."),
        default=True,
    )
    parser.add_argument(
        "--nonorm",
        dest="norm",
        action="store_false",
        help=("Do not normalize neighborhood by the variance."),
        default=True,
    )
    parser.add_argument(
        "--radius",
        action="store",
        type=float,
        metavar="RADIUS",
        help=(
            "Radius of the comparison, in voxels.  If not spherical, comparison neighborhood is "
            "cubic with (2 * RADIUS + 1)^3 voxels.  Must be 1.0 or greater.  Default is 2.0"
        ),
        default=2.0,
    )
    parser.add_argument(
        "--sigma",
        action="store",
        type=float,
        metavar="SIGMA",
        help=(
            "Width, in voxels, of a gaussian smoothing filter to apply to each input dataset.  "
            "Default is no filteriing."
        ),
        default=None,
    )
    parser.add_argument(
        "--kernelwidth",
        action="store",
        type=float,
        metavar="WIDTH",
        help=("Kernel width, in voxels, of gaussian neighborhood limit. Default is no kernel."),
        default=None,
    )
    parser.add_argument(
        "--spherical",
        dest="spherical",
        action="store_true",
        help="Use a spherical (rather than cubical) neighborhood (much slower).",
        default=False,
    )
    parser.add_argument(
        "--index1",
        action="store",
        type=int,
        metavar="INDEX1",
        help=(
            "If input file 1 is 4 dimensional, select timepoint INDEX1 for spatial mutual information calculation."
            "If not specified, the first image will be used."
        ),
        default=0,
    )
    parser.add_argument(
        "--index2",
        action="store",
        type=int,
        metavar="INDEX2",
        help=(
            "If input file 2 is 4 dimensional, select timepoint INDEX2 for spatial mutual information calculation."
            "If not specified, the first image will be used."
        ),
        default=0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print additional internal information.",
        default=False,
    )
    return parser


def getneighborhood(
    indata: NDArray[np.floating[Any]],
    xloc: Any,
    yloc: Any,
    zloc: Any,
    xsize: Any,
    ysize: Any,
    zsize: Any,
    radius: Any,
    spherical: bool = False,
    kernelwidth: float = 1.5,
    slop: float = 0.01,
    debug: bool = False,
) -> NDArray[np.floating[Any]]:
    """
    Extract a neighborhood from a 3D dataset, either as a weighted kernel or a spherical region.

    This function retrieves a local neighborhood around a specified 3D location in a dataset.
    The neighborhood can be extracted either as a cubic region with a Gaussian-weighted kernel
    (when `spherical=False`) or as a spherical region (when `spherical=True`).

    Parameters
    ----------
    indata : NDArray[np.floating[Any]]
        Input 3D dataset from which the neighborhood is extracted.
    xloc, yloc, zloc : float or int
        The center coordinates of the neighborhood in the dataset.
    xsize, ysize, zsize : int
        Dimensions of the input dataset along each axis.
    radius : float or int
        The radius of the neighborhood. For `spherical=False`, this defines the cubic
        kernel size. For `spherical=True`, it defines the spherical radius.
    spherical : bool, optional
        If True, extracts a spherical neighborhood. If False, extracts a cubic neighborhood
        with a Gaussian kernel. Default is False.
    kernelwidth : float, optional
        Width parameter for the Gaussian kernel used when `spherical=False`. Default is 1.5.
    slop : float, optional
        Tolerance for spherical radius checking. Default is 0.01.
    debug : bool, optional
        If True, prints debug information about the index list initialization. Default is False.

    Returns
    -------
    NDArray[np.floating[Any]]
        A flattened array of the neighborhood values. When `spherical=False`, the values
        are weighted by a Gaussian kernel. When `spherical=True`, the values are unweighted.

    Notes
    -----
    - The function uses global variables `kernel`, `usedwidth`, `indexlist`, and `usedradius`
      to cache computations for performance. These are initialized on first call.
    - For `spherical=False`, the function returns a flattened array of the kernel-weighted
      neighborhood.
    - For `spherical=True`, the function returns a flattened array of values within the
      spherical neighborhood.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 10, 10)
    >>> neighborhood = getneighborhood(data, 5, 5, 5, 10, 10, 10, 2, spherical=False)
    >>> print(neighborhood.shape)
    (125,)
    >>> neighborhood = getneighborhood(data, 5, 5, 5, 10, 10, 10, 2, spherical=True)
    >>> print(neighborhood.shape)
    (33,)
    """
    if not spherical:
        global usedwidth, kernel
        try:
            dummy = kernel
        except NameError:
            usedwidth = kernelwidth
            kernel = None
        if kernelwidth != usedwidth or kernel is None:
            kernel = np.zeros(
                (2 * int(radius) + 1, 2 * int(radius) + 1, 2 * int(radius) + 1), dtype=float
            )
            if kernelwidth is not None:
                for xval in range(-int(np.floor(radius)), int(np.ceil(radius + 1))):
                    for yval in range(-int(np.floor(radius)), int(np.ceil(radius + 1))):
                        for zval in range(-int(np.floor(radius)), int(np.ceil(radius + 1))):
                            kernel[xval, yval, zval] = tide_fit.gauss_eval(
                                np.sqrt(xval * xval + yval * yval + zval * zval),
                                [1.0, 0.0, kernelwidth],
                            )
            else:
                kernel[:, :, :] = 1.0
        if (
            (radius <= xloc < xsize - radius)
            and (radius <= yloc < ysize - radius)
            and (radius <= zloc < zsize - radius)
        ):
            fullrange = True
        else:
            fullrange = False

        if fullrange:
            return (
                kernel
                * indata[
                    int(xloc - radius) : int(xloc + radius + 1),
                    int(yloc - radius) : int(yloc + radius + 1),
                    int(zloc - radius) : int(zloc + radius + 1),
                ]
            ).flatten()
        else:
            return indata[
                np.max([0, int(np.ceil(xloc - radius))]) : np.min(
                    [xsize, int(np.floor(xloc + radius + 1))]
                ),
                np.max([0, int(np.ceil(yloc - radius))]) : np.min(
                    [ysize, int(np.floor(yloc + radius + 1))]
                ),
                np.max([0, int(np.ceil(zloc - radius))]) : np.min(
                    [zsize, int(np.floor(zloc + radius + 1))]
                ),
            ].flatten()
    else:
        global indexlist, usedradius
        try:
            usedradius
        except NameError:
            usedradius = radius
            indexlist = None
        if radius != usedradius or indexlist is None:
            indexlist = []
            for xval in range(-int(np.floor(radius)), int(np.ceil(radius + 1))):
                for yval in range(-int(np.floor(radius)), int(np.ceil(radius + 1))):
                    for zval in range(-int(np.floor(radius)), int(np.ceil(radius + 1))):
                        if np.sqrt(xval * xval + yval * yval + zval * zval) <= radius + slop:
                            indexlist.append([xval, yval, zval])
            if debug:
                print(f"index list initialized for radius {radius}")
                print(indexlist)
        outdata = []
        for position in indexlist:
            if (
                (0 <= xloc + position[0] < xsize)
                and (0 <= yloc + position[1] < ysize)
                and (0 <= zloc + position[2] < zsize)
            ):
                outdata.append(indata[xloc + position[0], yloc + position[1], zloc + position[2]])
        return np.array(outdata)


def getMI(
    x: Any,
    y: Any,
    norm: bool = True,
    bins: int = -1,
    init: bool = False,
    prebin: bool = True,
    sigma: float = 0.25,
    debug: bool = False,
) -> None:
    """
    Compute the mutual information between two variables using binned estimation.

    This function calculates the mutual information between two input arrays `x` and `y`,
    using a binned approach. It supports normalization, automatic bin selection, and
    optional pre-binning for performance optimization.

    Parameters
    ----------
    x : array-like
        First input variable.
    y : array-like
        Second input variable.
    norm : bool, optional
        If True, normalize the input variables using standard normalization
        (zero mean, unit variance). Default is True.
    bins : int, optional
        Number of bins to use for the 2D histogram. If less than 1, the number
        of bins is automatically determined as `max(int(sqrt(len(x) / 5)), 3)`.
        Default is -1.
    init : bool, optional
        If True, reinitialize the global binning structure. Default is False.
    prebin : bool, optional
        If True, use precomputed bin edges. If False, use the number of bins
        directly. Default is True.
    sigma : float, optional
        Standard deviation for Gaussian smoothing in the mutual information
        calculation. Default is 0.25.
    debug : bool, optional
        If True, print debugging information during execution. Default is False.

    Returns
    -------
    None
        The function returns the result of `tide_corr.mutual_info_2d`, which is
        not explicitly returned here but is the core output of the function.

    Notes
    -----
    The function uses a global variable `thebins` to store binning information.
    If `init` is True or `thebins` is None, bin edges are computed and stored
    in `thebins`. The function relies on `tide_math.stdnormalize` for normalization
    and `tide_corr.mutual_info_2d` for the actual mutual information computation.

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 6, 8, 10]
    >>> getMI(x, y, norm=True, bins=10, debug=False)
    """
    global thebins

    if norm:
        normx = tide_math.stdnormalize(x)
        normy = tide_math.stdnormalize(y)
    else:
        normx = x
        normy = y

    # see if we are using the default number of bins
    if bins < 1:
        bins = np.max([int(np.sqrt(len(x) / 5)), 3])
        if debug:
            print(f"cross_mutual_info: bins set to {bins}")

    if init or thebins is None:
        # find the bin locations
        if prebin:
            bins0 = np.linspace(-2.0, 2.0, bins, True)
            if debug:
                print(bins0, bins0)
            bins2d = (1.0 * bins0, 1.0 * bins0)
        else:
            bins2d = (bins + 0, bins + 0)
        thebins = bins2d

    if prebin:
        fast = True
    else:
        fast = False

    if debug:
        print(f"fast: {fast}")
        print(f"thebins: {thebins}")
        print(f"bins: {bins}")
        print(f"norm: {norm}")
        print(
            f"normx min, max, mean, std: {np.min(normx)}, {np.max(normx)}, {np.mean(normx)}, {np.std(normx)}"
        )
        print(
            f"normy min, max, mean, std: {np.min(normy)}, {np.max(normy)}, {np.mean(normy)}, {np.std(normy)}"
        )

    if fast:
        return tide_corr.mutual_info_2d_fast(
            normx,
            normy,
            thebins,
            normalized=norm,
            sigma=sigma,
            debug=debug,
        )
    else:
        return tide_corr.mutual_info_2d(
            normx,
            normy,
            thebins,
            normalized=norm,
            sigma=sigma,
            debug=debug,
        )


def spatialmi(args: Any) -> None:
    """
    Compute spatial mutual information (MI) between two 3D images over a specified neighborhood.

    This function reads two input NIfTI images and their corresponding masks, computes the
    mutual information between the images within a local neighborhood for each voxel,
    and saves the result as a NIfTI file. Optional spatial filtering can be applied
    before computing the MI.

    Parameters
    ----------
    args : Any
        Parsed command-line arguments containing:
        - inputfilename1 : str
            Path to the first input NIfTI image.
        - inputfilename2 : str
            Path to the second input NIfTI image.
        - maskfilename1 : str
            Path to the mask for the first image.
        - maskfilename2 : str
            Path to the mask for the second image.
        - index1 : int, optional
            Index of the time point to use from the first image (default is 0).
        - index2 : int, optional
            Index of the time point to use from the second image (default is 0).
        - radius : float
            Neighborhood radius for computing mutual information.
        - sigma : float, optional
            Standard deviation for spatial smoothing (default is None).
        - spherical : bool
            Whether to use a spherical neighborhood (default is False).
        - kernelwidth : float
            Width of the kernel for neighborhood computation (default is None).
        - norm : bool
            Whether to normalize the data before computing MI (default is False).
        - prebin : bool
            Whether to pre-bin the data (default is False).
        - debug : bool
            Enable debug output (default is False).
        - outputroot : str
            Root name for the output NIfTI file.

    Returns
    -------
    None
        The function writes the computed spatial mutual information to a NIfTI file
        named `<outputroot>_result.nii.gz`.

    Notes
    -----
    - The function requires that both input images and their masks have matching spatial dimensions.
    - If `sigma` is specified, spatial filtering is applied using a Gaussian kernel.
    - The neighborhood is defined by the `radius` and `spherical` parameters.
    - The output file contains mutual information values for each voxel in the masked region.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfilename1='image1.nii.gz',
    ...     inputfilename2='image2.nii.gz',
    ...     maskfilename1='mask1.nii.gz',
    ...     maskfilename2='mask2.nii.gz',
    ...     index1=0,
    ...     index2=0,
    ...     radius=5.0,
    ...     sigma=None,
    ...     spherical=True,
    ...     kernelwidth=None,
    ...     norm=False,
    ...     prebin=False,
    ...     debug=False,
    ...     outputroot='output'
    ... )
    >>> spatialmi(args)
    """
    global thebins
    thebins = None

    # read the arguments
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    if args.debug:
        print(f"Arguments: {args}")

    if args.radius < 1.0:
        print("radius must be >= 1.0")
        sys.exit()

    input1_img, input1_data, input1_hdr, thedims1, thesizes1 = tide_io.readfromnifti(
        args.inputfilename1
    )
    (
        mask1_img,
        mask1_data,
        mask1_hdr,
        themaskdims1,
        themasksizes1,
    ) = tide_io.readfromnifti(args.maskfilename1)

    if not tide_io.checkspacedimmatch(thedims1, themaskdims1):
        print("input image 1 dimensions do not match mask")
        exit()

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims1)
    if timepoints > 1:
        image1 = input1_data[:, :, :, args.index1]
    else:
        image1 = input1_data

    input2_img, input2_data, input2_hdr, thedims2, thesizes2 = tide_io.readfromnifti(
        args.inputfilename2
    )
    (
        mask2_img,
        mask2_data,
        mask2_hdr,
        themaskdims2,
        themasksizes2,
    ) = tide_io.readfromnifti(args.maskfilename2)

    if not tide_io.checkspacedimmatch(thedims2, themaskdims2):
        print("input image 2 dimensions do not match mask")
        exit()

    if not tide_io.checkspacedimmatch(thedims1, thedims2):
        print("input images 1 and 2 dimensions do not match")
        exit()

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims2)
    if timepoints > 1:
        image2 = input2_data[:, :, :, args.index2]
    else:
        image2 = input2_data

    totalmask = mask1_data * mask2_data
    print(f"totalmask.shape = {totalmask.shape}")
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims1)
    outputdata = np.zeros_like(image1)

    # spatial filter if desired
    if args.sigma is not None:
        print(f"filtering with sigma = {args.sigma}")
        image1[:, :, :] = tide_filt.ssmooth(
            1.0,
            1.0,
            1.0,
            args.sigma,
            image1[:, :, :],
        )
        image2[:, :, :] = tide_filt.ssmooth(
            1.0,
            1.0,
            1.0,
            args.sigma,
            image2[:, :, :],
        )

    # get the neighborhood size
    testneighborhood = getneighborhood(
        image1,
        int(xsize // 2),
        int(ysize // 2),
        int(numslices // 2),
        xsize,
        ysize,
        numslices,
        args.radius,
        spherical=args.spherical,
        kernelwidth=args.kernelwidth,
        debug=args.debug,
    )
    print("Neighborhood:")
    print(f"\tspherical: {args.spherical}")
    print(f"\tradius:    {args.radius}")
    print(f"\tsize:      {len(testneighborhood)}")

    # loop over all voxels
    for zloc in range(numslices):
        print("processing slice ", zloc)
        for yloc in range(ysize):
            for xloc in range(xsize):
                if totalmask[xloc, yloc, zloc] > 0.5:
                    neighborhood1 = getneighborhood(
                        image1,
                        xloc,
                        yloc,
                        zloc,
                        xsize,
                        ysize,
                        numslices,
                        args.radius,
                        spherical=args.spherical,
                        kernelwidth=args.kernelwidth,
                    )
                    neighborhood2 = getneighborhood(
                        image2,
                        xloc,
                        yloc,
                        zloc,
                        xsize,
                        ysize,
                        numslices,
                        args.radius,
                        spherical=args.spherical,
                        kernelwidth=args.kernelwidth,
                    )
                    outputdata[xloc, yloc, zloc] = getMI(
                        neighborhood1,
                        neighborhood2,
                        norm=args.norm,
                        prebin=args.prebin,
                        debug=args.debug,
                    )
    theoutheader = input1_hdr
    theoutheader["dim"][4] = 1
    tide_io.savetonifti(np.nan_to_num(outputdata), theoutheader, f"{args.outputroot}_result")
