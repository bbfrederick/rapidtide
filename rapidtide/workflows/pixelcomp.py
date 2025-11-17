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

import matplotlib as mpl
import numpy as np
from numpy.polynomial import Polynomial

import rapidtide.io as tide_io

mpl.use("Agg")
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from numpy.typing import NDArray


def _get_parser() -> Any:
    """
    Argument parser for pixelcomp.

    This function creates and configures an argument parser for the pixelcomp tool,
    which is used to compare two NIfTI files voxel by voxel and generate either
    a contour plot or a scatter plot of the differences.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments
        for the pixelcomp tool.

    Notes
    -----
    The parser supports the following positional arguments:

    - ``inputfilename1`` : str
        The name of the first input image NIfTI file.
    - ``maskfilename1`` : str
        The name of the first input mask NIfTI file.
    - ``inputfilename2`` : str
        The name of the second input image NIfTI file.
    - ``maskfilename2`` : str
        The name of the second input mask NIfTI file.
    - ``outputroot`` : str
        The root name of the output files.

    And the following optional arguments:

    - ``--scatter`` : bool, optional
        Do a scatter plot instead of a contour plot. Default is False.
    - ``--fitonly`` : bool, optional
        Perform fit only - do not generate graph. Default is False.
    - ``--nodisplay`` : bool, optional
        Save graphs to file only - do not display. Default is True.
    - ``--fitorder`` : int, optional
        Order of line fit - default is 1 (linear). Default is 1.
    - ``--usex`` : bool, optional
        Use x instead of (y + x)/2 in Bland-Altman plot. Default is False.
    - ``--histbins`` : int, optional
        Number of bins per dimension for the contour plot - default is 51.
        Default is 51.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        prog="pixelcomp",
        description=("Compare two nifti files, voxel by voxel, in a contour plot"),
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
        "--scatter",
        action="store_true",
        help=("Do a scatter plot instead of a contour plot."),
        default=False,
    )
    parser.add_argument(
        "--fitonly",
        action="store_true",
        help=("Perform fit only - do not generate graph."),
        default=False,
    )
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Save graphs to file only - do not display."),
        default=True,
    )
    parser.add_argument(
        "--fitorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=("Order of line fit - default is 1 (linear)."),
        default=1,
    )
    parser.add_argument(
        "--usex",
        dest="usex",
        action="store_true",
        help="Use x instead of (y + x)/2 in Bland-Altman plot.",
        default=False,
    )
    parser.add_argument(
        "--histbins",
        action="store",
        type=int,
        metavar="NUM",
        help=("Number of bins per dimension for the contour plot -Default is 51."),
        default=51,
    )
    return parser


def bland_altman_plot(
    data1: Any, data2: Any, usex: bool = False, *args: Any, **kwargs: Any
) -> None:
    """
    Create a Bland-Altman plot for comparing two sets of measurements.

    This function generates a scatter plot showing the difference between two
    measurements against their mean. The plot includes horizontal lines indicating
    the mean difference and ±2 standard deviations, which are commonly used to
    assess agreement between two measurement methods.

    Parameters
    ----------
    data1 : array-like
        First set of measurements (X values in the plot).
    data2 : array-like
        Second set of measurements (Y values in the plot).
    usex : bool, optional
        If True, use data1 as the x-values for the plot. If False (default),
        use the mean of data1 and data2 as x-values.
    *args : tuple
        Additional arguments to pass to matplotlib's scatter function.
    **kwargs : dict
        Additional keyword arguments to pass to matplotlib's scatter function.

    Returns
    -------
    None
        This function displays the plot but does not return any value.

    Notes
    -----
    The Bland-Altman plot is used to assess the agreement between two different
    measurement methods. The mean difference (MD) is plotted on the y-axis, and
    the mean of the two measurements is plotted on the x-axis. The horizontal
    lines represent:
    - Mean difference (MD)
    - Mean difference ± 2 standard deviations (±2SD)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data1 = np.array([1, 2, 3, 4, 5])
    >>> data2 = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    >>> bland_altman_plot(data1, data2)
    >>> plt.show()

    >>> # Using custom scatter plot properties
    >>> bland_altman_plot(data1, data2, c='red', alpha=0.7)
    >>> plt.show()
    """
    # data1 is X, data2 is Y
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    if usex:
        mean = np.mean(data1)
    else:
        mean = np.mean([data1, data2], axis=0)
    diff = data2 - data1  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color="gray", linestyle="--")
    plt.axhline(md + 2 * sd, color="gray", linestyle="--")
    plt.axhline(md - 2 * sd, color="gray", linestyle="--")


def pairdata(input1_data: Any, input2_data: Any, totalmask: Any) -> NDArray:
    """
    Pair corresponding elements from two 3D arrays based on a mask.

    This function extracts elements from two 3D input arrays where the mask
    has non-zero values, creating pairs of corresponding elements.

    Parameters
    ----------
    input1_data : array-like
        First 3D array from which elements will be extracted.
    input2_data : array-like
        Second 3D array from which elements will be extracted.
    totalmask : array-like
        3D mask array where non-zero values indicate positions to pair.

    Returns
    -------
    NDArray
        2D array where each row contains a pair of corresponding elements
        from input1_data and input2_data at positions where totalmask > 0.

    Notes
    -----
    - The function assumes all input arrays have the same shape
    - Only positions where totalmask > 0 are considered
    - The returned array has shape (n_pairs, 2) where n_pairs is the number
      of non-zero mask positions

    Examples
    --------
    >>> import numpy as np
    >>> input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> input2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    >>> mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    >>> pairdata(input1, input2, mask)
    array([[ 1,  9],
           [ 4, 12],
           [ 5, 13],
           [ 6, 14]])
    """
    nonzeropoints = np.where(totalmask > 0)
    pairlist = []
    for i in range(0, len(nonzeropoints[0])):
        pairlist.append(
            [
                input1_data[nonzeropoints[0][i], nonzeropoints[1][i], nonzeropoints[2][i]],
                input2_data[nonzeropoints[0][i], nonzeropoints[1][i], nonzeropoints[2][i]],
            ]
        )

    return np.asarray(pairlist)


def pixelcomp(args: Any) -> None:
    """
    Compare pixel values from two input images using masks and generate statistical plots.

    This function reads two NIfTI images and their corresponding masks, performs a pixel-wise
    comparison, and generates either a scatter plot or a 2D histogram of the paired data.
    It also fits a polynomial to the data and optionally produces a Bland-Altman plot.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - inputfilename1 : str
            Path to the first input NIfTI image file.
        - maskfilename1 : str
            Path to the first mask NIfTI file.
        - inputfilename2 : str
            Path to the second input NIfTI image file.
        - maskfilename2 : str
            Path to the second mask NIfTI file.
        - outputroot : str
            Root name for output files.
        - histbins : int
            Number of bins for the 2D histogram.
        - fitorder : int
            Order of the polynomial to fit.
        - display : bool
            If True, display plots; otherwise, save them to files.
        - scatter : bool
            If True, generate a scatter plot; otherwise, generate a contour plot.
        - fitonly : bool
            If True, only perform the polynomial fit and save coefficients.

    Returns
    -------
    None
        This function does not return any value. It saves plots and data to files.

    Notes
    -----
    - The function requires both input images and masks to have matching spatial dimensions.
    - The output includes:
        * A scatter or contour plot saved as PNG.
        * A file with polynomial coefficients.
        * Optionally, a Bland-Altman plot saved as PNG.
    - If a RankWarning occurs during polynomial fitting, the coefficients are set to [0.0, 0.0].

    Examples
    --------
    >>> class Args:
    ...     inputfilename1 = "image1.nii.gz"
    ...     maskfilename1 = "mask1.nii.gz"
    ...     inputfilename2 = "image2.nii.gz"
    ...     maskfilename2 = "mask2.nii.gz"
    ...     outputroot = "output"
    ...     histbins = 50
    ...     fitorder = 1
    ...     display = False
    ...     scatter = False
    ...     fitonly = False
    >>> args = Args()
    >>> pixelcomp(args)
    """
    if args.display:
        mpl.use("TkAgg")

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

    totalmask = mask1_data * mask2_data
    thearray = pairdata(input1_data, input2_data, totalmask)

    plt.figure()
    if args.scatter:
        plt.plot(thearray[:, 0], thearray[:, 1], "k.")
        theplotname = args.outputroot + "_scatterplot.png"
    else:
        # construct a 2d histogram
        H, xedges, yedges = np.histogram2d(
            thearray[:, 0], thearray[:, 1], bins=args.histbins, normed=True
        )
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        plt.contour(H, extent=extent)
        theplotname = args.outputroot + "_contourplot.png"
    if args.display:
        plt.show()
    else:
        plt.savefig(theplotname, bbox_inches="tight")

    # now fit the line
    try:
        thecoffs = (
            Polynomial.fit(thearray[:, 0], thearray[:, 1], args.fitorder).convert().coef[::-1]
        )
    except np.exceptions.RankWarning:
        thecoffs = np.asarray([0.0, 0.0])
    print("thecoffs=", thecoffs)
    with open(f"{args.outputroot}_order_{args.fitorder}_fit", "w") as file:
        file.writelines(str(thecoffs))

    if not args.fitonly:
        with open(args.outputroot, "w") as file:
            for pair in range(thearray.shape[0]):
                file.writelines(str(thearray[pair, 0]) + "\t" + str(thearray[pair, 1]) + "\n")
        plt.figure()
        bland_altman_plot(thearray[:, 0], thearray[:, 1])
        plt.title("Bland-Altman Plot")
        if args.display:
            plt.show()
        else:
            plt.savefig(args.outputroot + "_blandaltman.png", bbox_inches="tight")
