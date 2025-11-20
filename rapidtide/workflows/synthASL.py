#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2020-2025 Blaise Frederick
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
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
from rapidtide.RapidtideDataset import RapidtideDataset


def _get_parser() -> Any:
    """
    Get the argument parser for the synthASL command-line tool.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    with the necessary arguments for running the synthASL tool, which uses rapidtide
    output to predict ASL (Arterial Spin Labeling) images.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with required and optional arguments for synthASL.

    Notes
    -----
    The parser includes both required positional arguments and several optional
    arguments that control the ASL synthesis process. Default values are set according
    to typical parameters used in ASL analysis.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['dataset_name', 'output.nii.gz'])
    >>> print(args.dataset)
    'dataset_name'
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="synthASL",
        description="Use rapidtide output to predict ASL image.",
        allow_abbrev=False,
    )
    parser.add_argument("dataset", type=str, help="The name of the rapidtide dataset.")
    parser.add_argument("outputfilename", type=str, help="The name of the output nifti file.")

    # optional parameters
    parser.add_argument(
        "--tagoffset",
        dest="tagoffset",
        metavar="SECS",
        type=float,
        help="The assumed time of tagging, relative to the peak of the lag histogram (default is 2.945).",
        default=2.945,
    )
    parser.add_argument(
        "--pld",
        dest="pld",
        metavar="SECS",
        type=float,
        help="The postlabel delay (default is 1.8).",
        default=1.8,
    )
    parser.add_argument(
        "--labelduration",
        dest="labelduration",
        metavar="SECS",
        type=float,
        help="The length of the labelling period (default is 1.8).",
        default=1.8,
    )
    parser.add_argument(
        "--bloodT1",
        dest="bloodT1",
        metavar="SECS",
        type=float,
        help="The T1 of blood at this field strength (default is 1.841, for 3T).",
        default=1.841,
    )
    return parser


def calcASL(
    lags: Any,
    strengths: Any,
    widths: Any,
    mask: Any,
    tagoffset: float = 2.945,
    pld: float = 1.8,
    TI: float = 1.8,
    bloodT1: float = 1.841,
) -> None:
    """
    Calculate ASL (Arterial Spin Labeling) signal based on lags, strengths, and timing parameters.

    This function computes the ASL signal by applying temporal dynamics to the input lags and
    strengths, considering blood T1 relaxation effects and tagging timing parameters.

    Parameters
    ----------
    lags : array-like
        Time lags for the ASL signal calculation
    strengths : array-like
        Signal strengths corresponding to the lags
    widths : array-like
        Widths parameter (not used in current implementation)
    mask : array-like
        Binary mask for region of interest
    tagoffset : float, optional
        Tagging offset in seconds, default is 2.945
    pld : float, optional
        Preparation delay in seconds, default is 1.8
    TI : float, optional
        Inversion time in seconds, default is 1.8
    bloodT1 : float, optional
        Blood T1 relaxation time in seconds, default is 1.841

    Returns
    -------
    tuple
        A tuple containing:
        - theaslimage : array-like
          Calculated ASL image signal
        - tagdecayfac : array-like
          Tagging decay factor
        - oxyfac : array-like
          Oxygenation factor (constant value of 1.0)
        - cbvfac : array-like
          CBV (Cerebral Blood Volume) factor
        - calcmask : array-like
          Calculated mask with time constraints
        - offsets : array-like
          Time offsets after adding tagoffset

    Notes
    -----
    The function applies exponential decay based on blood T1 relaxation and only considers
    positive delays after the preparation delay (pld). The calculation uses a linear
    interpolation over 50 time points between pld and pld + TI.

    Examples
    --------
    >>> import numpy as np
    >>> lags = np.array([0.5, 1.0, 1.5])
    >>> strengths = np.array([1.0, 1.5, 2.0])
    >>> mask = np.array([1, 1, 1])
    >>> result = calcASL(lags, strengths, None, mask)
    """
    theaslimage = np.zeros_like(lags)

    # convert rapidtide delays to time from tagging, and only keep positive delays after pld
    offsets = lags + tagoffset
    calcmask = mask * np.where(offsets < pld, 0.0, 1.0)

    for imtime in np.linspace(pld, pld + TI, num=50, endpoint=True):
        tagdecayfac = np.exp(-(offsets + imtime) / bloodT1) * calcmask
        oxyfac = np.ones_like(strengths)
        cbvfac = np.fabs(strengths) * oxyfac
        theaslimage += tagdecayfac * cbvfac
    return theaslimage, tagdecayfac, oxyfac, cbvfac, calcmask, offsets


def synthASL(args: Any) -> None:
    """
    Generate synthetic ASL (Arterial Spin Labeling) images and associated parameters.

    This function reads ASL dataset parameters from a specified input dataset,
    computes synthetic ASL signal using the `calcASL` function, and saves the
    resulting images and intermediate outputs as NIfTI files.

    Parameters
    ----------
    args : Any
        Command-line arguments object containing the following attributes:
        - dataset : str
            Path to the input dataset.
        - outputfilename : str
            Base name for output NIfTI files.
        - bloodT1 : float
            Blood T1 relaxation time in seconds.
        - tagoffset : float
            Tag offset in seconds.
        - pld : float
            Post-labeling delay in seconds.
        - labelduration : float
            Labeling pulse duration in seconds.

    Returns
    -------
    None
        This function does not return a value but writes multiple NIfTI files
        to disk, including:
        - `<outputfilename>_ASL.nii.gz`
        - `<outputfilename>_tagdecayfac.nii.gz`
        - `<outputfilename>_oxyfac.nii.gz`
        - `<outputfilename>_cbvfac.nii.gz`
        - `<outputfilename>_calcmask.nii.gz`
        - `<outputfilename>_offsets.nii.gz`
        - `<outputfilename>_lagmask.nii.gz`
        - `<outputfilename>_lagtimes.nii.gz`
        - `<outputfilename>_lagstrengths.nii.gz`

    Notes
    -----
    The function relies on the `RapidtideDataset` class to load overlay data
    and uses `calcASL` to compute the synthetic ASL signal. All outputs are
    saved using `tide_io.savetonifti`.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     dataset="path/to/dataset",
    ...     outputfilename="output",
    ...     bloodT1=1.6,
    ...     tagoffset=0.0,
    ...     pld=1.5,
    ...     labelduration=1.0
    ... )
    >>> synthASL(args)
    """
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    thedataset = RapidtideDataset("main", args.dataset, init_LUT=False)

    themask = thedataset.overlays["lagmask"].data
    thelags = thedataset.overlays["lagtimes"].data
    thewidths = thedataset.overlays["lagsigma"].data
    thestrengths = thedataset.overlays["lagstrengths"].data

    print(
        "bloodT1, tagoffset, pld, and inversion time:",
        args.bloodT1,
        args.tagoffset,
        args.pld,
        args.labelduration,
    )
    theaslimage, tagdecayfac, oxyfac, cbvfac, calcmask, offsets = calcASL(
        thelags,
        thestrengths,
        thewidths,
        themask,
        tagoffset=args.tagoffset,
        pld=args.pld,
        TI=args.labelduration,
        bloodT1=args.bloodT1,
    )

    tide_io.savetonifti(
        theaslimage,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_ASL",
    )
    tide_io.savetonifti(
        tagdecayfac,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_tagdecayfac",
    )
    tide_io.savetonifti(
        oxyfac,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_oxyfac",
    )
    tide_io.savetonifti(
        cbvfac,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_cbvfac",
    )
    tide_io.savetonifti(
        calcmask,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_calcmask",
    )
    tide_io.savetonifti(
        offsets,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_offsets",
    )
    tide_io.savetonifti(
        themask,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_lagmask",
    )
    tide_io.savetonifti(
        thelags,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_lagtimes",
    )
    tide_io.savetonifti(
        thestrengths,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_lagstrengths",
    )
