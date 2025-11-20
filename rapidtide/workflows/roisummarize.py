#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2021-2025 Blaise Frederick
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.workflows.parser_funcs as pf


def _get_parser() -> Any:
    """
    Argument parser for roisummarize.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    for parsing command-line arguments used by the `roisummarize` tool. It defines
    required inputs, optional arguments for sampling frequency, filtering, normalization,
    and debugging options.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the roisummarize tool.

    Notes
    -----
    The parser supports two mutually exclusive ways to specify sampling frequency:
    either via `--samplerate` or `--sampletstep`. These are equivalent and both
    set the same internal `samplerate` parameter.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['--inputfilename', 'input.txt',
    ...                           '--templatefile', 'template.nii',
    ...                           '--outputfile', 'output.txt'])
    """
    parser = argparse.ArgumentParser(
        prog="filttc",
        description=("Extract summary timecourses from the regions in an atlas"),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "inputfilename")
    pf.addreqinputtextfile(parser, "templatefile")
    pf.addreqoutputtextfile(parser, "outputfile")

    # add optional arguments
    freq_group = parser.add_mutually_exclusive_group()
    freq_group.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="FREQ",
        help=(
            "Timecourses in file have sample "
            "frequency FREQ (default is 1.0Hz) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )
    freq_group.add_argument(
        "--sampletstep",
        dest="samplerate",
        action="store",
        type=lambda x: pf.invert_float(parser, x),
        metavar="TSTEP",
        help=(
            "Timecourses in file have sample "
            "timestep TSTEP (default is 1.0s) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )

    parser.add_argument(
        "--numskip",
        dest="numskip",
        action="store",
        type=int,
        metavar="NPTS",
        help=("Skip NPTS initial points to get past T1 relaxation. "),
        default=0,
    )

    # Filter arguments
    pf.addfilteropts(parser, defaultmethod="None", filtertarget="timecourses")

    # Normalization arguments
    pf.addnormalizationopts(parser, normtarget="timecourses", defaultmethod="None")

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    # Miscellaneous options

    return parser


def summarize4Dbylabel(
    inputvoxels: NDArray, templatevoxels: NDArray, normmethod: str = "z", debug: bool = False
) -> NDArray:
    """
    Summarize 4D voxel data by region labels from a template.

    This function extracts time series data for each region defined in a template
    and computes normalized mean time courses for each region across time points.

    Parameters
    ----------
    inputvoxels : NDArray
        4D array containing voxel data with shape (n_voxels, n_timepoints, n_other_dims)
    templatevoxels : NDArray
        3D array containing region labels with shape (n_voxels, 1, 1)
    normmethod : str, optional
        Normalization method to apply to time courses, default is "z"
        Supported methods depend on tide_math.normalize function
    debug : bool, optional
        If True, print debugging information including voxel counts and shapes,
        default is False

    Returns
    -------
    timecourses : NDArray
        2D array of shape (n_regions, n_timepoints) containing normalized mean
        time courses for each region

    Notes
    -----
    - Regions are assumed to be labeled starting from 1
    - Zero-valued voxels in template are ignored
    - NaN values are converted to zeros before computing means
    - The function uses tide_math.normalize for normalization

    Examples
    --------
    >>> import numpy as np
    >>> input_data = np.random.rand(100, 50, 1)
    >>> template = np.random.randint(1, 4, (100, 1, 1))
    >>> result = summarize4Dbylabel(input_data, template, normmethod="z")
    >>> print(result.shape)
    (3, 50)
    """
    numregions = np.max(templatevoxels)
    numtimepoints = inputvoxels.shape[1]
    timecourses = np.zeros((numregions, numtimepoints), dtype="float")
    for theregion in range(1, numregions + 1):
        thevoxels = inputvoxels[np.where(templatevoxels == theregion), :][0]
        print("extracting", thevoxels.shape[0], "voxels from region", theregion)
        if thevoxels.shape[1] > 0:
            regiontimecourse = np.nan_to_num(np.mean(thevoxels, axis=0))
        else:
            regiontimecourse = np.zeros_like(timecourses[0, :])
        if debug:
            print("thevoxels, data shape are:", thevoxels.shape, regiontimecourse.shape)
        timecourses[theregion - 1, :] = tide_math.normalize(regiontimecourse, method=normmethod)
    return timecourses


def summarize3Dbylabel(
    inputvoxels: NDArray, templatevoxels: NDArray, debug: bool = False
) -> Tuple[NDArray, list]:
    """
    Summarize 3D voxel data by label using mean, standard deviation, and median statistics.

    This function processes 3D voxel data by grouping voxels according to labels in a template
    and computes summary statistics for each labeled region. The input voxels are replaced
    with the mean value of each region, and statistics are returned for further analysis.

    Parameters
    ----------
    inputvoxels : NDArray
        3D array containing the voxel values to be summarized
    templatevoxels : NDArray
        3D array containing integer labels defining regions of interest
    debug : bool, optional
        Flag to enable debug output (default is False)

    Returns
    -------
    tuple
        A tuple containing:
        - outputvoxels : NDArray
          3D array with each labeled region replaced by its mean value
        - regionstats : list
          List of lists containing [mean, std, median] statistics for each region

    Notes
    -----
    - Regions are labeled starting from 1 to max(templatevoxels)
    - NaN values are converted to 0 during statistics calculation
    - The function modifies the input arrays in-place during processing

    Examples
    --------
    >>> import numpy as np
    >>> input_data = np.random.rand(10, 10, 10)
    >>> template = np.zeros((10, 10, 10), dtype=int)
    >>> template[2:5, 2:5, 2:5] = 1
    >>> template[6:8, 6:8, 6:8] = 2
    >>> result, stats = summarize3Dbylabel(input_data, template)
    >>> print(f"Region 1 mean: {stats[0][0]:.3f}")
    >>> print(f"Region 2 mean: {stats[1][0]:.3f}")
    """
    numregions = np.max(templatevoxels)
    outputvoxels = 0.0 * inputvoxels
    regionstats = []
    for theregion in range(1, numregions + 1):
        thevoxels = inputvoxels[np.where(templatevoxels == theregion)][0]
        regionmean = np.nan_to_num(np.mean(thevoxels))
        regionstd = np.nan_to_num(np.std(thevoxels))
        regionmedian = np.nan_to_num(np.median(thevoxels))
        regionstats.append([regionmean, regionstd, regionmedian])
        outputvoxels[np.where(templatevoxels == theregion)] = regionmean
    return outputvoxels, regionstats


def roisummarize(args: Any) -> None:
    """
    Summarize fMRI data by regions of interest (ROIs) using a template image.

    This function reads input fMRI and template NIfTI files, checks spatial
    compatibility, and computes either 3D or 4D summaries depending on the
    number of timepoints in the input data. For 4D data, it applies a filter
    and summarizes timecourses by ROI. For 3D data, it computes mean values and
    region statistics.

    Parameters
    ----------
    args : Any
        Command-line arguments parsed by `_get_parser()`. Expected attributes include:
        - `inputfilename` : str
            Path to the input fMRI NIfTI file.
        - `templatefile` : str
            Path to the template NIfTI file defining ROIs.
        - `samplerate` : str or float
            Sampling rate for filtering. If "auto", defaults to 1.0.
        - `numskip` : int
            Number of initial timepoints to skip when summarizing 4D data.
        - `normmethod` : str
            Normalization method for 4D summarization.
        - `debug` : bool
            Enable debug mode for additional output.
        - `outputfile` : str
            Base name for output files.

    Returns
    -------
    None
        The function writes output files to disk:
        - `<outputfile>_timecourses`: Timecourses for each ROI (4D case).
        - `<outputfile>_meanvals`: Mean values per ROI (3D case).
        - `<outputfile>_regionstats.txt`: Statistics for each ROI (3D case).

    Notes
    -----
    - The function assumes that the template file defines ROIs with integer labels.
    - For 4D data, the input is filtered using `pf.postprocessfilteropts`.
    - If the spatial dimensions of the input and template files do not match,
      the function exits with an error message.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfilename='fmri.nii',
    ...     templatefile='roi_template.nii',
    ...     samplerate='auto',
    ...     numskip=5,
    ...     normmethod='zscore',
    ...     debug=False,
    ...     outputfile='output'
    ... )
    >>> roisummarize(args)
    """
    # grab the command line arguments then pass them off.
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    # set the sample rate
    if args.samplerate == "auto":
        args.samplerate = 1.0
    else:
        samplerate = args.samplerate

    args, thefilter = pf.postprocessfilteropts(args, debug=args.debug)

    print("loading fmri data")
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    print("loading template data")
    template_img, template_data, template_hdr, templatedims, templatesizes = tide_io.readfromnifti(
        args.templatefile
    )

    print("checking dimensions")
    if not tide_io.checkspacematch(input_hdr, template_hdr):
        print("template file does not match spatial coverage of input fmri file")
        sys.exit()

    print("reshaping")
    xsize = thedims[1]
    ysize = thedims[2]
    numslices = thedims[3]
    numtimepoints = thedims[4]
    numvoxels = int(xsize) * int(ysize) * int(numslices)
    templatevoxels = np.reshape(template_data, numvoxels).astype(int)

    if numtimepoints > 1:
        inputvoxels = np.reshape(input_data, (numvoxels, numtimepoints))[:, args.numskip :]
        print("filtering")
        for thevoxel in range(numvoxels):
            if templatevoxels[thevoxel] > 0:
                inputvoxels[thevoxel, :] = thefilter.apply(
                    args.samplerate, inputvoxels[thevoxel, :]
                )

        print("summarizing")
        timecourses = summarize4Dbylabel(
            inputvoxels, templatevoxels, normmethod=args.normmethod, debug=args.debug
        )

        print("writing data")
        tide_io.writenpvecs(timecourses, args.outputfile + "_timecourses")
    else:
        inputvoxels = np.reshape(input_data, (numvoxels))
        numregions = np.max(templatevoxels)
        template_hdr["dim"][4] = numregions
        outputvoxels, regionstats = summarize3Dbylabel(
            inputvoxels, templatevoxels, debug=args.debug
        )
        tide_io.savetonifti(
            outputvoxels.reshape((xsize, ysize, numslices)),
            template_hdr,
            args.outputfile + "_meanvals",
        )
        tide_io.writenpvecs(np.array(regionstats), args.outputfile + "_regionstats.txt")
