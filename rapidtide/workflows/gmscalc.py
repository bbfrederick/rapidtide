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
import platform
import sys
import time
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


def _get_parser() -> Any:
    """
    Construct and return an argument parser for the gmscalc tool.

    This function sets up an `argparse.ArgumentParser` with required and optional
    arguments needed to run the global mean signal calculation and filtering
    pipeline. It includes support for specifying input data, output root, data mask,
    normalization options, smoothing, and debugging.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all required and optional arguments
        for the gmscalc tool.

    Notes
    -----
    The parser is configured with `allow_abbrev=False` to enforce full argument
    names and avoid ambiguity.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['data.nii', 'output_root'])
    >>> print(args.datafile)
    'data.nii'
    """
    parser = argparse.ArgumentParser(
        prog="gmscalc",
        description="Calculate the global mean signal, and filtered versions",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=str,
        help=(
            "The name of a 4 dimensional nifti files (3 spatial dimensions and a "
            "subject dimension)."
        ),
    )
    parser.add_argument("outputroot", type=str, help="The root name for the output nifti files.")

    # Optional arguments
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: pf.is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    pf.addnormalizationopts(parser, normtarget="timecourses", defaultmethod="None")

    parser.add_argument(
        "--normfirst",
        dest="normfirst",
        action="store_true",
        help=("Normalize before filtering, rather than after."),
        default=False,
    )
    parser.add_argument(
        "--smooth",
        dest="sigma",
        type=lambda x: pf.is_float(parser, x),
        action="store",
        metavar="SIGMA",
        help=("Spatially smooth the input data with a SIGMA mm kernel prior to calculation."),
        default=0.0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print out additional debugging information."),
        default=False,
    )
    return parser


def makecommandlinelist(
    arglist: Any, starttime: Any, endtime: Any, extra: Optional[Any] = None
) -> None:
    """
    Create a list of command line information for logging purposes.

    This function generates a list of descriptive strings containing processing
    information including date, duration, version details, and the actual command
    that was executed.

    Parameters
    ----------
    arglist : Any
        List of command line arguments to be joined into a command string.
    starttime : Any
        Start time of the process, typically a timestamp.
    endtime : Any
        End time of the process, typically a timestamp.
    extra : Any, optional
        Additional descriptive text to include in the output list. Default is None.

    Returns
    -------
    list of str
        List containing the following elements in order:
        - Processing date and time
        - Processing duration
        - Node and version information
        - Extra information (if provided)
        - The actual command line string

    Notes
    -----
    The function uses `time.strftime` to format the start time and `tide_util.version()`
    to retrieve version information. The command line is constructed by joining
    the `arglist` elements with spaces.

    Examples
    --------
    >>> import time
    >>> args = ['python', 'script.py', '--input', 'data.txt']
    >>> start = time.time()
    >>> # ... some processing ...
    >>> end = time.time()
    >>> info = makecommandlinelist(args, start, end)
    >>> print(info[0])
    '# Processed on Mon, 01 Jan 2024 12:00:00 UTC.'
    """
    # get the processing date
    dateline = (
        "# Processed on "
        + time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime(starttime))
        + "."
    )
    timeline = f"# Processing took {endtime - starttime:.3f} seconds."

    # diagnostic information about version
    (
        release_version,
        dummy,
        git_date,
        dummy,
    ) = tide_util.version()

    nodeline = "# " + " ".join(
        [
            "Using",
            platform.node(),
            "(",
            release_version + ",",
            git_date,
            ")",
        ]
    )

    # and the actual command issued
    commandline = " ".join(arglist)

    if extra is not None:
        return [dateline, timeline, nodeline, "# " + extra, commandline]
    else:
        return [dateline, timeline, nodeline, commandline]


def gmscalc_main() -> None:
    """
    Main function to calculate global mean signal (GMS) from fMRI data.

    This function reads NIfTI-formatted fMRI data, applies optional smoothing,
    masks the data if a mask is provided, and computes the global mean signal
    across valid voxels. It then applies low-frequency (LFO) and high-frequency
    (HF) filtering to the global signal and writes the results to text files.

    The function uses the `tide_io` module for reading and writing data, and
    `tide_filt` and `tide_math` for filtering and normalization.

    Notes
    -----
    The function expects a command-line interface to be set up with `_get_parser()`
    and uses `sys.argv` to parse arguments. It prints diagnostic information
    during execution.

    Examples
    --------
    Assuming the script is called as `gmscalc_main.py` and properly configured:

    >>> gmscalc_main()

    This will read the input data, perform processing, and write output files
    with names based on the `outputroot` argument.

    See Also
    --------
    tide_io.readfromnifti : Reads NIfTI files.
    tide_io.writevec : Writes vectors to text files.
    tide_filt.ssmooth : Applies spatial smoothing.
    tide_filt.NoncausalFilter : Applies non-causal filtering.
    tide_math.normalize : Normalizes a signal.
    """
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    runstarttime = time.time()

    # now read in the data
    print("reading datafile")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        thedims,
        thesizes,
    ) = tide_io.readfromnifti(args.datafile)

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    print(f"Datafile shape is {datafile_data.shape}")

    # smooth the data
    if args.sigma > 0.0:
        print("smoothing data")
        for i in range(timepoints):
            datafile_data[:, :, :, i] = tide_filt.ssmooth(
                xdim, ydim, slicethickness, args.sigma, datafile_data[:, :, :, i]
            )
        print("done smoothing data")

    if args.datamaskname is not None:
        print("reading in mask array")
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(args.datamaskname)
        if not tide_io.checkspacematch(datafile_hdr, datamask_hdr):
            print("Data and mask dimensions do not match - exiting.")
            sys.exit()
        print("done reading in mask array")
    else:
        datamask_data = np.ones_like(datafile_data[:, :, :, 0])

    # now reformat from x, y, z, time to voxelnumber, measurement, subject
    numvoxels = int(xsize) * int(ysize) * int(numslices)
    mask_in_vox = datamask_data.reshape((numvoxels))

    print("finding valid voxels")
    validvoxels = np.where(mask_in_vox > 0)[0]
    numvalid = int(len(validvoxels))

    data_in_voxacq = datafile_data.reshape((numvoxels, timepoints))
    valid_in_voxacq = data_in_voxacq[validvoxels, :]

    # calculate each voxel's mean over time
    mean_in_voxacq = np.mean(valid_in_voxacq, axis=1)

    # calculate mean timecourse
    gms = np.mean(valid_in_voxacq, axis=0)

    # now filter
    lfofilter = tide_filt.NoncausalFilter("lfo")
    hffilter = tide_filt.NoncausalFilter("arb")
    lowerpass = 0.175
    lowerstop = 0.95 * lowerpass
    hffilter.setfreqs(lowerstop, lowerpass, 10.0, 10.0)
    gms_lfo = tide_math.normalize(lfofilter.apply(1.0 / tr, gms), method=args.normmethod)
    gms_hf = tide_math.normalize(hffilter.apply(1.0 / tr, gms), method=args.normmethod)

    tide_io.writevec(gms, args.outputroot + "_gms.txt")
    tide_io.writevec(gms_lfo, args.outputroot + "_gmslfo.txt")
    tide_io.writevec(gms_hf, args.outputroot + "_gmshf.txt")
    runendtime = time.time()
    thecommandfilelines = makecommandlinelist(sys.argv, runstarttime, runendtime)
    tide_io.writevec(thecommandfilelines, args.outputroot + "_commandline.txt")
