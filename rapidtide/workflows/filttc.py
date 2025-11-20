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

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.workflows.parser_funcs as pf


def _get_parser() -> Any:
    """
    Argument parser for filttc.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    for parsing command-line arguments for the `filttc` tool, which filters timecourse
    data in text files. It includes support for specifying sampling rate or timestep,
    filter options, normalization methods, and miscellaneous flags.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the filttc tool.

    Notes
    -----
    The `--samplerate` and `--sampletstep` arguments are mutually exclusive and define
    the sampling frequency of the input timecourses. The sampling frequency can be
    specified either as a frequency (Hz) or as a timestep (seconds), with the latter
    being the inverse of the former.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['--inputfile', 'input.txt', '--outputfile', 'output.txt'])
    """
    parser = argparse.ArgumentParser(
        prog="filttc",
        description=("Filter timecourse data in text files"),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "inputfile")
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

    # Filter arguments
    pf.addfilteropts(parser, filtertarget="timecourses")

    # Normalization arguments
    pf.addnormalizationopts(parser, normtarget="timecourses", defaultmethod="None")

    parser.add_argument(
        "--normfirst",
        dest="normfirst",
        action="store_true",
        help=("Normalize before filtering, rather than after."),
        default=False,
    )
    parser.add_argument(
        "--demean",
        dest="demean",
        action="store_true",
        help=("Demean before filtering."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    # Miscellaneous options

    return parser


def filttc(args: Any) -> None:
    """
    Apply a filter to timecourse data read from a text file and write the filtered output.

    This function reads timecourse data from a specified input file, applies a filter
    to each timecourse, and writes the filtered data to an output file. It supports
    normalization, demeaning, and automatic sampling rate detection from the input file
    or command-line arguments.

    Parameters
    ----------
    args : Any
        An object containing command-line arguments. Expected attributes include:
        - `inputfile`: Path to the input text file containing timecourse data.
        - `outputfile`: Path to the output text file where filtered data will be written.
        - `samplerate`: Sampling rate of the data, or 'auto' to detect from file.
        - `normfirst`: Boolean indicating whether to normalize before filtering.
        - `normmethod`: Normalization method to use (e.g., 'zscore', 'minmax').
        - `demean`: Boolean indicating whether to remove the mean from the filtered data.

    Returns
    -------
    None
        This function does not return a value. It writes the filtered timecourse data
        to the specified output file.

    Notes
    -----
    - The input file must contain a header specifying the sampling rate, or the
      sampling rate must be provided via command-line arguments.
    - The filtering and normalization are applied independently to each timecourse.
    - If `normfirst` is True, normalization is applied before filtering; otherwise,
      filtering is applied before normalization.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfile='input.txt',
    ...     outputfile='output.txt',
    ...     samplerate='auto',
    ...     normfirst=False,
    ...     normmethod='zscore',
    ...     demean=True
    ... )
    >>> filttc(args)
    """
    args, thefilter = pf.postprocessfilteropts(args)

    # read in data
    (
        samplerate,
        starttime,
        colnames,
        invecs,
        compressed,
        filetype,
    ) = tide_io.readvectorsfromtextfile(args.inputfile)

    if samplerate is None:
        if args.samplerate == "auto":
            print(
                "sample rate must be specified, either by command line arguments or in the file header."
            )
            sys.exit()
        else:
            samplerate = args.samplerate
    else:
        if args.samplerate != "auto":
            samplerate = args.samplerate

    print("about to filter")
    numvecs = invecs.shape[0]
    if numvecs == 1:
        print("there is 1 timecourse")
    else:
        print("there are", numvecs, "timecourses")
    print("samplerate is", samplerate)
    outvecs = np.zeros_like(invecs)
    for i in range(numvecs):
        if args.normfirst:
            outvecs[i, :] = thefilter.apply(
                samplerate, tide_math.normalize(invecs[i, :], method=args.normmethod)
            )
        else:
            outvecs[i, :] = tide_math.normalize(
                thefilter.apply(samplerate, invecs[i, :]), method=args.normmethod
            )
        if args.demean:
            outvecs[i, :] -= np.mean(outvecs[i, :])

    tide_io.writevectorstotextfile(
        outvecs,
        args.outputfile,
        samplerate=samplerate,
        starttime=starttime,
        columns=colnames,
        compressed=compressed,
        filetype=filetype,
    )
    # tide_io.writenpvecs(outvecs, args.outputfile)
