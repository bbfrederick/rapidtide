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
import os
import sys
from typing import Any

import rapidtide.dlfiltertorch as tide_dlfilt
import rapidtide.happy_supportfuncs as happy_support
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.workflows.parser_funcs as pf

DEFAULT_MODEL = "model_cnn_pytorch"


def _get_parser() -> Any:
    """
    Argument parser for applydlfilter.

    This function creates and configures an argument parser for the `applydlfilter`
    command-line tool, which applies a deep learning filter to a timecourse.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with defined arguments for `applydlfilter`.

    Notes
    -----
    The parser includes the following required and optional arguments:

    - ``infilename``: Input text file (or list of files) containing timecourse data.
    - ``outfilename``: Output text file (or list of files) to save filtered data.
    - ``--model``: Model root name to use for filtering (default: ``model_revised``).
    - ``--filesarelists``: Flag indicating input file contains lists of filenames.
    - ``--nodisplay``: Flag to disable plotting (for non-interactive use).
    - ``--verbose``: Flag to enable verbose output.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.txt', 'output.txt', '--model', 'custom_model'])
    >>> print(args.infilename)
    'input.txt'
    >>> print(args.model)
    'custom_model'
    """
    parser = argparse.ArgumentParser(
        prog="applydlfilter",
        description=("Apply a deep learning filter to a timecourse."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "infilename",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the input text file (or a list of names of input files).",
    )
    parser.add_argument(
        "outfilename",
        help="The name of the output text file (or a list of names of output files).",
    )

    # add optional arguments
    parser.add_argument(
        "--model",
        dest="model",
        action="store",
        metavar="MODELROOT",
        type=str,
        help=(f"Use model named MODELROOT (default is {DEFAULT_MODEL})."),
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--plethfile",
        dest="plethfile",
        action="store",
        metavar="FILE",
        type=str,
        help=(f"Check agreement with an actual plethysmogram."),
        default=None,
    )
    parser.add_argument(
        "--filesarelists",
        dest="filesarelists",
        action="store_true",
        help=("Input file contains lists of filenames, rather than data."),
        default=False,
    )
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not plot the data (for noninteractive use)."),
        default=True,
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Print a lot of internal information."),
        default=False,
    )
    return parser


def applydlfilter(args: Any) -> None:
    """
    Apply a deep learning filter to fMRI data files.

    This function reads fMRI data from input files, applies a deep learning filter
    to denoise the data, and writes the filtered output to specified files. It supports
    processing multiple files either from lists or a single file, and optionally displays
    the filtering results using matplotlib.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - `infilename` : str
            Path to the input file or list of input files.
        - `outfilename` : str
            Path to the output file or list of output files.
        - `filesarelists` : bool
            If True, `infilename` and `outfilename` are treated as paths to text files
            containing lists of input and output filenames, respectively.
        - `model` : str
            Path to the deep learning model to be used for filtering.
        - `plethfile` : str
            Path to plethysmogram to evaluate performance.
        - `display` : bool
            If True, displays the original and filtered data using matplotlib.
        - `verbose` : bool
            If True, prints verbose output during processing.

    Returns
    -------
    None
        This function does not return any value. It writes filtered data to files
        and optionally displays plots.

    Notes
    -----
    - The function assumes that the input data has a sampling rate of 25.0 Hz.
    - If `filesarelists` is True, the input and output filenames are read from
      text files, where each line contains a single filename.
    - The function checks for matching list lengths when processing multiple files.
    - If a bad points file is specified, it is read and used during filtering.
    - The deep learning model is loaded from a predefined model path within the
      `rapidtide` package.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     infilename="input.txt",
    ...     outfilename="output.txt",
    ...     filesarelists=False,
    ...     model="model.h5",
    ...     display=False,
    ...     verbose=True
    ... )
    >>> applydlfilter(args)
    """
    if args.display:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

    if args.filesarelists:
        infilenamelist = []
        with open(args.infilename, "r") as f:
            inputlist = f.readlines()
            for line in inputlist:
                infilenamelist.append(line.strip())
                if args.verbose:
                    print(infilenamelist[-1])
        outfilenamelist = []
        with open(args.outfilename, "r") as f:
            inputlist = f.readlines()
            for line in inputlist:
                outfilenamelist.append(line.strip())
                if args.verbose:
                    print(outfilenamelist[-1])
        if len(infilenamelist) != len(outfilenamelist):
            print("list lengths do not match - exiting")
            sys.exit()
    else:
        infilenamelist = [args.infilename]
        outfilenamelist = [args.outfilename]

    # load the filter
    modelpath = os.path.join(
        os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0],
        "rapidtide",
        "data",
        "models",
    )
    thedlfilter = tide_dlfilt.DeepLearningFilter(modelpath=modelpath)
    thedlfilter.loadmodel(args.model)
    usebadpts = thedlfilter.usebadpts

    plethwave = None
    if args.plethfile is not None:
        (
            thesamplerate,
            thestarttime,
            thecolumns,
            plethwave,
            compressed,
            filetype,
        ) = tide_io.readvectorsfromtextfile(args.plethfile, onecol=True, debug=args.verbose)

    badpts = None
    if usebadpts:
        badptsname = f"{(args.infilename.split(':'))[0]}:badpts"
        try:
            (
                thesamplerate,
                thestarttime,
                thecolumns,
                badpts,
                compressed,
                filetype,
            ) = tide_io.readvectorsfromtextfile(badptsname, onecol=True, debug=args.verbose)
        except:
            print(
                "bad points file",
                badptsname,
                "not found!",
            )
            sys.exit()

    for idx, infilename in enumerate(infilenamelist):
        # read in the data
        if args.verbose:
            print("reading in", infilename)
        (
            thesamplerate,
            thestarttime,
            thecolumns,
            fmridata,
            compressed,
            filetype,
        ) = tide_io.readvectorsfromtextfile(infilename, onecol=True, debug=args.verbose)
        if args.verbose:
            print("data is read")
        if thesamplerate != 25.0:
            print("sampling rate", thesamplerate)
            sys.exit()
        if args.verbose:
            print("filtering...")
        predicteddata = thedlfilter.apply(fmridata, badpts=badpts)
        if args.verbose:
            print("done...")

        # performance metrics
        extradict = {}
        maxval, maxdelay, failreason = happy_support.checkcardmatch(
            fmridata, predicteddata, 25.0, debug=False
        )
        print(infilename, "max correlation of input to output:", maxval)
        extradict["corrtoinput"] = maxval + 0.0

        if plethwave is not None:
            maxval, maxdelay, failreason = happy_support.checkcardmatch(
                fmridata, plethwave, 25.0, debug=False
            )
            print(infilename, "max correlation of input to target plethysmogram:", maxval)
            extradict["corrtopleth"] = maxval + 0.0

        if args.verbose:
            print("writing to", outfilenamelist[idx])
        tide_io.writebidstsv(
            outfilenamelist[idx],
            predicteddata,
            25.0,
            extraheaderinfo=extradict,
            columns=["filtered_signal"],
            debug=args.verbose,
        )

        # normalize
        fmridata = tide_math.stdnormalize(fmridata)
        predicteddata = tide_math.stdnormalize(predicteddata)

        if args.display:
            plt.figure()
            plt.plot(fmridata + 1.5)
            plt.plot(predicteddata - 1.5)
            plt.show()
