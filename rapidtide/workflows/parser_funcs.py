#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2018-2021 Blaise Frederick
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
"""
Functions for parsers.
"""
import argparse
import os.path as op
import sys

import rapidtide.filter as tide_filt


class IndicateSpecifiedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_nondefault", True)


def setifnotset(thedict, thekey, theval):
    if (thekey + "_nondefault") not in thedict.keys():
        print("overriding " + thekey)
        thedict[thekey] = theval


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is not None:
        thefilename = arg.split(":")[0]
    else:
        thefilename = None

    if not op.isfile(thefilename) and thefilename is not None:
        parser.error("The file {0} does not exist!".format(thefilename))

    return arg


def invert_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    arg = is_float(parser, arg)

    if arg != "auto":
        arg = 1.0 / arg
    return arg


def is_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    if arg != "auto":
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))

    return arg


def is_int(parser, arg):
    """
    Check if argument is int or auto.
    """
    if arg != "auto":
        try:
            arg = int(arg)
        except parser.error:
            parser.error('Value {0} is not an int or "auto"'.format(arg))

    return arg


def is_range(parser, arg):
    """
    Check if argument is min/max pair.
    """
    if arg is not None and len(arg) != 2:
        parser.error("Argument must be min/max pair.")
    elif arg is not None and float(arg[0]) > float(arg[1]):
        parser.error("Argument min must be lower than max.")

    return arg


DEFAULT_FILTER_ORDER = 6
DEFAULT_PAD_SECONDS = 30.0
DEFAULT_PERMUTATIONMETHOD = "shuffle"
DEFAULT_NORMTYPE = "stddev"
DEFAULT_FILTERBAND = "lfo"
DEFAULT_FILTERTYPE = "trapezoidal"
DEFAULT_PADVAL = 0
DEFAULT_WINDOWFUNC = "hamming"


def addreqinputniftifile(parser, varname, addedtext=""):
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Input NIFTI file name.  " + addedtext,
    )


def addreqoutputniftifile(parser, varname, addedtext=""):
    parser.add_argument(
        varname, type=str, help="Output NIFTI file name.  " + addedtext,
    )


def addreqinputtextfile(parser, varname, onecol=False):
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)."
        )
    else:
        colspecline = (
            "Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an "
            "integer, a column separated list of ranges, or a comma "
            "separated set of column names (if input file is BIDS).  Default is to use all columns"
        )
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Text file containing one or more timeseries columns. " + colspecline,
    )


def addreqinputtextfiles(parser, varname, numreq="Two", nargs="*", onecol=False):
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)."
        )
    else:
        colspecline = (
            "Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an "
            "integer, a column separated list of ranges, or a comma "
            "separated set of column names (if input file is BIDS).  Default is to use all columns."
        )
    parser.add_argument(
        varname,
        nargs=nargs,
        type=lambda x: is_valid_file(parser, x),
        help=numreq + " text files containing one or more timeseries columns. " + colspecline,
    )


def addreqoutputtextfile(parser, varname, rootname=False):
    if rootname:
        helpline = "Root name for the output files"
    else:
        helpline = "Name of the output text file."
    parser.add_argument(
        varname, type=str, help=helpline,
    )


def addnormalizationopts(parser, normtarget="timecourse", defaultmethod=DEFAULT_NORMTYPE):
    norm_opts = parser.add_argument_group("Normalization options")
    norm_opts.add_argument(
        "--normmethod",
        dest="normmethod",
        action="store",
        type=str,
        choices=["None", "percent", "variance", "stddev", "z", "p2p", "mad"],
        help=(
            f"Demean and normalize {normtarget} "
            "using one of the following methods: "
            '"None" - demean only; '
            '"percent" - divide by mean; '
            '"variance" - divide by variance; '
            '"stddev" or "z" - divide by standard deviation; '
            '"p2p" - divide by range; '
            '"mad" - divide by median absolute deviation. '
            f'Default is "{defaultmethod}".'
        ),
        default=defaultmethod,
    )


def addfilteropts(
    parser, filtertarget="timecourses", defaultmethod=DEFAULT_FILTERBAND, details=False
):
    filt_opts = parser.add_argument_group("Filtering options")
    filt_opts.add_argument(
        "--filterband",
        dest="filterband",
        action="store",
        type=str,
        choices=["None", "vlf", "lfo", "resp", "cardiac", "lfo_legacy"],
        help=(
            f'Filter {filtertarget} to specific band. Use "None" to disable filtering.  '
            f'Default is "{defaultmethod}".'
        ),
        default=defaultmethod,
    )
    filt_opts.add_argument(
        "--filterfreqs",
        dest="passvec",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERPASS", "UPPERPASS"),
        help=(
            "Filter " + filtertarget + " to retain LOWERPASS to "
            "UPPERPASS. If --filterstopfreqs is not also specified, "
            "LOWERSTOP and UPPERSTOP will be calculated "
            "automatically. "
        ),
        default=None,
    )
    filt_opts.add_argument(
        "--filterstopfreqs",
        dest="stopvec",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERSTOP", "UPPERSTOP"),
        help=(
            "Filter " + filtertarget + " to with stop frequencies LOWERSTOP and UPPERSTOP. "
            "LOWERSTOP must be <= LOWERPASS, UPPERSTOP must be >= UPPERPASS. "
            "Using this argument requires the use of --filterfreqs."
        ),
        default=None,
    )
    if details:
        filt_opts.add_argument(
            "--filtertype",
            dest="filtertype",
            action="store",
            type=str,
            choices=["trapezoidal", "brickwall", "butterworth"],
            help=(
                f"Filter {filtertarget} "
                "using a trapezoidal FFT, brickwall FFT, or "
                "butterworth bandpass filter. "
                f'Default is "{DEFAULT_FILTERTYPE}".'
            ),
            default=DEFAULT_FILTERTYPE,
        )
        filt_opts.add_argument(
            "--butterorder",
            dest="filtorder",
            action="store",
            type=int,
            metavar="ORDER",
            help=(
                "Set order of butterworth filter (if used). " f"Default is {DEFAULT_FILTER_ORDER}."
            ),
            default=DEFAULT_FILTER_ORDER,
        )
        filt_opts.add_argument(
            "--padseconds",
            dest="padseconds",
            action="store",
            type=float,
            metavar="SECONDS",
            help=(
                "The number of seconds of padding to add to each end of a "
                "filtered timecourse "
                f"to reduce end effects.  Default is {DEFAULT_PAD_SECONDS}."
            ),
            default=DEFAULT_PAD_SECONDS,
        )


def postprocessfilteropts(args):
    # configure the filter
    # set the trapezoidal flag, if using
    try:
        thetype = args.filtertype
    except AttributeError:
        args.filtertype = "trapezoidal"
    try:
        theorder = args.filtorder
    except AttributeError:
        args.filtorder = DEFAULT_FILTER_ORDER
    try:
        thepadseconds = args.padseconds
    except AttributeError:
        args.padseconds = DEFAULT_PAD_SECONDS

    # if passvec, or passvec and stopvec, are set, we are going set up an arbpass filter
    args.arbvec = None
    if args.stopvec is not None:
        if args.passvec is not None:
            args.arbvec = [args.passvec[0], args.passvec[1], args.stopvec[0], args.stopvec[1]]
        else:
            raise ValueError("--filterfreqs must be used if --filterstopfreqs is specified")
    else:
        if args.passvec is not None:
            args.arbvec = [
                args.passvec[0],
                args.passvec[1],
                args.passvec[0] * 0.95,
                args.passvec[1] * 1.05,
            ]
    if args.arbvec is not None:
        # NOTE - this vector is LOWERPASS, UPPERPASS, LOWERSTOP, UPPERSTOP
        # setfreqs expects LOWERSTOP, LOWERPASS, UPPERPASS, UPPERSTOP
        theprefilter = tide_filt.NoncausalFilter("arb", transferfunc=args.filtertype,)
        theprefilter.setfreqs(args.arbvec[2], args.arbvec[0], args.arbvec[1], args.arbvec[3])
    else:
        theprefilter = tide_filt.NoncausalFilter(
            args.filterband, transferfunc=args.filtertype, padtime=args.padseconds,
        )

    # set the butterworth order
    theprefilter.setbutterorder(args.filtorder)

    (args.lowerstop, args.lowerpass, args.upperpass, args.upperstop,) = theprefilter.getfreqs()

    return args, theprefilter


def addwindowopts(parser, windowtype=DEFAULT_WINDOWFUNC):
    wfunc = parser.add_argument_group("Windowing options")
    wfunc.add_argument(
        "--windowfunc",
        dest="windowfunc",
        action="store",
        type=str,
        choices=["hamming", "hann", "blackmanharris", "None"],
        help=(
            "Window function to use prior to correlation. "
            "Options are hamming, hann, "
            f"blackmanharris, and None. Default is {windowtype}"
        ),
        default=windowtype,
    )
    wfunc.add_argument(
        "--nowindow",
        dest="windowfunc",
        action="store_const",
        const="None",
        help="Disable precorrelation windowing.",
        default=windowtype,
    )
    wfunc.add_argument(
        "--zeropadding",
        dest="zeropadding",
        action="store",
        type=int,
        metavar="PADVAL",
        help=(
            "Pad input functions to correlation with PADVAL zeros on each side. "
            "A PADVAL of 0 does circular correlations, positive values reduce edge artifacts. "
            f"Set PADVAL < 0 to set automatically. Default is {DEFAULT_PADVAL}."
        ),
        default=DEFAULT_PADVAL,
    )


def addplotopts(parser, multiline=True):
    plotopts = parser.add_argument_group("General plot appearance options")
    plotopts.add_argument(
        "--title",
        dest="thetitle",
        metavar="TITLE",
        type=str,
        action="store",
        help="Use TITLE as the overall title of the graph.",
        default="",
    )
    plotopts.add_argument(
        "--xlabel",
        dest="xlabel",
        metavar="LABEL",
        type=str,
        action="store",
        help="Label for the plot x axis.",
        default="",
    )
    plotopts.add_argument(
        "--ylabel",
        dest="ylabel",
        metavar="LABEL",
        type=str,
        action="store",
        help="Label for the plot y axis.",
        default="",
    )
    if multiline:
        plotopts.add_argument(
            "--legends",
            dest="legends",
            metavar="LEGEND[,LEGEND[,LEGEND...]]",
            type=str,
            action="store",
            help="Comma separated list of legends for each timecourse.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--legend",
            dest="legends",
            metavar="LEGEND",
            type=str,
            action="store",
            help="Legends for the timecourse.",
            default=None,
        )
    plotopts.add_argument(
        "--legendloc",
        dest="legendloc",
        metavar="LOC",
        type=int,
        action="store",
        help=(
            "Integer from 0 to 10 inclusive specifying legend location.  Legal values are: "
            "0: best, 1: upper right, 2: upper left, 3: lower left, 4: lower right, "
            "5: right, 6: center left, 7: center right, 8: lower center, 9: upper center, "
            "10: center.  Default is 2."
        ),
        default=2,
    )
    if multiline:
        plotopts.add_argument(
            "--colors",
            dest="colors",
            metavar="COLOR[,COLOR[,COLOR...]]",
            type=str,
            action="store",
            help="Comma separated list of colors for each timecourse.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--color",
            dest="colors",
            metavar="COLOR",
            type=str,
            action="store",
            help="Color of the timecourse plot.",
            default=None,
        )
    plotopts.add_argument(
        "--nolegend",
        dest="dolegend",
        action="store_false",
        help="Turn off legend label.",
        default=True,
    )
    plotopts.add_argument(
        "--noxax", dest="showxax", action="store_false", help="Do not show x axis.", default=True,
    )
    plotopts.add_argument(
        "--noyax", dest="showyax", action="store_false", help="Do not show y axis.", default=True,
    )
    if multiline:
        plotopts.add_argument(
            "--linewidth",
            dest="linewidths",
            metavar="LINEWIDTH[,LINEWIDTH[,LINEWIDTH...]]",
            type=str,
            help="A comma separated list of linewidths (in points) for plots.  Default is 1.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--linewidth",
            dest="linewidths",
            metavar="LINEWIDTH",
            type=str,
            help="Linewidth (in points) for plot.  Default is 1.",
            default=None,
        )
    plotopts.add_argument(
        "--tofile",
        dest="outputfile",
        metavar="FILENAME",
        type=str,
        action="store",
        help="Write figure to file FILENAME instead of displaying on the screen.",
        default=None,
    )
    plotopts.add_argument(
        "--fontscalefac",
        dest="fontscalefac",
        metavar="FAC",
        type=float,
        action="store",
        help="Scaling factor for annotation fonts (default is 1.0).",
        default=1.0,
    )
    plotopts.add_argument(
        "--saveres",
        dest="saveres",
        metavar="DPI",
        type=int,
        action="store",
        help="Write figure to file at DPI dots per inch (default is 1000).",
        default=1000,
    )


def addpermutationopts(parser, numreps=10000):
    permutationmethod = parser.add_mutually_exclusive_group()
    permutationmethod.add_argument(
        "--permutationmethod",
        dest="permutationmethod",
        action="store",
        type=str,
        choices=["shuffle", "phaserandom"],
        help=(
            "Permutation method for significance testing. "
            f'Default is "{DEFAULT_PERMUTATIONMETHOD}".'
        ),
        default=DEFAULT_PERMUTATIONMETHOD,
    )
    parser.add_argument(
        "--numnull",
        dest="numestreps",
        action="store",
        type=int,
        metavar="NREPS",
        help=(
            "Estimate significance threshold by running "
            f"NREPS null correlations (default is {numreps}, "
            "set to 0 to disable). "
        ),
        default=numreps,
    )
    parser.add_argument(
        "--skipsighistfit",
        dest="dosighistfit",
        action="store_false",
        help=("Do not fit significance histogram with a Johnson SB function."),
        default=True,
    )


def addsearchrangeopts(parser, details=False, defaultmin=-30.0, defaultmax=30.0):
    parser.add_argument(
        "--searchrange",
        dest="lag_extrema",
        action=IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=("LAGMIN", "LAGMAX"),
        help=(
            "Limit fit to a range of lags from LAGMIN to "
            "LAGMAX.  Default is -30.0 to 30.0 seconds. "
        ),
        default=(defaultmin, defaultmax),
    )
    if details:
        parser.add_argument(
            "--fixdelay",
            dest="fixeddelayvalue",
            action="store",
            type=float,
            metavar="DELAYTIME",
            help=("Don't fit the delay time - set it to " "DELAYTIME seconds for all voxels. "),
            default=None,
        )


def postprocesssearchrangeopts(args):
    # Additional argument parsing not handled by argparse
    # first handle fixed delay
    try:
        test = args.fixeddelayvalue
    except:
        args.fixeddelayvalue = None
    if args.fixeddelayvalue is not None:
        args.fixdelay = True
        args.lag_extrema = (args.fixeddelayvalue - 10.0, args.fixeddelayvalue + 10.0)
    else:
        args.fixdelay = False

    # now set the extrema
    try:
        test = args.lag_extrema_nondefault
        args.lagmin_nondefault = True
        args.lagmax_nondefault = True
    except AttributeError:
        pass
    args.lagmin = args.lag_extrema[0]
    args.lagmax = args.lag_extrema[1]
    return args


def addtimerangeopts(parser):
    parser.add_argument(
        "--timerange",
        dest="timerange",
        action="store",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Limit analysis to data between timepoints "
            "START and END in the input file. If END is set to -1, "
            "analysis will go to the last timepoint.  Negative values "
            "of START will be set to 0. Default is to use all timepoints."
        ),
        default=(-1, -1),
    )


def postprocesstimerangeopts(args):
    args.startpoint = int(args.timerange[0])
    if args.timerange[1] == -1:
        args.endpoint = 100000000
    else:
        args.endpoint = int(args.timerange[1])
    return args


def addsimilarityopts(parser):
    parser.add_argument(
        "--mutualinfosmoothingtime",
        dest="smoothingtime",
        action="store",
        type=float,
        metavar="TAU",
        help=(
            "Time constant of a temporal smoothing function to apply to the "
            "mutual information function. "
            "Default is 3.0 seconds.  TAU <=0.0 disables smoothing."
        ),
        default=3.0,
    )


def setargs(thegetparserfunc, inputargs=None):
    """
    Compile arguments for rapidtide workflow.
    """
    if inputargs is None:
        # get arguments from the command line
        # LGR.info("processing command line arguments")
        try:
            args = thegetparserfunc().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            thegetparserfunc().print_help()
            raise
    else:
        # get arguments from the passed list
        # LGR.info("processing passed argument list:")
        # LGR.info(inputargs)
        try:
            args = thegetparserfunc().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            thegetparserfunc().print_help()
            raise

    return args, argstowrite


def generic_init(theparser, themain, inputargs=None):
    """
    Compile arguments either from the command line, or from an argument list.
    """
    if inputargs is None:
        print("processing command line arguments")
        # write out the command used
        try:
            args = theparser().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            theparser().print_help()
            raise
    else:
        print("processing passed argument list:")
        try:
            args = theparser().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            theparser().print_help()
            raise

    # save the raw and formatted command lines
    args.commandline = " ".join(argstowrite)

    themain(args)
