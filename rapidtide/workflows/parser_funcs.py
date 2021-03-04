"""
Functions for parsers.
"""
import os.path as op
import argparse

import rapidtide.filter as tide_filt


class IndicateSpecifiedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_nondefault", True)


def setifnotset(thedict, thekey, theval):
    if (thekey + "_nondefault") not in thedict.keys():
        print("overriding " + thekey)
        thedict[thekey] = theval


def is_valid_filespec(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is None:
        parser.error("No file specified")

    thesplit = arg.split(":")
    if not op.isfile(thesplit[0]):
        parser.error("The file {0} does not exist!".format(thesplit[0]))

    return arg


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
    if arg != "auto":
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))

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


def addreqinputniftifile(parser, varname, addedtext=""):
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Input NIFTI file name.  " + addedtext,
    )


def addreqoutputniftifile(parser, varname, addedtext=""):
    parser.add_argument(
        varname,
        type=str,
        help="Output NIFTI file name.  " + addedtext,
    )


def addreqinputtextfile(parser, varname, onecol=False):
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)"
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


def addreqoutputtextfile(parser, varname, rootname=False):
    if rootname:
        helpline = "Root name for the output files"
    else:
        helpline = "Name of the output text file."
    parser.add_argument(
        varname,
        type=str,
        help=helpline,
    )


def addfilteropts(parser, filtertarget, details=False):
    filt_opts = parser.add_argument_group("Filtering options")
    filt_opts.add_argument(
        "--filterband",
        dest="filterband",
        action="store",
        type=str,
        choices=["vlf", "lfo", "resp", "cardiac", "lfo_legacy"],
        help=("Filter " + filtertarget + " to specific band. "),
        default="lfo",
    )
    filt_opts.add_argument(
        "--filterfreqs",
        dest="arbvec",
        action="store",
        nargs="+",
        type=lambda x: is_float(parser, x),
        metavar=("LOWERPASS UPPERPASS", "LOWERSTOP UPPERSTOP"),
        help=(
            "Filter " + filtertarget + " to retain LOWERPASS to "
            "UPPERPASS. LOWERSTOP and UPPERSTOP can also "
            "be specified, or will be calculated "
            "automatically. "
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
                "Filter "
                + filtertarget
                + " using a trapezoidal FFT filter (default), brickwall, or "
                "butterworth bandpass."
            ),
            default="trapezoidal",
        )
        filt_opts.add_argument(
            "--butterorder",
            dest="filtorder",
            action="store",
            type=int,
            metavar="ORDER",
            help=("Set order of butterworth filter (if used)."),
            default=DEFAULT_FILTER_ORDER,
        )
        filt_opts.add_argument(
            "--padseconds",
            dest="padseconds",
            action="store",
            type=float,
            metavar="SECONDS",
            help=(
                "The number of seconds of padding to add to each end of a " "filtered timecourse. "
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

    # if arbvec is set, we are going set up an arbpass filter
    if args.arbvec is not None:
        if len(args.arbvec) == 2:
            args.arbvec.append(args.arbvec[0] * 0.95)
            args.arbvec.append(args.arbvec[1] * 1.05)
        elif len(args.arbvec) != 4:
            raise ValueError("Argument '--arb' must be either two or four " "floats.")
        # NOTE - this vector is LOWERPASS, UPPERPASS, LOWERSTOP, UPPERSTOP
        # setfreqs expects LOWERSTOP, LOWERPASS, UPPERPASS, UPPERSTOP
        theprefilter = tide_filt.noncausalfilter(
            "arb",
            transferfunc=args.filtertype,
        )
        theprefilter.setfreqs(args.arbvec[2], args.arbvec[0], args.arbvec[1], args.arbvec[3])
    else:
        theprefilter = tide_filt.noncausalfilter(args.filterband, transferfunc=args.filtertype)

    # set the butterworth order
    theprefilter.setbutterorder(args.filtorder)

    (
        args.lowerstop,
        args.lowerpass,
        args.upperpass,
        args.upperstop,
    ) = theprefilter.getfreqs()

    return args, theprefilter


def addwindowopts(parser):
    wfunc = parser.add_argument_group("Windowing options")
    wfunc.add_argument(
        "--windowfunc",
        dest="windowfunc",
        action="store",
        type=str,
        choices=["hamming", "hann", "blackmanharris", "None"],
        help=(
            "Window function to use prior to correlation. "
            "Options are hamming (default), hann, "
            "blackmanharris, and None. "
        ),
        default="hamming",
    )


def addpermutationopts(parser):
    permutationmethod = parser.add_mutually_exclusive_group()
    permutationmethod.add_argument(
        "--permutationmethod",
        dest="permutationmethod",
        action="store",
        type=str,
        choices=["shuffle", "phaserandom"],
        help=("Permutation method for significance testing.  " "Default is shuffle."),
        default="shuffle",
    )
    parser.add_argument(
        "--numnull",
        dest="numestreps",
        action="store",
        type=int,
        metavar="NREPS",
        help=(
            "Estimate significance threshold by running "
            "NREPS null correlations (default is 10000, "
            "set to 0 to disable). "
        ),
        default=10000,
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
        args.fixdelayvalue = None
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
