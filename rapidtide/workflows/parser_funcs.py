"""
Functions for parsers.
"""
import os.path as op
import argparse

import rapidtide.filter as tide_filt


class indicatespecifiedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)

def setifnotset(thedict, thekey, theval):
    try:
        test = thedict[thekey + '_nondefault']
    except KeyError:
        print('overriding ' + thekey)
        thedict[thekey] = theval


def is_valid_filespec(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is None:
        parser.error('No file specified')

    thesplit = arg.split(':')
    if not op.isfile(thesplit[0]):
        parser.error('The file {0} does not exist!'.format(thesplit[0]))

    return arg


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


def invert_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    if arg != 'auto':
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))

    if arg != 'auto':
        arg = 1.0 / arg
    return arg


def is_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    if arg != 'auto':
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))

    return arg


def is_int(parser, arg):
    """
    Check if argument is int or auto.
    """
    if arg != 'auto':
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
        parser.error('Argument must be min/max pair.')
    elif arg is not None and float(arg[0]) > float(arg[1]):
        parser.error('Argument min must be lower than max.')

    return arg


def addfilteropts(parser, filtertarget, details=False):
    filt_opts = parser.add_argument_group('Filtering options')
    filt_opts.add_argument('--filterband',
                          dest='filterband',
                          action='store',
                          type=str,
                          choices=['vlf', 'lfo', 'resp', 'cardiac', 'lfo_legacy'],
                          help=('Filter ' + filtertarget + ' to specific band. '),
                          default='lfo')
    filt_opts.add_argument('--filterfreqs',
                          dest='arbvec',
                          action='store',
                          nargs='+',
                          type=lambda x: is_float(parser, x),
                          metavar=('LOWERPASS UPPERPASS',
                                   'LOWERSTOP UPPERSTOP'),
                          help=('Filter ' + filtertarget + ' to retain LOWERPASS to '
                                'UPPERPASS. LOWERSTOP and UPPERSTOP can also '
                                'be specified, or will be calculated '
                                'automatically. '),
                          default=None)
    if details:
        filt_opts.add_argument('--filtertype',
                              dest='filtertype',
                              action='store',
                              type=str,
                              choices=['trapezoidal', 'brickwall', 'butterworth'],
                              help=('Filter ' + filtertarget + ' using a trapezoidal FFT filter (default), brickwall, or butterworth bandpass.'),
                              default='trapezoidal')
        filt_opts.add_argument('--butterorder',
                             dest='filtorder',
                             action='store',
                             type=int,
                             metavar='ORDER',
                             help=('Set order of butterworth filter (if used).'),
                             default=6)
        filt_opts.add_argument('--padseconds',
                             dest='padseconds',
                             action='store',
                             type=float,
                             metavar='SECONDS',
                             help=('The number of seconds of padding to add to each end of a filtered timecourse. '),
                             default=30.0)


def addpermutationopts(parser):
    permutationmethod = parser.add_mutually_exclusive_group()
    permutationmethod.add_argument('--permutationmethod',
                          dest='permutationmethod',
                          action='store',
                          type=str,
                          choices=['shuffle', 'phaserandom'],
                          help=('Permutation method for significance testing.  Default is shuffle. '),
                          default='shuffle')

    parser.add_argument('--numnull',
                         dest='numestreps',
                         action='store',
                         type=int,
                         metavar='NREPS',
                         help=('Estimate significance threshold by running '
                               'NREPS null correlations (default is 10000, '
                               'set to 0 to disable). '),
                         default=10000)

def addlagrangeopts(parser, defaultmin=-30.0, defaultmax=30.0):
    parser.add_argument('--searchrange',
                          dest='lag_extrema',
                          action=indicatespecifiedAction,
                          nargs=2,
                          type=float,
                          metavar=('LAGMIN', 'LAGMAX'),
                          help=('Limit fit to a range of lags from LAGMIN to '
                                'LAGMAX.  Default is -30.0 to 30.0 seconds. '),
                          default=(defaultmin, defaultmax))


def postprocesssearchrangeopts(args):
    # Additional argument parsing not handled by argparse
    try:
        test = args.lag_extrema_nondefault
        args.lagmin_nondefault = True
        args.lagmax_nondefault = True
    except KeyError:
        pass
    args.lagmin = args.lag_extrema[0]
    args.lagmax = args.lag_extrema[1]
    return args


def postprocessfilteropts(args):
    # configure the filter
    # set the trapezoidal flag, if using
    try:
        thetype = args.filtertype
    except AttributeError:
        args.filtertype = 'trapezoidal'
    try:
        theorder = args.filtorder
    except AttributeError:
        args.filtorder = 6

    if args.filtertype == 'trapezoidal':
        inittrap = True
    else:
        inittrap = False

    # if arbvec is set, we are going set up an arbpass filter
    if args.arbvec is not None:
        if len(args.arbvec) == 2:
            args.arbvec.append(args.arbvec[0] * 0.95)
            args.arbvec.append(args.arbvec[1] * 1.05)
        elif len(args.arbvec) != 4:
            raise ValueError("Argument '--arb' must be either two "
                             "or four floats.")
        # NOTE - this vector is LOWERPASS, UPPERPASS, LOWERSTOP, UPPERSTOP
        # setfreqs expects LOWERSTOP, LOWERPASS, UPPERPASS, UPPERSTOP
        theprefilter = tide_filt.noncausalfilter('arb', usetrapfftfilt=inittrap)
        theprefilter.setfreqs(args.arbvec[2], args.arbvec[0], args.arbve[1], args.arbvec[3])
    else:
        theprefilter = tide_filt.noncausalfilter(args.filterband, usetrapfftfilt=inittrap)

    # make the filter a butterworth if selected
    if args.filtertype == 'butterworth':
        args.usebutterworthfilter = True
    else:
        args.usebutterworthfilter = False
    theprefilter.setbutter(args.usebutterworthfilter, args.filtorder)

    args.lowerstop, args.lowerpass, args.upperpass, args.upperstop = theprefilter.getfreqs()

    return args, theprefilter





