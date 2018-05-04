"""
"""
import os.path as op
import argparse


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


def is_range(parser, arg):
    """
    Check if argument is min/max pair.
    """
    if arg is not None and len(arg) != 2:
        parser.error('Argument must be min/max pair.')
    elif arg is not None and arg[0] > arg[1]:
        parser.error('Argument min must be lower than max.')

    return arg


def get_parser():
    """
    Argument parser for rapidtide
    """
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('datafilename',
                        dest='datafilename',
                        help='The input data file (BOLD fmri file or NIRS)',
                        required=True)
    parser.add_argument('outputname',
                        dest='outputname',
                        help='The root name for the output files',
                        required=True)

    # Macros
    macros = parser.add_argument_group('Macros')
    macros.add_argument('--venousrefine',
                        dest='venousrefine',
                        action='store_true')
    macros.add_argument('--nirs',
                        dest='nirs',
                        action='store_true')

    # Preprocessing options
    preproc = parser.add_argument_group('Preprocessing options')
    preproc.add_argument('-t',
                         dest='realtr',
                         action='store',
                         metavar='TSTEP',
                         type=float,
                         help=('Set the timestep of the data file to TSTEP '
                               '(or 1/FREQ). This will override the TR in an '
                               'fMRI file. NOTE: if using data from a text '
                               'file, for example with NIRS data, using one '
                               'of these options is mandatory.'))
    preproc.add_argument('--datatstep',
                         dest='realtr',
                         action='store',
                         metavar='TSTEP',
                         type=float,
                         help=('Set the timestep of the data file to TSTEP '
                               '(or 1/FREQ). This will override the TR in an '
                               'fMRI file. NOTE: if using data from a text '
                               'file, for example with NIRS data, using one '
                               'of these options is mandatory.'))
    # TODO: Compute 1 / FREQ here
    preproc.add_argument('--datafreq',
                         dest='realtr',
                         action='store',
                         metavar='FREQ',
                         type=float,
                         help=('Set the timestep of the data file to TSTEP '
                               '(or 1/FREQ). This will override the TR in an '
                               'fMRI file. NOTE: if using data from a text '
                               'file, for example with NIRS data, using one '
                               'of these options is mandatory.'))
    preproc.add_argument('-a',
                         dest='antialias',
                         action='store_false',
                         help='Disable antialiasing filter',
                         default=True)
    preproc.add_argument('--nodetrend',
                         dest='dodetrend',
                         action='store_false',
                         help='Disable linear trend removal',
                         default=True)
    preproc.add_argument('-I',
                         dest='invertregressor',
                         action='store_true',
                         help=('Invert the sign of the regressor before '
                               'processing'),
                         default=False)
    preproc.add_argument('-i',
                         dest='',
                         action='store',
                         type=str,
                         choices=['univariate', 'cubic', 'quadratic'],
                         help=("Use specified interpolation type. Options "
                               "are 'cubic','quadratic', and 'univariate' "
                               "(default)."),
                         default='univariate')
    preproc.add_argument('-o',
                         dest='',
                         action='store',
                         type=float,
                         metavar='OFFSETTIME',
                         help='Apply offset OFFSETTIME to the lag regressors',
                         default=None)
    preproc.add_argument('-b',
                         dest='',
                         action='store_true',
                         help=('Use butterworth filter for band splitting '
                               'instead of trapezoidal FFT filter'),
                         default=False)
    preproc.add_argument('-F',
                         dest='',
                         action='store',
                         nargs=2,
                         type=float,
                         metavar='LOWERFREQ UPPERFREQ',
                         help=('Filter data and regressors from LOWERFREQ to '
                               'UPPERFREQ. LOWERFREQ and UPPERFREQ can be '
                               'specified, or will be calculated '
                               'automatically'),
                         default=None)
    preproc.add_argument('-V',
                         dest='',
                         action='store_true',
                         help=('Filter data and regressors to VLF band'),
                         default=False)
    preproc.add_argument('-L',
                         dest='',
                         action='store_true',
                         help=('Filter data and regressors to LFO band'),
                         default=False)
    preproc.add_argument('-R',
                         dest='',
                         action='store_true',
                         help=('Filter data and regressors to respiratory '
                               'band'),
                         default=False)
    preproc.add_argument('-C',
                         dest='',
                         action='store_true',
                         help=('Filter data and regressors to cardiac band'),
                         default=False)
    preproc.add_argument('-N',
                         dest='',
                         action='store',
                         type=int,
                         metavar='NREPS',
                         help=('Estimate significance threshold by running '
                               'NREPS null correlations (default is 10000, '
                               'set to 0 to disable)'),
                         default=10000)
    preproc.add_argument('--skipsighistfit',
                         dest='',
                         action='store_false',
                         help=('Do not fit significance histogram with a '
                               'Johnson SB function'),
                         default=True)
    parser.add_argument('--windowfunc',
                        dest='windowfunc',
                        action='store',
                        type=str,
                        metavar='FUNC',
                        choices=['hamming', 'hann', 'blackmanharris', 'None'],
                        help=('Window funcion to use prior to correlation. '
                              'Options are hamming (default), hann, '
                              'blackmanharris, and None'),
                        default='hamming')
    preproc.add_argument('--nowindow',
                         dest='usewindowfunc',
                         action='store_false',
                         help='Disable precorrelation windowing',
                         default=True)
    preproc.add_argument('-f',
                         dest='',
                         action='store',
                         type=float,
                         metavar='GAUSSSIGMA',
                         help=('Spatially filter fMRI data prior to analysis '
                               'using GAUSSSIGMA in mm'),
                         default=0.)
    preproc.add_argument('-M',
                         dest='',
                         action='store_true',
                         help=('Generate a global mean regressor and use that '
                               'as the reference regressor'),
                         default=False)
    preproc.add_argument('-m',
                         dest='',
                         action='store_true',
                         help=('Mean scale regressors during global mean '
                               'estimation'),
                         default=False)
    preproc.add_argument('--slicetimes',
                         dest='',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Apply offset times from FILE to each slice in '
                               'the dataset'),
                         default=None)
    preproc.add_argument('--numskip',
                         dest='',
                         action='store',
                         type=int,
                         metavar='SKIP',
                         help=('SKIP TRs were previously deleted during '
                               'preprocessing (default is 0)'),
                         default=0)
    preproc.add_argument('--nothresh',
                         dest='',
                         action='store_false',
                         help=('Disable voxel intensity threshold (especially '
                               'useful for NIRS data)'),
                         default=True)

    # Correlation options
    corr = parser.add_argument_group('Correlation options')
    corr.add_argument('-O',
                      dest='',
                      action='store',
                      type=int,
                      metavar='OVERSAMPFAC',
                      help=('Oversample the fMRI data by the following '
                            'integral factor (default is 2)'),
                      default=2)
    corr.add_argument('--regressor',
                      dest='',
                      action='store',
                      type=lambda x: is_valid_file(parser, x),
                      metavar='FILENAME',
                      help=('Read probe regressor from file FILENAME (if none '
                            'specified, generate and use global regressor)'),
                      default=None)

    reg_group = parser.add_mutually_exclusive_group()
    reg_group.add_argument('--regressorfreq',
                           dest='',
                           action='store',
                           type=float,
                           metavar='FREQ',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default=None)
    reg_group.add_argument('--regressortstep',
                           dest='',
                           action='store',
                           type=float,
                           metavar='TSTEP',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default=None)

    corr.add_argument('--regressorstart',
                      dest='',
                      action='store',
                      type=float,
                      metavar='START',
                      help=('The time delay in seconds into the regressor '
                            'file, corresponding in the first TR of the fMRI '
                            'file (default is 0.0)'),
                      default=0.)

    cc_group = corr.add_mutually_exclusive_group()
    cc_group.add_argument('--phat',
                          dest='',
                          action='store_true',
                          help=('Use generalized cross-correlation with phase '
                                'alignment transform (PHAT) instead of '
                                'correlation'),
                          default=False)
    cc_group.add_argument('--liang',
                          dest='',
                          action='store_true',
                          help=('Use generalized cross-correlation with Liang '
                                'weighting function (Liang, et al., '
                                'doi:10.1109/IMCCC.2015.283)'),
                          default=False)
    cc_group.add_argument('--eckart',
                          dest='',
                          action='store_true',
                          help=('Use generalized cross-correlation with '
                                'Eckart weighting function'),
                          default=False)

    mask_group = corr.add_mutually_exclusive_group()
    mask_group.add_argument('--corrmaskthresh',
                            dest='',
                            action='store',
                            type=float,
                            metavar='PCT',
                            help=('Do correlations in voxels where the mean '
                                  'exceeds this percentage of the robust max '
                                  '(default is 1.0)'),
                            default=1.0)
    mask_group.add_argument('--corrmask',
                            dest='',
                            action='store',
                            type=lambda x: is_valid_file(parser, x),
                            metavar='MASK',
                            help=('Only do correlations in voxels in MASK '
                                  '(if set, corrmaskthresh is ignored).'),
                            default=0.)
    corr.add_argument('--accheck',
                      dest='',
                      action='store_true',
                      help=('Check for periodic components that corrupt the '
                            'autocorrelation'),
                      default=False)

    # Correlation fitting options
    corr_fit = parser.add_argument_group('Correlation fitting options')
    corr_fit.add_argument('-Z',
                          dest='',
                          action='store',
                          type=float,
                          metavar='DELAYTIME',
                          help=("Don't fit the delay time - set it to "
                                "DELAYTIME seconds for all voxels"),
                          default=None)
    corr_fit.add_argument('-r',
                          dest='',
                          action='store',
                          nargs=2,
                          type=int,
                          metavar='LAGMIN LAGMAX',
                          help=('Limit fit to a range of lags from LAGMIN to '
                                'LAGMAX'),
                          default=None)
    corr_fit.add_argument('-s',
                          dest='',
                          action='store',
                          type=float,
                          metavar='SIGMALIMIT',
                          help=('Reject lag fits with linewidth wider than '
                                'SIGMALIMIT'),
                          default=None)
    corr_fit.add_argument('-B',
                          dest='',
                          action='store_true',
                          help=('Bipolar mode - match peak correlation '
                                'ignoring sign'),
                          default=False)
    corr_fit.add_argument('--nofitfilt',
                          dest='',
                          action='store_false',
                          help=('Do not zero out peak fit values if fit '
                                'fails'),
                          default=True)
    corr_fit.add_argument('--maxfittype',
                          dest='',
                          action='store',
                          type=str,
                          choices=['gauss', 'quad'],
                          help=("Method for fitting the correlation peak "
                                "(default is 'gauss'). 'quad' uses a "
                                "quadratic fit.  Faster but not as well "
                                "tested"),
                          default='gauss')
    corr_fit.add_argument('--despecklepasses',
                          dest='',
                          action='store',
                          type=int,
                          metavar='PASSES',
                          help=('Detect and refit suspect correlations to '
                                'disambiguate peak locations in PASSES '
                                'passes'),
                          default=0)
    corr_fit.add_argument('--despecklethresh',
                          dest='',
                          action='store',
                          type=int,
                          metavar='VAL',
                          help=('Refit correlation if median discontinuity '
                                'magnitude exceeds VAL (default is 5s)'),
                          default=5)

    # Regressor refinement options
    reg_ref = parser.add_argument_group('Regressor refinement options')
    reg_ref.add_argument('--refineprenorm')
    reg_ref.add_argument('--refineweighting')
    reg_ref.add_argument('--passes')
    reg_ref.add_argument('--includemask')
    reg_ref.add_argument('--excludemask')
    reg_ref.add_argument('--lagminthresh')
    reg_ref.add_argument('--lagmaxthresh')
    reg_ref.add_argument('--ampthresh')
    reg_ref.add_argument('--sigmathresh')
    reg_ref.add_argument('--refineoffset')
    reg_ref.add_argument('--refineupperlag')
    reg_ref.add_argument('--refinelowerlag')
    reg_ref.add_argument('--pca')
    reg_ref.add_argument('--ica')
    reg_ref.add_argument('--weightedavg')
    reg_ref.add_argument('--avg')
    reg_ref.add_argument('--psdfilter')

    # Output options
    output = parser.add_argument_group('Output options')
    output.add_argument('--limitoutput')
    output.add_argument('-T')
    output.add_argument('-h')  # Need to replace with something else
    output.add_argument('--timerange')
    output.add_argument('--glmsourcefile')
    output.add_argument('--noglm')
    output.add_argument('--preservefiltering')

    # Miscellaneous options
    misc = parser.add_argument_group('Miscellaneous options')
    misc.add_argument('--noprogressbar')
    misc.add_argument('--wiener',
                      dest='dodeconv',
                      action='store_true',
                      help=('Do Wiener deconvolution to find voxel transfer '
                            'function'),
                      default=False)
    misc.add_argument('--usesp')
    misc.add_argument('-c',
                      dest='isgrayordinate',
                      action='store_true',
                      help='Data file is a converted CIFTI',
                      default=False)
    misc.add_argument('-S')
    misc.add_argument('-d')
    misc.add_argument('--nonumba')
    misc.add_argument('--nosharedmem')
    misc.add_argument('--memprofile')
    misc.add_argument('--multiproc')  # can probs be dropped if nprocs is used
    misc.add_argument('--nprocs')
    misc.add_argument('--debug')

    # Experimental options (not fully tested, may not work)
    experimental = parser.add_argument_group('Experimental options (not fully '
                                             'tested, may not work)')
    experimental.add_argument('--cleanrefined')
    experimental.add_argument('--dispersioncalc')
    experimental.add_argument('--acfix')
    experimental.add_argument('--tmask')
    experimental.add_argument('-p')
    experimental.add_argument('-P')
    experimental.add_argument('-A', '--AR',
                              dest='armodelorder',
                              action='store',
                              type=int,
                              help='Set AR model order (default is 1)',
                              default=1)
