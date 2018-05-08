"""
Arguments to maybe drop:
- --multiproc (redundant with nprocs)
- --datatstep or -t (redundant with one another)
- -i (appears unused)
- check_autocorrelation (not an argument, just a variable)
- shiftall (not an argument, just a variable)
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


def freq_to_step(parser, arg):
    """
    Check if argument is float or auto.
    """
    if not isinstance(arg, float) and arg != 'auto':
        parser.error('Value {0} is not a float or "auto"'.format(arg))

    if arg != 'auto':
        arg = 1. / arg
    return arg


def is_two_or_four_floats(parser, arg):
    if len(arg) not in (2, 4) or not all([isinstance(a, float) for a in arg]):
        parser.error('Value {0} must be tuple of two or four '
                     'integers'.format(arg))
    return arg


def is_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    if not isinstance(arg, float) and arg != 'auto':
        parser.error('Value {0} is not a float or "auto"'.format(arg))

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
                        type=lambda x: is_valid_file(parser, x),
                        help='The input data file (BOLD fmri file or NIRS)')
    parser.add_argument('outputname',
                        help='The root name for the output files')

    # Macros
    # TODO: Do macros
    macros = parser.add_argument_group('Macros')
    macros.add_argument('--venousrefine',
                        dest='venousrefine',
                        action='store_true')
    macros.add_argument('--nirs',
                        dest='nirs',
                        action='store_true')

    # Preprocessing options
    preproc = parser.add_argument_group('Preprocessing options')
    realtr = preproc.add_mutually_exclusive_group()
    # NOTE: Are both of the following necessary?
    realtr.add_argument('-t',
                        dest='realtr',
                        action='store',
                        metavar='TSTEP',
                        type=lambda x: is_float(parser, x),
                        help=('Set the timestep of the data file to TSTEP '
                              '(or 1/FREQ). This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    realtr.add_argument('--datatstep',
                        dest='realtr',
                        action='store',
                        metavar='TSTEP',
                        type=lambda x: is_float(parser, x),
                        help=('Set the timestep of the data file to TSTEP '
                              '(or 1/FREQ). This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    realtr.add_argument('--datafreq',
                        dest='realtr',
                        action='store',
                        metavar='FREQ',
                        type=lambda x: freq_to_step(parser, x),
                        help=('Set the timestep of the data file to TSTEP '
                              '(or 1/FREQ). This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    preproc.add_argument('-a',
                         dest='antialias',
                         action='store_false',
                         help='Disable antialiasing filter',
                         default=True)
    preproc.add_argument('-I',
                         dest='invertregressor',
                         action='store_true',
                         help=('Invert the sign of the regressor before '
                               'processing'),
                         default=False)
    # TODO: THIS APPEARS UNUSED
    preproc.add_argument('-i',
                         dest='interptype',
                         action='store',
                         type=str,
                         choices=['univariate', 'cubic', 'quadratic'],
                         help=("Use specified interpolation type. Options "
                               "are 'cubic','quadratic', and 'univariate' "
                               "(default)."),
                         default='univariate')
    # TODO: Set offsettime_total to negative offsettime
    preproc.add_argument('-o',
                         dest='offsettime',
                         action='store',
                         type=float,
                         metavar='OFFSETTIME',
                         help='Apply offset OFFSETTIME to the lag regressors',
                         default=None)
    preproc.add_argument('-b',
                         dest='usebutterworthfilter',
                         action='store_true',
                         help=('Use butterworth filter for band splitting '
                               'instead of trapezoidal FFT filter'),
                         default=False)

    filttype = preproc.add_mutually_exclusive_group()
    filttype.add_argument('-F',
                          dest='arbvec',
                          action='store',
                          nargs='*',
                          type=lambda x: is_two_or_four_floats(parser, x),
                          metavar='LOWERFREQ UPPERFREQ [LOWERSTOP UPPERSTOP]',
                          help=('Filter data and regressors from LOWERFREQ to '
                                'UPPERFREQ. LOWERSTOP and UPPERSTOP can also '
                                'be specified, or will be calculated '
                                'automatically'),
                          default=None)
    filttype.add_argument('-V',
                          dest='filtertype',
                          action='store_const',
                          const='vlf',
                          help=('Filter data and regressors to VLF band'),
                          default='arb')
    filttype.add_argument('-L',
                          dest='filtertype',
                          action='store_const',
                          const='lfo',
                          help=('Filter data and regressors to LFO band'),
                          default='arb')
    filttype.add_argument('-R',
                          dest='filtertype',
                          action='store_const',
                          const='resp',
                          help=('Filter data and regressors to respiratory '
                                'band'),
                          default='arb')
    filttype.add_argument('-C',
                          dest='filtertype',
                          action='store_const',
                          const='cardiac',
                          help=('Filter data and regressors to cardiac band'),
                          default='arb')

    preproc.add_argument('-N',
                         dest='numestreps',
                         action='store',
                         type=int,
                         metavar='NREPS',
                         help=('Estimate significance threshold by running '
                               'NREPS null correlations (default is 10000, '
                               'set to 0 to disable)'),
                         default=10000)
    preproc.add_argument('--skipsighistfit',
                         dest='dosighistfit',
                         action='store_false',
                         help=('Do not fit significance histogram with a '
                               'Johnson SB function'),
                         default=True)

    wfunc = preproc.add_mutually_exclusive_group()
    wfunc.add_argument('--windowfunc',
                       dest='windowfunc',
                       action='store',
                       type=str,
                       choices=['hamming', 'hann', 'blackmanharris', 'None'],
                       help=('Window funcion to use prior to correlation. '
                             'Options are hamming (default), hann, '
                             'blackmanharris, and None'),
                       default='hamming')
    wfunc.add_argument('--nowindow',
                       dest='windowfunc',
                       action='store_const',
                       const='None',
                       help='Disable precorrelation windowing',
                       default='hamming')

    preproc.add_argument('-f',
                         dest='gausssigma',
                         action='store',
                         type=float,
                         metavar='GAUSSSIGMA',
                         help=('Spatially filter fMRI data prior to analysis '
                               'using GAUSSSIGMA in mm'),
                         default=0.)
    preproc.add_argument('-M',
                         dest='useglobalref',
                         action='store_true',
                         help=('Generate a global mean regressor and use that '
                               'as the reference regressor'),
                         default=False)
    preproc.add_argument('-m',
                         dest='meanscaleglobal',
                         action='store_true',
                         help=('Mean scale regressors during global mean '
                               'estimation'),
                         default=False)
    preproc.add_argument('--slicetimes',
                         dest='slicetimes',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Apply offset times from FILE to each slice in '
                               'the dataset'),
                         default=None)
    preproc.add_argument('--numskip',
                         dest='preprocskip',
                         action='store',
                         type=int,
                         metavar='SKIP',
                         help=('SKIP TRs were previously deleted during '
                               'preprocessing (default is 0)'),
                         default=0)
    preproc.add_argument('--nothresh',
                         dest='nothresh',
                         action='store_false',
                         help=('Disable voxel intensity threshold (especially '
                               'useful for NIRS data)'),
                         default=True)

    # Correlation options
    corr = parser.add_argument_group('Correlation options')
    corr.add_argument('-O',
                      dest='oversampfactor',
                      action='store',
                      type=int,
                      metavar='OVERSAMPFAC',
                      help=('Oversample the fMRI data by the following '
                            'integral factor (default is 2)'),
                      default=2)
    corr.add_argument('--regressor',
                      dest='regressorfile',
                      action='store',
                      type=lambda x: is_valid_file(parser, x),
                      metavar='FILENAME',
                      help=('Read probe regressor from file FILENAME (if none '
                            'specified, generate and use global regressor)'),
                      default=None)

    reg_group = corr.add_mutually_exclusive_group()
    # TODO: Calculate default with TR
    reg_group.add_argument('--regressorfreq',
                           dest='inputfreq',
                           action='store',
                           type=lambda x: freq_to_step(parser, x),
                           metavar='FREQ',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default='auto')
    reg_group.add_argument('--regressortstep',
                           dest='inputstep',
                           action='store',
                           type=lambda x: is_float(parser, x),
                           metavar='TSTEP',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default='auto')

    corr.add_argument('--regressorstart',
                      dest='inputstarttime',
                      action='store',
                      type=float,
                      metavar='START',
                      help=('The time delay in seconds into the regressor '
                            'file, corresponding in the first TR of the fMRI '
                            'file (default is 0.0)'),
                      default=0.)

    cc_group = corr.add_mutually_exclusive_group()
    cc_group.add_argument('--nodetrend',
                          dest='dodetrend',
                          action='store_false',
                          help='Disable linear trend removal',
                          default=True)
    cc_group.add_argument('--phat',
                          dest='corrweighting',
                          action='store_const',
                          const='phat',
                          help=('Use generalized cross-correlation with phase '
                                'alignment transform (PHAT) instead of '
                                'correlation'),
                          default='eckart')
    cc_group.add_argument('--liang',
                          dest='corrweighting',
                          action='store_const',
                          const='liang',
                          help=('Use generalized cross-correlation with Liang '
                                'weighting function (Liang, et al., '
                                'doi:10.1109/IMCCC.2015.283)'),
                          default='none')
    cc_group.add_argument('--eckart',
                          dest='corrweighting',
                          action='store_const',
                          const='eckart',
                          help=('Use generalized cross-correlation with '
                                'Eckart weighting function'),
                          default='none')

    mask_group = corr.add_mutually_exclusive_group()
    mask_group.add_argument('--corrmaskthresh',
                            dest='corrmaskthreshpct',
                            action='store',
                            type=float,
                            metavar='PCT',
                            help=('Do correlations in voxels where the mean '
                                  'exceeds this percentage of the robust max '
                                  '(default is 1.0)'),
                            default=1.0)
    mask_group.add_argument('--corrmask',
                            dest='corrmaskname',
                            action='store',
                            type=lambda x: is_valid_file(parser, x),
                            metavar='MASK',
                            help=('Only do correlations in voxels in MASK '
                                  '(if set, corrmaskthresh is ignored).'),
                            default=None)

    corr.add_argument('--accheck',
                      dest='check_autocorrelation',
                      action='store_true',
                      help=('Check for periodic components that corrupt the '
                            'autocorrelation'),
                      default=False)

    # Correlation fitting options
    corr_fit = parser.add_argument_group('Correlation fitting options')

    fixdelay = corr_fit.add_mutually_exclusive_group()
    # TODO: Also adjust fixdelay, lagmin, and lagmax
    fixdelay.add_argument('-Z',
                          dest='fixeddelayvalue',
                          action='store',
                          type=float,
                          metavar='DELAYTIME',
                          help=("Don't fit the delay time - set it to "
                                "DELAYTIME seconds for all voxels"),
                          default=None)
    # TODO: Split into lagmin and lagmax
    fixdelay.add_argument('-r',
                          dest='lag_extrema',
                          action='store',
                          nargs=2,
                          type=int,
                          metavar=('LAGMIN', 'LAGMAX'),
                          help=('Limit fit to a range of lags from LAGMIN to '
                                'LAGMAX'),
                          default=(-30.0, 30.0))

    corr_fit.add_argument('-s',
                          dest='widthlimit',
                          action='store',
                          type=float,
                          metavar='SIGMALIMIT',
                          help=('Reject lag fits with linewidth wider than '
                                'SIGMALIMIT Hz'),
                          default=100.0)
    corr_fit.add_argument('-B',
                          dest='bipolar',
                          action='store_true',
                          help=('Bipolar mode - match peak correlation '
                                'ignoring sign'),
                          default=False)
    # TODO: Also set nohistzero to True (default is False)
    corr_fit.add_argument('--nofitfilt',
                          dest='zerooutbadfit',
                          action='store_false',
                          help=('Do not zero out peak fit values if fit '
                                'fails'),
                          default=True)
    corr_fit.add_argument('--maxfittype',
                          dest='findmaxtype',
                          action='store',
                          type=str,
                          choices=['gauss', 'quad'],
                          help=("Method for fitting the correlation peak "
                                "(default is 'gauss'). 'quad' uses a "
                                "quadratic fit.  Faster but not as well "
                                "tested"),
                          default='gauss')
    # TODO: Also change check_autocorrelation to True
    # NOTE: However, there's no way to set check_autocorrelation to False, and
    # True is the default
    corr_fit.add_argument('--despecklepasses',
                          dest='despeckle_passes',
                          action='store',
                          type=int,
                          metavar='PASSES',
                          help=('Detect and refit suspect correlations to '
                                'disambiguate peak locations in PASSES '
                                'passes'),
                          default=0)
    # TODO: Also set despeckle_passes to 1
    corr_fit.add_argument('--despecklethresh',
                          dest='despeckle_thresh',
                          action='store',
                          type=int,
                          metavar='VAL',
                          help=('Refit correlation if median discontinuity '
                                'magnitude exceeds VAL (default is 5s)'),
                          default=5)

    # Regressor refinement options
    reg_ref = parser.add_argument_group('Regressor refinement options')
    reg_ref.add_argument('--refineprenorm',
                         dest='refineprenorm',
                         action='store',
                         type=str,
                         choices=['None', 'mean', 'var', 'std', 'invlag'],
                         help=("Apply TYPE prenormalization to each "
                               "timecourse prior to refinement. Valid "
                               "weightings are 'None', 'mean' (default), "
                               "'var', and 'std'"),
                         default='mean')
    reg_ref.add_argument('--refineweighting',
                         dest='refineweighting',
                         action='store',
                         type=str,
                         choices=['None', 'NIRS', 'R', 'R2'],
                         help=("Apply TYPE weighting to each timecourse prior "
                               "to refinement. Valid weightings are "
                               "'None', 'NIRS', 'R', and 'R2' (default)"),
                         default='R2')
    reg_ref.add_argument('--passes',
                         dest='passes',
                         action='store',
                         type=int,
                         metavar='PASSES',
                         help=('Set the number of processing passes to '
                               'PASSES'),
                         default=1)
    reg_ref.add_argument('--includemask',
                         dest='includemaskname',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Only use voxels in NAME for global regressor '
                               'generation and regressor refinement'),
                         default=None)
    reg_ref.add_argument('--excludemask',
                         dest='excludemaskname',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Do not use voxels in NAME for global '
                               'regressor generation and regressor '
                               'refinement'),
                         default=None)
    # TODO: Also set passes to 2
    reg_ref.add_argument('--lagminthresh',
                         dest='lagminthresh',
                         action='store',
                         metavar='MIN',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'less than MIN (default is 0.5s)'),
                         default=0.5)
    # TODO: Also set passes to 2
    reg_ref.add_argument('--lagmaxthresh',
                         dest='lagmaxthresh',
                         action='store',
                         metavar='MAX',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'greater than MAX (default is 5s)'),
                         default=5.0)
    # TODO: Also set passes to 2
    # TODO: Also set ampthreshfromsig to False
    reg_ref.add_argument('--ampthresh',
                         dest='ampthresh',
                         action='store',
                         metavar='AMP',
                         type=float,
                         help=('or refinement, exclude voxels with '
                               'correlation coefficients less than AMP '
                               '(default is 0.3)'),
                         default=0.3)
    # TODO: Also set passes to 2
    reg_ref.add_argument('--sigmathresh',
                         dest='sigmathresh',
                         action='store',
                         metavar='SIGMA',
                         type=float,
                         help=('For refinement, exclude voxels with widths '
                               'greater than SIGMA (default is 100s)'),
                         default=100.0)
    # TODO: Also set passes to 2
    reg_ref.add_argument('--refineoffset',
                         dest='refineoffset',
                         action='store_true',
                         help=('Bipolar mode - match peak correlation '
                               'ignoring sign'),
                         default=False)
    reg_ref.add_argument('--psdfilter',
                         dest='psdfilter',
                         action='store_true',
                         help=('Apply a PSD weighted Wiener filter to '
                               'shifted timecourses prior to refinement'),
                         default=False)

    refine = reg_ref.add_mutually_exclusive_group()
    refine.add_argument('--refineupperlag',
                        dest='lagmaskside',
                        action='store_const',
                        const='upper',
                        help=('Only use positive lags for regressor '
                              'refinement'),
                        default='both')
    refine.add_argument('--refinelowerlag',
                        dest='lagmaskside',
                        action='store_const',
                        const='lower',
                        help=('Only use negative lags for regressor '
                              'refinement'),
                        default='both')
    rtype = reg_ref.add_mutually_exclusive_group()
    rtype.add_argument('--pca',
                       dest='refinetype',
                       action='store_const',
                       const='pca',
                       help=('Use PCA to derive refined regressor '
                             '(default is unweighted averaging)'),
                       default='unweighted_average')
    rtype.add_argument('--ica',
                       dest='refinetype',
                       action='store_const',
                       const='ica',
                       help=('Use ICA to derive refined regressor '
                             '(default is unweighted averaging)'),
                       default='unweighted_average')
    rtype.add_argument('--weightedavg',
                       dest='refinetype',
                       action='store_const',
                       const='weighted_average',
                       help=('Use weighted average to derive refined '
                             'regressor (default is unweighted averaging)'),
                       default='unweighted_average')
    rtype.add_argument('--avg',
                       dest='refinetype',
                       action='store_const',
                       const='unweighted_average',
                       help=('Use unweighted average to derive refined '
                             'regressor (default)'),
                       default='unweighted_average')

    # Output options
    output = parser.add_argument_group('Output options')
    output.add_argument('--limitoutput',
                        dest='savelagregressors',
                        action='store_false',
                        help=("Don't save some of the large and rarely used "
                              "files"),
                        default=True)
    output.add_argument('-T',
                        dest='savecorrtimes',
                        action='store_true',
                        help='Save a table of lagtimes used',
                        default=False)
    output.add_argument('--histlen',  # was -h
                        dest='histlen',
                        action='store',
                        type=int,
                        metavar='HISTLEN',
                        help=('Change the histogram length to HISTLEN '
                              '(default is 100)'),
                        default=100)
    # TODO: Split timerange into startpoint and endpoint
    output.add_argument('--timerange',
                        dest='timerange',
                        action='store',
                        nargs=2,
                        type=int,
                        metavar=('START', 'END'),
                        help=('Limit analysis to data between timepoints '
                              'START and END in the fmri file'),
                        default=(-1, 10000000))
    output.add_argument('--glmsourcefile',
                        dest='glmsourcefile',
                        action='store',
                        type=lambda x: is_valid_file(parser, x),
                        metavar='FILE',
                        help=('Regress delayed regressors out of FILE instead '
                              'of the initial fmri file used to estimate '
                              'delays'),
                        default=None)
    output.add_argument('--noglm',
                        dest='doglmfilt',
                        action='store_false',
                        help=('Turn off GLM filtering to remove delayed '
                              'regressor from each voxel (disables output of '
                              'fitNorm)'),
                        default=True)
    output.add_argument('--preservefiltering',
                        dest='preservefiltering',
                        action='store_true',
                        help="Don't reread data prior to GLM",
                        default=False)

    # Miscellaneous options
    misc = parser.add_argument_group('Miscellaneous options')
    misc.add_argument('--noprogressbar',
                      dest='showprogressbar',
                      action='store_false',
                      help='Will disable progress bars',
                      default=True)
    misc.add_argument('--wiener',
                      dest='dodeconv',
                      action='store_true',
                      help=('Do Wiener deconvolution to find voxel transfer '
                            'function'),
                      default=False)
    misc.add_argument('--usesp',
                      dest='internalprecision',
                      action='store_const',
                      const='single',
                      help=('Use single precision for internal calculations '
                            '(may be useful when RAM is limited)'),
                      default='double')
    misc.add_argument('-c',
                      dest='isgrayordinate',
                      action='store_true',
                      help='Data file is a converted CIFTI',
                      default=False)
    misc.add_argument('-S',
                      dest='fakerun',
                      action='store_true',
                      help='Simulate a run - just report command line options',
                      default=False)
    misc.add_argument('-d',
                      dest='displayplots',
                      action='store_true',
                      help='Display plots of interesting timecourses',
                      default=False)
    misc.add_argument('--nonumba',
                      dest='nonumba',
                      action='store_true',
                      help='Disable jit compilation with numba',
                      default=False)
    misc.add_argument('--nosharedmem',
                      dest='sharedmem',
                      action='store_false',
                      help=('Disable use of shared memory for large array '
                            'storage'),
                      default=True)
    misc.add_argument('--memprofile',
                      dest='memprofile',
                      action='store_true',
                      help=('Enable memory profiling for debugging - '
                            'warning: this slows things down a lot.'),
                      default=False)
    misc.add_argument('--multiproc',  # can probs be dropped if nprocs is used
                      dest='nprocs',
                      action='store_const',
                      const=-1,
                      help=('Enable multiprocessing versions of key '
                            'subroutines. This speeds things up dramatically. '
                            'Almost certainly will NOT work on Windows (due '
                            'to different forking behavior).'),
                      default=1)
    misc.add_argument('--nprocs',
                      dest='nprocs',
                      action='store',
                      type=int,
                      metavar='NPROCS',
                      help=('Use NPROCS worker processes for multiprocessing. '
                            'Setting NPROCS less than 1 sets the number of '
                            'worker processes to n_cpus - 1.'),
                      default=1)
    # TODO: Also set theprefilter.setdebug(True)
    misc.add_argument('--debug',
                      dest='debug',
                      action='store_true',
                      help=('Enable additional information output'),
                      default=False)

    # Experimental options (not fully tested, may not work)
    experimental = parser.add_argument_group('Experimental options (not fully '
                                             'tested, may not work)')
    # TODO: Also set shiftall to True, although shiftall is set to True anyway
    experimental.add_argument('--cleanrefined',
                              dest='cleanrefined',
                              action='store_true',
                              help=('Perform additional processing on refined '
                                    'regressor to remove spurious '
                                    'components.'),
                              default=False)
    experimental.add_argument('--dispersioncalc',
                              dest='dodispersioncalc',
                              action='store_true',
                              help=('Generate extra data during refinement to '
                                    'allow calculation of dispersion.'),
                              default=False)
    # TODO: Also set check_autocorrelation to True, although it's already True
    experimental.add_argument('--acfix',
                              dest='fix_autocorrelation',
                              action='store_true',
                              help=('Perform a secondary correlation to '
                                    'disambiguate peak location (enables '
                                    '--accheck). Experimental.'),
                              default=False)
    # TODO: Also set usetmask to True
    experimental.add_argument('--tmask',
                              dest='tmaskname',
                              action='store',
                              type=lambda x: is_valid_file(parser, x),
                              metavar='MASKFILE',
                              help=('Only correlate during epochs specified '
                                    'in MASKFILE (NB: each line of MASKFILE '
                                    'contains the time and duration of an '
                                    'epoch to include'),
                              default=None)
    exp_group = experimental.add_mutually_exclusive_group()
    exp_group.add_argument('-p',
                           dest='doprewhiten',
                           action='store_true',
                           help='Prewhiten and refit data',
                           default=False)
    # TODO: Also change doprewhiten to True
    exp_group.add_argument('-P',
                           dest='saveprewhiten',
                           action='store_true',
                           help=('Save prewhitened data (turns prewhitening '
                                 'on)'),
                           default=False)
    experimental.add_argument('-A', '--AR',
                              dest='armodelorder',
                              action='store',
                              type=int,
                              help='Set AR model order (default is 1)',
                              default=1)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    print(args)


if __name__ == '__main__':
    main()
