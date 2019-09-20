#!/usr/bin/env python
#
#   Copyright 2016-2019 Blaise Frederick
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
# $Author: frederic $
# $Date: 2016/07/11 14:50:43 $
# $Id: rapidtide,v 1.161 2016/07/11 14:50:43 frederic Exp $
#
#
#
from __future__ import print_function, division

import argparse
import nibabel as nib

from .parser_funcs import (is_valid_file, invert_float, is_float)


def _get_parser():
    """
    Argument parser for rapidtide2
    """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('in_file',
                        type=lambda x: is_valid_file(parser, x),
                        help='The input data file (BOLD fmri file or NIRS)')
    parser.add_argument('prefix',
                        help='The root name for the output files')

    # Macros
    macros = parser.add_argument_group('Macros').add_mutually_exclusive_group()
    macros.add_argument('--venousrefine',
                        dest='venousrefine',
                        action='store_true',
                        help=('This is a macro that sets --lagminthresh=2.5, '
                              '--lagmaxthresh=6.0, --ampthresh=0.5, and '
                              '--refineupperlag to bias refinement towards '
                              'voxels in the draining vasculature for an '
                              'fMRI scan.'),
                        default=False)
    macros.add_argument('--nirs',
                        dest='nirs',
                        action='store_true',
                        help=('This is a NIRS analysis - this is a macro that '
                              'sets --nothresh, --preservefiltering, '
                              '--refineprenorm=var, --ampthresh=0.7, and '
                              '--lagminthresh=0.1.'),
                        default=False)

    # Preprocessing options
    preproc = parser.add_argument_group('Preprocessing options')
    realtr = preproc.add_mutually_exclusive_group()
    realtr.add_argument('--datatstep',
                        dest='realtr',
                        action='store',
                        metavar='TSTEP',
                        type=lambda x: is_float(parser, x),
                        help=('Set the timestep of the data file to TSTEP. '
                              'This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    realtr.add_argument('--datafreq',
                        dest='realtr',
                        action='store',
                        metavar='FREQ',
                        type=lambda x: invert_float(parser, x),
                        help=('Set the timestep of the data file to 1/FREQ. '
                              'This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    preproc.add_argument('-a',
                         dest='antialias',
                         action='store_false',
                         help='Disable antialiasing filter',
                         default=True)
    preproc.add_argument('--invert',
                         dest='invertregressor',
                         action='store_true',
                         help=('Invert the sign of the regressor before '
                               'processing'),
                         default=False)
    preproc.add_argument('--interptype',
                         dest='interptype',
                         action='store',
                         type=str,
                         choices=['univariate', 'cubic', 'quadratic'],
                         help=("Use specified interpolation type. Options "
                               "are 'cubic','quadratic', and 'univariate' "
                               "(default)."),
                         default='univariate')
    preproc.add_argument('--offsettime',
                         dest='offsettime',
                         action='store',
                         type=float,
                         metavar='OFFSETTIME',
                         help='Apply offset OFFSETTIME to the lag regressors',
                         default=None)
    preproc.add_argument('--butterorder',
                         dest='butterorder',
                         action='store',
                         type=int,
                         metavar='ORDER',
                         help=('Use butterworth filter for band splitting '
                               'instead of trapezoidal FFT filter and set '
                               'filter order to ORDER.'),
                         default=None)

    filttype = preproc.add_mutually_exclusive_group()
    filttype.add_argument('-F', '--arb',
                          dest='arbvec',
                          action='store',
                          nargs='+',
                          type=lambda x: is_float(parser, x),
                          metavar=('LOWERFREQ UPPERFREQ',
                                   'LOWERSTOP UPPERSTOP'),
                          help=('Filter data and regressors from LOWERFREQ to '
                                'UPPERFREQ. LOWERSTOP and UPPERSTOP can also '
                                'be specified, or will be calculated '
                                'automatically'),
                          default=None)
    filttype.add_argument('--filtertype',
                          dest='filtertype',
                          action='store',
                          type=str,
                          choices=['arb', 'vlf', 'lfo', 'resp', 'cardiac'],
                          help=('Filter data and regressors to specific band'),
                          default='arb')
    filttype.add_argument('-V', '--vlf',
                          dest='filtertype',
                          action='store_const',
                          const='vlf',
                          help=('Filter data and regressors to VLF band'),
                          default='arb')
    filttype.add_argument('-L', '--lfo',
                          dest='filtertype',
                          action='store_const',
                          const='lfo',
                          help=('Filter data and regressors to LFO band'),
                          default='arb')
    filttype.add_argument('-R', '--resp',
                          dest='filtertype',
                          action='store_const',
                          const='resp',
                          help=('Filter data and regressors to respiratory '
                                'band'),
                          default='arb')
    filttype.add_argument('-C', '--cardiac',
                          dest='filtertype',
                          action='store_const',
                          const='cardiac',
                          help=('Filter data and regressors to cardiac band'),
                          default='arb')

    preproc.add_argument('-N', '--numnull',
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

    preproc.add_argument('-f', '--spatialfilt',
                         dest='gausssigma',
                         action='store',
                         type=float,
                         metavar='GAUSSSIGMA',
                         help=('Spatially filter fMRI data prior to analysis '
                               'using GAUSSSIGMA in mm'),
                         default=0.)
    preproc.add_argument('-M', '--globalmean',
                         dest='useglobalref',
                         action='store_true',
                         help=('Generate a global mean regressor and use that '
                               'as the reference regressor'),
                         default=False)
    preproc.add_argument('--meanscale',
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
    corr.add_argument('--oversampfac',
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
                      metavar='FILE',
                      help=('Read probe regressor from file FILE (if none '
                            'specified, generate and use global regressor)'),
                      default=None)

    reg_group = corr.add_mutually_exclusive_group()
    reg_group.add_argument('--regressorfreq',
                           dest='inputfreq',
                           action='store',
                           type=lambda x: is_float(parser, x),
                           metavar='FREQ',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default='auto')
    reg_group.add_argument('--regressortstep',
                           dest='inputfreq',
                           action='store',
                           type=lambda x: invert_float(parser, x),
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
    cc_group.add_argument('--corrweighting',
                          dest='corrweighting',
                          action='store',
                          type=str,
                          choices=['none', 'phat', 'liang', 'eckart'],
                          help=('Method to use for cross-correlation '
                                'weighting.'),
                          default='none')
    cc_group.add_argument('--nodetrend',
                          dest='dodetrend',
                          action='store_false',
                          help='Disable linear trend removal',
                          default=True)

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
                            metavar='FILE',
                            help=('Only do correlations in voxels in FILE '
                                  '(if set, corrmaskthresh is ignored).'),
                            default=None)

    # Correlation fitting options
    corr_fit = parser.add_argument_group('Correlation fitting options')

    fixdelay = corr_fit.add_mutually_exclusive_group()
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

    corr_fit.add_argument('--sigmalimit',
                          dest='widthlimit',
                          action='store',
                          type=float,
                          metavar='SIGMALIMIT',
                          help=('Reject lag fits with linewidth wider than '
                                'SIGMALIMIT Hz'),
                          default=100.0)
    corr_fit.add_argument('--bipolar',
                          dest='bipolar',
                          action='store_true',
                          help=('Bipolar mode - match peak correlation '
                                'ignoring sign'),
                          default=False)
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
    corr_fit.add_argument('--despecklepasses',
                          dest='despeckle_passes',
                          action='store',
                          type=int,
                          metavar='PASSES',
                          help=('Detect and refit suspect correlations to '
                                'disambiguate peak locations in PASSES '
                                'passes'),
                          default=0)
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
    reg_ref.add_argument('--lagminthresh',
                         dest='lagminthresh',
                         action='store',
                         metavar='MIN',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'less than MIN (default is 0.5s)'),
                         default=0.5)
    reg_ref.add_argument('--lagmaxthresh',
                         dest='lagmaxthresh',
                         action='store',
                         metavar='MAX',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'greater than MAX (default is 5s)'),
                         default=5.0)
    reg_ref.add_argument('--ampthresh',
                         dest='ampthresh',
                         action='store',
                         metavar='AMP',
                         type=float,
                         help=('or refinement, exclude voxels with '
                               'correlation coefficients less than AMP '
                               '(default is 0.3)'),
                         default=0.3)
    reg_ref.add_argument('--sigmathresh',
                         dest='sigmathresh',
                         action='store',
                         metavar='SIGMA',
                         type=float,
                         help=('For refinement, exclude voxels with widths '
                               'greater than SIGMA (default is 100s)'),
                         default=100.0)
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
    reg_ref.add_argument('--refinetype',
                         dest='refinetype',
                         action='store',
                         type=str,
                         choices=['avg', 'pca', 'ica', 'weightedavg'],
                         help=('Method with which to derive refined '
                               'regressor.'),
                         default='avg')

    # Output options
    output = parser.add_argument_group('Output options')
    output.add_argument('--limitoutput',
                        dest='savelagregressors',
                        action='store_false',
                        help=("Don't save some of the large and rarely used "
                              "files"),
                        default=True)
    output.add_argument('--savelags',
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
    misc.add_argument('--cifti',
                      dest='isgrayordinate',
                      action='store_true',
                      help='Data file is a converted CIFTI',
                      default=False)
    misc.add_argument('--simulate',
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
    misc.add_argument('--nprocs',
                      dest='nprocs',
                      action='store',
                      type=int,
                      metavar='NPROCS',
                      help=('Use NPROCS worker processes for multiprocessing. '
                            'Setting NPROCS to less than 1 sets the number of '
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
    experimental.add_argument('--acfix',
                              dest='fix_autocorrelation',
                              action='store_true',
                              help=('Perform a secondary correlation to '
                                    'disambiguate peak location (enables '
                                    '--accheck). Experimental.'),
                              default=False)
    experimental.add_argument('--tmask',
                              dest='tmaskname',
                              action='store',
                              type=lambda x: is_valid_file(parser, x),
                              metavar='FILE',
                              help=('Only correlate during epochs specified '
                                    'in MASKFILE (NB: each line of FILE '
                                    'contains the time and duration of an '
                                    'epoch to include'),
                              default=None)
    exp_group = experimental.add_mutually_exclusive_group()
    exp_group.add_argument('--prewhiten',
                           dest='doprewhiten',
                           action='store_true',
                           help='Prewhiten and refit data',
                           default=False)
    exp_group.add_argument('--saveprewhiten',
                           dest='saveprewhiten',
                           action='store_true',
                           help=('Save prewhitened data (turns prewhitening '
                                 'on)'),
                           default=False)
    experimental.add_argument('--AR',
                              dest='armodelorder',
                              action='store',
                              type=int,
                              help='Set AR model order (default is 1)',
                              default=1)
    return parser


def rapidtide_workflow(in_file, prefix, venousrefine=False, nirs=False,
                       realtr='auto', antialias=True, invertregressor=False,
                       interptype='univariate', offsettime=None,
                       butterorder=None, arbvec=None, filtertype='arb',
                       numestreps=10000, dosighistfit=True,
                       windowfunc='hamming', gausssigma=0.,
                       useglobalref=False, meanscaleglobal=False,
                       slicetimes=None, preprocskip=0, nothresh=True,
                       oversampfactor=2, regressorfile=None, inputfreq=1.,
                       inputstarttime=0., corrweighting='none',
                       dodetrend=True, corrmaskthreshpct=1.,
                       corrmaskname=None, fixeddelayvalue=None,
                       lag_extrema=(-30.0, 30.0), widthlimit=100.,
                       bipolar=False, zerooutbadfit=True, findmaxtype='gauss',
                       despeckle_passes=0, despeckle_thresh=5,
                       refineprenorm='mean', refineweighting='R2', passes=1,
                       includemaskname=None, excludemaskname=None,
                       lagminthresh=0.5, lagmaxthresh=5., ampthresh=0.3,
                       sigmathresh=100., refineoffset=False, psdfilter=False,
                       lagmaskside='both', refinetype='avg',
                       savelagregressors=True, savecorrtimes=False,
                       histlen=100, timerange=(-1, 10000000),
                       glmsourcefile=None, doglmfilt=True,
                       preservefiltering=False, showprogressbar=True,
                       dodeconv=False, internalprecision='double',
                       isgrayordinate=False, fakerun=False, displayplots=False,
                       nonumba=False, sharedmem=True, memprofile=False,
                       nprocs=1, debug=False, cleanrefined=False,
                       dodispersioncalc=False, fix_autocorrelation=False,
                       tmaskname=None, doprewhiten=False, saveprewhiten=False,
                       armodelorder=1, offsettime_total=None,
                       ampthreshfromsig=False, nohistzero=False,
                       fixdelay=False, usebutterworthfilter=False):
    """
    Run the full rapidtide workflow.
    """
    pass


def _main(argv=None):
    """
    Compile arguments for rapidtide workflow.
    """
    args = vars(_get_parser().parse_args(argv))

    # Additional argument parsing not handled by argparse
    if args['arbvec'] is not None:
        if len(args['arbvec']) == 2:
            args['arbvec'].append(args['arbvec'][0] * 0.9)
            args['arbvec'].append(args['arbvec'][1] * 1.1)
        elif len(args['arbvec']) != 4:
            raise ValueError("Argument '--arb' (or '-F') must be either two "
                             "or four floats.")

    if args['offsettime'] is not None:
        args['offsettime_total'] = -1 * args['offsettime']
    else:
        args['offsettime_total'] = None

    if args['saveprewhiten'] is True:
        args['doprewhiten'] = True

    if args['tmaskname'] is not None:
        args['usetmask'] = True

    reg_ref_used = ((args['lagminthresh'] != 0.5) or
                    (args['lagmaxthresh'] != 5.) or
                    (args['ampthresh'] != 0.3) or
                    (args['sigmathresh'] != 100.) or
                    (args['refineoffset']))
    if reg_ref_used and args['passes'] == 1:
        args['passes'] = 2

    if args['ampthresh'] != 100.:
        args['ampthreshfromsig'] = False
    else:
        args['ampthreshfromsig'] = True

    if args['despeckle_thresh'] != 5 and args['despeckle_passes'] == 0:
        args['despeckle_passes'] = 1

    if args['zerooutbadfit']:
        args['nohistzero'] = False
    else:
        args['nohistzero'] = True

    if args['fixeddelayvalue'] is not None:
        args['fixdelay'] = True
        args['lag_extrema'] = (args['fixeddelayvalue'] - 10.0,
                               args['fixeddelayvalue'] + 10.0)
    else:
        args['fixdelay'] = False

    if args['in_file'].endswith('txt') and args['realtr'] == 'auto':
        raise ValueError('Either --datatstep or --datafreq must be provided '
                         'if data file is a text file.')

    if args['realtr'] != 'auto':
        fmri_tr = args['realtr']
    else:
        fmri_tr = nib.load(args['in_file']).header.get_zooms()[3]

    if args['inputfreq'] == 'auto':
        args['inputfreq'] = 1. / fmri_tr

    args['usebutterworthfilter'] = bool(args['butterorder'])

    if args['venousrefine']:
        print('WARNING: Using "venousrefine" macro. Overriding any affected '
              'arguments.')
        args['lagminthresh'] = 2.5
        args['lagmaxthresh'] = 6.
        args['ampthresh'] = 0.5
        args['lagmaskside'] = 'upper'

    if args['nirs']:
        print('WARNING: Using "nirs" macro. Overriding any affected '
              'arguments.')
        args['nothresh'] = False
        args['preservefiltering'] = True
        args['refineprenorm'] = 'var'
        args['ampthresh'] = 0.7
        args['lagmaskthresh'] = 0.1

    rapidtide_workflow(**args)


if __name__ == '__main__':
    _main()
