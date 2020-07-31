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
import time
import multiprocessing as mp
import platform
import warnings
import sys

import numpy as np
from scipy import ndimage

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util

import rapidtide.nullcorrpassx as tide_nullcorr
import rapidtide.corrpassx as tide_corrpass
import rapidtide.corrfitx as tide_corrfit
import rapidtide.refine as tide_refine
import rapidtide.glmpass as tide_glmpass
import rapidtide.helper_classes as tide_classes
import rapidtide.wiener as tide_wiener

import nibabel as nib

import copy

from .parser_funcs import (is_valid_file, invert_float, is_float)

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False


def conditionalprofile():
    def resdec(f):
        if memprofilerexists:
            return profile(f)
        return f

    return resdec


global rt_floatset, rt_floattype


@conditionalprofile()
def memcheckpoint(message):
    print(message)


def maketmask(filename, timeaxis, maskvector, debug=False):
    inputdata = tide_io.readvecs(filename)
    theshape = np.shape(inputdata)
    if theshape[0] == 1:
        # this is simply a vector, one per TR.  If the value is nonzero, include the point, otherwise don't
        if theshape[1] == len(timeaxis):
            maskvector = np.where(inputdata[0, :] > 0.0, 1.0, 0.0)
        else:
            print('tmask length does not match fmri data')
            sys.exit(1)
    else:
        maskvector *= 0.0
        for idx in range(0, theshape[1]):
            starttime = inputdata[0, idx]
            endtime = starttime + inputdata[1, idx]
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(maskvector) - 1))
            maskvector[startindex:endindex] = 1.0
            print(starttime, startindex, endtime, endindex)
    if debug:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.set_title('temporal mask vector')
        plot(timeaxis, maskvector)
        show()
    return maskvector


class timerangeAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(timerangeAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        setattr(namespace, self.dest, values)

def processmaskspec(maskspec, spectext1, spectext2):
    thename, colspec = tide_io.parsefilespec(maskspec)
    if colspec is not None:
        thevals = tide_io.colspectolist(colspec)
    else:
        thevals = None
    if thevals is not None:
        print(spectext1,
              thename,
              ' = ',
              thevals,
              spectext2)
    return thename, thevals


def addmemprofiling(thefunc, memprofile, memfile, themessage):
    if memprofile:
        return profile(thefunc, precision=2)
    else:
        tide_util.logmem(themessage, file=memfile)
        return thefunc


def numpy2shared(inarray, thetype):
    thesize = inarray.size
    theshape = inarray.shape
    if thetype == np.float64:
        inarray_shared = mp.RawArray('d', inarray.reshape(thesize))
    else:
        inarray_shared = mp.RawArray('f', inarray.reshape(thesize))
    inarray = np.frombuffer(inarray_shared, dtype=thetype, count=thesize)
    inarray.shape = theshape
    return inarray, inarray_shared, theshape


def allocshared(theshape, thetype):
    thesize = int(1)
    for element in theshape:
        thesize *= int(element)
    if thetype == np.float64:
        outarray_shared = mp.RawArray('d', thesize)
    else:
        outarray_shared = mp.RawArray('f', thesize)
    outarray = np.frombuffer(outarray_shared, dtype=thetype, count=thesize)
    outarray.shape = theshape
    return outarray, outarray_shared, theshape


def readamask(maskfilename, nim_hdr, xsize, istext=False, valslist=None, maskname='the', verbose=False):
    if verbose:
        print('readamask called with filename:', maskfilename, 'vals:', valslist)
    if istext:
        maskarray = tide_io.readvecs(maskfilename).astype('int16')
        theshape = np.shape(maskarray)
        theincludexsize = theshape[0]
        if not theincludexsize == xsize:
            print('Dimensions of ' + maskname + ' mask do not match the fmri data - exiting')
            sys.exit()
    else:
        themask, maskarray, mask_hdr, maskdims, masksizes = tide_io.readfromnifti(maskfilename)
        if not tide_io.checkspacematch(mask_hdr, nim_hdr):
            print('Dimensions of ' + maskname + ' mask do not match the fmri data - exiting')
            sys.exit()
    if valslist is not None:
        tempmask = (0 * maskarray).astype('int16')
        for theval in valslist:
            if verbose:
                print('looking for voxels matching', theval)
            tempmask[np.where(np.fabs(maskarray - theval) < 0.1)] += 1
        maskarray = np.where(tempmask > 0, 1, 0)
    return maskarray


def getglobalsignal(indata, optiondict, includemask=None, excludemask=None):
    # mask to interesting voxels
    if optiondict['globalmaskmethod'] == 'mean':
        themask = tide_stats.makemask(np.mean(indata, axis=1), optiondict['corrmaskthreshpct'])
    elif optiondict['globalmaskmethod'] == 'variance':
        themask = tide_stats.makemask(np.var(indata, axis=1), optiondict['corrmaskthreshpct'])
    if optiondict['nothresh']:
        themask *= 0
        themask += 1
    if includemask is not None:
        themask = themask * includemask
    if excludemask is not None:
        themask = themask * (1 - excludemask)

    # add up all the voxels
    globalmean = rt_floatset(indata[0, :])
    thesize = np.shape(themask)
    numvoxelsused = 0
    for vox in range(0, thesize[0]):
        if themask[vox] > 0.0:
            numvoxelsused += 1
            if optiondict['meanscaleglobal']:
                themean = np.mean(indata[vox, :])
                if themean != 0.0:
                    globalmean = globalmean + indata[vox, :] / themean - 1.0
            else:
                globalmean = globalmean + indata[vox, :]
    print()
    print('used ', numvoxelsused, ' voxels to calculate global mean signal')
    return tide_math.stdnormalize(globalmean), themask




def _get_parser():
    """
    Argument parser for rapidtide
    """
    parser = argparse.ArgumentParser('rapidtideX - perform time delay analysis on a data file', usage=argparse.SUPPRESS)

    # Required arguments
    parser.add_argument('in_file',
                        type=lambda x: is_valid_file(parser, x),
                        help='The input data file (BOLD fmri file or NIRS text file)')
    parser.add_argument('outputname',
                        help='The root name for the output files')

    # Analysis types
    analysis_type = parser.add_argument_group('Analysis type').add_mutually_exclusive_group()
    analysis_type.add_argument('--denoising',
                        dest='denoising',
                        action='store_true',
                        help=('This is a macro that sets --passes=3, '
                              '--lagmaxthresh=6.0, --ampthresh=0.5, and '
                              '--refineupperlag to bias refinement towards '
                              'voxels in the draining vasculature for an '
                              'fMRI scan. '),
                        default=False)
    analysis_type.add_argument('--delaymapping',
                        dest='delaymapping',
                        action='store_true',
                        help=('This is a NIRS analysis - this is a macro that '
                              'sets --nothresh, --preservefiltering, '
                              '--refineprenorm=var, --ampthresh=0.7, and '
                              '--lagminthresh=0.1. '),
                        default=False)

    # Macros
    macros = parser.add_argument_group('Macros').add_mutually_exclusive_group()
    macros.add_argument('--venousrefine',
                        dest='venousrefine',
                        action='store_true',
                        help=('This is a macro that sets --lagminthresh=2.5, '
                              '--lagmaxthresh=6.0, --ampthresh=0.5, and '
                              '--refineupperlag to bias refinement towards '
                              'voxels in the draining vasculature for an '
                              'fMRI scan. '),
                        default=False)
    macros.add_argument('--nirs',
                        dest='nirs',
                        action='store_true',
                        help=('This is a NIRS analysis - this is a macro that '
                              'sets --nothresh, --preservefiltering, '
                              '--refineprenorm=var, --ampthresh=0.7, and '
                              '--lagminthresh=0.1. '),
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
                              'of these options is mandatory. '),
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
                              'of these options is mandatory. '),
                        default='auto')
    preproc.add_argument('--noantialias',
                         dest='antialias',
                         action='store_false',
                         help='Disable antialiasing filter. ',
                         default=True)
    preproc.add_argument('--invert',
                         dest='invertregressor',
                         action='store_true',
                         help=('Invert the sign of the regressor before '
                               'processing. '),
                         default=False)
    preproc.add_argument('--interptype',
                         dest='interptype',
                         action='store',
                         type=str,
                         choices=['univariate', 'cubic', 'quadratic'],
                         help=("Use specified interpolation type. Options "
                               "are 'cubic','quadratic', and 'univariate' "
                               "(default). "),
                         default='univariate')
    preproc.add_argument('--offsettime',
                         dest='offsettime',
                         action='store',
                         type=float,
                         metavar='OFFSETTIME',
                         help='Apply offset OFFSETTIME to the lag regressors. ',
                         default=0.0)

    filt_opts = parser.add_argument_group('Filtering options')
    filt_opts.add_argument('--filterfreqs',
                          dest='arbvec',
                          action='store',
                          nargs='+',
                          type=lambda x: is_float(parser, x),
                          metavar=('LOWERPASS UPPERPASS',
                                   'LOWERSTOP UPPERSTOP'),
                          help=('Filter data and regressors to retain LOWERPASS to '
                                'UPPERPASS. LOWERSTOP and UPPERSTOP can also '
                                'be specified, or will be calculated '
                                'automatically. '),
                          default=None)
    filt_opts.add_argument('--filterband',
                          dest='filterband',
                          action='store',
                          type=str,
                          choices=['vlf', 'lfo', 'resp', 'cardiac', 'lfo_legacy'],
                          help=('Filter data and regressors to specific band. '),
                          default='lfo')
    filt_opts.add_argument('--filtertype',
                          dest='filtertype',
                          action='store',
                          type=str,
                          choices=['trapezoidal', 'brickwall', 'butterworth'],
                          help=('Filter data and regressors using a trapezoidal FFT filter (default), brickwall, or butterworth bandpass.'),
                          default='trapezoidal')
    filt_opts.add_argument('--butterorder',
                         dest='butterorder',
                         action='store',
                         type=int,
                         metavar='ORDER',
                         help=('Set order of butterworth filter for band splitting. '),
                         default=6)
    filt_opts.add_argument('--padseconds',
                         dest='padseconds',
                         action='store',
                         type=float,
                         metavar='SECONDS',
                         help=('The number of seconds of padding to add to each end of a filtered timecourse. '),
                         default=30.0)


    permutationmethod = preproc.add_mutually_exclusive_group()
    permutationmethod.add_argument('--permutationmethod',
                          dest='permutationmethod',
                          action='store',
                          type=str,
                          choices=['shuffle', 'phaserandom'],
                          help=('Permutation method for significance testing.  Default is shuffle. '),
                          default='shuffle')

    preproc.add_argument('--numnull',
                         dest='numestreps',
                         action='store',
                         type=int,
                         metavar='NREPS',
                         help=('Estimate significance threshold by running '
                               'NREPS null correlations (default is 10000, '
                               'set to 0 to disable). '),
                         default=10000)
    preproc.add_argument('--skipsighistfit',
                         dest='dosighistfit',
                         action='store_false',
                         help=('Do not fit significance histogram with a '
                               'Johnson SB function. '),
                         default=True)

    wfunc = preproc.add_mutually_exclusive_group()
    wfunc.add_argument('--windowfunc',
                       dest='windowfunc',
                       action='store',
                       type=str,
                       choices=['hamming', 'hann', 'blackmanharris', 'None'],
                       help=('Window function to use prior to correlation. '
                             'Options are hamming (default), hann, '
                             'blackmanharris, and None. '),
                       default='hamming')
    wfunc.add_argument('--nowindow',
                       dest='windowfunc',
                       action='store_const',
                       const='None',
                       help='Disable precorrelation windowing. ',
                       default='hamming')

    preproc.add_argument('--detrendorder',
                         dest='detrendorder',
                         action='store',
                         type=int,
                         metavar='ORDER',
                         help=('Set order of trend removal (0 to disable, default is 1 - linear). '),
                         default=3)
    preproc.add_argument('--spatialfilt',
                         dest='gausssigma',
                         action='store',
                         type=float,
                         metavar='GAUSSSIGMA',
                         help=('Spatially filter fMRI data prior to analysis '
                               'using GAUSSSIGMA in mm. '),
                         default=0.0)
    preproc.add_argument('--globalmean',
                         dest='useglobalref',
                         action='store_true',
                         help=('Generate a global mean regressor and use that '
                               'as the reference regressor.  If no external regressor is specified, this'
                               'is enatbled by default. '),
                         default=False)

    globalmethod = preproc.add_mutually_exclusive_group()
    globalmethod.add_argument('--globalmaskmethod',
                       dest='globalmaskmethod',
                       action='store',
                       type=str,
                       choices=['mean', 'variance'],
                       help=('Select whether to use timecourse mean (default) or variance to mask voxels prior to generating global mean. '),
                       default='mean')

    preproc.add_argument('--globalmeaninclude',
                         dest='globalmeanincludespec',
                         metavar='MASK[:VALSPEC]',
                         help=('Only use voxels in NAME for global regressor '
                               'generation (if VALSPEC is given, only voxels '
                               'with integral values listed in VALSPEC are used). '),
                         default=None)
    preproc.add_argument('--globalmeanexclude',
                         dest='globalmeanexcludespec',
                         metavar='MASK[:VALSPEC]',
                         help=('Do not use voxels in NAME for global regressor '
                               'generation (if VALSPEC is given, only voxels '
                               'with integral values listed in VALSPEC are excluded). '),
                         default=None)
    preproc.add_argument('--motionfile',
                         dest='motionfilespec',
                         metavar='MASK[:VALSPEC]',
                         help=('Read 6 columns of motion regressors out of MOTFILE text file. '
                               '(with timepoints rows) and regress their derivatives '
                               'and delayed derivatives out of the data prior to analysis. '
                               'If COLSPEC is present, use the comma separated list of ranges to '
                               'specify X, Y, Z, RotX, RotY, and RotZ, in that order.  For  '
                               'example, :3-5,7,0,9 would use columns 3, 4, 5, 7, 0 and 9 '
                               'for X, Y, Z, RotX, RotY, RotZ, respectively. '),
                         default=None)
    preproc.add_argument('--motpos',
                         dest='mot_pos',
                         action='store_true',
                         help=('Toggle whether displacement regressors will be used in motion regression. Default is False. '),
                         default=False)
    preproc.add_argument('--motderiv',
                         dest='mot_deriv',
                         action='store_false',
                         help=('Toggle whether derivatives will be used in motion regression.  Default is True. '),
                         default=True)
    preproc.add_argument('--motdelayderiv',
                         dest='mot_delayderiv',
                         action='store_true',
                         help=('Toggle whether delayed derivative regressors will be used in motion regression.  Default is False. '),
                         default=False)


    preproc.add_argument('--meanscale',
                         dest='meanscaleglobal',
                         action='store_true',
                         help=('Mean scale regressors during global mean '
                               'estimation. '),
                         default=False)
    preproc.add_argument('--slicetimes',
                         dest='slicetimes',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Apply offset times from FILE to each slice in '
                               'the dataset. '),
                         default=None)
    preproc.add_argument('--numskip',
                         dest='preprocskip',
                         action='store',
                         type=int,
                         metavar='SKIP',
                         help=('SKIP TRs were previously deleted during '
                               'preprocessing (default is 0). '),
                         default=0)
    preproc.add_argument('--nothresh',
                         dest='nothresh',
                         action='store_true',
                         help=('Disable voxel intensity threshold (especially '
                               'useful for NIRS data). '),
                         default=False)

    # Correlation options
    corr = parser.add_argument_group('Correlation options')
    corr.add_argument('--oversampfac',
                      dest='oversampfactor',
                      action='store',
                      type=int,
                      metavar='OVERSAMPFAC',
                      help=('Oversample the fMRI data by the following '
                            'integral factor.  Set to -1 for automatic selection (default). '),
                      default=-1)
    corr.add_argument('--regressor',
                      dest='regressorfile',
                      action='store',
                      type=lambda x: is_valid_file(parser, x),
                      metavar='FILE',
                      help=('Read probe regressor from file FILE (if none '
                            'specified, generate and use global regressor). '),
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
                                 'are two ways to specify the same thing. '),
                           default='auto')
    reg_group.add_argument('--regressortstep',
                           dest='inputfreq',
                           action='store',
                           type=lambda x: invert_float(parser, x),
                           metavar='TSTEP',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing. '),
                           default='auto')

    corr.add_argument('--regressorstart',
                      dest='inputstarttime',
                      action='store',
                      type=float,
                      metavar='START',
                      help=('The time delay in seconds into the regressor '
                            'file, corresponding in the first TR of the fMRI '
                            'file (default is 0.0). '),
                      default=0.)

    cc_group = corr.add_mutually_exclusive_group()
    cc_group.add_argument('--corrweighting',
                          dest='corrweighting',
                          action='store',
                          type=str,
                          choices=['none', 'phat', 'liang', 'eckart'],
                          help=('Method to use for cross-correlation '
                                'weighting. Default is none. '),
                          default='none')

    mask_group = corr.add_mutually_exclusive_group()
    mask_group.add_argument('--corrmaskthresh',
                            dest='corrmaskthreshpct',
                            action='store',
                            type=float,
                            metavar='PCT',
                            help=('Do correlations in voxels where the mean '
                                  'exceeds this percentage of the robust max '
                                  '(default is 1.0). '),
                            default=1.0)
    mask_group.add_argument('--corrmask',
                            dest='corrmaskname',
                            action='store',
                            type=lambda x: is_valid_file(parser, x),
                            metavar='FILE',
                            help=('Only do correlations in voxels in FILE '
                                  '(if set, corrmaskthresh is ignored). '),
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
                                "DELAYTIME seconds for all voxels. "),
                          default=None)
    fixdelay.add_argument('--searchrange',
                          dest='lag_extrema',
                          action='store',
                          nargs=2,
                          type=float,
                          metavar=('LAGMIN', 'LAGMAX'),
                          help=('Limit fit to a range of lags from LAGMIN to '
                                'LAGMAX.  Default is -30.0 to 30.0 seconds. '),
                          default=(-30.0, 30.0))

    corr_fit.add_argument('--sigmalimit',
                          dest='widthlimit',
                          action='store',
                          type=float,
                          metavar='SIGMALIMIT',
                          help=('Reject lag fits with linewidth wider than '
                                'SIGMALIMIT Hz. Default is 100.0. '),
                          default=100.0)
    corr_fit.add_argument('--bipolar',
                          dest='bipolar',
                          action='store_true',
                          help=('Bipolar mode - match peak correlation '
                                'ignoring sign. '),
                          default=False)
    corr_fit.add_argument('--nofitfilt',
                          dest='zerooutbadfit',
                          action='store_false',
                          help=('Do not zero out peak fit values if fit '
                                'fails. '),
                          default=True)
    corr_fit.add_argument('--maxfittype',
                          dest='findmaxtype',
                          action='store',
                          type=str,
                          choices=['gauss', 'quad'],
                          help=("Method for fitting the correlation peak "
                                "(default is 'gauss'). 'quad' uses a "
                                "quadratic fit.  Faster but not as well "
                                "tested. "),
                          default='gauss')
    corr_fit.add_argument('--despecklepasses',
                          dest='despeckle_passes',
                          action='store',
                          type=int,
                          metavar='PASSES',
                          help=('Detect and refit suspect correlations to '
                                'disambiguate peak locations in PASSES '
                                'passes. '),
                          default=0)
    corr_fit.add_argument('--despecklethresh',
                          dest='despeckle_thresh',
                          action='store',
                          type=float,
                          metavar='VAL',
                          help=('Refit correlation if median discontinuity '
                                'magnitude exceeds VAL (default is 5.0s). '),
                          default=5.0)

    # Regressor refinement options
    reg_ref = parser.add_argument_group('Regressor refinement options')
    reg_ref.add_argument('--refineprenorm',
                         dest='refineprenorm',
                         action='store',
                         type=str,
                         choices=['None', 'mean', 'var', 'std', 'invlag'],
                         help=("Apply TYPE prenormalization to each "
                               "timecourse prior to refinement. "),
                         default='mean')
    reg_ref.add_argument('--refineweighting',
                         dest='refineweighting',
                         action='store',
                         type=str,
                         choices=['None', 'NIRS', 'R', 'R2'],
                         help=("Apply TYPE weighting to each timecourse prior "
                               "to refinement. Valid weightings are "
                               "'None', 'NIRS', 'R', and 'R2' (default). "),
                         default='R2')
    reg_ref.add_argument('--passes',
                         dest='passes',
                         action='store',
                         type=int,
                         metavar='PASSES',
                         help=('Set the number of processing passes to '
                               'PASSES.  Default is 3. '),
                         default=3)
    reg_ref.add_argument('--refineinclude',
                         dest='refineincludespec',
                         metavar='MASK[:VALSPEC]',
                         help=('Only use voxels in NAME for regressor refinement '
                               '(if VALSPEC is given, only voxels '
                               'with integral values listed in VALSPEC are used). '),
                         default=None)
    reg_ref.add_argument('--refineexclude',
                         dest='refineexcludespec',
                         metavar='MASK[:VALSPEC]',
                         help=('Do not use voxels in NAME for regressor refinement '
                               '(if VALSPEC is given, voxels '
                               'with integral values listed in VALSPEC are excluded). '),
                         default=None)
    reg_ref.add_argument('--lagminthresh',
                         dest='lagminthresh',
                         action='store',
                         metavar='MIN',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'less than MIN (default is 0.25s). '),
                         default=0.25)
    reg_ref.add_argument('--lagmaxthresh',
                         dest='lagmaxthresh',
                         action='store',
                         metavar='MAX',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'greater than MAX (default is 5s). '),
                         default=5.0)
    reg_ref.add_argument('--ampthresh',
                         dest='ampthresh',
                         action='store',
                         metavar='AMP',
                         type=float,
                         help=('or refinement, exclude voxels with '
                               'correlation coefficients less than AMP '
                               '(default is 0.3). '),
                         default=-1.0)
    reg_ref.add_argument('--sigmathresh',
                         dest='sigmathresh',
                         action='store',
                         metavar='SIGMA',
                         type=float,
                         help=('For refinement, exclude voxels with widths '
                               'greater than SIGMA (default is 100s). '),
                         default=100.0)
    reg_ref.add_argument('--norefineoffset',
                         dest='refineoffset',
                         action='store_false',
                         help=('Disable realigning refined regressor to zero lag. '),
                         default=True)
    reg_ref.add_argument('--psdfilter',
                         dest='psdfilter',
                         action='store_true',
                         help=('Apply a PSD weighted Wiener filter to '
                               'shifted timecourses prior to refinement. '),
                         default=False)
    reg_ref.add_argument('--pickleft',
                        dest='pickleft',
                        action='store_true',
                        help=('Will select the leftmost delay peak when setting the refine offset. '),
                        default=False)

    refine = reg_ref.add_mutually_exclusive_group()
    refine.add_argument('--refineupperlag',
                        dest='lagmaskside',
                        action='store_const',
                        const='upper',
                        help=('Only use positive lags for regressor '
                              'refinement. '),
                        default='both')
    refine.add_argument('--refinelowerlag',
                        dest='lagmaskside',
                        action='store_const',
                        const='lower',
                        help=('Only use negative lags for regressor '
                              'refinement. '),
                        default='both')
    reg_ref.add_argument('--refinetype',
                         dest='refinetype',
                         action='store',
                         type=str,
                         choices=['pca', 'ica', 'weighted_average', 'unweighted_average'],
                         help=('Method with which to derive refined '
                               'regressor. '),
                         default='unweighted_average')

    # Output options
    output = parser.add_argument_group('Output options')
    output.add_argument('--limitoutput',
                        dest='limitoutput',
                        action='store_true',
                        help=("Don't save some of the large and rarely used "
                              "files. "),
                        default=False)
    output.add_argument('--savelags',
                        dest='savecorrtimes',
                        action='store_true',
                        help='Save a table of lagtimes used. ',
                        default=False)
    output.add_argument('--histlen',  # was -h
                        dest='histlen',
                        action='store',
                        type=int,
                        metavar='HISTLEN',
                        help=('Change the histogram length to HISTLEN '
                              '(default is 100). '),
                        default=100)
    output.add_argument('--timerange',
                        dest='timerange',
                        action='store',
                        nargs=2,
                        type=int,
                        metavar=('START', 'END'),
                        help=('Limit analysis to data between timepoints '
                              'START and END in the fmri file. '),
                        default=(-1, 10000000))
    output.add_argument('--glmsourcefile',
                        dest='glmsourcefile',
                        action='store',
                        type=lambda x: is_valid_file(parser, x),
                        metavar='FILE',
                        help=('Regress delayed regressors out of FILE instead '
                              'of the initial fmri file used to estimate '
                              'delays. '),
                        default=None)
    output.add_argument('--noglm',
                        dest='doglmfilt',
                        action='store_false',
                        help=('Turn off GLM filtering to remove delayed '
                              'regressor from each voxel (disables output of '
                              'fitNorm). '),
                        default=True)
    output.add_argument('--preservefiltering',
                        dest='preservefiltering',
                        action='store_true',
                        help="Don't reread data prior to performing GLM. ",
                        default=False)

    # Miscellaneous options
    misc = parser.add_argument_group('Miscellaneous options')
    misc.add_argument('--noprogressbar',
                      dest='showprogressbar',
                      action='store_false',
                      help='Will disable showing progress bars (helpful if stdout is going to a file). ',
                      default=True)
    misc.add_argument('--checkpoint',
                      dest='checkpoint',
                      action='store_true',
                      help='Enable run checkpoints. ',
                      default=False)
    misc.add_argument('--wiener',
                      dest='dodeconv',
                      action='store_true',
                      help=('Do Wiener deconvolution to find voxel transfer '
                            'function. '),
                      default=False)
    misc.add_argument('--saveoptionsastext',
                      dest='saveoptionsasjson',
                      action='store_false',
                      help=('Save options as text, rather than as a json file. '),
                      default=True)
    misc.add_argument('--spcalculation',
                      dest='internalprecision',
                      action='store_const',
                      const='single',
                      help=('Use single precision for internal calculations '
                            '(may be useful when RAM is limited). '),
                      default='double')
    misc.add_argument('--dpoutput',
                      dest='outputprecision',
                      action='store_const',
                      const='double',
                      help=('Use double precision for output files. '),
                      default='single')
    misc.add_argument('--cifti',
                      dest='isgrayordinate',
                      action='store_true',
                      help='Data file is a converted CIFTI. ',
                      default=False)
    misc.add_argument('--simulate',
                      dest='fakerun',
                      action='store_true',
                      help='Simulate a run - just report command line options. ',
                      default=False)
    misc.add_argument('--displayplots',
                      dest='displayplots',
                      action='store_true',
                      help='Display plots of interesting timecourses. ',
                      default=False)
    misc.add_argument('--nonumba',
                      dest='nonumba',
                      action='store_true',
                      help='Disable jit compilation with numba. ',
                      default=False)
    misc.add_argument('--nosharedmem',
                      dest='sharedmem',
                      action='store_false',
                      help=('Disable use of shared memory for large array '
                            'storage. '),
                      default=True)
    misc.add_argument('--memprofile',
                      dest='memprofile',
                      action='store_true',
                      help=('Enable memory profiling for debugging - '
                            'warning: this slows things down a lot. '),
                      default=False)
    misc.add_argument('--mklthreads',
                      dest='mklthreads',
                      action='store',
                      type=int,
                      metavar='MKLTHREADS',
                      help=('Use no more than MKLTHREADS worker threads in accelerated numpy calls. '),
                      default=1)
    misc.add_argument('--nprocs',
                      dest='nprocs',
                      action='store',
                      type=int,
                      metavar='NPROCS',
                      help=('Use NPROCS worker processes for multiprocessing. '
                            'Setting NPROCS to less than 1 sets the number of '
                            'worker processes to n_cpus - 1. '),
                      default=1)
    misc.add_argument('--debug',
                      dest='debug',
                      action='store_true',
                      help=('Enable additional debugging output.'),
                      default=False)
    misc.add_argument('--verbose',
                      dest='verbose',
                      action='store_true',
                      help=('Enable additional runtime information output. '),
                      default=False)

    # Experimental options (not fully tested, may not work)
    experimental = parser.add_argument_group('Experimental options (not fully '
                                             'tested, may not work)')
    experimental.add_argument('--respdelete',
                              dest='respdelete',
                              action='store_true',
                              help=('Attempt to detect and remove respiratory signal that strays into the LFO band.'),
                              default=False)
    experimental.add_argument('--cleanrefined',
                              dest='cleanrefined',
                              action='store_true',
                              help=('Perform additional processing on refined '
                                    'regressor to remove spurious '
                                    'components. '),
                              default=False)
    experimental.add_argument('--dispersioncalc',
                              dest='dodispersioncalc',
                              action='store_true',
                              help=('Generate extra data during refinement to '
                                    'allow calculation of dispersion. '),
                              default=False)
    experimental.add_argument('--acfix',
                              dest='fix_autocorrelation',
                              action='store_true',
                              help=('Perform a secondary correlation to '
                                    'disambiguate peak location. Experimental. '),
                              default=False)
    experimental.add_argument('--tmask',
                              dest='tmaskname',
                              action='store',
                              type=lambda x: is_valid_file(parser, x),
                              metavar='FILE',
                              help=('Only correlate during epochs specified '
                                    'in MASKFILE (NB: each line of FILE '
                                    'contains the time and duration of an '
                                    'epoch to include. '),
                              default=None)
    return parser


def rapidtide_workflow(in_file, outputname, venousrefine=False, nirs=False,
                       realtr='auto', antialias=True, invertregressor=False,
                       interptype='univariate', offsettime=None,
                       butterorder=None, arbvec=None, filterband='lfo',
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
                       tmaskname=None,
                       offsettime_total=None,
                       ampthreshfromsig=False, nohistzero=False,
                       fixdelay=False, usebutterworthfilter=False, permutationmethod='shuffle'):
    """
    Run the full rapidtide workflow.
    """
    pass


def process_args():
    """
    Compile arguments for rapidtide workflow.
    """
    #args = vars(_get_parser().parse_args())
    try:
        args = vars(_get_parser().parse_args())
    except SystemExit:
        _get_parser().print_help()
        raise
    print(args)

    # some tunable parameters for internal debugging
    args['addedskip'] = 0
    args['dodemean'] = True
    args['edgebufferfrac'] = 0.0  # what fraction of the correlation window to avoid on either end when fitting
    args['enforcethresh'] = True  # only do fits in voxels that exceed threshhold
    args['lagmod']  = 1000.0  # if set to the location of the first autocorrelation sidelobe, this will fold back sidelobes
    args['fastgauss'] = False  # use a non-iterative gaussian peak fit (DOES NOT WORK)
    args['lthreshval'] = 0.0  # zero out peaks with correlations lower than this value
    args['uthreshval'] = 1.0  # zero out peaks with correlations higher than this value
    args['absmaxsigma'] = 100.0  # width of the reference autocorrelation function
    args['absminsigma'] = 0.25  # width of the reference autocorrelation function

    # correlation fitting
    args['hardlimit'] = True  # Peak value must be within specified range.  If false, allow max outside if maximum
                              # correlation value is that one end of the range.
    args['gaussrefine'] = True  # fit gaussian after initial guess at parameters
    args['findmaxtype'] = 'gauss'  # if set to 'gauss', use old gaussian fitting, if set to 'quad' use parabolic
    args['searchfrac'] = 0.5  # The fraction of the main peak over which points are included in the peak
    args['mp_chunksize'] = 50000

    # significance estimation
    args['sighistlen'] = 1000
    args['dosighistfit'] = True

    # output options
    args['savecorrmask'] = True
    args['savedespecklemasks'] = True
    args['saveglmfiltered'] = True
    args['savemotionfiltered'] = False
    args['savecorrmask'] = True
    args['histlen'] = 250

    # refinement options
    args['estimatePCAdims'] = False
    args['filterbeforePCA'] = True

    # autocorrelation processing
    args['check_autocorrelation'] = True
    args['acwidth'] = 0.0  # width of the reference autocorrelation function

    # diagnostic information about version
    args['release_version'], \
    args['git_longtag'], \
    args['git_date'],\
    args['git_isdirty'] = tide_util.version()
    args['python_version'] = str(sys.version_info)


    # configure the filter
    # set the trapezoidal flag, if using
    if args['filtertype'] == 'trapezoidal':
        inittrap = True
    else:
        inittrap = False

    # if arbvec is set, we are going set up an arbpass filter
    if args['arbvec'] is not None:
        if len(args['arbvec']) == 2:
            args['arbvec'].append(args['arbvec'][0] * 0.95)
            args['arbvec'].append(args['arbvec'][1] * 1.05)
        elif len(args['arbvec']) != 4:
            raise ValueError("Argument '--arb' must be either two "
                             "or four floats.")
        theprefilter = tide_filt.noncausalfilter('arb', usetrapfftfilt=inittrap)
        theprefilter.setfreqs(*args['arbvec'])
    else:
        theprefilter = tide_filt.noncausalfilter(args['filterband'], usetrapfftfilt=inittrap)

    # make the filter a butterworth if selected
    if args['filtertype'] == 'butterworth':
        args['usebutterworthfilter'] = True
    else:
        args['usebutterworthfilter'] = False
    theprefilter.setbutter(args['usebutterworthfilter'], args['butterorder'])


    # Additional argument parsing not handled by argparse
    args['lagmin'] = args['lag_extrema'][0]
    args['lagmax'] = args['lag_extrema'][1]
    args['startpoint'] = args['timerange'][0]
    args['endpoint'] = args['timerange'][1]


    if args['offsettime'] is not None:
        args['offsettime_total'] = -1 * args['offsettime']
    else:
        args['offsettime_total'] = None

    reg_ref_used = ((args['lagminthresh'] != 0.5) or
                    (args['lagmaxthresh'] != 5.) or
                    (args['ampthresh'] != 0.3) or
                    (args['sigmathresh'] != 100.) or
                    (args['refineoffset']))
    if reg_ref_used and args['passes'] == 1:
        args['passes'] = 2

    if args['numestreps'] == 0:
        args['ampthreshfromsig'] = False
    else:
        args['ampthreshfromsig'] = True

    if args['ampthresh'] < 0.0:
        args['ampthresh'] = 0.3
        args['ampthreshfromsig'] = True
    else:
        args['ampthreshfromsig'] = False

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

    if args['windowfunc'] is None:
        args['usewindowfunc'] = False
    else:
        args['usewindowfunc'] = True

    if args['in_file'].endswith('txt') and args['realtr'] == 'auto':
        raise ValueError('Either --datatstep or --datafreq must be provided '
                         'if data file is a text file.')

    if args['realtr'] != 'auto':
        fmri_tr = float(args['realtr'])
    else:
        fmri_tr = nib.load(args['in_file']).header.get_zooms()[3]
    args['realtr'] = fmri_tr

    if args['inputfreq'] == 'auto':
        args['inputfreq'] = 1. / fmri_tr

    # mask processing
    if args['globalmeanincludespec'] is not None:
        args['globalmeanincludename'], args['globalmeanincludevals'] = processmaskspec(args['globalmeanincludespec'],
                                                                                       'Including voxels where ',
                                                                                       'in global mean.')
    else:
        args['globalmeanincludename'] = None

    if args['globalmeanexcludespec'] is not None:
        args['globalmeanexcludename'], args['globalmeanexcludevals'] = processmaskspec(args['globalmeanexcludespec'],
                                                                                       'Excluding voxels where ',
                                                                                       'from global mean.')
    else:
        args['globalmeanexcludename'] = None

    if args['refineincludespec'] is not None:
        args['refineincludename'], args['refineincludevals'] = processmaskspec(args['refineincludespec'],
                                                                                       'Including voxels where ',
                                                                                       'in refinement.')
    else:
        args['refineincludename'] = None

    if args['refineexcludespec'] is not None:
        args['refineexcludename'], args['refineexcludevals'] = processmaskspec(args['refineexcludespec'],
                                                                                       'Excluding voxels where ',
                                                                                       'from refinement.')
    else:
        args['refineexcludename'] = None

    # motion processing
    if args['motionfilespec'] is not None:
        args['motionfilename'], args['motionfilevals'] = processmaskspec(args['motionfilespec'],
                                                                         'Using columns in ',
                                                                         'as motion regressors.')
    else:
        args['motionfilename'] = None

    if args['limitoutput']:
        args['savedatatoremove'] = False
        args['savelagregressors'] = False
    else:
        args['savedatatoremove'] = True
        args['savelagregressors'] = True


    if args['venousrefine']:
        print('WARNING: Using "venousrefine" macro. Overriding any affected '
              'arguments.')
        args['lagminthresh'] = 2.5
        args['lagmaxthresh'] = 6.
        args['ampthresh'] = 0.5
        args['ampthreshfromsig'] = False
        args['lagmaskside'] = 'upper'

    if args['nirs']:
        print('WARNING: Using "nirs" macro. Overriding any affected '
              'arguments.')
        args['nothresh'] = False
        args['preservefiltering'] = True
        args['refineprenorm'] = 'var'
        args['ampthresh'] = 0.7
        args['ampthreshfromsig'] = False
        args['lagmaskthresh'] = 0.1

    if args['delaymapping']:
        args['despecklepasses'] = 4
        args['lagmin'] = -10.0
        args['lagmax'] = 30.0
        args['passes'] = 3
        args['refineoffset'] = True
        args['pickleft'] = True
        args['doglmfilt'] = False

    if args['denoising']:
        args['despecklepasses'] = 0
        args['lagmin'] = -15.0
        args['lagmax'] = 15.0
        args['passes'] = 3
        args['refineoffset'] = True
        args['doglmfilt'] = True

    # start the clock!
    tide_util.checkimports(args)

    return args, theprefilter


def rapidtide_main():
    timings = [['Start', time.time(), None, None]]
    optiondict, theprefilter = process_args()

    fmrifilename = optiondict['in_file']
    outputname = optiondict['outputname']
    filename = optiondict['regressorfile']

    if optiondict['saveoptionsasjson']:
        tide_io.writedicttojson(optiondict, outputname + '_options_initial.json')
    else:
        tide_io.writedict(optiondict, outputname + '_options_initial.txt')

    optiondict['dispersioncalc_lower'] = optiondict['lagmin']
    optiondict['dispersioncalc_upper'] = optiondict['lagmax']
    optiondict['dispersioncalc_step'] = np.max(
        [(optiondict['dispersioncalc_upper'] - optiondict['dispersioncalc_lower']) / 25, 0.50])
    timings.append(['Argument parsing done', time.time(), None, None])

    # don't use shared memory if there is only one process
    if optiondict['nprocs'] == 1:
        optiondict['sharedmem'] = False
        print('running single process - disabled shared memory use')

    # disable numba now if we're going to do it (before any jits)
    if optiondict['nonumba']:
        tide_util.disablenumba()

    # set the internal precision
    global rt_floatset, rt_floattype
    if optiondict['internalprecision'] == 'double':
        print('setting internal precision to double')
        rt_floattype = 'float64'
        rt_floatset = np.float64
    else:
        print('setting internal precision to single')
        rt_floattype = 'float32'
        rt_floatset = np.float32

    # set the output precision
    if optiondict['outputprecision'] == 'double':
        print('setting output precision to double')
        rt_outfloattype = 'float64'
        rt_outfloatset = np.float64
    else:
        print('setting output precision to single')
        rt_outfloattype = 'float32'
        rt_outfloatset = np.float32

    # set set the number of worker processes if multiprocessing
    if optiondict['nprocs'] < 1:
        optiondict['nprocs'] = tide_multiproc.maxcpus()

    # set the number of MKL threads to use
    if mklexists:
        mkl.set_num_threads(optiondict['mklthreads'])

    # open up the memory usage file
    if not optiondict['memprofile']:
        memfile = open(outputname + '_memusage.csv', 'w')
        tide_util.logmem(None, file=memfile)

    # open the fmri datafile
    tide_util.logmem('before reading in fmri data', file=memfile)
    if tide_io.checkiftext(fmrifilename):
        print('input file is text - all I/O will be to text files')
        optiondict['textio'] = True
        if optiondict['gausssigma'] > 0.0:
            optiondict['gausssigma'] = 0.0
            print('gaussian spatial filter disabled for text input files')
    else:
        optiondict['textio'] = False

    if optiondict['textio']:
        nim_data = tide_io.readvecs(fmrifilename)
        theshape = np.shape(nim_data)
        xsize = theshape[0]
        ysize = 1
        numslices = 1
        fileiscifti = False
        timepoints = theshape[1]
        thesizes = [0, int(xsize), 1, 1, int(timepoints)]
        numspatiallocs = int(xsize)
        slicesize = numspatiallocs
    else:
        nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
        if nim_hdr['intent_code'] == 3002:
            print('input file is CIFTI')
            optiondict['isgrayordinate'] = True
            fileiscifti = True
            timepoints = nim_data.shape[4]
            numspatiallocs = nim_data.shape[5]
            slicesize = numspatiallocs
            outsuffix3d = '.dscalar'
            outsuffix4d = '.dtseries'
        else:
            print('input file is NIFTI')
            fileiscifti = False
            xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
            numspatiallocs = int(xsize) * int(ysize) * int(numslices)
            slicesize = numspatiallocs / int(numslices)
            outsuffix3d = ''
            outsuffix4d = ''
        xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    tide_util.logmem('after reading in fmri data', file=memfile)

    # correct some fields if necessary
    if optiondict['isgrayordinate']:
        fmritr = 0.72  # this is wrong and is a hack until I can parse CIFTI XML
    else:
        if optiondict['textio']:
            if optiondict['realtr'] <= 0.0:
                print('for text file data input, you must use the -t option to set the timestep')
                sys.exit()
        else:
            if nim_hdr.get_xyzt_units()[1] == 'msec':
                fmritr = thesizes[4] / 1000.0
            else:
                fmritr = thesizes[4]
    if optiondict['realtr'] > 0.0:
        fmritr = optiondict['realtr']

    # check to see if we need to adjust the oversample factor
    if optiondict['oversampfactor'] < 0:
        optiondict['oversampfactor'] = int(np.max([np.ceil(fmritr // 0.5), 1]))
        print('oversample factor set to', optiondict['oversampfactor'])

    oversamptr = fmritr / optiondict['oversampfactor']
    if optiondict['verbose']:
        print('fmri data: ', timepoints, ' timepoints, tr = ', fmritr, ', oversamptr =', oversamptr)
    print(numspatiallocs, ' spatial locations, ', timepoints, ' timepoints')
    timings.append(['Finish reading fmrifile', time.time(), None, None])

    # if the user has specified start and stop points, limit check, then use these numbers
    validstart, validend = tide_util.startendcheck(timepoints, optiondict['startpoint'], optiondict['endpoint'])
    if abs(optiondict['lagmin']) > (validend - validstart + 1) * fmritr / 2.0:
        print('magnitude of lagmin exceeds', (validend - validstart + 1) * fmritr / 2.0, ' - invalid')
        sys.exit()
    if abs(optiondict['lagmax']) > (validend - validstart + 1) * fmritr / 2.0:
        print('magnitude of lagmax exceeds', (validend - validstart + 1) * fmritr / 2.0, ' - invalid')
        sys.exit()
    if optiondict['gausssigma'] > 0.0:
        print('applying gaussian spatial filter to timepoints ', validstart, ' to ', validend)
        reportstep = 10
        for i in range(validstart, validend + 1):
            if (i % reportstep == 0 or i == validend) and optiondict['showprogressbar']:
                tide_util.progressbar(i - validstart + 1, timepoints, label='Percent complete')
            nim_data[:, :, :, i] = tide_filt.ssmooth(xdim, ydim, slicethickness, optiondict['gausssigma'],
                                                     nim_data[:, :, :, i])
        timings.append(['End 3D smoothing', time.time(), None, None])
        print()

    # reshape the data and trim to a time range, if specified.  Check for special case of no trimming to save RAM
    if (validstart == 0) and (validend == timepoints):
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))
    else:
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1]
        validtimepoints = validend - validstart + 1

    # read in the optional masks
    tide_util.logmem('before setting masks', file=memfile)
    internalglobalmeanincludemask = None
    internalglobalmeanexcludemask = None
    internalrefineincludemask = None
    internalrefineexcludemask = None

    if optiondict['globalmeanincludename'] is not None:
        print('constructing global mean include mask')
        theglobalmeanincludemask = readamask(optiondict['globalmeanincludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['globalmeanincludevals'],
                                             maskname='global mean include')
        internalglobalmeanincludemask = theglobalmeanincludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalglobalmeanincludemask) == 0:
            print('ERROR: there are no voxels in the global mean include mask - exiting')
            sys.exit()

    if optiondict['globalmeanexcludename'] is not None:
        print('constructing global mean exclude mask')
        theglobalmeanexcludemask = readamask(optiondict['globalmeanexcludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['globalmeanexcludevals'],
                                             maskname='global mean exclude')
        internalglobalmeanexcludemask = theglobalmeanexcludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalglobalmeanexcludemask) == numspatiallocs:
            print('ERROR: the global mean exclude mask does not leave any voxels - exiting')
            sys.exit()

    if (internalglobalmeanincludemask is not None) and (internalglobalmeanexcludemask is not None):
        if tide_stats.getmasksize(internalglobalmeanincludemask * (1 - internalglobalmeanexcludemask)) == 0:
            print('ERROR: the global mean include and exclude masks not leave any voxels between them - exiting')
            sys.exit()

    if optiondict['refineincludename'] is not None:
        print('constructing refine include mask')
        therefineincludemask = readamask(optiondict['refineincludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['refineincludevals'],
                                             maskname='refine include')
        internalrefineincludemask = therefineincludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalrefineincludemask) == 0:
            print('ERROR: there are no voxels in the refine include mask - exiting')
            sys.exit()

    if optiondict['refineexcludename'] is not None:
        print('constructing refine exclude mask')
        therefineexcludemask = readamask(optiondict['refineexcludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['refineexcludevals'],
                                             maskname='refine exclude')
        internalrefineexcludemask = therefineexcludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalrefineexcludemask) == numspatiallocs:
            print('ERROR: the refine exclude mask does not leave any voxels - exiting')
            sys.exit()

    tide_util.logmem('after setting masks', file=memfile)

    # read or make a mask of where to calculate the correlations
    tide_util.logmem('before selecting valid voxels', file=memfile)
    threshval = tide_stats.getfracvals(fmri_data[:, optiondict['addedskip']:], [0.98])[0] / 25.0
    print('constructing correlation mask')
    if optiondict['corrmaskname'] is not None:
        thecorrmask = readamask(optiondict['corrmaskname'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['corrmaskvals'],
                                             maskname='correlation')

        corrmask = np.uint16(np.where(thecorrmask > 0, 1, 0).reshape(numspatiallocs))
    else:
        # check to see if the data has been demeaned
        meanim = np.mean(fmri_data[:, optiondict['addedskip']:], axis=1)
        stdim = np.std(fmri_data[:, optiondict['addedskip']:], axis=1)
        if np.mean(stdim) < np.mean(meanim):
            print('generating correlation mask from mean image')
            corrmask = np.uint16(tide_stats.makemask(meanim, threshpct=optiondict['corrmaskthreshpct']))
        else:
            print('generating correlation mask from std image')
            corrmask = np.uint16(tide_stats.makemask(stdim, threshpct=optiondict['corrmaskthreshpct']))
    if tide_stats.getmasksize(corrmask) == 0:
        print('ERROR: there are no voxels in the correlation mask - exiting')
        sys.exit()
    if internalrefineincludemask is not None:
        if internalrefineexcludemask is not None:
            if tide_stats.getmasksize(corrmask * internalrefineincludemask * (1 - internalrefineexcludemask)) == 0:
                print('ERROR: the refine include and exclude masks not leave any voxels in the corrmask - exiting')
                sys.exit()
        else:
            if tide_stats.getmasksize(corrmask * internalrefineincludemask) == 0:
                print('ERROR: the refine include mask does not leave any voxels in the corrmask - exiting')
                sys.exit()
    else:
        if internalrefineexcludemask is not None:
            if tide_stats.getmasksize(corrmask * (1 - internalrefineexcludemask)) == 0:
                print('ERROR: the refine exclude mask does not leave any voxels in the corrmask - exiting')
                sys.exit()

    if optiondict['nothresh']:
        corrmask *= 0
        corrmask += 1
        threshval = -10000000.0
    if optiondict['savecorrmask']:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1
        tide_io.savetonifti(corrmask.reshape(xsize, ysize, numslices), theheader, outputname + '_corrmask')

    if optiondict['verbose']:
        print('image threshval =', threshval)
    validvoxels = np.where(corrmask > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    print('validvoxels shape =', numvalidspatiallocs)
    fmri_data_valid = fmri_data[validvoxels, :] + 0.0
    print('original size =', np.shape(fmri_data), ', trimmed size =', np.shape(fmri_data_valid))
    if internalglobalmeanincludemask is not None:
        internalglobalmeanincludemask_valid = 1.0 * internalglobalmeanincludemask[validvoxels]
        del internalglobalmeanincludemask
        print('internalglobalmeanincludemask_valid has size:', internalglobalmeanincludemask_valid.size)
    else:
        internalglobalmeanincludemask_valid = None
    if internalglobalmeanexcludemask is not None:
        internalglobalmeanexcludemask_valid = 1.0 * internalglobalmeanexcludemask[validvoxels]
        del internalglobalmeanexcludemask
        print('internalglobalmeanexcludemask_valid has size:', internalglobalmeanexcludemask_valid.size)
    else:
        internalglobalmeanexcludemask_valid = None
    if internalrefineincludemask is not None:
        internalrefineincludemask_valid = 1.0 * internalrefineincludemask[validvoxels]
        del internalrefineincludemask
        print('internalrefineincludemask_valid has size:', internalrefineincludemask_valid.size)
    else:
        internalrefineincludemask_valid = None
    if internalrefineexcludemask is not None:
        internalrefineexcludemask_valid = 1.0 * internalrefineexcludemask[validvoxels]
        del internalrefineexcludemask
        print('internalrefineexcludemask_valid has size:', internalrefineexcludemask_valid.size)
    else:
        internalrefineexcludemask_valid = None
    tide_util.logmem('after selecting valid voxels', file=memfile)

    # move fmri_data_valid into shared memory
    if optiondict['sharedmem']:
        print('moving fmri data to shared memory')
        timings.append(['Start moving fmri_data to shared memory', time.time(), None, None])
        numpy2shared_func = addmemprofiling(numpy2shared,
                                            optiondict['memprofile'],
                                            memfile,
                                            'before fmri data move')
        fmri_data_valid, fmri_data_valid_shared, fmri_data_valid_shared_shape = numpy2shared_func(fmri_data_valid,
                                                                                                  rt_floatset)
        timings.append(['End moving fmri_data to shared memory', time.time(), None, None])

    # get rid of memory we aren't using
    tide_util.logmem('before purging full sized fmri data', file=memfile)
    del fmri_data
    del nim_data
    tide_util.logmem('after purging full sized fmri data', file=memfile)

    # filter out motion regressors here
    if optiondict['motionfilename'] is not None:
        print('regressing out motion')

        timings.append(['Motion filtering start', time.time(), None, None])
        motionregressors, fmri_data_valid = tide_glmpass.motionregress(optiondict['motionfilename'],
                                                                    fmri_data_valid,
                                                                    tr,
                                                                    motstart=validstart,
                                                                    motend=validend + 1,
                                                                    position=optiondict['mot_pos'],
                                                                    deriv=optiondict['mot_deriv'],
                                                                    derivdelayed=optiondict['mot_delayderiv'])

        timings.append(['Motion filtering end', time.time(), fmri_data_valid.shape[0], 'voxels'])
        tide_io.writenpvecs(motionregressors, outputname + '_orthogonalizedmotion.txt')
        if optiondict['memprofile']:
            memcheckpoint('...done')
        else:
            tide_util.logmem('after motion glm filter', file=memfile)

        if optiondict['savemotionfiltered']:
            outfmriarray = np.zeros((numspatiallocs, validtimepoints), dtype=rt_floattype)
            outfmriarray[validvoxels, :] = fmri_data_valid[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outfmriarray.reshape((numspatiallocs, validtimepoints)),
                                outputname + '_motionfiltered' + '' + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape((xsize, ysize, numslices, validtimepoints)), nim_hdr,
                                outputname + '_motionfiltered' + '')


    # read in the timecourse to resample
    timings.append(['Start of reference prep', time.time(), None, None])
    if filename is None:
        print('no regressor file specified - will use the global mean regressor')
        optiondict['useglobalref'] = True

    if optiondict['useglobalref']:
        inputfreq = 1.0 / fmritr
        inputperiod = 1.0 * fmritr
        inputstarttime = 0.0
        inputvec, meanmask = getglobalsignal(fmri_data_valid, optiondict,
                                             includemask=internalglobalmeanincludemask_valid,
                                             excludemask=internalglobalmeanexcludemask_valid)
        fullmeanmask = np.zeros((numspatiallocs), dtype=rt_floattype)
        fullmeanmask[validvoxels] = meanmask[:]
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1
        tide_io.savetonifti(fullmeanmask.reshape((xsize, ysize, numslices)), theheader,
                            outputname + '_meanmask' + '')
        optiondict['preprocskip'] = 0
    else:
        if optiondict['inputfreq'] is None:
            print('no regressor frequency specified - defaulting to 1/tr')
            inputfreq = 1.0 / fmritr
        if optiondict['inputstarttime'] is None:
            print('no regressor start time specified - defaulting to 0.0')
            inputstarttime = 0.0
        inputperiod = 1.0 / inputfreq
        inputvec = tide_io.readvec(filename)
    numreference = len(inputvec)
    optiondict['inputfreq'] = inputfreq
    optiondict['inputstarttime'] = inputstarttime
    print('regressor start time, end time, and step', inputstarttime, inputstarttime + numreference * inputperiod,
          inputperiod)
    if optiondict['verbose']:
        print('input vector length', len(inputvec), 'input freq', inputfreq, 'input start time', inputstarttime)

    reference_x = np.arange(0.0, numreference) * inputperiod - (inputstarttime + optiondict['offsettime'])

    # Print out initial information
    if optiondict['verbose']:
        print('there are ', numreference, ' points in the original regressor')
        print('the timepoint spacing is ', 1.0 / inputfreq)
        print('the input timecourse start time is ', inputstarttime)

    # generate the time axes
    fmrifreq = 1.0 / fmritr
    optiondict['fmrifreq'] = fmrifreq
    skiptime = fmritr * (optiondict['preprocskip'] + optiondict['addedskip'])
    print('first fMRI point is at ', skiptime, ' seconds relative to time origin')
    initial_fmri_x = np.arange(0.0, validtimepoints - optiondict['addedskip']) * fmritr + skiptime
    os_fmri_x = np.arange(0.0, (validtimepoints - optiondict['addedskip']) * optiondict['oversampfactor'] - (
            optiondict['oversampfactor'] - 1)) * oversamptr + skiptime

    if optiondict['verbose']:
        print(np.shape(os_fmri_x)[0])
        print(np.shape(initial_fmri_x)[0])

    # generate the comparison regressor from the input timecourse
    # correct the output time points
    # check for extrapolation
    if os_fmri_x[0] < reference_x[0]:
        print('WARNING: extrapolating ', os_fmri_x[0] - reference_x[0], ' seconds of data at beginning of timecourse')
    if os_fmri_x[-1] > reference_x[-1]:
        print('WARNING: extrapolating ', os_fmri_x[-1] - reference_x[-1], ' seconds of data at end of timecourse')

    # invert the regressor if necessary
    if optiondict['invertregressor']:
        invertfac = -1.0
    else:
        invertfac = 1.0

    # detrend the regressor if necessary
    if optiondict['detrendorder'] > 0:
        reference_y = invertfac * tide_fit.detrend(inputvec[0:numreference],
                                                   order=optiondict['detrendorder'],
                                                   demean=optiondict['dodemean'])
    else:
        reference_y = invertfac * (inputvec[0:numreference] - np.mean(inputvec[0:numreference]))

    # write out the reference regressor prior to filtering
    tide_io.writenpvecs(reference_y, outputname + '_reference_origres_prefilt.txt')

    # band limit the regressor if that is needed
    print('filtering to ', theprefilter.gettype(), ' band')
    optiondict['lowerstop'], optiondict['lowerpass'], optiondict['upperpass'], optiondict['upperstop'] = theprefilter.getfreqs()
    reference_y_classfilter = theprefilter.apply(inputfreq, reference_y)
    reference_y = reference_y_classfilter

    # write out the reference regressor used
    tide_io.writenpvecs(tide_math.stdnormalize(reference_y), outputname + '_reference_origres.txt')

    # filter the input data for antialiasing
    if optiondict['antialias']:
        print('applying trapezoidal antialiasing filter')
        reference_y_filt = tide_filt.dolptrapfftfilt(inputfreq, 0.25 * fmrifreq, 0.5 * fmrifreq, reference_y,
                                                     padlen=int(inputfreq * optiondict['padseconds']),
                                                     debug=optiondict['debug'])
        reference_y = rt_floatset(reference_y_filt.real)

    warnings.filterwarnings('ignore', 'Casting*')

    if optiondict['fakerun']:
        return

    # generate the resampled reference regressors
    if optiondict['detrendorder'] > 0:
        resampnonosref_y = tide_fit.detrend(
            tide_resample.doresample(reference_x, reference_y, initial_fmri_x, method=optiondict['interptype']),
            order=optiondict['detrendorder'],
            demean=optiondict['dodemean'])
        resampref_y = tide_fit.detrend(
            tide_resample.doresample(reference_x, reference_y, os_fmri_x, method=optiondict['interptype']),
            order=optiondict['detrendorder'],
            demean=optiondict['dodemean'])
    else:
        resampnonosref_y = tide_resample.doresample(reference_x, reference_y, initial_fmri_x,
                                                    method=optiondict['interptype'])
        resampref_y = tide_resample.doresample(reference_x, reference_y, os_fmri_x, method=optiondict['interptype'])

    # prepare the temporal mask
    if optiondict['tmaskname'] is not None:
        tmask_y = maketmask(optiondict['tmaskname'], reference_x, rt_floatset(reference_y))
        tmaskos_y = tide_resample.doresample(reference_x, tmask_y, os_fmri_x, method=optiondict['interptype'])
        tide_io.writenpvecs(tmask_y, outputname + '_temporalmask.txt')
        resampnonosref_y *= tmask_y
        thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
        resampnonosref_y -= thefit[0, 1] * tmask_y
        resampref_y *= tmaskos_y
        thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
        resampref_y -= thefit[0, 1] * tmaskos_y

    if optiondict['passes'] > 1:
        nonosrefname = '_reference_fmrires_pass1.txt'
        osrefname = '_reference_resampres_pass1.txt'
    else:
        nonosrefname = '_reference_fmrires.txt'
        osrefname = '_reference_resampres.txt'

    tide_io.writenpvecs(tide_math.stdnormalize(resampnonosref_y), outputname + nonosrefname)
    tide_io.writenpvecs(tide_math.stdnormalize(resampref_y), outputname + osrefname)
    timings.append(['End of reference prep', time.time(), None, None])

    corrtr = oversamptr
    if optiondict['verbose']:
        print('corrtr=', corrtr)

    # initialize the correlator
    oversampfreq = optiondict['oversampfactor'] / fmritr
    thecorrelator = tide_classes.correlator(Fs=oversampfreq,
                                         ncprefilter=theprefilter,
                                         detrendorder=optiondict['detrendorder'],
                                         windowfunc=optiondict['windowfunc'],
                                         corrweighting=optiondict['corrweighting'])
    thecorrelator.setreftc(np.zeros((optiondict['oversampfactor'] * (validtimepoints - optiondict['addedskip'])),
                                    dtype=np.float))
    numccorrlags = thecorrelator.corrlen
    corrorigin = thecorrelator.corrorigin
    dummy, corrscale, dummy = thecorrelator.getcorrelation(trim=False)

    lagmininpts = int((-optiondict['lagmin'] / corrtr) - 0.5)
    lagmaxinpts = int((optiondict['lagmax'] / corrtr) + 0.5)

    if (lagmaxinpts + lagmininpts) < 3:
        print('correlation search range is too narrow - decrease lagmin, increase lagmax, or increase oversample factor')
        sys.exit(1)

    thecorrelator.setlimits(lagmininpts, lagmaxinpts)
    dummy, trimmedcorrscale, dummy = thecorrelator.getcorrelation()

    if optiondict['verbose']:
        print('corrorigin at point ', corrorigin, corrscale[corrorigin])
        print('corr range from ', corrorigin - lagmininpts, '(', corrscale[
            corrorigin - lagmininpts], ') to ', corrorigin + lagmaxinpts, '(', corrscale[corrorigin + lagmaxinpts], ')')

    if optiondict['savecorrtimes']:
        tide_io.writenpvecs(trimmedcorrscale, outputname + '_corrtimes.txt')

    # allocate all of the data arrays
    tide_util.logmem('before main array allocation', file=memfile)
    if optiondict['textio']:
        nativespaceshape = xsize
    else:
        if fileiscifti:
            nativespaceshape = (1, 1, 1, 1, numspatiallocs)
        else:
            nativespaceshape = (xsize, ysize, numslices)
    internalspaceshape = numspatiallocs
    internalvalidspaceshape = numvalidspatiallocs
    meanval = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagtimes = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagstrengths = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagsigma = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagmask = np.zeros(internalvalidspaceshape, dtype='uint16')
    failimage = np.zeros(internalvalidspaceshape, dtype='uint16')
    R2 = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    outmaparray = np.zeros(internalspaceshape, dtype=rt_floattype)
    tide_util.logmem('after main array allocation', file=memfile)

    corroutlen = np.shape(trimmedcorrscale)[0]
    if optiondict['textio']:
        nativecorrshape = (xsize, corroutlen)
    else:
        if fileiscifti:
            nativecorrshape = (1, 1, 1, corroutlen, numspatiallocs)
        else:
            nativecorrshape = (xsize, ysize, numslices, corroutlen)
    internalcorrshape = (numspatiallocs, corroutlen)
    internalvalidcorrshape = (numvalidspatiallocs, corroutlen)
    print('allocating memory for correlation arrays', internalcorrshape, internalvalidcorrshape)
    if optiondict['sharedmem']:
        corrout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        gaussout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        windowout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        outcorrarray, dummy, dummy = allocshared(internalcorrshape, rt_floatset)
    else:
        corrout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        gaussout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        windowout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        outcorrarray = np.zeros(internalcorrshape, dtype=rt_floattype)
    tide_util.logmem('after correlation array allocation', file=memfile)

    if optiondict['textio']:
        nativefmrishape = (xsize, np.shape(initial_fmri_x)[0])
    else:
        if fileiscifti:
            nativefmrishape = (1, 1, 1, np.shape(initial_fmri_x)[0], numspatiallocs)
        else:
            nativefmrishape = (xsize, ysize, numslices, np.shape(initial_fmri_x)[0])
    internalfmrishape = (numspatiallocs, np.shape(initial_fmri_x)[0])
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    tide_util.logmem('after lagtc array allocation', file=memfile)

    if optiondict['passes'] > 1:
        if optiondict['sharedmem']:
            shiftedtcs, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
            weights, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
        else:
            shiftedtcs = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
            weights = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        tide_util.logmem('after refinement array allocation', file=memfile)
    if optiondict['sharedmem']:
        outfmriarray, dummy, dummy = allocshared(internalfmrishape, rt_floatset)
    else:
        outfmriarray = np.zeros(internalfmrishape, dtype=rt_floattype)

    # prepare for fast resampling
    padvalue = max((-optiondict['lagmin'], optiondict['lagmax'])) + 30.0
    # print('setting up fast resampling with padvalue =',padvalue)
    numpadtrs = int(padvalue // fmritr)
    padvalue = fmritr * numpadtrs
    genlagtc = tide_resample.fastresampler(reference_x, reference_y, padvalue=padvalue)

    # cycle over all voxels
    refine = True
    if optiondict['verbose']:
        print('refine is set to ', refine)
    optiondict['edgebufferfrac'] = max([optiondict['edgebufferfrac'], 2.0 / np.shape(corrscale)[0]])
    if optiondict['verbose']:
        print('edgebufferfrac set to ', optiondict['edgebufferfrac'])

    # intitialize the correlation fitter
    thefitter = tide_classes.correlation_fitter(lagmod=optiondict['lagmod'],
                                             lthreshval=optiondict['lthreshval'],
                                             uthreshval=optiondict['uthreshval'],
                                             bipolar=optiondict['bipolar'],
                                             lagmin=optiondict['lagmin'],
                                             lagmax=optiondict['lagmax'],
                                             absmaxsigma=optiondict['absmaxsigma'],
                                             absminsigma=optiondict['absminsigma'],
                                             debug=optiondict['debug'],
                                             findmaxtype=optiondict['findmaxtype'],
                                             refine=optiondict['gaussrefine'],
                                             searchfrac=optiondict['searchfrac'],
                                             fastgauss=optiondict['fastgauss'],
                                             enforcethresh=optiondict['enforcethresh'],
                                             hardlimit=optiondict['hardlimit'])

    for thepass in range(1, optiondict['passes'] + 1):
        # initialize the pass
        if optiondict['passes'] > 1:
            print('\n\n*********************')
            print('Pass number ', thepass)

        referencetc = tide_math.corrnormalize(resampref_y,
                                              prewindow=optiondict['usewindowfunc'],
                                              detrendorder=optiondict['detrendorder'],
                                              windowfunc=optiondict['windowfunc'])

        # Step -1 - check the regressor for periodic components in the passband
        dolagmod = True
        doreferencenotch = True
        if optiondict['respdelete']:
            resptracker = tide_classes.freqtrack(nperseg=64)
            thetimes, thefreqs = resptracker.track(resampref_y, 1.0 / oversamptr)
            tide_io.writevec(thefreqs, outputname + '_peakfreaks_pass' + str(thepass) + '.txt')
            resampref_y = resptracker.clean(resampref_y, 1.0 / oversamptr, thetimes, thefreqs)
            tide_io.writevec(resampref_y, outputname + '_respfilt_pass' + str(thepass) + '.txt')
            referencetc = tide_math.corrnormalize(resampref_y,
                                                  prewindow=optiondict['usewindowfunc'],
                                                  detrendorder=optiondict['detrendorder'],
                                                  windowfunc=optiondict['windowfunc'])
        if optiondict['check_autocorrelation']:
            print('checking reference regressor autocorrelation properties')
            optiondict['lagmod'] = 1000.0
            lagindpad = corrorigin - 2 * np.max((lagmininpts, lagmaxinpts))
            acmininpts = lagmininpts + lagindpad
            acmaxinpts = lagmaxinpts + lagindpad
            thecorrelator.setreftc(referencetc)
            thecorrelator.setlimits(acmininpts, acmaxinpts)
            thexcorr, accheckcorrscale, dummy = thecorrelator.run(resampref_y)
            thefitter.setcorrtimeaxis(accheckcorrscale)
            maxindex, maxlag, maxval, acwidth, maskval, peakstart, peakend, failreason = \
                tide_corrfit.onecorrfitx(thexcorr,
                                         thefitter,
                                         despeckle_thresh=optiondict['despeckle_thresh'],
                                         lthreshval=optiondict['lthreshval'],
                                         fixdelay=optiondict['fixdelay'],
                                         rt_floatset=rt_floatset,
                                         rt_floattype=rt_floattype
                                         )
            outputarray = np.asarray([accheckcorrscale, thexcorr])
            tide_io.writenpvecs(outputarray, outputname + '_referenceautocorr_pass' + str(thepass) + '.txt')
            thelagthresh = np.max((abs(optiondict['lagmin']), abs(optiondict['lagmax'])))
            theampthresh = 0.1
            print('searching for sidelobes with amplitude >', theampthresh, 'with abs(lag) <', thelagthresh, 's')
            sidelobetime, sidelobeamp = tide_corr.autocorrcheck(
                accheckcorrscale,
                thexcorr,
                acampthresh=theampthresh,
                aclagthresh=thelagthresh,
                prewindow=optiondict['usewindowfunc'],
                detrendorder=optiondict['detrendorder'])
            optiondict['acwidth'] = acwidth + 0.0
            optiondict['absmaxsigma'] = acwidth * 10.0
            if sidelobetime is not None:
                passsuffix = '_pass' + str(thepass + 1)
                optiondict['acsidelobelag' + passsuffix] = sidelobetime
                optiondict['despeckle_thresh'] = np.max([optiondict['despeckle_thresh'], sidelobetime / 2.0])
                optiondict['acsidelobeamp' + passsuffix] = sidelobeamp
                print('\n\nWARNING: autocorrcheck found bad sidelobe at', sidelobetime, 'seconds (', 1.0 / sidelobetime,
                      'Hz)...')
                tide_io.writenpvecs(np.array([sidelobetime]),
                                    outputname + '_autocorr_sidelobetime' + passsuffix + '.txt')
                if optiondict['fix_autocorrelation']:
                    print('Removing sidelobe')
                    if dolagmod:
                        print('subjecting lag times to modulus')
                        optiondict['lagmod'] = sidelobetime / 2.0
                    if doreferencenotch:
                        print('removing spectral component at sidelobe frequency')
                        acstopfreq = 1.0 / sidelobetime
                        acfixfilter = tide_filt.noncausalfilter(debug=optiondict['debug'])
                        acfixfilter.settype('arb_stop')
                        acfixfilter.setfreqs(acstopfreq * 0.9, acstopfreq * 0.95, acstopfreq * 1.05, acstopfreq * 1.1)
                        cleaned_resampref_y = tide_math.corrnormalize(acfixfilter.apply(fmrifreq, resampref_y),
                                                                      prewindow=False,
                                                                      detrendorder=optiondict['detrendorder'])
                        cleaned_referencetc = tide_math.corrnormalize(cleaned_resampref_y,
                                                                      prewindow=optiondict['usewindowfunc'],
                                                                      detrendorder=optiondict['detrendorder'],
                                                                      windowfunc=optiondict['windowfunc'])
                        cleaned_nonosreferencetc = tide_math.stdnormalize(acfixfilter.apply(fmrifreq, resampnonosref_y))
                        tide_io.writenpvecs(cleaned_nonosreferencetc,
                                            outputname + '_cleanedreference_fmrires_pass' + str(thepass) + '.txt')
                        tide_io.writenpvecs(cleaned_referencetc,
                                            outputname + '_cleanedreference_pass' + str(thepass) + '.txt')
                        tide_io.writenpvecs(cleaned_resampref_y,
                                            outputname + '_cleanedresampref_y_pass' + str(thepass) + '.txt')
                else:
                    cleaned_resampref_y = 1.0 * tide_math.corrnormalize(resampref_y,
                                                                        prewindow=False,
                                                                        detrendorder=optiondict['detrendorder'])
                    cleaned_referencetc = 1.0 * referencetc
            else:
                print('no sidelobes found in range')
                cleaned_resampref_y = 1.0 * tide_math.corrnormalize(resampref_y,
                                                                    prewindow=False,
                                                                    detrendorder=optiondict['detrendorder'])
                cleaned_referencetc = 1.0 * referencetc
        else:
            cleaned_resampref_y = 1.0 * tide_math.corrnormalize(resampref_y,
                                                                prewindow=False,
                                                                detrendorder=optiondict['detrendorder'])
            cleaned_referencetc = 1.0 * referencetc

        # Step 0 - estimate significance
        if optiondict['numestreps'] > 0:
            timings.append(['Significance estimation start, pass ' + str(thepass), time.time(), None, None])
            print('\n\nSignificance estimation, pass ' + str(thepass))
            if optiondict['verbose']:
                print('calling getNullDistributionData with args:', oversampfreq, fmritr, corrorigin, lagmininpts,
                      lagmaxinpts)
            getNullDistributionData_func = addmemprofiling(tide_nullcorr.getNullDistributionDatax,
                                                           optiondict['memprofile'],
                                                           memfile,
                                                           'before getnulldistristributiondata')
            if optiondict['checkpoint']:
                tide_io.writenpvecs(cleaned_referencetc,
                                    outputname + '_cleanedreference_pass' + str(thepass) + '.txt')
                tide_io.writenpvecs(cleaned_resampref_y,
                                    outputname + '_cleanedresampref_y_pass' + str(thepass) + '.txt')

                plot(cleaned_resampref_y)
                plot(cleaned_referencetc)
                show()
                if optiondict['saveoptionsasjson']:
                    tide_io.writedicttojson(optiondict, outputname + '_options_pregetnull_pass' + str(thepass) + '.json')
                else:
                    tide_io.writedict(optiondict, outputname + '_options_pregetnull_pass' + str(thepass) + '.txt')
            thecorrelator.setlimits(lagmininpts, lagmaxinpts)
            thecorrelator.setreftc(cleaned_resampref_y)
            dummy, trimmedcorrscale, dummy = thecorrelator.getcorrelation()
            thefitter.setcorrtimeaxis(trimmedcorrscale)
            corrdistdata = getNullDistributionData_func(cleaned_resampref_y,
                                                         oversampfreq,
                                                         thecorrelator,
                                                         thefitter,
                                                         numestreps=optiondict['numestreps'],
                                                         nprocs=optiondict['nprocs'],
                                                         showprogressbar=optiondict['showprogressbar'],
                                                         chunksize=optiondict['mp_chunksize'],
                                                         permutationmethod=optiondict['permutationmethod'],
                                                         fixdelay=optiondict['fixdelay'],
                                                         fixeddelayvalue=optiondict['fixeddelayvalue'],
                                                         rt_floatset=np.float64,
                                                         rt_floattype='float64')
            tide_io.writenpvecs(corrdistdata, outputname + '_corrdistdata_pass' + str(thepass) + '.txt')

            # calculate percentiles for the crosscorrelation from the distribution data
            thepercentiles = np.array([0.95, 0.99, 0.995, 0.999])
            thepvalnames = []
            for thispercentile in thepercentiles:
                thepvalnames.append("{:.3f}".format(1.0 - thispercentile).replace('.', 'p'))

            pcts, pcts_fit, sigfit = tide_stats.sigFromDistributionData(corrdistdata, optiondict['sighistlen'],
                                                                        thepercentiles, twotail=optiondict['bipolar'],
                                                                        displayplots=optiondict['displayplots'],
                                                                        nozero=optiondict['nohistzero'],
                                                                        dosighistfit=optiondict['dosighistfit'])
            if optiondict['ampthreshfromsig']:
                if pcts is not None:
                    print('setting ampthresh to the p<', "{:.3f}".format(1.0 - thepercentiles[0]), ' threshhold')
                    optiondict['ampthresh'] = pcts[2]
                    tide_stats.printthresholds(pcts, thepercentiles, 'Crosscorrelation significance thresholds from data:')
                    if optiondict['dosighistfit']:
                        tide_stats.printthresholds(pcts_fit, thepercentiles,
                                                   'Crosscorrelation significance thresholds from fit:')
                        tide_stats.makeandsavehistogram(corrdistdata, optiondict['sighistlen'], 0,
                                                        outputname + '_nullcorrelationhist_pass' + str(thepass),
                                                        displaytitle='Null correlation histogram, pass' + str(thepass),
                                                        displayplots=optiondict['displayplots'], refine=False)
                else:
                    print('leaving ampthresh unchanged')

            del corrdistdata
            timings.append(['Significance estimation end, pass ' + str(thepass), time.time(), optiondict['numestreps'],
                            'repetitions'])

        # Step 1 - Correlation step
        print('\n\nCorrelation calculation, pass ' + str(thepass))
        timings.append(['Correlation calculation start, pass ' + str(thepass), time.time(), None, None])
        correlationpass_func = addmemprofiling(tide_corrpass.correlationpass,
                                               optiondict['memprofile'],
                                               memfile,
                                               'before correlationpass')

        thecorrelator.setlimits(lagmininpts, lagmaxinpts)
        voxelsprocessed_cp, theglobalmaxlist, trimmedcorrscale = correlationpass_func(fmri_data_valid[:,optiondict['addedskip']:],
                                                               cleaned_referencetc,
                                                               thecorrelator,
                                                               initial_fmri_x,
                                                               os_fmri_x,
                                                               corrorigin,
                                                               lagmininpts,
                                                               lagmaxinpts,
                                                               corrout,
                                                               meanval,
                                                               nprocs=optiondict['nprocs'],
                                                               oversampfactor=optiondict['oversampfactor'],
                                                               interptype=optiondict['interptype'],
                                                               showprogressbar=optiondict['showprogressbar'],
                                                               chunksize=optiondict['mp_chunksize'],
                                                               rt_floatset=rt_floatset,
                                                               rt_floattype=rt_floattype)

        for i in range(len(theglobalmaxlist)):
            theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]]
        tide_stats.makeandsavehistogram(np.asarray(theglobalmaxlist), len(corrscale), 0,
                                        outputname + '_globallaghist_pass' + str(thepass),
                                        displaytitle='lagtime histogram', displayplots=optiondict['displayplots'],
                                        therange=(corrscale[0], corrscale[-1]), refine=False)

        if optiondict['checkpoint']:
            outcorrarray[:, :] = 0.0
            outcorrarray[validvoxels, :] = corrout[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                                    outputname + '_corrout_prefit_pass' + str(thepass) + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                                    outputname + '_corrout_prefit_pass' + str(thepass)+ outsuffix4d)

        timings.append(['Correlation calculation end, pass ' + str(thepass), time.time(), voxelsprocessed_cp, 'voxels'])

        # Step 2 - correlation fitting and time lag estimation
        print('\n\nTime lag estimation pass ' + str(thepass))
        timings.append(['Time lag estimation start, pass ' + str(thepass), time.time(), None, None])
        fitcorr_func = addmemprofiling(tide_corrfit.fitcorrx,
                                       optiondict['memprofile'],
                                       memfile,
                                       'before fitcorr')
        thefitter.setcorrtimeaxis(trimmedcorrscale)
        voxelsprocessed_fc = fitcorr_func(genlagtc,
                                          initial_fmri_x,
                                          lagtc,
                                          trimmedcorrscale,
                                          thefitter,
                                          corrout,
                                          lagmask, failimage, lagtimes, lagstrengths, lagsigma,
                                          gaussout, windowout, R2,
                                          nprocs=optiondict['nprocs'],
                                          fixdelay=optiondict['fixdelay'],
                                          showprogressbar=optiondict['showprogressbar'],
                                          chunksize=optiondict['mp_chunksize'],
                                          despeckle_thresh=optiondict['despeckle_thresh'],
                                          rt_floatset=rt_floatset,
                                          rt_floattype=rt_floattype
                                          )

        timings.append(['Time lag estimation end, pass ' + str(thepass), time.time(), voxelsprocessed_fc, 'voxels'])

        # Step 2b - Correlation time despeckle
        if optiondict['despeckle_passes'] > 0:
            print('\n\nCorrelation despeckling pass ' + str(thepass))
            print('\tUsing despeckle_thresh =' + str(optiondict['despeckle_thresh']))
            timings.append(['Correlation despeckle start, pass ' + str(thepass), time.time(), None, None])

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            for despecklepass in range(optiondict['despeckle_passes']):
                print('\n\nCorrelation despeckling subpass ' + str(despecklepass + 1))
                outmaparray *= 0.0
                outmaparray[validvoxels] = eval('lagtimes')[:]
                medianlags = ndimage.median_filter(outmaparray.reshape(nativespaceshape), 3).reshape(numspatiallocs)
                initlags = \
                    np.where(np.abs(outmaparray - medianlags) > optiondict['despeckle_thresh'], medianlags, -1000000.0)[
                        validvoxels]
                if len(initlags) > 0:
                    if len(np.where(initlags != -1000000.0)[0]) > 0:
                        voxelsprocessed_fc_ds += fitcorr_func(genlagtc,
                                                              initial_fmri_x,
                                                              lagtc,
                                                              trimmedcorrscale,
                                                              thefitter,
                                                              corrout,
                                                              lagmask, failimage, lagtimes, lagstrengths, lagsigma,
                                                              gaussout, windowout, R2,
                                                              nprocs=optiondict['nprocs'],
                                                              fixdelay=optiondict['fixdelay'],
                                                              showprogressbar=optiondict['showprogressbar'],
                                                              chunksize=optiondict['mp_chunksize'],
                                                              despeckle_thresh=optiondict['despeckle_thresh'],
                                                              initiallags=initlags,
                                                              rt_floatset=rt_floatset,
                                                              rt_floattype=rt_floattype
                                                              )
                    else:
                        despecklingdone = True
                else:
                    despecklingdone = True
                if despecklingdone:
                    print('Nothing left to do! Terminating despeckling')
                    break

            if optiondict['savedespecklemasks']:
                theheader = copy.deepcopy(nim_hdr)
                if fileiscifti:
                    theheader['intent_code'] = 3006
                else:
                    theheader['dim'][0] = 3
                    theheader['dim'][4] = 1
                tide_io.savetonifti((np.where(np.abs(outmaparray - medianlags) > optiondict['despeckle_thresh'], medianlags, 0.0)).reshape(nativespaceshape), theheader,
                                 outputname + '_despecklemask_pass' + str(thepass))
            print('\n\n', voxelsprocessed_fc_ds, 'voxels despeckled in', optiondict['despeckle_passes'], 'passes')
            timings.append(
                ['Correlation despeckle end, pass ' + str(thepass), time.time(), voxelsprocessed_fc_ds, 'voxels'])

        # Step 3 - regressor refinement for next pass
        if thepass < optiondict['passes']:
            print('\n\nRegressor refinement, pass' + str(thepass))
            timings.append(['Regressor refinement start, pass ' + str(thepass), time.time(), None, None])
            if optiondict['refineoffset']:
                peaklag, peakheight, peakwidth = tide_stats.gethistprops(lagtimes[np.where(lagmask > 0)],
                                                                         optiondict['histlen'],
                                                                         pickleft=optiondict['pickleft'])
                optiondict['offsettime'] = peaklag
                optiondict['offsettime_total'] += peaklag
                print('offset time set to ', optiondict['offsettime'], ', total is ', optiondict['offsettime_total'])

            # regenerate regressor for next pass
            refineregressor_func = addmemprofiling(tide_refine.refineregressor,
                                                   optiondict['memprofile'],
                                                   memfile,
                                                   'before refineregressor')
            voxelsprocessed_rr, outputdata, refinemask = refineregressor_func(
                fmri_data_valid[:, :],
                fmritr,
                shiftedtcs,
                weights,
                thepass,
                lagstrengths,
                lagtimes,
                lagsigma,
                R2,
                theprefilter,
                optiondict,
                padtrs=numpadtrs,
                includemask=internalrefineincludemask_valid,
                excludemask=internalrefineexcludemask_valid,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype)
            normoutputdata = tide_math.stdnormalize(theprefilter.apply(fmrifreq, outputdata))
            tide_io.writenpvecs(normoutputdata, outputname + '_refinedregressor_pass' + str(thepass) + '.txt')

            if optiondict['detrendorder'] > 0:
                resampnonosref_y = tide_fit.detrend(
                    tide_resample.doresample(initial_fmri_x,
                                             normoutputdata,
                                             initial_fmri_x,
                                             method=optiondict['interptype']),
                    order=optiondict['detrendorder'],
                    demean=optiondict['dodemean'])
                resampref_y = tide_fit.detrend(
                    tide_resample.doresample(initial_fmri_x,
                                             normoutputdata,
                                             os_fmri_x,
                                             method=optiondict['interptype']),
                    order=optiondict['detrendorder'],
                    demean=optiondict['dodemean'])
            else:
                resampnonosref_y = tide_resample.doresample(initial_fmri_x,
                                                            normoutputdata,
                                                            initial_fmri_x,
                                                            method=optiondict['interptype'])
                resampref_y = tide_resample.doresample(initial_fmri_x,
                                                       normoutputdata,
                                                       os_fmri_x,
                                                       method=optiondict['interptype'])
            if optiondict['tmaskname'] is not None:
                resampnonosref_y *= tmask_y
                thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
                resampnonosref_y -= thefit[0, 1] * tmask_y
                resampref_y *= tmaskos_y
                thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
                resampref_y -= thefit[0, 1] * tmaskos_y

            # reinitialize lagtc for resampling
            genlagtc = tide_resample.fastresampler(initial_fmri_x, normoutputdata, padvalue=padvalue)
            nonosrefname = '_reference_fmrires_pass' + str(thepass + 1) + '.txt'
            osrefname = '_reference_resampres_pass' + str(thepass + 1) + '.txt'
            tide_io.writenpvecs(tide_math.stdnormalize(resampnonosref_y), outputname + nonosrefname)
            tide_io.writenpvecs(tide_math.stdnormalize(resampref_y), outputname + osrefname)
            timings.append(
                ['Regressor refinement end, pass ' + str(thepass), time.time(), voxelsprocessed_rr, 'voxels'])

    # Post refinement step 0 - Wiener deconvolution
    if optiondict['dodeconv']:
        timings.append(['Wiener deconvolution start', time.time(), None, None])
        print('\n\nWiener deconvolution')
        reportstep = 1000

        # now allocate the arrays needed for Wiener deconvolution
        wienerdeconv = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        wpeak = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)

        wienerpass_func = addmemprofiling(tide_wiener.wienerpass,
                                          optiondict['memprofile'],
                                          memfile,
                                          'before wienerpass')
        voxelsprocessed_wiener = wienerpass_func(numspatiallocs,
                                                 reportstep,
                                                 fmri_data_valid,
                                                 threshval,
                                                 optiondict,
                                                 wienerdeconv,
                                                 wpeak,
                                                 resampref_y,
                                                 rt_floatset=rt_floatset,
                                                 rt_floattype=rt_floattype
                                                 )
        timings.append(['Wiener deconvolution end', time.time(), voxelsprocessed_wiener, 'voxels'])

    # Post refinement step 1 - GLM fitting to remove moving signal
    if optiondict['doglmfilt']:
        timings.append(['GLM filtering start', time.time(), None, None])
        print('\n\nGLM filtering')
        reportstep = 1000
        if (optiondict['gausssigma'] > 0.0) or (optiondict['glmsourcefile'] is not None):
            if optiondict['glmsourcefile'] is not None:
                print('reading in ', optiondict['glmsourcefile'], 'for GLM filter, please wait')
                if optiondict['textio']:
                    nim_data = tide_io.readvecs(optiondict['glmsourcefile'])
                else:
                    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(optiondict['glmsourcefile'])
            else:
                print('rereading', fmrifilename, ' for GLM filter, please wait')
                if optiondict['textio']:
                    nim_data = tide_io.readvecs(fmrifilename)
                else:
                    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
            fmri_data_valid = (nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1])[validvoxels,
                              :] + 0.0

            # move fmri_data_valid into shared memory
            if optiondict['sharedmem']:
                print('moving fmri data to shared memory')
                timings.append(['Start moving fmri_data to shared memory', time.time(), None, None])
                numpy2shared_func = addmemprofiling(numpy2shared,
                                                    optiondict['memprofile'],
                                                    memfile,
                                                    'before movetoshared (glm)')
                fmri_data_valid, fmri_data_valid_shared, fmri_data_valid_shared_shape = numpy2shared_func(
                    fmri_data_valid, rt_floatset)
                timings.append(['End moving fmri_data to shared memory', time.time(), None, None])
            del nim_data

        # now allocate the arrays needed for GLM filtering
        meanvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitcoff = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        if optiondict['sharedmem']:
            datatoremove, dummy, dummy = allocshared(internalvalidfmrishape, rt_outfloatset)
            filtereddata, dummy, dummy = allocshared(internalvalidfmrishape, rt_outfloatset)
        else:
            datatoremove = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
            filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

        if optiondict['memprofile']:
            memcheckpoint('about to start glm noise removal...')
        else:
            tide_util.logmem('before glm', file=memfile)

        if optiondict['preservefiltering']:
            for i in range(len(validvoxels)):
                fmri_data_valid[i] = theprefilter.apply(optiondict['fmrifreq'], fmri_data_valid[i])
        glmpass_func = addmemprofiling(tide_glmpass.glmpass,
                                       optiondict['memprofile'],
                                       memfile,
                                       'before glmpass')
        voxelsprocessed_glm = glmpass_func(numvalidspatiallocs,
                                           fmri_data_valid,
                                           threshval,
                                           lagtc,
                                           meanvalue,
                                           rvalue,
                                           r2value,
                                           fitcoff,
                                           fitNorm,
                                           datatoremove,
                                           filtereddata,
                                           reportstep=reportstep,
                                           nprocs=optiondict['nprocs'],
                                           showprogressbar=optiondict['showprogressbar'],
                                           addedskip=optiondict['addedskip'],
                                           mp_chunksize=optiondict['mp_chunksize'],
                                           rt_floatset=rt_floatset,
                                           rt_floattype=rt_floattype
                                           )
        del fmri_data_valid

        timings.append(['GLM filtering end, pass ' + str(thepass), time.time(), voxelsprocessed_glm, 'voxels'])
        if optiondict['memprofile']:
            memcheckpoint('...done')
        else:
            tide_util.logmem('after glm filter', file=memfile)
        print('')
    else:
        # get the original data to calculate the mean
        print('rereading', fmrifilename, ' for GLM filter, please wait')
        if optiondict['textio']:
            nim_data = tide_io.readvecs(fmrifilename)
        else:
            nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1]
        meanvalue = np.mean(fmri_data, axis=1)


    # Post refinement step 2 - make and save interesting histograms
    timings.append(['Start saving histograms', time.time(), None, None])
    tide_stats.makeandsavehistogram(lagtimes[np.where(lagmask > 0)], optiondict['histlen'], 0, outputname + '_laghist',
                                    displaytitle='lagtime histogram', displayplots=optiondict['displayplots'],
                                    refine=False)
    tide_stats.makeandsavehistogram(lagstrengths[np.where(lagmask > 0)], optiondict['histlen'], 0,
                                    outputname + '_strengthhist',
                                    displaytitle='lagstrength histogram', displayplots=optiondict['displayplots'],
                                    therange=(0.0, 1.0))
    tide_stats.makeandsavehistogram(lagsigma[np.where(lagmask > 0)], optiondict['histlen'], 1,
                                    outputname + '_widthhist',
                                    displaytitle='lagsigma histogram', displayplots=optiondict['displayplots'])
    if optiondict['doglmfilt']:
        tide_stats.makeandsavehistogram(r2value[np.where(lagmask > 0)], optiondict['histlen'], 1, outputname + '_Rhist',
                                        displaytitle='correlation R2 histogram',
                                        displayplots=optiondict['displayplots'])
    timings.append(['Finished saving histograms', time.time(), None, None])

    # Post refinement step 3 - save out all of the important arrays to nifti files
    # write out the options used
    if optiondict['saveoptionsasjson']:
        tide_io.writedicttojson(optiondict, outputname + '_options.json')
    else:
        tide_io.writedict(optiondict, outputname + '_options.txt')

    # do ones with one time point first
    timings.append(['Start saving maps', time.time(), None, None])
    if not optiondict['textio']:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1

    # first generate the MTT map
    MTT = np.square(lagsigma) - (optiondict['acwidth'] * optiondict['acwidth'])
    MTT = np.where(MTT > 0.0, MTT, 0.0)
    MTT = np.sqrt(MTT)

    for mapname in ['lagtimes', 'lagstrengths', 'R2', 'lagsigma', 'lagmask', 'failimage', 'MTT']:
        if optiondict['memprofile']:
            memcheckpoint('about to write ' + mapname)
        else:
            tide_util.logmem('about to write ' + mapname, file=memfile)
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = eval(mapname)[:]
        if optiondict['textio']:
            tide_io.writenpvecs(outmaparray.reshape(nativespaceshape, 1),
                                outputname + '_' + mapname + outsuffix3d + '.txt')
        else:
            tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                outputname + '_' + mapname + outsuffix3d)

    if optiondict['doglmfilt']:
        for mapname, mapsuffix in [('rvalue', 'fitR'), ('r2value', 'fitR2'), ('meanvalue', 'mean'),
                                   ('fitcoff', 'fitcoff'), ('fitNorm', 'fitNorm')]:
            if optiondict['memprofile']:
                memcheckpoint('about to write ' + mapname)
            else:
                tide_util.logmem('about to write ' + mapname, file=memfile)
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = eval(mapname)[:]
            if optiondict['textio']:
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    outputname + '_' + mapsuffix + outsuffix3d + '.txt')
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                    outputname + '_' + mapsuffix + outsuffix3d)
        del rvalue
        del r2value
        del meanvalue
        del fitcoff
        del fitNorm
    else:
        for mapname, mapsuffix in [('meanvalue', 'mean')]:
            if optiondict['memprofile']:
                memcheckpoint('about to write ' + mapname)
            else:
                tide_util.logmem('about to write ' + mapname, file=memfile)
            outmaparray[:] = 0.0
            outmaparray = eval(mapname)[:]
            if optiondict['textio']:
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    outputname + '_' + mapsuffix + outsuffix3d + '.txt')
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                    outputname + '_' + mapsuffix + outsuffix3d)
        del meanvalue

    if optiondict['numestreps'] > 0:
        for i in range(0, len(thepercentiles)):
            pmask = np.where(np.abs(lagstrengths) > pcts[i], lagmask, 0 * lagmask)
            if optiondict['dosighistfit']:
                tide_io.writenpvecs(sigfit, outputname + '_sigfit' + '.txt')
            tide_io.writenpvecs(np.array([pcts[i]]), outputname + '_p_lt_' + thepvalnames[i] + '_thresh.txt')
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = pmask[:]
            if optiondict['textio']:
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    outputname + '_p_lt_' + thepvalnames[i] + '_mask' + outsuffix3d + '.txt')
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                    outputname + '_p_lt_' + thepvalnames[i] + '_mask' + outsuffix3d)

    if optiondict['passes'] > 1:
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = refinemask[:]
        if optiondict['textio']:
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_lagregressor' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                outputname + '_refinemask' + outsuffix3d)
        del refinemask

    # clean up arrays that will no longer be needed
    del lagtimes
    del lagstrengths
    del lagsigma
    del R2
    del lagmask

    # now do the ones with other numbers of time points
    if not optiondict['textio']:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3002
        else:
            theheader['dim'][4] = np.shape(corrscale)[0]
        theheader['toffset'] = corrscale[corrorigin - lagmininpts]
        theheader['pixdim'][4] = corrtr
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = gaussout[:, :]
    if optiondict['textio']:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            outputname + '_gaussout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            outputname + '_gaussout' + outsuffix4d)
    del gaussout
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = windowout[:, :]
    if optiondict['textio']:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            outputname + '_windowout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            outputname + '_windowout' + outsuffix4d)
    del windowout
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = corrout[:, :]
    if optiondict['textio']:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            outputname + '_corrout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            outputname + '_corrout' + outsuffix4d)
    del corrout

    if not optiondict['textio']:
        theheader = copy.deepcopy(nim_hdr)
        theheader['pixdim'][4] = fmritr
        theheader['toffset'] = 0.0
        if fileiscifti:
            theheader['intent_code'] = 3002
        else:
            theheader['dim'][4] = np.shape(initial_fmri_x)[0]

    if optiondict['savelagregressors']:
        outfmriarray[validvoxels, :] = lagtc[:, :]
        if optiondict['textio']:
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_lagregressor' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                outputname + '_lagregressor' + outsuffix4d)
        del lagtc

    if optiondict['passes'] > 1:
        if optiondict['savelagregressors']:
            outfmriarray[validvoxels, :] = shiftedtcs[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                    outputname + '_shiftedtcs' + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                    outputname + '_shiftedtcs' + outsuffix4d)
        del shiftedtcs

    if optiondict['doglmfilt'] and optiondict['saveglmfiltered']:
        if optiondict['savedatatoremove']:
            outfmriarray[validvoxels, :] = datatoremove[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_datatoremove' + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                outputname + '_datatoremove' + outsuffix4d)
        del datatoremove
        outfmriarray[validvoxels, :] = filtereddata[:, :]
        if optiondict['textio']:
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_filtereddata' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                outputname + '_filtereddata' + outsuffix4d)
        del filtereddata

    timings.append(['Finished saving maps', time.time(), None, None])
    memfile.close()
    print('done')

    if optiondict['displayplots']:
        show()
    timings.append(['Done', time.time(), None, None])

    # Post refinement step 5 - process and save timing information
    nodeline = ' '.join(['Processed on',
                         platform.node(),
                         '(',
                         optiondict['release_version'] + ',',
                         optiondict['git_date'],
                         ')'])
    tide_util.proctiminginfo(timings, outputfile=outputname + '_runtimings.txt', extraheader=nodeline)

if __name__ == '__main__':
    rapidtide_main()
