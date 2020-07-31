#!/usr/bin/env python
# -*- coding: latin-1 -*-
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

import bisect
import getopt
import multiprocessing as mp
import os
import platform
import sys
import time
import warnings

import numpy as np
from matplotlib.pyplot import figure, plot, show
from scipy import ndimage
from statsmodels.tsa.stattools import pacf_yw

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

import copy

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


def addmemprofiling(thefunc, memprofile, memfile, themessage):
    if memprofile:
        return profile(thefunc, precision=2)
    else:
        tide_util.logmem(themessage, file=memfile)
        return thefunc


def usage():
    print("usage: ", os.path.basename(sys.argv[0]), " datafilename outputname ")
    print(' '.join([
        "[-r LAGMIN,LAGMAX]",
        "[-s SIGMALIMIT]",
        "[-a]",
        "[--nowindow]",
        "[--phat]", "[--liang]", "[--eckart]",
        "[-f GAUSSSIGMA]",
        "[-O oversampfac]",
        "[-t TSTEP]", "[--datatstep=TSTEP]", "[--datafreq=FREQ]",
        "[-d]",
        "[-b]",
        "[-V]", "[-L]", "[-R]", "[-C]", "[-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]]",
        "[-o OFFSETTIME]",
        "[-T]",
        "[-p]",
        "[-P]",
        "[-B]",
        "[-h HISTLEN]",
        "[-i INTERPTYPE]",
        "[-I]",
        "[-Z DELAYTIME]",
        "[--nofitfilt]",
        "[--searchfrac=SEARCHFRAC]",
        "[-N NREPS]",
        "[--motionfile=MOTFILE]",
        "[--pickleft]",
        "[--numskip=SKIP]",
        "[--refineweighting=TYPE]",
        "[--refineprenorm=TYPE]",
        "[--passes=PASSES]",
        "[--refinepasses=PASSES]",
        "[--excluderefine=MASK]", "[--includerefine=MASK]", "[--includemean=MASK]", "[--excludemean=MASK]"
        "[--lagminthresh=MIN]", "[--lagmaxthresh=MAX]", "[--ampthresh=AMP]", "[--sigmathresh=SIGMA]",
        "[--corrmask=MASK]",
        "[--corrmaskthresh=PCT]",
        "[--refineoffset]",
        "[--pca]", "[--ica]", "[--weightedavg]", "[--avg]",
        "[--psdfilter]",
        "[--noprogressbar]",
        "[--despecklethresh=VAL]", "[--despecklepasses=PASSES]",
        "[--dispersioncalc]",
        "[--refineupperlag]", "[--refinelowerlag]",
        "[--nosharedmem]",
        "[--tmask=MASKFILE]",
        "[--limitoutput]",
        "[--motionfile=FILENAME[:COLSPEC]",
        "[--nogaussrefine]",
        "[--softlimit]",
        "[--timerange=START,END]",
        "[--skipsighistfit]",
        "[--accheck]", "[--acfix]"
                       "[--numskip=SKIP]",
        "[--slicetimes=FILE]",
        "[--glmsourcefile=FILE]",
        "[--regressorfreq=FREQ]", "[--regressortstep=TSTEP]" "[--regressor=FILENAME]", "[--regressorstart=STARTTIME]",
        "[--usesp]",
        "[--maxfittype=FITTYPE]",
        "[--mklthreads=NTHREADS]",
        "[--nprocs=NPROCS]",
        "[--nirs]",
        "[--venousrefine]"]))
    print("")
    print("Required arguments:")
    print("    datafilename               - The input data file (BOLD fmri file or NIRS)")
    print("    outputname                 - The root name for the output files")
    print("")
    print("Optional arguments:")
    print("    Arguments are processed in order of appearance.  Later options can override ones earlier on")
    print("    the command line")
    print("")
    print("Macros:")
    print("    --venousrefine                 - This is a macro that sets --lagminthresh=2.5, --lagmaxthresh=6.0,")
    print("                                     --ampthresh=0.5, and --refineupperlag to bias refinement towards ")
    print("                                     voxels in the draining vasculature for an fMRI scan.")
    print("    --nirs                         - This is a NIRS analysis - this is a macro that sets --nothresh,")
    print("                                     --preservefiltering, --refinenorm=var, --ampthresh=0.7, ")
    print("                                     and --lagminthresh=0.1.")
    print("")
    print("Preprocessing options:")
    print("    -t TSTEP,                      - Set the timestep of the data file to TSTEP (or 1/FREQ)")
    print("      --datatstep=TSTEP,             This will override the TR in an fMRI file.")
    print("      --datafreq=FREQ                NOTE: if using data from a text file, for example with")
    print("                                     NIRS data, using one of these options is mandatory.")
    print("    -a                             - Disable antialiasing filter")
    print("    --detrendorder=ORDER           - Set order of trend removal (0 to disable, default is 1 - linear)")
    print("    -I                             - Invert the sign of the regressor before processing")
    print("    -i                             - Use specified interpolation type (options are 'cubic',")
    print("                                     'quadratic', and 'univariate (default)')")
    print("    -o                             - Apply an offset OFFSETTIME to the lag regressors")
    print("    -b                             - Use butterworth filter for band splitting instead of")
    print("                                     trapezoidal FFT filter")
    print("    -F                             - Filter data and regressors from LOWERFREQ to UPPERFREQ.")
    print("                                     LOWERSTOP and UPPERSTOP can be specified, or will be")
    print("                                     calculated automatically")
    print("    -V                             - Filter data and regressors to VLF band")
    print("    -L                             - Filter data and regressors to LFO band")
    print("    -R                             - Filter data and regressors to respiratory band")
    print("    -C                             - Filter data and regressors to cardiac band")
    print("    -N                             - Estimate significance threshold by running NREPS null ")
    print("                                     correlations (default is 10000, set to 0 to disable)")
    print("    --permutationmethod=METHOD     - Method for permuting the regressor for significance estimation.  Default")
    print("                                     is shuffle")
    print("    --skipsighistfit               - Do not fit significance histogram with a Johnson SB function")
    print("    --windowfunc=FUNC              - Use FUNC window funcion prior to correlation.  Options are")
    print("                                     hamming (default), hann, blackmanharris, and None")
    print("    --nowindow                     - Disable precorrelation windowing")
    print("    -f GAUSSSIGMA                  - Spatially filter fMRI data prior to analysis using ")
    print("                                     GAUSSSIGMA in mm")
    print("    -M                             - Generate a global mean regressor and use that as the ")
    print("                                     reference regressor")
    print("    --globalmeaninclude=MASK[:VALSPEC]")
    print("                                   - Only use voxels in NAME for global regressor generation (if VALSPEC is")
    print("                                     given, only voxels with integral values listed in VALSPEC are used.)")
    print("    --globalmeanexclude=MASK[:VALSPEC]")
    print("                                   - Do not use voxels in NAME for global regressor generation (if VALSPEC is")
    print("                                     given, only voxels with integral values listed in VALSPEC are used.)")
    print("    -m                             - Mean scale regressors during global mean estimation")
    print("    --slicetimes=FILE              - Apply offset times from FILE to each slice in the dataset")
    print("    --numskip=SKIP                 - SKIP tr's were previously deleted during preprocessing")
    print("                                     (default is 0)")
    print("    --nothresh                     - Disable voxel intensity threshold (especially useful")
    print("                                     for NIRS data)")
    print("    --motionfile=MOTFILE[:COLSPEC] - Read 6 columns of motion regressors out of MOTFILE text file.")
    print("                                     (with timepoints rows) and regress their derivatives")
    print("                                     and delayed derivatives out of the data prior to analysis.")
    print("                                     If COLSPEC is present, use the comma separated list of ranges to")
    print("                                     specify X, Y, Z, RotX, RotY, and RotZ, in that order.  For")
    print("                                     example, :3-5,7,0,9 would use columns 3, 4, 5, 7, 0 and 9")
    print("                                     for X, Y, Z, RotX, RotY, RotZ, respectively")
    print("    --motpos                       - Toggle whether displacement regressors will be used in motion regression.")
    print("                                     Default is False.")
    print("    --motderiv                     - Toggle whether derivatives will be used in motion regression.")
    print("                                     Default is True.")
    print("    --motdelayderiv                - Toggle whether delayed derivative  regressors will be used in motion regression.")
    print("                                     Default is False.")
    print("")
    print("Correlation options:")
    print("    -O OVERSAMPFAC                 - Oversample the fMRI data by the following integral ")
    print("                                     factor.  Setting to -1 chooses the factor automatically (default)")
    print("    --regressor=FILENAME           - Read probe regressor from file FILENAME (if none ")
    print("                                     specified, generate and use global regressor)")
    print("    --regressorfreq=FREQ           - Probe regressor in file has sample frequency FREQ ")
    print("                                     (default is 1/tr) NB: --regressorfreq and --regressortstep")
    print("                                     are two ways to specify the same thing")
    print("    --regressortstep=TSTEP         - Probe regressor in file has sample time step TSTEP ")
    print("                                     (default is tr) NB: --regressorfreq and --regressortstep")
    print("                                     are two ways to specify the same thing")
    print("    --regressorstart=START         - The time delay in seconds into the regressor file, corresponding")
    print("                                     in the first TR of the fmri file (default is 0.0)")
    print("    --phat                         - Use generalized cross-correlation with phase alignment ")
    print("                                     transform (PHAT) instead of correlation")
    print("    --liang                        - Use generalized cross-correlation with Liang weighting function")
    print("                                     (Liang, et al, doi:10.1109/IMCCC.2015.283)")
    print("    --eckart                       - Use generalized cross-correlation with Eckart weighting function")
    print("    --corrmaskthresh=PCT           - Do correlations in voxels where the mean exceeeds this ")
    print("                                     percentage of the robust max (default is 1.0)")
    print("    --corrmask=MASK                - Only do correlations in voxels in MASK (if set, corrmaskthresh")
    print("                                     is ignored).")
    print("    --accheck                      - Check for periodic components that corrupt the autocorrelation")
    print("")
    print("Correlation fitting options:")
    print("    -Z DELAYTIME                   - Don't fit the delay time - set it to DELAYTIME seconds ")
    print("                                     for all voxels")
    print("    -r LAGMIN,LAGMAX               - Limit fit to a range of lags from LAGMIN to LAGMAX")
    print("    -s SIGMALIMIT                  - Reject lag fits with linewidth wider than SIGMALIMIT")
    print("    -B                             - Bipolar mode - match peak correlation ignoring sign")
    print("    --nofitfilt                    - Do not zero out peak fit values if fit fails")
    print("    --searchfrac=FRAC              - When peak fitting, include points with amplitude > FRAC * the")
    print("                                     maximum amplitude.")
    print("                                     (default value is 0.5)")
    print("    --maxfittype=FITTYPE           - Method for fitting the correlation peak (default is 'gauss'). ")
    print("                                     'quad' uses a quadratic fit.  Faster but not as well tested")
    print("    --despecklepasses=PASSES       - detect and refit suspect correlations to disambiguate peak")
    print("                                     locations in PASSES passes")
    print("    --despecklethresh=VAL          - refit correlation if median discontinuity magnitude exceeds")
    print("                                     VAL (default is 5s)")
    print("    --softlimit                    - Allow peaks outside of range if the maximum correlation is")
    print("                                     at an edge of the range.")
    print("    --nogaussrefine                - Use initial guess at peak parameters - do not perform fit to Gaussian.")
    print("")
    print("Regressor refinement options:")
    print("    --refineprenorm=TYPE           - Apply TYPE prenormalization to each timecourse prior ")
    print("                                     to refinement (valid weightings are 'None', ")
    print("                                     'mean' (default), 'var', and 'std'")
    print("    --refineweighting=TYPE         - Apply TYPE weighting to each timecourse prior ")
    print("                                     to refinement (valid weightings are 'None', ")
    print("                                     'R', 'R2' (default)")
    print("    --passes=PASSES,               - Set the number of processing passes to PASSES ")
    print("     --refinepasses=PASSES           (default is 1 pass - no refinement).")
    print("                                     NB: refinepasses is the wrong name for this option -")
    print("                                     --refinepasses is deprecated, use --passes from now on.")
    print("    --refineinclude=MASK[:VALSPEC] - Only use nonzero voxels in MASK for regressor refinement (if VALSPEC is")
    print("                                     given, only voxels with integral values listed in VALSPEC are used.)")
    print("    --refineexclude=MASK[:VALSPEC] - Do not use nonzero voxels in MASK for regressor refinement (if VALSPEC is")
    print("                                     given, only voxels with integral values listed in VALSPEC are used.)")
    print("    --lagminthresh=MIN             - For refinement, exclude voxels with delays less ")
    print("                                     than MIN (default is 0.5s)")
    print("    --lagmaxthresh=MAX             - For refinement, exclude voxels with delays greater ")
    print("                                     than MAX (default is 5s)")
    print("    --ampthresh=AMP                - For refinement, exclude voxels with correlation ")
    print("                                     coefficients less than AMP (default is 0.3)")
    print("    --sigmathresh=SIGMA            - For refinement, exclude voxels with widths greater ")
    print("                                     than SIGMA (default is 100s)")
    print("    --refineoffset                 - Adjust offset time during refinement to bring peak ")
    print("                                     delay to zero")
    print("    --pickleft                     - When setting refineoffset, always select the leftmost histogram peak")
    print("    --refineupperlag               - Only use positive lags for regressor refinement")
    print("    --refinelowerlag               - Only use negative lags for regressor refinement")
    print("    --pca                          - Use pca to derive refined regressor (default is ")
    print("                                     unweighted averaging)")
    print("    --ica                          - Use ica to derive refined regressor (default is ")
    print("                                     unweighted averaging)")
    print("    --weightedavg                  - Use weighted average to derive refined regressor ")
    print("                                     (default is unweighted averaging)")
    print("    --avg                          - Use unweighted average to derive refined regressor ")
    print("                                     (default)")
    print("    --psdfilter                    - Apply a PSD weighted Wiener filter to shifted")
    print("                                     timecourses prior to refinement")
    print("")
    print("Output options:")
    print("    --limitoutput                  - Don't save some of the large and rarely used files")
    print("    -T                             - Save a table of lagtimes used")
    print("    -h HISTLEN                     - Change the histogram length to HISTLEN (default is")
    print("                                     100)")
    print("    --timerange=START,END          - Limit analysis to data between timepoints START ")
    print("                                     and END in the fmri file")
    print("    --glmsourcefile=FILE           - Regress delayed regressors out of FILE instead of the ")
    print("                                     initial fmri file used to estimate delays")
    print("    --noglm                        - Turn off GLM filtering to remove delayed regressor ")
    print("                                     from each voxel (disables output of fitNorm)")
    print("    --preservefiltering            - don't reread data prior to GLM")
    print("")
    print("Miscellaneous options:")
    print("    --noprogressbar                - Disable progress bars - useful if saving output to files")
    print("    --wiener                       - Perform Wiener deconvolution to get voxel transfer functions")
    print("    --usesp                        - Use single precision for internal calculations (may")
    print("                                     be useful when RAM is limited)")
    print("    -c                             - Data file is a converted CIFTI")
    print("    -S                             - Simulate a run - just report command line options")
    print("    -d                             - Display plots of interesting timecourses")
    print("    --nonumba                      - Disable jit compilation with numba")
    print("    --nosharedmem                  - Disable use of shared memory for large array storage")
    print("    --memprofile                   - Enable memory profiling for debugging - warning:")
    print("                                     this slows things down a lot.")
    print("    --multiproc                    - Enable multiprocessing versions of key subroutines.  This")
    print("                                     speeds things up dramatically.  Almost certainly will NOT")
    print("                                     work on Windows (due to different forking behavior).")
    print("    --mklthreads=NTHREADS          - Use no more than NTHREADS worker threads in accelerated numpy calls.")
    print("    --nprocs=NPROCS                - Use NPROCS worker processes for multiprocessing.  Setting NPROCS")
    print("                                     less than 1 sets the number of worker processes to")
    print("                                     n_cpus - 1 (default).  Setting NPROCS enables --multiproc.")
    print("    --debug                        - Enable additional information output")
    print("    --saveoptionsasjson            - Save the options file in json format rather than text.  Will eventually")
    print("                                     become the default, but for now I'm just trying it out.")
    print("")
    print("Experimental options (not fully tested, may not work):")
    print("    --cleanrefined                 - perform additional processing on refined regressor to remove spurious")
    print("                                     components.")
    print("    --dispersioncalc               - Generate extra data during refinement to allow calculation of")
    print("                                     dispersion.")
    print("    --acfix                        - Perform a secondary correlation to disambiguate peak location")
    print("                                     (enables --accheck).  Experimental.")
    print("    --tmask=MASKFILE               - Only correlate during epochs specified in ")
    print("                                     MASKFILE (NB: if file has one colum, the length needs to match")
    print("                                     the number of TRs used.  TRs with nonzero values will be used")
    print("                                     in analysis.  If there are 2 or more columns, each line of MASKFILE")
    print("                                     contains the time (first column) and duration (second column) of an")
    print("                                     epoch to include.)")
    return ()


def process_args():
    nargs = len(sys.argv)
    if nargs < 3:
        usage()
        exit()

    # set default variable values
    optiondict = {}

    # file i/o file options
    optiondict['isgrayordinate'] = False
    optiondict['textio'] = False

    # preprocessing options
    optiondict['dogaussianfilter'] = False  # apply a spatial filter to the fmri data prior to analysis
    optiondict['gausssigma'] = 0.0  # the width of the spatial filter kernel in mm
    optiondict['antialias'] = True  # apply an antialiasing filter to any regressors prior to filtering
    optiondict['invertregressor'] = False  # invert the initial regressor during startup
    optiondict['slicetimes'] = None  # do not apply any slice order correction by default
    optiondict['startpoint'] = -1  # by default, analyze the entire length of the dataset
    optiondict['endpoint'] = 10000000  # by default, analyze the entire length of the dataset
    optiondict['preprocskip'] = 0  # number of trs skipped in preprocessing
    optiondict['meanscaleglobal'] = False
    optiondict['globalmaskmethod'] = 'mean'
    optiondict['globalmeanexcludename'] = None
    optiondict['globalmeanexcludevals'] = None    # list of integer values to use in the mask
    optiondict['globalmeanincludename'] = None
    optiondict['globalmeanincludevals'] = None    # list of integer values to use in the mask

    # correlation options
    optiondict['dodemean'] = True  # remove the mean from signals prior to correlation
    optiondict['detrendorder'] = 1  # remove linear trends prior to correlation
    optiondict['usewindowfunc'] = True  # apply a window prior to correlation
    optiondict['windowfunc'] = 'hamming'  # the particular window function to use for correlation
    optiondict['corrweighting'] = 'none'  # use a standard unweighted crosscorrelation for calculate time delays
    optiondict['usetmask'] = False  # premultiply the regressor with the tmask timecourse
    optiondict['tmaskname'] = None  # file name for tmask regressor
    optiondict['corrmaskthreshpct'] = 1.0  # percentage of robust maximum of mean to mask correlations
    optiondict['corrmaskname'] = None  # name of correlation mask
    optiondict['corrmaskvals'] = None  # list of integer values in the correlation mask file to use in the mask
    optiondict[
        'check_autocorrelation'] = True  # check for periodic components that corrupt the autocorrelation
    optiondict[
        'fix_autocorrelation'] = False  # remove periodic components that corrupt the autocorrelation
    optiondict[
        'despeckle_thresh'] = 5.0  # threshold value - despeckle if median discontinuity magnitude exceeds despeckle_thresh
    optiondict['despeckle_passes'] = 0  # despeckling passes to perform
    optiondict['nothresh'] = False  # disable voxel intensity threshholding

    # correlation fitting options
    optiondict['hardlimit'] = True  # Peak value must be within specified range.  If false, allow max outside if maximum
    # correlation value is that one end of the range.
    optiondict['bipolar'] = False  # find peak with highest magnitude, regardless of sign
    optiondict['gaussrefine'] = True  # fit gaussian after initial guess at parameters
    optiondict['fastgauss'] = False  # use a non-iterative gaussian peak fit (DOES NOT WORK)
    optiondict['lthreshval'] = 0.0  # zero out peaks with correlations lower than this value
    optiondict['uthreshval'] = 1.0  # zero out peaks with correlations higher than this value
    optiondict['edgebufferfrac'] = 0.0  # what fraction of the correlation window to avoid on either end when fitting
    optiondict['enforcethresh'] = True  # only do fits in voxels that exceed threshhold
    optiondict['zerooutbadfit'] = True  # if true zero out all fit parameters if the fit fails
    optiondict['searchfrac'] = 0.5  # The fraction of the main peak over which points are included in the peak
    optiondict['lagmod'] = 1000.0  # if set to the location of the first autocorrelation sidelobe, this should
    optiondict['findmaxtype'] = 'gauss'  # if set to 'gauss', use old gaussian fitting, if set to 'quad' use parabolic
    optiondict['acwidth'] = 0.0  # width of the reference autocorrelation function
    optiondict['absmaxsigma'] = 100.0  # width of the reference autocorrelation function
    optiondict['absminsigma'] = 0.25  # width of the reference autocorrelation function
    #     move delay peaks back to the correct position if they hit a sidelobe

    # postprocessing options
    optiondict['doglmfilt'] = True  # use a glm filter to remove the delayed regressor from the data in each voxel
    optiondict['preservefiltering'] = False
    optiondict[
        'glmsourcefile'] = None  # name of the file from which to regress delayed regressors (if not the original data)
    optiondict['dodeconv'] = False  # do Wiener deconvolution to find voxel transfer function
    optiondict['motionfilename'] = None  # by default do no motion regression
    optiondict['mot_pos'] = False  # do not do position
    optiondict['mot_deriv'] = True  # do use derivative
    optiondict['mot_delayderiv'] = False  # do not do delayed derivative
    optiondict['savemotionfiltered'] = False  # save motion filtered file for debugging

    # filter options
    optiondict['usebutterworthfilter'] = False
    optiondict['filtorder'] = 6
    optiondict['padseconds'] = 30.0  # the number of seconds of padding to add to each end of a filtered timecourse
    optiondict['filtertype'] = None
    optiondict['respdelete'] = False
    optiondict['lowerstop'] = None
    optiondict['lowerpass'] = None
    optiondict['upperpass'] = None
    optiondict['upperstop'] = None

    # output options
    optiondict['savelagregressors'] = True
    optiondict['savedatatoremove'] = True
    optiondict['saveglmfiltered'] = True
    optiondict['savecorrtimes'] = False
    optiondict['saveoptionsasjson'] = False

    optiondict['interptype'] = 'univariate'
    optiondict['useglobalref'] = False
    optiondict['fixdelay'] = False
    optiondict['fixeddelayvalue'] = 0.0

    # significance estimation options
    optiondict['numestreps'] = 10000  # the number of sham correlations to perform to estimate significance
    optiondict['permutationmethod'] = 'shuffle'
    optiondict['nohistzero'] = False  # if False, there is a spike at R=0 in the significance histogram
    optiondict['ampthreshfromsig'] = True
    optiondict['sighistlen'] = 1000
    optiondict['dosighistfit'] = True

    optiondict['histlen'] = 250
    optiondict['oversampfactor'] = -1
    optiondict['lagmin'] = -30.0
    optiondict['lagmax'] = 30.0
    optiondict['widthlimit'] = 100.0
    optiondict['offsettime'] = 0.0
    optiondict['offsettime_total'] = 0.0
    optiondict['addedskip'] = 0

    # refinement options
    optiondict['cleanrefined'] = False
    optiondict['lagmaskside'] = 'both'
    optiondict['refineweighting'] = 'R2'
    optiondict['refineprenorm'] = 'mean'
    optiondict['sigmathresh'] = 100.0
    optiondict['lagminthresh'] = 0.5
    optiondict['lagmaxthresh'] = 5.0
    optiondict['ampthresh'] = 0.3
    optiondict['passes'] = 1
    optiondict['refineoffset'] = False
    optiondict['pickleft'] = False
    optiondict['refineexcludename'] = None
    optiondict['refineexcludevals'] = None        # list of integer values to use in the mask
    optiondict['refineincludename'] = None
    optiondict['refineincludevals'] = None        # list of integer values to use in the mask
    optiondict['corrmaskvallist'] = None
    optiondict['refinetype'] = 'unweighted_average'
    optiondict['estimatePCAdims'] = False
    optiondict['filterbeforePCA'] = True
    optiondict['fmrifreq'] = 0.0
    optiondict['dodispersioncalc'] = False
    optiondict['dispersioncalc_lower'] = -4.0
    optiondict['dispersioncalc_upper'] = 4.0
    optiondict['dispersioncalc_step'] = 0.50
    optiondict['psdfilter'] = False

    # debugging options
    optiondict['internalprecision'] = 'double'
    optiondict['outputprecision'] = 'single'
    optiondict['nonumba'] = False
    optiondict['memprofile'] = False
    optiondict['sharedmem'] = True
    optiondict['fakerun'] = False
    optiondict['displayplots'] = False
    optiondict['debug'] = False
    optiondict['verbose'] = False
    optiondict['release_version'], \
    optiondict['git_longtag'], \
    optiondict['git_date'],\
    optiondict['git_isdirty'] = tide_util.version()
    optiondict['python_version'] = str(sys.version_info)
    optiondict['nprocs'] = 1
    optiondict['mklthreads'] = 1
    optiondict['mp_chunksize'] = 50000
    optiondict['showprogressbar'] = True
    optiondict['savecorrmask'] = True
    optiondict['savedespecklemasks'] = True
    optiondict['checkpoint'] = False                    # save checkpoint information for tracking program state

    # package options
    optiondict['memprofilerexists'] = memprofilerexists

    realtr = 0.0
    theprefilter = tide_filt.noncausalfilter()
    theprefilter.setbutter(optiondict['usebutterworthfilter'], optiondict['filtorder'])

    # start the clock!
    tide_util.checkimports(optiondict)

    # get the command line parameters
    optiondict['regressorfile'] = None
    optiondict['inputfreq'] = None
    optiondict['inputstarttime'] = None
    if len(sys.argv) < 3:
        usage()
        sys.exit()
    # handle required args first
    optiondict['fmrifilename'] = sys.argv[1]
    optiondict['outputname'] = sys.argv[2]
    optparsestart = 3

    # now scan for optional arguments
    try:
        opts, args = getopt.getopt(sys.argv[optparsestart:], 'abcdf:gh:i:mo:s:r:t:vBCF:ILMN:O:RSTVZ:', ['help',
                                                                                                          'nowindow',
                                                                                                          'windowfunc=',
                                                                                                          'datatstep=',
                                                                                                          'datafreq=',
                                                                                                          'lagminthresh=',
                                                                                                          'lagmaxthresh=',
                                                                                                          'ampthresh=',
                                                                                                          'skipsighistfit',
                                                                                                          'sigmathresh=',
                                                                                                          'refineweighting=',
                                                                                                          'refineprenorm=',
                                                                                                          'corrmaskthresh=',
                                                                                                          'despecklepasses=',
                                                                                                          'despecklethresh=',
                                                                                                          'accheck',
                                                                                                          'acfix',
                                                                                                          'noprogressbar',
                                                                                                          'refinepasses=',
                                                                                                          'passes=',
                                                                                                          'corrmask=',
                                                                                                          'motionfile=',
                                                                                                          'motpos',
                                                                                                          'motderiv',
                                                                                                          'motdelayderiv',
                                                                                                          'globalmeaninclude=',
                                                                                                          'globalmeanexclude=',
                                                                                                          'refineinclude=',
                                                                                                          'refineexclude=',
                                                                                                          'refineoffset',
                                                                                                          'pickleft',
                                                                                                          'nofitfilt',
                                                                                                          'cleanrefined',
                                                                                                          'pca',
                                                                                                          'ica',
                                                                                                          'weightedavg',
                                                                                                          'avg',
                                                                                                          'psdfilter',
                                                                                                          'saveoptionsasjson',
                                                                                                          'dispersioncalc',
                                                                                                          'noglm',
                                                                                                          'nosharedmem',
                                                                                                          'multiproc',
                                                                                                          'mklthreads=',
                                                                                                          'permutationmethod=',
                                                                                                          'nprocs=',
                                                                                                          'debug',
                                                                                                          'nonumba',
                                                                                                          'savemotionglmfilt',
                                                                                                          'tmask=',
                                                                                                          'detrendorder=',
                                                                                                          'slicetimes=',
                                                                                                          'glmsourcefile=',
                                                                                                          'preservefiltering',
                                                                                                          'globalmaskmethod=',
                                                                                                          'numskip=',
                                                                                                          'nirs',
                                                                                                          'venousrefine',
                                                                                                          'nothresh',
                                                                                                          'searchfrac=',
                                                                                                          'limitoutput',
                                                                                                          'softlimit',
                                                                                                          'regressor=',
                                                                                                          'regressorfreq=',
                                                                                                          'regressortstep=',
                                                                                                          'regressorstart=',
                                                                                                          'timerange=',
                                                                                                          'refineupperlag',
                                                                                                          'refinelowerlag',
                                                                                                          'fastgauss',
                                                                                                          'memprofile',
                                                                                                          'nogaussrefine',
                                                                                                          'usesp',
                                                                                                          'liang',
                                                                                                          'eckart',
                                                                                                          'phat',
                                                                                                          'wiener',
                                                                                                          'weiner',
                                                                                                          'respdelete',
                                                                                                          'checkpoint',
                                                                                                          'maxfittype='])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like 'option -a not recognized'
        usage()
        sys.exit(2)

    formattedcmdline = [sys.argv[0] + ' \\']
    for thearg in range(1, optparsestart):
        formattedcmdline.append('\t' + sys.argv[thearg] + ' \\')

    for o, a in opts:
        linkchar = ' '
        if o == '--nowindow':
            optiondict['usewindowfunc'] = False
            print('disable precorrelation windowing')
        elif o == '--checkpoint':
            optiondict['checkpoint'] = True
            print('Enabled run checkpoints')
        elif o == '--permutationmethod':
            themethod = a
            if (themethod != 'shuffle') and (themethod != 'phaserandom'):
                print('illegal permutation method', themethod)
                sys.exit()
            optiondict['permutationmethod'] = themethod
            linkchar = '='
            print('Will use', optiondict['permutationmethod'], 'as the permutation method for calculating null correlation threshold')
        elif o == '--windowfunc':
            optiondict['usewindowfunc'] = True
            thewindow = a
            if (thewindow != 'hamming') and (thewindow != 'hann') and (thewindow != 'blackmanharris') and (
                    thewindow != 'None'):
                print('illegal window function', thewindow)
                sys.exit()
            optiondict['windowfunc'] = thewindow
            linkchar = '='
            print('Will use', optiondict['windowfunc'], 'as the window function for correlation')
        elif o == '-v':
            optiondict['verbose'] = True
            print('Turned on verbose mode')
        elif o == '--liang':
            optiondict['corrweighting'] = 'Liang'
            print('Enabled Liang weighted crosscorrelation')
        elif o == '--eckart':
            optiondict['corrweighting'] = 'Eckart'
            print('Enabled Eckart weighted crosscorrelation')
        elif o == '--phat':
            optiondict['corrweighting'] = 'PHAT'
            print('Enabled GCC-PHAT fitting')
        elif o == '--weiner':
            print('It\'s spelled wiener, not weiner')
            print('The filter is named after Norbert Wiener, an MIT mathematician.')
            print('The name probably indicates that his family came from Vienna.')
            print('Spell it right and try again.  I mean, I know what you meant, and could just')
            print('call the routine you wanted anyway, but then how would you learn?')
            sys.exit()
        elif o == '--cleanrefined':
            optiondict['cleanrefined'] = True
            print('Will attempt to clean refined regressor')
        elif o == '--respdelete':
            optiondict['respdelete'] = True
            print('Will attempt to track and delete respiratory waveforms in the passband')
        elif o == '--wiener':
            optiondict['dodeconv'] = True
            print('Will perform Wiener deconvolution')
        elif o == '--usesp':
            optiondict['internalprecision'] = 'single'
            print('Will use single precision for internal calculations')
        elif o == '--preservefiltering':
            optiondict['preservefiltering'] = True
            print('Will not reread input file prior to GLM')
        elif o == '--glmsourcefile':
            optiondict['glmsourcefile'] = a
            linkchar = '='
            print('Will regress delayed regressors out of', optiondict['glmsourcefile'])
        elif o == '--corrmaskthresh':
            optiondict['corrmaskthreshpct'] = float(a)
            linkchar = '='
            print('Will perform correlations in voxels where mean exceeds', optiondict['corrmaskthreshpct'],
                  '% of robust maximum')
        elif o == '-I':
            optiondict['invertregressor'] = True
            print('Invert the regressor prior to running')
        elif o == '-B':
            optiondict['bipolar'] = True
            print('Enabled bipolar correlation fitting')
        elif o == '-S':
            optiondict['fakerun'] = True
            print('report command line options and quit')
        elif o == '-a':
            optiondict['antialias'] = False
            print('antialiasing disabled')
        elif o == '-M':
            optiondict['useglobalref'] = True
            print('using global mean timecourse as the reference regressor')
        elif o == '--globalmaskmethod':
            optiondict['globalmaskmethod'] = a
            if optiondict['globalmaskmethod'] == 'mean':
                print('will use mean value to mask voxels prior to generating global mean')
            elif optiondict['globalmaskmethod'] == 'variance':
                print('will use timecourse variance to mask voxels prior to generating global mean')
            else:
                print(optiondict['globalmaskmethod'],
                      'is not a valid masking method.  Valid methods are \'mean\' and \'variance\'')
                sys.exit()
        elif o == '-m':
            optiondict['meanscaleglobal'] = True
            print('mean scale voxels prior to generating global mean')
        elif o == '--limitoutput':
            optiondict['savelagregressors'] = False
            optiondict['savedatatoremove'] = False
            print('disabling output of lagregressors and some ancillary GLM timecourses')
        elif o == '--debug':
            optiondict['debug'] = True
            theprefilter.setdebug(optiondict['debug'])
            print('enabling additional data output for debugging')
        elif o == '--multiproc':
            optiondict['nprocs'] = -1
            print('enabling multiprocessing')
        elif o == '--softlimit':
            optiondict['hardlimit'] = False
            linkchar = '='
            print('will relax peak lag constraint for maximum correlations at edge of range')
        elif o == '--nosharedmem':
            optiondict['sharedmem'] = False
            linkchar = '='
            print('will not use shared memory for large array storage')
        elif o == '--mklthreads':
            optiondict['mklthreads'] = int(a)
            linkchar = '='
            if mklexists:
                print('will use', optiondict['mklthreads'], 'MKL threads for accelerated numpy processing.')
            else:
                print('MKL not present - ignoring --mklthreads')
        elif o == '--nprocs':
            optiondict['nprocs'] = int(a)
            linkchar = '='
            if optiondict['nprocs'] < 0:
                print('will use n_cpus - 1 processes for calculation')
            else:
                print('will use', optiondict['nprocs'], 'processes for calculation')
        elif o == '--saveoptionsasjson':
            optiondict['saveoptionsasjson'] = True
            print('saving options file as json rather than text')
        elif o == '--savemotionglmfilt':
            optiondict['savemotionfiltered'] = True
            print('saveing motion filtered data')
        elif o == '--nonumba':
            optiondict['nonumba'] = True
            print('disabling numba if present')
        elif o == '--memprofile':
            if memprofilerexists:
                optiondict['memprofile'] = True
                print('enabling memory profiling')
            else:
                print('cannot enable memory profiling - memory_profiler module not found')
        elif o == '--noglm':
            optiondict['doglmfilt'] = False
            print('disabling GLM filter')
        elif o == '-T':
            optiondict['savecorrtimes'] = True
            print('saving a table of correlation times used')
        elif o == '-V':
            theprefilter.settype('vlf')
            print('prefiltering to vlf band')
        elif o == '-L':
            theprefilter.settype('lfo')
            optiondict['filtertype'] = 'lfo'
            optiondict['despeckle_thresh'] = np.max(
                [optiondict['despeckle_thresh'], 0.5 / (theprefilter.getfreqs()[2])])
            print('prefiltering to lfo band')
        elif o == '-R':
            theprefilter.settype('resp')
            optiondict['filtertype'] = 'resp'
            optiondict['despeckle_thresh'] = np.max(
                [optiondict['despeckle_thresh'], 0.5 / (theprefilter.getfreqs()[2])])
            print('prefiltering to respiratory band')
        elif o == '-C':
            theprefilter.settype('cardiac')
            optiondict['filtertype'] = 'cardiac'
            optiondict['despeckle_thresh'] = np.max(
                [optiondict['despeckle_thresh'], 0.5 / (theprefilter.getfreqs()[2])])
            print('prefiltering to cardiac band')
        elif o == '-F':
            arbvec = a.split(',')
            if len(arbvec) != 2 and len(arbvec) != 4:
                usage()
                sys.exit()
            if len(arbvec) == 2:
                optiondict['arb_lower'] = float(arbvec[0])
                optiondict['arb_upper'] = float(arbvec[1])
                optiondict['arb_lowerstop'] = 0.9 * float(arbvec[0])
                optiondict['arb_upperstop'] = 1.1 * float(arbvec[1])
            if len(arbvec) == 4:
                optiondict['arb_lower'] = float(arbvec[0])
                optiondict['arb_upper'] = float(arbvec[1])
                optiondict['arb_lowerstop'] = float(arbvec[2])
                optiondict['arb_upperstop'] = float(arbvec[3])
            theprefilter.settype('arb')
            optiondict['filtertype'] = 'arb'
            theprefilter.setfreqs(optiondict['arb_lowerstop'], optiondict['arb_lower'],
                                optiondict['arb_upper'], optiondict['arb_upperstop'])
            optiondict['despeckle_thresh'] = np.max(
                [optiondict['despeckle_thresh'], 0.5 / (theprefilter.getfreqs()[2])])
            print('prefiltering to ', optiondict['arb_lower'], optiondict['arb_upper'],
                  '(stops at ', optiondict['arb_lowerstop'], optiondict['arb_upperstop'], ')')
        elif o == '-d':
            optiondict['displayplots'] = True
            print('displaying all plots')
        elif o == '-N':
            optiondict['numestreps'] = int(a)
            if optiondict['numestreps'] == 0:
                optiondict['ampthreshfromsig'] = False
                print('Will not estimate significance thresholds from null correlations')
            else:
                print('Will estimate p<0.05 significance threshold from ', optiondict['numestreps'],
                      ' null correlations')
        elif o == '--accheck':
            optiondict['check_autocorrelation'] = True
            print('Will check for periodic components in the autocorrelation function')
        elif o == '--despecklethresh':
            if optiondict['despeckle_passes'] == 0:
                optiondict['despeckle_passes'] = 1
            optiondict['check_autocorrelation'] = True
            optiondict['despeckle_thresh'] = float(a)
            linkchar = '='
            print('Forcing despeckle threshhold to ', optiondict['despeckle_thresh'])
        elif o == '--despecklepasses':
            optiondict['check_autocorrelation'] = True
            optiondict['despeckle_passes'] = int(a)
            if optiondict['despeckle_passes'] < 1:
                print("minimum number of despeckle passes is 1")
                sys.exit()
            linkchar = '='
            print('Will do ', optiondict['despeckle_passes'], ' despeckling passes')
        elif o == '--acfix':
            optiondict['fix_autocorrelation'] = True
            optiondict['check_autocorrelation'] = True
            print('Will remove periodic components in the autocorrelation function (experimental)')
        elif o == '--noprogressbar':
            optiondict['showprogressbar'] = False
            print('Will disable progress bars')
        elif o == '-s':
            optiondict['widthlimit'] = float(a)
            print('Setting gaussian fit width limit to ', optiondict['widthlimit'], 'Hz')
        elif o == '-b':
            optiondict['usebutterworthfilter'] = True
            theprefilter.setbutter(optiondict['usebutterworthfilter'], optiondict['filtorder'])
            print('Using butterworth bandlimit filter')
        elif o == '-Z':
            optiondict['fixeddelayvalue'] = float(a)
            optiondict['fixdelay'] = True
            optiondict['lagmin'] = optiondict['fixeddelayvalue'] - 10.0
            optiondict['lagmax'] = optiondict['fixeddelayvalue'] + 10.0
            print('Delay will be set to ', optiondict['fixeddelayvalue'], 'in all voxels')
        elif o == '--motionfile':
            optiondict['motionfilename'] = a
            print('Will regress derivatives and delayed derivatives of motion out of data prior to analysis')
        elif o == '--motpos':
            optiondict['mot_pos'] = not optiondict['mot_pos']
            print(optiondict['mot_pos'], 'set to', optiondict['mot_pos'])
        elif o == '--motderiv':
            optiondict['mot_deriv'] = not optiondict['mot_deriv']
            print(optiondict['mot_deriv'], 'set to', optiondict['mot_deriv'])
        elif o == '--motdelayderiv':
            optiondict['mot_delayderiv'] = not optiondict['mot_delayderiv']
            print(optiondict['mot_delayderiv'], 'set to', optiondict['mot_delayderiv'])
        elif o == '-f':
            optiondict['gausssigma'] = float(a)
            optiondict['dogaussianfilter'] = True
            print('Will prefilter fMRI data with a gaussian kernel of ', optiondict['gausssigma'], ' mm')
        elif o == '--timerange':
            limitvec = a.split(',')
            optiondict['startpoint'] = int(limitvec[0])
            optiondict['endpoint'] = int(limitvec[1])
            linkchar = '='
            print('Analysis will be performed only on data from point ', optiondict['startpoint'], ' to ',
                  optiondict['endpoint'])
        elif o == '-r':
            lagvec = a.split(',')
            if not optiondict['fixdelay']:
                optiondict['lagmin'] = float(lagvec[0])
                optiondict['lagmax'] = float(lagvec[1])
                if optiondict['lagmin'] >= optiondict['lagmax']:
                    print('lagmin must be less than lagmax - exiting')
                    sys.exit(1)
                print('Correlations will be calculated over range ', optiondict['lagmin'], ' to ', optiondict['lagmax'])
        elif o == '-y':
            optiondict['interptype'] = a
            if (optiondict['interptype'] != 'cubic') and (optiondict['interptype'] != 'quadratic') and (
                    optiondict['interptype'] != 'univariate'):
                print('unsupported interpolation type!')
                sys.exit()
        elif o == '-h':
            optiondict['histlen'] = int(a)
            print('Setting histogram length to ', optiondict['histlen'])
        elif o == '-o':
            optiondict['offsettime'] = float(a)
            optiondict['offsettime_total'] = -float(a)
            print('Applying a timeshift of ', optiondict['offsettime'], ' to regressor')
        elif o == '--datafreq':
            realtr = 1.0 / float(a)
            linkchar = '='
            print('Data time step forced to ', realtr)
        elif o == '--datatstep':
            realtr = float(a)
            linkchar = '='
            print('Data time step forced to ', realtr)
        elif o == '-t':
            print(
                'DEPRECATION WARNING: The -t option is obsolete and will be removed in a future version.  Use --datatstep=TSTEP or --datafreq=FREQ instead')
            realtr = float(a)
            print('Data time step forced to ', realtr)
        elif o == '-c':
            optiondict['isgrayordinate'] = True
            print('Input fMRI file is a converted CIFTI file')
        elif o == '-O':
            optiondict['oversampfactor'] = int(a)
            if 0 <= optiondict['oversampfactor'] < 1:
                print('oversampling factor must be an integer greater than or equal to 1 (or negative to set automatically)')
                sys.exit()
            print('oversampling factor set to ', optiondict['oversampfactor'])
        elif o == '--psdfilter':
            optiondict['psdfilter'] = True
            print('Will use a cross-spectral density filter on shifted timecourses prior to refinement')
        elif o == '--avg':
            optiondict['refinetype'] = 'unweighted_average'
            print('Will use unweighted average to refine regressor rather than simple averaging')
        elif o == '--weightedavg':
            optiondict['refinetype'] = 'weighted_average'
            print('Will use weighted average to refine regressor rather than simple averaging')
        elif o == '--ica':
            optiondict['refinetype'] = 'ica'
            print('Will use ICA procedure to refine regressor rather than simple averaging')
        elif o == '--dispersioncalc':
            optiondict['dodispersioncalc'] = True
            print('Will do dispersion calculation during regressor refinement')
        elif o == '--nofitfilt':
            optiondict['zerooutbadfit'] = False
            optiondict['nohistzero'] = True
            print('Correlation parameters will be recorded even if out of bounds')
        elif o == '--pca':
            optiondict['refinetype'] = 'pca'
            print('Will use PCA procedure to refine regressor rather than simple averaging')
        elif o == '--numskip':
            optiondict['preprocskip'] = int(a)
            linkchar = '='
            print('Setting preprocessing trs skipped to ', optiondict['preprocskip'])
        elif o == '--venousrefine':
            optiondict['lagmaskside'] = 'upper'
            optiondict['lagminthresh'] = 2.5
            optiondict['lagmaxthresh'] = 6.0
            optiondict['ampthresh'] = 0.5
            print('Biasing refinement to voxels in draining vasculature')
        elif o == '--nirs':
            optiondict['nothresh'] = True
            optiondict['corrmaskthreshpct'] = 0.0
            optiondict['preservefiltering'] = True
            optiondict['refineprenorm'] = 'var'
            optiondict['ampthresh'] = 0.7
            optiondict['lagminthresh'] = 0.1
            print('Setting NIRS mode')
        elif o == '--nothresh':
            optiondict['nothresh'] = True
            optiondict['corrmaskthreshpct'] = 0.0
            print('Disabling voxel threshhold')
        elif o == '--regressor':
            optiondict['regressorfile'] = a
            optiondict['useglobalref'] = False
            linkchar = '='
            print('Will use regressor file', a)
        elif o == '--regressorfreq':
            optiondict['inputfreq'] = float(a)
            linkchar = '='
            print('Setting regressor sample frequency to ', inputfreq)
        elif o == '--regressortstep':
            optiondict['inputfreq'] = 1.0 / float(a)
            linkchar = '='
            print('Setting regressor sample time step to ', float(a))
        elif o == '--regressorstart':
            optiondict['inputstarttime'] = float(a)
            linkchar = '='
            print('Setting regressor start time to ', optiondict['inputstarttime'])
        elif o == '--slicetimes':
            optiondict['slicetimes'] = tide_io.readvecs(a)
            linkchar = '='
            print('Using slicetimes from file', a)
        elif o == '--detrendorder':
            optiondict['detrendorder'] = int(a)
            print('Setting trend removal order to', optiondict['detrendorder'],
                  'for regressor generation and correlation preparation')
        elif o == '--refineupperlag':
            optiondict['lagmaskside'] = 'upper'
            print('Will only use lags between ', optiondict['lagminthresh'], ' and ', optiondict['lagmaxthresh'],
                  ' in refinement')
        elif o == '--refinelowerlag':
            optiondict['lagmaskside'] = 'lower'
            print('Will only use lags between ', -optiondict['lagminthresh'], ' and ', -optiondict['lagmaxthresh'],
                  ' in refinement')
        elif o == '--nogaussrefine':
            optiondict['gaussrefine'] = False
            print('Will not use gaussian correlation peak refinement')
        elif o == '--fastgauss':
            optiondict['fastgauss'] = True
            print('Will use alternative fast gauss refinement (does not work well)')
        elif o == '--refineoffset':
            optiondict['refineoffset'] = True
            if optiondict['passes'] == 1:
                optiondict['passes'] = 2
            print('Will refine offset time during subsequent passes')
        elif o == '--pickleft':
            optiondict['pickleft'] = True
            print('Will select the leftmost delay peak when setting refine offset')
        elif o == '--lagminthresh':
            optiondict['lagminthresh'] = float(a)
            if optiondict['passes'] == 1:
                optiondict['passes'] = 2
            linkchar = '='
            print('Using lagminthresh of ', optiondict['lagminthresh'])
        elif o == '--lagmaxthresh':
            optiondict['lagmaxthresh'] = float(a)
            if optiondict['passes'] == 1:
                optiondict['passes'] = 2
            linkchar = '='
            print('Using lagmaxthresh of ', optiondict['lagmaxthresh'])
        elif o == '--skipsighistfit':
            optiondict['dosighistfit'] = False
            print('will not fit significance histogram with a Johnson SB function')
        elif o == '--searchfrac':
            optiondict['searchfrac'] = float(a)
            linkchar = '='
            print('Points greater than', optiondict['ampthresh'],
                  '* the peak height will be used to fit peak parameters')
        elif o == '--ampthresh':
            optiondict['ampthresh'] = float(a)
            optiondict['ampthreshfromsig'] = False
            if optiondict['passes'] == 1:
                optiondict['passes'] = 2
            linkchar = '='
            if optiondict['ampthresh'] < 0.0:
                print('Setting ampthresh to the', -100.0 * optiondict['ampthresh'], 'th percentile')
            else:
                print('Using ampthresh of ', optiondict['ampthresh'])
        elif o == '--sigmathresh':
            optiondict['sigmathresh'] = float(a)
            if optiondict['passes'] == 1:
                optiondict['passes'] = 2
            linkchar = '='
            print('Using widththresh of ', optiondict['sigmathresh'])
        elif o == '--globalmeaninclude':

            optiondict['globalmeanincludename'], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict['globalmeanincludevals'] = tide_io.colspectolist(colspec)
            linkchar = '='
            if optiondict['globalmeanincludevals'] is not None:
                print('Using voxels where',
                      optiondict['globalmeanincludename'],
                      ' = ',
                      optiondict['globalmeanincludevals'],
                      ' for inclusion in global mean')
            else:
                print('Using ', optiondict['globalmeanincludename'], ' as include mask for global mean calculation')
        elif o == '--globalmeanexclude':
            optiondict['globalmeanexcludename'], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict['globalmeanexcludevals'] = tide_io.colspectolist(colspec)
            linkchar = '='
            if optiondict['globalmeanexcludevals'] is not None:
                print('Using voxels where',
                      optiondict['globalmeanexcludename'],
                      ' = ',
                      optiondict['globalmeanexcludevals'],
                      ' for exclusion from global mean')
            else:
                print('Using ', optiondict['globalmeanexcludename'], ' as exclude mask for global mean calculation')
        elif o == '--refineinclude':
            optiondict['refineincludename'], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict['refineincludevals'] = tide_io.colspectolist(colspec)
            linkchar = '='
            if optiondict['refineincludevals'] is not None:
                print('Using voxels where',
                      optiondict['refineincludename'],
                      ' = ',
                      optiondict['refineincludevals'],
                      ' for inclusion in refine mask')
            else:
                print('Using ', optiondict['refineincludename'], ' as include mask for probe regressor refinement')
        elif o == '--refineexclude':
            optiondict['refineexcludename'], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict['refineexcludevals'] = tide_io.colspectolist(colspec)
            linkchar = '='
            if optiondict['refineexcludevals'] is not None:
                print('Using voxels where',
                      optiondict['refineexcludename'],
                      ' = ',
                      optiondict['refineexcludevals'],
                      ' for exclusion from refine mask')
            else:
                print('Using ', optiondict['refineexcludename'], ' as exclude mask for probe regressor refinement')
        elif o == '--corrmask':
            optiondict['corrmaskname'], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict['corrmaskvallist'] = tide_io.colspectolist(colspec)
            linkchar = '='
            if optiondict['corrmaskvallist'] is not None:
                print('Using voxels where',
                      optiondict['corrmaskname'],
                      ' = ',
                      optiondict['corrmaskvallist'],
                      '- corrmaskthresh will be ignored')
            else:
                print('Using ', optiondict['corrmaskname'], ' as mask file - corrmaskthresh will be ignored')
        elif o == '--refineprenorm':
            optiondict['refineprenorm'] = a
            if (
                    optiondict['refineprenorm'] != 'None') and (
                    optiondict['refineprenorm'] != 'mean') and (
                    optiondict['refineprenorm'] != 'var') and (
                    optiondict['refineprenorm'] != 'std') and (
                    optiondict['refineprenorm'] != 'invlag'):
                print('unsupported refinement prenormalization mode!')
                sys.exit()
            linkchar = '='
        elif o == '--refineweighting':
            optiondict['refineweighting'] = a
            if (
                    optiondict['refineweighting'] != 'None') and (
                    optiondict['refineweighting'] != 'NIRS') and (
                    optiondict['refineweighting'] != 'R') and (
                    optiondict['refineweighting'] != 'R2'):
                print('unsupported refinement weighting!')
                sys.exit()
            linkchar = '='
        elif o == '--tmask':
            optiondict['usetmask'] = True
            optiondict['tmaskname'] = a
            linkchar = '='
            print('Will multiply regressor by timecourse in ', optiondict['tmaskname'])
        elif o == '--refinepasses' or o == '--passes':
            if o == '--refinepasses':
                print(
                    'DEPRECATION WARNING - refinepasses is deprecated and will be removed in a future version - use passes instead')
            optiondict['passes'] = int(a)
            linkchar = '='
            print('Will do ', optiondict['passes'], ' processing passes')
        elif o == '--maxfittype':
            optiondict['findmaxtype'] = a
            linkchar = '='
            print('Will do ', optiondict['findmaxtype'], ' peak fitting')
        elif o in ('-h', '--help'):
            usage()
            sys.exit()
        else:
            assert False, 'unhandled option: ' + o
        formattedcmdline.append('\t' + o + linkchar + a + ' \\')
    formattedcmdline[len(formattedcmdline) - 1] = formattedcmdline[len(formattedcmdline) - 1][:-2]

    # write out the command used
    tide_io.writevec(formattedcmdline, optiondict['outputname'] + '_formattedcommandline.txt')
    tide_io.writevec([' '.join(sys.argv)], optiondict['outputname'] + '_commandline.txt')

    # add additional information to option structure for debugging
    optiondict['realtr'] = realtr

    return optiondict, theprefilter

def rapidtide_main():
    timings = [['Start', time.time(), None, None]]
    optiondict, theprefilter = process_args()

    fmrifilename = optiondict['fmrifilename']
    outputname = optiondict['outputname']
    filename = optiondict['regressorfile']

    optiondict['dispersioncalc_lower'] = optiondict['lagmin']
    optiondict['dispersioncalc_upper'] = optiondict['lagmax']
    optiondict['dispersioncalc_step'] = np.max(
        [(optiondict['dispersioncalc_upper'] - optiondict['dispersioncalc_lower']) / 25,
         optiondict['dispersioncalc_step']])
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
        if optiondict['dogaussianfilter']:
            optiondict['dogaussianfilter'] = False
            print('gaussian spatial filter disabled for text input files')

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
    if optiondict['dogaussianfilter']:
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
    if optiondict['usetmask']:
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
            if optiondict['usetmask']:
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
        if optiondict['dogaussianfilter'] or (optiondict['glmsourcefile'] is not None):
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

