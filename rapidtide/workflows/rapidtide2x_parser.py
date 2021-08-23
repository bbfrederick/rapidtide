#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import getopt
import os
import sys

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.util as tide_util

global rt_floatset, rt_floattype

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


def usage():
    print("usage: ", os.path.basename(sys.argv[0]), " datafilename outputname ")
    print(
        " ".join(
            [
                "[-r LAGMIN,LAGMAX]",
                "[-s SIGMALIMIT]",
                "[-a]",
                "[--nowindow]",
                "[--phat]",
                "[--liang]",
                "[--eckart]",
                "[-f GAUSSSIGMA]",
                "[-O oversampfac]",
                "[-t TSTEP]",
                "[--datatstep=TSTEP]",
                "[--datafreq=FREQ]",
                "[-d]",
                "[-b]",
                "[-V]",
                "[-L]",
                "[-R]",
                "[-C]",
                "[-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]]",
                "[-o OFFSETTIME]",
                "[--autosync]",
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
                "[--excluderefine=MASK]",
                "[--includerefine=MASK]",
                "[--includemean=MASK]",
                "[--excludemean=MASK]" "[--lagminthresh=MIN]",
                "[--lagmaxthresh=MAX]",
                "[--ampthresh=AMP]",
                "[--sigmathresh=SIGMA]",
                "[--corrmask=MASK]",
                "[--corrmaskthresh=PCT]",
                "[--refineoffset]",
                "[--pca]",
                "[--ica]",
                "[--weightedavg]",
                "[--avg]",
                "[--psdfilter]",
                "[--noprogressbar]",
                "[--despecklethresh=VAL]",
                "[--despecklepasses=PASSES]",
                "[--dispersioncalc]",
                "[--refineupperlag]",
                "[--refinelowerlag]",
                "[--nosharedmem]",
                "[--tmask=MASKFILE]",
                "[--limitoutput]",
                "[--motionfile=FILENAME[:COLSPEC]",
                "[--softlimit]",
                "[--timerange=START,END]",
                "[--skipsighistfit]",
                "[--accheck]",
                "[--acfix]" "[--numskip=SKIP]",
                "[--slicetimes=FILE]",
                "[--glmsourcefile=FILE]",
                "[--regressorfreq=FREQ]",
                "[--regressortstep=TSTEP]" "[--regressor=FILENAME]",
                "[--regressorstart=STARTTIME]",
                "[--usesp]",
                "[--peakfittype=FITTYPE]",
                "[--mklthreads=NTHREADS]",
                "[--nprocs=NPROCS]",
                "[--nirs]",
                "[--venousrefine]",
            ]
        )
    )
    print("")
    print("Required arguments:")
    print("    datafilename               - The input data file (BOLD fmri file or NIRS)")
    print("    outputname                 - The root name for the output files")
    print("")
    print("Optional arguments:")
    print(
        "    Arguments are processed in order of appearance.  Later options can override ones earlier on"
    )
    print("    the command line")
    print("")
    print("Macros:")
    print(
        "    --venousrefine                 - This is a macro that sets --lagminthresh=2.5, --lagmaxthresh=6.0,"
    )
    print(
        "                                     --ampthresh=0.5, and --refineupperlag to bias refinement towards "
    )
    print(
        "                                     voxels in the draining vasculature for an fMRI scan."
    )
    print(
        "    --nirs                         - This is a NIRS analysis - this is a macro that sets --nothresh,"
    )
    print(
        "                                     --preservefiltering, --refinenorm=var, --ampthresh=0.7, "
    )
    print("                                     and --lagminthresh=0.1.")
    print("")
    print("Preprocessing options:")
    print(
        "    -t TSTEP,                      - Set the timestep of the data file to TSTEP (or 1/FREQ)"
    )
    print("      --datatstep=TSTEP,             This will override the TR in an fMRI file.")
    print(
        "      --datafreq=FREQ                NOTE: if using data from a text file, for example with"
    )
    print(
        "                                     NIRS data, using one of these options is mandatory."
    )
    print("    -a                             - Disable antialiasing filter")
    print(
        "    --detrendorder=ORDER           - Set order of trend removal (0 to disable, default is 1 - linear)"
    )
    print(
        "    -I                             - Invert the sign of the regressor before processing"
    )
    print(
        "    -i                             - Use specified interpolation type (options are 'cubic',"
    )
    print("                                     'quadratic', and 'univariate (default)')")
    print("    -o                             - Apply an offset OFFSETTIME to the lag regressors")
    print(
        "    --autosync                     - Calculate and apply offset time of an external regressor from "
    )
    print(
        "                                     the global crosscorrelation.  Overrides offsettime if specified."
    )
    print(
        "    -b                             - Use butterworth filter for band splitting instead of"
    )
    print("                                     trapezoidal FFT filter")
    print("    -F  LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]")
    print(
        "                                   - Filter data and regressors from LOWERFREQ to UPPERFREQ."
    )
    print(
        "                                     LOWERSTOP and UPPERSTOP can be specified, or will be"
    )
    print("                                     calculated automatically")
    print("    -V                             - Filter data and regressors to VLF band")
    print("    -L                             - Filter data and regressors to LFO band")
    print("    -R                             - Filter data and regressors to respiratory band")
    print("    -C                             - Filter data and regressors to cardiac band")
    print(
        "    --padseconds=SECONDS           - Set the filter pad time to SECONDS seconds.  Default"
    )
    print("                                     is 30.0")
    print(
        "    -N NREPS                       - Estimate significance threshold by running NREPS null "
    )
    print(
        "                                     correlations (default is 10000, set to 0 to disable).  If you are"
    )
    print(
        "                                     running multiple passes, 'ampthresh' will be set to the 0.05 significance."
    )
    print(
        "                                     level unless it is manually specified (see below)."
    )
    print(
        "    --permutationmethod=METHOD     - Method for permuting the regressor for significance estimation.  Default"
    )
    print("                                     is shuffle")
    print(
        "    --skipsighistfit               - Do not fit significance histogram with a Johnson SB function"
    )
    print(
        "    --windowfunc=FUNC              - Use FUNC window funcion prior to correlation.  Options are"
    )
    print("                                     hamming (default), hann, blackmanharris, and None")
    print("    --nowindow                     - Disable precorrelation windowing")
    print(
        "    -f GAUSSSIGMA                  - Spatially filter fMRI data prior to analysis using "
    )
    print("                                     GAUSSSIGMA in mm")
    print(
        "    -M                             - Generate a global mean regressor and use that as the "
    )
    print("                                     reference regressor")
    print("    --globalmeaninclude=MASK[:VALSPEC]")
    print(
        "                                   - Only use voxels in NAME for global regressor generation (if VALSPEC is"
    )
    print(
        "                                     given, only voxels with integral values listed in VALSPEC are used.)"
    )
    print("    --globalmeanexclude=MASK[:VALSPEC]")
    print(
        "                                   - Do not use voxels in NAME for global regressor generation (if VALSPEC is"
    )
    print(
        "                                     given, only voxels with integral values listed in VALSPEC are used.)"
    )
    print(
        "    -m                             - Mean scale regressors during global mean estimation"
    )
    print(
        "    --slicetimes=FILE              - Apply offset times from FILE to each slice in the dataset"
    )
    print(
        "    --numskip=SKIP                 - SKIP tr's were previously deleted during preprocessing (e.g. if you "
    )
    print(
        "                                     have done your preprocessing in FSL and set dummypoints to a "
    )
    print("                                     nonzero value.) Default is 0.")
    print("    --timerange=START,END          - Limit analysis to data between timepoints START ")
    print("                                     and END in the fmri file. If END is set to -1, ")
    print(
        "                                     analysis will go to the last timepoint.  Negative values "
    )
    print(
        "                                     of START will be set to 0. Default is to use all timepoints."
    )
    print(
        "    --nothresh                     - Disable voxel intensity threshold (especially useful"
    )
    print("                                     for NIRS data)")
    print(
        "    --motionfile=MOTFILE[:COLSPEC] - Read 6 columns of motion regressors out of MOTFILE text file."
    )
    print(
        "                                     (with timepoints rows) and regress their derivatives"
    )
    print(
        "                                     and delayed derivatives out of the data prior to analysis."
    )
    print(
        "                                     If COLSPEC is present, use the comma separated list of ranges to"
    )
    print(
        "                                     specify X, Y, Z, RotX, RotY, and RotZ, in that order.  For"
    )
    print(
        "                                     example, :3-5,7,0,9 would use columns 3, 4, 5, 7, 0 and 9"
    )
    print("                                     for X, Y, Z, RotX, RotY, RotZ, respectively")
    print(
        "    --motpos                       - Toggle whether displacement regressors will be used in motion regression."
    )
    print("                                     Default is False.")
    print(
        "    --motderiv                     - Toggle whether derivatives will be used in motion regression."
    )
    print("                                     Default is True.")
    print(
        "    --motdelayderiv                - Toggle whether delayed derivative  regressors will be used in motion regression."
    )
    print("                                     Default is False.")
    print("")
    print("Correlation options:")
    print(
        "    -O OVERSAMPFAC                 - Oversample the fMRI data by the following integral "
    )
    print(
        "                                     factor.  Setting to -1 chooses the factor automatically (default)"
    )
    print("    --regressor=FILENAME           - Read probe regressor from file FILENAME (if none ")
    print("                                     specified, generate and use global regressor)")
    print(
        "    --regressorfreq=FREQ           - Probe regressor in file has sample frequency FREQ "
    )
    print(
        "                                     (default is 1/tr) NB: --regressorfreq and --regressortstep"
    )
    print("                                     are two ways to specify the same thing")
    print(
        "    --regressortstep=TSTEP         - Probe regressor in file has sample time step TSTEP "
    )
    print(
        "                                     (default is tr) NB: --regressorfreq and --regressortstep"
    )
    print("                                     are two ways to specify the same thing")
    print(
        "    --regressorstart=START         - The time delay in seconds into the regressor file, corresponding"
    )
    print("                                     in the first TR of the fmri file (default is 0.0)")
    print(
        "    --phat                         - Use generalized cross-correlation with phase alignment "
    )
    print("                                     transform (PHAT) instead of correlation")
    print(
        "    --liang                        - Use generalized cross-correlation with Liang weighting function"
    )
    print("                                     (Liang, et al, doi:10.1109/IMCCC.2015.283)")
    print(
        "    --eckart                       - Use generalized cross-correlation with Eckart weighting function"
    )
    print(
        "    --corrmaskthresh=PCT           - Do correlations in voxels where the mean exceeeds this "
    )
    print("                                     percentage of the robust max (default is 1.0)")
    print(
        "    --corrmask=MASK                - Only do correlations in voxels in MASK (if set, corrmaskthresh"
    )
    print("                                     is ignored).")
    print(
        "    --accheck                      - Check for periodic components that corrupt the autocorrelation"
    )
    print("")
    print("Correlation fitting options:")
    print(
        "    -Z DELAYTIME                   - Don't fit the delay time - set it to DELAYTIME seconds "
    )
    print("                                     for all voxels")
    print(
        "    -r LAGMIN,LAGMAX               - Limit fit to a range of lags from LAGMIN to LAGMAX"
    )
    print(
        "    -s SIGMALIMIT                  - Reject lag fits with linewidth wider than SIGMALIMIT"
    )
    print(
        "    -B                             - Bipolar mode - match peak correlation ignoring sign"
    )
    print("    --nofitfilt                    - Do not zero out peak fit values if fit fails")
    print(
        "    --searchfrac=FRAC              - When peak fitting, include points with amplitude > FRAC * the"
    )
    print("                                     maximum amplitude.")
    print("                                     (default value is 0.5)")
    print(
        "    --peakfittype=FITTYPE          - Method for fitting the peak of the similarity function"
    )
    print(
        "                                     (default is 'gauss'). 'quad' uses a quadratic fit. Other options are "
    )
    print(
        "                                     'fastgauss' which is faster but not as well tested, and 'None'."
    )
    print(
        "    --despecklepasses=PASSES       - detect and refit suspect correlations to disambiguate peak"
    )
    print("                                     locations in PASSES passes")
    print(
        "    --despecklethresh=VAL          - refit correlation if median discontinuity magnitude exceeds"
    )
    print("                                     VAL (default is 5s)")
    print(
        "    --softlimit                    - Allow peaks outside of range if the maximum correlation is"
    )
    print("                                     at an edge of the range.")
    print("")
    print("Regressor refinement options:")
    print(
        "    --refineprenorm=TYPE           - Apply TYPE prenormalization to each timecourse prior "
    )
    print("                                     to refinement (valid weightings are 'None', ")
    print("                                     'mean' (default), 'var', and 'std'")
    print("    --refineweighting=TYPE         - Apply TYPE weighting to each timecourse prior ")
    print("                                     to refinement (valid weightings are 'None', ")
    print("                                     'R', 'R2' (default)")
    print("    --passes=PASSES,               - Set the number of processing passes to PASSES ")
    print("     --refinepasses=PASSES           (default is 1 pass - no refinement).")
    print(
        "                                     NB: refinepasses is the wrong name for this option -"
    )
    print(
        "                                     --refinepasses is deprecated, use --passes from now on."
    )
    print(
        "    --refineinclude=MASK[:VALSPEC] - Only use nonzero voxels in MASK for regressor refinement (if VALSPEC is"
    )
    print(
        "                                     given, only voxels with integral values listed in VALSPEC are used.)"
    )
    print(
        "    --refineexclude=MASK[:VALSPEC] - Do not use nonzero voxels in MASK for regressor refinement (if VALSPEC is"
    )
    print(
        "                                     given, only voxels with integral values listed in VALSPEC are used.)"
    )
    print("    --lagminthresh=MIN             - For refinement, exclude voxels with delays less ")
    print("                                     than MIN (default is 0.5s)")
    print(
        "    --lagmaxthresh=MAX             - For refinement, exclude voxels with delays greater "
    )
    print("                                     than MAX (default is 5s)")
    print("    --ampthresh=AMP                - For refinement, exclude voxels with correlation ")
    print(
        "                                     coefficients less than AMP (default is 0.3).  NOTE: ampthresh will"
    )
    print(
        "                                     automatically be set to the p<0.05 significance level determined by"
    )
    print(
        "                                     the -N option if -N is set greater than 0 and this is not "
    )
    print("                                     manually specified.")
    print(
        "    --sigmathresh=SIGMA            - For refinement, exclude voxels with widths greater "
    )
    print("                                     than SIGMA (default is 100s)")
    print(
        "    --refineoffset                 - Adjust offset time during refinement to bring peak "
    )
    print("                                     delay to zero")
    print(
        "    --pickleft                     - When setting refineoffset, always select the leftmost histogram peak"
    )
    print(
        "    --pickleftthresh=THRESH        - Set the threshold value (fraction of maximum) to decide something is a "
    )
    print("                                     peak in a histogram.  Default is 0.33.")
    print("    --refineupperlag               - Only use positive lags for regressor refinement")
    print("    --refinelowerlag               - Only use negative lags for regressor refinement")
    print("    --pca                          - Use pca to derive refined regressor (default is ")
    print("                                     unweighted averaging)")
    print("    --ica                          - Use ica to derive refined regressor (default is ")
    print("                                     unweighted averaging)")
    print("    --weightedavg                  - Use weighted average to derive refined regressor ")
    print("                                     (default is unweighted averaging)")
    print(
        "    --avg                          - Use unweighted average to derive refined regressor "
    )
    print("                                     (default)")
    print("    --psdfilter                    - Apply a PSD weighted Wiener filter to shifted")
    print("                                     timecourses prior to refinement")
    print("")
    print("Output options:")
    print(
        "    --limitoutput                  - Don't save some of the large and rarely used files"
    )
    print("    -T                             - Save a table of lagtimes used")
    print(
        "    -h HISTLEN                     - Change the histogram length to HISTLEN (default is"
    )
    print("                                     100)")
    print(
        "    --glmsourcefile=FILE           - Regress delayed regressors out of FILE instead of the "
    )
    print("                                     initial fmri file used to estimate delays")
    print(
        "    --noglm                        - Turn off GLM filtering to remove delayed regressor "
    )
    print("                                     from each voxel (disables output of fitNorm)")
    print("    --preservefiltering            - don't reread data prior to GLM")
    print("")
    print("Miscellaneous options:")
    print(
        "    --noprogressbar                - Disable progress bars - useful if saving output to files"
    )
    print(
        "    --wiener                       - Perform Wiener deconvolution to get voxel transfer functions"
    )
    print(
        "    --usesp                        - Use single precision for internal calculations (may"
    )
    print("                                     be useful when RAM is limited)")
    print("    -c                             - Data file is a converted CIFTI")
    print("    -S                             - Simulate a run - just report command line options")
    print("    -d                             - Display plots of interesting timecourses")
    print("    --nonumba                      - Disable jit compilation with numba")
    print(
        "    --nosharedmem                  - Disable use of shared memory for large array storage"
    )
    print("    --memprofile                   - Enable memory profiling for debugging - warning:")
    print("                                     this slows things down a lot.")
    print(
        "    --multiproc                    - Enable multiprocessing versions of key subroutines.  This"
    )
    print(
        "                                     speeds things up dramatically.  Almost certainly will NOT"
    )
    print(
        "                                     work on Windows (due to different forking behavior)."
    )
    print(
        "    --mklthreads=NTHREADS          - Use no more than NTHREADS worker threads in accelerated numpy calls."
    )
    print(
        "    --nprocs=NPROCS                - Use NPROCS worker processes for multiprocessing.  Setting NPROCS"
    )
    print(
        "                                     less than 1 sets the number of worker processes to"
    )
    print(
        "                                     n_cpus - 1 (default).  Setting NPROCS enables --multiproc."
    )
    print("    --debug                        - Enable additional information output")
    print(
        "                                     become the default, but for now I'm just trying it out."
    )
    print("")
    print("Experimental options (not fully tested, may not work):")
    print(
        "    --cleanrefined                 - perform additional processing on refined regressor to remove spurious"
    )
    print("                                     components.")
    print(
        "    --dispersioncalc               - Generate extra data during refinement to allow calculation of"
    )
    print("                                     dispersion.")
    print(
        "    --acfix                        - Perform a secondary correlation to disambiguate peak location"
    )
    print("                                     (enables --accheck).  Experimental.")
    print("    --tmask=MASKFILE               - Only correlate during epochs specified in ")
    print(
        "                                     MASKFILE (NB: if file has one colum, the length needs to match"
    )
    print(
        "                                     the number of TRs used.  TRs with nonzero values will be used"
    )
    print(
        "                                     in analysis.  If there are 2 or more columns, each line of MASKFILE"
    )
    print(
        "                                     contains the time (first column) and duration (second column) of an"
    )
    print("                                     epoch to include.)")
    return ()


def process_args(inputargs=None):
    nargs = len(sys.argv)
    if nargs < 3:
        usage()
        exit()

    # set default variable values
    optiondict = {}

    # file i/o file options
    optiondict["isgrayordinate"] = False
    optiondict["textio"] = False

    # preprocessing options
    optiondict["gausssigma"] = 0.0  # the width of the spatial filter kernel in mm
    optiondict[
        "antialias"
    ] = True  # apply an antialiasing filter to any regressors prior to filtering
    optiondict["invertregressor"] = False  # invert the initial regressor during startup
    optiondict["slicetimes"] = None  # do not apply any slice order correction by default
    optiondict["startpoint"] = -1  # by default, analyze the entire length of the dataset
    optiondict["endpoint"] = 10000000  # by default, analyze the entire length of the dataset
    optiondict["preprocskip"] = 0  # number of trs skipped in preprocessing
    optiondict["globalsignalmethod"] = "sum"
    optiondict["globalpcacomponents"] = 0.8
    optiondict["globalmaskmethod"] = "mean"
    optiondict["globalmeanexcludename"] = None
    optiondict["globalmeanexcludevals"] = None  # list of integer values to use in the mask
    optiondict["globalmeanincludename"] = None
    optiondict["globalmeanincludevals"] = None  # list of integer values to use in the mask

    # correlation options
    optiondict["similaritymetric"] = "correlation"
    optiondict["smoothingtime"] = 3.0
    optiondict["madnormMI"] = False
    optiondict["dodemean"] = True  # remove the mean from signals prior to correlation
    optiondict["detrendorder"] = 1  # remove linear trends prior to correlation
    optiondict["windowfunc"] = "hamming"  # the particular window function to use for correlation
    optiondict["zeropadding"] = 0  # by default, padding is 0 (circular correlation)
    optiondict[
        "corrweighting"
    ] = "None"  # use a standard unweighted crosscorrelation for calculate time delays
    optiondict["tmaskname"] = None  # file name for tmask regressor
    optiondict[
        "corrmaskthreshpct"
    ] = 1.0  # percentage of robust maximum of mean to mask correlations
    optiondict["corrmaskexcludename"] = None
    optiondict["corrmaskexcludevals"] = None  # list of integer values to use in the mask
    optiondict["corrmaskincludename"] = None
    optiondict["corrmaskincludevals"] = None  # list of integer values to use in the mask

    optiondict[
        "check_autocorrelation"
    ] = True  # check for periodic components that corrupt the autocorrelation
    optiondict[
        "fix_autocorrelation"
    ] = False  # remove periodic components that corrupt the autocorrelation
    optiondict[
        "despeckle_thresh"
    ] = 5.0  # threshold value - despeckle if median discontinuity magnitude exceeds despeckle_thresh
    optiondict["despeckle_passes"] = 0  # despeckling passes to perform
    optiondict["nothresh"] = False  # disable voxel intensity threshholding

    # correlation fitting options
    optiondict[
        "hardlimit"
    ] = True  # Peak value must be within specified range.  If false, allow max outside if maximum
    # correlation value is that one end of the range.
    optiondict["bipolar"] = False  # find peak with highest magnitude, regardless of sign
    optiondict["lthreshval"] = 0.0  # zero out peaks with correlations lower than this value
    optiondict["uthreshval"] = 1.0  # zero out peaks with correlations higher than this value
    optiondict[
        "edgebufferfrac"
    ] = 0.0  # what fraction of the correlation window to avoid on either end when fitting
    optiondict["enforcethresh"] = True  # only do fits in voxels that exceed threshhold
    optiondict["zerooutbadfit"] = True  # if true zero out all fit parameters if the fit fails
    optiondict[
        "searchfrac"
    ] = 0.5  # The fraction of the main peak over which points are included in the peak
    optiondict[
        "lagmod"
    ] = 1000.0  # if set to the location of the first autocorrelation sidelobe, this should
    optiondict[
        "peakfittype"
    ] = "gauss"  # if set to 'gauss', use old gaussian fitting, if set to 'quad' use parabolic
    optiondict["acwidth"] = 0.0  # width of the reference autocorrelation function
    optiondict["absmaxsigma"] = 100.0  # width of the reference autocorrelation function
    optiondict["absminsigma"] = 0.25  # width of the reference autocorrelation function
    #     move delay peaks back to the correct position if they hit a sidelobe

    # postprocessing options
    optiondict[
        "doglmfilt"
    ] = True  # use a glm filter to remove the delayed regressor from the data in each voxel
    optiondict["preservefiltering"] = False
    optiondict[
        "glmsourcefile"
    ] = None  # name of the file from which to regress delayed regressors (if not the original data)
    optiondict["dodeconv"] = False  # do Wiener deconvolution to find voxel transfer function
    optiondict["motionfilename"] = None  # by default do no motion regression
    optiondict["mot_pos"] = False  # do not do position
    optiondict["mot_deriv"] = True  # do use derivative
    optiondict["mot_delayderiv"] = False  # do not do delayed derivative
    optiondict["savemotionfiltered"] = False  # save motion filtered file for debugging

    # filter options
    optiondict["filtorder"] = 6
    optiondict[
        "padseconds"
    ] = 30.0  # the number of seconds of padding to add to each end of a filtered timecourse
    optiondict["filtertype"] = "trapezoidal"
    optiondict["respdelete"] = False
    optiondict["lowerstop"] = None
    optiondict["lowerpass"] = None
    optiondict["upperpass"] = None
    optiondict["upperstop"] = None

    # output options
    optiondict["savelagregressors"] = True
    optiondict["savemovingsignal"] = True
    optiondict["saveglmfiltered"] = True
    optiondict["savecorrtimes"] = False
    optiondict["saveintermediatemaps"] = False
    optiondict["bidsoutput"] = False

    optiondict["interptype"] = "univariate"
    optiondict["useglobalref"] = False
    optiondict["fixdelay"] = False
    optiondict["fixeddelayvalue"] = 0.0

    # significance estimation options
    optiondict[
        "numestreps"
    ] = 10000  # the number of sham correlations to perform to estimate significance
    optiondict["permutationmethod"] = "shuffle"
    optiondict[
        "nohistzero"
    ] = False  # if False, there is a spike at R=0 in the significance histogram
    optiondict["ampthreshfromsig"] = True
    optiondict["sighistlen"] = 1000
    optiondict["dosighistfit"] = True

    optiondict["histlen"] = 250
    optiondict["oversampfactor"] = -1
    optiondict["lagmin"] = -30.0
    optiondict["lagmax"] = 30.0
    optiondict["widthlimit"] = 100.0
    optiondict["offsettime"] = 0.0
    optiondict["autosync"] = False
    optiondict["offsettime_total"] = 0.0

    # refinement options
    optiondict["cleanrefined"] = False
    optiondict["lagmaskside"] = "both"
    optiondict["refineweighting"] = "R2"
    optiondict["refineprenorm"] = "mean"
    optiondict["sigmathresh"] = 100.0
    optiondict["lagminthresh"] = 0.5
    optiondict["lagmaxthresh"] = 5.0
    optiondict["ampthresh"] = 0.3
    optiondict["passes"] = 1
    optiondict["refineoffset"] = False
    optiondict["pickleft"] = False
    optiondict["pickleftthresh"] = 0.33
    optiondict["refineexcludename"] = None
    optiondict["refineexcludevals"] = None  # list of integer values to use in the mask
    optiondict["refineincludename"] = None
    optiondict["refineincludevals"] = None  # list of integer values to use in the mask
    optiondict["corrmaskvallist"] = None
    optiondict["refinetype"] = "unweighted_average"
    optiondict["convergencethresh"] = None
    optiondict["maxpasses"] = 10
    optiondict["pcacomponents"] = 0.8
    optiondict["filterbeforePCA"] = True
    optiondict["fmrifreq"] = 0.0
    optiondict["dodispersioncalc"] = False
    optiondict["dispersioncalc_lower"] = -4.0
    optiondict["dispersioncalc_upper"] = 4.0
    optiondict["dispersioncalc_step"] = 0.50
    optiondict["psdfilter"] = False

    # debugging options
    optiondict["singleproc_getNullDist"] = False
    optiondict["singleproc_calcsimilarity"] = False
    optiondict["singleproc_peakeval"] = False
    optiondict["singleproc_fitcorr"] = False
    optiondict["singleproc_glm"] = False
    optiondict["internalprecision"] = "double"
    optiondict["outputprecision"] = "single"
    optiondict["nonumba"] = False
    optiondict["memprofile"] = False
    optiondict["sharedmem"] = True
    optiondict["fakerun"] = False
    optiondict["displayplots"] = False
    optiondict["debug"] = False
    optiondict["verbose"] = False
    (
        optiondict["release_version"],
        optiondict["git_longtag"],
        optiondict["git_date"],
        optiondict["git_isdirty"],
    ) = tide_util.version()
    optiondict["python_version"] = str(sys.version_info)
    optiondict["nprocs"] = 1
    optiondict["mklthreads"] = 1
    optiondict["mp_chunksize"] = 50000
    optiondict["showprogressbar"] = True
    optiondict["savecorrmask"] = True
    optiondict["savedespecklemasks"] = True
    optiondict["checkpoint"] = False  # save checkpoint information for tracking program state
    optiondict["alwaysmultiproc"] = False
    optiondict["calccoherence"] = False

    # experimental options
    optiondict["echocancel"] = False
    optiondict["negativegradient"] = False
    optiondict["negativegradregressor"] = False

    # package options
    optiondict["memprofilerexists"] = memprofilerexists

    realtr = 0.0
    theprefilter = tide_filt.NoncausalFilter()
    theprefilter.setbutterorder(optiondict["filtorder"])

    # start the clock!
    tide_util.checkimports(optiondict)

    # get the command line parameters
    optiondict["regressorfile"] = None
    optiondict["inputfreq"] = None
    optiondict["inputstarttime"] = None
    if len(sys.argv) < 3:
        usage()
        sys.exit()
    # handle required args first
    optiondict["in_file"] = sys.argv[1]
    optiondict["outputname"] = sys.argv[2]
    optparsestart = 3

    # now scan for optional arguments
    try:
        opts, args = getopt.getopt(
            sys.argv[optparsestart:],
            "abcdf:gh:i:mo:s:r:t:vBCF:ILMN:O:RSTVZ:",
            [
                "help",
                "nowindow",
                "windowfunc=",
                "datatstep=",
                "datafreq=",
                "lagminthresh=",
                "lagmaxthresh=",
                "ampthresh=",
                "skipsighistfit",
                "sigmathresh=",
                "refineweighting=",
                "refineprenorm=",
                "corrmaskthresh=",
                "despecklepasses=",
                "despecklethresh=",
                "autosync",
                "accheck",
                "acfix",
                "padseconds",
                "noprogressbar",
                "refinepasses=",
                "passes=",
                "corrmask=",
                "motionfile=",
                "motpos",
                "motderiv",
                "motdelayderiv",
                "globalmeaninclude=",
                "globalmeanexclude=",
                "refineinclude=",
                "refineexclude=",
                "refineoffset",
                "pickleft",
                "pickleftthresh=",
                "nofitfilt",
                "cleanrefined",
                "pca",
                "ica",
                "weightedavg",
                "avg",
                "psdfilter",
                "dispersioncalc",
                "noglm",
                "nosharedmem",
                "multiproc",
                "mklthreads=",
                "permutationmethod=",
                "nprocs=",
                "debug",
                "nonumba",
                "savemotionglmfilt",
                "tmask=",
                "detrendorder=",
                "slicetimes=",
                "glmsourcefile=",
                "preservefiltering",
                "globalmaskmethod=",
                "numskip=",
                "nirs",
                "venousrefine",
                "nothresh",
                "searchfrac=",
                "limitoutput",
                "softlimit",
                "regressor=",
                "regressorfreq=",
                "regressortstep=",
                "regressorstart=",
                "timerange=",
                "refineupperlag",
                "refinelowerlag",
                "memprofile",
                "usesp",
                "liang",
                "eckart",
                "phat",
                "wiener",
                "weiner",
                "respdelete",
                "checkpoint",
                "peakfittype=",
            ],
        )
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like 'option -a not recognized'
        usage()
        sys.exit(2)

    formattedcmdline = [sys.argv[0] + " \\"]
    for thearg in range(1, optparsestart):
        formattedcmdline.append("\t" + sys.argv[thearg] + " \\")

    for o, a in opts:
        linkchar = " "
        if o == "--nowindow":
            optiondict["windowfunc"] = "None"
            print("disable precorrelation windowing")
        elif o == "--checkpoint":
            optiondict["checkpoint"] = True
            print("Enabled run checkpoints")
        elif o == "--permutationmethod":
            themethod = a
            if (themethod != "shuffle") and (themethod != "phaserandom"):
                print("illegal permutation method", themethod)
                sys.exit()
            optiondict["permutationmethod"] = themethod
            linkchar = "="
            print(
                "Will use",
                optiondict["permutationmethod"],
                "as the permutation method for calculating null correlation threshold",
            )
        elif o == "--windowfunc":
            thewindow = a
            if (
                (thewindow != "hamming")
                and (thewindow != "hann")
                and (thewindow != "blackmanharris")
                and (thewindow != "None")
            ):
                print("illegal window function", thewindow)
                sys.exit()
            optiondict["windowfunc"] = thewindow
            linkchar = "="
            print(
                "Will use", optiondict["windowfunc"], "as the window function for correlation",
            )
        elif o == "-v":
            optiondict["verbose"] = True
            print("Turned on verbose mode")
        elif o == "--liang":
            optiondict["corrweighting"] = "liang"
            print("Enabled Liang weighted crosscorrelation")
        elif o == "--eckart":
            optiondict["corrweighting"] = "eckart"
            print("Enabled Eckart weighted crosscorrelation")
        elif o == "--phat":
            optiondict["corrweighting"] = "phat"
            print("Enabled GCC-PHAT fitting")
        elif o == "--weiner":
            print("It's spelled wiener, not weiner")
            print("The filter is named after Norbert Wiener, an MIT mathematician.")
            print("The name probably indicates that his family came from Vienna.")
            print("Spell it right and try again.  I mean, I know what you meant, and could just")
            print("call the routine you wanted anyway, but then how would you learn?")
            sys.exit()
        elif o == "--cleanrefined":
            optiondict["cleanrefined"] = True
            print("Will attempt to clean refined regressor")
        elif o == "--respdelete":
            optiondict["respdelete"] = True
            print("Will attempt to track and delete respiratory waveforms in the passband")
        elif o == "--wiener":
            optiondict["dodeconv"] = True
            print("Will perform Wiener deconvolution")
        elif o == "--usesp":
            optiondict["internalprecision"] = "single"
            print("Will use single precision for internal calculations")
        elif o == "--preservefiltering":
            optiondict["preservefiltering"] = True
            print("Will not reread input file prior to GLM")
        elif o == "--glmsourcefile":
            optiondict["glmsourcefile"] = a
            linkchar = "="
            print("Will regress delayed regressors out of", optiondict["glmsourcefile"])
        elif o == "--corrmaskthresh":
            optiondict["corrmaskthreshpct"] = float(a)
            linkchar = "="
            print(
                "Will perform correlations in voxels where mean exceeds",
                optiondict["corrmaskthreshpct"],
                "% of robust maximum",
            )
        elif o == "-I":
            optiondict["invertregressor"] = True
            print("Invert the regressor prior to running")
        elif o == "-B":
            optiondict["bipolar"] = True
            print("Enabled bipolar correlation fitting")
        elif o == "-S":
            optiondict["fakerun"] = True
            print("report command line options and quit")
        elif o == "-a":
            optiondict["antialias"] = False
            print("antialiasing disabled")
        elif o == "-M":
            optiondict["useglobalref"] = True
            print("using global mean timecourse as the reference regressor")
        elif o == "--globalmaskmethod":
            optiondict["globalmaskmethod"] = a
            if optiondict["globalmaskmethod"] == "mean":
                print("will use mean value to mask voxels prior to generating global mean")
            elif optiondict["globalmaskmethod"] == "variance":
                print(
                    "will use timecourse variance to mask voxels prior to generating global mean"
                )
            else:
                print(
                    optiondict["globalmaskmethod"],
                    "is not a valid masking method.  Valid methods are 'mean' and 'variance'",
                )
                sys.exit()
        elif o == "-m":
            optiondict["globalsignalmethod"] = "meanscale"
            print("mean scale voxels prior to generating global mean")
        elif o == "--limitoutput":
            optiondict["savelagregressors"] = False
            optiondict["savemovingsignal"] = False
            print("disabling output of lagregressors and some ancillary GLM timecourses")
        elif o == "--debug":
            optiondict["debug"] = True
            theprefilter.setdebug(optiondict["debug"])
            print("enabling additional data output for debugging")
        elif o == "--multiproc":
            optiondict["nprocs"] = -1
            print("enabling multiprocessing")
        elif o == "--softlimit":
            optiondict["hardlimit"] = False
            linkchar = "="
            print("will relax peak lag constraint for maximum correlations at edge of range")
        elif o == "--nosharedmem":
            optiondict["sharedmem"] = False
            linkchar = "="
            print("will not use shared memory for large array storage")
        elif o == "--mklthreads":
            optiondict["mklthreads"] = int(a)
            linkchar = "="
            if mklexists:
                print(
                    "will use",
                    optiondict["mklthreads"],
                    "MKL threads for accelerated numpy processing.",
                )
            else:
                print("MKL not present - ignoring --mklthreads")
        elif o == "--nprocs":
            optiondict["nprocs"] = int(a)
            linkchar = "="
            if optiondict["nprocs"] < 0:
                print("will use n_cpus - 1 processes for calculation")
            else:
                print("will use", optiondict["nprocs"], "processes for calculation")
        elif o == "--savemotionglmfilt":
            optiondict["savemotionfiltered"] = True
            print("saveing motion filtered data")
        elif o == "--nonumba":
            optiondict["nonumba"] = True
            print("disabling numba if present")
        elif o == "--memprofile":
            if memprofilerexists:
                optiondict["memprofile"] = True
                print("enabling memory profiling")
            else:
                print("cannot enable memory profiling - memory_profiler module not found")
        elif o == "--noglm":
            optiondict["doglmfilt"] = False
            print("disabling GLM filter")
        elif o == "-T":
            optiondict["savecorrtimes"] = True
            print("saving a table of correlation times used")
        elif o == "-V":
            theprefilter.settype("vlf")
            print("prefiltering to vlf band")
        elif o == "-L":
            theprefilter.settype("lfo")
            optiondict["filtertype"] = "lfo"
            optiondict["despeckle_thresh"] = np.max(
                [optiondict["despeckle_thresh"], 0.5 / (theprefilter.getfreqs()[2])]
            )
            print("prefiltering to lfo band")
        elif o == "-R":
            theprefilter.settype("resp")
            optiondict["filtertype"] = "resp"
            optiondict["despeckle_thresh"] = np.max(
                [optiondict["despeckle_thresh"], 0.5 / (theprefilter.getfreqs()[2])]
            )
            print("prefiltering to respiratory band")
        elif o == "-C":
            theprefilter.settype("cardiac")
            optiondict["filtertype"] = "cardiac"
            optiondict["despeckle_thresh"] = np.max(
                [optiondict["despeckle_thresh"], 0.5 / (theprefilter.getfreqs()[2])]
            )
            print("prefiltering to cardiac band")
        elif o == "-F":
            arbvec = a.split(",")
            if len(arbvec) != 2 and len(arbvec) != 4:
                usage()
                sys.exit()
            if len(arbvec) == 2:
                optiondict["arb_lower"] = float(arbvec[0])
                optiondict["arb_upper"] = float(arbvec[1])
                optiondict["arb_lowerstop"] = 0.9 * float(arbvec[0])
                optiondict["arb_upperstop"] = 1.1 * float(arbvec[1])
            if len(arbvec) == 4:
                optiondict["arb_lower"] = float(arbvec[0])
                optiondict["arb_upper"] = float(arbvec[1])
                optiondict["arb_lowerstop"] = float(arbvec[2])
                optiondict["arb_upperstop"] = float(arbvec[3])
            theprefilter.settype("arb")
            optiondict["filtertype"] = "arb"
            theprefilter.setfreqs(
                optiondict["arb_lowerstop"],
                optiondict["arb_lower"],
                optiondict["arb_upper"],
                optiondict["arb_upperstop"],
            )
            optiondict["despeckle_thresh"] = np.max(
                [optiondict["despeckle_thresh"], 0.5 / (theprefilter.getfreqs()[2])]
            )
            print(
                "prefiltering to ",
                optiondict["arb_lower"],
                optiondict["arb_upper"],
                "(stops at ",
                optiondict["arb_lowerstop"],
                optiondict["arb_upperstop"],
                ")",
            )
        elif o == "--padseconds":
            optiondict["padseconds"] = float(a)
            print("Setting filter padding to", optiondict["padseconds"], "seconds")
        elif o == "-d":
            optiondict["displayplots"] = True
            print("displaying all plots")
        elif o == "-N":
            optiondict["numestreps"] = int(a)
            if optiondict["numestreps"] == 0:
                optiondict["ampthreshfromsig"] = False
                print("Will not estimate significance thresholds from null correlations")
            else:
                print(
                    "Will estimate p<0.05 significance threshold from ",
                    optiondict["numestreps"],
                    " null correlations",
                )
        elif o == "--accheck":
            optiondict["check_autocorrelation"] = True
            print("Will check for periodic components in the autocorrelation function")
        elif o == "--despecklethresh":
            if optiondict["despeckle_passes"] == 0:
                optiondict["despeckle_passes"] = 1
            optiondict["check_autocorrelation"] = True
            optiondict["despeckle_thresh"] = float(a)
            linkchar = "="
            print("Forcing despeckle threshhold to ", optiondict["despeckle_thresh"])
        elif o == "--despecklepasses":
            optiondict["check_autocorrelation"] = True
            optiondict["despeckle_passes"] = int(a)
            if optiondict["despeckle_passes"] < 0:
                print("minimum number of despeckle passes is 0")
                sys.exit()
            linkchar = "="
            print("Will do ", optiondict["despeckle_passes"], " despeckling passes")
        elif o == "--acfix":
            optiondict["fix_autocorrelation"] = True
            optiondict["check_autocorrelation"] = True
            print("Will remove periodic components in the autocorrelation function (experimental)")
        elif o == "--noprogressbar":
            optiondict["showprogressbar"] = False
            print("Will disable progress bars")
        elif o == "-s":
            optiondict["widthlimit"] = float(a)
            print("Setting gaussian fit width limit to ", optiondict["widthlimit"], "Hz")
        elif o == "-b":
            optiondict["filtertype"] = "butterworth"
            theprefilter.setbutterorder(optiondict["filtorder"])
            print("Using butterworth bandlimit filter")
        elif o == "-Z":
            optiondict["fixeddelayvalue"] = float(a)
            optiondict["fixdelay"] = True
            optiondict["lagmin"] = optiondict["fixeddelayvalue"] - 10.0
            optiondict["lagmax"] = optiondict["fixeddelayvalue"] + 10.0
            print("Delay will be set to ", optiondict["fixeddelayvalue"], "in all voxels")
        elif o == "--motionfile":
            optiondict["motionfilename"] = a
            print(
                "Will regress derivatives and delayed derivatives of motion out of data prior to analysis"
            )
        elif o == "--motpos":
            optiondict["mot_pos"] = not optiondict["mot_pos"]
            print(optiondict["mot_pos"], "set to", optiondict["mot_pos"])
        elif o == "--motderiv":
            optiondict["mot_deriv"] = not optiondict["mot_deriv"]
            print(optiondict["mot_deriv"], "set to", optiondict["mot_deriv"])
        elif o == "--motdelayderiv":
            optiondict["mot_delayderiv"] = not optiondict["mot_delayderiv"]
            print(optiondict["mot_delayderiv"], "set to", optiondict["mot_delayderiv"])
        elif o == "-f":
            optiondict["gausssigma"] = float(a)
            print(
                "Will prefilter fMRI data with a gaussian kernel of ",
                optiondict["gausssigma"],
                " mm",
            )
        elif o == "--timerange":
            limitvec = a.split(",")
            optiondict["startpoint"] = int(limitvec[0])
            optiondict["endpoint"] = int(limitvec[1])
            if optiondict["endpoint"] == -1:
                optiondict["endpoint"] = 100000000
            linkchar = "="
            print(
                "Analysis will be performed only on data from point ",
                optiondict["startpoint"],
                " to ",
                optiondict["endpoint"],
                ".",
            )
        elif o == "-r":
            lagvec = a.split(",")
            if not optiondict["fixdelay"]:
                optiondict["lagmin"] = float(lagvec[0])
                optiondict["lagmax"] = float(lagvec[1])
                if optiondict["lagmin"] >= optiondict["lagmax"]:
                    print("lagmin must be less than lagmax - exiting")
                    sys.exit(1)
                print(
                    "Correlations will be calculated over range ",
                    optiondict["lagmin"],
                    " to ",
                    optiondict["lagmax"],
                )
        elif o == "-y":
            optiondict["interptype"] = a
            if (
                (optiondict["interptype"] != "cubic")
                and (optiondict["interptype"] != "quadratic")
                and (optiondict["interptype"] != "univariate")
            ):
                print("unsupported interpolation type!")
                sys.exit()
        elif o == "-h":
            optiondict["histlen"] = int(a)
            print("Setting histogram length to ", optiondict["histlen"])
        elif o == "-o":
            optiondict["offsettime"] = float(a)
            optiondict["offsettime_total"] = float(a)
            print("Applying a timeshift of ", optiondict["offsettime"], " to regressor")
        elif o == "--autosync":
            optiondict["autosync"] = True
            print(
                "Will calculate and apply regressor synchronization from global correlation.  Overrides offsettime."
            )
        elif o == "--datafreq":
            realtr = 1.0 / float(a)
            linkchar = "="
            print("Data time step forced to ", realtr)
        elif o == "--datatstep":
            realtr = float(a)
            linkchar = "="
            print("Data time step forced to ", realtr)
        elif o == "-t":
            print(
                "DEPRECATION WARNING: The -t option is obsolete and will be removed in a future version.  Use --datatstep=TSTEP or --datafreq=FREQ instead"
            )
            realtr = float(a)
            print("Data time step forced to ", realtr)
        elif o == "-c":
            optiondict["isgrayordinate"] = True
            print("Input fMRI file is a converted CIFTI file")
        elif o == "-O":
            optiondict["oversampfactor"] = int(a)
            if 0 <= optiondict["oversampfactor"] < 1:
                print(
                    "oversampling factor must be an integer greater than or equal to 1 (or negative to set automatically)"
                )
                sys.exit()
            print("oversampling factor set to ", optiondict["oversampfactor"])
        elif o == "--psdfilter":
            optiondict["psdfilter"] = True
            print(
                "Will use a cross-spectral density filter on shifted timecourses prior to refinement"
            )
        elif o == "--avg":
            optiondict["refinetype"] = "unweighted_average"
            print("Will use unweighted average to refine regressor rather than simple averaging")
        elif o == "--weightedavg":
            optiondict["refinetype"] = "weighted_average"
            print("Will use weighted average to refine regressor rather than simple averaging")
        elif o == "--ica":
            optiondict["refinetype"] = "ica"
            print("Will use ICA procedure to refine regressor rather than simple averaging")
        elif o == "--dispersioncalc":
            optiondict["dodispersioncalc"] = True
            print("Will do dispersion calculation during regressor refinement")
        elif o == "--nofitfilt":
            optiondict["zerooutbadfit"] = False
            optiondict["nohistzero"] = True
            print("Correlation parameters will be recorded even if out of bounds")
        elif o == "--pca":
            optiondict["refinetype"] = "pca"
            print("Will use PCA procedure to refine regressor rather than simple averaging")
        elif o == "--numskip":
            optiondict["preprocskip"] = int(a)
            linkchar = "="
            print("Setting preprocessing trs skipped to ", optiondict["preprocskip"])
        elif o == "--venousrefine":
            optiondict["lagmaskside"] = "upper"
            optiondict["lagminthresh"] = 2.5
            optiondict["lagmaxthresh"] = 6.0
            optiondict["ampthresh"] = 0.5
            print("Biasing refinement to voxels in draining vasculature")
        elif o == "--nirs":
            optiondict["nothresh"] = True
            optiondict["corrmaskthreshpct"] = 0.0
            optiondict["preservefiltering"] = True
            optiondict["refineprenorm"] = "var"
            optiondict["ampthresh"] = 0.7
            optiondict["lagminthresh"] = 0.1
            print("Setting NIRS mode")
        elif o == "--nothresh":
            optiondict["nothresh"] = True
            optiondict["corrmaskthreshpct"] = 0.0
            print("Disabling voxel threshhold")
        elif o == "--regressor":
            optiondict["regressorfile"] = a
            optiondict["useglobalref"] = False
            linkchar = "="
            print("Will use regressor file", a)
        elif o == "--regressorfreq":
            optiondict["inputfreq"] = float(a)
            linkchar = "="
            print("Setting regressor sample frequency to ", float(a))
        elif o == "--regressortstep":
            optiondict["inputfreq"] = 1.0 / float(a)
            linkchar = "="
            print("Setting regressor sample time step to ", float(a))
        elif o == "--regressorstart":
            optiondict["inputstarttime"] = float(a)
            linkchar = "="
            print("Setting regressor start time to ", optiondict["inputstarttime"])
        elif o == "--slicetimes":
            optiondict["slicetimes"] = tide_io.readvecs(a)
            linkchar = "="
            print("Using slicetimes from file", a)
        elif o == "--detrendorder":
            optiondict["detrendorder"] = int(a)
            print(
                "Setting trend removal order to",
                optiondict["detrendorder"],
                "for regressor generation and correlation preparation",
            )
        elif o == "--refineupperlag":
            optiondict["lagmaskside"] = "upper"
            print(
                "Will only use lags between ",
                optiondict["lagminthresh"],
                " and ",
                optiondict["lagmaxthresh"],
                " in refinement",
            )
        elif o == "--refinelowerlag":
            optiondict["lagmaskside"] = "lower"
            print(
                "Will only use lags between ",
                -optiondict["lagminthresh"],
                " and ",
                -optiondict["lagmaxthresh"],
                " in refinement",
            )
        elif o == "--refineoffset":
            optiondict["refineoffset"] = True
            print("Will refine offset time during subsequent passes")
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
        elif o == "--pickleft":
            optiondict["pickleft"] = True
            print("Will select the leftmost delay peak when setting refine offset")
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
        elif o == "--pickleftthresh":
            optiondict["pickleftthresh"] = float(a)
            print(
                "Threshhold value for leftmost peak height set to", optiondict["pickleftthresh"],
            )
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
        elif o == "--lagminthresh":
            optiondict["lagminthresh"] = float(a)
            print("Using lagminthresh of ", optiondict["lagminthresh"])
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
            linkchar = "="
        elif o == "--lagmaxthresh":
            optiondict["lagmaxthresh"] = float(a)
            print("Using lagmaxthresh of ", optiondict["lagmaxthresh"])
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
            linkchar = "="
        elif o == "--skipsighistfit":
            optiondict["dosighistfit"] = False
            print("will not fit significance histogram with a Johnson SB function")
        elif o == "--searchfrac":
            optiondict["searchfrac"] = float(a)
            linkchar = "="
            print(
                "Points greater than",
                optiondict["ampthresh"],
                "* the peak height will be used to fit peak parameters",
            )
        elif o == "--ampthresh":
            optiondict["ampthresh"] = float(a)
            optiondict["ampthreshfromsig"] = False
            if optiondict["ampthresh"] < 0.0:
                print(
                    "Setting ampthresh to the", -100.0 * optiondict["ampthresh"], "th percentile",
                )
            else:
                print("Using ampthresh of ", optiondict["ampthresh"])
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
            linkchar = "="
        elif o == "--sigmathresh":
            optiondict["sigmathresh"] = float(a)
            print("Using widththresh of ", optiondict["sigmathresh"])
            if optiondict["passes"] == 1:
                print(
                    "WARNING: setting this value implies you are doing refinement; make sure if you want to do that, passes > 1"
                )
            linkchar = "="
        elif o == "--globalmeaninclude":

            optiondict["globalmeanincludename"], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict["globalmeanincludevals"] = tide_io.colspectolist(colspec)
            linkchar = "="
            if optiondict["globalmeanincludevals"] is not None:
                print(
                    "Using voxels where",
                    optiondict["globalmeanincludename"],
                    " = ",
                    optiondict["globalmeanincludevals"],
                    " for inclusion in global mean",
                )
            else:
                print(
                    "Using ",
                    optiondict["globalmeanincludename"],
                    " as include mask for global mean calculation",
                )
        elif o == "--globalmeanexclude":
            optiondict["globalmeanexcludename"], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict["globalmeanexcludevals"] = tide_io.colspectolist(colspec)
            linkchar = "="
            if optiondict["globalmeanexcludevals"] is not None:
                print(
                    "Using voxels where",
                    optiondict["globalmeanexcludename"],
                    " = ",
                    optiondict["globalmeanexcludevals"],
                    " for exclusion from global mean",
                )
            else:
                print(
                    "Using ",
                    optiondict["globalmeanexcludename"],
                    " as exclude mask for global mean calculation",
                )
        elif o == "--refineinclude":
            optiondict["refineincludename"], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict["refineincludevals"] = tide_io.colspectolist(colspec)
            linkchar = "="
            if optiondict["refineincludevals"] is not None:
                print(
                    "Using voxels where",
                    optiondict["refineincludename"],
                    " = ",
                    optiondict["refineincludevals"],
                    " for inclusion in refine mask",
                )
            else:
                print(
                    "Using ",
                    optiondict["refineincludename"],
                    " as include mask for probe regressor refinement",
                )
        elif o == "--refineexclude":
            optiondict["refineexcludename"], colspec = tide_io.parsefilespec(a)
            if colspec is not None:
                optiondict["refineexcludevals"] = tide_io.colspectolist(colspec)
            linkchar = "="
            if optiondict["refineexcludevals"] is not None:
                print(
                    "Using voxels where",
                    optiondict["refineexcludename"],
                    " = ",
                    optiondict["refineexcludevals"],
                    " for exclusion from refine mask",
                )
            else:
                print(
                    "Using ",
                    optiondict["refineexcludename"],
                    " as exclude mask for probe regressor refinement",
                )
        elif o == "--corrmask":
            (
                optiondict["corrmaskincludename"],
                optiondict["corrmaskincludevals"],
            ) = tide_io.processnamespec(a, "Using voxels where ", "in correlation calculations.")
        elif o == "--refineprenorm":
            optiondict["refineprenorm"] = a
            if (
                (optiondict["refineprenorm"] != "None")
                and (optiondict["refineprenorm"] != "mean")
                and (optiondict["refineprenorm"] != "var")
                and (optiondict["refineprenorm"] != "std")
                and (optiondict["refineprenorm"] != "invlag")
            ):
                print("unsupported refinement prenormalization mode!")
                sys.exit()
            linkchar = "="
        elif o == "--refineweighting":
            optiondict["refineweighting"] = a
            if (
                (optiondict["refineweighting"] != "None")
                and (optiondict["refineweighting"] != "NIRS")
                and (optiondict["refineweighting"] != "R")
                and (optiondict["refineweighting"] != "R2")
            ):
                print("unsupported refinement weighting!")
                sys.exit()
            linkchar = "="
        elif o == "--tmask":
            optiondict["tmaskname"] = a
            linkchar = "="
            print("Will multiply regressor by timecourse in ", optiondict["tmaskname"])
        elif o == "--refinepasses" or o == "--passes":
            if o == "--refinepasses":
                print(
                    "DEPRECATION WARNING - refinepasses is deprecated and will be removed in a future version - use passes instead"
                )
            optiondict["passes"] = int(a)
            linkchar = "="
            print("Will do ", optiondict["passes"], " processing passes")
        elif o == "--peakfittype":
            optiondict["peakfittype"] = a
            linkchar = "="
            print("Similarity function peak fitting method is ", optiondict["peakfittype"])
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option: " + o
        formattedcmdline.append("\t" + o + linkchar + a + " \\")
    formattedcmdline[len(formattedcmdline) - 1] = formattedcmdline[len(formattedcmdline) - 1][:-2]

    # store the filter limits
    (
        optiondict["lowerpass"],
        optiondict["upperpass"],
        optiondict["lowerstop"],
        optiondict["upperstop"],
    ) = theprefilter.getfreqs()

    # write out the command used
    tide_io.writevec(formattedcmdline, optiondict["outputname"] + "_formattedcommandline.txt")
    tide_io.writevec([" ".join(sys.argv)], optiondict["outputname"] + "_commandline.txt")
    optiondict["commandlineargs"] = sys.argv[1:]

    # add additional information to option structure for debugging
    optiondict["realtr"] = realtr

    optiondict["dispersioncalc_lower"] = optiondict["lagmin"]
    optiondict["dispersioncalc_upper"] = optiondict["lagmax"]
    optiondict["dispersioncalc_step"] = np.max(
        [
            (optiondict["dispersioncalc_upper"] - optiondict["dispersioncalc_lower"]) / 25,
            optiondict["dispersioncalc_step"],
        ]
    )

    return optiondict, theprefilter
