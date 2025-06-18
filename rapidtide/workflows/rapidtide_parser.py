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
import logging
import os
import sys
from argparse import Namespace

import nibabel as nib
import numpy as np

import rapidtide.io as tide_io
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


# Create a sentinel.
# from https://stackoverflow.com/questions/58594956/find-out-which-arguments-were-passed-explicitly-in-argparse
class _Sentinel:
    pass


sentinel = _Sentinel()

LGR = logging.getLogger(__name__)

# Some default settings
DEFAULT_SPATIALFILT = -1
DEFAULT_HISTLEN = 101
DEFAULT_DETREND_ORDER = 3
DEFAULT_INITREGRESSOR_PCACOMPONENTS = 0.8
DEFAULT_CORRMASK_THRESHPCT = 1.0
DEFAULT_MUTUALINFO_SMOOTHINGTIME = 3.0
DEFAULT_LAGMIN = -30.0
DEFAULT_LAGMAX = 30.0
DEFAULT_SIGMAMAX = 1000.0
DEFAULT_SIGMAMIN = 0.0
DEFAULT_DESPECKLE_PASSES = 4
DEFAULT_DESPECKLE_THRESH = 5.0
DEFAULT_PASSES = 3
DEFAULT_LAGMIN_THRESH = 0.25
DEFAULT_LAGMAX_THRESH = 3.0
DEFAULT_AMPTHRESH = 0.3
DEFAULT_PICKLEFT_THRESH = 0.33
DEFAULT_SIGMATHRESH = 100.0
DEFAULT_MAXPASSES = 15
DEFAULT_REFINE_TYPE = "pca"
DEFAULT_INTERPTYPE = "univariate"
DEFAULT_WINDOW_TYPE = "hamming"
DEFAULT_INITREGRESSOR_METHOD = "sum"
DEFAULT_CORRWEIGHTING = "phat"
DEFAULT_CORRTYPE = "linear"
DEFAULT_SIMILARITYMETRIC = "correlation"
DEFAULT_PEAKFIT_TYPE = "gauss"
DEFAULT_REFINE_PRENORM = "var"
DEFAULT_REFINE_WEIGHTING = "None"
DEFAULT_REFINE_PCACOMPONENTS = 0.8
DEFAULT_REGRESSIONFILTDERIVS = 0
DEFAULT_REFINEREGRESSDERIVS = 1

DEFAULT_DENOISING_LAGMIN = -10.0
DEFAULT_DENOISING_LAGMAX = 10.0
DEFAULT_DENOISING_PASSES = 3
DEFAULT_DENOISING_DESPECKLE_PASSES = 4
DEFAULT_DENOISING_PEAKFITTYPE = "gauss"
DEFAULT_DENOISING_SPATIALFILT = -1

DEFAULT_DELAYMAPPING_LAGMIN = -10.0
DEFAULT_DELAYMAPPING_LAGMAX = 30.0
DEFAULT_DELAYMAPPING_PASSES = 3
DEFAULT_DELAYMAPPING_DESPECKLE_PASSES = 4
DEFAULT_DELAYMAPPING_SPATIALFILT = 5

DEFAULT_CVRMAPPING_LAGMIN = -5.0
DEFAULT_CVRMAPPING_LAGMAX = 20.0
DEFAULT_CVRMAPPING_FILTER_LOWERPASS = 0.0
DEFAULT_CVRMAPPING_FILTER_UPPERPASS = 0.01
DEFAULT_CVRMAPPING_DESPECKLE_PASSES = 4

DEFAULT_OUTPUTLEVEL = "normal"

DEFAULT_SLFONOISEAMP_WINDOWSIZE = 40.0

DEFAULT_PATCHTHRESH = 3.0
DEFAULT_REFINEDELAYMINDELAY = -5.0
DEFAULT_REFINEDELAYMAXDELAY = 5.0
DEFAULT_REFINEDELAYNUMPOINTS = 501
DEFAULT_DELAYOFFSETSPATIALFILT = -1

DEFAULT_PATCHMINSIZE = 10
DEFAULT_PATCHFWHM = 5


def _get_parser():
    """
    Argument parser for rapidtide
    """
    parser = argparse.ArgumentParser(
        prog="rapidtide",
        description=("Perform a RIPTiDe time delay analysis on a dataset."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "in_file",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The input data file (BOLD fMRI file or NIRS text file).",
    )
    parser.add_argument(
        "outputname",
        type=str,
        help=(
            "The root name for the output files.  "
            "For BIDS compliance, this can only contain valid BIDS entities "
            "from the source data."
        ),
    )

    # Analysis types
    analysis_type = parser.add_argument_group(
        title="Analysis type",
        description=(
            "Single arguments that set several parameter values, tailored to particular analysis types. "
            "Any parameter set by an analysis type can be overridden "
            "by setting that parameter explicitly. "
            "Analysis types are mutually exclusive with one another."
        ),
    ).add_mutually_exclusive_group()
    analysis_type.add_argument(
        "--denoising",
        dest="denoising",
        action="store_true",
        help=(
            "Preset for hemodynamic denoising - this is a macro that "
            f"sets searchrange=({DEFAULT_DENOISING_LAGMIN}, {DEFAULT_DENOISING_LAGMAX}), "
            f"passes={DEFAULT_DENOISING_PASSES}, despeckle_passes={DEFAULT_DENOISING_DESPECKLE_PASSES}, "
            f"refineoffset=True, peakfittype={DEFAULT_DENOISING_PEAKFITTYPE}, "
            f"gausssigma={DEFAULT_DENOISING_SPATIALFILT}, nofitfilt=True, dolinfitfilt=True. "
            "Any of these options can be overridden with the appropriate "
            "additional arguments."
        ),
        default=False,
    )
    analysis_type.add_argument(
        "--delaymapping",
        dest="delaymapping",
        action="store_true",
        help=(
            "Preset for delay mapping analysis - this is a macro that "
            f"sets searchrange=({DEFAULT_DELAYMAPPING_LAGMIN}, {DEFAULT_DELAYMAPPING_LAGMAX}), "
            f"passes={DEFAULT_DELAYMAPPING_PASSES}, despeckle_passes={DEFAULT_DELAYMAPPING_DESPECKLE_PASSES}, "
            f"gausssigma={DEFAULT_DELAYMAPPING_SPATIALFILT}, "
            "refineoffset=True, refinedelay=True, outputlevel='normal', "
            "dolinfitfilt=False. "
            "Any of these options can be overridden with the appropriate "
            "additional arguments."
        ),
        default=False,
    )
    analysis_type.add_argument(
        "--CVR",
        dest="docvrmap",
        action="store_true",
        help=(
            "Preset for calibrated CVR mapping.  Given an input regressor that represents some measured "
            "quantity over time (e.g. mmHg CO2 in the EtCO2 trace), rapidtide will calculate and output a map of percent "
            "BOLD change in units of the input regressor.  To do this, this sets: "
            f"passes=1, despeckle_passes={DEFAULT_CVRMAPPING_DESPECKLE_PASSES}, "
            f"searchrange=({DEFAULT_CVRMAPPING_LAGMIN}, {DEFAULT_CVRMAPPING_LAGMAX}), "
            f"filterfreqs=({DEFAULT_CVRMAPPING_FILTER_LOWERPASS}, {DEFAULT_CVRMAPPING_FILTER_UPPERPASS}), "
            "and calculates a voxelwise regression fit using the optimally delayed "
            "input regressor and the percent normalized, demeaned BOLD data as inputs. This map is output as "
            "(XXX_desc-CVR_map.nii.gz).  If no input regressor is supplied, this will generate an error.  "
            "These options can be overridden with the appropriate additional arguments."
        ),
        default=False,
    )
    analysis_type.add_argument(
        "--globalpreselect",
        dest="initregressorpreselect",
        action="store_true",
        help=(
            "Treat this run as an initial pass to locate good candidate voxels for global mean "
            "regressor generation.  This sets: passes=1, despecklepasses=0, "
            "refinedespeckle=False, outputlevel='normal', dolinfitfilt=False, saveintermediatemaps=False."
        ),
        default=False,
    )

    # Macros
    macros = parser.add_argument_group(
        title="Macros",
        description=(
            "Single arguments that change default values for many "
            "arguments. "
            "Macros override individually set parameters. "
        ),
    ).add_mutually_exclusive_group()
    macros.add_argument(
        "--venousrefine",
        dest="venousrefine",
        action="store_true",
        help=(
            "This is a macro that sets lagminthresh=2.5, "
            "lagmaxthresh=6.0, ampthresh=0.5, and "
            "refineupperlag to bias refinement towards "
            "voxels in the draining vasculature for an "
            "fMRI scan."
        ),
        default=False,
    )
    macros.add_argument(
        "--nirs",
        dest="nirs",
        action="store_true",
        help=(
            "This is a NIRS analysis - this is a macro that "
            "sets nothresh, dataiszeromean=True, refineprenorm=var, ampthresh=0.7, and "
            "lagminthresh=0.1. "
        ),
        default=False,
    )

    anatomy = parser.add_argument_group(
        title="Anatomic information",
        description=(
            "These options allow you to tailor the analysis with some anatomic constraints.  You don't need to supply "
            "any of them, but if you do, rapidtide will try to make intelligent processing decisions based on "
            "these maps.  Any individual masks set with anatomic information will be overridden if you specify "
            "that mask directly."
        ),
    )
    anatomy.add_argument(
        "--brainmask",
        dest="brainmaskincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "This specifies the valid brain voxels.  No voxels outside of this mask will be used for global mean "
            "calculation, correlation, refinement, offset calculation, or denoising. "
            "If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise "
            "voxels with value > 0.1 are used.  If this option is set, "
            "rapidtide will limit the include mask used to 1) calculate the initial global mean regressor, "
            "2) decide which voxels in which to calculate delays, "
            "3) refine the regressor at the end of each pass, 4) determine the zero time offset value, and 5) process "
            "to remove sLFO signal. "
            "Setting --initregressorinclude, --corrmaskinclude, --refineinclude or --offsetinclude explicitly will "
            "override this for the given include mask."
        ),
        default=None,
    )
    anatomy.add_argument(
        "--graymattermask",
        dest="graymatterincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "This specifies a gray matter mask registered to the input functional data.  "
            "If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise "
            "voxels with value > 0.1 are used.  If this option is set, "
            "rapidtide will use voxels in the gray matter mask to calculate the initial global mean regressor. "
            "Setting --initregressorinclude  explicitly will override this for "
            "the given include mask."
        ),
        default=None,
    )
    anatomy.add_argument(
        "--whitemattermask",
        dest="whitematterincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "This specifies a white matter mask registered to the input functional data.  "
            "If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise "
            "voxels with value > 0.1 are used.  "
            "This will be used to calculate the white matter timecourse before running rapidtide, and after sLFO "
            "filtering (if enabled)."
        ),
        default=None,
    )
    anatomy.add_argument(
        "--csfmask",
        dest="csfincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "This specifies a CSF mask registered to the input functional data.  "
            "If VALSPEC is given, only voxels in the mask with integral values listed in VALSPEC are used, otherwise "
            "voxels with value > 0.1 are used.  "
            "This will be used to calculate the CSF timecourse before running rapidtide, and after sLFO "
            "filtering (if enabled)."
        ),
        default=None,
    )

    # Preprocessing options
    preproc = parser.add_argument_group("Preprocessing options")
    realtr = preproc.add_mutually_exclusive_group()
    realtr.add_argument(
        "--datatstep",
        dest="realtr",
        action="store",
        metavar="TSTEP",
        type=lambda x: pf.is_float(parser, x, minval=0.0),
        help=(
            "Set the timestep of the data file to TSTEP. "
            "This will override the TR in an "
            "fMRI file. NOTE: if using data from a text "
            "file, for example with NIRS data, using one "
            "of these options is mandatory. "
        ),
        default="auto",
    )
    realtr.add_argument(
        "--datafreq",
        dest="realtr",
        action="store",
        metavar="FREQ",
        type=lambda x: pf.invert_float(parser, x),
        help=(
            "Set the timestep of the data file to 1/FREQ. "
            "This will override the TR in an "
            "fMRI file. NOTE: if using data from a text "
            "file, for example with NIRS data, using one "
            "of these options is mandatory. "
        ),
        default="auto",
    )
    preproc.add_argument(
        "--noantialias",
        dest="antialias",
        action="store_false",
        help="Disable antialiasing filter. ",
        default=True,
    )
    preproc.add_argument(
        "--invert",
        dest="invertregressor",
        action="store_true",
        help=("Invert the sign of the regressor before processing."),
        default=False,
    )
    preproc.add_argument(
        "--interptype",
        dest="interptype",
        action="store",
        type=str,
        choices=["univariate", "cubic", "quadratic"],
        help=(
            "Use specified interpolation type. Options "
            'are "cubic", "quadratic", and "univariate". '
            f"Default is {DEFAULT_INTERPTYPE}. "
        ),
        default=DEFAULT_INTERPTYPE,
    )
    preproc.add_argument(
        "--offsettime",
        dest="offsettime",
        action="store",
        type=float,
        metavar="OFFSETTIME",
        help="Apply offset OFFSETTIME to the lag regressors.",
        default=0.0,
    )
    preproc.add_argument(
        "--autosync",
        dest="autosync",
        action="store_true",
        help=(
            "Estimate and apply the initial offsettime of an external "
            "regressor using the global crosscorrelation. "
            "Overrides offsettime if present."
        ),
        default=False,
    )
    preproc.add_argument(
        "--dataiszeromean",
        dest="dataiszeromean",
        action="store_true",
        help=(
            "Assume that the fMRI data is zero mean (this will be the case if you used AFNI for preprocessing).  "
            "This affects how masks are generated.  Rapidtide will attempt to detect this, but set explicitly "
            "if you know this is the case."
        ),
        default=False,
    )

    # Add filter options
    pf.addfilteropts(parser, filtertarget="data and regressors", details=True)

    # Add permutation options
    pf.addpermutationopts(parser)

    # add window options
    pf.addwindowopts(parser, windowtype=DEFAULT_WINDOW_TYPE)

    preproc.add_argument(
        "--detrendorder",
        dest="detrendorder",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=0),
        metavar="ORDER",
        help=(f"Set order of trend removal (0 to disable). Default is {DEFAULT_DETREND_ORDER}."),
        default=DEFAULT_DETREND_ORDER,
    )
    preproc.add_argument(
        "--spatialfilt",
        dest="gausssigma",
        action="store",
        type=float,
        metavar="GAUSSSIGMA",
        help=(
            "Spatially filter fMRI data prior to analysis "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to have rapidtide set it to half the mean voxel "
            "dimension (a rule of thumb for a good value). Set to 0 to disable."
        ),
        default=DEFAULT_SPATIALFILT,
    )
    preproc.add_argument(
        "--premask",
        dest="premask",
        action="store_true",
        help=(
            "Apply masking prior to spatial filtering to limit extracerebral sources (requires --brainmask)"
        ),
        default=False,
    )
    preproc.add_argument(
        "--premasktissueonly",
        dest="premasktissueonly",
        action="store_true",
        help=(
            "Apply more stringent masking prior to spatial filtering, removing CSF voxels (requires --graymattermask and --whitemattermask)."
        ),
        default=False,
    )
    preproc.add_argument(
        "--globalmean",
        dest="useinitregressorref",
        action="store_true",
        help=(
            "Generate a global mean regressor and use that as the reference "
            "regressor.  If no external regressor is specified, this "
            "is enabled by default."
        ),
        default=False,
    )
    preproc.add_argument(
        "--globalmeaninclude",
        dest="initregressorincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Only use voxels in mask file NAME for the initial regressor "
            "generation (if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are used)."
        ),
        default=None,
    )
    preproc.add_argument(
        "--globalmeanexclude",
        dest="initregressorexcludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Do not use voxels in mask file NAME for the initial regressor "
            "generation (if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are excluded)."
        ),
        default=None,
    )
    preproc.add_argument(
        "--motionfile",
        dest="motionfilespec",
        metavar="MOTFILE",
        help=(
            "Read 6 columns of motion regressors out of MOTFILE file (.par or BIDS .json) "
            "(with timepoints rows) and regress them and/or their derivatives "
            "out of the data prior to analysis. "
        ),
        default=None,
    )
    preproc.add_argument(
        "--motpowers",
        dest="mot_power",
        metavar="N",
        type=lambda x: pf.is_int(parser, x, minval=1),
        help=(
            "Include powers of each motion regressor up to order N. Default is 1 (no expansion). "
        ),
        default=1,
    )
    preproc.add_argument(
        "--nomotderiv",
        dest="mot_deriv",
        action="store_false",
        help=(
            "Do not use derivatives in motion regression.  Default is to use temporal derivatives."
        ),
        default=True,
    )
    preproc.add_argument(
        "--confoundfile",
        dest="confoundfilespec",
        metavar="CONFFILE",
        help=(
            "Read additional (non-motion) confound regressors out of CONFFILE file (which can be any type of "
            "multicolumn text file "
            "rapidtide reads as long as data is sampled at TR with timepoints rows).  Optionally do power expansion "
            "and/or calculate derivatives prior to regression. "
        ),
        default=None,
    )
    preproc.add_argument(
        "--confoundpowers",
        dest="confound_power",
        metavar="N",
        type=lambda x: pf.is_int(parser, x, minval=1),
        help=(
            "Include powers of each confound regressor up to order N. Default is 1 (no expansion). "
        ),
        default=1,
    )
    preproc.add_argument(
        "--noconfoundderiv",
        dest="confound_deriv",
        action="store_false",
        help=(
            "Do not use derivatives in confound regression.  Default is to use temporal derivatives."
        ),
        default=True,
    )
    preproc.add_argument(
        "--noconfoundorthogonalize",
        dest="orthogonalize",
        action="store_false",
        help=(
            "Do not orthogonalize confound regressors prior to regressing them out of the data. "
        ),
        default=True,
    )
    preproc.add_argument(
        "--globalsignalmethod",
        dest="initregressorsignalmethod",
        action="store",
        type=str,
        choices=["sum", "meanscale", "pca", "random"],
        help=(
            "The method for constructing the initial signal regressor - straight summation, "
            "mean scaling each voxel prior to summation, MLE PCA of the voxels in the initial regressor mask, "
            "or initializing using random noise."
            f'Default is "{DEFAULT_INITREGRESSOR_METHOD}."'
        ),
        default=DEFAULT_INITREGRESSOR_METHOD,
    )
    preproc.add_argument(
        "--globalpcacomponents",
        dest="initregressorpcacomponents",
        action="store",
        type=float,
        metavar="VALUE",
        help=(
            "Number of PCA components used for estimating the initial regressor.  If VALUE >= 1, will retain this"
            "many components.  If "
            "0.0 < VALUE < 1.0, enough components will be retained to explain the fraction VALUE of the "
            "total variance. If VALUE is negative, the number of components will be to retain will be selected "
            f"automatically using the MLE method.  Default is {DEFAULT_INITREGRESSOR_PCACOMPONENTS}."
        ),
        default=DEFAULT_INITREGRESSOR_PCACOMPONENTS,
    )
    preproc.add_argument(
        "--slicetimes",
        dest="slicetimes",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=("Apply offset times from FILE to each slice in the dataset."),
        default=None,
    )
    preproc.add_argument(
        "--numskip",
        dest="preprocskip",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=0),
        metavar="SKIP",
        help=(
            "SKIP TRs were previously deleted during "
            "preprocessing (e.g. if you have done your preprocessing "
            "in FSL and set dummypoints to a nonzero value.) Default is 0. "
        ),
        default=0,
    )
    preproc.add_argument(
        "--timerange",
        dest="timerange",
        action="store",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Limit analysis to data between timepoints "
            "START and END in the fmri file. If END is set to -1, "
            "analysis will go to the last timepoint.  Negative values "
            "of START will be set to 0. Default is to use all timepoints."
        ),
        default=(-1, -1),
    )
    preproc.add_argument(
        "--tincludemask",
        dest="tincludemaskname",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Only correlate during epochs specified "
            "in MASKFILE (NB: each line of FILE "
            "contains the time and duration of an "
            "epoch to include."
        ),
        default=None,
    )
    preproc.add_argument(
        "--texcludemask",
        dest="texcludemaskname",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Do not correlate during epochs specified "
            "in MASKFILE (NB: each line of FILE "
            "contains the time and duration of an "
            "epoch to exclude."
        ),
        default=None,
    )
    preproc.add_argument(
        "--nothresh",
        dest="nothresh",
        action="store_true",
        help=("Disable voxel intensity threshold (especially useful for NIRS " "data)."),
        default=False,
    )

    # Correlation options
    corr = parser.add_argument_group("Correlation options")
    corr.add_argument(
        "--oversampfac",
        dest="oversampfactor",
        action="store",
        type=int,
        metavar="OVERSAMPFAC",
        help=(
            "Oversample the fMRI data by the following "
            "integral factor.  Set to -1 for automatic selection (default)."
        ),
        default=-1,
    )
    corr.add_argument(
        "--regressor",
        dest="regressorfile",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Read the initial probe regressor from file FILE (if not "
            "specified, generate and use the global regressor)."
        ),
        default=None,
    )

    reg_group = corr.add_mutually_exclusive_group()
    reg_group.add_argument(
        "--regressorfreq",
        dest="inputfreq",
        action="store",
        type=lambda x: pf.is_float(parser, x, minval=0.0),
        metavar="FREQ",
        help=(
            "Probe regressor in file has sample "
            "frequency FREQ (default is 1/tr) "
            "NB: --regressorfreq and --regressortstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )
    reg_group.add_argument(
        "--regressortstep",
        dest="inputfreq",
        action="store",
        type=lambda x: pf.invert_float(parser, x),
        metavar="TSTEP",
        help=(
            "Probe regressor in file has sample "
            "frequency FREQ (default is 1/tr) "
            "NB: --regressorfreq and --regressortstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )

    corr.add_argument(
        "--regressorstart",
        dest="inputstarttime",
        action="store",
        type=float,
        metavar="START",
        help=(
            "The time delay in seconds into the regressor "
            "file, corresponding in the first TR of the fMRI "
            "file (default is 0.0)."
        ),
        default=None,
    )
    corr.add_argument(
        "--corrweighting",
        dest="corrweighting",
        action="store",
        type=str,
        choices=["None", "phat", "liang", "eckart", "regressor"],
        help=(
            "Method to use for cross-correlation weighting. "
            "'None' performs an unweighted correlation. "
            "'phat' weights the correlation by the magnitude of the product of the timecourse's FFTs. "
            "'liang' weights the correlation by the sum of the magnitudes of the timecourse's FFTs. "
            "'eckart' weights the correlation by the product of the magnitudes of the timecourse's FFTs. "
            "'regressor' weights the correlation by the magnitude of the sLFO regressor FFT. "
            f'Default is "{DEFAULT_CORRWEIGHTING}".'
        ),
        default=DEFAULT_CORRWEIGHTING,
    )
    corr.add_argument(
        "--corrtype",
        dest="corrtype",
        action="store",
        type=str,
        choices=["linear", "circular"],
        help=("Cross-correlation type (linear or circular). " f'Default is "{DEFAULT_CORRTYPE}".'),
        default=DEFAULT_CORRTYPE,
    )

    mask_group = corr.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--corrmaskthresh",
        dest="corrmaskthreshpct",
        action="store",
        type=float,
        metavar="PCT",
        help=(
            "Do correlations in voxels where the mean "
            "exceeds this percentage of the robust max. "
            f"Default is {DEFAULT_CORRMASK_THRESHPCT}. "
        ),
        default=DEFAULT_CORRMASK_THRESHPCT,
    )
    mask_group.add_argument(
        "--corrmask",
        dest="corrmaskincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Only do correlations in nonzero voxels in NAME "
            "(if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are used). "
        ),
        default=None,
    )
    corr.add_argument(
        "--similaritymetric",
        dest="similaritymetric",
        action="store",
        type=str,
        choices=["correlation", "mutualinfo", "hybrid"],
        help=(
            "Similarity metric for finding delay values.  "
            'Choices are "correlation", "mutualinfo", and "hybrid". '
            f"Default is {DEFAULT_SIMILARITYMETRIC}."
        ),
        default=DEFAULT_SIMILARITYMETRIC,
    )
    corr.add_argument(
        "--mutualinfosmoothingtime",
        dest="smoothingtime",
        action="store",
        type=float,
        metavar="TAU",
        help=(
            "Time constant of a temporal smoothing function to apply to the "
            "mutual information function. "
            f"Default is {DEFAULT_MUTUALINFO_SMOOTHINGTIME} seconds.  "
            "TAU <=0.0 disables smoothing."
        ),
        default=DEFAULT_MUTUALINFO_SMOOTHINGTIME,
    )
    corr.add_argument(
        "--simcalcrange",
        dest="simcalcrange",
        action="store",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Limit correlation calculation to data between timepoints "
            "START and END in the fmri file. If END is set to -1, "
            "analysis will go to the last timepoint.  Negative values "
            "of START will be set to 0. Default is to use all timepoints. "
            "NOTE: these offsets are relative to the start of the "
            "dataset AFTER any trimming done with '--timerange'."
        ),
        default=(-1, -1),
    )

    # Correlation fitting options
    corr_fit = parser.add_argument_group("Correlation fitting options")

    corr_fit.add_argument(
        "--initialdelay",
        dest="initialdelayvalue",
        type=lambda x: pf.is_valid_file_or_float(parser, x),
        metavar="DELAY",
        help=(
            "Start the analysis with predefined delays.  If DELAY is a filename, use it as an initial delay map; "
            "if DELAY is a float, initialize all voxels to a  delay of DELAY seconds."
        ),
        default=0.0,
    )

    fixdelay = corr_fit.add_mutually_exclusive_group()
    fixdelay.add_argument(
        "--searchrange",
        dest="lag_extrema",
        action=pf.IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=("LAGMIN", "LAGMAX"),
        help=(
            "Limit fit to a range of lags from LAGMIN to "
            f"LAGMAX.  Default is {DEFAULT_LAGMIN} to {DEFAULT_LAGMAX} seconds. "
        ),
        default=(DEFAULT_LAGMIN, DEFAULT_LAGMAX),
    )
    fixdelay.add_argument(
        "--nodelayfit",
        dest="fixdelay",
        action="store_true",
        help=("Fix the delay in every voxel to its initial value.  Do not perform any fitting."),
        default=False,
    )
    corr_fit.add_argument(
        "--sigmalimit",
        dest="widthmax",
        action="store",
        type=float,
        metavar="SIGMALIMIT",
        help=(
            "Reject lag fits with linewidth wider than "
            f"SIGMALIMIT Hz. Default is {DEFAULT_SIGMAMAX} Hz."
        ),
        default=DEFAULT_SIGMAMAX,
    )
    corr_fit.add_argument(
        "--bipolar",
        dest="bipolar",
        action="store_true",
        help=("Bipolar mode - match peak correlation ignoring sign."),
        default=False,
    )
    corr_fit.add_argument(
        "--nofitfilt",
        dest="zerooutbadfit",
        action="store_false",
        help=("Do not zero out peak fit values if fit fails."),
        default=True,
    )
    corr_fit.add_argument(
        "--peakfittype",
        dest="peakfittype",
        action="store",
        type=str,
        choices=["gauss", "fastgauss", "quad", "fastquad", "COM", "None"],
        help=(
            "Method for fitting the peak of the similarity function "
            '"gauss" performs a Gaussian fit, and is most accurate. '
            '"quad" and "fastquad" use a quadratic fit, '
            "which is faster, but not as well tested. "
            f'Default is "{DEFAULT_PEAKFIT_TYPE}".'
        ),
        default=DEFAULT_PEAKFIT_TYPE,
    )
    corr_fit.add_argument(
        "--despecklepasses",
        dest="despeckle_passes",
        action=pf.IndicateSpecifiedAction,
        type=lambda x: pf.is_int(parser, x, minval=0),
        metavar="PASSES",
        help=(
            "Detect and refit suspect correlations to "
            "disambiguate peak locations in PASSES "
            f"passes.  Default is to perform {DEFAULT_DESPECKLE_PASSES} passes. "
            "Set to 0 to disable."
        ),
        default=DEFAULT_DESPECKLE_PASSES,
    )
    corr_fit.add_argument(
        "--despecklethresh",
        dest="despeckle_thresh",
        action="store",
        type=float,
        metavar="VAL",
        help=(
            "Refit correlation if median discontinuity "
            "magnitude exceeds VAL. "
            f"Default is {DEFAULT_DESPECKLE_THRESH} seconds."
        ),
        default=DEFAULT_DESPECKLE_THRESH,
    )

    # Regressor refinement options
    reg_ref = parser.add_argument_group("Regressor refinement options")
    reg_ref.add_argument(
        "--refineprenorm",
        dest="refineprenorm",
        action="store",
        type=str,
        choices=["None", "mean", "var", "std", "invlag"],
        help=(
            "Apply TYPE prenormalization to each "
            "timecourse prior to refinement. "
            f'Default is "{DEFAULT_REFINE_PRENORM}".'
        ),
        default=DEFAULT_REFINE_PRENORM,
    )
    reg_ref.add_argument(
        "--refineweighting",
        dest="refineweighting",
        action="store",
        type=str,
        choices=["None", "NIRS", "R", "R2"],
        help=(
            "Apply TYPE weighting to each timecourse prior "
            f'to refinement. Default is "{DEFAULT_REFINE_WEIGHTING}".'
        ),
        default=DEFAULT_REFINE_WEIGHTING,
    )
    reg_ref.add_argument(
        "--passes",
        dest="passes",
        action=pf.IndicateSpecifiedAction,
        type=lambda x: pf.is_int(parser, x, minval=1),
        metavar="PASSES",
        help=("Set the number of processing passes to PASSES.  " f"Default is {DEFAULT_PASSES}."),
        default=DEFAULT_PASSES,
    )
    reg_ref.add_argument(
        "--refineinclude",
        dest="refineincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Only use voxels in file MASK for regressor refinement "
            "(if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are used). "
        ),
        default=None,
    )
    reg_ref.add_argument(
        "--refineexclude",
        dest="refineexcludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Do not use voxels in file MASK for regressor refinement "
            "(if VALSPEC is given, voxels "
            "with integral values listed in VALSPEC are excluded). "
        ),
        default=None,
    )
    reg_ref.add_argument(
        "--norefinedespeckled",
        dest="refinedespeckled",
        action="store_false",
        help=("Do not use despeckled pixels in calculating the refined regressor."),
        default=True,
    )
    reg_ref.add_argument(
        "--lagminthresh",
        dest="lagminthresh",
        action="store",
        metavar="MIN",
        type=float,
        help=(
            "For refinement, exclude voxels with delays "
            f"less than MIN. Default is {DEFAULT_LAGMIN_THRESH} seconds. "
        ),
        default=DEFAULT_LAGMIN_THRESH,
    )
    reg_ref.add_argument(
        "--lagmaxthresh",
        dest="lagmaxthresh",
        action="store",
        metavar="MAX",
        type=float,
        help=(
            "For refinement, exclude voxels with delays "
            f"greater than MAX. Default is {DEFAULT_LAGMAX_THRESH} seconds. "
        ),
        default=DEFAULT_LAGMAX_THRESH,
    )
    reg_ref.add_argument(
        "--ampthresh",
        dest="ampthresh",
        action="store",
        metavar="AMP",
        type=float,
        help=(
            "For refinement, exclude voxels with correlation "
            f"coefficients less than AMP (default is {DEFAULT_AMPTHRESH}).  "
            "NOTE: ampthresh will automatically be set to the p<0.05 "
            "significance level determined by the --numnull option if NREPS "
            "is set greater than 0 and this is not manually specified."
        ),
        default=-1.0,
    )
    reg_ref.add_argument(
        "--sigmathresh",
        dest="sigmathresh",
        action="store",
        metavar="SIGMA",
        type=float,
        help=(
            "For refinement, exclude voxels with widths "
            f"greater than SIGMA seconds. Default is {DEFAULT_SIGMATHRESH} seconds."
        ),
        default=DEFAULT_SIGMATHRESH,
    )
    reg_ref.add_argument(
        "--offsetinclude",
        dest="offsetincludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Only use voxels in file MASK for determining the zero time offset value "
            "(if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are used). "
        ),
        default=None,
    )
    reg_ref.add_argument(
        "--offsetexclude",
        dest="offsetexcludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Do not use voxels in file MASK for determining the zero time offset value "
            "(if VALSPEC is given, voxels "
            "with integral values listed in VALSPEC are excluded). "
        ),
        default=None,
    )
    reg_ref.add_argument(
        "--norefineoffset",
        dest="refineoffset",
        action="store_false",
        help=("Disable realigning refined regressor to zero lag."),
        default=True,
    )
    reg_ref.add_argument(
        "--nopickleft",
        dest="pickleft",
        action="store_false",
        help=("Disables selecting the leftmost delay peak when setting the refine offset."),
        default=True,
    )
    reg_ref.add_argument(
        "--pickleftthresh",
        dest="pickleftthresh",
        action="store",
        metavar="THRESH",
        type=float,
        help=(
            "Threshold value (fraction of maximum) in a histogram "
            f"to be considered the start of a peak.  Default is {DEFAULT_PICKLEFT_THRESH}."
        ),
        default=DEFAULT_PICKLEFT_THRESH,
    )
    reg_ref.add_argument(
        "--sLFOnoiseampwindow",
        dest="sLFOnoiseampwindow",
        action="store",
        metavar="SECONDS",
        type=float,
        help=(
            "Width of the averaging window for filtering the RMS sLFO waveform to calculate "
            f"amplitude change over time.  Default is {DEFAULT_SLFONOISEAMP_WINDOWSIZE}."
        ),
        default=DEFAULT_SLFONOISEAMP_WINDOWSIZE,
    )

    refine = reg_ref.add_mutually_exclusive_group()
    refine.add_argument(
        "--refineupperlag",
        dest="lagmaskside",
        action="store_const",
        const="upper",
        help=("Only use positive lags for regressor refinement."),
        default="both",
    )
    refine.add_argument(
        "--refinelowerlag",
        dest="lagmaskside",
        action="store_const",
        const="lower",
        help=("Only use negative lags for regressor refinement."),
        default="both",
    )
    reg_ref.add_argument(
        "--refinetype",
        dest="refinetype",
        action="store",
        type=str,
        choices=["pca", "ica", "weighted_average", "unweighted_average"],
        help=(
            "Method with which to derive refined regressor. "
            f'Default is "{DEFAULT_REFINE_TYPE}".'
        ),
        default=DEFAULT_REFINE_TYPE,
    )
    reg_ref.add_argument(
        "--pcacomponents",
        dest="pcacomponents",
        action="store",
        type=float,
        metavar="VALUE",
        help=(
            "Number of PCA components used for refinement.  If VALUE >= 1, will retain this many components.  If "
            "0.0 < VALUE < 1.0, enough components will be retained to explain the fraction VALUE of the "
            "total variance. If VALUE is negative, the number of components will be to retain will be selected "
            f"automatically using the MLE method.  Default is {DEFAULT_REFINE_PCACOMPONENTS}."
        ),
        default=DEFAULT_REFINE_PCACOMPONENTS,
    )
    reg_ref.add_argument(
        "--convergencethresh",
        dest="convergencethresh",
        action="store",
        type=float,
        metavar="THRESH",
        help=(
            "Continue refinement until the MSE between regressors becomes <= THRESH.  "
            "By default, this is not set, so refinement will run for the specified number of passes. "
        ),
        default=None,
    )
    reg_ref.add_argument(
        "--maxpasses",
        dest="maxpasses",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=1),
        metavar="MAXPASSES",
        help=(
            "Terminate refinement after MAXPASSES passes, whether or not convergence has occurred. "
            f"Default is {DEFAULT_MAXPASSES}."
        ),
        default=DEFAULT_MAXPASSES,
    )

    # sLFO noise removal options
    slfofilt = parser.add_argument_group("sLFO noise removal options")
    slfofilt.add_argument(
        "--nodenoise",
        dest="dolinfitfilt",
        action="store_false",
        help=(
            "Turn off regression filtering to remove delayed "
            "regressor from each voxel (disables output of "
            "fitNorm)."
        ),
        default=True,
    )
    slfofilt.add_argument(
        "--denoisesourcefile",
        dest="denoisesourcefile",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Regress delayed regressors out of FILE instead "
            "of the initial fmri file used to estimate "
            "delays."
        ),
        default=None,
    )
    slfofilt.add_argument(
        "--preservefiltering",
        dest="preservefiltering",
        action="store_true",
        help="Don't reread data prior to performing sLFO filtering.",
        default=False,
    )
    slfofilt.add_argument(
        "--regressderivs",
        dest="regressderivs",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=0),
        metavar="NDERIVS",
        help=(
            f"When doing final sLFO filtering, include derivatives up to NDERIVS order. Default is {DEFAULT_REGRESSIONFILTDERIVS}"
        ),
        default=DEFAULT_REGRESSIONFILTDERIVS,
    )
    slfofilt.add_argument(
        "--norefinedelay",
        dest="refinedelay",
        action="store_false",
        help=("Do not calculate a refined delay map using sLFO regression information."),
        default=True,
    )
    slfofilt.add_argument(
        "--nofilterwithrefineddelay",
        dest="filterwithrefineddelay",
        action="store_false",
        help=("Do not use the refined delay in sLFO filter."),
        default=True,
    )
    slfofilt.add_argument(
        "--delaypatchthresh",
        dest="delaypatchthresh",
        action="store",
        type=float,
        metavar="NUMMADs",
        help=(
            "Maximum number of robust standard deviations to permit in the offset delay refine map. "
            f"Default is {DEFAULT_PATCHTHRESH}"
        ),
        default=DEFAULT_PATCHTHRESH,
    )
    slfofilt.add_argument(
        "--delayoffsetspatialfilt",
        dest="delayoffsetgausssigma",
        action="store",
        type=float,
        metavar="GAUSSSIGMA",
        help=(
            "Spatially filter fMRI data prior to calculating delay offsets "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to have rapidtide set it to half the mean voxel "
            "dimension (a rule of thumb for a good value).  Set to 0 to disable."
        ),
        default=DEFAULT_DELAYOFFSETSPATIALFILT,
    )

    # Output options
    output = parser.add_argument_group("Output options")
    output.add_argument(
        "--outputlevel",
        dest="outputlevel",
        action="store",
        type=str,
        choices=["min", "less", "normal", "more", "max"],
        help=(
            "The level of file output produced.  'min' produces only absolutely essential files, 'less' adds in "
            "the sLFO filtered data (rather than just filter efficacy metrics), 'normal' saves what you "
            "would typically want around for interactive data exploration, "
            "'more' adds files that are sometimes useful, and 'max' outputs anything you might possibly want. "
            "Selecting 'max' will produce ~3x your input datafile size as output.  "
            f'Default is "{DEFAULT_OUTPUTLEVEL}."'
        ),
        default=DEFAULT_OUTPUTLEVEL,
    )
    output.add_argument(
        "--savelags",
        dest="savecorrtimes",
        action="store_true",
        help="Save a table of lagtimes used.",
        default=False,
    )
    output.add_argument(
        "--histlen",  # was -h
        dest="histlen",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=5),
        metavar="HISTLEN",
        help=(f"Change the histogram length to HISTLEN.  Default is {DEFAULT_HISTLEN}."),
        default=DEFAULT_HISTLEN,
    )
    output.add_argument(
        "--saveintermediatemaps",
        dest="saveintermediatemaps",
        action="store_true",
        help="Save lag times, strengths, widths, mask (and shiftedtcs, if they'd normally be saved) for each pass.",
        default=False,
    )
    output.add_argument(
        "--calccoherence",
        dest="calccoherence",
        action="store_true",
        help=("Calculate and save the coherence between the final regressor and the data."),
        default=False,
    )

    # Add version options
    pf.addversionopts(parser)

    # Performance options
    perf = parser.add_argument_group("Performance options")
    perf.add_argument(
        "--nprocs",
        dest="nprocs",
        action="store",
        type=int,
        metavar="NPROCS",
        help=(
            "Use NPROCS worker processes for multiprocessing. "
            "Setting NPROCS to less than 1 sets the number of "
            "worker processes to n_cpus (unless --reservecpu is used)."
        ),
        default=1,
    )
    perf.add_argument(
        "--reservecpu",
        dest="reservecpu",
        action="store_true",
        help=(
            "When automatically setting nprocs, reserve one CPU for "
            "process management rather than using them all for worker threads."
        ),
        default=False,
    )
    perf.add_argument(
        "--mklthreads",
        dest="mklthreads",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=1),
        metavar="MKLTHREADS",
        help=(
            "If mkl library is installed, use no more than MKLTHREADS worker "
            "threads in accelerated numpy calls.  Set to -1 to use the maximum available.  Default is 1."
        ),
        default=1,
    )
    perf.add_argument(
        "--nonumba",
        dest="nonumba",
        action="store_true",
        help=(
            "By default, numba is used if present.  Use this option to disable jit "
            "compilation with numba even if it is installed."
        ),
        default=False,
    )

    # Miscellaneous options
    misc = parser.add_argument_group("Miscellaneous options")
    misc.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help=("Will disable showing progress bars (helpful if stdout is going to a file)."),
        default=True,
    )
    misc.add_argument(
        "--makepseudofile",
        dest="makepseudofile",
        action="store_true",
        help=("Make a simulated input file from the mean and the movingsignal."),
        default=False,
    )
    misc.add_argument(
        "--checkpoint",
        dest="checkpoint",
        action="store_true",
        help="Enable run checkpoints.",
        default=False,
    )
    misc.add_argument(
        "--spcalculation",
        dest="internalprecision",
        action="store_const",
        const="single",
        help=(
            "Use single precision for internal calculations "
            "(may be useful when RAM is limited)."
        ),
        default="double",
    )
    misc.add_argument(
        "--dpoutput",
        dest="outputprecision",
        action="store_const",
        const="double",
        help=("Use double precision for output files."),
        default="single",
    )
    misc.add_argument(
        "--spatialtolerance",
        dest="spatialtolerance",
        action="store",
        type=float,
        metavar="EPSILON",
        help=(
            "When checking to see if the spatial dimensions of two NIFTI files match, allow a relative difference "
            "of EPSILON in any dimension.  By default, this is set to 0.0, requiring an exact match. "
        ),
        default=0.0,
    )
    misc.add_argument(
        "--cifti",
        dest="isgrayordinate",
        action="store_true",
        help="Data file is a converted CIFTI.",
        default=False,
    )
    misc.add_argument(
        "--simulate",
        dest="fakerun",
        action="store_true",
        help="Simulate a run - just report command line options.",
        default=False,
    )
    misc.add_argument(
        "--displayplots",
        dest="displayplots",
        action="store_true",
        help="Display plots of interesting timecourses.",
        default=False,
    )
    misc.add_argument(
        "--nosharedmem",
        dest="sharedmem",
        action="store_false",
        help=("Disable use of shared memory for large array storage."),
        default=True,
    )
    pf.addtagopts(misc)

    # Experimental options (not fully tested, may not work)
    experimental = parser.add_argument_group(
        "Experimental options (not fully tested, or not tested at all, may not work).  Beware!"
    )
    experimental.add_argument(
        "--refineregressderivs",
        dest="refineregressderivs",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=1),
        metavar="NDERIVS",
        help=(
            f"When doing regression fit for delay refinement, include derivatives up to NDERIVS order. Must be 1 or more.  "
            f"Default is {DEFAULT_REFINEREGRESSDERIVS}"
        ),
        default=DEFAULT_REFINEREGRESSDERIVS,
    )
    experimental.add_argument(
        "--dofinalrefine",
        dest="dofinalrefine",
        action="store_true",
        help=("Do regressor refinement on the final pass."),
        default=False,
    )
    experimental.add_argument(
        "--sLFOfiltmask",
        dest="sLFOfiltmask",
        action="store_true",
        help=("Limit sLFO filter to fit voxels."),
        default=False,
    )
    experimental.add_argument(
        "--territorymap",
        dest="territorymap",
        metavar="MAP[:VALSPEC]",
        help=(
            "This specifies a territory map.  Each territory is a set of voxels with the same integral value.  "
            "If VALSPEC is given, only territories in the mask with integral values listed in VALSPEC are used, otherwise "
            "all nonzero voxels are used.  If this option is set, certain output measures will be summarized over "
            "each territory in the map, in addition to over the whole brain.  Some interesting territory maps might be: "
            "a gray/white/csf segmentation image, an arterial territory map, lesion area vs. healthy "
            "tissue segmentation, etc.  NB: at the moment this is just a placeholder - it doesn't do anything."
        ),
        default=None,
    )
    experimental.add_argument(
        "--psdfilter",
        dest="psdfilter",
        action="store_true",
        help=("Apply a PSD weighted Wiener filter to shifted timecourses prior to refinement."),
        default=False,
    )
    experimental.add_argument(
        "--wiener",
        dest="dodeconv",
        action="store_true",
        help=("Do Wiener deconvolution to find voxel transfer function."),
        default=False,
    )
    experimental.add_argument(
        "--corrbaselinespatialsigma",
        dest="corrbaselinespatialsigma",
        action="store",
        type=float,
        metavar="SIGMA",
        help=("Spatial lowpass kernel, in mm, for filtering the correlation function baseline. "),
        default=0.0,
    )
    experimental.add_argument(
        "--corrbaselinetemphpfcutoff",
        dest="corrbaselinetemphpfcutoff",
        action="store",
        type=float,
        metavar="FREQ",
        help=(
            "Temporal highpass cutoff, in Hz, for filtering the correlation function baseline. "
        ),
        default=0.0,
    )
    experimental.add_argument(
        "--echocancel",
        dest="echocancel",
        action="store_true",
        help=("Attempt to perform echo cancellation on current moving regressor."),
        default=False,
    )
    experimental.add_argument(
        "--autorespdelete",
        dest="respdelete",
        action="store_true",
        help=("Attempt to detect and remove respiratory signal that strays into " "the LFO band."),
        default=False,
    )
    experimental.add_argument(
        "--acfix",
        dest="fix_autocorrelation",
        action="store_true",
        help=(
            "Check probe regressor for autocorrelations in order to disambiguate peak location."
        ),
        default=False,
    )
    experimental.add_argument(
        "--negativegradient",
        dest="negativegradient",
        action="store_true",
        help=(
            "Calculate the negative gradient of the fmri data after spectral filtering "
            "so you can look for CSF flow à la https://www.biorxiv.org/content/10.1101/2021.03.29.437406v1.full. "
        ),
        default=False,
    )
    experimental.add_argument(
        "--negativegradregressor",
        dest="negativegradregressor",
        action="store_true",
        help=argparse.SUPPRESS,
        default=False,
    )
    experimental.add_argument(
        "--cleanrefined",
        dest="cleanrefined",
        action="store_true",
        help=(
            "Perform additional processing on refined "
            "regressor to remove spurious "
            "components."
        ),
        default=False,
    )
    experimental.add_argument(
        "--donotfilterinputdata",
        dest="filterinputdata",
        action="store_false",
        help=(
            "Do not filter input data prior to similarity calculation.  May make processing faster."
        ),
        default=True,
    )
    experimental.add_argument(
        "--dispersioncalc",
        dest="dodispersioncalc",
        action="store_true",
        help=("Generate extra data during refinement to allow calculation of dispersion."),
        default=False,
    )
    experimental.add_argument(
        "--patchshift",
        dest="patchshift",
        action="store_true",
        help=("Perform patch shift correction."),
        default=False,
    )

    # Deprecated options
    deprecated = parser.add_argument_group(
        "Deprecated options.  These options will go away in the indeterminate future.  Don't use them."
    )
    deprecated.add_argument(
        "--refinedelay",
        dest="dummy",
        action="store_true",
        help=(
            "Calculate a refined delay map using regression coefficient information. ***DEPRECATED***: refinedelay is now on by default."
        ),
        default=True,
    )
    deprecated.add_argument(
        "--nolimitoutput",
        dest="limitoutput",
        action="store_false",
        help=(
            "Save some of the large and rarely used files.  "
            "***DEPRECATED***: Use '--outputlevel max' instead."
        ),
        default=True,
    )
    deprecated.add_argument(
        "--pickleft",
        dest="dummy",
        action="store_true",
        help=(
            "***DEPRECATED***: pickleft is now on by default. Use 'nopickleft' to disable it instead."
        ),
        default=True,
    )
    deprecated.add_argument(
        "--noglm",
        dest="dolinfitfilt",
        action="store_false",
        help=(
            "Turn off regression filtering to remove delayed "
            "regressor from each voxel (disables output of "
            "fitNorm)."
            "***DEPRECATED***: Use '--nodenoise' instead."
        ),
        default=True,
    )

    # Debugging options
    debugging = parser.add_argument_group(
        "Debugging options.  You probably don't want to use any of these unless I ask you to to help diagnose a problem"
    )
    debugging.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )
    debugging.add_argument(
        "--focaldebug",
        dest="focaldebug",
        action="store_true",
        help=("Enable targeted additional debugging output (used during development)."),
        default=False,
    )
    debugging.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Enable additional runtime information output."),
        default=False,
    )
    debugging.add_argument(
        "--disablecontainermemfix",
        dest="containermemfix",
        action="store_false",
        help=("Disable container memory limit setting."),
        default=True,
    )
    debugging.add_argument(
        "--alwaysmultiproc",
        dest="alwaysmultiproc",
        action="store_true",
        help=("Use the multiprocessing code path even when nprocs=1."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_confoundregress",
        dest="singleproc_confoundregress",
        action="store_true",
        help=("Force single proc path for confound regression."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_getNullDist",
        dest="singleproc_getNullDist",
        action="store_true",
        help=("Force single proc path for getNullDist."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_calcsimilarity",
        dest="singleproc_calcsimilarity",
        action="store_true",
        help=("Force single proc path for calcsimilarity."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_peakeval",
        dest="singleproc_peakeval",
        action="store_true",
        help=("Force single proc path for peakeval."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_fitcorr",
        dest="singleproc_fitcorr",
        action="store_true",
        help=("Force single proc path for fitcorr."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_refine",
        dest="singleproc_refine",
        action="store_true",
        help=("Force single proc path for refine."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_makelaggedtcs",
        dest="singleproc_makelaggedtcs",
        action="store_true",
        help=("Force single proc path for makelaggedtcs."),
        default=False,
    )
    debugging.add_argument(
        "--singleproc_regressionfilt",
        dest="singleproc_regressionfilt",
        action="store_true",
        help=("Force single proc path for regression filtering."),
        default=False,
    )
    debugging.add_argument(
        "--savecorrout",
        dest="savecorrout",
        action="store_true",
        help=("Save the corrout file even if you normally wouldn't."),
        default=False,
    )
    debugging.add_argument(
        "--isatest",
        dest="isatest",
        action="store_true",
        help=("This run of rapidtide is in a unit test."),
        default=False,
    )

    return parser


def process_args(inputargs=None):
    """
    Compile arguments for rapidtide workflow.
    """
    inargs, argstowrite = pf.setargs(_get_parser, inputargs=inputargs)
    args = vars(inargs)

    sh = logging.StreamHandler()
    if args["debug"]:
        logging.basicConfig(level=logging.DEBUG, handlers=[sh])
    else:
        logging.basicConfig(level=logging.INFO, handlers=[sh])

    # save the raw and formatted command lines
    args["commandlineargs"] = argstowrite[1:]
    thecommandline = " ".join(argstowrite)
    tide_io.writevec([thecommandline], args["outputname"] + "_commandline.txt")
    formattedcommandline = []
    for thetoken in argstowrite[0:3]:
        formattedcommandline.append(thetoken)
    for thetoken in argstowrite[3:]:
        if thetoken[0:2] == "--":
            formattedcommandline.append(thetoken)
        else:
            formattedcommandline[-1] += " " + thetoken
    for i in range(len(formattedcommandline)):
        if i > 0:
            prefix = "    "
        else:
            prefix = ""
        if i < len(formattedcommandline) - 1:
            suffix = " \\"
        else:
            suffix = ""
        formattedcommandline[i] = prefix + formattedcommandline[i] + suffix
    tide_io.writevec(formattedcommandline, args["outputname"] + "_formattedcommandline.txt")

    LGR.debug("\nbefore postprocessing:\n{}".format(args))

    # some tunable parameters for internal debugging
    args["dodemean"] = True
    # what fraction of the correlation window to avoid on either end when
    # fitting
    args["edgebufferfrac"] = 0.0
    # only do fits in voxels that exceed threshold
    args["enforcethresh"] = True
    # if set to the location of the first autocorrelation sidelobe,
    # this will fold back sidelobes
    args["lagmod"] = 1000.0
    # zero out peaks with correlations lower than this value
    args["lthreshval"] = 0.0
    # zero out peaks with correlations higher than this value
    args["uthreshval"] = 1.0
    # width of the reference autocorrelation function
    args["absmaxsigma"] = 10000.0
    # width of the reference autocorrelation function
    args["absminsigma"] = 0.05
    # number of MADs away from the median to consider an outlier
    args["sigdistoutlierfac"] = 10.0

    # correlation fitting
    # Peak value must be within specified range.
    # If false, allow max outside if maximum
    # correlation value is that one end of the range.
    args["hardlimit"] = True
    # The fraction of the main peak over which points are included in the peak
    args["searchfrac"] = 0.5
    args["mp_chunksize"] = 500
    args["patchminsize"] = DEFAULT_PATCHMINSIZE
    args["patchfwhm"] = DEFAULT_PATCHFWHM

    # significance estimation
    args["sighistlen"] = 1000
    if args["corrtype"] == "linear":
        args["corrpadding"] = -1
        # pf.setifnotset(args, "windowfunc", "None")
    else:
        args["corrpadding"] = 0

    # refinement options
    args["filterbeforePCA"] = True
    args["dispersioncalc_step"] = 0.50

    # autocorrelation processing
    args["check_autocorrelation"] = True
    args["acwidth"] = 0.0  # width of the reference autocorrelation function

    # delay refinement
    args["mindelay"] = DEFAULT_REFINEDELAYMINDELAY
    args["maxdelay"] = DEFAULT_REFINEDELAYMAXDELAY
    args["numpoints"] = DEFAULT_REFINEDELAYNUMPOINTS
    # diagnostic information about version
    (
        args["release_version"],
        args["git_sha"],
        args["git_date"],
        args["git_isdirty"],
    ) = tide_util.version()
    args["python_version"] = str(sys.version_info)

    # split infotags
    theobj = pf.postprocesstagopts(Namespace(**args))
    args = vars(theobj)

    # Additional argument parsing not handled by argparse
    args["passes"] = np.max([args["passes"], 1])

    args["despeckle_passes"] = np.max([args["despeckle_passes"], 0])

    if "lag_extrema_nondefault" in args.keys():
        args["lagmin_nondefault"] = True
        args["lagmax_nondefault"] = True

    args["lagmin"] = args["lag_extrema"][0]
    args["lagmax"] = args["lag_extrema"][1]

    # set startpoint and endpoint
    args["startpoint"], args["endpoint"] = pf.parserange(args["timerange"], descriptor="timerange")

    # set simcalc startpoint and endpoint
    args["simcalcstartpoint"], args["simcalcendpoint"] = pf.parserange(
        args["simcalcrange"], descriptor="simcalcrange"
    )

    args["offsettime_total"] = args["offsettime"] + 0.0

    reg_ref_used = (
        (args["lagminthresh"] != 0.5)
        or (args["lagmaxthresh"] != 5.0)
        or (args["ampthresh"] != DEFAULT_AMPTHRESH)
        or (args["sigmathresh"] != 100.0)
        or (args["refineoffset"])
    )
    if reg_ref_used and args["passes"] == 1:
        LGR.warning(
            "One or more arguments have been set that are only "
            "relevant if performing refinement.  "
            "If you want to do refinement, set passes > 1."
        )

    if args["numestreps"] == 0:
        args["ampthreshfromsig"] = False
    else:
        args["ampthreshfromsig"] = True

    if args["ampthresh"] < 0.0:
        args["ampthresh"] = DEFAULT_AMPTHRESH
    else:
        args["ampthreshfromsig"] = False

    if args["despeckle_thresh"] != 5.0 and args["despeckle_passes"] == 0:
        args["despeckle_passes"] = 1

    if args["zerooutbadfit"]:
        args["nohistzero"] = False
    else:
        args["nohistzero"] = True

    # sort out initial delay options
    if args["fixdelay"]:
        try:
            tempinitialdelayvalue = float(args["initialdelayvalue"])
        except ValueError:
            tempinitialdelayvalue = 0.0
        args["lag_extrema"] = (
            tempinitialdelayvalue - 10.0,
            tempinitialdelayvalue + 10.0,
        )
    else:
        args["initialdelayvalue"] = None

    if args["in_file"].endswith("txt") and args["realtr"] == "auto":
        raise ValueError(
            "Either --datatstep or --datafreq must be provided if data file is a text file."
        )

    if args["realtr"] != "auto":
        fmri_tr = float(args["realtr"])
    else:
        if tide_io.checkifcifti(args["in_file"]):
            fmri_tr, dummy = tide_io.getciftitr(nib.load(args["in_file"]).header)
        else:
            fmri_tr, dummy = tide_io.fmritimeinfo(args["in_file"])
    args["realtr"] = fmri_tr

    if args["inputfreq"] == "auto":
        args["inputfreq"] = 1.0 / fmri_tr
        args["inputfreq_nondefault"] = False
    else:
        args["inputfreq_nondefault"] = True

    # initial mask values from anatomical images (before any overrides)
    for key in [
        "corrmaskincludename",
        "corrmaskincludevals",
        "initregressorincludename",
        "initregressorincludevals",
        "initregressorexcludename",
        "initregressorexcludevals",
        "refineincludename",
        "refineincludevals",
        "refineexcludename",
        "refineexcludevals",
        "offsetincludename",
        "offsetincludevals",
        "offsetexcludename",
        "offsetexcludevals",
    ]:
        args[key] = None

    # if brainmaskincludespec is set, set initregressorinclude, corrmaskinclude, refineinclude, and offsetinclude to it.
    brainmasks = ["initregressor", "corrmask", "refine", "offset"]
    if args["brainmaskincludespec"] is not None:
        (
            args["brainmaskincludename"],
            args["brainmaskincludevals"],
        ) = tide_io.processnamespec(
            args["brainmaskincludespec"], "Limiting all processing to voxels ", "in brain mask."
        )
        if not os.path.isfile(args["brainmaskincludename"]):
            raise FileNotFoundError(f"file {args['brainmaskincludename']} does not exist.")
        for masktype in brainmasks:
            (
                args[f"{masktype}includename"],
                args[f"{masktype}includevals"],
            ) = (
                args["brainmaskincludename"],
                args["brainmaskincludevals"],
            )
            print(f"setting {masktype}include mask to gray matter mask")
    else:
        args["brainmaskincludename"] = None
        args["brainmaskincludevals"] = None

    # if graymatterincludespec is set, set initregressorinclude, offsetinclude to it.
    # graymasks = ["initregressor", "offset"]
    graymasks = ["initregressor"]
    if args["graymatterincludespec"] is not None:
        (
            args["graymatterincludename"],
            args["graymatterincludevals"],
        ) = tide_io.processnamespec(
            args["graymatterincludespec"], "Including voxels where ", "in gray matter mask."
        )
        if not os.path.isfile(args["graymatterincludename"]):
            raise FileNotFoundError(f"file {args['graymatterincludename']} does not exist.")
        for masktype in graymasks:
            (
                args[f"{masktype}includename"],
                args[f"{masktype}includevals"],
            ) = (
                args["graymatterincludename"],
                args["graymatterincludevals"],
            )
            print(f"setting {masktype}include mask to gray matter mask")
    else:
        args["graymatterincludename"] = None
        args["graymatterincludevals"] = None

    if args["whitematterincludespec"] is not None:
        (
            args["whitematterincludename"],
            args["whitematterincludevals"],
        ) = tide_io.processnamespec(
            args["whitematterincludespec"], "Including voxels where ", "in white matter mask."
        )
    else:
        args["whitematterincludename"] = None
        args["whitematterincludevals"] = None

    if args["csfincludespec"] is not None:
        (
            args["csfincludename"],
            args["csfincludevals"],
        ) = tide_io.processnamespec(
            args["csfincludespec"], "Including voxels where ", "in CSF mask."
        )
    else:
        args["csfincludename"] = None
        args["csfincludevals"] = None

    # individual mask processing (including overrides)
    if args["corrmaskincludespec"] is not None:
        (
            args["corrmaskincludename"],
            args["corrmaskincludevals"],
        ) = tide_io.processnamespec(
            args["corrmaskincludespec"],
            "Including voxels where ",
            "in correlation calculations.",
        )

    if args["initregressorincludespec"] is not None:
        (
            args["initregressorincludename"],
            args["initregressorincludevals"],
        ) = tide_io.processnamespec(
            args["initregressorincludespec"],
            "Including voxels where ",
            "in initial regressor calculation.",
        )

    if args["initregressorexcludespec"] is not None:
        (
            args["initregressorexcludename"],
            args["initregressorexcludevals"],
        ) = tide_io.processnamespec(
            args["initregressorexcludespec"],
            "Excluding voxels where ",
            "from initial regressor calculation.",
        )

    if args["refineincludespec"] is not None:
        (
            args["refineincludename"],
            args["refineincludevals"],
        ) = tide_io.processnamespec(
            args["refineincludespec"], "Including voxels where ", "in refinement."
        )

    if args["refineexcludespec"] is not None:
        (
            args["refineexcludename"],
            args["refineexcludevals"],
        ) = tide_io.processnamespec(
            args["refineexcludespec"], "Excluding voxels where ", "from refinement."
        )

    if args["offsetincludespec"] is not None:
        (
            args["offsetincludename"],
            args["offsetincludevals"],
        ) = tide_io.processnamespec(
            args["offsetincludespec"], "Including voxels where ", "in offset calculation."
        )

    if args["offsetexcludespec"] is not None:
        (
            args["offsetexcludename"],
            args["offsetexcludevals"],
        ) = tide_io.processnamespec(
            args["offsetexcludespec"], "Excluding voxels where ", "from offset calculation."
        )

    # motion processing
    if args["motionfilespec"] is not None:
        (args["motionfilename"], args["motionfilecolspec"]) = tide_io.parsefilespec(
            args["motionfilespec"]
        )
    else:
        args["motionfilename"] = None

    # process analysis modes
    if args["delaymapping"]:
        LGR.warning('Using "delaymapping" analysis mode. Overriding any affected arguments.')
        pf.setifnotset(args, "passes", DEFAULT_DELAYMAPPING_PASSES)
        pf.setifnotset(args, "despeckle_passes", DEFAULT_DELAYMAPPING_DESPECKLE_PASSES)
        pf.setifnotset(args, "lagmin", DEFAULT_DELAYMAPPING_LAGMIN)
        pf.setifnotset(args, "lagmax", DEFAULT_DELAYMAPPING_LAGMAX)
        pf.setifnotset(args, "gausssigma", DEFAULT_DELAYMAPPING_SPATIALFILT)
        args["refineoffset"] = True
        args["refinedelay"] = True
        args["outputlevel"] = "normal"
        pf.setifnotset(args, "dolinfitfilt", True)

    if args["denoising"]:
        LGR.warning('Using "denoising" analysis mode. Overriding any affected arguments.')
        pf.setifnotset(args, "passes", DEFAULT_DENOISING_PASSES)
        pf.setifnotset(args, "despeckle_passes", DEFAULT_DENOISING_DESPECKLE_PASSES)
        pf.setifnotset(args, "lagmin", DEFAULT_DENOISING_LAGMIN)
        pf.setifnotset(args, "lagmax", DEFAULT_DENOISING_LAGMAX)
        pf.setifnotset(args, "peakfittype", DEFAULT_DENOISING_PEAKFITTYPE)
        pf.setifnotset(args, "gausssigma", DEFAULT_DENOISING_SPATIALFILT)
        args["refineoffset"] = True
        args["refinedelay"] = True
        args["zerooutbadfit"] = False
        pf.setifnotset(args, "dolinfitfilt", True)

    if args["docvrmap"]:
        LGR.warning('Using "CVR" analysis mode. Overriding any affected arguments.')
        if args["regressorfile"] is None:
            raise ValueError(
                "CVR mapping requires an externally supplied regresssor file - terminating."
            )
        args["passvec"] = (
            DEFAULT_CVRMAPPING_FILTER_LOWERPASS,
            DEFAULT_CVRMAPPING_FILTER_UPPERPASS,
        )
        pf.setifnotset(args, "filterband", "arb")
        pf.setifnotset(args, "despeckle_passes", DEFAULT_CVRMAPPING_DESPECKLE_PASSES)
        pf.setifnotset(args, "lagmin", DEFAULT_CVRMAPPING_LAGMIN)
        pf.setifnotset(args, "lagmax", DEFAULT_CVRMAPPING_LAGMAX)
        args["preservefiltering"] = True
        args["passes"] = 1
        args["outputlevel"] = "min"
        args["dolinfitfilt"] = False

    if args["initregressorpreselect"]:
        LGR.warning('Using "globalpreselect" analysis mode. Overriding any affected arguments.')
        args["passes"] = 1
        args["despeckle_passes"] = 0
        args["refinedespeckle"] = False
        args["outputlevel"] = "normal"
        pf.setifnotset(args, "dolinfitfilt", False)
        args["saveintermediatemaps"] = False

    # configure the filter
    theobj, theprefilter = pf.postprocessfilteropts(Namespace(**args))
    args = vars(theobj)

    # process macros
    if args["venousrefine"]:
        LGR.warning('Using "venousrefine" macro. Overriding any affected arguments.')
        args["lagminthresh"] = 2.5
        args["lagmaxthresh"] = 6.0
        args["ampthresh"] = 0.5
        args["ampthreshfromsig"] = False
        args["lagmaskside"] = "upper"

    if args["nirs"]:
        LGR.warning('Using "nirs" macro. Overriding any affected arguments.')
        args["nothresh"] = True
        pf.setifnotset(args, "preservefiltering", False)
        args["dataiszeromean"] = True
        args["refineprenorm"] = "var"
        args["ampthresh"] = 0.7
        args["ampthreshfromsig"] = False
        args["lagmaskthresh"] = 0.1
        args["despeckle_passes"] = 0

    # process limitoutput
    if not args["limitoutput"]:
        args["outputlevel"] = "max"

    # output options
    if args["outputlevel"] == "min":
        args["saveconfoundfiltered"] = False
        args["savegaussout"] = False
        args["savecorrtimes"] = False
        args["savelagregressors"] = False
        args["savedespecklemasks"] = False
        args["saveminimumsLFOfiltfiles"] = False
        args["savenormalsLFOfiltfiles"] = False
        args["savemovingsignal"] = False
        args["saveallsLFOfiltfiles"] = False
    elif args["outputlevel"] == "less":
        args["saveconfoundfiltered"] = False
        args["savegaussout"] = False
        args["savecorrtimes"] = False
        args["savelagregressors"] = False
        args["savedespecklemasks"] = False
        args["saveminimumsLFOfiltfiles"] = True
        args["savenormalsLFOfiltfiles"] = False
        args["savemovingsignal"] = False
        args["saveallsLFOfiltfiles"] = False
    elif args["outputlevel"] == "normal":
        args["saveconfoundfiltered"] = False
        args["savegaussout"] = False
        args["savecorrtimes"] = False
        args["savelagregressors"] = False
        args["savedespecklemasks"] = False
        args["saveminimumsLFOfiltfiles"] = True
        args["savenormalsLFOfiltfiles"] = True
        args["savemovingsignal"] = False
        args["saveallsLFOfiltfiles"] = False
    elif args["outputlevel"] == "more":
        args["saveconfoundfiltered"] = False
        args["savegaussout"] = False
        args["savecorrtimes"] = False
        args["savelagregressors"] = True
        args["savedespecklemasks"] = False
        args["saveminimumsLFOfiltfiles"] = True
        args["savenormalsLFOfiltfiles"] = True
        args["savemovingsignal"] = True
        args["saveallsLFOfiltfiles"] = False
    elif args["outputlevel"] == "max":
        args["saveconfoundfiltered"] = True
        args["savegaussout"] = True
        args["savecorrtimes"] = True
        args["savelagregressors"] = True
        args["savedespecklemasks"] = True
        args["saveminimumsLFOfiltfiles"] = True
        args["savenormalsLFOfiltfiles"] = True
        args["savemovingsignal"] = True
        args["saveallsLFOfiltfiles"] = True
    else:
        print(f"illegal output level {args['outputlevel']}")
        sys.exit()

    # dispersion calculation
    args["dispersioncalc_lower"] = args["lagmin"]
    args["dispersioncalc_upper"] = args["lagmax"]
    args["dispersioncalc_step"] = np.max(
        [
            (args["dispersioncalc_upper"] - args["dispersioncalc_lower"]) / 25,
            args["dispersioncalc_step"],
        ]
    )

    if args["territorymap"] is not None:
        (
            args["territorymapname"],
            args["territorymapincludevals"],
        ) = tide_io.processnamespec(
            args["territorymap"], "Including voxels where ", "in initial regressor calculation."
        )
    else:
        args["territorymapname"] = None
        args["territorymapincludevals"] = None

    # this is new enough to do retrospective regression filtering
    args["retroregresscompatible"] = True

    LGR.debug("\nafter postprocessing\n{}".format(args))

    # start the clock!
    tide_util.checkimports(args)

    return args, theprefilter
