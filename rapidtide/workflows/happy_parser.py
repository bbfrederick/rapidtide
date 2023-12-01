#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2019-2021 Blaise Frederick
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
import sys

import numpy as np

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for happy
    """
    parser = argparse.ArgumentParser(
        prog="happy",
        description="Hypersampling by Analytic Phase Projection - Yay!.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "fmrifilename",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The input data file (BOLD fmri file or NIRS text file)",
    )
    parser.add_argument(
        "slicetimename",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "Text file containing the offset time in seconds of each slice relative "
            "to the start of the TR, one value per line, OR the BIDS sidecar JSON file."
            "NB: FSL slicetime files give slice times in fractions of a TR, BIDS sidecars "
            "give slice times in seconds. Non-json files are assumed to be the FSL style "
            "(fractions of a TR) UNLESS the --slicetimesareinseconds flag is used."
        ),
    )
    parser.add_argument("outputroot", help="The root name for the output files")

    # Processing steps
    processing_steps = parser.add_argument_group("Processing steps")
    processing_steps.add_argument(
        "--cardcalconly",
        dest="cardcalconly",
        action="store_true",
        help="Stop after all cardiac regressor calculation steps (before phase projection). ",
        default=False,
    )
    processing_steps.add_argument(
        "--skipdlfilter",
        dest="dodlfilter",
        action="store_false",
        help="Disable deep learning cardiac waveform filter.  ",
        default=True,
    )
    processing_steps.add_argument(
        "--usesuperdangerousworkaround",
        dest="mpfix",
        action="store_true",
        help=(
            "Some versions of tensorflow seem to have some weird conflict with MKL which"
            "I don't seem to be able to fix.  If the dl filter bombs complaining about "
            "multiple openmp libraries, try rerunning with the secret and inadvisable "
            "'--usesuperdangerousworkaround' flag.  Good luck! "
        ),
        default=False,
    )
    processing_steps.add_argument(
        "--slicetimesareinseconds",
        action="store_true",
        help=(
            "If a non-json slicetime file is specified, happy assumes the file is FSL style "
            "(slice times are specified in fractions of a TR).  Setting this flag overrides this "
            "assumption, and interprets the slice time file as being in seconds.  This does "
            "nothing when the slicetime file is a .json BIDS sidecar."
        ),
        default=False,
    )
    processing_steps.add_argument(
        "--model",
        dest="modelname",
        metavar="MODELNAME",
        help=(
            "Use model MODELNAME for dl filter (default is model_revised - "
            "from the revised NeuroImage paper. "
        ),
        default="model_revised",
    )

    # Performance
    performance_opts = parser.add_argument_group("Performance")
    performance_opts.add_argument(
        "--mklthreads",
        dest="mklthreads",
        action="store",
        metavar="NTHREADS",
        type=lambda x: pf.is_int(parser, x),
        help=(
            "Use NTHREADS MKL threads to accelerate processing (defaults to 1 - more "
            "threads up to the number of cores can accelerate processing a lot, but "
            "can really kill you on clusters unless you're very careful.  Use at your own risk"
        ),
        default=1,
    )

    # Preprocessing
    preprocessing_opts = parser.add_argument_group("Preprocessing")
    preprocessing_opts.add_argument(
        "--numskip",
        dest="numskip",
        action="store",
        metavar="SKIP",
        type=lambda x: pf.is_int(parser, x),
        help="Skip SKIP tr's at the beginning of the fMRI file (default is 0). ",
        default=0,
    )
    preprocessing_opts.add_argument(
        "--motskip",
        dest="motskip",
        action="store",
        metavar="SKIP",
        type=lambda x: pf.is_int(parser, x),
        help="Skip SKIP tr's at the beginning of the motion regressor file (default is 0). ",
        default=0,
    )
    preprocessing_opts.add_argument(
        "--motionfile",
        dest="motionfilename",
        metavar="MOTFILE",
        help=(
            "Read 6 columns of motion regressors out of MOTFILE file (.par or BIDS .json) "
            "(with timepoints rows) and regress them, their derivatives, "
            "and delayed derivatives out of the data prior to analysis. "
        ),
        default=None,
    )
    preprocessing_opts.add_argument(
        "--motionhp",
        dest="motionhp",
        action="store",
        metavar="HPFREQ",
        type=lambda x: pf.is_float(parser, x),
        help="Highpass filter motion regressors to HPFREQ Hz prior to regression. ",
        default=None,
    )
    preprocessing_opts.add_argument(
        "--motionlp",
        dest="motionlp",
        action="store",
        metavar="LPFREQ",
        type=lambda x: pf.is_float(parser, x),
        help="Lowpass filter motion regressors to LPFREQ Hz prior to regression. ",
        default=None,
    )
    preprocessing_opts.add_argument(
        "--nomotorthogonalize",
        dest="orthogonalize",
        action="store_false",
        help=(
            "Do not orthogonalize motion regressors prior to regressing them out of the " "data. "
        ),
        default=True,
    )
    preprocessing_opts.add_argument(
        "--motpos",
        dest="motfilt_pos",
        action="store_true",
        help=("Include motion position regressors. "),
        default=False,
    )
    preprocessing_opts.add_argument(
        "--nomotderiv",
        dest="motfilt_deriv",
        action="store_false",
        help=("Do not use motion derivative regressors. "),
        default=True,
    )
    preprocessing_opts.add_argument(
        "--nomotderivdelayed",
        dest="motfilt_derivdelayed",
        action="store_false",
        help=("Do not use delayed motion derivative regressors. "),
        default=True,
    )
    preprocessing_opts.add_argument(
        "--discardmotionfiltered",
        dest="savemotionglmfilt",
        action="store_false",
        help=("Do not save data after motion filtering. "),
        default=True,
    )

    # Cardiac estimation tuning
    cardiac_est_tuning = parser.add_argument_group("Cardiac estimation tuning")
    cardiac_est_tuning.add_argument(
        "--estmask",
        dest="estmaskname",
        action="store",
        metavar="MASKNAME",
        help=(
            "Generation of cardiac waveform from data will be restricted to "
            "voxels in MASKNAME and weighted by the mask intensity.  If this is "
            "selected, happy will only make a single pass through the data (the "
            "initial vessel mask generation pass will be skipped)."
        ),
        default=None,
    )
    cardiac_est_tuning.add_argument(
        "--minhr",
        dest="minhr",
        action="store",
        metavar="MINHR",
        type=lambda x: pf.is_float(parser, x),
        help="Limit lower cardiac frequency search range to MINHR BPM (default is 40). ",
        default=40.0,
    )
    cardiac_est_tuning.add_argument(
        "--maxhr",
        dest="maxhr",
        action="store",
        metavar="MAXHR",
        type=lambda x: pf.is_float(parser, x),
        help="Limit upper cardiac frequency search range to MAXHR BPM (default is 140). ",
        default=140.0,
    )
    cardiac_est_tuning.add_argument(
        "--minhrfilt",
        dest="minhrfilt",
        action="store",
        metavar="MINHR",
        type=lambda x: pf.is_float(parser, x),
        help="Highpass filter cardiac waveform estimate to MINHR BPM (default is 40). ",
        default=40.0,
    )
    cardiac_est_tuning.add_argument(
        "--maxhrfilt",
        dest="maxhrfilt",
        action="store",
        metavar="MAXHR",
        type=lambda x: pf.is_float(parser, x),
        help="Lowpass filter cardiac waveform estimate to MAXHR BPM (default is 1000). ",
        default=1000.0,
    )
    cardiac_est_tuning.add_argument(
        "--hilbertcomponents",
        dest="hilbertcomponents",
        action="store",
        metavar="NCOMPS",
        type=lambda x: pf.is_int(parser, x),
        help="Retain NCOMPS components of the cardiac frequency signal to Hilbert transform (default is 1). ",
        default=1,
    )
    cardiac_est_tuning.add_argument(
        "--envcutoff",
        dest="envcutoff",
        action="store",
        metavar="CUTOFF",
        type=lambda x: pf.is_float(parser, x),
        help="Lowpass filter cardiac normalization envelope to CUTOFF Hz (default is 0.4 Hz). ",
        default=0.4,
    )
    cardiac_est_tuning.add_argument(
        "--notchwidth",
        dest="notchpct",
        action="store",
        metavar="WIDTH",
        type=lambda x: pf.is_float(parser, x),
        help="Set the width of the notch filter, in percent of the notch frequency (default is 1.5). ",
        default=1.5,
    )
    cardiac_est_tuning.add_argument(
        "--invertphysiosign",
        dest="invertphysiosign",
        action="store_true",
        help=(
            "Invert the waveform extracted from the physiological signal.  "
            "Use this if there is a contrast agent in the blood. "
        ),
        default=False,
    )

    # External cardiac waveform options
    external_cardiac_opts = parser.add_argument_group("External cardiac waveform options")
    external_cardiac_opts.add_argument(
        "--cardiacfile",
        dest="cardiacfilename",
        metavar="FILE[:COL]",
        help=(
            "Read the cardiac waveform from file FILE.  If COL is an integer, "
            "and FILE is a text file, use the COL'th column.  If FILE is a BIDS "
            "format json file, use column named COL. If no file is specified, "
            "estimate the cardiac signal from the fMRI data."
        ),
        default=None,
    )
    cardiac_freq = external_cardiac_opts.add_mutually_exclusive_group()
    cardiac_freq.add_argument(
        "--cardiacfreq",
        dest="inputfreq",
        action="store",
        metavar="FREQ",
        type=lambda x: pf.is_float(parser, x),
        help=(
            "Cardiac waveform in cardiacfile has sample frequency FREQ "
            "(default is 32Hz). NB: --cardiacfreq and --cardiactstep "
            "are two ways to specify the same thing. "
        ),
        default=-32.0,
    )
    cardiac_freq.add_argument(
        "--cardiactstep",
        dest="inputfreq",
        action="store",
        metavar="TSTEP",
        type=lambda x: pf.invert_float(parser, x),
        help=(
            "Cardiac waveform in cardiacfile has time step TSTEP "
            "(default is 1/32 sec). NB: --cardiacfreq and --cardiactstep "
            "are two ways to specify the same thing. "
        ),
        default=-32.0,
    )
    external_cardiac_opts.add_argument(
        "--cardiacstart",
        dest="inputstart",
        metavar="START",
        action="store",
        type=float,
        help=(
            "The time delay in seconds into the cardiac file, corresponding "
            "to the first TR of the fMRI file (default is 0.0) "
        ),
        default=None,
    )
    external_cardiac_opts.add_argument(
        "--forcehr",
        dest="forcedhr",
        metavar="BPM",
        action="store",
        type=lambda x: pf.is_float(parser, x) / 60.0,
        help=(
            "Force heart rate fundamental detector to be centered at BPM "
            "(overrides peak frequencies found from spectrum).  Useful"
            "if there is structured noise that confuses the peak finder. "
        ),
        default=None,
    )

    respiration = True
    if respiration:
        # External respiration waveform options
        external_respiration_opts = parser.add_argument_group(
            "External respiration waveform options"
        )
        external_respiration_opts.add_argument(
            "--respirationfile",
            dest="respirationfilename",
            metavar="FILE[:COL]",
            help=(
                "Read the respiration waveform from file FILE.  If COL is an integer, "
                "and FILE is a text file, use the COL'th column.  If FILE is a BIDS "
                "format json file, use column named COL."
            ),
            default=None,
        )
        respiration_freq = external_respiration_opts.add_mutually_exclusive_group()
        respiration_freq.add_argument(
            "--respirationfreq",
            dest="respinputfreq",
            action="store",
            metavar="FREQ",
            type=lambda x: pf.is_float(parser, x),
            help=(
                "Respiration waveform in respirationfile has sample frequency FREQ "
                "(default is 32Hz). NB: --respirationfreq and --respirationtstep "
                "are two ways to specify the same thing. "
            ),
            default=-32.0,
        )
        respiration_freq.add_argument(
            "--respirationtstep",
            dest="respinputfreq",
            action="store",
            metavar="TSTEP",
            type=lambda x: pf.invert_float(parser, x),
            help=(
                "Respiration waveform in respirationfile has time step TSTEP "
                "(default is 1/32 sec). NB: --respirationfreq and --respirationtstep "
                "are two ways to specify the same thing. "
            ),
            default=-32.0,
        )
        external_respiration_opts.add_argument(
            "--respirationstart",
            dest="respinputstart",
            metavar="START",
            action="store",
            type=float,
            help=(
                "The time delay in seconds into the respiration file, corresponding "
                "to the first TR of the fMRI file (default is 0.0) "
            ),
            default=None,
        )
        external_respiration_opts.add_argument(
            "--forcerr",
            dest="forcedrr",
            metavar="BreathsPM",
            action="store",
            type=lambda x: pf.is_float(parser, x) / 60.0,
            help=(
                "Force respiratory rate fundamental detector to be centered at BreathsPM "
                "(overrides peak frequencies found from spectrum).  Useful"
                "if there is structured noise that confuses the peak finder. "
            ),
            default=None,
        )

    # Output processing
    output_proc = parser.add_argument_group("Output processing")
    output_proc.add_argument(
        "--spatialglm",
        dest="dospatialglm",
        action="store_true",
        help="Generate framewise cardiac signal maps and filter them out of the input data. ",
        default=False,
    )
    output_proc.add_argument(
        "--temporalglm",
        dest="dotemporalglm",
        action="store_true",
        help="Generate voxelwise aliased synthetic cardiac regressors and filter them out of the input data. ",
        default=False,
    )

    # Output options
    output = parser.add_argument_group("Output options")
    output.add_argument(
        "--stdfreq",
        dest="stdfreq",
        metavar="FREQ",
        action="store",
        type=float,
        help=(
            "Frequency to which the physiological signals are resampled for output. "
            "Default is 25. "
        ),
        default=25.0,
    )

    # Phase projection tuning
    phase_proj_tuning = parser.add_argument_group("Phase projection tuning")
    phase_proj_tuning.add_argument(
        "--outputbins",
        dest="destpoints",
        metavar="BINS",
        action="store",
        type=lambda x: pf.is_int(parser, x),
        help="Number of output phase bins (default is 32). ",
        default=32,
    )
    phase_proj_tuning.add_argument(
        "--gridbins",
        dest="congridbins",
        metavar="BINS",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help="Width of the gridding kernel in output phase bins (default is 3.0). ",
        default=3.0,
    )
    phase_proj_tuning.add_argument(
        "--gridkernel",
        dest="gridkernel",
        action="store",
        type=str,
        choices=["old", "gauss", "kaiser"],
        help="Convolution gridding kernel. Default is kaiser",
        default="kaiser",
    )
    phase_proj_tuning.add_argument(
        "--projmask",
        dest="projmaskname",
        metavar="MASKNAME",
        help=(
            "Phase projection will be restricted to voxels in MASKNAME "
            "(overrides normal intensity mask.) "
        ),
        default=None,
    )
    phase_proj_tuning.add_argument(
        "--projectwithraw",
        dest="projectwithraw",
        action="store_true",
        help="Use fMRI derived cardiac waveform as phase source for projection, even if a plethysmogram is supplied.",
        default=False,
    )
    phase_proj_tuning.add_argument(
        "--fliparteries",
        dest="fliparteries",
        action="store_true",
        help=(
            "Attempt to detect arterial signals and flip over the timecourses after phase projection "
            "(since relative arterial blood susceptibility is inverted relative to venous blood)."
        ),
        default=False,
    )
    phase_proj_tuning.add_argument(
        "--arteriesonly",
        dest="arteriesonly",
        action="store_true",
        help="Restrict cardiac waveform estimation to putative arteries only.",
        default=False,
    )

    # Add version options
    pf.addversionopts(parser)

    # Add miscellaneous options
    misc_opts = parser.add_argument_group("Miscellaneous options.")
    misc_opts.add_argument(
        "--aliasedcorrelation",
        dest="doaliasedcorrelation",
        action="store_true",
        help="Attempt to calculate absolute delay using an aliased correlation (experimental).",
        default=False,
    )
    misc_opts.add_argument(
        "--upsample",
        dest="doupsampling",
        action="store_true",
        help="Attempt to temporally upsample the fMRI data (experimental).",
        default=False,
    )
    misc_opts.add_argument(
        "--estimateflow",
        dest="doflowfields",
        action="store_true",
        help="Estimate blood flow using optical flow (experimental).",
        default=False,
    )
    misc_opts.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help="Will disable showing progress bars (helpful if stdout is going to a file). ",
        default=True,
    )
    pf.addtagopts(
        misc_opts,
        helptext="Additional key, value pairs to add to the info json file (useful for tracking analyses).",
    )

    # Debugging options
    debug_opts = parser.add_argument_group("Debugging options (probably not of interest to users)")
    debug_opts.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Turn on debugging information.",
        default=False,
    )
    debug_opts.add_argument(
        "--nodetrend",
        dest="detrendorder",
        action="store",
        type=lambda x: pf.is_int(parser, 0),
        help="Disable data detrending. ",
        default=3,
    )
    debug_opts.add_argument(
        "--noorthog",
        dest="orthogonalize",
        action="store_false",
        help="Disable orthogonalization of motion confound regressors. ",
        default=True,
    )
    debug_opts.add_argument(
        "--disablenotch",
        dest="disablenotch",
        action="store_true",
        help="Disable subharmonic notch filter. ",
        default=False,
    )
    debug_opts.add_argument(
        "--nomask",
        dest="usemaskcardfromfmri",
        action="store_false",
        help="Disable data masking for calculating cardiac waveform. ",
        default=True,
    )
    debug_opts.add_argument(
        "--nocensor",
        dest="censorbadpts",
        action="store_false",
        help="Bad points will not be excluded from analytic phase projection. ",
        default=True,
    )
    debug_opts.add_argument(
        "--noappsmooth",
        dest="smoothapp",
        action="store_false",
        help="Disable smoothing app file in the phase direction. ",
        default=True,
    )
    debug_opts.add_argument(
        "--nophasefilt",
        dest="filtphase",
        action="store_false",
        help="Disable the phase trend filter (probably not a good idea). ",
        default=True,
    )
    debug_opts.add_argument(
        "--nocardiacalign",
        dest="aligncardiac",
        action="store_false",
        help="Disable alignment of pleth signal to fMRI derived cardiac signal. ",
        default=True,
    )
    debug_opts.add_argument(
        "--saveinfoastext",
        dest="saveinfoasjson",
        action="store_false",
        help="Save the info file in text format rather than json. ",
        default=True,
    )
    debug_opts.add_argument(
        "--saveintermediate",
        dest="saveintermediate",
        action="store_true",
        help="Save some data from intermediate passes to help debugging. ",
        default=False,
    )
    debug_opts.add_argument(
        "--increaseoutputlevel",
        dest="inc_outputlevel",
        action="count",
        help="Increase the number of intermediate output files. ",
        default=0,
    )
    debug_opts.add_argument(
        "--decreaseoutputlevel",
        dest="dec_outputlevel",
        action="count",
        help="Decrease the number of intermediate output files. ",
        default=0,
    )

    return parser


def process_args(inputargs=None):
    """
    Compile arguments for rapidtide workflow.
    """
    if inputargs is None:
        print("processing command line arguments")
        # write out the command used
        try:
            args = _get_parser().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            _get_parser().print_help()
            raise
    else:
        print("processing passed argument list:")
        try:
            args = _get_parser().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            print("Use --help option for detailed informtion on options.")
            raise

    # save the raw and formatted command lines
    args.commandline = " ".join(argstowrite)
    tide_io.writevec([args.commandline], args.outputroot + "_commandline.txt")
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
    tide_io.writevec(formattedcommandline, args.outputroot + "_formattedcommandline.txt")

    if args.debug:
        print()
        print("before postprocessing")
        print(args)

    # some tunable parameters
    args.outputlevel = 1
    args.maskthreshpct = 10.0
    args.domadnorm = True
    args.nprocs = 1
    args.verbose = False
    args.smoothlen = 101
    args.envthresh = 0.2
    args.upsamplefac = 100
    args.centric = True
    args.pulsereconstepsize = 0.01
    args.aliasedcorrelationwidth = 3.0
    args.unnormvesselmap = True
    args.histlen = 100
    args.softvesselfrac = 0.4
    args.savecardiacnoise = True
    args.colnum = None
    args.colname = None

    # Additional argument parsing not handled by argparse
    # deal with notch filter logic
    if args.disablenotch:
        args.notchpct = None

    # process infotags
    args = pf.postprocesstagopts(args)

    # determine the outputlevel
    args.outputlevel = np.max([0, args.outputlevel + args.inc_outputlevel - args.dec_outputlevel])

    if args.debug:
        print()
        print("after postprocessing")
        print(args)

    # start the clock!
    # tide_util.checkimports(args)

    return args
