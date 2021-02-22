def addfilteropts(parser):
    filt_opts = parser.add_argument_group("Filtering options")
    filt_opts.add_argument(
        "--filterband",
        dest="filterband",
        action="store",
        type=str,
        choices=["vlf", "lfo", "resp", "cardiac", "lfo_legacy"],
        help=("Filter data and regressors to specific band. "),
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
            "Filter data and regressors to retain LOWERPASS to "
            "UPPERPASS. LOWERSTOP and UPPERSTOP can also "
            "be specified, or will be calculated "
            "automatically. "
        ),
        default=None,
    )
    filt_opts.add_argument(
        "--filtertype",
        dest="filtertype",
        action="store",
        type=str,
        choices=["trapezoidal", "brickwall", "butterworth"],
        help=(
            "Filter data and regressors using a trapezoidal FFT filter (default), brickwall, or butterworth bandpass."
        ),
        default="trapezoidal",
    )
    filt_opts.add_argument(
        "--butterorder",
        dest="filtorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=("Set order of butterworth filter for band splitting. "),
        default=6,
    )
    filt_opts.add_argument(
        "--padseconds",
        dest="padseconds",
        action="store",
        type=float,
        metavar="SECONDS",
        help=("The number of seconds of padding to add to each end of a filtered timecourse. "),
        default=30.0,
    )


def addsamplerateopts(parser):
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        metavar="FREQ",
        type=lambda x: is_float(parser, x),
        help=(
            "Set the sample rate of the data file to FREQ. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    sampling.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        metavar="TSTEP",
        type=lambda x: invert_float(parser, x),
        help=(
            "Set the sample rate of the data file to 1.0/TSTEP. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
