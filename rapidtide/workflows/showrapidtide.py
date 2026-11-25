"""Command-line rendering of 2D Rapidtide analysis results."""

import argparse
import math

import rapidtide.workflows.parser_funcs as pf
from rapidtide.maprender import render_rapidtide


def _positive_float(value):
    """
    Parse a command-line value as a strictly positive number.

    Parameters
    ----------
    value : str
        Text supplied on the command line.

    Returns
    -------
    float
        Parsed positive value.
    """
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _get_parser():
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for this command.
    """
    parser = argparse.ArgumentParser(
        prog="showrapidtide",
        description="Render delay/correlation maps and a corrout lag-sweep video from 2D TIFF/Zarr outputs.",
        allow_abbrev=False,
    )
    parser.add_argument("outputroot", help="Rapidtide output prefix (without _desc-* suffixes)")
    parser.add_argument("--outdir", help="Output directory (default: OUTPUTROOT_figures)")
    parser.add_argument(
        "--fps", type=_positive_float, default=30, help="Movie playback FPS (default: 30)"
    )
    parser.add_argument(
        "--minlag", type=float, help="Min delay color limit in seconds"
    )
    parser.add_argument(
        "--maxlag", type=float, help="Max delay color limit in seconds"
    )
    parser.add_argument(
        "--mincorr",
        type=float,
        default=-1.0,
        help="Lower correlation color limit for maps and movie (default: -1)",
    )
    parser.add_argument(
        "--maxcorr",
        type=float,
        default=1.0,
        help="Upper correlation color limit for maps and movie (default: 1)",
    )
    parser.add_argument(
        "--videolagrange",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Include only saved corrout lags from MIN to MAX seconds (inclusive)",
    )
    parser.add_argument("--nomovie", action="store_true", help="Generate only the static maps")
    parser.add_argument("--noprogressbar", action="store_true", help="Disable encoding progress")
    pf.addversionopts(parser)
    return parser


def main(args):
    """
    Render artifacts using the standard Rapidtide parser/workflow pattern.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options for the command.
    """
    #try:
    paths = render_rapidtide(
        args.outputroot,
        outdir=args.outdir,
        fps=args.fps,
        minlag=args.minlag,
        maxlag=args.maxlag,
        movie=not args.nomovie,
        showprogress=not args.noprogressbar,
        videolagrange=args.videolagrange,
        mincorr=args.mincorr,
        maxcorr=args.maxcorr,
    )
    #except (ImportError, OSError, ValueError, KeyError) as exc:
    #    raise SystemExit(f"showrapidtide: {exc}") from exc
    for path in paths.values():
        print(path)
