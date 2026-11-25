"""Command-line workflow for offline video-to-blood conversion."""

import argparse
import math

import rapidtide.workflows.parser_funcs as pf
from rapidtide.video import video_to_blood


def _positive_int(value: str) -> int:
    """
    Parse a command-line value as a strictly positive integer.

    Parameters
    ----------
    value : str
        Text supplied on the command line.

    Returns
    -------
    int
        Parsed positive integer.
    """
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_float(value: str) -> float:
    """
    Parse a command-line value as a nonnegative number.

    Parameters
    ----------
    value : str
        Text supplied on the command line.

    Returns
    -------
    float
        Parsed nonnegative value.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _get_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for this command.
    """
    parser = argparse.ArgumentParser(
        prog="videotoblood",
        description=(
            "Convert RGB video to mean-removed float32 Zarr blood signals using "
            "whole-recording per-pixel statistics and linear frame interpolation."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "infilename", type=lambda x: pf.is_valid_file(parser, x), help="Input video"
    )
    parser.add_argument("outfilename", help="Output Zarr directory (overwritten if it exists)")
    parser.add_argument(
        "--method", type=str.lower, choices=("grgb", "pos", "chrom"), default="grgb"
    )
    parser.add_argument(
        "--skinmask",
        action="store_true",
        help="Infer skin on original RGB; apply mask after --lpf and before --downsamp",
    )
    parser.add_argument(
        "--skinmaskmodel",
        metavar="FILE",
        type=lambda x: pf.is_valid_file(parser, x),
        help="Local multiclass model for --skinmask (default: download once to the user cache)",
    )
    parser.add_argument(
        "--downsamp",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Average N×N RGB blocks after --lpf and centered cropping; odd remainders favor the high side (default: 1)",
    )
    parser.add_argument(
        "--lpf",
        type=_nonnegative_float,
        default=0.0,
        metavar="SIGMA",
        help="2D Gaussian sigma in source pixels for each RGB channel, before --downsamp (default: 0)",
    )
    parser.add_argument(
        "--chunks",
        type=_positive_int,
        nargs=3,
        metavar=("Y", "X", "TIME"),
        default=(64, 64, 16),
        help="Zarr chunk shape; TIME also controls output buffering (default: 64 64 16)",
    )
    parser.add_argument(
        "--framerate",
        type=float,
        help="Output rate in Hz for linear frame interpolation (default: nominal video rate)",
    )
    parser.add_argument(
        "--bloodvideo",
        type=str,
        metavar="VIDEOFILE",
        default=None,
        help=(
            "Also write the blood signal as a monochrome 8-bit H.264 video (e.g. blood.mp4) "
            "for debugging and validation. Gray 128 is zero signal, invalid pixels are black, "
            "and the gray scale is fixed over the recording. Existing video is overwritten (default: off)"
        ),
    )
    parser.add_argument("--noprogressbar", action="store_true", help="Disable progress bars")
    pf.addversionopts(parser)
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Run the converter with parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options for the command.
    """
    try:
        output = video_to_blood(
            args.infilename,
            args.outfilename,
            method=args.method,
            chunks=tuple(args.chunks),
            framerate=args.framerate,
            showprogress=not args.noprogressbar,
            commandline=getattr(args, "commandline", None),
            bloodvideo=args.bloodvideo,
            downsamp=args.downsamp,
            lpf=args.lpf,
            skinmask=args.skinmask,
            skinmask_model=args.skinmaskmodel,
        )
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(f"videotoblood: {exc}") from exc
    print(f"Saved {args.method.upper()} blood signals to {output}")
    if args.bloodvideo is not None:
        print(f"Saved monochrome blood video to {args.bloodvideo}")
