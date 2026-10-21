"""Command-line workflow for offline video-to-blood conversion."""

import argparse

import rapidtide.workflows.parser_funcs as pf
from rapidtide.video import video_to_blood


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _get_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("outfilename", help="New output Zarr directory (must not exist)")
    parser.add_argument(
        "--method", type=str.lower, choices=("grgb", "pos", "chrom"), default="grgb"
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
    parser.add_argument("--noprogressbar", action="store_true", help="Disable progress bars")
    pf.addversionopts(parser)
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the converter with parsed command-line arguments."""
    try:
        output = video_to_blood(
            args.infilename,
            args.outfilename,
            method=args.method,
            chunks=tuple(args.chunks),
            framerate=args.framerate,
            showprogress=not args.noprogressbar,
            commandline=getattr(args, "commandline", None),
        )
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(f"videotoblood: {exc}") from exc
    print(f"Saved {args.method.upper()} blood signals to {output}")
