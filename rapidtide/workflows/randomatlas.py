#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
import numpy as np
import rapidtide.regionops as tide_regionops
import rapidtide.io as tide_io
from typing import Any
import rapidtide.workflows.parser_funcs as pf

DEFAULT_RNGSEED = 42
DEFAULT_ALPHA = 0.5
DEFAULT_ANISOTROPY_STRENGTH = 0.0


def _get_parser() -> Any:
    """
    Argument parser for randomatlas.

    Creates and configures an argument parser for the randomatlas command-line tool
    that splits a mask up into numregions random simply connected regions.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with required command-line arguments


    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="randomatlas",
        description="Partitions a NIFTI mask into numregions random regions.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputfileroot", type=str, help="The root of the output nifti file.")
    parser.add_argument(
        "numregions",
        type=int,
        help="The number of regions into which to segment the atlas.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        metavar="SEED",
        action="store",
        type=lambda x: pf.is_int(parser, x),
        help=f"Random number seed (default is {DEFAULT_RNGSEED}). ",
        default=DEFAULT_RNGSEED,
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        metavar="ALPHA",
        action="store",
        type=lambda x: pf.is_float(parser, x, minval=0.0),
        help=f"Balance factor to encourage similar volumes (default is {DEFAULT_ALPHA}). ",
        default=DEFAULT_ALPHA,
    )
    parser.add_argument(
        "--anisotropyfile",
        dest="anisotropyfile",
        metavar="ANISOFILE",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "Optional NIFTI tensor field used to bias growth along preferred directions. "
            "Accepted per-voxel layouts are 6 components [xx, yy, zz, xy, xz, yz], "
            "9 flattened matrix elements, or an explicit 3x3 tensor field."
        ),
        default=None,
    )
    parser.add_argument(
        "--anisotropystrength",
        dest="anisotropystrength",
        metavar="STRENGTH",
        action="store",
        type=lambda x: pf.is_float(parser, x, minval=0.0),
        help=(
            "Strength of anisotropic growth bias (default is "
            f"{DEFAULT_ANISOTROPY_STRENGTH})."
        ),
        default=DEFAULT_ANISOTROPY_STRENGTH,
    )
    return parser


def randomatlas(args: Any) -> None:
    infile, infile_data, infile_hdr, infiledims, infilesizes = tide_io.readfromnifti(
        args.inputfilename
    )
    anisotropy_data = None
    if args.anisotropyfile is not None:
        (
            _anisotropy_img,
            anisotropy_data,
            _anisotropy_hdr,
            _anisotropy_dims,
            _anisotropy_sizes,
        ) = tide_io.readfromnifti(args.anisotropyfile)
        if anisotropy_data.shape[:3] != infile_data.shape[:3]:
            raise ValueError(
                "anisotropy tensor field spatial dimensions must match the input mask "
                f"({anisotropy_data.shape[:3]} != {infile_data.shape[:3]})"
            )

    labels = tide_regionops.partition_3d(
        infile_data.astype(np.uint16),
        n_regions=args.numregions,
        connectivity=6,
        seed=args.seed,
        balance_alpha=args.alpha,  # encourages similar volumes
        jitter=0.1,  # reduces grid artifacts
        anisotropy_field=anisotropy_data,
        anisotropy_strength=args.anisotropystrength,
    )

    tide_io.savetonifti(
        labels,
        infile_hdr,
        f"{args.outputfileroot}_r{str(args.numregions).zfill(3)}_s{str(args.seed).zfill(4)}",
    )
