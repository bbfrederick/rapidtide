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
import glob
import os
import subprocess
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rapidtide.externaltools as tide_extern


def _get_parser() -> Any:
    """
    Create and configure an argument parser for the rapidtide2std command-line tool.

    This function sets up an `argparse.ArgumentParser` with specific arguments
    required for registering happy output maps to standard space. It defines
    positional and optional arguments for input file roots, output directories,
    and various transformation options.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with defined arguments for
        the rapidtide2std tool.

    Notes
    -----
    The parser is configured with a program name, description, and usage string
    specific to the rapidtide2std tool. It expects three positional arguments:
    inputfileroot, outputdir, and featdirectory, followed by several optional
    flags that control the registration process.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="rapidtide2std",
        description="Register happy output maps to standard space.",
        usage="%(prog)s inputfileroot outputdir featdirectory",
    )
    parser.add_argument(
        "inputfileroot",
        help="the root name of the input NIFTI files (up and including the 'desc' but not the underscore).",
    )
    parser.add_argument("outputdir", help="The location for the output files")
    parser.add_argument(
        "featdirectory",
        help=(
            "Either a feat-like directory (x.feat or x.ica) or an fmriprep derivatives"
            "anat directory where the information needed for registration to standard space can be found"
        ),
    )
    parser.add_argument(
        "--all",
        dest="all",
        action="store_true",
        help=("Also transform the corrout file (warning - file may be huge)."),
        default=False,
    )
    parser.add_argument(
        "--hires",
        dest="aligntohires",
        action="store_true",
        help=("Transform to match the high resolution anatomic image rather than the standard."),
        default=False,
    )
    parser.add_argument(
        "--linear",
        dest="forcelinear",
        action="store_true",
        help=("Only do linear transformation, even if warpfile exists."),
        default=False,
    )
    parser.add_argument(
        "--onefile",
        dest="onefilename",
        type=str,
        metavar="FILE",
        help="Align a single file, specified by name without extension (ignore INPUTFILEROOT).",
        default=None,
    )
    parser.add_argument(
        "--fake",
        dest="preponly",
        action="store_true",
        help=("Output, but do not execute, alignment commands."),
        default=False,
    )
    return parser


def transformmaps(
    thepath: Any,
    theoutputdir: Any,
    subjroot: Any,
    reftarget: Any,
    xformfuncmat: Any,
    warpfuncfile: Any,
    thefmrimaps: Optional[Any] = None,
    theanatmaps: Optional[Any] = None,
    preponly: bool = False,
) -> None:
    """
    Apply spatial transformations to fMRI and anatomical maps from a happy dataset using FLIRT and/or
    copy commands.

    This function applies rigid-body and non-linear transformations to a set of fMRI and/or
    anatomical maps, based on provided transformation matrices and warp fields. It supports
    both functional and anatomical image processing pipelines.

    Parameters
    ----------
    thepath : Any
        Path to the directory containing input fMRI maps.
    theoutputdir : Any
        Directory where transformed maps will be saved.
    subjroot : Any
        Subject identifier used to construct input and output file names.
    reftarget : Any
        Reference image to which maps are transformed.
    xformfuncmat : Any
        Transformation matrix for functional-to-standard space alignment.
    warpfuncfile : Any
        Warp field file for non-linear functional-to-standard space transformation.
    thefmrimaps : Optional[Any], default=None
        List of fMRI map names to transform. If None, no fMRI maps are processed.
    theanatmaps : Optional[Any], default=None
        List of anatomical map names to transform. If None, no anatomical maps are processed.
    preponly : bool, default=False
        If True, only print the commands without executing them.

    Returns
    -------
    None
        This function does not return any value; it performs file I/O operations.

    Notes
    -----
    - For fMRI maps, the function uses `tide_extern.makeflirtcmd` to generate and execute
      FLIRT commands for transformation.
    - For anatomical maps, it either copies the file or applies a transformation using
      `tide_extern.makeflirtcmd`, depending on the `aligntohires` flag.
    - If `preponly` is True, the commands are printed but not executed.
    - The function assumes that input files exist and are named according to the
      pattern: ``{subjroot}_{mapname}.nii.gz``.

    Examples
    --------
    >>> transformmaps(
    ...     thepath="/data/subj1",
    ...     theoutputdir="/data/subj1/transformed",
    ...     subjroot="subj1",
    ...     reftarget="/data/templates/MNI152_T1_2mm.nii.gz",
    ...     xformfuncmat="/data/subj1/reg/func2standard.mat",
    ...     warpfuncfile="/data/subj1/reg/warpfield.nii.gz",
    ...     thefmrimaps=["bold", "mask"],
    ...     theanatmaps=["brain"],
    ...     preponly=False
    ... )
    """
    print("entering transformmaps with:")
    print(thepath)
    print(subjroot)
    print(reftarget)
    print(xformfuncmat)
    print(warpfuncfile)
    if thefmrimaps is not None:
        for themap in thefmrimaps:
            inputname = os.path.abspath(os.path.join(thepath, subjroot + "_" + themap + ".nii.gz"))
            if os.path.isfile(inputname):
                outputname = os.path.abspath(
                    os.path.join(theoutputdir, subjroot + outputtag + themap + ".nii.gz")
                )
                thecommand = tide_extern.makeflirtcmd(
                    inputname,
                    reftarget,
                    xformfuncmat,
                    outputname,
                    warpfile=warpfuncfile,
                )

                if preponly:
                    print(" ".join(thecommand))
                else:
                    subprocess.call(thecommand)

    if theanatmaps is not None:
        for themap in theanatmaps:
            try:
                inputname = os.path.abspath(
                    glob.glob(os.path.join(xformdir, "reg", themap + ".nii.gz"))[0]
                )
                if os.path.isfile(inputname):
                    outputname = os.path.abspath(
                        os.path.join(
                            theoutputdir,
                            subjroot + outputtag + themap.replace("standard", "anat") + ".nii.gz",
                        )
                    )
                    if aligntohires:
                        thecommand = ["cp", inputname, outputname]
                    else:
                        xform = os.path.abspath(
                            glob.glob(os.path.join(xformdir, "reg", "highres2standard.mat"))[0]
                        )
                        thecommand = tide_extern.makeflirtcmd(
                            inputname, reftarget, xform, outputname
                        )

                    if preponly:
                        print(" ".join(thecommand))
                    else:
                        subprocess.call(thecommand)
            except:
                print("no hires anatomic found - skipping")


def happy2std(args: Any) -> None:
    """
    Apply FSL-based spatial transformation to fMRI data and anatomical maps.

    This function performs spatial normalization of fMRI data to either high-resolution
    or standard MNI152 space, using FSL transformation matrices and warp fields. It supports
    both linear and nonlinear transformations, and can process either a single file or
    a set of files based on input arguments.

    Parameters
    ----------
    args : Any
        An object containing command-line arguments. Expected attributes include:
        - featdirectory : str
            Path to the FEAT directory containing registration files.
        - aligntohires : bool
            If True, align to high-resolution anatomical space; otherwise to standard MNI152.
        - forcelinear : bool
            If True, forces use of a linear transformation even if warp files are present.
        - onefilename : str or None
            If provided, process only this single file.
        - preponly : bool
            If True, print the FSL command without executing it.
        - inputfileroot : str
            Root path to input files for batch processing.
        - outputdir : str
            Output directory for transformed files.
        - all : bool
            If True, include additional fMRI maps in processing.

    Returns
    -------
    None
        This function does not return a value but may exit the program on error.

    Notes
    -----
    This function requires the FSL environment variable ``FSLDIR`` to be set.
    It relies on FSL tools like `applywarp` and `flirt` for transformations.
    The function supports processing of both single files and batches of files
    using glob patterns.

    Examples
    --------
    >>> args = parse_args()  # Assume args is parsed from command line
    >>> happy2std(args)
    """
    # make sure the appropriate transformation matrix and targets exist
    fsldir = os.environ.get("FSLDIR")
    if fsldir is None:
        raise RuntimeError("FSLDIR not set")

    if args.aligntohires:
        reftarget = os.path.abspath(os.path.join(args.featdirectory, "reg", "highres.nii.gz"))
        warpfuncfile = ""
        xformfuncmat = os.path.abspath(
            os.path.join(args.featdirectory, "reg", "example_func2highres.mat")
        )
        outputtag = "_hires_"
        theanatmaps = ["highres"]
    else:
        xformfuncmat = os.path.abspath(
            os.path.join(args.featdirectory, "reg", "example_func2standard.mat")
        )
        warpfuncfile = os.path.abspath(
            os.path.join(args.featdirectory, "reg", "example_func2standard_warp.nii.gz")
        )
        reftarget = os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm.nii.gz")
        outputtag = "_std_"
        theanatmaps = ["highres", "standard"]

    if args.forcelinear:
        warpfuncfile += "ridiculous_suffix"

    if os.path.isfile(xformfuncmat):
        if os.path.isfile(reftarget):
            print("found alignment files - proceeding")
        else:
            print("cannot find reference file", reftarget, " - exiting")
            sys.exit(1)
        if os.path.isfile(warpfuncfile):
            print("found warp file - will do nonlinear transformation")
        else:
            print("no warp file found - will do linear transformation")
            warpfuncfile = None
    else:
        print("cannot find transform matrix", xformfuncmat, " - exiting")
        sys.exit(1)

    if args.onefilename is not None:
        inputname = os.path.abspath(args.onefilename + ".nii.gz")
        thepath, thebase = os.path.split(inputname)
        if os.path.isfile(inputname):
            outputname = os.path.abspath(os.path.join(thepath, outputtag[1:] + thebase))
            thecommand = makefslcmd(
                inputname, reftarget, xformfuncmat, outputname, warpfile=warpfuncfile
            )

            if args.preponly:
                print(" ".join(thecommand))
            else:
                subprocess.call(thecommand)
        else:
            print("file", inputname, "does not exist - exiting")
        sys.exit(0)

    theoutputdir = os.path.join(os.path.abspath("."), args.outputdir)
    thefileroot = glob.glob(os.path.join(args.inputfileroot + "*app.nii.gz"))[0]

    thefmrimaps = [
        "normapp_info",
        "processvoxels_mask",
        "arteries_map",
        "veins_map",
        "vessels_map",
        "vessels_mask",
    ]

    if args.all:
        thefmrimaps += ["app_info", "cine_info", "maxphase_map", "minphase_map", "rawapp_info"]

    absname = os.path.abspath(thefileroot)
    thepath, thebase = os.path.split(absname)
    theprevpath, theprevbase = os.path.split(thepath)
    subjroot = thebase[:-11]

    transformmaps(
        thepath,
        args.outputdir,
        subjroot,
        reftarget,
        xformfuncmat,
        warpfuncfile,
        thefmrimaps=thefmrimaps,
        theanatmaps=theanatmaps,
        preponly=args.preponly,
    )
