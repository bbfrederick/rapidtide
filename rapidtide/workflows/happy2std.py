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

import rapidtide.externaltools as tide_extern


def _get_parser():
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
    thepath,
    theoutputdir,
    subjroot,
    reftarget,
    xformfuncmat,
    warpfuncfile,
    thefmrimaps=None,
    theanatmaps=None,
    preponly=False,
):
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


def happy2std(args):
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
