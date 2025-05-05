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
import sys

import rapidtide.externaltools as tide_exttools
import rapidtide.io as tide_io


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="rapidtide2std",
        description="Register rapidtide output maps to standard space.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "inputfileroot",
        help="the root name of the input NIFTI files (up to but not including the underscore).",
    )
    parser.add_argument("outputdir", help="The location for the output files")
    parser.add_argument(
        "featdirectory",
        help=(
            "Either a feat directory (x.feat) or an fmriprep derivatives "
            "anat directory where the information needed for registration to standard space can be found"
        ),
    )
    parser.add_argument(
        "--corrout",
        dest="corrout",
        action="store_true",
        help=("Also transform the corrout file (warning - file may be huge)."),
        default=False,
    )
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        help=("Also transform the cleaned data file (warning - file may be huge)."),
        default=False,
    )
    parser.add_argument(
        "--confound",
        dest="confound",
        action="store_true",
        help=("Transform the counfound fit R2 map (if present)."),
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
        "--sequential",
        dest="sequential",
        action="store_true",
        help=("Execute commands sequentially - do not submit to the cluster."),
        default=False,
    )
    parser.add_argument(
        "--fake",
        dest="preponly",
        action="store_true",
        help=("Output, but do not execute, alignment commands."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Output additional debugging information."),
        default=False,
    )
    return parser


def rapidtide2std(args):
    if args.debug:
        print(args)

    fsldir = os.environ.get("FSLDIR")
    if fsldir is None:
        print("FSL directory not found - aborting")
        sys.exit()

    # make sure the appropriate transformation matrix and targets exist
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
            thecommand = tide_exttools.makeflirtcmd(
                inputname,
                reftarget,
                xformfuncmat,
                outputname,
                warpfile=warpfuncfile,
                cluster=(not args.sequential),
                debug=args.debug,
            )
            tide_exttools.runcmd(thecommand, fake=args.preponly)
        else:
            print("file", inputname, "does not exist - exiting")
        sys.exit(0)

    theoutputdir = os.path.join(os.path.abspath("."), args.outputdir)
    thefileroot = glob.glob(os.path.join(args.inputfileroot + "*desc-corrout_info.nii.gz"))[0]

    thetimecoursefiles = [
        "desc-initialmovingregressor_timeseries",
        "desc-oversampledmovingregressor_timeseries",
    ]

    thefmrimaps = [
        "desc-maxtime_map",
        "desc-maxtimerefined_map",
        "desc-timepercentile_map",
        "desc-maxcorr_map",
        "desc-maxcorrrefined_map",
        "desc-maxwidth_map",
        "desc-MTT_map",
        "desc-corrfit_mask",
        "desc-refine_mask",
        "desc-maxcorrsq_map",
        "desc-lfofilterNorm_map",
        "desc-lfofilterCoeff_map",
        "desc-lfofilterInbandVarianceBefore_map",
        "desc-lfofilterInbandVarianceAfter_map",
        "desc-lfofilterInbandVarianceChange_map",
        "desc-plt0p050_mask",
        "desc-plt0p010_mask",
        "desc-plt0p005_mask",
        "desc-plt0p001_mask",
    ]

    if args.corrout:
        thefmrimaps.append("desc-corrout_info")

    if args.clean:
        thefmrimaps.append("desc-lfofilterCleaned_bold")

    if args.confound:
        thefmrimaps.append("desc-confoundfilterR2_map")

    absname = os.path.abspath(thefileroot)
    thepath, thebase = os.path.split(absname)
    theprevpath, theprevbase = os.path.split(thepath)
    subjroot = thebase[:-25]
    print("SUBJROOT:", subjroot)

    # copy the options file
    inputname = os.path.abspath(os.path.join(thepath, subjroot + "_desc-runoptions_info"))
    theoptionsdict = tide_io.readoptionsfile(inputname)
    theoptionsdict["rapidtide2std_source"] = os.path.abspath(os.path.join(thepath, subjroot))
    outputname = os.path.abspath(
        os.path.join(theoutputdir, subjroot + outputtag + "desc-runoptions_info")
    )
    tide_io.writedicttojson(theoptionsdict, f"{outputname}.json")
    # thecommand = ["cp", f"{inputname}.json", f"{outputname}.json"]
    # tide_exttools.runcmd(thecommand, fake=args.preponly)

    # copy the timecourses
    for timecoursefile in thetimecoursefiles:
        for extension in [".tsv.gz", ".json"]:
            inputname = os.path.abspath(
                os.path.join(thepath, subjroot + "_" + timecoursefile + extension)
            )
            outputname = os.path.abspath(
                os.path.join(theoutputdir, subjroot + outputtag + timecoursefile + extension)
            )
            thecommand = ["cp", inputname, outputname]
            tide_exttools.runcmd(thecommand, fake=args.preponly)

    # transform the functional maps
    for themap in thefmrimaps:
        inputname = os.path.abspath(os.path.join(thepath, subjroot + "_" + themap + ".nii.gz"))
        if os.path.isfile(inputname):
            outputname = os.path.abspath(
                os.path.join(theoutputdir, subjroot + outputtag + themap + ".nii.gz")
            )
            if args.debug:
                print(f"Transforming {inputname}")
            thecommand = tide_exttools.makeflirtcmd(
                inputname,
                reftarget,
                xformfuncmat,
                outputname,
                warpfile=warpfuncfile,
                cluster=(not args.sequential),
                debug=args.debug,
            )
            tide_exttools.runcmd(thecommand, fake=args.preponly)

    # transform the anatomic maps
    for themap in theanatmaps:
        try:
            inputname = os.path.abspath(
                glob.glob(os.path.join(args.featdirectory, "reg", themap + ".nii.gz"))[0]
            )
            if os.path.isfile(inputname):
                outputname = os.path.abspath(
                    os.path.join(
                        theoutputdir,
                        subjroot + outputtag + themap.replace("standard", "anat") + ".nii.gz",
                    )
                )
                if args.aligntohires:
                    thecommand = ["cp", inputname, outputname]
                else:
                    xform = os.path.abspath(
                        glob.glob(os.path.join(args.featdirectory, "reg", "highres2standard.mat"))[
                            0
                        ]
                    )
                    thecommand = tide_exttools.makeflirtcmd(
                        inputname,
                        reftarget,
                        xform,
                        outputname,
                        cluster=(not args.sequential),
                        debug=args.debug,
                    )
                tide_exttools.runcmd(thecommand, fake=args.preponly)
        except:
            print("no hires anatomic found - skipping")
