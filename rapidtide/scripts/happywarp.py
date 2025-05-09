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
import copy
import glob
import os
from argparse import RawTextHelpFormatter

import rapidtide.externaltools as tide_exttools
import rapidtide.filter as tide_filt
import rapidtide.io as tide_io

""


def _get_parser():
    class FullPaths(argparse.Action):
        """Expand user- and relative-paths"""

        def __call__(self, parser, namespace, values, option_string=None):
            if values == "":
                setattr(namespace, self.dest, "__EMPTY__")
            else:
                setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    def is_file(filename):
        """Checks if a file exists"""
        if not os.path.exists(filename):
            msg = "{0} does not exist".format(filename)
            raise argparse.ArgumentTypeError(msg)
        else:
            return filename

    parser = argparse.ArgumentParser(
        description=(
            "Align an anatomic image with the preprocessed T1 in an fmriprep derivatives directory.\n"
            "fmriprep does not align other anatomic images to the T1 by default.  This fixes that.\n"
        ),
        formatter_class=RawTextHelpFormatter,
        allow_abbrev=False,
    )

    parser.add_argument(
        "sourceimage",
        help="cine output file from happy, in tide_io.nifti format",
        action=FullPaths,
        type=is_file,
    )

    parser.add_argument(
        "--scalefac",
        help="scale factor to exaggerate motion",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--threads",
        help="number of threads to use for registration",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--finaltarget",
        help="align all output images to this image in the final step",
        action=FullPaths,
        type=is_file,
        default=None,
    )

    parser.add_argument(
        "--splinesize",
        help="spline size in voxels",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--transformtype",
        help="Transform type.  Options are\n"
        "\tr: rigid\n"
        "\ta: rigid + affine\n"
        "\ts: rigid + affine + deformable syn\n"
        "\tb: rigid + affine + deformable b-spline syn\n",
        default="b",
    )

    parser.add_argument(
        "--rescaleonly",
        help="only perform the warp scaling (assumes a previous full run).",
        action="store_true",
    )

    parser.add_argument(
        "--redofinalalignment",
        help="recalculate the transformation between epi and anatomic",
        action="store_true",
    )

    parser.add_argument(
        "--fake",
        help="Print what operations will be performed, but don't do them.",
        action="store_true",
    )

    parser.add_argument(
        "--debug",
        help="output additional debugging information",
        action="store_true",
    )

    return parser


def filterintime(filelist, extractedfile, workdir, filtcycles, dumpinput=False):
    numpoints = len(filelist)
    filelist = filelist * filtcycles
    filterinput = os.path.join(workdir, "filterinput")
    filterdata, filter_hdr = tide_io.niftimerge(
        filelist, filterinput, returndata=True, writetodisk=dumpinput
    )
    outdata = filterdata * 0.0
    thedims = filterdata.shape
    print(thedims)
    numvoxels = thedims[0] * thedims[1] * thedims[2]
    thefilter = tide_filt.NoncausalFilter(filtertype="arb")
    hcutoff = 1.0 / 2.0
    lcutoff = 1.0 / 32.0
    thefilter.setfreqs(lcutoff, lcutoff, hcutoff, hcutoff)

    filterdata_byvoxel = filterdata.reshape((numvoxels, thedims[3], thedims[4]))
    outdata_byvoxel = outdata.reshape((numvoxels, thedims[3], thedims[4]))
    for direction in range(thedims[4]):
        print("filtering direction", direction)
        for voxel in range(numvoxels):
            filteredtc = thefilter.apply(1.0, filterdata_byvoxel[voxel, :, direction])
            outdata_byvoxel[voxel, :, direction] = filteredtc + 0.0

    tide_io.savetonifti(outdata[:, :, :, :, :], filter_hdr, "filteroutdata")

    theheader = copy.deepcopy(filter_hdr)
    theheader["dim"][4] = numpoints
    startpt = int((filtcycles // 2) * numpoints)
    endpt = startpt + numpoints
    tide_io.savetonifti(outdata[:, :, :, startpt:endpt, :], theheader, extractedfile)


def dohappywarp(
    sourceimage,
    transformtype="b",
    scalefac=20.0,
    threads=1,
    splinesize=10,
    rescaleonly=False,
    fake=False,
    finaltarget=None,
    redofinalalignment=False,
    debug=False,
):
    if debug:
        print("arguments to dohappywarp:")
        print("\tsourceimage:", sourceimage)
        print("\ttransformtype:", transformtype)
        print("\tscalefac:", scalefac)
        print("\tthreads:", threads)
        print("\tsplinesize:", splinesize)
        print("\trescaleonly:", rescaleonly)
        print("\tfinaltarget:", finaltarget)
        print("\tredofinalalignment:", redofinalalignment)
        print("\tfake:", fake)

    refimage = "cinemean.nii.gz"
    filtcycles = 5
    sourcedir, sourcefile = os.path.split(sourceimage)
    workdir = os.path.join(sourcedir, "warptemp")
    try:
        os.makedirs(workdir)
    except FileExistsError:
        pass
    reffullname = os.path.join(workdir, refimage)
    if debug:
        print("reffullname:", reffullname)

    # make a mean image
    fslmathscmd = ["fslmaths"]
    fslmathscmd += [sourceimage]
    fslmathscmd += ["-Tmean"]
    fslmathscmd += [reffullname]
    tide_exttools.runcmd(fslmathscmd, fake=fake)

    # intensity correct the mean image
    tide_exttools.n4correct(reffullname, workdir, fake=fake)
    correctedreffullname = reffullname.replace(".nii.gz", "_n4.nii.gz")

    # if there is a target image, align the mean image with it
    if finaltarget is not None and (not rescaleonly or redofinalalignment):
        tide_exttools.antsalign(
            finaltarget,
            correctedreffullname,
            os.path.join(workdir, "tofinal_"),
            transformtype="b",
            threads=threads,
            fake=fake,
        )

    if not rescaleonly:
        # split the cine file
        splitroot = os.path.join(workdir, "cineframe_")
        tide_io.niftisplit(sourceimage, splitroot)

        # intensity correct the split files
        filespec = splitroot + "[0123456789][0123456789][0123456789][0123456789].nii.gz"
        print(filespec)
        framefiles = glob.glob(filespec)
        print("framefiles:", framefiles)
        framefiles.sort()
        for framefile in framefiles:
            tide_exttools.n4correct(framefile, workdir, fake=fake)

        # make a sorted list of the intensity corrected cine frames
        correctedframefiles = glob.glob(splitroot + "*_n4.nii.gz")
        correctedframefiles.sort()
        numtimepoints = len(correctedframefiles)

        # find the alignment of every the mean image to every frame
        doinitalign = True
        alignroot = os.path.join(workdir, "alignedtomean_")
        if doinitalign:
            for correctedframefile in correctedframefiles:
                thenumber = correctedframefile[len(splitroot) : len(splitroot) + 4]
                theoutput = correctedframefile.replace(splitroot, alignroot).replace(
                    ".nii.gz", "_"
                )
                if debug:
                    print("thenumber:", thenumber)
                    print("initfile:", correctedframefile)
                    print("theoutput:", theoutput)
                tide_exttools.antsalign(
                    correctedframefile,
                    correctedreffullname,
                    theoutput,
                    transformtype=transformtype,
                    threads=threads,
                    splinesize=splinesize,
                    fake=fake,
                )

    # lowpass filter the warp data in time
    alignroot = os.path.join(workdir, "alignedtomean_")
    extractedfile = os.path.join(workdir, "filteredWarp")
    filelist = glob.glob(alignroot + "[0123456789]*_1Warp.nii.gz")
    filelist.sort()
    filterintime(filelist, extractedfile, workdir, filtcycles, dumpinput=True)

    # scale the warp file
    extractedscaledfile = os.path.join(workdir, "filteredWarp_scaled.nii.gz")
    scalecmd = []
    scalecmd += ["fslmaths"]
    scalecmd += [extractedfile]
    scalecmd += ["-mul", str(scalefac)]
    scalecmd += [extractedscaledfile]
    tide_exttools.runcmd(scalecmd, fake=fake)

    # split the filtered warps into timepoints
    splitunscaledwarproot = os.path.join(workdir, "filtered_Warp_")
    splitscaledwarproot = os.path.join(workdir, "filtered_Warp_scaled_")
    tide_io.niftisplit(extractedfile, splitunscaledwarproot)
    tide_io.niftisplit(extractedscaledfile, splitscaledwarproot)

    # now apply the warp files to the mean
    splitroot = os.path.join(workdir, "filtered_Warp_")
    alignedroot = os.path.join(workdir, "filtered_Warped_")

    warpfiles = glob.glob(splitroot + "[0123456789]*.nii.gz")
    warpfiles.sort()
    targetname = correctedreffullname

    for warpfile in warpfiles:
        theoutput = warpfile.replace(splitroot, alignedroot)
        translist = [warpfile]
        if finaltarget is not None:
            translist = [
                os.path.join(workdir, "tofinal_1Warp.nii.gz"),
                os.path.join(workdir, "tofinal_0GenericAffine.mat"),
            ] + translist
        tide_exttools.antsapply(targetname, targetname, theoutput, translist, fake=fake)
        tide_exttools.antsapply(
            targetname,
            targetname,
            theoutput.replace("Warped_", "Warped_scaled_"),
            [warpfile.replace("Warp_", "Warp_scaled_")],
            fake=fake,
        )

    # merge the warped files
    print("out of transform loop")
    mergedoutput = os.path.join(sourcedir, "warpedcine")
    mergedoutputscaled = os.path.join(sourcedir, "warpedcinescaled")
    filespec = alignedroot + "[0123456789]*.nii.gz"
    print(filespec)
    filelist = glob.glob(filespec)
    filelist.sort()
    tide_io.niftimerge(filelist, mergedoutput)
    filespec = alignedroot + "scaled_*.nii.gz"
    print(filespec)
    filelist = glob.glob(filespec)
    filelist.sort()
    tide_io.niftimerge(filelist, mergedoutputscaled)


def entrypoint():
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    if args.debug:
        print(args)

    dohappywarp(
        args.sourceimage,
        transformtype=args.transformtype,
        splinesize=args.splinesize,
        scalefac=args.scalefac,
        threads=args.threads,
        rescaleonly=args.rescaleonly,
        fake=args.fake,
        finaltarget=args.finaltarget,
        redofinalalignment=args.redofinalalignment,
        debug=args.debug,
    )


if __name__ == "__main__":
    entrypoint()
