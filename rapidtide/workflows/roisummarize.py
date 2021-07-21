#!/usr/bin/env python
#
#   Copyright 2021 Blaise Frederick
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

import argparse
import sys

import numpy as np

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for roisummarize
    """
    parser = argparse.ArgumentParser(
        prog="filttc",
        description=("Extract summary timecourses from the regions in an atlas"),
        usage="%(prog)s inputfilename templatefile outputfile [options]",
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "inputfilename")
    pf.addreqinputtextfile(parser, "templatefile")
    pf.addreqoutputtextfile(parser, "outputfile")

    # add optional arguments
    freq_group = parser.add_mutually_exclusive_group()
    freq_group.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="FREQ",
        help=(
            "Timecourses in file have sample "
            "frequency FREQ (default is 1.0Hz) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )
    freq_group.add_argument(
        "--sampletstep",
        dest="samplerate",
        action="store",
        type=lambda x: pf.invert_float(parser, x),
        metavar="TSTEP",
        help=(
            "Timecourses in file have sample "
            "timestep TSTEP (default is 1.0s) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )

    parser.add_argument(
        "--numskip",
        dest="numskip",
        action="store",
        type=int,
        metavar="NPTS",
        help=("Skip NPTS initial points to get past T1 relaxation. "),
        default=0,
    )

    # Filter arguments
    pf.addfilteropts(parser, defaultmethod="None", filtertarget="timecourses")

    # Normalization arguments
    pf.addnormalizationopts(parser, normtarget="timecourses", defaultmethod="None")

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    # Miscellaneous options

    return parser


def summarize4Dbylabel(inputvoxels, templatevoxels, normmethod="z", debug=False):
    numregions = np.max(templatevoxels)
    numtimepoints = inputvoxels.shape[1]
    timecourses = np.zeros((numregions, numtimepoints), dtype="float")
    for theregion in range(1, numregions + 1):
        thevoxels = inputvoxels[np.where(templatevoxels == theregion), :][0]
        print("extracting", thevoxels.shape[0], "voxels from region", theregion)
        if thevoxels.shape[1] > 0:
            regiontimecourse = np.nan_to_num(np.mean(thevoxels, axis=0))
        else:
            regiontimecourse = timecourses[0, :] * 0.0
        if debug:
            print("thevoxels, data shape are:", thevoxels.shape, regiontimecourse.shape)
        timecourses[theregion - 1, :] = tide_math.normalize(regiontimecourse, method=normmethod)
    return timecourses


def summarize3Dbylabel(inputvoxels, templatevoxels, debug=False):
    numregions = np.max(templatevoxels)
    outputvoxels = 0.0 * inputvoxels
    regionstats = []
    for theregion in range(1, numregions + 1):
        thevoxels = inputvoxels[np.where(templatevoxels == theregion)][0]
        regionmean = np.nan_to_num(np.mean(thevoxels))
        regionstd = np.nan_to_num(np.std(thevoxels))
        regionmedian = np.nan_to_num(np.median(thevoxels))
        regionstats.append([regionmean, regionstd, regionmedian])
        outputvoxels[np.where(templatevoxels == theregion)] = regionmean
    return outputvoxels, regionstats


def roisummarize(args):
    # grab the command line arguments then pass them off.
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    # set the sample rate
    if args.samplerate == "auto":
        args.samplerate = 1.0
    else:
        samplerate = args.samplerate

    args, thefilter = pf.postprocessfilteropts(args, debug=args.debug)

    print("loading fmri data")
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    print("loading template data")
    template_img, template_data, template_hdr, templatedims, templatesizes = tide_io.readfromnifti(
        args.templatefile
    )

    print("checking dimensions")
    if not tide_io.checkspacematch(input_hdr, template_hdr):
        print("template file does not match spatial coverage of input fmri file")
        sys.exit()

    print("reshaping")
    xsize = thedims[1]
    ysize = thedims[2]
    numslices = thedims[3]
    numtimepoints = thedims[4]
    numvoxels = int(xsize) * int(ysize) * int(numslices)
    templatevoxels = np.reshape(template_data, numvoxels).astype(int)

    if numtimepoints > 1:
        inputvoxels = np.reshape(input_data, (numvoxels, numtimepoints))[:, args.numskip :]
        print("filtering")
        for thevoxel in range(numvoxels):
            if templatevoxels[thevoxel] > 0:
                inputvoxels[thevoxel, :] = thefilter.apply(
                    args.samplerate, inputvoxels[thevoxel, :]
                )

        print("summarizing")
        timecourses = summarize4Dbylabel(
            inputvoxels, templatevoxels, normmethod=args.normmethod, debug=args.debug
        )

        print("writing data")
        tide_io.writenpvecs(timecourses, args.outputfile + "_timecourses")
    else:
        inputvoxels = np.reshape(input_data, (numvoxels))
        numregions = np.max(templatevoxels)
        template_hdr["dim"][4] = numregions
        outputvoxels, regionstats = summarize3Dbylabel(
            inputvoxels, templatevoxels, debug=args.debug
        )
        tide_io.savetonifti(
            outputvoxels.reshape((xsize, ysize, numslices)),
            template_hdr,
            args.outputfile + "_meanvals",
        )
        tide_io.writenpvecs(np.array(regionstats), args.outputfile + "_regionstats.txt")
