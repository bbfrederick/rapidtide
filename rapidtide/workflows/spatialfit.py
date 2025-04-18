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

from matplotlib.pyplot import *
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="spatialfit",
        description="Fit a 3D or 4D NIFTI file to a spatial template.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "datafile",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the 3D or 4D input NIFTI file.",
    )
    parser.add_argument(
        "templatefile",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the 3D template NIFTI file.",
    )
    parser.add_argument("outputroot", help="The root of the output file names.")
    parser.add_argument(
        "--datamask",
        dest="dmask",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="DATAMASK",
        help="DATAMASK specifies which voxels in the data to use.",
        default=None,
    )
    parser.add_argument(
        "--templatemask",
        dest="tmask",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="TEMPLATEMASK",
        help="TEMPLATEMASK specifies which voxels in the template to use.",
        default=None,
    )
    parser.add_argument(
        "--order",
        dest="order",
        action="store",
        type=int,
        metavar="ORDER",
        help="The order of the fit to the template.",
        default=1,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    return parser


def spatialfit(args):
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    # read in data
    print("reading in data arrays")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = tide_io.readfromnifti(args.datafile)
    if args.dmask is not None:
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(args.dmask)
    (
        templatefile_img,
        templatefile_data,
        templatefile_hdr,
        templatefiledims,
        templatefilesizes,
    ) = tide_io.readfromnifti(args.templatefile)
    if args.tmask is not None:
        (
            templatemask_img,
            templatemask_data,
            templatemask_hdr,
            templatemaskdims,
            templatemasksizes,
        ) = tide_io.readfromnifti(args.tmask)

    xsize = datafiledims[1]
    ysize = datafiledims[2]
    numslices = datafiledims[3]
    timepoints = datafiledims[4]

    if args.debug:
        print(datafiledims)
        print(templatefiledims)

    # check dimensions
    print("checking dimensions")
    if not tide_io.checkspacedimmatch(datafiledims, templatefiledims):
        print(templatefiledims, "template file spatial dimensions do not match image")
        exit()
    if not templatefiledims[4] == 1:
        print("template file time dimension is not equal to 1")
        exit()

    if args.dmask is not None:
        if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
            print("input mask spatial dimensions do not match image")
            exit()
        if not tide_io.checktimematch(datafiledims, datamaskdims):
            print("input mask time dimension does not match image")
            exit()

    if args.tmask is not None:
        if not tide_io.checkspacedimmatch(datafiledims, templatemaskdims):
            print("template mask spatial dimensions do not match image")
            exit()
        if not templatemaskdims[4] == 1:
            print("template mask time dimension is not equal to 1")
            exit()

    # allocating arrays
    print("allocating arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = datafile_data.reshape((numspatiallocs, timepoints))
    if args.dmask is not None:
        rs_datamask = datamask_data.reshape((numspatiallocs, timepoints))
    else:
        rs_datamask = np.ones((numspatiallocs, timepoints), dtype="float")
    bin_datamask = np.where(rs_datamask > 0.9, 1.0, 0.0)

    rs_templatefile = templatefile_data.reshape((numspatiallocs, 1))
    if args.tmask is not None:
        rs_templatemask = templatemask_data.reshape((numspatiallocs, 1))
    else:
        rs_templatemask = np.ones((numspatiallocs, timepoints), dtype="float")

    bin_templatemask = np.where(rs_templatemask > 0.1, 1.0, 0.0)

    fitdata = np.zeros((numspatiallocs, timepoints), dtype="float")
    residuals = np.zeros((numspatiallocs, timepoints), dtype="float")
    normalized = np.zeros((numspatiallocs, timepoints), dtype="float")
    newtemplate = np.zeros((numspatiallocs), dtype="float")
    newmask = np.zeros((numspatiallocs), dtype="float")
    lincoffs = np.zeros((timepoints), dtype="float")
    offsets = np.zeros((timepoints), dtype="float")
    r2vals = np.zeros((timepoints), dtype="float")

    if args.debug:
        print(fitdata.shape)
        print(residuals.shape)
        print(normalized.shape)
        print(newtemplate.shape)
        print(newmask.shape)
        print(lincoffs.shape)
        print(offsets.shape)
        print(r2vals.shape)

    # mask everything
    print("masking data and template")
    if args.debug:
        print(rs_datafile.shape, np.count_nonzero(np.isnan(rs_datafile)))
        print(bin_datamask.shape, np.count_nonzero(np.isnan(bin_datamask)))
        print(rs_templatefile.shape, np.count_nonzero(np.isnan(rs_templatefile)))
        print(bin_templatemask.shape, np.count_nonzero(np.isnan(bin_templatemask)))
    maskeddata = rs_datafile * bin_datamask
    if args.debug:
        print(maskeddata.shape, np.count_nonzero(np.isnan(maskeddata)))
    maskedtemplate = rs_templatefile * bin_templatemask
    if args.debug:
        print(maskedtemplate.shape, np.count_nonzero(np.isnan(maskedtemplate)))

    # cycle over all images
    print("now cycling over all images")
    for thetime in tqdm(
        range(0, timepoints),
        desc="Timepoint",
        unit="timepoints",
    ):
        if args.debug:
            print("fitting")
        thefit, R2 = tide_fit.mlregress(maskedtemplate[:, 0], maskeddata[:, thetime])
        lincoffs[thetime] = thefit[0, 1]
        offsets[thetime] = thefit[0, 0]
        r2vals[thetime] = R2
        if args.debug:
            print("generating fit data")
            print(
                lincoffs[thetime].shape,
                bin_datamask[:, thetime].shape,
                maskedtemplate.flatten().shape,
            )
        fitdata[:, thetime] = (
            lincoffs[thetime] * bin_datamask[:, thetime] * maskedtemplate.flatten()
        )
        newtemplate += (
            np.nan_to_num(maskeddata[:, thetime] / lincoffs[thetime]) * rs_datamask[:, thetime]
        )
        newmask += rs_datamask[:, thetime].flatten() * bin_templatemask.flatten()
        normalized[:, thetime] = (rs_datafile[:, thetime] - offsets[thetime]) / lincoffs[thetime]
    print()
    residuals = rs_datafile - fitdata

    # write out the data files
    print("writing time series")
    tide_io.writenpvecs(lincoffs, args.outputroot + "_lincoffs.txt")
    tide_io.writenpvecs(offsets, args.outputroot + "_offsets.txt")
    tide_io.writenpvecs(r2vals, args.outputroot + "_r2vals.txt")
    print("slope mean, std:", np.mean(lincoffs), np.std(lincoffs))
    print("offset mean, std:", np.mean(offsets), np.std(offsets))

    print("writing nifti series")
    tide_io.savetonifti(
        fitdata.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        args.outputroot + "_fit",
    )
    tide_io.savetonifti(
        residuals.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        args.outputroot + "_residuals",
    )
    tide_io.savetonifti(
        normalized.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        args.outputroot + "_normalized",
    )
    newtemplate = np.where(newmask > 0, newtemplate / newmask, 0.0)
    tide_io.savetonifti(
        newtemplate.reshape((xsize, ysize, numslices)),
        templatefile_hdr,
        args.outputroot + "_newtemplate",
    )
