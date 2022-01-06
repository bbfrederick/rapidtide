#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import sys

import numpy as np
from sklearn.decomposition import PCA, FastICA, SparsePCA

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_float, is_valid_file


def _get_parser(decompaxis):
    """
    Argument parser for spatialdecomp and temporaldecomp
    """
    if decompaxis == "temporal":
        parser = argparse.ArgumentParser(
            prog="temporaldecomp",
            description="Perform PCA or ICA decomposition on a data file in the time dimension.",
            usage="%(prog)s datafile outputroot [options]",
        )
    elif decompaxis == "spatial":
        parser = argparse.ArgumentParser(
            prog="spatialdecomp",
            description="Perform PCA or ICA decomposition on a data file in the spatial dimension.",
            usage="%(prog)s datafile outputroot [options]",
        )
    else:
        raise ValueError(f"Illegal decomposition type: {type}")

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit",
    )
    parser.add_argument("outputroot", help="The root name for the output nifti files")

    # Optional arguments
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    parser.add_argument(
        "--ncomp",
        dest="ncomp",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="NCOMPS",
        help=("The number of PCA/ICA components to return (default is to estimate the number)."),
        default=-1.0,
    )
    parser.add_argument(
        "--smooth",
        dest="sigma",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="SIGMA",
        help=("Spatially smooth the input data with a SIGMA mm kernel."),
        default=0.0,
    )
    parser.add_argument(
        "--type",
        dest="decomptype",
        action="store",
        type=str,
        choices=["pca", "sparse", "ica"],
        help=("Type of decomposition to perform. Default is pca."),
        default="pca",
    )
    parser.add_argument(
        "--nodemean",
        dest="demean",
        action="store_false",
        help=("Do not demean data prior to decomposition."),
        default=True,
    )
    parser.add_argument(
        "--novarnorm",
        dest="varnorm",
        action="store_false",
        help=("Do not variance normalize data prior to decomposition."),
        default=True,
    )

    return parser


def niftidecomp_workflow(
    decompaxis,
    datafile,
    outputroot,
    datamaskname=None,
    decomptype="pca",
    pcacomponents=0.5,
    icacomponents=None,
    varnorm=True,
    demean=True,
    sigma=0.0,
):

    print(f"Will perform {decomptype} analysis along the {decompaxis} axis")

    if decompaxis == "temporal":
        decompaxisnum = 1
        transposeifspatial = lambda *a, **k: None
    else:
        decompaxisnum = 0
        transposeifspatial = np.transpose

    # save the command line
    tide_io.writevec([" ".join(sys.argv)], outputroot + "_commandline.txt")

    # read in data
    print("reading in data arrays")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = tide_io.readfromnifti(datafile)

    if datamaskname is not None:
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(datamaskname)

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(datafiledims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(datafilesizes)

    # check dimensions
    if datamaskname is not None:
        print("checking mask dimensions")
        if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
            print("input mask spatial dimensions do not match image")
            exit()
        if not (tide_io.checktimematch(datafiledims, datamaskdims) or datamaskdims[4] == 1):
            print("input mask time dimension does not match image")
            exit()

    # save the command line
    tide_io.writevec([" ".join(sys.argv)], outputroot + "_commandline.txt")

    # smooth the data
    if sigma > 0.0:
        print("smoothing data")
        for i in range(timepoints):
            datafile_data[:, :, :, i] = tide_filt.ssmooth(
                xdim, ydim, slicethickness, sigma, datafile_data[:, :, :, i]
            )

    # allocating arrays
    print("reshaping arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = datafile_data.reshape((numspatiallocs, timepoints))

    print("masking arrays")
    maskthresh = 0.25
    if datamaskname is not None:
        if datamaskdims[4] == 1:
            proclocs = np.where(datamask_data.reshape(numspatiallocs) > maskthresh)
        else:
            proclocs = np.where(
                np.mean(datamask_data.reshape((numspatiallocs, timepoints)), axis=1) > maskthresh
            )
            rs_mask = datamask_data.reshape((numspatiallocs, timepoints))[proclocs, :]
            rs_mask = np.where(rs_mask > maskthresh, 1.0, 0.0)[0]
    else:
        datamaskdims = [1, xsize, ysize, numslices, 1]
        themaxes = np.max(rs_datafile, axis=1)
        themins = np.min(rs_datafile, axis=1)
        thediffs = (themaxes - themins).reshape(numspatiallocs)
        proclocs = np.where(thediffs > 0.0)
    procdata = rs_datafile[proclocs, :][0]
    print(rs_datafile.shape, procdata.shape)

    # normalize the individual images
    if demean:
        print("demeaning array")
        themean = np.mean(procdata, axis=decompaxisnum)
        print("shape of mean", themean.shape)
        for i in range(procdata.shape[1 - decompaxisnum]):
            if decompaxisnum == 1:
                procdata[i, :] -= themean[i]
            else:
                procdata[:, i] -= themean[i]
    else:
        themean = np.ones(procdata.shape[1 - decompaxisnum])

    if varnorm:
        print("variance normalizing array")
        thevar = np.var(procdata, axis=decompaxisnum)
        print("shape of var", thevar.shape)
        for i in range(procdata.shape[1 - decompaxisnum]):
            if decompaxisnum == 1:
                procdata[i, :] /= thevar[i]
            else:
                procdata[:, i] /= thevar[i]
        procdata = np.nan_to_num(procdata)
    else:
        thevar = np.ones(procdata.shape[1 - decompaxisnum])

    # applying mask
    if datamaskdims[4] > 1:
        procdata *= rs_mask

    # now perform the decomposition
    if decomptype == "ica":
        print("performing ica decomposition")
        if icacomponents is None:
            print("will return all significant components")
        else:
            print("will return", icacomponents, "components")
        thefit = FastICA(n_components=icacomponents).fit(
            transposeifspatial(procdata)
        )  # Reconstruct signals
        if icacomponents is None:
            thecomponents = transposeifspatial(thefit.components_[:])
            print(thecomponents.shape[1], "components found")
        else:
            thecomponents = transposeifspatial(thefit.components_[0:icacomponents])
            print("returning first", thecomponents.shape[1], "components found")
    else:
        print("performing pca decomposition")
        if pcacomponents < 1.0:
            print(
                "will return the components accounting for",
                pcacomponents * 100.0,
                "% of the variance",
            )
        else:
            print("will return", pcacomponents, "components")
        if decomptype == "pca":
            thepca = PCA(n_components=pcacomponents)
        else:
            thepca = SparsePCA(n_components=pcacomponents)
        thefit = thepca.fit(transposeifspatial(procdata))
        thetransform = thepca.transform(transposeifspatial(procdata))
        theinvtrans = transposeifspatial(thepca.inverse_transform(thetransform))
        if pcacomponents < 1.0:
            thecomponents = transposeifspatial(thefit.components_[:])
            print("returning", thecomponents.shape[1], "components")
        else:
            thecomponents = transposeifspatial(thefit.components_[0:pcacomponents])

        # save the eigenvalues
        print("variance explained by component:", 100.0 * thefit.explained_variance_ratio_)
        tide_io.writenpvecs(
            100.0 * thefit.explained_variance_ratio_,
            outputroot + "_explained_variance_pct.txt",
        )

        if decompaxis == "temporal":
            # save the components
            print("writing component timecourses")
            tide_io.writenpvecs(thecomponents, outputroot + "_components.txt")

            # save the singular values
            print("writing singular values")
            tide_io.writenpvecs(np.transpose(thesingvals), outputroot + "_singvals.txt")

            # save the coefficients
            print("writing out the coefficients")
            coefficients = thetransform
            print("coefficients shape:", coefficients.shape)
            theheader = datafile_hdr
            theheader["dim"][4] = coefficients.shape[1]
            tempout = np.zeros((numspatiallocs, coefficients.shape[1]), dtype="float")
            tempout[proclocs, :] = coefficients[:, :]
            tide_io.savetonifti(
                tempout.reshape((xsize, ysize, numslices, coefficients.shape[1])),
                datafile_hdr,
                outputroot + "_coefficients",
            )
            # unnormalize the dimensionality reduced data
            for i in range(numspatiallocs):
                theinvtrans[i, :] = thevar[i] * theinvtrans[i, :] + themean[i]

        else:
            # save the component images
            print("writing component images")
            theheader = datafile_hdr
            theheader["dim"][4] = thecomponents.shape[1]
            tempout = np.zeros((numspatiallocs, thecomponents.shape[1]), dtype="float")
            tempout[proclocs, :] = thecomponents[:, :]
            tide_io.savetonifti(
                tempout.reshape((xsize, ysize, numslices, thecomponents.shape[1])),
                datafile_hdr,
                outputroot + "_components",
            )

            # save the coefficients
            print("writing out the coefficients")
            coefficients = np.transpose(thetransform)
            tide_io.writenpvecs(coefficients, outputroot + "_coefficients.txt")
            tide_io.writenpvecs(coefficients * thevar[i], outputroot + "_denormcoefficients.txt")

            # unnormalize the dimensionality reduced data
            for i in range(timepoints):
                theinvtrans[:, i] = thevar[i] * theinvtrans[:, i] + themean[i]

        print("writing fit data")
        theheader = datafile_hdr
        theheader["dim"][4] = theinvtrans.shape[1]
        tempout = np.zeros((numspatiallocs, theinvtrans.shape[1]), dtype="float")
        tempout[proclocs, :] = theinvtrans[:, :]
        tide_io.savetonifti(
            tempout.reshape((xsize, ysize, numslices, theinvtrans.shape[1])),
            datafile_hdr,
            outputroot + "_fit",
        )


def getparameters(decompaxis):
    try:
        args = vars(_get_parser(decompaxis).parse_args())
    except SystemExit:
        _get_parser(decompaxis).print_help()
        raise

    if args["ncomp"] < 0.0:
        args["pcacomponents"] = 0.5
        args["icacomponents"] = None
    elif args["ncomp"] < 1.0:
        args["pcacomponents"] = args["ncomp"]
        args["icacomponents"] = None
    else:
        args["pcacomponents"] = int(args["ncomp"])
        args["icacomponents"] = int(args["ncomp"])

    return args


def main(decompaxis):
    args = getparameters(decompaxis)

    niftidecomp_workflow(
        decompaxis,
        args["datafile"],
        args["outputroot"],
        datamaskname=args["datamaskname"],
        decomptype=args["decomptype"],
        pcacomponents=args["pcacomponents"],
        icacomponents=args["icacomponents"],
        varnorm=args["varnorm"],
        demean=args["demean"],
        sigma=args["sigma"],
    )


if __name__ == "__main__":
    main("spatial")
