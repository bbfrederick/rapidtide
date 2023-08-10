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
import copy
import sys

import numpy as np
import pyfftw
from sklearn.decomposition import PCA, FastICA

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()


import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_valid_file


def P2R(radii, angles):
    return radii * np.exp(1j * angles)


def R2P(x):
    return np.absolute(x), np.angle(x)


def _get_parser():
    """
    Argument parser for fdica
    """
    parser = argparse.ArgumentParser(
        prog="fdica",
        description="Fit a spatial template to a 3D or 4D NIFTI file.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit.",
    )
    parser.add_argument(
        "datamask",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 dimensional nifti file voxel mask (must match datafile).",
    )
    parser.add_argument("outputroot", type=str, help="The root name for all output files.")

    parser.add_argument(
        "--spatialfilt",
        dest="gausssigma",
        action="store",
        type=float,
        metavar="GAUSSSIGMA",
        help=(
            "Spatially filter fMRI data prior to analysis "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to have rapidtide set it to half the mean voxel "
            "dimension (a rule of thumb for a good value)."
        ),
        default=0.0,
    )
    parser.add_argument(
        "--pcacomponents",
        metavar="NCOMP",
        dest="pcacomponents",
        type=float,
        help="Use NCOMP components for PCA fit of phase",
        default=0.9,
    )
    parser.add_argument(
        "--icacomponents",
        metavar="NCOMP",
        dest="icacomponents",
        type=int,
        help="Use NCOMP components for ICA decomposition",
        default=None,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Output additional debugging information.",
        default=False,
    )
    return parser


def fdica(
    datafile,
    datamask,
    outputroot,
    gausssigma=0.0,
    pcacomponents="mle",
    icacomponents=None,
    lowerfreq=0.009,
    upperfreq=0.15,
    debug=False,
):
    # read in data
    print("reading in data arrays")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = tide_io.readfromnifti(datafile)
    (
        datamask_img,
        datamask_data,
        datamask_hdr,
        datamaskdims,
        datamasksizes,
    ) = tide_io.readfromnifti(datamask)

    print(f"shape of datafile_data: {datafile_data.shape}")
    print(f"shape of datamask_data: {datamask_data.shape}")

    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(datafilesizes)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(datafiledims)

    if datafile_hdr.get_xyzt_units()[1] == "msec":
        fmritr = tr / 1000.0
    else:
        fmritr = tr
    nyquist = 0.5 / fmritr
    hzperpoint = nyquist / timepoints
    print(f"nyquist: {nyquist}Hz, hzperpoint: {hzperpoint}Hz")

    # figure out what bins we will retain
    if lowerfreq < 0.0:
        lowerbin = 0
    else:
        lowerbin = int(np.floor(lowerfreq / hzperpoint))
    if upperfreq < 0.0:
        upperbin = timepoints - 1
    else:
        upperbin = int(np.ceil(upperfreq / hzperpoint))
    trimmedsize = upperbin - lowerbin + 1
    print(f"will retain points {lowerbin} to {upperbin}")

    # check dimensions
    print("checking dimensions")
    if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
        print("input mask spatial dimensions do not match image")
        exit()
    if datamaskdims[4] != 1:
        print("specify a 3d data mask")
        sys.exit()

    # do spatial filtering if requested
    if gausssigma < 0.0:
        # set gausssigma automatically
        gausssigma = np.mean([xdim, ydim, slicethickness]) / 2.0
    if gausssigma > 0.0:
        print(f"applying gaussian spatial filter with sigma={gausssigma}")
        for i in range(timepoints):
            datafile_data[:, :, :, i] = tide_filt.ssmooth(
                xdim,
                ydim,
                slicethickness,
                gausssigma,
                datafile_data[:, :, :, i],
            )
        print("spatial filtering complete")

    # create arrays
    print("allocating arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    print(f"numspatiallocs: {numspatiallocs}")
    rs_datafile = datafile_data.reshape((numspatiallocs, timepoints))
    rs_datamask = datamask_data.reshape(numspatiallocs)
    rs_datamask_bin = np.where(rs_datamask > 0.9, 1.0, 0.0)
    savearray = np.zeros((xsize, ysize, numslices, trimmedsize), dtype="float")
    rs_savearray = savearray.reshape(numspatiallocs, trimmedsize)
    saveheader = copy.deepcopy(datafile_hdr)
    saveheader["dim"][4] = trimmedsize
    saveheader["pixdim"][4] = hzperpoint
    savesinglearray = np.zeros((xsize, ysize, numslices), dtype="float")
    rs_savesinglearray = savesinglearray.reshape(numspatiallocs)
    savesingleheader = copy.deepcopy(datafile_hdr)
    savesingleheader["dim"][4] = 1
    savesingleheader["pixdim"][4] = 1.0

    # select the voxels to process
    voxelstofit = np.where(rs_datamask_bin > 0.5)[0]
    numfitvoxels = len(voxelstofit)
    procvoxels = rs_datafile[voxelstofit, :]
    print(f"shape of procvoxels: {procvoxels.shape}")

    # calculating FFT
    print("calculating forward FFT")
    complexfftdata = fftpack.fft(procvoxels, axis=1)
    print(f"shape of complexfftdata: {complexfftdata.shape}")

    procvoxels2 = fftpack.ifft(complexfftdata, axis=1).real
    savefullarray = np.zeros((xsize, ysize, numslices, timepoints), dtype="float")
    rs_savefullarray = savefullarray.reshape(numspatiallocs, timepoints)
    rs_savefullarray[voxelstofit, :] = procvoxels2
    tide_io.savetonifti(
        savefullarray.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        outputroot + "_ifft",
    )

    # convert to polar
    fullmagdata, fullphasedata = R2P(complexfftdata)
    fullphasedata = np.unwrap(fullphasedata)

    # save the polar data
    rs_savefullarray[voxelstofit, :] = fullmagdata
    tide_io.savetonifti(
        savefullarray.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        outputroot + "_fullmagdata",
    )
    rs_savefullarray[voxelstofit, :] = fullphasedata
    tide_io.savetonifti(
        savefullarray.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        outputroot + "_fullphasedata",
    )

    # checking IFFT
    print("calculating forward FFT")
    procvoxels2 = fftpack.ifft(complexfftdata, axis=1).real

    # trim the data
    trimmeddata = complexfftdata[:, lowerbin : min(upperbin + 1, timepoints)]
    print(f"shape of trimmeddata: {trimmeddata.shape}")

    # convert to polar
    magdata = fullmagdata[:, lowerbin : min(upperbin + 1, timepoints)]
    phasedata = fullphasedata[:, lowerbin : min(upperbin + 1, timepoints)]
    print(f"shape of magdata: {magdata.shape}")
    print(f"shape of phasedata: {phasedata.shape}")

    # remove mean and linear component
    phasemeans = np.zeros((numfitvoxels), dtype="float")
    phaseslopes = np.zeros((numfitvoxels), dtype="float")
    detrendedphasedata = phasedata * 0.0
    print(f"shape of detrendedphasedata: {detrendedphasedata.shape}")
    X = np.linspace(lowerbin * hzperpoint, upperbin * hzperpoint, trimmedsize)
    for i in range(numfitvoxels):
        thecoffs = np.polyfit(X, phasedata[i, :], 1)
        phaseslopes[i], phasemeans[i] = thecoffs
        # detrendedphasedata[i, :] = phasedata[i, :] - tide_fit.trendgen(X, thecoffs, True)
        detrendedphasedata[i, :] = phasedata[i, :] - np.mean(phasedata[i, :])
    rs_savearray[:, :] = 0.0
    rs_savearray[voxelstofit, :] = detrendedphasedata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_detrendedphase",
    )
    rs_savesinglearray[:] = 0.0
    rs_savesinglearray[voxelstofit] = phasemeans
    tide_io.savetonifti(
        rs_savesinglearray.reshape((xsize, ysize, numslices)),
        savesingleheader,
        outputroot + "_phasemeans",
    )
    rs_savesinglearray[:] = 0.0
    rs_savesinglearray[voxelstofit] = phaseslopes
    tide_io.savetonifti(
        rs_savesinglearray.reshape((xsize, ysize, numslices)),
        savesingleheader,
        outputroot + "_phaseslopes",
    )

    # do PCA data reduction on phase data
    thepca = PCA(n_components=pcacomponents)
    thepcaphasefit = thepca.fit(detrendedphasedata)
    thepcaphasetransform = thepca.transform(phasedata)
    print(f"shape of thepcaphasetransform: {thepcaphasetransform.shape}")
    thepcaphaseinvtrans = thepca.inverse_transform(thepcaphasetransform)
    print(f"shape of thepcaphaseinvtrans: {thepcaphaseinvtrans.shape}")
    if pcacomponents < 1.0:
        thepcacomponents = thepcaphasefit.components_[:]
        print("returning", thepcacomponents.shape[1], "components")
    else:
        thepcacomponents = thepcaphasefit.components_[0 : int(pcacomponents)]
    print(f"shape of thepcacomponents: {thepcacomponents.shape}")
    print("writing pca component timecourses")
    tide_io.writenpvecs(thepcacomponents, outputroot + "_pcacomponents.txt")

    # save the eigenvalues
    print("variance explained by component:", 100.0 * thepcaphasefit.explained_variance_ratio_)
    tide_io.writenpvecs(
        100.0 * thepcaphasefit.explained_variance_ratio_,
        outputroot + "_explained_variance_pct.txt",
    )

    print("writing pca component timecourses")
    tide_io.writenpvecs(thepcacomponents, outputroot + "_pcacomponents.txt")

    # save the coefficients
    """print("writing out the coefficients")
    coefficients = thepcaphasetransform
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
        theinvtrans[i, :] = thevar[i] * theinvtrans[i, :] + themean[i]"""

    # save magnitude and phase data
    rs_savearray[:, :] = 0.0
    rs_savearray[voxelstofit, :] = magdata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_mag",
    )
    rs_savearray[:, :] = 0.0
    rs_savearray[voxelstofit, :] = phasedata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_phase",
    )
    rs_savearray[:, :] = 0.0
    rs_savearray[voxelstofit, :] = thepcaphaseinvtrans
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_pcaphase",
    )

    # run the ICA
    print("running ICA decomposition")
    temporalstack = True
    if temporalstack:
        icainput = np.hstack((magdata, phasedata))
    else:
        icainput = np.vstack((magdata, phasedata))

    print(f"shape of icainput: {icainput.shape}")
    theica = FastICA(n_components=icacomponents)
    theicafit = theica.fit_transform(icainput)
    theicaproj = theica.inverse_transform((theicafit))
    print(f"shape of theicafit: {theicafit.shape}")
    theicacomponents = theica.components_
    print(f"shape of theicacomponents: {theicacomponents.shape}")
    print("writing ica component timecourses")
    tide_io.writenpvecs(theicacomponents, outputroot + "_icacomponents.txt")
    tdicacomp = np.zeros((theicacomponents.shape[0], timepoints), dtype="float")

    # save magnitude and phase data
    if temporalstack:
        reconmagdata = theicaproj[:, :trimmedsize]
        reconphasedata = theicaproj[:, trimmedsize:]
    else:
        reconmagdata = theicaproj[:numfitvoxels, :]
        reconphasedata = theicaproj[numfitvoxels:, :]
    rs_savearray[:, :] = 0.0
    rs_savearray[voxelstofit, :] = reconmagdata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_reconmag",
    )
    rs_savearray[:, :] = 0.0
    rs_savearray[voxelstofit, :] = reconphasedata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_reconphase",
    )

    # put the data back into rectangular components and go back to time domain
    complexfftdata[:, :] = 0.0 + 0.0j
    complexfftdata[:, lowerbin : min(upperbin + 1, timepoints)] = P2R(reconmagdata, reconphasedata)
    procvoxels = fftpack.ifft(complexfftdata, axis=1).real
    print(f"shape of procvoxels: {procvoxels.shape}")

    savefullarray[:, :, :, :] = 0.0
    rs_savefullarray = savefullarray.reshape(numspatiallocs, timepoints)
    rs_savefullarray[voxelstofit, :] = procvoxels
    tide_io.savetonifti(
        savefullarray.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        outputroot + "_movingsignal",
    )


def main():
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    print(args)

    fdica(
        args.datafile,
        args.datamask,
        args.outputroot,
        gausssigma=args.gausssigma,
        icacomponents=args.icacomponents,
        pcacomponents=args.pcacomponents,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
