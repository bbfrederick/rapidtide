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
import copy
import json
import operator as op
import os
import platform
import sys

import nibabel as nib
import numpy as np
import pandas as pd

from rapidtide.tests.utils import mse


# ---------------------------------------- NIFTI file manipulation ---------------------------
def readfromnifti(inputfile, headeronly=False):
    r"""Open a nifti file and read in the various important parts

    Parameters
    ----------
    inputfile : str
        The name of the nifti file.

    Returns
    -------
    nim : nifti image structure
    nim_data : array-like
    nim_hdr : nifti header
    thedims : int array
    thesizes : float array

    """
    if os.path.isfile(inputfile):
        inputfilename = inputfile
    elif os.path.isfile(f"{inputfile}.nii.gz"):
        inputfilename = f"{inputfile}.nii.gz"
    elif os.path.isfile(f"{inputfile}.nii"):
        inputfilename = f"{inputfile}.nii"
    else:
        raise FileNotFoundError(f"nifti file {inputfile} does not exist")
    nim = nib.load(inputfilename)
    if headeronly:
        nim_data = None
    else:
        nim_data = nim.get_fdata()
    nim_hdr = nim.header.copy()
    thedims = nim_hdr["dim"].copy()
    thesizes = nim_hdr["pixdim"].copy()
    return nim, nim_data, nim_hdr, thedims, thesizes


def readfromcifti(inputfile, debug=False):
    r"""Open a cifti file and read in the various important parts

    Parameters
    ----------
    inputfile : str
        The name of the cifti file.

    Returns
    -------
    nim : nifti image structure
    nim_data : array-like
    nim_hdr : nifti header
    thedims : int array
    thesizes : float array

    """
    if os.path.isfile(inputfile):
        inputfilename = inputfile
    elif os.path.isfile(f"{inputfile}.nii"):
        inputfilename = f"{inputfile}.nii"
    else:
        raise FileNotFoundError(f"cifti file {inputfile} does not exist")

    cifti = nib.load(inputfilename)
    nifti_data = np.transpose(cifti.get_fdata(dtype=np.float32))
    cifti_hdr = cifti.header
    nifti_hdr = cifti.nifti_header

    if nifti_hdr["intent_code"] == 3002:
        timestep, starttime = getciftitr(cifti_hdr)
    else:
        timestep, starttime = None, None
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    if debug:
        for theaxis in axes:
            print(theaxis)

    thedims = nifti_hdr["dim"].copy()
    thesizes = nifti_hdr["pixdim"].copy()
    return cifti, cifti_hdr, nifti_data, nifti_hdr, thedims, thesizes, timestep


def getciftitr(cifti_hdr):
    seriesaxis = None
    for theaxis in cifti_hdr.matrix.mapped_indices:
        if isinstance(cifti_hdr.matrix.get_axis(theaxis), nib.cifti2.SeriesAxis):
            seriesaxis = theaxis
    if seriesaxis is not None:
        timepoint1 = cifti_hdr.matrix.get_axis(seriesaxis).get_element(1)
        starttime = cifti_hdr.matrix.get_axis(seriesaxis).get_element(0)
        timestep = timepoint1 - starttime
    else:
        print("No series axis found!  Exiting")
        sys.exit()
    return timestep, starttime


# dims are the array dimensions along each axis
def parseniftidims(thedims):
    r"""Split the dims array into individual elements

    Parameters
    ----------
    thedims : int array
        The nifti dims structure

    Returns
    -------
    nx, ny, nz, nt : int
        Number of points along each dimension
    """
    return int(thedims[1]), int(thedims[2]), int(thedims[3]), int(thedims[4])


# sizes are the mapping between voxels and physical coordinates
def parseniftisizes(thesizes):
    r"""Split the size array into individual elements

    Parameters
    ----------
    thesizes : float array
        The nifti voxel size structure

    Returns
    -------
    dimx, dimy, dimz, dimt : float
        Scaling from voxel number to physical coordinates
    """
    return thesizes[1], thesizes[2], thesizes[3], thesizes[4]


def dumparraytonifti(thearray, filename):
    outputaffine = np.zeros((4, 4), dtype=float)
    for i in range(4):
        outputaffine[i, i] = 1.0
    outputheader = nib.nifti1Header
    outputheader.set_affine(outputaffine)
    savetonifti(thearray, outputheader, filename)


def savetonifti(thearray, theheader, thename, debug=False):
    r"""Save a data array out to a nifti file

    Parameters
    ----------
    thearray : array-like
        The data array to save.
    theheader : nifti header
        A valid nifti header
    thename : str
        The name of the nifti file to save

    Returns
    -------

    """
    outputaffine = theheader.get_best_affine()
    qaffine, qcode = theheader.get_qform(coded=True)
    saffine, scode = theheader.get_sform(coded=True)
    thedtype = thearray.dtype
    if thedtype == np.uint8:
        thedatatypecode = 2
        thebitpix = 8
    elif thedtype == np.int16:
        thedatatypecode = 4
        thebitpix = 16
    elif thedtype == np.int32:
        thedatatypecode = 8
        thebitpix = 32
    elif thedtype == np.float32:
        thedatatypecode = 16
        thebitpix = 32
    elif thedtype == np.complex64:
        thedatatypecode = 32
        thebitpix = 64
    elif thedtype == np.float64:
        thedatatypecode = 64
        thebitpix = 64
    elif thedtype == np.int8:
        thedatatypecode = 256
        thebitpix = 8
    elif thedtype == np.uint16:
        thedatatypecode = 512
        thebitpix = 16
    elif thedtype == np.uint32:
        thedatatypecode = 768
        thebitpix = 32
    elif thedtype == np.int64:
        thedatatypecode = 1024
        thebitpix = 64
    elif thedtype == np.uint64:
        thedatatypecode = 1280
        thebitpix = 64
    # elif thedtype == np.float128:
    #        thedatatypecode = 1536
    #        thebitpix = 128
    elif thedtype == np.complex128:
        thedatatypecode = 1792
        thebitpix = 128
    # elif thedtype == np.complex256:
    #    thedatatypecode = 2048
    #    thebitpix = 256
    else:
        raise TypeError("type", thedtype, "is not legal")
    theheader["datatype"] = thedatatypecode
    theheader["bitpix"] = thebitpix
    if debug:
        print(f"savetonifti:")
        print(f"\tinput data array is type {thedtype}")
        targetdatatype = theheader["datatype"]
        print(f"\ttargetdatatype={targetdatatype}")

    if theheader["magic"] == "n+2":
        output_nifti = nib.Nifti2Image(thearray.astype(thedtype), outputaffine, header=theheader)
        suffix = ".nii"
    else:
        output_nifti = nib.Nifti1Image(thearray.astype(thedtype), outputaffine, header=theheader)
        suffix = ".nii.gz"
    output_nifti.set_qform(qaffine, code=int(qcode))
    output_nifti.set_sform(saffine, code=int(scode))

    output_nifti.to_filename(thename + suffix)
    output_nifti = None


def niftifromarray(data):
    return nib.Nifti1Image(data, affine=np.eye(4))


def niftihdrfromarray(data):
    return nib.Nifti1Image(data, affine=np.eye(4)).header.copy()


def makedestarray(
    destshape,
    filetype="nifti",
    rt_floattype="float64",
):
    if filetype == "text":
        try:
            internalspaceshape = destshape[0]
            timedim = destshape[1]
        except TypeError:
            internalspaceshape = destshape
            timedim = None
    elif filetype == "cifti":
        spaceindex = len(destshape) - 1
        timeindex = spaceindex - 1
        internalspaceshape = destshape[spaceindex]
        if destshape[timeindex] > 1:
            timedim = destshape[timeindex]
        else:
            timedim = None
    else:
        internalspaceshape = int(destshape[0]) * int(destshape[1]) * int(destshape[2])
        if len(destshape) == 3:
            timedim = None
        else:
            timedim = destshape[3]
    if timedim is None:
        outmaparray = np.zeros(internalspaceshape, dtype=rt_floattype)
    else:
        outmaparray = np.zeros((internalspaceshape, timedim), dtype=rt_floattype)
    return outmaparray, internalspaceshape


def populatemap(
    themap,
    internalspaceshape,
    validvoxels,
    outmaparray,
    debug=False,
):
    if len(outmaparray.shape) == 1:
        outmaparray[:] = 0.0
        if validvoxels is not None:
            outmaparray[validvoxels] = themap[:].reshape((np.shape(validvoxels)[0]))
        else:
            outmaparray = themap[:].reshape((internalspaceshape))
    else:
        outmaparray[:, :] = 0.0
        if validvoxels is not None:
            outmaparray[validvoxels, :] = themap[:, :].reshape(
                (np.shape(validvoxels)[0], outmaparray.shape[1])
            )
        else:
            outmaparray = themap[:, :].reshape((internalspaceshape, outmaparray.shape[1]))
    if debug:
        print(f"populatemap: output array shape is {outmaparray.shape}")
    return outmaparray


def savemaplist(
    outputname,
    maplist,
    validvoxels,
    destshape,
    theheader,
    bidsbasedict,
    filetype="nifti",
    rt_floattype="float64",
    cifti_hdr=None,
    savejson=True,
    debug=False,
):
    outmaparray, internalspaceshape = makedestarray(
        destshape,
        filetype=filetype,
        rt_floattype=rt_floattype,
    )
    for themap, mapsuffix, maptype, theunit, thedescription in maplist:
        # copy the data into the output array, remapping if warranted
        if debug:
            if validvoxels is None:
                print(f"savemaplist: saving {mapsuffix}  to {destshape}")
            else:
                print(
                    f"savemaplist: saving {mapsuffix}  to {destshape} from {np.shape(validvoxels)[0]} valid voxels"
                )
        outmaparray = populatemap(
            themap,
            internalspaceshape,
            validvoxels,
            outmaparray.astype(themap.dtype),
            debug=False,
        )

        # actually write out the data
        bidsdict = bidsbasedict.copy()
        if theunit is not None:
            bidsdict["Units"] = theunit
        if thedescription is not None:
            bidsdict["Description"] = thedescription
        if filetype == "text":
            writenpvecs(
                outmaparray.reshape(destshape),
                f"{outputname}_{mapsuffix}.txt",
            )
        else:
            savename = f"{outputname}_desc-{mapsuffix}_{maptype}"
            if savejson:
                writedicttojson(bidsdict, savename + ".json")
            if filetype == "nifti":
                savetonifti(outmaparray.reshape(destshape), theheader, savename)
            else:
                isseries = len(outmaparray.shape) != 1
                if isseries:
                    savetocifti(
                        outmaparray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=isseries,
                        names=[mapsuffix],
                    )
                else:
                    savetocifti(
                        outmaparray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=isseries,
                        start=theheader["toffset"],
                        step=theheader["pixdim"][4],
                    )


def savetocifti(
    thearray,
    theciftiheader,
    theniftiheader,
    thename,
    isseries=False,
    names=["placeholder"],
    start=0.0,
    step=1.0,
    debug=False,
):
    r"""Save a data array out to a cifti

    Parameters
    ----------
    thearray : array-like
        The data array to save.
    theciftiheader : cifti header
        A valid cifti header
    theniftiheader : nifti header
        A valid nifti header
    thename : str
        The name of the cifti file to save
    isseries: bool
        True if output is a dtseries, False if dtscalar
    start: float
        starttime in seconds
    step: float
        timestep in seconds
    debug: bool
        Print extended debugging information

    Returns
    -------

    """
    if debug:
        print("savetocifti:", thename)
    workingarray = np.transpose(thearray)
    if len(workingarray.shape) == 1:
        workingarray = workingarray.reshape((1, -1))

    # find the ModelAxis from the input file
    modelaxis = None
    for theaxis in theciftiheader.matrix.mapped_indices:
        if isinstance(theciftiheader.matrix.get_axis(theaxis), nib.cifti2.BrainModelAxis):
            modelaxis = theaxis
            parcellated = False
            if debug:
                print("axis", theaxis, "is the BrainModelAxis")
        elif isinstance(theciftiheader.matrix.get_axis(theaxis), nib.cifti2.ParcelsAxis):
            modelaxis = theaxis
            parcellated = True
            if debug:
                print("axis", theaxis, "is the ParcelsAxis")

    # process things differently for dscalar and dtseries files
    if isseries:
        # make a proper series header
        if parcellated:
            if debug:
                print("ptseries path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES"
            theintentname = "ConnParcelSries"
        else:
            if debug:
                print("dtseries path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES"
            theintentname = "ConnDenseSeries"
        if modelaxis is not None:
            seriesaxis = nib.cifti2.cifti2_axes.SeriesAxis(start, step, workingarray.shape[0])
            axislist = [seriesaxis, theciftiheader.matrix.get_axis(modelaxis)]
        else:
            raise KeyError("no ModelAxis found in source file - exiting")
    else:
        # make a proper scalar header
        if parcellated:
            if debug:
                print("pscalar path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SCALAR"
            theintentname = "ConnParcelScalr"
        else:
            if debug:
                print("dscalar path: workingarray shape", workingarray.shape)
            theintent = "NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS"
            theintentname = "ConnDenseScalar"
        if len(names) != workingarray.shape[0]:
            print("savetocifti - number of supplied names does not match array size - exiting.")
            sys.exit()
        if modelaxis is not None:
            scalaraxis = nib.cifti2.cifti2_axes.ScalarAxis(names)
            axislist = [scalaraxis, theciftiheader.matrix.get_axis(modelaxis)]
        else:
            raise KeyError("no ModelAxis found in source file - exiting")
    # now create the output file structure
    if debug:
        print("about to create cifti image - nifti header is:", theniftiheader)

    img = nib.cifti2.Cifti2Image(dataobj=workingarray, header=axislist)

    # make the header right
    img.nifti_header.set_intent(theintent, name=theintentname)
    img.update_headers()

    if isseries:
        if parcellated:
            suffix = ".ptseries.nii"
            if debug:
                print("\tPARCELLATED_SERIES")
        else:
            suffix = ".dtseries.nii"
            if debug:
                print("\tDENSE_SERIES")
    else:
        if parcellated:
            suffix = ".pscalar.nii"
            if debug:
                print("\tPARCELLATED_SCALARS")
        else:
            suffix = ".dscalar.nii"
            if debug:
                print("\tDENSE_SCALARS")

    if debug:
        print("after update_headers() - nifti header is:", theniftiheader)

    # save the data
    nib.cifti2.save(img, thename + suffix)


def checkifnifti(filename):
    r"""Check to see if a file name is a valid nifti name.

    Parameters
    ----------
    filename : str
        The file name

    Returns
    -------
    isnifti : bool
        True if name is a valid nifti file name.

    """
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return True
    else:
        return False


def niftisplitext(filename):
    r"""Split nifti filename into name base and extensionn.

    Parameters
    ----------
    filename : str
        The file name

    Returns
    -------
    name : str
        Base name of the nifti file.

    ext : str
        Extension of the nifti file.

    """
    firstsplit = os.path.splitext(filename)
    secondsplit = os.path.splitext(firstsplit[0])
    if secondsplit[1] is not None:
        return secondsplit[0], secondsplit[1] + firstsplit[1]
    else:
        return firstsplit[0], firstsplit[1]


def niftisplit(inputfile, outputroot, axis=3):
    infile, infile_data, infile_hdr, infiledims, infilesizes = readfromnifti(inputfile)
    theheader = copy.deepcopy(infile_hdr)
    numpoints = infiledims[axis + 1]
    print(infiledims)
    theheader["dim"][axis + 1] = 1
    for i in range(numpoints):
        if infiledims[0] == 5:
            if axis == 0:
                thisslice = infile_data[i : i + 1, :, :, :, :]
            elif axis == 1:
                thisslice = infile_data[:, i : i + 1, :, :, :]
            elif axis == 2:
                thisslice = infile_data[:, :, i : i + 1, :, :]
            elif axis == 3:
                thisslice = infile_data[:, :, :, i : i + 1, :]
            elif axis == 4:
                thisslice = infile_data[:, :, :, :, i : i + 1]
            else:
                raise ValueError("illegal axis")
        elif infiledims[0] == 4:
            if axis == 0:
                thisslice = infile_data[i : i + 1, :, :, :]
            elif axis == 1:
                thisslice = infile_data[:, i : i + 1, :, :]
            elif axis == 2:
                thisslice = infile_data[:, :, i : i + 1, :]
            elif axis == 3:
                thisslice = infile_data[:, :, :, i : i + 1]
            else:
                raise ValueError("illegal axis")
        savetonifti(thisslice, theheader, outputroot + str(i).zfill(4))


def niftimerge(inputlist, outputname, writetodisk=True, axis=3, returndata=False, debug=False):
    inputdata = []
    for thefile in inputlist:
        if debug:
            print("reading", thefile)
        infile, infile_data, infile_hdr, infiledims, infilesizes = readfromnifti(thefile)
        if infiledims[0] == 3:
            inputdata.append(
                infile_data.reshape((infiledims[1], infiledims[2], infiledims[3], 1)) + 0.0
            )
        else:
            inputdata.append(infile_data + 0.0)
    theheader = copy.deepcopy(infile_hdr)
    theheader["dim"][axis + 1] = len(inputdata)
    output_data = np.concatenate(inputdata, axis=axis)
    if writetodisk:
        savetonifti(output_data, theheader, outputname)
    if returndata:
        return output_data, infile_hdr


def niftiroi(inputfile, outputfile, startpt, numpoints):
    print(inputfile, outputfile, startpt, numpoints)
    infile, infile_data, infile_hdr, infiledims, infilesizes = readfromnifti(inputfile)
    theheader = copy.deepcopy(infile_hdr)
    theheader["dim"][4] = numpoints
    if infiledims[0] == 5:
        output_data = infile_data[:, :, :, startpt : startpt + numpoints, :]
    else:
        output_data = infile_data[:, :, :, startpt : startpt + numpoints]
    savetonifti(output_data, theheader, outputfile)


def checkifcifti(filename, debug=False):
    r"""Check to see if the specified file is CIFTI format

    Parameters
    ----------
    filename : str
        The file name

    Returns
    -------
    iscifti : bool
        True if the file header indicates this is a CIFTI file

    """
    theimg = nib.load(filename)
    thedict = vars(theimg)
    if debug:
        print("thedict:", thedict)
    try:
        intent = thedict["_nifti_header"]["intent_code"]
        if debug:
            print("intent found")
        return intent >= 3000 and intent < 3100
    except KeyError:
        if debug:
            print("intent not found")
        return False


def checkiftext(filename):
    r"""Check to see if the specified filename ends in '.txt'

    Parameters
    ----------
    filename : str
        The file name

    Returns
    -------
    istext : bool
        True if filename ends with '.txt'

    """
    if filename.endswith(".txt"):
        return True
    else:
        return False


def getniftiroot(filename):
    r"""Strip a nifti filename down to the root with no extensions

    Parameters
    ----------
    filename : str
        The file name to strip

    Returns
    -------
    strippedname : str
        The file name without any nifti extensions

    """
    if filename.endswith(".nii"):
        return filename[:-4]
    elif filename.endswith(".nii.gz"):
        return filename[:-7]
    else:
        return filename


def fmriheaderinfo(niftifilename):
    r"""Retrieve the header information from a nifti file

    Parameters
    ----------
    niftifilename : str
        The name of the nifti file

    Returns
    -------
    tr : float
        The repetition time, in seconds
    timepoints : int
        The number of points along the time axis

    """
    nim = nib.load(niftifilename)
    hdr = nim.header.copy()
    thedims = hdr["dim"].copy()
    thesizes = hdr["pixdim"].copy()
    if hdr.get_xyzt_units()[1] == "msec":
        thesizes[4] /= 1000.0
    return thesizes, thedims


def fmritimeinfo(niftifilename):
    r"""Retrieve the repetition time and number of timepoints from a nifti file

    Parameters
    ----------
    niftifilename : str
        The name of the nifti file

    Returns
    -------
    tr : float
        The repetition time, in seconds
    timepoints : int
        The number of points along the time axis

    """
    nim = nib.load(niftifilename)
    hdr = nim.header.copy()
    thedims = hdr["dim"].copy()
    thesizes = hdr["pixdim"].copy()
    if hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    timepoints = thedims[4]
    return tr, timepoints


def checkspacematch(hdr1, hdr2, tolerance=1.0e-3):
    r"""Check the headers of two nifti files to determine if the cover the same volume at the same resolution (within tolerance)

    Parameters
    ----------
    hdr1 : nifti header structure
        The header of the first file
    hdr2 : nifti header structure
        The header of the second file

    Returns
    -------
    ismatched : bool
        True if the spatial dimensions and resolutions of the two files match.

    """
    dimmatch = checkspaceresmatch(hdr1["pixdim"], hdr2["pixdim"], tolerance=tolerance)
    resmatch = checkspacedimmatch(hdr1["dim"], hdr2["dim"])
    return dimmatch and resmatch


def checkspaceresmatch(sizes1, sizes2, tolerance=1.0e-3):
    r"""Check the spatial pixdims of two nifti files to determine if they have the same resolution (within tolerance)

    Parameters
    ----------
    sizes1 : float array
        The size array from the first nifti file
    sizes2 : float array
        The size array from the second nifti file
    tolerance: float
        The fractional difference that is permissible between the two sizes that will still match

    Returns
    -------
    ismatched : bool
        True if the spatial resolutions of the two files match.

    """
    for i in range(1, 4):
        fracdiff = np.fabs(sizes1[i] - sizes2[i]) / sizes1[i]
        if fracdiff > tolerance:
            print(f"File spatial resolutions do not match within tolerance of {tolerance}")
            print(f"\tsize of dimension {i}: {sizes1[i]} != {sizes2[i]} ({fracdiff} difference)")
            return False
        else:
            return True


def checkspacedimmatch(dims1, dims2, verbose=False):
    r"""Check the dimension arrays of two nifti files to determine if the cover the same number of voxels in each dimension

    Parameters
    ----------
    dims1 : int array
        The dimension array from the first nifti file
    dims2 : int array
        The dimension array from the second nifti file

    Returns
    -------
    ismatched : bool
        True if the spatial dimensions of the two files match.
    """
    for i in range(1, 4):
        if dims1[i] != dims2[i]:
            if verbose:
                print("File spatial voxels do not match")
                print("dimension ", i, ":", dims1[i], "!=", dims2[i])
            return False
        else:
            return True


def checktimematch(dims1, dims2, numskip1=0, numskip2=0, verbose=False):
    r"""Check the dimensions of two nifti files to determine if the cover the same number of timepoints

    Parameters
    ----------
    dims1 : int array
        The dimension array from the first nifti file
    dims2 : int array
        The dimension array from the second nifti file
    numskip1 : int, optional
        Number of timepoints skipped at the beginning of file 1
    numskip2 : int, optional
        Number of timepoints skipped at the beginning of file 2

    Returns
    -------
    ismatched : bool
        True if the time dimension of the two files match.

    """
    if (dims1[4] - numskip1) != (dims2[4] - numskip2):
        if verbose:
            print("File numbers of timepoints do not match")
            print(
                "dimension ",
                4,
                ":",
                dims1[4],
                "(skip ",
                numskip1,
                ") !=",
                dims2[4],
                " (skip ",
                numskip2,
                ")",
            )
        return False
    else:
        return True


def checkdatamatch(data1, data2, absthresh=1e-12, msethresh=1e-12, debug=False):
    msediff = mse(data1, data2)
    absdiff = np.max(np.fabs(data1 - data2))
    if debug:
        print(f"msediff {msediff}, absdiff {absdiff}")
    return msediff < msethresh, absdiff < absthresh


def checkniftifilematch(
    filename1, filename2, absthresh=1e-12, msethresh=1e-12, spacetolerance=1e-3, debug=False
):
    im1, im1_data, im1_hdr, im1_dims, im1_sizes = readfromnifti(filename1)
    im2, im2_data, im2_hdr, im2_dims, im2_sizes = readfromnifti(filename2)
    spacematch = checkspacematch(im1_hdr, im2_hdr, tolerance=spacetolerance)
    if not spacematch:
        print(
            "file spatial dimensions or resolution do not match within tolerance {spacetolerance}"
        )
        return False
    timematch = checktimematch(im1_dims, im2_dims)
    if not timematch:
        print(f"file time dimensions do not match")
        return False
    msedatamatch, absdatamatch = checkdatamatch(
        im1_data,
        im2_data,
        absthresh=absthresh,
        msethresh=msethresh,
        debug=debug,
    )
    if not msedatamatch:
        print(f"file data mse does not match within tolerance {msethresh}")
        return False
    if not absdatamatch:
        print(f"files differ by at least {absthresh} in at least one voxel")
        return False
    return True


# --------------------------- non-NIFTI file I/O functions ------------------------------------------
def checkifparfile(filename):
    r"""Checks to see if a file is an FSL style motion parameter file

    Parameters
    ----------
    filename : str
        The name of the file in question.

    Returns
    -------
    isparfile : bool
        True if filename ends in '.par', False otherwise.

    """
    if filename.endswith(".par"):
        return True
    else:
        return False


def readconfounds(filename, debug=False):
    (
        thesamplerate,
        thestarttime,
        thecolumns,
        thedata,
        compressed,
        filetype,
    ) = readvectorsfromtextfile(filename, debug=debug)
    if thecolumns is None:
        thecolumns = []
        for i in range(thedata.shape[0]):
            thecolumns.append(f"confound_{str(i).zfill(3)}")
    theconfounddict = {}
    for i in range(thedata.shape[0]):
        theconfounddict[thecolumns[i]] = thedata[i]
    return theconfounddict


def readparfile(filename):
    r"""Checks to see if a file is an FSL style motion parameter file

    Parameters
    ----------
    filename : str
        The name of the file in question.

    Returns
    -------
    motiondict: dict
        All the timecourses in the file, keyed by name

    """
    labels = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
    motiontimeseries = readvecs(filename)
    motiondict = {}
    for j in range(0, 6):
        motiondict[labels[j]] = 1.0 * motiontimeseries[j, :]
    return motiondict


def readmotion(filename, tr=1.0, colspec=None):
    r"""Reads motion regressors from filename (from the columns specified in colspec, if given)

    Parameters
    ----------
    filename : str
        The name of the file in question.
    colspec: str, optional
        The column numbers from the input file to use for the 6 motion regressors

    Returns
    -------
    motiondict: dict
        All the timecourses in the file, keyed by name

    """
    # read in the motion timecourses
    print("reading motion timecourses...")
    filebase, extension = os.path.splitext(filename)
    if extension == ".par":
        allmotion = readvecs(filename)
        motiondict = {}
        motiondict["xtrans"] = allmotion[3, :] * 1.0
        motiondict["ytrans"] = allmotion[4, :] * 1.0
        motiondict["ztrans"] = allmotion[5, :] * 1.0
        motiondict["maxtrans"] = np.max(
            [
                np.max(motiondict["xtrans"]),
                np.max(motiondict["ytrans"]),
                np.max(motiondict["ztrans"]),
            ]
        )
        motiondict["mintrans"] = np.min(
            [
                np.min(motiondict["xtrans"]),
                np.min(motiondict["ytrans"]),
                np.min(motiondict["ztrans"]),
            ]
        )
        motiondict["xrot"] = allmotion[0, :] * 1.0
        motiondict["yrot"] = allmotion[1, :] * 1.0
        motiondict["zrot"] = allmotion[2, :] * 1.0
        motiondict["maxrot"] = np.max(
            [
                np.max(motiondict["xrot"]),
                np.max(motiondict["yrot"]),
                np.max(motiondict["zrot"]),
            ]
        )
        motiondict["minrot"] = np.min(
            [
                np.min(motiondict["xrot"]),
                np.min(motiondict["yrot"]),
                np.min(motiondict["zrot"]),
            ]
        )
    elif extension == ".tsv":
        allmotion = readlabelledtsv(filebase, compressed=False)
        motiondict = {}
        motiondict["xtrans"] = allmotion["trans_x"] * 1.0
        motiondict["ytrans"] = allmotion["trans_y"] * 1.0
        motiondict["ztrans"] = allmotion["trans_z"] * 1.0
        motiondict["maxtrans"] = np.max(
            [
                np.max(motiondict["xtrans"]),
                np.max(motiondict["ytrans"]),
                np.max(motiondict["ztrans"]),
            ]
        )
        motiondict["mintrans"] = np.min(
            [
                np.min(motiondict["xtrans"]),
                np.min(motiondict["ytrans"]),
                np.min(motiondict["ztrans"]),
            ]
        )
        motiondict["xrot"] = allmotion["rot_x"] * 1.0
        motiondict["yrot"] = allmotion["rot_y"] * 1.0
        motiondict["zrot"] = allmotion["rot_z"] * 1.0
        motiondict["maxrot"] = np.max(
            [
                np.max(motiondict["xrot"]),
                np.max(motiondict["yrot"]),
                np.max(motiondict["zrot"]),
            ]
        )
        motiondict["minrot"] = np.min(
            [
                np.min(motiondict["xrot"]),
                np.min(motiondict["yrot"]),
                np.min(motiondict["zrot"]),
            ]
        )
    else:
        # handle weird files gracefully
        allmotion = readvecs(filename, colspec=colspec)
        if allmotion.shape[0] != 6:
            print(
                "motion files without a .par or .tsv extension must either have 6 columns or have 6 columns specified"
            )
            sys.exit()
        # we are going to assume the columns are in FSL order, not that it really matters
        motiondict = {}
        motiondict["xtrans"] = allmotion[3, :] * 1.0
        motiondict["ytrans"] = allmotion[4, :] * 1.0
        motiondict["ztrans"] = allmotion[5, :] * 1.0
        motiondict["maxtrans"] = np.max(
            [
                np.max(motiondict["xtrans"]),
                np.max(motiondict["ytrans"]),
                np.max(motiondict["ztrans"]),
            ]
        )
        motiondict["mintrans"] = np.min(
            [
                np.min(motiondict["xtrans"]),
                np.min(motiondict["ytrans"]),
                np.min(motiondict["ztrans"]),
            ]
        )
        motiondict["xrot"] = allmotion[0, :] * 1.0
        motiondict["yrot"] = allmotion[1, :] * 1.0
        motiondict["zrot"] = allmotion[2, :] * 1.0
        motiondict["maxrot"] = np.max(
            [
                np.max(motiondict["xrot"]),
                np.max(motiondict["yrot"]),
                np.max(motiondict["zrot"]),
            ]
        )
        motiondict["minrot"] = np.min(
            [
                np.min(motiondict["xrot"]),
                np.min(motiondict["yrot"]),
                np.min(motiondict["zrot"]),
            ]
        )
    motiondict["tr"] = tr
    return motiondict


def sliceinfo(slicetimes, tr):
    r"""Find out what slicetimes we have, their spacing, and which timepoint each slice occurs at.  This assumes
    uniform slice time spacing, but supports any slice acquisition order and multiband acquisitions.

    Parameters
    ----------
    slicetimes : 1d float array
        List of all the slicetimes relative to the start of the TR
    tr: float
        The TR of the acquisition

    Returns
    -------
    numsteps : int
        The number of unique slicetimes in the list
    stepsize: float
        The stepsize in seconds between subsequent slice acquisitions
    sliceoffsets: 1d int array
        Which acquisition time each slice was acquired at
    """
    sortedtimes = np.sort(slicetimes)
    diffs = sortedtimes[1:] - sortedtimes[0:-1]
    minstep = np.max(diffs)
    numsteps = int(np.round(tr / minstep, 0))
    sliceoffsets = np.around(slicetimes / minstep).astype(int) % numsteps
    return numsteps, minstep, sliceoffsets


def getslicetimesfromfile(slicetimename):
    filebase, extension = os.path.splitext(slicetimename)
    if extension == ".json":
        jsoninfodict = readdictfromjson(slicetimename)
        try:
            slicetimelist = jsoninfodict["SliceTiming"]
            slicetimes = np.zeros((len(slicetimelist)), dtype=np.float64)
            for idx, thetime in enumerate(slicetimelist):
                slicetimes[idx] = float(thetime)
            normalizedtotr = False
            fileisbidsjson = True
        except KeyError:
            print(slicetimename, "is not a valid BIDS sidecar file")
            sys.exit()
    else:
        slicetimes = readvec(slicetimename)
        normalizedtotr = True
        fileisbidsjson = False
    return slicetimes, normalizedtotr, fileisbidsjson


def readbidssidecar(inputfilename):
    r"""Read key value pairs out of a BIDS sidecar file

    Parameters
    ----------
    inputfilename : str
        The name of the sidecar file (with extension)

    Returns
    -------
    thedict : dict
        The key value pairs from the json file

    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if os.path.exists(thefileroot + ".json"):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            return d
    else:
        print("sidecar file does not exist")
        return {}


def writedicttojson(thedict, thefilename):
    r"""Write key value pairs to a json file

    Parameters
    ----------
    thedict : dict
        The key value pairs from the json file
    thefilename : str
        The name of the json file (with extension)

    """
    thisdict = {}
    for key in thedict:
        if isinstance(thedict[key], np.integer):
            thisdict[key] = int(thedict[key])
        elif isinstance(thedict[key], np.floating):
            thisdict[key] = float(thedict[key])
        elif isinstance(thedict[key], np.ndarray):
            thisdict[key] = thedict[key].tolist()
        else:
            thisdict[key] = thedict[key]
    with open(thefilename, "wb") as fp:
        fp.write(
            json.dumps(thisdict, sort_keys=True, indent=4, separators=(",", ":")).encode("utf-8")
        )


def readdictfromjson(inputfilename):
    r"""Read key value pairs out of a json file

    Parameters
    ----------
    inputfilename : str
        The name of the json file (with extension)

    Returns
    -------
    thedict : dict
        The key value pairs from the json file

    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if os.path.exists(thefileroot + ".json"):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            return d
    else:
        print("specified json file does not exist")
        return {}


def readlabelledtsv(inputfilename, compressed=False):
    r"""Read time series out of an fmriprep confounds tsv file

    Parameters
    ----------
    inputfilename : str
        The root name of the tsv (no extension)

    Returns
    -------
        confounddict: dict
            All the timecourses in the file, keyed by the first row

    NOTE:  If file does not exist or is not valid, return an empty dictionary

    """
    confounddict = {}
    if compressed:
        theext = ".tsv.gz"
    else:
        theext = ".tsv"

    if not os.path.isfile(inputfilename + theext):
        raise FileNotFoundError(f"Labelled tsv file {inputfilename + theext} does not exist")

    df = pd.read_csv(inputfilename + theext, sep="\t", quotechar='"')

    # replace nans with 0
    df = df.fillna(0.0)

    for thecolname, theseries in df.items():
        confounddict[thecolname] = theseries.values
    return confounddict


def readcsv(inputfilename, debug=False):
    r"""Read time series out of an unlabelled csv file

    Parameters
    ----------
    inputfilename : str
        The root name of the csv (no extension)

    Returns
    -------
        timeseriesdict: dict
            All the timecourses in the file, keyed by the first row if it exists, by "col1, col2...colN"
            if not.

    NOTE:  If file does not exist or is not valid, return an empty dictionary

    """
    if not os.path.isfile(inputfilename + ".csv"):
        raise FileNotFoundError(f"csv file {inputfilename}.csv does not exist")

    timeseriesdict = {}

    # Read the data in initially with no header
    df = pd.read_csv(inputfilename + ".csv", sep=",", quotechar='"', header=0)

    # replace nans with 0
    df = df.fillna(0.0)

    # Check to see if the first element is a float
    try:
        dummy = float(df.columns[0])
    except ValueError:
        if debug:
            print("there is a header line")
        for thecolname, theseries in df.items():
            timeseriesdict[thecolname] = theseries.values
    else:
        # if we got to here, reread the data, but assume there is no header line
        df = pd.read_csv(inputfilename + ".csv", sep=",", quotechar='"', header=None)

        # replace nans with 0
        df = df.fillna(0.0)

        if debug:
            print("there is no header line")
        colnum = 0
        for dummy, theseries in df.items():
            timeseriesdict[makecolname(colnum, 0)] = theseries.values
            colnum += 1
        # timeseriesdict["columnsource"] = "synthetic"

    return timeseriesdict


def readfslmat(inputfilename, debug=False):
    r"""Read time series out of an FSL design.mat file

    Parameters
    ----------
    inputfilename : str
        The root name of the csv (no extension)

    Returns
    -------
        timeseriesdict: dict
            All the timecourses in the file, keyed by the first row if it exists, by "col1, col2...colN"
            if not.

    NOTE:  If file does not exist or is not valid, return an empty dictionary

    """
    if not os.path.isfile(inputfilename + ".mat"):
        raise FileNotFoundError(f"FSL mat file {inputfilename}.mat does not exist")

    timeseriesdict = {}

    # Read the data in with no header
    df = pd.read_csv(inputfilename + ".mat", delim_whitespace=True, header=None, skiprows=5)

    if debug:
        print(df)
    colnum = 0
    for dummy, theseries in df.items():
        timeseriesdict[makecolname(colnum, 0)] = theseries.values
        colnum += 1

    return timeseriesdict


def readoptionsfile(inputfileroot):
    if os.path.isfile(inputfileroot + ".json"):
        # options saved as json
        thedict = readdictfromjson(inputfileroot + ".json")
    elif os.path.isfile(inputfileroot + ".txt"):
        # options saved as text
        thedict = readdict(inputfileroot + ".txt")
    else:
        raise FileNotFoundError(f"options file {inputfileroot}(.json/.txt) does not exist")

    # correct behavior for older options files
    try:
        test = thedict["lowerpass"]
    except KeyError:
        print("no filter limits found in options file - filling in defaults")
        if thedict["filtertype"] == "None":
            thedict["lowerstop"] = 0.0
            thedict["lowerpass"] = 0.0
            thedict["upperpass"] = -1.0
            thedict["upperstop"] = -1.0
        elif thedict["filtertype"] == "vlf":
            thedict["lowerstop"] = 0.0
            thedict["lowerpass"] = 0.0
            thedict["upperpass"] = 0.009
            thedict["upperstop"] = 0.010
        elif thedict["filtertype"] == "lfo":
            thedict["lowerstop"] = 0.009
            thedict["lowerpass"] = 0.010
            thedict["upperpass"] = 0.15
            thedict["upperstop"] = 0.20
        elif thedict["filtertype"] == "resp":
            thedict["lowerstop"] = 0.15
            thedict["lowerpass"] = 0.20
            thedict["upperpass"] = 0.4
            thedict["upperstop"] = 0.5
        elif thedict["filtertype"] == "card":
            thedict["lowerstop"] = 0.4
            thedict["lowerpass"] = 0.5
            thedict["upperpass"] = 2.5
            thedict["upperstop"] = 3.0
        elif thedict["filtertype"] == "arb":
            thedict["lowerstop"] = thedict["arb_lowerstop"]
            thedict["lowerpass"] = thedict["arb_lower"]
            thedict["upperpass"] = thedict["arb_upper"]
            thedict["upperstop"] = thedict["arb_upperstop"]
        else:
            print("cannot determine filtering")
            thedict["lowerstop"] = 0.0
            thedict["lowerpass"] = 0.0
            thedict["upperpass"] = -1.0
            thedict["upperstop"] = -1.0
    return thedict


def makecolname(colnum, startcol):
    return f"col_{str(colnum + startcol).zfill(2)}"


def writebidstsv(
    outputfileroot,
    data,
    samplerate,
    extraheaderinfo=None,
    compressed=True,
    columns=None,
    xaxislabel="time",
    yaxislabel="arbitrary value",
    starttime=0.0,
    append=False,
    samplerate_tolerance=1e-6,
    starttime_tolerance=1e-6,
    colsinjson=True,
    colsintsv=False,
    omitjson=False,
    debug=False,
):
    """
    NB: to be strictly valid, a continuous BIDS tsv file (i.e. a "_physio" or "_stim" file) requires:
    1) The .tsv is compressed (.tsv.gz)
    2) "SamplingFrequency", "StartTime", "Columns" must exist and be in the .json file
    3) The tsv file does NOT have column headers.
    4) "_physio" or "_stim" has to be at the end of the name, although this seems a little flexible

    The first 3 are the defaults, but if you really want to override them, you can.

    :param outputfileroot:
    :param data:
    :param samplerate:
    :param compressed:
    :param columns:
    :param xaxislabel:
    :param yaxislabel:
    :param starttime:
    :param append:
    :param colsinjson:
    :param colsintsv:
    :param omitjson:
    :param debug:
    :return:
    """
    if debug:
        print("entering writebidstsv:")
        print("\toutputfileroot:", outputfileroot)
        print("\tdata.shape:", data.shape)
        print("\tsamplerate:", samplerate)
        print("\tcompressed:", compressed)
        print("\tcolumns:", columns)
        print("\txaxislabel:", xaxislabel)
        print("\tyaxislabel:", yaxislabel)
        print("\tstarttime:", starttime)
        print("\tappend:", append)
    if len(data.shape) == 1:
        reshapeddata = data.reshape((1, -1))
        if debug:
            print("input data reshaped from", data.shape, "to", reshapeddata.shape)
    else:
        reshapeddata = data
    if append:
        insamplerate, instarttime, incolumns, indata, incompressed, incolsource = readbidstsv(
            outputfileroot + ".json", neednotexist=True, debug=debug,
        )
        if debug:
            print("appending")
            print(insamplerate, instarttime, incolumns, indata, incompressed, incolsource)
        if insamplerate is None:
            # file does not already exist
            if debug:
                print("creating file:", data.shape, columns, samplerate)
            startcol = 0
        else:
            # file does already exist
            if debug:
                print(
                    "appending:",
                    insamplerate,
                    instarttime,
                    incolumns,
                    incolsource,
                    indata.shape,
                    reshapeddata.shape,
                )
            compressed = incompressed
            if (
                np.fabs(insamplerate - samplerate) < samplerate_tolerance
                and np.fabs(instarttime - starttime) < starttime_tolerance
                and reshapeddata.shape[1] == indata.shape[1]
            ):
                startcol = len(incolumns)
            else:
                print("data dimensions not compatible with existing dimensions")
                print(samplerate, insamplerate)
                print(starttime, instarttime)
                print(columns, incolumns, incolsource)
                print(indata.shape, reshapeddata.shape)
                sys.exit()
    else:
        startcol = 0

    if columns is None:
        columns = []
        for i in range(reshapeddata.shape[0]):
            columns.append(makecolname(i, startcol))
    else:
        if len(columns) != reshapeddata.shape[0]:
            raise ValueError(
                f"number of column names ({len(columns)}) ",
                f"does not match number of columns ({reshapeddata.shape[0]}) in data",
            )
    if startcol > 0:
        df = pd.DataFrame(data=np.transpose(indata), columns=incolumns)
        for i in range(len(columns)):
            df[columns[i]] = reshapeddata[i, :]
    else:
        df = pd.DataFrame(data=np.transpose(reshapeddata), columns=columns)
    if compressed:
        df.to_csv(
            outputfileroot + ".tsv.gz", sep="\t", compression="gzip", header=colsintsv, index=False
        )
    else:
        df.to_csv(
            outputfileroot + ".tsv", sep="\t", compression=None, header=colsintsv, index=False
        )
    headerdict = {}
    headerdict["SamplingFrequency"] = float(samplerate)
    headerdict["StartTime"] = float(starttime)
    headerdict["XAxisLabel"] = xaxislabel
    headerdict["YAxisLabel"] = yaxislabel
    if colsinjson:
        if startcol == 0:
            headerdict["Columns"] = columns
        else:
            headerdict["Columns"] = incolumns + columns
    if extraheaderinfo is not None:
        for key in extraheaderinfo:
            headerdict[key] = extraheaderinfo[key]

    if not omitjson:
        with open(outputfileroot + ".json", "wb") as fp:
            fp.write(
                json.dumps(headerdict, sort_keys=True, indent=4, separators=(",", ":")).encode(
                    "utf-8"
                )
            )


def readvectorsfromtextfile(fullfilespec, onecol=False, debug=False):
    r"""Read one or more time series from some sort of text file

    Parameters
    ----------
    fullfilespec : str
        The file name.  If extension is .tsv or .json, it will be assumed to be either a BIDS tsv, or failing that,
         a non-BIDS tsv.  If the extension is .csv, it will be assumed to be a csv file. If the extension is .mat,
         it will be assumed to be an FSL design.mat file.  If any other extension or
         no extension, it will be assumed to be a plain, whitespace separated text file.
    colspec:  A valid list and/or range of column numbers, or list of column names, or None
    debug : bool
        Output additional debugging information

    Returns
    -------
        samplerate : float
            Sample rate in Hz.  None if not knowable.
        starttime : float
            Time of first point, in seconds. None if not knowable.
        columns : str array
            Names of the timecourses contained in the file. None if not knowable.
        data : 2D numpy array
            Timecourses from the file
        compressed: bool
            True if time data is gzipped (as in a .tsv.gz file).
        filetype: str
            One of "text", "csv", "plaintsv", "bidscontinuous".

    NOTE:  If file does not exist or is not valid, all return values are None"""

    thefilename, colspec = parsefilespec(fullfilespec)
    thefileroot, theext = os.path.splitext(thefilename)

    if theext == ".gz":
        thefileroot, thenextext = os.path.splitext(thefileroot)
        if thenextext is not None:
            theext = thenextext + theext
    if debug:
        print("thefileroot:", thefileroot)
        print("theext:", theext)
        print("colspec:", colspec)
    jsonexists = os.path.exists(thefileroot + ".json")
    tsvexists = os.path.exists(thefileroot + ".tsv.gz") or os.path.exists(thefileroot + ".tsv")
    compressed = os.path.exists(thefileroot + ".tsv.gz")
    csvexists = os.path.exists(thefileroot + ".csv")
    matexists = os.path.exists(thefileroot + ".mat")
    if debug:
        print("jsonexists=", jsonexists)
        print("tsvexists=", tsvexists)
        print("compressed=", compressed)
    if tsvexists:
        if jsonexists:
            filetype = "bidscontinuous"
        else:
            filetype = "plaintsv"
    elif csvexists:
        filetype = "csv"
    elif matexists:
        filetype = "mat"
    else:
        filetype = "text"
    if debug:
        print(f"detected file type is {filetype}")
    if debug:
        print(f"filetype of {fullfilespec} determined to be", filetype)
    if filetype == "text":
        # colspec can only be None or a list of integer ranges
        thedata = readvecs(thefilename, colspec=colspec, debug=debug)
        if onecol and thedata.shape[0] > 1:
            print("specify a single column from", thefilename)
            sys.exit()
        thesamplerate = None
        thestarttime = None
        thecolumns = None
        compressed = None
    elif filetype == "bidscontinuous":
        # colspec can be None or a list of comma separated column names
        colspectouse = None
        if colspec is not None:
            try:
                colspectouse = makecolname(int(colspec), 0)
            except ValueError:
                colspectouse = colspec
        thesamplerate, thestarttime, thecolumns, thedata, compressed, colsource = readbidstsv(
            thefilename, colspec=colspectouse, debug=debug
        )
        if thedata is None:
            raise ValueError(f"specified column {colspectouse} does not exist")
        if onecol and thedata.shape[0] > 1:
            raise ValueError("specify a single column from", thefilename)
    elif filetype == "plaintsv":
        thedatadict = readlabelledtsv(thefileroot, compressed=compressed)
        if colspec is None:
            thecolumns = list(thedatadict.keys())
        else:
            try:
                thecolumns = [makecolname(int(colspec), 0)]
            except ValueError:
                thecolumns = colspec.split(",")
                colsource = "data"
            else:
                colsource = "synthetic"
        if onecol and len(thecolumns) > 1:
            print("specify a single column from", thefilename)
            sys.exit()
        thedatacols = []
        for thekey in thecolumns:
            try:
                thedatacols.append(thedatadict[thekey])
            except KeyError:
                print(thefilename, "does not contain column", thekey)
                sys.exit()
        thedata = np.array(thedatacols)
        thesamplerate = None
        thestarttime = None
        compressed = None
    elif (filetype == "csv") or (filetype == "mat"):
        if filetype == "csv":
            thedatadict = readcsv(thefileroot, debug=debug)
        else:
            thedatadict = readfslmat(thefileroot, debug=debug)
        if colspec is None:
            thecolumns = list(thedatadict.keys())
        else:
            thecolumns = colspec.split(",")
        if onecol and len(thecolumns) > 1:
            print("specify a single column from", thefilename)
            sys.exit()
        thedatacols = []
        for thekey in thecolumns:
            try:
                thedatacols.append(thedatadict[thekey])
            except KeyError:
                # try expanding the numbers
                try:
                    thedatacols.append(thedatadict[makecolname(int(thekey), 0)])
                except:
                    print(thefilename, "does not contain column", thekey)
                    sys.exit()
        thedata = np.array(thedatacols)
        thesamplerate = None
        thestarttime = None
        compressed = None
    else:
        print("illegal file type:", filetype)

    if onecol:
        thedata = thedata[0, :]

    if debug:
        print("\tthesamplerate:", thesamplerate)
        print("\tthestarttime:", thestarttime)
        print("\tthecolumns:", thecolumns)
        print("\tthedata.shape:", thedata.shape)
        print("\tcompressed:", compressed)
        print("\tfiletype:", filetype)

    return thesamplerate, thestarttime, thecolumns, thedata, compressed, filetype


def readbidstsv(inputfilename, colspec=None, warn=True, neednotexist=False, debug=False):
    r"""Read time series out of a BIDS tsv file

    Parameters
    ----------
    inputfilename : str
        The root name of the tsv and accompanying json file (no extension)
    colspec: list
        A comma separated list of column names to return
    debug : bool
        Output additional debugging information

    Returns
    -------
        samplerate : float
            Sample rate in Hz
        starttime : float
            Time of first point, in seconds
        columns : str array
            Names of the timecourses contained in the file
        data : 2D numpy array
            Timecourses from the file

    NOTE:  If file does not exist or is not valid, all return values are None

    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if theext == ".gz":
        thefileroot, thenextext = os.path.splitext(thefileroot)
        if thenextext is not None:
            theext = thenextext + theext

    if debug:
        print("thefileroot:", thefileroot)
        print("theext:", theext)
    if os.path.exists(thefileroot + ".json") and (
        os.path.exists(thefileroot + ".tsv.gz") or os.path.exists(thefileroot + ".tsv")
    ):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            try:
                samplerate = float(d["SamplingFrequency"])
            except:
                print("no samplerate found in json, setting to 1.0")
                samplerate = 1.0
                if warn:
                    print(
                        "Warning - SamplingFrequency not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            try:
                starttime = float(d["StartTime"])
            except:
                print("no starttime found in json, setting to 0.0")
                starttime = 0.0
                if warn:
                    print(
                        "Warning - StartTime not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            try:
                columns = d["Columns"]
            except:
                if debug:
                    print("no columns found in json, will take labels from the tsv file")
                columns = None
                if warn:
                    print(
                        "Warning - Columns not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            else:
                columnsource = "json"
        if os.path.exists(thefileroot + ".tsv.gz"):
            compression = "gzip"
            theextension = ".tsv.gz"
        else:
            compression = None
            theextension = ".tsv"
            if warn:
                print(
                    "Warning - "
                    + thefileroot
                    + ".tsv is uncompressed.  This is not BIDS compliant."
                )

        df = pd.read_csv(
            thefileroot + theextension,
            compression=compression,
            names=columns,
            header=None,
            sep="\t",
            quotechar='"',
        )

        # replace nans with 0
        df = df.fillna(0.0)

        # check for header line
        if any(df.iloc[0].apply(lambda x: isinstance(x, str))):
            headerlinefound = True
            # reread the data, skipping the first row
            df = pd.read_csv(
                thefileroot + theextension,
                compression=compression,
                names=columns,
                header=0,
                sep="\t",
                quotechar='"',
            )

            # replace nans with 0
            df = df.fillna(0.0)

            if warn:
                print(
                    "Warning - Column header line found in "
                    + thefileroot
                    + ".tsv.  This is not BIDS compliant."
                )
        else:
            headerlinefound = False

        if columns is None:
            columns = list(df.columns.values)
            columnsource = "tsv"
        if debug:
            print(
                samplerate,
                starttime,
                columns,
                np.transpose(df.to_numpy()).shape,
                (compression == "gzip"),
                warn,
                headerlinefound,
            )

        # select a subset of columns if they were specified
        if colspec is None:
            return (
                samplerate,
                starttime,
                columns,
                np.transpose(df.to_numpy()),
                (compression == "gzip"),
                columnsource,
            )
        else:
            collist = colspec.split(",")
            try:
                selectedcols = df[collist]
            except KeyError:
                print("specified column list cannot be found in", inputfilename)
                return [None, None, None, None, None, None]
            columns = list(selectedcols.columns.values)
            return (
                samplerate,
                starttime,
                columns,
                np.transpose(selectedcols.to_numpy()),
                (compression == "gzip"),
                columnsource,
            )
    else:
        if neednotexist:
            return [None, None, None, None, None, None]
        else:
            raise FileNotFoundError(f"file pair {thefileroot}(.json/.tsv[.gz]) does not exist")


def readcolfrombidstsv(inputfilename, columnnum=0, columnname=None, neednotexist=False, debug=False):
    r"""

    Parameters
    ----------
    inputfilename
    columnnum
    columnname

    Returns
    -------

    """
    samplerate, starttime, columns, data, compressed, colsource = readbidstsv(
        inputfilename, neednotexist=neednotexist, debug=debug
    )
    if data is None:
        print("no valid datafile found")
        return None, None, None
    else:
        if columnname is not None:
            # looking for a named column
            try:
                thecolnum = columns.index(columnname)
                return samplerate, starttime, data[thecolnum, :]
            except:
                print("no column named", columnname, "in", inputfilename)
                return None, None, None
        # we can only get here if columnname is undefined
        if not (0 < columnnum < len(columns)):
            print(
                "specified column number",
                columnnum,
                "is out of range in",
                inputfilename,
            )
            return None, None, None
        else:
            return samplerate, starttime, data[columnnum, :]


def parsefilespec(filespec, debug=False):
    inputlist = filespec.split(":")
    if debug:
        print(f"PARSEFILESPEC: input string >>>{filespec}<<<")
        print(f"PARSEFILESPEC: platform is {platform.system()}")

    specialcase = False
    if len(inputlist) > 1:
        if filespec[1] == ":" and platform.system() == "Windows":
            specialcase = True
    if specialcase:
        thefilename = ":".join([inputlist[0], inputlist[1]])
        if len(inputlist) == 3:
            thecolspec = inputlist[2]
        elif len(inputlist) == 2:
            thecolspec = None
        else:
            raise ValueError(
                f"PARSEFILESPEC: Badly formed file specification {filespec} - exiting"
            )
    else:
        thefilename = inputlist[0]
        if len(inputlist) == 2:
            thecolspec = inputlist[1]
        elif len(inputlist) == 1:
            thecolspec = None
        else:
            raise ValueError(
                f"PARSEFILESPEC: Badly formed file specification {filespec} - exiting"
            )
    if debug:
        print(f"PARSEFILESPEC: thefilename is >>>{filespec}<<<, thecolspec is >>>{thecolspec}<<<")
    return thefilename, thecolspec


def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if op.countOf(unique_list, x) == 0:
            unique_list.append(x)
    return unique_list


def colspectolist(colspec, debug=False):
    if colspec is None:
        print("COLSPECTOLIST: no range specification - exiting")
        return None
    collist = []
    theranges = colspec.split(",")

    def safeint(s):
        try:
            int(s)
            return int(s)
        except ValueError:
            print("COLSPECTOLIST:", s, "is not a legal integer - exiting")
            return None

    # convert value macros, if present
    macros = (
        ("APARC_SUBCORTGRAY", "8-13,17-20,26-28,47-56,58-60,96,97"),
        ("APARC_CORTGRAY", "1000-1035,2000-2035"),
        ("APARC_GRAY", "8-13,17-20,26-28,47-56,58-60,96,97,1000-1035,2000-2035"),
        ("APARC_WHITE", "2,7,41,46,177,219,3000-3035,4000-4035,5001,5002"),
        ("APARC_CSF", "4,5,14,15,24,31,43,44,63,72"),
        (
            "APARC_ALLBUTCSF",
            "2,7-13,17-20,26-28,41,46-56,58-60,96,97,177,219,1000-1035,2000-2035,3000-3035,4000-4035,5001,5002",
        ),
        ("SSEG_GRAY", "3,8,10-13,16-18,26,42,47,49-54,58"),
        ("SSEG_WHITE", "2,7,41,46"),
        ("SSEG_CSF", "4,5,14,15,24,43,44"),
    )
    preprocessedranges = []
    for thisrange in theranges:
        converted = False
        for themacro in macros:
            if thisrange == themacro[0]:
                if debug:
                    print(f"COLSPECTOLIST: macro {thisrange} detected.")
                preprocessedranges += themacro[1].split(",")
                converted = True
        if not converted:
            preprocessedranges.append(thisrange)
    for thisrange in preprocessedranges:
        if debug:
            print("processing range", thisrange)
        theendpoints = thisrange.split("-")
        if len(theendpoints) == 1:
            collist.append(safeint(theendpoints[0]))
        elif len(theendpoints) == 2:
            start = safeint(theendpoints[0])
            end = safeint(theendpoints[1])
            if start < 0:
                print("COLSPECTOLIST:", start, "must be greater than zero")
                return None
            if end < start:
                print("COLSPECTOLIST:", end, "must be greater than or equal to", start)
                return None
            for i in range(start, end + 1):
                collist.append(i)
        else:
            print("COLSPECTOLIST: bad range specification - exiting")
            return None

    return unique(sorted(collist))


def processnamespec(maskspec, spectext1, spectext2, debug=False):
    thename, colspec = parsefilespec(maskspec)
    if colspec is not None:
        thevals = colspectolist(colspec)
    else:
        thevals = None
    if (thevals is not None) and debug:
        print(spectext1, thename, " = ", thevals, spectext2)
    return thename, thevals


def readcolfromtextfile(inputfilespec):
    r"""

    Parameters
    ----------
    inputfilename
    colspec

    Returns
    -------
    """
    inputfilename, colspec = parsefilespec(inputfilespec)
    if inputfilename is None:
        print("Badly formed file specification", inputfilespec, "- exiting")
        sys.exit()

    inputdata = np.transpose(readvecs(inputfilename, colspec=colspec))
    if np.shape(inputdata)[1] > 1:
        print("specify only one column for input file 1")
        sys.exit()
    else:
        return inputdata[:, 0]


def readvecs(inputfilename, colspec=None, numskip=0, debug=False, thedtype=float):
    r"""

    Parameters
    ----------
    inputfilename

    Returns
    -------

    """
    if debug:
        print(f"inputfilename: {inputfilename}")
        print(f"colspec: {colspec}")
        print(f"numskip: {numskip}")
    with open(inputfilename, "r") as thefile:
        lines = thefile.readlines()
    if colspec is None:
        numvecs = len(lines[0].split())
        collist = range(0, numvecs)
    else:
        collist = colspectolist(colspec)
        if collist[-1] > len(lines[0].split()):
            print("READVECS: too many columns requested - exiting")
            sys.exit()
        if max(collist) > len(lines[0].split()) - 1:
            raise ValueError("READVECS: requested column", max(collist), "too large - exiting")
    inputvec = []
    if numskip == 0:
        try:
            test = float((lines[0].split())[0])
        except ValueError:
            numskip = 1
    for line in lines[numskip:]:
        if len(line) > 1:
            thetokens = line.split()
            thisvec = []
            for vecnum in collist:
                thisvec.append(thedtype(thetokens[vecnum]))
            inputvec.append(thisvec)
    theoutarray = np.transpose(np.asarray(inputvec, dtype=thedtype))
    return theoutarray


def readvec(inputfilename, numskip=0):
    r"""Read an array of floats in from a text file.

    Parameters
    ----------
    inputfilename : str
        The name of the text file

    Returns
    -------
    inputdata : 1D numpy float array
        The data from the file

    """
    inputvec = []
    with open(inputfilename, "r") as thefile:
        lines = thefile.readlines()
        for line in lines[numskip:]:
            if len(line) > 1:
                inputvec.append(np.float64(line))
    return np.asarray(inputvec, dtype=float)


def readtc(inputfilename, colnum=None, colname=None, debug=False):
    # check file type
    filebase, extension = os.path.splitext(inputfilename)
    inputfreq = None
    inputstart = None
    if debug:
        print("filebase:", filebase)
        print("extension:", extension)
    if extension == ".json":
        if (colnum is None) and (colname is None):
            raise ValueError("You must specify a column name or number to read a bidstsv file")
        if (colnum is not None) and (colname is not None):
            raise ValueError(
                "You must specify a column name or number, but not both, to read a bidstsv file"
            )
        inputfreq, inputstart, timecourse = readcolfrombidstsv(
            inputfilename, columnname=colname, columnnum=colnum, debug=debug
        )
    else:
        timecourse = np.transpose(readvecs(inputfilename))
        if debug:
            print(timecourse.shape)
        if len(timecourse.shape) != 1:
            if (colnum is None) or (colname is not None):
                raise TypeError(
                    "You must specify a column number (not a name) to read a column from a multicolumn file"
                )
            timecourse = timecourse[:, colnum]

    return timecourse, inputfreq, inputstart


def readlabels(inputfilename):
    r"""

    Parameters
    ----------
    inputfilename

    Returns
    -------

    """
    inputvec = []
    with open(inputfilename, "r") as thefile:
        lines = thefile.readlines()
        for line in lines:
            inputvec.append(line.rstrip())
    return inputvec


def writedict(thedict, outputfile, lineend="", machinereadable=False):
    r"""
    Write all the key value pairs from a dictionary to a text file.

    Parameters
    ----------
    thedict : dict
        A dictionary
    outputfile : str
        The name of the output file
    lineend : { 'mac', 'win', 'linux' }, optional
        Line ending style to use. Default is 'linux'.

    Returns
    -------

    """
    if lineend == "mac":
        thelineending = "\r"
        openmode = "wb"
    elif lineend == "win":
        thelineending = "\r\n"
        openmode = "wb"
    elif lineend == "linux":
        thelineending = "\n"
        openmode = "wb"
    else:
        thelineending = "\n"
        openmode = "w"
    with open(outputfile, openmode) as FILE:
        if machinereadable:
            FILE.writelines("{" + thelineending)
        for key, value in sorted(thedict.items()):
            if machinereadable:
                FILE.writelines('"' + str(key) + '"' + ":\t" + str(value) + thelineending)
            else:
                FILE.writelines(str(key) + ":\t" + str(value) + thelineending)
        if machinereadable:
            FILE.writelines("}" + thelineending)


def readdict(inputfilename):
    r"""Read key value pairs out of a text file

    Parameters
    ----------
    inputfilename : str
        The name of the json file (with extension)

    Returns
    -------
    thedict : dict
        The key value pairs from the json file

    """
    if os.path.exists(inputfilename):
        thedict = {}
        with open(inputfilename, "r") as f:
            for line in f:
                values = line.split()
                key = values[0][:-1]
                thevalues = values[1:]
                if len(thevalues) == 1:
                    thevalues = thevalues[0]
                thedict[key] = thevalues
        return thedict
    else:
        print("specified file does not exist")
        return {}


def writevec(thevec, outputfile, lineend=""):
    r"""Write a vector out to a text file.
    Parameters
    ----------
    thevec : 1D numpy or python array
        The array to write.
    outputfile : str
        The name of the output file
    lineend : { 'mac', 'win', 'linux' }, optional
        Line ending style to use. Default is 'linux'.

    Returns
    -------

    """
    if lineend == "mac":
        thelineending = "\r"
        openmode = "wb"
    elif lineend == "win":
        thelineending = "\r\n"
        openmode = "wb"
    elif lineend == "linux":
        thelineending = "\n"
        openmode = "wb"
    else:
        thelineending = "\n"
        openmode = "w"
    with open(outputfile, openmode) as FILE:
        for i in thevec:
            FILE.writelines(str(i) + thelineending)


def writevectorstotextfile(
    thevecs,
    outputfile,
    samplerate=1.0,
    starttime=0.0,
    columns=None,
    compressed=True,
    filetype="text",
    lineend="",
    debug=False,
):
    if filetype == "text":
        writenpvecs(thevecs, outputfile, headers=columns, lineend=lineend)
    elif filetype == "csv":
        writenpvecs(thevecs, outputfile, headers=columns, ascsv=True, lineend=lineend)
    elif filetype == "bidscontinuous":
        writebidstsv(
            niftisplitext(outputfile)[0],
            thevecs,
            samplerate,
            compressed=compressed,
            columns=columns,
            starttime=starttime,
            append=False,
            colsinjson=True,
            colsintsv=False,
            debug=debug,
        )
    elif filetype == "plaintsv":
        writebidstsv(
            niftisplitext(outputfile)[0],
            thevecs,
            samplerate,
            compressed=compressed,
            columns=columns,
            starttime=starttime,
            append=False,
            colsinjson=False,
            colsintsv=True,
            omitjson=True,
            debug=debug,
        )

    else:
        raise ValueError("illegal file type")


# rewritten to guarantee file closure, combines writenpvec and writenpvecs
def writenpvecs(thevecs, outputfile, ascsv=False, headers=None, altmethod=True, lineend=""):
    r"""Write out a two dimensional numpy array to a text file

    Parameters
    ----------
    thevecs: 1D or 2D numpy array
        The data to write to the file
    outputfile : str
        The name of the output file
    lineend : { 'mac', 'win', 'linux' }, optional
        Line ending style to use. Default is 'linux'.

    Returns
    -------

    """
    theshape = np.shape(thevecs)
    if lineend == "mac":
        thelineending = "\r"
        openmode = "wb"
    elif lineend == "win":
        thelineending = "\r\n"
        openmode = "wb"
    elif lineend == "linux":
        thelineending = "\n"
        openmode = "wb"
    else:
        thelineending = "\n"
        openmode = "w"
    if ascsv:
        theseparator = ","
    else:
        theseparator = "\t"
    if headers is not None:
        if thevecs.ndim == 2:
            if len(headers) != theshape[0]:
                raise ValueError("number of header lines must equal the number of data columns")
        else:
            if len(headers) != 1:
                raise ValueError("number of header lines must equal the number of data columns")
    with open(outputfile, openmode) as FILE:
        if headers is not None:
            FILE.writelines(theseparator.join(headers) + thelineending)
        if thevecs.ndim == 2:
            for i in range(0, theshape[1]):
                if altmethod:
                    outline = theseparator.join(thevecs[:, i].astype(str).tolist()) + thelineending
                    FILE.writelines(outline)
                else:
                    for j in range(0, theshape[0]):
                        FILE.writelines(str(thevecs[j, i]) + "\t")
                    FILE.writelines(thelineending)
        else:
            for i in range(0, theshape[0]):
                FILE.writelines(str(thevecs[i]) + thelineending)
