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
# $Author: frederic $
# $Date: 2016/07/12 13:50:29 $
# $Id: tide_funcs.py,v 1.4 2016/07/12 13:50:29 frederic Exp $
#
import bisect
import logging

import numpy as np
from nilearn import masking

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats

LGR = logging.getLogger("GENERAL")


def resampmask(themask, thetargetres):
    resampmask = themask
    return themask


def makeepimask(nim):
    return masking.compute_epi_mask(nim)


def maketmask(filename, timeaxis, maskvector, debug=False):
    inputdata = tide_io.readvecs(filename)
    theshape = np.shape(inputdata)
    if theshape[0] == 1:
        # this is simply a vector, one per TR.  If the value is nonzero, include the point, otherwise don't
        if theshape[1] == len(timeaxis):
            maskvector = np.where(inputdata[0, :] > 0.0, 1.0, 0.0)
        else:
            raise ValueError("tmask length does not match fmri data")
    else:
        maskvector *= 0.0
        for idx in range(0, theshape[1]):
            starttime = inputdata[0, idx]
            endtime = starttime + inputdata[1, idx]
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(maskvector) - 1))
            maskvector[startindex:endindex] = 1.0
            LGR.info(f"{starttime}, {startindex}, {endtime}, {endindex}")
    return maskvector


def readamask(
    maskfilename,
    nim_hdr,
    xsize,
    istext=False,
    valslist=None,
    thresh=None,
    maskname="the",
    tolerance=1.0e-3,
):
    LGR.debug(f"readamask called with filename: {maskfilename} vals: {valslist}")
    if istext:
        maskarray = tide_io.readvecs(maskfilename).astype("uint16")
        theshape = np.shape(maskarray)
        theincludexsize = theshape[0]
        if not theincludexsize == xsize:
            raise ValueError(
                f"Dimensions of {maskname} mask do not match the input data - exiting"
            )
    else:
        themask, maskarray, mask_hdr, maskdims, masksizes = tide_io.readfromnifti(maskfilename)
        if not tide_io.checkspacematch(mask_hdr, nim_hdr, tolerance=tolerance):
            raise ValueError(f"Dimensions of {maskname} mask do not match the fmri data - exiting")
        if thresh is None:
            maskarray = np.round(maskarray, 0).astype("uint16")
        else:
            maskarray = np.where(maskarray > thresh, 1, 0).astype("uint16")

    if valslist is not None:
        tempmask = (0 * maskarray).astype("uint16")
        for theval in valslist:
            LGR.debug(f"looking for voxels matching {theval}")
            tempmask[np.where(maskarray - theval == 0)] += 1
        maskarray = np.where(tempmask > 0, 1, 0)

    maskarray = np.where(maskarray > 0, 1, 0).astype("uint16")
    return maskarray


def getmaskset(
    maskname,
    includename,
    includevals,
    excludename,
    excludevals,
    datahdr,
    numspatiallocs,
    extramask=None,
    extramaskthresh=0.1,
    istext=False,
    tolerance=1.0e-3,
):
    internalincludemask = None
    internalexcludemask = None
    internalextramask = None

    if includename is not None:
        LGR.info(f"constructing {maskname} include mask")
        theincludemask = readamask(
            includename,
            datahdr,
            numspatiallocs,
            istext=istext,
            valslist=includevals,
            maskname=f"{maskname} include",
            tolerance=tolerance,
        )
        internalincludemask = theincludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalincludemask) == 0:
            raise ValueError(
                f"ERROR: there are no voxels in the {maskname} include mask - exiting"
            )

    if excludename is not None:
        LGR.info(f"constructing {maskname} exclude mask")
        theexcludemask = readamask(
            excludename,
            datahdr,
            numspatiallocs,
            istext=istext,
            valslist=excludevals,
            maskname=f"{maskname} exclude",
            tolerance=tolerance,
        )
        internalexcludemask = theexcludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalexcludemask) == numspatiallocs:
            raise ValueError(
                f"ERROR: the {maskname} exclude mask does not leave any voxels - exiting"
            )

    if extramask is not None:
        LGR.info(f"reading {maskname} extra mask")
        internalextramask = readamask(
            extramask,
            datahdr,
            numspatiallocs,
            istext=istext,
            valslist=None,
            thresh=extramaskthresh,
            maskname=f"{maskname} extra",
            tolerance=tolerance,
        )

    if (internalincludemask is not None) and (internalexcludemask is not None):
        if tide_stats.getmasksize(internalincludemask * (1 - internalexcludemask)) == 0:
            raise ValueError(
                f"ERROR: the {maskname} include and exclude masks do not leave any voxels between them - exiting"
            )
        if internalextramask is not None:
            if (
                tide_stats.getmasksize(
                    internalincludemask * (1 - internalexcludemask) * internalextramask
                )
                == 0
            ):
                raise ValueError(
                    f"ERROR: the {maskname} include, exclude, and extra masks do not leave any voxels between them - exiting"
                )

    return internalincludemask, internalexcludemask, internalextramask
