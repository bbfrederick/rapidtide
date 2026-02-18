#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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

import bisect
import logging
from typing import Any, Optional, Tuple

import numpy as np
from nilearn import masking
from numpy.typing import ArrayLike, NDArray

import rapidtide.core.signal.stats as tide_stats
import rapidtide.io as tide_io

LGR = logging.getLogger("GENERAL")


def resampmask(themask: ArrayLike, thetargetres: float) -> NDArray:
    # TODO: implement true resampling, currently passthrough behavior is preserved.
    return themask


def makeepimask(nim: Any) -> Any:
    return masking.compute_epi_mask(nim)


def maketmask(filename: str, timeaxis: ArrayLike, maskvector: NDArray, debug: bool = False) -> NDArray:
    inputdata = tide_io.readvecs(filename)
    theshape = np.shape(inputdata)
    if theshape[0] == 1:
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
    maskfilename: str,
    nim_hdr: Any,
    xsize: int,
    istext: bool = False,
    valslist: Optional[list] = None,
    thresh: Optional[float] = None,
    maskname: str = "the",
    tolerance: float = 1.0e-3,
    debug: bool = False,
) -> NDArray:
    LGR.debug(f"readamask called with filename: {maskfilename} vals: {valslist}")
    if debug:
        print("getmaskset:")
        print(f"{maskname=}")
        print(f"\tincludefilename={maskfilename}")
        print(f"\tincludevals={valslist}")
        print(f"\t{istext=}")
        print(f"\t{tolerance=}")
    if istext:
        maskarray = tide_io.readvecs(maskfilename).astype("uint16")
        theshape = np.shape(maskarray)
        theincludexsize = theshape[0]
        if not theincludexsize == xsize:
            raise ValueError(f"Dimensions of {maskname} mask do not match the input data - exiting")
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
    maskname: str,
    includename: Optional[str],
    includevals: Optional[list],
    excludename: Optional[str],
    excludevals: Optional[list],
    datahdr: Any,
    numspatiallocs: int,
    extramask: Optional[str] = None,
    extramaskthresh: float = 0.1,
    istext: bool = False,
    tolerance: float = 1.0e-3,
    debug: bool = False,
    readamask_fn=readamask,
) -> Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
    internalincludemask = None
    internalexcludemask = None
    internalextramask = None

    if debug:
        print("getmaskset:")
        print(f"{maskname=}")
        print(f"\t{includename=}")
        print(f"\t{includevals=}")
        print(f"\t{excludename=}")
        print(f"\t{excludevals=}")
        print(f"\t{istext=}")
        print(f"\t{tolerance=}")
        print(f"\t{extramask=}")
        print(f"\t{extramaskthresh=}")

    if includename is not None:
        LGR.info(f"constructing {maskname} include mask")
        theincludemask = readamask_fn(
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
            raise ValueError(f"ERROR: there are no voxels in the {maskname} include mask - exiting")

    if excludename is not None:
        LGR.info(f"constructing {maskname} exclude mask")
        theexcludemask = readamask_fn(
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
        internalextramask = readamask_fn(
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
