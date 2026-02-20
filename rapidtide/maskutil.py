#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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

import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
from nilearn import masking
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

import rapidtide.io as tide_io
from rapidtide.core.io.mask_io import (
    saveregionaltimeseries as _saveregionaltimeseries_impl,
)
from rapidtide.core.masks.mask_ops import getmaskset as _getmaskset_impl
from rapidtide.core.masks.mask_ops import makeepimask as _makeepimask_impl
from rapidtide.core.masks.mask_ops import maketmask as _maketmask_impl
from rapidtide.core.masks.mask_ops import readamask as _readamask_impl
from rapidtide.core.masks.mask_ops import resampmask as _resampmask_impl
from rapidtide.core.masks.region_signal import getregionsignal as _getregionsignal_impl

LGR = logging.getLogger("GENERAL")


def resampmask(themask: ArrayLike, thetargetres: float) -> NDArray:
    return _resampmask_impl(themask, thetargetres)


def makeepimask(nim: Any) -> Any:
    # Keep local masking symbol for compatibility with existing monkeypatch targets.
    return masking.compute_epi_mask(nim)


def maketmask(filename: str, timeaxis: ArrayLike, maskvector: NDArray, debug: bool = False) -> NDArray:
    return _maketmask_impl(filename, timeaxis, maskvector, debug=debug)


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
    return _readamask_impl(
        maskfilename,
        nim_hdr,
        xsize,
        istext=istext,
        valslist=valslist,
        thresh=thresh,
        maskname=maskname,
        tolerance=tolerance,
        debug=debug,
    )


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
) -> Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
    # Pass local readamask symbol so monkeypatching rapidtide.maskutil.readamask still works.
    return _getmaskset_impl(
        maskname,
        includename,
        includevals,
        excludename,
        excludevals,
        datahdr,
        numspatiallocs,
        extramask=extramask,
        extramaskthresh=extramaskthresh,
        istext=istext,
        tolerance=tolerance,
        debug=debug,
        readamask_fn=readamask,
    )


def getregionsignal(
    indata: NDArray,
    filter: Optional[Any] = None,
    Fs: float = 1.0,
    includemask: Optional[NDArray] = None,
    excludemask: Optional[NDArray] = None,
    signalgenmethod: str = "sum",
    pcacomponents: Union[float, str] = 0.8,
    signame: str = "global mean",
    rt_floattype: type = np.float64,
    debug: bool = False,
) -> Tuple[NDArray, NDArray]:
    # Pass local PCA symbol so monkeypatching rapidtide.maskutil.PCA still works.
    return _getregionsignal_impl(
        indata,
        filter=filter,
        Fs=Fs,
        includemask=includemask,
        excludemask=excludemask,
        signalgenmethod=signalgenmethod,
        pcacomponents=pcacomponents,
        signame=signame,
        rt_floattype=rt_floattype,
        debug=debug,
        pca_class=PCA,
    )


def saveregionaltimeseries(
    tcdesc: str,
    tcname: str,
    fmridata: NDArray,
    includemask: NDArray,
    fmrifreq: float,
    outputname: str,
    filter: Optional[Any] = None,
    initfile: bool = False,
    excludemask: Optional[NDArray] = None,
    filedesc: str = "regional",
    suffix: str = "",
    signalgenmethod: str = "sum",
    pcacomponents: Union[float, str] = 0.8,
    rt_floattype: type = np.float64,
    debug: bool = False,
) -> Tuple[NDArray, NDArray]:
    # Pass local symbols so monkeypatching rapidtide.maskutil.getregionsignal/tide_io.writebidstsv still works.
    return _saveregionaltimeseries_impl(
        tcdesc,
        tcname,
        fmridata,
        includemask,
        fmrifreq,
        outputname,
        filter=filter,
        initfile=initfile,
        excludemask=excludemask,
        filedesc=filedesc,
        suffix=suffix,
        signalgenmethod=signalgenmethod,
        pcacomponents=pcacomponents,
        rt_floattype=rt_floattype,
        debug=debug,
        getregionsignal_fn=getregionsignal,
        writebidstsv_fn=tide_io.writebidstsv,
    )


__all__ = [
    "resampmask",
    "makeepimask",
    "maketmask",
    "readamask",
    "getmaskset",
    "getregionsignal",
    "saveregionaltimeseries",
]
