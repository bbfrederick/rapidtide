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

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
from rapidtide.core.masks.region_signal import getregionsignal


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
    getregionsignal_fn=getregionsignal,
    writebidstsv_fn=tide_io.writebidstsv,
) -> Tuple[NDArray, NDArray]:
    thetimecourse, themask = getregionsignal_fn(
        fmridata,
        filter=filter,
        Fs=fmrifreq,
        includemask=includemask,
        excludemask=excludemask,
        signalgenmethod=signalgenmethod,
        pcacomponents=pcacomponents,
        signame=tcdesc,
        rt_floattype=rt_floattype,
        debug=debug,
    )
    writebidstsv_fn(
        f"{outputname}_desc-{filedesc}_timeseries",
        thetimecourse,
        fmrifreq,
        columns=[f"{tcname}{suffix}"],
        extraheaderinfo={
            "Description": "Regional timecourse averages",
        },
        append=(not initfile),
    )
    return thetimecourse, themask
