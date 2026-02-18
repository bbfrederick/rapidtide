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

import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

import rapidtide.miscmath as tide_math

LGR = logging.getLogger("GENERAL")


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
    pca_class=PCA,
) -> Tuple[NDArray, NDArray]:
    themask = np.ones_like(indata[:, 0])
    if includemask is not None:
        themask = themask * includemask
    if excludemask is not None:
        themask = themask * (1 - excludemask)

    globalmean = (indata[0, :]).astype(rt_floattype)
    thesize = np.shape(themask)
    numvoxelsused = int(np.sum(np.where(themask > 0.0, 1, 0)))
    selectedvoxels = indata[np.where(themask > 0.0), :][0]
    if debug:
        print(f"getregionsignal: {selectedvoxels.shape=}")
    LGR.info(f"constructing global mean signal using {signalgenmethod}")
    if signalgenmethod == "sum":
        globalmean = np.mean(selectedvoxels, axis=0)
        globalmean -= np.mean(globalmean)
    elif signalgenmethod == "meanscale":
        themean = np.mean(indata, axis=1)
        for vox in range(0, thesize[0]):
            if themask[vox] > 0.0 and themean[vox] != 0.0:
                globalmean += indata[vox, :] / themean[vox] - 1.0
    elif signalgenmethod == "pca":
        themean = np.mean(indata, axis=1)
        thevar = np.var(indata, axis=1)
        scaledvoxels = np.zeros_like(selectedvoxels)
        for vox in range(0, selectedvoxels.shape[0]):
            scaledvoxels[vox, :] = selectedvoxels[vox, :] - themean[vox]
            if thevar[vox] > 0.0:
                scaledvoxels[vox, :] = selectedvoxels[vox, :] / thevar[vox]
        try:
            thefit = pca_class(n_components=pcacomponents).fit(scaledvoxels)
        except ValueError:
            if pcacomponents == "mle":
                LGR.warning("mle estimation failed - falling back to pcacomponents=0.8")
                thefit = pca_class(n_components=0.8).fit(scaledvoxels)
            else:
                raise ValueError("unhandled math exception in PCA refinement - exiting")
        varex = 100.0 * np.cumsum(thefit.explained_variance_ratio_)[len(thefit.components_) - 1]
        thetransform = thefit.transform(scaledvoxels)
        cleanedvoxels = thefit.inverse_transform(thetransform) * thevar[:, None]
        globalmean = np.mean(cleanedvoxels, axis=0)
        globalmean -= np.mean(globalmean)
        LGR.info(
            f"Using {len(thefit.components_)} component(s), accounting for {varex:.2f}% of the variance"
        )
    elif signalgenmethod == "random":
        globalmean = np.random.standard_normal(size=len(globalmean))
    else:
        raise ValueError(f"illegal signal generation method: {signalgenmethod}")
    LGR.info(f"used {numvoxelsused} voxels to calculate {signame} signal")
    if filter is not None:
        globalmean = filter.apply(Fs, globalmean)
    return tide_math.stdnormalize(globalmean), themask
