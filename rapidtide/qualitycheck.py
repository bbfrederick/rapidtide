#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2024-2024 Blaise Frederick
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
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion

import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
import rapidtide.stats as tide_stats
from rapidtide.RapidtideDataset import RapidtideDataset


def prepmask(inputmask):
    erodedmask = binary_erosion(inputmask)
    return erodedmask


def checklag(
    themap,
    themask,
    histlen=101,
    minlag=-5.0,
    maxlag=10.0,
    maxgrad=3.0,
    savehist=False,
    debug=False,
):
    lagmetrics = {}

    gethistmetrics(
        themap,
        themask.data,
        lagmetrics,
        thehistlabel="lag time histogram",
        histlen=histlen,
        rangemin=minlag,
        rangemax=maxlag,
        nozero=False,
        savehist=savehist,
        debug=debug,
    )

    theerodedmask = prepmask(themask.data)
    thegradient = np.gradient(themap.data)
    thegradientamp = np.sqrt(
        np.square(thegradient[0] / themap.xsize)
        + np.square(thegradient[1] / themap.ysize)
        + np.square(thegradient[2] / themap.zsize)
    )
    maskedgradient = theerodedmask * thegradientamp
    if debug:
        tide_io.savetonifti(thegradientamp, themap.header, "laggradient")
        tide_io.savetonifti(maskedgradient, themap.header, "maskedlaggradient")

    maskedgradientdata = np.ravel(thegradientamp[np.where(theerodedmask > 0.0)])
    (
        lagmetrics["gradpct02"],
        lagmetrics["gradpct25"],
        lagmetrics["gradpct50"],
        lagmetrics["gradpct75"],
        lagmetrics["gradpct98"],
    ) = tide_stats.getfracvals(maskedgradientdata, [0.02, 0.25, 0.5, 0.75, 0.98], debug=debug)
    (
        gradhist,
        lagmetrics["gradpeakheight"],
        lagmetrics["gradpeakloc"],
        lagmetrics["gradpeakwidth"],
        lagmetrics["gradcenterofmass"],
        lagmetrics["gradpeakpercentile"],
    ) = tide_stats.makehistogram(
        maskedgradientdata,
        histlen,
        refine=False,
        therange=(0.0, maxgrad),
        normalize=True,
        ignorefirstpoint=True,
        debug=debug,
    )
    gradhistbincenters = ((gradhist[1][1:] + gradhist[1][0:-1]) / 2.0).tolist()
    gradhistvalues = (gradhist[0][-histlen:]).tolist()
    if savehist:
        lagmetrics["gradhistbincenters"] = gradhistbincenters
        lagmetrics["gradhistvalues"] = gradhistvalues
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("lag gradient magnitude histogram")
        plt.plot(gradhistbincenters, gradhistvalues)
        plt.show()
    return lagmetrics


def gethistmetrics(
    themap,
    themask,
    thedict,
    thehistlabel=None,
    histlen=101,
    rangemin=-1.0,
    rangemax=1.0,
    nozero=False,
    savehist=False,
    debug=False,
):
    thedict["pct02"] = themap.robustmin
    thedict["pct25"] = themap.quartiles[0]
    thedict["pct50"] = themap.quartiles[1]
    thedict["pct75"] = themap.quartiles[2]
    thedict["pct98"] = themap.robustmax

    dataforhist = np.ravel(themap.data[np.where(themask > 0.0)])
    if nozero:
        dataforhist = dataforhist[np.where(dataforhist != 0.0)]
    (
        thehist,
        thedict["peakheight"],
        thedict["peakloc"],
        thedict["peakwidth"],
        thedict["centerofmass"],
        thedict["peakpercentile"],
    ) = tide_stats.makehistogram(
        dataforhist,
        histlen,
        refine=False,
        therange=(rangemin, rangemax),
        normalize=True,
        ignorefirstpoint=True,
        debug=debug,
    )
    histbincenters = ((thehist[1][1:] + thehist[1][0:-1]) / 2.0).tolist()
    histvalues = (thehist[0][-histlen:]).tolist()
    if savehist:
        thedict["histbincenters"] = histbincenters
        thedict["histvalues"] = histvalues
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(thehistlabel)
        plt.plot(histbincenters, histvalues)
        plt.show()


def checkstrength(
    themap, themask, histlen=101, minstrength=0.0, maxstrength=1.0, savehist=False, debug=False
):
    strengthmetrics = {}

    gethistmetrics(
        themap,
        themask.data,
        strengthmetrics,
        thehistlabel="similarity metric histogram",
        histlen=histlen,
        rangemin=minstrength,
        rangemax=maxstrength,
        nozero=False,
        savehist=savehist,
        debug=debug,
    )
    return strengthmetrics


def checkMTT(themap, themask, histlen=101, minsMTT=0.0, maxMTT=10.0, savehist=False, debug=False):
    MTTmetrics = {}

    gethistmetrics(
        themap,
        themask.data,
        MTTmetrics,
        thehistlabel="MTT histogram",
        histlen=histlen,
        rangemin=minsMTT,
        rangemax=maxMTT,
        nozero=True,
        savehist=savehist,
        debug=debug,
    )
    return MTTmetrics


def checkregressors(theregressors, numpasses, filterlimits, debug=False):
    regressormetrics = {}
    firstregressor = theregressors["pass1"]
    lastregressor = theregressors[f"pass{numpasses}"]
    lowerlimindex = np.argmax(firstregressor.specaxis >= filterlimits[0])
    upperlimindex = np.argmin(firstregressor.specaxis <= filterlimits[1]) + 1
    if debug:
        print(f"{filterlimits=}")
        print(f"{lowerlimindex=}, {upperlimindex=}")
        print(firstregressor.specaxis)
        print(firstregressor.specdata[lowerlimindex: upperlimindex])
    for label, regressor in [["first", firstregressor],["last", lastregressor]]:
        regressormetrics[f"{label}_kurtosis"] = regressor.kurtosis
        regressormetrics[f"{label}_kurtosis_z"] = regressor.kurtosis_z
        regressormetrics[f"{label}_kurtosis_p"] = regressor.kurtosis_p
        regressormetrics[f"{label}_spectralflatness"] = tide_filt.spectralflatness(regressor.specdata[lowerlimindex: upperlimindex])
    return regressormetrics


def qualitycheck(
    datafileroot,
    graymaskspec=None,
    whitemaskspec=None,
    anatname=None,
    geommaskname=None,
    userise=False,
    usecorrout=False,
    useatlas=False,
    forcetr=False,
    forceoffset=False,
    offsettime=0.0,
    verbose=False,
    debug=False,
):
    # read in the dataset
    thedataset = RapidtideDataset(
        "main",
        datafileroot + "_",
        anatname=anatname,
        geommaskname=geommaskname,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        verbose=verbose,
        init_LUT=False,
    )

    outputdict = {}

    # put in some basic information
    outputdict["passes"] = thedataset.numberofpasses
    outputdict["filterlimits"] = thedataset.regressorfilterlimits

    themask = thedataset.overlays["lagmask"]

    thelags = thedataset.overlays["lagtimes"]
    thelags.setFuncMask(themask.data)
    thelags.updateStats()
    if debug:
        thelags.summarize()

    theMTTs = thedataset.overlays["MTT"]
    theMTTs.setFuncMask(themask.data)
    theMTTs.updateStats()
    if debug:
        theMTTs.summarize()

    thestrengths = thedataset.overlays["lagstrengths"]
    thestrengths.setFuncMask(themask.data)
    thestrengths.updateStats()
    if debug:
        thestrengths.summarize()

    theregressors = thedataset.regressors

    outputdict["lagmetrics"] = checklag(thelags, themask, debug=debug)
    outputdict["strengthmetrics"] = checkstrength(thestrengths, themask, debug=debug)
    outputdict["MTTmetrics"] = checkMTT(theMTTs, themask, debug=debug)
    outputdict["regressormetrics"] = checkregressors(theregressors, outputdict["passes"], outputdict["filterlimits"], debug=debug)

    return outputdict
