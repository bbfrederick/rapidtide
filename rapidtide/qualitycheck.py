#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2024-2025 Blaise Frederick
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

import rapidtide.filter as tide_filt
import rapidtide.stats as tide_stats
from rapidtide.RapidtideDataset import RapidtideDataset


def prepmask(inputmask):
    erodedmask = binary_erosion(inputmask)
    return erodedmask


def getmasksize(themask):
    return len(np.ravel(themask[np.where(themask > 0)]))


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
        print(firstregressor.specdata[lowerlimindex:upperlimindex])
    for label, regressor in [["first", firstregressor], ["last", lastregressor]]:
        regressormetrics[f"{label}_kurtosis"] = regressor.kurtosis
        regressormetrics[f"{label}_kurtosis_z"] = regressor.kurtosis_z
        regressormetrics[f"{label}_kurtosis_p"] = regressor.kurtosis_p
        regressormetrics[f"{label}_skewness"] = regressor.skewness
        regressormetrics[f"{label}_skewness_z"] = regressor.skewness_z
        regressormetrics[f"{label}_skewness_p"] = regressor.skewness_p
        regressormetrics[f"{label}_spectralflatness"] = tide_filt.spectralflatness(
            regressor.specdata[lowerlimindex:upperlimindex]
        )
    return regressormetrics


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
    ignorefirstpoint=False,
    debug=False,
):
    # mask and flatten the data
    maskisempty = False
    if len(np.where(themask > 0)) == 0:
        maskisempty = False
    if debug:
        print("num-nonzero in mask", len(np.where(themask > 0)[0]))
    if not maskisempty:
        dataforhist = np.ravel(themap[np.where(themask > 0.0)])
        if nozero:
            dataforhist = dataforhist[np.where(dataforhist != 0.0)]
            if len(dataforhist) == 0:
                maskisempty = True
            if debug:
                print("num-nonzero in dataforhist", len(dataforhist))
    else:
        maskisempty = True

    if not maskisempty:
        # get percentiles
        (
            thedict["pct02"],
            thedict["pct25"],
            thedict["pct50"],
            thedict["pct75"],
            thedict["pct98"],
        ) = tide_stats.getfracvals(dataforhist, [0.02, 0.25, 0.5, 0.75, 0.98], debug=debug)
        thedict["voxelsincluded"] = len(dataforhist)
        thedict["q1width"] = thedict["pct25"] - thedict["pct02"]
        thedict["q2width"] = thedict["pct50"] - thedict["pct25"]
        thedict["q3width"] = thedict["pct75"] - thedict["pct50"]
        thedict["q4width"] = thedict["pct98"] - thedict["pct75"]
        thedict["mid50width"] = thedict["pct75"] - thedict["pct25"]

        # get moments
        (
            thedict["kurtosis"],
            thedict["kurtosis_z"],
            thedict["kurtosis_p"],
        ) = tide_stats.kurtosisstats(dataforhist)
        (
            thedict["skewness"],
            thedict["skewness_z"],
            thedict["skewness_p"],
        ) = tide_stats.skewnessstats(dataforhist)
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
            ignorefirstpoint=ignorefirstpoint,
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
    else:
        thedict["voxelsincluded"] = 0
        taglist = ["pct02", "pct25", "pct50", "pct75", "pct98"]
        taglist += ["q1width", "q2width", "q3width", "q4width", "mid50width"]
        taglist += ["kurtosis", "kurtosis_z", "kurtosis_p", "skewness", "skewness_z", "skewness_p"]
        taglist += ["peakheight", "peakloc", "peakwidth", "centerofmass", "peakpercentile"]
        if savehist:
            taglist += ["histbincenters", "histvalues"]
        for tag in taglist:
            thedict[tag] = None


def checkmap(
    themap,
    themask,
    histlen=101,
    rangemin=0.0,
    rangemax=1.0,
    histlabel="similarity metric histogram",
    ignorefirstpoint=False,
    savehist=False,
    debug=False,
):
    themetrics = {}

    gethistmetrics(
        themap,
        themask,
        themetrics,
        thehistlabel=histlabel,
        histlen=histlen,
        rangemin=rangemin,
        rangemax=rangemax,
        nozero=False,
        savehist=savehist,
        ignorefirstpoint=ignorefirstpoint,
        debug=debug,
    )
    return themetrics


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
        graymaskspec=graymaskspec,
        whitemaskspec=whitemaskspec,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        verbose=verbose,
        init_LUT=False,
    )

    if debug:
        print(f"qualitycheck started on {datafileroot}")
    outputdict = {}
    if graymaskspec is not None:
        dograyonly = True
        thegraymask = (thedataset.overlays["graymask"]).data
    else:
        dograyonly = False
    if whitemaskspec is not None:
        dowhiteonly = True
        thewhitemask = (thedataset.overlays["whitemask"]).data
    else:
        dowhiteonly = False

    # put in some basic information
    outputdict["passes"] = thedataset.numberofpasses
    outputdict["filterlimits"] = thedataset.regressorfilterlimits
    outputdict["simcalclimits"] = thedataset.regressorsimcalclimits

    # process the masks
    outputdict["mask"] = {}
    thelagmask = (thedataset.overlays["lagmask"]).data
    theerodedmask = prepmask(thelagmask)
    outputdict["mask"]["lagmaskvoxels"] = len(np.ravel(thelagmask[np.where(thelagmask > 0)]))
    for maskname in [
        "refinemask",
        "meanmask",
        "preselectmask",
        "p_lt_0p050_mask",
        "p_lt_0p010_mask",
        "p_lt_0p005_mask",
        "p_lt_0p001_mask",
        "desc-plt0p001_mask",
    ]:
        try:
            thismask = (thedataset.overlays[maskname]).data
        except KeyError:
            print(f"{maskname} not found in dataset")
        else:
            outname = maskname.replace("_mask", "").replace("mask", "")
            outputdict["mask"][f"{outname}relsize"] = getmasksize(thismask) / (
                1.0 * outputdict["mask"]["lagmaskvoxels"]
            )

    # process the regressors
    theregressors = thedataset.regressors
    outputdict["regressor"] = checkregressors(
        theregressors, outputdict["passes"], outputdict["filterlimits"], debug=debug
    )

    # process the lag map
    thelags = thedataset.overlays["lagtimes"]
    thelags.setFuncMask(thelagmask)
    thelags.updateStats()
    if debug:
        thelags.summarize()
    outputdict["lag"] = checkmap(
        thelags.data,
        thelagmask,
        rangemin=-5.0,
        rangemax=10.0,
        histlabel="lag histogram",
        debug=debug,
    )

    # get the gradient of the lag map
    thegradient = np.gradient(thelags.data)
    thegradientamp = np.sqrt(
        np.square(thegradient[0] / thelags.xsize)
        + np.square(thegradient[1] / thelags.ysize)
        + np.square(thegradient[2] / thelags.zsize)
    )
    outputdict["laggrad"] = checkmap(
        thegradientamp,
        theerodedmask,
        rangemin=0.0,
        rangemax=3.0,
        histlabel="lag gradient amplitude histogram",
        ignorefirstpoint=True,
        debug=debug,
    )

    # process the strength map
    thestrengths = thedataset.overlays["lagstrengths"]
    thestrengths.setFuncMask(thelagmask)
    thestrengths.updateStats()
    if debug:
        thestrengths.summarize()
    outputdict["strength"] = checkmap(
        thestrengths.data,
        thelagmask,
        rangemin=0.0,
        rangemax=1.0,
        histlabel="similarity metric histogram",
        debug=debug,
    )

    # process the MTT map
    theMTTs = thedataset.overlays["MTT"]
    theMTTs.setFuncMask(thelagmask)
    theMTTs.updateStats()
    if debug:
        theMTTs.summarize()
    outputdict["MTT"] = checkmap(
        theMTTs.data,
        thelagmask,
        histlabel="MTT histogram",
        rangemin=0.0,
        rangemax=10.0,
        debug=debug,
    )

    if dograyonly:
        outputdict["grayonly-lag"] = checkmap(
            thelags.data,
            thelagmask * thegraymask,
            rangemin=-5.0,
            rangemax=10.0,
            histlabel="lag histogram - gray only",
            debug=debug,
        )
        outputdict["grayonly-laggrad"] = checkmap(
            thegradientamp,
            theerodedmask * thegraymask,
            rangemin=0.0,
            rangemax=3.0,
            histlabel="lag gradient amplitude histogram - gray only",
            debug=debug,
        )
        outputdict["grayonly-strength"] = checkmap(
            thestrengths.data,
            thelagmask * thegraymask,
            rangemin=0.0,
            rangemax=1.0,
            histlabel="similarity metric histogram - gray only",
            debug=debug,
        )
    if dowhiteonly:
        outputdict["whiteonly-lag"] = checkmap(
            thelags.data,
            thelagmask * thewhitemask,
            rangemin=-5.0,
            rangemax=10.0,
            histlabel="lag histogram - white only",
            debug=debug,
        )
        outputdict["whiteonly-laggrad"] = checkmap(
            thegradientamp,
            theerodedmask * thewhitemask,
            rangemin=0.0,
            rangemax=3.0,
            histlabel="lag gradient amplitude histogram - white only",
            debug=debug,
        )
        outputdict["whiteonly-strength"] = checkmap(
            thestrengths.data,
            thelagmask * thewhitemask,
            rangemin=0.0,
            rangemax=1.0,
            histlabel="similarity metric histogram - white only",
            debug=debug,
        )

    return outputdict
