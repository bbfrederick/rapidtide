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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from rapidtide.filter import NoncausalFilter


def spectralfilterprops(thefilter, debug=False):
    lowerstop, lowerpass, upperpass, upperstop = thefilter["filter"].getfreqs()
    freqspace = thefilter["frequencies"][1] - thefilter["frequencies"][0]
    lowerstopindex = int(np.floor(lowerstop / freqspace))
    lowerpassindex = int(np.ceil(lowerpass / freqspace))
    upperpassindex = int(np.floor(upperpass / freqspace))
    upperstopindex = int(
        np.min([np.ceil(upperstop / freqspace), len(thefilter["frequencies"]) - 1])
    )
    if debug:
        print("target freqs:", lowerstop, lowerpass, upperpass, upperstop)
        print(
            "actual freqs:",
            thefilter["frequencies"][lowerstopindex],
            thefilter["frequencies"][lowerpassindex],
            thefilter["frequencies"][upperpassindex],
            thefilter["frequencies"][upperstopindex],
        )
    response = {}

    passbandmean = np.mean(thefilter["transferfunc"][lowerpassindex:upperpassindex])
    passbandmax = np.max(thefilter["transferfunc"][lowerpassindex:upperpassindex])
    passbandmin = np.min(thefilter["transferfunc"][lowerpassindex:upperpassindex])

    response["passbandripple"] = (passbandmax - passbandmin) / passbandmean

    if lowerstopindex > 2:
        response["lowerstopmean"] = (
            np.mean(thefilter["transferfunc"][0:lowerstopindex]) / passbandmean
        )
        response["lowerstopmax"] = (
            np.max(np.abs(thefilter["transferfunc"][0:lowerstopindex])) / passbandmean
        )
    else:
        response["lowerstopmean"] = 0.0
        response["lowerstopmax"] = 0.0

    if len(thefilter["transferfunc"]) - upperstopindex > 2:
        response["upperstopmean"] = (
            np.mean(thefilter["transferfunc"][upperstopindex:-1]) / passbandmean
        )
        response["upperstopmax"] = (
            np.max(np.abs(thefilter["transferfunc"][upperstopindex:-1])) / passbandmean
        )
    else:
        response["upperstopmean"] = 0.0
        response["upperstopmax"] = 0.0
    return response


def eval_filterprops(sampletime=0.72, tclengthinsecs=300.0, numruns=100, displayplots=False):
    np.random.seed(12345)
    tclen = int(tclengthinsecs // sampletime)
    print("Testing transfer function:")
    lowestfreq = 1.0 / (sampletime * tclen)
    nyquist = 0.5 / sampletime
    print(
        "    sampletime=",
        sampletime,
        ", timecourse length=",
        tclengthinsecs,
        "s,  possible frequency range:",
        lowestfreq,
        nyquist,
    )
    timeaxis = np.arange(0.0, 1.0 * tclen) * sampletime

    overall = np.random.normal(size=tclen)
    nperseg = np.min([tclen, 256])
    f, dummy = sp.signal.welch(overall, fs=1.0 / sampletime, nperseg=nperseg)

    transferfunclist = ["brickwall", "trapezoidal", "butterworth"]

    allfilters = []

    # construct all the physiological filters
    for filtertype in ["lfo", "resp", "cardiac", "hrv_vlf", "hrv_lf", "hrv_hf", "hrv_vhf"]:
        testfilter = NoncausalFilter(filtertype=filtertype)
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            for transferfunc in transferfunclist:
                allfilters.append(
                    {
                        "name": filtertype + " " + transferfunc,
                        "filter": NoncausalFilter(
                            filtertype=filtertype,
                            transferfunc=transferfunc,
                            debug=False,
                        ),
                    }
                )

    """'# make the lowpass filters
    for transferfunc in transferfunclist:
        testfilter = NoncausalFilter(
                        filtertype='arb',
                        transferfunc=transferfunc,
                        initlowerstop=0.0, initlowerpass=0.0,
                        initupperpass=0.1, initupperstop=0.11)
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            allfilters.append(
                {
                    'name': '0.1Hz LP ' + transferfunc,
                    'filter': NoncausalFilter(
                                filtertype='arb',
                                transferfunc=transferfunc,
                                initlowerstop=0.0, initlowerpass=0.0,
                                initupperpass=0.1, initupperstop=0.11, debug=False)
                })

    # make the highpass filters
    for transferfunc in transferfunclist:
        testfilter = NoncausalFilter(
                        filtertype='arb',
                        transferfunc=transferfunc,
                        initlowerstop=0.09, initlowerpass=0.1,
                        initupperpass=-1.0, initupperstop=-1.0)
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            allfilters.append(
                {
                    'name': '0.1Hz HP ' + transferfunc,
                    'filter': NoncausalFilter(
                                filtertype='arb',
                                transferfunc=transferfunc,
                                initlowerstop=0.09, initlowerpass=0.1,
                                initupperpass=-1.0, initupperstop=-1.0, debug=False)
                })"""

    # calculate the transfer functions for the filters
    for index in range(0, len(allfilters)):
        psd_raw = 0.0 * dummy
        psd_filt = 0.0 * dummy
        for i in range(0, numruns):
            inputsig = np.random.normal(size=tclen)
            outputsig = allfilters[index]["filter"].apply(1.0 / sampletime, inputsig)
            f, raw = sp.signal.welch(inputsig, fs=1.0 / sampletime, nperseg=nperseg)
            f, filt = sp.signal.welch(outputsig, fs=1.0 / sampletime, nperseg=nperseg)
            psd_raw += raw
            psd_filt += filt
        allfilters[index]["frequencies"] = f
        allfilters[index]["transferfunc"] = psd_filt / psd_raw

    # show transfer functions
    if displayplots:
        legend = []
        plt.figure()
        plt.ylim([-1.0, 1.0 * len(allfilters)])
        offset = 0.0
        for thefilter in allfilters:
            plt.plot(thefilter["frequencies"], thefilter["transferfunc"] + offset)
            legend.append(thefilter["name"])
            offset += 1.0
        plt.legend(legend)
        plt.show()

    # test transfer function responses
    for thefilter in allfilters:
        response = spectralfilterprops(thefilter)
        print("    Evaluating", thefilter["name"], "transfer function")
        print("\tpassbandripple:", response["passbandripple"])
        print("\tlowerstopmax:", response["lowerstopmax"])
        print("\tlowerstopmean:", response["lowerstopmean"])
        print("\tupperstopmax:", response["upperstopmax"])
        print("\tupperstopmean:", response["upperstopmean"])
        # assert response['passbandripple'] < 0.45
        assert response["lowerstopmax"] < 1e4
        assert response["lowerstopmean"] < 1e4
        assert response["upperstopmax"] < 1e4
        assert response["upperstopmean"] < 1e4

    # construct some test waveforms for end effects
    testwaves = []
    testwaves.append(
        {
            "name": "constant high",
            "timeaxis": 1.0 * timeaxis,
            "waveform": np.ones((tclen), dtype="float"),
        }
    )
    testwaves.append(
        {
            "name": "white noise",
            "timeaxis": 1.0 * timeaxis,
            "waveform": 0.3 * np.random.normal(size=tclen),
        }
    )

    scratch = timeaxis * 0.0
    scratch[int(tclen / 5) : int(2 * tclen / 5)] = 1.0
    scratch[int(3 * tclen / 5) : int(4 * tclen / 5)] = 1.0
    testwaves.append(
        {
            "name": "block regressor",
            "timeaxis": 1.0 * timeaxis,
            "waveform": 1.0 * scratch,
        }
    )

    # show the end effects waveforms
    if displayplots:
        legend = []
        plt.figure()
        plt.ylim([-2.2, 2.2 * len(testwaves)])
        offset = 0.0
        for thewave in testwaves:
            for thefilter in allfilters:
                plt.plot(
                    thewave["timeaxis"],
                    offset + thefilter["filter"].apply(1.0 / sampletime, thewave["waveform"]),
                )
                legend.append(thewave["name"] + ": " + thefilter["name"])
                offset += 1.0
            # plt.plot(thewave['timeaxis'], thewave['waveform'] + offset)
            # legend.append(thewave['name'])
            # offset += 2.2
            plt.legend(legend)
            plt.show()


def test_filterprops(displayplots=False):
    eval_filterprops(sampletime=0.72, tclengthinsecs=300.0, numruns=100, displayplots=displayplots)
    eval_filterprops(sampletime=2.0, tclengthinsecs=300.0, numruns=100, displayplots=displayplots)
    eval_filterprops(sampletime=0.1, tclengthinsecs=500.0, numruns=10, displayplots=displayplots)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_filterprops(displayplots=True)
