#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from rapidtide.filter import NoncausalFilter
from rapidtide.helper_classes import FrequencyTracker
from rapidtide.io import writevec
from rapidtide.util import valtoindex


def spectralfilterprops(thefilter, debug=False):
    lowerstop, lowerpass, upperpass, upperstop = thefilter["filter"].getfreqs()
    lowerstopindex = valtoindex(thefilter["frequencies"], lowerstop)
    lowerpassindex = valtoindex(thefilter["frequencies"], lowerpass)
    upperpassindex = valtoindex(thefilter["frequencies"], upperpass)
    upperstopindex = np.min(
        [valtoindex(thefilter["frequencies"], upperstop), len(thefilter["frequencies"]) - 1]
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


def makewaves(sampletime=0.50, tclengthinsecs=300.0, display=False):
    tclen = int(tclengthinsecs // sampletime)
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

    # construct some test waveforms
    testwaves = []
    testwaves.append(
        {
            "name": "descending chirp",
            "timeaxis": 1.0 * timeaxis,
            "waveform": sp.signal.chirp(
                timeaxis, f0=0.3, f1=0.1, t1=timeaxis[-1], method="linear"
            ),
        }
    )
    if display:
        plt.figure()
        plt.plot(testwaves[-1]["timeaxis"], testwaves[-1]["waveform"])
        plt.legend([testwaves[-1]["name"]])
        plt.show()

    testwaves.append(
        {"name": "sinusoidal modulated", "timeaxis": 1.0 * timeaxis, "waveform": np.cos(timeaxis),}
    )
    if display:
        plt.figure()
        plt.plot(testwaves[-1]["timeaxis"], testwaves[-1]["waveform"])
        plt.legend([testwaves[-1]["name"]])
        plt.show()

    scratch = np.ones(len(timeaxis), dtype=float)
    freqs = [0.1, 0.12, 0.15, 0.2]
    seglen = int(len(scratch) // len(freqs))
    print("seglen:", seglen)
    for i in range(len(freqs)):
        scratch[i * seglen : i * seglen + seglen] = sampletime * 2.0 * np.pi * freqs[i]
    print(scratch)
    plt.figure()
    plt.plot(np.cumsum(scratch))
    plt.show()
    testwaves.append(
        {
            "name": "stepped freq",
            "timeaxis": 1.0 * timeaxis,
            "waveform": np.cos(np.cumsum(scratch)),
        }
    )
    if display:
        plt.figure()
        plt.plot(testwaves[-1]["timeaxis"], testwaves[-1]["waveform"])
        plt.legend([testwaves[-1]["name"]])
        plt.show()
    # writevec(testwaves[-1]['waveform'], 'stepped.txt')
    return testwaves


def eval_filterprops(sampletime=0.50, tclengthinsecs=300.0, numruns=100, display=False):
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

    allfilters = []

    # construct all the filters
    for filtertype in ["lfo", "resp", "cardiac"]:
        testfilter = NoncausalFilter(filtertype=filtertype)
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            allfilters.append(
                {
                    "name": filtertype + " brickwall",
                    "filter": NoncausalFilter(filtertype=filtertype, transferfunc="brickwall"),
                }
            )
            allfilters.append(
                {
                    "name": filtertype + " trapezoidal",
                    "filter": NoncausalFilter(filtertype=filtertype, transferfunc="trapezoidal"),
                }
            )
            allfilters.append(
                {
                    "name": filtertype + " gaussian",
                    "filter": NoncausalFilter(filtertype=filtertype, transferfunc="gaussian"),
                }
            )

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
    if display:
        legend = []
        plt.figure()
        plt.ylim([-1.1, 1.1 * len(allfilters)])
        offset = 0.0
        for thefilter in allfilters:
            plt.plot(thefilter["frequencies"], thefilter["transferfunc"] + offset)
            legend.append(thefilter["name"])
            offset += 1.1
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
        assert response["passbandripple"] < 0.45
        assert response["lowerstopmax"] < 1e4
        assert response["lowerstopmean"] < 1e4
        assert response["upperstopmax"] < 1e4
        assert response["upperstopmean"] < 1e4

    scratch = timeaxis * 0.0
    scratch[int(tclen / 5) : int(2 * tclen / 5)] = 1.0
    scratch[int(3 * tclen / 5) : int(4 * tclen / 5)] = 1.0
    testwaves.append(
        {"name": "block regressor", "timeaxis": 1.0 * timeaxis, "waveform": 1.0 * scratch,}
    )

    # show the end effects waveforms
    if display:
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
                offset += 1.1
            # plt.plot(thewave['timeaxis'], thewave['waveform'] + offset)
            # legend.append(thewave['name'])
            # offset += 2.2
            plt.legend(legend)
            plt.show()
    assert True


def test_filterprops(display=False):
    eval_filterprops(sampletime=0.72, tclengthinsecs=300.0, numruns=100, display=display)
    eval_filterprops(sampletime=2.0, tclengthinsecs=300.0, numruns=100, display=display)
    eval_filterprops(sampletime=0.1, tclengthinsecs=1000.0, numruns=10, display=display)


def main():
    makewaves(display=True)


if __name__ == "__main__":
    main()
