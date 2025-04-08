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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from rapidtide.filter import NoncausalFilter


def maketestwaves(timeaxis):
    # construct some test waveforms for end effects
    tclen = len(timeaxis)
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

    scratch = timeaxis * 0.0
    scratch[int(tclen / 2) :] = 1.0
    testwaves.append(
        {
            "name": "step regressor",
            "timeaxis": 1.0 * timeaxis,
            "waveform": 1.0 * scratch,
        }
    )
    return testwaves


def eval_padvecprops(
    sampletime=0.72, tclengthinsecs=300.0, numruns=100, displayplots=False, debug=False
):
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
    timeaxis = np.linspace(0.0, sampletime * tclen, num=tclen, endpoint=False)

    overall = np.random.normal(size=tclen)
    nperseg = np.min([tclen, 2048])
    f, dummy = sp.signal.welch(overall, fs=1.0 / sampletime, nperseg=nperseg)

    padtypelist = ["reflect", "zero", "constant", "constant+"]
    transferfunclist = ["brickwall", "trapezoidal", "butterworth"]

    # construct some test waveforms for end effects
    testwaves = maketestwaves(timeaxis)

    for padtype in padtypelist:
        allfilters = []
        # construct all the physiological filters
        for filtertype in ["None", "lfo", "resp", "cardiac", "hrv_lf", "hrv_hf", "hrv_vhf"]:
            testfilter = NoncausalFilter(filtertype=filtertype, padtype=padtype)
            lstest, lptest, uptest, ustest = testfilter.getfreqs()
            if lptest < nyquist:
                for transferfunc in transferfunclist:
                    allfilters.append(
                        {
                            "name": filtertype + " " + transferfunc + " " + padtype,
                            "filter": NoncausalFilter(
                                filtertype=filtertype,
                                transferfunc=transferfunc,
                                padtype=padtype,
                                debug=False,
                            ),
                        }
                    )

        # make the lowpass filters
        for transferfunc in transferfunclist:
            testfilter = NoncausalFilter(
                filtertype="arb",
                transferfunc=transferfunc,
                initlowerstop=0.0,
                initlowerpass=0.0,
                initupperpass=0.1,
                initupperstop=0.11,
            )
            lstest, lptest, uptest, ustest = testfilter.getfreqs()
            if lptest < nyquist:
                allfilters.append(
                    {
                        "name": "0.1Hz LP " + transferfunc,
                        "filter": NoncausalFilter(
                            filtertype="arb",
                            transferfunc=transferfunc,
                            initlowerstop=0.0,
                            initlowerpass=0.0,
                            initupperpass=0.1,
                            initupperstop=0.11,
                            debug=False,
                        ),
                    }
                )

        # make the highpass filters
        for transferfunc in transferfunclist:
            testfilter = NoncausalFilter(
                filtertype="arb",
                transferfunc=transferfunc,
                initlowerstop=0.09,
                initlowerpass=0.1,
                initupperpass=1.0e20,
                initupperstop=1.0e20,
            )
            lstest, lptest, uptest, ustest = testfilter.getfreqs()
            if lptest < nyquist:
                allfilters.append(
                    {
                        "name": "0.1Hz HP " + transferfunc,
                        "filter": NoncausalFilter(
                            filtertype="arb",
                            transferfunc=transferfunc,
                            initlowerstop=0.09,
                            initlowerpass=0.1,
                            initupperpass=1.0e20,
                            initupperstop=1.0e20,
                            debug=False,
                        ),
                    }
                )

        # show the end effects waveforms
        if displayplots:
            plt.figure()
            plt.ylim([-2.2, 2.2 * len(testwaves)])
            for thewave in testwaves:
                legend = []
                offset = 0.0
                for thefilter in allfilters:
                    plt.plot(
                        thewave["timeaxis"],
                        offset + thefilter["filter"].apply(1.0 / sampletime, thewave["waveform"]),
                    )
                    legend.append(thewave["name"] + ": " + thefilter["name"])
                    offset += 1.25
                plt.legend(legend)
                plt.show()


def test_padvec(displayplots=False, debug=False):
    eval_padvecprops(
        sampletime=0.72, tclengthinsecs=300.0, numruns=100, displayplots=displayplots, debug=debug
    )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_padvec(displayplots=True, debug=True)
