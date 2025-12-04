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
import numpy as np

import rapidtide.workflows.rapidtide_parser as rp
import rapidtide.workflows.parser_funcs as pf

testlist = {}
testlist["searchrange"] = {
    "command": ["--searchrange", "-7", "15.2"],
    "results": [["lagmin", -7.0, "isfloat"], ["lagmax", 15.2, "isfloat"]],
}
testlist["filterband"] = {
    "command": ["--filterband", "lfo"],
    "results": [
        ["filterband", "lfo"],
        ["lowerpass", 0.01, "isfloat"],
        ["upperpass", 0.15, "isfloat"],
    ],
}
testlist["filtertype"] = {
    "command": ["--filtertype", "trapezoidal"],
    "results": [["filtertype", "trapezoidal"]],
}
testlist["filterfreqs"] = {
    "command": ["--filterfreqs", "0.1", "0.2"],
    "results": [["arbvec", [0.1, 0.2, 0.095, 0.21], "isfloat"]],
}
testlist["pickleft"] = {"command": ["--pickleft"], "results": [["pickleft", True]]}
testlist["corrweighting"] = {
    "command": ["--corrweighting", "phat"],
    "results": [["corrweighting", "phat"]],
}
testlist["datatstep"] = {
    "command": ["--datatstep", "1.23"],
    "results": [["realtr", 1.23, "isfloat"]],
}
testlist["datafreq"] = {"command": ["--datafreq", "10.0"], "results": [["realtr", 0.1, "isfloat"]]}
testlist["noantialias"] = {
    "command": ["--noantialias"],
    "results": [["antialias", False]],
}
testlist["invert"] = {"command": ["--invert"], "results": [["invertregressor", True]]}
testlist["interptype"] = {
    "command": ["--interptype", "cubic"],
    "results": [["interptype", "cubic"]],
}
testlist["offsettime"] = {
    "command": ["--offsettime", "10.1"],
    "results": [["offsettime", 10.1, "isfloat"]],
}
testlist["timerange"] = {
    "command": ["--timerange", "2", "-1"],
    "results": [["startpoint", 2], ["endpoint", 100000000]],
}
testlist["numnull"] = {
    "command": ["--numnull", "0"],
    "results": [["numestreps", 0], ["ampthreshfromsig", False]],
}
testlist["initialdelay"] = {
    "command": ["--initialdelay", "0.0"],
    "results": [
        ["initialdelayvalue", 0.0, "isfloat"],
    ],
}
testlist["nodelayfit"] = {
    "command": ["--nodelayfit"],
    "results": [
        ["fixdelay", True],
    ],
}
testlist["delaymapping"] = {
    "command": ["--delaymapping"],
    "results": [
        ["fixdelay", True],
        ["passes", rp.DEFAULT_DELAYMAPPING_PASSES],
        ["despeckle_passes", rp.DEFAULT_DELAYMAPPING_DESPECKLE_PASSES],
        ["gausssigma", rp.DEFAULT_DELAYMAPPING_SPATIALFILT],
        ["lagmin", rp.DEFAULT_DELAYMAPPING_LAGMIN],
        ["lagmax", rp.DEFAULT_DELAYMAPPING_LAGMAX],
        ["refineoffset",True],
        ["refinedelay", True],
        ["outputlevel", "normal"],
        ["dolinfitfilt", True],
    ],
}


def checktests(thetestvec, testlist, theargs, epsilon):
    for thetest in thetestvec:
        for theresult in testlist[thetest]["results"]:
            print("testing", testlist[thetest]["command"], "effect on", theresult[0])
            if len(theresult) <= 2:
                # non-float argument
                print(theargs[theresult[0]], theresult[1])
                if isinstance(theresult[1], list):
                    for i in range(len(theresult[1])):
                        assert theargs[theresult[0]][i] == theresult[1][i]
                else:
                    assert theargs[theresult[0]] == theresult[1]
            else:
                print(theargs[theresult[0]], theresult[1])
                if isinstance(theresult[1], list):
                    for i in range(len(theresult[1])):
                        assert np.fabs(theargs[theresult[0]][i] - theresult[1][i]) < epsilon
                else:
                    assert np.fabs(theargs[theresult[0]] - theresult[1]) < epsilon

def checkavector(thetestvec, epsilon):
    print(testlist)
    print(thetestvec)

    # make the argument and results lists
    arglist = [
        "../data/examples/src/sub-RAPIDTIDETEST.nii.gz",
        "../data/examples/dst/parsertestdummy",
    ]
    resultlist = []
    for thetest in thetestvec:
        arglist += testlist[thetest]["command"]
        for theresult in testlist[thetest]["results"]:
            resultlist += [theresult]

    print(arglist)
    print(resultlist)

    theargs, ncprefilter = rp.process_args(inputargs=arglist)

    checktests(thetestvec, testlist, theargs, epsilon)


def main():
    epsilon = 0.00001

    # construct the first test vector
    thetestvec = []
    thetestvec.append("filterband")
    thetestvec.append("filtertype")
    thetestvec.append("searchrange")
    thetestvec.append("pickleft")
    thetestvec.append("corrweighting")
    thetestvec.append("datafreq")
    thetestvec.append("noantialias")
    thetestvec.append("invert")
    thetestvec.append("interptype")
    thetestvec.append("offsettime")
    thetestvec.append("datafreq")
    checkavector(thetestvec, epsilon)

    # construct the second test vector
    thetestvec = []
    thetestvec.append("filterfreqs")
    thetestvec.append("datatstep")
    thetestvec.append("timerange")
    thetestvec.append("numnull")
    thetestvec.append("initialdelay")
    thetestvec.append("nodelayfit")
    checkavector(thetestvec, epsilon)


if __name__ == "__main__":
    main()
