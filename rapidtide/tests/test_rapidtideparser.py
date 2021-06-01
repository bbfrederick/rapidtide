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
import numpy as np

from rapidtide.workflows.rapidtide_parser import process_args

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
testlist["fixdelay"] = {
    "command": ["--fixdelay", "0.0"],
    "results": [["fixeddelayvalue", 0.0, "isfloat"],],
}


def checktests(testvec, testlist, theargs, epsilon):
    for thetest in testvec:
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


def main():

    epsilon = 0.00001

    # construct the first test vector
    testvec = []
    testvec.append("filterband")
    testvec.append("filtertype")
    testvec.append("searchrange")
    testvec.append("pickleft")
    testvec.append("corrweighting")
    testvec.append("datafreq")
    testvec.append("noantialias")
    testvec.append("invert")
    testvec.append("interptype")
    testvec.append("offsettime")
    testvec.append("datafreq")

    print(testlist)
    print(testvec)

    # make the argument and results lists
    arglist = [
        "../data/examples/src/sub-RAPIDTIDETEST.nii.gz",
        "../data/examples/dst/parsertestdummy",
    ]
    resultlist = []
    for thetest in testvec:
        arglist += testlist[thetest]["command"]
        for theresult in testlist[thetest]["results"]:
            resultlist += [theresult]

    print(arglist)
    print(resultlist)

    theargs, ncprefilter = process_args(inputargs=arglist)

    checktests(testvec, testlist, theargs, epsilon)

    # construct the second test vector
    testvec = []
    testvec.append("filterfreqs")
    testvec.append("datatstep")
    testvec.append("timerange")
    testvec.append("numnull")
    testvec.append("fixdelay")

    print(testlist)
    print(testvec)

    # make the argument and results lists
    arglist = [
        "../data/examples/src/sub-RAPIDTIDETEST.nii.gz",
        "../data/examples/dst/parsertestdummy",
    ]
    resultlist = []
    for thetest in testvec:
        arglist += testlist[thetest]["command"]
        for theresult in testlist[thetest]["results"]:
            resultlist += [theresult]

    print(arglist)
    print(resultlist)

    theargs, ncprefilter = process_args(inputargs=arglist)

    checktests(testvec, testlist, theargs, epsilon)


if __name__ == "__main__":
    main()
