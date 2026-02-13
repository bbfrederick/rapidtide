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
#
import os

import numpy as np

import rapidtide.workflows.rapidtide_parser as rp
from rapidtide.tests.utils import get_examples_path, get_test_temp_path

global exampleroot, testtemproot  # set input and output directories
global testlist


def _resolve_examples_path(local=False):
    """Return the examples path, falling back to databundle/ if src/ has no data."""
    epath = get_examples_path(local)
    test_file = os.path.join(epath, "sub-RAPIDTIDETEST.nii.gz")
    if not os.path.isfile(test_file):
        # Try databundle/ as a fallback (local dev without CI extraction)
        from rapidtide.tests.utils import get_rapidtide_root

        alt = os.path.realpath(
            os.path.join(get_rapidtide_root(), "data", "examples", "databundle")
        )
        if os.path.isfile(os.path.join(alt, "sub-RAPIDTIDETEST.nii.gz")):
            return alt
    return epath


def setuptestlist():
    global testlist
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
    testlist["datafreq"] = {
        "command": ["--datafreq", "10.0"],
        "results": [["realtr", 0.1, "isfloat"]],
    }
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
    testlist["regressor"] = {
        "command": [
            "--regressor",
            f"{exampleroot}/sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:pass3",
        ],
        "results": [
            [
                "regressorfile",
                f"{exampleroot}/sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:pass3",
            ]
        ],
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
            ["passes", rp.DEFAULT_DELAYMAPPING_PASSES],
            ["despeckle_passes", rp.DEFAULT_DELAYMAPPING_DESPECKLE_PASSES],
            ["gausssigma", rp.DEFAULT_DELAYMAPPING_SPATIALFILT],
            ["lagmin", rp.DEFAULT_DELAYMAPPING_LAGMIN],
            ["lagmax", rp.DEFAULT_DELAYMAPPING_LAGMAX],
            ["refineoffset", True],
            ["refinedelay", True],
            ["outputlevel", "normal"],
            ["dolinfitfilt", True],
        ],
    }
    testlist["denoising"] = {
        "command": ["--denoising"],
        "results": [
            ["passes", rp.DEFAULT_DENOISING_PASSES],
            ["despeckle_passes", rp.DEFAULT_DENOISING_DESPECKLE_PASSES],
            ["gausssigma", rp.DEFAULT_DENOISING_SPATIALFILT],
            ["peakfittype", rp.DEFAULT_PEAKFIT_TYPE],
            ["lagmin", rp.DEFAULT_DENOISING_LAGMIN],
            ["lagmax", rp.DEFAULT_DENOISING_LAGMAX],
            ["refineoffset", True],
            ["refinedelay", True],
            ["zerooutbadfit", False],
            ["dolinfitfilt", True],
        ],
    }

    testlist["globalpreselect"] = {
        "command": ["--globalpreselect"],
        "results": [
            ["passes", 1],
            ["despeckle_passes", 0],
            ["refinedespeckled", False],
            ["outputlevel", "normal"],
            ["dolinfitfilt", False],
            ["saveintermediatemaps", False],
        ],
    }

    testlist["CVR"] = {
        "command": ["--CVR"],
        "results": [
            ["despeckle_passes", rp.DEFAULT_CVRMAPPING_DESPECKLE_PASSES],
            [
                "passvec",
                (rp.DEFAULT_CVRMAPPING_FILTER_LOWERPASS, rp.DEFAULT_CVRMAPPING_FILTER_UPPERPASS),
            ],
            ["filterband", "None"],
            ["lagmin", rp.DEFAULT_CVRMAPPING_LAGMIN],
            ["lagmax", rp.DEFAULT_CVRMAPPING_LAGMAX],
            ["preservefiltering", True],
            ["passes", 1],
            ["outputlevel", "min"],
            ["refinedelay", True],
            ["dolinfitfilt", False],
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


def checkavector(thetestvec, epsilon, debug=False, local=False):
    global testlist
    if debug:
        print(testlist)
        print(thetestvec)

    # make the argument and results lists
    arglist = [
        f"{exampleroot}/sub-RAPIDTIDETEST.nii.gz",
        f"{testtemproot}/parsertestdummy",
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


def _run_parser(extra_args=None, local=False):
    """Run process_args with the standard test input file and optional extra arguments."""
    exroot = _resolve_examples_path(local)
    ttroot = get_test_temp_path(local)
    arglist = [
        f"{exroot}/sub-RAPIDTIDETEST.nii.gz",
        f"{ttroot}/parsertestdummy",
    ]
    if extra_args:
        arglist += extra_args
    return rp.process_args(inputargs=arglist)


def test_rapidtideparser(debug=False, local=False):
    global testlist
    global exampleroot, testtemproot

    exampleroot = _resolve_examples_path(local)
    testtemproot = get_test_temp_path(local)

    epsilon = 0.00001
    setuptestlist()

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
    checkavector(thetestvec, epsilon, debug=debug, local=local)

    # construct the second test vector
    thetestvec = []
    thetestvec.append("filterfreqs")
    thetestvec.append("datatstep")
    thetestvec.append("timerange")
    thetestvec.append("numnull")
    thetestvec.append("initialdelay")
    thetestvec.append("nodelayfit")
    checkavector(thetestvec, epsilon, debug=debug, local=local)

    # construct the third test vector
    thetestvec = []
    thetestvec.append("delaymapping")
    checkavector(thetestvec, epsilon, debug=debug, local=local)

    # construct the fourth test vector
    thetestvec = []
    thetestvec.append("denoising")
    checkavector(thetestvec, epsilon, debug=debug, local=local)

    # construct the fifth test vector
    thetestvec = []
    thetestvec.append("globalpreselect")
    checkavector(thetestvec, epsilon, debug=debug, local=local)

    # construct the sixth test vector
    thetestvec = []
    thetestvec.append("regressor")
    thetestvec.append("CVR")
    checkavector(thetestvec, epsilon, debug=debug, local=local)


def test_auto_defaults(local=False):
    """When no preset or explicit flag is used, 'auto' parameters resolve to their original defaults."""
    args, _ = _run_parser(local=local)

    assert args["refineoffset"] is True
    assert args["refinedelay"] is True
    assert args["outputlevel"] == "normal"
    assert args["zerooutbadfit"] is True
    assert args["preservefiltering"] is False
    assert args["passes"] == rp.DEFAULT_PASSES
    assert args["despeckle_passes"] == rp.DEFAULT_DESPECKLE_PASSES
    assert args["refinedespeckled"] is True
    assert args["saveintermediatemaps"] is False
    assert args["passvec"] is None


def test_explicit_flags_without_preset(local=False):
    """User-explicit flags should set parameters correctly even without a preset."""
    # --nofitfilt sets zerooutbadfit to False
    args, _ = _run_parser(["--nofitfilt"], local=local)
    assert args["zerooutbadfit"] is False

    # --norefineoffset sets refineoffset to False
    args, _ = _run_parser(["--norefineoffset"], local=local)
    assert args["refineoffset"] is False

    # --norefinedelay sets refinedelay to False
    args, _ = _run_parser(["--norefinedelay"], local=local)
    assert args["refinedelay"] is False

    # --preservefiltering sets preservefiltering to True
    args, _ = _run_parser(["--preservefiltering"], local=local)
    assert args["preservefiltering"] is True

    # --norefinedespeckled sets refinedespeckled to False
    args, _ = _run_parser(["--norefinedespeckled"], local=local)
    assert args["refinedespeckled"] is False

    # --saveintermediatemaps sets saveintermediatemaps to True
    args, _ = _run_parser(["--saveintermediatemaps"], local=local)
    assert args["saveintermediatemaps"] is True

    # --outputlevel max overrides default
    args, _ = _run_parser(["--outputlevel", "max"], local=local)
    assert args["outputlevel"] == "max"

    # --passes overrides default
    args, _ = _run_parser(["--passes", "5"], local=local)
    assert args["passes"] == 5

    # --despecklepasses overrides default
    args, _ = _run_parser(["--despecklepasses", "2"], local=local)
    assert args["despeckle_passes"] == 2


def test_delaymapping_user_overrides(local=False):
    """User-explicit flags should override values that --delaymapping would set."""
    # Override passes
    args, _ = _run_parser(["--delaymapping", "--passes", "7"], local=local)
    assert args["passes"] == 7

    # Override despeckle_passes
    args, _ = _run_parser(["--delaymapping", "--despecklepasses", "1"], local=local)
    assert args["despeckle_passes"] == 1

    # Override refineoffset
    args, _ = _run_parser(["--delaymapping", "--norefineoffset"], local=local)
    assert args["refineoffset"] is False

    # Override refinedelay
    args, _ = _run_parser(["--delaymapping", "--norefinedelay"], local=local)
    assert args["refinedelay"] is False

    # Override outputlevel
    args, _ = _run_parser(["--delaymapping", "--outputlevel", "max"], local=local)
    assert args["outputlevel"] == "max"

    # Override searchrange (lag extents)
    args, _ = _run_parser(["--delaymapping", "--searchrange", "-5", "10"], local=local)
    assert args["lagmin"] == -5.0
    assert args["lagmax"] == 10.0

    # Param NOT touched by delaymapping still resolves to default
    assert args["zerooutbadfit"] is True
    assert args["preservefiltering"] is False


def test_denoising_user_overrides(local=False):
    """User-explicit flags should override values that --denoising would set."""
    # Override passes
    args, _ = _run_parser(["--denoising", "--passes", "5"], local=local)
    assert args["passes"] == 5

    # Override refineoffset
    args, _ = _run_parser(["--denoising", "--norefineoffset"], local=local)
    assert args["refineoffset"] is False
    # Other denoising values should still hold
    assert args["refinedelay"] is True
    assert args["zerooutbadfit"] is False

    # Override zerooutbadfit back to True (--nofitfilt sets to False, so
    # omitting it means denoising's False stays; to override we'd need a
    # positive flag, which doesn't exist — test that denoising's value holds)
    args, _ = _run_parser(["--denoising"], local=local)
    assert args["zerooutbadfit"] is False

    # Override searchrange
    args, _ = _run_parser(["--denoising", "--searchrange", "-3", "3"], local=local)
    assert args["lagmin"] == -3.0
    assert args["lagmax"] == 3.0


def test_cvr_user_overrides(local=False):
    """User-explicit flags should override values that --CVR would set."""
    exroot = _resolve_examples_path(local)
    regfile = (
        f"{exroot}/sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:pass3"
    )

    # Override passes
    args, _ = _run_parser(["--CVR", "--regressor", regfile, "--passes", "2"], local=local)
    assert args["passes"] == 2

    # Override outputlevel
    args, _ = _run_parser(
        ["--CVR", "--regressor", regfile, "--outputlevel", "normal"], local=local
    )
    assert args["outputlevel"] == "normal"

    # Override preservefiltering (CVR sets True, user sets False via omission — stays True)
    args, _ = _run_parser(["--CVR", "--regressor", regfile], local=local)
    assert args["preservefiltering"] is True

    # Override searchrange
    args, _ = _run_parser(
        ["--CVR", "--regressor", regfile, "--searchrange", "-2", "10"], local=local
    )
    assert args["lagmin"] == -2.0
    assert args["lagmax"] == 10.0


def test_globalpreselect_user_overrides(local=False):
    """User-explicit flags should override values that --globalpreselect would set."""
    # Override passes
    args, _ = _run_parser(["--globalpreselect", "--passes", "3"], local=local)
    assert args["passes"] == 3

    # Override outputlevel
    args, _ = _run_parser(["--globalpreselect", "--outputlevel", "max"], local=local)
    assert args["outputlevel"] == "max"

    # Override saveintermediatemaps
    args, _ = _run_parser(["--globalpreselect", "--saveintermediatemaps"], local=local)
    assert args["saveintermediatemaps"] is True

    # Override despeckle_passes
    args, _ = _run_parser(["--globalpreselect", "--despecklepasses", "2"], local=local)
    assert args["despeckle_passes"] == 2

    # Override refinedespeckled (globalpreselect sets False; omitting flag keeps False)
    args, _ = _run_parser(["--globalpreselect"], local=local)
    assert args["refinedespeckled"] is False


def test_venousrefine_macro(local=False):
    """The --venousrefine macro should set the expected parameter values."""
    args, _ = _run_parser(["--venousrefine"], local=local)

    assert args["lagminthresh"] == 2.5
    assert args["lagmaxthresh"] == 6.0
    assert args["ampthresh"] == 0.5
    assert args["ampthreshfromsig"] is False
    assert args["lagmaskside"] == "upper"


def test_nirs_macro(local=False):
    """The --nirs macro should set the expected parameter values."""
    args, _ = _run_parser(["--nirs"], local=local)

    assert args["nothresh"] is True
    assert args["preservefiltering"] is False
    assert args["dataiszeromean"] is True
    assert args["refineprenorm"] == "var"
    assert args["ampthresh"] == 0.7
    assert args["ampthreshfromsig"] is False
    assert args["lagminthresh"] == 0.1
    assert args["despeckle_passes"] == 0


def test_delaymapping_nofitfilt_interaction(local=False):
    """--nofitfilt should persist when --delaymapping does not touch zerooutbadfit."""
    args, _ = _run_parser(["--delaymapping", "--nofitfilt"], local=local)
    assert args["zerooutbadfit"] is False
    # delaymapping values should still hold
    assert args["refineoffset"] is True
    assert args["refinedelay"] is True
    assert args["passes"] == rp.DEFAULT_DELAYMAPPING_PASSES


def test_denoising_preserves_unrelated_user_flags(local=False):
    """User flags for parameters NOT set by --denoising should persist."""
    args, _ = _run_parser(["--denoising", "--saveintermediatemaps"], local=local)
    assert args["saveintermediatemaps"] is True

    args, _ = _run_parser(["--denoising", "--outputlevel", "max"], local=local)
    assert args["outputlevel"] == "max"
    # denoising values should still hold
    assert args["refineoffset"] is True
    assert args["zerooutbadfit"] is False


def test_nohistzero_follows_zerooutbadfit(local=False):
    """nohistzero should be derived from zerooutbadfit after auto-resolution."""
    # Default: zerooutbadfit=True → nohistzero=False
    args, _ = _run_parser(local=local)
    assert args["zerooutbadfit"] is True
    assert args["nohistzero"] is False

    # --nofitfilt: zerooutbadfit=False → nohistzero=True
    args, _ = _run_parser(["--nofitfilt"], local=local)
    assert args["zerooutbadfit"] is False
    assert args["nohistzero"] is True

    # --denoising: zerooutbadfit=False → nohistzero=True
    args, _ = _run_parser(["--denoising"], local=local)
    assert args["zerooutbadfit"] is False
    assert args["nohistzero"] is True


def test_despeckle_thresh_activates_despeckle_passes(local=False):
    """Setting --despecklethresh with despeckle_passes=0 should activate 1 pass."""
    args, _ = _run_parser(
        ["--globalpreselect", "--despecklethresh", "3.0"],
        local=local,
    )
    # globalpreselect sets despeckle_passes=0, but non-default despeckle_thresh
    # should bump it to 1
    assert args["despeckle_passes"] == 1


if __name__ == "__main__":
    test_rapidtideparser(debug=True, local=True)
    test_auto_defaults(local=True)
    test_explicit_flags_without_preset(local=True)
    test_delaymapping_user_overrides(local=True)
    test_denoising_user_overrides(local=True)
    test_cvr_user_overrides(local=True)
    test_globalpreselect_user_overrides(local=True)
    test_venousrefine_macro(local=True)
    test_nirs_macro(local=True)
    test_delaymapping_nofitfilt_interaction(local=True)
    test_denoising_preserves_unrelated_user_flags(local=True)
    test_nohistzero_follows_zerooutbadfit(local=True)
    test_despeckle_thresh_activates_despeckle_passes(local=True)
