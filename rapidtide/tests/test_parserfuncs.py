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
import argparse
import os
import tempfile
from unittest.mock import patch

import pytest

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def proccolspec(thecolspec):
    if thecolspec is not None:
        # see if this is a numeric or text list
        tokenlist = (thecolspec.split(",")[0]).split("-")
        try:
            firstelement = int(tokenlist[0])
            return tide_io.colspectolist(thecolspec)
        except ValueError:
            return thecolspec.split(",")
    else:
        return [None]


def _get_parser():
    """
    Argument parser for adjust offset
    """
    parser = argparse.ArgumentParser(
        prog="dummy",
        description="dummy",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputmap",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the rapidtide maxtime map.",
    )

    return parser


# ==================== argument validators ====================


def _makeparser():
    """A bare parser for the validators to report errors through.

    Returns
    -------
    argparse.ArgumentParser
        A parser whose error() raises SystemExit, as argparse's does.
    """
    return argparse.ArgumentParser()


def is_int_tests(debug=False):
    """is_int accepts integers and rejects anything else, with optional bounds."""
    if debug:
        print("is_int_tests")
    theparser = _makeparser()

    assert pf.is_int(theparser, "42") == 42
    assert pf.is_int(theparser, 42) == 42
    assert isinstance(pf.is_int(theparser, "42"), int)

    # "auto" is a legal sentinel and passes straight through
    assert pf.is_int(theparser, "auto") == "auto"

    # anything else that is not an integer is rejected with the intended message,
    # rather than a TypeError about the except clause
    for thebadvalue in ("4.5", "notanumber", ""):
        with pytest.raises(SystemExit):
            pf.is_int(theparser, thebadvalue)

    # bounds are inclusive on both ends
    assert pf.is_int(theparser, "5", minval=5, maxval=10) == 5
    assert pf.is_int(theparser, "10", minval=5, maxval=10) == 10
    with pytest.raises(SystemExit):
        pf.is_int(theparser, "4", minval=5)
    with pytest.raises(SystemExit):
        pf.is_int(theparser, "11", maxval=10)


def is_float_tests(debug=False):
    """is_float mirrors is_int: numbers through, "auto" through, anything else
    reported as a parser error rather than raised as a TypeError."""
    if debug:
        print("is_float_tests")
    theparser = _makeparser()

    assert pf.is_float(theparser, "3.5") == pytest.approx(3.5)
    assert pf.is_float(theparser, "auto") == "auto"
    assert pf.is_float(theparser, "4", minval=0.0, maxval=10.0) == pytest.approx(4.0)

    for thebadvalue in ("notanumber", ""):
        with pytest.raises(SystemExit):
            pf.is_float(theparser, thebadvalue)
    with pytest.raises(SystemExit):
        pf.is_float(theparser, "-1.0", minval=0.0)
    with pytest.raises(SystemExit):
        pf.is_float(theparser, "11.0", maxval=10.0)


def is_range_tests(debug=False):
    """is_range wants exactly two numbers in ascending order."""
    if debug:
        print("is_range_tests")
    theparser = _makeparser()

    # the argument is validated, not converted - it comes back exactly as supplied,
    # so argparse's own type conversion still applies afterwards
    assert pf.is_range(theparser, ["1.0", "2.0"]) == ["1.0", "2.0"]
    # equal endpoints are allowed, the check is min <= max
    assert pf.is_range(theparser, ["2.0", "2.0"]) == ["2.0", "2.0"]
    assert pf.is_range(theparser, None) is None

    for thebadvalue in (["1.0"], ["1.0", "2.0", "3.0"], ["2.0", "1.0"]):
        with pytest.raises(SystemExit):
            pf.is_range(theparser, thebadvalue)

    # A non-numeric pair reaches an unguarded float() and comes back as a bare
    # ValueError rather than a parser error.  Called through argparse that still
    # becomes an "invalid value" message, so it is a worse message rather than a
    # crash; pinned here so the difference from the checks above is visible.
    with pytest.raises(ValueError):
        pf.is_range(theparser, ["a", "b"])


def is_valid_tag_tests(debug=False):
    """A tag is a name and a value separated by the first comma; later commas belong
    to the value, so a value containing commas survives intact."""
    if debug:
        print("is_valid_tag_tests")
    theparser = _makeparser()

    assert pf.is_valid_tag(theparser, "environment,production") == ("environment", "production")
    # only the first comma splits
    assert pf.is_valid_tag(theparser, "type,web,app") == ("type", "web,app")

    # a bare name with no value is an error
    with pytest.raises(SystemExit):
        pf.is_valid_tag(theparser, "justaname")


def is_valid_file_or_float_tests(debug=False):
    """The argument may be a path to an existing file or a number.  Anything else is
    rejected, so a mistyped filename does not silently become a value."""
    if debug:
        print("is_valid_file_or_float_tests")
    theparser = _makeparser()

    assert pf.is_valid_file_or_float(theparser, "3.5") == 3.5
    assert pf.is_valid_file_or_float(theparser, "-2") == -2.0

    with tempfile.TemporaryDirectory() as thedir:
        thefile = os.path.join(thedir, "real.txt")
        with open(thefile, "w") as thehandle:
            thehandle.write("data")
        assert pf.is_valid_file_or_float(theparser, thefile) == thefile

        with pytest.raises(SystemExit):
            pf.is_valid_file_or_float(theparser, os.path.join(thedir, "missing.txt"))


def parserange_tests(debug=False):
    """parserange normalizes a (start, end) pair: negatives become the extremes, and
    an inverted range is an error rather than an empty selection."""
    if debug:
        print("parserange_tests")

    assert pf.parserange((10, 20)) == (10, 20)
    # a negative start means "from the beginning"
    assert pf.parserange((-5, 15)) == (0, 15)
    # a negative end means "to the end", represented by a very large number
    thestart, theend = pf.parserange((10, -5))
    assert thestart == 10
    assert theend > 1000000

    # both negative: the whole range
    thestart, theend = pf.parserange((-1, -1), debug=True)
    assert thestart == 0
    assert theend > 1000000

    # an inverted range selects nothing, which is a caller error
    with pytest.raises(ValueError, match="startpoint must be"):
        pf.parserange((20, 10))
    with pytest.raises(ValueError):
        pf.parserange((10, 10), descriptor="simcalcrange")


def detailedversion_tests(debug=False):
    """--detailedversion prints build metadata and exits; it must not raise whatever
    the version machinery reports, including when everything is UNKNOWN."""
    if debug:
        print("detailedversion_tests")

    # detailedversion prints and then exits, the way --version options do
    for theversioninfo in (
        ("2.9.0", "abc1234", "2026-01-01", False),
        ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"),
    ):
        with patch(
            "rapidtide.workflows.parser_funcs.tide_util.version", return_value=theversioninfo
        ):
            with pytest.raises(SystemExit):
                pf.detailedversion()


# ==================== filter option postprocessing ====================


def postprocessfilteropts_tests(debug=False):
    """postprocessfilteropts turns the filter arguments into a NoncausalFilter."""
    if debug:
        print("postprocessfilteropts_tests")

    theparser = argparse.ArgumentParser()
    pf.addfilteropts(theparser, filtertarget="timecourses", details=True)

    # a named band gives that band's frequencies
    theargs = theparser.parse_args(["--filterband", "lfo"])
    theargs, thefilter = pf.postprocessfilteropts(theargs)
    thelowerstop, thelowerpass, theupperpass, theupperstop = thefilter.getfreqs()
    assert thelowerpass == pytest.approx(0.01)
    assert theupperpass == pytest.approx(0.15)

    # an explicit passband overrides the named band
    theargs = theparser.parse_args(["--filterfreqs", "0.05", "0.2"])
    theargs, thefilter = pf.postprocessfilteropts(theargs, debug=True)
    assert theargs.arbvec is not None
    thelowerstop, thelowerpass, theupperpass, theupperstop = thefilter.getfreqs()
    assert thelowerpass == pytest.approx(0.05)
    assert theupperpass == pytest.approx(0.2)

    # a stopband without a passband is meaningless
    theargs = theparser.parse_args(["--filterstopfreqs", "0.01", "0.3"])
    try:
        pf.postprocessfilteropts(theargs)
    except ValueError:
        pass
    else:
        raise AssertionError("a stopband with no passband was accepted")

    # missing attributes fall back to the defaults rather than raising
    thebareargs = argparse.Namespace(
        filterband="lfo", filtertype="trapezoidal", passvec=None, stopvec=None
    )
    thebareargs, thefilter = pf.postprocessfilteropts(thebareargs)
    assert thebareargs.filtorder == pf.DEFAULT_FILTER_ORDER
    assert thebareargs.padseconds == pf.DEFAULT_PAD_SECONDS


def postprocessfilteropts_padtype_is_not_wired_up(debug=False):
    """--padtype is silently ignored.

    The parser stores the choice in args.ncfiltpadtype, but postprocessfilteropts
    reads args.prefilterpadtype.  That attribute does not exist, so the try/except
    swallows the AttributeError and substitutes the default - the user's choice never
    reaches the filter.  This is recorded as a known gotcha in the project's
    CLAUDE.md; the test pins the CURRENT behaviour so that fixing it is a deliberate,
    visible change rather than a silent one.
    """
    if debug:
        print("postprocessfilteropts_padtype_is_not_wired_up")

    theparser = argparse.ArgumentParser()
    pf.addfilteropts(theparser, filtertarget="timecourses", details=True)
    theargs = theparser.parse_args(["--filterband", "lfo", "--padtype", "zero"])

    # the parser did record the request, under a different name
    assert theargs.ncfiltpadtype == "zero"
    assert not hasattr(theargs, "prefilterpadtype")

    theargs, thefilter = pf.postprocessfilteropts(theargs)

    # ... and it is then discarded in favour of the default
    assert theargs.prefilterpadtype == pf.DEFAULT_PREFILTERPADTYPE
    assert thefilter.padtype == pf.DEFAULT_PREFILTERPADTYPE
    assert thefilter.padtype != "zero", "if this now passes, --padtype has been wired up"


def postprocesssearchrangeopts_tests(debug=False):
    """The search range is stored as a lag_extrema pair, with lagmin and lagmax
    broken out for the callers that want them separately."""
    if debug:
        print("postprocesssearchrangeopts_tests")

    theparser = argparse.ArgumentParser()
    pf.addsearchrangeopts(theparser, details=True)
    theargs = theparser.parse_args(["--searchrange", "-10.0", "20.0"])
    theargs = pf.postprocesssearchrangeopts(theargs)

    assert theargs.lagmin == pytest.approx(-10.0)
    assert theargs.lagmax == pytest.approx(20.0)


def postprocesssamplerateopts_tests(debug=False):
    """A sample rate of "auto" resolves to 1.0 - a sample per point, which is the
    right assumption for a file that carries no timing.  An explicit value is passed
    through untouched."""
    if debug:
        print("postprocesssamplerateopts_tests")

    theargs = argparse.Namespace(samplerate="auto")
    theargs = pf.postprocesssamplerateopts(theargs, debug=True)
    assert theargs.samplerate == pytest.approx(1.0)

    theargs = argparse.Namespace(samplerate=5.0)
    theargs = pf.postprocesssamplerateopts(theargs)
    assert theargs.samplerate == pytest.approx(5.0)


def postprocesstagopts_tests(debug=False):
    """--infotag pairs become INFO_ prefixed attributes on the namespace, which is
    how they reach the options JSON that records how a run was configured."""
    if debug:
        print("postprocesstagopts_tests")

    theargs = argparse.Namespace(infotag=[("version", "1.0"), ("build", "2023")])
    theargs = pf.postprocesstagopts(theargs)

    assert theargs.INFO_version == "1.0"
    assert theargs.INFO_build == "2023"
    # the raw list is consumed, so it does not also end up in the output
    assert not hasattr(theargs, "infotag")

    # no tags at all leaves the namespace alone
    theargs = argparse.Namespace(infotag=None)
    theargs = pf.postprocesstagopts(theargs)
    assert theargs.infotag is None


def test_parserfuncs(debug=False, local=False):
    # set input and output directories
    exampleroot = get_examples_path(local)
    testtemproot = get_test_temp_path(local)

    theparser = _get_parser()

    testvecs = [
        ["sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json", [None]],
        [
            "sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:acolname",
            ["acolname"],
        ],
        [
            "sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:acolname,bcolname",
            ["acolname", "bcolname"],
        ],
        [
            "sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.tsv.gz:1,2,5-10",
            [1, 2, 5, 6, 7, 8, 9, 10],
        ],
        [
            "sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:3,2,7,5-10,6-11",
            [2, 3, 5, 6, 7, 8, 9, 10, 11],
        ],
    ]
    for infile, expectedcols in testvecs:
        filename = os.path.join(exampleroot, infile)
        retval = pf.is_valid_file(theparser, filename)
        thename, thecolspec = tide_io.parsefilespec(retval)
        collist = proccolspec(thecolspec)
        if debug:
            print(filename, retval, thename, thecolspec, collist)
        assert collist == expectedcols


if __name__ == "__main__":
    test_parserfuncs(debug=True, local=True)


def test_parserfuncvalidators(debug=False, local=False):
    """Entry point for the validator and postprocessing tests."""
    is_int_tests(debug=debug)
    is_float_tests(debug=debug)
    is_range_tests(debug=debug)
    is_valid_tag_tests(debug=debug)
    is_valid_file_or_float_tests(debug=debug)
    parserange_tests(debug=debug)
    detailedversion_tests(debug=debug)
    postprocessfilteropts_tests(debug=debug)
    postprocessfilteropts_padtype_is_not_wired_up(debug=debug)
    postprocesssearchrangeopts_tests(debug=debug)
    postprocesssamplerateopts_tests(debug=debug)
    postprocesstagopts_tests(debug=debug)
