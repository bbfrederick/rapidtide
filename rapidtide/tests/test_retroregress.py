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
"""Argument handling tests for the retroregress workflow.

The workflow body itself needs a complete rapidtide output set to run against, and is
exercised end to end by test_fullrunrapidtide_v3, v6 and v7.  What those integration
tests do not touch is the argument surface - the option defaults, the validation of
paths, and the choices the parser accepts - which is what this file covers.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pytest

import rapidtide.workflows.retroregress as rr

# ==================== helpers ====================


def _makedummyinputs(tmpdir):
    """Create the two positional inputs the parser insists on.

    Parameters
    ----------
    tmpdir : str
        Directory to write into.

    Returns
    -------
    tuple of (str, str)
        The fmri file path and the dataset root.
    """
    import nibabel as nib

    fmripath = os.path.join(tmpdir, "sub-01.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 10), dtype=np.float32), np.eye(4)), fmripath)
    return fmripath, os.path.join(tmpdir, "sub-01")


# ==================== _get_parser ====================


def test_get_parser_returns_parser(debug=False):
    """The parser factory returns a real ArgumentParser."""
    if debug:
        print("get_parser_returns_parser")
    assert isinstance(rr._get_parser(), argparse.ArgumentParser)


def test_get_parser_requires_both_positionals(debug=False):
    """Both the fmri file and the dataset root are required."""
    if debug:
        print("get_parser_requires_both_positionals")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        for theargs in [[], [fmripath]]:
            with pytest.raises(SystemExit):
                rr._get_parser().parse_args(theargs)


def test_get_parser_rejects_missing_fmri_file(debug=False):
    """A nonexistent fmri file is caught at parse time."""
    if debug:
        print("get_parser_rejects_missing_fmri_file")
    with pytest.raises(SystemExit):
        rr._get_parser().parse_args(["/nonexistent/nosuch.nii.gz", "/tmp/someroot"])


def test_get_parser_defaults(debug=False):
    """The option defaults are what the workflow documents."""
    if debug:
        print("get_parser_defaults")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        args = rr._get_parser().parse_args([fmripath, theroot])
        assert args.alternateoutput is None
        assert args.regressderivs == 0
        assert args.nprocs == 1
        assert args.numskip == 0
        assert args.outputlevel == "normal"
        assert args.showprogressbar
        assert not args.makepseudofile
        assert args.refinedelay
        assert args.refinecorr
        assert args.filterwithrefineddelay
        assert args.delaypatchthresh == rr.DEFAULT_PATCHTHRESH
        assert args.delayoffsetgausssigma == rr.DEFAULT_DELAYOFFSETSPATIALFILT
        assert not args.debug
        assert not args.focaldebug
        assert not args.sLFOfiltmask


def test_get_parser_negating_flags(debug=False):
    """Each of the "no" flags turns its feature off."""
    if debug:
        print("get_parser_negating_flags")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        for theflag, thedest in [
            ("--noprogressbar", "showprogressbar"),
            ("--norefinedelay", "refinedelay"),
            ("--norefinecorr", "refinecorr"),
            ("--nofilterwithrefineddelay", "filterwithrefineddelay"),
        ]:
            args = rr._get_parser().parse_args([fmripath, theroot, theflag])
            assert getattr(args, thedest) is False, f"{theflag} did not clear {thedest}"


def test_get_parser_enabling_flags(debug=False):
    """Each of the plain store_true flags turns its feature on."""
    if debug:
        print("get_parser_enabling_flags")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        for theflag, thedest in [
            ("--makepseudofile", "makepseudofile"),
            ("--debug", "debug"),
            ("--focaldebug", "focaldebug"),
            ("--sLFOfiltmask", "sLFOfiltmask"),
        ]:
            args = rr._get_parser().parse_args([fmripath, theroot, theflag])
            assert getattr(args, thedest) is True, f"{theflag} did not set {thedest}"


def test_get_parser_numeric_options(debug=False):
    """The numeric options are parsed with the right types."""
    if debug:
        print("get_parser_numeric_options")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        args = rr._get_parser().parse_args(
            [
                fmripath,
                theroot,
                "--regressderivs",
                "2",
                "--nprocs",
                "4",
                "--numskip",
                "5",
                "--delaypatchthresh",
                "2.5",
                "--delayoffsetspatialfilt",
                "3",
            ]
        )
        assert args.regressderivs == 2
        assert args.nprocs == 4
        assert args.numskip == 5
        assert args.delaypatchthresh == 2.5
        assert args.delayoffsetgausssigma == 3


def test_get_parser_alternateoutput(debug=False):
    """An alternate output root is stored as given."""
    if debug:
        print("get_parser_alternateoutput")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        thealternate = os.path.join(tmpdir, "someotherroot")
        args = rr._get_parser().parse_args([fmripath, theroot, "--alternateoutput", thealternate])
        assert args.alternateoutput == thealternate


def test_get_parser_outputlevel_choices(debug=False):
    """Every documented output level is accepted, and nothing else is."""
    if debug:
        print("get_parser_outputlevel_choices")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        for thelevel in ["min", "less", "normal", "more", "max", "onlyregressors"]:
            args = rr._get_parser().parse_args([fmripath, theroot, "--outputlevel", thelevel])
            assert args.outputlevel == thelevel
        with pytest.raises(SystemExit):
            rr._get_parser().parse_args([fmripath, theroot, "--outputlevel", "bogus"])


def test_get_parser_rejects_abbreviations(debug=False):
    """The parser does not accept abbreviated option names."""
    if debug:
        print("get_parser_rejects_abbreviations")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        with pytest.raises(SystemExit):
            rr._get_parser().parse_args([fmripath, theroot, "--regressder", "1"])


# ==================== process_args ====================


def test_process_args_from_list(debug=False):
    """process_args parses an explicit argument list into a namespace."""
    if debug:
        print("process_args_from_list")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        args = rr.process_args(inputargs=[fmripath, theroot, "--regressderivs", "1"])
        assert isinstance(args, argparse.Namespace)
        assert args.regressderivs == 1
        assert args.fmrifile == fmripath


def test_process_args_reads_sys_argv(debug=False):
    """With no argument list, process_args falls back to the command line."""
    if debug:
        print("process_args_reads_sys_argv")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        oldargv = sys.argv
        sys.argv = ["retroregress", fmripath, theroot, "--nprocs", "2"]
        try:
            args = rr.process_args()
        finally:
            sys.argv = oldargv
        assert args.nprocs == 2


def test_process_args_rejects_bad_input(debug=False):
    """Invalid arguments still exit rather than returning a partial namespace."""
    if debug:
        print("process_args_rejects_bad_input")
    with pytest.raises(SystemExit):
        rr.process_args(inputargs=["/nonexistent/nosuch.nii.gz", "/tmp/root"])


# ==================== workflow input validation ====================


def test_retroregress_missing_dataset_root(debug=False):
    """A dataset root with no rapidtide output behind it fails loudly.

    The workflow's first act is to read the runoptions file written by the original
    rapidtide run; if that is not there, nothing downstream can work.
    """
    if debug:
        print("retroregress_missing_dataset_root")
    with tempfile.TemporaryDirectory() as tmpdir:
        fmripath, theroot = _makedummyinputs(tmpdir)
        args = rr.process_args(inputargs=[fmripath, theroot])
        # no rapidtide outputs exist under this root at all
        with pytest.raises((SystemExit, FileNotFoundError, OSError, ValueError, KeyError)):
            rr.retroregress(args)


def test_sentinel_is_a_distinct_object(debug=False):
    """The module sentinel is a unique marker, not something a user could pass by accident."""
    if debug:
        print("sentinel_is_a_distinct_object")
    assert isinstance(rr.sentinel, rr._Sentinel)
    assert rr.sentinel is not None
    assert rr.sentinel != 0
    assert rr.sentinel != ""
    # a fresh instance is a different object, so identity checks are meaningful
    assert rr.sentinel is not rr._Sentinel()


def test_retroregress(debug=False):
    test_get_parser_returns_parser(debug=debug)
    test_get_parser_requires_both_positionals(debug=debug)
    test_get_parser_rejects_missing_fmri_file(debug=debug)
    test_get_parser_defaults(debug=debug)
    test_get_parser_negating_flags(debug=debug)
    test_get_parser_enabling_flags(debug=debug)
    test_get_parser_numeric_options(debug=debug)
    test_get_parser_alternateoutput(debug=debug)
    test_get_parser_outputlevel_choices(debug=debug)
    test_get_parser_rejects_abbreviations(debug=debug)
    test_process_args_from_list(debug=debug)
    test_process_args_reads_sys_argv(debug=debug)
    test_process_args_rejects_bad_input(debug=debug)
    test_retroregress_missing_dataset_root(debug=debug)
    test_sentinel_is_a_distinct_object(debug=debug)


if __name__ == "__main__":
    test_retroregress(debug=True)
