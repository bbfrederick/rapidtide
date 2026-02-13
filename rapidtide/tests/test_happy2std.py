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
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rapidtide.workflows.happy2std import _get_parser, happy2std, transformmaps

# ==================== Helpers ====================


def _make_default_args(tmpdir, **overrides):
    """Create default args Namespace for happy2std."""
    defaults = dict(
        inputfileroot=os.path.join(tmpdir, "sub-01_task-rest_desc-"),
        outputdir=os.path.join(tmpdir, "output"),
        featdirectory=os.path.join(tmpdir, "sub-01.feat"),
        all=False,
        aligntohires=False,
        forcelinear=False,
        onefilename=None,
        preponly=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _setup_feat_dir(tmpdir, with_warp=False, with_hires=True, with_standard=True):
    """Create a minimal feat directory structure."""
    regdir = os.path.join(tmpdir, "sub-01.feat", "reg")
    os.makedirs(regdir, exist_ok=True)

    xform_func2std = os.path.join(regdir, "example_func2standard.mat")
    with open(xform_func2std, "w") as f:
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    xform_func2hires = os.path.join(regdir, "example_func2highres.mat")
    with open(xform_func2hires, "w") as f:
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    xform_hires2std = os.path.join(regdir, "highres2standard.mat")
    with open(xform_hires2std, "w") as f:
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    if with_warp:
        warpfile = os.path.join(regdir, "example_func2standard_warp.nii.gz")
        with open(warpfile, "w") as f:
            f.write("dummy warp")

    if with_hires:
        hiresfile = os.path.join(regdir, "highres.nii.gz")
        with open(hiresfile, "w") as f:
            f.write("dummy hires")

    if with_standard:
        stdfile = os.path.join(regdir, "standard.nii.gz")
        with open(stdfile, "w") as f:
            f.write("dummy standard")

    return regdir


def _setup_input_files(tmpdir, subjroot="sub-01_task-rest_desc-"):
    """Create minimal input files that happy2std expects to find."""
    # The glob pattern: inputfileroot + "*app.nii.gz"
    # With inputfileroot = "sub-01_task-rest_desc-", this matches "sub-01_task-rest_desc-normapp.nii.gz"
    # thebase[:-11] strips "_app.nii.gz" (11 chars?) Actually let me check:
    # thebase = "sub-01_task-rest_desc-normapp_info.nii.gz"? No.
    # thefileroot = glob.glob(args.inputfileroot + "*app.nii.gz")[0]
    # So files like "sub-01_task-rest_desc-normapp.nii.gz"
    # absname = os.path.abspath(thefileroot)
    # thebase = basename: "sub-01_task-rest_desc-normapp.nii.gz"
    # subjroot = thebase[:-11] → strips "app.nii.gz" (10 chars)... wait:
    # "app.nii.gz" is 10 chars, but [-11] removes one more char
    # Let me use a file that matches the glob pattern
    appfile = os.path.join(tmpdir, subjroot + "normapp.nii.gz")
    with open(appfile, "w") as f:
        f.write("dummy")

    # Create fmri maps
    # subjroot from code: thebase[:-11] where thebase = "sub-01_task-rest_desc-normapp.nii.gz"
    # len("app.nii.gz") = 10, so [:-11] strips "mapp.nii.gz"?
    # Actually: thebase = "sub-01_task-rest_desc-normapp.nii.gz"
    # thebase[:-11] = "sub-01_task-rest_desc-norm"
    # Then the map is subjroot + "_" + themap + ".nii.gz"
    # = "sub-01_task-rest_desc-norm_normapp_info.nii.gz"
    # That doesn't seem right. Let me trace more carefully.
    # glob matches *app.nii.gz, so appfile matches
    # subjroot = "sub-01_task-rest_desc-normapp.nii.gz"[:-11]
    # = "sub-01_task-rest_desc-no" ... that's 24 chars from 35
    # Actually: len("sub-01_task-rest_desc-normapp.nii.gz") = 36
    # 36 - 11 = 25: "sub-01_task-rest_desc-norm"
    # Then maps are: "sub-01_task-rest_desc-norm_normapp_info.nii.gz"
    # This seems odd but it's what the code does
    derived_root = subjroot + "normapp.nii.gz"
    derived_subjroot = derived_root[:-11]  # "sub-01_task-rest_desc-norm"

    fmrimaps = [
        "normapp_info",
        "processvoxels_mask",
        "arteries_map",
        "veins_map",
        "vessels_map",
        "vessels_mask",
        "app_info",
        "cine_info",
        "maxphase_map",
        "minphase_map",
        "rawapp_info",
    ]
    for m in fmrimaps:
        mapfile = os.path.join(tmpdir, derived_subjroot + "_" + m + ".nii.gz")
        with open(mapfile, "w") as f:
            f.write("dummy")

    return derived_subjroot


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that _get_parser returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "rapidtide2std"

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires all three positional arguments."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(["inputroot"])

    with pytest.raises(SystemExit):
        parser.parse_args(["inputroot", "outputdir"])

    if debug:
        print("parser_required_args passed")


def parser_all_positional(debug=False):
    """Test parser with all three positional arguments."""
    parser = _get_parser()
    args = parser.parse_args(["myinput", "myoutput", "myfeat.feat"])
    assert args.inputfileroot == "myinput"
    assert args.outputdir == "myoutput"
    assert args.featdirectory == "myfeat.feat"

    if debug:
        print("parser_all_positional passed")


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat"])

    assert getattr(args, "all") is False
    assert args.aligntohires is False
    assert args.forcelinear is False
    assert args.onefilename is None
    assert args.preponly is False

    if debug:
        print("parser_defaults passed")


def parser_all_flag(debug=False):
    """Test --all flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--all"])
    assert getattr(args, "all") is True

    if debug:
        print("parser_all_flag passed")


def parser_hires(debug=False):
    """Test --hires flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--hires"])
    assert args.aligntohires is True

    if debug:
        print("parser_hires passed")


def parser_linear(debug=False):
    """Test --linear flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--linear"])
    assert args.forcelinear is True

    if debug:
        print("parser_linear passed")


def parser_onefile(debug=False):
    """Test --onefile option."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--onefile", "myfile"])
    assert args.onefilename == "myfile"

    if debug:
        print("parser_onefile passed")


def parser_fake(debug=False):
    """Test --fake flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--fake"])
    assert args.preponly is True

    if debug:
        print("parser_fake passed")


# ==================== transformmaps tests ====================


def transformmaps_prints_entry(debug=False):
    """Test that transformmaps prints entry diagnostics."""
    if debug:
        print("transformmaps_prints_entry")

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        transformmaps(
            "/some/path",
            "/output",
            "sub-01",
            "/ref/target.nii.gz",
            "/xform.mat",
            "/warp.nii.gz",
            thefmrimaps=None,
            theanatmaps=None,
        )

    output = captured.getvalue()
    assert "entering transformmaps" in output
    assert "/some/path" in output
    assert "sub-01" in output

    if debug:
        print("transformmaps_prints_entry passed")


def transformmaps_no_maps(debug=False):
    """Test transformmaps with no fmri or anat maps (both None)."""
    if debug:
        print("transformmaps_no_maps")

    # Should complete without error when both map lists are None
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        transformmaps(
            "/path",
            "/output",
            "sub-01",
            "/ref.nii.gz",
            "/xform.mat",
            None,
            thefmrimaps=None,
            theanatmaps=None,
        )

    # Just verify it didn't crash
    assert "entering transformmaps" in captured.getvalue()

    if debug:
        print("transformmaps_no_maps passed")


def transformmaps_fmri_file_not_found(debug=False):
    """Test transformmaps skips fmri maps whose input files don't exist."""
    if debug:
        print("transformmaps_fmri_file_not_found")

    with tempfile.TemporaryDirectory() as tmpdir:
        # No actual map files exist in tmpdir
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            transformmaps(
                tmpdir,
                os.path.join(tmpdir, "output"),
                "sub-01",
                "/ref.nii.gz",
                "/xform.mat",
                None,
                thefmrimaps=["desc-maxtime_map", "desc-maxcorr_map"],
                theanatmaps=None,
            )

        # No makeflirtcmd should be called since files don't exist
        output = captured.getvalue()
        assert "entering transformmaps" in output

    if debug:
        print("transformmaps_fmri_file_not_found passed")


def transformmaps_fmri_with_outputtag(debug=False):
    """Test that transformmaps uses outputtag in fmri output filenames."""
    if debug:
        print("transformmaps_fmri_with_outputtag")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        # Create a map file so os.path.isfile returns True
        mapfile = os.path.join(tmpdir, "sub-01_testmap.nii.gz")
        with open(mapfile, "w") as f:
            f.write("dummy")

        flirt_calls = []

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append({"input": inputfile, "output": outputname})
            return ["flirt", "-in", inputfile]

        with (
            patch(
                "rapidtide.workflows.happy2std.tide_extern.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch("subprocess.call"),
        ):
            transformmaps(
                tmpdir,
                outdir,
                "sub-01",
                "/ref.nii.gz",
                "/xform.mat",
                None,
                thefmrimaps=["testmap"],
                theanatmaps=None,
                outputtag="_std_",
            )

        assert len(flirt_calls) == 1
        assert "_std_testmap.nii.gz" in flirt_calls[0]["output"]

    if debug:
        print("transformmaps_fmri_with_outputtag passed")


def transformmaps_anat_hires_copies(debug=False):
    """Test that transformmaps copies anat maps when aligntohires=True."""
    if debug:
        print("transformmaps_anat_hires_copies")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        # Set up a feat-like reg directory with highres.nii.gz
        featdir = os.path.join(tmpdir, "sub-01.feat")
        regdir = os.path.join(featdir, "reg")
        os.makedirs(regdir, exist_ok=True)
        hiresfile = os.path.join(regdir, "highres.nii.gz")
        with open(hiresfile, "w") as f:
            f.write("dummy hires")

        subprocess_calls = []

        def mock_subprocess_call(cmd):
            subprocess_calls.append(cmd)

        with patch("subprocess.call", side_effect=mock_subprocess_call):
            transformmaps(
                tmpdir,
                outdir,
                "sub-01",
                "/ref.nii.gz",
                "/xform.mat",
                None,
                thefmrimaps=None,
                theanatmaps=["highres"],
                outputtag="_hires_",
                xformdir=featdir,
                aligntohires=True,
            )

        # aligntohires=True should use "cp" for anat maps
        assert len(subprocess_calls) == 1
        assert subprocess_calls[0][0] == "cp"
        assert "highres.nii.gz" in subprocess_calls[0][1]

    if debug:
        print("transformmaps_anat_hires_copies passed")


def transformmaps_anat_standard_transforms(debug=False):
    """Test that transformmaps transforms anat maps when aligntohires=False."""
    if debug:
        print("transformmaps_anat_standard_transforms")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        featdir = os.path.join(tmpdir, "sub-01.feat")
        regdir = os.path.join(featdir, "reg")
        os.makedirs(regdir, exist_ok=True)
        hiresfile = os.path.join(regdir, "highres.nii.gz")
        with open(hiresfile, "w") as f:
            f.write("dummy hires")
        hires2std = os.path.join(regdir, "highres2standard.mat")
        with open(hires2std, "w") as f:
            f.write("1 0 0 0\n")

        flirt_calls = []

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append({"xform": xform, "output": outputname})
            return ["flirt", "-in", inputfile]

        with (
            patch(
                "rapidtide.workflows.happy2std.tide_extern.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch("subprocess.call"),
        ):
            transformmaps(
                tmpdir,
                outdir,
                "sub-01",
                "/ref.nii.gz",
                "/xform.mat",
                None,
                thefmrimaps=None,
                theanatmaps=["highres"],
                outputtag="_std_",
                xformdir=featdir,
                aligntohires=False,
            )

        # aligntohires=False should use makeflirtcmd with highres2standard.mat
        assert len(flirt_calls) == 1
        assert "highres2standard.mat" in flirt_calls[0]["xform"]

    if debug:
        print("transformmaps_anat_standard_transforms passed")


# ==================== happy2std workflow tests ====================


def workflow_no_fsldir(debug=False):
    """Test that happy2std raises RuntimeError when FSLDIR is not set."""
    if debug:
        print("workflow_no_fsldir")

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="FSLDIR"):
                happy2std(args)

    if debug:
        print("workflow_no_fsldir passed")


def workflow_missing_xform(debug=False):
    """Test that happy2std exits when transform matrix is missing."""
    if debug:
        print("workflow_missing_xform")

    with tempfile.TemporaryDirectory() as tmpdir:
        regdir = os.path.join(tmpdir, "sub-01.feat", "reg")
        os.makedirs(regdir, exist_ok=True)

        args = _make_default_args(tmpdir)
        with patch.dict(os.environ, {"FSLDIR": "/usr/local/fsl"}):
            with pytest.raises(SystemExit):
                happy2std(args)

    if debug:
        print("workflow_missing_xform passed")


def workflow_missing_reftarget(debug=False):
    """Test that happy2std exits when reference target is missing."""
    if debug:
        print("workflow_missing_reftarget")

    with tempfile.TemporaryDirectory() as tmpdir:
        regdir = os.path.join(tmpdir, "sub-01.feat", "reg")
        os.makedirs(regdir, exist_ok=True)
        xform = os.path.join(regdir, "example_func2standard.mat")
        with open(xform, "w") as f:
            f.write("1 0 0 0\n")

        args = _make_default_args(tmpdir)
        with patch.dict(os.environ, {"FSLDIR": tmpdir}):
            with pytest.raises(SystemExit):
                happy2std(args)

    if debug:
        print("workflow_missing_reftarget passed")


def workflow_onefile_missing(debug=False):
    """Test happy2std with --onefile pointing to a nonexistent file."""
    if debug:
        print("workflow_onefile_missing")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        args = _make_default_args(
            tmpdir,
            onefilename=os.path.join(tmpdir, "nonexistent_file"),
        )

        captured = io.StringIO()
        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch("sys.stdout", captured),
        ):
            with pytest.raises(SystemExit) as exc_info:
                happy2std(args)
            assert exc_info.value.code == 0

        output = captured.getvalue()
        assert "does not exist" in output

    if debug:
        print("workflow_onefile_missing passed")


def workflow_onefile_exists(debug=False):
    """Test happy2std --onefile with an existing file calls makeflirtcmd and exits."""
    if debug:
        print("workflow_onefile_exists")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        # Create the one file so it exists
        onefile = os.path.join(tmpdir, "mymap")
        with open(onefile + ".nii.gz", "w") as f:
            f.write("dummy nifti")

        args = _make_default_args(tmpdir, onefilename=onefile)

        flirt_calls = []

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append(inputfile)
            return ["flirt", "-in", inputfile]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.tide_extern.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch("subprocess.call"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                happy2std(args)
            assert exc_info.value.code == 0

        assert len(flirt_calls) == 1
        assert flirt_calls[0].endswith("mymap.nii.gz")

    if debug:
        print("workflow_onefile_exists passed")


def workflow_hires_setup(debug=False):
    """Test that happy2std with --hires sets correct reference and xform."""
    if debug:
        print("workflow_hires_setup")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False, with_hires=True)
        _setup_input_files(tmpdir)

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, aligntohires=True, preponly=True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": "/usr/local/fsl"}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        assert len(transformmaps_calls) == 1
        call = transformmaps_calls[0]
        # reftarget should point to highres.nii.gz
        assert "highres.nii.gz" in call["args"][3]
        # xformfuncmat should use example_func2highres.mat
        assert "example_func2highres.mat" in call["args"][4]
        # warpfuncfile should be None (no warp for hires)
        assert call["args"][5] is None
        # Only "highres" in anat maps
        assert call["kwargs"]["theanatmaps"] == ["highres"]

    if debug:
        print("workflow_hires_setup passed")


def workflow_standard_setup(debug=False):
    """Test that happy2std without --hires sets correct standard space refs."""
    if debug:
        print("workflow_standard_setup")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        assert len(transformmaps_calls) == 1
        call = transformmaps_calls[0]
        # reftarget should point to MNI152
        assert "MNI152_T1_2mm.nii.gz" in call["args"][3]
        # xformfuncmat should use example_func2standard.mat
        assert "example_func2standard.mat" in call["args"][4]
        # Standard mode has ["highres", "standard"] anat maps
        assert call["kwargs"]["theanatmaps"] == ["highres", "standard"]

    if debug:
        print("workflow_standard_setup passed")


def workflow_nonlinear(debug=False):
    """Test that happy2std detects warp file for nonlinear transformation."""
    if debug:
        print("workflow_nonlinear")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=True)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        captured = io.StringIO()
        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch("sys.stdout", captured),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        output = captured.getvalue()
        assert "nonlinear" in output

        # warpfuncfile should not be None
        warpfuncfile = transformmaps_calls[0]["args"][5]
        assert warpfuncfile is not None
        assert "warp" in warpfuncfile

    if debug:
        print("workflow_nonlinear passed")


def workflow_forcelinear(debug=False):
    """Test that --linear forces linear even when warp file exists."""
    if debug:
        print("workflow_forcelinear")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=True)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, forcelinear=True, preponly=True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        # forcelinear appends "ridiculous_suffix" which makes isfile False → warpfuncfile=None
        warpfuncfile = transformmaps_calls[0]["args"][5]
        assert warpfuncfile is None

    if debug:
        print("workflow_forcelinear passed")


def workflow_default_fmri_maps(debug=False):
    """Test that default (no --all) passes 6 fmri maps to transformmaps."""
    if debug:
        print("workflow_default_fmri_maps")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True)
        setattr(args, "all", False)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        fmrimaps = transformmaps_calls[0]["kwargs"]["thefmrimaps"]
        expected_default = [
            "normapp_info",
            "processvoxels_mask",
            "arteries_map",
            "veins_map",
            "vessels_map",
            "vessels_mask",
        ]
        assert fmrimaps == expected_default

    if debug:
        print("workflow_default_fmri_maps passed")


def workflow_all_fmri_maps(debug=False):
    """Test that --all adds extra fmri maps."""
    if debug:
        print("workflow_all_fmri_maps")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True)
        setattr(args, "all", True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        fmrimaps = transformmaps_calls[0]["kwargs"]["thefmrimaps"]
        # --all adds 5 extra maps
        assert len(fmrimaps) == 11
        assert "app_info" in fmrimaps
        assert "cine_info" in fmrimaps
        assert "maxphase_map" in fmrimaps
        assert "minphase_map" in fmrimaps
        assert "rawapp_info" in fmrimaps

    if debug:
        print("workflow_all_fmri_maps passed")


def workflow_preponly_passed(debug=False):
    """Test that preponly flag is passed through to transformmaps."""
    if debug:
        print("workflow_preponly_passed")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        assert transformmaps_calls[0]["kwargs"]["preponly"] is True

    if debug:
        print("workflow_preponly_passed passed")


def workflow_subjroot_derivation(debug=False):
    """Test that happy2std correctly derives subjroot from glob result."""
    if debug:
        print("workflow_subjroot_derivation")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False)

        # Create the app file that glob will match
        subjroot_prefix = "sub-01_task-rest_desc-"
        appfile = os.path.join(tmpdir, subjroot_prefix + "normapp.nii.gz")
        with open(appfile, "w") as f:
            f.write("dummy")

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True)

        transformmaps_calls = []

        def mock_transformmaps(*a, **kw):
            transformmaps_calls.append({"args": a, "kwargs": kw})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.happy2std.transformmaps",
                side_effect=mock_transformmaps,
            ),
        ):
            happy2std(args)

        # subjroot = thebase[:-11] where thebase = "sub-01_task-rest_desc-normapp.nii.gz"
        # [:-11] strips "mapp.nii.gz" (11 chars) → "sub-01_task-rest_desc-nor"
        passed_subjroot = transformmaps_calls[0]["args"][2]
        expected = "sub-01_task-rest_desc-nor"
        assert passed_subjroot == expected, (
            f"Expected subjroot '{expected}', got '{passed_subjroot}'"
        )

    if debug:
        print("workflow_subjroot_derivation passed")


# ==================== Main test function ====================


def test_happy2std(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_all_positional(debug=debug)
    parser_defaults(debug=debug)
    parser_all_flag(debug=debug)
    parser_hires(debug=debug)
    parser_linear(debug=debug)
    parser_onefile(debug=debug)
    parser_fake(debug=debug)

    # transformmaps tests
    transformmaps_prints_entry(debug=debug)
    transformmaps_no_maps(debug=debug)
    transformmaps_fmri_file_not_found(debug=debug)
    transformmaps_fmri_with_outputtag(debug=debug)
    transformmaps_anat_hires_copies(debug=debug)
    transformmaps_anat_standard_transforms(debug=debug)

    # happy2std workflow tests
    workflow_no_fsldir(debug=debug)
    workflow_missing_xform(debug=debug)
    workflow_missing_reftarget(debug=debug)
    workflow_onefile_missing(debug=debug)
    workflow_onefile_exists(debug=debug)
    workflow_hires_setup(debug=debug)
    workflow_standard_setup(debug=debug)
    workflow_nonlinear(debug=debug)
    workflow_forcelinear(debug=debug)
    workflow_default_fmri_maps(debug=debug)
    workflow_all_fmri_maps(debug=debug)
    workflow_preponly_passed(debug=debug)
    workflow_subjroot_derivation(debug=debug)


if __name__ == "__main__":
    test_happy2std(debug=True)
