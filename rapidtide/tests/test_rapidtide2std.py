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

import numpy as np
import pytest

from rapidtide.workflows.rapidtide2std import _get_parser, rapidtide2std

# ==================== Helpers ====================


def _make_default_args(tmpdir, **overrides):
    """Create default args Namespace for rapidtide2std."""
    defaults = dict(
        inputfileroot=os.path.join(tmpdir, "sub-01_task-rest"),
        outputdir=os.path.join(tmpdir, "output"),
        featdirectory=os.path.join(tmpdir, "sub-01.feat"),
        corrout=False,
        clean=False,
        confound=False,
        aligntohires=False,
        forcelinear=False,
        onefilename=None,
        sequential=True,
        preponly=True,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _setup_feat_dir(tmpdir, with_warp=False, with_hires=True, with_standard=True):
    """Create a minimal feat directory structure with registration files."""
    regdir = os.path.join(tmpdir, "sub-01.feat", "reg")
    os.makedirs(regdir, exist_ok=True)

    # Create transformation matrices
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


def _setup_input_files(tmpdir, subjroot="sub-01_task-rest"):
    """Create minimal input files that rapidtide2std expects to find."""
    # corrout_info file (used by glob to find fileroot)
    corrout = os.path.join(tmpdir, subjroot + "_desc-corrout_info.nii.gz")
    with open(corrout, "w") as f:
        f.write("dummy")

    # runoptions file
    optionsfile = os.path.join(tmpdir, subjroot + "_desc-runoptions_info.json")
    with open(optionsfile, "w") as f:
        f.write('{"filtertype": "lfo"}')

    # timecourse files
    for tc in [
        "desc-initialmovingregressor_timeseries",
        "desc-oversampledmovingregressor_timeseries",
    ]:
        for ext in [".tsv.gz", ".json"]:
            tcfile = os.path.join(tmpdir, subjroot + "_" + tc + ext)
            with open(tcfile, "w") as f:
                f.write("dummy")

    # functional maps
    fmrimaps = [
        "desc-maxtime_map",
        "desc-maxtimerefined_map",
        "desc-timepercentile_map",
        "desc-maxcorr_map",
        "desc-maxcorrrefined_map",
        "desc-maxwidth_map",
        "desc-MTT_map",
        "desc-corrfit_mask",
        "desc-refine_mask",
        "desc-maxcorrsq_map",
        "desc-lfofilterNorm_map",
        "desc-lfofilterCoeff_map",
        "desc-lfofilterInbandVarianceBefore_map",
        "desc-lfofilterInbandVarianceAfter_map",
        "desc-lfofilterInbandVarianceChange_map",
        "desc-plt0p050_mask",
        "desc-plt0p010_mask",
        "desc-plt0p005_mask",
        "desc-plt0p001_mask",
        "desc-corrout_info",
        "desc-lfofilterCleaned_bold",
        "desc-confoundfilterR2_map",
    ]
    for m in fmrimaps:
        mapfile = os.path.join(tmpdir, subjroot + "_" + m + ".nii.gz")
        with open(mapfile, "w") as f:
            f.write("dummy")

    return subjroot


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

    assert args.corrout is False
    assert args.clean is False
    assert args.confound is False
    assert args.aligntohires is False
    assert args.forcelinear is False
    assert args.onefilename is None
    assert args.sequential is False
    assert args.preponly is False
    assert args.debug is False

    if debug:
        print("parser_defaults passed")


def parser_corrout(debug=False):
    """Test --corrout flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--corrout"])
    assert args.corrout is True

    if debug:
        print("parser_corrout passed")


def parser_clean(debug=False):
    """Test --clean flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--clean"])
    assert args.clean is True

    if debug:
        print("parser_clean passed")


def parser_confound(debug=False):
    """Test --confound flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--confound"])
    assert args.confound is True

    if debug:
        print("parser_confound passed")


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


def parser_sequential(debug=False):
    """Test --sequential flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--sequential"])
    assert args.sequential is True

    if debug:
        print("parser_sequential passed")


def parser_fake(debug=False):
    """Test --fake flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--fake"])
    assert args.preponly is True

    if debug:
        print("parser_fake passed")


def parser_debug(debug=False):
    """Test --debug flag."""
    parser = _get_parser()
    args = parser.parse_args(["in", "out", "feat", "--debug"])
    assert args.debug is True

    if debug:
        print("parser_debug passed")


# ==================== rapidtide2std tests ====================


def workflow_no_fsldir(debug=False):
    """Test that rapidtide2std exits when FSLDIR is not set."""
    if debug:
        print("workflow_no_fsldir")

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                rapidtide2std(args)


def workflow_missing_xform(debug=False):
    """Test that rapidtide2std exits when transform matrix is missing."""
    if debug:
        print("workflow_missing_xform")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create feat dir without any transform files
        regdir = os.path.join(tmpdir, "sub-01.feat", "reg")
        os.makedirs(regdir, exist_ok=True)

        args = _make_default_args(tmpdir)
        with patch.dict(os.environ, {"FSLDIR": "/usr/local/fsl"}):
            with pytest.raises(SystemExit):
                rapidtide2std(args)


def workflow_missing_reftarget(debug=False):
    """Test that rapidtide2std exits when reference target is missing."""
    if debug:
        print("workflow_missing_reftarget")

    with tempfile.TemporaryDirectory() as tmpdir:
        regdir = os.path.join(tmpdir, "sub-01.feat", "reg")
        os.makedirs(regdir, exist_ok=True)
        # Create the transform matrix but not the MNI reference
        xform = os.path.join(regdir, "example_func2standard.mat")
        with open(xform, "w") as f:
            f.write("1 0 0 0\n")

        args = _make_default_args(tmpdir)
        # Use a fake FSLDIR so the MNI reference won't exist
        with patch.dict(os.environ, {"FSLDIR": tmpdir}):
            with pytest.raises(SystemExit):
                rapidtide2std(args)


def workflow_onefile_missing(debug=False):
    """Test rapidtide2std with --onefile pointing to a nonexistent file."""
    if debug:
        print("workflow_onefile_missing")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir)

        # Create a fake FSLDIR with standard reference
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
            with pytest.raises(SystemExit):
                rapidtide2std(args)

        output = captured.getvalue()
        assert "does not exist" in output

    if debug:
        print("workflow_onefile_missing passed")


def workflow_onefile_exists(debug=False):
    """Test rapidtide2std with --onefile pointing to an existing file."""
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

        # Create the one file to process
        onefile = os.path.join(tmpdir, "mymap")
        with open(onefile + ".nii.gz", "w") as f:
            f.write("dummy nifti")

        args = _make_default_args(tmpdir, onefilename=onefile)

        runcmd_calls = []

        def mock_runcmd(cmd, fake=False):
            runcmd_calls.append(cmd)

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                return_value=["flirt", "-in", "input"],
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                rapidtide2std(args)
            # --onefile path always calls sys.exit(0)
            assert exc_info.value.code == 0

        assert len(runcmd_calls) == 1

    if debug:
        print("workflow_onefile_exists passed")


def workflow_linear_standard(debug=False):
    """Test full workflow with linear transformation to standard space."""
    if debug:
        print("workflow_linear_standard")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False, with_hires=True, with_standard=True)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True, sequential=True)

        runcmd_calls = []
        flirt_calls = []
        written_json = {}

        def mock_runcmd(cmd, fake=False):
            runcmd_calls.append(cmd)

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append({
                "input": inputfile,
                "target": targetname,
                "xform": xform,
                "output": outputname,
                "kwargs": kwargs,
            })
            return ["flirt", "-in", inputfile, "-out", outputname]

        def mock_readoptionsfile(fname):
            return {"filtertype": "lfo", "key1": "val1"}

        def mock_writedicttojson(thedict, fname):
            written_json[fname] = thedict

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                side_effect=mock_readoptionsfile,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.writedicttojson",
                side_effect=mock_writedicttojson,
            ),
        ):
            rapidtide2std(args)

        # Check that options file was written with source key added
        assert len(written_json) == 1
        json_path = list(written_json.keys())[0]
        assert "_std_" in json_path
        json_data = list(written_json.values())[0]
        assert "rapidtide2std_source" in json_data

        # Check timecourse copy commands (2 files x 2 extensions = 4)
        cp_calls = [c for c in runcmd_calls if c[0] == "cp"]
        assert len(cp_calls) == 4

        # Check flirt calls for functional maps (19 default maps)
        assert len(flirt_calls) >= 19

        # All flirt calls should have warpfile=None (linear, no warp file)
        for call in flirt_calls:
            assert call["kwargs"].get("warpfile") is None

    if debug:
        print("workflow_linear_standard passed")


def workflow_nonlinear_standard(debug=False):
    """Test full workflow with nonlinear transformation (warp file present)."""
    if debug:
        print("workflow_nonlinear_standard")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=True, with_hires=True, with_standard=True)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True, sequential=True)

        flirt_calls = []

        def mock_runcmd(cmd, fake=False):
            pass

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append({"kwargs": kwargs})
            return ["applywarp", "-in", inputfile]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # Functional map flirt calls should have a warpfile set (nonlinear)
        func_flirt = [c for c in flirt_calls if c["kwargs"].get("warpfile") is not None]
        assert len(func_flirt) > 0, "Expected nonlinear transforms with warpfile set"

    if debug:
        print("workflow_nonlinear_standard passed")


def workflow_forcelinear(debug=False):
    """Test that --linear forces linear transformation even when warp file exists."""
    if debug:
        print("workflow_forcelinear")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=True, with_hires=True, with_standard=True)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, forcelinear=True, preponly=True, sequential=True)

        flirt_calls = []

        def mock_runcmd(cmd, fake=False):
            pass

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append({"kwargs": kwargs})
            return ["flirt", "-in", inputfile]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # All functional map transforms should have warpfile=None (forced linear)
        for call in flirt_calls:
            assert call["kwargs"].get("warpfile") is None

    if debug:
        print("workflow_forcelinear passed")


def workflow_hires(debug=False):
    """Test workflow with --hires (align to highres anatomical)."""
    if debug:
        print("workflow_hires")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False, with_hires=True)
        _setup_input_files(tmpdir)

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(
            tmpdir, aligntohires=True, preponly=True, sequential=True
        )

        runcmd_calls = []
        flirt_calls = []
        written_json = {}

        def mock_runcmd(cmd, fake=False):
            runcmd_calls.append(cmd)

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls.append({
                "target": targetname,
                "xform": xform,
                "output": outputname,
            })
            return ["flirt", "-in", inputfile]

        def mock_writedicttojson(thedict, fname):
            written_json[fname] = thedict

        with (
            patch.dict(os.environ, {"FSLDIR": "/usr/local/fsl"}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.writedicttojson",
                side_effect=mock_writedicttojson,
            ),
        ):
            rapidtide2std(args)

        # Hires mode: reference target should be highres.nii.gz
        for call in flirt_calls:
            assert "highres.nii.gz" in call["target"]

        # Hires mode: xform should use example_func2highres.mat
        for call in flirt_calls:
            assert "example_func2highres.mat" in call["xform"]

        # Output files should have _hires_ tag
        json_path = list(written_json.keys())[0]
        assert "_hires_" in json_path

        # Anatomic map handling: hires mode copies highres instead of transforming
        anat_cp_calls = [
            c for c in runcmd_calls if c[0] == "cp" and "highres" in c[1]
        ]
        assert len(anat_cp_calls) >= 1

    if debug:
        print("workflow_hires passed")


def workflow_corrout_clean_confound(debug=False):
    """Test that --corrout, --clean, --confound add extra maps."""
    if debug:
        print("workflow_corrout_clean_confound")

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

        # First run without extras
        args_no_extras = _make_default_args(tmpdir, preponly=True, sequential=True)
        flirt_calls_no_extras = []

        def mock_runcmd(cmd, fake=False):
            pass

        def capture_flirt_no_extras(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls_no_extras.append(inputfile)
            return ["flirt"]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=capture_flirt_no_extras,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args_no_extras)

        # Now run with all extras enabled
        args_extras = _make_default_args(
            tmpdir, corrout=True, clean=True, confound=True, preponly=True, sequential=True
        )
        flirt_calls_extras = []

        def capture_flirt_extras(inputfile, targetname, xform, outputname, **kwargs):
            flirt_calls_extras.append(inputfile)
            return ["flirt"]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=capture_flirt_extras,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args_extras)

        # Extras should produce more flirt calls (corrout, clean, confound = 3 extra maps)
        assert len(flirt_calls_extras) > len(flirt_calls_no_extras), (
            f"Expected more flirt calls with extras ({len(flirt_calls_extras)}) "
            f"than without ({len(flirt_calls_no_extras)})"
        )

        # Verify the specific extra maps are present
        extras_inputs = " ".join(flirt_calls_extras)
        assert "corrout_info" in extras_inputs
        assert "lfofilterCleaned_bold" in extras_inputs
        assert "confoundfilterR2_map" in extras_inputs

    if debug:
        print("workflow_corrout_clean_confound passed")


def workflow_debug_output(debug=False):
    """Test that --debug prints args."""
    if debug:
        print("workflow_debug_output")

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

        args = _make_default_args(tmpdir, debug=True, preponly=True, sequential=True)

        captured = io.StringIO()

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch("sys.stdout", captured),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                return_value=["flirt"],
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        output = captured.getvalue()
        # Debug mode should print the Namespace args
        assert "Namespace" in output or "debug=True" in output

    if debug:
        print("workflow_debug_output passed")


def workflow_preponly(debug=False):
    """Test that --fake (preponly) passes fake=True to runcmd."""
    if debug:
        print("workflow_preponly")

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

        args = _make_default_args(tmpdir, preponly=True, sequential=True)

        runcmd_kwargs = []

        def mock_runcmd(cmd, fake=False):
            runcmd_kwargs.append({"fake": fake})

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                return_value=["flirt"],
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # All runcmd calls should have fake=True
        assert len(runcmd_kwargs) > 0
        for call in runcmd_kwargs:
            assert call["fake"] is True

    if debug:
        print("workflow_preponly passed")


def workflow_sequential_flag(debug=False):
    """Test that --sequential passes cluster=False to makeflirtcmd."""
    if debug:
        print("workflow_sequential_flag")

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

        args = _make_default_args(tmpdir, sequential=True, preponly=True)

        flirt_kwargs = []

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_kwargs.append(kwargs)
            return ["flirt"]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch("rapidtide.workflows.rapidtide2std.tide_exttools.runcmd"),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # sequential=True means cluster=False (cluster = not args.sequential)
        for kw in flirt_kwargs:
            assert kw.get("cluster") is False

    if debug:
        print("workflow_sequential_flag passed")


def workflow_cluster_flag(debug=False):
    """Test that without --sequential, cluster=True is passed to makeflirtcmd."""
    if debug:
        print("workflow_cluster_flag")

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

        args = _make_default_args(tmpdir, sequential=False, preponly=True)

        flirt_kwargs = []

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_kwargs.append(kwargs)
            return ["flirt"]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch("rapidtide.workflows.rapidtide2std.tide_exttools.runcmd"),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # sequential=False means cluster=True
        for kw in flirt_kwargs:
            assert kw.get("cluster") is True

    if debug:
        print("workflow_cluster_flag passed")


def workflow_output_tags_standard(debug=False):
    """Test that standard space outputs have _std_ tag."""
    if debug:
        print("workflow_output_tags_standard")

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

        args = _make_default_args(tmpdir, preponly=True, sequential=True)

        flirt_outputs = []

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_outputs.append(outputname)
            return ["flirt"]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch("rapidtide.workflows.rapidtide2std.tide_exttools.runcmd"),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # All output filenames should contain _std_
        for outname in flirt_outputs:
            assert "_std_" in outname, f"Expected '_std_' in output name: {outname}"

    if debug:
        print("workflow_output_tags_standard passed")


def workflow_anat_maps_standard(debug=False):
    """Test that anatomic maps are transformed in standard mode."""
    if debug:
        print("workflow_anat_maps_standard")

    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_feat_dir(tmpdir, with_warp=False, with_hires=True, with_standard=True)
        _setup_input_files(tmpdir)

        fsldir = os.path.join(tmpdir, "fsl")
        stddir = os.path.join(fsldir, "data", "standard")
        os.makedirs(stddir, exist_ok=True)
        reftarget = os.path.join(stddir, "MNI152_T1_2mm.nii.gz")
        with open(reftarget, "w") as f:
            f.write("dummy")

        outdir = os.path.join(tmpdir, "output")
        os.makedirs(outdir, exist_ok=True)

        args = _make_default_args(tmpdir, preponly=True, sequential=True)

        flirt_outputs = []
        runcmd_calls = []

        def mock_runcmd(cmd, fake=False):
            runcmd_calls.append(cmd)

        def mock_makeflirtcmd(inputfile, targetname, xform, outputname, **kwargs):
            flirt_outputs.append(outputname)
            return ["flirt", "-in", inputfile]

        with (
            patch.dict(os.environ, {"FSLDIR": fsldir}),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.runcmd",
                side_effect=mock_runcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_exttools.makeflirtcmd",
                side_effect=mock_makeflirtcmd,
            ),
            patch(
                "rapidtide.workflows.rapidtide2std.tide_io.readoptionsfile",
                return_value={"filtertype": "lfo"},
            ),
            patch("rapidtide.workflows.rapidtide2std.tide_io.writedicttojson"),
        ):
            rapidtide2std(args)

        # Standard mode transforms both highres and standard anatomic maps
        # "standard" is renamed to "anat" in output
        anat_outputs = [o for o in flirt_outputs if "anat" in o or "highres" in o]
        assert len(anat_outputs) >= 1, "Expected at least one anatomic map transform"

    if debug:
        print("workflow_anat_maps_standard passed")


# ==================== Main test function ====================


def test_rapidtide2std(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_all_positional(debug=debug)
    parser_defaults(debug=debug)
    parser_corrout(debug=debug)
    parser_clean(debug=debug)
    parser_confound(debug=debug)
    parser_hires(debug=debug)
    parser_linear(debug=debug)
    parser_onefile(debug=debug)
    parser_sequential(debug=debug)
    parser_fake(debug=debug)
    parser_debug(debug=debug)

    # rapidtide2std workflow tests
    workflow_no_fsldir(debug=debug)
    workflow_missing_xform(debug=debug)
    workflow_missing_reftarget(debug=debug)
    workflow_onefile_missing(debug=debug)
    workflow_onefile_exists(debug=debug)
    workflow_linear_standard(debug=debug)
    workflow_nonlinear_standard(debug=debug)
    workflow_forcelinear(debug=debug)
    workflow_hires(debug=debug)
    workflow_corrout_clean_confound(debug=debug)
    workflow_debug_output(debug=debug)
    workflow_preponly(debug=debug)
    workflow_sequential_flag(debug=debug)
    workflow_cluster_flag(debug=debug)
    workflow_output_tags_standard(debug=debug)
    workflow_anat_maps_standard(debug=debug)


if __name__ == "__main__":
    test_rapidtide2std(debug=True)
