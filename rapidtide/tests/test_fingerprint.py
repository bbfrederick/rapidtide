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
import copy
import os
import sys
import tempfile
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.util as tide_util
import rapidtide.workflows.fingerprint as fp

# ==================== Helpers ====================


def _make_synthetic_atlas(shape=(10, 12, 8), numregions=3):
    """Create a synthetic atlas with numregions territories."""
    rng = np.random.RandomState(42)
    atlas = np.zeros(shape, dtype=int)
    nvox = shape[0] * shape[1] * shape[2]
    flat = atlas.reshape(nvox)
    # Assign each voxel to a random region (1..numregions), leave ~10% as 0
    for i in range(nvox):
        if rng.rand() > 0.1:
            flat[i] = rng.randint(1, numregions + 1)
    return atlas


def _make_synthetic_template(atlas, slope=2.0, offset=1.0):
    """Create a synthetic template that varies linearly within each region."""
    rng = np.random.RandomState(99)
    template = np.zeros_like(atlas, dtype=float)
    for region in range(1, np.max(atlas) + 1):
        voxels = np.where(atlas == region)
        nvox = len(voxels[0])
        template[voxels] = slope * rng.rand(nvox) + offset
    return template


def _make_synthetic_inputmap(template, atlas, noise_level=0.1):
    """Create a synthetic input map that is a linear function of the template + noise."""
    rng = np.random.RandomState(7)
    inputmap = np.zeros_like(template, dtype=float)
    for region in range(1, np.max(atlas) + 1):
        voxels = np.where(atlas == region)
        # Each region has a different slope and offset
        region_slope = 1.0 + 0.5 * region
        region_offset = -0.3 * region
        inputmap[voxels] = region_slope * template[voxels] + region_offset
    inputmap += noise_level * rng.randn(*inputmap.shape)
    return inputmap


def _make_nifti_file(tmpdir, data, name):
    """Write a NIfTI file and return the path."""
    img = nib.Nifti1Image(data, affine=np.eye(4))
    filepath = os.path.join(tmpdir, name)
    nib.save(img, filepath)
    return filepath


def _make_labelled_atlas(numregions, planesperregion=2):
    """Build a small atlas whose maximum value is exactly numregions.

    ``fingerprint`` sizes its output arrays from the region name file but sizes the fit from
    ``np.max(atlas)``, so the two must agree for the results to be transferable.

    Parameters
    ----------
    numregions : int
        Number of regions the atlas should contain.
    planesperregion : int, optional
        Thickness of each region along the first axis.  Default is 2.

    Returns
    -------
    ndarray
        Integer atlas of shape (numregions * planesperregion, 4, 4).
    """
    atlas = np.zeros((numregions * planesperregion, 4, 4), dtype=int)
    for region in range(1, numregions + 1):
        atlas[(region - 1) * planesperregion : region * planesperregion, :, :] = region
    return atlas


def _make_write_only_tmpdir_atlas(tmpdir, numregions, name="atlas"):
    """Write a synthetic atlas NIfTI plus its matching region name file.

    Parameters
    ----------
    tmpdir : str
        Directory to write into.
    numregions : int
        Number of regions in the atlas.
    name : str, optional
        Base name for the atlas pair.  Default is "atlas".

    Returns
    -------
    tuple of (str, ndarray)
        Path to the written atlas NIfTI and the atlas array itself.
    """
    atlas = _make_labelled_atlas(numregions)
    atlaspath = _make_nifti_file(tmpdir, atlas.astype(np.float64), f"{name}.nii.gz")
    with open(os.path.join(tmpdir, f"{name}.txt"), "w") as thefile:
        for i in range(numregions):
            thefile.write(f"Region_{i + 1}\n")
    return atlaspath, atlas


def _read_as_tuple(tmpdir, data, name):
    """Write an array to NIfTI and read it back in readfromnifti's 5-tuple form.

    Parameters
    ----------
    tmpdir : str
        Directory to stage the temporary NIfTI file in.
    data : ndarray
        Data to round trip.
    name : str
        File name to use.

    Returns
    -------
    tuple
        The (nim, data, header, dims, sizes) tuple returned by ``tide_io.readfromnifti``.
    """
    return tide_io.readfromnifti(_make_nifti_file(tmpdir, data.astype(np.float64), name))


def _run_with_stubbed_reads(tmpdir, whichtemplate, whichatlas, numregions, **kwargs):
    """Run fingerprint with all NIfTI reads stubbed, recording the paths it requests.

    This exercises the atlas and template filename resolution logic without needing the
    full sized reference volumes, which keeps the test fast while still asserting that the
    correct reference files are selected.

    Parameters
    ----------
    tmpdir : str
        Directory for staged inputs and generated outputs.
    whichtemplate : str
        Template selector passed through to ``fingerprint``.
    whichatlas : str
        Atlas selector passed through to ``fingerprint``.
    numregions : int
        Number of regions in the region name file of the selected atlas.
    **kwargs
        Additional keyword arguments forwarded to ``fingerprint``.

    Returns
    -------
    list of str
        The filenames passed to ``readfromnifti``, in call order.
    """
    rng = np.random.RandomState(3)
    atlas = _make_labelled_atlas(numregions)
    shape = atlas.shape
    atlastuple = _read_as_tuple(tmpdir, atlas, "stub_atlas.nii.gz")
    templatetuple = _read_as_tuple(tmpdir, rng.rand(*shape) + 1.0, "stub_template.nii.gz")
    maptuple = _read_as_tuple(tmpdir, rng.randn(*shape), "stub_map.nii.gz")

    sequence = [atlastuple]
    if whichtemplate != "constant":
        sequence.append(templatetuple)
    sequence.append(maptuple)

    recorded = []

    def _stubreader(thename, **readkwargs):
        recorded.append(thename)
        return copy.deepcopy(sequence[len(recorded) - 1])

    with patch.object(tide_io, "readfromnifti", side_effect=_stubreader):
        fp.fingerprint(
            "themap.nii.gz",
            whichtemplate,
            whichatlas,
            os.path.join(tmpdir, "stubout"),
            kwargs.pop("fitorder", 0),
            **kwargs,
        )
    return recorded


# ==================== _get_parser tests ====================


def get_parser_returns_parser(debug=False):
    """Test _get_parser returns an ArgumentParser."""
    if debug:
        print("get_parser_returns_parser")
    import argparse

    parser = fp._get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_required_args(debug=False):
    """Test _get_parser requires inputfile and outputroot."""
    if debug:
        print("get_parser_required_args")
    parser = fp._get_parser()
    # Should fail without required args
    try:
        parser.parse_args([])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_defaults(debug=False):
    """Test _get_parser default values."""
    if debug:
        print("get_parser_defaults")
    parser = fp._get_parser()
    # Use a tempfile to satisfy the is_valid_file check
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/outputroot"])
        assert args.atlas == "JHU1"
        assert args.fitorder == 1
        assert not args.nointercept
        assert not args.limittomask
        assert args.template == "lag"
        assert args.entropybins == 101
        assert args.entropyrange is None
        assert not args.debug
        assert args.includespec is None
        assert args.excludespec is None
        assert args.extramaskname is None
        assert args.customatlas is None


def get_parser_atlas_choices(debug=False):
    """Test _get_parser accepts all valid atlas choices."""
    if debug:
        print("get_parser_atlas_choices")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        for choice in ["ASPECTS", "ATT", "JHU1", "JHU2"]:
            args = parser.parse_args([tf.name, "/tmp/out", "--atlas", choice])
            assert args.atlas == choice


def get_parser_template_choices(debug=False):
    """Test _get_parser accepts all valid template choices."""
    if debug:
        print("get_parser_template_choices")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        for choice in ["lag", "strength", "sigma", "constant"]:
            args = parser.parse_args([tf.name, "/tmp/out", "--template", choice])
            assert args.template == choice


def get_parser_fitorder(debug=False):
    """Test _get_parser accepts --fitorder."""
    if debug:
        print("get_parser_fitorder")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--fitorder", "3"])
        assert args.fitorder == 3


def get_parser_nointercept(debug=False):
    """Test _get_parser accepts --nointercept."""
    if debug:
        print("get_parser_nointercept")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--nointercept"])
        assert args.nointercept


def get_parser_limittomask(debug=False):
    """Test _get_parser accepts --limittomask."""
    if debug:
        print("get_parser_limittomask")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--limittomask"])
        assert args.limittomask


def get_parser_entropybins(debug=False):
    """Test _get_parser accepts --entropybins."""
    if debug:
        print("get_parser_entropybins")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--entropybins", "51"])
        assert args.entropybins == 51


def get_parser_entropyrange(debug=False):
    """Test _get_parser accepts --entropyrange."""
    if debug:
        print("get_parser_entropyrange")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--entropyrange", "-5.0", "5.0"])
        assert args.entropyrange == [-5.0, 5.0]


def get_parser_debug(debug=False):
    """Test _get_parser accepts --debug."""
    if debug:
        print("get_parser_debug")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--debug"])
        assert args.debug


def get_parser_includemask(debug=False):
    """Test _get_parser accepts --includemask."""
    if debug:
        print("get_parser_includemask")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--includemask", "somemask.nii.gz:1,2,3"])
        assert args.includespec == "somemask.nii.gz:1,2,3"


def get_parser_excludemask(debug=False):
    """Test _get_parser accepts --excludemask."""
    if debug:
        print("get_parser_excludemask")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--excludemask", "excl.nii.gz"])
        assert args.excludespec == "excl.nii.gz"


def get_parser_invalid_atlas(debug=False):
    """Test _get_parser rejects invalid atlas."""
    if debug:
        print("get_parser_invalid_atlas")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        try:
            parser.parse_args([tf.name, "/tmp/out", "--atlas", "INVALID"])
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass


def get_parser_invalid_template(debug=False):
    """Test _get_parser rejects invalid template."""
    if debug:
        print("get_parser_invalid_template")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        try:
            parser.parse_args([tf.name, "/tmp/out", "--template", "INVALID"])
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass


def get_parser_fullpaths_expands(debug=False):
    """Test the FullPaths action turns a relative output root into an absolute path."""
    if debug:
        print("get_parser_fullpaths_expands")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "someroot"])
        assert os.path.isabs(args.outputroot)
        assert args.outputroot.endswith("someroot")


def get_parser_fullpaths_empty_string(debug=False):
    """Test the FullPaths action maps an empty path to the __EMPTY__ sentinel."""
    if debug:
        print("get_parser_fullpaths_empty_string")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, ""])
        assert args.outputroot == "__EMPTY__"


def get_parser_rejects_missing_input(debug=False):
    """Test _get_parser rejects an input file that does not exist."""
    if debug:
        print("get_parser_rejects_missing_input")
    parser = fp._get_parser()
    try:
        parser.parse_args(["/nonexistent/nosuchfile.nii.gz", "/tmp/out"])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_customatlas(debug=False):
    """Test _get_parser accepts --customatlas for an existing file."""
    if debug:
        print("get_parser_customatlas")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--customatlas", tf.name])
        assert args.customatlas == tf.name


def get_parser_extramask(debug=False):
    """Test _get_parser accepts --extramask for an existing file."""
    if debug:
        print("get_parser_extramask")
    parser = fp._get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
        args = parser.parse_args([tf.name, "/tmp/out", "--extramask", tf.name])
        assert args.extramaskname == tf.name


# ==================== atlas and template selection tests ====================


def fingerprint_selects_builtin_atlases(debug=False):
    """Test each built-in atlas name resolves to the expected pair of reference files."""
    if debug:
        print("fingerprint_selects_builtin_atlases")
    referencedir = tide_util.findreferencedir()
    cases = [
        ("ASPECTS", "ASPECTS_2mm.nii.gz", 24),
        ("ATT", "ATTbasedFlowTerritories_split_2mm.nii.gz", 18),
        ("JHU1", "JHU-ArterialTerritoriesNoVent-LVL1_space-MNI152NLin6Asym_2mm.nii.gz", 30),
        ("JHU2", "JHU-ArterialTerritoriesNoVent-LVL2_space-MNI152NLin6Asym_2mm.nii.gz", 8),
    ]
    for whichatlas, expectedfile, numregions in cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorded = _run_with_stubbed_reads(tmpdir, "constant", whichatlas, numregions)
            if debug:
                print(f"  {whichatlas} -> {recorded}")
            assert recorded[0] == os.path.join(referencedir, expectedfile)
            # constant template means no template file is read, so the map read comes second
            assert len(recorded) == 2
            assert recorded[1] == "themap.nii.gz"


def fingerprint_selects_templates(debug=False):
    """Test each named template resolves to the expected HCP1200 reference file."""
    if debug:
        print("fingerprint_selects_templates")
    referencedir = tide_util.findreferencedir()
    for whichtemplate, expectedfile in [
        ("lag", "HCP1200_lag_2mm.nii.gz"),
        ("strength", "HCP1200_strength_2mm.nii.gz"),
        ("sigma", "HCP1200_sigma_2mm.nii.gz"),
    ]:
        with tempfile.TemporaryDirectory() as tmpdir:
            # run with debug on so the template read is reported, mirroring the atlas read
            recorded = _run_with_stubbed_reads(
                tmpdir, whichtemplate, "JHU2", 8, fitorder=1, debug=True
            )
            if debug:
                print(f"  {whichtemplate} -> {recorded}")
            assert len(recorded) == 3
            assert recorded[1] == os.path.join(referencedir, expectedfile)


def fingerprint_user_atlas_derives_region_file(debug=False):
    """Test a USER_ atlas derives its region name file by swapping the extension."""
    if debug:
        print("fingerprint_user_atlas_derives_region_file")
    with tempfile.TemporaryDirectory() as tmpdir:
        atlaspath, _ = _make_write_only_tmpdir_atlas(tmpdir, 3, name="myatlas")
        recorded = _run_with_stubbed_reads(tmpdir, "constant", f"USER_{atlaspath}", 3)
        assert recorded[0] == atlaspath
        # the output root records the atlas as USER, not the full path
        assert os.path.exists(
            os.path.join(tmpdir, "stubout_template-constant_atlas-USER_O0_maskmap.nii.gz")
        )


def fingerprint_illegal_template(debug=False):
    """Test fingerprint exits on an unrecognized template name."""
    if debug:
        print("fingerprint_illegal_template")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            fp.fingerprint("themap.nii.gz", "bogus", "JHU1", os.path.join(tmpdir, "out"), 0)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass


def fingerprint_illegal_atlas(debug=False):
    """Test fingerprint exits on an unrecognized atlas name."""
    if debug:
        print("fingerprint_illegal_atlas")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            fp.fingerprint("themap.nii.gz", "constant", "bogus", os.path.join(tmpdir, "out"), 0)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass


# ==================== territorydecomp tests ====================


def territorydecomp_3d_basic(debug=False):
    """Test territorydecomp with 3D input, linear fit."""
    if debug:
        print("territorydecomp_3d_basic")
    shape = (10, 12, 8)
    atlas = _make_synthetic_atlas(shape, numregions=3)
    template = _make_synthetic_template(atlas)
    inputmap = _make_synthetic_inputmap(template, atlas, noise_level=0.01)

    fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        inputmap,
        template,
        atlas,
        fitorder=1,
        intercept=True,
        debug=debug,
    )
    assert fitmap.shape == inputmap.shape
    assert thecoffs.shape == (1, 3, 2)  # (nummaps=1, numregions=3, fitorder+1=2)
    assert theR2s.shape == (1, 3)
    # R2 should be high (good fit since input is linear function of template)
    for r in range(3):
        if debug:
            print(f"  region {r+1}: R2={theR2s[0, r]:.4f}, coffs={thecoffs[0, r, :]}")
        assert theR2s[0, r] > 0.9, f"R2 too low for region {r+1}: {theR2s[0, r]}"


def territorydecomp_3d_no_intercept(debug=False):
    """Test territorydecomp without intercept."""
    if debug:
        print("territorydecomp_3d_no_intercept")
    shape = (10, 12, 8)
    atlas = _make_synthetic_atlas(shape, numregions=2)
    template = _make_synthetic_template(atlas)
    inputmap = _make_synthetic_inputmap(template, atlas, noise_level=0.01)

    fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        inputmap,
        template,
        atlas,
        fitorder=1,
        intercept=False,
        debug=debug,
    )
    # mlregress always returns fitorder+1 coffs (intercept prepended, zero when intercept=False)
    assert thecoffs.shape == (1, 2, 2)
    assert theR2s.shape == (1, 2)
    # With intercept=False, the first coefficient (intercept) should be ~0
    for r in range(2):
        assert abs(thecoffs[0, r, 0]) < 1e-10


def territorydecomp_3d_fitorder0(debug=False):
    """Test territorydecomp with fitorder=0 (constant fit)."""
    if debug:
        print("territorydecomp_3d_fitorder0")
    shape = (8, 8, 6)
    atlas = _make_synthetic_atlas(shape, numregions=2)
    template = np.zeros(shape, dtype=float)
    # Just set up constant regions
    inputmap = np.zeros(shape, dtype=float)
    for region in range(1, 3):
        voxels = np.where(atlas == region)
        inputmap[voxels] = region * 5.0

    fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        inputmap,
        template,
        atlas,
        fitorder=0,
        intercept=True,
        debug=debug,
    )
    assert thecoffs.shape == (1, 2, 1)
    # R2 should be 1.0 for constant data
    for r in range(2):
        assert abs(theR2s[0, r] - 1.0) < 1e-10
        assert abs(thecoffs[0, r, 0] - (r + 1) * 5.0) < 1e-10


def territorydecomp_4d(debug=False):
    """Test territorydecomp with 4D input."""
    if debug:
        print("territorydecomp_4d")
    shape3d = (8, 8, 6)
    nummaps = 3
    atlas = _make_synthetic_atlas(shape3d, numregions=2)
    template = _make_synthetic_template(atlas)
    rng = np.random.RandomState(42)
    inputmap = np.zeros(shape3d + (nummaps,), dtype=float)
    for m in range(nummaps):
        inputmap[:, :, :, m] = (
            _make_synthetic_inputmap(template, atlas, noise_level=0.01) + m * 0.5
        )

    fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        inputmap,
        template,
        atlas,
        fitorder=1,
        intercept=True,
        debug=debug,
    )
    assert fitmap.shape == inputmap.shape
    assert thecoffs.shape == (nummaps, 2, 2)
    assert theR2s.shape == (nummaps, 2)


def territorydecomp_with_mask(debug=False):
    """Test territorydecomp with a mask."""
    if debug:
        print("territorydecomp_with_mask")
    shape = (10, 12, 8)
    atlas = _make_synthetic_atlas(shape, numregions=3)
    template = _make_synthetic_template(atlas)
    inputmap = _make_synthetic_inputmap(template, atlas, noise_level=0.01)
    # Create a mask that excludes half the voxels
    mask = np.ones(shape, dtype=float)
    mask[:5, :, :] = 0.0

    fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        inputmap,
        template,
        atlas,
        inputmask=mask,
        fitorder=1,
        intercept=True,
        debug=debug,
    )
    assert fitmap.shape == inputmap.shape
    assert thecoffs.shape == (1, 3, 2)
    # Masked region should have zero in fitmap
    assert np.all(fitmap[:5, :, :] == 0.0)


def territorydecomp_higher_order(debug=False):
    """Test territorydecomp with fitorder=2."""
    if debug:
        print("territorydecomp_higher_order")
    shape = (10, 12, 8)
    atlas = _make_synthetic_atlas(shape, numregions=2)
    template = _make_synthetic_template(atlas)
    inputmap = _make_synthetic_inputmap(template, atlas, noise_level=0.01)

    fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        inputmap,
        template,
        atlas,
        fitorder=2,
        intercept=True,
        debug=debug,
    )
    assert thecoffs.shape == (1, 2, 3)  # fitorder=2, intercept → 3 coffs
    assert theR2s.shape == (1, 2)


# ==================== territorystats tests ====================


def territorystats_3d_basic(debug=False):
    """Test territorystats with 3D input returns correct structure."""
    if debug:
        print("territorystats_3d_basic")
    shape = (10, 12, 8)
    numregions = 3
    atlas = _make_synthetic_atlas(shape, numregions=numregions)
    rng = np.random.RandomState(42)
    inputmap = rng.randn(*shape).astype(float)

    result = tide_fit.territorystats(inputmap, atlas, debug=debug)
    assert len(result) == 9
    statsmap, means, stds, medians, mads, variances, skewnesses, kurtoses, entropies = result
    assert statsmap.shape == inputmap.shape
    assert means.shape == (1, numregions)
    assert stds.shape == (1, numregions)
    assert medians.shape == (1, numregions)
    assert mads.shape == (1, numregions)
    assert variances.shape == (1, numregions)
    assert skewnesses.shape == (1, numregions)
    assert kurtoses.shape == (1, numregions)
    assert entropies.shape == (1, numregions)


def territorystats_known_values(debug=False):
    """Test territorystats returns correct mean/std for known data."""
    if debug:
        print("territorystats_known_values")
    shape = (4, 4, 4)
    atlas = np.ones(shape, dtype=int)
    inputmap = np.full(shape, 5.0, dtype=float)

    result = tide_fit.territorystats(inputmap, atlas, debug=debug)
    _, means, stds, medians, mads, variances, _, _, _ = result
    assert abs(means[0, 0] - 5.0) < 1e-10
    assert abs(stds[0, 0]) < 1e-10
    assert abs(medians[0, 0] - 5.0) < 1e-10
    assert abs(variances[0, 0]) < 1e-10


def territorystats_4d(debug=False):
    """Test territorystats with 4D input."""
    if debug:
        print("territorystats_4d")
    shape3d = (6, 6, 4)
    nummaps = 2
    numregions = 2
    atlas = _make_synthetic_atlas(shape3d, numregions=numregions)
    rng = np.random.RandomState(42)
    inputmap = rng.randn(*shape3d, nummaps).astype(float)

    result = tide_fit.territorystats(inputmap, atlas, debug=debug)
    _, means, stds, medians, mads, variances, skewnesses, kurtoses, entropies = result
    assert means.shape == (nummaps, numregions)
    assert stds.shape == (nummaps, numregions)


def territorystats_with_mask(debug=False):
    """Test territorystats with a mask."""
    if debug:
        print("territorystats_with_mask")
    shape = (6, 6, 4)
    numregions = 2
    atlas = _make_synthetic_atlas(shape, numregions=numregions)
    rng = np.random.RandomState(42)
    inputmap = rng.randn(*shape).astype(float)
    mask = np.ones(shape, dtype=float)
    mask[:3, :, :] = 0.0

    result = tide_fit.territorystats(inputmap, atlas, inputmask=mask, debug=debug)
    assert len(result) == 9


def territorystats_entropy_range(debug=False):
    """Test territorystats with custom entropy range."""
    if debug:
        print("territorystats_entropy_range")
    shape = (6, 6, 4)
    atlas = np.ones(shape, dtype=int)
    rng = np.random.RandomState(42)
    inputmap = rng.randn(*shape).astype(float)

    result = tide_fit.territorystats(
        inputmap,
        atlas,
        entropybins=50,
        entropyrange=(-3.0, 3.0),
        debug=debug,
    )
    _, _, _, _, _, _, _, _, entropies = result
    assert entropies.shape == (1, 1)
    # Entropy should be positive
    assert entropies[0, 0] > 0.0


def territorystats_entropy_bins(debug=False):
    """Test territorystats with different entropy bin count."""
    if debug:
        print("territorystats_entropy_bins")
    shape = (6, 6, 4)
    atlas = np.ones(shape, dtype=int)
    rng = np.random.RandomState(42)
    inputmap = rng.randn(*shape).astype(float)

    r1 = tide_fit.territorystats(inputmap, atlas, entropybins=10)
    r2 = tide_fit.territorystats(inputmap, atlas, entropybins=200)
    e1 = r1[8][0, 0]
    e2 = r2[8][0, 0]
    # Different bin counts should give different entropy values
    assert e1 != e2


# ==================== fingerprint workflow integration tests ====================


def fingerprint_with_constant_template(debug=False):
    """Test fingerprint workflow with constant template and synthetic data."""
    if debug:
        print("fingerprint_with_constant_template")
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (10, 12, 8)
        numregions = 3

        # Create atlas
        atlas = _make_synthetic_atlas(shape, numregions=numregions)
        atlas_path = _make_nifti_file(tmpdir, atlas.astype(np.float64), "atlas.nii.gz")

        # Create regions file
        regions_path = os.path.join(tmpdir, "atlas.txt")
        with open(regions_path, "w") as f:
            for i in range(numregions):
                f.write(f"Region_{i+1}\n")

        # Create input map
        rng = np.random.RandomState(42)
        inputmap = rng.randn(*shape).astype(np.float64)
        input_path = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")

        outputroot = os.path.join(tmpdir, "output")

        # Use constant template (fitorder=0) to avoid needing template file
        fp.fingerprint(
            input_path,
            "constant",
            f"USER_{atlas_path}",
            outputroot,
            0,
            intercept=True,
            debug=debug,
        )

        # Check that output files were created
        expected_suffix = "_template-constant_atlas-USER_O0"
        root = outputroot + expected_suffix

        # Check for the main output files
        assert os.path.exists(root + "_maskmap.nii.gz"), "maskmap not created"
        assert os.path.exists(root + "_allR2s.tsv"), "R2 tsv not created"
        assert os.path.exists(root + "_allR2s.nii.gz"), "R2 nifti not created"
        assert os.path.exists(root + "_allcoffs.nii.gz"), "coffs nifti not created"
        assert os.path.exists(root + "_fit_O0.tsv"), "fit coefficients tsv not created"
        assert os.path.exists(root + "_0000_fits.tsv"), "per-map fits tsv not created"

        # Check stats files
        for stat in [
            "mean",
            "std",
            "median",
            "mad",
            "variance",
            "skewness",
            "kurtosis",
            "entropy",
            "residual_mean",
            "residual_std",
            "residual_median",
            "residual_mad",
            "residual_variance",
            "residual_skewness",
            "residual_kurtosis",
            "residual_entropy",
        ]:
            assert os.path.exists(root + f"_all{stat}.tsv"), f"{stat} tsv not created"

        # Read back R2 tsv and verify structure
        r2_df = pd.read_csv(root + "_allR2s.tsv", sep="\t")
        assert len(r2_df.columns) == numregions
        assert r2_df.shape[0] == 1  # single map

        if debug:
            print("  Output files verified successfully")


def fingerprint_with_real_atlas(debug=False):
    """Test fingerprint workflow with a built-in atlas (JHU1) and constant template."""
    if debug:
        print("fingerprint_with_real_atlas")

    # Check that reference files exist
    referencedir = None
    try:
        import rapidtide.util as tide_util

        referencedir = tide_util.findreferencedir()
    except Exception:
        if debug:
            print("  Skipping: could not find reference directory")
        return

    atlas_path = os.path.join(
        referencedir,
        "JHU-ArterialTerritoriesNoVent-LVL1_space-MNI152NLin6Asym_2mm.nii.gz",
    )
    if not os.path.exists(atlas_path):
        if debug:
            print("  Skipping: atlas file not found")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Read atlas to get its shape
        _, atlas_data, atlas_hdr, atlasdims, _ = tide_io.readfromnifti(atlas_path)
        nx, ny, nz, _ = tide_io.parseniftidims(atlasdims)

        # Create an input map matching atlas dimensions
        rng = np.random.RandomState(42)
        inputmap = rng.randn(nx, ny, nz).astype(np.float64)
        # Zero out voxels outside atlas
        inputmap[atlas_data < 1] = 0.0
        input_path = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")

        outputroot = os.path.join(tmpdir, "output")

        fp.fingerprint(
            input_path,
            "constant",
            "JHU1",
            outputroot,
            0,
            intercept=True,
            debug=debug,
        )

        expected_suffix = "_template-constant_atlas-JHU1_O0"
        root = outputroot + expected_suffix
        assert os.path.exists(root + "_maskmap.nii.gz")
        assert os.path.exists(root + "_allR2s.tsv")
        if debug:
            print("  Real atlas test passed")


def fingerprint_output_r2_values(debug=False):
    """Test that fingerprint R2 values are reasonable for constant template."""
    if debug:
        print("fingerprint_output_r2_values")
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (8, 8, 6)
        numregions = 2
        atlas = _make_synthetic_atlas(shape, numregions=numregions)
        atlas_path = _make_nifti_file(tmpdir, atlas.astype(np.float64), "atlas.nii.gz")
        regions_path = os.path.join(tmpdir, "atlas.txt")
        with open(regions_path, "w") as f:
            for i in range(numregions):
                f.write(f"Region_{i+1}\n")

        # Constant input → R2 should be 1.0 for constant template
        inputmap = np.zeros(shape, dtype=np.float64)
        for region in range(1, numregions + 1):
            inputmap[atlas == region] = region * 3.0
        input_path = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        fp.fingerprint(
            input_path,
            "constant",
            f"USER_{atlas_path}",
            outputroot,
            0,
            intercept=True,
            debug=debug,
        )

        expected_suffix = "_template-constant_atlas-USER_O0"
        root = outputroot + expected_suffix
        r2_df = pd.read_csv(root + "_allR2s.tsv", sep="\t")
        # All R2 values should be 1.0 for constant data
        for col in r2_df.columns:
            assert (
                abs(r2_df[col].values[0] - 1.0) < 1e-10
            ), f"R2 for {col} is {r2_df[col].values[0]}"

        if debug:
            print("  R2 values verified successfully")


def fingerprint_stats_output(debug=False):
    """Test that fingerprint stats output files have correct structure."""
    if debug:
        print("fingerprint_stats_output")
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (8, 8, 6)
        numregions = 2
        atlas = _make_synthetic_atlas(shape, numregions=numregions)
        atlas_path = _make_nifti_file(tmpdir, atlas.astype(np.float64), "atlas.nii.gz")
        regions_path = os.path.join(tmpdir, "atlas.txt")
        with open(regions_path, "w") as f:
            for i in range(numregions):
                f.write(f"Region_{i+1}\n")

        rng = np.random.RandomState(42)
        inputmap = rng.randn(*shape).astype(np.float64)
        input_path = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        fp.fingerprint(
            input_path,
            "constant",
            f"USER_{atlas_path}",
            outputroot,
            0,
            intercept=True,
            debug=debug,
        )

        expected_suffix = "_template-constant_atlas-USER_O0"
        root = outputroot + expected_suffix

        # Check mean stats
        mean_df = pd.read_csv(root + "_allmean.tsv", sep="\t")
        assert mean_df.shape == (1, numregions)

        # Check std stats
        std_df = pd.read_csv(root + "_allstd.tsv", sep="\t")
        assert std_df.shape == (1, numregions)

        # Residual stats should also exist
        res_mean_df = pd.read_csv(root + "_allresidual_mean.tsv", sep="\t")
        assert res_mean_df.shape == (1, numregions)

        if debug:
            print("  Stats output verified")


def fingerprint_no_intercept(debug=False):
    """Test fingerprint with nointercept option."""
    if debug:
        print("fingerprint_no_intercept")
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (8, 8, 6)
        numregions = 2
        atlas = _make_synthetic_atlas(shape, numregions=numregions)
        atlas_path = _make_nifti_file(tmpdir, atlas.astype(np.float64), "atlas.nii.gz")
        regions_path = os.path.join(tmpdir, "atlas.txt")
        with open(regions_path, "w") as f:
            for i in range(numregions):
                f.write(f"Region_{i+1}\n")

        rng = np.random.RandomState(42)
        inputmap = rng.randn(*shape).astype(np.float64)
        input_path = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        # constant template with no intercept, fitorder 0 → still fitorder=0 w/ intercept
        # Use fitorder=1 to actually test no intercept
        # Create a template for linear fitting
        template = _make_synthetic_template(atlas)
        template_path = _make_nifti_file(tmpdir, template, "template.nii.gz")

        # Directly call territorydecomp with no intercept
        fitmap, thecoffs, theR2s = tide_fit.territorydecomp(
            inputmap,
            template,
            atlas,
            fitorder=1,
            intercept=False,
            debug=debug,
        )
        # mlregress always returns fitorder+1 coffs (intercept prepended as zero)
        assert thecoffs.shape == (1, numregions, 2)
        if debug:
            print("  No intercept test passed")


def fingerprint_no_intercept_workflow(debug=False):
    """Test the full workflow with intercept disabled.

    ``territorydecomp`` always returns ``fitorder + 1`` coefficients with the intercept
    prepended, so the output tables must drop that leading (identically zero) column rather
    than mislabelling it.  This regression test covers a length mismatch that made
    ``--nointercept`` unusable.
    """
    if debug:
        print("fingerprint_no_intercept_workflow")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 3
        atlaspath, _ = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        _run_with_stubbed_reads(
            tmpdir,
            "lag",
            f"USER_{atlaspath}",
            numregions,
            fitorder=2,
            intercept=False,
            debug=debug,
        )

        root = os.path.join(tmpdir, "stubout") + "_template-lag_atlas-USER_O2"
        # the order 0 (intercept) table must not be written when there is no intercept
        assert not os.path.exists(
            root + "_fit_O0.tsv"
        ), "intercept table written without intercept"
        assert os.path.exists(root + "_fit_O1.tsv")
        assert os.path.exists(root + "_fit_O2.tsv")

        fits = pd.read_csv(root + "_0000_fits.tsv", sep="\t")
        # columns are Region, R2, then one per fit order starting at 1
        assert list(fits.columns[-2:]) == ["1", "2"]
        assert fits.shape[0] == numregions


def fingerprint_high_fitorder_column_names(debug=False):
    """Test double digit fit orders produce one correctly named column per order.

    Building the name list by string concatenation split ``"10"`` into two entries, which
    silently produced garbage column names for fit orders of ten or more.
    """
    if debug:
        print("fingerprint_high_fitorder_column_names")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        atlaspath, _ = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(5)
        atlas = _make_labelled_atlas(numregions)
        inputmap = rng.randn(*atlas.shape)
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        fitorder = 11
        fp.fingerprint(
            inputpath,
            "constant",
            f"USER_{atlaspath}",
            outputroot,
            fitorder,
            intercept=True,
            debug=debug,
        )
        root = outputroot + f"_template-constant_atlas-USER_O{fitorder}"
        fits = pd.read_csv(root + "_0000_fits.tsv", sep="\t")
        coefcolumns = list(fits.columns[3:])
        assert coefcolumns == [str(i) for i in range(0, fitorder + 1)], coefcolumns
        # one coefficient table per order, orders 0 through fitorder inclusive
        for i in range(0, fitorder + 1):
            assert os.path.exists(root + f"_fit_O{i}.tsv"), f"missing order {i} table"


def fingerprint_4d_input(debug=False):
    """Test the workflow handles a 4D input map, producing one row per map."""
    if debug:
        print("fingerprint_4d_input")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        nummaps = 3
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(17)
        inputmap = rng.randn(*atlas.shape, nummaps)
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        fp.fingerprint(
            inputpath,
            "constant",
            f"USER_{atlaspath}",
            outputroot,
            0,
            intercept=True,
            debug=debug,
        )
        root = outputroot + "_template-constant_atlas-USER_O0"
        r2df = pd.read_csv(root + "_allR2s.tsv", sep="\t")
        assert r2df.shape == (nummaps, numregions)
        # one per-map fit table for each map in the series
        for whichmap in range(nummaps):
            assert os.path.exists(root + f"_{str(whichmap).zfill(4)}_fits.tsv")

        # at fit order 0 each coefficient is just the regional mean, so check the values
        # rather than only the shapes - a mask applied to the wrong map would show up here
        coeffs = pd.read_csv(root + "_fit_O0.tsv", sep="\t")
        assert coeffs.shape == (nummaps, numregions)
        for whichmap in range(nummaps):
            for whichregion in range(1, numregions + 1):
                expected = np.mean(inputmap[:, :, :, whichmap][atlas == whichregion])
                actual = coeffs.iloc[whichmap, whichregion - 1]
                assert (
                    abs(actual - expected) < 1e-10
                ), f"map {whichmap} region {whichregion}: {actual} != {expected}"


def fingerprint_limittomask_3d(debug=False):
    """Test --limittomask zeroes the 3D fit and difference maps outside the mask."""
    if debug:
        print("fingerprint_limittomask_3d")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(23)
        inputmap = rng.randn(*atlas.shape) + 10.0
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        # exclude the first plane so there is a region of the volume outside the mask
        extramask = np.ones(atlas.shape, dtype=np.float64)
        extramask[0, :, :] = 0.0
        extramaskpath = _make_nifti_file(tmpdir, extramask, "extramask.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        fp.fingerprint(
            inputpath,
            "constant",
            f"USER_{atlaspath}",
            outputroot,
            0,
            intercept=True,
            extramaskname=extramaskpath,
            limittomask=True,
            debug=debug,
        )
        root = outputroot + "_template-constant_atlas-USER_O0"
        _, fitdata, _, _, _ = tide_io.readfromnifti(root + "_fit.nii.gz")
        _, diffdata, _, _, _ = tide_io.readfromnifti(root + "_fitdiff.nii.gz")
        assert np.all(fitdata[0, :, :] == 0.0), "fit map not zeroed outside the mask"
        assert np.all(diffdata[0, :, :] == 0.0), "difference map not zeroed outside the mask"
        # inside the mask there is real signal
        assert np.any(fitdata[1:, :, :] != 0.0)


def fingerprint_limittomask_4d(debug=False):
    """Test --limittomask broadcasts the 3D mask across a 4D map."""
    if debug:
        print("fingerprint_limittomask_4d")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        nummaps = 2
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(29)
        inputmap = rng.randn(*atlas.shape, nummaps) + 10.0
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        extramask = np.ones(atlas.shape, dtype=np.float64)
        extramask[0, :, :] = 0.0
        extramaskpath = _make_nifti_file(tmpdir, extramask, "extramask.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        fp.fingerprint(
            inputpath,
            "constant",
            f"USER_{atlaspath}",
            outputroot,
            0,
            intercept=True,
            extramaskname=extramaskpath,
            limittomask=True,
            debug=debug,
        )
        root = outputroot + "_template-constant_atlas-USER_O0"
        _, fitdata, _, _, _ = tide_io.readfromnifti(root + "_fit.nii.gz")
        assert fitdata.shape[3] == nummaps
        assert np.all(fitdata[0, :, :, :] == 0.0), "fit map not zeroed outside the mask"


def fingerprint_include_and_exclude_masks(debug=False):
    """Test include and exclude mask specifications both restrict the fitted voxels."""
    if debug:
        print("fingerprint_include_and_exclude_masks")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 3
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(31)
        inputmap = rng.randn(*atlas.shape)
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")

        # include everything except the last plane
        includemask = np.ones(atlas.shape, dtype=np.float64)
        includemask[-1, :, :] = 0.0
        includepath = _make_nifti_file(tmpdir, includemask, "include.nii.gz")
        # exclude the first plane
        excludemask = np.zeros(atlas.shape, dtype=np.float64)
        excludemask[0, :, :] = 1.0
        excludepath = _make_nifti_file(tmpdir, excludemask, "exclude.nii.gz")

        outputroot = os.path.join(tmpdir, "output")
        fp.fingerprint(
            inputpath,
            "constant",
            f"USER_{atlaspath}",
            outputroot,
            0,
            intercept=True,
            includespec=includepath,
            excludespec=excludepath,
            debug=debug,
        )
        root = outputroot + "_template-constant_atlas-USER_O0"
        _, maskdata, _, _, _ = tide_io.readfromnifti(root + "_maskmap.nii.gz")
        assert np.all(maskdata[0, :, :] == 0.0), "excluded plane still in the mask"
        assert np.all(
            maskdata[-1, :, :] == 0.0
        ), "plane outside the include mask still in the mask"
        assert np.any(maskdata != 0.0), "mask is entirely empty"


def fingerprint_mask_spec_with_values(debug=False):
    """Test a MASK:VALSPEC include specification selects only the listed label values."""
    if debug:
        print("fingerprint_mask_spec_with_values")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 3
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(37)
        inputmap = rng.randn(*atlas.shape)
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")
        # use the atlas itself as a labelled mask, keeping only region 2
        labelmaskpath = _make_nifti_file(tmpdir, atlas.astype(np.float64), "labels.nii.gz")

        outputroot = os.path.join(tmpdir, "output")
        fp.fingerprint(
            inputpath,
            "constant",
            f"USER_{atlaspath}",
            outputroot,
            0,
            intercept=True,
            includespec=f"{labelmaskpath}:2",
            debug=debug,
        )
        root = outputroot + "_template-constant_atlas-USER_O0"
        _, maskdata, _, _, _ = tide_io.readfromnifti(root + "_maskmap.nii.gz")
        # only the voxels labelled 2 in the atlas survive
        assert np.all(maskdata[atlas != 2] == 0.0)
        assert np.any(maskdata[atlas == 2] != 0.0)


def fingerprint_entropy_options(debug=False):
    """Test custom entropy bin counts and ranges flow through to the stats tables."""
    if debug:
        print("fingerprint_entropy_options")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(41)
        inputmap = rng.randn(*atlas.shape)
        inputpath = _make_nifti_file(tmpdir, inputmap, "input.nii.gz")

        entropies = []
        for entropybins, entropyrange, tag in [
            (11, None, "coarse"),
            (201, (-3.0, 3.0), "fine"),
        ]:
            outputroot = os.path.join(tmpdir, f"output_{tag}")
            fp.fingerprint(
                inputpath,
                "constant",
                f"USER_{atlaspath}",
                outputroot,
                0,
                intercept=True,
                entropybins=entropybins,
                entropyrange=entropyrange,
                debug=debug,
            )
            root = outputroot + "_template-constant_atlas-USER_O0"
            entropies.append(pd.read_csv(root + "_allentropy.tsv", sep="\t").values)
        # different histogram parameters must produce different entropy estimates
        assert not np.allclose(entropies[0], entropies[1])


# ==================== entrypoint tests ====================


def entrypoint_missing_args(debug=False):
    """Test entrypoint raises SystemExit with no arguments."""
    if debug:
        print("entrypoint_missing_args")
    old_argv = sys.argv
    sys.argv = ["fingerprint"]
    try:
        fp.entrypoint()
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def _run_entrypoint(arglist):
    """Run the fingerprint entrypoint with a synthetic command line.

    Parameters
    ----------
    arglist : list of str
        Arguments to place on sys.argv after the program name.

    Returns
    -------
    None
    """
    old_argv = sys.argv
    sys.argv = ["fingerprint"] + arglist
    try:
        fp.entrypoint()
    finally:
        sys.argv = old_argv


def entrypoint_with_custom_atlas(debug=False):
    """Test the entrypoint runs end to end with a custom atlas."""
    if debug:
        print("entrypoint_with_custom_atlas")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(53)
        inputpath = _make_nifti_file(tmpdir, rng.randn(*atlas.shape), "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        _run_entrypoint(
            [inputpath, outputroot, "--customatlas", atlaspath, "--template", "constant"]
        )
        root = outputroot + "_template-constant_atlas-USER_O0"
        assert os.path.exists(root + "_maskmap.nii.gz")
        assert os.path.exists(root + "_allR2s.tsv")


def entrypoint_constant_template_forces_fitorder_zero(debug=False):
    """Test the constant template overrides an explicitly requested fit order.

    A constant template has no higher order terms to fit, so the entrypoint resets the fit
    order to 0 and re-enables the intercept regardless of what was asked for.
    """
    if debug:
        print("entrypoint_constant_template_forces_fitorder_zero")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(59)
        inputpath = _make_nifti_file(tmpdir, rng.randn(*atlas.shape), "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        _run_entrypoint(
            [
                inputpath,
                outputroot,
                "--customatlas",
                atlaspath,
                "--template",
                "constant",
                "--fitorder",
                "3",
                "--nointercept",
            ]
        )
        # the requested order 3 was overridden, so the O0 outputs are what exist
        assert os.path.exists(outputroot + "_template-constant_atlas-USER_O0_maskmap.nii.gz")
        assert not os.path.exists(outputroot + "_template-constant_atlas-USER_O3_maskmap.nii.gz")
        # --nointercept was also overridden, so the order 0 coefficient table is present
        assert os.path.exists(outputroot + "_template-constant_atlas-USER_O0_fit_O0.tsv")


def entrypoint_debug_mode(debug=False):
    """Test the entrypoint debug path prints its pre and post processing state."""
    if debug:
        print("entrypoint_debug_mode")
    with tempfile.TemporaryDirectory() as tmpdir:
        numregions = 2
        atlaspath, atlas = _make_write_only_tmpdir_atlas(tmpdir, numregions)
        rng = np.random.RandomState(61)
        inputpath = _make_nifti_file(tmpdir, rng.randn(*atlas.shape), "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        _run_entrypoint(
            [
                inputpath,
                outputroot,
                "--customatlas",
                atlaspath,
                "--template",
                "constant",
                "--debug",
            ]
        )
        assert os.path.exists(outputroot + "_template-constant_atlas-USER_O0_allR2s.tsv")


def entrypoint_uses_named_atlas_when_no_custom(debug=False):
    """Test the entrypoint falls back to the --atlas selection with no custom atlas given."""
    if debug:
        print("entrypoint_uses_named_atlas_when_no_custom")
    with tempfile.TemporaryDirectory() as tmpdir:
        # stub the volume reads so the full sized reference atlas is never actually loaded
        numregions = 8
        atlas = _make_labelled_atlas(numregions)
        rng = np.random.RandomState(67)
        atlastuple = _read_as_tuple(tmpdir, atlas, "ep_atlas.nii.gz")
        maptuple = _read_as_tuple(tmpdir, rng.randn(*atlas.shape), "ep_map.nii.gz")
        inputpath = _make_nifti_file(tmpdir, rng.randn(*atlas.shape), "input.nii.gz")
        outputroot = os.path.join(tmpdir, "output")

        recorded = []

        def _stubreader(thename, **readkwargs):
            recorded.append(thename)
            return copy.deepcopy([atlastuple, maptuple][len(recorded) - 1])

        with patch.object(tide_io, "readfromnifti", side_effect=_stubreader):
            _run_entrypoint([inputpath, outputroot, "--atlas", "JHU2", "--template", "constant"])
        assert recorded[0].endswith(
            "JHU-ArterialTerritoriesNoVent-LVL2_space-MNI152NLin6Asym_2mm.nii.gz"
        )
        assert os.path.exists(outputroot + "_template-constant_atlas-JHU2_O0_allR2s.tsv")


# ==================== Main test function ====================


def test_fingerprint(debug=False):
    # _get_parser tests
    if debug:
        print("Running _get_parser tests")
    get_parser_returns_parser(debug=debug)
    get_parser_required_args(debug=debug)
    get_parser_defaults(debug=debug)
    get_parser_atlas_choices(debug=debug)
    get_parser_template_choices(debug=debug)
    get_parser_fitorder(debug=debug)
    get_parser_nointercept(debug=debug)
    get_parser_limittomask(debug=debug)
    get_parser_entropybins(debug=debug)
    get_parser_entropyrange(debug=debug)
    get_parser_debug(debug=debug)
    get_parser_includemask(debug=debug)
    get_parser_excludemask(debug=debug)
    get_parser_invalid_atlas(debug=debug)
    get_parser_invalid_template(debug=debug)
    get_parser_fullpaths_expands(debug=debug)
    get_parser_fullpaths_empty_string(debug=debug)
    get_parser_rejects_missing_input(debug=debug)
    get_parser_customatlas(debug=debug)
    get_parser_extramask(debug=debug)

    # atlas and template selection tests
    if debug:
        print("Running atlas and template selection tests")
    fingerprint_selects_builtin_atlases(debug=debug)
    fingerprint_selects_templates(debug=debug)
    fingerprint_user_atlas_derives_region_file(debug=debug)
    fingerprint_illegal_template(debug=debug)
    fingerprint_illegal_atlas(debug=debug)

    # territorydecomp tests
    if debug:
        print("Running territorydecomp tests")
    territorydecomp_3d_basic(debug=debug)
    territorydecomp_3d_no_intercept(debug=debug)
    territorydecomp_3d_fitorder0(debug=debug)
    territorydecomp_4d(debug=debug)
    territorydecomp_with_mask(debug=debug)
    territorydecomp_higher_order(debug=debug)

    # territorystats tests
    if debug:
        print("Running territorystats tests")
    territorystats_3d_basic(debug=debug)
    territorystats_known_values(debug=debug)
    territorystats_4d(debug=debug)
    territorystats_with_mask(debug=debug)
    territorystats_entropy_range(debug=debug)
    territorystats_entropy_bins(debug=debug)

    # fingerprint workflow integration tests
    if debug:
        print("Running fingerprint integration tests")
    fingerprint_with_constant_template(debug=debug)
    fingerprint_with_real_atlas(debug=debug)
    fingerprint_output_r2_values(debug=debug)
    fingerprint_stats_output(debug=debug)
    fingerprint_no_intercept(debug=debug)
    fingerprint_no_intercept_workflow(debug=debug)
    fingerprint_high_fitorder_column_names(debug=debug)
    fingerprint_4d_input(debug=debug)
    fingerprint_limittomask_3d(debug=debug)
    fingerprint_limittomask_4d(debug=debug)
    fingerprint_include_and_exclude_masks(debug=debug)
    fingerprint_mask_spec_with_values(debug=debug)
    fingerprint_entropy_options(debug=debug)

    # entrypoint tests
    if debug:
        print("Running entrypoint tests")
    entrypoint_missing_args(debug=debug)
    entrypoint_with_custom_atlas(debug=debug)
    entrypoint_constant_template_forces_fitorder_zero(debug=debug)
    entrypoint_debug_mode(debug=debug)
    entrypoint_uses_named_atlas_when_no_custom(debug=debug)


if __name__ == "__main__":
    test_fingerprint(debug=True)
