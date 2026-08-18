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
"""
Utility functions for testing rapidtide.
"""

import os
from typing import Iterable

import numpy as np
import numpy.typing as npt


def get_rapidtide_root() -> str:
    """
    Returns the path to the base rapidtide directory, terminated with separator.
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    thisdir, thisfile = os.path.split(os.path.join(os.path.realpath(__file__)))
    return os.path.join(thisdir, "..") + os.path.sep


def get_scripts_path() -> str:
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "testdata".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.realpath(os.path.join(get_rapidtide_root(), "scripts")) + os.path.sep


def get_test_data_path() -> str:
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "testdata".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.realpath(os.path.join(get_rapidtide_root(), "tests", "testdata")) + os.path.sep


def get_test_temp_path(local: bool = False) -> str:
    """
    Returns the path to test temporary directory, terminated with separator.
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    if local:
        return "./tmp"
    else:
        return os.path.realpath(os.path.join(get_rapidtide_root(), "tests", "tmp")) + os.path.sep


def get_examples_path(local: bool = False) -> str:
    """
    Returns the path to examples src directory, where larger test files live, terminated with separator. Test-related
    data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    if local:
        return "../data/examples/src"
    else:
        return (
            os.path.realpath(os.path.join(get_rapidtide_root(), "data", "examples", "src"))
            + os.path.sep
        )


def create_dir(thedir: str, debug: bool = False) -> None:
    # create a directory if it doesn't exist
    try:
        os.makedirs(thedir)
        if debug:
            print(thedir, "created")
    except OSError:
        if debug:
            print(thedir, "exists")
        else:
            pass


def mse(ndarr1: npt.NDArray, ndarr2: npt.NDArray) -> np.floating:
    """
    Compute mean-squared error.
    """
    return np.mean(np.square(ndarr2 - ndarr1))


def get_example_and_temp_roots(local: bool = False) -> tuple[str, str]:
    """
    Return the standard example data and test temp roots as a tuple.
    """
    return get_examples_path(local), get_test_temp_path(local)


def run_happy(inputargs: list[str]) -> None:
    """
    Run happy workflow from CLI-style argument list.
    """
    import rapidtide.workflows.happy as happy_workflow
    import rapidtide.workflows.happy_parser as happy_parser

    happy_workflow.happy_main(happy_parser.process_args(inputargs=inputargs))


def run_rapidtide(inputargs: list[str]) -> None:
    """
    Run rapidtide workflow from CLI-style argument list.

    Parameters
    ----------
    inputargs : list of str
        The command line arguments, exactly as they would be typed, starting with the
        input file and the output root.

    Returns
    -------
    None

    Notes
    -----
    A run against one of the SMALL datasets must always pass an explicit ``--brainmask``.
    Those datasets are cropped to a box inside the head, and rapidtide's automatic brain
    mask thresholds against the mean of the whole volume - with the surrounding air gone
    that mean is far higher, and the automatic mask collapses to a few hundred voxels.
    The run still succeeds and the test still passes; it just stops exercising anything.
    Confirm the voxel count in the run log rather than trusting the exit status.
    """
    import rapidtide.workflows.rapidtide as rapidtide_workflow
    import rapidtide.workflows.rapidtide_parser as rapidtide_parser

    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))


def run_retroregress(inputargs: list[str]) -> None:
    """
    Run retroregress workflow from CLI-style argument list.
    """
    import rapidtide.workflows.retroregress as rapidtide_retroregress

    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))


def assert_output_maps_match(
    map_names: Iterable[str],
    output_root_1: str,
    output_root_2: str,
    temp_root: str,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    spacetolerance: float = 1e-3,
    debug: bool = False,
) -> None:
    """
    Assert that selected NIFTI output maps match between two output roots.

    Parameters
    ----------
    map_names : iterable of str
        The map descriptors to compare, without the surrounding "_desc-" and "_map".
    output_root_1 : str
        Output root of the first run.
    output_root_2 : str
        Output root of the second run.
    temp_root : str
        Directory holding both runs' output.
    rtol : float, optional
        Largest permitted difference in any voxel, as a fraction of that voxel's value.
        The default allows roughly eight float32 units in the last place.
    atol : float, optional
        Absolute floor added to the tolerance, which is what governs voxels at or near
        zero where a relative comparison means nothing.
    spacetolerance : float, optional
        Tolerance for the spatial geometry comparison.
    debug : bool, optional
        Print the measured differences for every map, not just failing ones.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any named map differs by more than the tolerance, naming the map and
        reporting how far apart the two versions actually are.

    Notes
    -----
    The tolerance is relative because these maps are stored as float32 and the two
    output roots come from separate computations rather than one being a saved
    reference.  Two such runs agree to within rounding, not exactly, and the smallest
    difference float32 can even express is about 1.2e-7 of the value being stored - so
    an absolute limit tighter than that is really a demand for bit-identical output,
    which does not survive a change of platform.  It also cannot be met uniformly by
    maps like regressderivratios, whose values span from well under one to over 1e5.

    The mean squared error is reported but deliberately not used as a pass/fail gate.
    It is an absolute quantity, so on a map spanning five orders of magnitude a single
    legitimate last-place difference at the largest voxel produces an mse of around
    5e-10 - which would fail any fixed limit tight enough to be worth having, for the
    same reason the absolute per-voxel limit did.  The per-voxel tolerance already
    covers everything the mse would have caught and more: mse dilutes a few badly wrong
    voxels across the whole volume, whereas the per-voxel check sees each of them.
    """
    import numpy as np

    import rapidtide.io as tide_io

    for map_name in map_names:
        filename1 = os.path.join(temp_root, f"{output_root_1}_desc-{map_name}_map.nii.gz")
        filename2 = os.path.join(temp_root, f"{output_root_2}_desc-{map_name}_map.nii.gz")

        dummy, thedata1, theheader1, thedims1, dummy2 = tide_io.readfromnifti(filename1)
        dummy3, thedata2, theheader2, thedims2, dummy4 = tide_io.readfromnifti(filename2)

        assert tide_io.checkspacematch(
            theheader1, theheader2, tolerance=spacetolerance
        ), f"map '{map_name}': spatial geometry differs between the two runs"
        assert tide_io.checktimematch(
            thedims1, thedims2
        ), f"map '{map_name}': time dimensions differ between the two runs"

        thefirst = thedata1.astype(np.float64)
        thesecond = thedata2.astype(np.float64)
        thediff = np.abs(thefirst - thesecond)
        theallowed = atol + rtol * np.abs(thesecond)
        thebad = thediff > theallowed
        themse = float(np.mean(thediff**2))

        if debug:
            print(
                f"{map_name}: maxabsdiff={np.max(thediff):.4e} mse={themse:.4e} "
                f"scale={np.max(np.abs(thefirst)):.4g} failing={int(np.sum(thebad))}"
            )

        if not np.any(thebad):
            continue

        # report the worst offender in the terms the tolerance is expressed in, so the
        # number in the message can be compared against rtol directly
        theworst = int(np.argmax(thediff - theallowed))
        thescale = abs(thesecond.flat[theworst])
        therelative = thediff.flat[theworst] / thescale if thescale > 0.0 else float("inf")
        raise AssertionError(
            f"map '{map_name}' differs between {output_root_1} and {output_root_2}. "
            f"maxabsdiff={np.max(thediff):.4e}, mse={themse:.4e}, "
            f"{int(np.sum(thebad))} of {thediff.size} voxels outside "
            f"rtol={rtol:.1e}/atol={atol:.1e}; worst voxel differs by "
            f"{therelative:.3e} of its value {thescale:.4g}, "
            f"map scale={np.max(np.abs(thefirst)):.4g}"
        )


def assert_text_vectors_match(
    infile_spec: str,
    outfile_spec: str,
    msethresh: float = 2e-6,
    aethresh: int = 2,
    debug: bool = False,
) -> None:
    """
    Assert that two one-column text vectors and metadata match.
    """
    import rapidtide.io as tide_io

    insamplerate, instarttime, incolumns, indata, incompressed, infiletype = (
        tide_io.readvectorsfromtextfile(infile_spec, onecol=True, debug=debug)
    )
    outsamplerate, outstarttime, outcolumns, outdata, outcompressed, outfiletype = (
        tide_io.readvectorsfromtextfile(outfile_spec, onecol=True, debug=debug)
    )

    if debug:
        print(f"{insamplerate=}, {outsamplerate=}")
    assert insamplerate == outsamplerate
    if debug:
        print(f"{instarttime=}, {outstarttime=}")
    assert instarttime == outstarttime
    if debug:
        print(f"{incompressed=}, {outcompressed=}")
    assert incompressed == outcompressed
    if debug:
        print(f"{infiletype=}, {outfiletype=}")
    assert infiletype == outfiletype
    if debug:
        print(f"{incolumns=}, {outcolumns=}")
    assert incolumns == outcolumns
    if debug:
        print(f"{indata.shape=}, {outdata.shape=}")
    assert indata.shape == outdata.shape
    if debug:
        print(f"{mse(indata, outdata)=}, {msethresh=}")
    assert mse(indata, outdata) < msethresh
    np.testing.assert_almost_equal(indata, outdata, aethresh)
