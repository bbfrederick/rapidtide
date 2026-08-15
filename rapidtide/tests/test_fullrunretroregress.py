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
"""Exercise the retroregress option branches the rapidtide full run tests leave alone.

test_fullrunrapidtide_v3, v6 and v7 all run retroregress, but only at the default and
"max"/"onlyregressors" output levels.  This file does one lean rapidtide run to produce
a dataset, then drives retroregress over the remaining option surface: the other output
levels, the delay refinement switches, voxel specific derivatives, the sLFO filter mask,
and the pseudofile path.
"""

import os

import matplotlib as mpl
import pytest

from rapidtide.tests.utils import get_example_and_temp_roots, run_rapidtide, run_retroregress

pytestmark = pytest.mark.slow


def _rapidtidedataset(exampleroot, testtemproot):
    """Produce a rapidtide output set for retroregress to consume.

    Kept deliberately lean - a single pass with no tissue masks - because nothing here
    depends on the quality of the fit, only on a complete and self consistent set of
    output files being present.

    Parameters
    ----------
    exampleroot : str
        Directory holding the example input data.
    testtemproot : str
        Directory to write the rapidtide outputs into.

    Returns
    -------
    tuple of (str, str)
        The fmri input file and the dataset root retroregress should be pointed at.

    Notes
    -----
    The rapidtide run is skipped if a complete dataset is already sitting at the target
    root, so the two tests in this file share one run when both are collected while each
    still works on its own.
    """
    fmrifile = os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz")
    theroot = os.path.join(testtemproot, "sub-RETROREGRESSTEST")
    # the runoptions file is written last, so its presence means the run finished
    if os.path.isfile(f"{theroot}_desc-runoptions_info.json"):
        return fmrifile, theroot
    run_rapidtide(
        [
            fmrifile,
            theroot,
            "--nprocs",
            "-1",
            "--passes",
            "1",
            "--brainmask",
            os.path.join(exampleroot, "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        ]
    )
    return fmrifile, theroot


def test_fullrunretroregress(debug=False, local=False, displayplots=False):
    """Drive retroregress across its option surface against one rapidtide dataset."""
    exampleroot, testtemproot = get_example_and_temp_roots(local)
    fmrifile, theroot = _rapidtidedataset(exampleroot, testtemproot)

    # the output levels the existing tests do not reach
    for thelevel in ["min", "less", "more"]:
        run_retroregress(
            [
                fmrifile,
                theroot,
                "--alternateoutput",
                os.path.join(testtemproot, f"retro_{thelevel}"),
                "--nprocs",
                "-1",
                "--outputlevel",
                thelevel,
            ]
        )
        assert os.path.isfile(
            os.path.join(testtemproot, f"retro_{thelevel}_desc-runoptions_info.json")
        ), f"no runoptions written at output level {thelevel}"

    # delay refinement turned off, which takes the unrefined lag branch
    run_retroregress(
        [
            fmrifile,
            theroot,
            "--alternateoutput",
            os.path.join(testtemproot, "retro_norefine"),
            "--nprocs",
            "-1",
            "--norefinedelay",
        ]
    )

    # voxel specific derivatives, which takes the multiple EV path, plus the debug
    # reporting that goes with it
    run_retroregress(
        [
            fmrifile,
            theroot,
            "--alternateoutput",
            os.path.join(testtemproot, "retro_derivs"),
            "--nprocs",
            "-1",
            "--regressderivs",
            "1",
            "--debug",
        ]
    )

    # the sLFO filter mask and the pseudofile output
    run_retroregress(
        [
            fmrifile,
            theroot,
            "--alternateoutput",
            os.path.join(testtemproot, "retro_maskpseudo"),
            "--nprocs",
            "-1",
            "--sLFOfiltmask",
            "--makepseudofile",
        ]
    )

    # NB: deliberately not exercising the no-alternateoutput case here.  With no alternate
    # root, retroregress writes back over the source dataset - including a modified
    # _desc-runoptions_info.json - so it would mutate the dataset the other tests share.


def test_fullrunretroregress_rejects_mismatched_fmri(debug=False, local=False, displayplots=False):
    """A dataset built from different geometry than the fmri file is rejected.

    Every mask and map retroregress reads has to agree with the fmri file it is handed;
    pairing a dataset with an fmri file of another shape must fail rather than silently
    reinterpret the voxels.
    """
    import nibabel as nib
    import numpy as np

    exampleroot, testtemproot = get_example_and_temp_roots(local)
    fmrifile, theroot = _rapidtidedataset(exampleroot, testtemproot)

    # a valid 4D file, but with geometry the dataset knows nothing about
    oddfmri = os.path.join(testtemproot, "sub-ODDSHAPE.nii.gz")
    nib.save(
        nib.Nifti1Image(np.zeros((5, 6, 7, 20), dtype=np.float32), np.eye(4)),
        oddfmri,
    )
    with pytest.raises(ValueError, match="do not match fmri dimensions"):
        run_retroregress(
            [
                oddfmri,
                theroot,
                "--alternateoutput",
                os.path.join(testtemproot, "retro_mismatch"),
                "--nprocs",
                "-1",
            ]
        )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunretroregress(debug=True, local=True, displayplots=True)
    test_fullrunretroregress_rejects_mismatched_fmri(debug=True, local=True, displayplots=True)
