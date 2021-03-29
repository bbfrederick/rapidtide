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
import os

import numpy as np

import rapidtide.io as tide_io
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path, mse


def test_io(debug=True, display=False):

    # create outputdir if it doesn't exist
    create_dir(get_test_temp_path())

    # test checkifnifti
    assert tide_io.checkifnifti("test.nii") == True
    assert tide_io.checkifnifti("test.nii.gz") == True
    assert tide_io.checkifnifti("test.txt") == False

    # test checkiftext
    assert tide_io.checkiftext("test.nii") == False
    assert tide_io.checkiftext("test.nii.gz") == False
    assert tide_io.checkiftext("test.txt") == True

    # test getniftiroot
    assert tide_io.getniftiroot("test.nii") == "test"
    assert tide_io.getniftiroot("test.nii.gz") == "test"
    assert tide_io.getniftiroot("test.txt") == "test.txt"

    # test fmritimeinfo
    fmritimeinfothresh = 1e-2
    tr, timepoints = tide_io.fmritimeinfo(
        os.path.join(get_examples_path(), "sub-HAPPYTEST.nii.gz")
    )
    assert np.fabs(tr - 1.16) < fmritimeinfothresh
    assert timepoints == 110
    tr, timepoints = tide_io.fmritimeinfo(
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST.nii.gz")
    )
    assert np.fabs(tr - 1.5) < fmritimeinfothresh
    assert timepoints == 260

    # test niftifile reading
    sizethresh = 1e-3
    happy_img, happy_data, happy_hdr, happydims, happysizes = tide_io.readfromnifti(
        os.path.join(get_examples_path(), "sub-HAPPYTEST.nii.gz")
    )
    fmri_img, fmri_data, fmri_hdr, fmridims, fmrisizes = tide_io.readfromnifti(
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST.nii.gz")
    )
    targetdims = [4, 65, 89, 64, 110, 1, 1, 1]
    targetsizes = [-1.00, 2.39583, 2.395830, 2.4, 1.16, 0.00, 0.00, 0.00]
    if debug:
        print("happydims:", happydims)
        print("targetdims:", targetdims)
        print("happysizes:", happysizes)
        print("targetsizes:", targetsizes)
    for i in range(len(targetdims)):
        assert targetdims[i] == happydims[i]
    assert mse(np.array(targetsizes), np.array(happysizes)) < sizethresh

    # test file writing
    datathresh = 2e-3  # relaxed threshold because sub-RAPIDTIDETEST has been converted to INT16
    tide_io.savetonifti(
        fmri_data, fmri_hdr, os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST_copy.nii.gz")
    )
    (
        fmricopy_img,
        fmricopy_data,
        fmricopy_hdr,
        fmricopydims,
        fmricopysizes,
    ) = tide_io.readfromnifti(os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST_copy.nii.gz"))
    assert tide_io.checkspacematch(fmri_hdr, fmricopy_hdr)
    assert tide_io.checktimematch(fmridims, fmridims)
    assert mse(fmri_data, fmricopy_data) < datathresh

    # test file header comparisons
    assert tide_io.checkspacematch(happy_hdr, happy_hdr)
    assert not tide_io.checkspacematch(happy_hdr, fmri_hdr)
    assert tide_io.checktimematch(happydims, happydims)
    assert not tide_io.checktimematch(happydims, fmridims)


def main():
    test_io(debug=True, display=True)


if __name__ == "__main__":
    main()
