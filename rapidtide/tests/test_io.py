#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

from rapidtide.tests.utils import mse
import rapidtide.io as tide_io
from rapidtide.tests.utils import get_test_data_path, get_test_target_path, get_test_temp_path, get_examples_path, get_rapidtide_root, get_scripts_path, create_dir


def test_io(debug=True, display=False):

    # test checkifnifti
    assert tide_io.checkifnifti('test.nii') == True
    assert tide_io.checkifnifti('test.nii.gz') == True
    assert tide_io.checkifnifti('test.txt') == False

    # test checkiftext
    assert tide_io.checkiftext('test.nii') == False
    assert tide_io.checkiftext('test.nii.gz') == False
    assert tide_io.checkiftext('test.txt') == True

    # test getniftiroot
    assert tide_io.getniftiroot('test.nii') == 'test'
    assert tide_io.getniftiroot('test.nii.gz') == 'test'
    assert tide_io.getniftiroot('test.txt') == 'test.txt'

    # test fmritimeinfo
    fmritimeinfothresh = 1e-2
    tr, timepoints = tide_io.fmritimeinfo(os.path.join(get_examples_path(), 'happyfmri.nii.gz'))
    assert np.fabs(tr - 1.16) < fmritimeinfothresh
    assert timepoints == 110
    tr, timepoints = tide_io.fmritimeinfo(os.path.join(get_examples_path(), 'fmri.nii.gz'))
    assert np.fabs(tr - 1.5) < fmritimeinfothresh
    assert timepoints == 260

def main():
    test_io(debug=True, display=True)


if __name__ == '__main__':
    main()
