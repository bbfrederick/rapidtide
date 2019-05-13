#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt 
from scipy import arange
import subprocess
import os

import rapidtide.io as tide_io
import rapidtide.fit as tide_fit
import rapidtide.util as tide_util
from rapidtide.tests.utils import get_test_data_path, get_test_target_path, get_test_temp_path, get_examples_path, get_rapidtide_root, get_scripts_path


def test_rapidtide2x(debug=False):
    recalculate = True
    if recalculate:
        # create outputdir if it doesn't exist
        try:
            os.makedirs(get_test_temp_path())
        except OSError:
            pass
    
        # and launch the processing
        theargs = []
        theargs += [os.path.join(get_examples_path(), 'fmri.nii.gz')]
        theargs += [os.path.join(get_test_temp_path(), 'rapidtide2x_testoutput')]
        theargs += ['--limitoutput']
        theargs += ['-s', '25.0']
        theargs += ['-L']
        theargs += ['-r', '-20,20']
        theargs += ['-f', '2']
        theargs += ['--refinepasses=3']
        theargs += ['--refineoffset']
        theargs += ['--despecklepasses=4']
        theargs += ['--accheck']
        theargs += ['--nprocs=-1']
        theargs += ['--saveoptionsasjson']
        theargs += ['--detrendorder=3']
        theargs += ['--pickleft']
        rapidtidecmd = [tide_util.findexecutable('rapidtide2x')] + theargs
        subprocess.call(rapidtidecmd)
    
    diffmaps = tide_util.comparerapidtideruns(os.path.join(get_test_temp_path(), 'rapidtide2x_testoutput'), os.path.join(get_test_target_path(), 'rapidtide2x_target'))

    for mapname, maps in diffmaps.items():
        print('examining', mapname)
        assert maps['relmindiff'] < 1e-4
        assert maps['relmaxdiff'] < 1e-4
        assert maps['relmeandiff'] < 1e-4
        assert maps['relmse'] < 1e-4

def main():
    test_rapidtide2x(debug=True)


if __name__ == '__main__':
    main()
