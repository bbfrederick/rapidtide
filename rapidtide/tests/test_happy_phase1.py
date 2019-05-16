#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt 
from scipy import arange
import os

import rapidtide.io as tide_io
import rapidtide.fit as tide_fit
import rapidtide.util as tide_util
import rapidtide.workflows.happy as happy_workflow
from rapidtide.tests.utils import get_test_data_path, get_test_target_path, get_test_temp_path, get_examples_path, get_rapidtide_root, get_scripts_path, create_dir


def test_happy_phase1(debug=False):
    recalculate = True
    if recalculate:
        # create outputdir if it doesn't exist
        create_dir(get_test_temp_path())
    
        # and launch the processing
        theargs = ['happy']
        theargs += [os.path.join(get_examples_path(), 'happyfmri.nii.gz')]
        theargs += [os.path.join(get_examples_path(), 'happyfmri.json')]
        theargs += [os.path.join(get_test_temp_path(), 'happy_phase1output')]
        theargs += ['--dodlfilter']
        theargs += ['--saveinfoasjson']
        theargs += ['--glm']
        theargs += ['--numskip=0']
        theargs += ['--gridbins=3.0']
        theargs += ['--gridkernel=kaiser']
        theargs += ['--model=model_revised']
        theargs += ['--estmask=' + os.path.join(get_test_target_path(), 'happy_phase1target_vesselmask.nii.gz')]
        happy_workflow.happy_main(theargs)
    
    diffmaps = tide_util.comparehappyruns(os.path.join(get_test_temp_path(), 'happy_phase1output'), os.path.join(get_test_target_path(), 'happy_phase1target'), debug=debug)

    for mapname, maps in diffmaps.items():
        print('checking', mapname)
        print('\trelmindiff', maps['relmindiff'])
        print('\trelmaxdiff', maps['relmaxdiff'])
        print('\trelmeandiff', maps['relmeandiff'])
        print('\trelmse', maps['relmse'])
        assert maps['relmindiff'] < 1e2
        assert maps['relmaxdiff'] < 1e2
        assert maps['relmeandiff'] < 1e-2
        assert maps['relmse'] < 1e2

def main():
    test_happy_phase1(debug=True)


if __name__ == '__main__':
    main()
