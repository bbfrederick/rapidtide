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
import rapidtide.workflows.rapidtide2x as rapidtide2x_workflow
from rapidtide.tests.utils import get_test_target_path, get_test_temp_path, get_examples_path, create_dir


def test_rapidtide2x_phase3(debug=False):
    recalculate = True
    if recalculate:
        # create outputdir if it doesn't exist
        create_dir(get_test_temp_path())

        # trigger the usage function
        rapidtide2x_workflow.usage()
    
        # and launch the processing
        theargs = ['rapidtide2x']
        theargs += [os.path.join(get_examples_path(), 'fmri.nii.gz')]
        theargs += [os.path.join(get_test_temp_path(), 'rapidtide2x_phase3output')]
        theargs += ['--nowindow']
        theargs += ['--windowfunc=hamming']
        theargs += ['--liang', '--eckart', '--phat']
        theargs += ['--usesp']
        theargs += ['--preservefiltering']
        theargs += ['--corrmaskthresh=0.25']
        theargs += ['-I', '-B', '-a', '-M', '-m']
        theargs += ['-C', '-R', '-L', '-V', '-F', '0.01,0.08']
        theargs += ['-v', '--debug']
        theargs += ['--globalmaskmethod=mean']
        theargs += ['--mklthreads=1']
        theargs += ['--nosharedmem']
        theargs += ['-S']
        rapidtide2x_workflow.rapidtide_main(theargs)
    
    diffmaps = tide_util.comparerapidtideruns(os.path.join(get_test_temp_path(), 'rapidtide2x_phase1output'), os.path.join(get_test_target_path(), 'rapidtide2x_phase1target'))

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
    test_rapidtide2x_phase3(debug=True)


if __name__ == '__main__':
    main()
