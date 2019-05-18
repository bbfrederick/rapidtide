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
from rapidtide.tests.utils import get_test_target_path, get_test_temp_path, get_examples_path, create_dir


def test_happy_phase2(debug=False):
    recalculate = True
    if recalculate:
        # create outputdir if it doesn't exist
        create_dir(get_test_temp_path())

        # trigger the usage function
        happy_workflow.usage()
    
        # and launch the processing
        theargs = ['happy']
        theargs += [os.path.join(get_examples_path(), 'happyfmri.nii.gz')]
        theargs += [os.path.join(get_examples_path(), 'happyfmri.json')]
        theargs += [os.path.join(get_test_temp_path(), 'happy_phase2output')]
        theargs += ['--debug']
        theargs += ['--numskip=0']
        happy_workflow.happy_main(theargs)

    assert True

def main():
    test_happy_phase2(debug=True)


if __name__ == '__main__':
    main()
