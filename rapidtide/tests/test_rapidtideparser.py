#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
from __future__ import print_function, division

import numpy as np

from rapidtide.workflows.rapidtide import process_args

testlist = {}
testlist['searchrange'] = {
    'command':    ['--searchrange', '-7', '15.2'],
    'results':    [['lagmin', -7.0], ['lagmax', 15.2]]
    }
testlist['pickleft'] = {
    'command':    ['--pickleft'],
    'results':    [['pickleft', True]]
    }

def main():

    # construct the test vector
    testvec = []
    testvec.append('searchrange')
    testvec.append('pickleft')

    print(testlist)
    print(testvec)
    
    # make the argument and results lists
    arglist = ['../data/examples/src/fmri.nii.gz', '../data/examples/dst/parsertestdummy']
    resultlist = []
    for thetest in testvec:
        arglist += testlist[thetest]['command']
        for theresult in testlist[thetest]['results']:
            resultlist += [theresult]
    

    print(arglist)
    print(resultlist)
        
    theargs, ncprefilter = process_args(inputargs=arglist)

    for thetest in testvec:
        for theresult in testlist[thetest]['results']:
            print('testing', testlist[thetest]['command'], 'effect on', theresult[0])
            print(theargs[theresult[0]], theresult[1])
            assert theargs[theresult[0]] == theresult[1]

if __name__ == "__main__":
    main()

