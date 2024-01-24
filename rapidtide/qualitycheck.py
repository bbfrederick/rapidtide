#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import argparse
import os
import sys

import numpy as np

from scipy.ndimage import binary_erosion

from rapidtide.RapidtideDataset import RapidtideDataset

def prepmask(inputmask):
    erodedmask = binary_erosion(inputmask)
    return erodedmask

def qualitycheck(
        datafileroot,
        anatname=None,
        geommaskname=None,
        userise=False,
        usecorrout=False,
        useatlas=False,
        forcetr=False,
        forceoffset=False,
        offsettime=0.0,
        verbose=False,
        debug=False,
):
    # read in the dataset
    thedataset = RapidtideDataset(
        "main",
        datafileroot,
        anatname=anatname,
        geommaskname=geommaskname,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        verbose=verbose,
        init_LUT = False,
    )
    themask = thedataset.overlays["lagmask"].data
    thelags = thedataset.overlays["lagtimes"].data
    thewidths = thedataset.overlays["lagsigma"].data
    thestrengths = thedataset.overlays["lagstrengths"].data

    theerodedmask = prepmask(themask)
