#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
import gc
import logging

import numpy as np
from tqdm import tqdm

import rapidtide.multiproc as tide_multiproc


def run_multiproc(
    voxelfunc,
    packfunc,
    unpackfunc,
    voxelargs,
    voxelproducts,
    inputshape,
    voxelmask,
    LGR,
    nprocs,
    alwaysmultiproc,
    showprogressbar,
    chunksize,
    indexaxis=0,
    procunit="voxels",
    debug=False,
    **kwargs,
):
    if debug:
        print(f"{len(voxelproducts)=}, {voxelproducts[0].shape}")
    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def theconsumerfunc(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(
                        voxelfunc(
                            val,
                            packfunc(val, voxelargs),
                            debug=debug,
                            **kwargs,
                        )
                    )
                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            theconsumerfunc,
            inputshape,
            voxelmask,
            indexaxis=indexaxis,
            procunit=procunit,
            verbose=(LGR is not None),
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for returnvals in data_out:
            volumetotal += 1
            unpackfunc(returnvals, voxelproducts)
        del data_out
    else:
        volumetotal = 0
        for vox in tqdm(
            range(0, inputshape[indexaxis]),
            desc="Voxel",
            unit=procunit,
            disable=(not showprogressbar),
        ):
            if voxelmask[vox] > 0:
                dothisone = True
            else:
                dothisone = False
            if dothisone:
                returnvals = voxelfunc(
                    vox,
                    packfunc(vox, voxelargs),
                    debug=debug,
                    **kwargs,
                )
                unpackfunc(returnvals, voxelproducts)
                volumetotal += 1

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        if LGR is not None:
            LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        if LGR is not None:
            LGR.info("garbage collected")

    return volumetotal
