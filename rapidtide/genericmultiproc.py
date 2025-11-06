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
from typing import Any, Callable

from numpy.typing import NDArray
from tqdm import tqdm

import rapidtide.multiproc as tide_multiproc


def run_multiproc(
    voxelfunc: Callable,
    packfunc: Callable,
    unpackfunc: Callable,
    voxelargs: list,
    voxelproducts: list,
    inputshape: tuple,
    voxelmask: NDArray,
    LGR: logging.Logger | None,
    nprocs: int,
    alwaysmultiproc: bool,
    showprogressbar: bool,
    chunksize: int,
    indexaxis: int = 0,
    procunit: str = "voxels",
    debug: bool = False,
    **kwargs: Any,
) -> int:
    """
    Execute voxel-wise processing with optional multiprocessing support.

    This function performs voxel-wise computations using the provided functions
    for processing, packing, and unpacking data. It supports both single-threaded
    and multi-process execution depending on the input parameters.

    Parameters
    ----------
    voxelfunc : callable
        Function to be applied to each voxel. It should accept a voxel index,
        packed arguments, and optional debug and keyword arguments.
    packfunc : callable
        Function to pack voxel arguments for use in `voxelfunc`. It takes a voxel
        index and `voxelargs` as input.
    unpackfunc : callable
        Function to unpack results from `voxelfunc` and store them in `voxelproducts`.
    voxelargs : list
        List of arguments to be passed to `packfunc`.
    voxelproducts : list
        List of arrays or data structures to be updated with results from `unpackfunc`.
    inputshape : tuple
        Shape of the input data, used to determine the number of voxels to process.
    voxelmask : ndarray
        Boolean or integer mask indicating which voxels to process.
    LGR : logging.Logger or None
        Logger instance for logging messages. If None, no logging is performed.
    nprocs : int
        Number of processes to use for multiprocessing. If 1, single-threaded execution is used.
    alwaysmultiproc : bool
        If True, forces multiprocessing even when `nprocs` is 1.
    showprogressbar : bool
        If True, displays a progress bar during processing.
    chunksize : int
        Size of chunks to process in each step for multiprocessing.
    indexaxis : int, optional
        Axis along which to iterate over voxels, default is 0.
    procunit : str, optional
        Unit of processing for progress bar, default is "voxels".
    debug : bool, optional
        If True, prints debug information, default is False.
    **kwargs : dict
        Additional keyword arguments passed to `voxelfunc`.

    Returns
    -------
    int
        Total number of voxels processed.

    Notes
    -----
    - If `nprocs` > 1 or `alwaysmultiproc` is True, multiprocessing is used.
    - Otherwise, a single-threaded loop is used with optional progress bar.
    - The function uses `tide_multiproc.run_multiproc` internally for multiprocessing.
    - Garbage collection is performed after processing.

    Examples
    --------
    >>> run_multiproc(
    ...     voxelfunc=my_voxel_func,
    ...     packfunc=pack_voxel_args,
    ...     unpackfunc=unpack_results,
    ...     voxelargs=[arg1, arg2],
    ...     voxelproducts=[output_array],
    ...     inputshape=(100, 100, 100),
    ...     voxelmask=mask_array,
    ...     LGR=None,
    ...     nprocs=4,
    ...     alwaysmultiproc=False,
    ...     showprogressbar=True,
    ...     chunksize=10,
    ... )
    1000
    """
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
