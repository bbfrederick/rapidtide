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
# $Author: frederic $
# $Date: 2016/07/11 14:50:43 $
# $Id: rapidtide,v 1.161 2016/07/11 14:50:43 frederic Exp $
#
#
#
import multiprocessing as mp
import sys
import threading as thread
from platform import python_version, system

from tqdm import tqdm

try:
    import queue as thrQueue
except ImportError:
    import Queue as thrQueue


def maxcpus(reservecpu=True):
    if reservecpu:
        return mp.cpu_count() - 1
    else:
        return mp.cpu_count()


def _process_data(data_in, inQ, outQ, showprogressbar=True, reportstep=1000, chunksize=10000):
    # send pos/data to workers
    data_out = []
    totalnum = len(data_in)
    numchunks = int(totalnum // chunksize)
    remainder = totalnum - numchunks * chunksize
    with tqdm(total=totalnum, desc="Voxel", disable=(not showprogressbar)) as pbar:
        # process all of the complete chunks
        for thechunk in range(numchunks):
            # queue the chunk
            for i, dat in enumerate(data_in[thechunk * chunksize : (thechunk + 1) * chunksize]):
                inQ.put(dat)
            offset = thechunk * chunksize

            # retrieve the chunk
            numreturned = 0
            while True:
                ret = outQ.get()
                if ret is not None:
                    data_out.append(ret)
                numreturned += 1
                pbar.update(1)
                if numreturned > chunksize - 1:
                    break

        # queue the remainder
        for i, dat in enumerate(
            data_in[numchunks * chunksize : numchunks * chunksize + remainder]
        ):
            inQ.put(dat)
        numreturned = 0
        offset = numchunks * chunksize

        # retrieve the remainder
        while True:
            ret = outQ.get()
            if ret is not None:
                data_out.append(ret)
            numreturned += 1
            pbar.update(1)
            if numreturned > remainder - 1:
                break

    print()

    return data_out


def run_multiproc(
    consumerfunc,
    inputshape,
    maskarray,
    nprocs=1,
    procbyvoxel=True,
    showprogressbar=True,
    chunksize=1000,
):
    # initialize the workers and the queues
    __spec__ = None
    n_workers = nprocs
    versioninfo = python_version().split(".")
    if (versioninfo[0] == "3") and (int(versioninfo[1]) >= 8) and (system() != "Windows"):
        cleanup = None
        ctx = mp.get_context("fork")
        inQ = ctx.Queue()
        outQ = ctx.Queue()
        workers = [ctx.Process(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
    else:
        """# try adding this magic incantation to get coverage to record multiprocessing properly
        # This fails for python 3.8 and above
        try:
            from pytest_cov.embed import cleanup
        except ImportError:
            cleanup = None
        """
        cleanup = None  # just disable this for now
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
    for i, w in enumerate(workers):
        w.start()

    if procbyvoxel:
        indexaxis = 0
        procunit = "voxels"
    else:
        indexaxis = 1
        procunit = "timepoints"

    # check that the mask array matches the index dimension
    if maskarray is not None:
        if inputshape[indexaxis] != len(maskarray):
            print(
                "run_multiproc: fatal error - maskarray dimension does not equal index axis dimension"
            )
            sys.exit()

    # pack the data and send to workers
    data_in = []
    for d in range(inputshape[indexaxis]):
        if maskarray is None:
            data_in.append(d)
        elif maskarray[d] > 0.5:
            data_in.append(d)
    print("processing", len(data_in), procunit + " with", n_workers, "processes")
    data_out = _process_data(
        data_in, inQ, outQ, showprogressbar=showprogressbar, chunksize=chunksize
    )

    # shut down workers
    for i in range(n_workers):
        inQ.put(None)
    for w in workers:
        w.join()
        w.close()
    if cleanup is not None:
        cleanup()

    return data_out


def run_multithread(
    consumerfunc,
    inputshape,
    maskarray,
    nprocs=1,
    procbyvoxel=True,
    showprogressbar=True,
    chunksize=1000,
):
    # initialize the workers and the queues
    n_workers = nprocs
    inQ = thrQueue.Queue()
    outQ = thrQueue.Queue()
    workers = [thread.Thread(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
    for i, w in enumerate(workers):
        w.start()

    if procbyvoxel:
        indexaxis = 0
        procunit = "voxels"
    else:
        indexaxis = 1
        procunit = "timepoints"

    # check that the mask array matches the index dimension
    if maskarray is not None:
        if inputshape[indexaxis] != len(maskarray):
            print(
                "run_multithread: fatal error - maskarray dimension does not equal index axis dimension"
            )
            sys.exit()

    # pack the data and send to workers
    data_in = []
    for d in range(inputshape[indexaxis]):
        if maskarray is None:
            data_in.append(d)
        elif maskarray[d] > 0:
            data_in.append(d)
    print("processing", len(data_in), procunit + " with", n_workers, "threads")
    data_out = _process_data(
        data_in, inQ, outQ, showprogressbar=showprogressbar, chunksize=chunksize
    )

    # shut down workers
    for i in range(n_workers):
        inQ.put(None)

    return data_out
