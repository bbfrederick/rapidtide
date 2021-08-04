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

try:
    import queue as thrQueue
except ImportError:
    import Queue as thrQueue

import rapidtide.util as tide_util


def maxcpus():
    return mp.cpu_count() - 1


def _process_data(data_in, inQ, outQ, showprogressbar=True, reportstep=1000, chunksize=10000):
    # send pos/data to workers
    data_out = []
    totalnum = len(data_in)
    numchunks = int(totalnum // chunksize)
    remainder = totalnum - numchunks * chunksize
    if showprogressbar:
        tide_util.progressbar(0, totalnum, label="Percent complete")

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
            if (((numreturned + offset + 1) % reportstep) == 0) and showprogressbar:
                tide_util.progressbar(numreturned + offset + 1, totalnum, label="Percent complete")
            if numreturned > chunksize - 1:
                break

    # queue the remainder
    for i, dat in enumerate(data_in[numchunks * chunksize : numchunks * chunksize + remainder]):
        inQ.put(dat)
    numreturned = 0
    offset = numchunks * chunksize

    # retrieve the remainder
    while True:
        ret = outQ.get()
        if ret is not None:
            data_out.append(ret)
        numreturned += 1
        if (((numreturned + offset + 1) % reportstep) == 0) and showprogressbar:
            tide_util.progressbar(numreturned + offset + 1, totalnum, label="Percent complete")
        if numreturned > remainder - 1:
            break
    if showprogressbar:
        tide_util.progressbar(totalnum, totalnum, label="Percent complete")
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
    n_workers = nprocs
    versioninfo = python_version().split(".")
    if (versioninfo[0] == "3") and (versioninfo[1] >= "8") and (system() != "Windows"):
        ctx = mp.get_context("fork")
        inQ = ctx.Queue()
        outQ = ctx.Queue()
        workers = [ctx.Process(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
    else:
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
        w.terminate()
        w.join()

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
    # for w in workers:
    #   #.terminate()
    #   w.join()

    return data_out
