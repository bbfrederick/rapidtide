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
import multiprocessing as mp
import sys
import threading as thread
from platform import python_version, system
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

try:
    import queue as thrQueue
except ImportError:
    import Queue as thrQueue


def maxcpus(reservecpu: bool = True) -> int:
    """Return the maximum number of CPUs that can be used for parallel processing.

    This function returns the total number of CPU cores available on the system,
    with an option to reserve one CPU core for system operations.

    Parameters
    ----------
    reservecpu : bool, default=True
        If True, reserves one CPU core for system operations by returning
        `cpu_count() - 1`. If False, returns the total number of CPU cores
        available without reservation.

    Returns
    -------
    int
        The maximum number of CPUs available for parallel processing.
        If `reservecpu=True`, returns `cpu_count() - 1`.
        If `reservecpu=False`, returns `cpu_count()`.

    Notes
    -----
    This function uses `multiprocessing.cpu_count()` to determine the number
    of available CPU cores. The reserved CPU core helps maintain system
    responsiveness during parallel processing tasks.

    Examples
    --------
    >>> maxcpus()
    7
    >>> maxcpus(reservecpu=False)
    8
    """
    if reservecpu:
        return mp.cpu_count() - 1
    else:
        return mp.cpu_count()


def _process_data(
    data_in: List[Any], inQ: Any, outQ: Any, showprogressbar: bool = True, chunksize: int = 10000
) -> List[Any]:
    """Process input data in chunks using multiprocessing queues.

    This function distributes data into chunks and processes them using
    provided input and output queues. It supports progress tracking and
    handles both complete chunks and a final remainder chunk.

    Parameters
    ----------
    data_in : List[Any]
        Input data to be processed.
    inQ : Any
        Input queue for sending data to worker processes.
    outQ : Any
        Output queue for receiving processed data from worker processes.
    showprogressbar : bool, optional
        If True, display a progress bar during processing. Default is True.
    chunksize : int, optional
        Size of data chunks to process at a time. Default is 10000.

    Returns
    -------
    List[Any]
        List of processed data items retrieved from the output queue.

    Notes
    -----
    This function assumes that `inQ` and `outQ` are properly configured
    multiprocessing queues and that worker processes are running and
    consuming from `inQ` and producing to `outQ`.

    Examples
    --------
    >>> from multiprocessing import Queue
    >>> data = list(range(1000))
    >>> in_q = Queue()
    >>> out_q = Queue()
    >>> result = _process_data(data, in_q, out_q)
    """
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
        if remainder != 0:
            for i, dat in enumerate(
                data_in[numchunks * chunksize : numchunks * chunksize + remainder]
            ):
                inQ.put(dat)
            numreturned = 0

            # retrieve the remainder
            while True:
                ret = outQ.get()
                if ret is not None:
                    data_out.append(ret)
                numreturned += 1
                pbar.update(1)
                if numreturned > remainder - 1:
                    break
    if showprogressbar:
        print()

    return data_out


def run_multiproc(
    consumerfunc: Callable[[Any, Any], None],
    inputshape: Tuple[int, ...],
    maskarray: Optional[NDArray] = None,
    nprocs: int = 1,
    verbose: bool = True,
    indexaxis: int = 0,
    procunit: str = "voxels",
    showprogressbar: bool = True,
    chunksize: int = 1000,
) -> List[Any]:
    """
    Execute a function in parallel across multiple processes using multiprocessing.

    This function initializes a set of worker processes and distributes input data
    across them for parallel processing. It supports optional masking of data
    along a specified axis and provides progress reporting.

    Parameters
    ----------
    consumerfunc : callable
        Function to be executed in parallel. Must accept two arguments: an input queue
        and an output queue for inter-process communication.
    inputshape : tuple of int
        Shape of the input data along all axes. The dimension along `indexaxis` is
        used to determine the number of items to process.
    maskarray : ndarray, optional
        Boolean or binary mask array used to filter indices. Only indices where
        `maskarray[d] > 0.5` are processed. If None, all indices are processed.
    nprocs : int, optional
        Number of worker processes to use. Default is 1 (single-threaded).
    verbose : bool, optional
        If True, print information about the number of units being processed.
        Default is True.
    indexaxis : int, optional
        Axis along which to iterate for processing. Default is 0.
    procunit : str, optional
        Unit of processing, used for logging messages. Default is "voxels".
    showprogressbar : bool, optional
        If True, display a progress bar during processing. Default is True.
    chunksize : int, optional
        Number of items to process in each chunk. Default is 1000.

    Returns
    -------
    list
        List of results returned by the worker processes.

    Notes
    -----
    - On Python 3.8+ and non-Windows systems, the function uses the 'fork' context
      for better performance.
    - The function will exit with an error if `maskarray` is provided but its
      length does not match the size of the `indexaxis` dimension of `inputshape`.

    Examples
    --------
    >>> def worker_func(inQ, outQ):
    ...     while True:
    ...         item = inQ.get()
    ...         if item is None:
    ...             break
    ...         outQ.put(item * 2)
    ...
    >>> shape = (100, 100)
    >>> result = run_multiproc(worker_func, shape, nprocs=4)
    """
    # initialize the workers and the queues
    __spec__ = None
    n_workers = nprocs
    versioninfo = python_version().split(".")
    if (versioninfo[0] == "3") and (int(versioninfo[1]) >= 8) and (system() != "Windows"):
        cleanup = None
        ctx = mp.get_context("fork")
        inQ = ctx.Queue()
        outQ = ctx.Queue()
        # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        workers = [ctx.Process(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
        # signal.signal(signal.SIGINT, original_sigint_handler)
    else:
        cleanup = None  # just disable this for now
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
    for i, w in enumerate(workers):
        w.start()

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
    if verbose:
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
    consumerfunc: Callable[[Any, Any], None],
    inputshape: Tuple[int, ...],
    maskarray: Optional[NDArray] = None,
    verbose: bool = True,
    nprocs: int = 1,
    indexaxis: int = 0,
    procunit: str = "voxels",
    showprogressbar: bool = True,
    chunksize: int = 1000,
) -> List[Any]:
    """
    Execute a multithreaded processing task using a specified consumer function.

    This function initializes a set of worker threads that process data in parallel
    according to the provided consumer function. It supports optional masking,
    progress tracking, and configurable chunking for efficient processing.

    Parameters
    ----------
    consumerfunc : callable
        A function that takes two arguments (input queue, output queue) and
        processes data in a loop until signaled to stop.
    inputshape : tuple of int
        Shape of the input data along all axes. The dimension along `indexaxis`
        determines how many items will be processed.
    maskarray : ndarray, optional
        Boolean or integer array used to filter which indices are processed.
        Must match the size of the axis specified by `indexaxis`.
    verbose : bool, optional
        If True, print information about the number of items being processed
        and the number of threads used. Default is True.
    nprocs : int, optional
        Number of worker threads to spawn. Default is 1.
    indexaxis : int, optional
        Axis along which the indexing is performed. Default is 0.
    procunit : str, optional
        Unit of processing, used in verbose output. Default is "voxels".
    showprogressbar : bool, optional
        If True, display a progress bar during processing. Default is True.
    chunksize : int, optional
        Number of items to process in each chunk. Default is 1000.

    Returns
    -------
    list
        A list of results returned by the consumer function for each processed item.

    Notes
    -----
    - The function uses `threading.Queue` for inter-thread communication.
    - If `maskarray` is provided, only indices where `maskarray[d] > 0` are processed.
    - The `consumerfunc` is expected to read from `inQ` and write to `outQ` until
      a `None` is received on `inQ`, signaling the end of processing.

    Examples
    --------
    >>> def my_consumer(inQ, outQ):
    ...     while True:
    ...         item = inQ.get()
    ...         if item is None:
    ...             break
    ...         result = item * 2
    ...         outQ.put(result)
    ...
    >>> shape = (100, 50)
    >>> result = run_multithread(my_consumer, shape, nprocs=4)
    """
    # initialize the workers and the queues
    n_workers = nprocs
    inQ = thrQueue.Queue()
    outQ = thrQueue.Queue()
    workers = [thread.Thread(target=consumerfunc, args=(inQ, outQ)) for i in range(n_workers)]
    for i, w in enumerate(workers):
        w.start()

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
    if verbose:
        print("processing", len(data_in), procunit + " with", n_workers, "threads")
    data_out = _process_data(
        data_in, inQ, outQ, showprogressbar=showprogressbar, chunksize=chunksize
    )

    # shut down workers
    for i in range(n_workers):
        inQ.put(None)

    return data_out
