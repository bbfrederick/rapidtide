#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
"""Tests for rapidtide.multiproc, the process and thread fan-out machinery."""

import multiprocessing as mp
import queue as thrQueue
import threading as thread
from unittest.mock import patch

import numpy as np
import pytest

import rapidtide.multiproc as tide_multiproc

# ==================== consumers ====================
#
# These live at module scope on purpose.  run_multiproc hands the consumer to a
# forked child, and a module level function is inherited cleanly by the child
# whichever start method is in play.


def _doublingconsumer(inQ, outQ):
    """Return (index, 2*index) for every index, until the TERM signal arrives.

    Parameters
    ----------
    inQ : Queue
        Source of voxel indices.  A None value means stop.
    outQ : Queue
        Destination for results.

    Returns
    -------
    None
    """
    while True:
        val = inQ.get()
        if val is None:
            break
        outQ.put((val, 2 * val))


def _skippingconsumer(inQ, outQ):
    """Return a result for even indices and None for odd ones.

    A consumer is allowed to report "nothing to see here" by returning None, which
    the collector counts as done but does not accumulate.  This exercises that.

    Parameters
    ----------
    inQ : Queue
        Source of voxel indices.  A None value means stop.
    outQ : Queue
        Destination for results.

    Returns
    -------
    None
    """
    while True:
        val = inQ.get()
        if val is None:
            break
        outQ.put(None if (val % 2) else (val, 2 * val))


# ==================== stand ins for workers ====================


class _FakeWorker:
    """A worker that is never actually started.

    run_multiproc and run_multithread both spawn their workers *before* validating
    the mask, so a test of that validation would strand real workers blocked on an
    empty queue - live threads would hang interpreter shutdown, and live processes
    would be orphaned.  Substituting this makes the failure path safe to test.

    Parameters
    ----------
    target : callable, optional
        Ignored; recorded so a test can assert what would have been run.
    args : tuple, optional
        Ignored; recorded alongside target.
    """

    def __init__(self, target=None, args=()):
        self.target = target
        self.args = args
        self.started = False

    def start(self):
        """Record the start without running anything."""
        self.started = True

    def join(self):
        """Accept a join request; there is nothing to wait for."""
        return None

    def close(self):
        """Accept a close request; there is nothing to release."""
        return None


class _ThreadAsProcess(thread.Thread):
    """A Thread wearing the part of the Process API that run_multiproc uses.

    Lets the non-fork branch of run_multiproc - the one taken on Windows, or on
    Python older than 3.8 - be executed for real on this machine, rather than left
    permanently unreached.  Thread supplies start and join; only close is missing.
    """

    def close(self):
        """Accept the close a Process would need, as a no-op."""
        return None


class _FakeContext:
    """Stands in for a multiprocessing context, handing out fake workers.

    Attributes
    ----------
    workers : list of _FakeWorker
        Every worker handed out, in order of creation.
    """

    def __init__(self):
        self.workers = []

    def Queue(self):
        """Return an ordinary in-process queue in place of a multiprocessing one."""
        return thrQueue.Queue()

    def Process(self, target=None, args=()):
        """Return a worker that does nothing, and remember it."""
        theworker = _FakeWorker(target=target, args=args)
        self.workers.append(theworker)
        return theworker


# ==================== maxcpus ====================


def maxcpus_reserves_a_core_unless_told_not_to(debug=False):
    """The reservation is what keeps a full-tilt run from starving the machine."""
    if debug:
        print("maxcpus_reserves_a_core_unless_told_not_to")

    thetotal = mp.cpu_count()
    assert tide_multiproc.maxcpus(reservecpu=False) == thetotal
    assert tide_multiproc.maxcpus(reservecpu=True) == thetotal - 1
    # reserving is the default, so an unqualified call must not claim the whole box
    assert tide_multiproc.maxcpus() == thetotal - 1


# ==================== _process_data ====================


def _collectwithathread(theconsumer, thedata, **thekwargs):
    """Run _process_data against a single live worker thread.

    Parameters
    ----------
    theconsumer : callable
        Consumer function taking (inQ, outQ).
    thedata : list
        Indices to feed through.
    **thekwargs
        Passed through to _process_data.

    Returns
    -------
    list
        Whatever _process_data collected.
    """
    inQ = thrQueue.Queue()
    outQ = thrQueue.Queue()
    theworker = thread.Thread(target=theconsumer, args=(inQ, outQ))
    theworker.start()
    try:
        theresults = tide_multiproc._process_data(thedata, inQ, outQ, **thekwargs)
    finally:
        inQ.put(None)
        theworker.join()
    return theresults


def process_data_handles_a_partial_final_chunk(debug=False):
    """Work is shipped a chunk at a time, and the last chunk is usually short.

    The remainder is handled by a separate block of code from the full chunks, so a
    count that divides evenly would leave it unexecuted.
    """
    if debug:
        print("process_data_handles_a_partial_final_chunk")

    # 25 items at 10 per chunk is two full chunks and a remainder of five
    theresults = _collectwithathread(
        _doublingconsumer, list(range(25)), showprogressbar=False, chunksize=10
    )

    assert len(theresults) == 25
    assert sorted(theresults) == [(i, 2 * i) for i in range(25)]


def process_data_handles_an_exact_multiple(debug=False):
    """With no remainder the trailing block must be skipped, not run on nothing."""
    if debug:
        print("process_data_handles_an_exact_multiple")

    theresults = _collectwithathread(
        _doublingconsumer, list(range(20)), showprogressbar=False, chunksize=10
    )

    assert len(theresults) == 20
    assert sorted(theresults) == [(i, 2 * i) for i in range(20)]


def process_data_counts_but_discards_empty_results(debug=False):
    """A None from a consumer means "no result", not "no work".

    If Nones were counted as missing the collector would wait forever for returns
    that are never coming, so this pins the accounting as well as the filtering.
    """
    if debug:
        print("process_data_counts_but_discards_empty_results")

    theresults = _collectwithathread(
        _skippingconsumer, list(range(21)), showprogressbar=False, chunksize=4
    )

    # the odd indices reported nothing, so only the eleven even ones survive
    assert sorted(theresults) == [(i, 2 * i) for i in range(0, 21, 2)]


def process_data_handles_a_chunk_larger_than_the_data(debug=False):
    """A chunk size above the item count means everything is one short remainder."""
    if debug:
        print("process_data_handles_a_chunk_larger_than_the_data")

    theresults = _collectwithathread(
        _doublingconsumer, list(range(5)), showprogressbar=False, chunksize=1000
    )

    assert sorted(theresults) == [(i, 2 * i) for i in range(5)]


def process_data_handles_no_data_at_all(debug=False):
    """An empty mask leaves nothing to do, and that has to return rather than block."""
    if debug:
        print("process_data_handles_no_data_at_all")

    assert _collectwithathread(_doublingconsumer, [], showprogressbar=False, chunksize=10) == []


def process_data_can_show_a_progress_bar(capsys, debug=False):
    """The progress bar path adds a trailing newline, and must not disturb results."""
    if debug:
        print("process_data_can_show_a_progress_bar")

    theresults = _collectwithathread(
        _doublingconsumer, list(range(6)), showprogressbar=True, chunksize=4
    )

    assert sorted(theresults) == [(i, 2 * i) for i in range(6)]
    assert capsys.readouterr().out == "\n"


# ==================== run_multithread ====================


def run_multithread_processes_every_index(debug=False):
    """With no mask, every index along the chosen axis is work."""
    if debug:
        print("run_multithread_processes_every_index")

    theresults = tide_multiproc.run_multithread(
        _doublingconsumer,
        (12, 40),
        nprocs=3,
        verbose=False,
        showprogressbar=False,
        chunksize=5,
    )

    assert sorted(theresults) == [(i, 2 * i) for i in range(12)]


def run_multithread_honours_the_mask(debug=False):
    """A mask is how a run skips voxels outside the brain, so only those survive."""
    if debug:
        print("run_multithread_honours_the_mask")

    themask = np.zeros(10, dtype=np.float64)
    themask[[1, 4, 7]] = 1.0
    theresults = tide_multiproc.run_multithread(
        _doublingconsumer,
        (10, 40),
        maskarray=themask,
        nprocs=2,
        verbose=False,
        showprogressbar=False,
        chunksize=2,
    )

    assert sorted(theresults) == [(i, 2 * i) for i in [1, 4, 7]]


def run_multithread_indexes_the_axis_it_is_told_to(debug=False):
    """indexaxis picks which dimension is iterated over; the other one is untouched."""
    if debug:
        print("run_multithread_indexes_the_axis_it_is_told_to")

    theresults = tide_multiproc.run_multithread(
        _doublingconsumer,
        (100, 7),
        indexaxis=1,
        nprocs=2,
        verbose=False,
        showprogressbar=False,
        chunksize=3,
    )

    assert sorted(theresults) == [(i, 2 * i) for i in range(7)]


def run_multithread_reports_what_it_is_doing(capsys, debug=False):
    """The verbose line names the unit, which is how a log tells voxels from slices."""
    if debug:
        print("run_multithread_reports_what_it_is_doing")

    tide_multiproc.run_multithread(
        _doublingconsumer,
        (6, 40),
        nprocs=2,
        verbose=True,
        procunit="slices",
        showprogressbar=False,
        chunksize=3,
    )

    theoutput = capsys.readouterr().out
    assert "processing 6 slices with 2 threads" in theoutput


def run_multithread_rejects_a_mismatched_mask(debug=False):
    """A mask of the wrong length means the caller has confused two axes, and
    processing it anyway would silently analyse the wrong voxels."""
    if debug:
        print("run_multithread_rejects_a_mismatched_mask")

    # the workers are started before the check, so they are stubbed out here; real
    # threads would be left blocked on an empty queue and would hang interpreter exit
    with patch.object(tide_multiproc.thread, "Thread", _FakeWorker):
        with pytest.raises(SystemExit):
            tide_multiproc.run_multithread(
                _doublingconsumer,
                (10, 40),
                maskarray=np.ones(9, dtype=np.float64),
                nprocs=1,
                verbose=False,
                showprogressbar=False,
            )


# ==================== run_multiproc ====================


def run_multiproc_processes_every_index(debug=False):
    """The real thing: forked worker processes, over a deliberately small problem."""
    if debug:
        print("run_multiproc_processes_every_index")

    theresults = tide_multiproc.run_multiproc(
        _doublingconsumer,
        (8, 40),
        nprocs=2,
        verbose=False,
        showprogressbar=False,
        chunksize=3,
    )

    assert sorted(theresults) == [(i, 2 * i) for i in range(8)]


def run_multiproc_honours_the_mask(debug=False):
    """Masked out indices are never queued, so no worker ever sees them."""
    if debug:
        print("run_multiproc_honours_the_mask")

    themask = np.zeros(8, dtype=np.float64)
    themask[[0, 3, 5]] = 1.0
    theresults = tide_multiproc.run_multiproc(
        _doublingconsumer,
        (8, 40),
        maskarray=themask,
        nprocs=2,
        verbose=False,
        showprogressbar=False,
        chunksize=2,
    )

    assert sorted(theresults) == [(i, 2 * i) for i in [0, 3, 5]]


def run_multiproc_thresholds_the_mask_at_a_half(debug=False):
    """Both entry points threshold the mask at 0.5, and must agree.

    A soft-edged mask - the kind a resampled or partial-volume mask produces - is
    where the two could diverge, so it is a fractional mask that is used here rather
    than a binary one.  The threshold matters: a workflow that switches between
    processes and threads must analyse the same voxels either way.
    """
    if debug:
        print("run_multiproc_thresholds_the_mask_at_a_half")

    themask = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    theprocresults = tide_multiproc.run_multiproc(
        _doublingconsumer,
        (5, 40),
        maskarray=themask,
        nprocs=2,
        verbose=False,
        showprogressbar=False,
        chunksize=2,
    )
    thethreadresults = tide_multiproc.run_multithread(
        _doublingconsumer,
        (5, 40),
        maskarray=themask,
        nprocs=2,
        verbose=False,
        showprogressbar=False,
        chunksize=2,
    )

    # 0.5 is not greater than 0.5, so only 0.75 and 1.0 are in
    assert sorted(theprocresults) == [(3, 6), (4, 8)]
    assert sorted(thethreadresults) == sorted(theprocresults)


def run_multiproc_reports_what_it_is_doing(capsys, debug=False):
    """The verbose line counts processes, not threads."""
    if debug:
        print("run_multiproc_reports_what_it_is_doing")

    tide_multiproc.run_multiproc(
        _doublingconsumer,
        (4, 40),
        nprocs=2,
        verbose=True,
        showprogressbar=False,
        chunksize=2,
    )

    assert "processing 4 voxels with 2 processes" in capsys.readouterr().out


def run_multiproc_rejects_a_mismatched_mask(debug=False):
    """As for run_multithread, a mask that does not match the index axis is fatal."""
    if debug:
        print("run_multiproc_rejects_a_mismatched_mask")

    thecontext = _FakeContext()
    with patch.object(tide_multiproc.mp, "get_context", return_value=thecontext):
        with pytest.raises(SystemExit):
            tide_multiproc.run_multiproc(
                _doublingconsumer,
                (10, 40),
                maskarray=np.ones(3, dtype=np.float64),
                nprocs=2,
                verbose=False,
                showprogressbar=False,
            )

    # the workers really were spawned before anyone checked the mask
    assert len(thecontext.workers) == 2
    assert all(w.started for w in thecontext.workers)


def run_multiproc_has_a_non_fork_path(debug=False):
    """Fork is unavailable on Windows and on Python before 3.8, and the fallback
    branch that covers those cases is otherwise never executed on this machine.

    Threads stand in for processes so the branch can be driven end to end; what is
    under test is the branch selection and the surrounding queueing, both of which
    are indifferent to whether the workers are processes or threads.
    """
    if debug:
        print("run_multiproc_has_a_non_fork_path")

    with (
        patch.object(tide_multiproc, "system", return_value="Windows"),
        patch.object(tide_multiproc.mp, "Queue", thrQueue.Queue),
        patch.object(tide_multiproc.mp, "Process", _ThreadAsProcess),
        patch.object(
            tide_multiproc.mp, "get_context", side_effect=AssertionError("took the fork path")
        ),
    ):
        theresults = tide_multiproc.run_multiproc(
            _doublingconsumer,
            (6, 40),
            nprocs=2,
            verbose=False,
            showprogressbar=False,
            chunksize=4,
        )

    assert sorted(theresults) == [(i, 2 * i) for i in range(6)]


def run_multiproc_has_a_legacy_python_path(debug=False):
    """The same fallback is selected by the Python version, independently of the OS."""
    if debug:
        print("run_multiproc_has_a_legacy_python_path")

    with (
        patch.object(tide_multiproc, "python_version", return_value="3.7.9"),
        patch.object(tide_multiproc.mp, "Queue", thrQueue.Queue),
        patch.object(tide_multiproc.mp, "Process", _ThreadAsProcess),
        patch.object(
            tide_multiproc.mp, "get_context", side_effect=AssertionError("took the fork path")
        ),
    ):
        theresults = tide_multiproc.run_multiproc(
            _doublingconsumer,
            (5, 40),
            nprocs=1,
            verbose=False,
            showprogressbar=False,
            chunksize=2,
        )

    assert sorted(theresults) == [(i, 2 * i) for i in range(5)]


# ==================== entry points ====================


def test_multiproc(debug=False):
    """Entry point for the sub-tests that need no fixtures."""
    maxcpus_reserves_a_core_unless_told_not_to(debug=debug)
    process_data_handles_a_partial_final_chunk(debug=debug)
    process_data_handles_an_exact_multiple(debug=debug)
    process_data_counts_but_discards_empty_results(debug=debug)
    process_data_handles_a_chunk_larger_than_the_data(debug=debug)
    process_data_handles_no_data_at_all(debug=debug)
    run_multithread_processes_every_index(debug=debug)
    run_multithread_honours_the_mask(debug=debug)
    run_multithread_indexes_the_axis_it_is_told_to(debug=debug)
    run_multithread_rejects_a_mismatched_mask(debug=debug)
    run_multiproc_processes_every_index(debug=debug)
    run_multiproc_honours_the_mask(debug=debug)
    run_multiproc_thresholds_the_mask_at_a_half(debug=debug)
    run_multiproc_rejects_a_mismatched_mask(debug=debug)
    run_multiproc_has_a_non_fork_path(debug=debug)
    run_multiproc_has_a_legacy_python_path(debug=debug)


def test_multiprocmessages(capsys):
    """Entry point for the sub-tests that capture printed output."""
    process_data_can_show_a_progress_bar(capsys)
    run_multithread_reports_what_it_is_doing(capsys)
    run_multiproc_reports_what_it_is_doing(capsys)


if __name__ == "__main__":
    test_multiproc(debug=True)
