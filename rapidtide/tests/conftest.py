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
"""Shared pytest fixtures for rapidtide tests."""

import inspect
import io
import sys
import tempfile
from contextlib import ExitStack
from unittest.mock import MagicMock

import numpy as np
import pytest
from rapidtide.tests._mplsetup import configure_matplotlib_env

configure_matplotlib_env()


@pytest.fixture
def case_runner(request):
    """Run legacy case-style test functions with debug defaults and fixture injection."""

    def _run(case_func, **overrides):
        kwargs = {"debug": False}
        kwargs.update(overrides)
        accepted_kwargs = {}
        sig = inspect.signature(case_func)
        for name, param in sig.parameters.items():
            if name in kwargs:
                accepted_kwargs[name] = kwargs[name]
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                try:
                    accepted_kwargs[name] = request.getfixturevalue(name)
                except pytest.FixtureLookupError:
                    if param.default is inspect.Parameter.empty:
                        raise
        return case_func(**accepted_kwargs)

    return _run


@pytest.fixture
def parse_with_temp_inputs():
    """Parse args with temporary input files for parser tests."""

    def _parse(parser, extra_args=None, num_inputs=2, suffix=".txt", include_output=False):
        if extra_args is None:
            extra_args = []
        with ExitStack() as stack:
            files = [stack.enter_context(tempfile.NamedTemporaryFile(suffix=suffix)) for _ in range(num_inputs)]
            argv = [f.name for f in files]
            if include_output:
                argv.append("out")
            argv.extend(extra_args)
            return parser.parse_args(argv)

    return _parse


@pytest.fixture
def capture_stdout():
    """Capture stdout from a callable and return it as a string."""

    def _capture(func, *args, **kwargs):
        stream = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stream
        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return stream.getvalue()

    return _capture


@pytest.fixture
def broadband_signal_factory():
    """Create deterministic broadband-like synthetic signals."""

    def _make(npts, samplerate, delay=0.0, seed=42, freqmin=0.01, freqmax=0.2, nfreqs=30):
        rng = np.random.RandomState(seed)
        t = np.arange(npts) / samplerate
        signal = np.zeros(npts, dtype=float)
        for _ in range(nfreqs):
            freq = rng.uniform(freqmin, freqmax)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal += np.sin(2.0 * np.pi * freq * (t - delay) + phase)
        return signal

    return _make


@pytest.fixture
def mock_nifti_header_factory():
    """Create a mutable mock NIfTI header with dict-like access and copy()."""

    def _make(pixdim=None, dim=None):
        if pixdim is None:
            pixdim = [0.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        if dim is None:
            dim = [4, 2, 2, 1, 10, 0, 0, 0]
        storage = {"pixdim": list(pixdim), "dim": list(dim)}

        hdr = MagicMock()
        hdr.__getitem__.side_effect = lambda key: storage[key]
        hdr.__setitem__.side_effect = lambda key, val: storage.__setitem__(key, val)

        def _copy():
            return _make(pixdim=list(storage["pixdim"]), dim=list(storage["dim"]))

        hdr.copy = _copy
        return hdr

    return _make


@pytest.fixture
def mock_nifti_data_factory():
    """Create synthetic 4D data with a stable sinusoidal component."""

    def _make(xsize=2, ysize=2, numslices=1, timepoints=10, inputtr=1.0, freq=0.1):
        data = np.zeros((xsize, ysize, numslices, timepoints), dtype=np.float64)
        t = np.arange(timepoints) * inputtr
        base = np.sin(2.0 * np.pi * freq * t)
        for x in range(xsize):
            for y in range(ysize):
                for z in range(numslices):
                    data[x, y, z, :] = base + x + y + z
        return data

    return _make
