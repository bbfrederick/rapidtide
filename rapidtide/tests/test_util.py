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
import os
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import rapidtide.util as tide_util

# ========================= disablenumba =========================


class TestDisablenumba:
    def test_sets_global_flag(self):
        # Save original value
        original = tide_util.donotusenumba
        tide_util.disablenumba()
        assert tide_util.donotusenumba is True
        # Restore
        tide_util.donotusenumba = original


# ========================= checkimports =========================


class TestCheckimports:
    def test_updates_optiondict(self, capsys):
        optiondict = {}
        tide_util.checkimports(optiondict)
        assert "pfftwexists" in optiondict
        assert "donotbeaggressive" in optiondict
        assert "donotusenumba" in optiondict
        assert isinstance(optiondict["pfftwexists"], bool)
        assert isinstance(optiondict["donotbeaggressive"], bool)
        assert isinstance(optiondict["donotusenumba"], bool)


# ========================= disablemkl / enablemkl =========================


class TestDisablemkl:
    def test_no_error_with_multiple_procs(self):
        # Should not raise any error
        tide_util.disablemkl(numprocs=4, debug=False)

    def test_no_error_with_single_proc(self):
        tide_util.disablemkl(numprocs=1, debug=False)


class TestEnablemkl:
    def test_no_error(self):
        # Should not raise any error
        tide_util.enablemkl(numthreads=4, debug=False)


# ========================= checkifincontainer =========================


class TestCheckifincontainer:
    def test_returns_correct_type(self, monkeypatch):
        # Clear all container env vars
        monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
        monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
        monkeypatch.delenv("CIRCLECI", raising=False)
        result = tide_util.checkifincontainer()
        assert result is None

    def test_detects_docker(self, monkeypatch):
        monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
        monkeypatch.delenv("CIRCLECI", raising=False)
        monkeypatch.setenv("RUNNING_IN_CONTAINER", "1")
        result = tide_util.checkifincontainer()
        assert result == "Docker"

    def test_detects_singularity(self, monkeypatch):
        monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
        monkeypatch.delenv("CIRCLECI", raising=False)
        monkeypatch.setenv("SINGULARITY_CONTAINER", "/path/to/container")
        result = tide_util.checkifincontainer()
        assert result == "Singularity"

    def test_detects_circleci(self, monkeypatch):
        monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
        monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
        monkeypatch.setenv("CIRCLECI", "true")
        result = tide_util.checkifincontainer()
        assert result == "CircleCI"

    def test_circleci_overrides_docker(self, monkeypatch):
        monkeypatch.setenv("RUNNING_IN_CONTAINER", "1")
        monkeypatch.setenv("CIRCLECI", "true")
        result = tide_util.checkifincontainer()
        assert result == "CircleCI"


# ========================= formatmemamt =========================


class TestFormatmemamt:
    def test_bytes(self):
        result = tide_util.formatmemamt(512)
        assert result == "512.000B"

    def test_kilobytes(self):
        result = tide_util.formatmemamt(1024)
        assert result == "1.000kB"

    def test_megabytes(self):
        result = tide_util.formatmemamt(1024 * 1024)
        assert result == "1.000MB"

    def test_gigabytes(self):
        result = tide_util.formatmemamt(1024 * 1024 * 1024)
        assert result == "1.000GB"

    def test_terabytes(self):
        result = tide_util.formatmemamt(1024 * 1024 * 1024 * 1024)
        assert result == "1.000TB"

    def test_fractional(self):
        result = tide_util.formatmemamt(1536)
        assert result == "1.500kB"


# ========================= format_bytes =========================


class TestFormatBytes:
    def test_bytes(self):
        size, unit = tide_util.format_bytes(512)
        assert size == 512
        assert unit == "bytes"

    def test_kilobytes(self):
        size, unit = tide_util.format_bytes(2048)
        assert size == 2.0
        assert unit == "kilobytes"

    def test_megabytes(self):
        # format_bytes uses > 1024 threshold, so 1024^2 = 1024 kilobytes
        size, unit = tide_util.format_bytes(1024 * 1024)
        assert size == 1024.0
        assert unit == "kilobytes"

    def test_gigabytes(self):
        # format_bytes uses > 1024 threshold, so 1024^3 = 1024 megabytes
        size, unit = tide_util.format_bytes(1024 * 1024 * 1024)
        assert size == 1024.0
        assert unit == "megabytes"


# ========================= findexecutable =========================


class TestFindexecutable:
    def test_finds_python(self):
        result = tide_util.findexecutable("python")
        assert result is not None
        assert "python" in result.lower()

    def test_returns_none_for_nonexistent(self):
        result = tide_util.findexecutable("this_command_does_not_exist_12345")
        assert result is None


# ========================= isexecutable =========================


class TestIsexecutable:
    def test_python_is_executable(self):
        assert tide_util.isexecutable("python") is True

    def test_nonexistent_is_not_executable(self):
        assert tide_util.isexecutable("this_command_does_not_exist_12345") is False


# ========================= makeadir =========================


class TestMakeadir:
    def test_creates_new_directory(self, tmp_path):
        new_dir = tmp_path / "newdir"
        result = tide_util.makeadir(str(new_dir))
        assert result is True
        assert new_dir.exists()

    def test_existing_directory_returns_true(self, tmp_path):
        result = tide_util.makeadir(str(tmp_path))
        assert result is True

    def test_nested_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        result = tide_util.makeadir(str(nested))
        assert result is True
        assert nested.exists()


# ========================= startendcheck =========================


class TestStartendcheck:
    def test_valid_range(self, capsys):
        start, end = tide_util.startendcheck(10, 2, 5)
        assert start == 2
        assert end == 5

    def test_negative_start_sets_to_zero(self, capsys):
        start, end = tide_util.startendcheck(10, -5, 5)
        assert start == 0
        assert end == 5

    def test_endpoint_minus_one_sets_to_max(self, capsys):
        start, end = tide_util.startendcheck(10, 2, -1)
        assert start == 2
        assert end == 9

    def test_endpoint_exceeds_max_clips(self, capsys):
        start, end = tide_util.startendcheck(10, 2, 100)
        assert start == 2
        assert end == 9


# ========================= valtoindex =========================


class TestValtoindex:
    def test_evenly_spaced_round(self):
        arr = np.linspace(0, 10, 11)
        idx = tide_util.valtoindex(arr, 5.4)
        assert idx == 5

    def test_evenly_spaced_floor(self):
        arr = np.linspace(0, 10, 11)
        idx = tide_util.valtoindex(arr, 5.9, discretization="floor")
        assert idx == 5

    def test_evenly_spaced_ceiling(self):
        arr = np.linspace(0, 10, 11)
        idx = tide_util.valtoindex(arr, 5.1, discretization="ceiling")
        assert idx == 6

    def test_uneven_spacing(self):
        arr = np.array([0, 1, 3, 6, 10])
        idx = tide_util.valtoindex(arr, 5.5, evenspacing=False)
        assert idx == 3  # closest to 6

    def test_clamps_to_bounds(self):
        arr = np.linspace(0, 10, 11)
        idx = tide_util.valtoindex(arr, 100)
        assert idx == 10

    def test_clamps_to_lower_bound(self):
        arr = np.linspace(0, 10, 11)
        idx = tide_util.valtoindex(arr, -100)
        assert idx == 0


# ========================= makelaglist =========================


class TestMakelaglist:
    def test_basic_range(self, capsys):
        lags = tide_util.makelaglist(0.0, 1.0, 0.25)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(lags, expected, atol=1e-10)

    def test_negative_range(self, capsys):
        lags = tide_util.makelaglist(-1.0, 1.0, 0.5)
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(lags, expected, atol=1e-10)


# ========================= version =========================


class TestVersion:
    def test_returns_tuple(self):
        result = tide_util.version()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_version_string(self):
        version, sha, date, isdirty = tide_util.version()
        assert isinstance(version, str)
        assert isinstance(sha, str)
        assert isinstance(date, str)


# ========================= timefmt =========================


class TestTimefmt:
    def test_basic_formatting(self):
        result = tide_util.timefmt(123.456)
        assert result.strip() == "123.46"

    def test_small_number(self):
        result = tide_util.timefmt(1.234)
        assert result.strip() == "1.23"

    def test_zero(self):
        result = tide_util.timefmt(0.0)
        assert result.strip() == "0.00"

    def test_width(self):
        result = tide_util.timefmt(1.0)
        assert len(result) == 10


# ========================= maketcfrom3col =========================


class TestMaketcfrom3col:
    def test_basic(self, capsys):
        timeaxis = np.linspace(0, 10, 11)
        # inputdata format: row 0 = start times, row 1 = durations, row 2 = values
        # Each column is one interval
        inputdata = np.array(
            [
                [1.0, 4.0],  # start times
                [2.0, 2.0],  # durations (end = start + duration)
                [5.0, 10.0],  # values
            ]
        )
        outputvector = np.zeros(11)
        result = tide_util.maketcfrom3col(inputdata, timeaxis, outputvector)
        # Interval 1: [1, 3) -> value 5
        # Interval 2: [4, 6) -> value 10
        assert result[1] == 5.0
        assert result[2] == 5.0
        assert result[4] == 10.0
        assert result[5] == 10.0

    def test_out_of_range_ignored(self, capsys):
        timeaxis = np.linspace(0, 5, 6)
        # start=10, duration=2, value=5 - starts after timeaxis ends
        inputdata = np.array([[10.0], [2.0], [5.0]])
        outputvector = np.zeros(6)
        result = tide_util.maketcfrom3col(inputdata, timeaxis, outputvector)
        np.testing.assert_allclose(result, 0.0)


# ========================= maketcfrom2col =========================


class TestMaketcfrom2col:
    def test_basic(self):
        timeaxis = np.arange(15)
        # Row 0: time boundaries, Row 1: values for each segment
        inputdata = np.array([[0, 5, 10], [1, 2, 3]]).astype(float)
        outputvector = np.zeros(15)
        result = tide_util.maketcfrom2col(inputdata, timeaxis, outputvector)
        # Segment 0-5: value 1, Segment 5-10: value 2, beyond 10: not filled
        assert result[0] == 1.0
        assert result[4] == 1.0
        assert result[5] == 2.0
        assert result[9] == 2.0
        # Note: the function only fills up to the boundaries specified
        assert result[10] == 0.0


# ========================= makeslicetimes =========================


class TestMakeslicetimes:
    def test_ascending(self):
        result = tide_util.makeslicetimes(4, "ascending", tr=1.0)
        assert result is not None
        assert len(result) == 4
        # ascending means times should increase
        assert result[0] < result[1] < result[2] < result[3]

    def test_descending(self):
        result = tide_util.makeslicetimes(4, "descending", tr=1.0)
        assert result is not None
        assert len(result) == 4
        # descending means times should decrease
        assert result[0] > result[1] > result[2] > result[3]

    def test_ascending_interleaved(self):
        result = tide_util.makeslicetimes(4, "ascending_interleaved", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_descending_interleaved(self):
        result = tide_util.makeslicetimes(4, "descending_interleaved", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_ascending_sparkplug(self):
        result = tide_util.makeslicetimes(4, "ascending_sparkplug", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_descending_sparkplug(self):
        result = tide_util.makeslicetimes(4, "descending_sparkplug", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_ascending_interleaved_siemens(self):
        result = tide_util.makeslicetimes(4, "ascending_interleaved_siemens", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_descending_interleaved_siemens(self):
        result = tide_util.makeslicetimes(4, "descending_interleaved_siemens", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_ascending_interleaved_philips(self):
        result = tide_util.makeslicetimes(4, "ascending_interleaved_philips", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_descending_interleaved_philips(self):
        result = tide_util.makeslicetimes(4, "descending_interleaved_philips", tr=1.0)
        assert result is not None
        assert len(result) == 4

    def test_multiband(self):
        result = tide_util.makeslicetimes(8, "ascending", tr=1.0, multibandfac=2)
        assert result is not None
        assert len(result) == 8

    def test_invalid_sliceorder_returns_none(self):
        result = tide_util.makeslicetimes(4, "invalid_type", tr=1.0)
        assert result is None

    def test_incompatible_multiband_returns_none(self):
        result = tide_util.makeslicetimes(5, "ascending", tr=1.0, multibandfac=2)
        assert result is None


# ========================= comparemap =========================


class TestComparemap:
    def test_identical_maps(self):
        map1 = np.array([1.0, 2.0, 3.0])
        map2 = np.array([1.0, 2.0, 3.0])
        mindiff, maxdiff, meandiff, mse, minreldiff, maxreldiff, meanreldiff, relmse = (
            tide_util.comparemap(map1, map2)
        )
        assert mindiff == 0.0
        assert maxdiff == 0.0
        assert meandiff == 0.0
        assert mse == 0.0

    def test_different_maps(self):
        map1 = np.array([1.0, 2.0, 3.0])
        map2 = np.array([1.1, 2.2, 2.9])
        mindiff, maxdiff, meandiff, mse, minreldiff, maxreldiff, meanreldiff, relmse = (
            tide_util.comparemap(map1, map2)
        )
        assert mindiff < 0
        assert maxdiff > 0

    def test_with_mask(self):
        map1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        map2 = np.array([[1.1, 2.0], [3.0, 4.1]])
        mask = np.array([[1, 0], [0, 1]])
        result = tide_util.comparemap(map1, map2, mask=mask)
        assert len(result) == 8


# ========================= numpy2shared =========================


class TestNumpy2shared:
    def test_basic(self):
        arr = np.array([1, 2, 3, 4, 5])
        shared_arr, shm = tide_util.numpy2shared(arr, np.int32)
        try:
            assert shared_arr.shape == arr.shape
            np.testing.assert_array_equal(shared_arr, arr)
        finally:
            shm.close()

    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        shared_arr, shm = tide_util.numpy2shared(arr, np.float64)
        try:
            assert shared_arr.shape == (2, 2)
            np.testing.assert_array_equal(shared_arr, arr)
        finally:
            shm.close()


# ========================= allocshared =========================


class TestAllocshared:
    def test_basic(self):
        arr, shm = tide_util.allocshared((3, 4), np.float64)
        try:
            assert arr.shape == (3, 4)
            assert arr.dtype == np.float64
        finally:
            shm.close()

    def test_1d(self):
        arr, shm = tide_util.allocshared((10,), np.int32)
        try:
            assert arr.shape == (10,)
            assert arr.dtype == np.int32
        finally:
            shm.close()


# ========================= allocarray =========================


class TestAllocarray:
    def test_non_shared(self):
        arr, shm = tide_util.allocarray((3, 4), np.float64, shared=False)
        assert arr.shape == (3, 4)
        assert shm is None
        np.testing.assert_allclose(arr, 0.0)

    def test_shared(self):
        arr, shm = tide_util.allocarray((3, 4), np.float64, shared=True)
        try:
            assert arr.shape == (3, 4)
            assert shm is not None
        finally:
            shm.close()


# ========================= cleanup_shm =========================


class TestCleanupShm:
    def test_with_none(self):
        # Should not raise
        tide_util.cleanup_shm(None)

    def test_with_shm(self):
        arr, shm = tide_util.allocshared((10,), np.float64)
        # Should not raise
        tide_util.cleanup_shm(shm)
        shm.close()


# ========================= logmem =========================


class TestLogmem:
    def test_header(self):
        # Should not raise on any platform
        tide_util.logmem()

    def test_with_message(self):
        # Should not raise on any platform
        tide_util.logmem("Test message")


# ========================= progressbar =========================


class TestProgressbar:
    def test_basic(self, capsys):
        tide_util.progressbar(50, 100, "Test", 20)
        captured = capsys.readouterr()
        assert "50.00%" in captured.out

    def test_complete(self, capsys):
        tide_util.progressbar(100, 100, "Test", 20)
        captured = capsys.readouterr()
        assert "100.00%" in captured.out


# ========================= findreferencedir =========================


class TestFindreferencedir:
    def test_returns_string(self):
        result = tide_util.findreferencedir()
        assert isinstance(result, str)
        assert "rapidtide" in result
        assert "reference" in result


# ========================= setmemlimit =========================


@pytest.mark.skipif(platform.system() == "Windows", reason="Not supported on Windows")
class TestSetmemlimit:
    def test_callable(self):
        # Just verify the function exists and is callable
        # Actually setting memory limits can cause issues in test environments
        assert callable(tide_util.setmemlimit)


# ========================= findavailablemem =========================


class TestFindavailablemem:
    def test_cgroup_branch(self, monkeypatch):
        monkeypatch.setattr(tide_util.os.path, "isfile", lambda p: True)

        class DummyFile:
            def __enter__(self):
                from io import StringIO

                self._f = StringIO("1048576")
                return self._f

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(tide_util, "open", lambda *args, **kwargs: DummyFile(), raising=False)
        mem, swap = tide_util.findavailablemem()
        assert mem == 1048576
        assert swap == 1048576

    def test_free_command_branch(self, monkeypatch):
        monkeypatch.setattr(tide_util.os.path, "isfile", lambda p: False)
        fake_stdout = b"              total        used        free      shared  buff/cache   available\nMem:          1000         500         200          20         300         400\nSwap:         1000         200         300\n"
        monkeypatch.setattr(
            tide_util.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout=fake_stdout),
        )
        mem, swap = tide_util.findavailablemem()
        assert mem == 200 * 1024 * 1024
        assert swap == 300 * 1024 * 1024


# ========================= savecommandline =========================


class TestSavecommandline:
    def test_calls_writevec(self):
        with patch("rapidtide.util.tide_io.writevec") as w:
            tide_util.savecommandline(["python", "prog.py", "--x", "1"], "testrun")
        w.assert_called_once()
        args, kwargs = w.call_args
        assert args[0] == ["python prog.py --x 1"]
        assert args[1].endswith("testrun_commandline.txt")


# ========================= startendcheck extra branches =========================


class TestStartendcheckErrors:
    def test_start_too_large_exits(self):
        with pytest.raises(SystemExit):
            tide_util.startendcheck(10, 20, 30)

    def test_start_greater_equal_end_exits(self):
        with pytest.raises(SystemExit):
            tide_util.startendcheck(10, 5, 5)


# ========================= valtoindex extra branches =========================


class TestValtoindexExtra:
    def test_continuous_position(self):
        arr = np.linspace(0, 10, 11)
        pos = tide_util.valtoindex(arr, 5.5, discrete=False)
        assert np.fabs(pos - 5.5) < 1e-12

    def test_illegal_discretization_raises(self):
        arr = np.linspace(0, 10, 11)
        with pytest.raises(TypeError):
            tide_util.valtoindex(arr, 5.5, discretization="illegal_mode")


# ========================= proctiminglogfile / proctiminginfo =========================


class TestTimingHelpers:
    def test_proctiminglogfile(self, tmp_path):
        logfile = tmp_path / "timing.tsv"
        logfile.write_text(
            "\n".join(
                [
                    "20260101T000000.000000\tStart\t\t",
                    "20260101T000001.000000\tStep1\t100\tvoxels",
                    "20260101T000003.000000\tStep2\t\t",
                ]
            )
            + "\n"
        )
        lines, total = tide_util.proctiminglogfile(str(logfile), timewidth=8)
        assert len(lines) >= 3
        assert "Description" in lines[0]
        assert np.fabs(total - 3.0) < 1e-6

    def test_proctiminginfo_with_output(self):
        timings = [
            ("start", 1000.0, None, None),
            ("step", 1002.0, 200.0, "voxels"),
            ("end", 1003.0, None, None),
        ]
        with patch("rapidtide.util.tide_io.writevec") as w:
            tide_util.proctiminginfo(timings, outputfile="timing_out.txt", extraheader="HEADER")
        w.assert_called_once()


# ========================= comparemap extra branches =========================


class TestComparemapNumerics:
    def test_zero_denominators_do_not_warn(self, recwarn):
        """The relative difference skips voxels whose reference value is zero.  It used
        to compute the division for every voxel and then throw the bad ones away, which
        is correct but fills the log with divide-by-zero warnings from a routine that
        runs over every map of every comparison."""
        themap1 = np.array([0.0, 1.0, 2.0, 0.0, 4.0])
        themap2 = themap1 + 1.0
        theresult = tide_util.comparemap(themap1, themap2)

        assert not [
            thewarning for thewarning in recwarn if issubclass(thewarning.category, RuntimeWarning)
        ], "comparemap warned about the zeros it was discarding"

        # and the answer is unchanged: zero-reference voxels contribute 0 relative diff
        mindiff, maxdiff, meandiff, mse, minrel, maxrel, meanrel, relmse = theresult
        assert np.isclose(meandiff, 1.0)
        assert np.all(np.isfinite(theresult))
        # 1/1, 1/2 and 1/4 for the nonzero references, 0 for the two zeros
        np.testing.assert_allclose(meanrel, (1.0 + 0.5 + 0.25) / 5.0)


class TestComparemapBranches:
    def test_shape_mismatch_exits(self):
        with pytest.raises(SystemExit):
            tide_util.comparemap(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_mask_one_less_dimension(self):
        map1 = np.arange(12, dtype=float).reshape(2, 2, 3)
        map2 = map1 + 1.0
        mask = np.array([[1, 0], [0, 1]], dtype=int)
        result = tide_util.comparemap(map1, map2, mask=mask)
        assert len(result) == 8
        assert result[1] > 0.0

    def test_incompatible_mask_exits(self):
        map1 = np.zeros((2, 2, 2))
        map2 = np.zeros((2, 2, 2))
        badmask = np.zeros((2,), dtype=int)
        with pytest.raises(SystemExit):
            tide_util.comparemap(map1, map2, mask=badmask)


# ========================= compare run helpers =========================


class TestCompareRunHelpers:
    def test_comparerapidtideruns_smoke(self):
        mask_hdr = {"dim": [0, 2, 2, 1, 1]}
        mask_data = np.ones((2, 2, 1), dtype=float)

        with (
            patch(
                "rapidtide.util.tide_io.readfromnifti",
                return_value=(None, mask_data, mask_hdr, None, None),
            ),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=False),
            patch("rapidtide.util.tide_io.readvectorsfromtextfile", side_effect=FileNotFoundError),
        ):
            out = tide_util.comparerapidtideruns("run1", "run2", debug=False)
        assert isinstance(out, dict)

    def test_comparehappyruns_smoke(self):
        mask_hdr = {"dim": [0, 2, 2, 1, 1]}
        mask_data = np.ones((2, 2, 1), dtype=float)
        with (
            patch(
                "rapidtide.util.tide_io.readfromnifti",
                return_value=(None, mask_data, mask_hdr, None, None),
            ),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=False),
        ):
            out = tide_util.comparehappyruns("happy1", "happy2", debug=False)
        assert isinstance(out, dict)


# ========================= version, container branch =========================


class TestVersionInContainer:
    """The container branch reads everything from environment variables.

    It used to be impossible to reach any of this from a test, because a container
    that set RUNNING_IN_CONTAINER without also setting GITDIRECTVERSION fell through
    every branch and raised UnboundLocalError on the return.
    """

    def test_no_git_variables_at_all(self):
        with patch.dict(os.environ, {"RUNNING_IN_CONTAINER": "1"}, clear=True):
            theversion, thesha, thedate, isdirty = tide_util.version()
        assert (theversion, thesha, thedate, isdirty) == (
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
            "UNKNOWN",
        )

    def test_gitversion_without_a_direct_version(self):
        theenv = {"RUNNING_IN_CONTAINER": "1", "GITVERSION": "2.9.0"}
        with patch.dict(os.environ, theenv, clear=True):
            theversion, thesha, thedate, isdirty = tide_util.version()
        assert theversion == "2.9.0"
        # nothing said whether the tree was dirty, so that must not be guessed
        assert isdirty == "UNKNOWN"

    def test_a_clean_tag_is_not_dirty(self):
        theenv = {
            "RUNNING_IN_CONTAINER": "1",
            "GITVERSION": "2.9.0",
            "GITDIRECTVERSION": "v2.9.0",
            "GITSHA": "abc1234",
            "GITDATE": "2026-01-01",
        }
        with patch.dict(os.environ, theenv, clear=True):
            theversion, thesha, thedate, isdirty = tide_util.version()
        assert theversion == "v2.9.0"
        assert thesha == "abc1234"
        assert thedate == "2026-01-01"
        assert isdirty is False

    def test_commits_past_the_tag_are_dirty(self):
        for thedirect, theexpected in (
            ("v2.9.0-3", "v2.9.0.3"),
            ("v2.9.0-3-gabc1234", "v2.9.0.3+gabc1234"),
        ):
            theenv = {
                "RUNNING_IN_CONTAINER": "1",
                "GITVERSION": "2.9.0",
                "GITDIRECTVERSION": thedirect,
            }
            with patch.dict(os.environ, theenv, clear=True):
                theversion, dummy, dummy2, isdirty = tide_util.version()
            assert theversion == theexpected, f"{thedirect} became {theversion}"
            assert isdirty is True

    def test_a_describe_dirty_suffix_is_handled(self):
        """`git describe --dirty` appends a fourth field.  That used to fall through
        to a bare pass, leaving isdirty unset and crashing on the return."""
        theenv = {
            "RUNNING_IN_CONTAINER": "1",
            "GITVERSION": "2.9.0",
            "GITDIRECTVERSION": "v2.9.0-3-gabc1234-dirty",
        }
        with patch.dict(os.environ, theenv, clear=True):
            theversion, dummy, dummy2, isdirty = tide_util.version()
        assert theversion == "v2.9.0.3+gabc1234"
        assert isdirty is True

    def test_the_direct_version_wins_over_gitversion(self):
        theenv = {
            "RUNNING_IN_CONTAINER": "1",
            "GITVERSION": "9.9.9",
            "GITDIRECTVERSION": "v2.9.0",
        }
        with patch.dict(os.environ, theenv, clear=True):
            theversion = tide_util.version()[0]
        assert theversion == "v2.9.0", "GITDIRECTVERSION did not take precedence"

    def test_versioneer_failure_falls_back_to_unknown(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("rapidtide.util.tide_versioneer.get_versions", side_effect=RuntimeError):
                assert tide_util.version() == ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")

    def test_missing_versioneer_fields_become_unknown(self):
        thefields = {"version": None, "full-revisionid": None, "date": None, "dirty": None}
        with patch.dict(os.environ, {}, clear=True):
            with patch("rapidtide.util.tide_versioneer.get_versions", return_value=thefields):
                assert tide_util.version() == ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")

    def test_a_zero_distance_local_version_is_trimmed(self):
        """versioneer appends +0.gSHA.dirty when the tag is exactly at HEAD; the
        leading zero means no commits past the tag, so the suffix is dropped."""
        thefields = {
            "version": "2.9.0+0.gabc1234.dirty",
            "full-revisionid": "abc1234",
            "date": "2026-01-01",
            "dirty": False,
        }
        with patch.dict(os.environ, {}, clear=True):
            with patch("rapidtide.util.tide_versioneer.get_versions", return_value=thefields):
                theversion = tide_util.version()[0]
        assert theversion == "2.9.0"

        # a nonzero distance keeps the whole thing, since it is a real dev version
        thefields["version"] = "2.9.0+3.gabc1234.dirty"
        with patch.dict(os.environ, {}, clear=True):
            with patch("rapidtide.util.tide_versioneer.get_versions", return_value=thefields):
                assert tide_util.version()[0] == "2.9.0+3.gabc1234.dirty"


# ========================= pyfftw configuration =========================


@pytest.fixture
def restorepyfftwconfig():
    """Put pyfftw's global config back after a test has changed it.

    pyfftw.config is process wide, and rapidtide.stats installs pyfftw as the backend
    for every scipy.fft call at import time.  A test that leaves NUM_THREADS somewhere
    other than it found it silently changes how every later test in the same pytest
    session transforms - which is exactly the kind of order dependent behaviour that
    only ever shows up in CI, where the whole suite runs in one process.
    """
    import pyfftw

    thesaved = (pyfftw.config.NUM_THREADS, pyfftw.config.PLANNER_EFFORT)
    yield
    pyfftw.config.NUM_THREADS, pyfftw.config.PLANNER_EFFORT = thesaved


class TestConfigurePyfftw:
    def test_returns_a_wisdom_path_under_home(self, tmp_path, restorepyfftwconfig):
        theenv = {"HOME": str(tmp_path)}
        with patch.dict(os.environ, theenv, clear=True):
            thewisdomfile = tide_util.configurepyfftw(threads=2)
        assert str(tmp_path) in thewisdomfile
        assert thewisdomfile.endswith(".txt")
        assert ".config" in thewisdomfile

    def test_thread_count_is_applied(self, tmp_path, restorepyfftwconfig):
        import pyfftw

        with patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            tide_util.configurepyfftw(threads=3)
        assert pyfftw.config.NUM_THREADS == 3

    def test_wisdom_round_trips_through_the_file(self, tmp_path, restorepyfftwconfig):
        """savewisdom writes what configurepyfftw reads back, so the two have to agree
        on the format.  They are only ever exercised as a pair."""
        with patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            thewisdomfile = tide_util.configurepyfftw(threads=1)
            assert not os.path.isfile(thewisdomfile), "no wisdom should exist yet"

            tide_util.savewisdom(thewisdomfile, debug=True)
            assert os.path.isfile(thewisdomfile), "savewisdom wrote nothing"

            # second call finds the file and loads it rather than starting fresh
            thesamefile = tide_util.configurepyfftw(threads=1, debug=True)
        assert thesamefile == thewisdomfile

    def test_works_without_HOME_set(self, restorepyfftwconfig):
        """A stripped environment has no HOME.  Building the wisdom path out of None
        used to raise TypeError before anything could report a useful problem."""
        theenv = {k: v for k, v in os.environ.items() if k != "HOME"}
        with patch.dict(os.environ, theenv, clear=True):
            thewisdomfile = tide_util.configurepyfftw(threads=1)
        assert isinstance(thewisdomfile, str) and thewisdomfile.endswith(".txt")

    def test_thread_count_from_the_environment_is_an_integer(self, tmp_path, restorepyfftwconfig):
        """PYFFTW_NUM_THREADS arrives as a string; NUM_THREADS is a count."""
        import pyfftw

        theenv = {"HOME": str(tmp_path), "PYFFTW_NUM_THREADS": "4"}
        with patch.dict(os.environ, theenv, clear=True):
            tide_util.configurepyfftw(threads=0)
        assert pyfftw.config.NUM_THREADS == 4
        assert isinstance(pyfftw.config.NUM_THREADS, int)

    def test_unparseable_wisdom_is_reported_not_executed(
        self, tmp_path, capsys, restorepyfftwconfig
    ):
        """The wisdom file is read back from disk, so it must be parsed as data rather
        than executed.  Anything that is not a Python literal is a corrupt file, and
        has to be reported rather than run or raised."""
        thewisdomdir = tmp_path / ".config"
        thewisdomdir.mkdir()
        with patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            thewisdomfile = tide_util.configurepyfftw(threads=1)
            with open(thewisdomfile, "w") as thefile:
                thefile.write("__import__('os').environ['PWNED'] = '1'")
            theresult = tide_util.configurepyfftw(threads=1)

        assert theresult == thewisdomfile
        assert "PWNED" not in os.environ, "the wisdom file was executed"
        assert "could not parse" in capsys.readouterr().out

    def test_savewisdom_ignores_a_none_path(self):
        # nothing to write to, and it must not raise trying
        assert tide_util.savewisdom(None) is None


# ========================= comparing whole runs =========================


def _make_comparerun_mocks(mapshape=(2, 2, 1), offset=0.5):
    """Build readfromnifti/isfile mocks where every map exists and differs by offset.

    Parameters
    ----------
    mapshape : tuple of int
        Shape of the mask and of each map.
    offset : float
        Constant added to the second run's maps, so the differences are known.

    Returns
    -------
    thereader : callable
        Stands in for tide_io.readfromnifti.
    """
    themaskdata = np.ones(mapshape, dtype=float)
    thehdr = {"dim": [3, mapshape[0], mapshape[1], mapshape[2], 1]}

    def thereader(thename, **thekwargs):
        thedata = np.ones(mapshape, dtype=float)
        # run2 filenames carry the second root, so key off that
        if "run2" in thename or "happy2" in thename:
            thedata = thedata + offset
        if "mask" in thename:
            return None, themaskdata, thehdr, None, None
        return None, thedata, thehdr, None, None

    return thereader


class TestCompareRapidtideRunsFull:
    def test_every_map_is_compared_when_present(self):
        """With all the maps present the comparison loop runs to completion and the
        differences it reports are the ones that were planted."""
        thereader = _make_comparerun_mocks(offset=0.5)
        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=True),
            patch(
                "rapidtide.util.tide_io.readvectorsfromtextfile",
                side_effect=FileNotFoundError,
            ),
        ):
            theresults = tide_util.comparerapidtideruns("run1", "run2", debug=False)

        assert "maxtime" in theresults, "the map loop did not produce any results"
        assert "maxcorr" in theresults
        for thekey in ("mindiff", "maxdiff", "meandiff", "mse", "relmse"):
            assert thekey in theresults["maxtime"]
        # every voxel differs by exactly the planted offset
        assert np.isclose(theresults["maxtime"]["meandiff"], 0.5)
        assert np.isclose(theresults["maxtime"]["mindiff"], 0.5)
        assert np.isclose(theresults["maxtime"]["mse"], 0.25)
        # the maps are all ones, so a 0.5 absolute difference is also a 0.5 relative one
        assert np.isclose(theresults["maxtime"]["relmeandiff"], 0.5)
        assert np.isclose(theresults["maxtime"]["relmse"], 0.25)

    def test_timecourses_are_compared_when_readable(self):
        thereader = _make_comparerun_mocks()
        thetc1 = np.arange(10, dtype=float)
        thetc2 = thetc1 + 1.0

        thecalls = {"n": 0}

        def thetcreader(thespec, **thekwargs):
            thecalls["n"] += 1
            thewhich = thetc2 if "run2" in thespec else thetc1
            return None, None, None, thewhich, None, None

        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=True),
            patch("rapidtide.util.tide_io.readvectorsfromtextfile", side_effect=thetcreader),
        ):
            theresults = tide_util.comparerapidtideruns("run1", "run2", debug=False)

        # the json:column spec is rewritten into a flat key
        thetckeys = [thekey for thekey in theresults if "movingregressor" in thekey]
        assert thetckeys, f"no timecourse results, keys were {sorted(theresults)}"
        assert all(":" not in thekey for thekey in thetckeys), "the column spec was not rewritten"
        assert np.isclose(theresults[thetckeys[0]]["meandiff"], 1.0)

    def test_mismatched_timecourse_lengths_are_skipped(self):
        thereader = _make_comparerun_mocks()

        def thetcreader(thespec, **thekwargs):
            thelength = 10 if "run1" in thespec else 11
            return None, None, None, np.zeros(thelength), None, None

        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=True),
            patch("rapidtide.util.tide_io.readvectorsfromtextfile", side_effect=thetcreader),
        ):
            theresults = tide_util.comparerapidtideruns("run1", "run2", debug=False)

        assert not [thekey for thekey in theresults if "movingregressor" in thekey]

    def test_mismatched_map_dimensions_abort(self):
        thereader = _make_comparerun_mocks()
        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=False),
            patch("rapidtide.util.os.path.isfile", return_value=True),
        ):
            with pytest.raises(SystemExit):
                tide_util.comparerapidtideruns("run1", "run2")


def _make_bidstsv_mock(offset=0.25, length=20):
    """Stand in for tide_io.readbidstsv, which comparehappyruns uses for timecourses.

    Note comparehappyruns calls this with no error handling at all, unlike its
    rapidtide counterpart which catches FileNotFoundError - so a missing timecourse
    file propagates out of the whole comparison.

    Parameters
    ----------
    offset : float
        Constant added to the second run's timecourses.
    length : int
        Number of samples in each timecourse.

    Returns
    -------
    callable
        Suitable as a side_effect for readbidstsv.
    """

    def thereader(thefilename, thecolumn, **thekwargs):
        thedata = np.arange(length, dtype=float)
        if "happy2" in thefilename:
            thedata = thedata + offset
        # samplerate, starttime, columns, data, compressed, columnsource, extra
        return 1.0, 0.0, [thecolumn], thedata, False, None, None

    return thereader


class TestCompareHappyRunsFull:
    def test_every_map_and_timecourse_is_compared_when_present(self):
        thereader = _make_comparerun_mocks(offset=0.25)
        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=True),
            patch("rapidtide.util.tide_io.readbidstsv", side_effect=_make_bidstsv_mock(0.25)),
        ):
            theresults = tide_util.comparehappyruns("happy1", "happy2", debug=False)

        assert theresults, "the happy map loop produced nothing"
        # the maps
        assert "app_info" in theresults or "vessels_mask" in theresults
        thefirstmap = theresults.get("app_info", theresults.get("vessels_mask"))
        assert np.isclose(thefirstmap["meandiff"], 0.25)
        assert np.isclose(thefirstmap["mse"], 0.0625)
        # and the timecourses
        assert "pleth" in theresults, f"no timecourse results, keys were {sorted(theresults)}"
        assert np.isclose(theresults["pleth"]["meandiff"], 0.25)

    def test_mismatched_sample_rates_abort(self):
        """Unlike the map loop, a timecourse mismatch here is fatal rather than skipped.

        The mismatch is in the sample rate rather than the length on purpose: equal
        length arrays get past comparemap's own shape check, so reaching SystemExit
        proves this function did the rejecting rather than borrowing comparemap's.
        """
        thereader = _make_comparerun_mocks()

        def thetcreader(thefilename, thecolumn, **thekwargs):
            thesamplerate = 1.0 if "happy1" in thefilename else 2.0
            return thesamplerate, 0.0, [thecolumn], np.zeros(20), False, None, None

        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=True),
            patch("rapidtide.util.os.path.isfile", return_value=True),
            patch("rapidtide.util.tide_io.readbidstsv", side_effect=thetcreader),
        ):
            with pytest.raises(SystemExit):
                tide_util.comparehappyruns("happy1", "happy2")

    def test_mismatched_dimensions_abort(self):
        thereader = _make_comparerun_mocks()
        with (
            patch("rapidtide.util.tide_io.readfromnifti", side_effect=thereader),
            patch("rapidtide.util.tide_io.checkspacematch", return_value=False),
            patch("rapidtide.util.os.path.isfile", return_value=True),
        ):
            with pytest.raises(SystemExit):
                tide_util.comparehappyruns("happy1", "happy2")


# ========================= library thread control =========================


class TestLibThreadControl:
    def test_disable_then_restore_round_trips(self):
        """disablelibthreads pins every BLAS-ish library to one thread; enablelibthreads
        puts back whatever findlibthreads reported.  rapidtide brackets its
        multiprocessing sections with these, so a mismatch between what is saved and
        what is restored leaves a run oversubscribed for everything that follows.
        """
        thevariables = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
        # deliberately distinct, so restoring the right value to the wrong
        # variable shows up rather than cancelling out
        thestartingvalues = dict(zip(thevariables, ("3", "4", "5", "6")))
        with patch.dict(os.environ, thestartingvalues, clear=False):
            thesaved = tide_util.findlibthreads()
            assert list(thesaved) == ["3", "4", "5", "6"]

            tide_util.disablelibthreads()
            for thename in thevariables:
                assert os.environ[thename] == "1", f"{thename} was not pinned"

            tide_util.enablelibthreads(*thesaved, debug=True)
            for thename, thevalue in thestartingvalues.items():
                assert (
                    os.environ[thename] == thevalue
                ), f"{thename} came back as {os.environ[thename]}, expected {thevalue}"

    def test_findlibthreads_defaults_when_unset(self):
        theenv = {
            thename: thevalue
            for thename, thevalue in os.environ.items()
            if thename
            not in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        with patch.dict(os.environ, theenv, clear=True):
            assert list(tide_util.findlibthreads()) == [1, 1, 1, 1]


class TestMklControl:
    def test_disablemkl_only_acts_with_multiple_procs(self):
        """Pinning MKL to one thread is only wanted when rapidtide is about to fan out
        across processes; with a single process MKL should keep its own threading."""
        themkl = SimpleNamespace(set_num_threads=lambda n: thecalls.append(n))
        thecalls = []
        with (
            patch("rapidtide.util.mklexists", True),
            patch("rapidtide.util.mkl", themkl, create=True),
        ):
            tide_util.disablemkl(1, debug=True)
            assert thecalls == [], "MKL was pinned for a single process run"
            tide_util.disablemkl(4, debug=True)
            assert thecalls == [1], "MKL was not pinned for a multiprocess run"

    def test_enablemkl_restores_the_requested_count(self):
        thecalls = []
        themkl = SimpleNamespace(set_num_threads=lambda n: thecalls.append(n))
        with (
            patch("rapidtide.util.mklexists", True),
            patch("rapidtide.util.mkl", themkl, create=True),
        ):
            tide_util.enablemkl(6, debug=True)
        assert thecalls == [6]

    def test_both_are_noops_without_mkl(self):
        with patch("rapidtide.util.mklexists", False):
            # must not raise even though rapidtide.util.mkl may not exist at all
            tide_util.disablemkl(8, debug=True)
            tide_util.enablemkl(8, debug=True)


# ========================= Run the tests =========================


def _runtests(debug=False, local=False):
    """Command line entry point, kept consistent with the other test_*.py files.

    Deliberately NOT named test_util.  Under that name pytest collected it as a test
    and it re-ran this whole module inside the run that was already running it, which
    doubled the wall clock of every invocation and made the module return an ExitCode
    instead of None (PytestReturnNotNoneWarning).  The classes above are what pytest
    should collect; this is only for `python test_util.py`.

    Parameters
    ----------
    debug : bool, optional
        Pass -s so print output is not captured.  Default is False.
    local : bool, optional
        Accepted for consistency with the other entry points; unused here.

    Returns
    -------
    int
        The pytest exit code.
    """
    # Use importlib mode to avoid basename collisions with installed package tests.
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    args = [thisfile, "-v", "--import-mode=importlib"]
    if debug:
        args.append("-s")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(_runtests(debug=True, local=True))
