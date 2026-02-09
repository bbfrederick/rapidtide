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
import platform

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
        inputdata = np.array([
            [1.0, 4.0],   # start times
            [2.0, 2.0],   # durations (end = start + duration)
            [5.0, 10.0],  # values
        ])
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


# ========================= Run the tests =========================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])