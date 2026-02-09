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
import sys
import time

import nibabel as nib
import numpy as np
import pytest

from rapidtide.workflows.niftistats import (
    _get_parser,
    makdcommandlinelist,
    niftistats_main,
    parsemeasurementlist,
)


# ---- helpers ----


def _create_4d_nifti(filepath, data, voxel_sizes=(2.0, 2.0, 2.0, 1.0)):
    """Write a 4D numpy array as a NIfTI file."""
    affine = np.eye(4)
    affine[0, 0] = voxel_sizes[0]
    affine[1, 1] = voxel_sizes[1]
    affine[2, 2] = voxel_sizes[2]
    img = nib.Nifti1Image(data.astype(np.float64), affine)
    img.header.set_zooms(voxel_sizes[: len(data.shape)])
    nib.save(img, filepath)


# ---- _get_parser tests ----


class TestGetParser:
    def test_icc_parser_returns_parser(self):
        parser = _get_parser("icc")
        assert parser is not None
        assert parser.prog == "calcicc"

    def test_ttest_parser_returns_parser(self):
        parser = _get_parser("ttest")
        assert parser is not None
        assert parser.prog == "calcttest"

    def test_invalid_calctype_raises(self):
        with pytest.raises(Exception):
            _get_parser("invalid")

    def test_icc_parser_has_measurementlist(self):
        parser = _get_parser("icc")
        # measurementlist is a required positional argument for icc
        action_names = [a.dest for a in parser._actions]
        assert "measurementlist" in action_names

    def test_ttest_parser_no_measurementlist(self):
        parser = _get_parser("ttest")
        action_names = [a.dest for a in parser._actions]
        assert "measurementlist" not in action_names

    def test_ttest_parser_has_paired(self):
        parser = _get_parser("ttest")
        action_names = [a.dest for a in parser._actions]
        assert "paired" in action_names

    def test_ttest_parser_has_alternative(self):
        parser = _get_parser("ttest")
        action_names = [a.dest for a in parser._actions]
        assert "alternative" in action_names

    def test_icc_parser_no_paired(self):
        parser = _get_parser("icc")
        action_names = [a.dest for a in parser._actions]
        assert "paired" not in action_names

    def test_icc_parser_has_nocache(self):
        parser = _get_parser("icc")
        action_names = [a.dest for a in parser._actions]
        assert "nocache" in action_names

    def test_ttest_parser_no_nocache(self):
        parser = _get_parser("ttest")
        action_names = [a.dest for a in parser._actions]
        assert "nocache" not in action_names

    def test_common_options_present(self):
        for calctype in ("icc", "ttest"):
            parser = _get_parser(calctype)
            action_names = [a.dest for a in parser._actions]
            assert "datamaskname" in action_names
            assert "sigma" in action_names
            assert "demedian" in action_names
            assert "demean" in action_names
            assert "showprogressbar" in action_names
            assert "debug" in action_names
            assert "deepdebug" in action_names

    def test_icc_parser_defaults(self, tmp_path):
        parser = _get_parser("icc")
        # measurementlist requires a valid file path
        measfile = str(tmp_path / "meas.txt")
        with open(measfile, "w") as f:
            f.write("0\t1\n")
        args = parser.parse_args(["data.nii", measfile, "output"])
        assert args.datamaskname is None
        assert args.sigma == 0.0
        assert args.demedian is False
        assert args.demean is False
        assert args.nocache is False
        assert args.showprogressbar is True
        assert args.debug is False
        assert args.deepdebug is False

    def test_ttest_parser_defaults(self):
        parser = _get_parser("ttest")
        args = parser.parse_args(["data.nii", "output"])
        assert args.paired is False
        assert args.alternative == "two-sided"
        assert args.sigma == 0.0

    def test_ttest_parser_alternative_choices(self):
        parser = _get_parser("ttest")
        for alt in ("two-sided", "less", "greater"):
            args = parser.parse_args(["data.nii", "output", "--alternative", alt])
            assert args.alternative == alt

    def test_ttest_parser_invalid_alternative_exits(self):
        parser = _get_parser("ttest")
        with pytest.raises(SystemExit):
            parser.parse_args(["data.nii", "output", "--alternative", "bogus"])


# ---- parsemeasurementlist tests ----


class TestParsemeasurementlist:
    def test_single_value_entries(self):
        """Single-integer entries default to file 0."""
        measlist = np.array([["3", "5"], ["7", "9"]])
        filesel, volumesel = parsemeasurementlist(measlist, numfiles=1)
        np.testing.assert_array_equal(filesel, np.array([[0, 0], [0, 0]]))
        np.testing.assert_array_equal(volumesel, np.array([[3, 5], [7, 9]]))

    def test_two_value_entries(self):
        """Two-integer entries parse file,volume."""
        measlist = np.array([["0,5", "1,3"], ["0,2", "1,7"]])
        filesel, volumesel = parsemeasurementlist(measlist, numfiles=2)
        np.testing.assert_array_equal(filesel, np.array([[0, 1], [0, 1]]))
        np.testing.assert_array_equal(volumesel, np.array([[5, 3], [2, 7]]))

    def test_mixed_entries(self):
        """Mix of single- and two-integer entries."""
        measlist = np.array([["0,1", "3"], ["2", "1,4"]])
        filesel, volumesel = parsemeasurementlist(measlist, numfiles=2)
        expected_filesel = np.array([[0, 0], [0, 1]])
        expected_volumesel = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(filesel, expected_filesel)
        np.testing.assert_array_equal(volumesel, expected_volumesel)

    def test_debug_mode(self, capsys):
        """Debug flag should produce printed output."""
        measlist = np.array([["1", "2"]])
        parsemeasurementlist(measlist, numfiles=1, debug=True)
        captured = capsys.readouterr()
        assert "element" in captured.out

    def test_invalid_file_number_exits(self):
        """File index exceeding numfiles should cause sys.exit."""
        measlist = np.array([["5,1"]])  # file 5, but numfiles=2
        with pytest.raises(SystemExit):
            parsemeasurementlist(measlist, numfiles=2)

    def test_too_many_commas_exits(self):
        """Entry with more than one comma should cause sys.exit."""
        measlist = np.array([["1,2,3"]])
        with pytest.raises(SystemExit):
            parsemeasurementlist(measlist, numfiles=5)

    def test_output_shape_matches_input(self):
        """Output arrays should have the same shape as input."""
        measlist = np.array([["0", "1", "2"], ["3", "4", "5"]])
        filesel, volumesel = parsemeasurementlist(measlist, numfiles=1)
        assert filesel.shape == measlist.shape
        assert volumesel.shape == measlist.shape

    def test_single_element(self):
        """Single-element measurement list."""
        measlist = np.array([["42"]])
        filesel, volumesel = parsemeasurementlist(measlist, numfiles=1)
        assert filesel[0, 0] == 0
        assert volumesel[0, 0] == 42


# ---- makdcommandlinelist tests ----


class TestMakdcommandlinelist:
    def test_basic_output_without_extra(self):
        starttime = time.time() - 5.0
        endtime = time.time()
        arglist = ["python", "script.py", "--input", "data.txt"]
        result = makdcommandlinelist(arglist, starttime, endtime)
        assert len(result) == 4
        assert result[0].startswith("# Processed on")
        assert "Processing took" in result[1]
        assert result[2].startswith("# Using")
        assert result[3] == "python script.py --input data.txt"

    def test_output_with_extra(self):
        starttime = time.time() - 2.0
        endtime = time.time()
        arglist = ["calcicc", "data.nii", "output"]
        result = makdcommandlinelist(arglist, starttime, endtime, extra="some extra info")
        assert len(result) == 5
        assert "# some extra info" in result[3]
        assert result[4] == "calcicc data.nii output"

    def test_processing_time_format(self):
        starttime = 1000.0
        endtime = 1010.5
        result = makdcommandlinelist(["cmd"], starttime, endtime)
        assert "10.500" in result[1]

    def test_command_line_reconstruction(self):
        arglist = ["prog", "-a", "val1", "--flag", "val2"]
        result = makdcommandlinelist(arglist, 0.0, 1.0)
        assert result[-1] == "prog -a val1 --flag val2"

    def test_version_info_present(self):
        result = makdcommandlinelist(["cmd"], 0.0, 1.0)
        # The node line should contain platform and version info
        assert "Using" in result[2]

    def test_empty_arglist(self):
        result = makdcommandlinelist([], 0.0, 1.0)
        assert result[-1] == ""

    def test_extra_none_gives_four_lines(self):
        result = makdcommandlinelist(["cmd"], 0.0, 1.0, extra=None)
        assert len(result) == 4

    def test_extra_notnone_gives_five_lines(self):
        result = makdcommandlinelist(["cmd"], 0.0, 1.0, extra="info")
        assert len(result) == 5


# ---- niftistats_main tests ----


class TestNiftistatsMainTtest:
    """Integration tests for niftistats_main with calctype='ttest'."""

    def test_ttest_independent(self, tmp_path):
        """Run an independent t-test on two synthetic 4D NIfTI files."""
        xsize, ysize, zsize, nsubj = 4, 4, 3, 10
        np.random.seed(42)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        data2 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 1.0

        file1 = str(tmp_path / "group1.nii.gz")
        file2 = str(tmp_path / "group2.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "ttest_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        # Check output files exist
        assert os.path.isfile(outputroot + "_t.nii.gz")
        assert os.path.isfile(outputroot + "_p.nii.gz")
        assert os.path.isfile(outputroot + "_commandline.txt")

        # Load and validate t-stat map
        t_img = nib.load(outputroot + "_t.nii.gz")
        t_data = np.asarray(t_img.dataobj)
        assert t_data.shape == (xsize, ysize, zsize)
        # group2 has mean ~1 higher, so most t values should be negative (group1 < group2)
        assert np.mean(t_data) < 0

        p_img = nib.load(outputroot + "_p.nii.gz")
        p_data = np.asarray(p_img.dataobj)
        assert p_data.shape == (xsize, ysize, zsize)
        # p-values should be in [0, 1]
        assert np.all(p_data >= 0.0)
        assert np.all(p_data <= 1.0)

    def test_ttest_paired(self, tmp_path):
        """Run a paired t-test."""
        xsize, ysize, zsize, nsubj = 3, 3, 2, 15
        np.random.seed(123)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        # Add a large shift with some noise so the paired difference has nonzero variance
        data2 = data1 + 3.0 + np.random.randn(xsize, ysize, zsize, nsubj) * 0.5

        file1 = str(tmp_path / "pre.nii.gz")
        file2 = str(tmp_path / "post.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "paired_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--paired",
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        assert os.path.isfile(outputroot + "_t.nii.gz")
        assert os.path.isfile(outputroot + "_p.nii.gz")

        # With a ~+3 shift, all t-values should be strongly negative (group1 < group2)
        t_data = np.asarray(nib.load(outputroot + "_t.nii.gz").dataobj)
        assert np.all(t_data < 0)

        # p-values should be very small
        p_data = np.asarray(nib.load(outputroot + "_p.nii.gz").dataobj)
        assert np.all(p_data < 0.05)

    def test_ttest_alternative_less(self, tmp_path):
        """Test one-sided alternative='less'."""
        xsize, ysize, zsize, nsubj = 3, 3, 2, 12
        np.random.seed(99)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        data2 = data1 + 3.0  # group1 mean < group2 mean

        file1 = str(tmp_path / "a.nii.gz")
        file2 = str(tmp_path / "b.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "less_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--alternative",
            "less",
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        p_data = np.asarray(nib.load(outputroot + "_p.nii.gz").dataobj)
        # With 'less' alternative and group1 < group2, p should be small
        assert np.all(p_data < 0.05)

    def test_ttest_alternative_greater(self, tmp_path):
        """Test one-sided alternative='greater'."""
        xsize, ysize, zsize, nsubj = 3, 3, 2, 12
        np.random.seed(77)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        data2 = data1 + 3.0

        file1 = str(tmp_path / "a.nii.gz")
        file2 = str(tmp_path / "b.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "greater_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--alternative",
            "greater",
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        p_data = np.asarray(nib.load(outputroot + "_p.nii.gz").dataobj)
        # With 'greater' alternative and group1 < group2, p should be large
        assert np.all(p_data > 0.5)

    def test_ttest_with_demean(self, tmp_path):
        """Test t-test with --demean flag."""
        xsize, ysize, zsize, nsubj = 3, 3, 2, 8
        np.random.seed(55)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 100.0
        data2 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 100.0

        file1 = str(tmp_path / "g1.nii.gz")
        file2 = str(tmp_path / "g2.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "demean_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--demean",
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        assert os.path.isfile(outputroot + "_t.nii.gz")
        assert os.path.isfile(outputroot + "_p.nii.gz")

    def test_ttest_with_demedian(self, tmp_path):
        """Test t-test with --demedian flag."""
        xsize, ysize, zsize, nsubj = 3, 3, 2, 8
        np.random.seed(44)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 50.0
        data2 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 50.0

        file1 = str(tmp_path / "g1.nii.gz")
        file2 = str(tmp_path / "g2.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "demedian_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--demedian",
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        assert os.path.isfile(outputroot + "_t.nii.gz")

    def test_ttest_with_mask(self, tmp_path):
        """Test t-test with a mask that zeros out some voxels."""
        xsize, ysize, zsize, nsubj = 4, 4, 2, 8
        np.random.seed(33)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        data2 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 2.0

        mask = np.zeros((xsize, ysize, zsize), dtype=np.float64)
        mask[1:3, 1:3, :] = 1.0  # only central voxels

        file1 = str(tmp_path / "g1.nii.gz")
        file2 = str(tmp_path / "g2.nii.gz")
        maskfile = str(tmp_path / "mask.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)
        _create_4d_nifti(maskfile, mask, voxel_sizes=(2.0, 2.0, 2.0))

        outputroot = str(tmp_path / "masked_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--dmask",
            maskfile,
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        t_data = np.asarray(nib.load(outputroot + "_t.nii.gz").dataobj)
        # Voxels outside mask should be zero
        assert t_data[0, 0, 0] == 0.0
        # Voxels inside mask should be nonzero
        assert t_data[1, 1, 0] != 0.0

    def test_ttest_with_smoothing(self, tmp_path):
        """Test that smoothing runs without error."""
        xsize, ysize, zsize, nsubj = 5, 5, 3, 6
        np.random.seed(22)
        data1 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        data2 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64) + 1.0

        file1 = str(tmp_path / "g1.nii.gz")
        file2 = str(tmp_path / "g2.nii.gz")
        _create_4d_nifti(file1, data1)
        _create_4d_nifti(file2, data2)

        outputroot = str(tmp_path / "smooth_out")
        sys.argv = [
            "calcttest",
            f"{file1},{file2}",
            outputroot,
            "--smooth",
            "2.0",
            "--noprogressbar",
        ]
        niftistats_main(calctype="ttest")

        assert os.path.isfile(outputroot + "_t.nii.gz")

    def test_ttest_wrong_number_of_files_exits(self, tmp_path):
        """t-test requires exactly 2 files; 1 or 3 should fail."""
        xsize, ysize, zsize, nsubj = 3, 3, 2, 5
        data = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        file1 = str(tmp_path / "f1.nii.gz")
        _create_4d_nifti(file1, data)

        outputroot = str(tmp_path / "bad_out")
        sys.argv = [
            "calcttest",
            file1,
            outputroot,
            "--noprogressbar",
        ]
        with pytest.raises(SystemExit):
            niftistats_main(calctype="ttest")


class TestNiftistatsMainICC:
    """Integration tests for niftistats_main with calctype='icc'."""

    def test_icc_basic(self, tmp_path):
        """Run ICC on synthetic data with 2 measurements x 4 subjects."""
        xsize, ysize, zsize = 3, 3, 2
        nsubj = 4
        nmeas = 2
        np.random.seed(7)

        # Build a single 4D file: volumes are [subj0_meas0, subj0_meas1, subj1_meas0, ...]
        nvols = nsubj * nmeas
        data = np.random.randn(xsize, ysize, zsize, nvols).astype(np.float64)
        # Add subject-specific signal so ICC is non-trivial
        for s in range(nsubj):
            subject_signal = np.random.randn(xsize, ysize, zsize) * 5.0
            for m in range(nmeas):
                data[:, :, :, s * nmeas + m] += subject_signal

        datafile = str(tmp_path / "iccdata.nii.gz")
        _create_4d_nifti(datafile, data)

        # Create measurement list file: rows=subjects, cols=measurements
        # readvecs transposes, so niftistats sees shape (nummeas, numsubjs)
        measfile = str(tmp_path / "measlist.txt")
        with open(measfile, "w") as f:
            for s in range(nsubj):
                vals = [str(s * nmeas + m) for m in range(nmeas)]
                f.write("\t".join(vals) + "\n")

        outputroot = str(tmp_path / "icc_out")
        sys.argv = [
            "calcicc",
            datafile,
            measfile,
            outputroot,
            "--noprogressbar",
        ]
        niftistats_main(calctype="icc")

        assert os.path.isfile(outputroot + "_ICC.nii.gz")
        assert os.path.isfile(outputroot + "_r_var.nii.gz")
        assert os.path.isfile(outputroot + "_e_var.nii.gz")
        assert os.path.isfile(outputroot + "_session_effect_F.nii.gz")
        assert os.path.isfile(outputroot + "_commandline.txt")

        icc_data = np.asarray(nib.load(outputroot + "_ICC.nii.gz").dataobj)
        assert icc_data.shape == (xsize, ysize, zsize)
        # With strong subject-specific signal, ICC should be positive on average
        assert np.mean(icc_data) > 0

    def test_icc_with_two_files(self, tmp_path):
        """Run ICC using two separate data files referenced in the measurement list."""
        xsize, ysize, zsize = 3, 3, 2
        nsubj = 3
        np.random.seed(11)

        # File 0: measurement 0 for all subjects
        data0 = np.random.randn(xsize, ysize, zsize, nsubj).astype(np.float64)
        # File 1: measurement 1 for all subjects (correlated with file 0)
        subject_signals = np.random.randn(nsubj, xsize, ysize, zsize) * 5.0
        for s in range(nsubj):
            data0[:, :, :, s] += subject_signals[s]
        data1 = data0 + np.random.randn(xsize, ysize, zsize, nsubj) * 0.5

        file0 = str(tmp_path / "meas0.nii.gz")
        file1 = str(tmp_path / "meas1.nii.gz")
        _create_4d_nifti(file0, data0)
        _create_4d_nifti(file1, data1)

        # Measurement list: 2 rows x 3 cols
        # Row 0 (meas 0): "0,0"  "0,1"  "0,2"  (file 0, volumes 0-2)
        # Row 1 (meas 1): "1,0"  "1,1"  "1,2"  (file 1, volumes 0-2)
        measfile = str(tmp_path / "measlist.txt")
        with open(measfile, "w") as f:
            f.write("\t".join([f"0,{s}" for s in range(nsubj)]) + "\n")
            f.write("\t".join([f"1,{s}" for s in range(nsubj)]) + "\n")

        outputroot = str(tmp_path / "icc2_out")
        sys.argv = [
            "calcicc",
            f"{file0},{file1}",
            measfile,
            outputroot,
            "--noprogressbar",
        ]
        niftistats_main(calctype="icc")

        assert os.path.isfile(outputroot + "_ICC.nii.gz")
        icc_data = np.asarray(nib.load(outputroot + "_ICC.nii.gz").dataobj)
        assert icc_data.shape == (xsize, ysize, zsize)

    def test_icc_with_demean(self, tmp_path):
        """Run ICC with --demean."""
        xsize, ysize, zsize = 3, 3, 2
        nsubj, nmeas = 4, 2
        np.random.seed(13)

        nvols = nsubj * nmeas
        data = np.random.randn(xsize, ysize, zsize, nvols).astype(np.float64) + 100.0

        datafile = str(tmp_path / "iccdata.nii.gz")
        _create_4d_nifti(datafile, data)

        measfile = str(tmp_path / "measlist.txt")
        with open(measfile, "w") as f:
            for m in range(nmeas):
                vals = [str(s * nmeas + m) for s in range(nsubj)]
                f.write("\t".join(vals) + "\n")

        outputroot = str(tmp_path / "icc_demean_out")
        sys.argv = [
            "calcicc",
            datafile,
            measfile,
            outputroot,
            "--demean",
            "--noprogressbar",
        ]
        niftistats_main(calctype="icc")

        assert os.path.isfile(outputroot + "_ICC.nii.gz")

    def test_icc_with_demedian(self, tmp_path):
        """Run ICC with --demedian."""
        xsize, ysize, zsize = 3, 3, 2
        nsubj, nmeas = 4, 2
        np.random.seed(17)

        nvols = nsubj * nmeas
        data = np.random.randn(xsize, ysize, zsize, nvols).astype(np.float64) + 50.0

        datafile = str(tmp_path / "iccdata.nii.gz")
        _create_4d_nifti(datafile, data)

        measfile = str(tmp_path / "measlist.txt")
        with open(measfile, "w") as f:
            for m in range(nmeas):
                vals = [str(s * nmeas + m) for s in range(nsubj)]
                f.write("\t".join(vals) + "\n")

        outputroot = str(tmp_path / "icc_demedian_out")
        sys.argv = [
            "calcicc",
            datafile,
            measfile,
            outputroot,
            "--demedian",
            "--noprogressbar",
        ]
        niftistats_main(calctype="icc")

        assert os.path.isfile(outputroot + "_ICC.nii.gz")


if __name__ == "__main__":
    test_args = [__file__, "-v"]
    pytest.main(test_args)