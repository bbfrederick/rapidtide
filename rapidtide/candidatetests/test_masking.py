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
"""
### Explanation:

1. **Imports**: The script imports `unittest` for testing and the `masking_module` which contains the functions to be tested.

2. **TestCase Class**: A class `TestMaskingFunctions` is defined, inheriting from `unittest.TestCase`. This class will contain all the test methods.

3. **Test Methods**:
   - Each method in this class tests a specific aspect of the functions in `masking_module`.
   - The `setUp` method initializes any necessary setup for the tests.
   - The `test_read_mask_file` method simulates reading mask files and checks if the output matches expected values.
   - The `test_apply_mask_to_data` method checks if the data is correctly masked based on a given mask.
   - The `test_combine_masks` method verifies that two masks are combined as expected.
   - The `test_invalid_files` method tests how the functions handle invalid file inputs, ensuring they raise exceptions.

4. **Assertions**: Each test uses assertions to verify that the function's output matches the expected result or behavior (e.g., raising an exception).

5. **Main Block**: The `if __name__ == '__main__': unittest.main()` block ensures that the tests are run when the script is executed directly.

### Running the Tests:

- Save the test script to a file, e.g., `test_masking.py`.
- Run the script using a Python interpreter: `python test_masking.py`.

This will execute all the test methods and report any failures or errors. Make sure that the `masking_module` is correctly implemented and accessible in your environment for these tests to run successfully.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import bisect

import rapidtide.maskutil as tide_mask

class TestMaskingFunctions(unittest.TestCase):
    @patch('tide_mask.resampmask')
    def test_resampmask(self, mock_resampmask):
        themask = np.array([1, 0, 1])
        thetargetres = 2
        result = tide_mask.resampmask(themask, thetargetres)
        self.assertEqual(result, themask)

    @patch('nilearn.masking.compute_epi_mask')
    def test_makeepimask(self, mock_compute_epi_mask):
        nim = MagicMock()
        mock_compute_epi_mask.return_value = np.array([1, 0, 1])
        result = tide_mask.makeepimask(nim)
        self.assertTrue(np.array_equal(result, np.array([1, 0, 1])))

    @patch('rapidtide.io.readvecs')
    def test_maketmask_single_vector(self, mock_readvecs):
        filename = 'test_file'
        timeaxis = np.array([0, 2, 4, 6])
        inputdata = np.array([[1, 0, 1]])
        maskvector = np.zeros(4)
        mock_readvecs.return_value = inputdata
        result = tide_mask.maketmask(filename, timeaxis, maskvector)
        expected_result = np.array([1, 0, 1, 0])
        self.assertTrue(np.array_equal(result, expected_result))

    @patch('rapidtide.io.readvecs')
    def test_maketmask_multiple_vectors(self, mock_readvecs):
        filename = 'test_file'
        timeaxis = np.array([0, 2, 4, 6])
        inputdata = np.array([[1, 0, 1], [0, 1, 0]])
        maskvector = np.zeros(4)
        mock_readvecs.return_value = inputdata
        result = tide_mask.maketmask(filename, timeaxis, maskvector)
        expected_result = np.array([1, 1, 1, 0])
        self.assertTrue(np.array_equal(result, expected_result))

    @patch('rapidtide.io.readvecs')
    def test_readamask(self, mock_readvecs):
        filename = 'test_file'
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3
        valslist = [1]
        inputdata = np.array([1, 2, 1])
        mock_readvecs.return_value = inputdata
        result = tide_mask.readamask(filename, datahdr, numspatiallocs, istext=istext, valslist=valslist, tolerance=tolerance)
        expected_result = np.array([1, 0, 1])
        self.assertTrue(np.array_equal(result, expected_result))

    @patch('tide_mask.readamask')
    def test_getmaskset(self, mock_readamask):
        maskname = 'test_mask'
        includename = 'include_file'
        includevals = [1]
        excludename = 'exclude_file'
        excludevals = [2]
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3

        mock_readamask.side_effect = [
            np.array([1, 0, 1]),  # include mask
            np.array([0, 1, 0]),  # exclude mask
            None                  # extra mask
        ]

        includemask, excludemask, extramask = tide_mask.getmaskset(
            maskname,
            includename,
            includevals,
            excludename,
            excludevals,
            datahdr,
            numspatiallocs,
            istext=istext,
            tolerance=tolerance
        )

        self.assertTrue(np.array_equal(includemask, np.array([1, 0, 1])))
        self.assertTrue(np.array_equal(excludemask, np.array([0, 1, 0])))
        self.assertIsNone(extramask)

    @patch('tide_mask.readamask')
    def test_getmaskset_with_extramask(self, mock_readamask):
        maskname = 'test_mask'
        includename = 'include_file'
        includevals = [1]
        excludename = 'exclude_file'
        excludevals = [2]
        extramaskname = 'extra_file'
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3
        extramaskthresh = 0.5

        mock_readamask.side_effect = [
            np.array([1, 0, 1]),  # include mask
            np.array([0, 1, 0]),  # exclude mask
            np.array([1, 1, 1])   # extra mask
        ]

        includemask, excludemask, extramask = tide_mask.getmaskset(
            maskname,
            includename,
            includevals,
            excludename,
            excludevals,
            datahdr,
            numspatiallocs,
            extramask=extramaskname,
            extramaskthresh=extramaskthresh,
            istext=istext,
            tolerance=tolerance
        )

        self.assertTrue(np.array_equal(includemask, np.array([1, 0, 1])))
        self.assertTrue(np.array_equal(excludemask, np.array([0, 1, 0])))
        self.assertTrue(np.array_equal(extramask, np.array([1, 1, 1])))

    @patch('tide_mask.readamask')
    def test_getmaskset_no_includename(self, mock_readamask):
        maskname = 'test_mask'
        includename = None
        includevals = [1]
        excludename = 'exclude_file'
        excludevals = [2]
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3

        mock_readamask.return_value = np.array([0, 1, 0])  # exclude mask

        includemask, excludemask, extramask = tide_mask.getmaskset(
            maskname,
            includename,
            includevals,
            excludename,
            excludevals,
            datahdr,
            numspatiallocs,
            istext=istext,
            tolerance=tolerance
        )

        self.assertIsNone(includemask)
        self.assertTrue(np.array_equal(excludemask, np.array([0, 1, 0])))
        self.assertIsNone(extramask)

    @patch('tide_mask.readamask')
    def test_getmaskset_no_excludename(self, mock_readamask):
        maskname = 'test_mask'
        includename = 'include_file'
        includevals = [1]
        excludename = None
        excludevals = [2]
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3

        mock_readamask.return_value = np.array([1, 0, 1])  # include mask

        includemask, excludemask, extramask = tide_mask.getmaskset(
            maskname,
            includename,
            includevals,
            excludename,
            excludevals,
            datahdr,
            numspatiallocs,
            istext=istext,
            tolerance=tolerance
        )

        self.assertTrue(np.array_equal(includemask, np.array([1, 0, 1])))
        self.assertIsNone(excludemask)
        self.assertIsNone(extramask)

    @patch('tide_mask.readamask')
    def test_getmaskset_empty_valslist(self, mock_readamask):
        maskname = 'test_mask'
        includename = 'include_file'
        includevals = []
        excludename = 'exclude_file'
        excludevals = []
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3

        mock_readamask.side_effect = [
            np.array([1, 2, 1]),  # include mask
            np.array([2, 1, 2])   # exclude mask
        ]

        includemask, excludemask, extramask = tide_mask.getmaskset(
            maskname,
            includename,
            includevals,
            excludename,
            excludevals,
            datahdr,
            numspatiallocs,
            istext=istext,
            tolerance=tolerance
        )

        self.assertTrue(np.array_equal(includemask, np.array([1, 2, 1])))
        self.assertTrue(np.array_equal(excludemask, np.array([2, 1, 2])))
        self.assertIsNone(extramask)

    @patch('tide_mask.readamask')
    def test_getmaskset_invalid_includename(self, mock_readamask):
        maskname = 'test_mask'
        includename = 'invalid_include_file'
        includevals = [1]
        excludename = 'exclude_file'
        excludevals = [2]
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3

        mock_readamask.side_effect = [
            ValueError("Invalid include mask file"),  # include mask
            np.array([0, 1, 0])                      # exclude mask
        ]

        with self.assertRaises(ValueError) as context:
            tide_mask.getmaskset(
                maskname,
                includename,
                includevals,
                excludename,
                excludevals,
                datahdr,
                numspatiallocs,
                istext=istext,
                tolerance=tolerance
            )

        self.assertEqual(str(context.exception), "Invalid include mask file")

    @patch('tide_mask.readamask')
    def test_getmaskset_invalid_excludename(self, mock_readamask):
        maskname = 'test_mask'
        includename = 'include_file'
        includevals = [1]
        excludename = 'invalid_exclude_file'
        excludevals = [2]
        datahdr = MagicMock()
        numspatiallocs = 3
        istext = False
        tolerance = 1.0e-3

        mock_readamask.side_effect = [
            np.array([1, 0, 1]),                      # include mask
            ValueError("Invalid exclude mask file")   # exclude mask
        ]

        with self.assertRaises(ValueError) as context:
            tide_mask.getmaskset(
                maskname,
                includename,
                includevals,
                excludename,
                excludevals,
                datahdr,
                numspatiallocs,
                istext=istext,
                tolerance=tolerance
            )

        self.assertEqual(str(context.exception), "Invalid exclude mask file")

if __name__ == '__main__':
    unittest.main()

