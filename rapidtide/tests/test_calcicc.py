#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import numpy as np

from rapidtide.stats import fast_ICC_rep_anova


def test_ICC_rep_anova():
    # see table 2 in P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass
    # Correlations: Uses in Assessing Rater Reliability". Psychological
    # Bulletin 86 (2): 420-428
    Y = np.array(
        [
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7],
        ]
    )

    icc, r_var, e_var, _, dfc, dfe = fast_ICC_rep_anova(Y)
    # see table 4
    assert round(icc, 2) == 0.71
    assert dfc == 3
    assert dfe == 15
    assert np.isclose(r_var / (r_var + e_var), icc)


if __name__ == "__main__":
    test_ICC_rep_anova()
