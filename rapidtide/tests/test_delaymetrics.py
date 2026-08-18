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
import numpy as np

from rapidtide.workflows.fitSimFuncMap import recorddelaymetrics


def _makecase(shape=(16, 16, 16)):
    """A mask with a planted outlier, a planted long-delay blob, and a planted jump."""
    themask = np.zeros(shape, dtype=bool)
    themask[2:14, 2:14, 2:14] = True
    thetau = np.where(themask, 1.0, 0.0)

    # one isolated voxel far from its neighbourhood: an outlier, and a jump source
    thetau[8, 8, 8] = 30.0

    # a coherent long-delay blob, 4x4x4 = 64 voxels, above the coherence floor of 27
    thetau[3:7, 3:7, 3:7] = 20.0

    thevalidvoxels = np.nonzero(themask.reshape(-1))[0]
    thelagtimes = thetau.reshape(-1)[thevalidvoxels]
    thefitmask = np.ones(len(thevalidvoxels))
    return thetau, themask, thevalidvoxels, thelagtimes, thefitmask


def test_recorddelaymetrics_records_expected_keys(debug=False):
    thetau, themask, thevalidvoxels, thelagtimes, thefitmask = _makecase()
    theoptiondict = {"lagmin": -10.0, "lagmax": 40.0}
    recorddelaymetrics(
        theoptiondict,
        2,
        thelagtimes,
        thefitmask,
        thevalidvoxels,
        thetau.shape,
        int(np.prod(thetau.shape)),
        5.0,
        corrscale=np.linspace(-10.0, 40.0, 101),
    )
    if debug:
        print({k: v for k, v in theoptiondict.items() if "_pass2" in k})

    for thekey in (
        "delayoutliers",
        "delayoutlierfrac",
        "delayjumps",
        "delayjumpfrac",
        "delayp5",
        "delayp50",
        "delayp99",
        "delayiqr",
        "delayrailedfrac",
        "delaylongcut",
        "delaylongfrac",
        "delaylongcoherent",
        "delaylonglargest",
        "delaylongdev",
    ):
        assert thekey + "_pass2" in theoptiondict, f"{thekey} was not recorded"

    # the planted isolated voxel must be caught as an outlier
    assert theoptiondict["delayoutliers_pass2"] >= 1
    # the planted 4x4x4 blob is above the 27 voxel coherence floor, so the long
    # delay voxels must come out coherent rather than scattered
    assert theoptiondict["delaylonglargest_pass2"] >= 27
    assert theoptiondict["delaylongcoherent_pass2"] > 0.9
    # nothing is anywhere near the search bounds
    assert theoptiondict["delayrailedfrac_pass2"] == 0.0
    # resolution did not run, so its metrics must be absent rather than zero
    assert not [k for k in theoptiondict if k.startswith("resolve")]


def test_recorddelaymetrics_selectivity_direction(debug=False):
    """With resolution changing every scattered long voxel and no coherent one,
    selectivity must come out high; the reverse must come out low."""
    thetau, themask, thevalidvoxels, dummy, thefitmask = _makecase()

    # scatter some isolated long voxels far from the coherent blob.  Note that the
    # lone 30 s voxel planted by _makecase is ALSO an isolated long voxel, so it
    # belongs to the scattered class too and has to move with the rest for the
    # scattered fraction to come out at 1.0.
    thescattered = [(10, 3, 3), (10, 6, 3), (10, 9, 3), (10, 12, 3), (8, 8, 8)]
    for thepoint in thescattered:
        thetau[thepoint] = 20.0
    thebefore = thetau.reshape(-1)[thevalidvoxels].copy()

    # move only the scattered ones
    theafter3d = thetau.copy()
    for thepoint in thescattered:
        theafter3d[thepoint] = 1.0
    theafter = theafter3d.reshape(-1)[thevalidvoxels]

    theoptiondict = {"lagmin": -10.0, "lagmax": 40.0}
    recorddelaymetrics(
        theoptiondict,
        1,
        theafter,
        thefitmask,
        thevalidvoxels,
        thetau.shape,
        int(np.prod(thetau.shape)),
        5.0,
        preresolvelags=thebefore,
        corrscale=np.linspace(-10.0, 40.0, 101),
    )
    if debug:
        print({k: v for k, v in theoptiondict.items() if k.startswith("resolve")})

    assert theoptiondict["resolvechgscatteredfrac_pass1"] == 1.0
    assert theoptiondict["resolvechgcoherentfrac_pass1"] == 0.0
    # coherent fraction is zero, so the ratio is undefined - the key must still be
    # present, holding None, so a tabulation can distinguish "undefined" from "absent"
    assert "resolveselectivity_pass1" in theoptiondict
    assert theoptiondict["resolveselectivity_pass1"] is None

    # now the reverse: move only the coherent blob
    theafter3d = thetau.copy()
    theafter3d[3:7, 3:7, 3:7] = 1.0
    theoptiondict2 = {"lagmin": -10.0, "lagmax": 40.0}
    recorddelaymetrics(
        theoptiondict2,
        1,
        theafter3d.reshape(-1)[thevalidvoxels],
        thefitmask,
        thevalidvoxels,
        thetau.shape,
        int(np.prod(thetau.shape)),
        5.0,
        preresolvelags=thebefore,
        corrscale=np.linspace(-10.0, 40.0, 101),
    )
    assert theoptiondict2["resolvechgcoherentfrac_pass1"] == 1.0
    assert theoptiondict2["resolvechgscatteredfrac_pass1"] == 0.0
    assert theoptiondict2["resolveselectivity_pass1"] == 0.0


def test_recorddelaymetrics_survives_bad_input(debug=False):
    """Instrumentation must never take down a run.  A shape mismatch is a bug, but it
    has to be a logged bug, not a crash seven minutes into processing."""
    theoptiondict = {}
    recorddelaymetrics(
        theoptiondict,
        1,
        np.zeros(10),
        np.ones(10),
        np.arange(10),
        (4, 4, 4),  # inconsistent with numspatiallocs below
        1000,
        5.0,
    )
    assert not theoptiondict, "a failed metrics pass must not leave partial values behind"


if __name__ == "__main__":
    test_recorddelaymetrics_records_expected_keys(debug=True)
    test_recorddelaymetrics_selectivity_direction(debug=True)
    test_recorddelaymetrics_survives_bad_input(debug=True)
