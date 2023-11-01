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
import os

import numpy as np

import rapidtide.io as tide_io
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path, mse


def test_io(debug=True, displayplots=False):
    # create outputdir if it doesn't exist
    create_dir(get_test_temp_path())

    # test checkifnifti
    assert tide_io.checkifnifti("test.nii") == True
    assert tide_io.checkifnifti("test.nii.gz") == True
    assert tide_io.checkifnifti("test.txt") == False

    # test checkiftext
    assert tide_io.checkiftext("test.nii") == False
    assert tide_io.checkiftext("test.nii.gz") == False
    assert tide_io.checkiftext("test.txt") == True

    # test getniftiroot
    assert tide_io.getniftiroot("test.nii") == "test"
    assert tide_io.getniftiroot("test.nii.gz") == "test"
    assert tide_io.getniftiroot("test.txt") == "test.txt"

    # test parsefilespec
    thefile, thespec = tide_io.parsefilespec("mymask.nii.gz:1,3,4-8,APARC_SUBCORTGRAY")
    assert thefile == "mymask.nii.gz"
    assert thespec == "1,3,4-8,APARC_SUBCORTGRAY"
    assert tide_io.colspectolist(thespec) == [
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        17,
        18,
        19,
        20,
        26,
        27,
        28,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        58,
        59,
        60,
        96,
        97,
    ]

    # test fmritimeinfo
    fmritimeinfothresh = 1e-2
    tr, timepoints = tide_io.fmritimeinfo(
        os.path.join(get_examples_path(), "sub-HAPPYTEST.nii.gz")
    )
    assert np.fabs(tr - 1.16) < fmritimeinfothresh
    assert timepoints == 110
    tr, timepoints = tide_io.fmritimeinfo(
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST.nii.gz")
    )
    assert np.fabs(tr - 1.5) < fmritimeinfothresh
    assert timepoints == 260

    # test niftifile reading
    sizethresh = 1e-3
    happy_img, happy_data, happy_hdr, happydims, happysizes = tide_io.readfromnifti(
        os.path.join(get_examples_path(), "sub-HAPPYTEST.nii.gz")
    )
    fmri_img, fmri_data, fmri_hdr, fmridims, fmrisizes = tide_io.readfromnifti(
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST.nii.gz")
    )
    targetdims = [4, 65, 89, 64, 110, 1, 1, 1]
    targetsizes = [-1.00, 2.39583, 2.395830, 2.4, 1.16, 0.00, 0.00, 0.00]
    if debug:
        print("happydims:", happydims)
        print("targetdims:", targetdims)
        print("happysizes:", happysizes)
        print("targetsizes:", targetsizes)
    for i in range(len(targetdims)):
        assert targetdims[i] == happydims[i]
    assert mse(np.array(targetsizes), np.array(happysizes)) < sizethresh

    # test file writing
    datathresh = 2e-3  # relaxed threshold because sub-RAPIDTIDETEST has been converted to INT16
    tide_io.savetonifti(
        fmri_data, fmri_hdr, os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST_copy.nii.gz")
    )
    (
        fmricopy_img,
        fmricopy_data,
        fmricopy_hdr,
        fmricopydims,
        fmricopysizes,
    ) = tide_io.readfromnifti(os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST_copy.nii.gz"))
    assert tide_io.checkspacematch(fmri_hdr, fmricopy_hdr)
    assert tide_io.checktimematch(fmridims, fmridims)
    assert mse(fmri_data, fmricopy_data) < datathresh

    # test file header comparisons
    assert tide_io.checkspacematch(happy_hdr, happy_hdr)
    assert not tide_io.checkspacematch(happy_hdr, fmri_hdr)
    assert tide_io.checktimematch(happydims, happydims)
    assert not tide_io.checktimematch(happydims, fmridims)

    # test writing and reading text files
    debug = False
    DESTDIR = get_test_temp_path()
    SOURCEDIR = get_examples_path()
    EPSILON = 1e-5
    numpoints = 10
    the2darray = np.zeros((6, numpoints), dtype=float)
    the2darray[0, :] = np.linspace(0, 1.0, numpoints, endpoint=False)
    the2darray[1, :] = np.sin(the2darray[0, :] * 2.0 * np.pi)
    the2darray[2, :] = np.cos(the2darray[0, :] * 2.0 * np.pi)
    the2darray[3, :] = np.cos(2.0 * the2darray[0, :] * 2.0 * np.pi)
    the2darray[4, :] = np.cos(3.0 * the2darray[0, :] * 2.0 * np.pi)
    the2darray[5, :] = np.cos(4.0 * the2darray[0, :] * 2.0 * np.pi)

    thecols = [
        "RotX",
        "RotY",
        "RotZ",
        "X",
        "Y",
        "Z",
    ]

    inputsamplerate = 10.0
    inputstarttime = 0.5

    thetests = [
        ["text", False, ".txt"],
        ["csv", False, ".csv"],
        ["bidscontinuous", False, ".tsv"],
        ["bidscontinuous", True, ".tsv.gz"],
        ["plaintsv", False, ".tsv"],
        ["plaintsv", True, ".tsv.gz"],
    ]

    print("writing files")
    for thistest in thetests:
        thetype = thistest[0]
        compressed = thistest[1]
        if compressed:
            compname = "compressed"
        else:
            compname = "uncompressed"

        thefileroot = os.path.join(DESTDIR, f"testout_withcol_{thetype}_{compname}")
        if thetype == "text":
            thefileroot += ".par"
        if thetype == "csv":
            thefileroot += ".csv"
        print(f"\t writing: {thefileroot}")
        tide_io.writevectorstotextfile(
            the2darray,
            thefileroot,
            samplerate=inputsamplerate,
            starttime=inputstarttime,
            columns=thecols,
            compressed=compressed,
            filetype=thetype,
            lineend="",
            debug=debug,
        )
        thefileroot = os.path.join(DESTDIR, f"testout_nocol_{thetype}_{compname}")
        if thetype == "text":
            thefileroot += ".par"
        if thetype == "csv":
            thefileroot += ".csv"
        print(f"\t writing: {thefileroot}")
        tide_io.writevectorstotextfile(
            the2darray,
            thefileroot,
            samplerate=inputsamplerate,
            starttime=inputstarttime,
            columns=None,
            compressed=compressed,
            filetype=thetype,
            lineend="",
            debug=debug,
        )

    print("reading complete files")
    for thistest in thetests:
        thetype = thistest[0]
        compressed = thistest[1]
        if compressed:
            compname = "compressed"
        else:
            compname = "uncompressed"

        for colspec in ["withcol", "nocol"]:
            if debug:
                print("\n\n\n")
            thefileroot = os.path.join(DESTDIR, f"testout_{colspec}_{thetype}_{compname}")

            if thetype == "text":
                theextension = ".par"
                if colspec == "nocol":
                    motionfilename = thefileroot + theextension
            elif thetype == "csv":
                theextension = ".csv"
            else:
                if compressed:
                    theextension = ".tsv.gz"
                else:
                    theextension = ".tsv"

            print("reading:", thetype, compressed, thefileroot)
            (
                thesamplerate,
                thestarttime,
                thecolumns,
                thedata,
                compressed,
                filetype,
            ) = tide_io.readvectorsfromtextfile(
                thefileroot + theextension, onecol=False, debug=debug
            )
            """
            print(f"\t{thesamplerate=}")
            print(f"\t{thestarttime=}")
            print(f"\t{thecolumns=}")
            print(f"\t{thedata=}")
            print(f"\t{compressed=}")
            print(f"\t{filetype=}")
            """

            if thetype == "text":
                assert thesamplerate is None
                assert thestarttime is None
                assert thecolumns is None
                assert filetype == "text"
            elif thetype == "csv":
                assert thesamplerate is None
                assert thestarttime is None
                if thefileroot.find("nocol") > 0:
                    assert len(thecolumns) == len(thecols)
                    for i in range(6):
                        assert thecolumns[i] == tide_io.makecolname(i, 0)
                else:
                    assert len(thecolumns) == len(thecols)
                    for i in range(6):
                        assert thecolumns[i] == thecols[i]
                assert filetype == "csv"
            elif thetype == "bidscontinuous":
                assert thesamplerate == inputsamplerate
                assert thestarttime == inputstarttime
                if thefileroot.find("nocol") > 0:
                    assert len(thecolumns) == len(thecols)
                    for i in range(6):
                        assert thecolumns[i] == tide_io.makecolname(i, 0)
                else:
                    assert len(thecolumns) == len(thecols)
                    for i in range(6):
                        assert thecolumns[i] == thecols[i]
                assert filetype == "bidscontinuous"
            elif thetype == "plaintsv":
                assert thesamplerate == None
                assert thestarttime == None
                if thefileroot.find("nocol") > 0:
                    assert len(thecolumns) == len(thecols)
                    for i in range(6):
                        assert thecolumns[i] == tide_io.makecolname(i, 0)
                else:
                    assert len(thecolumns) == len(thecols)
                    for i in range(6):
                        assert thecolumns[i] == thecols[i]
                assert filetype == "plaintsv"
            assert np.max(np.fabs(thedata - the2darray)) < EPSILON

    print("reading single columns")
    for thistest in thetests:
        print()
        thetype = thistest[0]
        compressed = thistest[1]
        if compressed:
            compname = "compressed"
        else:
            compname = "uncompressed"

        for colspec in ["withcol", "nocol"]:
            if debug:
                print("\n\n\n")
            thefileroot = os.path.join(DESTDIR, f"testout_{colspec}_{thetype}_{compname}")

            if thetype == "text":
                theextension = ".par"
                if colspec == "nocol":
                    motionfilename = thefileroot + theextension
            elif thetype == "csv":
                theextension = ".csv"
            else:
                if compressed:
                    theextension = ".tsv.gz"
                else:
                    theextension = ".tsv"

            if thetype == "text":
                thespec = ":4"
            elif thetype == "csv":
                if colspec == "nocol":
                    thespec = ":4"
                else:
                    thespec = ":Y"
            elif thetype == "bidscontinuous":
                if colspec == "nocol":
                    thespec = ":4"
                else:
                    thespec = ":Y"
            elif thetype == "plaintsv":
                if colspec == "nocol":
                    thespec = ":4"
                else:
                    thespec = ":Y"
            else:
                raise ValueError("illegal file type")
            print("reading:", thetype, compressed, thefileroot)
            (
                thesamplerate,
                thestarttime,
                thecolumns,
                thedata,
                compressed,
                filetype,
            ) = tide_io.readvectorsfromtextfile(
                thefileroot + theextension + thespec, onecol=True, debug=debug
            )
            if thetype == "text":
                assert thesamplerate is None
                assert thestarttime is None
                assert thecolumns is None
                assert filetype == "text"
            elif thetype == "csv":
                assert thesamplerate is None
                assert thestarttime is None
                assert filetype == "csv"
            elif thetype == "bidscontinuous":
                assert thesamplerate == inputsamplerate
                assert thestarttime == inputstarttime
                assert filetype == "bidscontinuous"
            elif thetype == "plaintsv":
                assert thesamplerate == None
                assert thestarttime == None
                assert filetype == "plaintsv"
            assert np.max(np.fabs(thedata - the2darray[4, :])) < EPSILON

    # now check motion file routines
    assert not tide_io.checkifparfile("afilename.txt")
    assert tide_io.checkifparfile("afilename.par")

    othermotiondict = tide_io.readparfile(motionfilename)
    theinitmotiondict = tide_io.readmotion(motionfilename)
    theexpandedmotionregressors, thelabels = tide_io.calcmotregressors(
        theinitmotiondict, derivdelayed=True
    )
    if debug:
        print(theexpandedmotionregressors.shape)
        print(theexpandedmotionregressors)
    assert np.max(np.fabs(theexpandedmotionregressors[3, :] - the2darray[0, :])) < EPSILON
    assert np.max(np.fabs(theexpandedmotionregressors[4, :] - the2darray[1, :])) < EPSILON
    assert np.max(np.fabs(theexpandedmotionregressors[5, :] - the2darray[2, :])) < EPSILON
    assert np.max(np.fabs(theexpandedmotionregressors[0, :] - the2darray[3, :])) < EPSILON
    assert np.max(np.fabs(theexpandedmotionregressors[1, :] - the2darray[4, :])) < EPSILON
    assert np.max(np.fabs(theexpandedmotionregressors[2, :] - the2darray[5, :])) < EPSILON
    assert np.max(np.fabs(theexpandedmotionregressors[3, :] - the2darray[0, :])) < EPSILON

    # test fmriheaderinfo
    fmritimeinfothresh = 1e-2
    thesizes, thedims = tide_io.fmriheaderinfo(os.path.join(SOURCEDIR, "sub-HAPPYTEST.nii.gz"))
    if debug:
        print(thedims)
        print(thesizes)
    targetdims = [4, 65, 89, 64, 110, 1, 1, 1]
    targetsizes = [-1.000000, 2.395830, 2.395830, 2.400000, 1.160000, 0.000000, 0.000000, 0.000000]
    for i in range(len(targetdims)):
        assert targetdims[i] == thedims[i]
        assert np.fabs(targetsizes[i] - thesizes[i]) < EPSILON

    tide_io.niftisplit(
        os.path.join(SOURCEDIR, "sub-HAPPYTEST.nii.gz"), os.path.join(DESTDIR, "splittest"), axis=3
    )

    mergelist = []
    for i in range(10):
        mergelist.append(os.path.join(DESTDIR, "splittest" + str(i).zfill(4)))
    if debug:
        print(mergelist)
    tide_io.niftimerge(mergelist, os.path.join(DESTDIR, "merged"))

    tide_io.niftiroi(
        os.path.join(SOURCEDIR, "sub-HAPPYTEST.nii.gz"), os.path.join(DESTDIR, "roi"), 2, 9
    )

    dummy, dataarray, theheader, thedims, thesizes = tide_io.readfromnifti(
        os.path.join(DESTDIR, "roi")
    )
    thetypes = [
        np.uint8,
        np.int16,
        np.int32,
        np.float32,
        np.complex64,
        np.float64,
        np.int8,
        np.uint16,
        np.uint32,
        np.int64,
        np.uint64,
        np.complex128,
    ]
    for thedtype in thetypes:
        tide_io.savetonifti(
            dataarray.astype(thedtype), theheader, os.path.join(DESTDIR, "dtypetest"), debug=debug
        )


if __name__ == "__main__":
    test_io(debug=True, displayplots=True)
