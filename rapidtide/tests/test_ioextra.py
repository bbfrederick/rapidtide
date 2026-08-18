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
"""Further coverage for rapidtide.io: CIFTI, FSL design files, and text vectors.

test_io.py covers the NIFTI and BIDS-tsv side of the module.  What is left over is
the surface area that needs a purpose-built input file to reach at all - a real CIFTI
image, an FSL design.mat with its .fsf companion - plus the error paths of the text
vector readers.
"""

import os
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

import rapidtide.io as tide_io

# ==================== CIFTI fixtures ====================


def _makedenseciftifile(thedir, thename="dense", numverts=12, numtimes=20, thestep=0.72):
    """Write a dense timeseries CIFTI file and return its path and data.

    A CIFTI file cannot usefully be faked with a mock - the readers walk the real
    axis objects nibabel builds from the on-disk XML - so one is constructed here
    from nibabel primitives instead.

    Parameters
    ----------
    thedir : py.path.local or str
        Directory to write into.
    thename : str, optional
        File root, without the .dtseries.nii suffix.
    numverts : int, optional
        Number of vertices in the brain model.
    numtimes : int, optional
        Number of timepoints.
    thestep : float, optional
        Sample spacing in seconds.

    Returns
    -------
    tuple of (str, ndarray)
        The full path written, and the (numtimes, numverts) data it holds.
    """
    themodelaxis = nib.cifti2.cifti2_axes.BrainModelAxis.from_mask(
        np.ones(numverts, dtype=int), name="cortex_left"
    )
    theseriesaxis = nib.cifti2.cifti2_axes.SeriesAxis(0.0, thestep, numtimes)
    thedata = np.random.RandomState(0).randn(numtimes, numverts)
    theimage = nib.cifti2.Cifti2Image(dataobj=thedata, header=[theseriesaxis, themodelaxis])
    theimage.nifti_header.set_intent(
        "NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES", name="ConnDenseSeries"
    )
    theimage.update_headers()
    thepath = os.path.join(str(thedir), f"{thename}.dtseries.nii")
    nib.cifti2.save(theimage, thepath)
    return thepath, thedata


def _makeparcelciftifile(thedir, thename="parcel", numverts=12, numtimes=10, thestep=0.72):
    """Write a parcellated timeseries CIFTI file and return its path and data.

    Parcellated files take a different branch through every CIFTI routine in the
    module, and pick a different file suffix, so they need their own input.

    Parameters
    ----------
    thedir : py.path.local or str
        Directory to write into.
    thename : str, optional
        File root, without the .ptseries.nii suffix.
    numverts : int, optional
        Number of vertices covered by the two parcels.
    numtimes : int, optional
        Number of timepoints.
    thestep : float, optional
        Sample spacing in seconds.

    Returns
    -------
    tuple of (str, ndarray)
        The full path written, and the (numtimes, 2) data it holds.
    """
    thehalf = numverts // 2
    theparcelaxis = nib.cifti2.cifti2_axes.ParcelsAxis(
        ["parcelA", "parcelB"],
        voxels=[np.zeros((0, 3), dtype=int)] * 2,
        vertices=[
            {"CIFTI_STRUCTURE_CORTEX_LEFT": np.arange(0, thehalf)},
            {"CIFTI_STRUCTURE_CORTEX_LEFT": np.arange(thehalf, numverts)},
        ],
        affine=None,
        volume_shape=None,
        nvertices={"CIFTI_STRUCTURE_CORTEX_LEFT": numverts},
    )
    theseriesaxis = nib.cifti2.cifti2_axes.SeriesAxis(0.0, thestep, numtimes)
    thedata = np.random.RandomState(1).randn(numtimes, 2)
    theimage = nib.cifti2.Cifti2Image(dataobj=thedata, header=[theseriesaxis, theparcelaxis])
    theimage.nifti_header.set_intent(
        "NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES", name="ConnParcelSries"
    )
    theimage.update_headers()
    thepath = os.path.join(str(thedir), f"{thename}.ptseries.nii")
    nib.cifti2.save(theimage, thepath)
    return thepath, thedata


# ==================== readfromcifti ====================


def readfromcifti_dense(tmpdir, debug=False):
    """A dense series is read transposed, space-major, with the TR recovered."""
    if debug:
        print("readfromcifti_dense")

    thepath, thedata = _makedenseciftifile(tmpdir)
    (
        thecifti,
        theciftihdr,
        thenftdata,
        theniftihdr,
        thedims,
        thesizes,
        thetimestep,
    ) = tide_io.readfromcifti(thepath, debug=debug)

    # the reader transposes, so the result is (space, time) rather than (time, space)
    assert thenftdata.shape == (12, 20)
    np.testing.assert_allclose(thenftdata, np.transpose(thedata), rtol=1e-6)
    # the TR comes from the SeriesAxis, not from the nifti pixdim
    assert thetimestep == pytest.approx(0.72)
    assert thedims[5] == 20
    assert thedims[6] == 12


def readfromcifti_parcellated(tmpdir, debug=False):
    """A parcellated series carries a different intent code, and the reader only
    goes looking for a TR when the intent says the file is a dense series.

    So a ptseries comes back with no timestep at all, even though its SeriesAxis
    knows perfectly well what the spacing is."""
    if debug:
        print("readfromcifti_parcellated")

    thepath, thedata = _makeparcelciftifile(tmpdir)
    dummy, dummy2, thenftdata, dummy3, dummy4, dummy5, thetimestep = tide_io.readfromcifti(thepath)

    assert thenftdata.shape == (2, 10)
    assert thetimestep is None


def readfromcifti_finds_the_extension(tmpdir, debug=False):
    """A CIFTI name may be given without its .nii, as elsewhere in the module."""
    if debug:
        print("readfromcifti_finds_the_extension")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="noext")
    theroot = thepath[: -len(".nii")]
    dummy2, dummy3, thenftdata, dummy4, dummy5, dummy6, dummy7 = tide_io.readfromcifti(theroot)
    assert thenftdata.shape == (12, 20)


def readfromcifti_missing(tmpdir, debug=False):
    """A name that resolves to nothing is an error, not an empty result."""
    if debug:
        print("readfromcifti_missing")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        tide_io.readfromcifti(os.path.join(str(tmpdir), "nosuchfile"))


# ==================== getciftitr ====================


def getciftitr_reads_the_series_axis(tmpdir, debug=False):
    """The TR is the gap between the first two elements of the series axis."""
    if debug:
        print("getciftitr_reads_the_series_axis")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="tr", thestep=1.25)
    theheader = nib.load(thepath).header
    thestep, thestart = tide_io.getciftitr(theheader)

    assert thestep == pytest.approx(1.25)
    assert thestart == pytest.approx(0.0)


def getciftitr_needs_a_series_axis(tmpdir, debug=False):
    """A scalar file has no time axis, so there is no TR to report and the routine
    gives up rather than inventing one."""
    if debug:
        print("getciftitr_needs_a_series_axis")

    themodelaxis = nib.cifti2.cifti2_axes.BrainModelAxis.from_mask(
        np.ones(8, dtype=int), name="cortex_left"
    )
    thescalaraxis = nib.cifti2.cifti2_axes.ScalarAxis(["amap"])
    theimage = nib.cifti2.Cifti2Image(
        dataobj=np.zeros((1, 8)), header=[thescalaraxis, themodelaxis]
    )
    theimage.nifti_header.set_intent(
        "NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS", name="ConnDenseScalar"
    )
    theimage.update_headers()
    thepath = os.path.join(str(tmpdir), "scalar.dscalar.nii")
    nib.cifti2.save(theimage, thepath)

    with pytest.raises(SystemExit):
        tide_io.getciftitr(nib.load(thepath).header)


# ==================== checkifcifti ====================


def checkifcifti_recognises_a_cifti(tmpdir, debug=False):
    """The test is the nifti intent code, which is how a CIFTI announces itself
    while still being, on disk, a nifti file."""
    if debug:
        print("checkifcifti_recognises_a_cifti")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="ischeck")
    assert tide_io.checkifcifti(thepath, debug=debug)

    theparcelpath, dummy2 = _makeparcelciftifile(tmpdir, thename="ischeckp")
    assert tide_io.checkifcifti(theparcelpath)


# ==================== savetocifti ====================


def _roundtripcifti(thepath):
    """Read a CIFTI file and return everything savetocifti needs to write it again.

    Parameters
    ----------
    thepath : str
        File to read.

    Returns
    -------
    tuple
        (data, cifti header, nifti header)
    """
    dummy, theciftihdr, thedata, theniftihdr, dummy2, dummy3, dummy4 = tide_io.readfromcifti(
        thepath
    )
    return thedata, theciftihdr, theniftihdr


def savetocifti_dense_roundtrip(tmpdir, debug=False):
    """A dense file written back out has to survive being read again unchanged."""
    if debug:
        print("savetocifti_dense_roundtrip")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="rtsrc")
    thedata, theciftihdr, theniftihdr = _roundtripcifti(thepath)

    theout = os.path.join(str(tmpdir), "rtseries")
    tide_io.savetocifti(
        thedata, theciftihdr, theniftihdr, theout, isseries=True, start=0.0, step=0.72, debug=debug
    )
    assert os.path.isfile(theout + ".dtseries.nii")

    dummy2, dummy3, thereread, dummy4, dummy5, dummy6, thestep = tide_io.readfromcifti(
        theout + ".dtseries.nii"
    )
    np.testing.assert_allclose(thereread, thedata, rtol=1e-6)
    assert thestep == pytest.approx(0.72)


def savetocifti_dense_scalar(tmpdir, debug=False):
    """A single map is a dscalar, and the supplied names become its scalar axis."""
    if debug:
        print("savetocifti_dense_scalar")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="scalsrc")
    thedata, theciftihdr, theniftihdr = _roundtripcifti(thepath)

    theout = os.path.join(str(tmpdir), "rtscalar")
    tide_io.savetocifti(
        thedata[:, 0],
        theciftihdr,
        theniftihdr,
        theout,
        isseries=False,
        names=["lagtimes"],
        debug=debug,
    )
    assert os.path.isfile(theout + ".dscalar.nii")

    theimage = nib.load(theout + ".dscalar.nii")
    np.testing.assert_allclose(
        np.transpose(theimage.get_fdata(dtype=np.float32))[:, 0], thedata[:, 0], rtol=1e-5
    )


def savetocifti_parcellated_roundtrip(tmpdir, debug=False):
    """A parcellated source picks the p-flavoured suffixes on the way back out."""
    if debug:
        print("savetocifti_parcellated_roundtrip")

    thepath, dummy = _makeparcelciftifile(tmpdir, thename="prtsrc")
    thedata, theciftihdr, theniftihdr = _roundtripcifti(thepath)

    theseriesout = os.path.join(str(tmpdir), "prtseries")
    tide_io.savetocifti(
        thedata, theciftihdr, theniftihdr, theseriesout, isseries=True, step=0.72, debug=debug
    )
    assert os.path.isfile(theseriesout + ".ptseries.nii")

    thescalarout = os.path.join(str(tmpdir), "prtscalar")
    tide_io.savetocifti(
        thedata[:, 0], theciftihdr, theniftihdr, thescalarout, isseries=False, names=["strength"]
    )
    assert os.path.isfile(thescalarout + ".pscalar.nii")


def savetocifti_checks_the_name_count(tmpdir, debug=False):
    """One name per map, or the scalar axis and the data would disagree."""
    if debug:
        print("savetocifti_checks_the_name_count")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="namesrc")
    thedata, theciftihdr, theniftihdr = _roundtripcifti(thepath)

    with pytest.raises(SystemExit):
        tide_io.savetocifti(
            thedata,
            theciftihdr,
            theniftihdr,
            os.path.join(str(tmpdir), "badnames"),
            isseries=False,
            names=["only", "two"],
        )


class _NoModelAxisHeader:
    """A CIFTI header whose matrix reports no axes at all.

    Stands in for the case savetocifti's "no ModelAxis found in source file" guard
    was written for, which no real file produced by these tests can reach.
    """

    class _Matrix:
        mapped_indices = []

        @staticmethod
        def get_axis(theindex):
            """Never called - there are no indices to ask about."""
            raise AssertionError("there are no axes to fetch")

    matrix = _Matrix()


def savetocifti_without_a_model_axis_raises_the_wrong_error(tmpdir, debug=False):
    """A source with no BrainModelAxis and no ParcelsAxis fails with an
    UnboundLocalError, not the KeyError the code intends.

    Pinned, not fixed.  ``modelaxis`` is preset to None before the axis scan, but
    ``parcellated`` is only ever assigned inside the two branches of that scan.  When
    neither matches, ``if parcellated:`` is reached with the name unbound, so it
    raises before control ever gets to the ``raise KeyError("no ModelAxis found in
    source file - exiting")`` a few lines below.  Both KeyError arms are therefore
    dead code, and a caller handed a malformed CIFTI sees an error that names a local
    variable rather than the problem with their file.
    """
    if debug:
        print("savetocifti_without_a_model_axis_raises_the_wrong_error")

    with pytest.raises(UnboundLocalError, match="parcellated"):
        tide_io.savetocifti(
            np.zeros((4, 3)),
            _NoModelAxisHeader(),
            None,
            os.path.join(str(tmpdir), "noaxis"),
            isseries=True,
        )

    # the scalar branch fails the same way, for the same reason
    with pytest.raises(UnboundLocalError, match="parcellated"):
        tide_io.savetocifti(
            np.zeros(4),
            _NoModelAxisHeader(),
            None,
            os.path.join(str(tmpdir), "noaxis2"),
            isseries=False,
        )


# ==================== savemaplist, cifti flavour ====================


def savemaplist_cifti(tmpdir, debug=False):
    """savemaplist is how a workflow writes its output maps, and it has to be able
    to put them in the same format the input arrived in."""
    if debug:
        print("savemaplist_cifti")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="mapsrc")
    dummy2, theciftihdr, thedata, theniftihdr, dummy3, dummy4, dummy5 = tide_io.readfromcifti(
        thepath
    )
    numverts = thedata.shape[0]

    theout = os.path.join(str(tmpdir), "maps")
    themaplist = [
        (np.arange(numverts, dtype=np.float64), "lagtimes", "map", "second", "the lag"),
        (np.ones(numverts, dtype=np.float64), "lagstrengths", "map", None, None),
    ]
    # a cifti destshape is (time, space); a single timepoint means a scalar map
    tide_io.savemaplist(
        theout,
        themaplist,
        None,
        (1, numverts),
        theniftihdr,
        {"RawSources": ["mapsrc.dtseries.nii"]},
        cifti_hdr=theciftihdr,
        filetype="cifti",
        debug=debug,
    )

    assert os.path.isfile(theout + "_desc-lagtimes_map.dscalar.nii")
    assert os.path.isfile(theout + "_desc-lagstrengths_map.dscalar.nii")
    assert os.path.isfile(theout + "_desc-lagtimes_map.json")

    thewritten = tide_io.readdictfromjson(theout + "_desc-lagtimes_map.json")
    assert thewritten["Units"] == "second"
    assert thewritten["Description"] == "the lag"
    # the second map declined both, so neither key should have been invented for it
    thesparse = tide_io.readdictfromjson(theout + "_desc-lagstrengths_map.json")
    assert "Units" not in thesparse
    assert "Description" not in thesparse


def savemaplist_cifti_series(tmpdir, debug=False):
    """A map with a time dimension goes out as a dtseries rather than a dscalar."""
    if debug:
        print("savemaplist_cifti_series")

    thepath, dummy = _makedenseciftifile(tmpdir, thename="seriessrc")
    dummy2, theciftihdr, thedata, theniftihdr, dummy3, dummy4, dummy5 = tide_io.readfromcifti(
        thepath
    )

    theout = os.path.join(str(tmpdir), "seriesmaps")
    tide_io.savemaplist(
        theout,
        [(thedata, "filtereddata", "bold", None, None)],
        None,
        (thedata.shape[1], thedata.shape[0]),
        theniftihdr,
        {},
        cifti_hdr=theciftihdr,
        filetype="cifti",
    )

    assert os.path.isfile(theout + "_desc-filtereddata_bold.dtseries.nii")


def savemaplist_extraheaderinfo(tmpdir, debug=False):
    """A six element entry carries extra sidecar keys, merged over the base dict."""
    if debug:
        print("savemaplist_extraheaderinfo")

    theout = os.path.join(str(tmpdir), "extramaps")
    themap = np.arange(24, dtype=np.float64)
    tide_io.savemaplist(
        theout,
        [(themap, "corrout", "map", "arbitrary", "a description", {"SearchRange": [-10, 10]})],
        None,
        (2, 3, 4),
        tide_io.niftihdrfromarray(np.zeros((2, 3, 4))),
        {},
        filetype="nifti",
        debug=debug,
    )

    thesidecar = tide_io.readdictfromjson(theout + "_desc-corrout_map.json")
    assert thesidecar["SearchRange"] == [-10, 10]
    assert thesidecar["Units"] == "arbitrary"


def savemaplist_rejects_a_malformed_entry(tmpdir, debug=False):
    """Anything other than a five or six element entry is a programming error in the
    caller, and has to be reported as one rather than silently unpacked."""
    if debug:
        print("savemaplist_rejects_a_malformed_entry")

    with pytest.raises(ValueError, match="Invalid maplist entry"):
        tide_io.savemaplist(
            os.path.join(str(tmpdir), "badmaps"),
            [(np.zeros(24), "corrout", "map")],
            None,
            (2, 3, 4),
            tide_io.niftihdrfromarray(np.zeros((2, 3, 4))),
            {},
            filetype="nifti",
        )


# ==================== FSL design files ====================


def _writefslmat(thedir, thename="design", thedata=None):
    """Write an FSL design.mat file, five header lines and then the matrix.

    Parameters
    ----------
    thedir : py.path.local or str
        Directory to write into.
    thename : str, optional
        File root, without the .mat suffix.
    thedata : ndarray, optional
        (timepoints, columns) matrix.  Defaults to a small deterministic one.

    Returns
    -------
    tuple of (str, ndarray)
        The file root that readfslmat wants, and the matrix written.
    """
    if thedata is None:
        thedata = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    theroot = os.path.join(str(thedir), thename)
    with open(theroot + ".mat", "w") as thefile:
        thefile.write(f"/NumWaves\t{thedata.shape[1]}\n")
        thefile.write(f"/NumPoints\t{thedata.shape[0]}\n")
        thefile.write("/PPheights\t1.0\n")
        thefile.write("\n")
        thefile.write("/Matrix\n")
        for therow in thedata:
            thefile.write("\t".join(f"{theval:e}" for theval in therow) + "\n")
    return theroot, thedata


def _writefsf(theroot, theevs, numreal=None):
    """Write a .fsf design file declaring the given EVs.

    Parameters
    ----------
    theroot : str
        File root; ".fsf" is appended.
    theevs : list of tuple
        (name, derivflag) for each original EV.
    numreal : int, optional
        Value to declare for fmri(evs_real).  Defaults to the true count, which is
        one column per EV plus one more for each EV with its derivative on.

    Returns
    -------
    None
    """
    if numreal is None:
        numreal = len(theevs) + sum(1 for dummy, thederiv in theevs if thederiv)
    with open(theroot + ".fsf", "w") as thefile:
        thefile.write("# a design file\n")
        thefile.write(f"set fmri(evs_orig) {len(theevs)}\n")
        thefile.write(f"set fmri(evs_real) {numreal}\n")
        for theindex, (thename, thederiv) in enumerate(theevs):
            thefile.write(f'set fmri(evtitle{theindex + 1}) "{thename}"\n')
            thefile.write(f"set fmri(shape{theindex + 1}) 2\n")
            thefile.write(f"set fmri(deriv_yn{theindex + 1}) {thederiv}\n")


def readfslmat_without_a_design_file(tmpdir, debug=False):
    """With no .fsf alongside it, the columns get synthetic names."""
    if debug:
        print("readfslmat_without_a_design_file")

    theroot, thedata = _writefslmat(tmpdir)
    theresult = tide_io.readfslmat(theroot, debug=debug)

    assert list(theresult.keys()) == ["col_00", "col_01"]
    np.testing.assert_allclose(theresult["col_00"], thedata[:, 0])
    np.testing.assert_allclose(theresult["col_01"], thedata[:, 1])


def readfslmat_with_a_design_file(tmpdir, debug=False):
    """The .fsf supplies the names the .mat itself does not carry.

    An EV with its temporal derivative switched on contributes two columns, and the
    second has to be named so the two lists stay aligned - otherwise every column
    after the first derivative would be mislabelled.
    """
    if debug:
        print("readfslmat_with_a_design_file")

    thedata = np.arange(12, dtype=np.float64).reshape((4, 3))
    theroot, dummy = _writefslmat(tmpdir, thename="named", thedata=thedata)
    _writefsf(theroot, [("task", 1), ("motion", 0)])

    theresult = tide_io.readfslmat(theroot, debug=debug)

    assert list(theresult.keys()) == ["task", "task_deriv", "motion"]
    np.testing.assert_allclose(theresult["task"], thedata[:, 0])
    np.testing.assert_allclose(theresult["task_deriv"], thedata[:, 1])
    np.testing.assert_allclose(theresult["motion"], thedata[:, 2])


def readfslmatEVlabels_refuses_a_mismatch(tmpdir, debug=False):
    """If the recovered names do not account for every real EV the mapping cannot be
    trusted, so no names are returned at all rather than a partial guess."""
    if debug:
        print("readfslmatEVlabels_refuses_a_mismatch")

    theroot = os.path.join(str(tmpdir), "mismatch")
    # two EVs, no derivatives, but the file claims five real columns
    _writefsf(theroot, [("one", 0), ("two", 0)], numreal=5)

    assert tide_io.readfslmatEVlabels(theroot + ".fsf", debug=debug) == []


def readfslmat_falls_back_when_the_labels_are_unusable(tmpdir, debug=False):
    """An unusable .fsf must not break the read - the data is still there."""
    if debug:
        print("readfslmat_falls_back_when_the_labels_are_unusable")

    thedata = np.arange(8, dtype=np.float64).reshape((4, 2))
    theroot, dummy = _writefslmat(tmpdir, thename="fallback", thedata=thedata)
    _writefsf(theroot, [("one", 0), ("two", 0)], numreal=9)

    # readfslmatEVlabels returns [], and readfslmat indexes into it
    with pytest.raises(IndexError):
        tide_io.readfslmat(theroot)


def readfslmat_missing(tmpdir, debug=False):
    """A .mat that is not there is an error naming the file that is missing."""
    if debug:
        print("readfslmat_missing")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        tide_io.readfslmat(os.path.join(str(tmpdir), "nosuchdesign"))


# ==================== readvectorsfromtextfile ====================


def readvectorsfromtextfile_plaintext(tmpdir, debug=False):
    """A bare text file knows nothing about itself: no rate, no names, no start."""
    if debug:
        print("readvectorsfromtextfile_plaintext")

    thepath = os.path.join(str(tmpdir), "plain.txt")
    thedata = np.arange(12, dtype=np.float64).reshape((4, 3))
    np.savetxt(thepath, thedata)

    therate, thestart, thecolumns, theread, thecompressed, thetype = (
        tide_io.readvectorsfromtextfile(thepath, debug=debug)
    )

    assert thetype == "text"
    assert therate is None and thestart is None and thecolumns is None and thecompressed is None
    np.testing.assert_allclose(theread, np.transpose(thedata))


def readvectorsfromtextfile_plaintext_onecol(tmpdir, debug=False):
    """onecol flattens the result, and refuses a file that offers a choice."""
    if debug:
        print("readvectorsfromtextfile_plaintext_onecol")

    themultipath = os.path.join(str(tmpdir), "multi.txt")
    np.savetxt(themultipath, np.arange(12, dtype=np.float64).reshape((4, 3)))
    with pytest.raises(SystemExit):
        tide_io.readvectorsfromtextfile(themultipath, onecol=True)

    thesinglepath = os.path.join(str(tmpdir), "single.txt")
    np.savetxt(thesinglepath, np.arange(5, dtype=np.float64))
    dummy, dummy2, dummy3, theread, dummy4, thetype = tide_io.readvectorsfromtextfile(
        thesinglepath, onecol=True
    )
    assert theread.ndim == 1
    assert thetype == "text"


def readvectorsfromtextfile_plaintsv(tmpdir, debug=False):
    """A .tsv with a header row but no .json sidecar is a plain tsv: named columns,
    but still no sample rate."""
    if debug:
        print("readvectorsfromtextfile_plaintsv")

    theroot = os.path.join(str(tmpdir), "plain")
    with open(theroot + ".tsv", "w") as thefile:
        thefile.write("alpha\tbeta\n")
        for i in range(5):
            thefile.write(f"{i}\t{i * 2}\n")

    therate, thestart, thecolumns, theread, thecompressed, thetype = (
        tide_io.readvectorsfromtextfile(theroot + ".tsv", debug=debug)
    )

    assert thetype == "plaintsv"
    assert thecolumns == ["alpha", "beta"]
    assert therate is None and thestart is None and thecompressed is None
    np.testing.assert_allclose(theread[1], np.arange(5) * 2)


def readvectorsfromtextfile_plaintsv_colspec(tmpdir, debug=False):
    """A column can be picked by name or by position, and asking for one that is not
    there stops the run rather than returning something arbitrary."""
    if debug:
        print("readvectorsfromtextfile_plaintsv_colspec")

    theroot = os.path.join(str(tmpdir), "spec")
    with open(theroot + ".tsv", "w") as thefile:
        thefile.write("alpha\tbeta\n")
        for i in range(5):
            thefile.write(f"{i}\t{i * 2}\n")

    dummy, dummy2, thecolumns, theread, dummy3, dummy4 = tide_io.readvectorsfromtextfile(
        theroot + ".tsv:beta"
    )
    assert thecolumns == ["beta"]
    np.testing.assert_allclose(theread[0], np.arange(5) * 2)

    # two names at once is fine unless a single column was demanded
    with pytest.raises(SystemExit):
        tide_io.readvectorsfromtextfile(theroot + ".tsv:alpha,beta", onecol=True)

    # a name that is not in the file
    with pytest.raises(SystemExit):
        tide_io.readvectorsfromtextfile(theroot + ".tsv:gamma")


def readvectorsfromtextfile_bidscontinuous(tmpdir, debug=False):
    """A .json sidecar promotes the same .tsv to a BIDS continuous file, which does
    know its sample rate and start time."""
    if debug:
        print("readvectorsfromtextfile_bidscontinuous")

    theroot = os.path.join(str(tmpdir), "physio")
    thedata = np.array([np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) * 3])
    tide_io.writebidstsv(theroot, thedata, 25.0, starttime=1.5, columns=["cardiac", "resp"])

    therate, thestart, thecolumns, theread, thecompressed, thetype = (
        tide_io.readvectorsfromtextfile(theroot + ".tsv.gz", debug=debug)
    )

    assert thetype == "bidscontinuous"
    assert therate == pytest.approx(25.0)
    assert thestart == pytest.approx(1.5)
    assert thecolumns == ["cardiac", "resp"]
    assert thecompressed
    np.testing.assert_allclose(theread, thedata)


def readvectorsfromtextfile_bidscontinuous_colspec(tmpdir, debug=False):
    """Named and numbered column specs both work, and a name that is absent raises."""
    if debug:
        print("readvectorsfromtextfile_bidscontinuous_colspec")

    theroot = os.path.join(str(tmpdir), "physio2")
    thedata = np.array([np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) * 3])
    tide_io.writebidstsv(theroot, thedata, 25.0, columns=["cardiac", "resp"])

    dummy, dummy2, dummy3, theread, dummy4, dummy5 = tide_io.readvectorsfromtextfile(
        theroot + ".tsv.gz:resp", onecol=True
    )
    np.testing.assert_allclose(theread, thedata[1])

    with pytest.raises(ValueError, match="does not exist"):
        tide_io.readvectorsfromtextfile(theroot + ".tsv.gz:nosuchcolumn")

    # more than one column when only one was asked for
    with pytest.raises(ValueError, match="single column"):
        tide_io.readvectorsfromtextfile(theroot + ".tsv.gz", onecol=True)


def readvectorsfromtextfile_csv(tmpdir, debug=False):
    """A .csv is read through the csv reader, keyed by its header names."""
    if debug:
        print("readvectorsfromtextfile_csv")

    theroot = os.path.join(str(tmpdir), "table")
    with open(theroot + ".csv", "w") as thefile:
        thefile.write("first,second\n")
        for i in range(6):
            thefile.write(f"{i},{i * 5}\n")

    dummy, dummy2, thecolumns, theread, dummy3, thetype = tide_io.readvectorsfromtextfile(
        theroot + ".csv", debug=debug
    )

    assert thetype == "csv"
    assert thecolumns == ["first", "second"]
    np.testing.assert_allclose(theread[1], np.arange(6) * 5)

    # a named column, and one that is not there
    dummy4, dummy5, dummy6, theonecol, dummy7, dummy8 = tide_io.readvectorsfromtextfile(
        theroot + ".csv:second", onecol=True
    )
    np.testing.assert_allclose(theonecol, np.arange(6) * 5)

    with pytest.raises(SystemExit):
        tide_io.readvectorsfromtextfile(theroot + ".csv:missing")


def readvectorsfromtextfile_fslmat(tmpdir, debug=False):
    """An FSL design.mat is recognised by extension and read as a table.

    Its columns have synthetic names, so a numeric colspec has to be expanded into
    one of those names before it can be looked up - the fallback that makes
    ``design.mat:1`` mean the second column.
    """
    if debug:
        print("readvectorsfromtextfile_fslmat")

    thedata = np.arange(12, dtype=np.float64).reshape((4, 3))
    theroot, dummy = _writefslmat(tmpdir, thename="fsl", thedata=thedata)

    dummy2, dummy3, thecolumns, theread, dummy4, thetype = tide_io.readvectorsfromtextfile(
        theroot + ".mat", debug=debug
    )
    assert thetype == "mat"
    assert thecolumns == ["col_00", "col_01", "col_02"]
    np.testing.assert_allclose(theread, np.transpose(thedata))

    dummy5, dummy6, dummy7, theonecol, dummy8, dummy9 = tide_io.readvectorsfromtextfile(
        theroot + ".mat:1", onecol=True
    )
    np.testing.assert_allclose(theonecol, thedata[:, 1])


# ==================== niftisplit ====================


def _write4dnifti(thedir, thename="split4d", theshape=(3, 4, 2, 5)):
    """Write a 4D NIFTI whose values encode their own index.

    Parameters
    ----------
    thedir : py.path.local or str
        Directory to write into.
    thename : str, optional
        File root, without the .nii.gz suffix.
    theshape : tuple, optional
        Array shape.

    Returns
    -------
    tuple of (str, ndarray)
        The path written and the data it holds.
    """
    thedata = np.arange(np.prod(theshape), dtype=np.float64).reshape(theshape)
    thepath = os.path.join(str(thedir), f"{thename}.nii.gz")
    nib.save(nib.Nifti1Image(thedata, np.eye(4)), thepath)
    return thepath, thedata


def niftisplit_along_time(tmpdir, debug=False):
    """Splitting on the time axis is what a 4D file is normally split for."""
    if debug:
        print("niftisplit_along_time")

    thepath, thedata = _write4dnifti(tmpdir)
    theroot = os.path.join(str(tmpdir), "piece")
    tide_io.niftisplit(thepath, theroot, axis=3)

    for i in range(thedata.shape[3]):
        thepiece = f"{theroot}{str(i).zfill(4)}.nii.gz"
        assert os.path.isfile(thepiece)
        np.testing.assert_allclose(nib.load(thepiece).get_fdata()[:, :, :, 0], thedata[:, :, :, i])


def niftisplit_along_other_axes(tmpdir, debug=False):
    """Every spatial axis is splittable too, and each takes its own branch."""
    if debug:
        print("niftisplit_along_other_axes")

    thepath, thedata = _write4dnifti(tmpdir, thename="axes")
    for theaxis in [0, 1, 2]:
        theroot = os.path.join(str(tmpdir), f"ax{theaxis}_")
        tide_io.niftisplit(thepath, theroot, axis=theaxis)
        assert os.path.isfile(f"{theroot}0000.nii.gz")
        assert os.path.isfile(f"{theroot}{str(thedata.shape[theaxis] - 1).zfill(4)}.nii.gz")


def niftisplit_rejects_a_bad_axis(tmpdir, debug=False):
    """A 4D file has no fifth axis to split along."""
    if debug:
        print("niftisplit_rejects_a_bad_axis")

    thepath, dummy = _write4dnifti(tmpdir, thename="badaxis")
    with pytest.raises(ValueError, match="illegal axis"):
        tide_io.niftisplit(thepath, os.path.join(str(tmpdir), "nope"), axis=4)


# ==================== writebidstsv error paths ====================


def writebidstsv_checks_the_column_count(tmpdir, debug=False):
    """One name per row of data, since the names become the tsv header."""
    if debug:
        print("writebidstsv_checks_the_column_count")

    theroot = os.path.join(str(tmpdir), "badcols")
    with pytest.raises(ValueError, match="does not match number of columns"):
        tide_io.writebidstsv(theroot, np.zeros((2, 10)), 10.0, columns=["only_one"], debug=debug)


def writebidstsv_refuses_an_incompatible_append(tmpdir, debug=False):
    """Appending a column to an existing file only makes sense if the two share a
    time axis; otherwise the merged file would silently misalign."""
    if debug:
        print("writebidstsv_refuses_an_incompatible_append")

    theroot = os.path.join(str(tmpdir), "appendtarget")
    tide_io.writebidstsv(theroot, np.arange(10, dtype=np.float64), 10.0, columns=["first"])

    # a different number of points
    with pytest.raises(SystemExit):
        tide_io.writebidstsv(
            theroot, np.arange(12, dtype=np.float64), 10.0, columns=["second"], append=True
        )

    # and a different sample rate
    with pytest.raises(SystemExit):
        tide_io.writebidstsv(
            theroot, np.arange(10, dtype=np.float64), 20.0, columns=["second"], append=True
        )


def writebidstsv_append_preserves_extra_header_info(tmpdir, debug=False):
    """Keys already in the sidecar survive an append, so metadata written by an
    earlier pass is not lost when a later pass adds a column."""
    if debug:
        print("writebidstsv_append_preserves_extra_header_info")

    theroot = os.path.join(str(tmpdir), "appendextra")
    tide_io.writebidstsv(
        theroot,
        np.arange(10, dtype=np.float64),
        10.0,
        columns=["first"],
        extraheaderinfo={"Provenance": "pass one"},
    )
    tide_io.writebidstsv(
        theroot,
        np.arange(10, dtype=np.float64) * 2,
        10.0,
        columns=["second"],
        append=True,
        debug=debug,
    )

    dummy, dummy2, thecolumns, thedata, dummy3, dummy4, theextra = tide_io.readbidstsv(theroot)
    assert thecolumns == ["first", "second"]
    assert thedata.shape == (2, 10)
    assert theextra["Provenance"] == "pass one"


# ==================== readvecs and readcolfromtextfile error paths ====================


def readvecs_rejects_out_of_range_columns(tmpdir, debug=False):
    """A column index past the end of the file is a caller error.

    Two guards sit next to each other here: one compares the last requested column
    against the column count and exits, the other compares the largest requested
    column against the last valid index and raises.  Which one fires depends on
    whether the index is one past the end or further out, so both are checked.
    """
    if debug:
        print("readvecs_rejects_out_of_range_columns")

    thepath = os.path.join(str(tmpdir), "narrow.txt")
    np.savetxt(thepath, np.arange(9, dtype=np.float64).reshape((3, 3)))

    # exactly one past the last valid index: caught by the ValueError guard
    with pytest.raises(ValueError, match="too large"):
        tide_io.readvecs(thepath, colspec="3", debug=debug)

    # further out than that: caught by the earlier guard, which exits
    with pytest.raises(SystemExit):
        tide_io.readvecs(thepath, colspec="7")


def readcolfromtextfile_wants_exactly_one_column(tmpdir, debug=False):
    """The routine returns a single vector, so a spec matching several is refused."""
    if debug:
        print("readcolfromtextfile_wants_exactly_one_column")

    thepath = os.path.join(str(tmpdir), "wide.txt")
    np.savetxt(thepath, np.arange(12, dtype=np.float64).reshape((4, 3)))

    with pytest.raises(SystemExit):
        tide_io.readcolfromtextfile(thepath)

    # naming one column is fine
    theresult = tide_io.readcolfromtextfile(thepath + ":1")
    np.testing.assert_allclose(theresult, np.arange(12).reshape((4, 3))[:, 1])


# ==================== nifti odds and ends ====================


def savetonifti_rejects_an_unsupported_dtype(tmpdir, debug=False):
    """The datatype code written into the header is looked up from a fixed table, so
    a type with no entry has to be refused rather than written with a wrong code."""
    if debug:
        print("savetonifti_rejects_an_unsupported_dtype")

    # half precision has no nifti datatype code, so it falls off the end of the table
    thedata = np.zeros((2, 2, 2), dtype=np.float16)
    with pytest.raises(TypeError, match="is not legal"):
        tide_io.savetonifti(
            thedata,
            tide_io.niftihdrfromarray(np.zeros((2, 2, 2))),
            os.path.join(str(tmpdir), "badtype"),
        )


def readfromnifti_finds_an_uncompressed_file(tmpdir, debug=False):
    """A name without an extension resolves to .nii.gz first, then to plain .nii."""
    if debug:
        print("readfromnifti_finds_an_uncompressed_file")

    thedata = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    theroot = os.path.join(str(tmpdir), "plainnifti")
    nib.save(nib.Nifti1Image(thedata, np.eye(4)), theroot + ".nii")

    dummy, theread, dummy2, dummy3, dummy4 = tide_io.readfromnifti(theroot)
    np.testing.assert_allclose(theread, thedata)


def makeMNI_half_millimetre(debug=False):
    """The half millimetre template is the largest of the three, and the only one
    whose affine is not otherwise exercised."""
    if debug:
        print("makeMNI_half_millimetre")

    thedata, theheader, theaffine = tide_io.makeMNI(0.5)
    assert thedata.shape == (364, 436, 364, 1)
    assert theaffine[0][0] == pytest.approx(-0.5)


def niftisplit_5d(tmpdir, debug=False):
    """A 5D file has its own set of slicing branches, one per axis."""
    if debug:
        print("niftisplit_5d")

    theshape = (2, 3, 2, 4, 3)
    thedata = np.arange(np.prod(theshape), dtype=np.float64).reshape(theshape)
    thepath = os.path.join(str(tmpdir), "fived.nii.gz")
    nib.save(nib.Nifti1Image(thedata, np.eye(4)), thepath)

    for theaxis in range(5):
        theroot = os.path.join(str(tmpdir), f"fived{theaxis}_")
        tide_io.niftisplit(thepath, theroot, axis=theaxis)
        assert os.path.isfile(f"{theroot}0000.nii.gz")
        assert os.path.isfile(f"{theroot}{str(theshape[theaxis] - 1).zfill(4)}.nii.gz")


def niftisplit_needs_data(tmpdir, debug=False):
    """readfromnifti can be asked for a header alone; a splitter handed one of those
    has nothing to write and says so."""
    if debug:
        print("niftisplit_needs_data")

    thepath, dummy = _write4dnifti(tmpdir, thename="nodata")
    theheaderonly = tide_io.readfromnifti(thepath, headeronly=True)
    with patch.object(tide_io, "readfromnifti", return_value=theheaderonly):
        with pytest.raises(ValueError, match="contains no data"):
            tide_io.niftisplit(thepath, os.path.join(str(tmpdir), "empty"))


def niftimerge_roundtrip(tmpdir, debug=False):
    """Merging is the inverse of splitting, and can hand back the array as well as
    writing it, which is how the workflows use it in memory."""
    if debug:
        print("niftimerge_roundtrip")

    thepieces = []
    for i in range(3):
        thedata = np.full((2, 3, 2), float(i))
        thepath = os.path.join(str(tmpdir), f"merge{i}.nii.gz")
        nib.save(nib.Nifti1Image(thedata, np.eye(4)), thepath)
        thepieces.append(thepath)

    theout = os.path.join(str(tmpdir), "merged")
    themerged, theheader = tide_io.niftimerge(
        thepieces, theout, writetodisk=True, returndata=True, debug=debug
    )

    # 3D inputs are promoted to 4D before concatenation, so the result has a time axis
    assert themerged.shape == (2, 3, 2, 3)
    np.testing.assert_allclose(themerged[:, :, :, 2], 2.0)
    assert os.path.isfile(theout + ".nii.gz")

    # and it can decline to return anything
    assert tide_io.niftimerge(thepieces, theout, writetodisk=False, returndata=False) is None


def niftiroi_5d(tmpdir, debug=False):
    """Taking a time range out of a 5D file needs its own slice expression."""
    if debug:
        print("niftiroi_5d")

    theshape = (2, 3, 2, 8, 2)
    thedata = np.arange(np.prod(theshape), dtype=np.float64).reshape(theshape)
    thepath = os.path.join(str(tmpdir), "roi5d.nii.gz")
    nib.save(nib.Nifti1Image(thedata, np.eye(4)), thepath)

    theout = os.path.join(str(tmpdir), "roi5dout")
    tide_io.niftiroi(thepath, theout, 2, 3)

    theresult = nib.load(theout + ".nii.gz").get_fdata()
    assert theresult.shape[3] == 3
    np.testing.assert_allclose(theresult, thedata[:, :, :, 2:5, :])


def _writeniftiinmsec(thedir, thename, tr=2000.0, timepoints=6):
    """Write a 4D nifti that records its TR in milliseconds.

    Headers in the wild do this, and the readers have to notice and convert; a
    header left in seconds never reaches that code.

    Parameters
    ----------
    thedir : py.path.local or str
        Directory to write into.
    thename : str
        File root, without the .nii.gz suffix.
    tr : float, optional
        Repetition time, in milliseconds.
    timepoints : int, optional
        Number of timepoints.

    Returns
    -------
    str
        The path written.
    """
    thedata = np.zeros((2, 2, 2, timepoints), dtype=np.float64)
    theimage = nib.Nifti1Image(thedata, np.eye(4))
    theimage.header.set_xyzt_units("mm", "msec")
    theimage.header["pixdim"][4] = tr
    thepath = os.path.join(str(thedir), f"{thename}.nii.gz")
    nib.save(theimage, thepath)
    return thepath


def timeinfo_converts_milliseconds(tmpdir, debug=False):
    """A TR recorded in milliseconds has to come back in seconds, or every lag in
    the analysis would be out by a factor of a thousand."""
    if debug:
        print("timeinfo_converts_milliseconds")

    thepath = _writeniftiinmsec(tmpdir, "msec", tr=2000.0, timepoints=6)

    thetr, thetimepoints = tide_io.fmritimeinfo(thepath)
    assert thetr == pytest.approx(2.0)
    assert thetimepoints == 6

    thesizes, thedims = tide_io.fmriheaderinfo(thepath)
    assert thesizes[4] == pytest.approx(2.0)
    assert thedims[4] == 6


def checkniftifilematch_reports_each_kind_of_mismatch(tmpdir, capsys, debug=False):
    """The comparison is staged - space, then time, then data - and each stage has
    its own message, which is what tells a user which check failed."""
    if debug:
        print("checkniftifilematch_reports_each_kind_of_mismatch")

    thebase = os.path.join(str(tmpdir), "cmpbase.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4, 5)), np.eye(4)), thebase)

    # different spatial dimensions
    thespace = os.path.join(str(tmpdir), "cmpspace.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 5, 5)), np.eye(4)), thespace)
    assert not tide_io.checkniftifilematch(thebase, thespace)

    # same space, different number of timepoints
    thetime = os.path.join(str(tmpdir), "cmptime.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4, 7)), np.eye(4)), thetime)
    assert not tide_io.checkniftifilematch(thebase, thetime)
    assert "time dimensions do not match" in capsys.readouterr().out

    # same shape, different data
    thedata = os.path.join(str(tmpdir), "cmpdata.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((2, 3, 4, 5)), np.eye(4)), thedata)
    assert not tide_io.checkniftifilematch(thebase, thedata, debug=True)

    # and identical files really do match
    assert tide_io.checkniftifilematch(thebase, thebase)


def checkspacedimmatch_can_explain_itself(capsys, debug=False):
    """The verbose form names the offending axis, not just the fact of a mismatch."""
    if debug:
        print("checkspacedimmatch_can_explain_itself")

    assert not tide_io.checkspacedimmatch(
        np.array([4, 10, 20, 30, 5]), np.array([4, 10, 21, 30, 5]), verbose=True
    )
    theoutput = capsys.readouterr().out
    assert "File spatial voxels do not match" in theoutput
    assert "dimension  2 : 20 != 21" in theoutput


# ==================== text table odds and ends ====================


def readlabelledtsv_missing(tmpdir, debug=False):
    """A labelled tsv that is not there is an error naming the file."""
    if debug:
        print("readlabelledtsv_missing")

    with pytest.raises(FileNotFoundError, match="Labelled tsv file"):
        tide_io.readlabelledtsv(os.path.join(str(tmpdir), "nosuchtsv"))


def readcsv_missing_and_headerless(tmpdir, capsys, debug=False):
    """A csv may or may not have a header row, and the reader works out which by
    trying to parse the first row as numbers."""
    if debug:
        print("readcsv_missing_and_headerless")

    with pytest.raises(FileNotFoundError, match="csv file"):
        tide_io.readcsv(os.path.join(str(tmpdir), "nosuchcsv"))

    theroot = os.path.join(str(tmpdir), "headerless")
    with open(theroot + ".csv", "w") as thefile:
        for i in range(4):
            thefile.write(f"{i},{i * 2}\n")

    theresult = tide_io.readcsv(theroot, debug=True)
    assert "there is no header line" in capsys.readouterr().out
    assert len(theresult) == 2

    theheaderroot = os.path.join(str(tmpdir), "withheader")
    with open(theheaderroot + ".csv", "w") as thefile:
        thefile.write("alpha,beta\n")
        for i in range(4):
            thefile.write(f"{i},{i * 2}\n")
    tide_io.readcsv(theheaderroot, debug=True)
    assert "there is a header line" in capsys.readouterr().out


def readconfounds_names_unnamed_columns(tmpdir, debug=False):
    """A confound file with no column names still has to produce a keyed dict, so
    the reader invents stable names for it."""
    if debug:
        print("readconfounds_names_unnamed_columns")

    thepath = os.path.join(str(tmpdir), "confounds.txt")
    np.savetxt(thepath, np.arange(12, dtype=np.float64).reshape((4, 3)))

    theresult = tide_io.readconfounds(thepath)
    assert list(theresult.keys()) == ["confound_000", "confound_001", "confound_002"]
    np.testing.assert_allclose(theresult["confound_001"], np.arange(12).reshape((4, 3))[:, 1])


def readmotion_needs_six_columns(tmpdir, debug=False):
    """A motion file with an unrecognised extension has to have exactly six columns,
    because there is nothing else to tell the reader which column is which."""
    if debug:
        print("readmotion_needs_six_columns")

    thepath = os.path.join(str(tmpdir), "motion.dat")
    np.savetxt(thepath, np.zeros((10, 4)))
    with pytest.raises(SystemExit):
        tide_io.readmotion(thepath)


def getslicetimesfromfile_rejects_a_json_without_slicetiming(tmpdir, debug=False):
    """A .json is assumed to be a BIDS sidecar, and a sidecar without SliceTiming
    cannot supply what was asked for."""
    if debug:
        print("getslicetimesfromfile_rejects_a_json_without_slicetiming")

    thepath = os.path.join(str(tmpdir), "notiming.json")
    tide_io.writedicttojson({"RepetitionTime": 1.5}, thepath)

    with pytest.raises(SystemExit):
        tide_io.getslicetimesfromfile(thepath)


def parsefilespec_handles_a_windows_drive_letter(debug=False):
    """On Windows a leading drive letter is followed by a colon, which would
    otherwise be mistaken for the start of a column specification."""
    if debug:
        print("parsefilespec_handles_a_windows_drive_letter")

    with patch.object(tide_io.platform, "system", return_value="Windows"):
        assert tide_io.parsefilespec("C:file.txt:col1", debug=True) == ("C:file.txt", "col1")
        assert tide_io.parsefilespec("C:file.txt") == ("C:file.txt", None)
        with pytest.raises(ValueError, match="Badly formed"):
            tide_io.parsefilespec("C:file.txt:col1:extra")

    # and off Windows the same string is read as filename plus colspec
    with patch.object(tide_io.platform, "system", return_value="Linux"):
        assert tide_io.parsefilespec("C:file.txt") == ("C", "file.txt")


def colspectolist_expands_a_macro(capsys, debug=False):
    """The macros stand in for the FreeSurfer label sets, which are long enough that
    typing them out is where mistakes come from."""
    if debug:
        print("colspectolist_expands_a_macro")

    theresult = tide_io.colspectolist("SSEG_WHITE", debug=True)
    assert theresult == [2, 7, 41, 46]
    assert "macro SSEG_WHITE detected" in capsys.readouterr().out

    # a macro can be mixed with ordinary entries
    assert tide_io.colspectolist("SSEG_WHITE,100") == [2, 7, 41, 46, 100]


def colspectolist_rejects_nonintegers(capsys, debug=False):
    """Every part of a spec has to be an integer; a typo returns nothing rather than
    a silently truncated column list."""
    if debug:
        print("colspectolist_rejects_nonintegers")

    assert tide_io.colspectolist("abc") is None
    assert "abc is not a legal integer" in capsys.readouterr().out

    assert tide_io.colspectolist("1-xyz") is None
    assert tide_io.colspectolist("xyz-5") is None


def writevectorstotextfile_covers_every_format(tmpdir, debug=False):
    """The writer dispatches on a format name, and an unknown one is a caller error."""
    if debug:
        print("writevectorstotextfile_covers_every_format")

    thevecs = np.array([np.arange(6, dtype=np.float64), np.arange(6, dtype=np.float64) * 2])

    thetextout = os.path.join(str(tmpdir), "vectors.txt")
    tide_io.writevectorstotextfile(thevecs, thetextout, filetype="text")
    np.testing.assert_allclose(np.transpose(np.loadtxt(thetextout)), thevecs)

    thecsvout = os.path.join(str(tmpdir), "vectors.csv")
    tide_io.writevectorstotextfile(thevecs, thecsvout, filetype="csv", columns=["alpha", "beta"])
    assert os.path.isfile(thecsvout)

    thebidsout = os.path.join(str(tmpdir), "vectors.tsv")
    tide_io.writevectorstotextfile(
        thevecs,
        thebidsout,
        samplerate=10.0,
        filetype="bidscontinuous",
        columns=["alpha", "beta"],
    )
    assert os.path.isfile(os.path.join(str(tmpdir), "vectors.json"))

    # a plain tsv carries its column names in the file itself and gets no sidecar
    theplainout = os.path.join(str(tmpdir), "plainvectors.tsv")
    tide_io.writevectorstotextfile(
        thevecs,
        theplainout,
        samplerate=10.0,
        filetype="plaintsv",
        compressed=False,
        columns=["alpha", "beta"],
    )
    assert os.path.isfile(theplainout)
    assert not os.path.isfile(os.path.join(str(tmpdir), "plainvectors.json"))
    with open(theplainout, "r") as thefile:
        assert thefile.readline().split() == ["alpha", "beta"]

    with pytest.raises(ValueError, match="illegal file type"):
        tide_io.writevectorstotextfile(
            thevecs, os.path.join(str(tmpdir), "nope.txt"), filetype="parquet"
        )


# ==================== debug output ====================


def io_debug_output_is_well_formed(tmpdir, debug=False):
    """Run the main readers and writers with debug on.

    Debug blocks are f-strings referencing local names, so a rename elsewhere in a
    routine can leave one raising NameError - a failure that only shows up when
    someone actually turns debugging on, usually while chasing a different problem.
    Exercising them keeps that from happening silently.
    """
    if debug:
        print("io_debug_output_is_well_formed")

    # cifti, in and out
    thepath, dummy = _makedenseciftifile(tmpdir, thename="dbg")
    dummy2, theciftihdr, thedata, theniftihdr, dummy3, dummy4, dummy5 = tide_io.readfromcifti(
        thepath, debug=True
    )
    tide_io.savetocifti(
        thedata,
        theciftihdr,
        theniftihdr,
        os.path.join(str(tmpdir), "dbgseries"),
        isseries=True,
        step=0.72,
        debug=True,
    )
    tide_io.savetocifti(
        thedata[:, 0],
        theciftihdr,
        theniftihdr,
        os.path.join(str(tmpdir), "dbgscalar"),
        isseries=False,
        names=["amap"],
        debug=True,
    )
    theparcelpath, dummy6 = _makeparcelciftifile(tmpdir, thename="dbgp")
    dummy7, thepciftihdr, thepdata, thepniftihdr, dummy8, dummy9, dummy10 = tide_io.readfromcifti(
        theparcelpath
    )
    tide_io.savetocifti(
        thepdata,
        thepciftihdr,
        thepniftihdr,
        os.path.join(str(tmpdir), "dbgpseries"),
        isseries=True,
        step=0.72,
        debug=True,
    )
    tide_io.savetocifti(
        thepdata[:, 0],
        thepciftihdr,
        thepniftihdr,
        os.path.join(str(tmpdir), "dbgpscalar"),
        isseries=False,
        names=["amap"],
        debug=True,
    )
    assert tide_io.checkifcifti(thepath, debug=True)

    # savemaplist, both with and without a valid voxel list
    numverts = thedata.shape[0]
    tide_io.savemaplist(
        os.path.join(str(tmpdir), "dbgmaps"),
        [(np.arange(numverts, dtype=np.float64), "lagtimes", "map", "second", "the lag")],
        None,
        (1, numverts),
        theniftihdr,
        {},
        cifti_hdr=theciftihdr,
        filetype="cifti",
        debug=True,
    )
    thevalid = np.arange(4)
    tide_io.savemaplist(
        os.path.join(str(tmpdir), "dbgvalidmaps"),
        [(np.arange(4, dtype=np.float64), "lagtimes", "map", None, None)],
        thevalid,
        (2, 3, 4),
        tide_io.niftihdrfromarray(np.zeros((2, 3, 4))),
        {},
        filetype="nifti",
        debug=True,
    )

    # populatemap on its own
    tide_io.populatemap(np.arange(4, dtype=np.float64), 24, thevalid, np.zeros(24), debug=True)

    # the text vector readers
    theroot = os.path.join(str(tmpdir), "dbgphysio")
    tide_io.writebidstsv(
        theroot,
        np.array([np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64) * 2]),
        10.0,
        columns=["alpha", "beta"],
        extraheaderinfo={"Provenance": "debugging"},
        debug=True,
    )
    tide_io.writebidstsv(
        theroot,
        np.arange(8, dtype=np.float64) * 3,
        10.0,
        columns=["gamma"],
        append=True,
        debug=True,
    )
    tide_io.readbidstsv(theroot, debug=True)
    tide_io.readbidstsv(theroot, colspec="beta", debug=True)
    tide_io.readcolfrombidstsv(theroot, columnname="beta", debug=True)
    tide_io.readvectorsfromtextfile(theroot + ".tsv.gz", debug=True)

    theplainpath = os.path.join(str(tmpdir), "dbgplain.txt")
    np.savetxt(theplainpath, np.arange(12, dtype=np.float64).reshape((4, 3)))
    tide_io.readvecs(theplainpath, colspec="0-1", debug=True)
    tide_io.readvectorsfromtextfile(theplainpath, debug=True)
    tide_io.readtc(theplainpath, colnum=0, debug=True)

    # and the fsl design reader
    theroot2, dummy11 = _writefslmat(tmpdir, thename="dbgfsl")
    _writefsf(theroot2, [("task", 1), ("motion", 0)])
    tide_io.readfslmat(theroot2, debug=True)


# ==================== last few branches ====================


def savetonifti_writes_nifti2(tmpdir, debug=False):
    """NIFTI-2 is what an array too large for NIFTI-1 needs, and asking for it
    explicitly does produce one."""
    if debug:
        print("savetonifti_writes_nifti2")

    thedata = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    theout = os.path.join(str(tmpdir), "asnifti2")
    tide_io.savetonifti(
        thedata, tide_io.niftihdrfromarray(np.zeros((2, 3, 4))), theout, nifti2=True, debug=True
    )

    assert os.path.isfile(theout + ".nii.gz")
    thereread = nib.load(theout + ".nii.gz")
    assert isinstance(thereread, nib.Nifti2Image)
    np.testing.assert_allclose(thereread.get_fdata(), thedata)


def savetonifti_honours_a_nifti2_header(tmpdir, debug=False):
    """A NIFTI-2 header produces a NIFTI-2 file without needing nifti2=True.

    This is the read-modify-write case: a caller who reads a NIFTI-2 file, changes the
    data and writes it back should not silently get a NIFTI-1 file.  The detection has
    to compare against bytes, since nibabel keeps the magic string as
    ``array(b'n+2', dtype='|S4')`` and comparing that to the str ``"n+2"`` is always
    False - which is exactly how this used to break.
    """
    if debug:
        print("savetonifti_honours_a_nifti2_header")

    thedata = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    thenifti2header = nib.Nifti2Image(np.zeros((2, 3, 4)), np.eye(4)).header.copy()
    # the trap that made the old check useless
    assert thenifti2header["magic"] == b"n+2"
    assert not (thenifti2header["magic"] == "n+2")

    theout = os.path.join(str(tmpdir), "fromheader")
    tide_io.savetonifti(thedata, thenifti2header, theout)

    # gzipped, like every other file this module writes
    assert os.path.isfile(theout + ".nii.gz")
    assert not os.path.isfile(theout + ".nii")
    thereread = nib.load(theout + ".nii.gz")
    assert isinstance(thereread, nib.Nifti2Image)
    np.testing.assert_allclose(thereread.get_fdata(), thedata)

    # and a NIFTI-1 header still gives NIFTI-1
    thenifti1out = os.path.join(str(tmpdir), "fromnifti1header")
    tide_io.savetonifti(thedata, tide_io.niftihdrfromarray(np.zeros((2, 3, 4))), thenifti1out)
    assert isinstance(nib.load(thenifti1out + ".nii.gz"), nib.Nifti1Image)


def savetonifti_roundtrips_nifti2(tmpdir, debug=False):
    """Reading a NIFTI-2 file and writing it back keeps it NIFTI-2.

    The header comparison is done on what readfromnifti hands back, which is a
    ``.copy()`` of the loaded header rather than the object nibabel built, so this
    checks the path an actual caller takes rather than a freshly constructed header.
    """
    if debug:
        print("savetonifti_roundtrips_nifti2")

    thedata = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    thesource = os.path.join(str(tmpdir), "source_nifti2.nii.gz")
    nib.save(nib.Nifti2Image(thedata, np.eye(4)), thesource)

    dummy, theread, theheader, dummy2, dummy3 = tide_io.readfromnifti(thesource)
    theout = os.path.join(str(tmpdir), "roundtripped")
    tide_io.savetonifti(theread * 2.0, theheader, theout)

    thereread = nib.load(theout + ".nii.gz")
    assert isinstance(thereread, nib.Nifti2Image)
    np.testing.assert_allclose(thereread.get_fdata(), thedata * 2.0)


def niftisplit_5d_rejects_a_bad_axis(tmpdir, debug=False):
    """Even a 5D file has only five axes."""
    if debug:
        print("niftisplit_5d_rejects_a_bad_axis")

    thedata = np.zeros((2, 2, 2, 2, 2), dtype=np.float64)
    thepath = os.path.join(str(tmpdir), "fivedbad.nii.gz")
    nib.save(nib.Nifti1Image(thedata, np.eye(4)), thepath)

    with pytest.raises(ValueError, match="illegal axis"):
        tide_io.niftisplit(thepath, os.path.join(str(tmpdir), "nope5d"), axis=5)


def checkifcifti_on_something_with_no_nifti_header(debug=False):
    """Not every image nibabel can load has a nifti header to interrogate, and the
    absence of one means the file is certainly not a CIFTI."""
    if debug:
        print("checkifcifti_on_something_with_no_nifti_header")

    thegifti = nib.gifti.GiftiImage()
    with patch.object(nib, "load", return_value=thegifti):
        assert not tide_io.checkifcifti("whatever.gii", debug=True)


def checktimematch_can_explain_itself(capsys, debug=False):
    """The verbose form reports the two lengths, after the skips are taken off."""
    if debug:
        print("checktimematch_can_explain_itself")

    assert not tide_io.checktimematch(
        np.array([4, 10, 10, 10, 20]), np.array([4, 10, 10, 10, 25]), verbose=True
    )
    assert "numbers of timepoints do not match" in capsys.readouterr().out

    # a skip on each side can bring two different lengths back into agreement
    assert tide_io.checktimematch(
        np.array([4, 10, 10, 10, 20]), np.array([4, 10, 10, 10, 25]), numskip2=5
    )


def checkniftifilematch_notices_a_single_bad_voxel(tmpdir, capsys, debug=False):
    """One wildly wrong voxel in a large volume barely moves the mean squared error,
    so the absolute difference check is what catches it.

    This is the case the two thresholds exist to separate: without the absolute
    check, a file with a single corrupted voxel would be declared a match.
    """
    if debug:
        print("checkniftifilematch_notices_a_single_bad_voxel")

    theshape = (10, 10, 10, 4)
    thedata = np.zeros(theshape, dtype=np.float64)
    thebase = os.path.join(str(tmpdir), "absbase.nii.gz")
    nib.save(nib.Nifti1Image(thedata, np.eye(4)), thebase)

    theoutlier = thedata.copy()
    theoutlier[5, 5, 5, 2] = 1.0
    theoutlierpath = os.path.join(str(tmpdir), "absoutlier.nii.gz")
    nib.save(nib.Nifti1Image(theoutlier, np.eye(4)), theoutlierpath)

    # the mse over 4000 voxels is 2.5e-4, comfortably inside a loose mse threshold
    assert not tide_io.checkniftifilematch(thebase, theoutlierpath, absthresh=1e-3, msethresh=1e-2)
    assert "differ by at least" in capsys.readouterr().out


def readcolfrombidstsv_selects_by_number_or_name(tmpdir, debug=False):
    """Columns are 0-based, so column 0 - which is also the default - has to work.

    The range check reads ``0 <= columnnum < len(columns)``; an earlier ``0 <``
    excluded the first column, which made the parameter's own default useless and
    contradicted the docstring's ``columnnum=0`` example.  Both ends of the range are
    checked here, since that guard is where an off-by-one lands.
    """
    if debug:
        print("readcolfrombidstsv_selects_by_number_or_name")

    theroot = os.path.join(str(tmpdir), "colzero")
    thedata = np.array(
        [
            np.arange(8, dtype=np.float64),
            np.arange(8, dtype=np.float64) * 2,
            np.arange(8, dtype=np.float64) * 3,
        ]
    )
    tide_io.writebidstsv(theroot, thedata, 10.0, columns=["alpha", "beta", "gamma"])

    # the first column, by number and by the default
    dummy, dummy2, thebyzero = tide_io.readcolfrombidstsv(theroot, columnnum=0)
    np.testing.assert_allclose(thebyzero, thedata[0])
    dummy3, dummy4, thebydefault = tide_io.readcolfrombidstsv(theroot)
    np.testing.assert_allclose(thebydefault, thedata[0])

    # and by name, which is the other way to reach it
    dummy5, dummy6, thebyname = tide_io.readcolfrombidstsv(theroot, columnname="alpha")
    np.testing.assert_allclose(thebyname, thedata[0])

    # the last column is in range too
    dummy7, dummy8, thebylast = tide_io.readcolfrombidstsv(theroot, columnnum=2)
    np.testing.assert_allclose(thebylast, thedata[2])

    # one past the end is not
    assert tide_io.readcolfrombidstsv(theroot, columnnum=3) == (None, None, None)
    # nor is a negative index, which would otherwise wrap round to the last column
    assert tide_io.readcolfrombidstsv(theroot, columnnum=-1) == (None, None, None)

    # a name that is not there
    assert tide_io.readcolfrombidstsv(theroot, columnname="nosuch") == (None, None, None)

    # and a file that is not there
    assert tide_io.readcolfrombidstsv(
        os.path.join(str(tmpdir), "nosuchtsv"), neednotexist=True
    ) == (None, None, None)


def writebidstsv_append_creates_a_missing_file(tmpdir, debug=False):
    """Appending to a file that does not exist yet just creates it, so a caller can
    append unconditionally without checking first."""
    if debug:
        print("writebidstsv_append_creates_a_missing_file")

    theroot = os.path.join(str(tmpdir), "appendnew")
    tide_io.writebidstsv(
        theroot, np.arange(6, dtype=np.float64), 10.0, columns=["first"], append=True, debug=True
    )

    dummy, dummy2, thecolumns, thedata, dummy3, dummy4, dummy5 = tide_io.readbidstsv(theroot)
    assert thecolumns == ["first"]
    assert thedata.shape == (1, 6)


def processnamespec_can_be_verbose(capsys, debug=False):
    """The report is how a run records which label values a mask actually selected."""
    if debug:
        print("processnamespec_can_be_verbose")

    thename, thevals = tide_io.processnamespec(
        "aseg.nii.gz:2,7,41", "including voxels where", "is nonzero", debug=True
    )
    assert thename == "aseg.nii.gz"
    assert thevals == [2, 7, 41]
    assert "including voxels where" in capsys.readouterr().out


# ==================== entry point ====================


def test_ioextra3(tmpdir, capsys, debug=False):
    """Entry point for the third wave of sub-tests."""
    savetonifti_writes_nifti2(tmpdir, debug=debug)
    savetonifti_honours_a_nifti2_header(tmpdir, debug=debug)
    savetonifti_roundtrips_nifti2(tmpdir, debug=debug)
    niftisplit_5d_rejects_a_bad_axis(tmpdir, debug=debug)
    checkifcifti_on_something_with_no_nifti_header(debug=debug)
    checktimematch_can_explain_itself(capsys, debug=debug)
    checkniftifilematch_notices_a_single_bad_voxel(tmpdir, capsys, debug=debug)
    readcolfrombidstsv_selects_by_number_or_name(tmpdir, debug=debug)
    writebidstsv_append_creates_a_missing_file(tmpdir, debug=debug)
    processnamespec_can_be_verbose(capsys, debug=debug)


def test_ioextra2(tmpdir, capsys, debug=False):
    """Entry point for the second wave, including the sub-tests that read stdout."""
    savetonifti_rejects_an_unsupported_dtype(tmpdir, debug=debug)
    readfromnifti_finds_an_uncompressed_file(tmpdir, debug=debug)
    makeMNI_half_millimetre(debug=debug)
    niftisplit_5d(tmpdir, debug=debug)
    niftisplit_needs_data(tmpdir, debug=debug)
    niftimerge_roundtrip(tmpdir, debug=debug)
    niftiroi_5d(tmpdir, debug=debug)
    timeinfo_converts_milliseconds(tmpdir, debug=debug)
    checkniftifilematch_reports_each_kind_of_mismatch(tmpdir, capsys, debug=debug)
    checkspacedimmatch_can_explain_itself(capsys, debug=debug)
    readlabelledtsv_missing(tmpdir, debug=debug)
    readcsv_missing_and_headerless(tmpdir, capsys, debug=debug)
    readconfounds_names_unnamed_columns(tmpdir, debug=debug)
    readmotion_needs_six_columns(tmpdir, debug=debug)
    getslicetimesfromfile_rejects_a_json_without_slicetiming(tmpdir, debug=debug)
    parsefilespec_handles_a_windows_drive_letter(debug=debug)
    colspectolist_expands_a_macro(capsys, debug=debug)
    colspectolist_rejects_nonintegers(capsys, debug=debug)
    writevectorstotextfile_covers_every_format(tmpdir, debug=debug)
    io_debug_output_is_well_formed(tmpdir, debug=debug)


def test_ioextra(tmpdir, debug=False):
    """Entry point for the CIFTI, FSL and text vector coverage."""
    readfromcifti_dense(tmpdir, debug=debug)
    readfromcifti_parcellated(tmpdir, debug=debug)
    readfromcifti_finds_the_extension(tmpdir, debug=debug)
    readfromcifti_missing(tmpdir, debug=debug)
    getciftitr_reads_the_series_axis(tmpdir, debug=debug)
    getciftitr_needs_a_series_axis(tmpdir, debug=debug)
    checkifcifti_recognises_a_cifti(tmpdir, debug=debug)
    savetocifti_dense_roundtrip(tmpdir, debug=debug)
    savetocifti_dense_scalar(tmpdir, debug=debug)
    savetocifti_parcellated_roundtrip(tmpdir, debug=debug)
    savetocifti_checks_the_name_count(tmpdir, debug=debug)
    savetocifti_without_a_model_axis_raises_the_wrong_error(tmpdir, debug=debug)
    savemaplist_cifti(tmpdir, debug=debug)
    savemaplist_cifti_series(tmpdir, debug=debug)
    savemaplist_extraheaderinfo(tmpdir, debug=debug)
    savemaplist_rejects_a_malformed_entry(tmpdir, debug=debug)
    readfslmat_without_a_design_file(tmpdir, debug=debug)
    readfslmat_with_a_design_file(tmpdir, debug=debug)
    readfslmatEVlabels_refuses_a_mismatch(tmpdir, debug=debug)
    readfslmat_falls_back_when_the_labels_are_unusable(tmpdir, debug=debug)
    readfslmat_missing(tmpdir, debug=debug)
    readvectorsfromtextfile_plaintext(tmpdir, debug=debug)
    readvectorsfromtextfile_plaintext_onecol(tmpdir, debug=debug)
    readvectorsfromtextfile_plaintsv(tmpdir, debug=debug)
    readvectorsfromtextfile_plaintsv_colspec(tmpdir, debug=debug)
    readvectorsfromtextfile_bidscontinuous(tmpdir, debug=debug)
    readvectorsfromtextfile_bidscontinuous_colspec(tmpdir, debug=debug)
    readvectorsfromtextfile_csv(tmpdir, debug=debug)
    readvectorsfromtextfile_fslmat(tmpdir, debug=debug)
    niftisplit_along_time(tmpdir, debug=debug)
    niftisplit_along_other_axes(tmpdir, debug=debug)
    niftisplit_rejects_a_bad_axis(tmpdir, debug=debug)
    writebidstsv_checks_the_column_count(tmpdir, debug=debug)
    writebidstsv_refuses_an_incompatible_append(tmpdir, debug=debug)
    writebidstsv_append_preserves_extra_header_info(tmpdir, debug=debug)
    readvecs_rejects_out_of_range_columns(tmpdir, debug=debug)
    readcolfromtextfile_wants_exactly_one_column(tmpdir, debug=debug)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as thedir:
        test_ioextra(thedir, debug=True)
