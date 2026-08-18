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
"""Tests for rapidtide.colorops.

colorops maps an sLFO power spectrum onto the visible band and converts it to a
colour, so the interesting assertions are about the mapping: which frequency lands
at which wavelength, and whether the resulting RGB is a usable colour.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# colour-science is not a declared dependency of rapidtide, and colorops imports it at
# module scope, so both the module and these tests are unavailable without it.  Skip
# the file rather than fail collection on an environment that does not have it.
colour = pytest.importorskip("colour")
tide_colorops = pytest.importorskip("rapidtide.colorops")


def _writespectrumfile(thedir, thename="spectrum.txt", withcomments=True):
    """Write a two column wavelength/flux file in Angstroms.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thename : str, optional
        Filename.
    withcomments : bool, optional
        Include comment lines that the reader is expected to skip.

    Returns
    -------
    thepath : str
        Full path of the file written.
    thewavelengths : NDArray
        The wavelengths written, in Angstroms.
    """
    thepath = os.path.join(thedir, thename)
    # dense enough for colour's spline interpolation to align the distribution
    thewavelengths = np.arange(3600, 8300, 50)
    theflux = np.exp(-(((thewavelengths - 5500) / 1500.0) ** 2))
    with open(thepath, "w") as thefile:
        if withcomments:
            thefile.write("# wavelength(A) flux\n")
            thefile.write("# a second comment\n")
        for thewavelength, thevalue in zip(thewavelengths, theflux):
            thefile.write(f"{thewavelength} {thevalue}\n")
    return thepath, thewavelengths


def test_read_cosmic_spectrum_converts_angstroms_to_nanometres():
    """The file is in Angstroms and colour works in nanometres, so every wavelength
    is divided by ten.  Skipping that conversion would put the whole spectrum a
    factor of ten outside the visible band."""
    with tempfile.TemporaryDirectory() as thedir:
        thepath, thewavelengths = _writespectrumfile(thedir)
        thespectrum = tide_colorops.read_cosmic_spectrum(thepath)

    assert isinstance(thespectrum, colour.SpectralDistribution)
    # aligned to the default shape, which is the visible band in nm
    assert thespectrum.shape.start == pytest.approx(colour.SPECTRAL_SHAPE_DEFAULT.start)
    assert thespectrum.shape.end == pytest.approx(colour.SPECTRAL_SHAPE_DEFAULT.end)
    # the input spanned 360-830 nm once converted, so the visible band is covered
    assert thewavelengths[0] / 10.0 <= thespectrum.shape.start
    assert thewavelengths[-1] / 10.0 >= thespectrum.shape.end


def test_read_cosmic_spectrum_skips_comment_lines():
    """Comment lines would otherwise fail to split into two floats."""
    with tempfile.TemporaryDirectory() as thedir:
        thewithcomments, dummy = _writespectrumfile(thedir, "withcomments.txt", True)
        thewithout, dummy2 = _writespectrumfile(thedir, "without.txt", False)
        thefirst = tide_colorops.read_cosmic_spectrum(thewithcomments)
        thesecond = tide_colorops.read_cosmic_spectrum(thewithout)

    np.testing.assert_allclose(thefirst.values, thesecond.values)


def test_spectrumtospecdist_maps_frequency_onto_wavelength():
    """Low frequencies map to long wavelengths and high frequencies to short ones -
    a red to blue ramp across the requested band.  The mapping is the whole point of
    the routine, so check the endpoints land where they were asked to."""
    thefrequencies = np.linspace(0.0, 0.5, 501)
    thespectrum = np.exp(-(((thefrequencies - 0.05) / 0.02) ** 2))

    thelowwave, thehighwave = 660.0, 420.0
    theresult = tide_colorops.spectrumtospecdist(
        thefrequencies,
        thespectrum,
        lowwave=thelowwave,
        highwave=thehighwave,
        lowfreq=0.009,
        highfreq=0.15,
    )

    assert isinstance(theresult, colour.SpectralDistribution)
    # the distribution spans exactly the requested wavelength interval
    assert theresult.shape.start == pytest.approx(thehighwave)
    assert theresult.shape.end == pytest.approx(thelowwave)


def test_spectrumtospecdist_respects_a_different_band():
    """Asking for a narrower wavelength interval has to give one."""
    thefrequencies = np.linspace(0.0, 0.5, 501)
    thespectrum = np.ones_like(thefrequencies)

    theresult = tide_colorops.spectrumtospecdist(
        thefrequencies, thespectrum, lowwave=600.0, highwave=500.0, debug=True
    )
    assert theresult.shape.start == pytest.approx(500.0)
    assert theresult.shape.end == pytest.approx(600.0)


def test_spectorgb_returns_three_components():
    """The conversion goes spectrum -> XYZ -> sRGB.  Saturated spectra fall outside
    the sRGB gamut, so a negative component is expected rather than an error."""
    thefrequencies = np.linspace(0.0, 0.5, 501)
    thespectrum = np.exp(-(((thefrequencies - 0.05) / 0.02) ** 2))
    thedistribution = tide_colorops.spectrumtospecdist(thefrequencies, thespectrum)

    thergb = tide_colorops.spectorgb(thedistribution)

    assert np.asarray(thergb).shape == (3,)
    assert np.all(np.isfinite(thergb))


def test_spectorgb_tracks_the_dominant_wavelength():
    """A spectrum concentrated at the red end must come out redder than one
    concentrated at the blue end.  Without this the conversion could return any
    fixed triple and still look plausible."""
    thefrequencies = np.linspace(0.0, 0.5, 501)

    # low frequency energy maps to long wavelengths, which is the red end
    theredspectrum = np.exp(-(((thefrequencies - 0.012) / 0.004) ** 2))
    thebluespectrum = np.exp(-(((thefrequencies - 0.14) / 0.004) ** 2))

    theredrgb = tide_colorops.spectorgb(
        tide_colorops.spectrumtospecdist(thefrequencies, theredspectrum)
    )
    thebluergb = tide_colorops.spectorgb(
        tide_colorops.spectrumtospecdist(thefrequencies, thebluespectrum)
    )

    # red channel relative to blue channel, compared between the two
    assert (theredrgb[0] - theredrgb[2]) > (
        thebluergb[0] - thebluergb[2]
    ), f"the red end gave {theredrgb} and the blue end {thebluergb}"


def test_normalizergb_returns_both_an_array_and_a_hex_string():
    """normalizergb hands back a pair, not the single array its signature claims.

    The annotation says NDArray and the docstring describes one return value, but the
    function returns (normalized, hexstring).  Pinned because callers unpack two.
    """
    thergb = np.array([0.9291, 0.5895, -0.6863])
    theresult = tide_colorops.normalizergb(thergb, factor=1.0)

    assert isinstance(theresult, tuple) and len(theresult) == 2
    thenormalized, thehexstring = theresult

    # the maximum is scaled to the factor, and negatives are clipped away
    assert np.max(thenormalized) == pytest.approx(1.0)
    assert np.min(thenormalized) >= 0.0

    # the hex string is the normalized triple in 8 bit form
    assert thehexstring.startswith("#")
    assert len(thehexstring) == 7
    theexpected = "#" + "".join(f"{int(255.0 * thevalue):02x}" for thevalue in thenormalized)
    assert thehexstring == theexpected


def test_normalizergb_factor_scales_the_result():
    """The factor sets what the maximum component becomes."""
    thergb = np.array([0.8, 0.4, 0.2])
    for thefactor in (0.25, 0.5, 1.0):
        thenormalized, dummy = tide_colorops.normalizergb(thergb, factor=thefactor)
        assert np.max(thenormalized) == pytest.approx(thefactor)
        # the ratios between channels are preserved
        np.testing.assert_allclose(
            thenormalized / np.max(thenormalized), thergb / np.max(thergb), rtol=1e-6
        )


def test_normalizergb_hex_string_overflows_above_a_factor_of_one():
    """A factor above 1 pushes components past 255 and the hex field overflows.

    "%02x" of a value over 255 produces three digits, so the string comes back nine
    characters long and is not a valid colour.  Pinned rather than fixed: main() only
    ever calls this with factor=1.0, and clipping would change what any caller using
    a larger factor currently gets.
    """
    thergb = np.array([0.9291, 0.5895, -0.6863])
    dummy, thehexstring = tide_colorops.normalizergb(thergb, factor=2.0)

    assert len(thehexstring) != 7, "the overflow appears to have been fixed"
    assert thehexstring == "#1fe14300"


def test_the_plotting_helpers_run(monkeypatch):
    """plot_sd and plot_swatch are called for their side effects; they still have to
    accept what the rest of the module produces."""
    import matplotlib

    matplotlib.use("Agg")

    thecalls = []
    monkeypatch.setattr(colour.plotting, "plot_single_sd", lambda *a, **k: thecalls.append("sd"))
    monkeypatch.setattr(
        colour.plotting, "plot_single_colour_swatch", lambda *a, **k: thecalls.append("swatch")
    )

    thefrequencies = np.linspace(0.0, 0.5, 501)
    thespectrum = np.exp(-(((thefrequencies - 0.05) / 0.02) ** 2))
    thedistribution = tide_colorops.spectrumtospecdist(thefrequencies, thespectrum)

    tide_colorops.plot_sd(thedistribution)
    tide_colorops.plot_sd(thedistribution, modulate=False)
    tide_colorops.plot_swatch(tide_colorops.spectorgb(thedistribution))

    assert thecalls == ["sd", "sd", "swatch"]


if __name__ == "__main__":
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
