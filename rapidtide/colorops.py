#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
"""Utilities to map physiological spectra into visible color representations."""

from __future__ import annotations

import colour
import numpy as np
from numpy.typing import NDArray

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


def read_cosmic_spectrum(path: str = "spectrum-Z-0.06e.txt") -> colour.SpectralDistribution:
    """Read a tabulated cosmic spectrum and return a spectral distribution.

    Parameters
    ----------
    path
        Path to a two-column text file containing wavelength and flux values.
        Wavelength values are expected in Angstrom units.

    Returns
    -------
    colour.SpectralDistribution
        Spectral distribution aligned to ``colour.SPECTRAL_SHAPE_DEFAULT``.
    """
    data = {}
    with open(path) as file:
        for line in file:
            if line.startswith("#"):
                continue

            wavelength, value = line.strip().split()
            # Current data is in angstrom and need to be converted to
            # nanometers.
            data[float(wavelength) / 10] = float(value)

    return colour.SpectralDistribution(data, name="Cosmic Spectrum at redshift 0.06").align(
        colour.SPECTRAL_SHAPE_DEFAULT
    )


def spectrumtospecdist(
    freqs: NDArray[np.float64],
    spec: NDArray[np.float64],
    lowwave: float = 660.0,
    highwave: float = 400.0,
    lowfreq: float = 0.009,
    highfreq: float = 0.15,
    debug: bool = False,
) -> colour.SpectralDistribution:
    """Map a frequency-domain spectrum onto a visible wavelength interval.

    Parameters
    ----------
    freqs
        Frequency axis values in Hz.
    spec
        Spectrum amplitudes corresponding to ``freqs``.
    lowwave
        Wavelength in nm assigned to ``lowfreq``.
    highwave
        Wavelength in nm assigned to ``highfreq``.
    lowfreq
        Lower frequency bound in Hz used for mapping.
    highfreq
        Upper frequency bound in Hz used for mapping.
    debug
        If ``True``, print detailed per-bin mapping information.

    Returns
    -------
    colour.SpectralDistribution
        Spectral distribution aligned to ``colour.SPECTRAL_SHAPE_DEFAULT``.
    """
    data = {}
    freqstep = freqs[1] - freqs[0]
    lowfreqindex = int(np.floor(lowfreq / freqstep))
    highfreqindex = int(np.ceil(highfreq / freqstep))

    theslope = (highwave - lowwave) / ((highfreq - lowfreq) / freqstep)
    intercept = lowwave - theslope * (lowfreq / freqstep)

    if debug:
        print(f"{theslope=}, {intercept=}")
    for i in range(lowfreqindex, highfreqindex + 1):
        wavelength = theslope * i + intercept
        data[wavelength] = spec[i]
        if debug:
            print(f"{freqs[i]=}, {wavelength=}, {spec[i]=}")
    return colour.SpectralDistribution(data, name="sLFO to color").align(
        colour.SPECTRAL_SHAPE_DEFAULT
    )


def plot_sd(thespectrum: colour.SpectralDistribution) -> None:
    """Plot a spectral distribution using colour-science plotting utilities.

    Parameters
    ----------
    thespectrum
        Spectral distribution to plot.

    Returns
    -------
    None
        This function is called for its plotting side effects.
    """
    colour.plotting.plot_single_sd(
        thespectrum,
        modulate_colours_with_sd_amplitude=True,
        y_label="Relative Flux / $F_\\lambda$",
    )


def spectorgb(thespectrum: colour.SpectralDistribution) -> NDArray[np.float64]:
    """Convert a spectral distribution to an sRGB triplet.

    Parameters
    ----------
    thespectrum
        Spectral distribution to convert.

    Returns
    -------
    numpy.ndarray
        A 3-element array containing sRGB values.
    """
    with colour.utilities.domain_range_scale("1"):
        XYZ = colour.sd_to_XYZ(thespectrum.align(colour.SPECTRAL_SHAPE_DEFAULT))
        RGB = colour.XYZ_to_sRGB(XYZ, illuminant=colour.CCS_ILLUMINANTS["cie_2_1931"]["E"])
    return RGB


def plot_swatch(rgb: NDArray[np.float64], label: str = "sLFO color") -> None:
    """Display an RGB swatch for a color sample.

    Parameters
    ----------
    rgb
        A 3-element RGB array.

    Returns
    -------
    None
        This function is called for its plotting side effects.
    """
    colour.plotting.plot_single_colour_swatch(
        colour.plotting.ColourSwatch(colour.algebra.normalise_maximum(rgb), label),
        text_parameters={"size": "x-large"},
    )


def main(debug: bool = False):
    thissamplerate, thisstartoffset, colnames, invec, compression, columnsource, extrainfo = (
        tide_io.readbidstsv(
            "data/examples/dst/sub-RAPIDTIDETEST_band3_desc-oversampledmovingregressor_timeseries.json",
            "pass3",
        )
    )

    freqaxis, spectrum = tide_filt.spectrum(
        tide_filt.hamming(len(invec[0])) * invec[0],
        Fs=thissamplerate,
        mode="power",
    )

    sLFO_spectrum = spectrumtospecdist(freqaxis, spectrum, debug=False)
    plot_sd(sLFO_spectrum)
    sLFO_color = spectorgb(sLFO_spectrum)
    print(f"{sLFO_color=}")
    plot_swatch(sLFO_color, label="sLFO color")


if __name__ == "__main__":
    main(debug=True)
