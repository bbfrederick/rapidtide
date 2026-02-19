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
import colour
import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


def read_cosmic_spectrum(path='spectrum-Z-0.06e.txt'):
    data = {}
    with open(path) as file:
        for line in file:
            if line.startswith('#'):
                continue

            wavelength, value = line.strip().split()
            # Current data is in angstrom and need to be converted to
            # nanometers.
            data[float(wavelength) / 10] = float(value)

    return colour.SpectralDistribution(
        data, name='Cosmic Spectrum at redshift 0.06').align(colour.SPECTRAL_SHAPE_DEFAULT)


def spectrumtospecdist(freqs, spec, lowwave=660, highwave=400, lowfreq=0.009, highfreq=0.15, debug=False):
    data = {}
    freqstep = freqs[1] - freqs[0]
    lowfreqindex = int(np.floor(lowfreq / freqstep))
    highfreqindex = int(np.ceil(highfreq / freqstep))

    theslope = (highwave - lowwave) / (highfreq / lowfreq)
    intercept = freqs[lowfreqindex] - theslope * lowfreqindex

    print(theslope)
    for i in range(lowfreqindex, highfreqindex + 1):
        wavelength = theslope * i + intercept
        data[wavelength] = spec[i]
        if debug:
            print(f"{freqs[i]=}, {wavelength=}, {spec[i]=}")
    return colour.SpectralDistribution(
        data, name='sLFO to color').align(colour.SPECTRAL_SHAPE_DEFAULT)


def plot_sd(thespectrum):
    colour.plotting.plot_single_sd(
        thespectrum,
        modulate_colours_with_sd_amplitude=True,
        y_label='Relative Flux / $F_\\lambda$')

def plot_swatch(thespectrum):
    with colour.utilities.domain_range_scale('1'):
        XYZ = colour.sd_to_XYZ(thespectrum.align(colour.SPECTRAL_SHAPE_DEFAULT))
        RGB = colour.XYZ_to_sRGB(
            XYZ, illuminant=colour.CCS_ILLUMINANTS['cie_2_1931']['E'])

    colour.plotting.plot_single_colour_swatch(
        colour.plotting.ColourSwatch(
            colour.algebra.normalise_maximum(RGB), 'The Universe Colour'),
        text_parameters={'size': 'x-large'})

if __name__ == "__main__":
    #thespectrum = read_cosmic_spectrum(path='data/examples/src/spectrum-Z-0.06e.txt')
    #plot_sd(thespectrum)
    #plot_swatch(thespectrum)

    thissamplerate, thisstartoffset, colnames, invec, compression, columnsource, extrainfo = tide_io.readbidstsv("data/examples/dst/sub-RAPIDTIDETEST_band3_desc-oversampledmovingregressor_timeseries.json", "pass3")

    print(thissamplerate, thisstartoffset, colnames, invec, compression, columnsource, extrainfo)
    freqaxis, spectrum = tide_filt.spectrum(
        tide_filt.hamming(len(invec[0])) * invec[0],
        Fs=thissamplerate,
        mode="power",
    )
    print(freqaxis)
    print(spectrum)

    sLFO_spectrum = spectrumtospecdist(freqaxis, spectrum, debug=True)
    plot_sd(sLFO_spectrum)
    plot_swatch(sLFO_spectrum)
