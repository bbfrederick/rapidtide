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
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import check_NOLA, stft

import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_float, is_int, is_valid_file


def _get_parser():
    """
    Argument parser for spectrogram
    """
    parser = argparse.ArgumentParser(
        prog="spectrogram",
        description="Computes and shows the spectrogram of a text file.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "textfilename",
        type=lambda x: is_valid_file(parser, x),
        help="The input data file (text file containing a timecourse, one point per line).",
    )
    parser.add_argument(
        "samplerate", type=lambda x: is_float(parser, x), help="Sample rate in Hz."
    )
    parser.add_argument(
        "--nperseg",
        dest="nperseg",
        type=lambda x: is_int(parser, x),
        action="store",
        help=("The number of points to include in each spectrogram (default is 128)."),
        default=128,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    return parser


def calcspecgram(x, time, nperseg=32, windowtype="hann"):
    """Make and plot a log-scaled spectrogram"""
    dt = np.diff(time)[0]  # In days...
    fs = 1.0 / dt
    nfft = nperseg
    noverlap = nperseg - 1

    freq, segtimes, thestft = stft(
        x,
        fs=fs,
        window=windowtype,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="linear",
        return_onesided=True,
        boundary="zeros",
        padded=True,
        axis=-1,
    )

    isinvertable = check_NOLA(windowtype, nperseg, noverlap, tol=1e-10)
    return freq, segtimes, thestft, isinvertable


def showspecgram(thestft, time, freq, ax, fig, mode="mag"):
    # Log scaling for amplitude values
    if mode == "mag":
        spec_img = np.log10(np.abs(thestft))
        themax = np.max(spec_img)
        themin = themax - 3.0
    elif mode == "phase":
        spec_img = np.log10(np.angle(thestft))
        themax = np.pi
        themin = -np.pi
    elif mode == "real":
        spec_img = np.real(thestft)
        themax = np.max(spec_img)
        themin = np.min(spec_img)
    elif mode == "imag":
        spec_img = np.imag(thestft)
        themax = np.max(spec_img)
        themin = np.min(spec_img)
    else:
        print("illegal spectrogram mode:", mode)
        sys.exit()

    t = np.linspace(time.min(), time.max(), spec_img.shape[1])
    print(len(t), len(time))

    # Log scaling for frequency values (y-axis)
    # ax.set_yscale('log')

    # Plot amplitudes
    im = ax.pcolormesh(t, freq, spec_img, vmin=themin, vmax=themax)

    # Add the colorbar in a separate axis
    cax = make_legend_axes(ax)
    if mode == "mag":
        cbar = fig.colorbar(im, cax=cax, format=r"$10^{%0.1f}$")
        cbar.set_label("Magnitude", rotation=-90)
    elif mode == "phase":
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Phase", rotation=-90)
    elif mode == "real":
        cbar = fig.colorbar(im, cax=cax, format=r"$10^{%0.1f}$")
        cbar.set_label("Real amplitude", rotation=-90)
    elif mode == "imag":
        cbar = fig.colorbar(im, cax=cax, format=r"${%0.1f}$")
        cbar.set_label("Imag amplitude", rotation=-90)

    ax.set_ylim([freq[1], freq.max()])

    # Hide x-axis tick labels
    plt.setp(ax.get_xticklabels(), visible=False)

    return im, cbar


def make_legend_axes(ax):
    divider = make_axes_locatable(ax)
    legend_ax = divider.append_axes("right", 0.4, pad=0.2)
    return legend_ax


def ndplot(x, time, thelabel, nperseg=32):
    print("arrived in ndplot")
    fig = plt.figure()

    freq, segtimes, thestft, isinvertable = calcspecgram(x, time, nperseg=nperseg)
    print("Is the spectrgram invertable?", isinvertable)

    # -- Panel 1 Magnitude
    ax1 = fig.add_subplot(311)
    im1, cbar1 = showspecgram(thestft, time, freq, ax1, fig, mode="mag")
    ax1.set_ylabel("X Freq. $(Hz)$")
    ax1.set_title(thelabel)

    # -- Panel 2 Phase
    ax2 = fig.add_subplot(312, sharex=ax1)
    im2, cbar2 = showspecgram(thestft, time, freq, ax2, fig, mode="phase")
    ax2.set_ylabel("X Freq. $(Hz)$")
    ax2.set_title(thelabel)

    """# -- Panel 3 Real
    ax3 = fig.add_subplot(513, sharex=ax1)
    im3, cbar3 = showspecgram(thestft, time, freq, ax3, fig, mode='real')
    ax3.set_ylabel('X Freq. $(Hz)$')
    ax3.set_title(thelabel)

    # -- Panel 4 Imaginary
    ax4 = fig.add_subplot(514, sharex=ax1)
    im4, cbar4 = showspecgram(thestft, time, freq, ax4, fig, mode='imag')
    ax4.set_ylabel('X Freq. $(Hz)$')
    ax4.set_title(thelabel)"""

    # Make an invisible spacer...
    ax5 = fig.add_subplot(313, sharex=ax1)
    cax = make_legend_axes(ax5)
    ax5.plot(time, x, "r")
    ax5.set_xlim(time[0], time[-1])
    plt.setp(cax, visible=False)

    # Set the labels to be rotated at 20 deg and aligned left to use less space
    # plt.setp(ax5.get_xticklabels(), rotation=-20, horizontalalignment='left')

    # Remove space between subplots
    # plt.subplots_adjust(hspace=0.0)


def spectrogram(args):
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise
    if args.debug:
        print(args)

    # handle required args first
    timestep = 1.0 / args.samplerate

    yvec = tide_io.readvec(args.textfilename)
    xvec = np.arange(0.0, len(yvec), 1.0) * timestep

    thelabel = args.textfilename
    ndplot(yvec, xvec, thelabel, nperseg=args.nperseg)
    plt.show()
