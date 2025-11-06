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
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT, check_NOLA, stft

import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_float, is_int, is_valid_file

NPFloat2DArray = NDArray[Union[np.float32, np.float64]]


def _get_parser() -> Any:
    """
    Argument parser for spectrogram.

    This function creates and configures an argument parser for the spectrogram
    command-line tool that computes and displays spectrograms from text files
    containing timecourse data.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional
        arguments for spectrogram computation.

    Notes
    -----
    The parser expects a text file containing one data point per line and a
    specified sample rate. The spectrogram is computed using the Short-Time
    Fourier Transform (STFT) with configurable segment length.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.txt', '44100'])
    >>> print(args.textfilename)
    'input.txt'
    >>> print(args.samplerate)
    44100

    See Also
    --------
    is_valid_file : Validates input file existence and readability
    is_float : Validates float conversion
    is_int : Validates integer conversion
    """
    parser = argparse.ArgumentParser(
        prog="spectrogram",
        description="Computes and shows the spectrogram of a text file.",
        allow_abbrev=False,
    )

    # Required argument
    parser.add_argument(
        "inputfile",
        type=lambda x: is_valid_file(parser, x),
        help="The input data file (text file containing a timecourse, one point per line).",
    )

    # Optional arguments
    parser.add_argument(
        "--nperseg",
        dest="nperseg",
        type=lambda x: is_int(parser, x),
        action="store",
        help=("The number of points to include in each spectrogram (default is 128)."),
        default=128,
    )
    parser.add_argument(
        "--samplerate",
        dest="samplerate",
        metavar="RATE",
        action="store",
        type=lambda x: is_float(parser, x),
        help="Sample rate in Hz.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    return parser


def calcspecgram(
    x: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    nperseg: int = 32,
    windowtype: str = "hann",
) -> Tuple[NDArray, NDArray, NDArray, bool]:
    """
    Make and plot a log-scaled spectrogram.

    This function computes a spectrogram using the Short-Time Fourier Transform (STFT)
    with configurable window parameters and returns frequency, time, and STFT data
    along with a check for invertibility.

    Parameters
    ----------
    x : NDArray[np.floating[Any]]
        Input signal data to compute the spectrogram for.
    time : NDArray[np.floating[Any]]
        Time vector corresponding to the input signal. Used to calculate sampling
        frequency from the time differences.
    nperseg : int, optional
        Length of each segment for the STFT, by default 32.
    windowtype : str, optional
        Type of window to use for the STFT, by default "hann".

    Returns
    -------
    freq : ndarray
        Array of frequencies corresponding to the spectrogram rows.
    segtimes : ndarray
        Array of time points corresponding to the spectrogram columns.
    thestft : ndarray
        Short-Time Fourier Transform of the input signal.
    isinvertable : bool
        Boolean flag indicating whether the window satisfies the Non-Overlap
        Additivity of Overlapped Windows (NOLA) constraint for invertibility.

    Notes
    -----
    The function calculates the sampling frequency from the time vector and uses
    the ShortTimeFFT library to compute the STFT. The window overlap is set to
    ensure maximum overlap (nperseg - 1) for optimal spectrogram resolution.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import chirp
    >>> t = np.linspace(0, 1, 1000)
    >>> x = chirp(t, f0=1, f1=10, t1=1, method='linear')
    >>> freq, times, stft, invertible = calcspecgram(x, t)
    >>> print(f"Frequency range: {freq[0]:.2f} to {freq[-1]:.2f} Hz")
    >>> print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
    """
    dt = np.diff(time)[0]  # In days...
    fs = 1.0 / dt
    noverlap = nperseg - 1

    SFT = ShortTimeFFT.from_window(
        windowtype,
        fs,
        nperseg,
        noverlap,
        mfft=nperseg,
        fft_mode="onesided",
        scale_to="magnitude",
        phase_shift=None,
    )
    freq = SFT.f
    thestft = SFT.stft(x, p0=0, p1=(len(x) - noverlap) // SFT.hop, k_offset=nperseg // 2)
    segtimes = SFT.t(len(x), p0=0, p1=(len(x) - noverlap) // SFT.hop, k_offset=nperseg // 2)

    isinvertable = check_NOLA(windowtype, nperseg, noverlap, tol=1e-10)
    return freq, segtimes, thestft, isinvertable


def showspecgram(
    thestft: Any, time: Any, freq: Any, ax: Any, fig: Any, mode: str = "mag"
) -> Tuple[Any, Any]:
    """
    Display a spectrogram plot based on the provided STFT data.

    This function visualizes the Short-Time Fourier Transform (STFT) of a signal
    in the form of a spectrogram. It supports multiple display modes including
    magnitude, phase, real, and imaginary components. The function also handles
    logarithmic scaling for amplitude values and includes a colorbar with appropriate
    labeling.

    Parameters
    ----------
    thestft : Any
        The Short-Time Fourier Transform (STFT) of the signal. Typically a 2D array
        where rows correspond to frequency bins and columns to time frames.
    time : Any
        Time vector corresponding to the STFT columns. Used for x-axis labeling.
    freq : Any
        Frequency vector corresponding to the STFT rows. Used for y-axis labeling.
    ax : Any
        Matplotlib axis object on which the spectrogram will be plotted.
    fig : Any
        Matplotlib figure object used to create the colorbar.
    mode : str, optional
        The mode of visualization. Options are:
        - "mag": Magnitude of the STFT (default)
        - "phase": Phase of the STFT
        - "real": Real part of the STFT
        - "imag": Imaginary part of the STFT

    Returns
    -------
    Tuple[Any, Any]
        A tuple containing:
        - im: The image object returned by `pcolormesh`
        - cbar: The colorbar object added to the plot

    Notes
    -----
    - For magnitude mode, the data is log-scaled using `log10`.
    - For phase mode, the data is scaled between -π and π.
    - The y-axis is set to range from `freq[1]` to `freq.max()` to avoid potential
      issues with zero frequency bins.
    - X-axis tick labels are hidden for cleaner visualization.
    - If an invalid `mode` is provided, the function will print an error and exit.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Assuming `sft` is your STFT data, `t` is time, `f` is frequency
    >>> fig, ax = plt.subplots()
    >>> im, cbar = showspecgram(sft, t, f, ax, fig, mode="mag")
    >>> plt.show()
    """
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


def make_legend_axes(ax: Any) -> object:
    """
    Create a new axes for legend placement next to the given axes.

    This function creates a new axes object positioned to the right of the
    provided axes, which can be used for placing legends. The new axes is
    created using matplotlib's make_axes_locatable utility.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The original axes object to which the legend axes will be appended.

    Returns
    -------
    matplotlib.axes.Axes
        The newly created axes object that can be used for legend placement.

    Notes
    -----
    The legend axes is positioned to the right of the original axes with:
    - Size: 0.4 inches width
    - Padding: 0.2 inches from the original axes

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> legend_ax = make_legend_axes(ax)
    >>> ax.legend(['data'], bbox_to_anchor=(1.05, 1), loc='upper left')
    >>> plt.show()
    """
    divider = make_axes_locatable(ax)
    legend_ax = divider.append_axes("right", 0.4, pad=0.2)
    return legend_ax


def ndplot(x: Any, time: Any, thelabel: Any, nperseg: int = 32) -> None:
    """
    Plot spectrogram magnitude, phase, and time-domain signal.

    This function computes and displays a spectrogram of the input signal `x` using
    the Short-Time Fourier Transform (STFT). It shows the magnitude and phase of
    the spectrogram in separate subplots, and overlays the original time-domain
    signal on a third subplot.

    Parameters
    ----------
    x : array-like
        Input signal to be analyzed.
    time : array-like
        Time vector corresponding to the signal `x`.
    thelabel : str
        Label used for plotting titles.
    nperseg : int, optional
        Length of each segment for the STFT. Default is 32.

    Returns
    -------
    None
        This function does not return any value. It displays the plot directly.

    Notes
    -----
    - The function uses `calcspecgram` to compute the STFT and checks if the
      spectrogram is invertible.
    - The magnitude and phase are plotted in the first two subplots.
    - The time-domain signal is overlaid in the third subplot.
    - The function relies on `showspecgram` for plotting the spectrograms.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 1000)
    >>> x = np.sin(2 * np.pi * 50 * t)
    >>> ndplot(x, t, "Example Signal")
    """
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


def spectrogram(args: Any) -> None:
    """
    Compute and display spectrogram of time series data from text file.

    This function reads time series data from a text file, computes its spectrogram
    using Welch's method, and displays the result. The spectrogram shows how
    the frequency content of the signal changes over time.

    Parameters
    ----------
    args : Any
        Command line arguments containing:
        - textfilename : str
            Path to the text file containing time series data
        - samplerate : float
            Sampling rate of the tidal data in Hz
        - nperseg : int, optional
            Length of each segment for FFT (default: 256)
        - debug : bool, optional
            Enable debug output (default: False)

    Returns
    -------
    None
        This function displays the spectrogram plot and does not return any value.

    Notes
    -----
    The function uses Welch's method for spectral estimation, which divides the
    signal into overlapping segments and averages their periodograms. The time
    axis is automatically calculated based on the sampling rate and data length.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(textfilename='tidal_data.txt',
    ...                          samplerate=10.0, nperseg=512, debug=False)
    >>> spectrogram(args)
    """
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise
    if args.debug:
        print(args)

    # read in data
    (
        samplerate,
        starttime,
        colnames,
        yvec,
        compressed,
        filetype,
    ) = tide_io.readvectorsfromtextfile(args.inputfile, onecol=True)

    # get the sample rate squared away
    if args.samplerate is not None:
        samplerate = args.samplerate
    else:
        if samplerate is None:
            print("Sample rate must be specified in input file or with --samplerate")
            sys.exit()
    timestep = 1.0 / samplerate

    xvec = np.arange(0.0, len(yvec), 1.0) * timestep + starttime

    thelabel = args.inputfile
    ndplot(yvec, xvec, thelabel, nperseg=args.nperseg)
    plt.show()
