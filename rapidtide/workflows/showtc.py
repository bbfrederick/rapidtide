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
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import sys

import matplotlib.cm as cm
import numpy as np
from matplotlib.pyplot import figure, savefig, setp, show

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


def phase(mcv):
    return np.arctan2(mcv.imag, mcv.real)


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="showtc",
        description="Plots the data in text files.",
        usage="%(prog)s texfilename[:col1,col2...,coln] [textfilename]... [options]",
    )

    parser.add_argument(
        "textfilenames",
        type=str,
        nargs="+",
        help="One or more input files, with optional column specifications",
    )

    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        metavar="FREQ",
        type=lambda x: pf.is_float(parser, x),
        help=(
            "Set the sample rate of the data file to FREQ. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    sampling.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        metavar="TSTEP",
        type=lambda x: pf.invert_float(parser, x),
        help=(
            "Set the sample rate of the data file to 1.0/TSTEP. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )

    parser.add_argument(
        "--displaytype",
        dest="displaymode",
        action="store",
        type=str,
        choices=["time", "power", "phase"],
        help=("Display data as time series (default), power spectrum, or phase spectrum."),
        default="time",
    )
    parser.add_argument(
        "--format",
        dest="plotformat",
        action="store",
        type=str,
        choices=["overlaid", "separate", "separatelinked"],
        help=(
            "Display data overlaid (default), in individually scaled windows, or in separate windows with linked scaling."
        ),
        default="overlaid",
    )
    parser.add_argument(
        "--waterfall",
        action="store_true",
        dest="dowaterfall",
        help="Display multiple timecourses in a waterfall plot.",
        default=False,
    )
    parser.add_argument(
        "--voffset",
        dest="voffset",
        metavar="OFFSET",
        type=float,
        action="store",
        help="Plot multiple timecourses with OFFSET between them (use negative OFFSET to set automatically).",
        default=0.0,
    )

    parser.add_argument(
        "--transpose",
        action="store_true",
        dest="dotranspose",
        help="Swap rows and columns in the input files.",
        default=False,
    )

    # add plot appearance options
    pf.addplotopts(parser)

    parser.add_argument(
        "--starttime",
        dest="thestarttime",
        metavar="START",
        type=float,
        help="Start plotting at START seconds (default is the start of the data).",
        default=None,
    )
    parser.add_argument(
        "--endtime",
        dest="theendtime",
        metavar="END",
        type=float,
        help="Finish plotting at END seconds (default is the end of the data).",
        default=None,
    )
    parser.add_argument(
        "--numskip",
        dest="numskip",
        metavar="NUM",
        type=int,
        help="Skip NUM lines at the beginning of each file (to get past header lines).",
        default=0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Output additional debugging information.",
        default=False,
    )

    return parser


def showtc(args):
    # set the sample rate
    if args.samplerate == "auto":
        samplerate = 1.0
        args.samplerate = samplerate
    else:
        samplerate = args.samplerate

    # set the appropriate display mode
    if args.displaymode == "time":
        dospectrum = False
        specmode = "power"
    elif args.displaymode == "power":
        dospectrum = True
        specmode = "power"
    elif args.displaymode == "phase":
        dospectrum = True
        specmode = "phase"
    else:
        print("illegal display mode")
        sys.exit()

    # determine how to composite multiple plots
    if args.plotformat == "overlaid":
        separate = False
        linky = True
    elif args.plotformat == "separate":
        separate = True
        linky = False
    elif args.plotformat == "separatelinked":
        separate = True
        linky = True
    else:
        print("illegal formatting mode")
        sys.exit()

    # set various cosmetic aspects of the plots
    if args.colors is not None:
        colornames = args.colors.split(",")
    else:
        colornames = []

    if args.legends is not None:
        legends = args.legends.split(",")
        legendset = True
    else:
        legends = []
        legendset = False
    dolegend = args.dolegend

    if args.linewidths is not None:
        thelinewidth = []
        for thestring in args.linewidths.split(","):
            thelinewidth.append(float(thestring))
    else:
        thelinewidth = [1.0]
    numlinewidths = len(thelinewidth)

    if 0 <= args.legendloc <= 10:
        legendloc = args.legendloc
    else:
        print("illegal legend location:", args.legendloc)
        sys.exit()

    savespec = False
    detrendorder = 1
    demean = False
    useHamming = True

    # check range
    if args.theendtime is None:
        args.theendtime = 100000000.0
    if args.thestarttime is not None:
        if args.thestarttime >= args.theendtime:
            print("endtime must be greater then starttime;")
            sys.exit()

    # handle required args first
    xvecs = []
    yvecs = []
    linelabels = []
    samplerates = []
    numvecs = 0

    minlen = 100000000
    shortcolnames = True
    # read in all the data
    for i in range(0, len(args.textfilenames)):
        thisfilename, thiscolspec = tide_io.parsefilespec(args.textfilenames[i])

        # check file type
        (
            thissamplerate,
            thisstartoffset,
            colnames,
            invecs,
            dummy,
            dummy,
        ) = tide_io.readvectorsfromtextfile(args.textfilenames[i], debug=args.debug)

        if args.debug:
            print("On return from readvectorsfromtextfile:")
            print(f"\targs.samplerate: {args.samplerate}")
            print(f"\tthissamplerate: {thissamplerate}")
            print(f"\targs.thestarttime: {args.thestarttime}")
            print(f"\tthisstartoffset: {thisstartoffset}")
            print("input data dimensions:", invecs.shape)

        if thissamplerate is None:
            thissamplerate = samplerate

        if thisstartoffset is None:
            # print("thisstartoffset is None")
            if args.thestarttime is None:
                if args.debug:
                    print("args.thestarttime is None")
                args.thestarttime = 0.0
            else:
                if args.debug:
                    print(f"args.thestarttime is {args.thestarttime}")
            thisstartoffset = args.thestarttime
        else:
            # print(f"thisstartoffset is {thisstartoffset}")
            if args.thestarttime is None:
                if args.debug:
                    print("args.thestarttime is None")
                args.thestarttime = thisstartoffset
            else:
                if args.debug:
                    print(f"args.thestarttime is {args.thestarttime}")
                thisstartoffset = args.thestarttime

        if args.debug:
            print("After preprocessing time variables:")
            print(f"\targs.samplerate: {args.samplerate}")
            print(f"\tthissamplerate: {thissamplerate}")
            print(f"\targs.thestarttime: {args.thestarttime}")
            print(f"\tthisstartoffset: {thisstartoffset}")

        if args.debug:
            print(f"file {args.textfilenames[i]} colnames: {colnames}")

        if args.dotranspose:
            invecs = np.transpose(invecs)
        if args.debug:
            print("   ", invecs.shape[0], " columns")

        for j in range(0, invecs.shape[0]):
            if args.debug:
                print("appending vector number ", j)
            if dospectrum:
                if invecs.shape[1] % 2 == 1:
                    invec = invecs[j, :-1]
                else:
                    invec = invecs[j, :]
                if detrendorder > 0:
                    invec = tide_fit.detrend(invec, order=detrendorder, demean=True)
                elif demean:
                    invec = invec - np.mean(invec)

                if useHamming:
                    freqaxis, spectrum = tide_filt.spectrum(
                        tide_filt.hamming(len(invec)) * invec, Fs=thissamplerate, mode=specmode,
                    )
                else:
                    freqaxis, spectrum = tide_filt.spectrum(
                        invec, Fs=thissamplerate, mode=specmode
                    )
                if savespec:
                    tide_io.writenpvecs(
                        np.transpose(np.stack([freqaxis, spectrum], axis=1)), "thespectrum.txt",
                    )
                xvecs.append(freqaxis)
                yvecs.append(spectrum)
            else:
                yvecs.append(invecs[j] * 1.0)
                xvecs.append(
                    thisstartoffset + np.arange(0.0, len(yvecs[-1]), 1.0) / thissamplerate
                )
            if len(yvecs[-1]) < minlen:
                minlen = len(yvecs[-1])
            if not legendset:
                if invecs.shape[0] > 1:
                    if colnames is None:
                        if shortcolnames:
                            linelabels.append("column" + str(j).zfill(2))
                        else:
                            linelabels.append(thisfilename + "_column" + str(j).zfill(2))

                    else:
                        if shortcolnames:
                            linelabels.append(colnames[j])
                        else:
                            linelabels.append(thisfilename + "_" + colnames[j])
                else:
                    linelabels.append(thisfilename)
            else:
                linelabels.append(legends[i % len(legends)])
                """if invecs.shape[0] > 1:
                    linelabels.append(legends[i % len(legends)] + '_column' + str(j).zfill(2))
                else:
                    linelabels.append(legends[i % len(legends)])"""
            samplerates.append(thissamplerate + 0.0)
            if args.debug:
                print(
                    "timecourse:",
                    j,
                    ", len:",
                    len(xvecs[-1]),
                    ", timerange:",
                    xvecs[-1][0],
                    xvecs[-1][-1],
                )
            numvecs += 1

    thestartpoint = tide_util.valtoindex(xvecs[0], args.thestarttime)
    theendpoint = tide_util.valtoindex(xvecs[0], args.theendtime)
    args.thestarttime = xvecs[0][thestartpoint]
    args.theendtime = xvecs[0][theendpoint]
    if args.debug:
        print("full range (pts):", thestartpoint, theendpoint)
        print("full range (time):", args.thestarttime, args.theendtime)
    overallxmax = -1e38
    overallxmin = 1e38
    for thevec in xvecs:
        overallxmax = np.max([np.max(thevec), overallxmax])
        overallxmin = np.min([np.min(thevec), overallxmin])
    xrange = (np.max([overallxmin, args.thestarttime]), np.min([overallxmax, args.theendtime]))
    ymins = []
    ymaxs = []
    for thevec in yvecs:
        ymins.append(np.min(np.asarray(thevec[thestartpoint:theendpoint], dtype="float")))
        ymaxs.append(np.max(np.asarray(thevec[thestartpoint:theendpoint], dtype="float")))
    overallymax = -1e38
    overallymin = 1e38
    for thevec in yvecs:
        overallymax = np.max([np.max(thevec), overallymax])
        overallymin = np.min([np.min(thevec), overallymin])
    yrange = (overallymin, overallymax)
    if args.debug:
        print("xrange:", xrange)
        print("yrange:", yrange)
    if args.voffset < 0.0:
        args.voffset = yrange[1] - yrange[0]
    if args.debug:
        print("voffset:", args.voffset)
    if not separate:
        for i in range(0, numvecs):
            yvecs[i] += (numvecs - i - 1) * args.voffset
        overallymax = -1e38
        overallymin = 1e38
        for thevec in yvecs:
            overallymax = np.max([np.max(thevec), overallymax])
            overallymin = np.min([np.min(thevec), overallymin])
        yrange = (overallymin, overallymax)

        if args.dowaterfall:
            xstep = (xrange[1] - xrange[0]) / numvecs
            ystep = yrange[1] - yrange[0]
            for i in range(numvecs):
                xvecs[i] = xvecs[i] + i * xstep
                yvecs[i] = 10.0 * yvecs[i] / ystep + i * ystep

    # now plot it out
    if separate:
        thexaxfontsize = 6 * args.fontscalefac
        theyaxfontsize = 6 * args.fontscalefac
        thexlabelfontsize = 6 * args.fontscalefac
        theylabelfontsize = 6 * args.fontscalefac
        thelegendfontsize = 5 * args.fontscalefac
        thetitlefontsize = 6 * args.fontscalefac
        thesuptitlefontsize = 10 * args.fontscalefac
    else:
        thexaxfontsize = 10 * args.fontscalefac
        theyaxfontsize = 10 * args.fontscalefac
        thexlabelfontsize = 10 * args.fontscalefac
        theylabelfontsize = 10 * args.fontscalefac
        thelegendfontsize = 8 * args.fontscalefac
        thetitlefontsize = 10 * args.fontscalefac
        thesuptitlefontsize = 10 * args.fontscalefac

    if len(colornames) > 0:
        colorlist = [colornames[i % len(colornames)] for i in range(numvecs)]
    else:
        colorlist = [cm.nipy_spectral(float(i) / numvecs) for i in range(numvecs)]

    fig = figure()
    if separate:
        if args.thetitle is not None:
            fig.suptitle(args.thetitle, fontsize=thesuptitlefontsize)
        if linky:
            axlist = fig.subplots(numvecs, sharex=True, sharey=True)[:]
        else:
            axlist = fig.subplots(numvecs, sharex=True, sharey=False)[:]
    else:
        ax = fig.add_subplot(1, 1, 1)
        if args.thetitle is not None:
            ax.set_title(args.thetitle, fontsize=thetitlefontsize)

    for i in range(0, numvecs):
        if separate:
            ax = axlist[i]
        ax.plot(
            xvecs[i],
            yvecs[i],
            color=colorlist[i],
            label=linelabels[i],
            linewidth=thelinewidth[i % numlinewidths],
        )
        if dolegend:
            ax.legend(fontsize=thelegendfontsize, loc=legendloc)
        ax.set_xlim(xrange)
        if linky:
            # print(yrange)
            ax.set_ylim(yrange)
        else:
            themax = np.max(yvecs[i])
            themin = np.min(yvecs[i])
            thediff = themax - themin
            # print(themin, themax, thediff)
            ax.set_ylim(top=(themax + thediff / 20.0), bottom=(themin - thediff / 20.0))
        if args.showxax:
            ax.tick_params(axis="x", labelsize=thexlabelfontsize, which="both")
        if args.showyax:
            ax.tick_params(axis="y", labelsize=theylabelfontsize, which="both")

        if separate:
            fig.subplots_adjust(hspace=0)
            setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        if dospectrum:
            if args.xlabel is None:
                args.xlabel = "Frequency (Hz)"
            if specmode == "power":
                if args.ylabel is None:
                    args.ylabel = "Signal power"
            else:
                if args.ylabel is None:
                    args.ylabel = "Signal phase"
        else:
            if args.xlabel is None:
                args.xlabel = "Time (s)"
        if args.showxax:
            ax.set_xlabel(args.xlabel, fontsize=thexlabelfontsize, fontweight="bold")
        else:
            ax.xaxis.set_visible(False)
        if args.showyax:
            ax.set_ylabel(args.ylabel, fontsize=theylabelfontsize, fontweight="bold")
        else:
            ax.yaxis.set_visible(False)

    # fig.tight_layout()

    if args.outputfile is None:
        show()
    else:
        savefig(args.outputfile, bbox_inches="tight", dpi=args.saveres)
