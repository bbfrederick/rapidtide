#!/usr/bin/env python
#
#   Copyright 2016 Blaise Frederick
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
# $Author: frederic $
# $Date: 2016/06/10 20:27:34 $
# $Id: tide_funcs.py,v 1.1 2016/06/10 20:27:34 frederic Exp $
#
from __future__ import print_function, division

import numpy as np
import scipy as sp
from scipy import fftpack
import pylab as pl
import warnings
import time
import sys
import bisect

from scipy import signal

#from memory_profiler import profile

try:
    from numba import jit
    numbaexists = True
except ImportError:
    numbaexists = False

try:
    import nibabel as nib

    nibabelexists = True
except ImportError:
    nibabelexists = False

def conditionaljit():
    def resdec(f):
        if not numbaexists:
            return f
        return jit(f)
    return resdec


# ---------------------------------------- Global constants -------------------------------------------

# ---------------------------------------- Debugging/profiling functions ------------------------------
#@profile(precision=4)
def checkpoint1():
    pass

#@profile(precision=4)
def checkpoint2():
    pass

#@profile(precision=4)
def checkpoint3():
    pass

#@profile(precision=4)
def checkpoint4():
    pass

#@profile(precision=4)
def checkpoint5():
    pass

#@profile(precision=4)
def checkpoint6():
    pass

# ---------------------------------------- Global defaults ----------------------------------
defaultbutterorder = 3
MAXLINES = 10000000

THRESHFRAC = 0.1

VLF_UPPERPASS = 0.009
VLF_UPPERSTOP = 0.010

LF_LOWERSTOP = VLF_UPPERPASS
LF_LOWERPASS = VLF_UPPERSTOP
LF_UPPERPASS = 0.15
LF_UPPERSTOP = 0.20

RESP_LOWERSTOP = LF_UPPERPASS
RESP_LOWERPASS = LF_UPPERSTOP
RESP_UPPERPASS = 0.4
RESP_UPPERSTOP = 0.5

CARD_LOWERSTOP = RESP_UPPERPASS
CARD_LOWERPASS = RESP_UPPERSTOP
CARD_UPPERPASS = 2.5
CARD_UPPERSTOP = 3.0

def version():
    return '$Id: tide_funcs.py,v 1.1 2016/06/10 20:27:34 frederic Exp $'


# ---------------------------------------- NIFTI file manipulation ---------------------------
if nibabelexists:
    def readfromnifti(inputfile):
        nim = nib.load(inputfile)
        nim_data = nim.get_data()
        nim_hdr = nim.get_header()
        thedims = nim_hdr['dim']
        thesizes = nim_hdr['pixdim']
        return nim, nim_data, nim_hdr, thedims, thesizes


    def parseniftidims(thedims):
        return thedims[1], thedims[2], thedims[3], thedims[4]


    def parseniftisizes(thesizes):
        return thesizes[1], thesizes[2], thesizes[3], thesizes[4]


    def savetonifti(thearray, theheader, thepixdim, thename):
        outputaffine = theheader.get_best_affine()
        qaffine, qcode = theheader.get_qform(coded=True)
        saffine, scode = theheader.get_sform(coded=True)
        if theheader['magic']=='n+2':
            output_nifti = nib.Nifti2Image(thearray, outputaffine, header=theheader)
            suffix='.nii'
        else:
            output_nifti = nib.Nifti1Image(thearray, outputaffine, header=theheader)
            suffix='.nii.gz'
        output_nifti.set_qform(qaffine, code=int(qcode))
        output_nifti.set_sform(saffine, code=int(scode))
        output_nifti.to_filename(thename+suffix)


    def checkifnifti(filename):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            return True
        else:
            return False


    def fmritimeinfo(niftifilename):
        nim = nib.load(niftifilename)
        hdr = nim.get_header()
        thedims = hdr['dim']
        thesizes = hdr['pixdim']
        if hdr.get_xyzt_units()[1] == 'msec':
            tr = thesizes[4]/1000.0
        else:
            tr = thesizes[4]
        timepoints = thedims[4]
        return tr, timepoints


    def checkspacematch(dims1, dims2):
        for i in range(1, 4):
            if dims1[i] != dims2[i]:
                print("File spatial voxels do not match")
                print("dimension ", i, ":", dims1[i], "!=", dims2[i])
                return False
            else:
                return True


    def checktimematch(dims1, dims2, numskip1, numskip2):
        if (dims1[4] - numskip1) != (dims2[4] - numskip2):
            print("File numbers of timepoints do not match")
            print("dimension ", 4, ":", dims1[4],
                  "(skip ", numskip1, ") !=",
                  dims2[4],
                  " (skip ", numskip2, ")")
            return False
        else:
            return True


# --------------------------- timing functions -------------------------------------------------
def timefmt(thenumber):
    return "{:10.2f}".format(thenumber)


def proctiminginfo(thetimings, outputfile='', extraheader=None):
    theinfolist = []
    start = thetimings[0]
    starttime = float(start[1])
    lasteventtime = starttime
    if extraheader is not None:
        print(extraheader)
        theinfolist.append(extraheader)
    headerstring = 'Clock time\tProgram time\tDuration\tDescription'
    print(headerstring)
    theinfolist.append(headerstring)
    for theevent in thetimings:
        theduration = float(theevent[1] - lasteventtime)
        outstring = time.strftime("%Y%m%dT%H%M%S", time.localtime(theevent[1])) + \
                    timefmt(float(theevent[1]) - starttime) + \
                    '\t' + timefmt(theduration) + '\t' + theevent[0]
        if theevent[2] is not None:
            outstring += " ({0:.2f} {1}/second)".format(float(theevent[2]) / theduration, theevent[3])
        print(outstring)
        theinfolist.append(outstring)
        lasteventtime = float(theevent[1])
    if outputfile != '':
        writevec(theinfolist, outputfile)


# --------------------------- histogram functions -------------------------------------------------
def gethistprops(indata, histlen, refine=False, therange=None):
    thestore = np.zeros((2, histlen))
    if therange is None:
        thehist = np.histogram(indata, histlen)
    else:
        thehist = np.histogram(indata, histlen, therange)
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (thestore[1, peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = gaussfit(peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :])
    return peaklag, peakheight, peakwidth


def makehistogram(indata, histlen, therange=None):
    if therange is None:
        thehist = np.histogram(indata, histlen)
    else:
        thehist = np.histogram(indata, histlen, therange)
    return thehist


def makeandsavehistogram(indata, histlen, endtrim, outname,
                         displaytitle='histogram', displayplots=False,
                         refine=False, therange=None):
    thestore = np.zeros((2, histlen))
    thehist = makehistogram(indata, histlen, therange)
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    # get starting values for the peak, ignoring first and last point of histogram
    peakindex = np.argmax(thestore[1, 1:-2])
    peaklag = thestore[0, peakindex + 1]
    # peakheight = max(thestore[1,:])
    peakheight = thestore[1, peakindex + 1]
    numbins = 1
    while (peakindex + numbins < histlen - 1) and (thestore[1, peakindex + numbins] > peakheight / 2.0):
        numbins += 1
    peakwidth = (thestore[0, peakindex + numbins] - thestore[0, peakindex]) * 2.0
    if refine:
        peakheight, peaklag, peakwidth = gaussfit(peakheight, peaklag, peakwidth, thestore[0, :], thestore[1, :])
    writenpvecs(np.array([peaklag]), outname + '_peak.txt')
    writenpvecs(thestore, outname + '.txt')
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        pl.plot(thestore[0, :(-1 - endtrim)], thestore[1, :(-1 - endtrim)])


# --------------------------- probability functions -------------------------------------------------
def printthresholds(pcts, thepercentiles, labeltext):
    print(labeltext)
    for i in range(0, len(pcts)):
        print('\tp <', "{:1.4f}".format(1.0 - thepercentiles[i]), ': ', pcts[i])


def fitpdf(thehist, histlen, endtrim, thedata, displayplots=False):
    thestore = np.zeros((2, histlen))
    thestore[0, :] = thehist[1][-histlen:]
    thestore[1, :] = thehist[0][-histlen:]
    theamp = max(thestore[1, :])
    themean = thedata.mean()
    thestd = thedata.std()
    theskew = sp.stats.stats.skew(thedata)
    # print('initial histogram stats:', theamp, themean, thestd, theskew)
    thefit = gaussfitsk(theamp, themean, thestd, theskew, thestore[0, :], thestore[1, :])
    # print('final histogram stats:', thefit[0], thefit[1], thefit[2], thefit[3])
    if displayplots:
        displaytitle = 'histogram fit to skewed normal distribution'
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title(displaytitle)
        pl.plot(thestore[0, :(-1 - endtrim)], thestore[1, :(-1 - endtrim)])
        pl.plot(thestore[0, :(-1 - endtrim)], gausssk_eval(thestore[0, :(-1 - endtrim)], thefit))
        pl.show()
    return thefit


def sigFromDistributionData(vallist, histlen, thepercentiles, displayplots=False, twotail=False, nozero=False):
    thehistogram = makehistogram(vallist, histlen)
    histfit = fitpdf(thehistogram, histlen, 0, vallist, displayplots=displayplots)
    pcts_data = getfracvals(vallist, thepercentiles, numbins=int(np.sqrt(len(vallist)) * 5.0), nozero=nozero)
    pcts_fit = getfracvalsfromfit(histfit, thepercentiles, numbins=100000)
    return pcts_data, pcts_fit


def tfromr(r, nsamps, dfcorrfac=1.0, oversampfactor=1.0, returnp=False):
    if r >= 1.0:
        tval = float("inf")
        pval = 0.0
    else:
        dof = (dfcorrfac * nsamps) / oversampfactor
        tval = r * np.sqrt(dof / (1 - r * r))
        pval = sp.stats.t.sf(abs(tval), dof) * 2.0
    if returnp:
        return tval, pval
    else:
        return tval


def zfromr(r, nsamps, dfcorrfac=1.0, oversampfactor=1.0, returnp=False):
    if r >= 1.0:
        zval = float("inf")
        pval = 0.0
    else:
        dof = (dfcorrfac * nsamps) / oversampfactor
        zval = r / np.sqrt(1.0 / (dof - 3))
        pval = 1.0 - sp.stats.norm.cdf(abs(zval))
    if returnp:
        return zval, pval
    else:
        return zval


def fisher(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def symmetrize(a, antisymmetric=False, zerodiagonal=False):
    if antisymmetric:
        intermediate = (a - a.T) / 2.0
    else:
        intermediate = (a + a.T) / 2.0
    if zerodiagonal:
        return intermediate - np.diag(intermediate.diagonal())
    else:
        return intermediate


### I don't remember where this came from.  Need to check license
def mlregress(x, y, intercept=True):
    """Return the coefficients from a multiple linear regression, along with R, the coefficient of determination.

    x: The independent variables (pxn or nxp).
    y: The dependent variable (1xn or nx1).
    intercept: Specifies whether or not the slope intercept should be considered.

    The routine computes the coefficients (b_0, b_1, ..., b_p) from the data (x,y) under
    the assumption that y = b0 + b_1 * x_1 + b_2 * x_2 + ... + b_p * x_p.

    If intercept is False, the routine assumes that b0 = 0 and returns (b_1, b_2, ..., b_p).
    """

    warnings.filterwarnings("ignore", "invalid*")
    y = np.atleast_1d(y)
    n = y.shape[0]

    x = np.atleast_2d(x)
    p, nx = x.shape

    if nx != n:
        x = x.transpose()
        p, nx = x.shape
        if nx != n:
            raise AttributeError('x and y must have have the same number of samples (%d and %d)' % (nx, n))

    if intercept is True:
        xc = np.vstack((np.ones(n), x))
        beta = np.ones(p + 1)
    else:
        xc = x
        beta = np.ones(p)

    solution = np.linalg.lstsq(np.mat(xc).T, np.mat(y).T)

    # Computation of the coefficient of determination.
    Rx = np.atleast_2d(np.corrcoef(x, rowvar=1))
    c = np.corrcoef(x, y, rowvar=1)[-1, :p]
    R2 = np.dot(np.dot(c, np.linalg.inv(Rx)), c.T)
    R = np.sqrt(R2)

    return np.atleast_1d(solution[0].T), R


# --------------------------- non-NIFTI file I/O functions ------------------------------------------
def checkifparfile(filename):
    if filename.endswith(".par"):
        return True
    else:
        return False


def readvecs(inputfilename):
    thefile = open(inputfilename, 'rU')
    lines = thefile.readlines()
    numvecs = len(lines[0].split())
    inputvec = np.zeros((numvecs, MAXLINES))
    numvals = 0
    for line in lines:
        numvals += 1
        thetokens = line.split()
        for vecnum in range(0, numvecs):
            inputvec[vecnum, numvals - 1] = float(thetokens[vecnum])
    return 1.0 * inputvec[:, 0:numvals]


def readvec(inputfilename):
    inputvec = np.zeros(MAXLINES)
    numvals = 0
    with open(inputfilename, 'rU') as thefile:
        lines = thefile.readlines()
        for line in lines:
            numvals += 1
            inputvec[numvals - 1] = float(line)
    return 1.0 * inputvec[0:numvals]


def writedict(thedict, outputfile, lineend=''):
    if lineend == 'mac':
        thelineending = '\r'
        openmode = 'wb'
    elif lineend == 'win':
        thelineending = '\r\n'
        openmode = 'wb'
    elif lineend == 'linux':
        thelineending = '\n'
        openmode = 'wb'
    else:
        thelineending = '\n'
        openmode = 'w'
    with open(outputfile, openmode) as FILE:
        for key, value in sorted(thedict.items()):
            FILE.writelines(str(key) + ':\t' + str(value) + thelineending)


def writevec(thevec, outputfile, lineend=''):
    if lineend == 'mac':
        thelineending = '\r'
        openmode = 'wb'
    elif lineend == 'win':
        thelineending = '\r\n'
        openmode = 'wb'
    elif lineend == 'linux':
        thelineending = '\n'
        openmode = 'wb'
    else:
        thelineending = '\n'
        openmode = 'w'
    with open(outputfile, openmode) as FILE:
        for i in thevec:
            FILE.writelines(str(i) + thelineending)


# rewritten to guarantee file closure, combines writenpvec and writenpvecs
def writenpvecs(thevecs, outputfile, lineend=''):
    theshape = np.shape(thevecs)
    if lineend == 'mac':
        thelineending = '\r'
        openmode = 'wb'
    elif lineend == 'win':
        thelineending = '\r\n'
        openmode = 'wb'
    elif lineend == 'linux':
        thelineending = '\n'
        openmode = 'wb'
    else:
        thelineending = '\n'
        openmode = 'w'
    with open(outputfile, openmode) as FILE:
        if len(theshape) == 2:
            for i in range(0, theshape[1]):
                for j in range(0, theshape[0]):
                    FILE.writelines(str(thevecs[j, i]) + '\t')
                FILE.writelines(thelineending)
        else:
            for i in range(0, theshape[0]):
                FILE.writelines(str(thevecs[i]) + thelineending)


# --------------------------- correlation functions -------------------------------------------------
def quickcorr(data1, data2):
    thepcorr = sp.stats.stats.pearsonr(corrnormalize(data1, True, True), corrnormalize(data2, True, True))
    return thepcorr


def shorttermcorr_1D(data1, data2, sampletime, windowtime, prewindow=False, dodetrend=False):
    windowsize = windowtime // sampletime
    halfwindow = int((windowsize + 1) // 2)
    corrpertime = data1 * 0.0
    ppertime = data1 * 0.0
    for i in range(halfwindow, len(data1) - halfwindow):
        dataseg1 = corrnormalize(data1[i - halfwindow:i + halfwindow], prewindow, dodetrend)
        dataseg2 = corrnormalize(data2[i - halfwindow:i + halfwindow], prewindow, dodetrend)
        thepcorr = sp.stats.stats.pearsonr(dataseg1, dataseg2)
        corrpertime[i] = thepcorr[0]
        ppertime[i] = thepcorr[1]
    return corrpertime, ppertime


def shorttermcorr_2D(data1, data2, sampletime, windowtime, prewindow=False, dodetrend=False, display=False):
    windowsize = windowtime // sampletime
    halfwindow = int((windowsize + 1) // 2)

    dataseg1 = corrnormalize(data1[0:2 * halfwindow], prewindow, dodetrend)
    dataseg2 = corrnormalize(data2[0:2 * halfwindow], prewindow, dodetrend)
    thexcorr = fastcorrelate(dataseg1, dataseg2)
    xcorrlen = len(thexcorr)
    xcorr_x = np.r_[0.0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    corrzero = xcorrlen // 2
    xcorrpertime = np.zeros((xcorrlen, len(data1)))
    Rvals = np.zeros((len(data1)))
    valid = np.zeros((len(data1)))
    delayvals = np.zeros((len(data1)))
    for i in range(halfwindow, len(data1) - halfwindow):
        dataseg1 = corrnormalize(data1[i - halfwindow:i + halfwindow], prewindow, dodetrend)
        dataseg2 = corrnormalize(data2[i - halfwindow:i + halfwindow], prewindow, dodetrend)
        xcorrpertime[:, i] = fastcorrelate(dataseg1, dataseg2)
        maxindex, delayvals[i], Rvals[i], maxsigma, maskval, failreason = findmaxlag(
            xcorr_x,
            xcorrpertime[:,i],
            -windowtime/2.0,windowtime/2.0,1000.0,
            refine=True,
            useguess=False,
            fastgauss=False,
            displayplots=False)
        if failreason == 0:
            valid[i] = 1
    if display:
        pl.imshow(xcorrpertime)
    return xcorrpertime, Rvals, delayvals, valid


def eckart(input1, input2, doplot=False):
    g1 = eckartraw(input1, input1, doplot=False)
    g2 = eckartraw(input2, input2, doplot=False)
    # normfac = 2.0*np.sqrt(np.sqrt(g1[len(g1) / 2 + 1] * g2[len(g2) / 2 + 1]))
    normfac = 2.0 * np.sqrt(np.sqrt(max(g1) * max(g2)))
    print("normfac=", normfac)
    # normfac = g1[len(g1) / 2 + 1] * g2[len(g2) / 2 + 1]
    return gccphatraw(input1, input2, doplot=doplot) / normfac


def eckartraw(input1, input2, doplot=False):
    fft1 = fftpack.fft(input1)
    fft2 = fftpack.fft(input2)
    G12 = fft1 * np.conj(fft2)
    G11 = fft1 * np.conj(fft1)
    G22 = fft2 * np.conj(fft2)
    denom = G11 * G22
    absdenom = abs(denom)
    print("max(abs(denom))", max(absdenom))
    thresh = max(absdenom) * THRESHFRAC
    print("thresh", thresh)
    G = np.where(absdenom > thresh, G12 / denom, 0.0)
    g = np.real(fftpack.fftshift(fftpack.ifft(G)))
    # g=np.real(fftpack.ifft(G))
    if doplot:
        xvec = range(0, len(fft1))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('fft1')
        pl.plot(xvec, abs(fft1))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('fft2')
        pl.plot(xvec, abs(fft2))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('G12')
        pl.plot(xvec, abs(G12))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('G11')
        pl.plot(xvec, abs(G11))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('G22')
        pl.plot(xvec, abs(G22))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('denom')
        pl.plot(xvec, abs(denom))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('G')
        pl.plot(xvec, abs(G))
    return g


# http://stackoverflow.com/questions/12323959/fast-cross-correlation-method-in-python
def fastcorrelate(input1, input2, usefft=True):
    if usefft:
        # Do an array flipped convolution, which is a correlation.
        return sp.signal.fftconvolve(input1, input2[::-1], mode='full')
    else:
        return np.correlate(input1, input2, mode='full')


def gccphat(input1, input2, doplot=False):
    g1 = gccphatraw(input1, input1, doplot=False)
    g2 = gccphatraw(input2, input2, doplot=False)
    normfac = np.sqrt(g1[len(g1) / 2 + 1] * g2[len(g2) / 2 + 1])
    return gccphatraw(input1, input2, doplot=doplot) / normfac


def gccphatraw(input1, input2, doplot):
    fft1 = fftpack.fft(input1)
    fft2 = fftpack.fft(input2)
    G12 = fft1 * np.conjugate(fft2)
    # denom=G12.real
    denom = G12
    absdenom = abs(denom)
    thresh = max(abs(denom)) * THRESHFRAC
    G = np.where(absdenom > thresh, G12 / absdenom, 0.0)
    g = np.real(fftpack.fftshift(fftpack.ifft(G)))
    if doplot:
        xvec = range(0, len(fft1))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('fft1')
        pl.plot(xvec, abs(fft1))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('fft2')
        pl.plot(xvec, abs(fft2))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('G12')
        pl.plot(xvec, abs(G12))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('denom')
        pl.plot(xvec, abs(denom))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('angle(G)')
        pl.plot(xvec, np.angle(G))

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('abs(G)')
        pl.plot(xvec, abs(G))
    return g


#### taken from filtfilt from scipy.org Cookbook http://www.scipy.org/Cookbook/FiltFilt
def lfilter_zi(b, a):
    # compute the zi state from the filter parameters. see [Gust96].

    # Based on:
    # [Gust96] Fredrik Gustafsson, Determining the initial states in forward-backward 
    # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996, 
    # Volume 44, Issue 4

    n = max(len(a), len(b))

    zin = (np.eye(n - 1) - np.hstack((-a[1:n, np.newaxis],
                                      np.vstack((np.eye(n - 2), np.zeros(n - 2))))))

    zid = b[1:n] - a[1:n] * b[0]

    zi_matrix = np.linalg.inv(zin) * (np.matrix(zid).transpose())
    zi_return = []

    # convert the result into a regular array (not a matrix)
    for i in range(len(zi_matrix)):
        zi_return.append(float(zi_matrix[i][0]))

    return np.array(zi_return)


#### adapted from filtfilt from scipy.org Cookbook http://www.scipy.org/Cookbook/FiltFilt
def fastfiltfiltinit(b, a, x):
    # For now only accepting 1d arrays
    ntaps = max(len(a), len(b))
    edge = ntaps * 10

    if x.ndim != 1:
        raise ValueError("filtfilt is only accepting 1 dimension arrays.")

    # x must be bigger than edge
    if x.size < edge:
        raise ValueError("Input vector needs to be bigger than 3 * max(len(a),len(b).")

    if len(a) < ntaps:
        a = np.r_[a, np.zeros(len(b) - len(a))]

    if len(b) < ntaps:
        b = np.r_[b, np.zeros(len(a) - len(b))]

    zi = sp.signal.lfilter_zi(b, a)

    return b, a, zi, edge


#### adapted from filtfilt from scipy.org Cookbook http://www.scipy.org/Cookbook/FiltFilt
def fastfiltfilt(b, a, zi, edge, x):
    # Grow the signal to have edges for stabilizing
    # the filter with inverted replicas of the signal
    s = np.r_[2 * x[0] - x[edge:1:-1], x, 2 * x[-1] - x[-1:-edge:-1]]
    # in the case of one go we only need one of the extrems
    # both are needed for filtfilt

    (y, zf) = sp.signal.lfilter(b, a, s, -1, zi * s[0])

    (y, zf) = sp.signal.lfilter(b, a, np.flipud(y), -1, zi * y[-1])

    return np.flipud(y[edge - 1:-edge + 1])


def gaussresidualssk(p, y, x):
    err = y - gausssk_eval(x, p)
    return err


def gaussskresiduals(p, y, x):
    return y - gausssk_eval(x, p)


@conditionaljit()
def gaussresiduals(p, y, x):
    return y - gauss_eval(x, p)


def risetimeresiduals(p, y, x):
    return y - risetime_eval_loop(x, p)


def thederiv(y):
    dyc = [0.0] * len(y)
    dyc[0] = (y[0] - y[1]) / 2.0
    for i in range(1, len(y) - 1):
        dyc[i] = (y[i + 1] - y[i - 1]) / 2.0
    dyc[-1] = (y[-1] - y[-2]) / 2.0
    return dyc


def gausssk_eval(x, p):
    t = (x - p[1]) / p[2]
    return p[0] * sp.stats.norm.pdf(t) * sp.stats.norm.cdf(p[3] * t)

@conditionaljit()
def gauss_eval(x, p):
    return p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))


def risetime_eval_loop(x, p):
    r = np.zeros(len(x))
    for i in range(0, len(x)):
        r[i] = risetime_eval(x[i], p)
    return r


@conditionaljit()
def risetime_eval(x, p):
    corrx = x - p[0]
    if corrx < 0.0:
        return 0.0
    else:
        return p[1] * (1.0 - np.exp(-corrx/p[2]))


# Find the image intensity value which thefrac of the non-zero voxels in the image exceed
def getfracval(datamat, thefrac, numbins=200):
    themax = datamat.max()
    themin = datamat.min()
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    cummeanhist = np.cumsum(meanhist)
    target = cummeanhist[numbins - 1] * thefrac
    for i in range(0, numbins):
        if cummeanhist[i] >= target:
            return bins[i]
    return 0.0


def getfracvals(datamat, thefracs, numbins=200, displayplots=False, nozero=False):
    themax = datamat.max()
    themin = datamat.min()
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    cummeanhist = np.cumsum(meanhist)
    if nozero:
        cummeanhist = cummeanhist - cummeanhist[0]
    thevals = []
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('cumulative mean sum of histogram')
        plot(bins[-numbins:], cummeanhist[-numbins:])
        pl.show()
    for thisfrac in thefracs:
        target = cummeanhist[numbins - 1] * thisfrac
        thevals.append(0.0)
        for i in range(0, numbins):
            if cummeanhist[i] >= target:
                thevals[-1] = bins[i]
                break
    return thevals


def getfracvalsfromfit(histfit, thefracs, numbins=2000, displayplots=False):
    themax = 1.0
    themin = -1.0
    bins = np.arange(themin, themax, (themax - themin) / numbins)
    meanhist = gausssk_eval(bins, histfit)
    cummeanhist = np.cumsum(meanhist)
    thevals = []
    if displayplots:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('cumulative mean sum of histogram')
        pl.plot(bins[-numbins:], cummeanhist[-numbins:])
        pl.show()
    for thisfrac in thefracs:
        target = cummeanhist[numbins - 1] * thisfrac
        thevals.append(0.0)
        for i in range(0, numbins):
            if cummeanhist[i] >= target:
                thevals[-1] = bins[i]
                break
    return thevals


def makemask(image, threshpct=25.0, verbose=False):
    fracval = getfracval(image, 0.98)
    threshval = (threshpct / 100.0) * fracval
    if verbose:
        print('fracval:', fracval, ' threshpct:', threshpct, ' mask threshhold:', threshval)
    themask = np.where(image > threshval, 1.0, 0.0)
    return themask


def makelaglist(lagstart, lagend, lagstep):
    numsteps = np.floor((lagend - lagstart) / lagstep) + 1
    lagend = lagstart + lagstep * (numsteps - 1)
    print("creating list of ", numsteps, " lag steps (", lagstart, " to ", lagend, " in steps of ", lagstep, ")")
    thelags = np.r_[0.0:1.0 * numsteps] * lagstep + lagstart
    return thelags


# --------------------------- Fitting functions -------------------------------------------------
def locpeak(data, samplerate, lastpeaktime, winsizeinsecs=5.0, thresh=0.75, hysteresissecs=0.4):
    # look at a limited time window
    winsizeinsecs = 5.0
    numpoints = int(winsizeinsecs * samplerate)
    startpoint = max((0, len(data) - numpoints))
    currenttime = (len(data) - 1) / samplerate

    # find normative limits
    recentmax = max(data[startpoint:])
    recentmin = min(data[startpoint:])
    recentrange = recentmax - recentmin

    # screen out obvious non-peaks
    if data[-1] < recentmin + recentrange * thresh:
        # print(currenttime,'below thresh')
        return -1.0
    if currenttime - lastpeaktime < hysteresissecs:
        # print(currenttime,'too soon')
        return -1.0

    # now check to see if we have just passed a peak
    if data[-1] < data[-2]:
        if data[-2] <= data[-3]:
            fitstart = -5
            fitdata = data[fitstart:]
            X = currenttime + (np.arange(0.0, len(fitdata)) - len(fitdata) + 1.0) / samplerate
            maxtime = sum(X * fitdata) / sum(fitdata)
            maxsigma = np.sqrt(abs(sum((X - maxtime) ** 2 * fitdata) / sum(fitdata)))
            maxval = fitdata.max()
            peakheight, peakloc, peakwidth = gaussfit(maxval, maxtime, maxsigma, X, fitdata)
            # print(currenttime,fitdata,X,peakloc)
            return peakloc
        else:
            # print(currenttime,'not less enough')
            return -1.0
    else:
        # print(currenttime,'not less')
        return -1.0


# generate the polynomial fit timecourse from the coefficients
@conditionaljit()
def trendgen(thexvals, thefitcoffs, demean):
    theshape = thefitcoffs.shape
    order = theshape[0] - 1
    thepoly = thexvals
    thefit = 0.0 * thexvals
    if order > 0:
        for i in range(1, order + 1):
            thefit = thefit + thefitcoffs[order - i] * thepoly
            thepoly = np.multiply(thepoly, thexvals)
    if demean:
        thefit = thefit + thefitcoffs[order]
    return thefit

@conditionaljit()
def detrend(inputdata, order=1, demean=False):
    thetimepoints = np.arange(0.0, len(inputdata), 1.0)
    thecoffs = np.polyfit(thetimepoints, inputdata, order)
    thefittc = trendgen(thetimepoints, thecoffs, demean)
    return inputdata - thefittc


@conditionaljit()
def findfirstabove(theyvals, thevalue):
    for i in range(0, len(theyvals)):
        if theyvals[i] >= thevalue:
            return i
    return i


def findrisetimefunc(thexvals, theyvals, initguess=None, debug=False, 
    minrise=0.0, maxrise=200.0, minstart=-100.0, maxstart=100.0, refine=False, displayplots=False):
    # guess at parameters: risestart, riseamplitude, risetime
    if initguess is None:
        initstart = 0.0
        initamp = np.mean(theyvals[-10:-1])
        initrisetime = 5.0
    else:
        initstart = initguess[0]
        initamp = initguess[1]
        initrisetime = initguess[2]

    p0 = np.array([initstart, initamp, initrisetime])
    if debug:
        for i in range(0,len(theyvals)):
            print(thexvals[i], theyvals[i])
    plsq, dummy = sp.optimize.leastsq(risetimeresiduals, p0, args=(theyvals, thexvals), maxfev=5000)
    #except ValueError:
    #    return 0.0, 0.0, 0.0, 0
    if (minrise <= plsq[2] <= maxrise) and (minstart <= plsq[0] <= maxstart):
        return plsq[0], plsq[1], plsq[2], 1
    else:
        return 0.0, 0.0, 0.0, 0
        

@conditionaljit()
def findmaxlag(thexcorr_x, thexcorr_y, lagmin, lagmax, widthlimit, edgebufferfrac=0.0, threshval=0.0, uthreshval=30.0,
               debug=False, refine=False, maxguess=0.0, useguess=False, fastgauss=False, enforcethresh=True, displayplots=False):
    # set initial parameters 
    # widthlimit is in seconds
    # maxsigma is in Hz
    # maxlag is in seconds
    warnings.filterwarnings("ignore", "Number*")
    maskval = 1
    maxval = 0.0
    maxlag = 0.0
    maxsigma = 0.0
    failreason = 0
    numlagbins = len(thexcorr_y)
    binwidth = thexcorr_x[1] - thexcorr_x[0]
    searchbins = int(widthlimit / binwidth)
    lowerlim = int(numlagbins * edgebufferfrac)
    upperlim = numlagbins - lowerlim - 1
    FML_BADAMP = 0x01
    FML_BADLAG = 0x02
    FML_BADWIDTH = 0x04
    FML_HITEDGE = 0x08
    FML_FITFAIL = 0x0f

    # make an initial guess at the fit parameters for the gaussian
    # start with finding the maximum value
    if useguess:
        nlowerlim = int(maxguess - widthlimit / 2.0)
        nupperlim = int(maxguess + widthlimit / 2.0)
        if nlowerlim < lowerlim:
            nlowerlim = lowerlim
            nupperlim = lowerlim + int(widthlimit)
        if nupperlim > upperlim:
            nlowerlim = upperlim
            nlowerlim = upperlim - int(widthlimit)
        maxindex = np.argmax(thexcorr_y[nlowerlim:nupperlim]) + nlowerlim
        maxval_init = 1.0 * max(thexcorr_y[nlowerlim:nupperlim])
    else:
        maxindex = np.argmax(thexcorr_y[lowerlim:upperlim]) + lowerlim
        maxval_init = thexcorr_y[maxindex]

    # now get a location for that value
    maxlag_init = 1.0 * thexcorr_x[maxindex]

    # and calculate the width of the peak
    maxsigma_init = 0.0
    upperlimit = len(thexcorr_y) - 1
    lowerlimit = 0
    i = 0
    j = 0
    searchfrac=0.5
    while (maxindex + i <= upperlimit) and (thexcorr_y[maxindex + i] > searchfrac * maxval_init) and (i < searchbins):
        i += 1
    i -= 1
    while (maxindex - j >= lowerlimit) and (thexcorr_y[maxindex - j] > searchfrac * maxval_init) and (j < searchbins):
        j += 1
    j -= 1
    maxsigma_init = (2.0 * searchfrac) * 2.0 * (i + j + 1) * binwidth / 2.355
    if (debug and (maxval_init != 0.0)) or displayplots:
        print("maxval_init=", maxval_init, "maxindex=", maxindex, "maxlag_init=", maxlag_init, "maxsigma_init=",
              maxsigma_init, "maskval=", maskval, lagmin, lagmax, widthlimit, threshval)
        print(maxlag_init, lagmax, "if 1 gt 2 reject")
        print(maxlag_init, lagmin, "if 1 lt 2 reject")
        print(maxsigma_init, widthlimit, "if 1 gt 2 reject")
        print(maxval_init, threshval, "if 2 gt 1 reject")

    # now check the values for errors and refine if necessary
    if not ((lagmin + binwidth) <= maxlag_init <= (lagmax - binwidth)):
        failreason += FML_HITEDGE
    if not (binwidth / 2.355 < maxsigma_init < widthlimit):
        failreason += FML_BADWIDTH
    if (maxval_init < threshval) and enforcethresh:
        failreason += FML_BADAMP
    #if (not ((lagmin + binwidth) <= maxlag_init <= (lagmax - binwidth)) or \
    #            not (binwidth / 2.355 < maxsigma_init < widthlimit) or \
    #                maxval_init < threshval) and enforcethresh:
    if failreason > 0:
        maxlag = 0.0
        maskval = 0.0
        maxsigma = 0.0
        maskval = 0
    else:
        if refine:
            fitend = min(maxindex + i + 1, upperlimit)
            fitstart = max(1, maxindex - j)
            data = thexcorr_y[fitstart:fitend]
            X = thexcorr_x[fitstart:fitend]
            if fastgauss:
                # do a non-iterative fit over the top of the peak
                # 6/12/2015  This is just broken.  Gives quantized maxima
                maxlag = 1.0 * sum(X * data) / sum(data)
                maxsigma = np.sqrt(abs(sum((X - maxlag) ** 2 * data) / sum(data)))
                maxval = data.max()
            else:
                # do a least squares fit over the top of the peak
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init])
                if fitend - fitstart >= 3:
                    plsq, dummy = sp.optimize.leastsq(gaussresiduals, p0,
                                          args=(data,X), maxfev=5000)
                    maxval = 1.0 * plsq[0]
                    maxlag = 1.0 * plsq[1]
                    maxsigma = 1.0 * plsq[2]
                else:
                    maxval = 0.0
        else:
            maxval = maxval_init
            maxlag = maxlag_init
            maxsigma = maxsigma_init
        #if (maxval == 0.0) or (maxlag >= lagmax) or (maxlag <= lagmin):
        if maxval == 0.0:
            failreason += FML_FITFAIL
        if not (lagmin <= maxlag <= lagmax):
            failreason += FML_BADLAG
        if failreason > 0:
            maxval = 0.0
            maskval = 0
            maxlag = 0.0
            maxsigma = 0.0
    if debug or displayplots:
        print("init to final: maxval", maxval_init, maxval, ", maxlag:", maxlag_init, maxlag, ", width:", maxsigma_init,
              maxsigma)
    if displayplots and refine and (maskval != 0.0):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Data and fit')
        hiresx=np.arange(X[0],X[-1],(X[1]-X[0])/10.0)
        pl.plot(X, data, 'ro', hiresx, gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])), 'b-')
        pl.show()
    return maxindex, maxlag, maxval, maxsigma, maskval, failreason


def gaussfitsk(height, loc, width, skewness, xvals, yvals):
    plsq, dummy = sp.optimize.leastsq(gaussresidualssk, np.array([height, loc, width, skewness]),
                          args=(yvals, xvals), maxfev=5000)
    return plsq


def gaussfit(height, loc, width, xvals, yvals):
    plsq, dummy = sp.optimize.leastsq(gaussresiduals, np.array([height, loc, width]), args=(yvals, xvals), maxfev=5000)
    return plsq[0], plsq[1], plsq[2]


# --------------------------- Resampling and time shifting functions -------------------------------------------
class fastresampler:
    def __init__(self, timeaxis, timecourse, padvalue=30.0, upsampleratio=100, doplot=False):
        #print('initializing fastresampler with padvalue =',padvalue)
        self.upsampleratio = upsampleratio
        self.initstep = timeaxis[1] - timeaxis[0]
        self.hiresstep = self.initstep / self.upsampleratio
        self.hires_x = np.r_[timeaxis[0]-padvalue:self.initstep * len(timeaxis) + padvalue:self.hiresstep]
        self.hiresstart = self.hires_x[0]
        self.hires_y = doresample(timeaxis, timecourse, self.hires_x, method='univariate')
        self.hires_y[:int(padvalue // self.hiresstep)] = 0.0
        self.hires_y[-int(padvalue // self.hiresstep):] = 0.0
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('fastresampler initial timecourses')
            pl.plot(timeaxis, timecourse, self.hires_x, self.hires_y)
            pl.legend(('input', 'hires'))
            pl.show()
    
    def yfromx(self, newtimeaxis, doplot=False):
        outindices = ((newtimeaxis -  self.hiresstart) // self.hiresstep).astype(int)
        try:
            out_y = self.hires_y[outindices]
        except IndexError:
            print('')
            print('indexing out of bounds in fastresampler')
            print('    hirestart,hiresstep,hiresend:',self.hiresstart,self.hiresstep,self.hires_x[-1])
            print('    requested axis limits:',newtimeaxis[0],newtimeaxis[-1])
            sys.exit()
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('fastresampler timecourses')
            pl.plot(self.hires_x, self.hires_y, newtimeaxis, out_y)
            pl.legend(('hires', 'output'))
            pl.show()
        return 1.0 * out_y
        

def prepforfastresample(orig_x, orig_y, numtrs, fmritr, padvalue, upsampleratio, doplot=False):
    hiresstep = fmritr / upsampleratio
    hires_x_padded = np.r_[-padvalue:fmritr * numtrs + padvalue:hiresstep]
    hiresstart = hires_x_padded[0]
    hires_y = doresample(orig_x, orig_y, hires_x_padded, method='univariate')
    hires_y[:padvalue//hiresstep] = 0.0
    hires_y[-padvalue//hiresstep:] = 0.0
    if doplot:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial resampled vector')
        pl.plot(hires_x_padded, hires_y)
        pl.show()
    return hires_x_padded, hires_y, hiresstep, hiresstart


#@profile(precision=4)
def dofastresample(orig_x, orig_y, new_x, hrstep, hrstart, upsampleratio):
    starthrindex = int((new_x[0] - hrstart) / hrstep)
    stride = int(upsampleratio)
    endhrindex = starthrindex + stride * len(new_x) - 1
    return 1.0 * orig_y[starthrindex:endhrindex:stride]


def doresample(orig_x, orig_y, new_x, method='cubic'):
    if method == 'cubic':
        cj = sp.signal.cspline1d(orig_y)
        return sp.signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])
    elif method == 'quadratic':
        qj = sp.signal.qspline1d(orig_y)
        return sp.signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])
    elif method == 'univariate':
        interpolator = sp.interpolate.UnivariateSpline(orig_x, orig_y, k=3, s=0)  # s=0 interpolates
        return interpolator(new_x)
    else:
        print('invalid interpolation method')
        return None


def dotwostepresample(orig_x, orig_y, intermed_freq, final_freq, method='univariate'):
    if intermed_freq <= final_freq:
        print('intermediate frequency must be higher than final frequency')
        sys.exit()

    # upsample
    endpoint = orig_x[-1] - orig_x[0]
    intermed_ts = 1.0 / intermed_freq
    numresamppts = np.floor(endpoint / intermed_ts) + 1
    intermed_x = np.arange(0.0, intermed_ts * numresamppts, intermed_ts)
    intermed_y = doresample(orig_x, orig_y, intermed_x, method=method)

    # antialias
    aafilter = noncausalfilter(type='arb', usebutterworth=True)
    aafilter.setarb(0.0, 0.0, 0.95 * final_freq, final_freq)
    antialias_y = aafilter.apply(intermed_freq, intermed_y)
    # antialias_y = dolptrapfftfilt(intermed_freq,0.9*final_freq,final_freq,intermed_y)

    # downsample
    final_ts = 1.0 / final_freq
    numresamppts = np.ceil(endpoint / final_ts) + 1
    final_x = np.arange(0.0, final_ts * numresamppts, final_ts)
    return doresample(intermed_x, antialias_y, final_x, method=method)


def calcsliceoffset(sotype, slicenum, numslices, tr, multiband=1):
    # Slice timing correction
    # 0 : None
    # 1 : Regular up (0, 1, 2, 3, ...)
    # 2 : Regular down
    # 3 : Use slice order file
    # 4 : Use slice timings file
    # 5 : Standard Interleaved (0, 2, 4 ... 1, 3, 5 ... )
    # 6 : Siemens Interleaved (0, 2, 4 ... 1, 3, 5 ... for odd number of slices)
    # (1, 3, 5 ... 0, 2, 4 ... for even number of slices)
    # 7 : Siemens Multiband Interleaved

    # default value of zero
    slicetime = 0.0

    # None
    if sotype == 0:
        slicetime = 0.0

    # Regular up
    if type == 1:
        slicetime = slicenum * (tr / numslices)

    # Regular down
    if sotype == 2:
        slicetime = (numslices - slicenum - 1) * (tr / numslices)

    # Slice order file not supported - do nothing
    if sotype == 3:
        slicetime = 0.0

    # Slice timing file not supported - do nothing
    if sotype == 4:
        slicetime = 0.0

    # Standard interleave
    if sotype == 5:
        if (slicenum % 2) == 0:
            # even slice number
            slicetime = (tr / numslices) * (slicenum / 2)
        else:
            # odd slice number
            slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens interleave format
    if sotype == 6:
        if (numslices % 2) == 0:
            # even number of slices - slices go 1,3,5,...,0,2,4,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (numslices / 2 + slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((slicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens multiband interleave format
    if sotype == 7:
        numberofshots = numslices / multiband
        modslicenum = slicenum % numberofshots
        if (numberofshots % 2) == 0:
            # even number of shots - slices go 1,3,5,...,0,2,4,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (numberofshots / 2 + modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((modslicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((numberofshots + 1) / 2 + (modslicenum - 1) / 2)
    return slicetime


# NB: a positive value of shifttrs delays the signal, a negative value advances it
# timeshift using fourier phase multiplication
def timeshift(inputtc, shifttrs, padtrs, doplot=False):
    # set up useful parameters
    thelen = len(inputtc)
    thepaddedlen = thelen + 2 * padtrs
    imag = 1.j

    # initialize variables
    preshifted_y = np.zeros(thepaddedlen)  # initialize the working buffer (with pad)
    weights = np.zeros(thepaddedlen)  # initialize the weight buffer (with pad)

    # now do the math
    preshifted_y[padtrs:padtrs + thelen] = inputtc[:]  # copy initial data into shift buffer
    weights[padtrs:padtrs + thelen] = 1.0  # put in the weight vector
    revtc = inputtc[::-1]  # reflect data around ends to
    preshifted_y[0:padtrs] = revtc[-padtrs:]  # eliminate discontinuities
    preshifted_y[padtrs + thelen:] = revtc[0:padtrs]

    # finish initializations
    osfac = 8
    fftlen = len(preshifted_y)
    osfftlen = osfac * fftlen

    # create the phase modulation timecourse
    initargvec = (np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(osfftlen)) - np.pi)
    argvec = np.roll(initargvec * osfac * shifttrs, -int(osfftlen / 2))
    modvec = np.cos(argvec) - imag * np.sin(argvec)

    # process the data (fft->oversample->modulate->ifft->filter->downsample)
    fftdata = fftpack.fft(preshifted_y)  # do the actual shifting
    osfftdata = (1.0 + imag) * np.zeros(fftlen * osfac)
    osfftdata[0:int(fftlen // 2)] = fftdata[0:int(fftlen // 2)]
    osfftdata[-int(fftlen // 2):] = fftdata[-int(fftlen // 2):]
    shifted_y = fftpack.ifft(modvec * osfftdata).real
    butterorder=4
    filt_shifted_y = dolpfiltfilt(2.0 * osfac, 1.0, shifted_y, butterorder)
    ds_shifted_y = filt_shifted_y[::osfac] * osfac

    # process the weights
    w_fftdata = fftpack.fft(weights)  # do the actual shifting
    w_osfftdata = (1.0 + imag) * np.zeros(fftlen * osfac)
    w_osfftdata[0:int(fftlen // 2)] = w_fftdata[0:int(fftlen // 2)]
    w_osfftdata[-int(fftlen // 2):] = w_fftdata[-int(fftlen // 2):]
    shifted_weights = fftpack.ifft(modvec * w_osfftdata).real
    filt_shifted_weights = dolpfiltfilt(2.0 * osfac, 1.0, shifted_weights, butterorder)
    ds_shifted_weights = filt_shifted_weights[::osfac] * osfac

    if doplot:
        xvec = range(0, thepaddedlen)  # make a ramp vector (with pad)
        print("shifttrs:", shifttrs)
        print("offset:", padtrs)
        print("thelen:", thelen)
        print("thepaddedlen:", thepaddedlen)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial vector')
        pl.plot(xvec, preshifted_y)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial and shifted vector')
        pl.plot(xvec, preshifted_y, xvec, ds_shifted_y)

        pl.show()

    return ([ds_shifted_y[padtrs:padtrs + thelen], ds_shifted_weights[padtrs:padtrs + thelen], ds_shifted_y,
             ds_shifted_weights])


# timeshift using direct resampling
def timeshift2(inputtc, shifttrs, padtrs, doplot=False, dopostfilter=False):
    # set up useful parameters
    thelen = len(inputtc)
    thepaddedlen = thelen + 2 * padtrs
    offset = padtrs

    # initialize the postfilter
    theringfilter = noncausalfilter(type='ringstop')

    # initialize variables
    preshifted_y = np.zeros(thepaddedlen)  # initialize the working buffer (with pad)
    weights = np.zeros(thepaddedlen)  # initialize the weight buffer (with pad)

    # now do the math
    preshifted_x = np.arange(0.0, len(preshifted_y), 1.0)
    shifted_x = preshifted_x - shifttrs
    preshifted_y[offset:offset + thelen] = inputtc[:]  # copy initial data into shift buffer
    revtc = inputtc[::-1]
    preshifted_y[0:offset] = revtc[-offset:]
    preshifted_y[offset + thelen:] = revtc[0:offset]
    weights[offset:offset + thelen] = 1.0  # put in the weight vector
    shifted_y = doresample(preshifted_x, preshifted_y, shifted_x, method='univariate')  # do the actual shifting
    shifted_weights = doresample(preshifted_x, weights, shifted_x, method='univariate')  # do the actual shifting
    if dopostfilter:
        shifted_y = theringfilter.apply(1.0, shifted_y)
        shifted_weights = theringfilter.apply(1.0, shifted_weights)

    if doplot:
        print("shifttrs:", shifttrs)
        print("offset:", offset)
        print("thelen:", thelen)
        print("thepaddedlen:", thepaddedlen)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial vector')
        pl.plot(preshifted_x, preshifted_y)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Shifted vector')
        pl.plot(shifted_x, shifted_y)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial and shifted vector')
        pl.plot(preshifted_x, preshifted_y, shifted_x, shifted_y)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial and shifted weight vector')
        pl.plot(preshifted_x, weights, shifted_x, shifted_weights)

        pl.show()

    return [shifted_y[offset:offset + thelen], shifted_weights[offset:offset + thelen], shifted_y, shifted_weights]


# --------------------------- Window functions -------------------------------------------------
def blackmanharris(length):
    argvec = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(length))
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    return a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)


def hann(length):
    return 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))


def hamming(length):
    return 0.54 - 0.46 * np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi)


def envdetect(vector, filtwidth=3.0):
    demeaned = vector - np.mean(vector)
    sigabs = abs(demeaned)
    return dolptrapfftfilt(1.0, 1.0 / (2.0 * filtwidth), 1.1 / (2.0 * filtwidth), sigabs)


# --------------------------- Normalization functions -------------------------------------------------
def znormalize(vector):
    return stdnormalize(vector)


@conditionaljit()
def stdnormalize(vector):
    demeaned = vector - np.mean(vector)
    sigstd = np.std(demeaned)
    if sigstd > 0.0:
        return demeaned / sigstd
    else:
        return demeaned


def pcnormalize(vector):
    sigmean = np.mean(vector)
    if sigmean > 0.0:
        return vector / sigmean - 1.0
    else:
        return vector


def ppnormalize(vector):
    demeaned = vector - np.mean(vector)
    sigpp = max(demeaned) - min(demeaned)
    if sigpp > 0.0:
        return demeaned / sigpp
    else:
        return demeaned

@conditionaljit()
def corrnormalize(thedata, prewindow, dodetrend):
    # detrend first
    if dodetrend:
        intervec = stdnormalize(detrend(thedata, demean=True))
    else:
        intervec = stdnormalize(thedata)

    # then window
    if prewindow:
        return stdnormalize(hamming(len(thedata)) * intervec) / np.sqrt(len(thedata))
    else:
        return stdnormalize(intervec) / np.sqrt(len(thedata))


# --------------------------- Filtering functions -------------------------------------------------
# NB: No automatic padding for precalculated filters

def padvec(indata, padlen=20):
    return np.concatenate((indata[::-1][-padlen:], indata, indata[::-1][0:padlen]))


def unpadvec(indata, padlen=20):
    return indata[padlen:-padlen]


def ssmooth(xsize, ysize, zsize, sigma, thedata):
    return sp.ndimage.gaussian_filter(thedata, [sigma / xsize, sigma / ysize, sigma / zsize])


# - butterworth filters
def dolpfiltfilt(samplefreq, cutofffreq, indata, order, padlen=20):
    if cutofffreq > samplefreq/2.0:
        cutofffreq = samplefreq/2.0
    [b, a] = sp.signal.butter(order, 2.0 * cutofffreq / samplefreq)
    return unpadvec(sp.signal.filtfilt(b, a, padvec(indata, padlen=padlen)).real, padlen=padlen)


def dohpfiltfilt(samplefreq, cutofffreq, indata, order, padlen=20):
    if cutofffreq < 0.0:
        cutofffreq = 0.0
    [b, a] = sp.signal.butter(order, 2.0 * cutofffreq / samplefreq, 'highpass')
    return unpadvec(sp.signal.filtfilt(b, a, padvec(indata, padlen=padlen)).real, padlen=padlen)


def dobpfiltfilt(samplefreq, cutofffreq_low, cutofffreq_high, indata, order, padlen=20):
    if cutofffreq_high > samplefreq/2.0:
        cutofffreq_high = samplefreq/2.0
    if cutofffreq_log < 0.0:
        cutofffreq_low = 0.0
    [b, a] = sp.signal.butter(order, [2.0 * cutofffreq_low / samplefreq, 2.0 * cutofffreq_high / samplefreq], 'bandpass')
    return unpadvec(sp.signal.filtfilt(b, a, padvec(indata, padlen=padlen)).real, padlen=padlen)


def doprecalcfiltfilt(b, a, indata):
    return sp.signal.filtfilt(b, a, indata).real


def dolpfastfiltfiltinit(samplefreq, cutofffreq, indata, order):
    [b, a] = sp.signal.butter(order, cutofffreq / samplefreq)
    return fastfiltfiltinit(b, a, indata)


def dohpfastfiltfiltinit(samplefreq, cutofffreq, indata, order):
    [b, a] = sp.signal.butter(order, cutofffreq / samplefreq, 'highpass')
    return fastfiltfiltinit(b, a, indata)


def dobpfastfiltfiltinit(samplefreq, cutofffreq_low, cutofffreq_high, indata, order):
    [b, a] = sp.signal.butter(order, [cutofffreq_low / samplefreq, cutofffreq_high / samplefreq], 'bandpass')
    return fastfiltfiltinit(b, a, indata)


# - fft brickwall filters
def getlpfftfunc(samplefreq, cutofffreq, indata):
    filterfunc = np.ones(len(indata))
    # cutoffbin = int((cutofffreq / samplefreq) * len(filterfunc) / 2.0)
    cutoffbin = int((cutofffreq / samplefreq) * len(filterfunc))
    filterfunc[cutoffbin:-cutoffbin] = 0.0
    return filterfunc


def doprecalcfftfilt(filterfunc, indata):
    indata_trans = fftpack.fft(indata)
    indata_trans = indata_trans * filterfunc
    return fftpack.ifft(indata_trans).real


def dolpfftfilt(samplefreq, cutofffreq, indata, padlen=20):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = getlpfftfunc(samplefreq, cutofffreq, padindata)
    indata_trans = indata_trans * filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dohpfftfilt(samplefreq, cutofffreq, indata, padlen=20):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = 1.0 - getlpfftfunc(samplefreq, cutofffreq, padindata)
    indata_trans = indata_trans * filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dobpfftfilt(samplefreq, cutofffreq_low, cutofffreq_high, indata, padlen=20):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = getlpfftfunc(samplefreq, cutofffreq_high, padindata) * (
        1.0 - getlpfftfunc(samplefreq, cutofffreq_low, padindata))
    indata_trans = indata_trans * filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


# - fft trapezoidal filters
def getlptrapfftfunc(samplefreq, passfreq, stopfreq, indata):
    filterfunc = np.ones(len(indata))
    passbin = int((passfreq / samplefreq) * len(filterfunc))
    cutoffbin = int((stopfreq / samplefreq) * len(filterfunc))
    translength = cutoffbin - passbin
    if translength > 0:
        transvector = np.arange(1.0 * translength) / translength
        filterfunc[passbin:cutoffbin] = 1.0 - transvector
        filterfunc[-cutoffbin:-passbin] = transvector
    if cutoffbin > 0:
        filterfunc[cutoffbin:-cutoffbin] = 0.0
    return filterfunc


def dolptrapfftfilt(samplefreq, passfreq, stopfreq, indata, padlen=20):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = getlptrapfftfunc(samplefreq, passfreq, stopfreq, padindata)
    indata_trans = indata_trans * filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dohptrapfftfilt(samplefreq, stopfreq, passfreq, indata, padlen=20):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    filterfunc = 1.0 - getlptrapfftfunc(samplefreq, stopfreq, passfreq, padindata)
    indata_trans = indata_trans * filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def dobptrapfftfilt(samplefreq, stopfreq_low, passfreq_low, passfreq_high, stopfreq_high, indata, padlen=20):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = fftpack.fft(padindata)
    if False:
        print("samplefreq=", samplefreq, " Fstopl=", stopfreq_low, " Fpassl=", passfreq_low, " Fpassu=", passfreq_high,
              " Fstopu=", stopfreq_high)
    filterfunc = getlptrapfftfunc(samplefreq, passfreq_high, stopfreq_high, padindata) * (
        1.0 - getlptrapfftfunc(samplefreq, stopfreq_low, passfreq_low, padindata))
    if False:
        freqs = np.arange(0.0, samplefreq, samplefreq / len(filterfunc))
        pl.plot(freqs, filterfunc)
        pl.show()
        sys.exit()
    indata_trans = indata_trans * filterfunc
    return unpadvec(fftpack.ifft(indata_trans).real, padlen=padlen)


def specsplit(samplerate, inputdata, bandwidth, usebutterworth=False):
    lowestfreq = samplerate / (2.0 * len(inputdata))
    highestfreq = samplerate / 2.0
    if lowestfreq < 0.01:
        lowestfreq = 0.01
    if highestfreq > 5.0:
        highestfreq = 5.00
    freqfac = highestfreq / lowestfreq
    print("spectral range=", lowestfreq, " to ", highestfreq, ", factor of ", freqfac)
    lowerlim = lowestfreq
    upperlim = lowerlim * bandwidth
    numbands = 1
    while upperlim < highestfreq:
        lowerlim = lowerlim * bandwidth
        upperlim = upperlim * bandwidth
        numbands += 1
    print("dividing into ", numbands, " bands")
    lowerlim = lowestfreq
    upperlim = lowerlim * bandwidth
    alldata = np.zeros((len(inputdata), numbands))
    bandcenters = np.zeros(numbands)
    print(alldata.shape)
    for theband in range(0, numbands):
        print("filtering from ", lowerlim, " to ", upperlim)
        if usebutterworth:
            alldata[:, theband] = dobpfiltfilt(samplerate, lowerlim, upperlim, inputdata, 2)
        else:
            alldata[:, theband] = dobpfftfilt(samplerate, lowerlim, upperlim, inputdata)
        bandcenters[theband] = np.sqrt(upperlim * lowerlim)
        lowerlim = lowerlim * bandwidth
        upperlim = upperlim * bandwidth
    return bandcenters, lowestfreq, upperlim, alldata

@conditionaljit()
def arb_pass(samplerate, inputdata, arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop, 
            usebutterworth=False, butterorder=defaultbutterorder,
            usetrapfftfilt=True, padlen=20):
    # check filter limits to see if we should do a lowpass, bandpass, or highpass
    if arb_lowerpass <= 0.0:
        # set up for lowpass
        if usebutterworth:
            return dolpfiltfilt(samplerate, arb_upperpass, inputdata, butterorder, padlen=padlen)
        else:
            if usetrapfftfilt:
                return dolptrapfftfilt(samplerate, arb_upperpass, arb_upperstop, inputdata, padlen=padlen)
            else:
                return dolpfftfilt(samplerate, arb_upperpass, inputdata, padlen=padlen)
    elif (arb_upperpass >= samplerate / 2.0) or (arb_upperpass <= 0.0):
        # set up for highpass
        if usebutterworth:
            return dohpfiltfilt(samplerate, arb_lowerpass, inputdata, butterorder, padlen=padlen)
        else:
            if usetrapfftfilt:
                return dohptrapfftfilt(samplerate, arb_lowerstop, arb_lowerpass, inputdata, padlen=padlen)
            else:
                return dohpfftfilt(samplerate, arb_lowerpass, inputdata, padlen=padlen)
    else:
        # set up for bandpass
        if usebutterworth:
            return (dohpfiltfilt(samplerate, arb_lowerpass,
                                 dolpfiltfilt(samplerate, arb_upperpass, inputdata, butterorder, padlen=padlen),
                                 butterorder, padlen=padlen))
        else:
            if usetrapfftfilt:
                return (
                    dobptrapfftfilt(samplerate, arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop, inputdata, padlen=padlen))
            else:
                return dobpfftfilt(samplerate, arb_lowerpass, arb_upperpass, inputdata, padlen=padlen)


def ringstop(samplerate, inputdata, usebutterworth=False, butterorder=defaultbutterorder, usetrapfftfilt=True):
    if usebutterworth:
        return dolpfiltfilt(samplerate, samplerate / 4.0, inputdata, butterorder), 2
    else:
        if usetrapfftfilt:
            return dolptrapfftfilt(samplerate, samplerate / 4.0, 1.1 * samplerate / 4.0, inputdata)
        else:
            return dolpfftfilt(samplerate, samplerate / 4.0, inputdata)


def vlf_pass(samplerate, inputdata, usebutterworth=False, butterorder=defaultbutterorder, usetrapfftfilt=True):
    if usebutterworth:
        return dolpfiltfilt(samplerate, VLF_UPPERPASS, inputdata, butterorder), 2
    else:
        if usetrapfftfilt:
            return dolptrapfftfilt(samplerate, VLF_UPPERPASS, VLF_UPPERSTOP, inputdata)
        else:
            return dolpfftfilt(samplerate, VLF_UPPERPASS, inputdata)


def lfo_pass(samplerate, inputdata, usebutterworth=False, butterorder=defaultbutterorder, usetrapfftfilt=True):
    if usebutterworth:
        return (
            dohpfiltfilt(samplerate, LF_LOWERPASS,
                         dolpfiltfilt(samplerate, LF_UPPERPASS, inputdata, butterorder),
                         2))
    else:
        if usetrapfftfilt:
            return dobptrapfftfilt(samplerate, LF_LOWERSTOP, LF_LOWERPASS, LF_UPPERPASS, LF_UPPERSTOP, inputdata)
        else:
            return dobpfftfilt(samplerate, LF_LOWERPASS, LF_UPPERPASS, inputdata)


def resp_pass(samplerate, inputdata, usebutterworth=False, butterorder=defaultbutterorder, usetrapfftfilt=True):
    if usebutterworth:
        return dobpfiltfilt(samplerate, RESP_LOWERPASS, RESP_UPPERPASS, inputdata, butterorder)
    else:
        if usetrapfftfilt:
            return (
                dobptrapfftfilt(samplerate, RESP_LOWERSTOP, RESP_LOWERPASS, RESP_UPPERPASS, RESP_UPPERSTOP, inputdata))
        else:
            return dobpfftfilt(samplerate, RESP_LOWERPASS, RESP_UPPERPASS, inputdata)


def card_pass(samplerate, inputdata, usebutterworth=False, butterorder=defaultbutterorder, usetrapfftfilt=True):
    if usebutterworth:
        return dobpfiltfilt(samplerate, CARD_LOWERPASS, CARD_UPPERPASS, inputdata, butterorder)
    else:
        if usetrapfftfilt:
            return (
                dobptrapfftfilt(samplerate, CARD_LOWERSTOP, CARD_LOWERPASS, CARD_UPPERPASS, CARD_UPPERSTOP, inputdata))
        else:
            return dobpfftfilt(samplerate, CARD_LOWERPASS, CARD_UPPERPASS, inputdata)


class noncausalfilter:
    def __init__(self, filtertype='none', usebutterworth=False, butterworthorder=3, usetrapfftfilt=True, correctfreq=True, padtime=30.0, debug=False):
        self.filtertype = filtertype
        self.arb_lowerpass = 0.05
        self.arb_lowerstop = 0.9 * self.arb_lowerpass
        self.arb_upperpass = 0.20
        self.arb_upperstop = 1.1 * self.arb_upperpass
        self.lowerstop = 0.0
        self.lowerpass = 0.0
        self.upperpass = -1.0
        self.upperstop = -1.0
        self.usebutterworth = usebutterworth
        self.butterworthorder = butterworthorder
        self.usetrapfftfilt = usetrapfftfilt
        self.correctfreq = correctfreq
        self.padtime = padtime
        self.debug = debug
        self.settype(self.filtertype)

    def settype(self, thetype):
        self.filtertype = thetype
        if self.filtertype == 'vlf' or self.filtertype == 'vlf_stop':
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = 1.0 * VLF_UPPERPASS
            self.upperstop = 1.0 * VLF_UPPERSTOP
        elif self.filtertype == 'lfo' or self.filtertype == 'lfo_stop':
            self.lowerstop = 1.0 * LF_LOWERSTOP
            self.lowerpass = 1.0 * LF_LOWERPASS
            self.upperpass = 1.0 * LF_UPPERPASS
            self.upperstop = 1.0 * LF_UPPERSTOP
        elif self.filtertype == 'resp' or self.filtertype == 'resp_stop':
            self.lowerstop = 1.0 * RESP_LOWERSTOP
            self.lowerpass = 1.0 * RESP_LOWERPASS
            self.upperpass = 1.0 * RESP_UPPERPASS
            self.upperstop = 1.0 * RESP_UPPERSTOP
        elif self.filtertype == 'cardiac' or self.filtertype == 'cardiac_stop':
            self.lowerstop = 1.0 * CARD_LOWERSTOP
            self.lowerpass = 1.0 * CARD_LOWERPASS
            self.upperpass = 1.0 * CARD_UPPERPASS
            self.upperstop = 1.0 * CARD_UPPERSTOP
        elif self.filtertype == 'arb' or self.filtertype == 'arb_stop':
            self.lowerstop = 1.0 * self.arb_lowerstop
            self.lowerpass = 1.0 * self.arb_lowerpass
            self.upperpass = 1.0 * self.arb_upperpass
            self.upperstop = 1.0 * self.arb_upperstop
        else:
            self.lowerstop = 0.0
            self.lowerpass = 0.0
            self.upperpass = -1.0
            self.upperstop = -1.0

    def gettype(self):
        return self.filtertype

    def getfreqlimits(self):
        return self.lowerstop, self.lowerpass, self.upperpass, self.upperstop

    def setbutter(self, useit, order=3):
        self.usebutterworth = useit
        self.butterworthorder = order

    def setpadtime(self, padtime):
        self.padtime = padtime

    def getpadtime(self):
        return self.padtime

    def settrapfft(self, useit):
        self.usetrapfftfilt = useit

    def setarb(self, lowerstop, lowerpass, upperpass, upperstop):
        if not (lowerstop <= lowerpass < upperpass):
            print('noncausalfilter error: lowerpass must be between lowerstop and upperpass')
            sys.exit()
        if not (lowerpass < upperpass <= upperstop):
            print('noncausalfilter error: upperpass must be between lowerpass and upperstop')
            sys.exit()
        self.arb_lowerstop = 1.0 * lowerstop
        self.arb_lowerpass = 1.0 * lowerpass
        self.arb_upperpass = 1.0 * upperpass
        self.arb_upperstop = 1.0 * upperstop
        self.lowerstop = 1.0 * self.arb_lowerstop
        self.lowerpass = 1.0 * self.arb_lowerpass
        self.upperpass = 1.0 * self.arb_upperpass
        self.upperstop = 1.0 * self.arb_upperstop

    def apply(self, samplerate, data):
        # do some bounds checking
        nyquistlimit = 0.5 * samplerate
        lowestfreq = 2.0 * samplerate / len(data)
        if self.upperpass >= nyquistlimit:
            if self.correctfreq:
                self.upperpass = nyquistlimit
            else:
                print('noncausalfilter error: filter upper pass ', self.upperpass, ' exceeds nyquist frequency ',
                      nyquistlimit)
                sys.exit()
        if self.upperstop > nyquistlimit:
            if self.correctfreq:
                self.upperstop = nyquistlimit
            else:
                print('noncausalfilter error: filter upper stop ', self.upperstop, ' exceeds nyquist frequency ',
                      nyquistlimit)
                sys.exit()
        if self.lowerpass < lowestfreq:
            if self.correctfreq:
                self.lowerpass = lowestfreq
            else:
                print('noncausalfilter error: filter lower pass ', self.lowerpass, ' is below minimum frequency ',
                      lowestfreq)
                sys.exit()
        if self.lowerstop < lowestfreq:
            if self.correctfreq:
                self.lowerstop = lowestfreq
            else:
                print('noncausalfilter error: filter lower stop ', self.lowerstop, ' is below minimum frequency ',
                      lowestfreq)
                sys.exit()

        padlen = int(self.padtime * samplerate)
        if self.debug:
            print('samplerate=',samplerate)
            print('lowerstop=',self.lowerstop)
            print('lowerpass=',self.lowerpass)
            print('upperpass=',self.upperpass)
            print('upperstop=',self.upperstop)
            print('usebutterworth=',self.usebutterworth)
            print('butterworthorder=',self.butterworthorder)
            print('usetrapfftfilt=',self.usetrapfftfilt)
            print('padtime=',self.padtime)
            print('padlen=',padlen)

        # now do the actual filtering
        if self.filtertype == 'none':
            return data
        elif self.filtertype == 'ringstop':
            return (arb_pass(samplerate, data,
                             0.0, 0.0, samplerate / 4.0, 1.1 * samplerate / 4.0,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder, 
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen))
        elif self.filtertype == 'vlf' or self.filtertype == 'lfo' \
                or self.filtertype == 'resp' or self.filtertype == 'cardiac':
            return (arb_pass(samplerate, data,
                             self.lowerstop, self.lowerpass, self.upperpass, self.upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder, 
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen))
        elif self.filtertype == 'vlf_stop' or self.filtertype == 'lfo_stop' \
                or self.filtertype == 'resp_stop' or self.filtertype == 'cardiac_stop':
            return (data - arb_pass(samplerate, data,
                             self.lowerstop, self.lowerpass, self.upperpass, self.upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder, 
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen))
        elif self.filtertype == 'arb':
            return (arb_pass(samplerate, data,
                             self.arb_lowerstop, self.arb_lowerpass, self.arb_upperpass, self.arb_upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder, 
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen))
        elif self.filtertype == 'arb_stop':
            return (data - arb_pass(samplerate, data,
                             self.arb_lowerstop, self.arb_lowerpass, self.arb_upperpass, self.arb_upperstop,
                             usebutterworth=self.usebutterworth, butterorder=self.butterworthorder, 
                             usetrapfftfilt=self.usetrapfftfilt, padlen=padlen))
        else:
            print("bad filter type")
            sys.exit()


# --------------------------- Utility functions -------------------------------------------------
def progressbar(thisval, end_val, label='Percent', barsize=60):
    percent = float(thisval) / end_val
    hashes = '#' * int(round(percent * barsize))
    spaces = ' ' * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.3f}%".format(label, hashes + spaces, 100.0 * percent))
    sys.stdout.flush()


