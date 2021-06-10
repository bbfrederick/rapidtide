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
# $Author: frederic $
# $Date: 2016/07/12 13:50:29 $
# $Id: tide_funcs.py,v 1.4 2016/07/12 13:50:29 frederic Exp $
#
import bisect
import logging
import os
import resource
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import pyfftw.interfaces.scipy_fftpack as fftpack
from numba import jit

import rapidtide._version as tide_versioneer
import rapidtide.io as tide_io

LGR = logging.getLogger(__name__)
TimingLGR = logging.getLogger("TIMING")
MemoryLGR = logging.getLogger("MEMORY")

pyfftw.interfaces.cache.enable()

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotusenumba = True
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False


def checkimports(optiondict):
    from numpy.distutils.system_info import get_info

    optiondict["blas_opt"] = get_info("blas_opt")
    optiondict["lapack_opt"] = get_info("lapack_opt")

    if memprofilerexists:
        print("memprofiler exists")
    else:
        print("memprofiler does not exist")
    optiondict["memprofilerexists"] = memprofilerexists

    if donotbeaggressive:
        print("no aggressive optimization")
    else:
        print("aggressive optimization")
    optiondict["donotbeaggressive"] = donotbeaggressive

    global donotusenumba
    if donotusenumba:
        print("will not use numba even if present")
    else:
        print("using numba if present")
    optiondict["donotusenumba"] = donotusenumba


# ----------------------------------------- Conditional jit handling ----------------------------------
def conditionaljit():
    def resdec(f):
        if donotusenumba:
            return f
        return jit(f, nopython=False)

    return resdec


def conditionaljit2():
    def resdec(f):
        if donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=False)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Utility functions -------------------------------------------------
def logmem(msg=None):
    """Log memory usage with a logging object.

    Parameters
    ----------
    msg : str or None, optional
        A message to include in the first column.
        If None, the column headers are logged.
        Default is None.
    """
    global lastmaxrss_parent, lastmaxrss_child
    if msg is None:
        outvals = [
            "",
            "Self Max RSS",
            "Self Diff RSS",
            "Self Shared Mem",
            "Self Unshared Mem",
            "Self Unshared Stack",
            "Self Non IO Page Fault",
            "Self IO Page Fault",
            "Self Swap Out",
            "Children Max RSS",
            "Children Diff RSS",
            "Children Shared Mem",
            "Children Unshared Mem",
            "Children Unshared Stack",
            "Children Non IO Page Fault",
            "Children IO Page Fault",
            "Children Swap Out",
        ]
        lastmaxrss_parent = 0
        lastmaxrss_child = 0
    else:
        rcusage = resource.getrusage(resource.RUSAGE_SELF)
        outvals = [msg]
        outvals.append(str(rcusage.ru_maxrss))
        outvals.append(str(rcusage.ru_maxrss - lastmaxrss_parent))
        lastmaxrss_parent = rcusage.ru_maxrss
        outvals.append(str(rcusage.ru_ixrss))
        outvals.append(str(rcusage.ru_idrss))
        outvals.append(str(rcusage.ru_isrss))
        outvals.append(str(rcusage.ru_minflt))
        outvals.append(str(rcusage.ru_majflt))
        outvals.append(str(rcusage.ru_nswap))
        rcusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        outvals.append(str(rcusage.ru_maxrss))
        outvals.append(str(rcusage.ru_maxrss - lastmaxrss_child))
        lastmaxrss_child = rcusage.ru_maxrss
        outvals.append(str(rcusage.ru_ixrss))
        outvals.append(str(rcusage.ru_idrss))
        outvals.append(str(rcusage.ru_isrss))
        outvals.append(str(rcusage.ru_minflt))
        outvals.append(str(rcusage.ru_majflt))
        outvals.append(str(rcusage.ru_nswap))

    MemoryLGR.info("\t".join(outvals))


def findexecutable(command):
    """

    Parameters
    ----------
    command

    Returns
    -------

    """
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        return shutil.which(command)
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            if os.access(os.path.join(path, command), os.X_OK):
                return os.path.join(path, command)
        return None


def isexecutable(command):
    """

    Parameters
    ----------
    command

    Returns
    -------

    """
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        if shutil.which(command) is not None:
            return True
        else:
            return False
    else:
        return any(
            os.access(os.path.join(path, command), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )


def savecommandline(theargs, thename):
    """

    Parameters
    ----------
    theargs
    thename

    Returns
    -------

    """
    tide_io.writevec([" ".join(theargs)], thename + "_commandline.txt")


def startendcheck(timepoints, startpoint, endpoint):
    """

    Parameters
    ----------
    timepoints
    startpoint
    endpoint

    Returns
    -------

    """
    if startpoint > timepoints - 1:
        print("startpoint is too large (maximum is ", timepoints - 1, ")")
        sys.exit()
    if startpoint < 0:
        realstart = 0
        print("startpoint set to minimum, (0)")
    else:
        realstart = startpoint
        print("startpoint set to ", startpoint)
    if endpoint == -1:
        endpoint = 100000000
    if endpoint > timepoints - 1:
        realend = timepoints - 1
        print("endppoint set to maximum, (", timepoints - 1, ")")
    else:
        realend = endpoint
        print("endpoint set to ", endpoint)
    if realstart >= realend:
        print("endpoint (", realend, ") must be greater than startpoint (", realstart, ")")
        sys.exit()
    return realstart, realend


def valtoindex(
    thearray, thevalue, evenspacing=True, discrete=True, discretization="round", debug=False,
):
    """

    Parameters
    ----------
    thearray: array-like
        An ordered list of values (does not need to be equally spaced)
    thevalue: float
        The value to search for in the array
    evenspacing: boolean, optional
        If True (default), assume data is evenly spaced for faster calculation.
    discrete: boolean, optional
        If True make the index an integer (round by default).
    discretization: string, optional
        Select rounding method - floor, ceiling, or round(default)

    Returns
    -------
    closestidx: int
        The index of the sample in thearray that is closest to val

    """
    if evenspacing:
        limval = np.max([thearray[0], np.min([thearray[-1], thevalue])])
        position = (limval - thearray[0]) / (thearray[1] - thearray[0])
        if debug:
            print("valtoindex:")
            print("\tthevalue:", thevalue)
            print("\tarraymin:", thearray[0])
            print("\tarraymax:", thearray[-1])
            print("\tlimval:", limval)
            print(
                "\tindex:", int(np.round((limval - thearray[0]) / (thearray[1] - thearray[0]), 0)),
            )
        if discrete:
            if discretization == "round":
                position = int(np.round(position, 0))
            elif discretization == "floor":
                position = int(np.floor(position))
            elif discretization == "ceiling":
                position = int(np.ceil(position))
            else:
                print("valtoindex - illegal discretization mode")
                position = None
            position = int(np.min([len(thearray) - 1, np.max([0, position])]))
        return position
    else:
        return int((np.abs(thearray - thevalue)).argmin())


def progressbar(thisval, end_val, label="Percent", barsize=60):
    """

    Parameters
    ----------
    thisval
    end_val
    label
    barsize

    Returns
    -------

    """
    percent = float(thisval) / end_val
    hashes = "#" * int(round(percent * barsize))
    spaces = " " * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.2f}%".format(label, hashes + spaces, 100.0 * percent))
    sys.stdout.flush()


def makelaglist(lagstart, lagend, lagstep):
    """

    Parameters
    ----------
    lagstart
    lagend
    lagstep

    Returns
    -------

    """
    numsteps = int((lagend - lagstart) // lagstep + 1)
    lagend = lagstart + lagstep * (numsteps - 1)
    print(
        "creating list of ",
        numsteps,
        " lag steps (",
        lagstart,
        " to ",
        lagend,
        " in steps of ",
        lagstep,
        ")",
    )
    # thelags = np.r_[0.0:1.0 * numsteps] * lagstep + lagstart
    thelags = np.arange(0.0, 1.0 * numsteps) * lagstep + lagstart
    return thelags


# ------------------------------------------ Version function ----------------------------------
def version():
    """

    Returns
    -------

    """
    try:
        versioninfo = tide_versioneer.get_versions()

    except:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"

    version = versioninfo["version"]
    if version is None:
        version = "UNKNOWN"
    longgittag = versioninfo["full-revisionid"]
    if longgittag is None:
        longgittag = "UNKNOWN"
    thedate = versioninfo["date"]
    if thedate is None:
        thedate = "UNKNOWN"
    isdirty = versioninfo["dirty"]
    if isdirty is None:
        isdirty = "UNKNOWN"
    return version, longgittag, thedate, isdirty


# --------------------------- timing functions -------------------------------------------------
def timefmt(thenumber):
    """

    Parameters
    ----------
    thenumber

    Returns
    -------

    """
    return "{:10.2f}".format(thenumber)


def proctiminginfo(thetimings, outputfile="", extraheader=None):
    """

    Parameters
    ----------
    thetimings
    outputfile
    extraheader

    Returns
    -------

    """
    theinfolist = []
    start = thetimings[0]
    starttime = float(start[1])
    lasteventtime = starttime
    if extraheader is not None:
        print(extraheader)
        theinfolist.append(extraheader)
    headerstring = "Clock time\tProgram time\tDuration\tDescription"
    print(headerstring)
    theinfolist.append(headerstring)
    for theevent in thetimings:
        theduration = float(theevent[1] - lasteventtime)
        outstring = (
            time.strftime("%Y%m%dT%H%M%S", time.localtime(theevent[1]))
            + timefmt(float(theevent[1]) - starttime)
            + "\t"
            + timefmt(theduration)
            + "\t"
            + theevent[0]
        )
        if theevent[2] is not None:
            outstring += " ({0:.2f} {1}/second)".format(
                float(theevent[2]) / theduration, theevent[3]
            )
        print(outstring)
        theinfolist.append(outstring)
        lasteventtime = float(theevent[1])
    if outputfile != "":
        tide_io.writevec(theinfolist, outputfile)


# timecourse functions
def maketcfrom3col(inputdata, timeaxis, outputvector, debug=False):
    theshape = np.shape(inputdata)
    for idx in range(0, theshape[1]):
        starttime = inputdata[0, idx]
        endtime = starttime + inputdata[1, idx]
        if (starttime <= timeaxis[-1]) and (endtime >= 0.0) and (endtime > starttime):
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(outputvector)))
            outputvector[startindex:endindex] = inputdata[2, idx]
            print(starttime, startindex, endtime, endindex)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("temporal output vector")
        plt.plot(timeaxis, outputvector)
        plt.show()
    return outputvector


# --------------------------- simulation functions ----------------------------------------------
def makeslicetimes(numslices, sliceordertype, tr=1.0, multibandfac=1, debug=False):
    outlist = np.zeros((numslices), dtype=np.float)
    if (numslices % multibandfac) != 0:
        print("ERROR: numslices is not evenly divisible by multband factor")
        return None
    mbcycle = int(numslices / multibandfac)
    normal = True
    if sliceordertype == "ascending":
        controllist = [[0, 1]]
    elif sliceordertype == "descending":
        controllist = [[mbcycle - 1, -1]]
    elif sliceordertype == "ascending_interleaved":
        controllist = [[0, 2], [1, 2]]
    elif sliceordertype == "descending_interleaved":
        controllist = [[mbcycle - 1, -2], [mbcycle - 2, -2]]
    elif sliceordertype == "ascending_sparkplug":
        normal = False
        controllist = [[0, int(mbcycle // 2) - 1]]
    elif sliceordertype == "descending_sparkplug":
        normal = False
        controllist = [[mbcycle - 1, -int(mbcycle // 2) - 1]]
    elif sliceordertype == "ascending_interleaved_siemens":
        if numslices % 2 == 0:
            controllist = [[0, 2], [1, 2]]
        else:
            controllist = [[1, 2], [0, 2]]
    elif sliceordertype == "descending_interleaved_siemens":
        if numslices % 2 == 0:
            controllist = [[mbcycle - 1, -2], [mbcycle - 2, -2]]
        else:
            controllist = [[mbcycle - 2, -2], [mbcycle - 1, -2]]
    elif sliceordertype == "ascending_interleaved_philips":
        controllist = []
        numgroups = int(np.floor(np.sqrt(numslices)))
        for i in range(numgroups):
            controllist.append([i, numgroups])
    elif sliceordertype == "descending_interleaved_philips":
        controllist = []
        numgroups = int(np.floor(np.sqrt(numslices)))
        for i in range(numgroups):
            controllist.append([mbcycle - i - 1, -numgroups])
    else:
        print("ERROR: illegal sliceordertype")
        return None

    # now make the slicetimes
    timelist = np.linspace(0, tr, num=mbcycle, endpoint=False)
    slicelist = []
    if debug:
        print("sliceordertype:", sliceordertype)
        print("number of mbcycles:", numslices // mbcycle)
        print("size of mbcycles:", mbcycle)
    for thecontrollist in controllist:
        start = thecontrollist[0]
        step = thecontrollist[1]
        theindex = start
        if normal:
            while 0 <= theindex < mbcycle:
                slicelist.append(theindex)
                theindex += step
        else:
            while len(slicelist) < mbcycle:
                slicelist.append(theindex)
                theindex = (theindex + step) % mbcycle

    if debug:
        print(slicelist)
    for index in range(numslices):
        posinmbcycle = index % mbcycle
        outlist[index] = timelist[slicelist[posinmbcycle]] + 0.0
    return outlist


# --------------------------- testing functions -------------------------------------------------
def comparemap(map1, map2, mask=None, debug=False):
    ndims = len(map1.shape)
    if debug:
        print("map has", ndims, "axes")
    if map1.shape != map2.shape:
        print("comparemap: maps do not have the same shape - aborting")
        sys.exit()
    if ndims == 1:
        if debug:
            print("dealing with ndims == 1 case")
        map1valid = map1
        map2valid = map2
    else:
        if mask is None:
            map1valid = map1
            map2valid = map2
        else:
            if debug:
                print("mask is not None")
            ndims_mask = len(mask.shape)
            if debug:
                print("mask has", ndims_mask, "axes")
            if ndims_mask == ndims:
                if debug:
                    print("dealing with ndims == ndims_mask case")
                if map1.shape != mask.shape:
                    print("comparemap: mask does not have the same shape as the maps - aborting")
                    sys.exit()
                validvoxels = np.where(mask > 0)[0]
                map1valid = map1[validvoxels, :]
                map2valid = map2[validvoxels, :]
            elif ndims_mask == ndims - 1:
                # need to make expanded mask
                if debug:
                    print("dealing with ndims == ndims_mask + 1 case")
                    print("shape of map:", map1.shape)
                    print("shape of mask:", mask.shape)
                numvox = 1
                for i in range(ndims - 1):
                    numvox *= mask.shape[i]
                reshapemask = mask.reshape(numvox)
                reshapemap1 = map1.reshape(numvox, -1)
                reshapemap2 = map2.reshape(numvox, -1)
                validvoxels = np.where(reshapemask > 0)[0]
                map1valid = reshapemap1[validvoxels, :]
                map2valid = reshapemap2[validvoxels, :]
            else:
                print("mask is not compatible with map")
                sys.exit()

    # at this point, map2valid and map1valid are the same dimensions
    diff = map2valid - map1valid
    reldiff = np.where(map1valid != 0.0, diff / map1valid, 0.0)
    maxdiff = np.max(diff)
    mindiff = np.min(diff)
    meandiff = np.mean(diff)
    mse = np.mean(np.square(diff))

    maxreldiff = np.max(reldiff)
    minreldiff = np.min(reldiff)
    meanreldiff = np.mean(reldiff)
    relmse = np.mean(np.square(reldiff))

    return mindiff, maxdiff, meandiff, mse, minreldiff, maxreldiff, meanreldiff, relmse


def comparerapidtideruns(root1, root2):
    results = {}
    for map in ["lagtimes", "lagstrengths", "lagsigma", "MTT", "fitCoff"]:
        filename1 = root1 + "_" + map + ".nii.gz"
        maskname1 = root1 + "_lagmask.nii.gz"
        filename2 = root2 + "_" + map + ".nii.gz"
        maskname2 = root2 + "_lagmask.nii.gz"
        (masknim1, maskdata1, maskhdr1, themaskdims1, themasksizes1,) = tide_io.readfromnifti(
            maskname1
        )
        (masknim2, maskdata2, maskhdr2, themaskdims2, themasksizes2,) = tide_io.readfromnifti(
            maskname2
        )
        if tide_io.checkspacematch(maskhdr1, maskhdr2):
            mask = maskdata1 * maskdata2
            if os.path.isfile(filename1) and os.path.isfile(filename2):
                # files exist - read them in and process them
                nim1, data1, hdr1, thedims1, thesizes1 = tide_io.readfromnifti(filename1)
                nim2, data2, hdr2, thedims2, thesizes2 = tide_io.readfromnifti(filename2)
                if tide_io.checkspacematch(hdr1, hdr2) and tide_io.checkspacematch(hdr1, maskhdr1):
                    # files match in size
                    results[map] = {}
                    (
                        results[map]["mindiff"],
                        results[map]["maxdiff"],
                        results[map]["meandiff"],
                        results[map]["mse"],
                        results[map]["relmindiff"],
                        results[map]["relmaxdiff"],
                        results[map]["relmeandiff"],
                        results[map]["relmse"],
                    ) = comparemap(data1, data2, mask=mask)
                else:
                    print("mask dimensions don't match - aborting")
                    sys.exit()
            else:
                print("map", map, "does not exist - skipping")
        else:
            print("mask dimensions don't match - aborting")
            sys.exit()
    return results


def comparehappyruns(root1, root2, debug=False):
    results = {}
    if debug:
        print("comparehappyruns rootnames:", root1, root2)
    for map in ["app", "mask", "vesselmask"]:
        filename1 = root1 + "_" + map + ".nii.gz"
        maskname1 = root1 + "_mask.nii.gz"
        filename2 = root2 + "_" + map + ".nii.gz"
        maskname2 = root2 + "_mask.nii.gz"
        (masknim1, maskdata1, maskhdr1, themaskdims1, themasksizes1,) = tide_io.readfromnifti(
            maskname1
        )
        (masknim2, maskdata2, maskhdr2, themaskdims2, themasksizes2,) = tide_io.readfromnifti(
            maskname2
        )
        if tide_io.checkspacematch(maskhdr1, maskhdr2):
            mask = maskdata1 * maskdata2
            if os.path.isfile(filename1) and os.path.isfile(filename2):
                # files exist - read them in and process them
                if debug:
                    print("comparing maps:")
                    print("\t", filename1)
                    print("\t", filename2)
                nim1, data1, hdr1, thedims1, thesizes1 = tide_io.readfromnifti(filename1)
                nim2, data2, hdr2, thedims2, thesizes2 = tide_io.readfromnifti(filename2)
                if tide_io.checkspacematch(hdr1, hdr2) and tide_io.checkspacematch(hdr1, maskhdr1):
                    # files match in size
                    results[map] = {}
                    (
                        results[map]["mindiff"],
                        results[map]["maxdiff"],
                        results[map]["meandiff"],
                        results[map]["mse"],
                        results[map]["relmindiff"],
                        results[map]["relmaxdiff"],
                        results[map]["relmeandiff"],
                        results[map]["relmse"],
                    ) = comparemap(data1, data2, mask=mask, debug=debug)
                else:
                    print("mask dimensions don't match - aborting")
                    sys.exit()
            else:
                print("map", map, "does not exist - skipping")
        else:
            print("mask dimensions don't match - aborting")
            sys.exit()
        if debug:
            print("done processing", map)
    for timecourse in [
        "cardfromfmri_25.0Hz.txt",
        "cardfromfmri_dlfiltered_25.0Hz.txt",
        "cardfromfmrienv_25.0Hz.txt",
    ]:
        filename1 = root1 + "_" + timecourse
        filename2 = root2 + "_" + timecourse
        if os.path.isfile(filename1) and os.path.isfile(filename2):
            if debug:
                print("comparing timecourses:")
                print("\t", filename1)
                print("\t", filename2)
            data1 = np.transpose(tide_io.readvecs(filename1))
            data2 = np.transpose(tide_io.readvecs(filename2))
            if len(data1) == len(data2):
                # files match in size
                results[timecourse] = {}
                (
                    results[timecourse]["mindiff"],
                    results[timecourse]["maxdiff"],
                    results[timecourse]["meandiff"],
                    results[timecourse]["mse"],
                    results[timecourse]["relmindiff"],
                    results[timecourse]["relmaxdiff"],
                    results[timecourse]["relmeandiff"],
                    results[timecourse]["relmse"],
                ) = comparemap(data1, data2, debug=debug)
            else:
                print("timecourse lengths don't match - aborting")
                sys.exit()
        else:
            print("timecourse", timecourse, "does not exist - skipping")
        if debug:
            print("done processing", timecourse)

    return results
