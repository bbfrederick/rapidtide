#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
from __future__ import print_function, division

import numpy as np
import time
import sys
import bisect
import os
import resource
import matplotlib.pyplot as plt

import rapidtide.io as tide_io

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False

try:
    from numba import jit

    numbaexists = True
except ImportError:
    numbaexists = False
numbaexists = False

try:
    import nibabel as nib

    nibabelexists = True
except ImportError:
    nibabelexists = False

donotusenumba = False

try:
    import pyfftw

    pyfftwexists = True
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()
except ImportError:
    pyfftwexists = False


def checkimports(optiondict):
    from numpy.distutils.system_info import get_info
    optiondict['blas_opt'] = get_info('blas_opt')
    optiondict['lapack_opt'] = get_info('lapack_opt')

    if pyfftwexists:
        print('monkey patched scipy.fftpack to use pyfftw')
    else:
        print('using standard scipy.fftpack')
    optiondict['pyfftwexists'] = pyfftwexists

    if numbaexists:
        print('numba exists')
    else:
        print('numba does not exist')
    optiondict['numbaexists'] = numbaexists

    if memprofilerexists:
        print('memprofiler exists')
    else:
        print('memprofiler does not exist')
    optiondict['memprofilerexists'] = memprofilerexists

    if nibabelexists:
        print('nibabel exists')
    else:
        print('nibabel does not exist')
    optiondict['nibabelexists'] = nibabelexists

    if donotbeaggressive:
        print('no aggressive optimization')
    else:
        print('aggressive optimization')
    optiondict['donotbeaggressive'] = donotbeaggressive

    global donotusenumba
    if donotusenumba:
        print('will not use numba even if present')
    else:
        print('using numba if present')
    optiondict['donotusenumba'] = donotusenumba


def conditionaljit():
    def resdec(f):
        if (not numbaexists) or donotusenumba:
            return f
        return jit(f, nopython=False)

    return resdec


def conditionaljit2():
    def resdec(f):
        if (not numbaexists) or donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=False)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Utility functions -------------------------------------------------
def logmem(msg, file=None):
    """

    Parameters
    ----------
    msg
    file

    Returns
    -------

    """
    global lastmaxrss_parent, lastmaxrss_child
    if msg is None:
        lastmaxrss_parent = 0
        lastmaxrss_child = 0
        logline = ','.join([
            '',
            'Self Max RSS',
            'Self Diff RSS',
            'Self Shared Mem',
            'Self Unshared Mem',
            'Self Unshared Stack',
            'Self Non IO Page Fault'
            'Self IO Page Fault'
            'Self Swap Out',
            'Children Max RSS',
            'Children Diff RSS',
            'Children Shared Mem',
            'Children Unshared Mem',
            'Children Unshared Stack',
            'Children Non IO Page Fault'
            'Children IO Page Fault'
            'Children Swap Out'])
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
        logline = ','.join(outvals)
    if file is None:
        print(logline)
    else:
        file.writelines(logline + "\n")


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
    tide_io.writevec([' '.join(theargs)], thename + '_commandline.txt')


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
        print('startpoint is too large (maximum is ', timepoints - 1, ')')
        sys.exit()
    if startpoint < 0:
        realstart = 0
        print('startpoint set to minimum, (0)')
    else:
        realstart = startpoint
        print('startpoint set to ', startpoint)
    if endpoint == -1:
        endpoint = 100000000
    if endpoint > timepoints - 1:
        realend = timepoints - 1
        print('endppoint set to maximum, (', timepoints - 1, ')')
    else:
        realend = endpoint
        print('endpoint set to ', endpoint)
    if realstart >= realend:
        print('endpoint (', realend, ') must be greater than startpoint (', realstart, ')')
        sys.exit()
    return realstart, realend



def valtoindex(thearray, thevalue, evenspacing=True):
    """

    Parameters
    ----------
    thearray: array-like
        An ordered list of values (does not need to be equally spaced)
    thevalue: float
        The value to search for in the array
    evenspacing: boolean, optional
        If True (default), assume data is evenly spaced for faster calculation.

    Returns
    -------
    closestidx: int
        The index of the sample in thearray that is closest to val

    """
    if evenspacing:
        limval = np.max([thearray[0], np.min([thearray[-1], thevalue])])
        return int(np.round((limval - thearray[0]) / (thearray[1] - thearray[0]), 0))
    else:
        return (np.abs(thearray - thevalue)).argmin()


def progressbar(thisval, end_val, label='Percent', barsize=60):
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
    hashes = '#' * int(round(percent * barsize))
    spaces = ' ' * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.3f}%".format(label, hashes + spaces, 100.0 * percent))
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
    print("creating list of ", numsteps, " lag steps (", lagstart, " to ", lagend, " in steps of ", lagstep, ")")
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
        return 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

    version = versioninfo['version']
    longgittag = versioninfo['full-revisionid']
    thedate = versioninfo['date']
    isdirty = versioninfo['dirty']
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


def proctiminginfo(thetimings, outputfile='', extraheader=None):
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
        ax.set_title('temporal output vector')
        plt.plot(timeaxis, outputvector)
        plt.show()
    return outputvector


# --------------------------- testing functions -------------------------------------------------
def comparemap(map1, map2, mask=None, debug=False):
    ndims = len(map1.shape)
    if debug:
        print('map has', ndims, 'axes')
    if map1.shape != map2.shape:
        print('comparemap: maps do not have the same shape - aborting')
        sys.exit()
    if ndims == 1:
        if debug:
            print('dealing with ndims == 1 case')
        map1valid = map1
        map2valid = map2
    else:
        if mask is  None:
            map1valid = map1
            map2valid = map2
        else:
            if debug:
                print('mask is not None')
            ndims_mask = len(mask.shape)
            if debug:
                print('mask has', ndims_mask, 'axes')
            if ndims_mask == ndims:
                if debug:
                    print('dealing with ndims == ndims_mask case')
                if map1.shape != mask.shape:
                    print('comparemap: mask does not have the same shape as the maps - aborting')
                    sys.exit()
                validvoxels = np.where(mask > 0)[0]
                map1valid = map1[validvoxels, :]
                map2valid = map2[validvoxels, :]
            elif ndims_mask == ndims - 1:
                # need to make expanded mask
                if debug:
                    print('dealing with ndims == ndims_mask + 1 case')
                    print('shape of map:', map1.shape)
                    print('shape of mask:', mask.shape)
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
                print('mask is not compatible with map')
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
    for map in ['lagtimes', 'lagstrengths', 'lagsigma', 'MTT', 'fitCoff']:
        filename1 = root1 + '_' + map + '.nii.gz'
        maskname1 = root1 + '_lagmask.nii.gz'
        filename2 = root2 + '_' + map + '.nii.gz'
        maskname2 = root2 + '_lagmask.nii.gz'
        masknim1, maskdata1, maskhdr1, themaskdims1, themasksizes1 = tide_io.readfromnifti(maskname1)
        masknim2, maskdata2, maskhdr2, themaskdims2, themasksizes2 = tide_io.readfromnifti(maskname2)
        if tide_io.checkspacematch(maskhdr1, maskhdr2):
            mask = maskdata1 * maskdata2
            if os.path.isfile(filename1) and os.path.isfile(filename2):
                # files exist - read them in and process them
                nim1, data1, hdr1, thedims1, thesizes1 = tide_io.readfromnifti(filename1)
                nim2, data2, hdr2, thedims2, thesizes2 = tide_io.readfromnifti(filename2)
                if tide_io.checkspacematch(hdr1, hdr2) and tide_io.checkspacematch(hdr1, maskhdr1):
                    # files match in size
                    results[map] = {}
                    results[map]['mindiff'], results[map]['maxdiff'], results[map]['meandiff'], results[map]['mse'], \
                        results[map]['relmindiff'], results[map]['relmaxdiff'], results[map]['relmeandiff'], results[map]['relmse'] = comparemap(data1, data2, mask=mask)
                else:
                    print('mask dimensions don\'t match - aborting')
                    sys.exit()
            else:
                print('map', map, 'does not exist - skipping')
        else:
            print('mask dimensions don\'t match - aborting')
            sys.exit()
    return results


def comparehappyruns(root1, root2, debug=False):
    results = {}
    if debug:
        print('comparehappyruns rootnames:', root1, root2)
    for map in ['app', 'mask', 'vesselmask']:
        filename1 = root1 + '_' + map + '.nii.gz'
        maskname1 = root1 + '_mask.nii.gz'
        filename2 = root2 + '_' + map + '.nii.gz'
        maskname2 = root2 + '_mask.nii.gz'
        masknim1, maskdata1, maskhdr1, themaskdims1, themasksizes1 = tide_io.readfromnifti(maskname1)
        masknim2, maskdata2, maskhdr2, themaskdims2, themasksizes2 = tide_io.readfromnifti(maskname2)
        if tide_io.checkspacematch(maskhdr1, maskhdr2):
            mask = maskdata1 * maskdata2
            if os.path.isfile(filename1) and os.path.isfile(filename2):
                # files exist - read them in and process them
                if debug:
                    print('comparing maps:')
                    print('\t', filename1)
                    print('\t', filename2)
                nim1, data1, hdr1, thedims1, thesizes1 = tide_io.readfromnifti(filename1)
                nim2, data2, hdr2, thedims2, thesizes2 = tide_io.readfromnifti(filename2)
                if tide_io.checkspacematch(hdr1, hdr2) and tide_io.checkspacematch(hdr1, maskhdr1):
                    # files match in size
                    results[map] = {}
                    results[map]['mindiff'], results[map]['maxdiff'], results[map]['meandiff'], results[map]['mse'], \
                    results[map]['relmindiff'], results[map]['relmaxdiff'], results[map]['relmeandiff'], \
                    results[map]['relmse'] \
                        = comparemap(data1, data2, mask=mask, debug=debug)
                else:
                    print('mask dimensions don\'t match - aborting')
                    sys.exit()
            else:
                print('map', map, 'does not exist - skipping')
        else:
            print('mask dimensions don\'t match - aborting')
            sys.exit()
        if debug:
            print('done processing', map)
    for timecourse in ['cardfromfmri_25.0Hz.txt', 'cardfromfmri_dlfiltered_25.0Hz.txt', 'cardfromfmrienv_25.0Hz.txt']:
        filename1 = root1 + '_' + timecourse
        filename2 = root2 + '_' + timecourse
        if os.path.isfile(filename1) and os.path.isfile(filename2):
            if debug:
                print('comparing timecourses:')
                print('\t', filename1)
                print('\t', filename2)
            data1 = np.transpose(tide_io.readvecs(filename1))
            data2 = np.transpose(tide_io.readvecs(filename2))
            if len(data1) == len(data2):
                # files match in size
                results[timecourse] = {}
                results[timecourse]['mindiff'], results[timecourse]['maxdiff'], results[timecourse]['meandiff'], \
                results[timecourse]['mse'], results[timecourse]['relmindiff'], results[timecourse]['relmaxdiff'], \
                results[timecourse]['relmeandiff'], results[timecourse]['relmse'] \
                    = comparemap(data1, data2, debug=debug)
            else:
                print('timecourse lengths don\'t match - aborting')
                sys.exit()
        else:
            print('timecourse', timecourse, 'does not exist - skipping')
        if debug:
            print('done processing', timecourse)

    return results
