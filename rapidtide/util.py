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
        return jit(f)

    return resdec


def conditionaljit2():
    def resdec(f):
        if (not numbaexists) or donotusenumba or donotbeaggressive:
            return f
        return jit(f)

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



def valtoindex(thearray, thevalue, circular=True, toleft=True):
    """

    Parameters
    ----------
    thearray
    thevalue
    toleft

    Returns
    -------

    """
    closestidx = (np.abs(thearray - thevalue)).argmin()
    if (thevalue - thearray[closestidx]) >= 0.0:
        if toleft:
            return np.max([0, closestidx - 1])
        else:
            return closestidx
    else:
        if toleft:
            return closestidx
        else:
            return np.max([0, closestidx - 1])
    return closestidx

    """
    if toleft:
        return bisect.bisect_left(thearray, thevalue)
    else:
        return bisect.bisect_right(thearray, thevalue)
    """


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
    thispath, thisfile = os.path.split(__file__)
    print(thispath)
    fulltag = 'UNKNOWN', 'UNKNOWN'
    if os.path.isfile(os.path.join(thispath, '_gittag.py')):
        with open(os.path.join(thispath, '_gittag.py')) as f:
            for line in f:
                if line.startswith('__gittag__'):
                    fulltag = (line.split()[2]).split('-')
                    break
        return fulltag[0][1:], '-'.join(fulltag[1:])[:-1]
    else:
        return 'UNKNOWN', 'UNKNOWN'


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


# --------------------------- testing functions -------------------------------------------------
def comparemap(map1, map2, mask=None):
    if map1.shape != map2.shape:
        print('comparemap: maps do not have the same shape - aborting')
        sys.exit()
    if mask is not None:
        if map1.shape != mask.shape:
            print('comparemap: mask does not have the same shape as the maps - aborting')
            sys.exit()
    else:  
        mask = map1 * 0 + 1

    validvoxels = np.where(mask > 0)[0]
    map1valid = map1[validvoxels]
    map2valid = map2[validvoxels]
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
