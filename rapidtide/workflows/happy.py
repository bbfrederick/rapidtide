#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2018-2025 Blaise Frederick
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
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.happy_supportfuncs as happy_support
import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.maskutil as tide_mask
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.voxelData as tide_voxelData

from .utils import setup_logger

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

try:
    import rapidtide.dlfilter as tide_dlfilt

    dlfilterexists = True
    print("dlfilter exists")
except ImportError:
    dlfilterexists = False
    print("dlfilter does not exist")


def happy_main(argparsingfunc):
    timings = [["Start", time.time(), None, None]]

    # get the command line parameters
    args = argparsingfunc
    infodict = vars(args)

    fmrifilename = args.fmrifilename
    slicetimename = args.slicetimename
    outputroot = args.outputroot

    # delete the old completion file, if present
    Path(f"{outputroot}_DONE.txt").unlink(missing_ok=True)

    # create the canary file
    Path(f"{outputroot}_ISRUNNING.txt").touch()

    # if running in Docker or Apptainer/Singularity, this is necessary to enforce memory limits properly
    # otherwise likely to  error out in gzip.py or at voxelnormalize step.  But do nothing if running in CircleCI
    # because it does NOT like you messing with the container.
    args.containertype = tide_util.checkifincontainer()
    if args.containertype is not None:
        if args.containertype != "CircleCI":
            args.containermemfree, args.containermemswap = tide_util.findavailablemem()
            tide_util.setmemlimit(args.containermemfree)
        else:
            print("running in CircleCI environment - not messing with memory")

    # Set up loggers for workflow
    setup_logger(
        logger_filename=f"{outputroot}_log.txt",
        timing_filename=f"{outputroot}_runtimings.tsv",
        memory_filename=f"{outputroot}_memusage.tsv",
        verbose=args.verbose,
        debug=args.debug,
    )

    timings.append(["Argument parsing done", time.time(), None, None])

    """print(
        "***********************************************************************************************************************************")
    print("NOTICE:  This program is NOT released yet - it's a work in progress and is nowhere near done.  That's why")
    print("there's no documentation or mention in the release notes.  If you want to play with it, be my guest, but be")
    print("aware of the following:")
    print("    1) Any given version of this program may or may not work, or may work in a way different than ")
    print("       a) previous versions, b) what I say it does, c) what I think it does, and d) what you want it to do.")
    print(
        "    2) I am intending to write a paper on this, and if you take this code and scoop me, I'll be peeved. That's just rude.")
    print("    3) For all I know this program might burn down your house, leave your milk out of the refrigerator, or ")
    print("       poison your dog.  USE AT YOUR OWN RISK.")
    print(
        "***********************************************************************************************************************************")
    print("")"""

    infodict["fmrifilename"] = fmrifilename
    infodict["slicetimename"] = slicetimename
    infodict["outputroot"] = outputroot

    # save program version
    (
        infodict["release_version"],
        infodict["git_sha"],
        infodict["git_date"],
        infodict["git_isdirty"],
    ) = tide_util.version()

    # record the machine we ran on
    infodict["hostname"] = platform.node()

    print("running version", infodict["release_version"], "on host", infodict["hostname"])

    # construct the BIDS base dictionary
    outputpath = os.path.dirname(infodict["outputroot"])
    rawsources = [os.path.relpath(infodict["fmrifilename"], start=outputpath)]
    bidsbasedict = {
        "RawSources": rawsources,
        "Units": "arbitrary",
        "CommandLineArgs": args.commandline,
    }

    # save the information file
    if args.saveinfoasjson:
        tide_io.writedicttojson(infodict, outputroot + "_info.json")
    else:
        tide_io.writedict(infodict, outputroot + "_info.txt")

    tide_util.logmem()
    tide_util.logmem("Start")

    # set the number of MKL threads to use
    if mklexists:
        mklmaxthreads = mkl.get_max_threads()
        if not (1 <= args.mklthreads <= mklmaxthreads):
            args.mklthreads = mklmaxthreads
        mkl.set_num_threads(args.mklthreads)

    # if we are going to do noise regression, make sure we are generating app matrix
    if (args.dotemporalregression or args.dospatialregression) and args.cardcalconly:
        print("doing glm fit requires phase projection - setting cardcalconly to False")
        args.cardcalconly = False

    # set up cardiac filter
    arb_lower = args.minhrfilt / 60.0
    arb_upper = args.maxhrfilt / 60.0
    thecardbandfilter = tide_filt.NoncausalFilter()
    thecardbandfilter.settype("arb")
    arb_lowerstop = arb_lower * 0.9
    arb_upperstop = arb_upper * 1.1
    thecardbandfilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
    therespbandfilter = tide_filt.NoncausalFilter()
    therespbandfilter.settype("resp")
    infodict["filtermaxbpm"] = arb_upper * 60.0
    infodict["filterminbpm"] = arb_lower * 60.0
    infodict["notchpct"] = args.notchpct
    timings.append(["Argument parsing done", time.time(), None, None])

    # read in the image data
    tide_util.logmem("before reading in fmri data")
    input_data = tide_voxelData.VoxelData(fmrifilename, validstart=args.numskip)
    xsize, ysize, numslices, dummy = input_data.getdims()
    timepoints = input_data.realtimepoints
    xdim, ydim, slicethickness, tr = input_data.getsizes()
    numspatiallocs = input_data.numspatiallocs
    mrsamplerate = 1.0 / tr
    print("tr is", tr, "seconds, mrsamplerate is", "{:.3f}".format(mrsamplerate))
    infodict["tr"] = tr
    infodict["mrsamplerate"] = mrsamplerate
    timings.append(["Image data read in", time.time(), None, None])

    # remap to space by time
    fmri_data = input_data.byvoxel()
    print("fmri_data has shape", fmri_data.shape)

    # make and save a mask of the voxels to process based on image intensity
    tide_util.logmem("before mask creation")
    mask = np.uint16(tide_mask.makeepimask(input_data.nim).dataobj.reshape(numspatiallocs))
    theheader = input_data.copyheader(numtimepoints=1)
    timings.append(["Mask created", time.time(), None, None])
    if args.outputlevel > 0:
        maskfilename = outputroot + "_desc-processvoxels_mask"
        bidsdict = bidsbasedict.copy()
        tide_io.writedicttojson(bidsdict, maskfilename + ".json")
        tide_io.savetonifti(mask.reshape((xsize, ysize, numslices)), theheader, maskfilename)
    timings.append(["Mask saved", time.time(), None, None])
    mask_byslice = mask.reshape((xsize * ysize, numslices))

    # read in projection mask if present otherwise fall back to intensity mask
    if args.projmaskname is not None:
        tide_util.logmem("before reading in projmask")
        projmask = happy_support.readextmask(
            args.projmaskname, input_data.nim_hdr, xsize, ysize, numslices, args.debug
        )
        # * np.float64(mask_byslice)
        projmask_byslice = projmask.reshape(xsize * ysize, numslices)
    else:
        projmask = mask.reshape(xsize * ysize, numslices)
        projmask_byslice = mask_byslice

    # output mask size
    validprojvoxels = np.where(projmask.reshape(numspatiallocs) > 0)[0]
    print(f"projmask has {len(validprojvoxels)} voxels above threshold.")

    # filter out motion regressors here
    if args.motionfilename is not None:
        timings.append(["Motion filtering start", time.time(), None, None])
        motiondict = tide_io.readmotion(args.motionfilename, tr=tr, colspec=args.motionfilecolspec)
        confoundregressors, confoundregressorlabels = tide_fit.calcexpandedregressors(
            motiondict,
            labels=["xtrans", "ytrans", "ztrans", "xrot", "yrot", "zrot"],
            deriv=args.motfilt_deriv,
            order=args.motfilt_order,
        )
        tide_util.disablemkl(args.nprocs)
        (motionregressors, motionregressorlabels, filtereddata, confoundr2) = (
            tide_linfitfiltpass.confoundregress(
                confoundregressors,
                confoundregressorlabels,
                fmri_data[validprojvoxels, :],
                tr,
                orthogonalize=args.orthogonalize,
                tcstart=args.motskip,
                tchp=args.motionhp,
                tclp=args.motionlp,
                nprocs=args.nprocs,
            )
        )
        tide_util.enablemkl(args.mklthreads)
        if confoundr2 is None:
            print("There are no nonzero confound regressors - exiting")
            sys.exit()
        fmri_data[validprojvoxels, :] = filtereddata[:, :]
        infodict["numorthogmotregressors"] = motionregressors.shape[0]
        timings.append(["Motion filtering end", time.time(), numspatiallocs, "voxels"])
        if args.orthogonalize:
            motiontype = "orthogonalizedmotion"
        else:
            motiontype = "motion"
        tide_io.writebidstsv(
            outputroot + "_desc-" + motiontype + "_timeseries",
            motionregressors,
            mrsamplerate,
            columns=motionregressorlabels,
            append=False,
            debug=args.debug,
        )
        # save motionr2 map
        theheader = input_data.copyheader(numtimepoints=1)
        motionr2filename = outputroot + "_desc-motionr2_map"
        bidsdict = bidsbasedict.copy()
        tide_io.writedicttojson(bidsdict, motionr2filename + ".json")
        outarray = np.zeros((xsize, ysize, numslices), dtype=float)
        outarray.reshape(numspatiallocs)[validprojvoxels] = confoundr2
        tide_io.savetonifti(
            outarray.reshape((xsize, ysize, numslices)), theheader, motionr2filename
        )
        if args.savemotionglmfilt:
            print("prior to save - fmri_data has shape", fmri_data.shape)
            motionfilteredfilename = outputroot + "_desc-motionfiltered_bold"
            bidsdict = bidsbasedict.copy()
            bidsdict["Units"] = "second"
            tide_io.writedicttojson(bidsdict, motionfilteredfilename + ".json")
            tide_io.savetonifti(
                fmri_data.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                motionfilteredfilename,
            )
            timings.append(["Motion filtered data saved", time.time(), numspatiallocs, "voxels"])

    # get slice times
    slicetimes, normalizedtotr, fileisbidsjson = tide_io.getslicetimesfromfile(slicetimename)
    if normalizedtotr and not args.slicetimesareinseconds:
        slicetimes *= tr
    if args.teoffset is not None:
        teoffset = float(args.teoffset)
    else:
        if fileisbidsjson:
            jsoninfodict = tide_io.readdictfromjson(slicetimename)
            try:
                teoffset = jsoninfodict["EchoTime"]
            except KeyError:
                teoffset = 0.0
        else:
            teoffset = 0.0
    infodict["teoffset"] = teoffset
    timings.append(["Slice times determined", time.time(), None, None])

    # normalize the input data
    tide_util.logmem("before normalization")
    normdata, demeandata, means, medians, mads = happy_support.normalizevoxels(
        fmri_data,
        args.detrendorder,
        validprojvoxels,
        time,
        timings,
        LGR=None,
        mpcode=args.mpdetrend,
        nprocs=args.nprocs,
        showprogressbar=args.showprogressbar,
        debug=args.debug,
    )
    normdata_byslice = normdata.reshape((xsize * ysize, numslices, timepoints))

    # save means, medians, and mads
    theheader = input_data.copyheader(numtimepoints=1)
    meansfilename = outputroot + "_desc-means_map"
    mediansfilename = outputroot + "_desc-medians_map"
    madsfilename = outputroot + "_desc-mads_map"
    bidsdict = bidsbasedict.copy()
    tide_io.writedicttojson(bidsdict, meansfilename + ".json")
    tide_io.writedicttojson(bidsdict, mediansfilename + ".json")
    tide_io.writedicttojson(bidsdict, madsfilename + ".json")
    tide_io.savetonifti(means.reshape((xsize, ysize, numslices)), theheader, meansfilename)
    tide_io.savetonifti(medians.reshape((xsize, ysize, numslices)), theheader, mediansfilename)
    tide_io.savetonifti(mads.reshape((xsize, ysize, numslices)), theheader, madsfilename)

    # read in estimation mask if present. Otherwise, otherwise use intensity mask.
    infodict["estweightsname"] = args.estweightsname
    if args.debug:
        print(args.estweightsname)
    if args.estweightsname is not None:
        tide_util.logmem("before reading in estweights")
        estweights = happy_support.readextmask(
            args.estweightsname, input_data.nim_hdr, xsize, ysize, numslices, args.debug
        )
        estweights_byslice = estweights.reshape(xsize * ysize, numslices)
        print("using estweights from file", args.estweightsname)
        numpasses = 1
    else:
        # just fall back to the intensity mask
        estweights_byslice = mask_byslice.astype("float64")
        numpasses = 2
        print("Not using separate estimation mask - doing initial estimate using intensity mask")

    # add another pass to refine the waveform after getting the new appflips
    if args.fliparteries or args.doaliasedcorrelation:
        numpasses += 1
        print("Adding a pass to regenerate cardiac waveform using better vessel specification")

    # output mask size
    print(
        f"estweights has {len(np.where(estweights_byslice[:, :] > 0)[0])} voxels above threshold."
    )

    infodict["numpasses"] = numpasses

    # if we have an estimation mask, run procedure once.  If not, run once to get a vessel mask, then rerun.
    appflips_byslice = None
    for thispass in range(numpasses):
        if numpasses > 1:
            print()
            print()
            print("starting pass", thispass + 1, "of", numpasses)
            passstring = " - pass " + str(thispass + 1)
        else:
            passstring = ""
        # now get an estimate of the cardiac signal
        print("estimating cardiac signal from fmri data")
        tide_util.logmem("before cardiacfromimage")
        tide_util.disablemkl(args.nprocs)
        (
            cardfromfmri_sliceres,
            cardfromfmri_normfac,
            respfromfmri_sliceres,
            respfromfmri_normfac,
            slicesamplerate,
            numsteps,
            sliceoffsets,
            cycleaverage,
            slicenorms,
        ) = happy_support.cardiacfromimage(
            normdata_byslice,
            estweights_byslice,
            numslices,
            timepoints,
            tr,
            slicetimes,
            thecardbandfilter,
            therespbandfilter,
            invertphysiosign=args.invertphysiosign,
            madnorm=args.domadnorm,
            nprocs=args.nprocs,
            notchpct=args.notchpct,
            fliparteries=args.fliparteries,
            arteriesonly=args.arteriesonly,
            usemask=args.usemaskcardfromfmri,
            appflips_byslice=appflips_byslice,
            debug=args.debug,
            verbose=args.verbose,
        )
        tide_util.enablemkl(args.mklthreads)
        timings.append(
            [
                "Cardiac signal generated from image data" + passstring,
                time.time(),
                None,
                None,
            ]
        )
        infodict["cardfromfmri_normfac"] = cardfromfmri_normfac
        slicetimeaxis = (
            np.linspace(0.0, tr * timepoints, num=(timepoints * numsteps), endpoint=False)
            + teoffset
        )
        if (thispass == 0) and args.doupsampling:
            happy_support.upsampleimage(
                input_data, numsteps, sliceoffsets, slicesamplerate, outputroot
            )
            sys.exit(0)

        if thispass == numpasses - 1:
            tide_io.writebidstsv(
                outputroot + "_desc-cycleaverage_timeseries",
                cycleaverage,
                slicesamplerate,
                columns=["cycleaverage"],
                append=False,
                debug=args.debug,
            )
            tide_io.writebidstsv(
                outputroot + "_desc-slicerescardfromfmri_timeseries",
                cardfromfmri_sliceres,
                slicesamplerate,
                columns=["cardiacfromfmri"],
                append=False,
                debug=args.debug,
            )
            tide_io.writebidstsv(
                outputroot + "_desc-sliceresrespfromfmri_timeseries",
                respfromfmri_sliceres,
                slicesamplerate,
                columns=["respfromfmri"],
                append=False,
                debug=args.debug,
            )

        # stash away a copy of the waveform if we need it later
        raw_cardfromfmri_sliceres = np.array(cardfromfmri_sliceres)

        # find bad points in cardiac from fmri
        thebadcardpts = happy_support.findbadpts(
            cardfromfmri_sliceres,
            "cardfromfmri_sliceres",
            outputroot,
            slicesamplerate,
            infodict,
        )

        cardiacwaveform = np.array(cardfromfmri_sliceres)
        badpointlist = np.array(thebadcardpts)

        infodict["slicesamplerate"] = slicesamplerate
        infodict["numcardpts_sliceres"] = timepoints * numsteps
        infodict["numsteps"] = numsteps
        infodict["slicenorms"] = slicenorms

        # find key components of cardiac waveform
        print("extracting harmonic components")
        if args.outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    cardfromfmri_sliceres * (1.0 - thebadcardpts),
                    slicesamplerate,
                    columns=["cardiacfromfmri_censored"],
                    append=True,
                    debug=args.debug,
                )
                tide_io.writebidstsv(
                    outputroot + "_desc-sliceresrespfromfmri_timeseries",
                    respfromfmri_sliceres * (1.0 - thebadcardpts),
                    slicesamplerate,
                    columns=["respfromfmri_censored"],
                    append=False,
                    debug=args.debug,
                )
        peakfreq_bold = happy_support.getcardcoeffs(
            (1.0 - thebadcardpts) * cardiacwaveform,
            slicesamplerate,
            minhr=args.minhr,
            maxhr=args.maxhr,
            smoothlen=args.smoothlen,
            debug=args.debug,
        )
        infodict["cardiacbpm_bold"] = np.round(peakfreq_bold * 60.0, 2)
        infodict["cardiacfreq_bold"] = peakfreq_bold
        timings.append(
            [
                "Cardiac signal from image data analyzed" + passstring,
                time.time(),
                None,
                None,
            ]
        )

        # resample to standard frequency
        cardfromfmri_stdres = tide_math.madnormalize(
            tide_resample.arbresample(
                cardfromfmri_sliceres,
                slicesamplerate,
                args.stdfreq,
                decimate=True,
                debug=False,
            )
        )

        if thispass == numpasses - 1:
            tide_io.writebidstsv(
                outputroot + "_desc-stdrescardfromfmri_timeseries",
                cardfromfmri_stdres,
                args.stdfreq,
                columns=["cardiacfromfmri_" + str(args.stdfreq) + "Hz"],
                append=False,
                debug=args.debug,
            )
        infodict["numcardpts_stdres"] = len(cardfromfmri_stdres)

        # normalize the signal to remove envelope effects
        (
            filtcardfromfmri_stdres,
            normcardfromfmri_stdres,
            cardfromfmrienv_stdres,
            envmean,
        ) = happy_support.cleanphysio(
            args.stdfreq,
            cardfromfmri_stdres,
            iscardiac=True,
            cutoff=args.envcutoff,
            nyquist=slicesamplerate / 2.0,
            thresh=args.envthresh,
        )
        if thispass == numpasses - 1:
            tide_io.writebidstsv(
                outputroot + "_desc-stdrescardfromfmri_timeseries",
                normcardfromfmri_stdres,
                args.stdfreq,
                columns=["normcardiac_" + str(args.stdfreq) + "Hz"],
                append=True,
                debug=args.debug,
            )
            tide_io.writebidstsv(
                outputroot + "_desc-stdrescardfromfmri_timeseries",
                cardfromfmrienv_stdres,
                args.stdfreq,
                columns=["envelope_" + str(args.stdfreq) + "Hz"],
                append=True,
            )

        # calculate quality metrics
        happy_support.calcplethquality(
            normcardfromfmri_stdres,
            args.stdfreq,
            infodict,
            "_bold",
            outputroot,
            outputlevel=args.outputlevel,
            initfile=True,
            debug=args.debug,
        )

        thebadcardpts_stdres = happy_support.findbadpts(
            cardfromfmri_stdres,
            "cardfromfmri_" + str(args.stdfreq) + "Hz",
            outputroot,
            args.stdfreq,
            infodict,
        )

        timings.append(
            [
                "Cardiac signal from image data resampled and saved" + passstring,
                time.time(),
                None,
                None,
            ]
        )

        # apply the deep learning filter if we're going to do that
        if args.dodlfilter:
            if dlfilterexists:
                if args.mpfix:
                    print("performing super dangerous openmp workaround")
                    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                modelpath = os.path.join(
                    os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0],
                    "rapidtide",
                    "data",
                    "models",
                )
                thedlfilter = tide_dlfilt.DeepLearningFilter(modelpath=modelpath)
                thedlfilter.loadmodel(args.modelname)
                updatemodels = False
                if updatemodels:
                    updatedmodelname = f"{args.modelname}_tf2"
                    newmodeldir = os.path.join(
                        "/Users/frederic/code/rapidtide/rapidtide/data/models", updatedmodelname
                    )
                    print(f"creating {newmodeldir}")
                    tide_util.makeadir(newmodeldir)
                    thedlfilter.savemodel(altname=newmodeldir)
                infodict["dlfiltermodel"] = args.modelname
                normdlfilteredcard_stdres = thedlfilter.apply(normcardfromfmri_stdres)
                dlfilteredcard_stdres = thedlfilter.apply(cardfromfmri_stdres)
                if thispass == numpasses - 1:
                    tide_io.writebidstsv(
                        outputroot + "_desc-stdrescardfromfmri_timeseries",
                        normdlfilteredcard_stdres,
                        args.stdfreq,
                        columns=["normcardiacfromfmri_dlfiltered_" + str(args.stdfreq) + "Hz"],
                        append=True,
                        debug=args.debug,
                    )
                    tide_io.writebidstsv(
                        outputroot + "_desc-stdrescardfromfmri_timeseries",
                        dlfilteredcard_stdres,
                        args.stdfreq,
                        columns=["cardiacfromfmri_dlfiltered_" + str(args.stdfreq) + "Hz"],
                        append=True,
                        debug=args.debug,
                    )

                # calculate quality metrics
                happy_support.calcplethquality(
                    dlfilteredcard_stdres,
                    args.stdfreq,
                    infodict,
                    "_dlfiltered",
                    outputroot,
                    outputlevel=args.outputlevel,
                    initfile=False,
                    debug=args.debug,
                )

                # downsample to sliceres from stdres
                # cardfromfmri_sliceres = tide_math.madnormalize(
                #    tide_resample.arbresample(dlfilteredcard_stdres, args.stdfreq, slicesamplerate, decimate=True, debug=False))
                stdtimeaxis = (1.0 / args.stdfreq) * np.linspace(
                    0.0,
                    len(dlfilteredcard_stdres),
                    num=(len(dlfilteredcard_stdres)),
                    endpoint=False,
                )
                arb_lowerstop = 0.0
                arb_lowerpass = 0.0
                arb_upperpass = slicesamplerate / 2.0
                arb_upperstop = slicesamplerate / 2.0
                theaafilter = tide_filt.NoncausalFilter(filtertype="arb")
                theaafilter.setfreqs(arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop)

                cardfromfmri_sliceres = tide_math.madnormalize(
                    tide_resample.doresample(
                        stdtimeaxis,
                        theaafilter.apply(args.stdfreq, dlfilteredcard_stdres),
                        slicetimeaxis,
                        method="univariate",
                        padlen=0,
                    )
                )
                if thispass == numpasses - 1:
                    tide_io.writebidstsv(
                        outputroot + "_desc-slicerescardfromfmri_timeseries",
                        cardfromfmri_sliceres,
                        slicesamplerate,
                        columns=["cardiacfromfmri_dlfiltered"],
                        append=True,
                        debug=args.debug,
                    )
                infodict["used_dlreconstruction_filter"] = True
                peakfreq_dlfiltered = happy_support.getcardcoeffs(
                    cardfromfmri_sliceres,
                    slicesamplerate,
                    minhr=args.minhr,
                    maxhr=args.maxhr,
                    smoothlen=args.smoothlen,
                    debug=args.debug,
                )
                infodict["cardiacbpm_dlfiltered"] = np.round(peakfreq_dlfiltered * 60.0, 2)
                infodict["cardiacfreq_dlfiltered"] = peakfreq_dlfiltered

                # check the match between the raw and filtered cardiac signals
                maxval, maxdelay, failreason = happy_support.checkcardmatch(
                    raw_cardfromfmri_sliceres,
                    cardfromfmri_sliceres,
                    slicesamplerate,
                    debug=args.debug,
                )
                print(
                    "Filtered cardiac fmri waveform delay is",
                    "{:.3f}".format(maxdelay),
                    "relative to raw fMRI data",
                )
                print(
                    "Correlation coefficient between cardiac regressors:",
                    "{:.3f}".format(maxval),
                )
                infodict["corrcoeff_raw2filt"] = maxval + 0
                infodict["delay_raw2filt"] = maxdelay + 0
                infodict["failreason_raw2filt"] = failreason + 0

                timings.append(
                    [
                        "Deep learning filter applied" + passstring,
                        time.time(),
                        None,
                        None,
                    ]
                )
            else:
                print("dlfilter could not be loaded - skipping")

        # get the cardiac signal from a file, if specified
        if args.cardiacfilename is not None:
            tide_util.logmem("before cardiacfromfile")
            (
                pleth_sliceres,
                pleth_stdres,
                returnedinputfreq,
                fullrespts,
            ) = happy_support.getphysiofile(
                args.cardiacfilename,
                args.inputfreq,
                args.inputstart,
                slicetimeaxis,
                args.stdfreq,
                len(cardfromfmri_stdres),
                args.envcutoff,
                args.envthresh,
                timings,
                outputroot,
                iscardiac=True,
                outputlevel=args.outputlevel,
                debug=args.debug,
            )
            infodict["cardiacfromfmri"] = False
            infodict["numplethpts_sliceres"] = len(pleth_sliceres)
            infodict["numplethpts_stdres"] = len(pleth_stdres)
            infodict["plethsamplerate"] = returnedinputfreq
            infodict["numplethpts_fullres"] = fullrespts

            if args.dodlfilter and dlfilterexists:
                maxval, maxdelay, failreason = happy_support.checkcardmatch(
                    pleth_sliceres,
                    cardfromfmri_sliceres,
                    slicesamplerate,
                    debug=args.debug,
                )
                print(
                    "Input cardiac waveform delay is",
                    "{:.3f}".format(maxdelay),
                    "relative to filtered fMRI data",
                )
                print(
                    "Correlation coefficient between cardiac regressors is",
                    "{:.3f}".format(maxval),
                )
                infodict["corrcoeff_filt2pleth"] = maxval + 0
                infodict["delay_filt2pleth"] = maxdelay + 0
                infodict["failreason_filt2pleth"] = failreason + 0

            # check the match between the bold and physio cardiac signals
            maxval, maxdelay, failreason = happy_support.checkcardmatch(
                pleth_sliceres,
                raw_cardfromfmri_sliceres,
                slicesamplerate,
                debug=args.debug,
            )
            print(
                "Input cardiac waveform delay is",
                "{:.3f}".format(maxdelay),
            )
            print(
                "Correlation coefficient between cardiac regressors is",
                "{:.3f}".format(maxval),
            )
            infodict["corrcoeff_raw2pleth"] = maxval + 0
            infodict["delay_raw2pleth"] = maxdelay + 0
            infodict["failreason_raw2pleth"] = failreason + 0

            # align the pleth signal with the cardiac signal derived from the data
            if args.aligncardiac:
                alignpts_sliceres = -maxdelay / slicesamplerate  # maxdelay is in seconds
                pleth_sliceres, dummy1, dummy2, dummy2 = tide_resample.timeshift(
                    pleth_sliceres, alignpts_sliceres, int(10.0 * slicesamplerate)
                )
                alignpts_stdres = -maxdelay * args.stdfreq  # maxdelay is in seconds
                pleth_stdres, dummy1, dummy2, dummy3 = tide_resample.timeshift(
                    pleth_stdres, alignpts_stdres, int(10.0 * args.stdfreq)
                )
            if thispass == numpasses - 1:
                if args.debug:
                    print("about to do the thing that causes the crash")
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    pleth_sliceres,
                    slicesamplerate,
                    columns=["pleth"],
                    append=True,
                    debug=args.debug,
                )
                tide_io.writebidstsv(
                    outputroot + "_desc-stdrescardfromfmri_timeseries",
                    pleth_stdres,
                    args.stdfreq,
                    columns=["pleth"],
                    append=True,
                    debug=args.debug,
                )

            # now clean up cardiac signal
            (
                filtpleth_stdres,
                normpleth_stdres,
                plethenv_stdres,
                envmean,
            ) = happy_support.cleanphysio(
                args.stdfreq,
                pleth_stdres,
                iscardiac=True,
                cutoff=args.envcutoff,
                thresh=args.envthresh,
            )
            if thispass == numpasses - 1:
                tide_io.writebidstsv(
                    outputroot + "_desc-stdrescardfromfmri_timeseries",
                    normpleth_stdres,
                    args.stdfreq,
                    columns=["normpleth"],
                    append=True,
                    debug=args.debug,
                )
                tide_io.writebidstsv(
                    outputroot + "_desc-stdrescardfromfmri_timeseries",
                    plethenv_stdres,
                    args.stdfreq,
                    columns=["plethenv"],
                    append=True,
                    debug=args.debug,
                )

            # calculate quality metrics
            happy_support.calcplethquality(
                filtpleth_stdres,
                args.stdfreq,
                infodict,
                "_pleth",
                outputroot,
                outputlevel=args.outputlevel,
                initfile=False,
                debug=args.debug,
            )

            if args.dodlfilter and dlfilterexists:
                dlfilteredpleth = thedlfilter.apply(pleth_stdres)
                if thispass == numpasses - 1:
                    tide_io.writebidstsv(
                        outputroot + "_desc-stdrescardfromfmri_timeseries",
                        dlfilteredpleth,
                        args.stdfreq,
                        columns=["pleth_dlfiltered"],
                        append=True,
                        debug=args.debug,
                    )
                    maxval, maxdelay, failreason = happy_support.checkcardmatch(
                        pleth_stdres, dlfilteredpleth, args.stdfreq, debug=args.debug
                    )
                    print(
                        "Filtered pleth cardiac waveform delay is",
                        "{:.3f}".format(maxdelay),
                        "relative to raw pleth data",
                    )
                    print(
                        "Correlation coefficient between pleth regressors:",
                        "{:.3f}".format(maxval),
                    )
                    infodict["corrcoeff_pleth2filtpleth"] = maxval + 0
                    infodict["delay_pleth2filtpleth"] = maxdelay + 0
                    infodict["failreason_pleth2filtpleth"] = failreason + 0

            # find bad points in plethysmogram
            thebadplethpts_sliceres = happy_support.findbadpts(
                pleth_sliceres,
                "pleth_sliceres",
                outputroot,
                slicesamplerate,
                infodict,
                thetype="fracval",
            )

            thebadplethpts_stdres = happy_support.findbadpts(
                pleth_stdres,
                "pleth_" + str(args.stdfreq) + "Hz",
                outputroot,
                args.stdfreq,
                infodict,
                thetype="fracval",
            )
            timings.append(
                [
                    "Cardiac signal from physiology data resampled to standard and saved"
                    + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )

            # find key components of cardiac waveform
            filtpleth = tide_math.madnormalize(
                thecardbandfilter.apply(slicesamplerate, pleth_sliceres)
            )
            peakfreq_file = happy_support.getcardcoeffs(
                (1.0 - thebadplethpts_sliceres) * filtpleth,
                slicesamplerate,
                minhr=args.minhr,
                maxhr=args.maxhr,
                smoothlen=args.smoothlen,
                debug=args.debug,
            )
            timings.append(
                [
                    "Cardiac coefficients calculated from pleth waveform" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )
            infodict["cardiacbpm_pleth"] = np.round(peakfreq_file * 60.0, 2)
            infodict["cardiacfreq_pleth"] = peakfreq_file
            timings.append(
                [
                    "Cardiac signal from physiology data analyzed" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )
            timings.append(
                [
                    "Cardiac parameters extracted from physiology data" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )

            if not args.projectwithraw:
                cardiacwaveform = np.array(pleth_sliceres)
                badpointlist = 1.0 - (1.0 - thebadplethpts_sliceres) * (1.0 - badpointlist)

            infodict["pleth"] = True
            peakfreq = peakfreq_file
        else:
            infodict["pleth"] = False
            peakfreq = peakfreq_bold
        if args.outputlevel > 0:
            if thispass == numpasses - 1:
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    badpointlist,
                    slicesamplerate,
                    columns=["badpts"],
                    append=True,
                    debug=args.debug,
                )
                badpointlist_stdres = np.round(
                    tide_resample.arbresample(
                        badpointlist,
                        slicesamplerate,
                        args.stdfreq,
                        decimate=True,
                        debug=False,
                    ),
                    0,
                )
                tide_io.writebidstsv(
                    outputroot + "_desc-stdrescardfromfmri_timeseries",
                    badpointlist_stdres,
                    args.stdfreq,
                    columns=["badpts"],
                    append=True,
                    debug=args.debug,
                )

        #  extract the fundamental
        if args.forcedhr is not None:
            peakfreq = args.forcedhr
            infodict["forcedhr"] = peakfreq
        if args.cardiacfilename is None:
            filthiresfund = tide_math.madnormalize(
                happy_support.getperiodic(
                    cardiacwaveform * (1.0 - thebadcardpts),
                    slicesamplerate,
                    peakfreq,
                    ncomps=args.hilbertcomponents,
                )
            )
        else:
            filthiresfund = tide_math.madnormalize(
                happy_support.getperiodic(
                    cardiacwaveform,
                    slicesamplerate,
                    peakfreq,
                    ncomps=args.hilbertcomponents,
                )
            )
        if args.outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    filthiresfund,
                    slicesamplerate,
                    columns=["cardiacfundamental"],
                    append=True,
                    debug=args.debug,
                )

        # now calculate the phase waveform
        tide_util.logmem("before analytic phase analysis")
        instantaneous_cardiacphase, amplitude_envelope, analytic_signal = tide_fit.phaseanalysis(
            filthiresfund
        )
        if args.outputlevel > 0:
            if thispass == numpasses - 1:
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    amplitude_envelope,
                    slicesamplerate,
                    columns=["envelope"],
                    append=True,
                    debug=args.debug,
                )
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    analytic_signal.real,
                    slicesamplerate,
                    columns=["analytic_real"],
                    append=True,
                    debug=args.debug,
                )
                tide_io.writebidstsv(
                    outputroot + "_desc-slicerescardfromfmri_timeseries",
                    instantaneous_cardiacphase,
                    slicesamplerate,
                    columns=["instphase_unwrapped"],
                    append=True,
                    debug=args.debug,
                )

        if args.filtphase:
            print("Filtering phase waveform")
            instantaneous_cardiacphase = tide_math.trendfilt(
                instantaneous_cardiacphase, debug=False
            )
            if args.outputlevel > 1:
                if thispass == numpasses - 1:
                    tide_io.writebidstsv(
                        outputroot + "_desc-slicerescardfromfmri_timeseries",
                        instantaneous_cardiacphase,
                        slicesamplerate,
                        columns=["filtered_instphase_unwrapped"],
                        append=True,
                        debug=args.debug,
                    )
        initialphase = instantaneous_cardiacphase[0]
        infodict["phi0"] = initialphase
        timings.append(["Phase waveform generated" + passstring, time.time(), None, None])

        # get the respiration signal from a file, if specified
        respiration = True
        if respiration:
            if args.respirationfilename is not None:
                tide_util.logmem("before respirationfromfile")
                (
                    pleth_sliceres,
                    pleth_stdres,
                    returnedinputfreq,
                    fullrespts,
                ) = happy_support.getphysiofile(
                    args.cardiacfilename,
                    args.respinputfreq,
                    args.respinputstart,
                    slicetimeaxis,
                    args.stdfreq,
                    len(cardfromfmri_stdres),
                    args.envcutoff,
                    args.envthresh,
                    timings,
                    outputroot,
                    iscardiac=False,
                    outputlevel=args.outputlevel,
                    debug=args.debug,
                )
                infodict["respirationfromfmri"] = False
                infodict["numresppts_sliceres"] = len(pleth_sliceres)
                infodict["numresppts_stdres"] = len(pleth_stdres)
                infodict["respsamplerate"] = returnedinputfreq
                infodict["numresppts_fullres"] = fullrespts

        # account for slice time offsets
        offsets_byslice = np.zeros((xsize * ysize, numslices), dtype=np.float64)
        for i in range(numslices):
            offsets_byslice[:, i] = slicetimes[i]

        # remap offsets to space by time
        fmri_offsets = offsets_byslice.reshape(numspatiallocs)

        # save the information file
        if args.saveinfoasjson:
            tide_io.writedicttojson(infodict, outputroot + "_info.json")
        else:
            tide_io.writedict(infodict, outputroot + "_info.txt")

        # interpolate the instantaneous phase
        print(f"{tr=}")
        print(f"{timepoints=}")
        print(f"{numsteps=}")
        print(f"{args.upsamplefac=}")
        upsampledslicetimeaxis = np.linspace(
            0.0,
            tr * timepoints,
            num=(timepoints * numsteps * args.upsamplefac),
            endpoint=False,
        )
        interpphase = tide_math.phasemod(
            tide_resample.doresample(
                slicetimeaxis,
                instantaneous_cardiacphase,
                upsampledslicetimeaxis,
                method="univariate",
                padlen=0,
            ),
            centric=args.centric,
        )
        if args.outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.writebidstsv(
                    outputroot + "_desc-interpinstphase_timeseries",
                    interpphase,
                    1.0 / (upsampledslicetimeaxis[1] - upsampledslicetimeaxis[0]),
                    starttime=upsampledslicetimeaxis[0],
                    columns=["instphase"],
                    append=False,
                    debug=args.debug,
                )

        if args.cardcalconly:
            print("Cardiac waveform calculations done - exiting")
            # Process and save timing information
            nodeline = "Processed on " + platform.node()
            tide_util.proctiminginfo(
                timings, outputfile=outputroot + "_runtimings.txt", extraheader=nodeline
            )
            tide_util.logmem("final")

        # find the phase values for all timepoints in all slices
        cardphasevals = np.zeros((numslices, timepoints), dtype=np.float64)
        thetimes = []
        for theslice in range(numslices):
            thetimes.append(
                np.linspace(0.0, tr * timepoints, num=timepoints, endpoint=False)
                + slicetimes[theslice]
            )
            cardphasevals[theslice, :] = tide_math.phasemod(
                tide_resample.doresample(
                    slicetimeaxis,
                    instantaneous_cardiacphase,
                    thetimes[-1],
                    method="univariate",
                    padlen=0,
                ),
                centric=args.centric,
            )
            if args.debug:
                if thispass == numpasses - 1:
                    tide_io.writevec(
                        thetimes[-1],
                        outputroot + "_times_" + str(theslice).zfill(2) + ".txt",
                    )
                    tide_io.writevec(
                        cardphasevals[theslice, :],
                        outputroot + "_cardphasevals_" + str(theslice).zfill(2) + ".txt",
                    )
        timings.append(
            [
                "Slice phases determined for all timepoints" + passstring,
                time.time(),
                None,
                None,
            ]
        )

        # construct the destination arrays
        tide_util.logmem("before making destination arrays")
        app = np.zeros((xsize, ysize, numslices, args.destpoints), dtype=np.float64)
        cine = np.zeros((xsize, ysize, numslices, args.destpoints), dtype=np.float64)
        rawapp = np.zeros((xsize, ysize, numslices, args.destpoints), dtype=np.float64)
        corrected_rawapp = np.zeros((xsize, ysize, numslices, args.destpoints), dtype=np.float64)
        normapp = np.zeros((xsize, ysize, numslices, args.destpoints), dtype=np.float64)
        weights = np.zeros((xsize, ysize, numslices, args.destpoints), dtype=np.float64)
        derivatives = np.zeros((xsize, ysize, numslices, 4), dtype=np.float64)
        timings.append(["Output arrays allocated" + passstring, time.time(), None, None])

        if args.centric:
            outphases = np.linspace(-np.pi, np.pi, num=args.destpoints, endpoint=False)
        else:
            outphases = np.linspace(0.0, 2.0 * np.pi, num=args.destpoints, endpoint=False)
        phasestep = outphases[1] - outphases[0]

        #######################################################################################################
        #
        # now do the phase projection
        #
        #
        app_byslice = app.reshape((xsize * ysize, numslices, args.destpoints))
        rawapp_byslice = rawapp.reshape((xsize * ysize, numslices, args.destpoints))
        corrected_rawapp_byslice = corrected_rawapp.reshape(
            (xsize * ysize, numslices, args.destpoints)
        )
        normapp_byslice = normapp.reshape((xsize * ysize, numslices, args.destpoints))
        weights_byslice = weights.reshape((xsize * ysize, numslices, args.destpoints))
        cine_byslice = cine.reshape((xsize * ysize, numslices, args.destpoints))
        derivatives_byslice = derivatives.reshape((xsize * ysize, numslices, 4))

        demeandata_byslice = demeandata.reshape((xsize * ysize, numslices, timepoints))
        means_byslice = means.reshape((xsize * ysize, numslices))

        timings.append(
            ["Phase projection to image prep started" + passstring, time.time(), None, None]
        )
        print("Starting phase projection")
        proctrs = range(timepoints)  # proctrs is the list of all fmri trs to be projected
        procpoints = range(
            timepoints * numsteps
        )  # procpoints is the list of all sliceres datapoints to be projected
        if args.censorbadpts:
            censortrs = np.zeros(timepoints, dtype="int")
            censorpoints = np.zeros(timepoints * numsteps, dtype="int")
            censortrs[np.where(badpointlist > 0.0)[0] // numsteps] = 1
            censorpoints[np.where(badpointlist > 0.0)[0]] = 1
            proctrs = np.where(censortrs < 1)[0]
            procpoints = np.where(censorpoints < 1)[0]

        # preload congrid
        if args.preloadcongrid:
            print("Preloading congrid values...")
            happy_support.preloadcongrid(
                outphases, args.congridbins, gridkernel=args.gridkernel, cyclic=True, debug=False
            )
            print("done")

        # do phase averaging
        app_bypoint, weights_bypoint = happy_support.cardiaccycleaverage(
            instantaneous_cardiacphase,
            outphases,
            cardfromfmri_sliceres,
            procpoints,
            args.congridbins,
            args.gridkernel,
            args.centric,
            cache=args.congridcache,
            cyclic=True,
        )
        if thispass == numpasses - 1:
            tide_io.writebidstsv(
                outputroot + "_desc-cardiaccyclefromfmri_timeseries",
                app_bypoint,
                1.0 / (outphases[1] - outphases[0]),
                starttime=outphases[0],
                columns=["cardiaccyclefromfmri"],
                append=False,
                debug=args.debug,
            )
            tide_io.writebidstsv(
                outputroot + "_desc-cardiaccycleweightfromfmri_timeseries",
                weights_bypoint,
                1.0 / (outphases[1] - outphases[0]),
                starttime=outphases[0],
                columns=["cardiaccycleweightfromfmri"],
                append=False,
                debug=args.debug,
            )

        # now do time averaging
        lookaheadval = int(slicesamplerate / 4.0)
        print("lookaheadval = ", lookaheadval)
        wrappedcardiacphase = tide_math.phasemod(instantaneous_cardiacphase, centric=args.centric)
        max_peaks, min_peaks = tide_fit.peakdetect(wrappedcardiacphase, lookahead=lookaheadval)
        # start on a maximum
        if max_peaks[0][0] > min_peaks[0][0]:
            min_peaks = min_peaks[1:]
        # work only with pairs
        if len(max_peaks) > len(min_peaks):
            max_peaks = max_peaks[:-1]

        zerophaselocs = []
        for idx, peak in enumerate(max_peaks):
            minloc = min_peaks[idx][0]
            maxloc = max_peaks[idx][0]
            minval = min_peaks[idx][1]
            maxval = max_peaks[idx][1]
            if minloc > 0:
                if wrappedcardiacphase[minloc - 1] < wrappedcardiacphase[minloc]:
                    minloc -= 1
                    minval = wrappedcardiacphase[minloc]
            phasediff = minval - (maxval - 2.0 * np.pi)
            timediff = minloc - maxloc
            zerophaselocs.append(1.0 * minloc - (minval - outphases[0]) * timediff / phasediff)
            # print(idx, [maxloc, maxval], [minloc, minval], phasediff, timediff, zerophaselocs[-1])
        instantaneous_cardiactime = instantaneous_cardiacphase * 0.0

        whichpeak = 0
        for t in procpoints:
            if whichpeak < len(zerophaselocs) - 1:
                if t > zerophaselocs[whichpeak + 1]:
                    whichpeak += 1
            if t > zerophaselocs[whichpeak]:
                instantaneous_cardiactime[t] = (t - zerophaselocs[whichpeak]) / slicesamplerate
            # print(t, whichpeak, zerophaselocs[whichpeak], instantaneous_cardiactime[t])
        maxtime = (
            np.ceil(
                int(
                    1.02
                    * tide_stats.getfracval(instantaneous_cardiactime, 0.98)
                    // args.pulsereconstepsize
                )
            )
            * args.pulsereconstepsize
        )
        outtimes = np.linspace(
            0.0, maxtime, num=int(maxtime / args.pulsereconstepsize), endpoint=False
        )
        atp_bypoint, atpweights_bypoint = happy_support.cardiaccycleaverage(
            instantaneous_cardiactime,
            outtimes,
            cardfromfmri_sliceres,
            procpoints,
            args.congridbins,
            args.gridkernel,
            False,
            cyclic=True,
        )
        if thispass == numpasses - 1:
            tide_io.writebidstsv(
                outputroot + "_desc-cardpulsefromfmri_timeseries",
                atp_bypoint,
                1.0 / (outtimes[1] - outtimes[0]),
                starttime=outtimes[0],
                columns=["pulsefromfmri"],
                append=False,
                debug=args.debug,
            )

        timings.append(
            ["Phase projection to image prep ended" + passstring, time.time(), None, None]
        )
        if not args.verbose:
            print("Setting up for phase projection...")

        # make a vessel map using Wright's method
        if args.wrightiterations > 0:
            timings.append(
                [
                    "Wright mask generation started" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )
            wrightcorrs = happy_support.wrightmap(
                input_data,
                demeandata_byslice,
                rawapp_byslice,
                projmask_byslice,
                outphases,
                cardphasevals,
                proctrs,
                args.congridbins,
                args.gridkernel,
                args.destpoints,
                iterations=args.wrightiterations,
                nprocs=args.nprocs,
                verbose=False,
                debug=args.debug,
            )

            theheader = input_data.copyheader(numtimepoints=1)
            wrightfilename = f"{outputroot}_desc-wrightcorrspass{thispass + 1}_map"
            tide_io.writedicttojson(bidsbasedict, wrightfilename + ".json")
            tide_io.savetonifti(wrightcorrs, theheader, wrightfilename)
            timings.append(
                [
                    "Wright mask generation completed" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )

        timings.append(["Phase projection to image started" + passstring, time.time(), None, None])
        # make a lowpass filter for the projected data. Limit frequency to 3 cycles per 2pi (1/6th Fs)
        phaseFs = 1.0 / phasestep
        phaseFc = phaseFs / 6.0
        appsmoothingfilter = tide_filt.NoncausalFilter("arb", cyclic=True, padtime=0.0)
        appsmoothingfilter.setfreqs(0.0, 0.0, phaseFc, phaseFc)

        # setup for aliased correlation if we're going to do it
        if args.doaliasedcorrelation:
            if args.cardiacfilename and False:
                signal_sliceres = pleth_sliceres
            else:
                signal_sliceres = cardfromfmri_sliceres

            # zero out bad points
            signal_sliceres *= 1.0 - badpointlist

            theAliasedCorrelator = tide_corr.AliasedCorrelator(
                signal_sliceres,
                slicesamplerate,
                numsteps,
            )
            thealiasedcorrx = theAliasedCorrelator.getxaxis()
            corrstartloc = tide_util.valtoindex(
                thealiasedcorrx, -args.aliasedcorrelationwidth / 2.0
            )
            correndloc = tide_util.valtoindex(thealiasedcorrx, args.aliasedcorrelationwidth / 2.0)
            aliasedcorrelationpts = correndloc - corrstartloc + 1
            if thispass == 0:
                thecorrfunc = np.zeros(
                    (xsize, ysize, numslices, aliasedcorrelationpts), dtype=np.float64
                )
                thecorrfunc_byslice = thecorrfunc.reshape(
                    (xsize * ysize, numslices, aliasedcorrelationpts)
                )
                wavedelay = np.zeros((xsize, ysize, numslices), dtype=np.float64)
                wavedelay_byslice = wavedelay.reshape((xsize * ysize, numslices))
                wavedelayCOM = np.zeros((xsize, ysize, numslices), dtype=np.float64)
                wavedelayCOM_byslice = wavedelayCOM.reshape((xsize * ysize, numslices))
                waveamp = np.zeros((xsize, ysize, numslices), dtype=np.float64)
                waveamp_byslice = waveamp.reshape((xsize * ysize, numslices))
            else:
                thecorrfunc *= 0.0
                wavedelay *= 0.0
                wavedelayCOM *= 0.0
                waveamp *= 0.0
        else:
            thecorrfunc_byslice = None
            waveamp_byslice = None
            wavedelay_byslice = None
            wavedelayCOM_byslice = None
            corrstartloc = None
            correndloc = None
            thealiasedcorrx = None
            theAliasedCorrelator = None

        # now project the data
        appflips_byslice = happy_support.phaseproject(
            input_data,
            demeandata_byslice,
            means_byslice,
            rawapp_byslice,
            app_byslice,
            normapp_byslice,
            weights_byslice,
            cine_byslice,
            projmask_byslice,
            derivatives_byslice,
            proctrs,
            thispass,
            args,
            sliceoffsets,
            cardphasevals,
            outphases,
            appsmoothingfilter,
            phaseFs,
            thecorrfunc_byslice,
            waveamp_byslice,
            wavedelay_byslice,
            wavedelayCOM_byslice,
            corrected_rawapp_byslice,
            corrstartloc,
            correndloc,
            thealiasedcorrx,
            theAliasedCorrelator,
        )
        if not args.verbose:
            print(" done")
        timings.append(
            [
                "Phase projection to image completed" + passstring,
                time.time(),
                None,
                None,
            ]
        )
        print("Phase projection done")

        # calculate the flow field from the normapp
        if args.doflowfields:
            print("calculating flow fields")
            flowhdr = input_data.copyheader(numtimepoints=3, tr=1.0, toffset=0.0)

            flowfield = happy_support.calc_3d_optical_flow(
                app,
                projmask.reshape(xsize, ysize, numslices),
                flowhdr,
                outputroot,
                window_size=3,
                debug=True,
            )
            print(f"flow field shape: {flowfield.shape}")

        # save the analytic phase projection image
        theheader = input_data.copyheader(
            numtimepoints=args.destpoints, tr=-np.pi, toffset=2.0 * np.pi / args.destpoints
        )
        if thispass == numpasses - 1:
            appfilename = outputroot + "_desc-app_info"
            normappfilename = outputroot + "_desc-normapp_info"
            cinefilename = outputroot + "_desc-cine_info"
            rawappfilename = outputroot + "_desc-rawapp_info"
            bidsdict = bidsbasedict.copy()
            bidsdict["Units"] = "second"
            tide_io.writedicttojson(bidsdict, appfilename + ".json")
            tide_io.writedicttojson(bidsdict, normappfilename + ".json")
            tide_io.writedicttojson(bidsdict, cinefilename + ".json")
            if args.outputlevel > 0:
                tide_io.writedicttojson(bidsdict, rawappfilename + ".json")
            tide_io.savetonifti(app, theheader, appfilename)
            tide_io.savetonifti(normapp, theheader, normappfilename)
            tide_io.savetonifti(cine, theheader, cinefilename)
            if args.outputlevel > 0:
                tide_io.savetonifti(rawapp, theheader, rawappfilename)

        timings.append(["Phase projected data saved" + passstring, time.time(), None, None])

        if args.doaliasedcorrelation and thispass == numpasses - 1:
            theheader = input_data.copyheader(
                numtimepoints=aliasedcorrelationpts,
                tr=(thealiasedcorrx[1] - thealiasedcorrx[0]),
                toffset=0.0,
            )
            corrfuncfilename = outputroot + "_desc-corrfunc_info"
            wavedelayfilename = outputroot + "_desc-wavedelay_map"
            wavedelayCOMfilename = outputroot + "_desc-wavedelayCOM_map"
            waveampfilename = outputroot + "_desc-waveamp_map"
            bidsdict = bidsbasedict.copy()
            tide_io.writedicttojson(bidsdict, waveampfilename + ".json")
            bidsdict["Units"] = "second"
            tide_io.writedicttojson(bidsdict, corrfuncfilename + ".json")
            tide_io.writedicttojson(bidsdict, wavedelayfilename + ".json")
            tide_io.writedicttojson(bidsdict, wavedelayCOMfilename + ".json")
            tide_io.savetonifti(thecorrfunc, theheader, corrfuncfilename)
            theheader["dim"][4] = 1
            tide_io.savetonifti(wavedelay, theheader, wavedelayfilename)
            tide_io.savetonifti(wavedelayCOM, theheader, wavedelayCOMfilename)
            tide_io.savetonifti(waveamp, theheader, waveampfilename)

        # make and save a voxel intensity histogram
        if args.unnormvesselmap:
            app2d = app.reshape((numspatiallocs, args.destpoints))
        else:
            app2d = normapp.reshape((numspatiallocs, args.destpoints))
        validlocs = np.where(mask > 0)[0]
        histinput = app2d[validlocs, :].reshape((len(validlocs), args.destpoints))
        if args.outputlevel > 0:
            namesuffix = "_desc-apppeaks_hist"
            tide_stats.makeandsavehistogram(
                histinput,
                args.histlen,
                0,
                outputroot + namesuffix,
                debug=args.debug,
            )

        # find vessel thresholds
        tide_util.logmem("before making vessel masks")
        hardvesselthresh = tide_stats.getfracvals(np.max(histinput, axis=1), [0.98])[0] / 2.0
        softvesselthresh = args.softvesselfrac * hardvesselthresh
        print(
            "hard, soft vessel thresholds set to",
            "{:.3f}".format(hardvesselthresh),
            "{:.3f}".format(softvesselthresh),
        )

        # save a vessel masked version of app
        if args.unnormvesselmap:
            vesselmask = np.where(np.max(app, axis=3) > softvesselthresh, 1, 0)
        else:
            vesselmask = np.where(np.max(normapp, axis=3) > softvesselthresh, 1, 0)
        maskedapp2d = np.array(app2d)
        maskedapp2d[np.where(vesselmask.reshape(numspatiallocs) == 0)[0], :] = 0.0
        if args.outputlevel > 1:
            if thispass == numpasses - 1:
                maskedappfilename = outputroot + "_desc-maskedapp_info"
                bidsdict = bidsbasedict.copy()
                bidsdict["Units"] = "second"
                tide_io.writedicttojson(bidsdict, maskedappfilename + ".json")
                tide_io.savetonifti(
                    maskedapp2d.reshape((xsize, ysize, numslices, args.destpoints)),
                    theheader,
                    maskedappfilename,
                )
        del maskedapp2d
        timings.append(
            [
                "Vessel masked phase projected data saved" + passstring,
                time.time(),
                None,
                None,
            ]
        )

        # save multiple versions of the hard vessel mask
        if args.unnormvesselmap:
            vesselmask = np.where(np.max(app, axis=3) > hardvesselthresh, 1, 0)
            minphase = np.argmin(app, axis=3) * 2.0 * np.pi / args.destpoints - np.pi
            maxphase = np.argmax(app, axis=3) * 2.0 * np.pi / args.destpoints - np.pi
        else:
            vesselmask = np.where(np.max(normapp, axis=3) > hardvesselthresh, 1, 0)
            minphase = np.argmin(normapp, axis=3) * 2.0 * np.pi / args.destpoints - np.pi
            maxphase = np.argmax(normapp, axis=3) * 2.0 * np.pi / args.destpoints - np.pi
        risediff = (maxphase - minphase) * vesselmask
        arteries = np.where(appflips_byslice.reshape((xsize, ysize, numslices)) < 0, vesselmask, 0)
        veins = np.where(appflips_byslice.reshape((xsize, ysize, numslices)) > 0, vesselmask, 0)
        theheader = input_data.copyheader(numtimepoints=1)
        if thispass == numpasses - 1:
            vesselmaskfilename = outputroot + "_desc-vessels_mask"
            minphasefilename = outputroot + "_desc-minphase_map"
            maxphasefilename = outputroot + "_desc-maxphase_map"
            arterymapfilename = outputroot + "_desc-arteries_map"
            veinmapfilename = outputroot + "_desc-veins_map"
            bidsdict = bidsbasedict.copy()
            tide_io.writedicttojson(bidsdict, vesselmaskfilename + ".json")
            tide_io.savetonifti(vesselmask, theheader, vesselmaskfilename)
            if args.outputlevel > 0:
                tide_io.writedicttojson(bidsdict, arterymapfilename + ".json")
                tide_io.writedicttojson(bidsdict, veinmapfilename + ".json")
                bidsdict["Units"] = "radians"
                tide_io.writedicttojson(bidsdict, minphasefilename + ".json")
                tide_io.writedicttojson(bidsdict, maxphasefilename + ".json")
                tide_io.savetonifti(minphase, theheader, minphasefilename)
                tide_io.savetonifti(maxphase, theheader, maxphasefilename)
                tide_io.savetonifti(arteries, theheader, arterymapfilename)
                tide_io.savetonifti(veins, theheader, veinmapfilename)
        timings.append(["Masks saved" + passstring, time.time(), None, None])

        # save the mask we used for this pass
        tide_io.savetonifti(
            estweights_byslice.reshape((xsize, ysize, numslices)),
            theheader,
            f"{outputroot}_desc-estweightspass{thispass}_map",
        )

        # now get ready to start again with a new mask
        if args.doaliasedcorrelation and thispass > 0:
            estweights_byslice = waveamp_byslice * vesselmask.reshape((xsize * ysize, numslices))
        else:
            estweights_byslice = vesselmask.reshape((xsize * ysize, numslices)) + 0

    # save a vessel image
    if args.unnormvesselmap:
        vesselmap = np.max(app, axis=3)
    else:
        vesselmap = np.max(normapp, axis=3)
    vesselmapfilename = outputroot + "_desc-vessels_map"
    arterymapfilename = outputroot + "_desc-arteries_map"
    veinmapfilename = outputroot + "_desc-veins_map"
    tide_io.savetonifti(vesselmap, theheader, vesselmapfilename)
    tide_io.savetonifti(
        np.where(appflips_byslice.reshape((xsize, ysize, numslices)) < 0, vesselmap, 0.0),
        theheader,
        arterymapfilename,
    )
    tide_io.savetonifti(
        np.where(appflips_byslice.reshape((xsize, ysize, numslices)) > 0, vesselmap, 0.0),
        theheader,
        veinmapfilename,
    )

    # now generate aliased cardiac signals and regress them out of the data
    if args.dotemporalregression or args.dospatialregression:
        # generate the signals
        timings.append(["Cardiac signal regression started", time.time(), None, None])
        tide_util.logmem("before cardiac regression")
        print("Generating cardiac regressors")
        cardiacnoise = fmri_data * 0.0
        cardiacnoise_byslice = cardiacnoise.reshape((xsize * ysize, numslices, timepoints))
        phaseindices = (cardiacnoise * 0.0).astype(np.int16)
        phaseindices_byslice = phaseindices.reshape((xsize * ysize, numslices, timepoints))
        for theslice in range(numslices):
            print("Calculating cardiac noise for slice", theslice)
            validlocs = np.where(projmask_byslice[:, theslice] > 0)[0]
            for t in range(timepoints):
                phaseindices_byslice[validlocs, theslice, t] = tide_util.valtoindex(
                    outphases, cardphasevals[theslice, t]
                )
                cardiacnoise_byslice[validlocs, theslice, t] = rawapp_byslice[
                    validlocs, theslice, phaseindices_byslice[validlocs, theslice, t]
                ]
        theheader = input_data.copyheader()
        timings.append(["Cardiac signal generated", time.time(), None, None])
        if args.savecardiacnoise:
            cardiacnoisefilename = outputroot + "_desc-cardiacnoise_info"
            phaseindexfilename = outputroot + "_desc-phaseindices_info"
            tide_io.savetonifti(
                cardiacnoise.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                cardiacnoisefilename,
            )
            tide_io.savetonifti(
                phaseindices.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                phaseindexfilename,
            )
            timings.append(["Cardiac signal saved", time.time(), None, None])

        # now remove them
        tide_util.logmem("before cardiac removal")
        print("Removing cardiac signal with GLM")
        filtereddata = 0.0 * fmri_data
        validlocs = np.where(mask > 0)[0]
        numvalidspatiallocs = len(validlocs)
        threshval = 0.0
        if args.dospatialregression:
            meanvals = np.zeros(timepoints, dtype=np.float64)
            rvals = np.zeros(timepoints, dtype=np.float64)
            r2vals = np.zeros(timepoints, dtype=np.float64)
            fitcoffs = np.zeros(timepoints, dtype=np.float64)
            fitNorm = np.zeros(timepoints, dtype=np.float64)
            datatoremove = 0.0 * fmri_data
            print("Running spatial regression on", timepoints, "timepoints")
            tide_util.disablemkl(args.nprocs)
            tide_linfitfiltpass.linfitfiltpass(
                timepoints,
                fmri_data[validlocs, :],
                threshval,
                cardiacnoise[validlocs, :],
                meanvals,
                rvals,
                r2vals,
                fitcoffs,
                fitNorm,
                datatoremove[validlocs, :],
                filtereddata[validlocs, :],
                mp_chunksize=10,
                procbyvoxel=False,
                nprocs=args.nprocs,
            )
            tide_util.enablemkl(args.mklthreads)
            print(datatoremove.shape, cardiacnoise.shape, fitcoffs.shape)
            # datatoremove[validlocs, :] = np.multiply(cardiacnoise[validlocs, :], fitcoffs[:, None])
            filtereddata = fmri_data - datatoremove
            timings.append(
                [
                    "Cardiac signal spatial regression finished",
                    time.time(),
                    timepoints,
                    "timepoints",
                ]
            )
            tide_io.writevec(fitcoffs, outputroot + "_fitcoff.txt")
            tide_io.writevec(meanvals, outputroot + "_fitmean.txt")
            tide_io.writevec(rvals, outputroot + "_fitR.txt")
            theheader = input_data.copyheader()
            cardfiltresultfilename = outputroot + "_desc-cardfiltResult_bold"
            cardfiltremovedfilename = outputroot + "_desc-cardfiltRemoved_bold"
            tide_io.savetonifti(
                filtereddata.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                cardfiltresultfilename,
            )
            tide_io.savetonifti(
                datatoremove.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                cardfiltremovedfilename,
            )
            timings.append(
                [
                    "Cardiac signal spatial regression files written",
                    time.time(),
                    None,
                    None,
                ]
            )

        if args.dotemporalregression:
            meanvals = np.zeros(numspatiallocs, dtype=np.float64)
            rvals = np.zeros(numspatiallocs, dtype=np.float64)
            r2vals = np.zeros(numspatiallocs, dtype=np.float64)
            fitcoffs = np.zeros(numspatiallocs, dtype=np.float64)
            fitNorm = np.zeros(numspatiallocs, dtype=np.float64)
            datatoremove = 0.0 * fmri_data
            print("Running temporal regression on", numvalidspatiallocs, "voxels")
            tide_util.disablemkl(args.nprocs)
            tide_linfitfiltpass.linfitfiltpass(
                numvalidspatiallocs,
                fmri_data[validlocs, :],
                threshval,
                cardiacnoise[validlocs, :],
                meanvals[validlocs],
                rvals[validlocs],
                r2vals[validlocs],
                fitcoffs[validlocs],
                fitNorm[validlocs],
                datatoremove[validlocs, :],
                filtereddata[validlocs, :],
                procbyvoxel=True,
                nprocs=args.nprocs,
            )
            tide_util.enablemkl(args.mklthreads)
            datatoremove[validlocs, :] = np.multiply(
                cardiacnoise[validlocs, :], fitcoffs[validlocs, None]
            )
            filtereddata[validlocs, :] = fmri_data[validlocs, :] - datatoremove[validlocs, :]
            timings.append(
                [
                    "Cardiac signal temporal regression finished",
                    time.time(),
                    numspatiallocs,
                    "voxels",
                ]
            )
            theheader = input_data.copyheader(numtimepoints=1)
            cardfiltcoeffsfilename = outputroot + "_desc-cardfiltCoeffs_map"
            cardfiltmeanfilename = outputroot + "_desc-cardfiltMean_map"
            cardfiltRfilename = outputroot + "_desc-cardfiltR_map"
            tide_io.savetonifti(
                fitcoffs.reshape((xsize, ysize, numslices)),
                theheader,
                cardfiltcoeffsfilename,
            )
            tide_io.savetonifti(
                meanvals.reshape((xsize, ysize, numslices)),
                theheader,
                cardfiltmeanfilename,
            )
            tide_io.savetonifti(
                rvals.reshape((xsize, ysize, numslices)), theheader, cardfiltRfilename
            )

            theheader = input_data.copyheader()
            cardfiltresultfilename = outputroot + "_desc-cardfiltResult_bold"
            cardfiltremovedfilename = outputroot + "_desc-cardfiltRemoved_bold"
            tide_io.savetonifti(
                filtereddata.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                cardfiltresultfilename,
            )
            tide_io.savetonifti(
                datatoremove.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                cardfiltremovedfilename,
            )
            timings.append(
                [
                    "Cardiac signal temporal regression files written",
                    time.time(),
                    None,
                    None,
                ]
            )

    timings.append(["Done", time.time(), None, None])

    # Process and save timing information
    nodeline = "Processed on " + platform.node()
    tide_util.proctiminginfo(
        timings, outputfile=outputroot + "_runtimings.txt", extraheader=nodeline
    )

    tide_util.logmem("final")

    # delete the canary file
    Path(f"{outputroot}_ISRUNNING.txt").unlink()

    # create the finished file
    Path(f"{outputroot}_DONE.txt").touch()
