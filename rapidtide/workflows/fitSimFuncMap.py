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
import numpy as np
from scipy import ndimage

import rapidtide.io as tide_io
import rapidtide.patchmatch as tide_patch
import rapidtide.peakeval as tide_peakeval
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.util as tide_util


def fitSimFunc(
    fmri_data_valid,
    validsimcalcstart,
    validsimcalcend,
    osvalidsimcalcstart,
    osvalidsimcalcend,
    initial_fmri_x,
    os_fmri_x,
    theMutualInformationator,
    cleaned_referencetc,
    corrout,
    outputname,
    validvoxels,
    nativespaceshape,
    bidsbasedict,
    numspatiallocs,
    gaussout,
    theinitialdelay,
    windowout,
    R2,
    thesizes,
    internalspaceshape,
    numvalidspatiallocs,
    theinputdata,
    theheader,
    theFitter,
    fitmask,
    lagtimes,
    lagstrengths,
    lagsigma,
    failreason,
    outmaparray,
    trimmedcorrscale,
    similaritytype,
    thepass,
    optiondict,
    LGR,
    TimingLGR,
    simplefit=False,
    upsampfac=8,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    # Do a peak prefit if doing hybrid
    if optiondict["similaritymetric"] == "hybrid":
        LGR.info(f"\n\nPeak prefit calculation, pass {thepass}")
        TimingLGR.info(f"Peak prefit calculation start, pass {thepass}")

        tide_util.disablemkl(optiondict["nprocs_peakeval"], debug=optiondict["threaddebug"])
        voxelsprocessed_pe, thepeakdict = tide_peakeval.peakevalpass(
            fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
            cleaned_referencetc,
            initial_fmri_x[validsimcalcstart : validsimcalcend + 1],
            os_fmri_x[osvalidsimcalcstart : osvalidsimcalcend + 1],
            theMutualInformationator,
            trimmedcorrscale,
            corrout,
            nprocs=optiondict["nprocs_peakeval"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            bipolar=optiondict["bipolar"],
            oversampfactor=optiondict["oversampfactor"],
            interptype=optiondict["interptype"],
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )
        tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])

        TimingLGR.info(
            f"Peak prefit end, pass {thepass}",
            {
                "message2": voxelsprocessed_pe,
                "message3": "voxels",
            },
        )
        mipeaks = lagtimes * 0.0
        for i in range(numvalidspatiallocs):
            if len(thepeakdict[str(i)]) > 0:
                mipeaks[i] = thepeakdict[str(i)][0][0]
    else:
        thepeakdict = None

    if simplefit:
        basedelay = trimmedcorrscale[0]
        delaystep = (trimmedcorrscale[1] - trimmedcorrscale[0]) / upsampfac
        for thevox in range(numvalidspatiallocs):
            fitmask[thevox] = 1
            upsampcorrout = tide_resample.upsample(corrout[thevox,:],1, upsampfac, intfac=True, dofilt=False)
            if optiondict["bipolar"]:
                thismax = np.argmax(np.fabs(upsampcorrout))
            else:
                thismax = np.argmax(upsampcorrout)
            lagtimes[thevox] = basedelay + thismax * delaystep
            lagstrengths[thevox] = upsampcorrout[thismax]
            lagsigma[thevox] = 1.0
        internaldespeckleincludemask = None
    else:
        # Similarity function fitting and time lag estimation
        # write out the current version of the run options
        optiondict["currentstage"] = f"presimfuncfit_pass{thepass}"
        tide_io.writedicttojson(optiondict, f"{outputname}_desc-runoptions_info.json")
        LGR.info(f"\n\nTime lag estimation pass {thepass}")
        TimingLGR.info(f"Time lag estimation start, pass {thepass}")

        theFitter.setfunctype(optiondict["similaritymetric"])
        theFitter.setcorrtimeaxis(trimmedcorrscale)

        # use initial lags if this is a hybrid fit
        if optiondict["similaritymetric"] == "hybrid" and thepeakdict is not None:
            initlags = mipeaks
        else:
            initlags = None

        tide_util.disablemkl(optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"])
        voxelsprocessed_fc = tide_simfuncfit.fitcorr(
            trimmedcorrscale,
            theFitter,
            corrout,
            fitmask,
            failreason,
            lagtimes,
            lagstrengths,
            lagsigma,
            gaussout,
            windowout,
            R2,
            despeckling=False,
            peakdict=thepeakdict,
            nprocs=optiondict["nprocs_fitcorr"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            fixdelay=optiondict["fixdelay"],
            initialdelayvalue=theinitialdelay,
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            despeckle_thresh=optiondict["despeckle_thresh"],
            initiallags=initlags,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )
        tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])

        TimingLGR.info(
            f"Time lag estimation end, pass {thepass}",
            {
                "message2": voxelsprocessed_fc,
                "message3": "voxels",
            },
        )

        # Correlation time despeckle
        if optiondict["despeckle_passes"] > 0:
            LGR.info(f"\n\n{similaritytype} despeckling pass {thepass}")
            LGR.info(f"\tUsing despeckle_thresh = {optiondict['despeckle_thresh']:.3f}")
            TimingLGR.info(f"{similaritytype} despeckle start, pass {thepass}")

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            lastnumdespeckled = 1000000
            for despecklepass in range(optiondict["despeckle_passes"]):
                LGR.info(f"\n\n{similaritytype} despeckling subpass {despecklepass + 1}")
                outmaparray *= 0.0
                outmaparray[validvoxels] = eval("lagtimes")[:]

                # find voxels to despeckle
                medianlags = ndimage.median_filter(outmaparray.reshape(nativespaceshape), 3).reshape(
                    numspatiallocs
                )
                # voxels that we're happy with have initlags set to -1000000.0
                initlags = np.where(
                    np.abs(outmaparray - medianlags) > optiondict["despeckle_thresh"],
                    medianlags,
                    -1000000.0,
                )[validvoxels]

                if len(initlags) > 0:
                    numdespeckled = len(np.where(initlags != -1000000.0)[0])
                    if lastnumdespeckled > numdespeckled > 0:
                        lastnumdespeckled = numdespeckled
                        tide_util.disablemkl(
                            optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"]
                        )
                        voxelsprocessed_thispass = tide_simfuncfit.fitcorr(
                            trimmedcorrscale,
                            theFitter,
                            corrout,
                            fitmask,
                            failreason,
                            lagtimes,
                            lagstrengths,
                            lagsigma,
                            gaussout,
                            windowout,
                            R2,
                            despeckling=True,
                            peakdict=thepeakdict,
                            nprocs=optiondict["nprocs_fitcorr"],
                            alwaysmultiproc=optiondict["alwaysmultiproc"],
                            fixdelay=optiondict["fixdelay"],
                            initialdelayvalue=theinitialdelay,
                            showprogressbar=optiondict["showprogressbar"],
                            chunksize=optiondict["mp_chunksize"],
                            despeckle_thresh=optiondict["despeckle_thresh"],
                            initiallags=initlags,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                        tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])

                        voxelsprocessed_fc_ds += voxelsprocessed_thispass
                        optiondict[
                            "despecklemasksize_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                        ] = voxelsprocessed_thispass
                        optiondict[
                            "despecklemaskpct_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                        ] = (100.0 * voxelsprocessed_thispass / optiondict["corrmasksize"])
                    else:
                        despecklingdone = True
                else:
                    despecklingdone = True
                if despecklingdone:
                    LGR.info("Nothing left to do! Terminating despeckling")
                    break

            internaldespeckleincludemask = np.where(
                np.abs(outmaparray - medianlags) > optiondict["despeckle_thresh"],
                medianlags,
                0.0,
            )
            if optiondict["savedespecklemasks"] and (optiondict["despeckle_passes"] > 0):
                despecklesavemask = np.where(internaldespeckleincludemask[validvoxels] == 0.0, 0, 1)
                if thepass == optiondict["passes"]:
                    if theinputdata.filetype != "text":
                        if theinputdata.filetype == "cifti":
                            timeindex = theheader["dim"][0] - 1
                            spaceindex = theheader["dim"][0]
                            theheader["dim"][timeindex] = 1
                            theheader["dim"][spaceindex] = numspatiallocs
                        else:
                            theheader["dim"][0] = 3
                            theheader["dim"][4] = 1
                            theheader["pixdim"][4] = 1.0
                    masklist = [
                        (
                            despecklesavemask,
                            "despeckle",
                            "mask",
                            None,
                            "Voxels that underwent despeckling in the final pass",
                        )
                    ]
                    tide_io.savemaplist(
                        outputname,
                        masklist,
                        validvoxels,
                        nativespaceshape,
                        theheader,
                        bidsbasedict,
                        filetype=theinputdata.filetype,
                        rt_floattype=rt_floattype,
                        cifti_hdr=theinputdata.cifti_hdr,
                    )
            LGR.info(
                f"\n\n{voxelsprocessed_fc_ds} voxels despeckled in "
                f"{optiondict['despeckle_passes']} passes"
            )
            TimingLGR.info(
                f"{similaritytype} despeckle end, pass {thepass}",
                {
                    "message2": voxelsprocessed_fc_ds,
                    "message3": "voxels",
                },
            )
        else:
            internaldespeckleincludemask = None

        # Patch shifting
        if optiondict["patchshift"]:
            outmaparray *= 0.0
            outmaparray[validvoxels] = eval("lagtimes")[:]
            # new method
            masklist = [
                (
                    outmaparray[validvoxels],
                    f"lagtimes_prepatch_pass{thepass}",
                    "map",
                    None,
                    f"Input lagtimes map prior to patch map generation pass {thepass}",
                ),
            ]
            tide_io.savemaplist(
                outputname,
                masklist,
                validvoxels,
                nativespaceshape,
                theheader,
                bidsbasedict,
                filetype=theinputdata.filetype,
                rt_floattype=rt_floattype,
                cifti_hdr=theinputdata.cifti_hdr,
            )

            # create list of anomalous 3D regions that don't match surroundings
            if theinputdata.nim_affine is not None:
                # make an atlas of anomalous patches - each patch shares the same integer value
                step1 = tide_patch.calc_DoG(
                    outmaparray.reshape(nativespaceshape).copy(),
                    theinputdata.nim_affine,
                    thesizes,
                    fwhm=optiondict["patchfwhm"],
                    ratioopt=False,
                    debug=True,
                )
                masklist = [
                    (
                        step1.reshape(internalspaceshape)[validvoxels],
                        f"DoG_pass{thepass}",
                        "map",
                        None,
                        f"DoG map for pass {thepass}",
                    ),
                ]
                tide_io.savemaplist(
                    outputname,
                    masklist,
                    validvoxels,
                    nativespaceshape,
                    theheader,
                    bidsbasedict,
                    filetype=theinputdata.filetype,
                    rt_floattype=rt_floattype,
                    cifti_hdr=theinputdata.cifti_hdr,
                )
                step2 = tide_patch.invertedflood3D(
                    step1,
                    1,
                )
                masklist = [
                    (
                        step2.reshape(internalspaceshape)[validvoxels],
                        f"invertflood_pass{thepass}",
                        "map",
                        None,
                        f"Inverted flood map for pass {thepass}",
                    ),
                ]
                tide_io.savemaplist(
                    outputname,
                    masklist,
                    validvoxels,
                    nativespaceshape,
                    theheader,
                    bidsbasedict,
                    filetype=theinputdata.filetype,
                    rt_floattype=rt_floattype,
                    cifti_hdr=theinputdata.cifti_hdr,
                )

                patchmap = tide_patch.separateclusters(
                    step2,
                    sizethresh=optiondict["patchminsize"],
                    debug=True,
                )
                # patchmap = tide_patch.getclusters(
                #   outmaparray.reshape(nativespaceshape),
                #    theinputdata.nim_affine,
                #    thesizes,
                #    fwhm=optiondict["patchfwhm"],
                #    ratioopt=True,
                #    sizethresh=optiondict["patchminsize"],
                #    debug=True,
                # )
                masklist = [
                    (
                        patchmap[validvoxels],
                        f"patch_pass{thepass}",
                        "map",
                        None,
                        f"Patch map for despeckling pass {thepass}",
                    ),
                ]
                tide_io.savemaplist(
                    outputname,
                    masklist,
                    validvoxels,
                    nativespaceshape,
                    theheader,
                    bidsbasedict,
                    filetype=theinputdata.filetype,
                    rt_floattype=rt_floattype,
                    cifti_hdr=theinputdata.cifti_hdr,
                )

            # now shift the patches to align with the majority of the image
            tide_patch.interppatch(lagtimes, patchmap[validvoxels])

    return internaldespeckleincludemask
