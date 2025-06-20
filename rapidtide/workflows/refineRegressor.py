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
import sys

import numpy as np

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def refineRegressor(
    LGR,
    TimingLGR,
    thepass,
    optiondict,
    fitmask,
    internaloffsetincludemask_valid,
    internaloffsetexcludemask_valid,
    internalrefineincludemask_valid,
    internalrefineexcludemask_valid,
    internaldespeckleincludemask,
    validvoxels,
    theRegressorRefiner,
    lagtimes,
    lagstrengths,
    lagsigma,
    fmri_data_valid,
    fmritr,
    R2,
    theprefilter,
    previousnormoutputdata,
    theinputdata,
    numpadtrs,
    outputname,
    nativefmrishape,
    bidsbasedict,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    LGR.info(f"\n\nRegressor refinement, pass {thepass}")
    TimingLGR.info(f"Regressor refinement start, pass {thepass}")
    if optiondict["refineoffset"]:
        # check that we won't end up excluding all voxels from offset calculation before accepting mask
        offsetmask = np.uint16(fitmask)
        if internaloffsetincludemask_valid is not None:
            offsetmask[np.where(internaloffsetincludemask_valid == 0)] = 0
        if internaloffsetexcludemask_valid is not None:
            offsetmask[np.where(internaloffsetexcludemask_valid != 0.0)] = 0
        if tide_stats.getmasksize(offsetmask) == 0:
            LGR.warning(
                "NB: cannot exclude voxels from offset calculation mask - including for this pass"
            )
            offsetmask = fitmask + 0

        peaklag, dummy, dummy = tide_stats.gethistprops(
            lagtimes[np.where(offsetmask > 0)],
            optiondict["histlen"],
            pickleft=optiondict["pickleft"],
            peakthresh=optiondict["pickleftthresh"],
        )
        optiondict["offsettime"] = peaklag
        optiondict["offsettime_total"] += peaklag
        optiondict[f"offsettime_pass{thepass}"] = optiondict["offsettime"]
        optiondict[f"offsettime_total_pass{thepass}"] = optiondict["offsettime_total"]
        LGR.info(
            f"offset time set to {optiondict['offsettime']:.3f}, "
            f"total is {optiondict['offsettime_total']:.3f}"
        )

    if optiondict["refinedespeckled"] or (optiondict["despeckle_passes"] == 0):
        # if refinedespeckled is true, or there is no despeckling, masks are unaffected
        thisinternalrefineexcludemask_valid = internalrefineexcludemask_valid
    else:
        # if refinedespeckled is false and there is despeckling, need to make a proper mask
        if internalrefineexcludemask_valid is None:
            # if there is currently no exclude mask, set exclude mask = despeckle mask
            thisinternalrefineexcludemask_valid = np.where(
                internaldespeckleincludemask[validvoxels] == 0.0, 0, 1
            )
        else:
            # if there is a current exclude mask, add any voxels that are being despeckled
            thisinternalrefineexcludemask_valid = np.where(
                internalrefineexcludemask_valid > 0, 1, 0
            )
            thisinternalrefineexcludemask_valid[
                np.where(internaldespeckleincludemask[validvoxels] != 0.0)
            ] = 1

        # now check that we won't end up excluding all voxels from refinement before accepting mask
        overallmask = np.uint16(fitmask)
        if internalrefineincludemask_valid is not None:
            overallmask[np.where(internalrefineincludemask_valid == 0)] = 0
        if thisinternalrefineexcludemask_valid is not None:
            overallmask[np.where(thisinternalrefineexcludemask_valid != 0.0)] = 0
        if tide_stats.getmasksize(overallmask) == 0:
            LGR.warning(
                "NB: cannot exclude despeckled voxels from refinement - including for this pass"
            )
            thisinternalrefineexcludemask_valid = internalrefineexcludemask_valid
    theRegressorRefiner.setmasks(
        internalrefineincludemask_valid, thisinternalrefineexcludemask_valid
    )

    # regenerate regressor for next pass
    # create the refinement mask
    LGR.info("making refine mask")
    createdmask = theRegressorRefiner.makemask(lagstrengths, lagtimes, lagsigma, fitmask)
    print(f"Refine mask has {theRegressorRefiner.refinemaskvoxels} voxels")
    if not createdmask:
        print("no voxels qualify for refinement - exiting")
        sys.exit()

    # align timecourses to prepare for refinement
    LGR.info("aligning timecourses")
    tide_util.disablemkl(optiondict["nprocs_refine"], debug=optiondict["threaddebug"])
    voxelsprocessed_rra = theRegressorRefiner.alignvoxels(fmri_data_valid, fmritr, lagtimes)
    tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])
    LGR.info(f"align complete: {voxelsprocessed_rra=}")

    # prenormalize
    LGR.info("prenormalizing timecourses")
    theRegressorRefiner.prenormalize(lagtimes, lagstrengths, R2)

    # now doing the refinement
    (
        voxelsprocessed_rr,
        outputdict,
        previousnormoutputdata,
        resampref_y,
        resampnonosref_y,
        stoprefining,
        refinestopreason,
        genlagtc,
    ) = theRegressorRefiner.refine(
        theprefilter,
        fmritr,
        thepass,
        lagstrengths,
        lagtimes,
        previousnormoutputdata,
        optiondict["corrmasksize"],
    )
    TimingLGR.info(
        f"Regressor refinement end, pass {thepass}",
        {
            "message2": voxelsprocessed_rr,
            "message3": "voxels",
        },
    )
    for key, value in outputdict.items():
        optiondict[key] = value

    # Save shifted timecourses for César
    if optiondict["saveintermediatemaps"] and optiondict["savelagregressors"]:
        theheader = theinputdata.copyheader()
        bidspasssuffix = f"_intermediatedata-pass{thepass}"
        maplist = [
            (
                (theRegressorRefiner.getpaddedshiftedtcs())[:, numpadtrs:-numpadtrs],
                "shiftedtcs",
                "bold",
                None,
                "The filtered input fMRI data, in voxels used for refinement, time shifted by the negated delay in every voxel so that the moving blood component is aligned.",
            ),
        ]
        tide_io.savemaplist(
            f"{outputname}{bidspasssuffix}",
            maplist,
            validvoxels,
            nativefmrishape,
            theheader,
            bidsbasedict,
            filetype=theinputdata.filetype,
            rt_floattype=rt_floattype,
            cifti_hdr=theinputdata.cifti_hdr,
            debug=True,
        )
    # We are done with refinement.
