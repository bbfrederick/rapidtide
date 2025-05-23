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
import copy
import os
import sys

import numpy as np
import pandas as pd

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.maskutil as tide_mask
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    class FullPaths(argparse.Action):
        """Expand user- and relative-paths"""

        def __call__(self, parser, namespace, values, option_string=None):
            if values == "":
                setattr(namespace, self.dest, "__EMPTY__")
            else:
                setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="fingerprint",
        description=(
            "Fit a rapidtide output map to a canonical map, by vascular territory, and calculate statistical metrics.\n"
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "inputfile",
        help="Rapidtide output map to decompose by vascular territory.  Must be in MNI152NLin6Asym coordinates, 2mm resolution",
        action=FullPaths,
        type=lambda x: pf.is_valid_file(parser, x),
    )
    parser.add_argument(
        "outputroot",
        help="name root for output files",
        action=FullPaths,
        type=str,
    )

    maskopts = parser.add_argument_group("Masking options")
    maskopts.add_argument(
        "--includemask",
        dest="includespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Only use atlas voxels that are also in file MASK in calculating the fit values "
            "(if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are used). "
        ),
        default=None,
    )
    maskopts.add_argument(
        "--excludemask",
        dest="excludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Do not use atlas voxels that are also in file MASK in calculating the fit values "
            "(if VALSPEC is given, voxels "
            "with integral values listed in VALSPEC are excluded). "
        ),
        default=None,
    )
    maskopts.add_argument(
        "--extramask",
        dest="extramaskname",
        metavar="MASK",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "Additional mask to apply to select voxels for fitting.  Zero voxels in this mask will be excluded."
        ),
        default=None,
    )

    atlasopts = parser.add_argument_group("Atlas options")
    atlasopts.add_argument(
        "--atlas",
        help=(
            "Atlas.  Options are: "
            "ASPECTS - ASPECTS territory atlas; "
            "ATT - Arterial transit time flow territories; "
            "JHU1 - Johns Hopkins level 1 probabilistic arterial flow territories, without ventricles (default); "
            "JHU2 - Johns Hopkins level 2 probabilistic arterial flow territories, without ventricles."
        ),
        choices=["ASPECTS", "ATT", "JHU1", "JHU2"],
        default="JHU1",
    )
    atlasopts.add_argument(
        "--customatlas",
        help=(
            "The name of a custom atlas file.  The file must be a 3D NIFTI "
            "file in MNI152NLin6Asym space, 2mm isotropic resolution.  The file values "
            "must be consecutive integers from 1 to N with no missing values.  There "
            "must be a text file with an identical path and name (but with .txt instead of .nii.gz "
            "as the extension) containing the name of every region on a separate line.  This argument "
            "overrides the --atlas argument."
        ),
        type=lambda x: pf.is_valid_file(parser, x),
        default=None,
    )

    fitopts = parser.add_argument_group("Fitting options")
    fitopts.add_argument(
        "--fitorder",
        help="order of polynomial fit to template (default is 1).",
        type=int,
        default=1,
    )
    fitopts.add_argument(
        "--nointercept",
        help="do not use the zeroth order component when fitting",
        action="store_true",
    )
    fitopts.add_argument(
        "--limittomask",
        help="only calculate fitdiff in masked region",
        action="store_true",
    )
    fitopts.add_argument(
        "--template",
        help=(
            "Template to fit.  Default is 'lag'.  Options are\n"
            "\tlag:      time lag in seconds\n"
            "\tstrength: correlation coefficient\n"
            "\tsigma:    correlation peak width in seconds\n"
            "\tconstant: constant value (forces fit order to 0).\n"
        ),
        choices=["lag", "strength", "sigma", "constant"],
        default="lag",
    )
    fitopts.add_argument(
        "--entropybins",
        help="number of bins in the entropy histogram (default is 101).",
        type=int,
        default=101,
    )

    fitopts.add_argument(
        "--entropyrange",
        dest="entropyrange",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERLIM", "UPPERLIM"),
        help=(
            "Upper and lower limits of the range for the entropy histogram "
            "(default is to use data min and max)."
        ),
        default=None,
    )

    miscopts = parser.add_argument_group("Fitting options")
    miscopts.add_argument(
        "--debug",
        help="output additional debugging information",
        action="store_true",
        default=False,
    )

    return parser


def fingerprint_main(
    themapname,
    whichtemplate,
    whichatlas,
    outputroot,
    fitorder,
    includespec=None,
    excludespec=None,
    extramaskname=None,
    intercept=True,
    entropybins=101,
    entropyrange=None,
    debug=False,
):
    # read the data
    referencedir = tide_util.findreferencedir()
    if debug:
        print(f"Reference directory is {referencedir}")

    if whichtemplate == "lag":
        thetemplatename = os.path.join(referencedir, "HCP1200_lag_2mm.nii.gz")
    elif whichtemplate == "strength":
        thetemplatename = os.path.join(referencedir, "HCP1200_strength_2mm.nii.gz")
    elif whichtemplate == "sigma":
        thetemplatename = os.path.join(referencedir, "HCP1200_sigma_2mm.nii.gz")
    elif whichtemplate == "constant":
        thetemplatename = None
    else:
        print("illegal template:", whichtemplate)
        sys.exit()

    if whichatlas == "ASPECTS":
        theatlasname = os.path.join(referencedir, "ASPECTS_2mm.nii.gz")
        theatlasregionsname = os.path.join(referencedir, "ASPECTS_regions.txt")
    elif whichatlas == "ATT":
        theatlasname = os.path.join(referencedir, "ATTbasedFlowTerritories_split_2mm.nii.gz")
        theatlasregionsname = os.path.join(
            referencedir, "ATTbasedFlowTerritories_split_regions.txt"
        )
    elif whichatlas == "JHU1":
        theatlasname = os.path.join(
            referencedir, "JHU-ArterialTerritoriesNoVent-LVL1_space-MNI152NLin6Asym_2mm.nii.gz"
        )
        theatlasregionsname = os.path.join(
            referencedir, "JHU-ArterialTerritoriesNoVent-LVL1_regions.txt"
        )
    elif whichatlas == "JHU2":
        theatlasname = os.path.join(
            referencedir, "JHU-ArterialTerritoriesNoVent-LVL2_space-MNI152NLin6Asym_2mm.nii.gz"
        )
        theatlasregionsname = os.path.join(
            referencedir, "JHU-ArterialTerritoriesNoVent-LVL2_regions.txt"
        )
    elif whichatlas[:5] == "USER_":
        theatlasname = whichatlas[5:]
        theatlasregionsname = theatlasname.replace(".nii.gz", ".txt")
        whichatlas = "USER"
    else:
        print("illegal atlas:", whichatlas)
        sys.exit()

    outputroot += f"_template-{whichtemplate}_atlas-{whichatlas}_O{fitorder}"

    # read the atlas
    if debug:
        print(f"Reading atlas {theatlasname}")
    atlas, atlas_data, atlas_hdr, atlasdims, atlassizes = tide_io.readfromnifti(theatlasname)
    atlaslabelsinput = pd.read_csv(
        theatlasregionsname, delimiter="\t", header=None, names=["Region"]
    )

    # read the template
    if thetemplatename is not None:
        if debug:
            print(f"reading atlas file {theatlasname}")
        (
            thetemplate,
            thetemplate_data,
            thetemplate_hdr,
            thetemplatedims,
            thetemplatesizes,
        ) = tide_io.readfromnifti(thetemplatename)
    else:
        thetemplate_data = atlas_data * 0.0
        thetemplate_data[np.where(atlas_data > 0)] = 1.0

    if debug:
        print(f"reading map file {themapname}")
    (
        themap,
        themap_data,
        themap_hdr,
        themapdims,
        thetemplatesizes,
    ) = tide_io.readfromnifti(themapname)
    nx, ny, nz, nummaps = tide_io.parseniftidims(themapdims)

    # process masks
    if includespec is not None:
        (
            includename,
            includevals,
        ) = tide_io.processnamespec(
            includespec, "Including voxels where ", "in offset calculation."
        )
    else:
        includename = None
        includevals = None
    if excludespec is not None:
        (
            excludename,
            excludevals,
        ) = tide_io.processnamespec(
            excludespec, "Excluding voxels where ", "from offset calculation."
        )
    else:
        excludename = None
        excludevals = None

    numspatiallocs = int(nx) * int(ny) * int(nz)
    includemask, excludemask, extramask = tide_mask.getmaskset(
        "anatomic",
        includename,
        includevals,
        excludename,
        excludevals,
        themap_hdr,
        numspatiallocs,
        extramask=extramaskname,
    )

    theflatmask = themap_data.reshape((numspatiallocs)) * 0 + 1
    if includemask is not None:
        theflatmask = theflatmask * includemask.reshape((numspatiallocs))
    if excludemask is not None:
        theflatmask = theflatmask * (1 - excludemask.reshape((numspatiallocs)))
    if extramask is not None:
        theflatmask = theflatmask * extramask.reshape((numspatiallocs))

    # generate the mask
    themask_data = theflatmask.reshape((nx, ny, nz))
    maskmap = themask_data
    maskmap[np.where(atlas_data < 1)] = 0.0

    # save the maskmap as nifti
    themaskmaphdr = copy.deepcopy(themap_hdr)
    themaskmaphdr["dim"][0] = 3
    themaskmaphdr["dim"][4] = 1
    tide_io.savetonifti(maskmap, themaskmaphdr, outputroot + "_maskmap")

    # get ready to do the fitting
    numregions = len(atlaslabelsinput)
    if intercept:
        numcoffs = fitorder + 1
    else:
        numcoffs = fitorder
    coff_array = np.zeros((numcoffs, numregions, nummaps), dtype="float")
    R2_array = np.zeros((numregions, nummaps), dtype="float")

    # do the fit
    if debug:
        print("starting decomposition")
    thefitmap, thecoffs, theR2s = tide_fit.territorydecomp(
        themap_data,
        thetemplate_data,
        atlas_data.astype(int),
        inputmask=themask_data,
        fitorder=fitorder,
        intercept=intercept,
        debug=debug,
    )

    # transfer the data into arrays
    for whichmap in range(nummaps):
        R2_array[:, whichmap] = theR2s[whichmap]
        coff_array[:, :, whichmap] = np.transpose(thecoffs[whichmap, :, :])

    # save the Rs as tsv
    newcols = pd.DataFrame(np.transpose(R2_array[:, :]))
    newcols.columns = atlaslabelsinput["Region"]
    newcols.to_csv(f"{outputroot}_allR2s.tsv", index=False, sep="\t")

    # save the fits as tsv
    if intercept:
        endpoint = fitorder + 1
        offset = 0
    else:
        endpoint = fitorder
        offset = 1
    for i in range(0, endpoint):
        newcols = pd.DataFrame(np.transpose(coff_array[i, :, :]))
        newcols.columns = atlaslabelsinput["Region"]
        newcols.to_csv(f"{outputroot}_fit_O{str(i + offset)}.tsv", index=False, sep="\t")

    # save the fit data as nifti
    if args.limittomask:
        if nummaps == 1:
            savemap = thefitmap * maskmap
        else:
            savemap = thefitmap * maskmap[:, :, :, None]
    else:
        savemap = thefitmap
    tide_io.savetonifti(savemap, themap_hdr, outputroot + "_fit")

    # save the fit error
    if args.limittomask:
        if nummaps == 1:
            diffmap = (themap_data - thefitmap) * maskmap
        else:
            diffmap = (themap_data - thefitmap) * maskmap[:, :, :, None]
    else:
        diffmap = themap_data - thefitmap
    tide_io.savetonifti(diffmap, themap_hdr, outputroot + "_fitdiff")

    # save the Rs as nifti
    thehdr = copy.deepcopy(themap_hdr)
    print(f"shape of R2_array: {R2_array.shape}")
    print(f"thehdr before: {thehdr['dim']}")
    thehdr["dim"][0] = 2
    thehdr["dim"][1] = R2_array.shape[0]
    thehdr["dim"][2] = R2_array.shape[1]
    thehdr["dim"][3] = 1
    thehdr["dim"][4] = 1
    print(f"thehdr after: {thehdr['dim']}")
    thehdr["pixdim"][0] = 1.0
    thehdr["pixdim"][1] = 1.0
    thehdr["pixdim"][2] = 1.0
    thehdr["pixdim"][3] = 1.0
    thehdr["pixdim"][4] = 1.0
    tide_io.savetonifti(R2_array, thehdr, outputroot + "_allR2s")

    # save the fit coefficients as nifti
    thehdr = copy.deepcopy(themap_hdr)
    print(f"shape of coff_array: {coff_array.shape}")
    print(f"thehdr before: {thehdr['dim']}")
    thehdr["dim"][0] = 3
    thehdr["dim"][1] = coff_array.shape[0]
    thehdr["dim"][2] = coff_array.shape[1]
    thehdr["dim"][3] = coff_array.shape[2]
    thehdr["dim"][4] = 1
    print(f"thehdr after: {thehdr['dim']}")
    thehdr["pixdim"][0] = 1.0
    thehdr["pixdim"][1] = 1.0
    thehdr["pixdim"][2] = 1.0
    thehdr["pixdim"][3] = 1.0
    thehdr["pixdim"][4] = 1.0
    tide_io.savetonifti(coff_array, thehdr, outputroot + "_allcoffs")

    # now do the stats
    #   first on the initial map
    (
        statsmap,
        themeans,
        thestds,
        themedians,
        themads,
        thevariances,
        theskewnesses,
        thekurtoses,
        theentropies,
    ) = tide_fit.territorystats(
        themap_data,
        atlas_data.astype(int),
        inputmask=themask_data,
        entropybins=entropybins,
        entropyrange=entropyrange,
        debug=debug,
    )

    #   then on the residuals after fitting
    (
        residualstatsmap,
        theresidualmeans,
        theresidualstds,
        theresidualmedians,
        theresidualmads,
        theresidualvariances,
        theresidualskewnesses,
        theresidualkurtoses,
        theresidualentropies,
    ) = tide_fit.territorystats(
        diffmap,
        atlas_data.astype(int),
        inputmask=themask_data,
        entropybins=entropybins,
        entropyrange=entropyrange,
        debug=debug,
    )

    # Organize the data
    mean_array = np.zeros((numregions, nummaps), dtype="float")
    std_array = np.zeros((numregions, nummaps), dtype="float")
    median_array = np.zeros((numregions, nummaps), dtype="float")
    variance_array = np.zeros((numregions, nummaps), dtype="float")
    skewness_array = np.zeros((numregions, nummaps), dtype="float")
    kurtosis_array = np.zeros((numregions, nummaps), dtype="float")
    mad_array = np.zeros((numregions, nummaps), dtype="float")
    entropy_array = np.zeros((numregions, nummaps), dtype="float")
    residual_mean_array = np.zeros((numregions, nummaps), dtype="float")
    residual_std_array = np.zeros((numregions, nummaps), dtype="float")
    residual_median_array = np.zeros((numregions, nummaps), dtype="float")
    residual_mad_array = np.zeros((numregions, nummaps), dtype="float")
    residual_variance_array = np.zeros((numregions, nummaps), dtype="float")
    residual_skewness_array = np.zeros((numregions, nummaps), dtype="float")
    residual_kurtosis_array = np.zeros((numregions, nummaps), dtype="float")
    residual_entropy_array = np.zeros((numregions, nummaps), dtype="float")

    for whichmap in range(nummaps):
        atlaslabels = atlaslabelsinput.copy()

        mean_array[:, whichmap] = themeans[whichmap]
        std_array[:, whichmap] = thestds[whichmap]
        median_array[:, whichmap] = themedians[whichmap]
        mad_array[:, whichmap] = themads[whichmap]
        variance_array[:, whichmap] = thevariances[whichmap]
        skewness_array[:, whichmap] = theskewnesses[whichmap]
        kurtosis_array[:, whichmap] = thekurtoses[whichmap]
        entropy_array[:, whichmap] = theentropies[whichmap]

        residual_mean_array[:, whichmap] = theresidualmeans[whichmap]
        residual_std_array[:, whichmap] = theresidualstds[whichmap]
        residual_median_array[:, whichmap] = theresidualmedians[whichmap]
        residual_mad_array[:, whichmap] = theresidualmads[whichmap]
        residual_variance_array[:, whichmap] = theresidualvariances[whichmap]
        residual_skewness_array[:, whichmap] = theresidualskewnesses[whichmap]
        residual_kurtosis_array[:, whichmap] = theresidualkurtoses[whichmap]
        residual_entropy_array[:, whichmap] = theresidualentropies[whichmap]

        newcols = pd.DataFrame(thecoffs[whichmap, :, :])
        columnnames = []
        if intercept:
            startpt = 0
        else:
            startpt = 1
        for i in range(startpt, fitorder + 1):
            columnnames += str(i)
        newcols.columns = columnnames
        atlaslabels["R2"] = theR2s[whichmap]
        atlaslabels = pd.concat([atlaslabels, newcols], axis=1)
        atlaslabels.to_csv(f"{outputroot}_{str(whichmap).zfill(4)}_fits.tsv", sep="\t")
        if debug:
            print(atlaslabels)

    # save the stats as tsv
    for thestat in [
        "mean",
        "std",
        "median",
        "mad",
        "variance",
        "skewness",
        "kurtosis",
        "entropy",
        "residual_mean",
        "residual_std",
        "residual_median",
        "residual_mad",
        "residual_variance",
        "residual_skewness",
        "residual_kurtosis",
        "residual_entropy",
    ]:
        newcols = pd.DataFrame(np.transpose(eval(f"{thestat}_array")[:, :]))
        newcols.columns = atlaslabelsinput["Region"]
        newcols.to_csv(f"{outputroot}_all{thestat}.tsv", index=False, sep="\t")


def entrypoint():
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    if args.debug:
        print("before preprocessing:", args)

    if args.customatlas is not None:
        theatlas = f"USER_{args.customatlas}"
    else:
        theatlas = args.atlas

    if args.template == "constant":
        args.fitorder = 0
        args.nointercept = False

    if args.debug:
        print("after preprocessing:", args)

    if args.debug:
        print(f"Using atlas {theatlas}")

    fingerprint_main(
        args.inputfile,
        args.template,
        theatlas,
        args.outputroot,
        args.fitorder,
        intercept=not (args.nointercept),
        includespec=args.includespec,
        excludespec=args.excludespec,
        extramaskname=args.extramaskname,
        entropybins=args.entropybins,
        entropyrange=args.entropyrange,
        debug=args.debug,
    )


if __name__ == "__main__":
    entrypoint()
