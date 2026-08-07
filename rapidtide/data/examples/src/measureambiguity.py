#!/usr/bin/env python
"""
Measure per-voxel delay ambiguity in rapidtide output, one line of TSV per run.

WHAT IT MEASURES
    For each voxel, the local maxima of its similarity function are the candidate
    delays.  A voxel is AMBIGUOUS when its second strongest peak is nearly as tall
    as the strongest AND lies far enough away in lag to be a genuine alternative
    rather than a shoulder of the same peak.  The reported number is the fraction
    of in-mask voxels that are ambiguous.

WHY
    rapidtide currently decides whether delays are at risk of sidelobe wrapping
    from acsidelobeamp, which is a property of the REGRESSOR.  Wrapping is a
    property of INDIVIDUAL VOXELS.  On HCP data these disagree: runs where
    rapidtide reports no sidelobe at all still had 4-5 percent of voxels facing a
    near-tied competing peak, and unwrapping fixed thousands of voxels on exactly
    those runs.  This script measures the voxel-level quantity directly so a
    better gate can be calibrated.

WHAT IT NEEDS
    Only standard rapidtide output, from ordinary runs:
        XXX_desc-corrout_info.nii.gz     (requires --savecorrout)
        XXX_desc-corrfit_mask.nii.gz
        XXX_desc-runoptions_info.json    (for despeckle_thresh; optional)
        XXX_desc-autocorr_timeseries.*   (optional, for comparison only)
    It does NOT need --resolvedelays.  The measurement is made on the similarity
    function before any repair, so it is identical either way.

    Dependencies: numpy and nibabel only.  rapidtide itself is not imported.

USAGE
    python measureambiguity.py /path/to/rapidtide/outputs -o ambiguity.tsv

    Searches recursively for *_desc-corrout_info.nii.gz.  Writes incrementally, so
    it can be interrupted and the partial TSV is still usable.  Roughly 5-15
    seconds per run, dominated by reading corrout.

    Send back only the TSV - it is a few KB regardless of how many runs you feed
    it.  No image data needs to move.

OUTPUT COLUMNS
    ambig_r<RATIO>_s<SEP>   fraction of mask voxels ambiguous at that
                            second/first amplitude ratio and that minimum lag
                            separation in seconds.  A small grid is computed so
                            the operating point can be explored without rerunning.
    ambig_r80_sdesp         the headline number: ratio 0.8, separation taken from
                            the run's own despeckle_thresh
"""
import argparse
import glob
import json
import os
import sys
import traceback

import nibabel as nib
import numpy as np

RATIOS = (0.6, 0.7, 0.8, 0.9)
SEPARATIONS = (4.0, 6.0, 8.0, 10.0)


def toppeaks(thedata, thelagaxis):
    """Amplitude and lag of the two strongest local maxima along the last axis.

    Vectorised over every leading axis.  No sub-sample refinement: the separation
    threshold is several seconds, so a fraction of a lag step does not matter here.
    """
    theismax = np.zeros(thedata.shape, dtype=bool)
    theismax[..., 1:-1] = (thedata[..., 1:-1] > thedata[..., :-2]) & (
        thedata[..., 1:-1] >= thedata[..., 2:]
    )
    theamps = np.where(theismax, thedata, -np.inf)
    theorder = np.argsort(-theamps, axis=-1)[..., :2]
    thebest = np.take_along_axis(theamps, theorder[..., 0:1], axis=-1)[..., 0]
    thesecond = np.take_along_axis(theamps, theorder[..., 1:2], axis=-1)[..., 0]
    thebestlag = thelagaxis[theorder[..., 0]]
    thesecondlag = thelagaxis[theorder[..., 1]]
    return thebest, thesecond, thebestlag, thesecondlag


def measuresidelobe(theroot, srmax=15.0):
    """rapidtide's autocorrelation sidelobe, measured independently, for comparison."""
    try:
        import pandas as pd

        thejson = json.load(open(theroot + "_desc-autocorr_timeseries.json"))
        they = pd.read_csv(
            theroot + "_desc-autocorr_timeseries.tsv.gz", sep="\t", header=None
        ).values[:, 0].astype(float)
    except Exception:
        return None, None
    thet = thejson["StartTime"] + np.arange(len(they)) / thejson["SamplingFrequency"]
    thezero = int(np.argmin(np.abs(thet)))
    if they[thezero] == 0:
        return None, None
    they = they / they[thezero]
    thesel = (thet > 1.0) & (thet <= srmax)
    if thesel.sum() < 5:
        return None, None
    thesub = they[thesel]
    theismax = np.zeros(thesub.shape, dtype=bool)
    theismax[1:-1] = (thesub[1:-1] > thesub[:-2]) & (thesub[1:-1] >= thesub[2:])
    if not theismax.any():
        return None, None
    thelags, theamps = thet[thesel][theismax], thesub[theismax]
    thepos = theamps > 0
    if not thepos.any():
        return None, float(theamps.max())
    thebest = int(np.argmax(np.where(thepos, theamps, -np.inf)))
    return float(thelags[thebest]), float(theamps[thebest])


def processrun(theroot, chunk=8):
    thecorrfile = theroot + "_desc-corrout_info.nii.gz"
    themaskfile = theroot + "_desc-corrfit_mask.nii.gz"
    if not (os.path.isfile(thecorrfile) and os.path.isfile(themaskfile)):
        return None

    theoptions = {}
    try:
        theoptions = json.load(open(theroot + "_desc-runoptions_info.json"))
    except Exception:
        pass
    thedespecklethresh = float(theoptions.get("despeckle_thresh", 6.0))

    theimg = nib.load(thecorrfile)
    thelagstep = float(theimg.header["pixdim"][4])
    thelagstart = float(theimg.header["toffset"])
    thenumlags = theimg.shape[3]
    if thelagstep <= 0:
        return None
    thelagaxis = thelagstart + np.arange(thenumlags) * thelagstep

    themask = np.asarray(nib.load(themaskfile).dataobj) > 0
    thenummask = int(themask.sum())
    if thenummask < 100:
        return None

    theseps = tuple(sorted(set(SEPARATIONS + (thedespecklethresh,))))
    thecounts = {(r, s): 0 for r in RATIOS for s in theseps}

    # stream over slabs so peak memory stays small even for large volumes
    for z0 in range(0, theimg.shape[2], chunk):
        z1 = min(z0 + chunk, theimg.shape[2])
        theslabmask = themask[:, :, z0:z1]
        if not theslabmask.any():
            continue
        theslab = np.asarray(theimg.dataobj[:, :, z0:z1, :], dtype=np.float32)
        theslab = np.nan_to_num(theslab)
        thebest, thesecond, thebestlag, thesecondlag = toppeaks(theslab, thelagaxis)
        theusable = theslabmask & np.isfinite(thesecond) & (thebest > 0)
        theratio = np.where(theusable, thesecond / np.maximum(thebest, 1.0e-9), 0.0)
        thegap = np.abs(thesecondlag - thebestlag)
        for r in RATIOS:
            for s in theseps:
                thecounts[(r, s)] += int((theusable & (theratio > r) & (thegap > s)).sum())
        del theslab

    thesidelobelag, thesidelobeamp = measuresidelobe(theroot)
    therow = {
        "run": os.path.basename(theroot),
        "maskvox": thenummask,
        "numlags": thenumlags,
        "lagstep": round(thelagstep, 4),
        "lagstart": round(thelagstart, 3),
        "despeckle_thresh": round(thedespecklethresh, 3),
        "rt_sidelobeamp": theoptions.get("acsidelobeamp_pass3")
        or theoptions.get("acsidelobeamp_pass1"),
        "rt_sidelobelag": theoptions.get("acsidelobelag_pass3")
        or theoptions.get("acsidelobelag_pass1"),
        "meas_sidelobeamp": round(thesidelobeamp, 4) if thesidelobeamp is not None else "",
        "meas_sidelobelag": round(thesidelobelag, 3) if thesidelobelag is not None else "",
        "ambig_r80_sdesp": round(thecounts[(0.8, thedespecklethresh)] / thenummask, 6),
    }
    for r in RATIOS:
        for s in SEPARATIONS:
            therow[f"ambig_r{int(r * 100)}_s{int(s)}"] = round(
                thecounts[(r, s)] / thenummask, 6
            )
    return therow


def main():
    theparser = argparse.ArgumentParser(
        description="Measure per-voxel delay ambiguity in rapidtide output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Needs only corrout, corrfit_mask and runoptions.  --resolvedelays NOT required.",
    )
    theparser.add_argument("searchdirs", nargs="+", help="Directories to search recursively.")
    theparser.add_argument("-o", "--output", default="ambiguity.tsv", help="Output TSV.")
    theparser.add_argument(
        "--chunk", type=int, default=8, help="Slices per slab; lower uses less memory."
    )
    theargs = theparser.parse_args()

    theroots = []
    for thedir in theargs.searchdirs:
        for thefile in glob.glob(
            os.path.join(thedir, "**", "*_desc-corrout_info.nii.gz"), recursive=True
        ):
            theroots.append(thefile.replace("_desc-corrout_info.nii.gz", ""))
    theroots = sorted(set(theroots))
    print(f"found {len(theroots)} runs with a saved similarity function", flush=True)
    if not theroots:
        print("nothing to do - was rapidtide run with --savecorrout?", file=sys.stderr)
        return 1

    therows = []
    thecolumns = None
    for thenum, theroot in enumerate(theroots):
        try:
            therow = processrun(theroot, chunk=theargs.chunk)
        except Exception:
            print(f"[{thenum + 1}/{len(theroots)}] {os.path.basename(theroot)}: FAILED", flush=True)
            traceback.print_exc()
            continue
        if therow is None:
            print(
                f"[{thenum + 1}/{len(theroots)}] {os.path.basename(theroot)}: skipped "
                f"(missing files or empty mask)",
                flush=True,
            )
            continue
        therows.append(therow)
        if thecolumns is None:
            thecolumns = list(therow.keys())
        with open(theargs.output, "w") as thefile:
            thefile.write("\t".join(thecolumns) + "\n")
            for r in therows:
                thefile.write("\t".join(str(r.get(c, "")) for c in thecolumns) + "\n")
        print(
            f"[{thenum + 1}/{len(theroots)}] {os.path.basename(theroot)}: "
            f"ambiguous {100 * therow['ambig_r80_sdesp']:.2f}%  "
            f"(rt sidelobe amp {therow['rt_sidelobeamp']})",
            flush=True,
        )

    print(f"\nwrote {len(therows)} rows to {theargs.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
