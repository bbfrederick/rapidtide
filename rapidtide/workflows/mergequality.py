#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2024-2025 Blaise Frederick
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

import pandas as pd

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats


def _get_parser():
    """
    Argument parser for mergequality
    """
    parser = argparse.ArgumentParser(
        prog="mergequality",
        description=("Merge rapidtide quality check data from several runs."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument("--input", action="store", type=str, nargs="*", required=True)
    parser.add_argument("--outputroot", action="store", type=str, required=True)

    # add optional
    parser.add_argument("--keyfile", action="store", type=str, default=None)
    parser.add_argument(
        "--showhists",
        action="store_true",
        help=("Display the histograms of the tracked quantities."),
        default=False,
    )
    parser.add_argument(
        "--addgraymetrics",
        action="store_true",
        help=("Include gray matter only metrics."),
        default=False,
    )
    parser.add_argument(
        "--addwhitemetrics",
        action="store_true",
        help=("Include white matter only metrics."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Output additional debugging information."),
        default=False,
    )

    return parser


def mergequality(args):
    if args.debug:
        print(f"{args.input=}")
        print(f"{args.outputroot=}")
        print(f"{args.keyfile=}")

    if args.keyfile is not None:
        thekeydict = tide_io.readdictfromjson(args.keyfile)
    else:
        thekeydict = {
            "mask": {
                "lagmaskvoxels": 215891,
                "meanrelsize": 4.180947793099295,
                "p_lt_0p001relsize": 0.6806073435205728,
                "p_lt_0p005relsize": 0.745116748729683,
                "p_lt_0p010relsize": 0.7726074732156505,
                "p_lt_0p050relsize": 0.838487940673766,
                "refinerelsize": 0.5496616348064532,
            },
            "regressor": {
                "first_kurtosis": 2.911833576023774,
                "first_kurtosis_p": 1.143835967291008e-38,
                "first_kurtosis_z": 13.00514376674551,
                "first_skewness": -0.3219158734451699,
                "first_skewness_p": 2.9429479496726994e-10,
                "first_skewness_z": -6.301754636834285,
                "first_spectralflatness": 0.9306835675389035,
                "last_kurtosis": 2.6687828914360496,
                "last_kurtosis_p": 1.2610242749531053e-35,
                "last_kurtosis_z": 12.458253243613367,
                "last_skewness": 0.06902106526612979,
                "last_skewness_p": 0.1667298705677166,
                "last_skewness_z": 1.3827880292121477,
                "last_spectralflatness": 0.9214471595162257,
            },
            "lag": {
                "centerofmass": 0.18365522576293233,
                "kurtosis": 2.6365118239988172,
                "kurtosis_p": 0.0,
                "kurtosis_z": 115.68766626074614,
                "pct02": -3.5552243631332305,
                "pct25": -1.114490891184175,
                "pct50": -0.1257154777049373,
                "pct75": 1.0872777685344683,
                "pct98": 5.428974111640412,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 0.268346902434662,
                "peakloc": -0.32178217821782207,
                "peakpercentile": 44.776762347666185,
                "peakwidth": 3.2673267326732676,
                "skewness": 0.8500928388366628,
                "skewness_p": 0.0,
                "skewness_z": 140.97401621947097,
                "voxelsincluded": 215891,
            },
            "laggrad": {
                "centerofmass": 0.571792758578445,
                "kurtosis": 9.628956515105777,
                "kurtosis_p": 0.0,
                "kurtosis_z": 178.87307892867446,
                "pct02": 0.08644029467314696,
                "pct25": 0.2526212777361018,
                "pct50": 0.41670054269474405,
                "pct75": 0.7196355014183266,
                "pct98": 2.445977359086096,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 1.7495682765313678,
                "peakloc": 0.2524752475247525,
                "peakpercentile": 24.97466346179356,
                "peakwidth": 0.6534653465346536,
                "skewness": 2.6454479455818714,
                "skewness_p": 0.0,
                "skewness_z": 273.76650916159275,
                "voxelsincluded": 188463,
            },
            "strength": {
                "centerofmass": 0.3718351617952105,
                "kurtosis": -0.7116698199891451,
                "kurtosis_p": 0.0,
                "kurtosis_z": -109.8631499714451,
                "pct02": 0.0463983482697488,
                "pct25": 0.2195563043221062,
                "pct50": 0.35605876676999876,
                "pct75": 0.5108963405705843,
                "pct98": 0.7675960511390326,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 1.9180975584901625,
                "peakloc": 0.2722772277227723,
                "peakpercentile": 34.364100402517934,
                "peakwidth": 0.7326732673267327,
                "skewness": 0.2767447291065046,
                "skewness_p": 0.0,
                "skewness_z": 51.540795677466,
                "voxelsincluded": 215891,
            },
            "MTT": {
                "centerofmass": 1.1854362572800032,
                "kurtosis": 6.48403822013551,
                "kurtosis_p": 0.0,
                "kurtosis_z": 169.01367062971818,
                "pct02": 0.0,
                "pct25": 0.0,
                "pct50": 0.0,
                "pct75": 0.9117823500153743,
                "pct98": 2.835547357601372,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 0.33323899429333736,
                "peakloc": 0.8415841584158417,
                "peakpercentile": 39.06788519856343,
                "peakwidth": 1.386138613861386,
                "skewness": 2.0750539699859707,
                "skewness_p": 0.0,
                "skewness_z": 257.5035937509046,
                "voxelsincluded": 215891,
            },
        }

        thewhitematterdict = {
            "whiteonly-lag": {
                "centerofmass": 0.12863780691943305,
                "kurtosis": 6.051218980924041,
                "kurtosis_p": 0.0,
                "kurtosis_z": 97.74313967555645,
                "pct02": -3.447084496826285,
                "pct25": -1.153943456786113,
                "pct50": 0.40468708058815667,
                "pct75": 2.5020675667656325,
                "pct98": 35.65786745845405,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 0.20154056103108517,
                "peakloc": -0.7673267326732676,
                "peakpercentile": 31.311237290373697,
                "peakwidth": 7.7227722772277225,
                "skewness": 2.583265856366246,
                "skewness_p": 0.0,
                "skewness_z": 171.4754157543282,
                "voxelsincluded": 75730,
            },
            "whiteonly-laggrad": {
                "centerofmass": 0.4400415073580781,
                "kurtosis": 4.8057153283315746,
                "kurtosis_p": 0.0,
                "kurtosis_z": 88.82511535725718,
                "pct02": 0.07577299562385723,
                "pct25": 0.2357562409434768,
                "pct50": 0.4080402253691755,
                "pct75": 0.9014255193901806,
                "pct98": 12.105429775628355,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 2.137460662083183,
                "peakloc": 0.22277227722772275,
                "peakpercentile": 22.70194611181395,
                "peakwidth": 0.5346534653465347,
                "skewness": 2.3423104651022015,
                "skewness_p": 0.0,
                "skewness_z": 161.40631688712926,
                "voxelsincluded": 74302,
            },
            "whiteonly-strength": {
                "centerofmass": 0.4427689302668548,
                "kurtosis": -0.3709348443577771,
                "kurtosis_p": 1.4319828709989065e-146,
                "kurtosis_z": -25.781443647052853,
                "pct02": 0.062435511959463494,
                "pct25": 0.3373922124837407,
                "pct50": 0.4652726794639804,
                "pct75": 0.5657461961283083,
                "pct98": 0.7482491589950304,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 2.6820414630925633,
                "peakloc": 0.4900990099009901,
                "peakpercentile": 56.33302522118051,
                "peakwidth": 0.3168316831683169,
                "skewness": -0.4091197098383096,
                "skewness_p": 0.0,
                "skewness_z": -44.23408729903814,
                "voxelsincluded": 75730,
            },
        }

        thegraymatterdict = {
            "grayonly-lag": {
                "centerofmass": 0.12863780691943305,
                "kurtosis": 6.051218980924041,
                "kurtosis_p": 0.0,
                "kurtosis_z": 97.74313967555645,
                "pct02": -3.447084496826285,
                "pct25": -1.153943456786113,
                "pct50": 0.40468708058815667,
                "pct75": 2.5020675667656325,
                "pct98": 35.65786745845405,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 0.20154056103108517,
                "peakloc": -0.7673267326732676,
                "peakpercentile": 31.311237290373697,
                "peakwidth": 7.7227722772277225,
                "skewness": 2.583265856366246,
                "skewness_p": 0.0,
                "skewness_z": 171.4754157543282,
                "voxelsincluded": 75730,
            },
            "grayonly-laggrad": {
                "centerofmass": 0.4400415073580781,
                "kurtosis": 4.8057153283315746,
                "kurtosis_p": 0.0,
                "kurtosis_z": 88.82511535725718,
                "pct02": 0.07577299562385723,
                "pct25": 0.2357562409434768,
                "pct50": 0.4080402253691755,
                "pct75": 0.9014255193901806,
                "pct98": 12.105429775628355,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 2.137460662083183,
                "peakloc": 0.22277227722772275,
                "peakpercentile": 22.70194611181395,
                "peakwidth": 0.5346534653465347,
                "skewness": 2.3423104651022015,
                "skewness_p": 0.0,
                "skewness_z": 161.40631688712926,
                "voxelsincluded": 74302,
            },
            "grayonly-strength": {
                "centerofmass": 0.4427689302668548,
                "kurtosis": -0.3709348443577771,
                "kurtosis_p": 1.4319828709989065e-146,
                "kurtosis_z": -25.781443647052853,
                "pct02": 0.062435511959463494,
                "pct25": 0.3373922124837407,
                "pct50": 0.4652726794639804,
                "pct75": 0.5657461961283083,
                "pct98": 0.7482491589950304,
                "q1width": -3.5552243631332305,
                "q2width": -1.114490891184175,
                "q3width": -0.1257154777049373,
                "q4width": 1.0872777685344683,
                "mid50width": 5.428974111640412,
                "peakheight": 2.6820414630925633,
                "peakloc": 0.4900990099009901,
                "peakpercentile": 56.33302522118051,
                "peakwidth": 0.3168316831683169,
                "skewness": -0.4091197098383096,
                "skewness_p": 0.0,
                "skewness_z": -44.23408729903814,
                "voxelsincluded": 75730,
            },
        }
        if args.addgraymetrics:
            thekeydict.update(thegraymatterdict)
        if args.addwhitemetrics:
            thekeydict.update(thewhitematterdict)

    thecolumns = ["datasource"]
    thedatadict = {"datasource": []}
    for key in thekeydict.keys():
        for subkey in thekeydict[key]:
            thecolumns.append(key + "_" + str(subkey))
            thedatadict[thecolumns[-1]] = []

    if args.debug:
        print(thecolumns)

    for theinput in args.input:
        inputdict = tide_io.readdictfromjson(theinput)
        thedatadict["datasource"].append(theinput)
        for column in thecolumns[1:]:
            keyparts = column.split("_")
            try:
                thedataitem = inputdict[keyparts[0]]["_".join(keyparts[1:])]
            except KeyError:
                thedataitem = None
            thedatadict[column].append(thedataitem)
    df = pd.DataFrame(thedatadict, columns=thecolumns)
    df.to_csv(args.outputroot + ".csv", index=False)

    for column in thecolumns[1:]:
        tide_stats.makeandsavehistogram(
            df[column].to_numpy(),
            51,
            0,
            args.outputroot + "_" + column,
            displaytitle=column,
            displayplots=args.showhists,
            normalize=True,
            append=False,
            debug=False,
        )
