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

from rapidtide.workflows.adjustoffset import _get_parser as adjustoffset_getparser
from rapidtide.workflows.aligntcs import _get_parser as aligntcs_getparser
from rapidtide.workflows.applydlfilter import _get_parser as applydlfilter_getparser
from rapidtide.workflows.atlasaverage import _get_parser as atlasaverage_getparser
from rapidtide.workflows.atlastool import _get_parser as atlastool_getparser
from rapidtide.workflows.calctexticc import _get_parser as calctexticc_getparser
from rapidtide.workflows.ccorrica import _get_parser as ccorrica_getparser
from rapidtide.workflows.delayvar import _get_parser as delayvar_getparser
from rapidtide.workflows.diffrois import _get_parser as diffrois_getparser
from rapidtide.workflows.endtidalproc import _get_parser as endtidalproc_getparser
from rapidtide.workflows.fdica import _get_parser as fdica_getparser
from rapidtide.workflows.filtnifti import _get_parser as filtnifti_getparser
from rapidtide.workflows.filttc import _get_parser as filttc_getparser
from rapidtide.workflows.fixtr import _get_parser as fixtr_getparser
from rapidtide.workflows.gmscalc import _get_parser as gmscalc_getparser
from rapidtide.workflows.happy_parser import _get_parser as happy_parser_getparser
from rapidtide.workflows.happy2std import _get_parser as happy2std_getparser
from rapidtide.workflows.histnifti import _get_parser as histnifti_getparser
from rapidtide.workflows.histtc import _get_parser as histtc_getparser
from rapidtide.workflows.linfitfilt import _get_parser as linfitfilt_getparser
from rapidtide.workflows.localflow import _get_parser as localflow_getparser
from rapidtide.workflows.mergequality import _get_parser as mergequality_getparser
from rapidtide.workflows.niftidecomp import _get_parser_temporal as niftidecomp_getparser_temporal
from rapidtide.workflows.niftidecomp import _get_parser_spatial as niftidecomp_getparser_spatial
from rapidtide.workflows.niftistats import _get_parser as niftistats_getparser
from rapidtide.workflows.pairproc import _get_parser as pairproc_getparser
from rapidtide.workflows.pairwisemergenifti import _get_parser as pairwisemergenifti_getparser
from rapidtide.workflows.physiofreq import _get_parser as physiofreq_getparser
from rapidtide.workflows.pixelcomp import _get_parser as pixelcomp_getparser
from rapidtide.workflows.plethquality import _get_parser as plethquality_getparser
from rapidtide.workflows.polyfitim import _get_parser as polyfitim_getparser
from rapidtide.workflows.proj2flow import _get_parser as proj2flow_getparser
from rapidtide.workflows.rankimage import _get_parser as rankimage_getparser
from rapidtide.workflows.rapidtide2std import _get_parser as rapidtide2std_getparser
from rapidtide.workflows.resamplenifti import _get_parser as resamplenifti_getparser
from rapidtide.workflows.resampletc import _get_parser as resampletc_getparser
from rapidtide.workflows.retrolagtcs import _get_parser as retrolagtcs_getparser
from rapidtide.workflows.retroregress import _get_parser as retroregress_getparser
from rapidtide.workflows.roisummarize import _get_parser as roisummarize_getparser
from rapidtide.workflows.runqualitycheck import _get_parser as runqualitycheck_getparser
from rapidtide.workflows.showarbcorr import _get_parser as showarbcorr_getparser
from rapidtide.workflows.showhist import _get_parser as showhist_getparser
from rapidtide.workflows.showstxcorr import _get_parser as showstxcorr_getparser
from rapidtide.workflows.showtc import _get_parser as showtc_getparser
from rapidtide.workflows.showxcorrx import _get_parser as showxcorrx_getparser
from rapidtide.workflows.showxy import _get_parser as showxy_getparser
from rapidtide.workflows.simdata import _get_parser as simdata_getparser
from rapidtide.workflows.spatialfit import _get_parser as spatialfit_getparser
from rapidtide.workflows.spatialmi import _get_parser as spatialmi_getparser
from rapidtide.workflows.spectrogram import _get_parser as spectrogram_getparser
from rapidtide.workflows.synthASL import _get_parser as synthASL_getparser
from rapidtide.workflows.tcfrom2col import _get_parser as tcfrom2col_getparser
from rapidtide.workflows.tcfrom3col import _get_parser as tcfrom3col_getparser
from rapidtide.workflows.variabilityizer import _get_parser as variabilityizer_getparser


def test_parsers(debug=False):
    parserlist = [ adjustoffset_getparser,
        aligntcs_getparser,
        applydlfilter_getparser,
        atlasaverage_getparser,
        atlastool_getparser,
        calctexticc_getparser,
        ccorrica_getparser,
        delayvar_getparser,
        diffrois_getparser,
        endtidalproc_getparser,
        fdica_getparser,
        filtnifti_getparser,
        filttc_getparser,
        fixtr_getparser,
        gmscalc_getparser,
        happy_parser_getparser,
        happy2std_getparser,
        histnifti_getparser,
        histtc_getparser,
        linfitfilt_getparser,
        localflow_getparser,
        mergequality_getparser,
        niftidecomp_getparser_temporal,
        niftidecomp_getparser_spatial,
        niftistats_getparser,
        pairproc_getparser,
        pairwisemergenifti_getparser,
        physiofreq_getparser,
        pixelcomp_getparser,
        plethquality_getparser,
        polyfitim_getparser,
        proj2flow_getparser,
        rankimage_getparser,
        rapidtide2std_getparser,
        resamplenifti_getparser,
        resampletc_getparser,
        retrolagtcs_getparser,
        retroregress_getparser,
        roisummarize_getparser,
        runqualitycheck_getparser,
        showarbcorr_getparser,
        showhist_getparser,
        showstxcorr_getparser,
        showtc_getparser,
        showxcorrx_getparser,
        showxy_getparser,
        simdata_getparser,
        spatialfit_getparser,
        spatialmi_getparser,
        spectrogram_getparser,
        synthASL_getparser,
        tcfrom2col_getparser,
        tcfrom3col_getparser,
        variabilityizer_getparser ]

    for thegetparser in parserlist:
        theusage = thegetparser().format_help()
        if debug:
            print(theusage)


if __name__ == "__main__":
    test_parsers(debug=True)
