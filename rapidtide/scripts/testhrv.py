#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import heartpy as hp
from matplotlib.pyplot import figure, plot, savefig, setp, show


def main():
    data = hp.get_data("../data/examples/dst/happy_flip_cardfromfmri_dlfiltered_25.0Hz.txt")
    Fs = 25.0
    filtered = hp.filtersignal(data, cutoff=5, sample_rate=Fs, order=3)
    working_data, measures = hp.process(
        filtered, Fs, calc_freq=True, high_precision=True, high_precision_fs=1000
    )

    hp.plotter(working_data, measures)
    peaktimes = working_data["peaklist"][1:] / Fs

    plot(peaktimes, working_data["RR_list"] / 1000.0)
    show()
    print(measures)
    print(working_data)
    for metric in [
        "bpm",
        "ibi",
        "sdnn",
        "sdsd",
        "rmssd",
        "pnn20",
        "pnn50",
        "hr_mad",
        "breathingrate",
        "lf",
        "hf",
        "lf/hf",
    ]:
        print(metric + ":", measures[metric])

    working_data, measures = hp.process_segmentwise(
        data, Fs, segment_width=10, segment_overlap=0.25
    )
    hp.segment_plotter(working_data, measures, title="Segmentwise plot", path=".")
    # print(measures)
    # print(working_data)


if __name__ == "__main__":
    main()
