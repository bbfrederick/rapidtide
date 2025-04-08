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
import rapidtide.Colortables as tide_colortables


def test_colortables():
    thestate = tide_colortables.gen_thermal_state()
    thedjustedstate = tide_colortables.setendalpha(thestate, 0.5, debug=True)
    thestate = tide_colortables.gen_flame_state()
    thestate = tide_colortables.gen_yellowy_state()
    thestate = tide_colortables.gen_bipolar_state()
    thestate = tide_colortables.gen_spectrum_state()
    thestate = tide_colortables.gen_turbo_state()
    thestate = tide_colortables.gen_gray_state()
    thestate = tide_colortables.gen_grey_state()
    thedjustedstate = tide_colortables.setendalpha(thestate, 0.5, debug=True)
    thestate = tide_colortables.gen_viridis_state()
    thestate = tide_colortables.gen_inferno_state()
    thestate = tide_colortables.gen_plasma_state()
    thestate = tide_colortables.gen_magma_state()
    thestate = tide_colortables.gen_g2y2r_state()
    thestate = tide_colortables.gen_mask_state()
    thestate = tide_colortables.gen_greyclip_state()
    thewidget = tide_colortables.getagradient()


if __name__ == "__main__":
    test_colortables()
