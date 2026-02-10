#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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

import rapidtide.Colortables as ct

# ==================== Helper ====================


def _validate_gradient_state(state, expected_name, must_have_ticks=True):
    """Validate that a gradient state dict has the expected structure."""
    assert isinstance(state, dict)
    assert state["name"] == expected_name
    assert "mode" in state
    if must_have_ticks:
        assert "ticks" in state
        assert isinstance(state["ticks"], list)
        assert len(state["ticks"]) >= 2
        for tick in state["ticks"]:
            assert len(tick) == 2
            pos, color = tick
            assert 0.0 <= float(pos) <= 1.0
            assert len(color) == 4


# ==================== setendalpha tests ====================


def setendalpha_first_at_zero_rgb(debug=False):
    """Test setendalpha when first tick is at 0.0, RGB mode."""
    if debug:
        print("setendalpha_first_at_zero_rgb")
    thestate = {
        "ticks": [
            (0.0, (100, 150, 200, 255)),
            (0.5, (50, 50, 50, 255)),
            (1.0, (200, 100, 50, 255)),
        ],
        "mode": "rgb",
    }
    result = ct.setendalpha(thestate, alpha=128, debug=debug)
    assert isinstance(result, dict)
    assert result["mode"] == "rgb"
    ticks = result["ticks"]
    # First tick should be at 0.0 with alpha=128
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 128
    # Second tick should be at 0.0001 with original color
    assert ticks[1][0] == 0.0001
    assert ticks[1][1] == (100, 150, 200, 255)
    # Last tick should be at 1.0 with alpha=128
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1][3] == 128
    # Second-to-last should be at 0.9999 with original last color
    assert ticks[-2][0] == 0.9999
    assert ticks[-2][1] == (200, 100, 50, 255)


def setendalpha_first_not_at_zero_rgb(debug=False):
    """Test setendalpha when first tick is not at 0.0, RGB mode."""
    if debug:
        print("setendalpha_first_not_at_zero_rgb")
    thestate = {
        "ticks": [
            (0.2, (255, 0, 0, 255)),
            (0.8, (0, 255, 0, 255)),
        ],
        "mode": "rgb",
    }
    result = ct.setendalpha(thestate, alpha=64, debug=debug)
    ticks = result["ticks"]
    # First tick at 0.0 with black + specified alpha
    assert ticks[0][0] == 0.0
    assert ticks[0][1] == (0, 0, 0, 64)
    # Second tick at 0.0001 with black + original alpha
    assert ticks[1][0] == 0.0001
    assert ticks[1][1] == (0, 0, 0, 255)
    # Middle tick preserved
    assert ticks[2][0] == 0.2
    assert ticks[2][1] == (255, 0, 0, 255)
    # End ticks: 0.9999 with white + original alpha, 1.0 with white + specified alpha
    assert ticks[-2][0] == 0.9999
    assert ticks[-2][1] == (255, 255, 255, 255)
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1] == (255, 255, 255, 64)


def setendalpha_last_not_at_one_hsv(debug=False):
    """Test setendalpha when last tick is not at 1.0, HSV mode."""
    if debug:
        print("setendalpha_last_not_at_one_hsv")
    thestate = {
        "ticks": [
            (0.0, (100, 200, 50, 200)),
            (0.7, (50, 100, 150, 200)),
        ],
        "mode": "hsv",
    }
    result = ct.setendalpha(thestate, alpha=0, debug=debug)
    ticks = result["ticks"]
    assert result["mode"] == "hsv"
    # First tick: 0.0 with alpha=0
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 0
    # Last tick: 1.0 with red + alpha=0 (HSV mode)
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1] == (255, 0, 0, 0)
    # Second-to-last: 0.9999 with red + original alpha
    assert ticks[-2][0] == 0.9999
    assert ticks[-2][1] == (255, 0, 0, 200)


def setendalpha_preserves_middle_ticks(debug=False):
    """Test that setendalpha preserves all middle ticks."""
    if debug:
        print("setendalpha_preserves_middle_ticks")
    thestate = {
        "ticks": [
            (0.0, (10, 20, 30, 255)),
            (0.25, (40, 50, 60, 255)),
            (0.5, (70, 80, 90, 255)),
            (0.75, (100, 110, 120, 255)),
            (1.0, (130, 140, 150, 255)),
        ],
        "mode": "rgb",
    }
    result = ct.setendalpha(thestate, alpha=100, debug=debug)
    ticks = result["ticks"]
    # Middle ticks (0.25, 0.5, 0.75) should be preserved
    middle = ticks[2:-2]
    assert len(middle) == 3
    assert middle[0] == (0.25, (40, 50, 60, 255))
    assert middle[1] == (0.5, (70, 80, 90, 255))
    assert middle[2] == (0.75, (100, 110, 120, 255))


def setendalpha_unsorted_ticks(debug=False):
    """Test that setendalpha handles unsorted ticks correctly."""
    if debug:
        print("setendalpha_unsorted_ticks")
    thestate = {
        "ticks": [
            (1.0, (255, 255, 255, 255)),
            (0.0, (0, 0, 0, 255)),
            (0.5, (128, 128, 128, 255)),
        ],
        "mode": "rgb",
    }
    result = ct.setendalpha(thestate, alpha=50, debug=debug)
    ticks = result["ticks"]
    # Should still produce valid output with sorted positions
    positions = [t[0] for t in ticks]
    assert positions == sorted(positions)
    assert ticks[0][0] == 0.0
    assert ticks[-1][0] == 1.0


def setendalpha_alpha_zero(debug=False):
    """Test setendalpha with alpha=0 (fully transparent)."""
    if debug:
        print("setendalpha_alpha_zero")
    thestate = {
        "ticks": [
            (0.0, (255, 0, 0, 255)),
            (1.0, (0, 0, 255, 255)),
        ],
        "mode": "rgb",
    }
    result = ct.setendalpha(thestate, alpha=0)
    assert result["ticks"][0][1][3] == 0
    assert result["ticks"][-1][1][3] == 0


def setendalpha_alpha_255(debug=False):
    """Test setendalpha with alpha=255 (fully opaque)."""
    if debug:
        print("setendalpha_alpha_255")
    thestate = {
        "ticks": [
            (0.0, (255, 0, 0, 128)),
            (1.0, (0, 0, 255, 128)),
        ],
        "mode": "rgb",
    }
    result = ct.setendalpha(thestate, alpha=255)
    assert result["ticks"][0][1][3] == 255
    assert result["ticks"][-1][1][3] == 255


# ==================== gen_gray_state tests ====================


def gen_gray_state_basic(debug=False):
    """Test gen_gray_state returns correct structure."""
    if debug:
        print("gen_gray_state_basic")
    state = ct.gen_gray_state()
    _validate_gradient_state(state, "gray")
    assert state["mode"] == "rgb"
    assert len(state["ticks"]) == 2
    assert state["ticks"][0] == (0.0, (0, 0, 0, 255))
    assert state["ticks"][1] == (1.0, (255, 255, 255, 255))


# ==================== gen_grey_state tests ====================


def gen_grey_state_basic(debug=False):
    """Test gen_grey_state returns correct structure."""
    if debug:
        print("gen_grey_state_basic")
    state = ct.gen_grey_state()
    _validate_gradient_state(state, "grey")
    assert state["mode"] == "rgb"
    assert len(state["ticks"]) == 2
    assert state["ticks"][0] == (0.0, (0, 0, 0, 255))
    assert state["ticks"][1] == (1.0, (255, 255, 255, 255))


def gen_gray_grey_equivalent(debug=False):
    """Test that gray and grey have identical tick data."""
    if debug:
        print("gen_gray_grey_equivalent")
    gray = ct.gen_gray_state()
    grey = ct.gen_grey_state()
    assert gray["ticks"] == grey["ticks"]
    assert gray["mode"] == grey["mode"]
    # Names differ
    assert gray["name"] != grey["name"]


# ==================== gen_g2y2r_state tests ====================


def gen_g2y2r_state_basic(debug=False):
    """Test gen_g2y2r_state returns correct structure."""
    if debug:
        print("gen_g2y2r_state_basic")
    state = ct.gen_g2y2r_state()
    _validate_gradient_state(state, "g2y2r")
    assert state["mode"] == "rgb"
    assert len(state["ticks"]) == 5
    # Check transparency at endpoints
    assert state["ticks"][0][1][3] == 0  # transparent start
    assert state["ticks"][-1][1][3] == 0  # transparent end
    # Check colors at key positions
    assert state["ticks"][1][1] == (0, 255, 0, 255)  # green
    assert state["ticks"][2][1] == (255, 255, 0, 255)  # yellow
    assert state["ticks"][3][1] == (255, 0, 0, 255)  # red


# ==================== gen_mask_state tests ====================


def gen_mask_state_basic(debug=False):
    """Test gen_mask_state returns correct structure."""
    if debug:
        print("gen_mask_state_basic")
    state = ct.gen_mask_state()
    _validate_gradient_state(state, "mask")
    assert state["mode"] == "rgb"
    assert len(state["ticks"]) == 2
    assert state["ticks"][0] == (0.0, (0, 0, 0, 255))
    assert state["ticks"][1] == (1.0, (255, 255, 255, 0))


# ==================== gen_greyclip_state tests ====================


def gen_greyclip_state_basic(debug=False):
    """Test gen_greyclip_state returns correct structure."""
    if debug:
        print("gen_greyclip_state_basic")
    state = ct.gen_greyclip_state()
    _validate_gradient_state(state, "greyclip")
    assert state["mode"] == "rgb"
    assert len(state["ticks"]) == 3
    assert state["ticks"][0] == (0.0, (0, 0, 0, 255))
    assert state["ticks"][1] == (0.99, (255, 255, 255, 255))
    assert state["ticks"][2] == (1.0, (255, 0, 0, 255))


# ==================== pyqtgraph gradient generator tests ====================


def gen_thermal_state_basic(debug=False):
    """Test gen_thermal_state returns correct structure."""
    if debug:
        print("gen_thermal_state_basic")
    state = ct.gen_thermal_state()
    _validate_gradient_state(state, "thermal")
    assert state["mode"] in ("rgb", "hsv")


def gen_flame_state_basic(debug=False):
    """Test gen_flame_state returns correct structure."""
    if debug:
        print("gen_flame_state_basic")
    state = ct.gen_flame_state()
    _validate_gradient_state(state, "flame")


def gen_yellowy_state_basic(debug=False):
    """Test gen_yellowy_state returns correct structure."""
    if debug:
        print("gen_yellowy_state_basic")
    state = ct.gen_yellowy_state()
    _validate_gradient_state(state, "yellowy")


def gen_bipolar_state_basic(debug=False):
    """Test gen_bipolar_state returns correct structure."""
    if debug:
        print("gen_bipolar_state_basic")
    state = ct.gen_bipolar_state()
    _validate_gradient_state(state, "bipolar")


def gen_spectrum_state_basic(debug=False):
    """Test gen_spectrum_state returns correct structure."""
    if debug:
        print("gen_spectrum_state_basic")
    state = ct.gen_spectrum_state()
    _validate_gradient_state(state, "spectrum")


def gen_turbo_state_basic(debug=False):
    """Test gen_turbo_state returns correct structure."""
    if debug:
        print("gen_turbo_state_basic")
    state = ct.gen_turbo_state()
    # turbo uses colorMap, not ticks
    assert isinstance(state, dict)
    assert state["name"] == "turbo"


def gen_viridis_state_basic(debug=False):
    """Test gen_viridis_state returns correct structure."""
    if debug:
        print("gen_viridis_state_basic")
    state = ct.gen_viridis_state()
    assert isinstance(state, dict)
    assert state["name"] == "viridis"


def gen_inferno_state_basic(debug=False):
    """Test gen_inferno_state returns correct structure."""
    if debug:
        print("gen_inferno_state_basic")
    state = ct.gen_inferno_state()
    assert isinstance(state, dict)
    assert state["name"] == "inferno"


def gen_plasma_state_basic(debug=False):
    """Test gen_plasma_state returns correct structure."""
    if debug:
        print("gen_plasma_state_basic")
    state = ct.gen_plasma_state()
    assert isinstance(state, dict)
    assert state["name"] == "plasma"


def gen_magma_state_basic(debug=False):
    """Test gen_magma_state returns correct structure."""
    if debug:
        print("gen_magma_state_basic")
    state = ct.gen_magma_state()
    assert isinstance(state, dict)
    assert state["name"] == "magma"


# ==================== getagradient tests ====================


def getagradient_basic(debug=False):
    """Test getagradient returns a GradientWidget."""
    if debug:
        print("getagradient_basic")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    gw = ct.getagradient()
    from pyqtgraph import GradientWidget

    assert isinstance(gw, GradientWidget)


# ==================== setendalpha with real gradients ====================


def setendalpha_with_gray(debug=False):
    """Test setendalpha with the gray gradient."""
    if debug:
        print("setendalpha_with_gray")
    state = ct.gen_gray_state()
    result = ct.setendalpha(state, alpha=128, debug=debug)
    ticks = result["ticks"]
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 128
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1][3] == 128
    assert result["mode"] == "rgb"


def setendalpha_with_g2y2r(debug=False):
    """Test setendalpha with the g2y2r gradient."""
    if debug:
        print("setendalpha_with_g2y2r")
    state = ct.gen_g2y2r_state()
    result = ct.setendalpha(state, alpha=50, debug=debug)
    ticks = result["ticks"]
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 50
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1][3] == 50


def setendalpha_with_greyclip(debug=False):
    """Test setendalpha with the greyclip gradient."""
    if debug:
        print("setendalpha_with_greyclip")
    state = ct.gen_greyclip_state()
    result = ct.setendalpha(state, alpha=200, debug=debug)
    ticks = result["ticks"]
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 200
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1][3] == 200
    # Intermediate tick at 0.99 should be preserved
    middle_positions = [t[0] for t in ticks[2:-2]]
    assert 0.99 in middle_positions


def setendalpha_with_thermal(debug=False):
    """Test setendalpha with a pyqtgraph thermal gradient."""
    if debug:
        print("setendalpha_with_thermal")
    state = ct.gen_thermal_state()
    result = ct.setendalpha(state, alpha=100, debug=debug)
    ticks = result["ticks"]
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 100
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1][3] == 100


def setendalpha_with_mask(debug=False):
    """Test setendalpha with the mask gradient."""
    if debug:
        print("setendalpha_with_mask")
    state = ct.gen_mask_state()
    result = ct.setendalpha(state, alpha=0, debug=debug)
    ticks = result["ticks"]
    assert ticks[0][0] == 0.0
    assert ticks[0][1][3] == 0
    assert ticks[-1][0] == 1.0
    assert ticks[-1][1][3] == 0


# ==================== Main test function ====================


def test_colortables(debug=False):
    # setendalpha
    if debug:
        print("Running setendalpha tests")
    setendalpha_first_at_zero_rgb(debug=debug)
    setendalpha_first_not_at_zero_rgb(debug=debug)
    setendalpha_last_not_at_one_hsv(debug=debug)
    setendalpha_preserves_middle_ticks(debug=debug)
    setendalpha_unsorted_ticks(debug=debug)
    setendalpha_alpha_zero(debug=debug)
    setendalpha_alpha_255(debug=debug)

    # hardcoded gradient generators
    if debug:
        print("Running hardcoded gradient tests")
    gen_gray_state_basic(debug=debug)
    gen_grey_state_basic(debug=debug)
    gen_gray_grey_equivalent(debug=debug)
    gen_g2y2r_state_basic(debug=debug)
    gen_mask_state_basic(debug=debug)
    gen_greyclip_state_basic(debug=debug)

    # pyqtgraph gradient generators
    if debug:
        print("Running pyqtgraph gradient tests")
    gen_thermal_state_basic(debug=debug)
    gen_flame_state_basic(debug=debug)
    gen_yellowy_state_basic(debug=debug)
    gen_bipolar_state_basic(debug=debug)
    gen_spectrum_state_basic(debug=debug)
    gen_turbo_state_basic(debug=debug)
    gen_viridis_state_basic(debug=debug)
    gen_inferno_state_basic(debug=debug)
    gen_plasma_state_basic(debug=debug)
    gen_magma_state_basic(debug=debug)

    # getagradient (needs QApp)
    if debug:
        print("Running getagradient test")
    getagradient_basic(debug=debug)

    # setendalpha with real gradients
    if debug:
        print("Running setendalpha integration tests")
    setendalpha_with_gray(debug=debug)
    setendalpha_with_g2y2r(debug=debug)
    setendalpha_with_greyclip(debug=debug)
    setendalpha_with_thermal(debug=debug)
    setendalpha_with_mask(debug=debug)


if __name__ == "__main__":
    test_colortables(debug=True)
