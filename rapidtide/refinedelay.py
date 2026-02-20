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
from rapidtide.core.delay import refinedelay as _impl
from rapidtide.core.delay.refinedelay import *  # noqa: F401,F403

tide_filt = _impl.tide_filt
tide_io = _impl.tide_io
tide_regressfrommaps = _impl.tide_regressfrommaps

CubicSpline = _impl.CubicSpline
UnivariateSpline = _impl.UnivariateSpline
median_filter = _impl.median_filter
mad = _impl.mad
poly = _impl.poly

__all__ = [name for name in dir(_impl) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_impl, name)
