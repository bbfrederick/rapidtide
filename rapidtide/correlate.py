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
from rapidtide.core.signal import correlate as _impl
from rapidtide.core.signal.correlate import *  # noqa: F401,F403

tide_fit = _impl.tide_fit
tide_math = _impl.tide_math
tide_resample = _impl.tide_resample
tide_stats = _impl.tide_stats
tide_util = _impl.tide_util

# Preserve selected private helpers used by existing tests/callers.
_centered = _impl._centered
_check_valid_mode_shapes = _impl._check_valid_mode_shapes

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
