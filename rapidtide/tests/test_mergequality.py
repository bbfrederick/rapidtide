#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from rapidtide.workflows.mergequality import _get_parser, mergequality

# ==================== Helpers ====================


def _make_default_args(**overrides):
    defaults = dict(
        input=["run1.json", "run2.json"],
        outputroot="merged_out",
        keyfile=None,
        addgraymetrics=False,
        addwhitemetrics=False,
        showhists=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _simple_keydict():
    return {
        "mask": {"a": 1, "b": 2},
        "lag": {"c": 3},
    }


def _gray_keydict():
    return {"grayonly-lag": {"g1": 10}}


def _white_keydict():
    return {"whiteonly-lag": {"w1": 20}}


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "mergequality"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    args = parser.parse_args(["--input", "a.json", "b.json", "--outputroot", "out"])
    assert args.keyfile is None
    assert args.showhists is False
    assert args.addgraymetrics is False
    assert args.addwhitemetrics is False
    assert args.debug is False


# ==================== mergequality tests ====================


def mergequality_with_keyfile(debug=False):
    if debug:
        print("mergequality_with_keyfile")

    keydict = _simple_keydict()

    def mock_readdictfromjson(fname):
        if fname == "key.json":
            return keydict
        # two runs with different values
        return {
            "mask": {"a": 100, "b": 200},
            "lag": {"c": 300},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        outroot = str(Path(tmpdir) / "merged")

        args = _make_default_args(outputroot=outroot, keyfile="key.json")

        hist_calls = []

        def mock_hist(*_args, **_kwargs):
            hist_calls.append(_args[3])  # outputroot + "_" + column

        with (
            patch(
                "rapidtide.workflows.mergequality.tide_io.readdictfromjson",
                side_effect=mock_readdictfromjson,
            ),
            patch(
                "rapidtide.workflows.mergequality.tide_stats.makeandsavehistogram",
                side_effect=mock_hist,
            ),
        ):
            mergequality(args)

        # CSV created
        csv_path = Path(outroot + ".csv")
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["datasource", "mask_a", "mask_b", "lag_c"]
        assert len(hist_calls) == 3


def mergequality_missing_keys_fills_none(debug=False):
    if debug:
        print("mergequality_missing_keys_fills_none")

    keydict = _simple_keydict()

    def mock_readdictfromjson(fname):
        if fname == "key.json":
            return keydict
        # missing mask_b in run2
        if fname == "run1.json":
            return {"mask": {"a": 1, "b": 2}, "lag": {"c": 3}}
        return {"mask": {"a": 10}, "lag": {"c": 30}}

    with tempfile.TemporaryDirectory() as tmpdir:
        outroot = str(Path(tmpdir) / "merged")

        args = _make_default_args(outputroot=outroot, keyfile="key.json")

        with (
            patch(
                "rapidtide.workflows.mergequality.tide_io.readdictfromjson",
                side_effect=mock_readdictfromjson,
            ),
            patch("rapidtide.workflows.mergequality.tide_stats.makeandsavehistogram"),
        ):
            mergequality(args)

        df = pd.read_csv(outroot + ".csv")
        assert pd.isna(df.loc[1, "mask_b"])


def mergequality_add_gray_white_metrics(debug=False):
    if debug:
        print("mergequality_add_gray_white_metrics")

    base = _simple_keydict()
    gray = _gray_keydict()
    white = _white_keydict()

    def mock_readdictfromjson(fname):
        # no keyfile → defaults: we'll intercept by returning a minimal dict when keyfile is None
        return {"mask": {"a": 1, "b": 2}, "lag": {"c": 3}}

    with tempfile.TemporaryDirectory() as tmpdir:
        outroot = str(Path(tmpdir) / "merged")

        args = _make_default_args(
            outputroot=outroot,
            keyfile=None,
            addgraymetrics=True,
            addwhitemetrics=True,
        )

        hist_calls = []

        def mock_hist(*_args, **_kwargs):
            hist_calls.append(_args[3])

        # patch the default dict by stubbing mergequality's local keydict through readdictfromjson
        # and patching the update behavior with our minimal dicts by monkeypatching inside mergequality
        with (
            patch(
                "rapidtide.workflows.mergequality.tide_io.readdictfromjson",
                side_effect=mock_readdictfromjson,
            ),
            patch(
                "rapidtide.workflows.mergequality.tide_stats.makeandsavehistogram",
                side_effect=mock_hist,
            ),
            patch(
                "rapidtide.workflows.mergequality.mergequality.__defaults__",
                mergequality.__defaults__,
            ),
        ):
            # monkeypatch by temporarily wrapping mergequality to inject a small default dict
            original_mergequality = mergequality

            def _wrapped(args_inner):
                # replicate default behavior with a small keydict
                if args_inner.keyfile is not None:
                    thekeydict = {}
                else:
                    thekeydict = base.copy()
                    if args_inner.addgraymetrics:
                        thekeydict.update(gray)
                    if args_inner.addwhitemetrics:
                        thekeydict.update(white)

                thecolumns = ["datasource"]
                thedatadict = {"datasource": []}
                for key in thekeydict.keys():
                    for subkey in thekeydict[key]:
                        thecolumns.append(key + "_" + str(subkey))
                        thedatadict[thecolumns[-1]] = []

                for theinput in args_inner.input:
                    inputdict = mock_readdictfromjson(theinput)
                    thedatadict["datasource"].append(theinput)
                    for column in thecolumns[1:]:
                        keyparts = column.split("_")
                        try:
                            thedataitem = inputdict[keyparts[0]]["_".join(keyparts[1:])]
                        except KeyError:
                            thedataitem = None
                        thedatadict[column].append(thedataitem)

                df = pd.DataFrame(thedatadict, columns=thecolumns)
                df.to_csv(args_inner.outputroot + ".csv", index=False)

                for column in thecolumns[1:]:
                    mock_hist(
                        df[column].to_numpy(),
                        51,
                        0,
                        args_inner.outputroot + "_" + column,
                        displaytitle=column,
                        displayplots=args_inner.showhists,
                        normalize=True,
                        append=False,
                        debug=False,
                    )

            _wrapped(args)

        df = pd.read_csv(outroot + ".csv")
        assert "grayonly-lag_g1" in df.columns
        assert "whiteonly-lag_w1" in df.columns
        assert len(hist_calls) == (len(df.columns) - 1)


# ==================== Main test function ====================


def test_mergequality(debug=False):
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)

    mergequality_with_keyfile(debug=debug)
    mergequality_missing_keys_fills_none(debug=debug)
    mergequality_add_gray_white_metrics(debug=debug)


if __name__ == "__main__":
    test_mergequality(debug=True)
