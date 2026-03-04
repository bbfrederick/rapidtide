#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Benchmark SimilarityFunctionFitter Gaussian performance and robust-fallback rate.
#
import argparse
import os
import sys
import time

import numpy as np

# Force imports from the local workspace checkout.
REPOROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPOROOT not in sys.path:
    sys.path.insert(0, REPOROOT)

import rapidtide.fit as tide_fit
import rapidtide.simFuncClasses as tide_simfunc


def _oldstyle_xaxis(npts: int, sampletime: float) -> np.ndarray:
    return np.r_[0.0:npts] * sampletime - (npts * sampletime) / 2.0 + sampletime / 2.0


def _newstyle_xaxis(npts: int, sampletime: float) -> np.ndarray:
    return (np.arange(npts, dtype="float64") - (npts // 2)) * sampletime


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_similarityfitter",
        description="Benchmark SimilarityFunctionFitter gaussian fit runtime and robust fallback rate.",
    )
    parser.add_argument("--nfits", type=int, default=10000, help="Number of fits to run.")
    parser.add_argument(
        "--problem-fraction",
        type=float,
        default=0.1,
        help="Fraction of fits using old-style x-axis to stress tiny-center fallback.",
    )
    parser.add_argument("--npts", type=int, default=201, help="Number of x points.")
    parser.add_argument("--sampletime", type=float, default=0.5, help="Sample spacing in seconds.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    x_new = _newstyle_xaxis(args.npts, args.sampletime)
    x_old = _oldstyle_xaxis(args.npts, args.sampletime)

    fitter_new = tide_simfunc.SimilarityFunctionFitter(
        corrtimeaxis=x_new,
        lagmin=-20.0,
        lagmax=20.0,
        peakfittype="gauss",
        zerooutbadfit=False,
    )
    fitter_old = tide_simfunc.SimilarityFunctionFitter(
        corrtimeaxis=x_old,
        lagmin=-20.0,
        lagmax=20.0,
        peakfittype="gauss",
        zerooutbadfit=False,
    )

    timings = np.zeros(args.nfits, dtype="float64")
    problem_prob = np.clip(args.problem_fraction, 0.0, 1.0)
    problem_count = 0

    for i in range(args.nfits):
        use_problem = rng.random() < problem_prob
        fitter = fitter_old if use_problem else fitter_new
        x = x_old if use_problem else x_new
        if use_problem:
            problem_count += 1

        # mostly realistic peaks, with occasional asymmetry to trigger fallback.
        amp = 0.8 + 0.15 * (rng.random() - 0.5)
        lag = rng.uniform(-12.0, 12.0)
        sigma = np.exp(rng.uniform(np.log(0.2), np.log(20.0)))
        y = tide_fit.gauss_eval(x, np.array([amp, lag, sigma], dtype="float64"))

        t0 = time.perf_counter()
        fitter.fit(y)
        timings[i] = time.perf_counter() - t0

    total_calls = fitter_new.gauss_fit_calls + fitter_old.gauss_fit_calls
    fast_calls = fitter_new.gauss_fastpath_calls + fitter_old.gauss_fastpath_calls
    robust_calls = (
        fitter_new.gauss_robust_fallback_calls + fitter_old.gauss_robust_fallback_calls
    )

    mean_ms = 1.0e3 * np.mean(timings)
    p95_ms = 1.0e3 * np.percentile(timings, 95.0)
    p99_ms = 1.0e3 * np.percentile(timings, 99.0)
    robust_pct = 100.0 * robust_calls / max(total_calls, 1)

    print("SimilarityFunctionFitter benchmark")
    print(f"nfits: {args.nfits}")
    print(f"npts: {args.npts}")
    print(f"problem_fraction_target: {problem_prob:.3f}")
    print(f"problem_fraction_actual: {problem_count / max(args.nfits, 1):.3f}")
    print(f"mean_ms_per_fit: {mean_ms:.4f}")
    print(f"p95_ms_per_fit: {p95_ms:.4f}")
    print(f"p99_ms_per_fit: {p99_ms:.4f}")
    print(f"gauss_fit_calls: {total_calls}")
    print(f"fastpath_calls: {fast_calls}")
    print(f"robust_fallback_calls: {robust_calls}")
    print(f"robust_fallback_rate_pct: {robust_pct:.3f}")


if __name__ == "__main__":
    main()
