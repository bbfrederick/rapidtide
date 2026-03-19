⏺ Everything is working. Here's a summary of what was implemented:

  ---
  Implementation Summary

  Feature 1: --preppass (internal preparation pass)

  rapidtide/workflows/rapidtide_parser.py — 3 new args in experimental group:
  - --preppass (bool, default False)
  - --preppass-lag-window SECONDS (float, default 3.0)
  - --preppass-r2-threshold FLOAT (float, default 0.3)

  rapidtide/workflows/rapidtide.py — inline prep pass block inserted before the main loop. Steps:
  1. Runs correlationpass with the current resampref_y
  2. Runs fitSimFunc with despeckle_passes=0
  3. Computes modal lag via makehistogram(refine=True)
  4. Selects voxels with lag ∈ [mode − window, mode] and R² ≥ threshold
  5. Warns if < 50 good voxels; skips rebuild
  6. Calls tide_mask.getregionsignal on fmri_data_valid → filtered + resampled → new resampref_y
  7. Resets lagtimes, lagstrengths, lagsigma, R2, fitmask, failreason to zero

  Feature 2: --sharpenregressor (Wiener deconvolution)

  rapidtide/correlate.py — new function find_all_acf_sidelobes():
  - Returns all (lag, amplitude) sidelobe peaks from an ACF (both ±lags), sorted by descending |amplitude|

  rapidtide/workflows/rapidtide.py — two new module-level functions:
  - _compute_acf(signal, oversamptr, lagmax) — fast normalised ACF via np.correlate
  - _sharpen_regressor(...) — Wiener deconvolution with multi-echo iterative fallback; saves original + sharpened to TSV

  Inline call block inserted between echocancel and prewhitening blocks. Uses optiondict["sharpenregressor_noise_level"] and optiondict["sharpenregressor_max_iters"].

  Both features are opt-in (defaults off) and compose independently.
