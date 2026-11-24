showrapidtide
=============

``showrapidtide`` renders native 2D TIFF analysis results produced by running
``rapidtide`` on ``videotoblood`` data. Give it the output root (the prefix before
``_desc-``), rather than an individual map or a directory of results::

    showrapidtide /path/to/blood_skin_rapidtide
    showrapidtide /path/to/blood_skin_rapidtide --fps 15 --minlag -0.25 --maxlag 0.25
    showrapidtide /path/to/blood_skin_rapidtide --videolagrange -0.25 0.25
    showrapidtide /path/to/blood_skin_rapidtide --mincorr 0 --maxcorr 0.6
    showrapidtide /path/to/blood_skin_rapidtide --outdir figures --nomovie

Install the optional TIFF, Zarr, and video dependencies with ``pip install 'rapidtide[video]'``.
The original source video and blood Zarr are not required. This command currently
supports native ``(y, x)`` analysis maps, rather than volumetric NIfTI/CIFTI outputs.
TIFF files are preferred when present; legacy Zarr analysis results are also supported.
The movie reads only the required TIFF pages when selecting a lag interval.

Artifacts
---------

By default the command writes to ``<outputroot>_figures``:

* ``static-maps.png`` and ``.pdf`` contain peak-fit delay and peak correlation,
  plus each of these maps that exists: refined delay and correlation, squared
  peak correlation, peak width, mean transit time, significance, time
  percentile, mean, unfiltered mean, standard deviation, and coefficient of
  variation.
* ``corrout-lag-sweep.mp4`` sweeps through the saved ``desc-corrout_info`` lag
  coordinate, with each frame labeled in seconds. Playback is 30 frames per
  second by default, independently of acquisition sampling. Negative correlations
  are retained. This is a sweep of correlation lag, not an acquisition-time movie.
* ``corrout-zero-lag.png`` shows the selected saved lag nearest zero.
* ``render-info.json`` records the input root, geometry, display limits, panels,
  playback rate, and exact movie lag coordinates.

Delay maps use **viridis** with shared symmetric limits covering all included
finite delays, never narrower than ±0.25 seconds. ``--minlag SECONDS`` and
``--maxlag SECONDS`` override these limits, separately or together, and values
outside the selected display range saturate. The resolved limits must satisfy
``minlag < maxlag``. Static correlation maps, the correlation lag-sweep movie,
and its zero-lag preview use **hot metal** (Matplotlib ``hot``). ``--mincorr`` and ``--maxcorr``
set the correlation color limits for
all these outputs, defaulting to **−1 and 1**, including when a static map has
only positive values. For example, ``--mincorr 0 --maxcorr 0.6`` maps values at
or below zero to the low endpoint and values at or above 0.6 to the high
endpoint. Limits must be finite with ``mincorr < maxcorr``. These settings
change display colors only, preserving the stored values and delay color scales.
The movie uses the same limits for every frame; the requested limits are recorded
in its metadata and ``render-info.json``.

``--videolagrange MIN MAX`` limits the movie to saved lags within the inclusive
interval in seconds. For example, ``--videolagrange -0.25 0.25`` focuses on the
region near zero without rerunning ``rapidtide``. Bounds may be asymmetric;
the selected frames retain their original lag labels, with no interpolation.
Bounds beyond the available lags use the overlapping portion. An interval with
no saved frames is an error. Only the selected frames and their nearest-zero
preview are rendered; static maps and their color limits are unaffected.
Playback FPS is unchanged, so selecting fewer frames shortens the video.
The JSON manifest records the requested bounds and exact selected coordinates.

Native row/column geometry is preserved. Peak maps use ``desc-corrfit_mask``;
all other maps and the movie use ``desc-processed_mask``. Excluded and nonfinite
pixels are gray. Nonfinite included map values also produce a warning.

Existing generated files are overwritten after successful rendering; failed
rendering leaves previous artifacts in place. Analysis outputs are read only.
If ``corrout`` is absent, maps are still generated with a warning; save correlation
functions during analysis with ``rapidtide --savecorrout`` to enable the movie.
``--nomovie`` requests static maps only. ``--noprogressbar`` suppresses encoding
progress. ``--fps`` controls movie playback speed.
