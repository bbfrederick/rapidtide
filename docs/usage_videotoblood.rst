videotoblood
------------

Convert RGB video into a single-channel, mean-removed blood-sensitive image
time series on a regular sampling grid. This offline preprocessing tool
interpolates missing frames before signal extraction. The main ``rapidtide``
command reads this Zarr output directly.

Installation and example
^^^^^^^^^^^^^^^^^^^^^^^^

Install the optional video dependencies (PyAV and Zarr)::

    pip install -e '.[video]'

Run the standalone command::

    videotoblood recording.mp4 blood.zarr --method grgb
    videotoblood recording.mp4 blood_pos.zarr --method pos --chunks 64 64 16
    videotoblood recording.mp4 blood_30hz.zarr --framerate 30
    rapidtide blood.zarr analysis

The output directory must not already exist. The first video stream is decoded
three times: to establish frame timing, to estimate whole-recording statistics
on the regularized RGB sequence, and to write the projected signals. Only the
timestamp vectors scale with recording duration; image memory scales with frame
dimensions and the temporal chunk length. Only a completed store is published;
a failed conversion removes its temporary store.

Sampling regularization
^^^^^^^^^^^^^^^^^^^^^^^

The output grid starts at the first presentation timestamp. Its rate defaults
to the decoder's nominal (guessed) frame rate, falling back to the reciprocal
median observed frame interval when unavailable. ``--framerate HZ`` explicitly
sets the output rate. Unlike a timestamp override, this interpolates actual
RGB intensities between neighboring observed frames, filling missing frames
and regularizing timing jitter. No frames are extrapolated beyond the observed
interval, so a fractional final interval is omitted. Large gaps are also
linearly interpolated; the maximum observed gap is recorded in metadata.

Interpolation precedes projection and estimation of whole-recording means and
variance weights. It uses two neighboring frames at a time, preserves fractional
RGB values internally, and adds no temporal bandpass filter. Missing, nonfinite,
or nonincreasing source timestamps fail conversion because missing frames
cannot be placed reliably without acquisition timing. Original presentation
timestamps are saved in ``source_timestamps`` for provenance.

Signal extraction
^^^^^^^^^^^^^^^^^

Each pixel is processed independently, using TelePlethy's projection formulas
with whole-recording statistics instead of real-time sliding windows. RGB
channels are divided by their respective temporal means before projection.

* **GRGB** (default): ``g/r + g/b`` on mean-normalized channels, followed by
  whole-recording mean removal and multiplication by 100.
* **POS**: TelePlethy's adaptive basis, constructed by projecting
  ``[0, 1, -1]/sqrt(2)`` onto the plane orthogonal to the mean skin-color vector.
  The second basis vector is the cross product of the skin normal and the
  first basis vector. The two normalized-RGB projections are centered and
  combined as ``100 * (s1 + alpha*s2) / sqrt(1 + alpha**2)``, where
  ``alpha = std(s1)/std(s2)``.
* **CHROM**: ``x = 3*r - 2*g`` and ``y = 1.5*r + g - 1.5*b``. The centered
  projections are combined using TelePlethy's polarity and normalization:
  ``-100 * (x - alpha*y) / sqrt(1 + alpha**2)``, with
  ``alpha = std(x)/std(y)``.

The means, skin normals, and variance weights are estimated separately at each
pixel over all regularly sampled frames. Float64 running means and centered
covariances are used internally; output samples are float32. Values are relative RGB
projections, not calibrated blood volume or oxygenation measurements.

There is no temporal bandpass, moving mean, detrending, spatial smoothing,
motion correction, or skin segmentation. Temporal interpolation can affect
high-frequency content, especially across long gaps; band selection remains
the responsibility of downstream analysis. Inputs should already have the
spatial alignment needed for pixelwise analysis. Decoding uses 16-bit RGB to
retain up to 16-bit channel precision;
it cannot recover precision or remove artifacts lost during source encoding.

Output layout
^^^^^^^^^^^^^

The output is a Zarr format 2 group, written using Zarr-Python 3, containing:

* ``blood``: float32 array with shape ``(height, width, frames)`` and axes
  ``(y, x, time)``. Original decoded row/column orientation is retained.
* ``timestamps``: float64 regularly spaced sample times in seconds, retaining
  the original start time.
* ``source_timestamps``: original float64 presentation times in seconds,
  including jitter and gaps, with one entry per decoded source frame.
* ``valid_mask``: boolean ``(height, width)`` array. Pixels with undefined
  projections contain NaN for their entire blood time course.

Nonpositive channel means invalidate all methods. GRGB also requires positive
red and blue denominators throughout the recording (normalized values greater
than float32 epsilon). POS and CHROM require a nondegenerate second projection
(standard deviation greater than float32 epsilon). Constant pixels therefore
produce zero for GRGB and are invalid for POS/CHROM. Conversion fails if fewer
than two frames are decoded or no pixels have a valid projection.

Group attributes record the method, whole-recording statistics, scaling,
software versions, source filename, command line, axes, timing, and lack of
temporal bandpass filtering. ``uniform_sampling`` is true and
``sampling_interval_seconds`` records the output step. ``resampled`` is true,
and ``interpolation`` records ``linear_rgb_before_projection``. The requested
output rate, source-frame count, and maximum source-frame gap are also recorded.

Analysis in Rapidtide
^^^^^^^^^^^^^^^^^^^^^

``rapidtide blood.zarr analysis`` infers the sampling interval from the stored
timestamps. It checks their regularity independently of metadata flags and
rejects conflicting ``--datatstep`` or ``--datafreq`` settings. Old stores
containing irregular timestamps must be regenerated from video with the updated
converter; analysis never silently replaces timestamps or fills gaps itself.

The validity mask excludes undefined pixels from correlation and initial
reference estimation, including with ``--nothresh``. Nonfinite time courses are
also excluded. MRI's default spatial smoothing is disabled for this input;
an explicit positive ``--spatialfilt`` is interpreted in pixels.

Analysis maps are Zarr groups with a ``data`` array in the original ``(y, x)``
pixel geometry. Time-series outputs use ``(y, x, time)`` with a ``timestamps``
array that includes the source start time and any ``--timerange`` trimming.
Correlation-function outputs identify their final axis as lag. Each store has
schema ``rapidtide.analysis_map`` and records pixel units and input provenance.
The analysis still loads input signals and working arrays into RAM; supporting
Zarr input does not make the full Rapidtide workflow operate out of core.

Monochrome blood video (debugging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--bloodvideo blood.mp4`` additionally writes the projected blood signal as a
monochrome 8-bit H.264 video, after the Zarr store is complete. This is for
visual inspection and data validation only; downstream analysis should always
use the Zarr store. The gray mapping is fixed for the whole recording: gray 128
is zero signal, and the 99.5th percentile of absolute finite signal maps to 255
(positive) or 1 (negative); larger values are clipped. Invalid pixels are black
(0), and an odd height or width is padded with one black row or column. The
display limit is recorded in the video's ``comment`` metadata. A video can also
be made from an existing store::

    from rapidtide.video import write_blood_video

    write_blood_video('blood.zarr', 'blood.mp4')

For example, read the output with::

    import zarr

    data = zarr.open_group('blood.zarr', mode='r')
    pixel_timecourse = data['blood'][100, 120, :]
    times = data['timestamps'][:]

Usage
^^^^^

.. argparse::
   :ref: rapidtide.workflows.videotoblood._get_parser
   :prog: videotoblood
   :func: _get_parser
