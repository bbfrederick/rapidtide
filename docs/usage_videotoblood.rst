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

For optional MediaPipe skin masking, install the skinmask extra (which also
includes the video dependencies)::

    pip install -e '.[skinmask]'

Run the standalone command::

    videotoblood recording.mp4 blood.zarr --method grgb
    videotoblood recording.mp4 blood_pos.zarr --method pos --chunks 64 64 16
    videotoblood recording.mp4 blood_30hz.zarr --framerate 30
    videotoblood recording.mp4 blood_small.zarr --lpf 1.5 --downsamp 4
    videotoblood recording.mp4 blood_skin.zarr --skinmask --lpf 2 --downsamp 3
    rapidtide blood.zarr analysis

Existing output Zarr stores and optional blood videos are overwritten. Each
previous output is retained until its replacement is complete. The first video stream is decoded
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

There is no temporal bandpass, moving mean, detrending, automatic spatial smoothing,
or motion correction. Skin segmentation is optional and requires ``--skinmask``.
Temporal interpolation can affect
high-frequency content, especially across long gaps; band selection remains
the responsibility of downstream analysis. Inputs should already have the
spatial alignment needed for pixelwise analysis. Decoding uses 16-bit RGB to
retain up to 16-bit channel precision;
it cannot recover precision or remove artifacts lost during source encoding.

Spatial preprocessing
^^^^^^^^^^^^^^^^^^^^^

``--skinmask`` runs Google's MediaPipe multiclass selfie segmenter on each
original RGB frame. It retains the union of the winning ``face-skin`` and
``body-skin`` categories and sets all other pixels to RGB zero. This is the
unrefined segmentation: there are no eye, glasses, mouth, beard, skin-color,
or seed-region exclusions. The processing order is **skin-mask inference on
original RGB, Gaussian filtering, skin-mask application, centered cropping,
block averaging, temporal interpolation, whole-recording statistics, then blood
extraction**. Filtering uses the complete original RGB frame. The previously
calculated mask then zeros non-skin pixels at source resolution, before cropping
and block averaging. With no Gaussian filter, the mask is applied directly to
the original RGB frame.

The 16-bit decoded RGB values are retained for blood extraction. MediaPipe uses
an 8-bit copy solely for inference. A fresh video-mode segmenter starts at the
beginning of each of the two RGB processing passes. No model is loaded or
downloaded when this flag is absent. All inference runs locally.

On first use, the approximately 16 MB versioned model is downloaded from Google
and verified with SHA-256. It is cached in
``$XDG_CACHE_HOME/rapidtide/models/`` or ``~/.cache/rapidtide/models/``. For offline
use, supply a compatible local model with ``--skinmaskmodel FILE`` together
with ``--skinmask``. Missing dependencies, invalid models, or download errors
fail conversion instead of silently disabling masking.

Masking zeros RGB samples before the existing projection calculations; it does
not change their validity rules. Fully masked pixels have undefined blood
projections: their Zarr blood values are NaN, their ``valid_mask`` is false, and
they appear black in ``--bloodvideo`` output. A pixel that changes skin/non-skin
classification can also fail the existing GRGB denominator checks. POS/CHROM
continue to use their original mean and variance checks. No temporal signal
quality or cardiac-power threshold is used to select skin.

``--lpf SIGMA`` optionally applies a 2D Gaussian lowpass filter independently to
each R, G, and B plane. Sigma is measured in original-frame pixels, boundaries
are reflected, and channels are never mixed. Zero (the default) disables it;
negative or nonfinite values are rejected.

``--downsamp N`` averages each nonoverlapping N×N RGB block into one output
pixel. N must be a positive integer (default 1). Cropping is centered in each
dimension: for remainder ``r = dimension % N``, remove ``r // 2`` pixels from
the low side and ``r - r // 2`` from the high side. Odd remainders therefore
remove one extra pixel from the high side. The retained height and width are
``H - H % N`` and ``W - W % N``; output dimensions are ``H // N`` by ``W // N``.
N cannot exceed either input dimension. For example, a 723×1282 frame with
``--downsamp 4`` loses one top row, two bottom rows, and one column from each
side, producing a 180×320 frame from the centered 720×1280 region.

When both spatial-filter/downsampling options are given, Gaussian filtering
runs on the full RGB frame after optional skin-mask inference. Mask application
then precedes cropping and block averaging. Both use float64 arithmetic to
retain fractional RGB intensities. This preprocessing precedes temporal
interpolation, whole-recording statistics, mean normalization, and projection,
and is identical in both processing passes. Blood-video output uses the same
reduced geometry (with the existing even-dimension padding for video encoding).

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
``spatial_preprocessing`` records the Gaussian sigma and boundary mode,
downsampling factor and method, original and cropped dimensions, operation
order, pixels removed from each side, and output pixel sizes and centers in
source-pixel coordinates (including the crop offset).
``skin_masking`` is null when disabled; when enabled, it records the model path,
checksum, MediaPipe version, included class names, absence of landmark refinement,
and masking order. ``calculated_before``, ``applied_after``, and ``applied_before``
distinguish inference from application. The spatial preprocessing order records
``skin_mask_inference``, ``gaussian_lowpass``, ``skin_mask``, ``crop``, and
``block_mean`` when skin masking is enabled.

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

Rapidtide writes video analysis maps and masks as single-page **float32 TIFF**
files (``_desc-*_map.tif`` and ``_desc-*_mask.tif``). Sequences, including cleaned
blood signals and correlation functions, are multi-page float32 TIFF stacks
(``_desc-*_bold.tif``, ``_desc-corrout_info.tif``). Each page is one grayscale
image in the original ``(y, x)`` pixel geometry; the file array order is
``(sample, y, x)``. No display scaling or integer quantization is applied.
DEFLATE compression is lossless. BigTIFF is selected automatically when the
uncompressed size plus metadata allowance approaches the classic TIFF 4 GiB limit.

JSON metadata is embedded in the first page and included in the ordinary JSON
sidecar. It records logical ``(y, x[, time/lag/frequency/sample])`` axes, exact
sample coordinates and units, sampling interval, start coordinate, geometry,
and input provenance. Time coordinates include the source start time and any
``--timerange`` trimming. TIFF storage axes are ``YX`` for maps, ``TYX`` for time
series, and ``QYX`` for other sequences, so lag or frequency is not mislabeled as
time. Masks and label maps are also float32. Undefined floating-point values
are retained. ``rapidtide.tiffio.open_map`` presents native ``(y, x[, sample])``
arrays with page-wise sequence reads. ``showrapidtide`` reads the TIFF outputs;
it can still display legacy Zarr results, preferring TIFF if both exist.
The ``videotoblood`` input store remains Zarr.

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
