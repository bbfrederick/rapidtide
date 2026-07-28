corrflow
--------

Description:
^^^^^^^^^^^^

Infers a blood flow map directly from the 4D similarity function movie, without
ever picking a correlation peak.

``delayflow`` starts from the delay map and solves the eikonal relation
grad(tau).v = 1.  That requires committing to a single peak location per voxel,
and peak finding is the fragile step on noisy data.  ``corrflow`` goes back one
step and works on the similarity function itself.

The two programs solve the same equation.  If the movie is I(x, l) = f(l -
tau(x)), then dI/dl = f' and grad(I) = -f' grad(tau), so the optical flow
brightness constancy constraint becomes::

    dI/dl + v . grad(I) = f' (1 - v . grad(tau)) = 0   ==>   v . grad(tau) = 1

which is the eikonal relation exactly.  The difference is entirely in the
estimation - optical flow accumulates the constraint over the whole waveform and
over every lag, so it never forms tau and never inherits a peak fitting failure.

What it does *not* fix: the constraint is identical, so the aperture problem is
identical.  Velocity tangent to the isochrone surfaces is invisible here too.

The estimator is three dimensional Lucas-Kanade, accumulating the structure
tensor over both a spatial neighborhood and the whole lag axis, solved through
its eigendecomposition.

Choosing a solver:
^^^^^^^^^^^^^^^^^^

``--solver ols`` (the default) is ordinary least squares.  It assumes the spatial
gradient is noise free, which it is not, so the recovered speed is biased toward
zero as noise rises - the classic errors-in-variables attenuation.  The bias is
large but graceful and monotone.

``--solver tls`` is total least squares, which spreads the error over all four
derivative components and removes that bias on clean data.  It is unstable at
low SNR, where it fails by producing confident nonsense rather than by degrading.

Measured on a synthetic radial wave with a true speed of 5.00 mm/s, against
``delayflow`` on the delay map derived from the same simulation:

===========  ==============  ==============  ==============  ==============
noise        ols direction   ols speed       tls direction   tls speed
===========  ==============  ==============  ==============  ==============
0.00         1.58 deg        4.96            1.60 deg        5.07
0.05         1.60 deg        4.86            1.62 deg        5.07
0.10         2.35 deg        4.56            13.5 deg        5.37
0.20         4.68 deg        3.47            70.8 deg        16.8
0.40         7.92 deg        1.68            69.7 deg        16.5
===========  ==============  ==============  ==============  ==============

Use ``tls`` only after checking it against ``ols`` on your own data.  Note that
the noise levels of the movie and of the delay map are coupled by an arbitrary
factor in this simulation, so the comparison with ``delayflow`` is informative
about the *shape* of the degradation, not about absolute equivalence.

Lag sampling and kernel balance:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The velocity is a ratio of two estimated derivatives, and each carries its own
smoothing.  For a wavefront at speed c, a spatial kernel of ``derivsigma`` voxels
is equivalent to a temporal kernel of ``derivsigma * voxdim / c`` seconds, and the
estimate is unbiased only when::

    lagsmooth * lagstep  ==  derivsigma * voxdim / c

Too wide a temporal kernel reads the speed LOW; too narrow reads it HIGH.  On a
synthetic wave at 20 mm/s with 2 mm voxels and a 0.4 s lag step, where balance
calls for ``derivsigma`` 4, the recovered/true ratio runs 0.76, 0.83, 1.08, 1.46
for ``derivsigma`` 1, 2, 4, 6.

``--lagoversamp N`` interpolates the lag axis N times finer at run time.  This is
exact rather than approximate: the correlation of bandlimited data is bandlimited
to the same band, so rapidtide already samples the lag axis far above its Nyquist
rate.  There is no need to pay for a higher ``--oversampfac`` in rapidtide - on
real data, interpolating a 0.5 s movie 2x reproduces a natively sampled 0.25 s
movie to a median direction difference of 1.02 degrees and a speed ratio of
1.002, against a 28.8 degree difference between the two methods.

It defaults to 1 (off), because narrowing the temporal kernel without also
reducing ``--derivsigma`` merely moves the imbalance to the other side.  Note also
that interpolation alone saturates - on the test above it gives 0.76, 0.85, 0.87,
0.88 for factors 1, 2, 4, 8, and never reaches 1.  Balancing adaptively from a
first pass speed estimate is the proper fix and is not yet implemented.

Do not lower ``--mineigenvalue``.  For a coherent travelling wavefront the
structure tensor is genuinely rank one, and admitting the near null eigenvectors
just divides noise by a small number: on the noiseless synthetic movie a
threshold of 1e-4 gives 13 degrees of direction error where 0.1 gives 1.6.

Inputs:
^^^^^^^

The rapidtide similarity function file, normally
``XXX_desc-corrout_info.nii.gz``.  The fourth dimension is the lag axis, whose
step and origin are read from the header ``pixdim[4]`` and ``toffset``.

``XXX_desc-corrfit_mask.nii.gz`` is picked up automatically as the mask if
present, and ``XXX_desc-maxtime_map.nii.gz`` as the ordering field, if present.

The CBV proxy is computed from the input itself, as the peak to peak amplitude of
the correlation function in each voxel.  No separate file is needed.

Outputs:
^^^^^^^^

Named ``OUTPUTROOT_desc-XXX_map.nii.gz``:

    **velocity**, **direction**, **speed**: the recovered flow field, in mm/s.

    **cbvproxy**: the peak to peak amplitude of the correlation function.

    **flowrank**: the number of structure tensor eigenvalues surviving the
    threshold, 0 to 3.  Read this as local geometry, not as a quality score -
    rank 1 is the expected, healthy regime for a clean propagating wavefront.

    **flowcoherence**: ratio of smallest to largest eigenvalue.

    **flowresidual**: rms brightness constancy residual.

    **orderingfield**: the scalar field used for routing and territories, either
    read from ``--delayfile`` or derived from the movie.  Used only for topology,
    never for velocity.

    **flowaccum**, **logflowaccum**, **territories**, **fluxdivergence**,
    **streamlinedensity**: as in ``delayflow``.

Plus ``OUTPUTROOT_desc-flow_streamlines.trk`` and
``OUTPUTROOT_desc-runoptions_info.json``.

See the ``delayflow`` documentation for how to view the streamlines - in
particular, FSLeyes needs ``-ds world`` or the tractogram will not align with the
NIfTI maps.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.corrflow._get_parser
   :prog: corrflow
   :func: _get_parser

   Debugging options : @skip
      skip debugging options
