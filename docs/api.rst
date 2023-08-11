.. _api_ref:

.. currentmodule:: rapidtide

API
===

.. _api_workflows_ref:

:mod:`rapidtide.workflows`: Rapidtide workflows
--------------------------------------------------

.. automodule:: rapidtide.workflows
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.workflows

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.workflows.rapidtide.rapidtide_main


.. _api_correlate_ref:

:mod:`rapidtide.correlate`: Correlation functions
--------------------------------------------------
.. automodule:: rapidtide.correlate
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.correlate

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.correlate.check_autocorrelation
    rapidtide.correlate.shorttermcorr_1D
    rapidtide.correlate.shorttermcorr_2D
    rapidtide.correlate.calc_MI
    rapidtide.correlate.mutual_info_2d
    rapidtide.correlate.cross_mutual_info
    rapidtide.correlate.mutual_info_to_r
    rapidtide.correlate.delayedcorr
    rapidtide.correlate.cepstraldelay
    rapidtide.correlate.arbcorr
    rapidtide.correlate.faststcorrelate
    rapidtide.correlate.fastcorrelate
    rapidtide.correlate._centered
    rapidtide.correlate._check_valid_mode_shapes
    rapidtide.correlate.convolve_weighted_fft
    rapidtide.correlate.gccproduct


.. _api_filter_ref:

:mod:`rapidtide.filter`: Filters
--------------------------------------------------
.. automodule:: rapidtide.filter
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.filter

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.filter.padvec
    rapidtide.filter.unpadvec
    rapidtide.filter.ssmooth
    rapidtide.filter.dolpfiltfilt
    rapidtide.filter.dohpfiltfilt
    rapidtide.filter.dobpfiltfilt
    rapidtide.filter.transferfuncfilt
    rapidtide.filter.getlpfftfunc
    rapidtide.filter.dolpfftfilt
    rapidtide.filter.dohpfftfilt
    rapidtide.filter.dobpfftfilt
    rapidtide.filter.getlptrapfftfunc
    rapidtide.filter.dolptrapfftfilt
    rapidtide.filter.dohptrapfftfilt
    rapidtide.filter.dobptrapfftfilt
    rapidtide.filter.wiener_deconvolution
    rapidtide.filter.pspec
    rapidtide.filter.spectrum
    rapidtide.filter.csdfilter
    rapidtide.filter.arb_pass
    rapidtide.filter.blackmanharris
    rapidtide.filter.hann
    rapidtide.filter.hamming
    rapidtide.filter.windowfunction

    :template: class.rst

    rapidtide.filter.NoncausalFilter


.. _api_fit_ref:

:mod:`rapidtide.fit`: Fitting functions
--------------------------------------------------
.. automodule:: rapidtide.fit
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.fit

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.fit.gaussresidualssk
    rapidtide.fit.gaussskresiduals
    rapidtide.fit.gaussresiduals
    rapidtide.fit.trapezoidresiduals
    rapidtide.fit.risetimeresiduals
    rapidtide.fit.gausssk_eval
    rapidtide.fit.gauss_eval
    rapidtide.fit.trapezoid_eval_loop
    rapidtide.fit.risetime_eval_loop
    rapidtide.fit.trapezoid_eval
    rapidtide.fit.risetime_eval
    rapidtide.fit.locpeak
    rapidtide.fit.trendgen
    rapidtide.fit.detrend
    rapidtide.fit.findfirstabove
    rapidtide.fit.findtrapezoidfunc
    rapidtide.fit.findrisetimefunc
    rapidtide.fit.findmaxlag_gauss
    rapidtide.fit.maxindex_noedge
    rapidtide.fit.findmaxlag_gauss_rev
    rapidtide.fit.findmaxlag_quad
    rapidtide.fit.gaussfitsk
    rapidtide.fit.gaussfit
    rapidtide.fit.mlregress
    rapidtide.fit.parabfit
    rapidtide.fit._datacheck_peakdetect
    rapidtide.fit.peakdetect


.. _api_io_ref:

:mod:`rapidtide.io`: Input/output functions
--------------------------------------------------
.. automodule:: rapidtide.io
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.io

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.io.readfromnifti
    rapidtide.io.readfromcifti
    rapidtide.io.getciftitr
    rapidtide.io.parseniftidims
    rapidtide.io.parseniftisizes
    rapidtide.io.savetonifti
    rapidtide.io.savetocifti
    rapidtide.io.checkifnifti
    rapidtide.io.niftisplitext
    rapidtide.io.niftisplit
    rapidtide.io.niftimerge
    rapidtide.io.niftiroi
    rapidtide.io.checkifcifti
    rapidtide.io.checkiftext
    rapidtide.io.getniftiroot
    rapidtide.io.fmriheaderinfo
    rapidtide.io.fmritimeinfo
    rapidtide.io.checkspacematch
    rapidtide.io.checkspaceresmatch
    rapidtide.io.checkspacedimmatch
    rapidtide.io.checktimematch
    rapidtide.io.checkifparfile
    rapidtide.io.readparfile
    rapidtide.io.readmotion
    rapidtide.io.calcmotregressors
    rapidtide.io.sliceinfo
    rapidtide.io.getslicetimesfromfile
    rapidtide.io.readbidssidecar
    rapidtide.io.writedicttojson
    rapidtide.io.readdictfromjson
    rapidtide.io.readlabelledtsv
    rapidtide.io.readoptionsfile
    rapidtide.io.writebidstsv
    rapidtide.io.readvectorsfromtextfile
    rapidtide.io.readbidstsv
    rapidtide.io.readcolfrombidstsv
    rapidtide.io.parsefilespec
    rapidtide.io.colspectolist
    rapidtide.io.processnamespec
    rapidtide.io.readcolfromtextfile
    rapidtide.io.readvecs
    rapidtide.io.readvec
    rapidtide.io.readtc
    rapidtide.io.readlabels
    rapidtide.io.writedict
    rapidtide.io.readdict
    rapidtide.io.writevec
    rapidtide.io.writenpvecs


.. _api_miscmath_ref:

:mod:`rapidtide.miscmath`: Miscellaneous math functions
-------------------------------------------------------
.. automodule:: rapidtide.miscmath
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.miscmath

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.miscmath.phase
    rapidtide.miscmath.polarfft
    rapidtide.miscmath.complex_cepstrum
    rapidtide.miscmath.real_cepstrum
    rapidtide.miscmath.thederiv
    rapidtide.miscmath.primes
    rapidtide.miscmath.largestfac
    rapidtide.miscmath.znormalize
    rapidtide.miscmath.stdnormalize
    rapidtide.miscmath.varnormalize
    rapidtide.miscmath.pcnormalize
    rapidtide.miscmath.ppnormalize
    rapidtide.miscmath.corrnormalize
    rapidtide.miscmath.rms
    rapidtide.miscmath.envdetect


.. _api_resample_ref:

:mod:`rapidtide.resample`: Resampling functions
--------------------------------------------------
.. automodule:: rapidtide.resample
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.resample

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.resample.congrid
    rapidtide.resample.doresample
    rapidtide.resample.arbresample
    rapidtide.resample.dotwostepresample
    rapidtide.resample.calcsliceoffset
    rapidtide.resample.timeshift

    :template: class.rst

    rapidtide.resample.FastResampler


.. _api_stats_ref:

:mod:`rapidtide.stats`: Statistical functions
--------------------------------------------------
.. automodule:: rapidtide.stats
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.stats

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.stats.fitjsbpdf
    rapidtide.stats.getjohnsonppf
    rapidtide.stats.sigFromDistributionData
    rapidtide.stats.rfromp
    rapidtide.stats.tfromr
    rapidtide.stats.zfromr
    rapidtide.stats.fisher
    rapidtide.stats.gethistprops
    rapidtide.stats.makehistogram
    rapidtide.stats.makeandsavehistogram
    rapidtide.stats.symmetrize
    rapidtide.stats.getfracval
    rapidtide.stats.makepmask
    rapidtide.stats.getfracvals
    rapidtide.stats.getfracvalsfromfit_old
    rapidtide.stats.getfracvalsfromfit
    rapidtide.stats.makemask


.. _api_util_ref:

:mod:`rapidtide.util`: Utility functions
--------------------------------------------------
.. automodule:: rapidtide.util
    :no-members:
    :no-inherited-members:

.. currentmodule:: rapidtide.util

.. autosummary::
    :toctree: generated/
    :template: function.rst

    rapidtide.util.logmem
    rapidtide.util.findexecutable
    rapidtide.util.isexecutable
    rapidtide.util.savecommandline
    rapidtide.util.valtoindex
    rapidtide.util.progressbar
    rapidtide.util.makelaglist
    rapidtide.util.version
    rapidtide.util.timefmt
    rapidtide.util.proctiminginfo
