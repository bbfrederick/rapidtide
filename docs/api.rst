API
===


:mod:`rapidtide.workflows`: Rapidtide workflows
--------------------------------------------------
.. automodule:: rapidtide.workflows
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.workflows
    :toctree: generated/
    :template: function.rst

    rapidtide.workflows.rapidtide_workflow
    rapidtide.workflows.showxcorrx_workflow

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.correlate`: Correlation functions
--------------------------------------------------
.. automodule:: rapidtide.correlate
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.correlate
    :toctree: generated/
    :template: function.rst

    rapidtide.correlate.autocorrcheck
    rapidtide.correlate.quickcorr
    rapidtide.correlate.shorttermcorr_1D
    rapidtide.correlate.shorttermcorr_2D
    rapidtide.correlate.delayedcorr
    rapidtide.correlate.cepstraldelay
    rapidtide.correlate.fastcorrelate
    rapidtide.correlate._centered
    rapidtide.correlate._check_valid_mode_shapes
    rapidtide.correlate.weightedfftconvolve
    rapidtide.correlate.gccproduct

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.filter`: Filters
--------------------------------------------------
.. automodule:: rapidtide.filter
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.filter
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
    rapidtide.filter.noncausalfilter

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.fit`: Fitting functions
--------------------------------------------------
.. automodule:: rapidtide.fit
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.fit
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

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.io`: Input/output functions
--------------------------------------------------
.. automodule:: rapidtide.io
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.io
    :toctree: generated/
    :template: function.rst

    rapidtide.io.readfromnifti
    rapidtide.io.parseniftidims
    rapidtide.io.parseniftisizes
    rapidtide.io.savetonifti
    rapidtide.io.checkifnifti
    rapidtide.io.checkiftext
    rapidtide.io.getniftiroot
    rapidtide.io.fmritimeinfo
    rapidtide.io.checkspacematch
    rapidtide.io.checktimematch
    rapidtide.io.checkifparfile
    rapidtide.io.readbidssidecar
    rapidtide.io.readbidstsv
    rapidtide.io.readcolfrombidstsv
    rapidtide.io.readvecs
    rapidtide.io.readvec
    rapidtide.io.readlabels
    rapidtide.io.writedict
    rapidtide.io.writevec
    rapidtide.io.writenpvecs

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.miscmath`: Miscellaneous math functions
-------------------------------------------------------
.. automodule:: rapidtide.miscmath
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.miscmath
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

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.resample`: Resampling functions
--------------------------------------------------
.. automodule:: rapidtide.resample
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.resample
    :toctree: generated/
    :template: function.rst

    rapidtide.resample.congrid
    rapidtide.resample.doresample
    rapidtide.resample.arbresample
    rapidtide.resample.dotwostepresample
    rapidtide.resample.calcsliceoffset
    rapidtide.resample.timeshift

    :template: class.rst
    rapidtide.resample.fastresampler

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.stats`: Statistical functions
--------------------------------------------------
.. automodule:: rapidtide.stats
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.stats
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

.. currentmodule:: rapidtide

.. _calibration_ref:


:mod:`rapidtide.util`: Utility functions
--------------------------------------------------
.. automodule:: rapidtide.util
    :no-members:
    :no-inherited-members:

.. autosummary:: rapidtide.util
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


.. currentmodule:: rapidtide

.. _calibration_ref:
