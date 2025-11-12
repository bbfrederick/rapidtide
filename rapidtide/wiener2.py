#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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

# Simple example of Wiener deconvolution in Python.
# We use a fixed SNR across all frequencies in this example.
#
# Written 2015 by Dan Stowell. Public domain.

import matplotlib
import matplotlib.cm as cm

# matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.fft import fft, ifft, ifftshift
from numpy.typing import NDArray

plt.rcParams.update({"font.size": 6})

##########################
# user config
sonlen = 128
irlen = 64

lambd_est = 1e-3  # estimated noise lev

##########################


def gen_son(length: int) -> NDArray:
    """
    Generate a synthetic un-reverberated 'sound event' template.

    This function creates a synthetic sound template by generating white noise,
    integrating it, applying an envelope, and normalizing the result.

    Parameters
    ----------
    length : int
        The length of the output array in samples.

    Returns
    -------
    NDArray
        A normalized synthetic sound event template of shape (length,) containing
        floating point values.

    Notes
    -----
    The generated sound template follows these steps:
    1. Generate white noise using random.randn
    2. Integrate the noise using cumulative sum
    3. Apply a triangular envelope with 12.5% attack time
    4. Normalize the result to unit energy

    Examples
    --------
    >>> import numpy as np
    >>> son = gen_son(1000)
    >>> print(son.shape)
    (1000,)
    >>> print(f"Energy: {np.sum(son * son):.2f}")
    Energy: 1.00
    """
    # "Generate a synthetic un-reverberated 'sound event' template"
    # (whitenoise -> integrate -> envelope -> normalise)
    son = np.cumsum(np.random.randn(length))
    # apply envelope
    attacklen = int(length // 8)
    env = np.hstack((np.linspace(0.1, 1, attacklen), np.linspace(1, 0.1, length - attacklen)))
    son *= env
    son /= np.sqrt(np.sum(son * son))
    return son


def gen_ir(length: int) -> NDArray:
    """
    Generate a synthetic impulse response.

    This function creates a synthetic impulse response with a quiet tail, attack envelope,
    direct signal component, and early reflection spikes. The resulting impulse response
    is normalized to unit energy.

    Parameters
    ----------
    length : int
        The length of the impulse response array to generate.

    Returns
    -------
    NDArray
        A normalized numpy array of shape (length,) representing the synthetic impulse response.

    Notes
    -----
    The generated impulse response includes:
    - A quiet tail with random noise
    - An attack envelope that rises from 0.1 to 1 and then falls back to 0.1
    - A direct signal component at index 5 with amplitude 1
    - 10 early reflection spikes with random positions and amplitudes
    - Normalization to unit energy (L2 norm equals 1)

    Examples
    --------
    >>> import numpy as np
    >>> ir = gen_ir(100)
    >>> print(ir.shape)
    (100,)
    >>> print(np.isclose(np.sum(ir * ir), 1.0))
    True
    """
    # "Generate a synthetic impulse response"
    # First we generate a quietish tail
    son = np.random.randn(length)
    attacklen = int(length // 2)
    env = np.hstack((np.linspace(0.1, 1, attacklen), np.linspace(1, 0.1, length - attacklen)))
    son *= env
    son *= 0.05
    # Here we add the "direct" signal
    son[5] = 1
    # Now some early reflection spikes
    for _ in range(10):
        son[int(length * (np.random.rand() ** 2))] += np.random.randn() * 0.5
    # Normalise and return
    son /= np.sqrt(np.sum(son * son))
    return son


def wiener_deconvolution(signal: NDArray, kernel: NDArray, lambd: float) -> NDArray:
    """
    Perform Wiener deconvolution on a signal.

    Wiener deconvolution is a method for reversing the effects of convolution
    in the presence of noise. It uses a regularization parameter to balance
    between deconvolution accuracy and noise amplification.

    Parameters
    ----------
    signal : NDArray
        Input signal to be deconvolved, assumed to be 1D.
    kernel : NDArray
        Convolution kernel (point spread function), assumed to be 1D.
    lambd : float
        Regularization parameter (signal-to-noise ratio). Higher values
        result in more smoothing and less noise amplification.

    Returns
    -------
    NDArray
        Deconvolved signal with same length as input signal.

    Notes
    -----
    The function zero-pads the kernel to match the signal length before
    performing frequency domain operations. The Wiener filter is applied
    in the frequency domain using the formula:
    output = real(ifft(fft(signal) * conj(H) / (|H|² + λ²)))

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 3, 2, 1])
    >>> kernel = np.array([1, 0.5, 0.25])
    >>> result = wiener_deconvolution(signal, kernel, lambd=0.1)
    """
    # "lambd is the SNR"
    kernel = np.hstack(
        (kernel, np.zeros(len(signal) - len(kernel)))
    )  # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal) * np.conj(H) / (H * np.conj(H) + lambd**2)))
    return deconvolved


if __name__ == "__main__":
    # "simple test: get one soundtype and one impulse response, convolve them, deconvolve them, and check the result (plot it!)"
    son = gen_son(sonlen)
    ir = gen_ir(irlen)
    obs = np.convolve(son, ir, mode="full")
    # let's add some noise to the obs
    obs += np.random.randn(*obs.shape) * lambd_est
    son_est = wiener_deconvolution(obs, ir, lambd=lambd_est)[:sonlen]
    ir_est = wiener_deconvolution(obs, son, lambd=lambd_est)[:irlen]
    # calc error
    son_err = np.sqrt(np.mean((son - son_est) ** 2))
    ir_err = np.sqrt(np.mean((ir - ir_est) ** 2))
    print("single_example_test(): RMS errors son %g, IR %g" % (son_err, ir_err))
    # plot
    pdf = PdfPages("wiener_deconvolution_example.pdf")
    plt.figure(frameon=False)
    #
    plt.subplot(3, 2, 1)
    plt.plot(son)
    plt.title("son")
    plt.subplot(3, 2, 3)
    plt.plot(son_est)
    plt.title("son_est")
    plt.subplot(3, 2, 2)
    plt.plot(ir)
    plt.title("ir")
    plt.subplot(3, 2, 4)
    plt.plot(ir_est)
    plt.title("ir_est")
    plt.subplot(3, 1, 3)
    plt.plot(obs)
    plt.title("obs")
    #
    pdf.savefig()
    plt.close()
    pdf.close()
