#!/usr/bin/env python

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

plt.rcParams.update({"font.size": 6})

##########################
# user config
sonlen = 128
irlen = 64

lambd_est = 1e-3  # estimated noise lev

##########################


def gen_son(length):
    "Generate a synthetic un-reverberated 'sound event' template"
    # (whitenoise -> integrate -> envelope -> normalise)
    son = np.cumsum(np.random.randn(length))
    # apply envelope
    attacklen = int(length // 8)
    env = np.hstack((np.linspace(0.1, 1, attacklen), np.linspace(1, 0.1, length - attacklen)))
    son *= env
    son /= np.sqrt(np.sum(son * son))
    return son


def gen_ir(length):
    "Generate a synthetic impulse response"
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


def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR"
    kernel = np.hstack(
        (kernel, np.zeros(len(signal) - len(kernel)))
    )  # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal) * np.conj(H) / (H * np.conj(H) + lambd**2)))
    return deconvolved


if __name__ == "__main__":
    "simple test: get one soundtype and one impulse response, convolve them, deconvolve them, and check the result (plot it!)"
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
