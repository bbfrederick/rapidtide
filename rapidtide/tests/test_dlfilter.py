#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
import os

import numpy as np
import pytest

from rapidtide.tests.utils import get_test_temp_path, mse

try:
    import tensorflow as tf

    tensorflowexists = True
except ImportError:
    tensorflowexists = False


def create_dummy_data(window_size=64):
    """Create dummy training data for testing."""
    num_samples = 100
    train_x = np.random.randn(num_samples, window_size, 1).astype(np.float32)
    train_y = np.random.randn(num_samples, window_size, 1).astype(np.float32)
    val_x = np.random.randn(20, window_size, 1).astype(np.float32)
    val_y = np.random.randn(20, window_size, 1).astype(np.float32)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "window_size": window_size,
    }


# ============================================================
# CNN DLFilter tests
# ============================================================
def cnn_dlfilter_initialization(testtemproot):
    """Test CNNDLFilter initialization and attribute setting."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
    )

    assert filter_obj.window_size == 64
    assert filter_obj.num_filters == 10
    assert filter_obj.kernel_size == 5
    assert filter_obj.infodict["nettype"] == "cnn"
    assert not filter_obj.initialized


def cnn_dlfilter_getname_makenet(testtemproot):
    """Test CNNDLFilter getname and makenet methods."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    filter_obj.getname()
    assert "cnn" in filter_obj.modelname
    assert "tf2" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None


def cnn_predict_model(testtemproot, dummy_data):
    """Test CNNDLFilter predict_model method."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=dummy_data["window_size"],
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
    )

    filter_obj.getname()
    filter_obj.makenet()

    predictions = filter_obj.predict_model(
        tf.constant(dummy_data["val_x"])
    ).numpy()

    assert predictions.shape == dummy_data["val_y"].shape
    assert isinstance(predictions, np.ndarray)


def cnn_apply_method(testtemproot):
    """Test CNNDLFilter apply method for filtering a signal."""
    import rapidtide.dlfilter as dlfilter

    window_size = 64
    signal_length = 500

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=window_size,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
    )

    filter_obj.getname()
    filter_obj.makenet()

    input_signal = np.random.randn(signal_length).astype(np.float32)
    filtered_signal = filter_obj.apply(input_signal)

    assert filtered_signal.shape == input_signal.shape
    assert isinstance(filtered_signal, np.ndarray)


def cnn_apply_with_badpts(testtemproot):
    """Test CNNDLFilter apply method with bad points."""
    import rapidtide.dlfilter as dlfilter

    window_size = 64
    signal_length = 500

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=window_size,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        usebadpts=True,
    )

    filter_obj.getname()
    filter_obj.makenet()

    input_signal = np.random.randn(signal_length).astype(np.float32)
    badpts = np.zeros(signal_length, dtype=np.float32)
    badpts[100:120] = 1.0

    filtered_signal = filter_obj.apply(input_signal, badpts=badpts)

    assert filtered_signal.shape == input_signal.shape
    assert isinstance(filtered_signal, np.ndarray)


def cnn_save_and_load(testtemproot):
    """Test CNNDLFilter save and load model."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="saveloadtest",
    )

    filter_obj.getname()
    filter_obj.makenet()
    filter_obj.initmetadata()
    filter_obj.savemodel()

    original_modelname = os.path.basename(filter_obj.modelpath)

    # Get original weights
    original_weights = [w.numpy() for w in filter_obj.model.weights]

    # Create new filter and load saved model
    filter_obj2 = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        modelpath=testtemproot,
    )

    filter_obj2.loadmodel(original_modelname)

    assert filter_obj2.window_size == 64
    assert filter_obj2.infodict["window_size"] == 64
    assert "modelname" in filter_obj2.infodict

    # Verify weights match
    loaded_weights = [w.numpy() for w in filter_obj2.model.weights]
    for orig, loaded in zip(original_weights, loaded_weights):
        assert np.allclose(orig, loaded)


def cnn_different_activations(testtemproot):
    """Test CNNDLFilter with different activation functions."""
    import rapidtide.dlfilter as dlfilter

    for activation in ["relu", "tanh"]:
        filter_obj = dlfilter.CNNDLFilter(
            num_filters=10,
            kernel_size=5,
            window_size=64,
            num_layers=3,
            num_epochs=1,
            activation=activation,
            modelroot=testtemproot,
        )
        filter_obj.getname()
        filter_obj.makenet()
        assert filter_obj.model is not None


# ============================================================
# DenseAutoencoder DLFilter tests
# ============================================================
def dense_autoencoder_dlfilter(testtemproot):
    """Test DenseAutoencoderDLFilter initialization, getname, makenet."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.DenseAutoencoderDLFilter(
        encoding_dim=10,
        window_size=64,
        num_layers=4,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.infodict["nettype"] == "autoencoder"
    assert filter_obj.encoding_dim == 10

    filter_obj.getname()
    assert "denseautoencoder" in filter_obj.modelname
    assert "tf2" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None


def dense_autoencoder_predict(testtemproot):
    """Test DenseAutoencoderDLFilter predict method."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.DenseAutoencoderDLFilter(
        encoding_dim=10,
        window_size=64,
        num_layers=4,
        num_epochs=1,
        modelroot=testtemproot,
    )

    filter_obj.getname()
    filter_obj.makenet()

    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(tf.constant(val_x)).numpy()
    assert predictions.shape == val_x.shape


# ============================================================
# ConvAutoencoder DLFilter tests
# ============================================================
def conv_autoencoder_dlfilter(testtemproot):
    """Test ConvAutoencoderDLFilter initialization, getname, makenet."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.ConvAutoencoderDLFilter(
        encoding_dim=8,
        num_filters=4,
        kernel_size=3,
        window_size=64,
        num_layers=4,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.infodict["nettype"] == "autoencoder"

    filter_obj.getname()
    assert "convautoencoder" in filter_obj.modelname
    assert "tf2" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None


def conv_autoencoder_predict(testtemproot):
    """Test ConvAutoencoderDLFilter predict method."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.ConvAutoencoderDLFilter(
        encoding_dim=8,
        num_filters=4,
        kernel_size=3,
        window_size=64,
        num_layers=4,
        num_epochs=1,
        modelroot=testtemproot,
    )

    filter_obj.getname()
    filter_obj.makenet()

    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(tf.constant(val_x)).numpy()
    assert predictions.shape == val_x.shape


# ============================================================
# CRNN DLFilter tests
# ============================================================
def crnn_dlfilter(testtemproot):
    """Test CRNNDLFilter initialization, getname, makenet."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CRNNDLFilter(
        encoding_dim=16,
        num_filters=8,
        kernel_size=3,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    # Note: CRNNDLFilter sets nettype to "cnn" in its __init__
    assert filter_obj.infodict["nettype"] == "cnn"

    filter_obj.getname()
    assert "crnn" in filter_obj.modelname
    assert "tf2" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None


def crnn_predict(testtemproot):
    """Test CRNNDLFilter predict method."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CRNNDLFilter(
        encoding_dim=16,
        num_filters=8,
        kernel_size=3,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
    )

    filter_obj.getname()
    filter_obj.makenet()

    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(tf.constant(val_x)).numpy()
    assert predictions.shape == val_x.shape


# ============================================================
# LSTM DLFilter tests
# ============================================================
def lstm_dlfilter(testtemproot):
    """Test LSTMDLFilter initialization, getname, makenet."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.LSTMDLFilter(
        num_units=16,
        window_size=64,
        num_layers=2,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.infodict["nettype"] == "lstm"
    assert filter_obj.num_units == 16

    filter_obj.getname()
    assert "lstm" in filter_obj.modelname
    assert "tf2" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None


def lstm_predict(testtemproot):
    """Test LSTMDLFilter predict method."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.LSTMDLFilter(
        num_units=16,
        window_size=64,
        num_layers=2,
        num_epochs=1,
        modelroot=testtemproot,
    )

    filter_obj.getname()
    filter_obj.makenet()

    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(tf.constant(val_x)).numpy()
    assert predictions.shape == val_x.shape


# ============================================================
# Hybrid DLFilter tests
# ============================================================
def hybrid_dlfilter_lstm_first(testtemproot):
    """Test HybridDLFilter with LSTM first (invert=False)."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.HybridDLFilter(
        invert=False,
        num_filters=8,
        kernel_size=3,
        num_units=16,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.infodict["nettype"] == "hybrid"
    assert filter_obj.invert is False

    filter_obj.getname()
    assert "hybrid" in filter_obj.modelname
    assert "tf2" in filter_obj.modelname
    assert "_invert" not in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None

    # Test predict
    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(tf.constant(val_x)).numpy()
    assert predictions.shape == val_x.shape


def hybrid_dlfilter_cnn_first(testtemproot):
    """Test HybridDLFilter with CNN first (invert=True)."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.HybridDLFilter(
        invert=True,
        num_filters=8,
        kernel_size=3,
        num_units=16,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="testinv",
    )

    assert filter_obj.invert is True

    filter_obj.getname()
    assert "_invert" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None

    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(tf.constant(val_x)).numpy()
    assert predictions.shape == val_x.shape


# ============================================================
# DeepLearningFilter base class tests
# ============================================================
def base_filter_inputsize_usebadpts(testtemproot):
    """Test that inputsize is set correctly based on usebadpts flag."""
    import rapidtide.dlfilter as dlfilter

    # Without badpts
    f1 = dlfilter.CNNDLFilter(
        num_filters=10, kernel_size=5, window_size=64,
        num_epochs=1, modelroot=testtemproot, usebadpts=False,
    )
    assert f1.inputsize == 1

    # With badpts
    f2 = dlfilter.CNNDLFilter(
        num_filters=10, kernel_size=5, window_size=64,
        num_epochs=1, modelroot=testtemproot, usebadpts=True,
    )
    assert f2.inputsize == 2


def base_filter_infodict(testtemproot):
    """Test that infodict is properly populated."""
    import rapidtide.dlfilter as dlfilter

    filter_obj = dlfilter.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        dropout_rate=0.3,
        num_epochs=5,
        excludethresh=4.0,
        corrthresh_rp=0.5,
        modelroot=testtemproot,
    )

    assert filter_obj.infodict["window_size"] == 64
    assert filter_obj.infodict["usebadpts"] is False
    assert filter_obj.infodict["num_epochs"] == 5
    assert filter_obj.infodict["dropout_rate"] == 0.3
    assert filter_obj.infodict["nettype"] == "cnn"
    assert filter_obj.infodict["num_filters"] == 10
    assert filter_obj.infodict["kernel_size"] == 5


# ============================================================
# Standalone function tests (filtscale, tobadpts, targettoinput)
# ============================================================
def filtscale_forward():
    """Test filtscale function in forward direction."""
    import rapidtide.dlfilter as dlfilter

    data = np.random.randn(64)

    # Test without log normalization
    scaled_data, scalefac = dlfilter.filtscale(data, reverse=False, lognormalize=False)
    assert scaled_data.shape == (64, 2)
    assert isinstance(scalefac, (float, np.floating))

    # Test with log normalization
    scaled_data_log, scalefac_log = dlfilter.filtscale(data, reverse=False, lognormalize=True)
    assert scaled_data_log.shape == (64, 2)


def filtscale_reverse():
    """Test filtscale function in reverse direction."""
    import rapidtide.dlfilter as dlfilter

    data = np.random.randn(64)

    # Forward then reverse
    scaled_data, scalefac = dlfilter.filtscale(data, reverse=False, lognormalize=False)
    reconstructed = dlfilter.filtscale(
        scaled_data, scalefac=scalefac, reverse=True, lognormalize=False
    )

    assert reconstructed.shape == data.shape
    assert mse(data, reconstructed) < 1e-10


def filtscale_hybrid():
    """Test filtscale in hybrid mode."""
    import rapidtide.dlfilter as dlfilter

    data = np.random.randn(64)

    # Forward hybrid mode
    scaled_data, scalefac = dlfilter.filtscale(data, hybrid=True, lognormalize=True)
    assert scaled_data.shape == (64, 2)
    # First column should be original data
    assert np.allclose(scaled_data[:, 0], data)

    # Reverse hybrid mode returns just the first column
    reconstructed = dlfilter.filtscale(
        scaled_data, scalefac=scalefac, reverse=True, hybrid=True
    )
    assert np.allclose(reconstructed, data)


def filtscale_roundtrip():
    """Test filtscale forward/reverse roundtrip with log normalization."""
    import rapidtide.dlfilter as dlfilter

    data = np.random.randn(128)

    scaled_log, sf_log = dlfilter.filtscale(data, reverse=False, lognormalize=True)
    recon_log = dlfilter.filtscale(
        scaled_log, scalefac=sf_log, reverse=True, lognormalize=True
    )
    assert recon_log.shape == data.shape
    assert mse(data, recon_log) < 0.1


def tobadpts_test():
    """Test tobadpts helper function."""
    import rapidtide.dlfilter as dlfilter

    result = dlfilter.tobadpts("test_file.txt")
    assert result == "test_file_badpts.txt"


def targettoinput_test():
    """Test targettoinput helper function."""
    import rapidtide.dlfilter as dlfilter

    result = dlfilter.targettoinput("test_xyz_file.txt", targetfrag="xyz", inputfrag="abc")
    assert result == "test_abc_file.txt"


# ============================================================
# Main test function
# ============================================================
@pytest.mark.skipif(not tensorflowexists, reason="TensorFlow not installed")
def test_dlfilterops(debug=False, local=False):
    # set input and output directories
    testtemproot = get_test_temp_path(local)

    thedummydata = create_dummy_data()

    # CNN DLFilter tests
    if debug:
        print("cnn_dlfilter_initialization(testtemproot)")
    cnn_dlfilter_initialization(testtemproot)

    if debug:
        print("cnn_dlfilter_getname_makenet(testtemproot)")
    cnn_dlfilter_getname_makenet(testtemproot)

    if debug:
        print("cnn_predict_model(testtemproot, thedummydata)")
    cnn_predict_model(testtemproot, thedummydata)

    if debug:
        print("cnn_apply_method(testtemproot)")
    cnn_apply_method(testtemproot)

    if debug:
        print("cnn_apply_with_badpts(testtemproot)")
    cnn_apply_with_badpts(testtemproot)

    if debug:
        print("cnn_save_and_load(testtemproot)")
    cnn_save_and_load(testtemproot)

    if debug:
        print("cnn_different_activations(testtemproot)")
    cnn_different_activations(testtemproot)

    # DenseAutoencoder DLFilter tests
    if debug:
        print("dense_autoencoder_dlfilter(testtemproot)")
    dense_autoencoder_dlfilter(testtemproot)

    if debug:
        print("dense_autoencoder_predict(testtemproot)")
    dense_autoencoder_predict(testtemproot)

    # ConvAutoencoder DLFilter tests
    if debug:
        print("conv_autoencoder_dlfilter(testtemproot)")
    conv_autoencoder_dlfilter(testtemproot)

    if debug:
        print("conv_autoencoder_predict(testtemproot)")
    conv_autoencoder_predict(testtemproot)

    # CRNN DLFilter tests
    if debug:
        print("crnn_dlfilter(testtemproot)")
    crnn_dlfilter(testtemproot)

    if debug:
        print("crnn_predict(testtemproot)")
    crnn_predict(testtemproot)

    # LSTM DLFilter tests
    if debug:
        print("lstm_dlfilter(testtemproot)")
    lstm_dlfilter(testtemproot)

    if debug:
        print("lstm_predict(testtemproot)")
    lstm_predict(testtemproot)

    # Hybrid DLFilter tests
    if debug:
        print("hybrid_dlfilter_lstm_first(testtemproot)")
    hybrid_dlfilter_lstm_first(testtemproot)

    if debug:
        print("hybrid_dlfilter_cnn_first(testtemproot)")
    hybrid_dlfilter_cnn_first(testtemproot)

    # Base class tests
    if debug:
        print("base_filter_inputsize_usebadpts(testtemproot)")
    base_filter_inputsize_usebadpts(testtemproot)

    if debug:
        print("base_filter_infodict(testtemproot)")
    base_filter_infodict(testtemproot)

    # Standalone function tests
    if debug:
        print("filtscale_forward()")
    filtscale_forward()

    if debug:
        print("filtscale_reverse()")
    filtscale_reverse()

    if debug:
        print("filtscale_hybrid()")
    filtscale_hybrid()

    if debug:
        print("filtscale_roundtrip()")
    filtscale_roundtrip()

    if debug:
        print("tobadpts_test()")
    tobadpts_test()

    if debug:
        print("targettoinput_test()")
    targettoinput_test()


if __name__ == "__main__":
    test_dlfilterops(debug=True, local=True)
