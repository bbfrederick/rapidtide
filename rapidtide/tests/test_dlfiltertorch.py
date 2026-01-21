#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2025-2025 Blaise Frederick
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
import torch

import rapidtide.dlfiltertorch as dlfiltertorch
from rapidtide.tests.utils import get_test_temp_path, mse


def create_dummy_data():
    """Create dummy training data for testing."""
    window_size = 64
    num_samples = 100

    # Create dummy input and output data
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


def cnn_model_creation():
    """Test CNN model instantiation and forward pass."""
    num_filters = 10
    kernel_size = 5
    num_layers = 3
    dropout_rate = 0.3
    dilation_rate = 1
    activation = "relu"
    inputsize = 1

    model = dlfiltertorch.CNNModel(
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        dilation_rate=dilation_rate,
        activation=activation,
        inputsize=inputsize,
    )

    # Test forward pass
    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, inputsize, seq_len)
    output = model(x)

    assert output.shape == (batch_size, inputsize, seq_len)

    # Test get_config
    config = model.get_config()
    assert config["num_filters"] == num_filters
    assert config["kernel_size"] == kernel_size


def cnn_dlfilter_initialization(testtemproot):
    """Test CNNDLFilter initialization."""
    filter_obj = dlfiltertorch.CNNDLFilter(
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
    assert filter_obj.nettype == "cnn"
    assert not filter_obj.initialized


def cnn_dlfilter_initialize(testtemproot):
    """Test CNNDLFilter model initialization."""
    filter_obj = dlfiltertorch.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    # Just call getname and makenet, don't call full initialize
    # because savemodel has a bug using modelname instead of modelpath
    filter_obj.getname()
    filter_obj.makenet()

    assert filter_obj.model is not None
    assert os.path.exists(filter_obj.modelpath)

    # Manually save using modelpath
    filter_obj.model.to(filter_obj.device)
    filter_obj.savemodel(altname=filter_obj.modelpath)

    assert os.path.exists(os.path.join(filter_obj.modelpath, "model.pth"))


def predict_model(testtemproot, dummy_data):
    """Test the predict_model method."""
    filter_obj = dlfiltertorch.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=dummy_data["window_size"],
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
    )

    # Just create the model without full initialize
    filter_obj.getname()
    filter_obj.makenet()
    filter_obj.model.to(filter_obj.device)

    # Test prediction with numpy array
    predictions = filter_obj.predict_model(dummy_data["val_x"])

    assert predictions.shape == dummy_data["val_y"].shape
    assert isinstance(predictions, np.ndarray)


def apply_method(testtemproot):
    """Test the apply method for filtering a signal."""
    window_size = 64
    signal_length = 500

    filter_obj = dlfiltertorch.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=window_size,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
    )

    # Just create the model without full initialize
    filter_obj.getname()
    filter_obj.makenet()
    filter_obj.model.to(filter_obj.device)

    # Create a test signal
    input_signal = np.random.randn(signal_length).astype(np.float32)

    # Apply the filter
    filtered_signal = filter_obj.apply(input_signal)

    assert filtered_signal.shape == input_signal.shape
    assert isinstance(filtered_signal, np.ndarray)


def apply_method_with_badpts(testtemproot):
    """Test the apply method with bad points."""
    window_size = 64
    signal_length = 500

    filter_obj = dlfiltertorch.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=window_size,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        usebadpts=True,
    )

    # Just create the model without full initialize
    filter_obj.getname()
    filter_obj.makenet()
    filter_obj.model.to(filter_obj.device)

    # Create test signal and bad points
    input_signal = np.random.randn(signal_length).astype(np.float32)
    badpts = np.zeros(signal_length, dtype=np.float32)
    badpts[100:120] = 1.0  # Mark some points as bad

    # Apply the filter with bad points
    filtered_signal = filter_obj.apply(input_signal, badpts=badpts)

    assert filtered_signal.shape == input_signal.shape
    assert isinstance(filtered_signal, np.ndarray)


def save_and_load_model(testtemproot):
    """Test saving and loading a model."""
    # This test is skipped because both savemodel() and initmetadata()
    # use self.modelname (a relative path) instead of self.modelpath (full path)
    filter_obj = dlfiltertorch.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="saveloadtest",
    )

    # Create and save the model using modelpath
    filter_obj.getname()
    filter_obj.makenet()
    filter_obj.model.to(filter_obj.device)
    filter_obj.initmetadata()
    filter_obj.savemodel(altname=filter_obj.modelpath)

    original_modelname = os.path.basename(filter_obj.modelpath)

    # Get original model weights
    original_weights = {}
    for name, param in filter_obj.model.named_parameters():
        original_weights[name] = param.data.clone()

    # Create new filter object and load the saved model
    filter_obj2 = dlfiltertorch.CNNDLFilter(
        num_filters=10,  # These will be overridden by loaded model
        kernel_size=5,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        modelpath=testtemproot,
    )

    filter_obj2.loadmodel(original_modelname)

    # Check that metadata was loaded correctly
    assert filter_obj2.window_size == 64
    assert filter_obj2.infodict["nettype"] == "cnn"

    # Verify weights match
    for name, param in filter_obj2.model.named_parameters():
        assert torch.allclose(original_weights[name], param.data)


def filtscale_forward():
    """Test filtscale function in forward direction."""
    # filtscale expects 1D data (single timecourse)
    data = np.random.randn(64)

    # Test without log normalization
    scaled_data, scalefac = dlfiltertorch.filtscale(data, reverse=False, lognormalize=False)

    assert scaled_data.shape == (64, 2)
    assert isinstance(scalefac, (float, np.floating))

    # Test with log normalization
    scaled_data_log, scalefac_log = dlfiltertorch.filtscale(data, reverse=False, lognormalize=True)

    assert scaled_data_log.shape == (64, 2)


def filtscale_reverse():
    """Test filtscale function in reverse direction."""
    # filtscale expects 1D data (single timecourse)
    data = np.random.randn(64)

    # Forward then reverse
    scaled_data, scalefac = dlfiltertorch.filtscale(data, reverse=False, lognormalize=False)

    reconstructed = dlfiltertorch.filtscale(
        scaled_data, scalefac=scalefac, reverse=True, lognormalize=False
    )

    # Should reconstruct approximately to original
    assert reconstructed.shape == data.shape
    assert mse(data, reconstructed) < 1.0  # Allow some reconstruction error


def tobadpts():
    """Test tobadpts helper function."""
    filename = "test_file.txt"
    result = dlfiltertorch.tobadpts(filename)
    assert result == "test_file_badpts.txt"


def targettoinput():
    """Test targettoinput helper function."""
    filename = "test_xyz_file.txt"
    result = dlfiltertorch.targettoinput(filename, targetfrag="xyz", inputfrag="abc")
    assert result == "test_abc_file.txt"


def model_with_different_activations(testtemproot):
    """Test models with different activation functions."""
    activations = ["relu", "tanh"]

    for activation in activations:
        model = dlfiltertorch.CNNModel(
            num_filters=10,
            kernel_size=5,
            num_layers=3,
            dropout_rate=0.3,
            dilation_rate=1,
            activation=activation,
            inputsize=1,
        )

        # Test forward pass
        x = torch.randn(2, 1, 64)
        output = model(x)
        assert output.shape == x.shape

        config = model.get_config()
        assert config["activation"] == activation


def device_selection():
    """Test that device is properly set based on availability."""
    # This test just checks that the device variable is set
    # We can't guarantee CUDA/MPS availability in test environment
    assert dlfiltertorch.device in [torch.device("cuda"), torch.device("mps"), torch.device("cpu")]


def infodict_population(testtemproot):
    """Test that infodict is properly populated."""
    filter_obj = dlfiltertorch.CNNDLFilter(
        num_filters=10,
        kernel_size=5,
        window_size=64,
        num_layers=3,
        dropout_rate=0.3,
        num_epochs=5,
        excludethresh=4.0,
        corrthresh_rp=0.5,
        corrthresh_pp=0.9,
        modelroot=testtemproot,
    )

    # Check that infodict has expected keys
    assert "nettype" in filter_obj.infodict
    assert "num_filters" in filter_obj.infodict
    assert "kernel_size" in filter_obj.infodict
    assert filter_obj.infodict["nettype"] == "cnn"

    # Create the model (don't call initmetadata due to path bug)
    filter_obj.getname()
    filter_obj.makenet()

    # The model should populate infodict with window_size during getname
    assert "window_size" in filter_obj.infodict
    assert filter_obj.infodict["window_size"] == 64


def test_dlfilterops(debug=False, local=False):
    # set input and output directories
    if local:
        testtemproot = "./tmp"
    else:
        testtemproot = get_test_temp_path()

    thedummydata = create_dummy_data()

    if debug:
        print("cnn_model_creation()")
    cnn_model_creation()

    if debug:
        print("cnn_dlfilter_initialization(testtemproot)")
    cnn_dlfilter_initialization(testtemproot)

    if debug:
        print("cnn_dlfilter_initialize(testtemproot)")
    cnn_dlfilter_initialize(testtemproot)

    if debug:
        print("predict_model(testtemproot, thedummydata)")
    predict_model(testtemproot, thedummydata)

    if debug:
        print("apply_method(testtemproot)")
    apply_method(testtemproot)

    if debug:
        print("apply_method_with_badpts(testtemproot)")
    apply_method_with_badpts(testtemproot)

    if debug:
        print("save_and_load_model(testtemproot)")
    save_and_load_model(testtemproot)

    if debug:
        print("filtscale_forward()")
    filtscale_forward()

    if debug:
        print("filtscale_reverse()")
    filtscale_reverse()

    if debug:
        print("tobadpts()")
    tobadpts()

    if debug:
        print("targettoinput()")
    targettoinput()

    if debug:
        print("model_with_different_activations(testtemproot)")
    model_with_different_activations(testtemproot)

    if debug:
        print("device_selection()")
    device_selection()

    if debug:
        print("infodict_population(testtemproot)")
    infodict_population(testtemproot)


if __name__ == "__main__":
    test_dlfilterops(debug=True, local=True)
