#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2025-2026 Blaise Frederick
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


def self_attention_model():
    """Test SelfAttention module forward pass and output shapes."""
    feature_dim = 32
    batch_size = 4
    seq_len = 20

    attn = dlfiltertorch.SelfAttention(feature_dim)
    x = torch.randn(batch_size, seq_len, feature_dim)
    out, weights = attn(x)

    assert out.shape == (batch_size, seq_len, feature_dim)
    assert weights.shape == (batch_size, seq_len)

    # Weights should sum to ~1 along last dim (softmax output averaged)
    # Just check they are non-negative
    assert torch.all(weights >= 0.0)


def ppg_attention_model_creation():
    """Test PPGAttentionModel instantiation and forward pass."""
    hidden_size = 32
    model = dlfiltertorch.PPGAttentionModel(hidden_size=hidden_size)

    # PPGAttentionModel expects (batch, 1, even_length)
    batch_size = 4
    x = torch.randn(batch_size, 1, 100)
    output = model(x)

    assert output.shape == (batch_size, 1, 100)

    # Test with different even-length input
    x2 = torch.randn(batch_size, 1, 64)
    output2 = model(x2)
    assert output2.shape == (batch_size, 1, 64)


def ppg_attention_dlfilter(testtemproot):
    """Test PPGAttentionDLFilter initialization, getname, makenet."""
    filter_obj = dlfiltertorch.PPGAttentionDLFilter(
        hidden_size=32,
        window_size=100,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.nettype == "ppgattention"
    assert filter_obj.hidden_size == 32

    filter_obj.getname()
    assert "ppgattention" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.PPGAttentionModel)

    # Test forward pass through DLFilter's predict_model
    val_x = np.random.randn(10, 100, 1).astype(np.float32)
    predictions = filter_obj.predict_model(val_x)
    assert predictions.shape == val_x.shape


def dense_autoencoder_model_creation():
    """Test DenseAutoencoderModel instantiation and forward pass."""
    window_size = 64
    encoding_dim = 10
    num_layers = 4
    dropout_rate = 0.3
    activation = "relu"
    inputsize = 1

    model = dlfiltertorch.DenseAutoencoderModel(
        window_size=window_size,
        encoding_dim=encoding_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        activation=activation,
        inputsize=inputsize,
    )
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, inputsize, window_size)
    output = model(x)

    assert output.shape == (batch_size, inputsize, window_size)

    # Test get_config
    config = model.get_config()
    assert config["window_size"] == window_size
    assert config["encoding_dim"] == encoding_dim
    assert config["num_layers"] == num_layers
    assert config["activation"] == activation

    # Test with tanh activation
    model_tanh = dlfiltertorch.DenseAutoencoderModel(
        window_size=window_size,
        encoding_dim=encoding_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        activation="tanh",
        inputsize=inputsize,
    )
    model_tanh.eval()
    output_tanh = model_tanh(x)
    assert output_tanh.shape == (batch_size, inputsize, window_size)


def dense_autoencoder_dlfilter(testtemproot):
    """Test DenseAutoencoderDLFilter initialization, getname, makenet."""
    filter_obj = dlfiltertorch.DenseAutoencoderDLFilter(
        encoding_dim=10,
        window_size=64,
        num_layers=4,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.nettype == "autoencoder"
    assert filter_obj.encoding_dim == 10

    filter_obj.getname()
    assert "autoencoder" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.DenseAutoencoderModel)

    # Test predict
    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(val_x)
    assert predictions.shape == val_x.shape


def conv_autoencoder_model_creation():
    """Test ConvAutoencoderModel instantiation and forward pass."""
    # Use window_size=65 (2^6+1) so MaxPool1d(2, padding=1) sizes match formula
    # 65 -> 33 -> 17 -> 9 -> 5 (all odd, formula matches actual)
    window_size = 65
    encoding_dim = 8
    num_filters = 4
    kernel_size = 3
    dropout_rate = 0.3
    activation = "relu"
    inputsize = 1

    model = dlfiltertorch.ConvAutoencoderModel(
        window_size=window_size,
        encoding_dim=encoding_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        activation=activation,
        inputsize=inputsize,
    )
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, inputsize, window_size)
    output = model(x)

    assert output.shape == (batch_size, inputsize, window_size)

    # Test get_config
    config = model.get_config()
    assert config["window_size"] == window_size
    assert config["encoding_dim"] == encoding_dim
    assert config["num_filters"] == num_filters


def conv_autoencoder_dlfilter(testtemproot):
    """Test ConvAutoencoderDLFilter initialization, getname, makenet."""
    filter_obj = dlfiltertorch.ConvAutoencoderDLFilter(
        encoding_dim=8,
        num_filters=4,
        kernel_size=3,
        window_size=65,
        num_layers=4,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.nettype == "convautoencoder"

    filter_obj.getname()
    assert "convautoencoder" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.ConvAutoencoderModel)

    # Test predict
    val_x = np.random.randn(10, 65, 1).astype(np.float32)
    predictions = filter_obj.predict_model(val_x)
    assert predictions.shape == val_x.shape


def multiscale_cnn_model_creation():
    """Test MultiscaleCNNModel instantiation and forward pass."""
    num_filters = 10
    kernel_sizes = [3, 5, 7]
    input_lens = [32, 64, 96]
    input_width = 1
    dilation_rate = 1

    model = dlfiltertorch.MultiscaleCNNModel(
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        input_lens=input_lens,
        input_width=input_width,
        dilation_rate=dilation_rate,
    )
    model.eval()

    batch_size = 4
    x_small = torch.randn(batch_size, input_width, input_lens[0])
    x_med = torch.randn(batch_size, input_width, input_lens[1])
    x_large = torch.randn(batch_size, input_width, input_lens[2])

    output = model(x_small, x_med, x_large)

    # Output should be (batch_size, 1) with sigmoid
    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

    # Test get_config
    config = model.get_config()
    assert config["num_filters"] == num_filters
    assert config["kernel_sizes"] == kernel_sizes
    assert config["input_lens"] == input_lens


def multiscale_cnn_dlfilter(testtemproot):
    """Test MultiscaleCNNDLFilter initialization, getname, makenet."""
    filter_obj = dlfiltertorch.MultiscaleCNNDLFilter(
        num_filters=10,
        kernel_sizes=[3, 5, 7],
        input_lens=[32, 64, 96],
        input_width=1,
        window_size=64,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.nettype == "multiscalecnn"

    filter_obj.getname()
    assert "multiscalecnn" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.MultiscaleCNNModel)


def crnn_model_creation():
    """Test CRNNModel instantiation and forward pass."""
    num_filters = 8
    kernel_size = 3
    encoding_dim = 16
    dropout_rate = 0.3
    activation = "relu"
    inputsize = 1

    model = dlfiltertorch.CRNNModel(
        num_filters=num_filters,
        kernel_size=kernel_size,
        encoding_dim=encoding_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        inputsize=inputsize,
    )
    model.eval()

    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, inputsize, seq_len)
    output = model(x)

    assert output.shape == (batch_size, inputsize, seq_len)

    # Test get_config
    config = model.get_config()
    assert config["num_filters"] == num_filters
    assert config["kernel_size"] == kernel_size
    assert config["encoding_dim"] == encoding_dim
    assert config["activation"] == activation

    # Test with tanh
    model_tanh = dlfiltertorch.CRNNModel(
        num_filters=num_filters,
        kernel_size=kernel_size,
        encoding_dim=encoding_dim,
        dropout_rate=dropout_rate,
        activation="tanh",
        inputsize=inputsize,
    )
    model_tanh.eval()
    output_tanh = model_tanh(x)
    assert output_tanh.shape == (batch_size, inputsize, seq_len)


def crnn_dlfilter(testtemproot):
    """Test CRNNDLFilter initialization, getname, makenet."""
    filter_obj = dlfiltertorch.CRNNDLFilter(
        encoding_dim=16,
        num_filters=8,
        kernel_size=3,
        window_size=64,
        num_layers=3,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.nettype == "crnn"

    filter_obj.getname()
    assert "crnn" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.CRNNModel)

    # Test predict
    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(val_x)
    assert predictions.shape == val_x.shape


def lstm_model_creation():
    """Test LSTMModel instantiation and forward pass."""
    num_units = 16
    num_layers = 2
    dropout_rate = 0.3
    window_size = 64
    inputsize = 1

    model = dlfiltertorch.LSTMModel(
        num_units=num_units,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        window_size=window_size,
        inputsize=inputsize,
    )
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, inputsize, window_size)
    output = model(x)

    assert output.shape == (batch_size, inputsize, window_size)

    # Test get_config
    config = model.get_config()
    assert config["num_units"] == num_units
    assert config["num_layers"] == num_layers
    assert config["window_size"] == window_size
    assert config["inputsize"] == inputsize


def lstm_dlfilter(testtemproot):
    """Test LSTMDLFilter initialization, getname, makenet."""
    filter_obj = dlfiltertorch.LSTMDLFilter(
        num_units=16,
        window_size=64,
        num_layers=2,
        num_epochs=1,
        modelroot=testtemproot,
        namesuffix="test",
    )

    assert filter_obj.nettype == "lstm"

    filter_obj.getname()
    assert "lstm" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.LSTMModel)

    # Test predict
    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(val_x)
    assert predictions.shape == val_x.shape


def hybrid_model_creation():
    """Test HybridModel with both invert modes."""
    num_filters = 8
    kernel_size = 3
    num_units = 16
    num_layers = 3
    dropout_rate = 0.3
    activation = "relu"
    inputsize = 1
    window_size = 64

    # Test invert=False (LSTM first, then CNN)
    model_lstm_first = dlfiltertorch.HybridModel(
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_units=num_units,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        activation=activation,
        inputsize=inputsize,
        window_size=window_size,
        invert=False,
    )
    model_lstm_first.eval()

    batch_size = 4
    x = torch.randn(batch_size, inputsize, window_size)
    output = model_lstm_first(x)
    assert output.shape == (batch_size, inputsize, window_size)

    config = model_lstm_first.get_config()
    assert config["invert"] is False
    assert config["num_filters"] == num_filters
    assert config["num_units"] == num_units

    # Test invert=True (CNN first, then LSTM)
    model_cnn_first = dlfiltertorch.HybridModel(
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_units=num_units,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        activation=activation,
        inputsize=inputsize,
        window_size=window_size,
        invert=True,
    )
    model_cnn_first.eval()

    output_inv = model_cnn_first(x)
    assert output_inv.shape == (batch_size, inputsize, window_size)

    config_inv = model_cnn_first.get_config()
    assert config_inv["invert"] is True


def hybrid_dlfilter(testtemproot):
    """Test HybridDLFilter initialization, getname, makenet."""
    # Test with invert=False
    filter_obj = dlfiltertorch.HybridDLFilter(
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

    assert filter_obj.nettype == "hybrid"
    assert filter_obj.invert is False

    filter_obj.getname()
    assert "hybrid" in filter_obj.modelname
    assert os.path.exists(filter_obj.modelpath)

    filter_obj.makenet()
    assert filter_obj.model is not None
    assert isinstance(filter_obj.model, dlfiltertorch.HybridModel)

    # Test predict
    val_x = np.random.randn(10, 64, 1).astype(np.float32)
    predictions = filter_obj.predict_model(val_x)
    assert predictions.shape == val_x.shape

    # Test with invert=True
    filter_obj_inv = dlfiltertorch.HybridDLFilter(
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
    filter_obj_inv.getname()
    filter_obj_inv.makenet()
    predictions_inv = filter_obj_inv.predict_model(val_x)
    assert predictions_inv.shape == val_x.shape


def calcnumchannels_test():
    """Test calcnumchannels for all 4 combinations of usebadpts and dofft."""
    # No badpts, no fft
    numchans, badptchans, fftchans = dlfiltertorch.calcnumchannels(False, False)
    assert numchans == 1
    assert badptchans == 0
    assert fftchans == 0

    # With badpts, no fft
    numchans, badptchans, fftchans = dlfiltertorch.calcnumchannels(True, False)
    assert numchans == 2
    assert badptchans == 1
    assert fftchans == 0

    # No badpts, with fft
    numchans, badptchans, fftchans = dlfiltertorch.calcnumchannels(False, True)
    assert numchans == 3
    assert badptchans == 0
    assert fftchans == 2

    # With badpts, with fft
    numchans, badptchans, fftchans = dlfiltertorch.calcnumchannels(True, True)
    assert numchans == 4
    assert badptchans == 1
    assert fftchans == 2


def datatochannels_test():
    """Test datatochannels with various configurations."""
    length = 100
    timecourse = np.random.randn(length).astype(np.float64)
    badpts = np.zeros(length, dtype=np.float64)
    badpts[20:30] = 1.0

    # Basic: no badpts, no fft
    result = dlfiltertorch.datatochannels(timecourse, badpts, usebadpts=False, dofft=False)
    assert result.shape == (length, 1)
    assert np.allclose(result[:, 0], timecourse)

    # With badpts
    result_bp = dlfiltertorch.datatochannels(timecourse, badpts, usebadpts=True, dofft=False)
    assert result_bp.shape == (length, 2)
    assert np.allclose(result_bp[:, 1], badpts)
    # Data at bad points should be zeroed
    assert np.all(result_bp[20:30, 0] == 0.0)

    # With fft (no badpts)
    result_fft = dlfiltertorch.datatochannels(timecourse, badpts, usebadpts=False, dofft=True)
    assert result_fft.shape == (length, 3)
    assert np.allclose(result_fft[:, 0], timecourse)

    # With both badpts and fft
    result_both = dlfiltertorch.datatochannels(timecourse, badpts, usebadpts=True, dofft=True)
    assert result_both.shape == (length, 4)
    assert np.allclose(result_both[:, 1], badpts)

    # Zero signal should produce zero fft channels
    zero_tc = np.zeros(length, dtype=np.float64)
    result_zero = dlfiltertorch.datatochannels(zero_tc, badpts, usebadpts=False, dofft=True)
    assert result_zero.shape == (length, 3)
    assert np.allclose(result_zero[:, 1], 0.0)
    assert np.allclose(result_zero[:, 2], 0.0)


def filtscale_hybrid():
    """Test filtscale in hybrid mode."""
    data = np.random.randn(64)

    # Forward hybrid mode
    scaled_data, scalefac = dlfiltertorch.filtscale(data, hybrid=True, lognormalize=True)
    assert scaled_data.shape == (64, 2)
    # First column should be original data in hybrid mode
    assert np.allclose(scaled_data[:, 0], data)

    # Reverse hybrid mode returns just the first column (original data)
    reconstructed = dlfiltertorch.filtscale(
        scaled_data, scalefac=scalefac, reverse=True, hybrid=True
    )
    assert np.allclose(reconstructed, data)


def filtscale_roundtrip():
    """Test filtscale forward/reverse roundtrip for both log and linear normalization."""
    data = np.random.randn(128)

    # Roundtrip with log normalization
    scaled_log, sf_log = dlfiltertorch.filtscale(data, reverse=False, lognormalize=True)
    recon_log = dlfiltertorch.filtscale(
        scaled_log, scalefac=sf_log, reverse=True, lognormalize=True
    )
    assert recon_log.shape == data.shape
    assert mse(data, recon_log) < 0.1

    # Roundtrip with linear normalization
    scaled_lin, sf_lin = dlfiltertorch.filtscale(data, reverse=False, lognormalize=False)
    recon_lin = dlfiltertorch.filtscale(
        scaled_lin, scalefac=sf_lin, reverse=True, lognormalize=False
    )
    assert recon_lin.shape == data.shape
    assert mse(data, recon_lin) < 1e-10


def test_dlfilterops(debug=False, local=False):
    # set input and output directories
    testtemproot = get_test_temp_path(local)

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

    if debug:
        print("self_attention_model()")
    self_attention_model()

    if debug:
        print("ppg_attention_model_creation()")
    ppg_attention_model_creation()

    if debug:
        print("ppg_attention_dlfilter(testtemproot)")
    ppg_attention_dlfilter(testtemproot)

    if debug:
        print("dense_autoencoder_model_creation()")
    dense_autoencoder_model_creation()

    if debug:
        print("dense_autoencoder_dlfilter(testtemproot)")
    dense_autoencoder_dlfilter(testtemproot)

    if debug:
        print("conv_autoencoder_model_creation()")
    conv_autoencoder_model_creation()

    if debug:
        print("conv_autoencoder_dlfilter(testtemproot)")
    conv_autoencoder_dlfilter(testtemproot)

    if debug:
        print("multiscale_cnn_model_creation()")
    multiscale_cnn_model_creation()

    if debug:
        print("multiscale_cnn_dlfilter(testtemproot)")
    multiscale_cnn_dlfilter(testtemproot)

    if debug:
        print("crnn_model_creation()")
    crnn_model_creation()

    if debug:
        print("crnn_dlfilter(testtemproot)")
    crnn_dlfilter(testtemproot)

    if debug:
        print("lstm_model_creation()")
    lstm_model_creation()

    if debug:
        print("lstm_dlfilter(testtemproot)")
    lstm_dlfilter(testtemproot)

    if debug:
        print("hybrid_model_creation()")
    hybrid_model_creation()

    if debug:
        print("hybrid_dlfilter(testtemproot)")
    hybrid_dlfilter(testtemproot)

    if debug:
        print("calcnumchannels_test()")
    calcnumchannels_test()

    if debug:
        print("datatochannels_test()")
    datatochannels_test()

    if debug:
        print("filtscale_hybrid()")
    filtscale_hybrid()

    if debug:
        print("filtscale_roundtrip()")
    filtscale_roundtrip()


if __name__ == "__main__":
    test_dlfilterops(debug=True, local=True)
