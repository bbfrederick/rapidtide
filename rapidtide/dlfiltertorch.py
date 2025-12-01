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
import glob
import logging
import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numpy.typing import NDArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

from scipy import fftpack
from statsmodels.robust.scale import mad

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import rapidtide.io as tide_io

LGR = logging.getLogger("GENERAL")
LGR.debug("setting backend to Agg")
mpl.use("Agg")

# Disable GPU if desired
if torch.cuda.is_available():
    device = torch.device("cuda")
    LGR.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    LGR.info(f"Using MPS device")
else:
    device = torch.device("cpu")
    LGR.info("Using CPU")

LGR.debug(f"pytorch version: >>>{torch.__version__}<<<")


class DeepLearningFilter:
    """Base class for deep learning filter"""

    thesuffix = "sliceres"
    thedatadir = "/Users/frederic/Documents/MR_data/physioconn/timecourses"
    inputfrag = "abc"
    targetfrag = "xyz"
    namesuffix = None
    modelroot = "."
    excludethresh = 4.0
    modelname = None
    intermediatemodelpath = None
    usebadpts = False
    activation = "tanh"
    dofft = False
    readlim = None
    countlim = None
    lossfilename = None
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    model = None
    modelpath = None
    inputsize = None
    infodict = {}

    def __init__(
        self,
        window_size: int = 128,
        num_layers: int = 5,
        dropout_rate: float = 0.3,
        num_pretrain_epochs: int = 0,
        num_epochs: int = 1,
        activation: str = "relu",
        modelroot: str = ".",
        dofft: bool = False,
        excludethresh: float = 4.0,
        usebadpts: bool = False,
        thesuffix: str = "25.0Hz",
        modelpath: str = ".",
        thedatadir: str = "/Users/frederic/Documents/MR_data/physioconn/timecourses",
        inputfrag: str = "abc",
        targetfrag: str = "xyz",
        corrthresh: float = 0.5,
        excludebysubject: bool = True,
        startskip: int = 200,
        endskip: int = 200,
        step: int = 1,
        namesuffix: str | None = None,
        readlim: int | None = None,
        readskip: int | None = None,
        countlim: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the DeepLearningFilter with specified parameters.

        This constructor sets up the configuration for a deep learning model used
        for filtering physiological timecourses. It initializes various hyperparameters,
        paths, and flags that control the behavior of the model and data processing.

        Parameters
        ----------
        window_size : int, optional
            Size of the sliding window used for processing time series data. Default is 128.
        num_layers : int, optional
            Number of layers in the neural network model. Default is 5.
        dropout_rate : float, optional
            Dropout rate for regularization during training. Default is 0.3.
        num_pretrain_epochs : int, optional
            Number of pre-training epochs. Default is 0.
        num_epochs : int, optional
            Number of training epochs. Default is 1.
        activation : str, optional
            Activation function to use in the model. Default is "relu".
        modelroot : str, optional
            Root directory for model storage. Default is ".".
        dofft : bool, optional
            Whether to apply FFT transformation to input data. Default is False.
        excludethresh : float, optional
            Threshold for excluding data points based on correlation. Default is 4.0.
        usebadpts : bool, optional
            Whether to include bad points in the input. Default is False.
        thesuffix : str, optional
            Suffix to append to filenames. Default is "25.0Hz".
        modelpath : str, optional
            Path to save or load the model. Default is ".".
        thedatadir : str, optional
            Directory containing the physiological data files. Default is
            "/Users/frederic/Documents/MR_data/physioconn/timecourses".
        inputfrag : str, optional
            Fragment identifier for input data. Default is "abc".
        targetfrag : str, optional
            Fragment identifier for target data. Default is "xyz".
        corrthresh : float, optional
            Correlation threshold for filtering. Default is 0.5.
        excludebysubject : bool, optional
            Whether to exclude data by subject. Default is True.
        startskip : int, optional
            Number of samples to skip at the beginning of each timecourse. Default is 200.
        endskip : int, optional
            Number of samples to skip at the end of each timecourse. Default is 200.
        step : int, optional
            Step size for sliding window. Default is 1.
        namesuffix : str, optional
            Suffix to append to model name. Default is None.
        readlim : int, optional
            Limit on number of samples to read. Default is None.
        readskip : int, optional
            Number of samples to skip when reading data. Default is None.
        countlim : int, optional
            Limit on number of timecourses to process. Default is None.
        **kwargs
            Additional keyword arguments passed to the parent class.

        Notes
        -----
        The `inputsize` is dynamically set based on the `usebadpts` flag:
        - If `usebadpts` is True, input size is 2.
        - Otherwise, input size is 1.

        Examples
        --------
        >>> filter = DeepLearningFilter(
        ...     window_size=256,
        ...     num_layers=6,
        ...     dropout_rate=0.2,
        ...     modelroot="/models",
        ...     dofft=True
        ... )
        """
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.num_pretrain_epochs = num_pretrain_epochs
        self.num_epochs = num_epochs
        self.usebadpts = usebadpts
        self.num_layers = num_layers
        if self.usebadpts:
            self.inputsize = 2
        else:
            self.inputsize = 1
        self.activation = activation
        self.modelroot = modelroot
        self.dofft = dofft
        self.thesuffix = thesuffix
        self.thedatadir = thedatadir
        self.modelpath = modelpath
        LGR.info(f"modeldir from DeepLearningFilter: {self.modelpath}")
        self.corrthresh = corrthresh
        self.excludethresh = excludethresh
        self.readlim = readlim
        self.readskip = readskip
        self.countlim = countlim
        self.model = None
        self.initialized = False
        self.trained = False
        self.usetensorboard = False
        self.inputfrag = inputfrag
        self.targetfrag = targetfrag
        self.namesuffix = namesuffix
        self.startskip = startskip
        self.endskip = endskip
        self.step = step
        self.excludebysubject = excludebysubject
        self.device = device

        # populate infodict
        self.infodict["window_size"] = self.window_size
        self.infodict["usebadpts"] = self.usebadpts
        self.infodict["dofft"] = self.dofft
        self.infodict["corrthresh"] = self.corrthresh
        self.infodict["excludethresh"] = self.excludethresh
        self.infodict["num_pretrain_epochs"] = self.num_pretrain_epochs
        self.infodict["num_epochs"] = self.num_epochs
        self.infodict["modelname"] = self.modelname
        self.infodict["dropout_rate"] = self.dropout_rate
        self.infodict["startskip"] = self.startskip
        self.infodict["endskip"] = self.endskip
        self.infodict["step"] = self.step
        self.infodict["train_arch"] = sys.platform

    def loaddata(self) -> None:
        """
        Load and preprocess data for training and validation.

        This method initializes the data loading process by calling the `prep` function
        with a set of parameters derived from the instance attributes. It handles both
        FFT and non-FFT modes of data preprocessing. The loaded data is stored in
        instance variables for use in subsequent training steps.

        Parameters
        ----------
        self : object
            The instance of the class containing the following attributes:
            - initialized : bool
                Indicates whether the model has been initialized.
            - dofft : bool
                Whether to apply FFT transformation to the data.
            - window_size : int
                Size of the sliding window used for data segmentation.
            - thesuffix : str
                Suffix to append to filenames when reading data.
            - thedatadir : str
                Directory path where the data files are located.
            - inputfrag : str
                Fragment identifier for input data.
            - targetfrag : str
                Fragment identifier for target data.
            - startskip : int
                Number of samples to skip at the beginning of each file.
            - endskip : int
                Number of samples to skip at the end of each file.
            - corrthresh : float
                Correlation threshold for filtering data.
            - step : int
                Step size for sliding window.
            - usebadpts : bool
                Whether to include bad points in the data.
            - excludethresh : float
                Threshold for excluding data points.
            - excludebysubject : bool
                Whether to exclude data by subject.
            - readlim : int
                Limit on the number of samples to read.
            - readskip : int
                Number of samples to skip while reading.
            - countlim : int
                Limit on the number of data points to process.

        Returns
        -------
        None
            This method does not return any value. It modifies the instance attributes
            in place.

        Raises
        ------
        Exception
            If the model is not initialized prior to calling this method.

        Notes
        -----
        The method assigns the following attributes to the instance after loading:
        - train_x : array-like
            Training input data.
        - train_y : array-like
            Training target data.
        - val_x : array-like
            Validation input data.
        - val_y : array-like
            Validation target data.
        - Ns : int
            Number of samples.
        - tclen : int
            Length of time series.
        - thebatchsize : int
            Batch size for training.

        Examples
        --------
        >>> model = MyModel()
        >>> model.initialized = True
        >>> model.loaddata()
        >>> print(model.train_x.shape)
        (1000, 10)
        """
        if not self.initialized:
            raise Exception("model must be initialized prior to loading data")

        if self.dofft:
            (
                self.train_x,
                self.train_y,
                self.val_x,
                self.val_y,
                self.Ns,
                self.tclen,
                self.thebatchsize,
                dummy,
                dummy,
            ) = prep(
                self.window_size,
                thedatadir=self.thedatadir,
                inputfrag=self.inputfrag,
                targetfrag=self.targetfrag,
                startskip=self.startskip,
                endskip=self.endskip,
                corrthresh=self.corrthresh,
                step=self.step,
                dofft=self.dofft,
                usebadpts=self.usebadpts,
                excludethresh=self.excludethresh,
                excludebysubject=self.excludebysubject,
                readlim=self.readlim,
                readskip=self.readskip,
                countlim=self.countlim,
            )
        else:
            (
                self.train_x,
                self.train_y,
                self.val_x,
                self.val_y,
                self.Ns,
                self.tclen,
                self.thebatchsize,
            ) = prep(
                self.window_size,
                thedatadir=self.thedatadir,
                inputfrag=self.inputfrag,
                targetfrag=self.targetfrag,
                startskip=self.startskip,
                endskip=self.endskip,
                corrthresh=self.corrthresh,
                step=self.step,
                dofft=self.dofft,
                usebadpts=self.usebadpts,
                excludethresh=self.excludethresh,
                excludebysubject=self.excludebysubject,
                readlim=self.readlim,
                readskip=self.readskip,
                countlim=self.countlim,
            )

    def predict_model(self, X: NDArray) -> NDArray:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : NDArray
            Input features for prediction. Shape should be (n_samples, n_features)
            where n_samples is the number of samples and n_features is the number
            of features expected by the model.

        Returns
        -------
        NDArray
            Model predictions. Shape will depend on the specific model type but
            typically follows (n_samples,) for regression or (n_samples, n_classes)
            for classification.

        Notes
        -----
        This method sets the model to inference mode by calling with training=False.
        The predictions are made without computing gradients, making it efficient
        for inference tasks. Input data is automatically converted to PyTorch tensors
        and moved to the appropriate device. Special handling is included for
        tensor dimension permutation to match model expectations.

        Examples
        --------
        >>> # Assuming model is already trained
        >>> X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> predictions = model.predict_model(X_test)
        >>> print(predictions)
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float().to(self.device)
            # PyTorch expects (batch, channels, length) but we have (batch, length, channels)
            X = X.permute(0, 2, 1)
            output = self.model(X)
            # Convert back to (batch, length, channels)
            output = output.permute(0, 2, 1)
            return output.cpu().numpy()

    def evaluate(self) -> tuple[list, list, float, float]:
        """
        Evaluate the model performance on validation data and compute loss metrics.

        This method performs model evaluation by computing prediction errors and
        saving training/validation loss curves. It calculates both prediction error
        (difference between predicted and actual values) and raw error (difference
        between input and actual values). The method also generates and saves a
        plot of the training and validation loss over epochs.

        Parameters
        ----------
        self : object
            The instance of the class containing the model and data attributes.

        Returns
        -------
        tuple[list, list, float, float]
            A tuple containing:
            - training_loss : list
                List of training loss values per epoch
            - validation_loss : list
                List of validation loss values per epoch
            - prediction_error : float
                Mean squared error between predicted and actual values
            - raw_error : float
                Mean squared error between input features and actual values

        Notes
        -----
        This method modifies the instance attributes:
        - self.lossfilename: Path to the saved loss plot
        - self.pred_error: Computed prediction error
        - self.raw_error: Computed raw error
        - self.loss: Training loss history
        - self.val_loss: Validation loss history

        The method saves:
        - Loss plot as PNG file
        - Loss metrics as text file

        Examples
        --------
        >>> model = MyModel()
        >>> train_loss, val_loss, pred_error, raw_error = model.evaluate()
        >>> print(f"Prediction Error: {pred_error}")
        Prediction Error: 0.1234
        """
        self.lossfilename = os.path.join(self.modelname, "loss.png")
        LGR.info(f"lossfilename: {self.lossfilename}")

        YPred = self.predict_model(self.val_x)

        error = self.val_y - YPred
        self.pred_error = np.mean(np.square(error))

        error2 = self.val_x - self.val_y
        self.raw_error = np.mean(np.square(error2))
        LGR.info(f"Prediction Error: {self.pred_error}\tRaw Error: {self.raw_error}")

        f = open(os.path.join(self.modelname, "loss.txt"), "w")
        f.write(
            self.modelname
            + ": Prediction Error: "
            + str(self.pred_error)
            + " Raw Error: "
            + str(self.raw_error)
            + "\n"
        )
        f.close()

        epochs = range(len(self.loss))

        self.updatemetadata()

        plt.figure()
        plt.plot(epochs, self.loss, "bo", label="Training loss")
        plt.plot(epochs, self.val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.savefig(self.lossfilename)
        plt.close()

        return self.loss, self.val_loss, self.pred_error, self.raw_error

    def initmetadata(self) -> None:
        """
        Initialize and store metadata information for the model.

        This function creates a dictionary containing various model configuration parameters
        and writes them to a JSON file for future reference and reproducibility.

        Parameters
        ----------
        self : object
            The instance of the class containing the metadata attributes. Expected to have
            the following attributes:
            - `nettype`: Type of neural network
            - `window_size`: Size of the window used for processing
            - `usebadpts`: Flag indicating whether bad points are handled
            - `dofft`: Flag indicating whether FFT is used
            - `excludethresh`: Threshold for exclusion
            - `num_epochs`: Number of training epochs
            - `num_layers`: Number of layers in the model
            - `dropout_rate`: Dropout rate for regularization
            - `modelname`: Name of the model

        Returns
        -------
        None
            This function does not return any value but writes metadata to a JSON file.

        Notes
        -----
        The metadata includes:
        - Window size for processing
        - Bad point handling flag
        - FFT usage flag
        - Exclusion threshold
        - Number of epochs and layers
        - Dropout rate
        - Operating system platform
        - Model name

        The metadata is saved to ``{modelname}/model_meta.json`` where ``modelname``
        is the model's name attribute.

        Examples
        --------
        >>> model = MyModel()
        >>> model.initmetadata()
        >>> # Metadata stored in modelname/model_meta.json
        """

        self.infodict = {}
        self.infodict["nettype"] = self.nettype
        self.infodict["window_size"] = self.window_size
        self.infodict["usebadpts"] = self.usebadpts
        self.infodict["dofft"] = self.dofft
        self.infodict["excludethresh"] = self.excludethresh
        self.infodict["num_epochs"] = self.num_epochs
        self.infodict["num_layers"] = self.num_layers
        self.infodict["dropout_rate"] = self.dropout_rate
        self.infodict["train_arch"] = sys.platform
        self.infodict["modelname"] = self.modelname
        tide_io.writedicttojson(self.infodict, os.path.join(self.modelname, "model_meta.json"))

    def updatemetadata(self) -> None:
        """
        Update metadata dictionary with model metrics and save to JSON file.

        This method updates the internal information dictionary with various model
        performance metrics and writes the complete metadata to a JSON file for
        model persistence and tracking.

        Parameters
        ----------
        self : object
            The instance of the class containing the metadata and model information.
            Expected to have the following attributes:
            - infodict : dict
                Dictionary containing model metadata.
            - loss : float
                Training loss value.
            - val_loss : float
                Validation loss value.
            - raw_error : float
                Raw error metric.
            - pred_error : float
                Prediction error metric.
            - modelname : str
                Name/path of the model for file output.

        Returns
        -------
        None
            This method does not return any value but modifies the `infodict` in-place
            and writes to a JSON file.

        Notes
        -----
        The method writes metadata to ``{modelname}/model_meta.json`` where
        ``modelname`` is the model name attribute of the instance.

        Examples
        --------
        >>> model = MyModel()
        >>> model.updatemetadata()
        >>> # Creates model_meta.json with loss, val_loss, raw_error, and pred_error
        """
        self.infodict["loss"] = self.loss
        self.infodict["val_loss"] = self.val_loss
        self.infodict["raw_error"] = self.raw_error
        self.infodict["prediction_error"] = self.pred_error
        tide_io.writedicttojson(self.infodict, os.path.join(self.modelname, "model_meta.json"))

    def savemodel(self, altname: str | None = None) -> None:
        """
        Save the model to disk with the specified name.

        This method saves the current model to a Keras file format (.keras) in a
        directory named according to the model name or an alternative name provided.

        Parameters
        ----------
        altname : str, optional
            Alternative name to use for saving the model. If None, uses the
            model's default name stored in `self.modelname`. Default is None.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The model is saved in the Keras format (.keras) and stored in a directory
        with the same name as the model. The method logs the saving operation
        using the logger instance `LGR`.

        Examples
        --------
        >>> # Save model with default name
        >>> savemodel()
        >>>
        >>> # Save model with alternative name
        >>> savemodel(altname="my_custom_model")
        """
        if altname is None:
            modelsavename = self.modelname
        else:
            modelsavename = altname
        LGR.info(f"saving {modelsavename}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": (
                    self.model.get_config() if hasattr(self.model, "get_config") else None
                ),
            },
            os.path.join(modelsavename, "model.pth"),
        )

    def loadmodel(self, modelname: str, verbose: bool = False) -> None:
        """
        Load a trained model from disk and initialize model parameters.

        Load a Keras model from the specified model directory, along with associated
        metadata and configuration information. The function attempts to load the model
        in Keras format first, falling back to HDF5 format if the Keras format is not found.

        Parameters
        ----------
        modelname : str
            Name of the model to load, corresponding to a subdirectory in ``self.modelpath``.
        verbose : bool, optional
            If True, print model summary and metadata information. Default is False.

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return anything.

        Notes
        -----
        The function attempts to load the model in the following order:
        1. Keras format (model.keras)
        2. HDF5 format (model.h5)

        If neither format is found, the function exits with an error message.

        The loaded model metadata is stored in ``self.infodict``, and model configuration
        is stored in ``self.config``. Additional attributes like ``window_size`` and
        ``usebadpts`` are extracted from the metadata and stored as instance attributes.

        Examples
        --------
        >>> loader = ModelLoader()
        >>> loader.loadmodel("my_model", verbose=True)
        loading my_model
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #
        =================================================================
        ...
        >>> print(loader.window_size)
        100
        """
        # read in the data
        LGR.info(f"loading {modelname}")

        # load additional information first to reconstruct model
        self.infodict = tide_io.readdictfromjson(
            os.path.join(self.modelpath, modelname, "model_meta.json")
        )
        if verbose:
            print(self.infodict)
        self.window_size = self.infodict["window_size"]
        self.usebadpts = self.infodict["usebadpts"]
        self.dofft = self.infodict["dofft"]

        # Load the model as a dict
        checkpoint = torch.load(
            os.path.join(self.modelpath, modelname, "model.pth"), map_location=self.device
        )

        # Reconstruct the model architecture (must be done by subclass)
        if self.infodict["nettype"] == "cnn":
            self.num_filters = checkpoint["model_config"]["num_filters"]
            self.kernel_size = checkpoint["model_config"]["kernel_size"]
            self.num_layers = checkpoint["model_config"]["num_layers"]
            self.dropout_rate = checkpoint["model_config"]["dropout_rate"]
            self.dilation_rate = checkpoint["model_config"]["dilation_rate"]
            self.activation = checkpoint["model_config"]["activation"]
            self.inputsize = checkpoint["model_config"]["inputsize"]

            self.model = CNNModel(
                self.num_filters,
                self.kernel_size,
                self.num_layers,
                self.dropout_rate,
                self.dilation_rate,
                self.activation,
                self.inputsize,
            )
        elif self.infodict["nettype"] == "autoencoder":
            self.encoding_dim = checkpoint["model_config"]["encoding_dim"]
            self.num_layers = checkpoint["model_config"]["num_layers"]
            self.dropout_rate = checkpoint["model_config"]["dropout_rate"]
            self.activation = checkpoint["model_config"]["activation"]
            self.inputsize = checkpoint["model_config"]["inputsize"]

            self.model = DenseAutoencoderModel(
                self.window_size,
                self.encoding_dim,
                self.num_layers,
                self.dropout_rate,
                self.activation,
                self.inputsize,
            )
        elif self.infodict["nettype"] == "multiscalecnn":
            self.num_filters = checkpoint["model_config"]["num_filters"]
            self.kernel_sizes = checkpoint["model_config"]["kernel_sizes"]
            self.input_lens = checkpoint["model_config"]["input_lens"]
            self.input_width = checkpoint["model_config"]["input_width"]
            self.dilation_rate = checkpoint["model_config"]["dilation_rate"]

            self.model = MultiscaleCNNModel(
                self.num_filters,
                self.kernel_sizes,
                self.input_lens,
                self.input_width,
                self.dilation_rate,
            )
        elif self.infodict["nettype"] == "convautoencoder":
            self.encoding_dim = checkpoint["model_config"]["encoding_dim"]
            self.num_filters = checkpoint["model_config"]["num_filters"]
            self.kernel_size = checkpoint["model_config"]["kernel_size"]
            self.dropout_rate = checkpoint["model_config"]["dropout_rate"]
            self.activation = checkpoint["model_config"]["activation"]
            self.inputsize = checkpoint["model_config"]["inputsize"]

            self.model = ConvAutoencoderModel(
                self.window_size,
                self.encoding_dim,
                self.num_filters,
                self.kernel_size,
                self.dropout_rate,
                self.activation,
                self.inputsize,
            )
        elif self.infodict["nettype"] == "crnn":
            self.num_filters = checkpoint["model_config"]["num_filters"]
            self.kernel_size = checkpoint["model_config"]["kernel_size"]
            self.encoding_dim = checkpoint["model_config"]["encoding_dim"]
            self.dropout_rate = checkpoint["model_config"]["dropout_rate"]
            self.activation = checkpoint["model_config"]["activation"]
            self.inputsize = checkpoint["model_config"]["inputsize"]

            self.model = CRNNModel(
                self.num_filters,
                self.kernel_size,
                self.encoding_dim,
                self.dropout_rate,
                self.activation,
                self.inputsize,
            )
        elif self.infodict["nettype"] == "lstm":
            self.num_units = checkpoint["model_config"]["num_units"]
            self.num_layers = checkpoint["model_config"]["num_layers"]
            self.dropout_rate = checkpoint["model_config"]["dropout_rate"]
            self.inputsize = checkpoint["model_config"]["inputsize"]

            self.model = LSTMModel(
                self.num_units,
                self.num_layers,
                self.dropout_rate,
                self.window_size,
                self.inputsize,
            )
        elif self.infodict["nettype"] == "hybrid":
            self.num_filters = checkpoint["model_config"]["num_filters"]
            self.kernel_size = checkpoint["model_config"]["kernel_size"]
            self.num_units = checkpoint["model_config"]["num_units"]
            self.num_layers = checkpoint["model_config"]["num_layers"]
            self.dropout_rate = checkpoint["model_config"]["dropout_rate"]
            self.activation = checkpoint["model_config"]["activation"]
            self.inputsize = checkpoint["model_config"]["inputsize"]
            self.invert = checkpoint["model_config"]["invert"]

            self.model = HybridModel(
                self.num_filters,
                self.kernel_size,
                self.num_units,
                self.num_layers,
                self.dropout_rate,
                self.activation,
                self.inputsize,
                self.window_size,
                self.invert,
            )
        else:
            print(f"nettype {self.infodict['nettype']} is not supported!")
            sys.exit()

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

    def initialize(self) -> None:
        """
        Initialize the model by setting up network architecture and metadata.

        This method performs a series of initialization steps including retrieving
        the model name, creating the network architecture, displaying model summary,
        saving the model configuration, initializing metadata, and setting appropriate
        flags to indicate initialization status.

        Parameters
        ----------
        self : object
            The instance of the model class being initialized.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method should be called before any training or prediction operations.
        The initialization process sets `self.initialized` to True and `self.trained`
        to False, indicating that the model is ready for training but has not been
        trained yet.

        Examples
        --------
        >>> model = MyModel()
        >>> model.initialize()
        >>> print(model.initialized)
        True
        >>> print(model.trained)
        False
        """
        self.getname()
        self.makenet()
        print(self.model)
        self.savemodel()
        self.initmetadata()
        self.initialized = True
        self.trained = False

    def train(self) -> None:
        """
        Train the model using the provided training and validation datasets.

        This method performs model training with optional pretraining and logging. It supports
        TensorBoard logging, model checkpointing, early stopping, and NaN termination. The trained
        model is saved at the end of training.

        Parameters
        ----------
        self : object
            The instance of the class containing the model and training configuration.
            Expected attributes include:
            - `model`: PyTorch model to be trained.
            - `train_x`, `train_y`, `val_x`, `val_y`: Training and validation data as numpy arrays.
            - `device`: Device to run the training on (e.g., 'cpu' or 'cuda').
            - `num_pretrain_epochs`: Number of pretraining epochs (default: 0).
            - `num_epochs`: Number of main training epochs.
            - `modelname`: Directory name to save model checkpoints and logs.
            - `usetensorboard`: Boolean flag to enable TensorBoard logging.
            - `savemodel()`: Method to save the final trained model.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - If `self.usetensorboard` is True, TensorBoard logging is enabled.
        - If `self.num_pretrain_epochs` is greater than 0, a pretraining phase is performed
          before the main training loop.
        - The model is saved after training using the `savemodel()` method.
        - Training uses `ModelCheckpoint`, `EarlyStopping`, and `TerminateOnNaN` callbacks
          to manage training process and prevent overfitting or NaN issues.
        - Intermediate model checkpoints are saved during training.
        - The best model (based on validation loss) is retained and restored upon early stopping.

        Examples
        --------
        >>> trainer = ModelTrainer(model, train_x, train_y, val_x, val_y)
        >>> trainer.train()
        """
        self.model.train()
        self.model.to(self.device)

        # Convert numpy arrays to PyTorch tensors and transpose for Conv1d
        print("converting tensors")
        train_x_tensor = torch.from_numpy(self.train_x).float().permute(0, 2, 1)
        train_y_tensor = torch.from_numpy(self.train_y).float().permute(0, 2, 1)
        val_x_tensor = torch.from_numpy(self.val_x).float().permute(0, 2, 1)
        val_y_tensor = torch.from_numpy(self.val_y).float().permute(0, 2, 1)

        print("setting data")
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        print("setting criterion")
        criterion = nn.MSELoss()

        print("setting optimizer")
        optimizer = optim.RMSprop(self.model.parameters())

        self.loss = []
        self.val_loss = []

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        total_epochs = self.num_pretrain_epochs + self.num_epochs

        for epoch in range(total_epochs):
            print(f"Epoch {epoch+1}/{total_epochs}")
            # Training phase
            self.model.train()
            train_loss_epoch = 0.0
            # for batch_x, batch_y in train_loader:
            for batch_x, batch_y in tqdm.tqdm(
                train_loader,
                desc="Batch",
                unit="batches",
                disable=False,
            ):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                if torch.isnan(loss):
                    LGR.error("NaN loss detected, terminating training")
                    break

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()

            train_loss_epoch /= len(train_loader)
            self.loss.append(train_loss_epoch)

            # Validation phase
            self.model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss_epoch += loss.item()

            val_loss_epoch /= len(val_loader)
            self.val_loss.append(val_loss_epoch)

            LGR.info(
                f"Epoch {epoch+1}/{total_epochs} - Loss: {train_loss_epoch:.4f} - Val Loss: {val_loss_epoch:.4f}"
            )

            # Save checkpoint
            self.intermediatemodelpath = os.path.join(
                self.modelname, f"model_e{epoch+1:02d}_v{val_loss_epoch:.4f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss_epoch,
                    "val_loss": val_loss_epoch,
                },
                self.intermediatemodelpath,
            )

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(self.modelname, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    LGR.info(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best weights
                    self.model.load_state_dict(
                        torch.load(
                            os.path.join(self.modelname, "best_model.pth"), weights_only=True
                        )
                    )
                    break
            self.evaluate()

        self.savemodel()
        self.trained = True

    def apply(self, inputdata: NDArray, badpts: NDArray | None = None) -> NDArray:
        """
        Apply a sliding-window prediction model to the input data, optionally incorporating bad points.

        This function performs a sliding-window prediction using a pre-trained model. It scales the input
        data using the median absolute deviation (MAD), applies the model to overlapping windows of data,
        and aggregates predictions with a weighted scheme. Optionally, bad points can be included in
        the prediction process to influence the model's behavior.

        Parameters
        ----------
        inputdata : NDArray
            Input data array of shape (N,) to be processed.
        badpts : NDArray | None, optional
            Array of same shape as `inputdata` indicating bad or invalid points. If None, no bad points
            are considered. Default is None.

        Returns
        -------
        NDArray
            Predicted data array of the same shape as `inputdata`, with predictions aggregated and
            weighted across overlapping windows.

        Notes
        -----
        - The function uses a sliding window of size `self.window_size` to process input data.
        - Predictions are aggregated by summing over overlapping windows.
        - A triangular weight scheme is applied to the aggregated predictions to reduce edge effects.
        - If `self.usebadpts` is True, `badpts` are included as an additional feature in the model input.

        Examples
        --------
        >>> model = MyModel(window_size=10, usebadpts=True)
        >>> input_data = np.random.randn(100)
        >>> bad_points = np.zeros_like(input_data)
        >>> result = model.apply(input_data, bad_points)
        """
        initscale = mad(inputdata)
        scaleddata = inputdata / initscale
        predicteddata = np.zeros_like(scaleddata)
        weightarray = np.zeros_like(scaleddata)

        # make an X array with the proper number of channels
        if self.usebadpts is None:
            badptchans = 0
        else:
            badptchans = 1
        if self.dofft:
            fftchans = 2
        else:
            fftchans = 0
        numchans = 1 + badptchans + fftchans
        N_pts = len(scaleddata)
        X = np.zeros(((N_pts - self.window_size - 1), self.window_size, numchans))

        # make sure we have a valid badpts vector
        if badpts is None:
            badpts = np.zeros_like(scaleddata)

        # now populate all channels
        for i in range(X.shape[0]):
            X[i, :, 0] = scaleddata[i : i + self.window_size]
            if self.usebadpts:
                X[i, :, 1] = badpts[i : i + self.window_size]
                # zero out data in badpts regions
                X[i, np.where(X[i, :, 1] != 0.0), 0] = 0.0
            if self.dofft:
                specvals = fftpack.fft(X[i, :, 0])
                scalefac = np.std(X[i, :, 0])
                if scalefac != 0.0:
                    X[i, :, 1 + badptchans] = np.absolute(specvals) / scalefac
                    X[i, :, 1 + badptchans + 1] = np.angle(specvals)
                else:
                    X[i, :, 1 + badptchans] = 0.0
                    X[i, :, 1 + badptchans + 1] = 0.0

        Y = self.predict_model(X)
        for i in range(X.shape[0]):
            predicteddata[i : i + self.window_size] += Y[i, :, 0]

        weightarray[:] = self.window_size
        weightarray[0 : self.window_size] = np.linspace(
            1.0, self.window_size, self.window_size, endpoint=False
        )
        weightarray[-(self.window_size + 1) : -1] = np.linspace(
            self.window_size, 1.0, self.window_size, endpoint=False
        )
        return initscale * predicteddata / weightarray


class CNNModel(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        num_layers: int,
        dropout_rate: float,
        dilation_rate: int,
        activation: str,
        inputsize: int,
    ) -> None:
        """
        Initialize the CNNModel with specified architecture parameters.

        Parameters
        ----------
        num_filters : int
            Number of convolutional filters in each layer.
        kernel_size : int
            Size of the convolutional kernel.
        num_layers : int
            Total number of layers in the network.
        dropout_rate : float
            Dropout rate for regularization.
        dilation_rate : int
            Dilation rate for dilated convolutions in intermediate layers.
        activation : str
            Activation function to use; options are 'relu' or 'tanh'.
        inputsize : int
            Size of the input features.

        Returns
        -------
        None
            This method initializes the model in-place and does not return any value.

        Notes
        -----
        The model consists of an input layer, intermediate layers with dilated convolutions,
        and an output layer. Batch normalization and dropout are applied after each convolutional
        layer except the output layer. The activation function is applied after each convolutional
        layer based on the `activation` parameter.

        Examples
        --------
        >>> model = CNNModel(
        ...     num_filters=64,
        ...     kernel_size=3,
        ...     num_layers=4,
        ...     dropout_rate=0.2,
        ...     dilation_rate=2,
        ...     activation="relu",
        ...     inputsize=10
        ... )
        """
        super(CNNModel, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.inputsize = inputsize

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv1d(inputsize, num_filters, kernel_size, padding="same"))
        self.layers.append(nn.BatchNorm1d(num_filters))
        self.layers.append(nn.Dropout(dropout_rate))
        if activation == "relu":
            self.layers.append(nn.ReLU())
        elif activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.ReLU())

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Conv1d(
                    num_filters,
                    num_filters,
                    kernel_size,
                    dilation=dilation_rate,
                    padding="same",
                )
            )
            self.layers.append(nn.BatchNorm1d(num_filters))
            self.layers.append(nn.Dropout(dropout_rate))
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "tanh":
                self.layers.append(nn.Tanh())
            else:
                self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Conv1d(num_filters, 1, kernel_size, padding="same"))

    def forward(self, x):
        """
        Forward pass through all layers.

        Applies each layer in the network sequentially to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the forward pass. Shape should be compatible with the
            first layer's expected input dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through all layers. Shape will depend on
            the output dimensions of the last layer in the network.

        Notes
        -----
        This method applies layers in the order they were added to the network.
        Each layer's forward method is called sequentially, with the output of
        one layer becoming the input to the next.

        Examples
        --------
        >>> import torch
        >>> model = MyNetwork()
        >>> input_tensor = torch.randn(32, 10)
        >>> output = model.forward(input_tensor)
        >>> print(output.shape)
        torch.Size([32, 5])
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        """
        Get the configuration parameters of the model.

        Returns
        -------
        dict
            A dictionary containing all configuration parameters with their current values:
            - num_filters: int, number of filters in the convolutional layers
            - kernel_size: int, size of the convolutional kernel
            - num_layers: int, number of layers in the network
            - dropout_rate: float, dropout rate for regularization
            - dilation_rate: int, dilation rate for dilated convolution
            - activation: str, activation function used in layers
            - inputsize: tuple, input dimensions of the model

        Notes
        -----
        This method returns a copy of the current configuration. Modifications to the
        returned dictionary will not affect the original model configuration.

        Examples
        --------
        >>> config = model.get_config()
        >>> print(config['num_filters'])
        32
        """
        return {
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "dilation_rate": self.dilation_rate,
            "activation": self.activation,
            "inputsize": self.inputsize,
        }


class CNNDLFilter(DeepLearningFilter):
    def __init__(
        self,
        num_filters: int = 10,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize CNN deep learning filter.

        Parameters
        ----------
        num_filters : int, optional
            Number of convolutional filters to use, by default 10
        kernel_size : int, optional
            Size of the convolutional kernel, by default 5
        dilation_rate : int, optional
            Dilation rate for the convolutional layers, by default 1
        *args
            Variable length argument list passed to parent class
        **kwargs
            Arbitrary keyword arguments passed to parent class

        Returns
        -------
        None
            This method initializes the instance and does not return any value

        Notes
        -----
        This constructor sets up the basic configuration for a CNN filter with
        specified number of filters, kernel size, and dilation rate. The network
        type is automatically set to "cnn" and information is stored in infodict
        for later reference.

        Examples
        --------
        >>> filter = CNNDLFilter(num_filters=32, kernel_size=3, dilation_rate=2)
        >>> print(filter.num_filters)
        32
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.nettype = "cnn"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        super(CNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure the model name and path based on current parameters.

        This method constructs a descriptive model name string using various instance
        attributes and creates the corresponding directory path. The generated name
        includes information about model architecture, hyperparameters, and configuration
        options. The method also ensures the model directory exists by creating it if
        necessary.

        Parameters
        ----------
        self : object
            The instance containing model configuration parameters.

        Returns
        -------
        None
            This method does not return a value but modifies instance attributes:
            - self.modelname: Generated model name string
            - self.modelpath: Full path to the model directory

        Notes
        -----
        The generated model name follows a specific format:
        "model_cnn_pytorch_wXXX_lYY_fnZZ_flZZ_eXXX_tY_ctZ_sZ_dZ_activation[options]"

        Where:
        - XXX: window_size (3 digits zero-padded)
        - YY: num_layers (2 digits zero-padded)
        - ZZ: num_filters (2 digits zero-padded)
        - ZZ: kernel_size (2 digits zero-padded)
        - XXX: num_epochs (3 digits zero-padded)
        - Y: excludethresh (single digit)
        - Z: corrthresh (single digit)
        - Z: step (single digit)
        - Z: dilation_rate (single digit)

        Options are appended if corresponding boolean flags are True:
        - _usebadpts: when usebadpts is True
        - _excludebysubject: when excludebysubject is True

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 128
        >>> model.num_layers = 3
        >>> model.num_filters = 16
        >>> model.kernel_size = 3
        >>> model.num_epochs = 100
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 1
        >>> model.dilation_rate = 2
        >>> model.activation = "relu"
        >>> model.usebadpts = True
        >>> model.excludebysubject = False
        >>> model.namesuffix = "test"
        >>> model.getname()
        >>> print(model.modelname)
        'model_cnn_pytorch_w128_l03_fn16_fl03_e100_t0_ct0_s1_d2_relu_usebadpts_test'
        """
        self.modelname = "_".join(
            [
                "model",
                "cnn",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "l" + str(self.num_layers).zfill(2),
                "fn" + str(self.num_filters).zfill(2),
                "fl" + str(self.kernel_size).zfill(2),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
                "d" + str(self.dilation_rate),
                self.activation,
            ]
        )
        if self.usebadpts:
            self.modelname += "_usebadpts"
        if self.dofft:
            self.modelname += "_dofft"
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        if self.namesuffix is not None:
            self.modelname += "_" + self.namesuffix
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and configure a CNN model for neural network training.

        This method initializes a CNNModel with the specified parameters and moves
        it to the designated device (CPU or GPU). The model configuration is
        determined by the instance attributes set prior to calling this method.

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - num_filters : int
                Number of filters in each convolutional layer
            - kernel_size : int or tuple
                Size of the convolutional kernel
            - num_layers : int
                Number of convolutional layers in the network
            - dropout_rate : float
                Dropout rate for regularization
            - dilation_rate : int or tuple
                Dilation rate for dilated convolutions
            - activation : str or callable
                Activation function to use
            - inputsize : tuple
                Input dimensions for the model
            - device : torch.device
                Device to move the model to (CPU or GPU)

        Returns
        -------
        None
            This method does not return any value. It modifies the instance
            by setting the `model` attribute to the created CNNModel.

        Notes
        -----
        The method assumes that all required attributes are properly initialized
        before calling. The model is automatically moved to the specified device
        using the `.to()` method.

        Examples
        --------
        >>> # Assuming all required attributes are set
        >>> makenet()
        >>> # Model is now available as self.model
        >>> print(self.model)
        CNNModel(...)
        """
        self.model = CNNModel(
            self.num_filters,
            self.kernel_size,
            self.num_layers,
            self.dropout_rate,
            self.dilation_rate,
            self.activation,
            self.inputsize,
        )
        self.model.to(self.device)


class DenseAutoencoderModel(nn.Module):
    def __init__(
        self,
        window_size: int,
        encoding_dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: str,
        inputsize: int,
    ) -> None:
        """
        Initialize a dense autoencoder model with configurable architecture.

        This constructor builds a symmetric dense autoencoder with customizable number of layers,
        encoding dimension, dropout rate, and activation function. The model architecture follows
        a symmetric encoder-decoder structure, where the bottleneck layer reduces the feature space
        to `encoding_dim`, and then expands back to the original input size.

        Parameters
        ----------
        window_size : int
            The size of the input window (number of time steps or features per sample).
        encoding_dim : int
            The dimension of the latent space (bottleneck layer size).
        num_layers : int
            Total number of layers in the autoencoder (including input and output).
        dropout_rate : float
            Dropout rate applied to all hidden layers to prevent overfitting.
        activation : str
            Activation function to use in hidden layers. Supported values: 'relu', 'tanh'.
            Defaults to 'relu' if an unsupported value is provided.
        inputsize : int
            The number of features per time step in the input data.

        Returns
        -------
        None
            This method initializes the model in-place and does not return any value.

        Notes
        -----
        - The model uses batch normalization after each linear layer except the output layer.
        - The architecture is symmetric: the number of neurons decreases from input to bottleneck
          and then increases back to the output size.
        - Logging is performed at various stages to track layer sizes during construction.

        Examples
        --------
        >>> model = DenseAutoencoderModel(
        ...     window_size=10,
        ...     encoding_dim=5,
        ...     num_layers=4,
        ...     dropout_rate=0.2,
        ...     activation="relu",
        ...     inputsize=3
        ... )
        """
        super(DenseAutoencoderModel, self).__init__()

        self.window_size = window_size
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inputsize = inputsize

        self.layers = nn.ModuleList()

        # Calculate initial size factor
        sizefac = 2
        for i in range(1, num_layers - 1):
            sizefac = int(sizefac * 2)
        LGR.info(f"input layer - sizefac: {sizefac}")

        # Input layer
        self.layers.append(nn.Linear(window_size * inputsize, sizefac * encoding_dim))
        self.layers.append(nn.BatchNorm1d(sizefac * encoding_dim))
        self.layers.append(nn.Dropout(dropout_rate))
        if activation == "relu":
            self.layers.append(nn.ReLU())
        elif activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.ReLU())

        # Encoding layers
        for i in range(1, num_layers - 1):
            sizefac = int(sizefac // 2)
            LGR.info(f"encoder layer {i + 1}, sizefac: {sizefac}")
            self.layers.append(nn.Linear(sizefac * 2 * encoding_dim, sizefac * encoding_dim))
            self.layers.append(nn.BatchNorm1d(sizefac * encoding_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "tanh":
                self.layers.append(nn.Tanh())
            else:
                self.layers.append(nn.ReLU())

        # Encoding layer (bottleneck)
        sizefac = int(sizefac // 2)
        LGR.info(f"encoding layer - sizefac: {sizefac}")
        self.layers.append(nn.Linear(sizefac * 2 * encoding_dim, encoding_dim))
        self.layers.append(nn.BatchNorm1d(encoding_dim))
        self.layers.append(nn.Dropout(dropout_rate))
        if activation == "relu":
            self.layers.append(nn.ReLU())
        elif activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.ReLU())

        # Decoding layers
        for i in range(1, num_layers):
            sizefac = int(sizefac * 2)
            LGR.info(f"decoding layer {i}, sizefac: {sizefac}")
            if i == 1:
                self.layers.append(nn.Linear(encoding_dim, sizefac * encoding_dim))
            else:
                self.layers.append(nn.Linear(sizefac // 2 * encoding_dim, sizefac * encoding_dim))
            self.layers.append(nn.BatchNorm1d(sizefac * encoding_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "tanh":
                self.layers.append(nn.Tanh())
            else:
                self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(sizefac * encoding_dim, window_size * inputsize))

    def forward(self, x):
        """
        Forward pass through the network layers.

        Applies a series of layers to the input tensor, flattening and reshaping
        as needed for processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, length) where batch is the
            number of samples, channels is the number of channels, and length is
            the sequence length.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch, channels, length) where the
            dimensions match the input shape after processing through the layers.

        Notes
        -----
        The function first flattens the input from (batch, channels, length) to
        (batch, channels*length) to enable processing through linear layers,
        then reshapes the output back to the original format.

        Examples
        --------
        >>> import torch
        >>> # Assuming self.layers contains appropriate layer definitions
        >>> x = torch.randn(32, 3, 100)  # batch_size=32, channels=3, length=100
        >>> output = model.forward(x)
        >>> output.shape
        torch.Size([32, 3, 100])
        """
        # Flatten input from (batch, channels, length) to (batch, channels*length)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        for layer in self.layers:
            x = layer(x)

        # Reshape back to (batch, channels, length)
        x = x.reshape(batch_size, self.inputsize, self.window_size)
        return x

    def get_config(self):
        """
        Get the configuration parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the model configuration parameters with the following keys:
            - "window_size" (int): The size of the sliding window used for input sequences
            - "encoding_dim" (int): The dimensionality of the encoding layer
            - "num_layers" (int): The number of layers in the model
            - "dropout_rate" (float): The dropout rate for regularization
            - "activation" (str): The activation function used in the model
            - "inputsize" (int): The size of the input features

        Notes
        -----
        This method returns a copy of the internal configuration parameters. Modifications
        to the returned dictionary will not affect the original model configuration.

        Examples
        --------
        >>> model = MyModel()
        >>> config = model.get_config()
        >>> print(config['window_size'])
        10
        """
        return {
            "window_size": self.window_size,
            "encoding_dim": self.encoding_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "inputsize": self.inputsize,
        }


class DenseAutoencoderDLFilter(DeepLearningFilter):
    def __init__(self, encoding_dim: int = 10, *args, **kwargs) -> None:
        """
        Initialize DenseAutoencoderDLFilter instance.

        Parameters
        ----------
        encoding_dim : int, default=10
            Dimension of the encoding layer in the autoencoder. This determines
            the size of the latent representation learned by the model.
        *args
            Variable length argument list passed to the parent class constructor.
        **kwargs
            Arbitrary keyword arguments passed to the parent class constructor.

        Returns
        -------
        None
            This method initializes the instance and does not return any value.

        Notes
        -----
        This constructor sets up the autoencoder architecture by:
        1. Storing the encoding dimension as an instance attribute
        2. Setting the network type to "autoencoder"
        3. Updating the info dictionary with network type and encoding dimension
        4. Calling the parent class constructor with any additional arguments

        Examples
        --------
        >>> filter = DenseAutoencoderDLFilter(encoding_dim=15)
        >>> print(filter.encoding_dim)
        15
        >>> print(filter.nettype)
        'autoencoder'
        """
        self.encoding_dim = encoding_dim
        self.nettype = "autoencoder"
        self.infodict["nettype"] = self.nettype
        self.infodict["encoding_dim"] = self.encoding_dim
        super(DenseAutoencoderDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return the model name and path based on current configuration parameters.

        This method constructs a descriptive model name string using various configuration
        parameters and creates the corresponding directory path for model storage. The
        generated name includes information about window size, encoding dimensions, epochs,
        thresholds, step size, activation function, and additional flags.

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - window_size : int
                Size of the sliding window
            - encoding_dim : int
                Dimension of the encoding layer
            - num_epochs : int
                Number of training epochs
            - excludethresh : float
                Threshold for excluding data points
            - corrthresh : float
                Correlation threshold for filtering
            - step : int
                Step size for sliding window
            - activation : str
                Activation function name
            - usebadpts : bool
                Flag indicating whether to use bad points
            - excludebysubject : bool
                Flag indicating whether to exclude by subject
            - namesuffix : str, optional
                Additional suffix to append to model name
            - modelroot : str
                Root directory for model storage

        Returns
        -------
        None
            This method modifies the instance attributes in-place:
            - self.modelname : str
                Generated model name string
            - self.modelpath : str
                Full path to the model directory

        Notes
        -----
        The generated model name follows a specific format:
        "model_denseautoencoder_pytorch_wXXX_enXXX_eXXX_tX.XX_ctX.XX_sX_activation[flags]"

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.encoding_dim = 50
        >>> model.num_epochs = 1000
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 10
        >>> model.activation = 'relu'
        >>> model.usebadpts = True
        >>> model.excludebysubject = False
        >>> model.namesuffix = 'test'
        >>> model.modelroot = '/path/to/models'
        >>> model.getname()
        >>> print(model.modelname)
        'model_denseautoencoder_pytorch_w100_en050_e1000_t0.5_ct0.8_s10_relu_usebadpts_test'
        """
        self.modelname = "_".join(
            [
                "model",
                "denseautoencoder",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "en" + str(self.encoding_dim).zfill(3),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
                self.activation,
            ]
        )
        if self.usebadpts:
            self.modelname += "_usebadpts"
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        if self.namesuffix is not None:
            self.modelname += "_" + self.namesuffix
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and configure a dense autoencoder model.

        This method initializes a DenseAutoencoderModel with the specified parameters
        and moves it to the designated device (CPU or GPU).

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Expected to have the
            following attributes:
            - window_size : int
            - encoding_dim : int
            - num_layers : int
            - dropout_rate : float
            - activation : str or callable
            - inputsize : int
            - device : torch.device

        Returns
        -------
        None
            This method does not return any value. It sets the model attribute
            of the instance to the created DenseAutoencoderModel.

        Notes
        -----
        The model is automatically moved to the device specified by self.device.
        This method should be called after all required parameters have been set
        on the instance.

        Examples
        --------
        >>> # Assuming a class with the makenet method
        >>> instance = MyClass()
        >>> instance.window_size = 100
        >>> instance.encoding_dim = 32
        >>> instance.num_layers = 3
        >>> instance.dropout_rate = 0.2
        >>> instance.activation = 'relu'
        >>> instance.inputsize = 100
        >>> instance.device = torch.device('cuda')
        >>> instance.makenet()
        >>> # Model is now available as instance.model
        """
        self.model = DenseAutoencoderModel(
            self.window_size,
            self.encoding_dim,
            self.num_layers,
            self.dropout_rate,
            self.activation,
            self.inputsize,
        )
        self.model.to(self.device)


class MultiscaleCNNModel(nn.Module):
    def __init__(
        self,
        num_filters,
        kernel_sizes,
        input_lens,
        input_width,
        dilation_rate,
    ):
        """
        Initialize the MultiscaleCNNModel.

        This constructor initializes a multiscale CNN model with three parallel branches
        processing input at different scales. Each branch uses dilated convolutions to
        capture features at different receptive fields.

        Parameters
        ----------
        num_filters : int
            Number of filters (channels) to use in the convolutional layers.
        kernel_sizes : list of int
            List of three kernel sizes for the three branches (small, medium, large scales).
        input_lens : list of int
            List of input lengths for each branch, corresponding to the input sequence lengths.
        input_width : int
            Width of the input features (number of input channels).
        dilation_rate : int
            Dilation rate to use in the dilated convolutional layers.

        Returns
        -------
        None
            This method initializes the model instance and does not return any value.

        Notes
        -----
        The model creates three parallel branches with different kernel sizes to capture
        multi-scale temporal features. Each branch uses dilated convolutions to increase
        the receptive field without increasing the number of parameters significantly.

        The final dense layer reduces the combined features to a single output value,
        followed by a sigmoid activation for binary classification.

        Examples
        --------
        >>> model = MultiscaleCNNModel(
        ...     num_filters=64,
        ...     kernel_sizes=[3, 5, 7],
        ...     input_lens=[100, 100, 100],
        ...     input_width=10,
        ...     dilation_rate=2
        ... )
        """
        super(MultiscaleCNNModel, self).__init__()

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.input_lens = input_lens
        self.input_width = input_width
        self.dilation_rate = dilation_rate

        # Create three separate branches for different scales
        self.branch_small = self._make_branch(kernel_sizes[0])
        self.branch_med = self._make_branch(kernel_sizes[1])
        self.branch_large = self._make_branch(kernel_sizes[2])

        # Final dense layer
        self.fc = nn.Linear(150, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_branch(self, kernel_size):
        """
        Create a convolutional branch for the neural network architecture.

        This method constructs a sequential neural network branch consisting of
        convolutional, pooling, flattening, linear, activation, and dropout layers.

        Parameters
        ----------
        kernel_size : int
            The size of the convolutional kernel to be used in the Conv1d layer.

        Returns
        -------
        torch.nn.Sequential
            A sequential container containing the following layers:
            - Conv1d: 1D convolutional layer with input_width as input channels
            - AdaptiveMaxPool1d: 1D adaptive max pooling with output size 1
            - Flatten: Flattens the tensor for linear layer input
            - Linear: Linear layer with num_filters input features and 50 output features
            - Tanh: Hyperbolic tangent activation function
            - Dropout: 30% dropout regularization

        Notes
        -----
        The branch is designed to process 1D input data through a convolutional
        feature extraction pathway followed by a fully connected classifier head.
        The padding="same" parameter ensures the output size matches the input size
        for the convolutional layer.

        Examples
        --------
        >>> branch = self._make_branch(kernel_size=3)
        >>> print(type(branch))
        <class 'torch.nn.modules.container.Sequential'>
        """
        return nn.Sequential(
            nn.Conv1d(self.input_width, self.num_filters, kernel_size, padding="same"),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(self.num_filters, 50),
            nn.Tanh(),
            nn.Dropout(0.3),
        )

    def forward(self, x_small, x_med, x_large):
        """
        Forward pass of the multi-scale feature extraction network.

        This function processes input tensors through three parallel branches with different
        receptive fields and concatenates the outputs before applying a final fully connected
        layer with sigmoid activation.

        Parameters
        ----------
        x_small : torch.Tensor
            Input tensor for the small-scale branch with shape (batch_size, channels, height, width)
        x_med : torch.Tensor
            Input tensor for the medium-scale branch with shape (batch_size, channels, height, width)
        x_large : torch.Tensor
            Input tensor for the large-scale branch with shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, num_classes) containing sigmoid-activated
            predictions for each class

        Notes
        -----
        The function assumes that `self.branch_small`, `self.branch_med`, `self.branch_large`,
        `self.fc`, and `self.sigmoid` are properly initialized components of the class.

        Examples
        --------
        >>> import torch
        >>> # Assuming model is initialized
        >>> x_small = torch.randn(1, 3, 32, 32)
        >>> x_med = torch.randn(1, 3, 64, 64)
        >>> x_large = torch.randn(1, 3, 128, 128)
        >>> output = model.forward(x_small, x_med, x_large)
        >>> print(output.shape)
        torch.Size([1, num_classes])
        """
        # Process each branch
        out_small = self.branch_small(x_small)
        out_med = self.branch_med(x_med)
        out_large = self.branch_large(x_large)

        # Concatenate outputs
        merged = torch.cat([out_small, out_med, out_large], dim=1)

        # Final output
        out = self.fc(merged)
        out = self.sigmoid(out)

        return out

    def get_config(self):
        """
        Get the configuration parameters of the layer.

        Returns
        -------
        dict
            A dictionary containing the layer configuration parameters with the following keys:
            - "num_filters" (int): Number of filters in the layer
            - "kernel_sizes" (list of int): Size of the convolutional kernels
            - "input_lens" (list of int): Lengths of input sequences
            - "input_width" (int): Width of the input data
            - "dilation_rate" (int): Dilation rate for dilated convolution

        Notes
        -----
        This method returns a copy of the internal configuration parameters
        that can be used to reconstruct the layer with the same settings.

        Examples
        --------
        >>> config = layer.get_config()
        >>> print(config['num_filters'])
        32
        """
        return {
            "num_filters": self.num_filters,
            "kernel_sizes": self.kernel_sizes,
            "input_lens": self.input_lens,
            "input_width": self.input_width,
            "dilation_rate": self.dilation_rate,
        }


class MultiscaleCNNDLFilter(DeepLearningFilter):
    def __init__(
        self,
        num_filters=10,
        kernel_sizes=[4, 8, 12],
        input_lens=[64, 128, 192],
        input_width=1,
        dilation_rate=1,
        *args,
        **kwargs,
    ):
        """
        Initialize the MultiscaleCNNDLFilter layer.

        This constructor initializes a multiscale CNN filter with configurable
        kernel sizes, input lengths, and dilation rates for multi-scale feature extraction.

        Parameters
        ----------
        num_filters : int, optional
            Number of filters to use in each convolutional layer, default is 10
        kernel_sizes : list of int, optional
            List of kernel sizes for different convolutional layers, default is [4, 8, 12]
        input_lens : list of int, optional
            List of input sequence lengths for different scales, default is [64, 128, 192]
        input_width : int, optional
            Width of the input data, default is 1
        dilation_rate : int, optional
            Dilation rate for the convolutional layers, default is 1
        *args : tuple
            Variable length argument list passed to parent class
        **kwargs : dict
            Arbitrary keyword arguments passed to parent class

        Returns
        -------
        None
            This method initializes the object and does not return any value

        Notes
        -----
        The multiscale CNN filter uses multiple convolutional layers with different
        kernel sizes and dilation rates to capture features at multiple scales.
        The input data is processed through parallel convolutional branches,
        each with different kernel sizes and dilation rates.

        Examples
        --------
        >>> filter_layer = MultiscaleCNNDLFilter(
        ...     num_filters=20,
        ...     kernel_sizes=[3, 6, 9],
        ...     input_lens=[32, 64, 128],
        ...     input_width=2,
        ...     dilation_rate=2
        ... )
        """
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.input_lens = input_lens
        self.input_width = input_width
        self.dilation_rate = dilation_rate
        self.nettype = "multiscalecnn"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_sizes"] = self.kernel_sizes
        self.infodict["input_lens"] = self.input_lens
        self.infodict["input_width"] = self.input_width
        super(MultiscaleCNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return the model name and path based on current configuration parameters.

        This method constructs a descriptive model name string by joining various configuration
        parameters with specific prefixes and zero-padded numeric values. The resulting name
        is used to create a unique directory path for model storage.

        Parameters
        ----------
        self : object
            The instance containing model configuration parameters.

        Returns
        -------
        None
            This method does not return a value but sets the following attributes:
            - self.modelname: str, the generated model name
            - self.modelpath: str, the full path to the model directory

        Notes
        -----
        The generated model name includes the following components:
        - Model type: "model_multiscalecnn_pytorch"
        - Window size: "w" + zero-padded window size
        - Number of layers: "l" + zero-padded layer count
        - Number of filters: "fn" + zero-padded filter count
        - First kernel size: "fl" + zero-padded kernel size
        - Number of epochs: "e" + zero-padded epoch count
        - Exclusion threshold: "t" + threshold value
        - Correlation threshold: "ct" + threshold value
        - Step size: "s" + zero-padded step value
        - Dilation rate: "d" + dilation rate value
        - Activation function name

        Additional suffixes are appended if:
        - usebadpts is True: "_usebadpts"
        - excludebysubject is True: "_excludebysubject"
        - namesuffix is not None: "_{namesuffix}"

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 128
        >>> model.num_layers = 5
        >>> model.num_filters = 32
        >>> model.kernel_sizes = [3, 5, 7]
        >>> model.num_epochs = 100
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 16
        >>> model.dilation_rate = 2
        >>> model.activation = "relu"
        >>> model.usebadpts = True
        >>> model.excludebysubject = False
        >>> model.namesuffix = "exp1"
        >>> model.getname()
        >>> print(model.modelname)
        'model_multiscalecnn_pytorch_w128_l05_fn32_fl03_e100_t0.5_ct0.8_s16_d2_relu_usebadpts_exp1'
        """
        self.modelname = "_".join(
            [
                "model",
                "multiscalecnn",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "l" + str(self.num_layers).zfill(2),
                "fn" + str(self.num_filters).zfill(2),
                "fl" + str(self.kernel_sizes[0]).zfill(2),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
                "d" + str(self.dilation_rate),
                self.activation,
            ]
        )
        if self.usebadpts:
            self.modelname += "_usebadpts"
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        if self.namesuffix is not None:
            self.modelname += "_" + self.namesuffix
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and initialize a multiscale CNN model for network construction.

        This method initializes a MultiscaleCNNModel with the specified parameters
        and moves the model to the designated device (CPU or GPU).

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - num_filters : int
                Number of filters for the CNN layers
            - kernel_sizes : list of int
                List of kernel sizes for different scales
            - input_lens : list of int
                List of input lengths for different scales
            - input_width : int
                Width of the input data
            - dilation_rate : int
                Dilation rate for the convolutional layers
            - device : torch.device
                Device to move the model to (e.g., 'cuda' or 'cpu')

        Returns
        -------
        None
            This method does not return any value but modifies the instance
            by setting the `model` attribute to the created MultiscaleCNNModel.

        Notes
        -----
        The method assumes that all required attributes are properly initialized
        in the instance before calling this method. The model is automatically
        moved to the specified device using the `.to()` method.

        Examples
        --------
        >>> # Assuming instance with required attributes is created
        >>> instance.makenet()
        >>> print(instance.model)
        MultiscaleCNNModel(...)
        """
        self.model = MultiscaleCNNModel(
            self.num_filters,
            self.kernel_sizes,
            self.input_lens,
            self.input_width,
            self.dilation_rate,
        )
        self.model.to(self.device)


class ConvAutoencoderModel(nn.Module):
    def __init__(
        self,
        window_size,
        encoding_dim,
        num_filters,
        kernel_size,
        dropout_rate,
        activation,
        inputsize,
    ):
        """
        Initialize the ConvAutoencoderModel.

        This class implements a convolutional autoencoder for time series data. The model
        consists of an encoder and a decoder, with symmetric architecture. The encoder
        reduces the input dimensionality through convolutional and pooling layers, while
        the decoder reconstructs the input from the encoded representation.

        Parameters
        ----------
        window_size : int
            The length of the input time series window.
        encoding_dim : int
            The dimensionality of the latent space representation.
        num_filters : int
            The number of filters in the first convolutional layer.
        kernel_size : int
            The size of the convolutional kernels.
        dropout_rate : float
            The dropout rate applied after each convolutional layer.
        activation : str
            The activation function to use. Supported values are "relu" and "tanh".
        inputsize : int
            The number of input channels (e.g., number of features in the time series).

        Returns
        -------
        None
            This method initializes the model in-place and does not return any value.

        Notes
        -----
        The model uses a symmetric encoder-decoder architecture. The encoder reduces
        the input size through 4 max-pooling layers, and the decoder reconstructs
        the input using upsample and convolutional layers. The final layer uses
        a convolution with padding to match the input size.

        Examples
        --------
        >>> model = ConvAutoencoderModel(
        ...     window_size=100,
        ...     encoding_dim=32,
        ...     num_filters=32,
        ...     kernel_size=3,
        ...     dropout_rate=0.2,
        ...     activation="relu",
        ...     inputsize=1
        ... )
        """
        super(ConvAutoencoderModel, self).__init__()

        self.window_size = window_size
        self.encoding_dim = encoding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inputsize = inputsize

        # Get activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU

        # Initial conv block
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Conv1d(inputsize, num_filters, kernel_size, padding="same"))
        self.encoder_layers.append(nn.BatchNorm1d(num_filters))
        self.encoder_layers.append(nn.Dropout(dropout_rate))
        self.encoder_layers.append(act_fn())
        self.encoder_layers.append(nn.MaxPool1d(2, padding=1))

        # Encoding path (3 layers)
        nfilters = num_filters
        self.filter_list = []
        for _ in range(3):
            nfilters *= 2
            self.filter_list.append(nfilters)
            self.encoder_layers.append(
                nn.Conv1d(nfilters // 2, nfilters, kernel_size, padding="same")
            )
            self.encoder_layers.append(nn.BatchNorm1d(nfilters))
            self.encoder_layers.append(nn.Dropout(dropout_rate))
            self.encoder_layers.append(act_fn())
            self.encoder_layers.append(nn.MaxPool1d(2, padding=1))

        # Calculate size after pooling
        self.encoded_size = window_size
        for _ in range(4):  # 4 pooling layers
            self.encoded_size = (self.encoded_size + 1) // 2

        # Bottleneck
        self.flatten = nn.Flatten()
        self.encode_fc = nn.Linear(nfilters * self.encoded_size, encoding_dim)
        self.encode_act = act_fn()
        self.decode_fc = nn.Linear(encoding_dim, nfilters * self.encoded_size)
        self.decode_act = act_fn()
        self.unflatten_size = (nfilters, self.encoded_size)

        # Decoding path (mirror)
        self.decoder_layers = nn.ModuleList()
        for i, filters in enumerate(reversed(self.filter_list)):
            self.decoder_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            if i == 0:
                self.decoder_layers.append(
                    nn.Conv1d(nfilters, filters, kernel_size, padding="same")
                )
            else:
                self.decoder_layers.append(
                    nn.Conv1d(self.filter_list[-i], filters, kernel_size, padding="same")
                )
            self.decoder_layers.append(nn.BatchNorm1d(filters))
            self.decoder_layers.append(nn.Dropout(dropout_rate))
            self.decoder_layers.append(act_fn())

        # Final upsampling
        self.decoder_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.decoder_layers.append(nn.Conv1d(num_filters, inputsize, kernel_size, padding="same"))

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Applies encoding, bottleneck processing, and decoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width) where
            height and width should match the expected window size.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, channels, window_size, width)
            with the same spatial dimensions as the input.

        Notes
        -----
        The forward pass consists of three main stages:
        1. Encoding: Input is passed through encoder layers
        2. Bottleneck: Flattening, encoding, and decoding with activation functions
        3. Decoding: Reconstructed features are passed through decoder layers

        The output is cropped or padded to match the original window size.

        Examples
        --------
        >>> import torch
        >>> model = AutoEncoder()
        >>> x = torch.randn(1, 3, 64, 64)
        >>> output = model.forward(x)
        >>> print(output.shape)
        torch.Size([1, 3, 64, 64])
        """
        # Encoding
        for layer in self.encoder_layers:
            x = layer(x)

        # Bottleneck
        x = self.flatten(x)
        x = self.encode_fc(x)
        x = self.encode_act(x)
        x = self.decode_fc(x)
        x = self.decode_act(x)
        x = x.view(x.size(0), *self.unflatten_size)

        # Decoding
        for layer in self.decoder_layers:
            x = layer(x)

        # Crop/pad to original window size
        if x.size(2) > self.window_size:
            x = x[:, :, : self.window_size]
        elif x.size(2) < self.window_size:
            pad_size = self.window_size - x.size(2)
            x = nn.functional.pad(x, (0, pad_size))

        return x

    def get_config(self):
        """
        Get the configuration parameters of the model.

        Returns
        -------
        dict
            A dictionary containing all configuration parameters with their current values:
            - "window_size" (int): Size of the sliding window
            - "encoding_dim" (int): Dimension of the encoding layer
            - "num_filters" (int): Number of filters in the convolutional layers
            - "kernel_size" (int): Size of the convolutional kernel
            - "dropout_rate" (float): Dropout rate for regularization
            - "activation" (str): Activation function to use
            - "inputsize" (int): Size of the input data

        Notes
        -----
        This method returns a copy of the current configuration. Modifications to the
        returned dictionary will not affect the original model configuration.

        Examples
        --------
        >>> config = model.get_config()
        >>> print(config['window_size'])
        100
        """
        return {
            "window_size": self.window_size,
            "encoding_dim": self.encoding_dim,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "inputsize": self.inputsize,
        }


class ConvAutoencoderDLFilter(DeepLearningFilter):
    def __init__(
        self,
        encoding_dim: int = 10,
        num_filters: int = 5,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize ConvAutoencoderDLFilter instance.

        Parameters
        ----------
        encoding_dim : int, optional
            Dimension of the encoded representation, by default 10
        num_filters : int, optional
            Number of filters in the convolutional layers, by default 5
        kernel_size : int, optional
            Size of the convolutional kernel, by default 5
        dilation_rate : int, optional
            Dilation rate for the convolutional layers, by default 1
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments

        Returns
        -------
        None
            This method does not return any value

        Notes
        -----
        This constructor initializes a convolutional autoencoder with dilated filters.
        The network type is set to "convautoencoder" and various configuration parameters
        are stored in the infodict for later reference.

        Examples
        --------
        >>> autoencoder = ConvAutoencoderDLFilter(
        ...     encoding_dim=15,
        ...     num_filters=8,
        ...     kernel_size=3,
        ...     dilation_rate=2
        ... )
        """
        self.encoding_dim = encoding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.nettype = "convautoencoder"
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        self.infodict["nettype"] = self.nettype
        self.infodict["encoding_dim"] = self.encoding_dim
        super(ConvAutoencoderDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure the model name and path based on current parameters.

        This method constructs a descriptive model name string using various
        configuration parameters and creates the corresponding directory path.
        The generated name includes information about window size, encoding dimensions,
        filters, kernel size, epochs, thresholds, and other model configuration options.

        Parameters
        ----------
        self : object
            The instance of the class containing model configuration parameters.
            Expected attributes include:
            - window_size : int
            - encoding_dim : int
            - num_filters : int
            - kernel_size : int
            - num_epochs : int
            - excludethresh : float
            - corrthresh : float
            - step : int
            - activation : str
            - usebadpts : bool
            - excludebysubject : bool
            - namesuffix : str, optional
            - modelroot : str

        Returns
        -------
        None
            This method does not return a value but modifies the instance attributes:
            - self.modelname : str
            - self.modelpath : str

        Notes
        -----
        The model name is constructed with the following components:
        - "model_convautoencoder_pytorch" as base identifier
        - Window size with 3-digit zero-padded formatting (wXXX)
        - Encoding dimension with 3-digit zero-padded formatting (enXXX)
        - Number of filters with 2-digit zero-padded formatting (fnXX)
        - Kernel size with 2-digit zero-padded formatting (flXX)
        - Number of epochs with 3-digit zero-padded formatting (eXXX)
        - Exclusion threshold (tX.XX)
        - Correlation threshold (ctX.XX)
        - Step size (sX)
        - Activation function name

        Additional suffixes are appended based on:
        - usebadpts flag
        - excludebysubject flag
        - namesuffix parameter

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.encoding_dim = 50
        >>> model.getname()
        >>> print(model.modelname)
        'model_convautoencoder_pytorch_w100_en050_fn10_fl05_e001_t0.5_ct0.8_s1_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "convautoencoder",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "en" + str(self.encoding_dim).zfill(3),
                "fn" + str(self.num_filters).zfill(2),
                "fl" + str(self.kernel_size).zfill(2),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
                self.activation,
            ]
        )
        if self.usebadpts:
            self.modelname += "_usebadpts"
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        if self.namesuffix is not None:
            self.modelname += "_" + self.namesuffix
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and initialize a convolutional autoencoder model.

        This method constructs a ConvAutoencoderModel with the specified parameters
        and moves it to the designated device (CPU or GPU).

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Expected to have the
            following attributes:
            - window_size : int
                Size of the input window
            - encoding_dim : int
                Dimension of the encoded representation
            - num_filters : int
                Number of filters in the convolutional layers
            - kernel_size : int
                Size of the convolutional kernel
            - dropout_rate : float
                Dropout rate for regularization
            - activation : str or callable
                Activation function to use
            - inputsize : tuple
                Input size dimensions
            - device : torch.device
                Device to move the model to (CPU or GPU)

        Returns
        -------
        None
            This method does not return any value. It initializes the model
            attribute of the class instance.

        Notes
        -----
        The method assumes that `ConvAutoencoderModel` is a valid class that accepts
        the specified parameters. The model is automatically moved to the device
        specified by `self.device`.

        Examples
        --------
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.window_size = 100
        ...         self.encoding_dim = 32
        ...         self.num_filters = 64
        ...         self.kernel_size = 3
        ...         self.dropout_rate = 0.2
        ...         self.activation = 'relu'
        ...         self.inputsize = (1, 100)
        ...         self.device = torch.device('cpu')
        ...         self.model = None
        ...
        ...     def makenet(self):
        ...         self.model = ConvAutoencoderModel(
        ...             self.window_size,
        ...             self.encoding_dim,
        ...             self.num_filters,
        ...             self.kernel_size,
        ...             self.dropout_rate,
        ...             self.activation,
        ...             self.inputsize,
        ...         )
        ...         self.model.to(self.device)
        ...
        >>> model = MyModel()
        >>> model.makenet()
        >>> print(model.model)
        """
        self.model = ConvAutoencoderModel(
            self.window_size,
            self.encoding_dim,
            self.num_filters,
            self.kernel_size,
            self.dropout_rate,
            self.activation,
            self.inputsize,
        )
        self.model.to(self.device)


class CRNNModel(nn.Module):
    def __init__(
        self, num_filters, kernel_size, encoding_dim, dropout_rate, activation, inputsize
    ):
        """
        Initialize the CRNNModel.

        This function initializes a Convolutional Recurrent Neural Network (CRNN) model
        with convolutional front-end, bidirectional LSTM layers, and output mapping.
        The model processes sequential data through convolutional layers, applies
        bidirectional LSTM encoding, and maps the output back to the original input size.

        Parameters
        ----------
        num_filters : int
            Number of filters in the convolutional layers
        kernel_size : int
            Size of the convolutional kernel
        encoding_dim : int
            Dimension of the LSTM encoding (hidden state size)
        dropout_rate : float
            Dropout rate for regularization
        activation : str
            Activation function to use ('relu' or 'tanh')
        inputsize : int
            Size of the input features

        Returns
        -------
        None
            Initializes the CRNNModel instance

        Notes
        -----
        The model uses a bidirectional LSTM with batch_first=True.
        The convolutional layers use 'same' padding to maintain sequence length.
        Default activation function is ReLU if an invalid activation is provided.

        Examples
        --------
        >>> model = CRNNModel(
        ...     num_filters=32,
        ...     kernel_size=3,
        ...     encoding_dim=64,
        ...     dropout_rate=0.2,
        ...     activation='relu',
        ...     inputsize=128
        ... )
        """
        super(CRNNModel, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inputsize = inputsize

        # Get activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU

        # Convolutional front-end
        self.conv1 = nn.Conv1d(inputsize, num_filters, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.act1 = act_fn()

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.act2 = act_fn()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(num_filters * 2, encoding_dim, batch_first=True, bidirectional=True)

        # Output mapping
        self.fc_out = nn.Linear(encoding_dim * 2, inputsize)

    def forward(self, x):
        """
        Forward pass through the neural network architecture.

        This function processes input data through a convolutional neural network
        followed by an LSTM layer and a fully connected output layer. The input
        is first processed through two convolutional blocks, then reshaped for
        LSTM processing, and finally converted back to the original output format.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length) containing
            the input sequence data to be processed.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, channels, length) containing
            the processed sequence data after passing through all layers.

        Notes
        -----
        The function performs the following operations in sequence:
        1. Two convolutional blocks with batch normalization, dropout, and activation
        2. Permute operation to reshape data for LSTM processing (batch, seq_len, features)
        3. LSTM layer processing
        4. Fully connected output layer
        5. Final permutation to restore original shape (batch, channels, length)

        Examples
        --------
        >>> import torch
        >>> model = YourModelClass()
        >>> x = torch.randn(32, 1, 100)  # batch_size=32, channels=1, length=100
        >>> output = model.forward(x)
        >>> print(output.shape)  # torch.Size([32, 1, 100])
        """
        # Conv layers expect (batch, channels, length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.act2(x)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        # Output layer
        x = self.fc_out(x)

        # Convert back to (batch, channels, length)
        x = x.permute(0, 2, 1)

        return x

    def get_config(self):
        """
        Get the configuration parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the model configuration parameters with the following keys:
            - "num_filters" (int): Number of filters in the convolutional layers
            - "kernel_size" (int): Size of the convolutional kernel
            - "encoding_dim" (int): Dimension of the encoding layer
            - "dropout_rate" (float): Dropout rate for regularization
            - "activation" (str): Activation function used in the layers
            - "inputsize" (int): Size of the input data

        Notes
        -----
        This method returns a copy of the current configuration parameters. Modifications
        to the returned dictionary will not affect the original model configuration.

        Examples
        --------
        >>> model = MyModel()
        >>> config = model.get_config()
        >>> print(config['num_filters'])
        32
        """
        return {
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "encoding_dim": self.encoding_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "inputsize": self.inputsize,
        }


class CRNNDLFilter(DeepLearningFilter):
    def __init__(
        self,
        encoding_dim: int = 10,
        num_filters: int = 10,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize CRNNDLFilter layer.

        Parameters
        ----------
        encoding_dim : int, optional
            Dimension of the encoding layer, by default 10
        num_filters : int, optional
            Number of filters in the convolutional layer, by default 10
        kernel_size : int, optional
            Size of the convolutional kernel, by default 5
        dilation_rate : int, optional
            Dilation rate for the convolutional layer, by default 1
        *args : tuple
            Variable length argument list
        **kwargs : dict
            Arbitrary keyword arguments

        Returns
        -------
        None
            This method does not return any value

        Notes
        -----
        This constructor initializes a CRNN (Convolutional Recurrent Neural Network)
        with dilated filters. The layer type is set to "crnn" and configuration
        parameters are stored in infodict for later reference.

        Examples
        --------
        >>> layer = CRNNDLFilter(encoding_dim=20, num_filters=15, kernel_size=3)
        >>> print(layer.nettype)
        'crnn'
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.encoding_dim = encoding_dim
        self.nettype = "crnn"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        self.infodict["encoding_dim"] = self.encoding_dim
        super(CRNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure model name and path based on configuration parameters.

        This method constructs a descriptive model name string based on various configuration
        parameters and creates the corresponding model directory path. The generated name
        includes information about window size, encoding dimensions, filters, kernel size,
        epochs, thresholds, step size, and activation function.

        Parameters
        ----------
        self : object
            The instance containing configuration parameters for model naming.

        Returns
        -------
        None
            This method modifies instance attributes in-place and does not return a value.

        Notes
        -----
        The generated model name follows a consistent naming convention:
        'model_crnn_pytorch_wXXX_enXXX_fnXX_flXX_eXXX_tX_ctX_sX_activation'
        where XXX represents zero-padded numeric values and X represents single digits.

        Additional suffixes are appended based on:
        - usebadpts: '_usebadpts' if True
        - excludebysubject: '_excludebysubject' if True
        - namesuffix: '_{suffix}' if not None

        Examples
        --------
        >>> model = ModelClass()
        >>> model.window_size = 100
        >>> model.encoding_dim = 128
        >>> model.num_filters = 32
        >>> model.kernel_size = 5
        >>> model.num_epochs = 100
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 10
        >>> model.activation = 'relu'
        >>> model.modelroot = '/path/to/models'
        >>> model.getname()
        >>> print(model.modelname)
        'model_crnn_pytorch_w100_en128_fn32_fl05_e100_t0.5_ct0.8_s10_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "crnn",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "en" + str(self.encoding_dim).zfill(3),
                "fn" + str(self.num_filters).zfill(2),
                "fl" + str(self.kernel_size).zfill(2),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
                self.activation,
            ]
        )
        if self.usebadpts:
            self.modelname += "_usebadpts"
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        if self.namesuffix is not None:
            self.modelname += "_" + self.namesuffix
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and initialize a CRNN model for neural network training.

        This method initializes a CRNN (Convolutional Recurrent Neural Network) model
        using the specified configuration parameters and moves it to the designated
        device (CPU or GPU).

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Expected to have the
            following attributes:
            - num_filters : int
                Number of filters in the convolutional layers
            - kernel_size : int or tuple
                Size of the convolutional kernel
            - encoding_dim : int
                Dimension of the encoding layer
            - dropout_rate : float
                Dropout rate for regularization
            - activation : str or callable
                Activation function to use
            - inputsize : tuple
                Input dimensions for the model
            - device : torch.device
                Device to move the model to (CPU or GPU)

        Returns
        -------
        None
            This method does not return any value. It initializes the model attribute
            of the class instance.

        Notes
        -----
        The method assumes that the CRNNModel class is properly imported and available
        in the namespace. The model is automatically moved to the device specified
        in self.device.

        Examples
        --------
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.num_filters = 32
        ...         self.kernel_size = 3
        ...         self.encoding_dim = 128
        ...         self.dropout_rate = 0.2
        ...         self.activation = 'relu'
        ...         self.inputsize = (1, 28, 28)
        ...         self.device = torch.device('cpu')
        ...
        ...     def makenet(self):
        ...         self.model = CRNNModel(
        ...             self.num_filters,
        ...             self.kernel_size,
        ...             self.encoding_dim,
        ...             self.dropout_rate,
        ...             self.activation,
        ...             self.inputsize,
        ...         )
        ...         self.model.to(self.device)
        ...
        >>> model = MyModel()
        >>> model.makenet()
        >>> print(model.model)
        """
        self.model = CRNNModel(
            self.num_filters,
            self.kernel_size,
            self.encoding_dim,
            self.dropout_rate,
            self.activation,
            self.inputsize,
        )
        self.model.to(self.device)


class LSTMModel(nn.Module):
    def __init__(self, num_units, num_layers, dropout_rate, window_size, inputsize):
        """
        Initialize the LSTMModel with specified architecture parameters.

        Parameters
        ----------
        num_units : int
            Number of units in each LSTM layer
        num_layers : int
            Number of LSTM layers in the model
        dropout_rate : float
            Dropout rate for LSTM layers (applied only if num_layers > 1)
        window_size : int
            Size of the sliding window used for sequence processing
        inputsize : int
            Dimensionality of input features

        Returns
        -------
        None
            Initializes the LSTMModel instance with the specified architecture

        Notes
        -----
        This constructor creates a bidirectional LSTM model with residual connections.
        The model uses LSTM layers with bidirectional processing and time-distributed
        dense layers for output transformation. Dropout is applied between layers
        when multiple layers are present.

        Examples
        --------
        >>> model = LSTMModel(num_units=128, num_layers=2, dropout_rate=0.2,
        ...                   window_size=10, inputsize=20)
        >>> print(model)
        """
        super(LSTMModel, self).__init__()

        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        self.inputsize = inputsize

        self.lstm_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()

        for _ in range(num_layers):
            # Bidirectional LSTM
            self.lstm_layers.append(
                nn.LSTM(
                    inputsize if len(self.lstm_layers) == 0 else inputsize,
                    num_units,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout_rate if num_layers > 1 else 0,
                )
            )
            # Time-distributed dense layer
            self.dense_layers.append(nn.Linear(num_units * 2, inputsize))

    def forward(self, x):
        """
        Forward pass through LSTM and dense layers.

        Apply a sequence of LSTM layers followed by dense layers to the input tensor,
        with appropriate dimension permutations to maintain correct data flow.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, length) containing the sequential data.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch, channels, length) after processing through
            LSTM and dense layers.

        Notes
        -----
        The function performs the following operations:
        1. Permutes input from (batch, channels, length) to (batch, length, channels)
        2. Processes through LSTM layers sequentially
        3. Applies dense layers to each time step
        4. Permutes output back to (batch, channels, length)

        Examples
        --------
        >>> import torch
        >>> # Assuming self.lstm_layers and self.dense_layers are initialized
        >>> x = torch.randn(32, 128, 100)  # batch=32, channels=128, length=100
        >>> output = model.forward(x)
        >>> output.shape
        torch.Size([32, 128, 100])
        """
        # x is (batch, channels, length), convert to (batch, length, channels)
        x = x.permute(0, 2, 1)

        for lstm, dense in zip(self.lstm_layers, self.dense_layers):
            x, _ = lstm(x)
            # Apply dense layer across time steps
            x = dense(x)

        # Convert back to (batch, channels, length)
        x = x.permute(0, 2, 1)

        return x

    def get_config(self):
        """
        Get the configuration parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the model configuration parameters with the following keys:
            - "num_units" (int): Number of units in each layer
            - "num_layers" (int): Number of layers in the model
            - "dropout_rate" (float): Dropout rate for regularization
            - "window_size" (int): Size of the sliding window for sequence processing
            - "inputsize" (int): Size of the input features

        Notes
        -----
        This method returns a copy of the internal configuration parameters.
        The returned dictionary can be used to recreate the model with the same configuration.

        Examples
        --------
        >>> config = model.get_config()
        >>> print(config['num_units'])
        128
        >>> new_model = ModelClass(**config)
        """
        return {
            "num_units": self.num_units,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "window_size": self.window_size,
            "inputsize": self.inputsize,
        }


class LSTMDLFilter(DeepLearningFilter):
    def __init__(self, num_units: int = 16, *args, **kwargs) -> None:
        """
        Initialize the LSTMDLFilter layer.

        Parameters
        ----------
        num_units : int, optional
            Number of units in the LSTM layer, by default 16
        *args
            Variable length argument list passed to parent class
        **kwargs
            Arbitrary keyword arguments passed to parent class

        Returns
        -------
        None
            This method initializes the instance and does not return any value

        Notes
        -----
        This constructor sets up the LSTM layer with specified number of units and
        initializes the network type identifier. The infodict is updated with both
        the network type and number of units for tracking purposes.

        Examples
        --------
        >>> layer = LSTMDLFilter(num_units=32)
        >>> print(layer.num_units)
        32
        >>> print(layer.nettype)
        'lstm'
        """
        self.num_units = num_units
        self.nettype = "lstm"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_units"] = self.num_units
        super(LSTMDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure model name and path based on current parameters.

        This method constructs a descriptive model name string using various
        hyperparameters and configuration settings. It then creates the
        corresponding directory path and ensures it exists.

        Parameters
        ----------
        self : object
            The instance containing model configuration attributes.

        Returns
        -------
        None
            This method modifies instance attributes in-place and does not return a value.

        Notes
        -----
        The generated model name follows a specific format:
        "model_lstm_pytorch_wXXX_lYY_nuZZZ_dDD_rdDD_eFFF_tT_ctTT_sS"
        where XXX, YY, ZZZ, DD, FF, T, TT, S represent formatted parameter values.

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.num_layers = 2
        >>> model.num_units = 128
        >>> model.dropout_rate = 0.2
        >>> model.num_epochs = 100
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 1
        >>> model.excludebysubject = True
        >>> model.getname()
        >>> print(model.modelname)
        'model_lstm_pytorch_w100_l02_nu128_d02_rd02_e100_t05_ct08_s1_excludebysubject'
        """
        self.modelname = "_".join(
            [
                "model",
                "lstm",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "l" + str(self.num_layers).zfill(2),
                "nu" + str(self.num_units),
                "d" + str(self.dropout_rate),
                "rd" + str(self.dropout_rate),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
            ]
        )
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and initialize an LSTM model for neural network training.

        This method initializes an LSTMModel with the specified architecture parameters
        and moves the model to the designated device (CPU or GPU).

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - num_units : int
                Number of units in each LSTM layer
            - num_layers : int
                Number of LSTM layers in the model
            - dropout_rate : float
                Dropout rate for regularization
            - window_size : int
                Size of the input window for time series data
            - inputsize : int
                Size of the input features
            - device : torch.device
                Device to move the model to (e.g., 'cpu' or 'cuda')

        Returns
        -------
        None
            This method does not return any value. It initializes the model attribute
            and moves it to the specified device.

        Notes
        -----
        The method creates an LSTMModel instance with the following parameters:
        - num_units: Number of hidden units in LSTM layers
        - num_layers: Number of stacked LSTM layers
        - dropout_rate: Dropout probability for regularization
        - window_size: Input sequence length
        - inputsize: Feature dimension of input data

        Examples
        --------
        >>> # Assuming self is an instance with required attributes
        >>> self.makenet()
        >>> # Model is now initialized and moved to specified device
        """
        self.model = LSTMModel(
            self.num_units,
            self.num_layers,
            self.dropout_rate,
            self.window_size,
            self.inputsize,
        )
        self.model.to(self.device)


class HybridModel(nn.Module):
    def __init__(
        self,
        num_filters,
        kernel_size,
        num_units,
        num_layers,
        dropout_rate,
        activation,
        inputsize,
        window_size,
        invert,
    ):
        """
        Initialize the HybridModel with configurable CNN-LSTM architecture.

        Parameters
        ----------
        num_filters : int
            Number of filters in the convolutional layers.
        kernel_size : int
            Size of the convolutional kernel.
        num_units : int
            Number of units in the LSTM layers.
        num_layers : int
            Total number of layers in the model.
        dropout_rate : float
            Dropout rate for regularization.
        activation : str
            Activation function to use; options are 'relu' or 'tanh'.
        inputsize : int
            Size of the input features.
        window_size : int
            Size of the sliding window for input data.
        invert : bool
            If True, applies CNN first followed by LSTM. Otherwise, applies LSTM first followed by CNN.

        Returns
        -------
        None
            This method initializes the model's layers and components but does not return any value.

        Notes
        -----
        The model supports two architectures:
        - If `invert=False`: LSTM → CNN
        - If `invert=True`: CNN → LSTM

        Examples
        --------
        >>> model = HybridModel(
        ...     num_filters=64,
        ...     kernel_size=3,
        ...     num_units=128,
        ...     num_layers=3,
        ...     dropout_rate=0.2,
        ...     activation="relu",
        ...     inputsize=10,
        ...     window_size=100,
        ...     invert=True
        ... )
        """
        super(HybridModel, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inputsize = inputsize
        self.window_size = window_size
        self.invert = invert

        # Get activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU

        self.layers = nn.ModuleList()

        if invert:
            # CNN first, then LSTM
            # Input layer
            self.layers.append(nn.Conv1d(inputsize, num_filters, kernel_size, padding="same"))
            self.layers.append(nn.BatchNorm1d(num_filters))
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(act_fn())

            # Intermediate CNN layers
            for _ in range(num_layers - 2):
                self.layers.append(
                    nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
                )
                self.layers.append(nn.BatchNorm1d(num_filters))
                self.layers.append(nn.Dropout(dropout_rate))
                self.layers.append(act_fn())

            # LSTM layer
            self.lstm = nn.LSTM(
                num_filters, num_units, batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.lstm_dense = nn.Linear(num_units * 2, inputsize)

        else:
            # LSTM first, then CNN
            self.lstm = nn.LSTM(
                inputsize, num_units, batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.lstm_dense = nn.Linear(num_units * 2, inputsize)
            self.lstm_dropout = nn.Dropout(dropout_rate)

            # Intermediate CNN layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Conv1d(inputsize, num_filters, kernel_size, padding="same"))
                self.layers.append(nn.BatchNorm1d(num_filters))
                self.layers.append(nn.Dropout(dropout_rate))
                self.layers.append(act_fn())

            # Output layer
            self.output_conv = nn.Conv1d(
                num_filters if num_layers > 2 else inputsize,
                inputsize,
                kernel_size,
                padding="same",
            )

    def forward(self, x):
        """
        Forward pass of the model with optional CNN-LSTM hybrid architecture.

        This method implements a flexible forward pass that can operate in two modes
        depending on the `invert` flag. When `invert` is True, the sequence processing
        follows CNN → LSTM → CNN structure. When `invert` is False, the sequence
        processing follows LSTM → CNN structure.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length) or
            (batch_size, sequence_length, channels) depending on the processing path.

        Returns
        -------
        torch.Tensor
            Output tensor with the same batch dimension as input, with shape
            dependent on the specific layers and processing path used.

        Notes
        -----
        The function handles different tensor permutations based on the processing
        order:
        - CNN → LSTM path: permutes from (B, C, L) to (B, L, C) for LSTM, then back
        - LSTM → CNN path: permutes from (B, C, L) to (B, L, C) for LSTM, then back
        The `invert` flag determines which processing order is used.

        Examples
        --------
        >>> model = MyModel()
        >>> x = torch.randn(32, 10, 128)  # batch_size=32, seq_len=10, features=128
        >>> output = model.forward(x)
        >>> print(output.shape)
        torch.Size([32, 10, 256])
        """
        if self.invert:
            # Apply CNN layers
            for layer in self.layers:
                x = layer(x)

            # LSTM expects (batch, seq_len, features)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = self.lstm_dense(x)

            # Convert back to (batch, channels, length)
            x = x.permute(0, 2, 1)

        else:
            # LSTM first
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = self.lstm_dense(x)
            x = self.lstm_dropout(x)
            x = x.permute(0, 2, 1)

            # CNN layers
            for layer in self.layers:
                x = layer(x)

            # Output layer
            if hasattr(self, "output_conv"):
                x = self.output_conv(x)

        return x

    def get_config(self):
        """
        Get the configuration parameters of the model.

        Returns
        -------
        dict
            A dictionary containing all configuration parameters with their current values:
            - num_filters: int, number of filters in the convolutional layers
            - kernel_size: int, size of the convolutional kernel
            - num_units: int, number of units in the dense layers
            - num_layers: int, number of layers in the model
            - dropout_rate: float, dropout rate for regularization
            - activation: str or callable, activation function to use
            - inputsize: int, size of the input features
            - window_size: int, size of the sliding window
            - invert: bool, whether to invert the input data

        Notes
        -----
        This method returns a copy of the internal configuration dictionary.
        Modifications to the returned dictionary will not affect the original model configuration.

        Examples
        --------
        >>> config = model.get_config()
        >>> print(config['num_filters'])
        32
        """
        return {
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "num_units": self.num_units,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "inputsize": self.inputsize,
            "window_size": self.window_size,
            "invert": self.invert,
        }


class HybridDLFilter(DeepLearningFilter):
    def __init__(
        self,
        invert: bool = False,
        num_filters: int = 10,
        kernel_size: int = 5,
        num_units: int = 16,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize HybridDLFilter layer.

        Parameters
        ----------
        invert : bool, default=False
            If True, inverts the filter response. If False, applies normal filtering.
        num_filters : int, default=10
            Number of filters to apply in the convolutional layer.
        kernel_size : int, default=5
            Size of the convolutional kernel.
        num_units : int, default=16
            Number of units in the dense layer.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
            This method initializes the HybridDLFilter instance and does not return any value.

        Notes
        -----
        This constructor sets up a hybrid deep learning filter that combines convolutional
        and dense layers. The infodict dictionary is populated with configuration parameters
        for tracking and logging purposes.

        Examples
        --------
        >>> filter_layer = HybridDLFilter(
        ...     invert=True,
        ...     num_filters=20,
        ...     kernel_size=3,
        ...     num_units=32
        ... )
        """
        self.invert = invert
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_units = num_units
        self.nettype = "hybrid"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        self.infodict["invert"] = self.invert
        self.infodict["num_units"] = self.num_units
        super(HybridDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure the model name and path based on current parameters.

        This method constructs a descriptive model name string using various
        hyperparameters and configuration settings. The generated name follows
        a standardized format that includes window size, layer count, filter count,
        kernel size, number of units, dropout rates, number of epochs, threshold
        values, step size, and activation function. The method also creates the
        corresponding model directory path and ensures it exists.

        Parameters
        ----------
        self : object
            The instance of the class containing the model configuration attributes.
            Required attributes include:
            - window_size : int
            - num_layers : int
            - num_filters : int
            - kernel_size : int
            - num_units : int
            - dropout_rate : float
            - num_epochs : int
            - excludethresh : float
            - corrthresh : float
            - step : int
            - activation : str
            - invert : bool
            - excludebysubject : bool
            - modelroot : str

        Returns
        -------
        None
            This method modifies the instance attributes in-place:
            - self.modelname : str
            - self.modelpath : str

        Notes
        -----
        The model name format follows this pattern:
        "model_hybrid_pytorch_wXXX_lYY_fnZZ_flZZ_nuZZ_dZZ_rdZZ_eXXX_tX_ctX_sX_activation"
        where XXX, YY, ZZ, etc. represent zero-padded numerical values.

        Additional suffixes are appended based on:
        - "_invert" if self.invert is True
        - "_excludebysubject" if self.excludebysubject is True

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.num_layers = 2
        >>> model.getname()
        >>> print(model.modelname)
        'model_hybrid_pytorch_w100_l02_fn08_fl08_nu128_d05_rd05_e100_t05_ct08_s1_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "hybrid",
                "pytorch",
                "w" + str(self.window_size).zfill(3),
                "l" + str(self.num_layers).zfill(2),
                "fn" + str(self.num_filters).zfill(2),
                "fl" + str(self.kernel_size).zfill(2),
                "nu" + str(self.num_units),
                "d" + str(self.dropout_rate),
                "rd" + str(self.dropout_rate),
                "e" + str(self.num_epochs).zfill(3),
                "t" + str(self.excludethresh),
                "ct" + str(self.corrthresh),
                "s" + str(self.step),
                self.activation,
            ]
        )
        if self.invert:
            self.modelname += "_invert"
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        """
        Create and initialize a hybrid neural network model.

        This method constructs a HybridModel with the specified architecture parameters
        and moves it to the designated device (CPU or GPU).

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - num_filters : int
                Number of filters in the convolutional layers
            - kernel_size : int
                Size of the convolutional kernels
            - num_units : int
                Number of units in the dense layers
            - num_layers : int
                Number of layers in the model
            - dropout_rate : float
                Dropout rate for regularization
            - activation : str or callable
                Activation function to use
            - inputsize : int
                Size of the input features
            - window_size : int
                Size of the sliding window
            - invert : bool
                Whether to invert the model architecture

        Returns
        -------
        None
            This method does not return any value. It initializes the model attribute
            and moves it to the specified device.

        Notes
        -----
        The method assumes that the instance has all required attributes set before
        calling. The model is moved to the device specified by `self.device`.

        Examples
        --------
        >>> model = MyModel()
        >>> model.num_filters = 32
        >>> model.kernel_size = 3
        >>> model.num_units = 64
        >>> model.num_layers = 2
        >>> model.dropout_rate = 0.2
        >>> model.activation = 'relu'
        >>> model.inputsize = 10
        >>> model.window_size = 5
        >>> model.invert = False
        >>> model.device = 'cuda'
        >>> model.makenet()
        >>> print(model.model)
        """
        self.model = HybridModel(
            self.num_filters,
            self.kernel_size,
            self.num_units,
            self.num_layers,
            self.dropout_rate,
            self.activation,
            self.inputsize,
            self.window_size,
            self.invert,
        )
        self.model.to(self.device)


def filtscale(
    data: NDArray,
    scalefac: float = 1.0,
    reverse: bool = False,
    hybrid: bool = False,
    lognormalize: bool = True,
    epsilon: float = 1e-10,
    numorders: int = 6,
) -> tuple[NDArray, float] | NDArray:
    """
    Apply or reverse a scaling transformation to spectral data.

    This function performs either forward or inverse scaling of input data,
    typically used in signal processing or spectral analysis. In forward mode,
    it computes the FFT of the input data and applies normalization and scaling
    to the magnitude and phase components. In reverse mode, it reconstructs
    the original time-domain signal from scaled magnitude and phase components.

    Parameters
    ----------
    data : NDArray
        Input time-domain signal or scaled spectral data depending on `reverse` flag.
    scalefac : float, optional
        Scaling factor used in normalization. Default is 1.0.
    reverse : bool, optional
        If True, performs inverse transformation to reconstruct the original signal.
        If False, performs forward transformation. Default is False.
    hybrid : bool, optional
        If True, returns a hybrid output combining original signal and magnitude.
        Only applicable in forward mode. Default is False.
    lognormalize : bool, optional
        If True, applies logarithmic normalization to the magnitude. Default is True.
    epsilon : float, optional
        Small constant added to magnitude before log to avoid log(0). Default is 1e-10.
    numorders : int, optional
        Number of orders used in normalization scaling. Default is 6.

    Returns
    -------
    tuple[NDArray, float] or NDArray
        - If `reverse` is False: Returns a tuple of (scaled_data, scalefac).
          `scaled_data` is a stacked array of magnitude and phase (or original signal
          and magnitude in hybrid mode).
        - If `reverse` is True: Returns the reconstructed time-domain signal as
          a numpy array.

    Notes
    -----
    - In forward mode, the function computes the FFT of `data`, normalizes the
      magnitude, and scales it to a range suitable for further processing.
    - In reverse mode, the function reconstructs the time-domain signal using
      inverse FFT from the provided scaled magnitude and phase components.
    - The `hybrid` mode is useful for certain types of signal visualization or
      feature extraction where both time-domain and frequency-domain information
      are needed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> x = np.random.randn(1024)
    >>> scaled_data, scalefac = filtscale(x)
    >>> reconstructed = filtscale(scaled_data, scalefac=scalefac, reverse=True)
    """
    if not reverse:
        specvals = fftpack.fft(data)
        if lognormalize:
            themag = np.log(np.absolute(specvals) + epsilon)
            scalefac = np.max(themag)
            themag = (themag - scalefac + numorders) / numorders
            themag[np.where(themag < 0.0)] = 0.0
        else:
            scalefac = np.std(data)
            themag = np.absolute(specvals) / scalefac
        thephase = np.angle(specvals)
        thephase = thephase / (2.0 * np.pi) - 0.5
        if hybrid:
            return np.stack((data, themag), axis=1), scalefac
        else:
            return np.stack((themag, thephase), axis=1), scalefac
    else:
        if hybrid:
            return data[:, 0]
        else:
            thephase = (data[:, 1] + 0.5) * 2.0 * np.pi
            if lognormalize:
                themag = np.exp(data[:, 0] * numorders - numorders + scalefac)
            else:
                themag = data[:, 0] * scalefac
            specvals = themag * np.exp(1.0j * thephase)
            return fftpack.ifft(specvals).real


def tobadpts(name: str) -> str:
    """
    Convert a filename to its corresponding bad points filename.

    This function takes a filename string and replaces the '.txt' extension
    with '_badpts.txt' to create a new filename for bad points data.

    Parameters
    ----------
    name : str
        The input filename string, typically ending with '.txt'.

    Returns
    -------
    str
        The converted filename with '_badpts.txt' extension instead of '.txt'.

    Notes
    -----
    This function is useful for creating consistent naming conventions for
    bad points data files that correspond to original data files.

    Examples
    --------
    >>> tobadpts("data.txt")
    'data_badpts.txt'

    >>> tobadpts("results.txt")
    'results_badpts.txt'

    >>> tobadpts("output.txt")
    'output_badpts.txt'
    """
    return name.replace(".txt", "_badpts.txt")


def targettoinput(name: str, targetfrag: str = "xyz", inputfrag: str = "abc") -> str:
    """
    Replace target fragment with input fragment in a string.

    Parameters
    ----------
    name : str
        The input string to perform replacement on.
    targetfrag : str, default='xyz'
        The fragment to be replaced in the input string.
    inputfrag : str, default='abc'
        The fragment to replace the target fragment with.

    Returns
    -------
    str
        The modified string with targetfrag replaced by inputfrag.

    Notes
    -----
    This function uses Python's built-in string replace method, which replaces
    all occurrences of the target fragment with the input fragment.

    Examples
    --------
    >>> targettoinput("hello xyz world")
    'hello abc world'

    >>> targettoinput("test xyz xyz test", "xyz", "123")
    'test 123 123 test'

    >>> targettoinput("abcdef", "cde", "XXX")
    'abXXXf'
    """
    LGR.debug(f"replacing {targetfrag} with {inputfrag}")
    return name.replace(targetfrag, inputfrag)


def getmatchedtcs(
    searchstring: str,
    usebadpts: bool = False,
    targetfrag: str = "xyz",
    inputfrag: str = "abc",
    debug: bool = False,
) -> tuple[list[str], int]:
    """
    Find and validate matched timecourse files based on a search pattern.

    This function searches for timecourse files matching the given search string,
    verifies their completeness by checking for associated info files, and
    determines the length of the timecourses from the first valid file.

    Parameters
    ----------
    searchstring : str
        A glob pattern to match target timecourse files.
    usebadpts : bool, optional
        Flag indicating whether bad points should be used (default is False).
    targetfrag : str, optional
        Target fragment identifier (default is "xyz").
    inputfrag : str, optional
        Input fragment identifier (default is "abc").
    debug : bool, optional
        If True, prints debug information including matched files (default is False).

    Returns
    -------
    tuple[list[str], int]
        A tuple containing:
        - List of matched and validated file paths.
        - Length of the timecourses (number of timepoints).

    Notes
    -----
    The function expects timecourse files to have a corresponding info file
    with the same base name but with "_info" appended. Only files with complete
    info files are considered valid.

    Examples
    --------
    >>> matched_files, tc_length = getmatchedtcs("data/*cardiac*.tsv")
    >>> print(f"Found {len(matched_files)} files with {tc_length} timepoints")
    """
    # list all the target files
    fromfile = sorted(glob.glob(searchstring))
    if debug:
        print(f"searchstring: {searchstring} -> {fromfile}")

    # make sure all timecourses exist
    # we need cardiacfromfmri_25.0Hz as x, normpleth as y, and perhaps badpts
    matchedfilelist = []
    for targetname in fromfile:
        infofile = targetname.replace("_desc-stdrescardfromfmri_timeseries", "_info")
        if os.path.isfile(infofile):
            matchedfilelist.append(targetname)
            print(f"{targetname} is complete")
            LGR.debug(matchedfilelist[-1])
        else:
            print(f"{targetname} is incomplete")
    print(f"found {len(matchedfilelist)} matched files")

    # find out how long the files are
    (
        samplerate,
        starttime,
        columns,
        inputarray,
        compression,
        columnsource,
    ) = tide_io.readbidstsv(
        matchedfilelist[0],
        colspec="cardiacfromfmri_25.0Hz,normpleth,badpts",
    )
    print(f"{inputarray.shape=}")
    tclen = inputarray.shape[1]
    LGR.info(f"tclen set to {tclen}")
    return matchedfilelist, tclen


def readindata(
    matchedfilelist: list[str],
    tclen: int,
    targetfrag: str = "xyz",
    inputfrag: str = "abc",
    usebadpts: bool = False,
    startskip: int = 0,
    endskip: int = 0,
    corrthresh: float = 0.5,
    readlim: int | None = None,
    readskip: int | None = None,
    debug: bool = False,
) -> tuple[NDArray, NDArray, list[str], NDArray | None]:
    """
    Read and process time-series data from a list of matched files.

    This function reads cardiac and plethysmographic time-series data from a list of
    files, performs quality checks, and returns the data in arrays suitable for
    training or analysis. It supports filtering based on correlation thresholds,
    NaN values, and signal standard deviations, and allows for optional skipping
    of data at the start and end of each time series.

    Parameters
    ----------
    matchedfilelist : list of str
        List of file paths to be processed. Each file should contain time-series data
        in a format compatible with `tide_io.readbidstsv`.
    tclen : int
        Length of the time series to be read from each file.
    targetfrag : str, optional
        Fragment identifier for target files, used in naming conversions. Default is "xyz".
    inputfrag : str, optional
        Fragment identifier for input files, used in naming conversions. Default is "abc".
    usebadpts : bool, optional
        If True, include a third array with bad point indicators. Default is False.
    startskip : int, optional
        Number of samples to skip at the beginning of each time series. Default is 0.
    endskip : int, optional
        Number of samples to skip at the end of each time series. Default is 0.
    corrthresh : float, optional
        Minimum correlation threshold between raw and plethysmographic signals.
        Files with lower correlation are excluded. Default is 0.5.
    readlim : int, optional
        Maximum number of files to read. If None, all files are read. Default is None.
    readskip : int, optional
        Number of files to skip at the beginning of the file list. If None, no files are skipped. Default is None.
    debug : bool, optional
        If True, print debug information for each file. Default is False.

    Returns
    -------
    tuple of (NDArray, NDArray, list[str]) or (NDArray, NDArray, list[str], NDArray)
        - `x1`: Array of shape `(tclen, count)` containing x-time series data.
        - `y1`: Array of shape `(tclen, count)` containing y-time series data.
        - `names`: List of file names that passed quality checks.
        - `bad1`: Optional array of shape `(tclen, count)` with bad point indicators if `usebadpts=True`.  Otherwise None.

    Notes
    -----
    - Files with NaNs, short data, extreme standard deviations, or low correlation are excluded.
    - The function logs information about excluded files for debugging and quality control.
    - The `startskip` and `endskip` parameters are applied after filtering and before returning the data.

    Examples
    --------
    >>> x, y, names, bad = readindata(filelist, tclen=1000)
    """
    LGR.info(
        "readindata called with usebadpts, startskip, endskip, readlim, readskip, targetfrag, inputfrag = "
        f"{usebadpts} {startskip} {endskip} {readlim} {readskip} {targetfrag} {inputfrag}"
    )
    # allocate target arrays
    LGR.info("allocating arrays")
    s = len(matchedfilelist[readskip:])
    if readlim is not None:
        if s > readlim:
            LGR.info(f"trimming read list to {readlim} from {s}")
            s = readlim
    x1 = np.zeros((tclen, s))
    y1 = np.zeros((tclen, s))
    names = []
    if usebadpts:
        bad1 = np.zeros((tclen, s))

    # now read the data in
    count = 0
    badcount = 0
    LGR.info("checking data")
    lowcorrfiles = []
    nanfiles = []
    shortfiles = []
    strangemagfiles = []
    for i in range(readskip, readskip + s):
        lowcorrfound = False
        nanfound = False
        LGR.info(f"processing {matchedfilelist[i]}")

        # read the info dict first
        infodict = tide_io.readdictfromjson(
            matchedfilelist[i].replace("_desc-stdrescardfromfmri_timeseries", "_info")
        )
        if infodict["corrcoeff_raw2pleth"] < corrthresh:
            lowcorrfound = True
            lowcorrfiles.append(matchedfilelist[i])
        thecolspec = "cardiacfromfmri_25.0Hz,normpleth"
        if usebadpts:
            thecolspec = thecolspec + ",badpts"
        (
            samplerate,
            starttime,
            columns,
            inputarray,
            compression,
            columnsource,
        ) = tide_io.readbidstsv(
            matchedfilelist[i],
            colspec=thecolspec,
        )
        tempy = inputarray[1, :]
        tempx = inputarray[0, :]

        if np.any(np.isnan(tempy)):
            LGR.info(f"NaN found in file {matchedfilelist[i]} - discarding")
            nanfound = True
            nanfiles.append(matchedfilelist[i])
        if np.any(np.isnan(tempx)):
            nan_fname = targettoinput(
                matchedfilelist[i], targetfrag=targetfrag, inputfrag=inputfrag
            )
            LGR.info(f"NaN found in file {nan_fname} - discarding")
            nanfound = True
            nanfiles.append(nan_fname)
        strangefound = False
        if not (0.5 < np.std(tempx) < 20.0):
            strange_fname = matchedfilelist[i]
            LGR.info(
                f"file {strange_fname} has an extreme cardiacfromfmri standard deviation - discarding"
            )
            strangefound = True
            strangemagfiles.append(strange_fname)
        if not (0.5 < np.std(tempy) < 20.0):
            LGR.info(
                f"file {matchedfilelist[i]} has an extreme normpleth standard deviation - discarding"
            )
            strangefound = True
            strangemagfiles.append(matchedfilelist[i])
        shortfound = False
        ntempx = tempx.shape[0]
        ntempy = tempy.shape[0]
        if ntempx < tclen:
            short_fname = matchedfilelist[i]
            LGR.info(f"file {short_fname} is short - discarding")
            shortfound = True
            shortfiles.append(short_fname)
        if ntempy < tclen:
            LGR.info(f"file {matchedfilelist[i]} is short - discarding")
            shortfound = True
            shortfiles.append(matchedfilelist[i])
        if (
            (ntempx >= tclen)
            and (ntempy >= tclen)
            and (not nanfound)
            and (not shortfound)
            and (not strangefound)
            and (not lowcorrfound)
        ):
            x1[:tclen, count] = tempx[:tclen]
            y1[:tclen, count] = tempy[:tclen]
            names.append(matchedfilelist[i])
            if debug:
                print(f"{matchedfilelist[i]} included:")
            if usebadpts:
                bad1[:tclen, count] = tempx[:tclen]
            count += 1
        else:
            print(f"{matchedfilelist[i]} excluded:")
            badcount += 1
            if ntempx < tclen:
                print("\tx data too short")
            if ntempy < tclen:
                print("\ty data too short")
            print(f"\t{nanfound=}")
            print(f"\t{shortfound=}")
            print(f"\t{strangefound=}")
            print(f"\t{lowcorrfound=}")
    LGR.info(f"{count} runs pass file length check")
    if len(lowcorrfiles) > 0:
        LGR.info("files with low raw/pleth correlations:")
        for thefile in lowcorrfiles:
            LGR.info(f"\t{thefile}")
    if len(nanfiles) > 0:
        LGR.info("files with NaNs:")
        for thefile in nanfiles:
            LGR.info(f"\t{thefile}")
    if len(shortfiles) > 0:
        LGR.info("short files:")
        for thefile in shortfiles:
            LGR.info(f"\t{thefile}")
    if len(strangemagfiles) > 0:
        LGR.info("files with extreme standard deviations:")
        for thefile in strangemagfiles:
            LGR.info(f"\t{thefile}")

    print(f"training set contains {count} runs of length {tclen}")
    print(f"{badcount} runs were excluded")
    if usebadpts:
        item4 = bad1[startskip:-endskip, :count]
    else:
        item4 = None
    return (
        x1[startskip:-endskip, :count],
        y1[startskip:-endskip, :count],
        names[:count],
        item4,
    )

def prep(
    window_size: int,
    step: int = 1,
    excludethresh: float = 4.0,
    usebadpts: bool = False,
    startskip: int = 200,
    endskip: int = 200,
    excludebysubject: bool = True,
    thedatadir: str = "/data/frederic/physioconn/output_2025",
    inputfrag: str = "abc",
    targetfrag: str = "xyz",
    corrthresh: float = 0.5,
    dofft: bool = False,
    readlim: int | None = None,
    readskip: int | None = None,
    countlim: int | None = None,
    debug: bool = False,
) -> (
    tuple[NDArray, NDArray, NDArray, NDArray, int, int, int]
    | tuple[NDArray, NDArray, NDArray, NDArray, int, int, int, NDArray, NDArray]
):
    """
    Prepare time-series data for training and validation by reading, normalizing,
    windowing, and splitting into batches.

    This function reads physiological time-series data from JSON files, normalizes
    the data, and organizes it into overlapping windows for model training and
    validation. It supports filtering by subject or by window, and can optionally
    apply FFT transformations to the data.

    Parameters
    ----------
    window_size : int
        Size of the sliding window used to segment time series data.
    step : int, optional
        Step size for sliding window (default is 1).
    excludethresh : float, optional
        Threshold for excluding data points based on maximum absolute value
        (default is 4.0).
    usebadpts : bool, optional
        If True, includes bad points in the data processing (default is False).
    startskip : int, optional
        Number of time points to skip at the beginning of each time series
        (default is 200).
    endskip : int, optional
        Number of time points to skip at the end of each time series
        (default is 200).
    excludebysubject : bool, optional
        If True, exclude subjects with any region exceeding `excludethresh`;
        otherwise, exclude windows (default is True).
    thesuffix : str, optional
        Suffix used in file search pattern (default is "sliceres").
    thedatadir : str, optional
        Directory path where the data files are stored (default is
        "/data/frederic/physioconn/output_2025").
    inputfrag : str, optional
        Fragment identifier for input data (default is "abc").
    targetfrag : str, optional
        Fragment identifier for target data (default is "xyz").
    corrthresh : float, optional
        Correlation threshold for data filtering (default is 0.5).
    dofft : bool, optional
        If True, apply FFT transformation to the data (default is False).
    readlim : int, optional
        Limit on number of time points to read (default is None).
    readskip : int, optional
        Number of time points to skip when reading data (default is None).
    countlim : int, optional
        Maximum number of subjects to include (default is None).
    debug : bool, optional
        If True, enable debug logging (default is False).

    Returns
    -------
    tuple of (NDArray, NDArray, NDArray, NDArray, int, int, int)
        If `dofft` is False:
            - train_x : Training input data (shape: [n_windows, window_size, 1])
            - train_y : Training target data (shape: [n_windows, window_size, 1])
            - val_x : Validation input data (shape: [n_windows, window_size, 1])
            - val_y : Validation target data (shape: [n_windows, window_size, 1])
            - N_subjs : Number of subjects
            - tclen : Total time points after skipping
            - batchsize : Number of windows per subject

        tuple of (NDArray, NDArray, NDArray, NDArray, int, int, int, NDArray, NDArray)
        If `dofft` is True:
            - train_x : Training input data (shape: [n_windows, window_size, 2])
            - train_y : Training target data (shape: [n_windows, window_size, 2])
            - val_x : Validation input data (shape: [n_windows, window_size, 2])
            - val_y : Validation target data (shape: [n_windows, window_size, 2])
            - N_subjs : Number of subjects
            - tclen : Total time points after skipping
            - batchsize : Number of windows per subject
            - Xscale_fourier : Fourier scaling for input data
            - Yscale_fourier : Fourier scaling for target data

    Notes
    -----
    - Data normalization is performed using median absolute deviation (MAD).
    - The function supports both window-based and subject-based exclusion strategies.
    - If `usebadpts` is True, bad points are included in the output arrays.
    - FFT transformations are applied using a helper function `filtscale`.

    Examples
    --------
    >>> train_x, train_y, val_x, val_y, N_subjs, tclen, batchsize = prep(
    ...     window_size=100,
    ...     step=10,
    ...     excludethresh=3.0,
    ...     excludebysubject=True,
    ...     dofft=False
    ... )
    """
    searchstring = os.path.join(thedatadir, "*", "*_desc-stdrescardfromfmri_timeseries.json")

    # find matched files
    matchedfilelist, tclen = getmatchedtcs(
        searchstring,
        usebadpts=usebadpts,
        targetfrag=targetfrag,
        inputfrag=inputfrag,
        debug=debug,
    )
    # print("matchedfilelist", matchedfilelist)
    print("tclen", tclen)

    # read in the data from the matched files
    print("about to read in data")
    x, y, names, bad = readindata(
        matchedfilelist,
        tclen,
        corrthresh=corrthresh,
        targetfrag=targetfrag,
        inputfrag=inputfrag,
        usebadpts=usebadpts,
        startskip=startskip,
        endskip=endskip,
        readlim=readlim,
        readskip=readskip,
    )

    print("finished reading in data")
    LGR.info(f"xshape, yshape: {x.shape} {y.shape}")

    # normalize input and output data
    LGR.info("normalizing data")
    LGR.info(f"count: {x.shape[1]}")
    if LGR.getEffectiveLevel() <= logging.DEBUG:
        # Only take these steps if the logger is set to DEBUG.
        for thesubj in range(x.shape[1]):
            LGR.debug(
                f"prenorm sub {thesubj} min, max, mean, std, MAD x, y: "
                f"{thesubj} "
                f"{np.min(x[:, thesubj])} {np.max(x[:, thesubj])} {np.mean(x[:, thesubj])} "
                f"{np.std(x[:, thesubj])} {mad(x[:, thesubj])} {np.min(y[:, thesubj])} "
                f"{np.max(y[:, thesubj])} {np.mean(y[:, thesubj])} {np.std(x[:, thesubj])} "
                f"{mad(y[:, thesubj])}"
            )

    y -= np.mean(y, axis=0)
    themad = mad(y, axis=0)
    for thesubj in range(themad.shape[0]):
        if themad[thesubj] > 0.0:
            y[:, thesubj] /= themad[thesubj]

    x -= np.mean(x, axis=0)
    themad = mad(x, axis=0)
    for thesubj in range(themad.shape[0]):
        if themad[thesubj] > 0.0:
            x[:, thesubj] /= themad[thesubj]

        if LGR.getEffectiveLevel() <= logging.DEBUG:
            # Only take these steps if the logger is set to DEBUG.
            for thesubj in range(x.shape[1]):
                LGR.debug(
                    f"postnorm sub {thesubj} min, max, mean, std, MAD x, y: "
                    f"{thesubj} "
                    f"{np.min(x[:, thesubj])} {np.max(x[:, thesubj])} {np.mean(x[:, thesubj])} "
                    f"{np.std(x[:, thesubj])} {mad(x[:, thesubj])} {np.min(y[:, thesubj])} "
                    f"{np.max(y[:, thesubj])} {np.mean(y[:, thesubj])} {np.std(x[:, thesubj])} "
                    f"{mad(y[:, thesubj])}"
                )

    # now decide what to keep and what to exclude
    thefabs = np.fabs(x)
    if not excludebysubject:
        N_pts = x.shape[0]
        N_subjs = x.shape[1]
        windowspersubject = np.int64((N_pts - window_size - 1) // step)
        LGR.info(
            f"{N_subjs} subjects with {N_pts} points will be evaluated with "
            f"{windowspersubject} windows per subject with step {step}"
        )
        usewindow = np.zeros(N_subjs * windowspersubject, dtype=np.int64)
        subjectstarts = np.zeros(N_subjs, dtype=np.int64)
        # check each window
        numgoodwindows = 0
        LGR.info("checking windows")
        subjectnames = []
        for subj in range(N_subjs):
            subjectstarts[subj] = numgoodwindows
            subjectnames.append(names[subj])
            LGR.info(f"{names[subj]} starts at {numgoodwindows}")
            for windownumber in range(windowspersubject):
                if (
                    np.max(
                        thefabs[
                            step * windownumber : (step * windownumber + window_size),
                            subj,
                        ]
                    )
                    <= excludethresh
                ):
                    usewindow[subj * windowspersubject + windownumber] = 1
                    numgoodwindows += 1
        LGR.info(
            f"found {numgoodwindows} out of a potential {N_subjs * windowspersubject} "
            f"({100.0 * numgoodwindows / (N_subjs * windowspersubject)}%)"
        )

        for subj in range(N_subjs):
            LGR.info(f"{names[subj]} starts at {subjectstarts[subj]}")

        LGR.info("copying data into windows")
        if usebadpts:
            Xb = np.zeros((numgoodwindows, window_size, 2))
        else:
            Xb = np.zeros((numgoodwindows, window_size, 1))
        Yb = np.zeros((numgoodwindows, window_size, 1))
        LGR.info(f"dimensions of Xb: {Xb.shape}")
        thiswindow = 0
        for subj in range(N_subjs):
            for windownumber in range(windowspersubject):
                if usewindow[subj * windowspersubject + windownumber] == 1:
                    if usebadpts:
                        Xb[thiswindow, :, 1] = bad[
                            step * windownumber : (step * windownumber + window_size),
                            subj,
                        ]
                    Xb[thiswindow, :, 0] = x[
                        step * windownumber : (step * windownumber + window_size), subj
                    ]
                    Yb[thiswindow, :, 0] = y[
                        step * windownumber : (step * windownumber + window_size), subj
                    ]
                    thiswindow += 1

    else:
        # now check for subjects that have regions that exceed the target
        themax = np.max(thefabs, axis=0)

        cleansubjs = np.where(themax < excludethresh)[0]

        totalcount = x.shape[1] + 0
        cleancount = len(cleansubjs)
        if countlim is not None:
            if cleancount > countlim:
                LGR.info(f"reducing count to {countlim} from {cleancount}")
                cleansubjs = cleansubjs[:countlim]

        x = x[:, cleansubjs]
        y = y[:, cleansubjs]
        cleannames = []
        for theindex in cleansubjs:
            cleannames.append(names[theindex])
        if usebadpts:
            bad = bad[:, cleansubjs]
        subjectnames = cleannames

        LGR.info(f"after filtering, shape of x is {x.shape}")

        N_pts = y.shape[0]
        N_subjs = y.shape[1]

        X = np.zeros((1, N_pts, N_subjs))
        Y = np.zeros((1, N_pts, N_subjs))
        if usebadpts:
            BAD = np.zeros((1, N_pts, N_subjs))

        X[0, :, :] = x
        Y[0, :, :] = y
        if usebadpts:
            BAD[0, :, :] = bad

        windowspersubject = int((N_pts - window_size - 1) // step)
        LGR.info(
            f"found {windowspersubject * cleancount} out of a potential "
            f"{windowspersubject * totalcount} "
            f"({100.0 * cleancount / totalcount}%)"
        )
        LGR.info(f"{windowspersubject} {cleancount} {totalcount}")

        if usebadpts:
            Xb = np.zeros((N_subjs * windowspersubject, window_size, 2))
        else:
            Xb = np.zeros((N_subjs * windowspersubject, window_size, 1))
        LGR.info(f"dimensions of Xb: {Xb.shape}")
        for j in range(N_subjs):
            LGR.info(
                f"sub {j} ({cleannames[j]}) min, max X, Y: "
                f"{j} {np.min(X[0, :, j])} {np.max(X[0, :, j])} {np.min(Y[0, :, j])} "
                f"{np.max(Y[0, :, j])}"
            )
            for i in range(windowspersubject):
                Xb[j * windowspersubject + i, :, 0] = X[0, step * i : (step * i + window_size), j]
                Xb[j * windowspersubject + i, :, 1] = BAD[
                    0, step * i : (step * i + window_size), j
                ]

        Yb = np.zeros((N_subjs * windowspersubject, window_size, 1))
        LGR.info(f"dimensions of Yb: {Yb.shape}")
        for j in range(N_subjs):
            for i in range(windowspersubject):
                Yb[j * windowspersubject + i, :, 0] = Y[0, step * i : (step * i + window_size), j]

        subjectstarts = [i * windowspersubject for i in range(N_subjs)]
        for subj in range(N_subjs):
            LGR.info(f"{names[subj]} starts at {subjectstarts[subj]}")

    print(f"Xb.shape: {Xb.shape}")
    print(f"Yb.shape: {Yb.shape}")
    print(f"{usebadpts=}, {dofft=}")
    if dofft:
        Xb_fourier = np.zeros((N_subjs * windowspersubject, window_size, 2))
        LGR.info(f"dimensions of Xb_fourier: {Xb_fourier.shape}")
        Xscale_fourier = np.zeros((N_subjs, windowspersubject))
        LGR.info(f"dimensions of Xscale_fourier: {Xscale_fourier.shape}")
        Yb_fourier = np.zeros((N_subjs * windowspersubject, window_size, 2))
        LGR.info(f"dimensions of Yb_fourier: {Yb_fourier.shape}")
        Yscale_fourier = np.zeros((N_subjs, windowspersubject))
        LGR.info(f"dimensions of Yscale_fourier: {Yscale_fourier.shape}")
        for j in range(N_subjs):
            LGR.info(f"transforming subject {j}")
            for i in range((N_pts - window_size - 1)):
                (
                    Xb_fourier[j * windowspersubject + i, :, :],
                    Xscale_fourier[j, i],
                ) = filtscale(X[0, step * i : (step * i + window_size), j])
                (
                    Yb_fourier[j * windowspersubject + i, :, :],
                    Yscale_fourier[j, i],
                ) = filtscale(Y[0, step * i : (step * i + window_size), j])

    limit = np.int64(0.8 * Xb.shape[0])
    LGR.info(f"limit: {limit} out of {len(subjectstarts)}")
    # find nearest subject start
    firstvalsubject = np.abs(subjectstarts - limit).argmin()
    LGR.info(f"firstvalsubject: {firstvalsubject}")
    perm_train = np.random.permutation(np.int64(np.arange(subjectstarts[firstvalsubject])))
    perm_val = np.random.permutation(
        np.int64(np.arange(subjectstarts[firstvalsubject], Xb.shape[0]))
    )

    LGR.info("training subjects:")
    for i in range(0, firstvalsubject):
        LGR.info(f"\t{i} {subjectnames[i]}")
    LGR.info("validation subjects:")
    for i in range(firstvalsubject, len(subjectstarts)):
        LGR.info(f"\t{i} {subjectnames[i]}")

    perm = range(Xb.shape[0])

    batchsize = windowspersubject

    if dofft:
        train_x = Xb_fourier[perm[:limit], :, :]
        train_y = Yb_fourier[perm[:limit], :, :]

        val_x = Xb_fourier[perm[limit:], :, :]
        val_y = Yb_fourier[perm[limit:], :, :]
        LGR.info(f"train, val dims: {train_x.shape} {train_y.shape} {val_x.shape} {val_y.shape}")
        return (
            train_x,
            train_y,
            val_x,
            val_y,
            N_subjs,
            tclen - startskip - endskip,
            batchsize,
            Xscale_fourier,
            Yscale_fourier,
        )
    else:
        train_x = Xb[perm_train, :, :]
        train_y = Yb[perm_train, :, :]

        val_x = Xb[perm_val, :, :]
        val_y = Yb[perm_val, :, :]

        LGR.info(f"train, val dims: {train_x.shape} {train_y.shape} {val_x.shape} {val_y.shape}")
        print(f"in prep: {train_x.shape=}, {train_y.shape=}, {val_x.shape}, {val_y.shape=}")

        LGR.info(f"train, val dims: {train_x.shape} {train_y.shape} {val_x.shape} {val_y.shape}")
        return (
            train_x,
            train_y,
            val_x,
            val_y,
            N_subjs,
            tclen - startskip - endskip,
            batchsize,
        )
