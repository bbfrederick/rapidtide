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
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

import tensorflow as tf
import tf_keras.backend as K
from tf_keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    TerminateOnNaN,
)
from tf_keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Convolution1D,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPool1D,
    Input,
    MaxPooling1D,
    Reshape,
    TimeDistributed,
    UpSampling1D,
)
from tf_keras.models import Model, Sequential, load_model
from tf_keras.optimizers.legacy import RMSprop

import rapidtide.io as tide_io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LGR = logging.getLogger("GENERAL")
LGR.debug("setting backend to Agg")
mpl.use("Agg")

# Disable GPU if desired
# figure out what sorts of devices we have
physical_devices = tf.config.list_physical_devices()
print(physical_devices)
# try:
#    tf.config.set_visible_devices([], "GPU")
# except Exception as e:
#    LGR.warning(f"Failed to disable GPU: {e}")

LGR.debug(f"tensorflow version: >>>{tf.__version__}<<<")


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
                thesuffix=self.thesuffix,
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
                thesuffix=self.thesuffix,
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

    @tf.function
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
        for inference tasks.

        Examples
        --------
        >>> # Assuming model is already trained
        >>> X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> predictions = model.predict_model(X_test)
        >>> print(predictions)
        """
        return self.model(X, training=False)

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

        YPred = self.predict_model(self.val_x).numpy()

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

        self.loss = self.history.history["loss"]
        self.val_loss = self.history.history["val_loss"]

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
            The instance of the class containing the metadata attributes.

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
        self.model.save(os.path.join(modelsavename, "model.keras"))

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
        1. Keras format (``model.keras``)
        2. HDF5 format (``model.h5``)

        If neither format is found, the function exits with an error message.

        The loaded model metadata is stored in ``self.infodict``, and model configuration
        is stored in ``self.config``. Additional attributes like ``window_size`` and ``usebadpts``
        are extracted from the metadata and stored as instance attributes.

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
        try:
            # load the keras format model if it exists
            self.model = load_model(os.path.join(self.modelpath, modelname, "model.keras"))
            self.config = self.model.get_config()
        except OSError:
            # load in the model with weights from hdf
            try:
                self.model = load_model(os.path.join(self.modelpath, modelname, "model.h5"))
            except OSError:
                print(f"Could not load {modelname}")
                sys.exit()

        if verbose:
            self.model.summary()

        # now load additional information
        self.infodict = tide_io.readdictfromjson(
            os.path.join(self.modelpath, modelname, "model_meta.json")
        )
        if verbose:
            print(self.infodict)
        self.window_size = self.infodict["window_size"]
        self.usebadpts = self.infodict["usebadpts"]

        # model is ready to use
        self.initialized = True
        self.trained = True
        LGR.info(f"{modelname} loaded")

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
        self.model.summary()
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
            - `model`: The Keras model to be trained.
            - `train_x`, `train_y`: Training data inputs and labels.
            - `val_x`, `val_y`: Validation data inputs and labels.
            - `modelname`: Name of the model for saving purposes.
            - `usetensorboard`: Boolean flag to enable TensorBoard logging.
            - `num_pretrain_epochs`: Number of epochs for pretraining phase.
            - `num_epochs`: Total number of training epochs.
            - `savemodel()`: Method to save the trained model.

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

        Examples
        --------
        >>> trainer = ModelTrainer(model, train_x, train_y, val_x, val_y)
        >>> trainer.train()
        """
        self.intermediatemodelpath = os.path.join(
            self.modelname, "model_e{epoch:02d}_v{val_loss:.4f}.keras"
        )
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
            .shuffle(2048)
            .batch(1024)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices((self.val_x, self.val_y)).batch(1024)
        if self.usetensorboard:
            tensorboard = TensorBoard(
                log_dir=os.path.join(self.intermediatemodelpath, "logs", str(int(time.time())))
            )
            self.model.fit(self.train_x, self.train_y, verbose=1, callbacks=[tensorboard])
        else:
            if self.num_pretrain_epochs > 0:
                LGR.info("pretraining model to reproduce input data")
                self.history = self.model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=self.num_pretrain_epochs,
                    verbose=1,
                    callbacks=[
                        TerminateOnNaN(),
                        ModelCheckpoint(self.intermediatemodelpath, save_format="keras"),
                        EarlyStopping(
                            monitor="val_loss",  # or 'val_mae', etc.
                            patience=10,  # number of epochs to wait
                            restore_best_weights=True,
                        ),
                    ],
                )
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.num_epochs,
                verbose=1,
                callbacks=[
                    TerminateOnNaN(),
                    ModelCheckpoint(self.intermediatemodelpath, save_format="keras"),
                    EarlyStopping(
                        monitor="val_loss",  # or 'val_mae', etc.
                        patience=10,  # number of epochs to wait
                        restore_best_weights=True,
                    ),
                ],
            )
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
        N_pts = len(scaleddata)
        if self.usebadpts:
            if badpts is None:
                badpts = np.zeros_like(scaleddata)
            X = np.zeros(((N_pts - self.window_size - 1), self.window_size, 2))
            for i in range(X.shape[0]):
                X[i, :, 0] = scaleddata[i : i + self.window_size]
                X[i, :, 1] = badpts[i : i + self.window_size]
        else:
            X = np.zeros(((N_pts - self.window_size - 1), self.window_size, 1))
            for i in range(X.shape[0]):
                X[i, :, 0] = scaleddata[i : i + self.window_size]

        Y = self.predict_model(X).numpy()
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


class MultiscaleCNNDLFilter(DeepLearningFilter):
    # from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D
    # from keras.models import Model
    # this base model is one branch of the main model
    # it takes a time series as an input, performs 1-D convolution, and returns it as an output ready for concatenation
    def __init__(
        self,
        num_filters: int = 10,
        kernel_sizes: list[int] = [4, 8, 12],
        input_lens: list[int] = [64, 128, 192],
        input_width: int = 1,
        dilation_rate: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the MultiscaleCNNDLFilter.

        This constructor initializes a multiscale CNN filter with specified parameters
        for processing sequential data with multiple kernel sizes and dilation rates.

        Parameters
        ----------
        num_filters : int, optional
            Number of filters to use in the convolutional layers. Default is 10.
        kernel_sizes : list of int, optional
            List of kernel sizes to use for different convolutional layers.
            Default is [4, 8, 12].
        input_lens : list of int, optional
            List of input sequence lengths to process. Default is [64, 128, 192].
        input_width : int, optional
            Width of the input data (number of input channels). Default is 1.
        dilation_rate : int, optional
            Dilation rate for the convolutional layers. Default is 1.
        *args
            Variable length argument list passed to parent class.
        **kwargs
            Arbitrary keyword arguments passed to parent class.

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Notes
        -----
        The initialized object will store network configuration information in
        `infodict` including network type, number of filters, kernel sizes, input lengths,
        and input width. The parent class initialization is called using `super()`.

        Examples
        --------
        >>> filter = MultiscaleCNNDLFilter(
        ...     num_filters=20,
        ...     kernel_sizes=[3, 6, 9],
        ...     input_lens=[32, 64, 128],
        ...     input_width=2,
        ...     dilation_rate=2
        ... )
        """
        """
            Initialize the MultiscaleCNNDLFilter.
    
            This constructor initializes a multiscale CNN filter with specified parameters
            for processing sequential data with multiple kernel sizes and dilation rates.
    
            Parameters
            ----------
            num_filters : int, optional
                Number of filters to use in the convolutional layers, default is 10
            kernel_sizes : list of int, optional
                List of kernel sizes to use for different convolutional layers, 
                default is [4, 8, 12]
            input_lens : list of int, optional
                List of input sequence lengths to process, default is [64, 128, 192]
            input_width : int, optional
                Width of the input data (number of input channels), default is 1
            dilation_rate : int, optional
                Dilation rate for the convolutional layers, default is 1
            *args
                Variable length argument list passed to parent class
            **kwargs
                Arbitrary keyword arguments passed to parent class
    
            Returns
            -------
            None
                This method initializes the object and does not return any value
    
            Notes
            -----
            The initialized object will store network configuration information in
            `infodict` including network type, number of filters, kernel sizes, input lengths,
            and input width. The parent class initialization is called using super().
    
            Examples
            --------
            >>> filter = MultiscaleCNNDLFilter(
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
        self.infodict["nettype"] = "multscalecnn"
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_sizes"] = self.kernel_sizes
        self.infodict["input_lens"] = self.input_lens
        self.infodict["input_width"] = self.input_width
        super(MultiscaleCNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return model name based on configuration parameters.

        This method constructs a descriptive model name by joining various configuration
        parameters with specific prefixes and zero-padded numbers. The generated name
        is used to create a unique model identifier and corresponding directory path.

        Parameters
        ----------
        self : object
            The instance of the class containing the following attributes:

            - window_size : int
                Size of the input window
            - num_layers : int
                Number of layers in the model
            - num_filters : int
                Number of filters in the model
            - kernel_size : int
                Size of the convolutional kernel
            - num_epochs : int
                Number of training epochs
            - excludethresh : float
                Threshold for excluding data points
            - corrthresh : float
                Correlation threshold for filtering
            - step : int
                Step size for processing
            - dilation_rate : int
                Dilation rate for convolutional layers
            - activation : str
                Activation function name
            - usebadpts : bool
                Whether to use bad points in training
            - excludebysubject : bool
                Whether to exclude data by subject
            - namesuffix : str, optional
                Additional suffix to append to the model name
            - modelroot : str
                Root directory for model storage

        Returns
        -------
        None
            This method modifies the instance attributes `modelname` and `modelpath`
            but does not return any value.

        Notes
        -----
        The generated model name follows this format:
        "model_multiscalecnn_tf2_wXxx_lYy_fnZz_flZz_eXxx_tY_ctZ_sZ_dZ_activation[_usebadpts][_excludebysubject][_suffix]"

        Where Xxx, Yy, Zz, etc. represent zero-padded numbers based on the parameter values.

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.num_layers = 5
        >>> model.getname()
        >>> print(model.modelname)
        'model_multiscalecnn_tf2_w100_l05_fn05_fl05_e001_t0.5_ct0.8_s1_d1_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "multiscalecnn",
                "tf2",
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
        if self.excludebysubject:
            self.modelname += "_excludebysubject"
        if self.namesuffix is not None:
            self.modelname += "_" + self.namesuffix
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makesubnet(self, inputlen, kernelsize):
        """
        Create a 1D convolutional neural network submodel for time series processing.

        This function constructs a neural network submodel that processes time series data
        through 1D convolutional layers followed by global max pooling and dense layers
        with dropout regularization. The model is designed for feature extraction from
        sequential data.

        Parameters
        ----------
        inputlen : int
            Length of the input time series sequence.
        kernelsize : int
            Size of the convolutional kernel for 1D convolution operation.

        Returns
        -------
        Model
            Keras Model object representing the constructed submodel with the following structure:
            Input -> Conv1D -> GlobalMaxPool1D -> Dense -> Dropout -> Dense -> Output

        Notes
        -----
        The model architecture includes:
        - 1D convolution with tanh activation
        - Global max pooling for sequence reduction
        - Two dense layers with tanh activation
        - Dropout regularization (0.3) after first dense layer
        - Uses 'same' padding for convolutional layer

        Examples
        --------
        >>> model = makesubnet(inputlen=100, kernelsize=3)
        >>> model.summary()
        """
        # the input is a time series of length input_len and width input_width
        input_seq = Input(shape=(inputlen, self.input_width))

        # 1-D convolution and global max-pooling
        convolved = Convolution1D(
            filters=self.num_filters, kernel_size=kernelsize, padding="same", activation="tanh"
        )(input_seq)
        processed = GlobalMaxPool1D()(convolved)

        # dense layer with dropout regularization
        compressed = Dense(50, activation="tanh")(processed)
        compressed = Dropout(0.3)(compressed)
        basemodel = Model(inputs=input_seq, outputs=compressed)
        return basemodel

    def makenet(self):
        """
        Create a multi-scale neural network for time series analysis.

        This function constructs a neural network model that processes time series data at
        multiple resolutions (original, medium, and small scale). The model uses separate
        sub-networks for each resolution level and concatenates their outputs to make
        predictions. The architecture leverages different kernel sizes for different
        down-sampled versions to capture features at multiple temporal scales.

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - inputs_lens : list of int
                Lengths of input sequences for small, medium, and original scales.
            - input_width : int
                Width of input features.
            - kernel_sizes : list of int
                Kernel sizes for convolutional layers corresponding to each scale.
            - makesubnet : callable
                Function to create sub-network for a given sequence length and kernel size.

        Returns
        -------
        None
            This method modifies the instance in-place by setting the `model` attribute
            to the constructed Keras model.

        Notes
        -----
        The model architecture:
        1. Takes three inputs at different resolutions.
        2. Processes each input through separate sub-networks with scale-specific kernels.
        3. Concatenates the embeddings from all branches.
        4. Applies a final dense layer with sigmoid activation for binary classification.

        Examples
        --------
        >>> # Assuming self is an instance with proper attributes set
        >>> self.makenet()
        >>> print(self.model.summary())
        """
        # the inputs to the branches are the original time series, and its down-sampled versions
        input_smallseq = Input(shape=(self.inputs_lens[0], self.input_width))
        input_medseq = Input(shape=(self.inputs_lens[1], self.input_width))
        input_origseq = Input(shape=(self.inputs_lens[2], self.input_width))

        # the more down-sampled the time series, the shorter the corresponding filter
        base_net_small = self.makesubnet(self.inputs_lens[0], self.kernel_sizes[0])
        base_net_med = self.makesubnet(self.inputs_lens[1], self.kernel_sizes[1])
        base_net_original = self.makesubnet(self.inputs_lens[2], self.kernel_sizes[2])
        embedding_small = base_net_small(input_smallseq)
        embedding_med = base_net_med(input_medseq)
        embedding_original = base_net_original(input_origseq)

        # concatenate all the outputs
        merged = Concatenate()([embedding_small, embedding_med, embedding_original])
        out = Dense(1, activation="sigmoid")(merged)
        self.model = Model(inputs=[input_smallseq, input_medseq, input_origseq], outputs=out)
        self.model.compile(optimizer=RMSprop(), loss="mse")


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
        Initialize the CNNDLFilter layer.

        Parameters
        ----------
        num_filters : int, optional
            Number of convolutional filters to use, by default 10.
        kernel_size : int, optional
            Size of the convolutional kernel, by default 5.
        dilation_rate : int, optional
            Dilation rate for the convolutional layer, by default 1.
        *args
            Variable length argument list passed to parent class.
        **kwargs
            Arbitrary keyword arguments passed to parent class.

        Returns
        -------
        None
            This method initializes the layer and does not return any value.

        Notes
        -----
        This constructor sets up a convolutional layer with specified parameters
        and registers the network type as "cnn" in the infodict. The dilation rate
        controls the spacing between kernel points, allowing for wider receptive
        fields without increasing the number of parameters.

        Examples
        --------
        >>> layer = CNNDLFilter(num_filters=32, kernel_size=3, dilation_rate=2)
        >>> print(layer.num_filters)
        32
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.infodict["nettype"] = "cnn"
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        super(CNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return the model name based on configuration parameters.

        This method constructs a descriptive model name by joining various configuration
        parameters with specific prefixes and zero-padded numbers. The resulting name
        is used to create a unique directory path for model storage.

        Parameters
        ----------
        self : object
            The instance of the class containing model configuration parameters.
            Expected attributes include:
            - window_size : int
            - num_layers : int
            - num_filters : int
            - kernel_size : int
            - num_epochs : int
            - excludethresh : float
            - corrthresh : float
            - step : int
            - dilation_rate : int
            - activation : str
            - modelroot : str
            - usebadpts : bool
            - excludebysubject : bool
            - namesuffix : str, optional

        Returns
        -------
        None
            This method does not return a value but sets the following attributes:
            - `self.modelname`: str, the constructed model name
            - `self.modelpath`: str, the full path to the model directory

        Notes
        -----
        The generated model name follows this format:
        "model_cnn_tf2_wXXX_lYY_fnZZ_flZZ_eXXX_tY_ctZ_sZ_dZ_activation[_usebadpts][_excludebysubject][_suffix]"

        Where:
        - XXX: window_size (3-digit zero-padded)
        - YY: num_layers (2-digit zero-padded)
        - ZZ: num_filters (2-digit zero-padded)
        - ZZ: kernel_size (2-digit zero-padded)
        - XXX: num_epochs (3-digit zero-padded)
        - Y: excludethresh (single digit)
        - Z: corrthresh (single digit)
        - Z: step (single digit)
        - Z: dilation_rate (single digit)

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.num_layers = 5
        >>> model.num_filters = 32
        >>> model.kernel_size = 3
        >>> model.num_epochs = 1000
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 1
        >>> model.dilation_rate = 2
        >>> model.activation = "relu"
        >>> model.modelroot = "./models"
        >>> model.getname()
        >>> print(model.modelname)
        'model_cnn_tf2_w100_l05_fn32_fl03_e1000_t05_ct08_s1_d2_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "cnn",
                "tf2",
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
        Create and configure a 1D convolutional neural network model.

        This method builds a sequential CNN architecture with multiple convolutional layers,
        batch normalization, dropout regularization, and ReLU activation functions. The network
        is designed for sequence-to-sequence mapping with skip connections.

        Parameters
        ----------
        self : object
            The instance of the class containing the model configuration parameters.
            Expected attributes include:
            - num_filters : int
                Number of convolutional filters in each layer.
            - kernel_size : int
                Size of the convolutional kernel.
            - inputsize : int
                Size of the input sequence.
            - num_layers : int
                Total number of layers in the network.
            - dilation_rate : int
                Dilation rate for dilated convolutions.
            - dropout_rate : float
                Dropout rate for regularization.
            - activation : str or callable
                Activation function to use.

        Returns
        -------
        None
            This method modifies the instance's model attribute in-place and does not return anything.

        Notes
        -----
        The network architecture follows this pattern:
        - Input layer with Conv1D, BatchNormalization, Dropout, and Activation
        - Intermediate layers with Conv1D, BatchNormalization, Dropout, and Activation
        - Output layer with Conv1D matching the input size
        - Model compiled with RMSprop optimizer and MSE loss

        Examples
        --------
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.num_filters = 64
        ...         self.kernel_size = 3
        ...         self.inputsize = 100
        ...         self.num_layers = 5
        ...         self.dilation_rate = 2
        ...         self.dropout_rate = 0.2
        ...         self.activation = 'relu'
        ...         self.model = None
        ...
        >>> model = MyModel()
        >>> model.makenet()
        >>> print(model.model.summary())
        """
        self.model = Sequential()

        # make the input layer
        self.model.add(
            Convolution1D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                padding="same",
                input_shape=(None, self.inputsize),
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Activation(self.activation))

        # make the intermediate layers
        for layer in range(self.num_layers - 2):
            self.model.add(
                Convolution1D(
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    dilation_rate=self.dilation_rate,
                    padding="same",
                )
            )
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=self.dropout_rate))
            self.model.add(Activation(self.activation))

        # make the output layer
        self.model.add(
            Convolution1D(filters=self.inputsize, kernel_size=self.kernel_size, padding="same")
        )
        self.model.compile(optimizer=RMSprop(), loss="mse")


class DenseAutoencoderDLFilter(DeepLearningFilter):
    def __init__(self, encoding_dim: int = 10, *args, **kwargs) -> None:
        """
        Initialize the DenseAutoencoderDLFilter.

        Parameters
        ----------
        encoding_dim : int, default=10
            The dimensionality of the encoding layer in the autoencoder.
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
        2. Updating the infodict with network type and encoding dimension information
        3. Calling the parent class constructor with passed arguments

        Examples
        --------
        >>> filter = DenseAutoencoderDLFilter(encoding_dim=20)
        >>> print(filter.encoding_dim)
        20
        """
        self.encoding_dim = encoding_dim
        self.infodict["nettype"] = "autoencoder"
        self.infodict["encoding_dim"] = self.encoding_dim
        super(DenseAutoencoderDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return model name based on configuration parameters.

        This method constructs a descriptive model name by joining various configuration
        parameters with specific prefixes and formatting conventions. The generated name
        is used to create a unique identifier for the model and its corresponding directory
        path.

        Parameters
        ----------
        self : object
            The instance of the class containing the model configuration attributes.
            Expected attributes include:
            - `window_size`: int, size of the window
            - `encoding_dim`: int, dimension of the encoding layer
            - `num_epochs`: int, number of training epochs
            - `excludethresh`: float, threshold for exclusion
            - `corrthresh`: float, correlation threshold
            - `step`: int, step size
            - `activation`: str, activation function name
            - `modelroot`: str, root directory for models
            - `usebadpts`: bool, flag to include bad points
            - `excludebysubject`: bool, flag to exclude by subject
            - `namesuffix`: str, optional suffix to append to the model name

        Returns
        -------
        None
            This method does not return a value but sets the following attributes:
            - `self.modelname`: str, the generated model name
            - `self.modelpath`: str, the full path to the model directory

        Notes
        -----
        The model name is constructed using the following components:
        - "model_denseautoencoder_tf2" as base identifier
        - Window size with 3-digit zero-padded formatting (wXXX)
        - Encoding dimension with 3-digit zero-padded formatting (enXXX)
        - Number of epochs with 3-digit zero-padded formatting (eXXX)
        - Exclusion threshold (tX.XX)
        - Correlation threshold (ctX.XX)
        - Step size (sX)
        - Activation function name
        - Optional suffixes based on configuration flags

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
        >>> model.modelroot = '/models'
        >>> model.getname()
        >>> print(model.modelname)
        'model_denseautoencoder_tf2_w100_en050_e1000_t0.5_ct00.8_s10_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "denseautoencoder",
                "tf2",
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
        Create and compile a neural network model for autoencoding.

        This function constructs a symmetric autoencoder architecture with configurable
        number of layers, encoding dimension, and activation function. The network
        consists of an input layer, multiple encoding layers, an encoding layer,
        decoding layers, and an output layer. Batch normalization, dropout, and
        activation functions are applied at each layer as specified by the instance
        attributes.

        Parameters
        ----------
        self : object
            The instance of the class containing the following attributes:
            - num_layers : int
                Number of layers in the network.
            - encoding_dim : int
                Dimension of the encoding layer.
            - inputsize : int
                Size of the input data.
            - dropout_rate : float
                Dropout rate applied to all layers.
            - activation : str or callable
                Activation function used in all layers.
            - model : keras.Sequential
                The constructed Keras model (assigned within the function).

        Returns
        -------
        None
            This function does not return a value but assigns the constructed model
            to the instance attribute `self.model`.

        Notes
        -----
        - The network architecture is symmetric, with the number of units in each
          layer following a pattern that doubles or halves based on the layer index.
        - The model is compiled with the RMSprop optimizer and mean squared error loss.

        Examples
        --------
        >>> autoencoder = AutoEncoder()
        >>> autoencoder.num_layers = 4
        >>> autoencoder.encoding_dim = 32
        >>> autoencoder.inputsize = 784
        >>> autoencoder.dropout_rate = 0.2
        >>> autoencoder.activation = 'relu'
        >>> autoencoder.makenet()
        >>> autoencoder.model.summary()
        """
        self.model = Sequential()

        # make the input layer
        sizefac = 2
        for i in range(1, self.num_layers - 1):
            sizefac = int(sizefac * 2)
        LGR.info(f"input layer - sizefac: {sizefac}")

        self.model.add(Dense(sizefac * self.encoding_dim, input_shape=(None, self.inputsize)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Activation(self.activation))

        # make the intermediate encoding layers
        for i in range(1, self.num_layers - 1):
            sizefac = int(sizefac // 2)
            LGR.info(f"encoder layer {i + 1}, sizefac: {sizefac}")
            self.model.add(Dense(sizefac * self.encoding_dim))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=self.dropout_rate))
            self.model.add(Activation(self.activation))

        # make the encoding layer
        sizefac = int(sizefac // 2)
        LGR.info(f"encoding layer - sizefac: {sizefac}")
        self.model.add(Dense(self.encoding_dim))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Activation(self.activation))

        # make the intermediate decoding layers
        for i in range(1, self.num_layers):
            sizefac = int(sizefac * 2)
            LGR.info(f"decoding layer {i}, sizefac: {sizefac}")
            self.model.add(Dense(sizefac * self.encoding_dim))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=self.dropout_rate))
            self.model.add(Activation(self.activation))

        # make the output layer
        self.model.add(Dense(self.inputsize))
        self.model.compile(optimizer=RMSprop(), loss="mse")


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
        Initialize ConvAutoencoderDLFilter.

        Parameters
        ----------
        encoding_dim : int, optional
            Dimension of the encoding layer, by default 10.
        num_filters : int, optional
            Number of filters in the convolutional layers, by default 5.
        kernel_size : int, optional
            Size of the convolutional kernel, by default 5.
        dilation_rate : int, optional
            Dilation rate for the convolutional layers, by default 1.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This constructor initializes the ConvAutoencoderDLFilter with specified
        convolutional parameters and sets up the infodict with network metadata.

        Examples
        --------
        >>> model = ConvAutoencoderDLFilter(
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
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        self.infodict["nettype"] = "autoencoder"
        self.infodict["encoding_dim"] = self.encoding_dim
        super(ConvAutoencoderDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure the model name and path based on current configuration parameters.

        This method constructs a descriptive model name string using various configuration
        parameters and creates the corresponding directory path for model storage. The generated
        name includes information about window size, encoding dimensions, filter settings,
        training parameters, and optional flags.

        Parameters
        ----------
        self : object
            The instance containing configuration parameters for model naming.
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
            - namesuffix : str or None
            - modelroot : str

        Returns
        -------
        None
            This method modifies the instance attributes in-place and does not return a value.
            Modifies the following instance attributes:
            - modelname : str
            - modelpath : str

        Notes
        -----
        The generated model name follows a specific naming convention:
        "model_convautoencoder_tf2_wXXX_enYYY_fnZZ_flZZ_eXXX_tY_ctZ_sZ_activation"
        where XXX, YYY, ZZ, and Z represent zero-padded numerical values.

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.encoding_dim = 50
        >>> model.num_filters = 32
        >>> model.kernel_size = 5
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
        'model_convautoencoder_tf2_w100_en050_fn32_fl05_e1000_t0.5_ct0.8_s10_relu_usebadpts_test'
        """
        self.modelname = "_".join(
            [
                "model",
                "convautoencoder",
                "tf2",
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
        Build and compile a 1D convolutional autoencoder model.

        This function constructs a symmetric encoder-decoder architecture using 1D convolutions,
        batch normalization, dropout, and pooling layers. The model is designed for sequence
        processing tasks such as time-series or signal denoising.

        Parameters
        ----------
        self : object
            Instance of the class containing the following attributes:
            - window_size : int
                The length of the input sequence.
            - inputsize : int
                The number of input features at each time step.
            - num_filters : int
                Initial number of filters in the first convolutional layer.
            - kernel_size : int
                Size of the convolutional kernel.
            - dropout_rate : float
                Dropout rate applied after batch normalization.
            - activation : str or callable
                Activation function used in layers.
            - encoding_dim : int
                Dimensionality of the encoded representation.

        Returns
        -------
        None
            This method does not return a value but sets the `self.model` attribute to the
            compiled Keras model.

        Notes
        -----
        The model architecture includes:
        - An initial convolutional block followed by max pooling.
        - Three encoding layers with increasing filter sizes and halving the sequence length.
        - A bottleneck layer that compresses the representation.
        - A symmetric decoding path that mirrors the encoding path.
        - Final upsampling to reconstruct the original sequence dimensions.

        Examples
        --------
        >>> model = MyClass()
        >>> model.window_size = 100
        >>> model.inputsize = 10
        >>> model.num_filters = 32
        >>> model.kernel_size = 3
        >>> model.dropout_rate = 0.2
        >>> model.activation = 'relu'
        >>> model.encoding_dim = 16
        >>> model.makenet()
        >>> model.model.summary()
        """
        input_layer = Input(shape=(self.window_size, self.inputsize))
        x = input_layer

        # Initial conv block
        x = Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding="same")(
            x
        )
        x = BatchNormalization()(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation(self.activation)(x)
        x = MaxPooling1D(pool_size=2, padding="same")(x)

        layersize = self.window_size
        nfilters = self.num_filters
        filter_list = []

        # Encoding path (3 layers)
        for _ in range(3):
            layersize = int(np.ceil(layersize / 2))
            nfilters *= 2
            filter_list.append(nfilters)
            x = Convolution1D(filters=nfilters, kernel_size=self.kernel_size, padding="same")(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=self.dropout_rate)(x)
            x = Activation(self.activation)(x)
            x = MaxPooling1D(pool_size=2, padding="same")(x)

        # Save shape for reshaping later
        shape_before_flatten = K.int_shape(x)[1:]  # (timesteps, channels)

        # Bottleneck
        x = Flatten()(x)
        x = Dense(self.encoding_dim, activation=self.activation, name="encoded")(x)
        x = Dense(np.prod(shape_before_flatten), activation=self.activation)(x)
        x = Reshape(shape_before_flatten)(x)

        # Decoding path (mirror)
        for filters in reversed(filter_list):
            layersize *= 2
            x = UpSampling1D(size=2)(x)
            x = Convolution1D(filters=filters, kernel_size=self.kernel_size, padding="same")(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=self.dropout_rate)(x)
            x = Activation(self.activation)(x)

        # Final upsampling (to match initial maxpool)
        x = UpSampling1D(size=2)(x)
        x = Convolution1D(filters=self.inputsize, kernel_size=self.kernel_size, padding="same")(x)

        output_layer = x
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer="adam", loss="mse")


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
        encoding_dim : int, default=10
            Dimension of the encoding layer.
        num_filters : int, default=10
            Number of filters in the convolutional layer.
        kernel_size : int, default=5
            Size of the convolutional kernel.
        dilation_rate : int, default=1
            Dilation rate for the convolutional layer.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This constructor initializes the CRNNDLFilter layer with specified parameters
        and sets up the network type information in the infodict.

        Examples
        --------
        >>> filter_layer = CRNNDLFilter(
        ...     encoding_dim=15,
        ...     num_filters=20,
        ...     kernel_size=3,
        ...     dilation_rate=2
        ... )
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.encoding_dim = encoding_dim
        self.infodict["nettype"] = "cnn"
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        self.infodict["encoding_dim"] = self.encoding_dim
        super(CRNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and configure the model name and path based on current parameters.

        This method constructs a descriptive model name string using various instance
        attributes and creates the corresponding directory path for model storage.

        Parameters
        ----------
        self : object
            The instance containing model configuration parameters

        Attributes Used
        ---------------
        window_size : int
            Size of the sliding window
        encoding_dim : int
            Dimension of the encoding layer
        num_filters : int
            Number of filters in the convolutional layers
        kernel_size : int
            Size of the convolutional kernel
        num_epochs : int
            Number of training epochs
        excludethresh : float
            Threshold for excluding data points
        corrthresh : float
            Correlation threshold for filtering
        step : int
            Step size for sliding window
        activation : str
            Activation function used
        usebadpts : bool
            Whether to use bad points in training
        excludebysubject : bool
            Whether to exclude data by subject
        namesuffix : str, optional
            Additional suffix to append to model name
        modelroot : str
            Root directory for model storage

        Returns
        -------
        None
            This method modifies instance attributes in-place:
            - self.modelname: constructed model name string
            - self.modelpath: full path to model directory

        Notes
        -----
        The generated model name follows a consistent format:
        "model_crnn_tf2_wXxx_enXxx_fnXX_flXX_eXXX_tX_ctX_sX_activation"
        where Xxx represents zero-padded numbers and XX represents zero-padded two-digit numbers.

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.encoding_dim = 50
        >>> model.getname()
        >>> print(model.modelname)
        'model_crnn_tf2_w100_en050_fn10_fl10_e001_t0.5_ct0.8_s1_relu'
        """
        self.modelname = "_".join(
            [
                "model",
                "crnn",
                "tf2",
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
        Create and compile a 1D convolutional neural network for temporal feature extraction and reconstruction.

        This function builds a neural network architecture consisting of:
        - A convolutional front-end for feature extraction
        - A bidirectional LSTM layer for temporal modeling
        - A dense output layer mapping to the input size

        The model is compiled with Adam optimizer and mean squared error loss.

        Parameters
        ----------
        self : object
            The class instance containing the following attributes:
            - window_size : int
                The size of the input time window.
            - inputsize : int
                The number of input channels/features.
            - num_filters : int
                Number of filters in the convolutional layers.
            - kernel_size : int
                Size of the convolutional kernel.
            - dropout_rate : float
                Dropout rate for regularization.
            - activation : str or callable
                Activation function for convolutional layers.
            - encoding_dim : int
                Number of units in the LSTM layer.

        Returns
        -------
        None
            This method modifies the instance's `model` attribute in-place.

        Notes
        -----
        The network architecture follows this pipeline:
        Input -> Conv1D -> BatchNorm -> Dropout -> Activation ->
        Conv1D -> BatchNorm -> Dropout -> Activation ->
        Bidirectional LSTM -> Dense -> Output

        The model is compiled with:
        - Optimizer: Adam
        - Loss: Mean Squared Error (MSE)

        Examples
        --------
        >>> # Assuming a class with the required attributes
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.inputsize = 10
        >>> model.num_filters = 32
        >>> model.kernel_size = 3
        >>> model.dropout_rate = 0.2
        >>> model.activation = 'relu'
        >>> model.encoding_dim = 64
        >>> model.makenet()
        >>> model.model.summary()
        """
        input_layer = Input(shape=(self.window_size, self.inputsize))
        x = input_layer

        # Convolutional front-end: feature extraction
        x = Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding="same")(
            x
        )
        x = BatchNormalization()(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation(self.activation)(x)

        x = Convolution1D(
            filters=self.num_filters * 2, kernel_size=self.kernel_size, padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation(self.activation)(x)

        # Recurrent layer: temporal modeling
        x = Bidirectional(LSTM(units=self.encoding_dim, return_sequences=True))(x)

        # Output mapping to inputsize channels
        output_layer = Dense(self.inputsize)(x)

        # Model definition
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer="adam", loss="mse")


class LSTMDLFilter(DeepLearningFilter):
    def __init__(self, num_units: int = 16, *args, **kwargs) -> None:
        """
        Initialize the LSTMDLFilter layer.

        Parameters
        ----------
        num_units : int, optional
            Number of units in the LSTM layer, by default 16
        *args : tuple
            Additional positional arguments passed to the parent class
        **kwargs : dict
            Additional keyword arguments passed to the parent class

        Returns
        -------
        None
            This method initializes the layer in-place and does not return any value

        Notes
        -----
        This method sets up the LSTM layer with the specified number of units and
        configures the infodict with the network type and unit count. The parent
        class initialization is called with any additional arguments provided.

        Examples
        --------
        >>> layer = LSTMDLFilter(num_units=32)
        >>> print(layer.num_units)
        32
        """
        self.num_units = num_units
        self.infodict["nettype"] = "lstm"
        self.infodict["num_units"] = self.num_units
        super(LSTMDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return the model name and path based on current configuration parameters.

        This method constructs a standardized model name string using various configuration
        parameters and creates the corresponding directory path. The generated name includes
        information about the model architecture, training parameters, and preprocessing settings.

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - window_size : int
                Size of the sliding window for time series data
            - num_layers : int
                Number of LSTM layers in the model
            - num_units : int
                Number of units in each LSTM layer
            - dropout_rate : float
                Dropout rate for regularization
            - num_epochs : int
                Number of training epochs
            - excludethresh : float
                Threshold for exclusion criteria
            - corrthresh : float
                Correlation threshold for filtering
            - step : int
                Step size for data processing
            - excludebysubject : bool
                Whether to exclude data by subject
            - modelroot : str
                Root directory for model storage

        Returns
        -------
        None
            This method modifies the instance attributes `modelname` and `modelpath` in place.
            It does not return any value.

        Notes
        -----
        The generated model name follows a specific naming convention:
        "model_lstm_tf2_wXxx_lYY_nuZZZ_dDD_rdDD_eEEE_tT_ctTT_sS"
        where Xxx, YY, ZZZ, DD, EEEE, T, TT, S represent zero-padded numerical values.

        If `excludebysubject` is True, "_excludebysubject" is appended to the model name.

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
        >>> model.modelroot = "/path/to/models"
        >>> model.getname()
        >>> print(model.modelname)
        'model_lstm_tf2_w100_l02_nu128_d02_rd02_e100_t05_ct08_s1_excludebysubject'
        """
        self.modelname = "_".join(
            [
                "model",
                "lstm",
                "tf2",
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
        Create and configure a bidirectional LSTM neural network model.

        This function builds a sequential neural network architecture using bidirectional LSTM layers
        followed by time-distributed dense layers. The model is compiled with Adam optimizer and MSE loss.

        Parameters
        ----------
        self : object
            The instance containing the following attributes:
            - num_layers : int
                Number of LSTM layers in the model
            - num_units : int
                Number of units in each LSTM layer
            - dropout_rate : float
                Dropout rate for both dropout and recurrent dropout
            - window_size : int
                Size of the input window for time series data

        Returns
        -------
        None
            This method modifies the instance in-place by setting the `model` attribute.

        Notes
        -----
        The model architecture consists of:
        1. Bidirectional LSTM layers with specified number of units and dropout rates
        2. TimeDistributed Dense layers to map outputs back to window size
        3. Compilation with Adam optimizer and MSE loss function

        Examples
        --------
        >>> model_instance = MyModel()
        >>> model_instance.num_layers = 2
        >>> model_instance.num_units = 50
        >>> model_instance.dropout_rate = 0.2
        >>> model_instance.window_size = 10
        >>> model_instance.makenet()
        >>> print(model_instance.model.summary())
        """
        self.model = Sequential()

        # each layer consists of an LSTM followed by a dense time distributed layer to get it back to the window size
        for layer in range(self.num_layers):
            self.model.add(
                Bidirectional(
                    LSTM(
                        self.num_units,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        return_sequences=True,
                    ),
                    input_shape=(self.window_size, 1),
                )
            )
            self.model.add(TimeDistributed(Dense(1)))

        self.model.compile(optimizer="adam", loss="mse")


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
        Initialize the HybridDLFilter layer.

        Parameters
        ----------
        invert : bool, optional
            If True, inverts the filter operation, by default False
        num_filters : int, optional
            Number of filters to use in the convolutional layer, by default 10
        kernel_size : int, optional
            Size of the convolutional kernel, by default 5
        num_units : int, optional
            Number of units in the dense layer, by default 16
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments

        Returns
        -------
        None
            This method does not return a value

        Notes
        -----
        This method initializes the hybrid deep learning filter by setting up
        the convolutional and dense layer parameters. The infodict is populated
        with configuration information for tracking and debugging purposes.

        Examples
        --------
        >>> layer = HybridDLFilter(
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
        self.infodict["nettype"] = "hybrid"
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        self.infodict["invert"] = self.invert
        self.infodict["num_units"] = self.num_units
        super(HybridDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        """
        Generate and return the model name and path based on current configuration parameters.

        This method constructs a descriptive model name string by joining various configuration
        parameters with specific prefixes and formatting conventions. The resulting model name
        is used to create a unique directory path for model storage.

        Parameters
        ----------
        self : object
            The instance containing model configuration parameters. Expected attributes include:
            - `window_size` : int
            - `num_layers` : int
            - `num_filters` : int
            - `kernel_size` : int
            - `num_units` : int
            - `dropout_rate` : float
            - `num_epochs` : int
            - `excludethresh` : float
            - `corrthresh` : float
            - `step` : int
            - `activation` : str
            - `invert` : bool
            - `excludebysubject` : bool
            - `modelroot` : str

        Returns
        -------
        None
            This method does not return a value but modifies instance attributes:
            - `self.modelname`: The constructed model name string
            - `self.modelpath`: The full path to the model directory

        Notes
        -----
        The model name is constructed using the following components:
        - "model_hybrid_tf2_" prefix
        - Window size with 3-digit zero-padded formatting
        - Number of layers with 2-digit zero-padded formatting
        - Number of filters with 2-digit zero-padded formatting
        - Kernel size with 2-digit zero-padded formatting
        - Number of units
        - Dropout rate (appears twice with different prefixes)
        - Number of epochs with 3-digit zero-padded formatting
        - Exclusion threshold
        - Correlation threshold
        - Step size
        - Activation function name

        Additional suffixes are appended based on boolean flags:
        - "_invert" if `self.invert` is True
        - "_excludebysubject" if `self.excludebysubject` is True

        Examples
        --------
        >>> model = MyModel()
        >>> model.window_size = 100
        >>> model.num_layers = 3
        >>> model.num_filters = 16
        >>> model.kernel_size = 5
        >>> model.num_units = 64
        >>> model.dropout_rate = 0.2
        >>> model.num_epochs = 100
        >>> model.excludethresh = 0.5
        >>> model.corrthresh = 0.8
        >>> model.step = 1
        >>> model.activation = "relu"
        >>> model.invert = True
        >>> model.excludebysubject = False
        >>> model.getname()
        >>> print(model.modelname)
        'model_hybrid_tf2_w100_l03_fn16_fl05_nu64_d0.2_rd0.2_e100_t0.5_ct0.8_s01_relu_invert'
        """
        self.modelname = "_".join(
            [
                "model",
                "hybrid",
                "tf2",
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
        Build and compile a neural network model with configurable CNN and LSTM layers.

        This function constructs a neural network model using Keras, with the architecture
        determined by the `invert` flag. If `invert` is True, the model starts with a
        Conv1D layer followed by LSTM layers; otherwise, it starts with LSTM layers
        followed by Conv1D layers. The model is compiled with the RMSprop optimizer
        and mean squared error loss.

        Parameters
        ----------
        self : object
            The instance of the class containing the model configuration attributes.

        Attributes Used
        ---------------
        self.invert : bool
            If True, the model begins with a Conv1D layer and ends with an LSTM layer.
            If False, the model begins with an LSTM layer and ends with a Conv1D layer.
        self.num_filters : int
            Number of filters in each Conv1D layer.
        self.kernel_size : int
            Size of the kernel in Conv1D layers.
        self.padding : str, default='same'
            Padding mode for Conv1D layers.
        self.window_size : int
            Length of the input sequence.
        self.inputsize : int
            Number of features in the input data.
        self.num_layers : int
            Total number of layers in the model.
        self.num_units : int
            Number of units in the LSTM layers.
        self.dropout_rate : float
            Dropout rate for regularization.
        self.activation : str or callable
            Activation function for Conv1D layers.

        Returns
        -------
        None
            This method modifies the instance's `self.model` attribute in place.

        Notes
        -----
        - The model uses `Sequential` from Keras.
        - Batch normalization and dropout are applied after each Conv1D layer (except the last).
        - The final layer is a Dense layer wrapped in `TimeDistributed` for sequence-to-sequence output.
        - The model is compiled using `RMSprop` optimizer and `mse` loss.

        Examples
        --------
        >>> model_builder = MyModelClass()
        >>> model_builder.invert = True
        >>> model_builder.num_filters = 32
        >>> model_builder.kernel_size = 3
        >>> model_builder.window_size = 100
        >>> model_builder.inputsize = 1
        >>> model_builder.num_layers = 5
        >>> model_builder.num_units = 64
        >>> model_builder.dropout_rate = 0.2
        >>> model_builder.activation = 'relu'
        >>> model_builder.makenet()
        >>> model_builder.model.summary()
        """
        self.model = Sequential()

        if self.invert:
            # make the input layer
            self.model.add(
                Convolution1D(
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    input_shape=(self.window_size, self.inputsize),
                )
            )
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=self.dropout_rate))
            self.model.add(Activation(self.activation))

            # then make make the intermediate CNN layers
            for layer in range(self.num_layers - 2):
                self.model.add(
                    Convolution1D(
                        filters=self.num_filters,
                        kernel_size=self.kernel_size,
                        padding="same",
                    )
                )
                self.model.add(BatchNormalization())
                self.model.add(Dropout(rate=self.dropout_rate))
                self.model.add(Activation(self.activation))

            # finish with an LSTM layer to find hidden states
            self.model.add(
                Bidirectional(
                    LSTM(
                        self.num_units,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        return_sequences=True,
                    ),
                    input_shape=(self.window_size, 1),
                )
            )
            self.model.add(TimeDistributed(Dense(1)))

        else:
            # start with an LSTM layer to find hidden states
            self.model.add(
                Bidirectional(
                    LSTM(
                        self.num_units,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        return_sequences=True,
                    ),
                    input_shape=(self.window_size, 1),
                )
            )
            self.model.add(TimeDistributed(Dense(1)))
            self.model.add(Dropout(rate=self.dropout_rate))

            # then make make the intermediate CNN layers
            for layer in range(self.num_layers - 2):
                self.model.add(
                    Convolution1D(
                        filters=self.num_filters,
                        kernel_size=self.kernel_size,
                        padding="same",
                    )
                )
                self.model.add(BatchNormalization())
                self.model.add(Dropout(rate=self.dropout_rate))
                self.model.add(Activation(self.activation))

            # make the output layer
            self.model.add(
                Convolution1D(filters=self.inputsize, kernel_size=self.kernel_size, padding="same")
            )

        self.model.compile(optimizer=RMSprop(), loss="mse")


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
    Apply or reverse a frequency-domain scaling and normalization to input data.

    This function performs either forward or inverse transformation of the input
    data in the frequency domain, applying scaling, normalization, and optionally
    hybrid encoding. It supports both logarithmic and standard normalization
    modes.

    Parameters
    ----------
    data : NDArray
        Input signal or transformed data (depending on `reverse` flag).
    scalefac : float, optional
        Scaling factor used for normalization. Default is 1.0.
    reverse : bool, optional
        If True, performs inverse transformation from frequency domain back
        to time domain. Default is False.
    hybrid : bool, optional
        If True, returns a hybrid representation combining original data
        and magnitude spectrum. Default is False.
    lognormalize : bool, optional
        If True, applies logarithmic normalization to the magnitude spectrum.
        Default is True.
    epsilon : float, optional
        Small constant added to avoid log(0). Default is 1e-10.
    numorders : int, optional
        Number of orders used for scaling in log normalization. Default is 6.

    Returns
    -------
    tuple[NDArray, float] or NDArray
        - If `reverse=False`: A tuple of (transformed data, scale factor).
          The transformed data is a stacked array of magnitude and phase components
          (or original data in hybrid mode).
        - If `reverse=True`: Reconstructed time-domain signal.

    Notes
    -----
    - Forward mode applies FFT, normalizes the magnitude, and stacks magnitude
      and phase.
    - In hybrid mode, the output includes the original time-domain signal
      instead of the phase component.
    - In reverse mode, the phase and magnitude components are used to reconstruct
      the original signal using inverse FFT.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> x = np.random.randn(1024)
    >>> scaled_data, scale = filtscale(x)
    >>> reconstructed = filtscale(scaled_data, scale, reverse=True)
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

    Replaces the '.txt' extension with '_badpts.txt' to create a standardized
    bad points filename pattern.

    Parameters
    ----------
    name : str
        Input filename, typically ending with '.txt' extension.

    Returns
    -------
    str
        Filename with '.txt' replaced by '_badpts.txt' extension.

    Notes
    -----
    This function assumes the input filename ends with '.txt' extension.
    If the input does not contain '.txt', the function will append '_badpts.txt'
    to the end of the string.

    Examples
    --------
    >>> tobadpts("data.txt")
    'data_badpts.txt'

    >>> tobadpts("results.txt")
    'results_badpts.txt'

    >>> tobadpts("config")
    'config_badpts.txt'
    """
    return name.replace(".txt", "_badpts.txt")


def targettoinput(name: str, targetfrag: str = "xyz", inputfrag: str = "abc") -> str:
    """
    Replace occurrences of a target fragment with an input fragment in a string.

    Parameters
    ----------
    name : str
        The input string to process.
    targetfrag : str, default='xyz'
        The fragment to search for and replace. Defaults to 'xyz'.
    inputfrag : str, default='abc'
        The fragment to replace targetfrag with. Defaults to 'abc'.

    Returns
    -------
    str
        The modified string with targetfrag replaced by inputfrag.

    Notes
    -----
    This function uses Python's built-in string replace method, which replaces
    all occurrences of targetfrag with inputfrag in the input string.

    Examples
    --------
    >>> targettoinput("hello xyz world")
    'hello abc world'

    >>> targettoinput("xyzxyzxyz", "xyz", "123")
    '123123123'

    >>> targettoinput("abcdef", "xyz", "123")
    'abcdef'
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
    Find and validate timecourse files matching a search pattern, and determine the length of the timecourses.

    This function searches for files matching the given `searchstring`, checks for the existence of
    corresponding info files to ensure completeness, and reads the first matched file to determine
    the length of the timecourses. It is intended for use with BIDS-compatible timecourse files.

    Parameters
    ----------
    searchstring : str
        A glob pattern to match target files. Typically includes a path and file name pattern.
    usebadpts : bool, optional
        Flag indicating whether to use bad points in processing. Default is False.
    targetfrag : str, optional
        Fragment identifier for target files. Default is "xyz".
    inputfrag : str, optional
        Fragment identifier for input files. Default is "abc".
    debug : bool, optional
        If True, print debug information including matched files and processing steps.
        Default is False.

    Returns
    -------
    tuple[list[str], int]
        A tuple containing:
        - List of matched and validated file paths.
        - Length of the timecourses (number of timepoints) in the first matched file.

    Notes
    -----
    - The function expects files to have a corresponding `_info` file for validation.
    - Timecourse data is read from the first matched file using `tide_io.readbidstsv`.
    - The function currently only reads the first matched file to determine `tclen`, assuming all
      matched files have the same timecourse length.

    Examples
    --------
    >>> matched_files, length = getmatchedtcs("sub-*/func/*cardiacfromfmri_25.0Hz*")
    >>> print(f"Found {len(matched_files)} files with {length} timepoints each.")
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
        colspec="cardiacfromfmri_25.0Hz,normpleth",
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
) -> tuple[NDArray, NDArray, list[str]] | tuple[NDArray, NDArray, list[str], NDArray]:
    """
    Read and process time-series data from a list of matched files.

    This function reads cardiac and plethysmographic data from a set of files, applies
    filtering based on data quality metrics (e.g., correlation thresholds, NaN values,
    data length, and standard deviation), and returns processed arrays for training.

    Parameters
    ----------
    matchedfilelist : list of str
        List of file paths to be processed. Each file should contain time-series data
        in a format compatible with `tide_io.readbidstsv`.
    tclen : int
        The length of the time series to be read from each file.
    targetfrag : str, optional
        Fragment identifier used for mapping filenames (default is "xyz").
    inputfrag : str, optional
        Fragment identifier used for mapping filenames (default is "abc").
    usebadpts : bool, optional
        If True, include bad point data in the output (default is False).
    startskip : int, optional
        Number of samples to skip at the beginning of each time series (default is 0).
    endskip : int, optional
        Number of samples to skip at the end of each time series (default is 0).
    corrthresh : float, optional
        Threshold for correlation between raw and plethysmographic signals.
        Files with correlations below this value are excluded (default is 0.5).
    readlim : int, optional
        Maximum number of files to read. If None, all files are read (default is None).
    readskip : int, optional
        Number of files to skip at the beginning of the list. If None, no files are skipped (default is None).
    debug : bool, optional
        If True, print debug information during processing (default is False).

    Returns
    -------
    tuple of (NDArray, NDArray, list[str]) or (NDArray, NDArray, list[str], NDArray)
        - `x1[startskip:-endskip, :count]`: Array of x-axis time series data.
        - `y1[startskip:-endskip, :count]`: Array of y-axis time series data.
        - `names[:count]`: List of file names that passed quality checks.
        - `bad1[startskip:-endskip, :count]`: Optional array of bad point data if `usebadpts` is True.

    Notes
    -----
    - Files are filtered based on:
        - Correlation threshold (`corrthresh`)
        - Presence of NaN values
        - Data length (must be at least `tclen`)
        - Standard deviation of data (must be between 0.5 and 20.0)
    - Excluded files are logged with reasons.
    - If `usebadpts` is True, bad point data is included in the returned tuple.

    Examples
    --------
    >>> x, y, names = readindata(
    ...     matchedfilelist=["file1.tsv", "file2.tsv"],
    ...     tclen=1000,
    ...     corrthresh=0.6,
    ...     readlim=10
    ... )
    >>> print(f"Loaded {len(names)} files")
    """
    """
        Read and process time-series data from a list of matched files.

        This function reads cardiac and plethysmographic data from a set of files, applies
        filtering based on data quality metrics (e.g., correlation thresholds, NaN values,
        data length, and standard deviation), and returns processed arrays for training.

        Parameters
        ----------
        matchedfilelist : list of str
            List of file paths to be processed. Each file should contain time-series data
            in a format compatible with `tide_io.readbidstsv`.
        tclen : int
            The length of the time series to be read from each file.
        targetfrag : str, optional
            Fragment identifier used for mapping filenames (default is "xyz").
        inputfrag : str, optional
            Fragment identifier used for mapping filenames (default is "abc").
        usebadpts : bool, optional
            If True, include bad point data in the output (default is False).
        startskip : int, optional
            Number of samples to skip at the beginning of each time series (default is 0).
        endskip : int, optional
            Number of samples to skip at the end of each time series (default is 0).
        corrthresh : float, optional
            Threshold for correlation between raw and plethysmographic signals.
            Files with correlations below this value are excluded (default is 0.5).
        readlim : int, optional
            Maximum number of files to read. If None, all files are read (default is None).
        readskip : int, optional
            Number of files to skip at the beginning of the list. If None, no files are skipped (default is None).
        debug : bool, optional
            If True, print debug information during processing (default is False).

        Returns
        -------
        tuple of (NDArray, NDArray, list[str]) or (NDArray, NDArray, list[str], NDArray)
            - `x1[startskip:-endskip, :count]`: Array of x-axis time series data.
            - `y1[startskip:-endskip, :count]`: Array of y-axis time series data.
            - `names[:count]`: List of file names that passed quality checks.
            - `bad1[startskip:-endskip, :count]`: Optional array of bad point data if `usebadpts` is True.

        Notes
        -----
        - Files are filtered based on:
            - Correlation threshold (`corrthresh`)
            - Presence of NaN values
            - Data length (must be at least `tclen`)
            - Standard deviation of data (must be between 0.5 and 20.0)
        - Excluded files are logged with reasons.
        - If `usebadpts` is True, bad point data is included in the returned tuple.

        Examples
        --------
        >>> x, y, names = readindata(
        ...     matchedfilelist=["file1.tsv", "file2.tsv"],
        ...     tclen=1000,
        ...     corrthresh=0.6,
        ...     readlim=10
        ... )
        >>> print(f"Loaded {len(names)} files")
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
        (
            samplerate,
            starttime,
            columns,
            inputarray,
            compression,
            columnsource,
        ) = tide_io.readbidstsv(
            matchedfilelist[i],
            colspec="cardiacfromfmri_25.0Hz,normpleth",
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
                bad1[:tclen, count] = inputarray[2, :]
            count += 1
        else:
            print(f"{matchedfilelist[i]} excluded:")
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
    if usebadpts:
        return (
            x1[startskip:-endskip, :count],
            y1[startskip:-endskip, :count],
            names[:count],
            bad1[startskip:-endskip, :count],
        )
    else:
        return (
            x1[startskip:-endskip, :count],
            y1[startskip:-endskip, :count],
            names[:count],
        )


def prep(
    window_size: int,
    step: int = 1,
    excludethresh: float = 4.0,
    usebadpts: bool = False,
    startskip: int = 200,
    endskip: int = 200,
    excludebysubject: bool = True,
    thesuffix: str = "sliceres",
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

    This function reads time-series data from JSON files, normalizes the data,
    applies windowing to create input-output pairs, and splits the data into
    training and validation sets based on subject-wise or window-wise exclusion
    criteria.

    Parameters
    ----------
    window_size : int
        Size of the sliding window used to create input-output pairs.
    step : int, optional
        Step size for sliding window (default is 1).
    excludethresh : float, optional
        Threshold for excluding windows or subjects based on maximum absolute
        value of input data (default is 4.0).
    usebadpts : bool, optional
        Whether to include bad points in the data (default is False).
    startskip : int, optional
        Number of time points to skip at the beginning of each time series
        (default is 200).
    endskip : int, optional
        Number of time points to skip at the end of each time series
        (default is 200).
    excludebysubject : bool, optional
        If True, exclude entire subjects based on maximum absolute value;
        otherwise, exclude individual windows (default is True).
    thesuffix : str, optional
        Suffix used to identify files (default is "sliceres").
    thedatadir : str, optional
        Directory where the data files are located (default is
        "/data/frederic/physioconn/output_2025").
    inputfrag : str, optional
        Fragment identifier for input data (default is "abc").
    targetfrag : str, optional
        Fragment identifier for target data (default is "xyz").
    corrthresh : float, optional
        Correlation threshold for matching time series (default is 0.5).
    dofft : bool, optional
        Whether to perform FFT on the data (default is False).
    readlim : int or None, optional
        Limit on number of time points to read (default is None).
    readskip : int or None, optional
        Number of time points to skip when reading data (default is None).
    countlim : int or None, optional
        Limit on number of subjects to process (default is None).
    debug : bool, optional
        Whether to enable debug logging (default is False).

    Returns
    -------
    tuple of (NDArray, NDArray, NDArray, NDArray, int, int, int)
        If `dofft` is False:
            - train_x : ndarray of shape (n_train, window_size, 1)
            - train_y : ndarray of shape (n_train, window_size, 1)
            - val_x : ndarray of shape (n_val, window_size, 1)
            - val_y : ndarray of shape (n_val, window_size, 1)
            - N_subjs : int
            - tclen : int
            - batchsize : int

        tuple of (NDArray, NDArray, NDArray, NDArray, int, int, int,
                  NDArray, NDArray)
        If `dofft` is True:
            - train_x : ndarray of shape (n_train, window_size, 2)
            - train_y : ndarray of shape (n_train, window_size, 2)
            - val_x : ndarray of shape (n_val, window_size, 2)
            - val_y : ndarray of shape (n_val, window_size, 2)
            - N_subjs : int
            - tclen : int
            - batchsize : int
            - Xscale_fourier : ndarray of shape (N_subjs, windowspersubject)
            - Yscale_fourier : ndarray of shape (N_subjs, windowspersubject)

    Notes
    -----
    - Data normalization is performed using median absolute deviation (MAD).
    - Windows are created based on sliding window approach.
    - Training and validation sets are split based on subject-wise partitioning.
    - If `usebadpts` is True, bad points are included in the returned data.
    - If `dofft` is True, data is transformed using a filtering scale function.

    Examples
    --------
    >>> train_x, train_y, val_x, val_y, N_subjs, tclen, batchsize = prep(
    ...     window_size=100, step=10, excludethresh=3.0, dofft=False
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
    if usebadpts:
        x, y, names, bad = readindata(
            matchedfilelist,
            tclen,
            corrthresh=corrthresh,
            targetfrag=targetfrag,
            inputfrag=inputfrag,
            usebadpts=True,
            startskip=startskip,
            endskip=endskip,
            readlim=readlim,
            readskip=readskip,
        )
    else:
        x, y, names = readindata(
            matchedfilelist,
            tclen,
            corrthresh=corrthresh,
            targetfrag=targetfrag,
            inputfrag=inputfrag,
            startskip=startskip,
            endskip=endskip,
            readlim=readlim,
            readskip=readskip,
        )
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
        Xb = np.zeros((numgoodwindows, window_size, 1))
        Yb = np.zeros((numgoodwindows, window_size, 1))
        if usebadpts:
            Xb_withbad = np.zeros((numgoodwindows, window_size, 1))
        LGR.info(f"dimensions of Xb: {Xb.shape}")
        thiswindow = 0
        for subj in range(N_subjs):
            for windownumber in range(windowspersubject):
                if usewindow[subj * windowspersubject + windownumber] == 1:
                    Xb[thiswindow, :, 0] = x[
                        step * windownumber : (step * windownumber + window_size), subj
                    ]
                    Yb[thiswindow, :, 0] = y[
                        step * windownumber : (step * windownumber + window_size), subj
                    ]
                    if usebadpts:
                        Xb_withbad[thiswindow, :, 0] = bad[
                            step * windownumber : (step * windownumber + window_size),
                            subj,
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

        Yb = np.zeros((N_subjs * windowspersubject, window_size, 1))
        LGR.info(f"dimensions of Yb: {Yb.shape}")
        for j in range(N_subjs):
            for i in range(windowspersubject):
                Yb[j * windowspersubject + i, :, 0] = Y[0, step * i : (step * i + window_size), j]

        if usebadpts:
            Xb_withbad = np.zeros((N_subjs * windowspersubject, window_size, 2))
            LGR.info(f"dimensions of Xb_withbad: {Xb_withbad.shape}")
            for j in range(N_subjs):
                LGR.info(f"packing data for subject {j}")
                for i in range(windowspersubject):
                    Xb_withbad[j * windowspersubject + i, :, 0] = X[
                        0, step * i : (step * i + window_size), j
                    ]
                    Xb_withbad[j * windowspersubject + i, :, 1] = BAD[
                        0, step * i : (step * i + window_size), j
                    ]
            Xb = Xb_withbad

        subjectstarts = range(N_subjs) * windowspersubject
        for subj in range(N_subjs):
            LGR.info(f"{names[subj]} starts at {subjectstarts[subj]}")

    LGR.info(f"Xb.shape: {Xb.shape}")
    LGR.info(f"Yb.shape: {Yb.shape}")

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
        return (
            train_x,
            train_y,
            val_x,
            val_y,
            N_subjs,
            tclen - startskip - endskip,
            batchsize,
        )
