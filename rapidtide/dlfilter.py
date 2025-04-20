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
        window_size=128,
        num_layers=5,
        dropout_rate=0.3,
        num_pretrain_epochs=0,
        num_epochs=1,
        activation="relu",
        modelroot=".",
        dofft=False,
        excludethresh=4.0,
        usebadpts=False,
        thesuffix="25.0Hz",
        modelpath=".",
        thedatadir="/Users/frederic/Documents/MR_data/physioconn/timecourses",
        inputfrag="abc",
        targetfrag="xyz",
        corrthresh=0.5,
        excludebysubject=True,
        startskip=200,
        endskip=200,
        step=1,
        namesuffix=None,
        readlim=None,
        readskip=None,
        countlim=None,
        **kwargs,
    ):
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

    def loaddata(self):
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
    def predict_model(self, X):
        return self.model(X, training=False)

    def evaluate(self):
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

    def initmetadata(self):
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

    def updatemetadata(self):
        self.infodict["loss"] = self.loss
        self.infodict["val_loss"] = self.val_loss
        self.infodict["raw_error"] = self.raw_error
        self.infodict["prediction_error"] = self.pred_error
        tide_io.writedicttojson(self.infodict, os.path.join(self.modelname, "model_meta.json"))

    def savemodel(self, altname=None):
        if altname is None:
            modelsavename = self.modelname
        else:
            modelsavename = altname
        LGR.info(f"saving {modelsavename}")
        self.model.save(os.path.join(modelsavename, "model.keras"))

    def loadmodel(self, modelname, verbose=False):
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

    def initialize(self):
        self.getname()
        self.makenet()
        self.model.summary()
        self.savemodel()
        self.initmetadata()
        self.initialized = True
        self.trained = False

    def train(self):
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

    def apply(self, inputdata, badpts=None):
        initscale = mad(inputdata)
        scaleddata = inputdata / initscale
        predicteddata = scaleddata * 0.0
        weightarray = scaleddata * 0.0
        N_pts = len(scaleddata)
        if self.usebadpts:
            if badpts is None:
                badpts = scaleddata * 0.0
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
        num_filters=10,
        kernel_sizes=[4, 8, 12],
        input_lens=[64, 128, 192],
        input_width=1,
        dilation_rate=1,
        *args,
        **kwargs,
    ):
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
        self.modelname = "_".join(
            [
                "model",
                "multiscalecnn",
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
    def __init__(self, num_filters=10, kernel_size=5, dilation_rate=1, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.infodict["nettype"] = "cnn"
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        super(CNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        self.modelname = "_".join(
            [
                "model",
                "cnn",
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
    def __init__(self, encoding_dim=10, *args, **kwargs):
        self.encoding_dim = encoding_dim
        self.infodict["nettype"] = "autoencoder"
        self.infodict["encoding_dim"] = self.encoding_dim
        super(DenseAutoencoderDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        self.modelname = "_".join(
            [
                "model",
                "denseautoencoder",
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
        self, encoding_dim=10, num_filters=5, kernel_size=5, dilation_rate=1, *args, **kwargs
    ):
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
        self.modelname = "_".join(
            [
                "model",
                "convautoencoder",
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
        self, encoding_dim=10, num_filters=10, kernel_size=5, dilation_rate=1, *args, **kwargs
    ):
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
        self.modelname = "_".join(
            [
                "model",
                "crnn",
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
    def __init__(self, num_units=16, *args, **kwargs):
        self.num_units = num_units
        self.infodict["nettype"] = "lstm"
        self.infodict["num_units"] = self.num_units
        super(LSTMDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
        self.modelname = "_".join(
            [
                "model",
                "lstm",
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
    def __init__(self, invert=False, num_filters=10, kernel_size=5, num_units=16, *args, **kwargs):
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
        self.modelname = "_".join(
            [
                "model",
                "hybrid",
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
    data,
    scalefac=1.0,
    reverse=False,
    hybrid=False,
    lognormalize=True,
    epsilon=1e-10,
    numorders=6,
):
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


def tobadpts(name):
    return name.replace(".txt", "_badpts.txt")


def targettoinput(name, targetfrag="xyz", inputfrag="abc"):
    LGR.debug(f"replacing {targetfrag} with {inputfrag}")
    return name.replace(targetfrag, inputfrag)


def getmatchedtcs(searchstring, usebadpts=False, targetfrag="xyz", inputfrag="abc", debug=False):
    # list all of the target files
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
    matchedfilelist,
    tclen,
    targetfrag="xyz",
    inputfrag="abc",
    usebadpts=False,
    startskip=0,
    endskip=0,
    corrthresh=0.5,
    readlim=None,
    readskip=None,
    debug=False,
):
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
    window_size,
    step=1,
    excludethresh=4.0,
    usebadpts=False,
    startskip=200,
    endskip=200,
    excludebysubject=True,
    thesuffix="sliceres",
    thedatadir="/data/frederic/physioconn/output_2025",
    inputfrag="abc",
    targetfrag="xyz",
    corrthresh=0.5,
    dofft=False,
    readlim=None,
    readskip=None,
    countlim=None,
    debug=False,
):
    """
    prep - reads in training and validation data for 1D filter

    Parameters
    ----------
    window_size
    step
    excludethresh
    excludebysubject
    usebadpts
    startskip
    endskip
    thesuffix
    thedatadir
    inputfrag
    targetfrag
    corrthresh
    dofft
    readlim
    readskip
    countlim

    Returns
    -------
    train_x, train_y, val_x, val_y, N_subjs, tclen - startskip, batchsize

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
