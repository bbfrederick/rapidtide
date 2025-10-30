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

    def predict_model(self, X):
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

    def evaluate(self):
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

    def initmetadata(self):
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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": (
                    self.model.get_config() if hasattr(self.model, "get_config") else None
                ),
            },
            os.path.join(modelsavename, "model.pth"),
        )

    def loadmodel(self, modelname, verbose=False):
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

    def initialize(self):
        self.getname()
        self.makenet()
        print(self.model)
        self.savemodel()
        self.initmetadata()
        self.initialized = True
        self.trained = False

    def train(self):
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
        num_filters,
        kernel_size,
        num_layers,
        dropout_rate,
        dilation_rate,
        activation,
        inputsize,
    ):
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
        self.layers.append(nn.Conv1d(num_filters, inputsize, kernel_size, padding="same"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
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
    def __init__(self, num_filters=10, kernel_size=5, dilation_rate=1, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.nettype = "cnn"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_filters"] = self.num_filters
        self.infodict["kernel_size"] = self.kernel_size
        super(CNNDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
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
    def __init__(self, window_size, encoding_dim, num_layers, dropout_rate, activation, inputsize):
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
        # Flatten input from (batch, channels, length) to (batch, channels*length)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        for layer in self.layers:
            x = layer(x)

        # Reshape back to (batch, channels, length)
        x = x.reshape(batch_size, self.inputsize, self.window_size)
        return x

    def get_config(self):
        return {
            "window_size": self.window_size,
            "encoding_dim": self.encoding_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "inputsize": self.inputsize,
        }


class DenseAutoencoderDLFilter(DeepLearningFilter):
    def __init__(self, encoding_dim=10, *args, **kwargs):
        self.encoding_dim = encoding_dim
        self.nettype = "autoencoder"
        self.infodict["nettype"] = self.nettype
        self.infodict["encoding_dim"] = self.encoding_dim
        super(DenseAutoencoderDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
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
        return nn.Sequential(
            nn.Conv1d(self.input_width, self.num_filters, kernel_size, padding="same"),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(self.num_filters, 50),
            nn.Tanh(),
            nn.Dropout(0.3),
        )

    def forward(self, x_small, x_med, x_large):
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
        self, encoding_dim=10, num_filters=5, kernel_size=5, dilation_rate=1, *args, **kwargs
    ):
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
        self, encoding_dim=10, num_filters=10, kernel_size=5, dilation_rate=1, *args, **kwargs
    ):
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
        return {
            "num_units": self.num_units,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "window_size": self.window_size,
            "inputsize": self.inputsize,
        }


class LSTMDLFilter(DeepLearningFilter):
    def __init__(self, num_units=16, *args, **kwargs):
        self.num_units = num_units
        self.nettype = "lstm"
        self.infodict["nettype"] = self.nettype
        self.infodict["num_units"] = self.num_units
        super(LSTMDLFilter, self).__init__(*args, **kwargs)

    def getname(self):
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
    def __init__(self, invert=False, num_filters=10, kernel_size=5, num_units=16, *args, **kwargs):
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
    data: np.ndarray,
    scalefac: float = 1.0,
    reverse: bool = False,
    hybrid: bool = False,
    lognormalize: bool = True,
    epsilon: float = 1e-10,
    numorders: int = 6,
) -> tuple[np.ndarray, float] | np.ndarray:
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
    return name.replace(".txt", "_badpts.txt")


def targettoinput(name: str, targetfrag: str = "xyz", inputfrag: str = "abc") -> str:
    LGR.debug(f"replacing {targetfrag} with {inputfrag}")
    return name.replace(targetfrag, inputfrag)


def getmatchedtcs(
    searchstring: str,
    usebadpts: bool = False,
    targetfrag: str = "xyz",
    inputfrag: str = "abc",
    debug: bool = False,
) -> tuple[list[str], int]:
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
) -> (
    tuple[np.ndarray, np.ndarray, list[str]] | tuple[np.ndarray, np.ndarray, list[str], np.ndarray]
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
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray]
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
    print("about to read in data")
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

        subjectstarts = [i * windowspersubject for i in range(N_subjs)]
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
