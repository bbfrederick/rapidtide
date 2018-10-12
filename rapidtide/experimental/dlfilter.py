#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 23:01:07 2018

@author: neuro
"""

import matplotlib.pyplot as plt
import dataload as dl
import numpy as np
import os
import sys
from statsmodels.robust.scale import mad

import rapidtide.io as tide_io

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Bidirectional, Convolution1D, Dense, Activation, Dropout, BatchNormalization, MaxPooling1D, LSTM, TimeDistributed
from keras.callbacks import TerminateOnNaN, ModelCheckpoint
from keras.models import load_model

class dlfilter:
    """Base class for deep learning filter"""
    thesuffix = 'sliceres'
    thedatadir = '/Users/frederic/Documents/MR_data/physioconn/timecourses'
    modelroot = '.'
    excludethresh = 4.0
    modelname = None
    modelpath = None
    intermediatemodelpath = None
    usebadpts = False
    activation = 'relu'
    dofft = False
    debug = False
    readlim = None
    countlim = None
    lossfilename = None
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    model = None
    modelname = None
    inputsize = None
    infodict = {}


    def __init__(self,
        window_size=128,
        num_layers=5,
        num_units=16,
        num_filters=10,
        kernel_size=5,
        dropout_rate=0.3,
        num_epochs=1,
        activation='relu',
        modelroot='.',
        dofft=False,
        debug=False,
        excludethresh=4.0,
        usebadpts=False,
        thesuffix='25.0Hz',
        thedatadir='/Users/frederic/Documents/MR_data/physioconn/timecourses',
        readlim=None,
        countlim=None):

        self.window_size = window_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.usebadpts = usebadpts
        if self.usebadpts:
            self.inputsize = 2
        else:
            self.inputsize = 1
        self.activation = activation
        self.modelroot = modelroot
        self.dofft = dofft
        self.debug = debug
        self.thesuffix = thesuffix
        self.thedatadir = thedatadir
        self.excludethresh = excludethresh
        self.readlim = readlim
        self.countlim = countlim
        self.model = None
        self.initialized = False
        self.trained = False

        # populate infodict
        self.infodict['window_size'] = self.window_size
        self.infodict['usebadpts'] = self.usebadpts
        self.infodict['dofft'] = self.dofft
        self.infodict['excludethresh'] = self.excludethresh
        self.infodict['num_epochs'] = self.num_epochs
        self.infodict['num_layers'] = self.num_layers
        self.infodict['modelname'] = self.modelname
        self.infodict['dropout_rate'] = self.dropout_rate
        self.infodict['train_arch'] = sys.platform


    def loaddata(self):
        if not self.initialized:
            print('model must be initialized prior to loading data')
            sys.exit()

        if self.dofft:
            self.train_x, self.train_y, self.val_x, self.val_y, self.Ns, self.tclen, self.thebatchsize, dummy, dummy = dl.prep(self.window_size,
                                                                        thesuffix=self.thesuffix,
                                                                        thedatadir=self.thedatadir,
                                                                        dofft=self.dofft,
                                                                        debug=self.debug,
                                                                        usebadpts=self.usebadpts,
                                                                        excludethresh=self.excludethresh,
                                                                        readlim=self.readlim,
                                                                        countlim=self.countlim)
        else:
            self.train_x, self.train_y, self.val_x, self.val_y, self.Ns, self.tclen, self.thebatchsize = dl.prep(self.window_size,
                                                                        thesuffix=self.thesuffix,
                                                                        thedatadir=self.thedatadir,
                                                                        dofft=self.dofft,
                                                                        debug=self.debug,
                                                                        usebadpts=self.usebadpts,
                                                                        excludethresh=self.excludethresh,
                                                                        readlim=self.readlim,
                                                                        countlim=self.countlim)

    def evaluate(self):
        self.lossfilename = os.path.join(self.modelname, 'loss.png')
        print('lossfilename:', self.lossfilename)
    
        YPred = self.model.predict(self.val_x)

        error = self.val_y - YPred
        self.pred_error = (np.mean(np.square(error)))
    
        error2 = self.val_x - self.val_y
        self.raw_error = (np.mean(np.square(error2)))
        print('Prediction Error: ', self.pred_error, 'Raw Error: ', self.raw_error)
    
        f = open(os.path.join(self.modelname, 'loss.txt'), 'w')
        f.write(self.modelname + ': Prediction Error: ' + str(self.pred_error) + ' Raw Error: ' + str(self.raw_error) + '\n')
        f.close()
    
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
    
        epochs = range(len(self.loss))
    
        plt.plot(epochs, self.loss, 'bo', label='Training loss')
        plt.plot(epochs, self.val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(self.lossfilename)
        plt.close()
        self.updatemetadata()
    
        return self.loss, self.val_loss, self.pred_error, self.raw_error


    def initmetadata(self):
        self.infodict = {}
        self.infodict['window_size'] = self.window_size
        self.infodict['usebadpts'] = self.usebadpts
        self.infodict['dofft'] = self.dofft
        self.infodict['excludethresh'] = self.excludethresh
        self.infodict['num_epochs'] = self.num_epochs
        self.infodict['num_layers'] = self.num_layers
        self.infodict['dropout_rate'] = self.dropout_rate
        self.infodict['train_arch'] = sys.platform
        self.infodict['modelname'] = self.modelname
        tide_io.writedicttojson(self.infodict, os.path.join(self.modelname, 'model_meta.json'))


    def updatemetadata(self):
        self.infodict['loss'] = self.loss
        self.infodict['val_loss'] = self.val_loss
        self.infodict['raw_error'] = self.raw_error
        self.infodict['prediction_error'] = self.pred_error
        tide_io.writedicttojson(self.infodict, os.path.join(self.modelname, 'model_meta.json'))


    def savemodel(self):
        # save the trained model
        self.model.save(os.path.join(self.modelname, 'model.h5'))


    def loadmodel(self, modelname):
        # read in the data

        # load in the model with weights
        self.model = load_model(os.path.join(modelname, 'model.h5'))
        self.model.summary()

        # now load additional information
        self.infodict = tide_io.readdictfromjson(os.path.join(modelname, 'model_meta.json'))
        self.window_size = self.infodict['window_size']
        self.usebadpts = self.infodict['usebadpts']

        # model is ready to use
        self.initialized = True
        self.trained = True


    def initialize(self):
        self.getname()
        self.makenet()
        self.model.summary()
        self.savemodel()
        self.initmetadata()
        self.initialized = True
        self.trained = False


    def train(self):
        self.intermediatemodelpath = os.path.join(self.modelname, 'model_e{epoch:02d}_v{val_loss:.4f}.h5')
        self.history = self.model.fit(
                        self.train_x,
                        self.train_y,
                        batch_size=1024,
                        epochs=self.num_epochs,
                        shuffle=True,
                        verbose=1,
                        callbacks=[TerminateOnNaN(), ModelCheckpoint(self.intermediatemodelpath)],
                        validation_data=(self.val_x, self.val_y))
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
                X[i, :, 0] = scaleddata[i:i + self.window_size]
                X[i, :, 1] = badpts[i:i + self.window_size]
        else:
            X = np.zeros(((N_pts - self.window_size - 1), self.window_size, 1))
            for i in range(X.shape[0]):
                X[i, :, 0] = scaleddata[i:i + self.window_size]
        
        Y = self.model.predict(X)
        for i in range(X.shape[0]):
            predicteddata[i:i + self.window_size] += Y[i, :, 0]
        
        weightarray[:] = self.window_size
        weightarray[0:self.window_size] = np.linspace(1.0, self.window_size, self.window_size, endpoint=False)
        weightarray[-(self.window_size + 1):-1] = np.linspace(self.window_size, 1.0, self.window_size, endpoint=False)
        return initscale * predicteddata / weightarray


class cnn(dlfilter):
    def __init__(self, num_filters=10, kernel_size=5, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.infodict['nettype'] = 'cnn'
        self.infodict['num_filters'] = self.num_filters
        self.infodict['kernel_size'] = self.kernel_size
        super(cnn, self).__init__(*args, **kwargs)


    def getname(self):
        self.modelname = '_'.join([  'model',
                            'cnn',
                            'w' + str(self.window_size),
                            'l' + str(self.num_layers),
                            'fn' + str(self.num_filters),
                            'fl' + str(self.kernel_size),
                            'd' + str(self.dropout_rate),
                            'e' + str(self.num_epochs),
                            't' + str(self.excludethresh),
                            self.activation])
        if self.usebadpts:
            self.modelname += '_usebadpts'
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass


    def makenet(self):
        self.model = Sequential()

        # make the input layer
        self.model.add(Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same', input_shape=(None, self.inputsize)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Activation(self.activation))
    
        # make the intermediate layers
        for layer in range(self.num_layers - 2):
            self.model.add(Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=self.dropout_rate))
            self.model.add(Activation(self.activation))
    
        # make the output layer
        self.model.add(Convolution1D(filters=self.inputsize, kernel_size=self.kernel_size, padding='same'))
        self.model.compile(optimizer=RMSprop(),
                            loss='mse')


class lstm(dlfilter):
    def __init__(self, num_units=16, *args, **kwargs):
        self.num_units = num_units
        self.infodict['nettype'] = 'cnn'
        self.infodict['num_units'] = self.num_units
        super(lstm, self).__init__(*args, **kwargs)


    def getname(self):
        self.modelname = '_'.join([  'model',
                            'lstm',
                            'w' + str(self.window_size),
                            'l' + str(self.num_layers),
                            'nu' + str(self.num_units),
                            'd' + str(self.dropout_rate),
                            'rd' + str(self.dropout_rate),
                            'e' + str(self.num_epochs),
                            't' + str(self.excludethresh)
                            ])
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass


    def makenet(self):
        self.model = Sequential()

        # each layer consists of an LSTM followed by a dense time distributed layer to get it back to the window size
        for layer in range(self.num_layers):
            self.model.add(Bidirectional(LSTM(self.num_units,
                     dropout=self.dropout_rate,
                     recurrent_dropout=self.dropout_rate,
                     return_sequences=True),
                     input_shape=(self.window_size, 1)))
            self.model.add(TimeDistributed(Dense(1)))

        self.model.compile(optimizer='adam',
                            loss='mse')


class hybrid(dlfilter):
    def __init__(self, invert=False, num_units=16, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.invert = invert
        self.num_units = num_units
        self.infodict['nettype'] = 'hybrid'
        self.infodict['num_filters'] = self.num_filters
        self.infodict['kernel_size'] = self.kernel_size
        self.infodict['invert'] = self.invert
        self.infodict['num_units'] = self.num_units
        super(hybrid, self).__init__(*args, **kwargs)


    def getname(self):
        self.modelname = '_'.join([  'model',
                            'hybrid',
                            'w' + str(self.window_size),
                            'l' + str(self.num_layers),
                            'fn' + str(self.num_filters),
                            'fl' + str(self.kernel_size),
                            'nu' + str(self.num_units),
                            'd' + str(self.dropout_rate),
                            'rd' + str(self.dropout_rate),
                            'e' + str(self.num_epochs),
                            't' + str(self.excludethresh),
                            self.activation])
        if self.invert:
            self.modelname += '_invert'
        self.modelpath = os.path.join(self.modelroot, self.modelname)

        try:
            os.makedirs(self.modelpath)
        except OSError:
            pass

    def makenet(self):
        self.model = Sequential()

        if self.invert:
            # make the input layer
            self.model.add(Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same', input_shape=(self.window_size, self.inputsize)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=self.dropout_rate))
            self.model.add(Activation(self.activation))
    
            # then make make the intermediate CNN layers
            for layer in range(self.num_layers - 2):
                self.model.add(Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same'))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(rate=self.dropout_rate))
                self.model.add(Activation(self.activation))
    
            # finish with an LSTM layer to find hidden states
            self.model.add(Bidirectional(LSTM(self.num_units,
                         dropout=self.dropout_rate,
                         recurrent_dropout=self.dropout_rate,
                         return_sequences=True),
                         input_shape=(self.window_size, 1)))
            self.model.add(TimeDistributed(Dense(1)))
    
        else:
            # start with an LSTM layer to find hidden states
            self.model.add(Bidirectional(LSTM(self.num_units,
                         dropout=self.dropout_rate,
                         recurrent_dropout=self.dropout_rate,
                         return_sequences=True),
                        input_shape=(self.window_size, 1)))
            self.model.add(TimeDistributed(Dense(1)))
            self.model.add(Dropout(rate=self.dropout_rate))
    
            # then make make the intermediate CNN layers
            for layer in range(self.num_layers - 2):
                self.model.add(Convolution1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same'))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(rate=self.dropout_rate))
                self.model.add(Activation(self.activation))
    
            # make the output layer
            self.model.add(Convolution1D(filters=self.inputsize, kernel_size=self.kernel_size, padding='same'))
    
        self.model.compile(optimizer=RMSprop(),
                        loss='mse')
