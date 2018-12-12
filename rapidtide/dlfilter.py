#!/usr/bin/env python
# 
#   Copyright 2016 Blaise Frederick
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
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 23:01:07 2018

@author: neuro
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from statsmodels.robust.scale import mad
import glob
from scipy import fftpack

try:
    import pyfftw

    pyfftwexists = True
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()
except ImportError:
    pyfftwexists = False

import rapidtide.io as tide_io

try:
    import plaidml.keras
    plaidml.keras.install_backend("plaidml")
except:
    pass

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Bidirectional, Convolution1D, Dense, Activation, Dropout, BatchNormalization, MaxPooling1D, LSTM, TimeDistributed
from keras.callbacks import TerminateOnNaN, ModelCheckpoint
from keras.models import load_model


class dlfilter:
    """Base class for deep learning filter"""
    thesuffix = 'sliceres'
    thedatadir = '/Users/frederic/Documents/MR_data/physioconn/timecourses'
    inputfrag='cardfromfmri'
    targetfrag='normpleth'
    modelroot = '.'
    excludethresh = 4.0
    modelname = None
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
    modelpath = None
    modelname = '.'
    inputsize = None
    infodict = {}


    def __init__(self,
        window_size=128,
        num_layers=5,
        dropout_rate=0.3,
        num_epochs=1,
        activation='relu',
        modelroot='.',
        dofft=False,
        debug=False,
        excludethresh=4.0,
        usebadpts=False,
        thesuffix='25.0Hz',
        modelpath='.',
        thedatadir='/Users/frederic/Documents/MR_data/physioconn/timecourses',
        inputfrag='cardfromfmri',
        targetfrag='normpleth',
        readlim=None,
        countlim=None):

        self.window_size = window_size
        self.dropout_rate = dropout_rate
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
        self.debug = debug
        self.thesuffix = thesuffix
        self.thedatadir = thedatadir
        self.modelpath = modelpath
        print('modeldir from dlfilter:', self.modelpath)
        self.excludethresh = excludethresh
        self.readlim = readlim
        self.countlim = countlim
        self.model = None
        self.initialized = False
        self.trained = False
        self.usetensorboard = False
        self.inputfrag = inputfrag
        self.targetfrag = targetfrag

        # populate infodict
        self.infodict['window_size'] = self.window_size
        self.infodict['usebadpts'] = self.usebadpts
        self.infodict['dofft'] = self.dofft
        self.infodict['excludethresh'] = self.excludethresh
        self.infodict['num_epochs'] = self.num_epochs
        self.infodict['modelname'] = self.modelname
        self.infodict['dropout_rate'] = self.dropout_rate
        self.infodict['train_arch'] = sys.platform


    def loaddata(self):
        if not self.initialized:
            print('model must be initialized prior to loading data')
            sys.exit()

        if self.dofft:
            self.train_x, self.train_y, self.val_x, self.val_y, self.Ns, self.tclen, self.thebatchsize, dummy, dummy = prep(self.window_size,
                                                                        thesuffix=self.thesuffix,
                                                                        thedatadir=self.thedatadir,
                                                                        inputfrag=self.inputfrag,
                                                                        targetfrag=self.targetfrag,
                                                                        dofft=self.dofft,
                                                                        debug=self.debug,
                                                                        usebadpts=self.usebadpts,
                                                                        excludethresh=self.excludethresh,
                                                                        readlim=self.readlim,
                                                                        countlim=self.countlim)
        else:
            self.train_x, self.train_y, self.val_x, self.val_y, self.Ns, self.tclen, self.thebatchsize = prep(self.window_size,
                                                                        thesuffix=self.thesuffix,
                                                                        thedatadir=self.thedatadir,
                                                                        inputfrag=self.inputfrag,
                                                                        targetfrag=self.targetfrag,
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
    
        plt.figure()
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
        print('loading', modelname)

        # load in the model with weights
        self.model = load_model(os.path.join(self.modelpath, modelname, 'model.h5'))
        self.model.summary()

        # now load additional information
        self.infodict = tide_io.readdictfromjson(os.path.join(self.modelpath, modelname, 'model_meta.json'))
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
        if self.usetensorboard:
            tensorboard = TensorBoard(log_dir=self.intermediatemodelpath + "logs/{}".format(time()))
            model.fit(x_train, y_train, verbose=1, callbacks=[tensorboard])
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
    def __init__(self, invert=False, num_filters=10, kernel_size=5, num_units=16, *args, **kwargs):
        self.invert = invert
        self.num_filters = num_filters
        self.kernel_size = kernel_size
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






def filtscale(data, scalefac=1.0, reverse=False, hybrid=False, lognormalize=True, epsilon=1e-10, numorders=6):
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
            return np.stack((thedata, themag), axis=1), scalefac
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
            return  fftpack.ifft(specvals).real

def tobadpts(name):
    return name.replace('.txt', '_badpts.txt')


def targettoinput(name, targetfrag='normpleth', inputfrag='cardfromfmri'):
    return name.replace(targetfrag, inputfrag)


def getmatchedfiles(searchstring, usebadpts=False, targetfrag='normpleth', inpufrag='cardfromfmri'):
    fromfile = sorted(glob.glob(searchstring))

    # make sure all files exist
    matchedfilelist = []
    for targetname in fromfile:
        if os.path.isfile(targettoinput(targetname)):
            if usebadpts:
                if os.path.isfile(tobadpts(targetname.replace('normpleth', 'pleth'))) \
                    and os.path.isfile(tobadpts(targettoinput(targetname))):
                    matchedfilelist.append(targetname)
                    print(matchedfilelist[-1])
            else:
                matchedfilelist.append(targetname)
                print(matchedfilelist[-1])
    if usebadpts:
        print(len(matchedfilelist), 'runs pass all 4 files present check')
    else:
        print(len(matchedfilelist), 'runs pass both files present check')

    # find out how long the files are
    tempy = np.loadtxt(matchedfilelist[0])
    tempx = np.loadtxt(targettoinput(matchedfilelist[0]))
    tclen = np.min([tempx.shape[0], tempy.shape[0]])
    print('tclen set to', tclen)
    return matchedfilelist, tclen


def readindata(matchedfilelist, tclen, usebadpts=False, startskip=0, readlim=None):

    print('readindata called with usebadpts, startskip, readlim =', usebadpts, startskip, readlim)
    # allocate target arrays
    print('allocating arrays')
    s = len(matchedfilelist)
    if readlim is not None:
        if s > readlim:
            print('trimming read list to', readlim, 'from', s)
            s = readlim
    x1 = np.zeros([tclen, s])
    y1 = np.zeros([tclen, s])
    names = []
    if usebadpts:
        bad1 = np.zeros([tclen, s])

    # now read the data in
    count = 0
    print('checking data')
    for i in range(s):
        print('processing ', matchedfilelist[i])
        tempy = np.loadtxt(matchedfilelist[i])
        tempx = np.loadtxt(targettoinput(matchedfilelist[i]))
        ntempx = tempx.shape[0]
        ntempy = tempy.shape[0]
        if (ntempx >= tclen) and (ntempy >= tclen):
            x1[:tclen, count] = tempx[:tclen]
            y1[:tclen, count] = tempy[:tclen]
            names.append(matchedfilelist[i])
            if usebadpts:
                tempbad1 = np.loadtxt(tobadpts(matchedfilelist[i].replace('normpleth', 'pleth')))
                tempbad2 = np.loadtxt(tobadpts(targettoinput(matchedfilelist[i])))
                bad1[:tclen, count] = 1.0 - (1.0 - tempbad1[:tclen]) * (1.0 - tempbad2[:tclen])
            count += 1
    print(count, 'runs pass file length check')

    if usebadpts:
        return x1[startskip:, :count], y1[startskip:, :count], names[:count], bad1[startskip:, :count]
    else:
        return x1[startskip:, :count], y1[startskip:, :count], names[:count]


def prep(window_size,
        step=1,
        excludethresh=4.0,
        usebadpts=False,
        startskip=200,
        endskip=0,
        excludesubject=True,
        thesuffix='sliceres',
        thedatadir='/data1/frederic/test/output',
        inputfrag='cardfromfmri',
        targetfrag='normpleth',
        dofft=False,
        debug=False,
        readlim=None,
        countlim=None):
    '''
    prep - reads in training and validation data for 1D filter

    Parameters
    ----------
    window_size
    step
    excludethresh
    usebadpts
    startskip
    endskip
    thesuffix
    thedatadir
    inputfrag
    targetfrag
    dofft
    debug
    readlim
    countlim

    Returns
    -------
    train_x, train_y, val_x, val_y, N_subjs, tclen - startskip, batchsize

    '''

    searchstring = os.path.join(thedatadir, '*normpleth_' + thesuffix + '.txt')

    # find matched files
    matchedfilelist, tclen = getmatchedfiles(searchstring, usebadpts=usebadpts, targetfrag=targetfrag, inpufrag=inputfrag)

    # read in the data from the matched files
    if usebadpts:
        x, y, names, bad = readindata(matchedfilelist, tclen, usebadpts=True, startskip=startskip, readlim=readlim)
    else:
        x, y, names = readindata(matchedfilelist, tclen, startskip=startskip, readlim=readlim)
    print('xshape, yshape:', x.shape, y.shape)

    # normalize input and output data
    print('normalizing data')
    print('count:', x.shape[1])
    if debug:
        for thesubj in range(x.shape[1]):
            print('prenorm sub', thesubj, 'min, max, mean, std, MAD x, y:', thesubj,
                  np.min(x[:, thesubj]), np.max(x[:, thesubj]), np.mean(x[:, thesubj]), np.std(x[:, thesubj]), mad(x[:, thesubj]),
                  np.min(y[:, thesubj]), np.max(y[:, thesubj]), np.mean(y[:, thesubj]), np.std(x[:, thesubj]), mad(y[:, thesubj]))

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

    if debug:
        for thesubj in range(x.shape[1]):
            print('postnorm sub', thesubj, 'min, max, mean, std, MAD x, y:', thesubj,
                  np.min(x[:, thesubj]), np.max(x[:, thesubj]), np.mean(x[:, thesubj]), np.std(x[:, thesubj]), mad(x[:, thesubj]),
                  np.min(y[:, thesubj]), np.max(y[:, thesubj]), np.mean(y[:, thesubj]), np.std(x[:, thesubj]), mad(y[:, thesubj]))

    thefabs = np.fabs(x)
    themax = np.max(thefabs, axis=0)

    if excludesubject:
        pass

    cleansubjs = np.where(themax < excludethresh)[0]

    cleancount = len(cleansubjs)
    if countlim is not None:
        if cleancount > countlim:
            print('reducing count to', countlim, 'from', cleancount)
            cleansubjs = cleansubjs[:countlim]

    x = x[:, cleansubjs]
    y = y[:, cleansubjs]
    cleannames = []
    for theindex in cleansubjs:
        cleannames.append(names[theindex])
    if usebadpts:
        bad = bad[:, cleansubjs]

    print('after filtering, shape of x is', x.shape)

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

    Xb = np.zeros((N_subjs * (N_pts - window_size - 1), window_size, 1))
    print('dimensions of Xb:', Xb.shape)
    for j in range(N_subjs):
        print('sub', j, '(', cleannames[j], '), min, max X, Y:', j, np.min(X[0, :, j]), np.max(X[0, :, j]), np.min(Y[0, :, j]),
              np.max(Y[0, :, j]))
        for i in range((N_pts - window_size - 1)):
            Xb[j * ((N_pts - window_size - 1)) + i, :, 0] = X[0, step * i:(step * i + window_size), j]

    Yb = np.zeros((N_subjs * (N_pts - window_size - 1), window_size, 1))
    print('dimensions of Yb:', Yb.shape)
    for j in range(N_subjs):
        for i in range((N_pts - window_size - 1)):
            Yb[j * ((N_pts - window_size - 1)) + i, :, 0] = Y[0, step * i:(step * i + window_size), j]

    if usebadpts:
        Xb_withbad = np.zeros((N_subjs * (N_pts - window_size - 1), window_size, 2))
        print('dimensions of Xb_withbad:', Xb_withbad.shape)
        for j in range(N_subjs):
            print('packing data for subject',j)
            for i in range((N_pts - window_size - 1)):
                Xb_withbad[j * ((N_pts - window_size - 1)) + i, :, 0] = \
                    X[0, step * i:(step * i + window_size), j]
                Xb_withbad[j * ((N_pts - window_size - 1)) + i, :, 1] = \
                    BAD[0, step * i:(step * i + window_size), j]
        Xb = Xb_withbad
    
    perm = np.arange(Xb.shape[0])

    if dofft:
        Xb_fourier = np.zeros((N_subjs * (N_pts - window_size - 1), window_size, 2))
        print('dimensions of Xb_fourier:', Xb_fourier.shape)
        Xscale_fourier = np.zeros((N_subjs, N_pts - window_size - 1))
        print('dimensions of Xscale_fourier:', Xscale_fourier.shape)
        Yb_fourier = np.zeros((N_subjs * (N_pts - window_size - 1), window_size, 2))
        print('dimensions of Yb_fourier:', Yb_fourier.shape)
        Yscale_fourier = np.zeros((N_subjs, N_pts - window_size - 1))
        print('dimensions of Yscale_fourier:', Yscale_fourier.shape)
        for j in range(N_subjs):
            print('transforming subject',j)
            for i in range((N_pts - window_size - 1)):
                Xb_fourier[j * ((N_pts - window_size - 1)) + i, :, :], Xscale_fourier[j, i] = \
                    filtscale(X[0, step * i:(step * i + window_size), j])
                Yb_fourier[j * ((N_pts - window_size - 1)) + i, :, :], Yscale_fourier[j, i] = \
                    filtscale(Y[0, step * i:(step * i + window_size), j])
    
    perm = np.arange(Xb.shape[0])
    limit = int(0.8 * Xb.shape[0])

    batchsize = N_pts - window_size - 1

    if dofft:
        train_x = Xb_fourier[perm[:limit], :, :]
        train_y = Yb_fourier[perm[:limit], :, :]

        val_x = Xb_fourier[perm[limit:], :, :]
        val_y = Yb_fourier[perm[limit:], :, :]
        print('train, val dims:', train_x.shape, train_y.shape, val_x.shape, val_y.shape)
        return train_x, train_y, val_x, val_y, N_subjs, tclen - startskip, batchsize, Xscale_fourier, Yscale_fourier
    else:
        train_x = Xb[perm[:limit], :, :]
        train_y = Yb[perm[:limit], :, :]

        val_x = Xb[perm[limit:], :, :]
        val_y = Yb[perm[limit:], :, :]
        print('train, val dims:', train_x.shape, train_y.shape, val_x.shape, val_y.shape)
        return train_x, train_y, val_x, val_y, N_subjs, tclen - startskip, batchsize
