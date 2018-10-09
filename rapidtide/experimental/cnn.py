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

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Bidirectional, Convolution1D, Dense, Activation, Dropout, BatchNormalization, MaxPooling1D
from keras.callbacks import TerminateOnNaN, ModelCheckpoint


def cnn(window_size, num_layers, num_filters, kernel_size, dropout_rate, num_epochs,
        thesuffix='sliceres',
        thedatadir='/data1/frederic/test/output',
        excludethresh=4.0,
        modelname='model',
        usebadpts=False,
        activation='relu',
        dofft=False,
        readlim=None,
        countlim=None):

    folder = 'batch'
    lossfilename = os.path.join(modelname, 'loss.png')
    print('lossfilename:', lossfilename)

    print('cnn - loading data')
    if dofft:
        train_x, train_y, val_x, val_y, Ns, tclen, thebatchsize, dummy, dummy = dl.prep(window_size,
                                                                                    thesuffix=thesuffix,
                                                                                    thedatadir=thedatadir,
                                                                                    dofft=True,
                                                                                    usebadpts=usebadpts,
                                                                                    excludethresh=excludethresh,
                                                                                    readlim=readlim,
                                                                                    countlim=countlim)
    else:
        train_x, train_y, val_x, val_y, Ns, tclen, thebatchsize = dl.prep(window_size,
                                                                        thesuffix=thesuffix,
                                                                        thedatadir=thedatadir,
                                                                        dofft=False,
                                                                        usebadpts=usebadpts,
                                                                        excludethresh=excludethresh,
                                                                        readlim=readlim,
                                                                        countlim=countlim)
    model = Sequential()

    print('data shape:', train_x.shape)
    model.add(Convolution1D(filters=num_filters, kernel_size=kernel_size, padding='same', input_shape=(None, train_x.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    model.add(Activation(activation))
    #model.add(MaxPooling1D())

    # make the intermediate layers
    for layer in range(num_layers - 2):
        model.add(Convolution1D(filters=num_filters, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_rate))
        model.add(Activation(activation))
        #model.add(MaxPooling1D())

    # make the output layer
    model.add(Convolution1D(filters=train_y.shape[2], kernel_size=kernel_size, padding='same'))
    #model.add(Dense(units=window_size, input_shape=((train_x.shape[2],)), activation='linear'))

    model.summary()
    model.compile(optimizer=RMSprop(), loss='mse')
    modelpath = os.path.join(modelname, 'model_e{epoch:02d}_v{val_loss:.4f}.h5')
    history = model.fit(train_x, train_y,
                        batch_size=1024,
                        epochs=num_epochs,
                        shuffle=True,
                        verbose=1,
                        callbacks=[TerminateOnNaN(), ModelCheckpoint(modelpath)],
                        validation_data=(val_x, val_y))

    # save the trained model
    model.save(os.path.join(modelname, 'model.h5'))

    YPred = model.predict(val_x)

    error = val_y - YPred
    sq_error = (np.mean(np.square(error)))

    error2 = val_x - val_y
    sq_error2 = (np.mean(np.square(error2)))
    description = ' '.join([
        'Num layers: ', str(num_layers),
        'Num filters: ', str(num_filters),
        'Dropout prob: ', str(dropout_rate),
        'Window size: ', str(window_size)
    ])
    print(description)
    print('Prediction Error: ', sq_error, 'Raw Error: ', sq_error2)

    f = open(os.path.join(modelname, "loss.txt"), "a")
    f.write(description + '\n')
    f.write('Prediction Error: ' + str(sq_error) + ' Raw Error: ' + str(sq_error2) + '\n')
    f.close()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(lossfilename)
    plt.close()

    return loss, val_loss, sq_error, sq_error2
