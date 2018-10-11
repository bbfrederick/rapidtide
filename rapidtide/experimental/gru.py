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
from keras.layers import Bidirectional, Convolution1D, Dense, Activation, Dropout, BatchNormalization, GRU, TimeDistributed
from keras.callbacks import TerminateOnNaN, ModelCheckpoint


def gru(window_size=128,
        num_layers=3,
        num_units=1,
        filter_length=5,
        dropout_rate=0.3,
        excludethresh=4.0,
        num_epochs=3,
        dofft=False,
        thesuffix='25.0Hz',
        modelname='model',
        debug=False,
        thedatadir='/data1/frederic/test/output',
        readlim=None,
        countlim=None):

    lossfilename = os.path.join(modelname, 'loss.png')
    print('lossfilename:', lossfilename)

    print('lstm - loading data')
    if dofft:
        train_x, train_y, val_x, val_y, Ns, tclen, thebatchsize, dummy, dummy = dl.prep(window_size,
                                                                                      thesuffix=thesuffix,
                                                                                      thedatadir=thedatadir,
                                                                                      excludethresh=excludethresh,
                                                                                      dofft=True,
                                                                                      debug=debug,
                                                                                      readlim=readlim,
                                                                                      countlim=countlim)
    else:
        train_x, train_y, val_x, val_y, Ns, tclen, thebatchsize = dl.prep(window_size,
                                                                          thesuffix=thesuffix,
                                                                          thedatadir=thedatadir,
                                                                          excludethresh=excludethresh,
                                                                          dofft=False,
                                                                          debug=debug,
                                                                          readlim=readlim,
                                                                          countlim=countlim)

    print('dimension of input data', train_x.shape)
    print('dimension of output data', train_y.shape)
    model = Sequential()

    # each layer consists of an GRU followed by a dense time distributed layer to get it back to the window size
    for layer in range(num_layers):
        model.add(Bidirectional(GRU(num_units,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     return_sequences=True),
                     input_shape=(window_size, 1)))
        model.add(TimeDistributed(Dense(1)))

    model.summary()

    # now compile and train
    model.compile (loss ="mean_squared_error" , optimizer="adam")  
    modelpath = os.path.join(modelname, 'model_e{epoch:02d}_v{val_loss:.4f}.h5')
    history = model.fit(train_x, train_y,
                        batch_size=1024,
                        epochs=num_epochs,
                        shuffle=True,
                        callbacks=[TerminateOnNaN(), ModelCheckpoint(modelpath)],
                        validation_data=(val_x, val_y))

    # save the trained model
    model.save(os.path.join(modelname, 'model.h5'))

    print('doing prediction')
    YPred = model.predict(val_x)
    print('prediction finished')

    error = val_y - YPred
    sq_error = (np.mean(np.square(error)))

    error2 = val_x - val_y
    sq_error2 = (np.mean(np.square(error2)))
    description = ' '.join([
        'Num layers: ', str(num_layers),
        'Num units: ', str(num_units),
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

    # print('loss, val_loss', loss, val_loss)
    return loss, val_loss, sq_error, sq_error2
