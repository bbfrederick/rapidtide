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


def cnn(window_size, num_layers, num_filters, kernel_size, dropout_prob, num_epochs,
        thesuffix='sliceres',
        thedatadir='/data1/frederic/test/output',
        modelname='model',
        dofft=False):
    folder = './batch/'

    print('cnn - loading data')
    if dofft:
        train_x, train_y, val_x, val_y, Ns, tclen, dummy, dummy = dl.prep(window_size, thesuffix=thesuffix, thedatadir=thedatadir, dofft=True)
    else:
        train_x, train_y, val_x, val_y, Ns, tclen = dl.prep(window_size, thesuffix=thesuffix, thedatadir=thedatadir, dofft=False)
    model = Sequential()

    print('data shape:', train_x.shape)
    model.add(Convolution1D(filters=num_filters, kernel_size=kernel_size, padding='same', input_shape=(None, train_x.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_prob))
    model.add(Activation('relu'))
    #model.add(MaxPooling1D())

    # make the intermediate layers
    for layer in range(num_layers - 2):
        model.add(Convolution1D(filters=num_filters, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_prob))
        model.add(Activation('relu'))
        #model.add(MaxPooling1D())

    # make the output layer
    model.add(Convolution1D(filters=train_y.shape[2], kernel_size=kernel_size, padding='same'))
    #model.add(Dense(units=window_size, input_shape=((train_x.shape[2],)), activation='linear'))

    model.summary()
    model.compile(optimizer=RMSprop(), loss='mse')
    history = model.fit(train_x, train_y,
                        batch_size=1024,
                        epochs=num_epochs,
                        shuffle=True,
                        verbose=1,
                        validation_data=(val_x, val_y))

    # save the model structure to a json file
    model_json = model.to_json()
    with open(modelname + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(modelname + '_weights.h5')

    YPred = model.predict(val_x)

    error = val_y - YPred
    sq_error = (np.mean(np.square(error)))

    error2 = val_x - val_y
    sq_error2 = (np.mean(np.square(error2)))
    description = ' '.join([
        'Num layers: ', str(num_layers),
        'Num filters: ', str(num_filters),
        'Dropout prob: ', str(dropout_prob),
        'Window size: ', str(window_size)
    ])
    print(description)
    print('Prediction Error: ', sq_error, 'Raw Error: ', sq_error2)

    f = open("loss.txt", "a")
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
    plt.savefig(
        folder + 'loss' + '_layer_' + str(num_layers) + '_filter_num_' + str(num_filters) + '_dropout_prob_' + str(
            dropout_prob) + '_window_size_' + str(window_size) + '.png')
    plt.close()

    # print('loss, val_loss', loss, val_loss)
    return loss, val_loss
