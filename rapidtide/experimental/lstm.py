#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 23:01:07 2018

@author: neuro
"""

import matplotlib.pyplot as plt
import dataload as dl
import numpy as np

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Bidirectional, Convolution1D, Dense, Activation, Dropout, BatchNormalization, LSTM


def lstm(window_size, num_layers, num_units, filter_length, dropout_prob, num_epochs,
        thesuffix='sliceres',
        thedatadir='/data1/frederic/test/output'):
    folder = './batch/'
    #    train_x, train_y, val_x, val_y, Ns = dl.prep_ind(w)
    print('cnn - loading data')
    train_x, train_y, val_x, val_y, Ns = dl.prep(window_size, thesuffix=thesuffix, thedatadir=thedatadir)
    model = Sequential()

    model.add(LSTM(num_units , activation='tanh', input_shape=(window_size + 1, 1), recurrent_activation='hard_sigmoid'))
    model.add(Dropout(rate=dropout_prob))
    model.add(Dense(units=window_size, activation='linear'))
    model.compile (loss ="mean_squared_error" , optimizer="adam")  
    history = model.fit(train_x, train_y,
                        batch_size=1024,
                        nb_epoch=num_epochs,
                        shuffle=False,
                        validation_data=(val_x, val_y))

    # save the model structure to a json file
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('model_whole.h5')

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
