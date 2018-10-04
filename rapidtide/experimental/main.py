#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 23:09:24 2018

@author: neuro
"""

import cnn
import lstm
import numpy as np

num_epochs = 10
thewindow_sizes = [128]
thelayer_nums = [4]
thefilter_nums = [10]
thefilter_lengths = [10]
thedropout_rates = [0.3]
dofft = True

loss = np.zeros(
    [len(thewindow_sizes), len(thelayer_nums), len(thefilter_nums), len(thefilter_lengths), len(thedropout_rates),
     num_epochs])
loss_val = np.zeros(
    [len(thewindow_sizes), len(thelayer_nums), len(thefilter_nums), len(thefilter_lengths), len(thedropout_rates),
     num_epochs])

for c1, window_size in list(enumerate(thewindow_sizes)):
    for c2, num_layers in list(enumerate(thelayer_nums)):
        for c3, num_filters in list(enumerate(thefilter_nums)):
            for c4, filter_length in list(enumerate(thefilter_lengths)):
                for c5, dropout_rate in list(enumerate(thedropout_rates)):
                    # print('layer numbers: ', num_layers,'filter numers: ', num_filters, 'Dropout Prob: ',p, 'window Size: ', window_size)
                    loss[c1, c2, c3, c4, c5, :], loss_val[c1, c2, c3, c4, c5, :] = cnn.cnn(window_size,
                                                                                           num_layers,
                                                                                           num_filters,
                                                                                           filter_length,
                                                                                           dropout_rate,
                                                                                           num_epochs,
                                                                                           thesuffix='25.0Hz',
                                                                                           fft=dofft,
                                                                                           thedatadir='/Users/frederic/Documents/MR_data/physioconn/timecourses')
                    #                                                                       thedatadir='/data1/frederic/test/output')

                    #loss[c1, c2, c3, c4, c5, :], loss_val[c1, c2, c3, c4, c5, :] = lstm.lstm(window_size,
                    #                                                                       num_layers,
                    #                                                                       num_filters,
                    #                                                                       filter_length,
                    #                                                                       dropout_rate,
                    #                                                                       num_epochs,
                    #                                                                       thesuffix='25.0Hz',
                    #                                                                       thedatadir='/data1/frederic/test/output')

np.save('loss.npy', loss)
np.save('loss_val.npy', loss_val)
