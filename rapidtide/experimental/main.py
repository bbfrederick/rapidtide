#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 23:09:24 2018

@author: neuro
"""

import cnn
import lstm
import numpy as np
import sys
import json
import io as tide_io

num_epochs = 10
thewindow_sizes = [128]
thelayer_nums = [4]
thefilter_nums = [64]
thefilter_lengths = [5]
thedropout_rates = [0.3]
dofft = False
nettype = 'cnn'

infodict = {}
infodict['window_size'] = thewindow_sizes[0]
if dofft:
    infodict['dofft'] = 1
else:
    infodict['dofft'] = 0
infodict['nettype'] = nettype
infodict['lag'] = 0
tide_io.writedict(infodict, 'model_meta')


if sys.platform == 'darwin':
    thedatadir = '/Users/frederic/Documents/MR_data/physioconn/timecourses'
else:
    thedatadir = '/data1/frederic/test/output'

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
                    if nettype == 'cnn':
                        loss[c1, c2, c3, c4, c5, :], loss_val[c1, c2, c3, c4, c5, :] = cnn.cnn(window_size,
                                                                                           num_layers,
                                                                                           num_filters,
                                                                                           filter_length,
                                                                                           dropout_rate,
                                                                                           num_epochs,
                                                                                           thesuffix='25.0Hz',
                                                                                           dofft=dofft,
                                                                                           thedatadir=thedatadir)

                    elif nettype == 'lstm':
                        loss[c1, c2, c3, c4, c5, :], loss_val[c1, c2, c3, c4, c5, :] = lstm.lstm(window_size,
                                                                                           num_layers,
                                                                                           num_filters,
                                                                                           filter_length,
                                                                                           dropout_rate,
                                                                                           num_epochs,
                                                                                           thesuffix='25.0Hz',
                                                                                           thedatadir=thedatadir)
                    else:
                        print('unknown network type:', nettype)
                        sys.exit()

np.save('loss.npy', loss)
np.save('loss_val.npy', loss_val)
