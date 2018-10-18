#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:01:29 2018

@author: neuro
"""

import matplotlib.pyplot as plt
import numpy as np
import os 
import dataload25 as dl


w=60
train_x, train_y, val_x, val_y, Ns = dl.prep(w)


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Bidirectional

model = Sequential()


model.add(Bidirectional(layers.LSTM(100,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     return_sequences=True),
                     input_shape=(None, 1)))


model.add(layers.TimeDistributed(layers.Dense(1)))

model.compile(optimizer=RMSprop(), loss='mse')

history = model.fit(train_x,train_y,
#                    steps_per_epoch=500,
                    epochs=40,
                    validation_data=(val_x, val_y))
#                    validation_split=0.2)

import matplotlib.pyplot as plt

#YPred=model.predict(X)
YPred2=model.predict(val_x)


#error=Y[0,8480:,0]-YPred[0,8480:,0]
error=YPred2-val_y
sq_error=(np.mean(np.square(error)))

error2=Y[0,8480:,0]-X[0,8480:,0]
sq_error2=(np.mean(np.square(error2)))

print(sq_error)
print(sq_error2)

#plt.plot(YPred[0,:,0])
#plt.plot(Y[0,:,0])
#plt.plot(X[0,:,0])
#plt.legend(['pred', 'groundtruth', 'raw'])
#plt.show()

plt.plot(YPred2[60,:,0])
plt.plot(val_y[60,:,0])
plt.plot(val_x[60,:,0])
#plt.plot(X[0,:,0])
plt.legend(['pred', 'groundtruth', 'raw'])
plt.show()



loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
   