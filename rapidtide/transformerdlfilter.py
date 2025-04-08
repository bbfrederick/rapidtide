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
"""This module contains all an alternate filter routine"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model


# Positional Encoding to add temporal information to the sequence
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_len, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, sequence_len, d_model):
        angle_rads = self.get_angles(
            tf.range(sequence_len)[:, tf.newaxis], tf.range(d_model)[tf.newaxis, :], d_model
        )

        # Apply sin to even indices in the array; cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]


# Transformer block implementation
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),  # Feed Forward Network
                Dense(d_model),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.att(x, x)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Add & Norm

        ffn_output = self.ffn(out1)  # Feed forward
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm


# Building the Transformer-based Cardiac Waveform Filter model
def build_transformer_model(
    input_shape, d_model=128, num_heads=4, ff_dim=512, num_layers=3, dropout_rate=0.1
):
    inputs = Input(shape=input_shape)

    # Positional Encoding
    x = PositionalEncoding(input_shape[0], d_model)(inputs)

    # Stack Transformer layers
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, rate=dropout_rate)(x)

    # Output layer for waveform filtering (regression to the original signal)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Compile and train the model
def compile_and_train_model(
    model, train_data, train_labels, val_data, val_labels, epochs=50, batch_size=32
):
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=batch_size,
    )
    return history


# Example usage:
# Assuming train_data and train_labels are the time-series data of cardiac signals
input_shape = (1000, 1)  # For example, 1000 time points with 1 feature (cardiac waveform)
transformer_model = build_transformer_model(input_shape)
transformer_model.summary()

# After this, you would use your train_data and val_data (cardiac waveforms) to train the model.
# Example: compile_and_train_model(transformer_model, train_data, train_labels, val_data, val_labels)
