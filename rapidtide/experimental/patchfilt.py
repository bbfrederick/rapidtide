#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# Generate synthetic noisy pseudoperiodic signal
def generate_signal(n_samples=1000, noise_level=0.1):
    """
    Generate a noisy pseudoperiodic signal.

    Parameters
    ----------
    n_samples : int, optional
        Number of synthetic signal samples.
    noise_level : float, optional
        Amplitude of the noise added to the synthetic signal.

    Returns
    -------
    numpy.ndarray
        Synthetic signal with additive noise.
    """
    t = np.linspace(0, 10 * np.pi, n_samples)
    # Multiple frequency components with slowly varying amplitude
    signal = np.sin(t) + 0.5 * np.sin(2.3 * t) + 0.3 * np.sin(0.5 * t) * np.cos(0.1 * t)
    noise = np.random.randn(n_samples) * noise_level
    return signal + noise


# Create sequences for training
def create_sequences(data, seq_length):
    """
    Create input-output pairs from time series data.

    Parameters
    ----------
    data : array-like
        Input data to filter.
    seq_length : int
        Number of time steps per training sequence.

    Returns
    -------
    tuple of torch.Tensor
        Input sequences and their next-sample targets.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return torch.FloatTensor(X), torch.FloatTensor(y)


# Define LSTM-based predictor model
class SignalPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize `SignalPredictor`.

        Parameters
        ----------
        input_size : int, optional
            Number of features per time step.
        hidden_size : int, optional
            Number of features in the hidden state.
        num_layers : int, optional
            Number of layers in the model.
        dropout : float, optional
            Dropout probability.
        """
        super(SignalPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        """
        Run a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor passed to the model.

        Returns
        -------
        torch.Tensor
            Model output for the input tensor.
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_out = lstm_out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction


# Training function
def train_model(model, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    """
    Train the signal prediction model.

    Parameters
    ----------
    model : object
        Model instance or model file used by this routine.
    X_train : array-like
        Training input sequences.
    y_train : array-like
        Training targets.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Number of training examples per batch.
    lr : float, optional
        Optimizer learning rate.

    Returns
    -------
    object
        The losses.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    return losses


# Prediction function
def predict_next_point(model, sequence):
    """
    Predict the next point given a sequence.

    Parameters
    ----------
    model : object
        Model instance or model file used by this routine.
    sequence : array-like
        Recent samples used to predict the next point.

    Returns
    -------
    float
        Predicted value of the next sample.
    """
    model.eval()
    with torch.no_grad():
        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0).unsqueeze(-1)
        elif len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(-1)
        prediction = model(sequence)
    return prediction.item()


# Multi-step prediction
def predict_future(model, initial_sequence, n_steps):
    """
    Predict multiple future points.

    Parameters
    ----------
    model : object
        Model instance or model file used by this routine.
    initial_sequence : array-like
        Input samples used to seed autoregressive prediction.
    n_steps : int
        Number of future samples to predict.

    Returns
    -------
    object
        The predictions.
    """
    model.eval()
    predictions = []
    current_seq = initial_sequence.clone()

    for _ in range(n_steps):
        next_val = predict_next_point(model, current_seq)
        predictions.append(next_val)
        # Update sequence: remove first element, append prediction
        current_seq = torch.cat([current_seq[1:], torch.FloatTensor([next_val])])

    return predictions


# Main execution
if __name__ == "__main__":
    # Configuration
    SEQ_LENGTH = 50
    TRAIN_SIZE = 800

    # Generate data
    print("Generating signal...")
    signal = generate_signal(n_samples=1000, noise_level=0.15)

    # Create sequences
    X, y = create_sequences(signal, SEQ_LENGTH)
    X = X.unsqueeze(-1)  # Add feature dimension

    # Train-test split
    X_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
    X_test, y_test = X[TRAIN_SIZE:], y[TRAIN_SIZE:]

    # Initialize model
    print("\nInitializing model...")
    model = SignalPredictor(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)

    # Train
    print("\nTraining model...")
    losses = train_model(model, X_train, y_train, epochs=100, batch_size=32, lr=0.001)

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze()
        test_loss = nn.MSELoss()(test_pred, y_test)
        print(f"Test Loss: {test_loss.item():.6f}")

    # Predict future points
    print("\nPredicting future...")
    last_sequence = torch.FloatTensor(signal[TRAIN_SIZE : TRAIN_SIZE + SEQ_LENGTH])
    future_pred = predict_future(model, last_sequence, n_steps=100)

    # Visualize results
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal, label="Original Signal", alpha=0.7)
    start_idx = TRAIN_SIZE + SEQ_LENGTH
    plt.plot(
        range(start_idx, start_idx + len(future_pred)),
        future_pred,
        label="Predictions",
        color="red",
        linewidth=2,
    )
    plt.axvline(x=TRAIN_SIZE, color="green", linestyle="--", label="Train/Test Split")
    plt.title("Signal Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nDone! Model can predict next points using predict_next_point() function.")
