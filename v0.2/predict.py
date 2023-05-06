import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
from config import MODELS_DIR, DATA_DIR

# Define the number of previous data points to use for prediction
lookback = 1000

# Load the trained model
model = load_model(os.path.join(MODELS_DIR, 'best_model_1.h5'))

# Load the latest normalized data from the csv file
all_stock_data = pd.read_csv(os.path.join(DATA_DIR, 'stock_data_normalized.csv'), index_col=0)

# Load the scaler from the saved file
with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Get the most recent data
recent_data = all_stock_data.iloc[-lookback:]

# Create sequences from the recent data
recent_sequences = np.array([recent_data[-lookback:]])

# Use the model to predict the future movements of the index
recent_sequences = np.array(recent_sequences).astype(np.float32)
predicted_movements = model.predict(recent_sequences)

# Inverse transform the predicted movements using the scaler
predicted_movements = scaler.inverse_transform(predicted_movements)

# Define the filter parameters
cutoff_freq = 0.1  # Cutoff frequency in Hz
nyquist_freq = 0.5 * 1  # Nyquist frequency is half the sampling rate (1 Hz in this case)
order = 2  # Filter order

# Calculate the filter coefficients
b, a = butter(order, cutoff_freq/nyquist_freq, btype='low')

# Apply the filter to the predicted values
filtered_movements = filtfilt(b, a, predicted_movements.squeeze())

# Divide the predicted movements by the first value to make the graph start from 1
normalized_movements = predicted_movements[0] / predicted_movements[0][0]
filtered_movements = filtered_movements / filtered_movements[0]

# Plot the predicted movements, the graph should start from number 1
plt.plot(normalized_movements, label='Predictions')
plt.plot(filtered_movements, label='Filtered Predictions')
plt.legend()
plt.show()