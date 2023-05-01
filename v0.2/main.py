import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from utils.data_processing import process_data, normalize_data
from utils.load_data import fetch_stock_data
import os
from config import MODELS_DIR
import numpy as np


# Load stock symbols from JSON file
with open('symbols.json') as f:
    symbols = json.load(f)

def create_sequences(input_data, train_targets, sequence_length):
    sequences = []
    corresponding_targets = []

    for i in range(len(input_data) - sequence_length - 30):
        sequences.append(input_data[i : i + sequence_length])
        corresponding_targets.append(train_targets[i + sequence_length : i + sequence_length + 30])

    return np.array(sequences), np.array(corresponding_targets)


# Load the data from the CSV file in the data folder
all_stock_data = fetch_stock_data(symbols)

# Process and normalize the data
all_stock_data = process_data(all_stock_data)
all_stock_data = normalize_data(all_stock_data)

# After processing and normalizing the data
print("Number of features after data processing: ", all_stock_data.shape[1])

# Split the data into training and validation sets
train_data, val_data = train_test_split(all_stock_data, test_size=0.2, shuffle=False)

# Define the lookback period and batch size
lookback = 300

# Create a new DataFrame for the features, KEEPING the 'Index Movement' column
train_features = train_data.values
val_features = val_data.values

train_targets = train_data['Index Movement'].values
val_targets = val_data['Index Movement'].values

train_sequences, train_sequence_targets = create_sequences(train_features, train_targets, lookback)
val_sequences, val_sequence_targets = create_sequences(val_features, val_targets, lookback)

# save the training and testing data to csv files
train_data.to_csv(os.path.join(MODELS_DIR, 'train_data.csv'))
val_data.to_csv(os.path.join(MODELS_DIR, 'val_data.csv'))

# Train the model
model, history = train_model(train_sequences, train_sequence_targets, val_sequences, val_sequence_targets, num_epochs=10)

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.show()