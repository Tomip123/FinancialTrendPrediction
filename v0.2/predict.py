from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from config import MODELS_DIR, DATA_DIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


lookback = 300

# load the model
model = load_model(os.path.join(MODELS_DIR, 'best_model.h5'))

# load the latest normalized data from the csv file
all_stock_data = pd.read_csv(os.path.join(DATA_DIR, 'stock_data_normalized.csv'), index_col=0)

recent_data = all_stock_data.iloc[-lookback:]

# Create sequences from the recent data
recent_sequences = np.array([recent_data[-lookback:]])

# Use the model to predict the future movements of the index
recent_sequences = np.array(recent_sequences).astype(np.float32)
predicted_movements = model.predict(recent_sequences)

print(predicted_movements.shape)

print(predicted_movements)

# Plot the predicted movements
plt.plot(predicted_movements[0], label='Predicted')
plt.title('Predicted Index Movements')
plt.xlabel('Days')
plt.ylabel('Index Movement')
plt.legend()
plt.show()