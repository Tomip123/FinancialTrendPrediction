import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to create input-output pairs for the model
def create_dataset(dataset, lookback=60):
    data_x, data_y = [], []
    for i in range(len(dataset) - lookback):
        data_x.append(dataset[i:i + lookback, :])
        data_y.append(dataset[i + lookback, -1])
    return np.array(data_x), np.array(data_y)

# Load the test data
test_data = pd.read_csv('stock_data_test.csv', header=[0, 1], index_col=0)

# Combine the close prices of all symbols
test_data_combined = test_data.xs('close', level=1, axis=1).dropna().values

# Calculate the overall trend by summing the close prices
test_data_trend = np.sum(test_data_combined, axis=1).reshape(-1, 1)

# Concatenate the combined data and trend to form the dataset
test_data_symbol = np.hstack((test_data_combined, test_data_trend))

# Create input-output pairs for the test set
lookback = 60
x_test, y_test = create_dataset(test_data_symbol, lookback)

# Load the best model
best_model = tf.keras.models.load_model('best_complex_model.h5')

# Make predictions on the test set
y_pred = best_model.predict(x_test)

# Reshape y_pred array
y_pred = y_pred.reshape(-1, 1)

print("Predicted values:", y_pred)

# Calculate the Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error (MAPE):", mape)