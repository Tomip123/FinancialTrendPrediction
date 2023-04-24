import numpy as np
import pandas as pd
import tensorflow as tf
from utils.get_data import get_stock_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import os


lookback = 60

# Function to fetch the latest data for all symbols
def fetch_latest_data(symbols):
    outputsize = 'compact'  # Use 'compact' to get the latest 100 data points

    stock_data_dict = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}")
        df = get_stock_data(symbol, outputsize)

        if df is not None:
            stock_data_dict[symbol] = df
        else:
            print(f"Error fetching data for {symbol}")

    # Create a DataFrame with a multi-level column index
    all_stock_data = pd.concat(stock_data_dict, axis=1)

    # Fill missing data points (e.g., weekends, holidays) with the previous day's value
    all_stock_data.fillna(method='ffill', inplace=True)

    return all_stock_data


def preprocess_data(data, symbols, window_size=60):

    # Ensure the input DataFrame has the correct multi-level column structure
    for symbol in symbols:
        if symbol not in data.columns.get_level_values(0):
            print(f"Warning: {symbol} not found in input DataFrame")
            continue
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in data.columns.get_level_values(1):
                print(f"Warning: {col} not found in input DataFrame for {symbol}")
                continue

    # Calculate the moving average for a given window size
    def moving_average(df, window_size):
        return df.rolling(window=window_size).mean()

    # Calculate the relative strength index (RSI) for a given period
    def relative_strength_index(df, period):
        delta = df.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Create new features
    window_size = 14

    # Add moving averages
    for symbol in symbol_list:
        data[(symbol, 'ma')] = moving_average(data[(symbol, 'close')], window_size)

    # Add RSI
    for symbol in symbol_list:
        data[(symbol, 'rsi')] = relative_strength_index(data[(symbol, 'close')], window_size)

    # Normalize the data
    scaler = MinMaxScaler()

    for symbol in symbol_list:
        data[(symbol, 'open')] = scaler.fit_transform(data[(symbol, 'open')].values.reshape(-1, 1))
        data[(symbol, 'high')] = scaler.fit_transform(data[(symbol, 'high')].values.reshape(-1, 1))
        data[(symbol, 'low')] = scaler.fit_transform(data[(symbol, 'low')].values.reshape(-1, 1))
        data[(symbol, 'close')] = scaler.fit_transform(data[(symbol, 'close')].values.reshape(-1, 1))
        data[(symbol, 'volume')] = scaler.fit_transform(data[(symbol, 'volume')].values.reshape(-1, 1))
        data[(symbol, 'ma')] = scaler.fit_transform(data[(symbol, 'ma')].values.reshape(-1, 1))
        data[(symbol, 'rsi')] = scaler.fit_transform(data[(symbol, 'rsi')].values.reshape(-1, 1))

    return data

# Fetch the latest data for all symbols
symbol_list = ['IBM', 'NVDA', 'TSLA', 'NFLX', 'INTC']
latest_data = fetch_latest_data(symbol_list)

print(latest_data)

# Preprocess the data (use your existing preprocessing function)
preprocessed_latest_data = preprocess_data(latest_data, symbol_list)

print(preprocessed_latest_data)

# Combine the close prices of all symbols
latest_data_combined = preprocessed_latest_data.xs('close', level=1, axis=1).dropna().values

# Calculate the overall trend by summing the close prices
latest_data_trend = np.sum(latest_data_combined, axis=1).reshape(-1, 1)

# Concatenate the combined data and trend to form the dataset
latest_data_symbol = np.hstack((latest_data_combined, latest_data_trend))

# Extract the most recent window of data
latest_window = latest_data_symbol[-lookback:, :]

# Reshape the input for the model
latest_window_input = latest_window.reshape(1, lookback, latest_data_combined.shape[1] + 1)

# Load the best model
best_model = tf.keras.models.load_model('best_complex_model.h5')

# Initialize the list to store the predictions
predicted_trend = []

# Loop through the dataset to make predictions one day at a time
for i in range(len(latest_data_symbol) - lookback):
    window = latest_data_symbol[i:i + lookback, :]
    window_input = window.reshape(1, lookback, latest_data_combined.shape[1] + 1)
    prediction = best_model.predict(window_input)
    predicted_trend.append(prediction[0][0])

# Convert the list of predictions to a NumPy array
predicted_trend_array = np.array(predicted_trend)

# normalize the predictions so that the first index value is 1 (i.e., no change)
predicted_trend_array = predicted_trend_array / predicted_trend_array[0]

print("Predicted overall trend array:", predicted_trend_array)

# Plot the predicted overall trend
def plot_predicted_trend(predicted_trend_array, symbol_list):
    # Create a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a folder to store the predictions
    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    # Plot the predicted overall trend
    plt.plot(predicted_trend_array)
    plt.title(f"Predicted overall trend for {symbol_list}")
    plt.xlabel("Days")
    plt.ylabel("Overall trend")
    plt.savefig(f"predictions/predicted_trend_{timestamp}.png")
    plt.show()

plot_predicted_trend(predicted_trend_array, symbol_list)
