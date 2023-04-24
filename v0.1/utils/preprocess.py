import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Replace 'symbol_list' with the list of NYSE stock symbols you want to fetch
symbol_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

# Load the data from the CSV file
data = pd.read_csv('./data/stock_data_ml.csv', header=[0, 1], index_col=0)
data.index = pd.to_datetime(data.index)

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

# Save the preprocessed data
data.to_csv('stock_data_preprocessed.csv')
