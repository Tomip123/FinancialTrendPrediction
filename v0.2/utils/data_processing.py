import pandas as pd
from sklearn import preprocessing
import os
from config import DATA_DIR
from utils.load_data import load_data

def process_data(all_stock_data):

    if os.path.exists(os.path.join(DATA_DIR, 'processed_data.csv')):
        return load_data(DATA_DIR, 'processed_data.csv')

    symbol_list = all_stock_data.columns.levels[0].tolist()
    for symbol in symbol_list:
        all_stock_data[(symbol, 'MA20')] = all_stock_data[symbol]['Close'].rolling(window=20).mean()
        delta = all_stock_data[symbol]['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        all_stock_data[(symbol, 'RSI')] = 100 - (100 / (1 + rs))
        all_stock_data[(symbol, 'Bolinger High')] = all_stock_data[(symbol, 'MA20')] + (all_stock_data[symbol]['Close'].rolling(window=20).std() * 2)
        all_stock_data[(symbol, 'Bolinger Low')] = all_stock_data[(symbol, 'MA20')] - (all_stock_data[symbol]['Close'].rolling(window=20).std() * 2)
        all_stock_data[(symbol, 'MACD')] = all_stock_data[symbol]['Close'].ewm(span=12, adjust=False).mean() - all_stock_data[symbol]['Close'].ewm(span=26, adjust=False).mean()
        
    # Calculate the overall index of the portfolio
    all_stock_data['Index Movement'] = all_stock_data.xs('Close', level=1, axis=1).sum(axis=1)
    # all_stock_data['Index Movement'] = all_stock_data['Index Movement'].pct_change()
    # all_stock_data['Index Movement'] = all_stock_data['Index Movement'].shift(-1)
    all_stock_data = all_stock_data.fillna(0)

    # need to callculate now the combined RSI, MACD, Bolinger Bands, MA20
    # RSI
    all_stock_data['Combined RSI'] = all_stock_data.xs('RSI', level=1, axis=1).sum(axis=1)
    all_stock_data['Combined RSI'] = all_stock_data['Combined RSI'] / len(symbol_list)
    # MACD
    all_stock_data['Combined MACD'] = all_stock_data.xs('MACD', level=1, axis=1).sum(axis=1)
    all_stock_data['Combined MACD'] = all_stock_data['Combined MACD'] / len(symbol_list)
    # Bolinger Bands
    all_stock_data['Combined Bolinger High'] = all_stock_data.xs('Bolinger High', level=1, axis=1).sum(axis=1)
    all_stock_data['Combined Bolinger High'] = all_stock_data['Combined Bolinger High'] / len(symbol_list)
    all_stock_data['Combined Bolinger Low'] = all_stock_data.xs('Bolinger Low', level=1, axis=1).sum(axis=1)
    all_stock_data['Combined Bolinger Low'] = all_stock_data['Combined Bolinger Low'] / len(symbol_list)
    # MA20
    all_stock_data['Combined MA20'] = all_stock_data.xs('MA20', level=1, axis=1).sum(axis=1)
    all_stock_data['Combined MA20'] = all_stock_data['Combined MA20'] / len(symbol_list)

    all_stock_data = all_stock_data.fillna(0)

    all_stock_data.to_csv(os.path.join(DATA_DIR, 'processed_data.csv'))
    return all_stock_data

def normalize_data(all_stock_data):

    if os.path.exists(os.path.join(DATA_DIR, 'stock_data_normalized.csv')):
        return load_data(DATA_DIR, 'stock_data_normalized.csv')

    scaler = preprocessing.MinMaxScaler()
    # all exclude the last 6 columns
    symbol_list = all_stock_data.columns.levels[0].tolist()[0:-6]
    print(all_stock_data.columns.levels[0].tolist()[0:-6])
    for symbol in symbol_list:
        all_stock_data[(symbol, 'Open')] = scaler.fit_transform(all_stock_data[(symbol, 'Open')].values.reshape(-1,1))
        all_stock_data[(symbol, 'High')] = scaler.fit_transform(all_stock_data[(symbol, 'High')].values.reshape(-1,1))
        all_stock_data[(symbol, 'Low')] = scaler.fit_transform(all_stock_data[(symbol, 'Low')].values.reshape(-1,1))
        all_stock_data[(symbol, 'Close')] = scaler.fit_transform(all_stock_data[(symbol, 'Close')].values.reshape(-1,1))
        all_stock_data[(symbol, 'Volume')] = scaler.fit_transform(all_stock_data[(symbol, 'Volume')].values.reshape(-1,1))
        all_stock_data[(symbol, 'MA20')] = scaler.fit_transform(all_stock_data[(symbol, 'MA20')].values.reshape(-1,1))
        all_stock_data[(symbol, 'RSI')] = scaler.fit_transform(all_stock_data[(symbol, 'RSI')].values.reshape(-1,1))
        all_stock_data[(symbol, 'Bolinger High')] = scaler.fit_transform(all_stock_data[(symbol, 'Bolinger High')].values.reshape(-1,1))
        all_stock_data[(symbol, 'Bolinger Low')] = scaler.fit_transform(all_stock_data[(symbol, 'Bolinger Low')].values.reshape(-1,1))
        all_stock_data[(symbol, 'MACD')] = scaler.fit_transform(all_stock_data[(symbol, 'MACD')].values.reshape(-1,1))
    all_stock_data['Index Movement'] = scaler.fit_transform(all_stock_data['Index Movement'].values.reshape(-1,1))
    all_stock_data['Combined RSI'] = scaler.fit_transform(all_stock_data['Combined RSI'].values.reshape(-1,1))
    all_stock_data['Combined MACD'] = scaler.fit_transform(all_stock_data['Combined MACD'].values.reshape(-1,1))
    all_stock_data['Combined Bolinger High'] = scaler.fit_transform(all_stock_data['Combined Bolinger High'].values.reshape(-1,1))
    all_stock_data['Combined Bolinger Low'] = scaler.fit_transform(all_stock_data['Combined Bolinger Low'].values.reshape(-1,1))
    all_stock_data['Combined MA20'] = scaler.fit_transform(all_stock_data['Combined MA20'].values.reshape(-1,1))
    all_stock_data.to_csv(os.path.join(DATA_DIR, 'stock_data_normalized.csv'))
    return all_stock_data
