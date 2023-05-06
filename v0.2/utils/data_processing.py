import pandas as pd
from sklearn import preprocessing
import os
from config import DATA_DIR
from utils.load_data import load_data
import pickle

def process_data(all_stock_data):
    """
    This function processes the stock data by calculating technical indicators such as RSI, MACD, Bolinger Bands, and MA20.
    It also calculates the overall index of the portfolio and combines the technical indicators for all stocks into a single value.
    The processed data is saved to a CSV file and returned as a pandas DataFrame.

    Parameters:
    all_stock_data (pandas.DataFrame): A DataFrame containing the stock data for all symbols.

    Returns:
    pandas.DataFrame: A DataFrame containing the processed stock data.
    """

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
        all_stock_data[(symbol, 'Daily Return')] = all_stock_data[symbol]['Close'].pct_change()
        
    # Calculate the overall index of the portfolio
    all_stock_data['Index Movement'] = all_stock_data.xs('Daily Return', level=1, axis=1).mean(axis=1)
    all_stock_data = all_stock_data.fillna(0)

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
    """
    This function normalizes the stock data using MinMaxScaler from sklearn.preprocessing.
    It takes in a pandas dataframe containing the stock data and returns the normalized data.
    If the normalized data already exists, it loads and returns it instead of normalizing again.
    The function also saves the scaler object to a file for future use.
    """
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
    
    # Save the scaler to a file
    with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    return all_stock_data
