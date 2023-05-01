import json
import os
import pandas as pd
import yfinance as yf
from config import DATA_DIR

def fetch_stock_data(symbols):
    # Define the function to fetch stock data for a symbol
    def get_stock_data(symbol):
        # Fetch stock data from Yahoo Finance API
        start_date = '2000-01-01'
        end_date = '2023-04-28'
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
        except ValueError as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        df.drop(columns=['Dividends', 'Stock Splits'], axis=1, inplace=True)

        df.index = pd.to_datetime(df.index).date

        return df
    
    # if the data is already downloaded, load it from the CSV file
    if os.path.exists(os.path.join(DATA_DIR, 'stock_data.csv')):
        return load_data(DATA_DIR, 'stock_data.csv')


    # Fetch stock data for each symbol and store in a dictionary
    stock_data = {}
    for symbol in symbols:
        try:
            df = get_stock_data(symbol)
            stock_data[symbol] = df
        except ValueError as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

    # Combine stock data into a single DataFrame
    all_stock_data = pd.concat(stock_data, axis=1)

    # Fill missing data points with the previous day's value
    all_stock_data.fillna(method='ffill', inplace=True)

    # Save the data to a CSV file in the data folder
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, 'stock_data.csv')
    all_stock_data.to_csv(file_path)

    return all_stock_data


def load_data(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    all_stock_data = pd.read_csv(file_path, header=[0,1], index_col=0)
    return all_stock_data