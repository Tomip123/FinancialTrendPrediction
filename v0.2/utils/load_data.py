import os
import pandas as pd
import yfinance as yf
from config import DATA_DIR


def fetch_stock_data(symbols):
    """
    Fetches stock data for a list of symbols from Yahoo Finance API and stores it in a CSV file.
    If the data is already downloaded, it loads it from the CSV file.

    Args:
    symbols (list): A list of stock symbols to fetch data for.

    Returns:
    all_stock_data (pandas.DataFrame): A DataFrame containing the stock data for all the symbols.
    """

    def get_stock_data(symbol):
        """
        Fetches stock data for a symbol from Yahoo Finance API.

        Args:
        symbol (str): The stock symbol to fetch data for.

        Returns:
        df (pandas.DataFrame): A DataFrame containing the stock data for the symbol.
        """

        start_date = '2001-07-19'
        end_date = '2023-05-05'
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

    if os.path.exists(os.path.join(DATA_DIR, 'stock_data.csv')):
        return load_data(DATA_DIR, 'stock_data.csv')

    stock_data = {}
    for symbol in symbols:
        try:
            df = get_stock_data(symbol)
            stock_data[symbol] = df
        except ValueError as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

    all_stock_data = pd.concat(stock_data, axis=1)

    all_stock_data.fillna(method='ffill', inplace=True)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, 'stock_data.csv')
    all_stock_data.to_csv(file_path)

    return all_stock_data


def load_data(folder_path, file_name):
    """
    Loads stock data from a CSV file.

    Args:
    folder_path (str): The path to the folder containing the CSV file.
    file_name (str): The name of the CSV file.

    Returns:
    all_stock_data (pandas.DataFrame): A DataFrame containing the stock data.
    """

    file_path = os.path.join(folder_path, file_name)
    all_stock_data = pd.read_csv(file_path, header=[0,1], index_col=0)
    return all_stock_data