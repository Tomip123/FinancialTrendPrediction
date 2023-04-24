import requests
import pandas as pd
import time
import os
import dotenv

# Load the API key from the .env file
API_KEY = dotenv.get_key('.env', 'ALPHA_VANTAGE_API_KEY')

def get_stock_data(symbol, outputsize='full'):
    base_url = "https://www.alphavantage.co/query?"
    function = "TIME_SERIES_DAILY_ADJUSTED"
    
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': API_KEY,
        'datatype': 'json'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        key = 'Time Series (Daily)'
        
        if key not in data:
            print(f"Error retrieving data for {symbol}: {data}")
            return None
        
        df = pd.DataFrame(data[key]).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
        df = df.astype(float)
        df.drop(['dividend_amount', 'split_coefficient', "adjusted_close"], axis=1, inplace=True)
        return df
    else:
        print(f"Error retrieving data for {symbol}: {response.status_code}")
        return None

# Replace 'symbol_list' with the list of NYSE stock symbols you want to fetch
symbol_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
outputsize = 'full'

stock_data_dict = {}

for symbol in symbol_list:
    print(f"Fetching data for {symbol}")
    df = get_stock_data(symbol, outputsize)
    
    if df is not None:
        stock_data_dict[symbol] = df
    else:
        print(f"Error fetching data for {symbol}")

    # sleep for 12 seconds to avoid exceeding the API call limit
    time.sleep(13)

# Create a DataFrame with a multi-level column index
all_stock_data = pd.concat(stock_data_dict, axis=1)

# Fill missing data points (e.g., weekends, holidays) with the previous day's value
all_stock_data.fillna(method='ffill', inplace=True)

# Save the processed DataFrame to a CSV file
all_stock_data.to_csv('stock_data_ml.csv')