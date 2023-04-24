import requests
import pandas as pd
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