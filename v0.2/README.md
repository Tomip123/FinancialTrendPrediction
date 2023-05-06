This Python program utilises Long Short-Term Memory (LSTM) neural networks to forecast the 100 technology stocks in the NYSE. The program takes stock data, normalises it, trains an LSTM model, and then utilises it to predict future stock index movements. Here are the central parts of the program:

main.py: script is responsible for importing the needed libraries, loading and processing the stock data, normalising it, dividing it into training and validation batches, forming sequences for the LSTM model, saving the training and validation data to CSV files, training the LSTM model, and showing the training and validation loss via plots.

model_training.py: This script includes the function train_model() to train the LSTM model on the supplied data. It makes an LSTM model with three layers and a dense forecast layer. It then compiles and fits the model using the training and validation data.

data_processing.py: script has two key functions: process_data() and normalize_data(). The process_data() function enhances the raw stock data by computing new features such as moving averages, relative strength index (RSI), Bollinger Bands, and moving average convergence divergence (MACD). Meanwhile, the normalize_data() function uses the MinMaxScaler from sci-kit-learn to standardise the processed data.

load_data.py: This script holds functions to fetch stock data using the Yahoo Finance API (yfinance) and to load stock data from CSV files.

predict.py: the script uses a trained LSTM model to make future projections. It loads the latest normalised data, trained model, and scaler and then generates sequences from recent data. The model forecasts future stock index movements, and a Butterworth low-pass filter ensures smooth data predictions.

This code is developed to predict stock market movements utilising an LSTM neural network model. It fetches stock data, processes and normalises it, trains an LSTM model, and makes predictions based on the trained model.