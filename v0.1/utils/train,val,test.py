import pandas as pd

# Load the preprocessed data
data = pd.read_csv('./data/stock_data_preprocessed.csv', header=[0, 1], index_col=0)
data.index = pd.to_datetime(data.index)

# Calculate the number of samples for each set
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

# Split the data
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Save the datasets as CSV files
train_data.to_csv('stock_data_train.csv')
val_data.to_csv('stock_data_val.csv')
test_data.to_csv('stock_data_test.csv')
