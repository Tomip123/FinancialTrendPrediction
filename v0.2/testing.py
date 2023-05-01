import os
import pandas as pd
from config import DATA_DIR
import matplotlib.pyplot as plt

# print the head of the data stock_data_normalized.csv

# load the data
preprocessed_data = pd.read_csv(os.path.join(DATA_DIR, 'stock_data_normalized.csv'), index_col=0, header=[0, 1])

# visualize the data index movement
preprocessed_data['Index Movement'].plot()
plt.xlabel('Days')
plt.ylabel('Index Movement')
plt.title('Index Movement over the years')
plt.show()

