from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Reshape, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import MODELS_DIR
import os

def train_model(train_data, train_targets, val_data, val_targets, num_epochs=100, forecast_horizon=30, model_name='best_model.h5'):

    # calculate the number of features and the number of timesteps
    num_features = train_data.shape[2]
    num_timesteps = train_data.shape[1]

    # define the LSTM model
    model = Sequential()

    # Encoder
    model.add(LSTM(units=512, activation='tanh', input_shape=(num_timesteps, num_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, activation='tanh', return_sequences=False))
    
    # Prediction Layer
    model.add(Dense(forecast_horizon))  # predict forecast_horizon trend values directly

    # compile the model
    model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])

    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath=os.path.join(MODELS_DIR, model_name), monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # fit the model to the training data
    model.fit(train_data, train_targets, epochs=num_epochs, validation_data=(val_data, val_targets), shuffle=False, callbacks=[early_stopping, checkpoint])

    return model, model.history

