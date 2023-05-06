import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import MODELS_DIR


def train_model(train_data, train_targets, val_data, val_targets, num_epochs=100, forecast_horizon=30, model_name='best_model.h5'):
    """
    Trains an LSTM model to predict financial trends.

    Args:
        train_data (numpy.ndarray): Training data of shape (num_samples, num_timesteps, num_features).
        train_targets (numpy.ndarray): Training targets of shape (num_samples, forecast_horizon).
        val_data (numpy.ndarray): Validation data of shape (num_samples, num_timesteps, num_features).
        val_targets (numpy.ndarray): Validation targets of shape (num_samples, forecast_horizon).
        num_epochs (int): Number of epochs to train the model for. Default is 100.
        forecast_horizon (int): Number of timesteps to predict into the future. Default is 30.
        model_name (str): Name of the file to save the best model as. Default is 'best_model.h5'.

    Returns:
        tuple: A tuple containing the trained model and its training history.
    """

    # Check if GPU is available
    if tf.test.is_built_with_gpu_support():
        print("Training on device: ", tf.test.gpu_device_name())
    else:
        print("GPU not available, training on CPU...")

    # Get the number of features and timesteps
    num_features = train_data.shape[2]
    num_timesteps = train_data.shape[1]

    # Define the LSTM model
    with tf.device(tf.test.gpu_device_name()):
        model = Sequential()

        # Encoder
        model.add(LSTM(units=512, activation='tanh', input_shape=(num_timesteps, num_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=256, activation='tanh', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128, activation='tanh', return_sequences=False))

        # Prediction Layer
        model.add(Dense(forecast_horizon))  # predict forecast_horizon trend values directly

        # Compile the model
        model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath=os.path.join(MODELS_DIR, model_name), monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Fit the model to the training data
    model.fit(train_data, train_targets, epochs=num_epochs, validation_data=(val_data, val_targets), shuffle=False, callbacks=[early_stopping, checkpoint])

    return model, model.history
