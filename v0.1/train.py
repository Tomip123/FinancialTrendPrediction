import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Bidirectional, LeakyReLU)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        LearningRateScheduler)
from tensorflow.keras.models import Model
import os


# Function to create input-output pairs for the model
def create_dataset(dataset, lookback=60):
    data_x, data_y = [], []
    for i in range(len(dataset) - lookback):
        data_x.append(dataset[i:i + lookback, :])
        data_y.append(dataset[i + lookback, -1])
    return np.array(data_x), np.array(data_y)

# Define the model architecture
def transformer_encoder_block(embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return Model(inputs=inputs, outputs=out2)

def build_model(input_shape, num_blocks=6, num_heads=32, ff_dim=1024, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder_block(input_shape[-1], num_heads, ff_dim, dropout_rate)(x)

    x = Bidirectional(LSTM(units=512, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units=256, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=128, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(512)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model

# Learning rate scheduler function
def lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        lr = lr * 0.5
    return lr

# Load the training and validation data
train_data = pd.read_csv('stock_data_train.csv', header=[0, 1], index_col=0)
val_data = pd.read_csv('stock_data_val.csv', header=[0, 1], index_col=0)

# Combine the close prices of all symbols
train_data_combined = train_data.xs('close', level=1, axis=1).dropna().values
val_data_combined = val_data.xs('close', level=1, axis=1).dropna().values

# Calculate the overall trend by summing the close prices
train_data_trend = np.sum(train_data_combined, axis=1).reshape(-1, 1)
val_data_trend = np.sum(val_data_combined, axis=1).reshape(-1, 1)

# Concatenate the combined data and trend to form the dataset
train_data_symbol = np.hstack((train_data_combined, train_data_trend))
val_data_symbol = np.hstack((val_data_combined, val_data_trend))

# Create input-output pairs for the training and validation sets
lookback = 60
x_train, y_train = create_dataset(train_data_symbol, lookback)
x_val, y_val = create_dataset(val_data_symbol, lookback)

# Build the model
input_shape = (lookback, train_data_combined.shape[1] + 1)  # Add 1 to account for the new overall trend column
model = build_model(input_shape)

# Train the model
epochs = 100
batch_size = 32

# Set up early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_complex_model.h5', monitor='val_loss', save_best_only=True)

# Set up the learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [early_stopping, model_checkpoint, lr_scheduler]

# Select the GPU device
gpu_device = '/gpu:0'

with tf.device(gpu_device):
    # Print the device being used
    print("Device being used:", tf.test.gpu_device_name() if tf.test.gpu_device_name() else "CPU")
    
    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# show the loss and accuracy curves for training and validation
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(history.history['loss'], label='train')
ax[0].plot(history.history['val_loss'], label='validation')
ax[0].set_title('Loss')
ax[0].legend()

ax[1].plot(history.history['mean_absolute_error'], label='train')
ax[1].plot(history.history['val_mean_absolute_error'], label='validation')
ax[1].set_title('Mean Absolute Error')
ax[1].legend()

# save the figure with a time stamp, in the folder 'figures', create folder if it doesn't exist
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join("figures", fig_id + "." + fig_extension)
    if not os.path.isdir("figures"):
        os.makedirs("figures")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

save_fig('loss_and_accuracy_curves')