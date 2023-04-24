from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Bidirectional, LeakyReLU)
from tensorflow.keras.models import Model

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

# Define the input shape for the model
n_features = 7  # Number of features, i.e., open, high, low, close, adjusted_close, volume, rsi
lookback = 60  # Number of historical data points to use for prediction

input_shape = (lookback, n_features)

# Build the model
model = build_model(input_shape)
model.summary()
