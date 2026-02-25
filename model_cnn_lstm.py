import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense


def create_sequences(X, y, seq_length=14):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)


def train_model():

    df = pd.read_csv("data/features.csv")

    target = "river_discharge"

    features = df.drop(columns=["date", "location", target])
    y = df[target].values
    X = features.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = create_sequences(X_scaled, y)

    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_seq, y_seq, epochs=10, batch_size=32)

    model.save("data/cnn_lstm_model.h5")

    return model