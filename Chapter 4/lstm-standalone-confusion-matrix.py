# lstm_model.py

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# CONFIG
DATA_PATH = "data/features_daily.csv"
SEQUENCE_LENGTH = 14
TEST_RATIO = 0.2
EPOCHS = 30
BATCH_SIZE = 32
TARGET_COL = "river_discharge_m3s"
FLOOD_THRESHOLD = 100

np.random.seed(42)
tf.random.set_seed(42)

# =========================
# HELPERS
# =========================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_lstm(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN
# =========================
def train():
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    df["flood_label"] = (df[TARGET_COL] >= FLOOD_THRESHOLD).astype(int)

    df = df[df["data_type"] == "historical"]

    feature_cols = [
        c for c in df.columns
        if c not in ["date", "location", TARGET_COL, "data_type", "flood_label"]
    ]

    df = df.dropna(subset=feature_cols + ["flood_label"])

    scaler = StandardScaler()
    scaler.fit(df[feature_cols])

    X_all, y_all = [], []

    for _, g in df.groupby("location"):
        g = g.sort_values("date")

        X_scaled = scaler.transform(g[feature_cols])
        y = g["flood_label"].values

        X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)

        if len(X_seq) > 0:
            X_all.append(X_seq)
            y_all.append(y_seq)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(SEQUENCE_LENGTH, len(feature_cols))

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Evaluation
    y_pred = (model.predict(X_test).flatten() >= 0.5).astype(int)

    print("\n📊 LSTM Results")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train()