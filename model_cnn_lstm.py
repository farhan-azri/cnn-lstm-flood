import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATA_PATH = "data/features_daily.csv"
RUN_DIR = "run"
MODEL_PATH = f"{RUN_DIR}/cnn_lstm_model.h5"
SCALER_PATH = f"{RUN_DIR}/scaler.pkl"
FEATURES_PATH = f"{RUN_DIR}/feature_columns.pkl"

SEQUENCE_LENGTH = 14
TEST_RATIO = 0.2
EPOCHS = 40
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train():
    Path(RUN_DIR).mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    target = "river_discharge_m3s"
    feature_cols = df.drop(columns=["date", "location", target]).columns.tolist()

    # IMPORTANT: train per location then concat sequences (prevents mixing sequences across locations)
    X_all, y_all = [], []

    scaler = StandardScaler()

    # Fit scaler on all features (across locations) for consistency
    scaler.fit(df[feature_cols].values)

    for loc, g in df.groupby("location"):
        g = g.sort_values("date")
        X_scaled = scaler.transform(g[feature_cols].values)
        y = g[target].values

        X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
        if len(X_seq) == 0:
            continue

        X_all.append(X_seq)
        y_all.append(y_seq)

    X_seq = np.concatenate(X_all, axis=0)
    y_seq = np.concatenate(y_all, axis=0)

    # Time-series split (simple)
    split = int(len(X_seq) * (1 - TEST_RATIO))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    ckpt = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt, es],
        verbose=1
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    model.save(MODEL_PATH)

    print(f"✅ Saved model: {MODEL_PATH}")
    print(f"✅ Saved scaler: {SCALER_PATH}")
    print(f"✅ Saved features: {FEATURES_PATH}")
    print(f"📊 RMSE={rmse:.3f}  MAE={mae:.3f}")

    return rmse, mae


if __name__ == "__main__":
    train()