import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, Input, LSTM,
    Bidirectional, BatchNormalization, Concatenate,
    Softmax, Multiply, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/features_daily.csv"
RUN_DIR = Path("run")

MODEL_PATH = RUN_DIR / "cnn_lstm_model.keras"
SCALER_STATS_PATH = RUN_DIR / "scaler_stats.npz"
FEATURES_PATH = RUN_DIR / "feature_columns.json"
METADATA_PATH = RUN_DIR / "training_metadata.json"

SEQUENCE_LENGTH = 14
TEST_RATIO = 0.2
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

TARGET_COL = "river_discharge_m3s"
DROP_COLS = ["date", "location", TARGET_COL]

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# HELPERS
# ============================================================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def save_scaler_stats(scaler, path):
    np.savez(
        path,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        var_=scaler.var_,
        n_features_in_=np.array([scaler.n_features_in_])
    )


def attention_block(inputs):
    score = Dense(1, activation="tanh")(inputs)
    weights = Softmax(axis=1)(score)
    context = Multiply()([inputs, weights])
    context = GlobalAveragePooling1D()(context)
    return context


# ============================================================
# MODEL
# ============================================================
def build_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))

    # Multi-scale CNN
    conv1 = Conv1D(64, 2, padding="same", activation="relu")(inputs)
    conv2 = Conv1D(64, 3, padding="same", activation="relu")(inputs)
    conv3 = Conv1D(64, 5, padding="same", activation="relu")(inputs)

    x = Concatenate()([conv1, conv2, conv3])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Attention (FIXED)
    x = attention_block(x)

    # Dense
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss=Huber(),
        metrics=["mae", "mse"],
    )

    return model


# ============================================================
# TRAINING
# ============================================================
def train():
    RUN_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # -------------------------
    # Basic cleaning
    # -------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    # -------------------------
    # Discharge Features 🔥
    # -------------------------
    df["discharge_lag_1"] = df.groupby("location")[TARGET_COL].shift(1)
    df["discharge_lag_3"] = df.groupby("location")[TARGET_COL].shift(3)

    df["discharge_delta"] = df[TARGET_COL] - df["discharge_lag_1"]
    df["discharge_delta"] = df["discharge_delta"].fillna(0)

    # -------------------------
    # Cyclical Time Features 🔥
    # -------------------------
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # -------------------------
    # Feature selection
    # -------------------------
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    model_df = df.dropna(subset=[TARGET_COL] + feature_cols)

    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    scaler.fit(model_df[feature_cols])

    X_all, y_all = [], []

    for loc, g in model_df.groupby("location"):
        g = g.sort_values("date")

        X_scaled = scaler.transform(g[feature_cols])
        y = g[TARGET_COL].values

        X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)

        if len(X_seq) > 0:
            X_all.append(X_seq)
            y_all.append(y_seq)

    X = np.concatenate(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.float32)

    # -------------------------
    # Shuffle + Split
    # -------------------------
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    split = int(len(X) * (1 - TEST_RATIO))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -------------------------
    # Save artifacts
    # -------------------------
    save_scaler_stats(scaler, SCALER_STATS_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(feature_cols),
        "rows": len(model_df),
        "samples": len(X),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # -------------------------
    # Train model
    # -------------------------
    model = build_model(SEQUENCE_LENGTH, len(feature_cols))

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=7, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE : {mae:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "model_path": str(MODEL_PATH)
    }


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    train()