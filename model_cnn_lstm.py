import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ============================================================
# PATHS / CONFIG
# ============================================================
DATA_PATH = "data/features_daily.csv"
RUN_DIR = Path("run")

MODEL_PATH = RUN_DIR / "cnn_lstm_model.keras"
SCALER_STATS_PATH = RUN_DIR / "scaler_stats.npz"
FEATURES_PATH = RUN_DIR / "feature_columns.json"
METADATA_PATH = RUN_DIR / "training_metadata.json"

SEQUENCE_LENGTH = 14
TEST_RATIO = 0.2
EPOCHS = 50 # Increased slightly since we have a dynamic learning rate now
BATCH_SIZE = 32
SEED = 42
TARGET_COL = "river_discharge_m3s"
DROP_COLS = ["date", "location", TARGET_COL]

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HELPERS
# ============================================================
def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def save_scaler_stats(scaler: StandardScaler, out_path: Path):
    """
    Save only the scaler numeric attributes as plain NumPy arrays.
    This avoids pickle/joblib environment issues in deployment.
    """
    np.savez(
        out_path,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        var_=scaler.var_,
        n_features_in_=np.array([scaler.n_features_in_], dtype=np.int64),
        with_mean=np.array([int(scaler.with_mean)], dtype=np.int64),
        with_std=np.array([int(scaler.with_std)], dtype=np.int64),
    )


def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        
        # 1. Enhanced Feature Extraction
        Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        # 2. Bidirectional Sequence Modeling
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        
        # 3. Deeper Dense Block for final regression
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    # 4. Huber Loss for robustness against extreme flood outliers
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(delta=1.0), 
        metrics=["mae", "mse"],
    )
    return model


# ============================================================
# TRAINING
# ============================================================
def train():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # =========================
    # Basic cleaning
    # =========================
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["location"] = df["location"].astype(str).str.strip()
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # =========================
    # 🔥 IMPORTANT: Use ONLY historical data
    # =========================
    if "data_type" not in df.columns:
        raise ValueError("Column 'data_type' not found. Please regenerate features.")

    model_df = df[df["data_type"] == "historical"].copy()

    # =========================
    # Feature selection
    # =========================
    feature_cols = [
        c for c in model_df.columns
        if c not in DROP_COLS + ["data_type"]  # exclude data_type from model
    ]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding identifiers.")

    # =========================
    # Drop missing
    # =========================
    model_df = model_df.dropna(subset=[TARGET_COL] + feature_cols)

    if model_df.empty:
        raise ValueError("No valid rows left after filtering historical data.")

    # =========================
    # Scaling
    # =========================
    scaler = StandardScaler()
    scaler.fit(model_df[feature_cols].values)

    X_all, y_all = [], []
    location_sequence_counts = {}

    # =========================
    # Sequence creation per location
    # =========================
    for loc, g in model_df.groupby("location"):
        g = g.sort_values("date").reset_index(drop=True)

        X_scaled = scaler.transform(g[feature_cols].values)
        y = g[TARGET_COL].values.astype(np.float32)

        X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
        location_sequence_counts[loc] = int(len(X_seq))

        if len(X_seq) == 0:
            continue

        X_all.append(X_seq)
        y_all.append(y_seq)

    if not X_all:
        raise ValueError("No sequences created. Check SEQUENCE_LENGTH or data size.")

    X_seq = np.concatenate(X_all).astype(np.float32)
    y_seq = np.concatenate(y_all).astype(np.float32)

    # =========================
    # Train/Test split
    # =========================
    split_idx = int(len(X_seq) * (1 - TEST_RATIO))

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # =========================
    # Save artifacts
    # =========================
    save_scaler_stats(scaler, SCALER_STATS_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "target_column": TARGET_COL,
        "feature_count": len(feature_cols),
        "rows_used": int(len(model_df)),
        "total_sequences": int(len(X_seq)),
        "train_sequences": int(len(X_train)),
        "test_sequences": int(len(X_test)),
        "location_sequence_counts": location_sequence_counts,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # =========================
    # Train model
    # =========================
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

    # =========================
    # Evaluation
    # =========================
    y_pred = model.predict(X_test).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print(f"\n✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE : {mae:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
    }

if __name__ == "__main__":
    train()