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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["location"] = df["location"].astype(str).str.strip()
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding date/location/target.")

    # Drop rows with missing target or missing features
    model_df = df.dropna(subset=[TARGET_COL] + feature_cols).copy()

    if model_df.empty:
        raise ValueError("No valid rows left after dropping missing target/features.")

    # Fit one scaler across all rows for consistency
    scaler = StandardScaler()
    scaler.fit(model_df[feature_cols].values)

    X_all, y_all = [], []
    location_sequence_counts = {}

    # Build sequences per location to avoid crossing location boundaries
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
        raise ValueError(
            "No sequences were created. Check data size per location and SEQUENCE_LENGTH."
        )

    X_seq = np.concatenate(X_all, axis=0).astype(np.float32)
    y_seq = np.concatenate(y_all, axis=0).astype(np.float32)

    split_idx = int(len(X_seq) * (1 - TEST_RATIO))
    if split_idx <= 0 or split_idx >= len(X_seq):
        raise ValueError("Train/test split produced an invalid partition.")

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Save plain, deployment-friendly artifacts
    save_scaler_stats(scaler, SCALER_STATS_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "target_column": TARGET_COL,
        "feature_count": len(feature_cols),
        "feature_columns_path": str(FEATURES_PATH),
        "scaler_stats_path": str(SCALER_STATS_PATH),
        "model_path": str(MODEL_PATH),
        "rows_used": int(len(model_df)),
        "total_sequences": int(len(X_seq)),
        "train_sequences": int(len(X_train)),
        "test_sequences": int(len(X_test)),
        "location_sequence_counts": location_sequence_counts,
        "seed": SEED,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    model = build_model(SEQUENCE_LENGTH, len(feature_cols))

    ckpt = ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    es = EarlyStopping(
        monitor="val_loss",
        patience=7, # Increased patience to allow LR scheduler to kick in
        restore_best_weights=True,
        verbose=1,
    )
    
    # 5. Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt, es, lr_scheduler],
        verbose=1,
    )

    # Ensure final saved model exists in .keras format
    model.save(MODEL_PATH)

    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "model_path": str(MODEL_PATH),
        "scaler_stats_path": str(SCALER_STATS_PATH),
        "features_path": str(FEATURES_PATH),
        "metadata_path": str(METADATA_PATH),
    }


if __name__ == "__main__":
    train()