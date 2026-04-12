import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/features_daily.csv"
RUN_DIR = Path("run")

MODEL_PATH = RUN_DIR / "cnn_lstm_classification.keras"
SCALER_PATH = RUN_DIR / "scaler.npz"
FEATURES_PATH = RUN_DIR / "feature_columns.json"

SEQUENCE_LENGTH = 14
TEST_RATIO = 0.2
EPOCHS = 30
BATCH_SIZE = 32
SEED = 42

TARGET_COL = "river_discharge_m3s"
DROP_COLS = ["date", "location", TARGET_COL]

# 🔥 Flood threshold (IMPORTANT)
FLOOD_THRESHOLD = 100  # adjust based on domain

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_model(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        Conv1D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Conv1D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),

        # 🔥 Classification output
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAINING PIPELINE
# ============================================================
def train():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # =========================
    # Basic Cleaning
    # =========================
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["location"] = df["location"].astype(str).str.strip()
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    # =========================
    # 🔥 Create Classification Label
    # =========================
    df["flood_label"] = (df[TARGET_COL] >= FLOOD_THRESHOLD).astype(int)

    # =========================
    # Use historical data only
    # =========================
    if "data_type" not in df.columns:
        raise ValueError("Column 'data_type' not found.")

    df = df[df["data_type"] == "historical"].copy()

    # =========================
    # Feature selection
    # =========================
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS + ["data_type", "flood_label"]
    ]

    df = df.dropna(subset=feature_cols + ["flood_label"])

    # =========================
    # Scaling
    # =========================
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)

    X_all, y_all = [], []

    # =========================
    # Sequence creation per location
    # =========================
    for loc, g in df.groupby("location"):
        g = g.sort_values("date").reset_index(drop=True)

        X_scaled = scaler.transform(g[feature_cols].values)
        y = g["flood_label"].values.astype(np.float32)

        X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)

        if len(X_seq) == 0:
            continue

        X_all.append(X_seq)
        y_all.append(y_seq)

    X_seq = np.concatenate(X_all)
    y_seq = np.concatenate(y_all)

    # =========================
    # Train/Test Split
    # =========================
    split_idx = int(len(X_seq) * (1 - TEST_RATIO))

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"✅ Train size: {len(X_train)}")
    print(f"✅ Test size : {len(X_test)}")

    # =========================
    # Build Model
    # =========================
    model = build_model(SEQUENCE_LENGTH, len(feature_cols))

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    # =========================
    # Train
    # =========================
    print("🚀 Training model...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # =========================
    # Evaluation
    # =========================
    print("\n📊 Evaluating model...")

    y_pred_prob = model.predict(X_test).flatten()
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

    print(f"\n✅ Accuracy : {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall   : {recall:.4f}")
    print(f"✅ F1-Score : {f1:.4f}")

    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_class))

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_class))

    # =========================
    # Save Artifacts
    # =========================
    model.save(MODEL_PATH)

    np.savez(SCALER_PATH,
             mean_=scaler.mean_,
             scale_=scaler.scale_)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    results = train()
    print("\n🎯 Final Results:", results)