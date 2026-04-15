# model_comparison.py

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, Input,
    BatchNormalization, Bidirectional, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/features_daily.csv"

SEQUENCE_LENGTH = 14
TEST_RATIO = 0.2
# EPOCHS = 25
BATCH_SIZE = 32

TARGET_COL = "river_discharge_m3s"
FLOOD_THRESHOLD = 100

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# HELPERS
# ============================================================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def evaluate_model(name, model, X_test, y_test):
    y_pred = (model.predict(X_test).flatten() >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 {name} Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }


# ============================================================
# MODELS
# ============================================================
def build_cnn(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Conv1D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.2),

        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.2),

        Flatten(),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


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

    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_lstm(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        Conv1D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.2),

        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    # Classification label
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

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    results = []

    # =========================
    # CNN
    # =========================
    print("\n🚀 Training CNN...")
    cnn = build_cnn(SEQUENCE_LENGTH, X.shape[2])
    cnn.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            # epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1)

    results.append(evaluate_model("CNN", cnn, X_test, y_test))

    # =========================
    # LSTM
    # =========================
    print("\n🚀 Training LSTM...")
    lstm = build_lstm(SEQUENCE_LENGTH, X.shape[2])
    lstm.fit(X_train, y_train,
             validation_data=(X_test, y_test),
            #  epochs=EPOCHS,
             batch_size=BATCH_SIZE,
             callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
             verbose=1)

    results.append(evaluate_model("LSTM", lstm, X_test, y_test))

    # =========================
    # CNN-LSTM
    # =========================
    print("\n🚀 Training CNN-LSTM...")
    cnn_lstm = build_cnn_lstm(SEQUENCE_LENGTH, X.shape[2])
    cnn_lstm.fit(X_train, y_train,
                 validation_data=(X_test, y_test),
                #  epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                 verbose=1)

    results.append(evaluate_model("CNN-LSTM", cnn_lstm, X_test, y_test))

    # =========================
    # Comparison Table
    # =========================
    results_df = pd.DataFrame(results)

    print("\n📊 FINAL COMPARISON TABLE")
    print(results_df)

    # Save for thesis
    results_df.to_csv("model_comparison_results.csv", index=False)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()