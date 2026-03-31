import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ============================================================
# PATHS
# ============================================================
RUN_DIR = Path("run")
MODEL_PATH = RUN_DIR / "cnn_lstm_model.keras"
SCALER_STATS_PATH = RUN_DIR / "scaler_stats.npz"
FEATURES_PATH = RUN_DIR / "feature_columns.json"

SEQUENCE_LENGTH = 14
TARGET_COL = "river_discharge_m3s"

# ============================================================
# LOAD ARTIFACTS
# ============================================================
def load_scaler(stats_path):
    data = np.load(stats_path)

    scaler = StandardScaler()
    scaler.mean_ = data["mean_"]
    scaler.scale_ = data["scale_"]
    scaler.var_ = data["var_"]
    scaler.n_features_in_ = int(data["n_features_in_"][0])
    scaler.with_mean = bool(data["with_mean"][0])
    scaler.with_std = bool(data["with_std"][0])

    return scaler

model = tf.keras.models.load_model(MODEL_PATH)
scaler = load_scaler(SCALER_STATS_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_cols = json.load(f)

# ============================================================
# STEP 1: LOAD HOURLY RAIN DATA
# ============================================================
rain_df = pd.read_csv("data/weather_forecast_hourly.csv")

rain_df["datetime"] = pd.to_datetime(rain_df["datetime"])
rain_df["date"] = rain_df["datetime"].dt.date

# ============================================================
# STEP 2: AGGREGATE TO DAILY
# ============================================================
daily_rain = (
    rain_df.groupby(["location", "date"])
    .agg(
        rain_sum=("rain_member0", "sum"),
        rain_max=("rain_member0", "max"),
        rain_mean=("rain_member0", "mean"),
    )
    .reset_index()
)

# ============================================================
# STEP 3: CREATE TIME FEATURES
# ============================================================
daily_rain["date"] = pd.to_datetime(daily_rain["date"])
daily_rain = daily_rain.sort_values(["location", "date"])

# Lag features
for lag in [1, 2, 3]:
    daily_rain[f"rain_sum_lag{lag}"] = (
        daily_rain.groupby("location")["rain_sum"].shift(lag)
    )

# Rolling features
daily_rain["rain_3d_avg"] = (
    daily_rain.groupby("location")["rain_sum"]
    .rolling(3).mean().reset_index(0, drop=True)
)

daily_rain = daily_rain.dropna()

# ============================================================
# STEP 4: ALIGN WITH TRAINING FEATURES
# ============================================================
missing_cols = [c for c in feature_cols if c not in daily_rain.columns]

for col in missing_cols:
    daily_rain[col] = 0  # fallback if feature missing

X = daily_rain[feature_cols].values
X_scaled = scaler.transform(X)

# ============================================================
# STEP 5: CREATE SEQUENCE (LAST 14 DAYS)
# ============================================================
def create_sequences(X, seq_len):
    sequences = []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i + seq_len])
    return np.array(sequences)

X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)

if len(X_seq) == 0:
    raise ValueError("Not enough data to create sequences")

# ============================================================
# STEP 6: PREDICT DISCHARGE
# ============================================================
y_pred = model.predict(X_seq).flatten()

# Align prediction with dates
pred_df = daily_rain.iloc[SEQUENCE_LENGTH:].copy()
pred_df["predicted_discharge"] = y_pred

# ============================================================
# STEP 7: FLOOD RISK CLASSIFICATION
# ============================================================
def classify_flood(discharge):
    if discharge > 300:
        return "HIGH"
    elif discharge > 150:
        return "MEDIUM"
    else:
        return "LOW"

pred_df["flood_risk"] = pred_df["predicted_discharge"].apply(classify_flood)

# ============================================================
# OUTPUT
# ============================================================
print(pred_df[["date", "location", "predicted_discharge", "flood_risk"]])

pred_df.to_csv("data/flood_forecast_output.csv", index=False)