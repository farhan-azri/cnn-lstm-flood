import joblib
import streamlit as st
import tensorflow as tf

MODEL_PATH = "run/cnn_lstm_model.h5"
SCALER_PATH = "run/scaler.pkl"
FEATURES_PATH = "run/feature_columns.pkl"


@st.cache_resource
def load_artifacts_safe():
    """
    Robust model loader for TF/Keras version mismatches.
    Tries:
      1) load_model(..., compile=False)
      2) load_model(..., compile=False, custom_objects=...)
      3) load_model(..., custom_objects=...)  # last resort
    """
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    custom_objects = {
        # Common aliases that break across TF/Keras versions
        "mse": tf.keras.losses.MeanSquaredError(),
        "mae": tf.keras.metrics.MeanAbsoluteError(),
        "mean_squared_error": tf.keras.losses.MeanSquaredError(),
        "mean_absolute_error": tf.keras.metrics.MeanAbsoluteError(),
    }

    errors = []

    # Try 1: best practice for inference
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, scaler, feature_cols
    except Exception as e:
        errors.append(f"Try 1 (compile=False) failed: {e}")

    # Try 2: compile=False + custom objects
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects
        )
        return model, scaler, feature_cols
    except Exception as e:
        errors.append(f"Try 2 (compile=False + custom_objects) failed: {e}")

    # Try 3: custom objects without compile=False (rarely needed)
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects
        )
        return model, scaler, feature_cols
    except Exception as e:
        errors.append(f"Try 3 (custom_objects) failed: {e}")

    # If all failed, show helpful message
    debug_msg = "\n\n".join(errors)
    raise RuntimeError(
        "❌ Failed to load model. This is almost always a TensorFlow/Keras version mismatch.\n\n"
        f"Details:\n{debug_msg}\n\n"
        "✅ Fix (recommended):\n"
        "1) Uninstall standalone keras:\n"
        "   pip uninstall -y keras\n"
        "2) Reinstall TensorFlow pinned:\n"
        "   pip install 'tensorflow==2.15.*' 'numpy<2' 'protobuf<5'\n\n"
        "✅ Best long-term fix:\n"
        "- Re-save model in .keras format:\n"
        "  model.save('run/cnn_lstm_model.keras')\n"
        "- Then load with compile=False."
    )