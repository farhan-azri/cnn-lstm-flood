import streamlit as st
import pandas as pd
import tensorflow as tf


def run_app():

    st.title("🌊 Flood Prediction Dashboard")

    df = pd.read_csv("data/features.csv")

    st.subheader("EDA Preview")
    st.line_chart(df["river_discharge"])

    model = tf.keras.models.load_model("data/cnn_lstm_model.h5")

    st.subheader("Future Prediction")

    last_row = df.drop(columns=["date","location"]).tail(14).values
    pred = model.predict(last_row.reshape(1, last_row.shape[0], last_row.shape[1]))

    st.metric("Predicted River Discharge", float(pred[0][0]))