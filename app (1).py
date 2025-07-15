import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# App title
st.set_page_config(layout="wide")
st.title("📈 Beer Demand Forecast & Anomaly Detection")

# Load and display predictions
st.header("📊 Forecasted Data with Anomalies")
try:
    df_predictions = pd.read_csv("beer_sales_predictions_with_anomalies.csv")
    st.dataframe(df_predictions)

    # Export CSV
    csv = df_predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast & Anomaly Data as CSV",
        data=csv,
        file_name="beer_predictions.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.error("Prediction file not found.")

# Load and display feature importance
st.header("📌 Feature Importance")
try:
    df_feat = pd.read_csv("feature_importance.csv")
    st.dataframe(df_feat)

    feat_csv = df_feat.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Feature Importance as CSV",
        data=feat_csv,
        file_name="feature_importance.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("Feature importance file not found.")
