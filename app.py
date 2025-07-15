import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Streamlit settings
st.set_page_config(page_title="Beer Forecasting", layout="wide")
st.title("🍺 Beer Demand Forecast & Anomaly Detection")

# --- Upload CSV Section ---
st.sidebar.header("📤 Upload New Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with new sales data", type="csv")

if uploaded_file:
    try:
        new_data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
        st.subheader("🔎 Preview of Uploaded Data")
        st.dataframe(new_data.head())
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")

# --- Plot Section ---
st.header("📈 Forecast Plot with Anomalies")
try:
    df = pd.read_csv("beer_sales_predictions_with_anomalies.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(x="date", y="actual", data=df, label="Actual", ax=ax)
    sns.lineplot(x="date", y="predicted", data=df, label="Forecast", ax=ax)

    # Highlight anomalies
    anomalies = df[df["anomaly"] == 1]
    if not anomalies.empty:
        plt.scatter(anomalies["date"], anomalies["actual"], color="red", label="Anomaly", zorder=5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("Forecast dataset not found. Please generate it first.")

# --- Download Forecast ---
st.subheader("📥 Export Forecast Data")
try:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast & Anomalies as CSV", csv, "beer_predictions.csv", "text/csv")
except:
    st.info("Forecast not available for export.")

# --- Feature Importance ---
st.header("🧠 Feature Importance")
try:
    df_feat = pd.read_csv("feature_importance.csv")
    st.dataframe(df_feat)

    feat_csv = df_feat.to_csv(index=False).encode("utf-8")
    st.download_button("Download Feature Importance CSV", feat_csv, "feature_importance.csv", "text/csv")
except FileNotFoundError:
    st.warning("Feature importance file not found.")

# --- Feature Interaction Tree ---
st.header("🌲 Feature Interaction Tree")
try:
    image = Image.open("feature_tree_high_res.png")
    st.image(image, caption="Feature Interaction Tree", use_column_width=True)

    with open("feature_tree_high_res.png", "rb") as img_file:
        st.download_button("Download Feature Tree", img_file, "feature_tree.png", "image/png")
except FileNotFoundError:
    st.warning("Feature interaction image not found.")

# --- Retraining Trigger (Placeholder) ---
st.header("🔁 Retrain Model (WIP)")
if st.button("Start Retraining"):
    with st.spinner("Retraining model..."):
        # Placeholder: logic can be added later
        st.success("✅ Model retraining completed (placeholder)")

