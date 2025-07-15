import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("beer_sales_predictions_with_anomalies.csv", parse_dates=["date"])

@st.cache_data
def load_importance():
    return pd.read_csv("feature_importance.csv")

# --- Page Setup ---
st.set_page_config(page_title="Beer Demand Forecasting", layout="wide")
st.title("🍺 AI-Driven Beer Demand Forecasting Dashboard")

# --- Load Data ---
df = load_data()
importance_df = load_importance()

# --- Date Filter ---
st.sidebar.header("📅 Filter by Date")
date_range = st.sidebar.date_input(
    "Select date range", [df["date"].min(), df["date"].max()]
)
if len(date_range) == 2:
    df = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]

# --- Actual vs Predicted ---
st.subheader("📈 Actual vs Predicted Beer Sales")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["units_sold"], label="Actual", color="blue")
ax.plot(df["date"], df["predicted"], label="Predicted", color="orange")
ax.set_title("Beer Sales Forecasting")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Anomaly Table ---
st.subheader("🚨 Detected Anomalies")
anomalies = df[df["anomaly"] == True]
st.dataframe(anomalies[["date", "units_sold", "predicted", "root_cause_hint"]])

# --- Feature Importance ---
st.subheader("📊 Feature Importance (Categorized)")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, y="feature", x="importance", hue="category", ax=ax2, dodge=False)
ax2.set_title("Categorized Feature Importance")
ax2.set_xlabel("Importance Score")
st.pyplot(fig2)

# --- Feature Tree ---
st.subheader("🌳 Feature Interaction Tree")
st.image("feature_tree_high_res.png", caption="XGBoost First Tree", use_column_width=True)
