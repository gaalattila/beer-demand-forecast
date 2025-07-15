import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

st.set_page_config(page_title="🍺 Beer Forecast & Root Cause Analyzer", layout="wide")
st.title("🍺 Beer Demand Forecast & Anomaly Detection")

# Upload file
uploaded_file = st.file_uploader("📤 Upload raw input file (raw_beer_sales_data.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])

        required_cols = {"date", "units_sold", "is_weekend", "temperature", "football_match", "holiday", "season"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must include columns: {required_cols}")
        else:
            st.success("✅ Raw input loaded")

            # --- Model Training and Prediction ---
            features = ["is_weekend", "temperature", "football_match", "holiday", "season"]
            X = df[features]
            y = df["units_sold"]

            model = XGBRegressor(n_estimators=100, max_depth=3)
            model.fit(X, y)
            df["predicted"] = model.predict(X)

            # --- Anomaly Detection ---
            df["error"] = abs(df["units_sold"] - df["predicted"])
            threshold = df["error"].mean() + 2 * df["error"].std()
            df["anomaly"] = df["error"] > threshold

            # --- Root Cause Hints ---
            def root_cause(row):
                if row["football_match"] and row["temperature"] > 25:
                    return "Hot day + football match"
                elif row["football_match"]:
                    return "Football match"
                elif row["temperature"] > 25:
                    return "Hot day"
                elif row["is_weekend"]:
                    return "Weekend spike"
                return "Unexplained"

            df["root_cause_hint"] = df.apply(lambda row: root_cause(row) if row["anomaly"] else "", axis=1)

            # --- Forecast Plot ---
            st.subheader("📈 Actual vs Predicted Sales")
            fig1, ax1 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df, x="date", y="units_sold", label="Actual", ax=ax1)
            sns.lineplot(data=df, x="date", y="predicted", label="Predicted", ax=ax1)
            ax1.set_ylabel("Units Sold")
            st.pyplot(fig1)

            # --- Anomalies ---
            st.subheader("🚨 Detected Anomalies")
            anomalies = df[df["anomaly"] == True]
            root_causes = sorted(anomalies["root_cause_hint"].unique())
            selected_causes = st.multiselect("Filter by Root Cause", root_causes, default=root_causes)
            filtered = anomalies[anomalies["root_cause_hint"].isin(selected_causes)]

            if not filtered.empty:
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                sns.lineplot(data=df, x="date", y="units_sold", label="Actual", ax=ax2)
                sns.scatterplot(data=filtered, x="date", y="units_sold", color="red", label="Anomaly", s=100, marker="X", ax=ax2)
                ax2.set_title("Filtered Anomalies")
                st.pyplot(fig2)

                st.dataframe(filtered[["date", "units_sold", "predicted", "root_cause_hint"]])
            else:
                st.info("No anomalies match the selected root causes.")

            # --- Feature Importance ---
            st.subheader("📊 Feature Importance (retrained model)")
            importance = model.feature_importances_
            categories = {
                "is_weekend": "Temporal",
                "temperature": "Weather",
                "football_match": "Event",
                "holiday": "Holiday",
                "season": "Seasonal"
            }

            importance_df = pd.DataFrame({
                "feature": features,
                "importance": importance,
                "category": [categories[f] for f in features]
            }).sort_values(by="importance", ascending=False)

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=importance_df, x="importance", y="feature", hue="category", ax=ax3)
            ax3.set_title("Feature Importance")
            st.pyplot(fig3)
            with st.expander("📄 Feature Importance Table"):
                st.dataframe(importance_df)

            # --- Download Enriched Output ---
            st.download_button(
                label="📥 Download Forecast + Anomaly CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="beer_forecast_with_anomalies.csv",
                mime="text/csv",
            )

            with st.expander("🧾 Show Enriched Data"):
                st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("Please upload the raw beer sales dataset to get started.")


