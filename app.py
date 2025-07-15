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

        required_cols = {"date", "units_sold", "is_weekend", "temperature", "football_match", "holiday", "season", 
                        "precipitation", "lead_time", "beer_type", "promotion", "stock_level"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must include columns: {required_cols}")
        else:
            st.success("✅ Raw input loaded")

            # --- Feature Engineering ---
            df["day_of_week"] = df["date"].dt.dayofweek
            df["units_sold_lag1"] = df["units_sold"].shift(1).fillna(df["units_sold"].mean())
            df["units_sold_7d_avg"] = df["units_sold"].rolling(window=7, min_periods=1).mean().fillna(df["units_sold"].mean())
            df = pd.get_dummies(df, columns=["beer_type", "season"], prefix=["beer", "season"])

            # --- Model Training and Prediction ---
            features = ["is_weekend", "temperature", "football_match", "holiday", 
                        "precipitation", "lead_time", "promotion", "day_of_week", 
                        "units_sold_lag1", "units_sold_7d_avg"] + \
                       [col for col in df.columns if col.startswith("beer_") or col.startswith("season_")]
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
                elif row["precipitation"] > 10:
                    return "Rainy day"
                elif row["promotion"] == 1:
                    return "Promotion active"
                elif row["is_weekend"]:
                    return "Weekend spike"
                elif row["lead_time"] > 5:
                    return "Delayed restock"
                return "Unexplained"

            df["root_cause_hint"] = df.apply(lambda row: root_cause(row) if row["anomaly"] else "", axis=1)

            # --- Reorder Quantity ---
            df["reorder_quantity"] = (df["predicted"] * (df["lead_time"] + 1) - df["stock_level"]).clip(lower=0)

            # --- Forecast Plot ---
            st.subheader("📈 Actual vs Predicted Sales")
            # Debug: Check data integrity
            if df["units_sold"].isna().any() or df["predicted"].isna().any():
                st.warning("NaN values detected in units_sold or predicted. Filling with mean for plotting.")
                df["units_sold"] = df["units_sold"].fillna(df["units_sold"].mean())
                df["predicted"] = df["predicted"].fillna(df["predicted"].mean())
            st.write(f"Debug: units_sold - min: {df['units_sold'].min()}, max: {df['units_sold'].max()}, mean: {df['units_sold'].mean():.2f}")
            st.write(f"Debug: predicted - min: {df['predicted'].min()}, max: {df['predicted'].max()}, mean: {df['predicted'].mean():.2f}")

            fig1, ax1 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df, x="date", y="units_sold", label="Actual Sales", ax=ax1, color="#1f77b4", linewidth=2.5)
            sns.lineplot(data=df, x="date", y="predicted", label="Predicted Sales", ax=ax1, color="#ff7f0e", linestyle="--", linewidth=2.5)
            ax1.set_ylabel("Units")
            ax1.set_title("Actual vs Predicted Sales")
            ax1.legend()
            plt.tight_layout()
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

            # --- Stock vs Demand Plot ---
            st.subheader("📦 Stock Levels vs Predicted Demand")
            fig4, ax4 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df, x="date", y="units_sold", label="Actual Sales", ax=ax4, color="#1f77b4")
            sns.lineplot(data=df, x="date", y="predicted", label="Predicted Demand", ax=ax4, color="#ff7f0e", linestyle="--")
            sns.lineplot(data=df, x="date", y="stock_level", label="Stock Level", ax=ax4, color="#2ca02c")
            ax4.set_title("Stock Levels vs Actual and Predicted Demand")
            ax4.set_ylabel("Units")
            st.pyplot(fig4)

            # --- Reorder Recommendations ---
            st.subheader("📦 Reorder Recommendations")
            st.dataframe(df[["date", "predicted", "stock_level", "reorder_quantity"]])

            # --- Feature Importance ---
            st.subheader("📊 Feature Importance (retrained model)")
            importance = model.feature_importances_
            categories = {
                "is_weekend": "Temporal",
                "temperature": "Weather",
                "football_match": "Event",
                "holiday": "Holiday",
                "precipitation": "Weather",
                "lead_time": "Inventory",
                "promotion": "Marketing",
                "day_of_week": "Temporal",
                "units_sold_lag1": "Historical",
                "units_sold_7d_avg": "Historical"
            }
            for col in [c for c in df.columns if c.startswith("beer_")]:
                categories[col] = "Product"
            for col in [c for c in df.columns if c.startswith("season_")]:
                categories[col] = "Seasonal"

            importance_df = pd.DataFrame({
                "feature": features,
                "importance": importance,
                "category": [categories.get(f, "Other") for f in features]
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