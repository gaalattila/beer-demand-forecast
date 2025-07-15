import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

st.set_page_config(page_title="🍺 Beer Forecast & Anomaly Detection", layout="wide")
st.title("🍺 Beer Demand Forecast & Anomaly Insights")

# Upload file
uploaded_file = st.file_uploader("Upload forecast file (beer_sales_predictions_with_anomalies.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        required_cols = {"date", "units_sold", "predicted", "anomaly", "root_cause_hint"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must include: {required_cols}")
        else:
            st.success("✅ Forecast data loaded")

            # --- Forecast Plot ---
            st.subheader("📈 Actual vs Predicted Beer Demand")
            fig1, ax1 = plt.subplots(figsize=(14, 4))
            sns.lineplot(x="date", y="units_sold", data=df, label="Actual", ax=ax1)
            sns.lineplot(x="date", y="predicted", data=df, label="Forecast", ax=ax1)
            ax1.set_ylabel("Units Sold")
            ax1.set_xlabel("Date")
            st.pyplot(fig1)

            # --- Anomalies ---
            st.subheader("🚨 Detected Anomalies")
            anomalies = df[df["anomaly"] == True]
            root_causes = sorted(anomalies["root_cause_hint"].unique())
            selected_causes = st.multiselect("Filter by Root Cause", root_causes, default=root_causes)
            filtered_anomalies = anomalies[anomalies["root_cause_hint"].isin(selected_causes)]

            if not filtered_anomalies.empty:
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                sns.lineplot(x="date", y="units_sold", data=df, label="Actual", ax=ax2)
                sns.scatterplot(
                    x="date", y="units_sold", data=filtered_anomalies,
                    color="red", label="Anomaly", marker="X", s=100, ax=ax2
                )
                ax2.set_title("Filtered Anomalies")
                st.pyplot(fig2)

                st.markdown("### 🧠 Root Cause Hints")
                st.dataframe(filtered_anomalies[["date", "units_sold", "predicted", "root_cause_hint"]])
            else:
                st.info("No anomalies match the selected root causes.")

            # --- Feature Importance from On-the-Fly Model ---
            st.subheader("📊 Feature Importance (from retrained model)")
            feature_cols = ["is_weekend", "temperature", "football_match", "holiday", "season"]

            # Ensure required features exist
            if set(feature_cols).issubset(df.columns):
                X = df[feature_cols]
                y = df["units_sold"]

                model = XGBRegressor(n_estimators=100, max_depth=3)
                model.fit(X, y)
                importance = model.feature_importances_

                category_map = {
                    "is_weekend": "Temporal",
                    "temperature": "Weather",
                    "football_match": "Event",
                    "holiday": "Holiday",
                    "season": "Seasonal"
                }

                importance_df = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": importance,
                    "category": [category_map[f] for f in feature_cols]
                }).sort_values(by="importance", ascending=False)

                fig3, ax3 = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=importance_df,
                    x="importance", y="feature", hue="category", ax=ax3
                )
                ax3.set_title("Feature Importance by Category")
                st.pyplot(fig3)

                with st.expander("📄 Feature Importance Data"):
                    st.dataframe(importance_df)
            else:
                st.warning(f"Missing features: {set(feature_cols) - set(df.columns)}")

            # --- Download full dataset ---
            st.download_button(
                label="📥 Download Full Forecast CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="beer_sales_predictions_with_anomalies.csv",
                mime="text/csv",
            )

            with st.expander("🧾 Show Raw Data"):
                st.dataframe(df)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload beer_sales_predictions_with_anomalies.csv to begin.")

