import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🍺 Beer Demand Forecasting & Anomaly Detection", layout="wide")

st.title("🍺 Beer Demand Forecast & Anomaly Insights")

# Upload CSVs
uploaded_main = st.file_uploader("Upload forecast file (beer_sales_predictions_with_anomalies.csv)", type=["csv"])
uploaded_importance = st.file_uploader("Upload feature importance file (feature_importance.csv)", type=["csv"])

if uploaded_main:
    try:
        df = pd.read_csv(uploaded_main, parse_dates=["date"])
        required = {"date", "units_sold", "predicted", "anomaly", "root_cause_hint"}
        if not required.issubset(df.columns):
            st.error(f"CSV must include: {required}")
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

            # --- Anomaly Detection ---
            st.subheader("🚨 Anomalies")

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

            # --- Feature Importance ---
            if uploaded_importance:
                st.subheader("📊 Feature Importance")

                importance_df = pd.read_csv(uploaded_importance)
                if {"feature", "importance", "category"}.issubset(importance_df.columns):
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    sns.barplot(
                        data=importance_df.sort_values("importance", ascending=False),
                        x="importance", y="feature", hue="category", ax=ax3
                    )
                    ax3.set_title("Feature Importance by Category")
                    st.pyplot(fig3)
                    st.dataframe(importance_df)
                else:
                    st.warning("Feature importance file is missing required columns.")
            else:
                st.info("Upload feature_importance.csv to see feature rankings.")

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
