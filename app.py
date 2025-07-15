import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 Beer Demand Forecasting & Anomaly Detection", layout="wide")

st.title("🍺 Beer Demand Forecast & Anomaly Insights")

# Upload CSV
uploaded_file = st.file_uploader("Upload forecast file (e.g., beer_sales_predictions_with_anomalies.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])

        # Validate required columns
        required = {"date", "units_sold", "predicted", "anomaly", "root_cause_hint"}
        if not required.issubset(df.columns):
            st.error(f"CSV must include: {required}")
        else:
            st.success("✅ Forecast data loaded successfully")

            # --- Plot Forecast vs Actual ---
            st.subheader("📈 Actual vs Predicted Beer Demand")
            fig1, ax1 = plt.subplots(figsize=(14, 5))
            sns.lineplot(x="date", y="units_sold", data=df, label="Actual", ax=ax1)
            sns.lineplot(x="date", y="predicted", data=df, label="Forecast", ax=ax1)
            ax1.set_ylabel("Units Sold")
            ax1.set_xlabel("Date")
            ax1.set_title("Daily Beer Demand Forecast")
            st.pyplot(fig1)

            # --- Plot Anomalies ---
            st.subheader("🚨 Detected Anomalies")
            anomalies = df[df["anomaly"] == True]
            if not anomalies.empty:
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                sns.lineplot(x="date", y="units_sold", data=df, label="Actual", ax=ax2)
                sns.scatterplot(
                    x="date", y="units_sold", data=anomalies, color="red", label="Anomaly", ax=ax2, marker="X", s=100
                )
                ax2.set_title("Anomalous Days in Beer Demand")
                st.pyplot(fig2)

                st.markdown("### 🧠 Root Cause Hints")
                st.dataframe(anomalies[["date", "units_sold", "predicted", "root_cause_hint"]])
            else:
                st.info("No anomalies detected in the dataset.")

            # --- Download Button ---
            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Forecast CSV",
                data=csv_download,
                file_name="beer_sales_predictions_with_anomalies.csv",
                mime="text/csv",
            )

            # --- Optional: Show raw data ---
            with st.expander("🧾 Show Raw Data"):
                st.dataframe(df)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a forecast CSV file to begin.")

