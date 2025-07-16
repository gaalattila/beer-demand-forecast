import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Note: To prevent inotify errors on Streamlit Cloud, create a config.toml file in the project root with:
# [server]
# fileWatcherType = "none"
# Alternatively, run locally with: streamlit run app.py --server.fileWatcherType none

# Set Matplotlib dark theme for Chart.js-like visuals
plt.style.use("dark_background")

st.set_page_config(page_title="🍺 Beer Forecast & Root Cause Analyzer", layout="wide")
st.title("🍺 Beer Demand Forecast & Anomaly Detection")

# Cache data loading and feature engineering
@st.cache_data
def load_and_process_data(file):
    try:
        df = pd.read_csv(file, parse_dates=["date"])
        
        required_cols = {"date", "units_sold", "is_weekend", "temperature", "football_match", "holiday", "season", 
                        "precipitation", "lead_time", "beer_type", "promotion", "stock_level", 
                        "customer_sentiment", "competitor_promotion", "region", "supply_chain_disruption", "units_sold_30d_avg"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must include columns: {required_cols}")
        
        # Feature Engineering
        df["day_of_week"] = df["date"].dt.dayofweek
        df["units_sold_lag1"] = df["units_sold"].shift(1).fillna(df["units_sold"].mean())
        df["units_sold_7d_avg"] = df["units_sold"].rolling(window=7, min_periods=1).mean().fillna(df["units_sold"].mean())
        df = pd.get_dummies(df, columns=["beer_type", "season", "region"], prefix=["beer", "season", "region"])
        
        return df
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

# Upload file
uploaded_file = st.file_uploader("📤 Upload raw input file (raw_beer_sales_data.csv)", type=["csv"])

if uploaded_file:
    try:
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load and process data
        status_text.text("Loading and processing data...")
        df = load_and_process_data(uploaded_file)
        progress_bar.progress(20)
        st.success("✅ Raw input loaded")

        # --- Model Training and Prediction ---
        status_text.text("Training model...")
        features = ["is_weekend", "temperature", "football_match", "holiday", 
                    "precipitation", "lead_time", "promotion", "day_of_week", 
                    "units_sold_lag1", "units_sold_7d_avg", "customer_sentiment", 
                    "competitor_promotion", "supply_chain_disruption", "units_sold_30d_avg"] + \
                   [col for col in df.columns if col.startswith("beer_") or col.startswith("season_") or col.startswith("region_")]
        X = df[features]
        y = df["units_sold"]

        model = XGBRegressor(n_estimators=50, max_depth=2, reg_alpha=0.1)
        model.fit(X, y)
        df["predicted"] = model.predict(X)
        progress_bar.progress(40)

        # --- Anomaly Detection ---
        status_text.text("Detecting anomalies...")
        df["error"] = abs(df["units_sold"] - df["predicted"])
        threshold = df["error"].mean() + 2 * df["error"].std()
        df["anomaly"] = df["error"] > threshold
        progress_bar.progress(60)

        # --- Root Cause Hints ---
        def root_cause(row):
            if row["football_match"] and row["temperature"] > 25:
                return "Hot day + football match"
            elif row["football_match"] and row["customer_sentiment"] > 70:
                return "Football match + high sentiment"
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
            elif row["competitor_promotion"] == 1:
                return "Competitor promotion"
            elif row["supply_chain_disruption"] == 1:
                return "Supply chain disruption"
            return "Unexplained"

        df["root_cause_hint"] = df.apply(lambda row: root_cause(row) if row["anomaly"] else "", axis=1)

        # --- Reorder Quantity (Enhanced) ---
        # Adjust reorder based on region and supply chain disruptions
        df["reorder_quantity"] = (df["predicted"] * (df["lead_time"] + 1) * 
                                (1.2 if df["region_Urban"] == 1 else 1.0) *  # Higher buffer for Urban
                                (1.5 if df["supply_chain_disruption"] == 1 else 1.0) -  # Extra buffer for disruptions
                                df["stock_level"]).clip(lower=0)
        progress_bar.progress(80)

        # --- Forecast Plot ---
        st.subheader("📈 Actual vs Predicted Sales")
        if df["units_sold"].isna().any() or df["predicted"].isna().any():
            st.warning("NaN values detected in units_sold or predicted. Filling with mean for plotting.")
            df["units_sold"] = df["units_sold"].fillna(df["units_sold"].mean())
            df["predicted"] = df["predicted"].fillna(df["predicted"].mean())
        st.write(f"Debug: units_sold - min: {df['units_sold'].min()}, max: {df['units_sold'].max()}, mean: {df['units_sold'].mean():.2f}")
        st.write(f"Debug: predicted - min: {df['predicted'].min()}, max: {df['predicted'].max()}, mean: {df['predicted'].mean():.2f}")
        st.write(f"Debug: Mean Absolute Error (MAE) between units_sold and predicted: {mean_absolute_error(df['units_sold'], df['predicted']):.2f}")

        status_text.text("Generating plots...")
        try:
            fig1, ax1 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df, x="date", y="units_sold", label="Actual Sales", ax=ax1, color="#1f77b4", linewidth=3, alpha=0.8)
            sns.lineplot(data=df, x="date", y="predicted", label="Predicted Sales", ax=ax1, color="#ff7f0e", linestyle="--", linewidth=3, alpha=0.8)
            ax1.set_ylabel("Units", color="#ffffff", fontsize=12)
            ax1.set_title("Actual vs Predicted Sales", color="#ffffff", fontsize=16)
            ax1.tick_params(axis="x", colors="#ffffff", rotation=45)
            ax1.tick_params(axis="y", colors="#ffffff")
            ax1.legend(labelcolor="#ffffff")
            plt.tight_layout()
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Error generating forecast plot: {str(e)}")

        # --- Anomalies ---
        st.subheader("🚨 Detected Anomalies")
        anomalies = df[df["anomaly"] == True]
        root_causes = sorted(anomalies["root_cause_hint"].unique())
        selected_causes = st.multiselect("Filter by Root Cause", root_causes, default=root_causes)
        filtered = anomalies[anomalies["root_cause_hint"].isin(selected_causes)]

        if not filtered.empty:
            try:
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                sns.lineplot(data=df, x="date", y="units_sold", label="Actual Sales", ax=ax2, color="#1f77b4", linewidth=3, alpha=0.8)
                sns.scatterplot(data=filtered, x="date", y="units_sold", color="red", label="Anomaly", s=100, marker="X", ax=ax2)
                ax2.set_title("Filtered Anomalies", color="#ffffff", fontsize=16)
                ax2.set_ylabel("Units", color="#ffffff", fontsize=12)
                ax2.tick_params(axis="x", colors="#ffffff", rotation=45)
                ax2.tick_params(axis="y", colors="#ffffff")
                ax2.legend(labelcolor="#ffffff")
                plt.tight_layout()
            except Exception as e:
                st.error(f"Error generating anomalies plot: {str(e)}")
            st.pyplot(fig2)
            st.dataframe(filtered[["date", "units_sold", "predicted", "root_cause_hint"]])
        else:
            st.info("No anomalies match the selected root causes.")

        # --- Stock vs Demand Plot ---
        st.subheader("📦 Stock Levels vs Predicted Demand")
        try:
            fig4, ax4 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df, x="date", y="units_sold", label="Actual Sales", ax=ax4, color="#1f77b4", linewidth=3, alpha=0.8)
            sns.lineplot(data=df, x="date", y="predicted", label="Predicted Demand", ax=ax4, color="#ff7f0e", linestyle="--", linewidth=3, alpha=0.8)
            sns.lineplot(data=df, x="date", y="stock_level", label="Stock Level", ax=ax4, color="#2ca02c", linewidth=3, alpha=0.8)
            ax4.set_title("Stock Levels vs Actual and Predicted Demand", color="#ffffff", fontsize=16)
            ax4.set_ylabel("Units", color="#ffffff", fontsize=12)
            ax4.tick_params(axis="x", colors="#ffffff", rotation=45)
            ax4.tick_params(axis="y", colors="#ffffff")
            ax4.legend(labelcolor="#ffffff")
            plt.tight_layout()
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error generating stock plot: {str(e)}")

        # --- Reorder Recommendations ---
        st.subheader("📦 Reorder Recommendations")
        st.dataframe(df[["date", "predicted", "stock_level", "reorder_quantity"]])

        # --- Feature Importance ---
        st.subheader("📊 Feature Importance (retrained model)")
        try:
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
                "units_sold_7d_avg": "Historical",
                "customer_sentiment": "Social",
                "competitor_promotion": "Market",
                "supply_chain_disruption": "Logistics",
                "units_sold_30d_avg": "Historical"
            }
            for col in [c for c in df.columns if c.startswith("beer_")]:
                categories[col] = "Product"
            for col in [c for c in df.columns if c.startswith("season_")]:
                categories[col] = "Seasonal"
            for col in [c for c in df.columns if c.startswith("region_")]:
                categories[col] = "Regional"

            importance_df = pd.DataFrame({
                "feature": features,
                "importance": importance,
                "category": [categories.get(f, "Other") for f in features]
            }).sort_values(by="importance", ascending=False)

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=importance_df, x="importance", y="feature", hue="category", ax=ax3)
            ax3.set_title("Feature Importance", color="#ffffff", fontsize=16)
            ax3.set_xlabel("Importance", color="#ffffff", fontsize=12)
            ax3.set_ylabel("Feature", color="#ffffff", fontsize=12)
            ax3.tick_params(axis="x", colors="#ffffff")
            ax3.tick_params(axis="y", colors="#ffffff")
            ax3.legend(labelcolor="#ffffff")
            plt.tight_layout()
            st.pyplot(fig3)
            with st.expander("📄 Feature Importance Table"):
                st.dataframe(importance_df)
        except Exception as e:
            st.error(f"Error generating feature importance plot: {str(e)}")

        # --- Prediction Model Equation ---
        st.subheader("🧮 Prediction Model Equation")
        try:
            st.write("The prediction model is an XGBoost ensemble of 50 decision trees, each with a maximum depth of 2, and L1 regularization (reg_alpha=0.1).")
            st.write("The predicted units_sold is a weighted sum of contributions from the following features, based on their importance:")
            
            top_features = importance_df.head(5)[["feature", "importance"]]
            equation = "Predicted units_sold ≈ "
            equation += " + ".join([f"{row['importance']:.3f} * {row['feature']}" for _, row in top_features.iterrows()])
            equation += " + other features"
            st.write(equation)
            with st.expander("📄 Top Feature Contributions"):
                st.dataframe(top_features)
        except Exception as e:
            st.error(f"Error generating model equation: {str(e)}")

        # --- Download Enriched Output ---
        st.download_button(
            label="📥 Download Forecast + Anomaly CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="beer_forecast_with_anomalies.csv",
            mime="text/csv",
        )

        with st.expander("🧾 Show Enriched Data"):
            st.dataframe(df)

        progress_bar.progress(100)
        status_text.text("Processing complete!")

    except Exception as e:
        st.error(f"❌ Error processing app: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

else:
    st.info("Please upload the raw beer sales dataset to get started.")