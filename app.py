import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Note: To prevent inotify errors on Streamlit Cloud, ensure a config.toml file exists in the project root with:
# [server]
# fileWatcherType = "none"

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
        df["hot_day"] = (df["temperature"] > 25).astype(int)
        df = pd.get_dummies(df, columns=["beer_type", "season", "region"], prefix=["beer", "season", "region"])
        
        return df
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

# Cache correlation matrix computation
@st.cache_data
def compute_correlation_matrix(df, features, threshold):
    corr_matrix = df[features].corr()
    if threshold > 0:
        mask = (abs(corr_matrix) < threshold) & (corr_matrix != 1.0)
        corr_matrix = corr_matrix.where(~mask, 0)
    return corr_matrix

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

        # --- Regional Dashboard Filter ---
        st.subheader("🌍 Regional Dashboard")
        st.write("Select a region to view tailored forecasts, anomalies, and inventory recommendations. Choose 'All' to see network-wide data.")
        region_filter = st.selectbox("Select Region", ["All", "Urban", "Suburban", "Rural"])
        if region_filter != "All":
            df_filtered = df[df[f"region_{region_filter}"] == 1]
        else:
            df_filtered = df

        # --- Model Training and Prediction ---
        status_text.text("Training model...")
        features = ["is_weekend", "temperature", "football_match", "holiday", 
                    "precipitation", "lead_time", "promotion", "day_of_week", 
                    "units_sold_lag1", "units_sold_7d_avg", "customer_sentiment", 
                    "competitor_promotion", "supply_chain_disruption", "units_sold_30d_avg", 
                    "hot_day"] + \
                   [col for col in df.columns if col.startswith("beer_") or col.startswith("season_") or col.startswith("region_")]
        X = df[features]
        y = df["units_sold"]

        model = XGBRegressor(n_estimators=50, max_depth=2, reg_alpha=0.1)
        model.fit(X, y)
        df["predicted"] = model.predict(X)
        df_filtered["predicted"] = df["predicted"][df_filtered.index]
        progress_bar.progress(40)

        # --- Feature Importance (Computed Early for Correlation Matrix) ---
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
            "units_sold_30d_avg": "Historical",
            "hot_day": "Weather"
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

        # --- Anomaly Detection ---
        status_text.text("Detecting anomalies...")
        df["error"] = abs(df["units_sold"] - df["predicted"])
        threshold = df["error"].mean() + 2 * df["error"].std()
        df["anomaly"] = df["error"] > threshold
        df_filtered["anomaly"] = df["anomaly"][df_filtered.index]
        progress_bar.progress(60)

        # --- Root Cause Hints ---
        def root_cause(row):
            if row["hot_day"] == 1:
                return "Hot day"
            elif row["football_match"]:
                return "Football match"
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
        df_filtered["root_cause_hint"] = df["root_cause_hint"][df_filtered.index]

        # --- Reorder Quantity ---
        urban_buffer = np.where(df["region_Urban"] == 1, 1.2, 1.0)
        disruption_buffer = np.where(df["supply_chain_disruption"] == 1, 1.5, 1.0)
        df["reorder_quantity"] = (df["predicted"] * (df["lead_time"] + 1) * 
                                 urban_buffer * disruption_buffer - 
                                 df["stock_level"]).clip(lower=0)
        df_filtered["reorder_quantity"] = df["reorder_quantity"][df_filtered.index]
        progress_bar.progress(80)

        # --- Correlation Matrix ---
        st.subheader("🔗 Correlation Matrix")
        st.write("This heatmap shows correlations between the top 5 most important features (from the model) and sales. Values range from -1 (negative) to 1 (positive). Strong correlations (>0.5) indicate key demand drivers and are listed in the table below. Use the slider to filter weak correlations and the checkbox to focus on sales-related features.")
        try:
            # Select top 5 features from importance_df
            top_5_features = importance_df["feature"].head(5).tolist()
            corr_features = ["units_sold"] + top_5_features
            # Ensure all features exist in df_filtered
            corr_features = [f for f in corr_features if f in df_filtered.columns]
            
            show_full_matrix = st.checkbox("Show Full Correlation Matrix", value=True)
            corr_threshold = st.slider("Correlation Threshold (show values above this magnitude)", 0.0, 1.0, 0.3, 0.1)
            
            if not show_full_matrix:
                corr_matrix_temp = df_filtered[corr_features].corr()
                units_sold_corr = corr_matrix_temp["units_sold"].abs()
                corr_features = [f for f in corr_features if f == "units_sold" or units_sold_corr[f] > 0.3]
            
            corr_matrix = compute_correlation_matrix(df_filtered, corr_features, corr_threshold)
            fig5, ax5 = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0, fmt=".2f", 
                        linewidths=0.5, ax=ax5, cbar_kws={"label": "Correlation"},
                        annot_kws={"size": 10, "weight": "bold"})
            ax5.set_title(f"Correlation Matrix (Top 5 Features, {region_filter})", color="#ffffff", fontsize=16)
            ax5.tick_params(axis="x", colors="#ffffff", rotation=45)
            ax5.tick_params(axis="y", colors="#ffffff")
            plt.tight_layout()
            st.pyplot(fig5)

            # Table of strong correlations with units_sold
            st.write("**Strong Correlations with Units Sold (|correlation| > 0.5)**")
            units_sold_corr = corr_matrix["units_sold"].drop("units_sold")
            strong_corr = units_sold_corr[units_sold_corr.abs() > 0.5].sort_values(ascending=False)
            if not strong_corr.empty:
                strong_corr_df = pd.DataFrame({
                    "Feature": strong_corr.index,
                    "Correlation with Units Sold": strong_corr.values
                })
                st.dataframe(strong_corr_df)
            else:
                st.info("No correlations with units_sold exceed |0.5|.")
        except Exception as e:
            st.error(f"Error generating correlation matrix: {str(e)}")

        # --- Forecast Plot ---
        st.subheader("📈 Actual vs Predicted Sales")
        st.write("This plot compares actual beer sales (blue) with AI-predicted sales (orange) to assess forecast accuracy. Look for alignment and note spikes (e.g., Summer or football matches).")
        if df_filtered["units_sold"].isna().any() or df_filtered["predicted"].isna().any():
            st.warning("NaN values detected in units_sold or predicted. Filling with mean for plotting.")
            df_filtered["units_sold"] = df_filtered["units_sold"].fillna(df_filtered["units_sold"].mean())
            df_filtered["predicted"] = df_filtered["predicted"].fillna(df_filtered["predicted"].mean())
        st.write(f"Debug: units_sold - min: {df_filtered['units_sold'].min()}, max: {df_filtered['units_sold'].max()}, mean: {df_filtered['units_sold'].mean():.2f}")
        st.write(f"Debug: predicted - min: {df_filtered['predicted'].min()}, max: {df_filtered['predicted'].max()}, mean: {df_filtered['predicted'].mean():.2f}")
        st.write(f"Debug: Mean Absolute Error (MAE) between units_sold and predicted: {mean_absolute_error(df_filtered['units_sold'], df_filtered['predicted']):.2f}")

        status_text.text("Generating plots...")
        try:
            fig1, ax1 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df_filtered, x="date", y="units_sold", label="Actual Sales", ax=ax1, color="#1f77b4", linewidth=3, alpha=0.8)
            sns.lineplot(data=df_filtered, x="date", y="predicted", label="Predicted Sales", ax=ax1, color="#ff7f0e", linestyle="--", linewidth=3, alpha=0.8)
            ax1.set_ylabel("Units", color="#ffffff", fontsize=12)
            ax1.set_title(f"Actual vs Predicted Sales ({region_filter})", color="#ffffff", fontsize=16)
            ax1.tick_params(axis="x", colors="#ffffff", rotation=45)
            ax1.tick_params(axis="y", colors="#ffffff")
            ax1.legend(labelcolor="#ffffff")
            plt.tight_layout()
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Error generating forecast plot: {str(e)}")

        # --- Anomalies ---
        st.subheader("🚨 Detected Anomalies")
        st.write("This section highlights unexpected sales spikes or drops (red markers) with potential causes (e.g., hot days, football matches). Filter by cause to investigate.")
        anomalies = df_filtered[df_filtered["anomaly"] == True]
        root_causes = sorted(anomalies["root_cause_hint"].unique())
        selected_causes = st.multiselect("Filter by Root Cause", root_causes, default=root_causes, key="anomaly_filter")
        filtered_anomalies = anomalies[anomalies["root_cause_hint"].isin(selected_causes)]

        if not filtered_anomalies.empty:
            try:
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                sns.lineplot(data=df_filtered, x="date", y="units_sold", label="Actual Sales", ax=ax2, color="#1f77b4", linewidth=3, alpha=0.8)
                sns.scatterplot(data=filtered_anomalies, x="date", y="units_sold", color="red", label="Anomaly", s=100, marker="X", ax=ax2)
                ax2.set_title(f"Filtered Anomalies ({region_filter})", color="#ffffff", fontsize=16)
                ax2.set_ylabel("Units", color="#ffffff", fontsize=12)
                ax2.tick_params(axis="x", colors="#ffffff", rotation=45)
                ax2.tick_params(axis="y", colors="#ffffff")
                ax2.legend(labelcolor="#ffffff")
                plt.tight_layout()
                st.pyplot(fig2)
                st.dataframe(filtered_anomalies[["date", "units_sold", "predicted", "root_cause_hint"]])
            except Exception as e:
                st.error(f"Error generating anomalies plot: {str(e)}")
        else:
            st.info("No anomalies match the selected root causes.")

        # --- Stock vs Demand Plot ---
        st.subheader("📦 Stock Levels vs Predicted Demand")
        st.write("This plot compares stock levels (green) with actual (blue) and predicted (orange) demand to identify potential stockouts or overstocking.")
        try:
            fig4, ax4 = plt.subplots(figsize=(14, 4))
            sns.lineplot(data=df_filtered, x="date", y="units_sold", label="Actual Sales", ax=ax4, color="#1f77b4", linewidth=3, alpha=0.8)
            sns.lineplot(data=df_filtered, x="date", y="predicted", label="Predicted Demand", ax=ax4, color="#ff7f0e", linestyle="--", linewidth=3, alpha=0.8)
            sns.lineplot(data=df_filtered, x="date", y="stock_level", label="Stock Level", ax=ax4, color="#2ca02c", linewidth=3, alpha=0.8)
            ax4.set_title(f"Stock Levels vs Actual and Predicted Demand ({region_filter})", color="#ffffff", fontsize=16)
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
        st.write("This table suggests reorder quantities to maintain optimal inventory, accounting for predicted demand, lead time, and buffers for Urban regions (20% extra) and supply chain disruptions (50% extra).")
        st.dataframe(df_filtered[["date", "predicted", "stock_level", "reorder_quantity"]])

        # --- Feature Importance ---
        st.subheader("📊 Feature Importance (retrained model)")
        st.write("This chart shows which factors (e.g., hot days, football matches) most influence sales predictions, helping prioritize inventory strategies.")
        try:
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
        st.write("This equation summarizes how key factors (e.g., past sales, hot days) contribute to the AI’s sales predictions, guiding inventory planning.")
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
        st.subheader("📥 Download Forecast Data")
        st.write("Download the enriched dataset with predictions, anomalies, and reorder quantities for further analysis.")
        st.download_button(
            label="Download Forecast + Anomaly CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"beer_forecast_with_anomalies_{region_filter.lower()}.csv",
            mime="text/csv",
        )

        with st.expander("🧾 Show Enriched Data"):
            st.write("This table shows the full dataset with predictions, anomalies, and reorder quantities for the selected region.")
            st.dataframe(df_filtered)

        progress_bar.progress(100)
        status_text.text("Processing complete!")

    except Exception as e:
        st.error(f"❌ Error processing app: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

else:
    st.info("Please upload the raw beer sales dataset to get started.")