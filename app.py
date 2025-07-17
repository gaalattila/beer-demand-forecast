import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import io

# Note: To prevent inotify errors on Streamlit Cloud, ensure a config.toml file exists in the project root with:
# [server]
# fileWatcherType = "none"

# Set Matplotlib dark theme for Chart.js-like visuals
plt.style.use("dark_background")

st.set_page_config(page_title="🍺 Beer Forecast & Root Cause Analyzer", layout="wide")
st.title("🍺 Beer Demand Forecast & Anomaly Detection")

# Cache data loading and feature engineering
@st.cache_data
def load_and_process_data(file, is_future=False):
    try:
        df = pd.read_csv(file, parse_dates=["date"])
        
        required_cols = {"date", "is_weekend", "temperature", "football_match", "holiday", "season", 
                        "precipitation", "lead_time", "beer_type", "promotion", "stock_level", 
                        "customer_sentiment", "competitor_promotion", "region", "supply_chain_disruption", "units_sold_30d_avg"}
        if not is_future:
            required_cols.add("units_sold")
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            # For future data, allow defaults for non-critical columns
            if is_future:
                for col in missing_cols:
                    if col in ["customer_sentiment", "competitor_promotion", "promotion", "supply_chain_disruption"]:
                        df[col] = 0  # Default for optional columns
                    else:
                        raise ValueError(f"Missing critical column(s) in future data: {missing_cols}")
            else:
                raise ValueError(f"CSV must include columns: {required_cols}")
        
        # Feature Engineering
        df["day_of_week"] = df["date"].dt.dayofweek
        if not is_future:
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

# Function to align future data features with training data
def align_features(future_df, train_df, features):
    # Ensure all training features are present
    for col in features:
        if col not in future_df.columns:
            if col.startswith("beer_") or col.startswith("season_") or col.startswith("region_"):
                future_df[col] = 0  # Set missing dummy variables to 0
            elif col in ["units_sold_lag1", "units_sold_7d_avg"]:
                future_df[col] = train_df["units_sold"].mean()  # Default to historical mean
    # Remove extra columns not in training features
    future_df = future_df[[col for col in future_df.columns if col in features or col == "date"]]
    return future_df

# Upload historical data
uploaded_file = st.file_uploader("📤 Upload raw input file (raw_beer_sales_data.csv)", type=["csv"])

if uploaded_file:
    try:
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load and process historical data
        status_text.text("Loading and processing historical data...")
        df = load_and_process_data(uploaded_file, is_future=False)
        progress_bar.progress(20)
        st.success("✅ Historical data loaded")

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
        st.subheader("⚙️ Customize Model Hyperparameters")
        st.write("Adjust the settings below to tune the AI model's predictions for beer sales. These control how the model learns patterns from the data.")
        
        n_estimators = st.slider(
            "Number of Trees (n_estimators)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Controls how many decision trees the model uses. More trees can improve accuracy but may slow down the app and risk overfitting (memorizing data instead of generalizing)."
        )
        max_depth = st.slider(
            "Maximum Tree Depth (max_depth)",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Sets how complex each tree can be. Deeper trees capture more detailed patterns but may overfit if set too high."
        )
        reg_alpha = st.slider(
            "L1 Regularization (reg_alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Reduces model complexity to prevent overfitting. Higher values make the model simpler, which can improve predictions on new data."
        )
        
        features = ["is_weekend", "temperature", "football_match", "holiday", 
                    "precipitation", "lead_time", "promotion", "day_of_week", 
                    "units_sold_lag1", "units_sold_7d_avg", "customer_sentiment", 
                    "competitor_promotion", "supply_chain_disruption", "units_sold_30d_avg", 
                    "hot_day"] + \
                   [col for col in df.columns if col.startswith("beer_") or col.startswith("season_") or col.startswith("region_")]
        X = df[features]
        y = df["units_sold"]

        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, reg_alpha=reg_alpha)
        model.fit(X, y)
        df["predicted"] = model.predict(X)
        df_filtered["predicted"] = df["predicted"][df_filtered.index]
        historical_mae = mean_absolute_error(df_filtered["units_sold"], df_filtered["predicted"])
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
        st.write(f"Debug: Mean Absolute Error (MAE) between units_sold and predicted: {historical_mae:.2f}")

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

        # --- Prediction Model Equation ---
        st.subheader("🧮 Prediction Model Equation")
        st.write("This equation summarizes how key factors (e.g., past sales, hot days) contribute to the AI’s sales predictions, guiding inventory planning.")
        try:
            st.write(f"The prediction model is an XGBoost ensemble of {n_estimators} decision trees, each with a maximum depth of {max_depth}, and L1 regularization (reg_alpha={reg_alpha}).")
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

        # --- Future Data Predictions ---
        st.subheader("🔮 Future Data Predictions")
        st.write("Upload a CSV with future data (e.g., weather, football matches) to predict future beer sales. Lagged features (e.g., past sales averages) are computed using historical data and predictions. A sample template is available below.")
        
        # Provide sample CSV template
        sample_data = pd.DataFrame({
            "date": ["2025-08-01", "2025-08-02"],
            "is_weekend": [1, 0],
            "temperature": [28.5, 22.0],
            "football_match": [1, 0],
            "holiday": [0, 0],
            "season": ["Summer", "Summer"],
            "precipitation": [0.0, 5.0],
            "lead_time": [3, 3],
            "beer_type": ["Lager", "IPA"],
            "promotion": [1, 0],
            "stock_level": [100, 120],
            "customer_sentiment": [0.5, 0.0],
            "competitor_promotion": [0, 1],
            "region": ["Urban", "Rural"],
            "supply_chain_disruption": [0, 0],
            "units_sold_30d_avg": [150.0, 150.0]
        })
        sample_csv = sample_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample Future Data CSV",
            data=sample_csv,
            file_name="sample_future_data.csv",
            mime="text/csv"
        )

        # Upload future data
        future_file = st.file_uploader("📤 Upload future data CSV", type=["csv"], key="future_data")
        if future_file:
            try:
                status_text.text("Processing future data...")
                future_df = load_and_process_data(future_file, is_future=True)
                
                # Append future data to historical data for lagged features
                combined_df = pd.concat([df[["date", "units_sold"]], future_df.assign(units_sold=np.nan)], ignore_index=True)
                combined_df = combined_df.sort_values("date")
                combined_df["units_sold_lag1"] = combined_df["units_sold"].shift(1).fillna(df["units_sold"].mean())
                combined_df["units_sold_7d_avg"] = combined_df["units_sold"].rolling(window=7, min_periods=1).mean().fillna(df["units_sold"].mean())
                
                # Update future_df with computed lagged features
                future_df = future_df.merge(
                    combined_df[["date", "units_sold_lag1", "units_sold_7d_avg"]],
                    on="date",
                    how="left"
                )
                
                # Align features with training data
                future_df = align_features(future_df, df, features)
                
                # Predict future sales
                X_future = future_df[features]
                future_df["predicted"] = model.predict(X_future)
                future_df["uncertainty_range"] = f"±{historical_mae:.2f} (based on historical MAE)"
                
                # Filter by region
                future_region_filter = st.selectbox("Select Region for Future Predictions", ["All", "Urban", "Suburban", "Rural"], key="future_region")
                if future_region_filter != "All":
                    future_df_filtered = future_df[future_df[f"region_{future_region_filter}"] == 1]
                else:
                    future_df_filtered = future_df

                # Display predictions
                st.write("**Future Sales Predictions**")
                st.write("Note: Lagged features (e.g., past sales averages) for future dates are computed using historical sales and predicted values, which may introduce some uncertainty.")
                display_cols = ["date", "predicted", "uncertainty_range", "temperature", "football_match", "promotion", 
                                "beer_type", "season", "region", "is_weekend", "holiday", "precipitation"]
                display_cols = [col for col in display_cols if col in future_df_filtered.columns or col == "predicted" or col == "uncertainty_range"]
                st.dataframe(future_df_filtered[display_cols])

                # Plot future predictions
                try:
                    fig6, ax6 = plt.subplots(figsize=(14, 4))
                    sns.lineplot(data=future_df_filtered, x="date", y="predicted", label="Predicted Sales", ax=ax6, color="#ff7f0e", linewidth=3, alpha=0.8)
                    ax6.set_ylabel("Units", color="#ffffff", fontsize=12)
                    ax6.set_title(f"Future Sales Predictions ({future_region_filter})", color="#ffffff", fontsize=16)
                    ax6.tick_params(axis="x", colors="#ffffff", rotation=45)
                    ax6.tick_params(axis="y", colors="#ffffff")
                    ax6.legend(labelcolor="#ffffff")
                    plt.tight_layout()
                    st.pyplot(fig6)
                except Exception as e:
                    st.error(f"Error generating future predictions plot: {str(e)}")

                # Download future predictions
                st.download_button(
                    label="📥 Download Future Predictions CSV",
                    data=future_df_filtered.to_csv(index=False).encode("utf-8"),
                    file_name=f"future_beer_predictions_{future_region_filter.lower()}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error processing future data: {str(e)}")

        # --- What-If Scenario Analysis ---
        st.subheader("🔍 What-If Scenario Analysis")
        st.write("Enter data for a single future date to predict sales under specific conditions (e.g., a football match with a promotion).")
        
        with st.form(key="what_if_form"):
            col1, col2 = st.columns(2)
            with col1:
                scenario_date = st.date_input("Date")
                scenario_is_weekend = st.checkbox("Is Weekend")
                scenario_temperature = st.slider("Temperature (°C)", 0.0, 40.0, 20.0, 0.5)
                scenario_football_match = st.checkbox("Football Match")
                scenario_holiday = st.checkbox("Holiday")
                scenario_season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
                scenario_precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
            with col2:
                scenario_lead_time = st.number_input("Lead Time (days)", 0, 10, 3)
                scenario_beer_type = st.selectbox("Beer Type", df["beer_type"].unique())
                scenario_promotion = st.checkbox("Promotion Active")
                scenario_stock_level = st.number_input("Stock Level", 0, 1000, 100)
                scenario_customer_sentiment = st.slider("Customer Sentiment", -1.0, 1.0, 0.0, 0.1)
                scenario_competitor_promotion = st.checkbox("Competitor Promotion")
                scenario_region = st.selectbox("Region", ["Urban", "Suburban", "Rural"])
                scenario_supply_chain_disruption = st.checkbox("Supply Chain Disruption")
                scenario_units_sold_30d_avg = st.number_input("30-Day Avg Sales", 0.0, 1000.0, df["units_sold"].mean())
            
            # Submit button
            submitted = st.form_submit_button("Predict Sales")
            
            if submitted:
                try:
                    # Create DataFrame for scenario
                    scenario_df = pd.DataFrame({
                        "date": [pd.to_datetime(scenario_date)],
                        "is_weekend": [1 if scenario_is_weekend else 0],
                        "temperature": [scenario_temperature],
                        "football_match": [1 if scenario_football_match else 0],
                        "holiday": [1 if scenario_holiday else 0],
                        "season": [scenario_season],
                        "precipitation": [scenario_precipitation],
                        "lead_time": [scenario_lead_time],
                        "beer_type": [scenario_beer_type],
                        "promotion": [1 if scenario_promotion else 0],
                        "stock_level": [scenario_stock_level],
                        "customer_sentiment": [scenario_customer_sentiment],
                        "competitor_promotion": [1 if scenario_competitor_promotion else 0],
                        "region": [scenario_region],
                        "supply_chain_disruption": [1 if scenario_supply_chain_disruption else 0],
                        "units_sold_30d_avg": [scenario_units_sold_30d_avg]
                    })
                    
                    # Process scenario data
                    scenario_df = load_and_process_data(io.StringIO(scenario_df.to_csv(index=False)), is_future=True)
                    scenario_df = align_features(scenario_df, df, features)
                    
                    # Append to historical data for lagged features
                    combined_df = pd.concat([df[["date", "units_sold"]], scenario_df.assign(units_sold=np.nan)], ignore_index=True)
                    combined_df = combined_df.sort_values("date")
                    combined_df["units_sold_lag1"] = combined_df["units_sold"].shift(1).fillna(df["units_sold"].mean())
                    combined_df["units_sold_7d_avg"] = combined_df["units_sold"].rolling(window=7, min_periods=1).mean().fillna(df["units_sold"].mean())
                    scenario_df = scenario_df.merge(
                        combined_df[["date", "units_sold_lag1", "units_sold_7d_avg"]],
                        on="date",
                        how="left"
                    )
                    
                    # Predict
                    X_scenario = scenario_df[features]
                    scenario_pred = model.predict(X_scenario)[0]
                    st.success(f"**Predicted Sales for {scenario_date}:** {scenario_pred:.2f} units ±{historical_mae:.2f} (based on historical MAE)")
                except Exception as e:
                    st.error(f"❌ Error processing scenario: {str(e)}")
            else:
                st.info("Fill in the form and click 'Predict Sales' to see the prediction.")

        # --- Download Historical Forecast Data ---
        st.subheader("📥 Download Historical Forecast Data")
        st.write("Download the enriched historical dataset with predictions, anomalies, and reorder quantities for further analysis.")
        st.download_button(
            label="Download Historical Forecast + Anomaly CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"beer_forecast_with_anomalies_{region_filter.lower()}.csv",
            mime="text/csv",
        )

        with st.expander("🧾 Show Enriched Historical Data"):
            st.write("This table shows the full historical dataset with predictions, anomalies, and reorder quantities for the selected region.")
            st.dataframe(df_filtered)

        progress_bar.progress(100)
        status_text.text("Processing complete!")

    except Exception as e:
        st.error(f"❌ Error processing app: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

else:
    st.info("Please upload the raw beer sales dataset to get started.")