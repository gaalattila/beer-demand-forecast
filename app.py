import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io
import os
from scipy import stats

# Set Matplotlib style for iOS-inspired graphs
plt.rcParams['font.family'] = 'San Francisco'  # iOS default font (approximated with system sans-serif)
plt.rcParams['text.color'] = '#6e6e6e'
plt.rcParams['axes.labelcolor'] = '#6e6e6e'
plt.rcParams['xtick.color'] = '#6e6e6e'
plt.rcParams['ytick.color'] = '#6e6e6e'
plt.rcParams['legend.labelcolor'] = '#6e6e6e'
plt.rcParams['axes.titlecolor'] = '#007aff'
plt.rcParams['axes.edgecolor'] = '#e0e0e0'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#f0f0f0'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5

# Custom CSS for iOS-style UX with readable text and styled graphs/tables
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f7;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp * {
        color: #6e6e6e !important;
    }
    .stApp > h1 {
        color: #6e6e6e !important;
    }
    .card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin-bottom: 15px;
    }
    .card h3 {
        color: #007aff;
        font-size: 1.2em;
        margin-bottom: 5px;
    }
    .card p {
        color: #6e6e6e;
        font-size: 0.9em;
        margin: 0;
    }
    .stButton>button {
        background-color: #007aff;
        color: white;
        border-radius: 10px;
        padding: 5px 15px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    .stSlider, .stSelectbox, .stCheckbox, .stNumberInput {
        border-radius: 10px;
        background-color: #f0f0f0;
        padding: 5px;
    }
    .stExpander {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* Style for Matplotlib graphs */
    .matplotlib-plot {
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: white !important;
        padding: 10px;
    }
    /* Style for tables */
    .stDataFrame {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0 8px;
        font-size: 0.9em;
    }
    .stDataFrame th, .stDataFrame td {
        border: 1px solid #e0e0e0;
        padding: 10px 12px;
        background-color: white;
        text-align: left;
        border-radius: 6px;
    }
    .stDataFrame th {
        background-color: #f0f0f0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Style for container-based sections */
    .stContainer {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin-bottom: 15px;
    }
    .what-if-section h4 {
        color: #007aff;
        font-size: 1em;
        margin-bottom: 5px;
        padding-bottom: 5px;
        border-bottom: 1px solid #e0e0e0;
    }
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6e6e6e;
        padding: 8px 16px;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #007aff;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page config with iOS-inspired title
st.set_page_config(page_title="🍺 Beer Forecast", layout="wide")
st.markdown('<h1 style="color: #6e6e6e;">🍺 Beer Demand Forecast & Anomaly Detection</h1>', unsafe_allow_html=True)

# Initialize session state for metrics
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = []

@st.cache_data
def load_and_process_data(file_path, is_future=False):
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path, parse_dates=["date"])
        else:
            df = pd.read_csv(file_path, parse_dates=["date"])
        
        required_cols = {"date", "is_weekend", "temperature", "football_match", "holiday", "season",
                        "precipitation", "lead_time", "beer_type", "promotion", "stock_level",
                        "customer_sentiment", "competitor_promotion", "region", "supply_chain_disruption", "units_sold_30d_avg"}
        if not is_future:
            required_cols.add("units_sold")
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            if is_future:
                for col in missing_cols:
                    if col in ["customer_sentiment", "competitor_promotion", "promotion", "supply_chain_disruption"]:
                        df[col] = 0
                    elif col == "beer_type":
                        df[col] = "Lager"
                    else:
                        raise ValueError(f"Missing column(s): {missing_cols}")
            else:
                raise ValueError(f"Required columns: {required_cols}")
        
        df["day_of_week"] = df["date"].dt.dayofweek
        if not is_future:
            df["units_sold_lag1"] = df["units_sold"].shift(1).fillna(df["units_sold"].mean())
            df["units_sold_7d_avg"] = df["units_sold"].rolling(7, min_periods=1).mean().fillna(df["units_sold"].mean())
        df["hot_day"] = (df["temperature"] > 25).astype(int)
        df = pd.get_dummies(df, columns=["beer_type", "season", "region"], prefix=["beer", "season", "region"])
        
        return df
    except Exception as e:
        st.error(f"Data load error: {str(e)}")
        return None

@st.cache_data
def compute_correlation_matrix(df, features, threshold):
    if df is None or features is None:
        return pd.DataFrame()
    corr_matrix = df[features].corr()
    if threshold > 0:
        mask = (abs(corr_matrix) < threshold) & (corr_matrix != 1.0)
        corr_matrix = corr_matrix.where(~mask, 0)
    return corr_matrix

def align_features(future_df, train_df, features):
    if future_df is None or train_df is None:
        return pd.DataFrame()
    for col in features:
        if col not in future_df.columns:
            if col.startswith(("beer_", "season_", "region_")):
                future_df[col] = 0
            elif col in ["units_sold_lag1", "units_sold_7d_avg"]:
                future_df[col] = train_df["units_sold"].mean()
    return future_df[[col for col in future_df.columns if col in features or col == "date"]]

# Load data from repo or uploader
data_file = "raw_beer_sales_data.csv"
if os.path.exists(data_file):
    df = load_and_process_data(data_file, is_future=False)
else:
    df = None
    uploaded_file = st.file_uploader("📤 Upload raw_beer_sales_data.csv", type=["csv"])
    if uploaded_file:
        df = load_and_process_data(uploaded_file, is_future=False)

if df is not None:
    try:
        # Main tabs
        tabs = st.tabs(["🌍 Regional Dashboard", "⚙️ Model Hyperparameters", "📊 Feature Importance", 
                       "🔗 Correlation Matrix", "📈 Actual vs Predicted", "🚨 Anomalies", 
                       "📦 Stock vs Demand", "📦 Reorder Recommendations", "🧮 Model Equation", 
                       "📅 Forecast & Simulation", "📥 Download Historical Data"])

        with tabs[0]:  # Regional Dashboard
            st.markdown('<div class="card"><h3>🌍 Regional Dashboard</h3><p>Filter sales data by region to analyze trends. Select "All" for overall data or a specific region (Urban, Suburban, Rural) to focus on localized patterns. Interpret the filtered data as a subset of total sales influenced by regional factors.</p></div>', unsafe_allow_html=True)
            region_filter = st.selectbox("Region", ["All", "Urban", "Suburban", "Rural"])
            df_filtered = df if region_filter == "All" else df[df[f"region_{region_filter}"] == 1]

        with tabs[1]:  # Model Hyperparameters
            st.markdown('<div class="card"><h3>⚙️ Model Hyperparameters</h3><p>Adjust the model"s complexity. "Trees" controls the number of decision trees (for XGB/RF), "Depth" limits tree depth (for XGB/RF), and "L1 Reg" adds regularization (for XGB). Higher values increase accuracy but may overfit; interpret as a trade-off between precision and generalization.</p></div>', unsafe_allow_html=True)
            model_type = st.selectbox("Model", ["XGBRegressor", "RandomForestRegressor", "LinearRegression"])
            n_estimators = st.slider("Trees", 10, 200, 50, 10) if model_type in ["XGBRegressor", "RandomForestRegressor"] else 0
            max_depth = st.slider("Depth", 1, 10, 2, 1) if model_type in ["XGBRegressor", "RandomForestRegressor"] else 0
            reg_alpha = st.slider("L1 Reg", 0.0, 1.0, 0.1, 0.05) if model_type == "XGBRegressor" else 0.0
        
        features = ["is_weekend", "temperature", "football_match", "holiday", 
                   "precipitation", "lead_time", "promotion", "day_of_week", 
                   "units_sold_lag1", "units_sold_7d_avg", "customer_sentiment", 
                   "competitor_promotion", "supply_chain_disruption", "units_sold_30d_avg", 
                   "hot_day"] + [col for col in df.columns if col.startswith(("beer_", "season_", "region_"))]
        X = df[features]
        y = df["units_sold"]

        @st.cache_resource
        def train_model(model_type, n_estimators, max_depth, reg_alpha, X, y):
            if model_type == "XGBRegressor":
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, reg_alpha=reg_alpha)
            elif model_type == "RandomForestRegressor":
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            else:  # LinearRegression
                model = LinearRegression()
            model.fit(X, y)
            return model

        model = train_model(model_type, n_estimators, max_depth, reg_alpha, X, y)
        df["predicted"] = model.predict(X)
        df_filtered["predicted"] = df["predicted"][df_filtered.index]
        mae = mean_absolute_error(df_filtered["units_sold"], df_filtered["predicted"])
        st.session_state.model_metrics.append({"model": model_type, "mae": mae, "time": pd.Timestamp.now().strftime("%H:%M:%S")})

        importance = model.feature_importances_ if hasattr(model, "feature_importances_") else np.zeros(len(features))
        categories = {k: v for k, v in {
            "is_weekend": "Temporal", "temperature": "Weather", "football_match": "Event",
            "holiday": "Holiday", "precipitation": "Weather", "lead_time": "Inventory",
            "promotion": "Marketing", "day_of_week": "Temporal", "units_sold_lag1": "Historical",
            "units_sold_7d_avg": "Historical", "customer_sentiment": "Social",
            "competitor_promotion": "Market", "supply_chain_disruption": "Logistics",
            "units_sold_30d_avg": "Historical", "hot_day": "Weather"
        }.items()}
        for col in [c for c in df.columns if c.startswith("beer_")]: categories[col] = "Product"
        for col in [c for c in df.columns if c.startswith("season_")]: categories[col] = "Seasonal"
        for col in [c for c in df.columns if c.startswith("region_")]: categories[col] = "Regional"

        importance_df = pd.DataFrame({"feature": features, "importance": importance,
                                    "category": [categories.get(f, "Other") for f in features]}).sort_values("importance", ascending=False)

        # Anomaly detection with Isolation Forest
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df["anomaly_score"] = iso_forest.fit_predict(X_scaled)
        df["anomaly"] = df["anomaly_score"] == -1
        df_filtered["anomaly"] = df["anomaly"][df_filtered.index]
        sensitivity = st.slider("Anomaly Sensitivity", 0.01, 0.2, 0.1, 0.01, key="anomaly_sensitivity")
        df["anomaly"] = iso_forest.fit_predict(X_scaled, sample_weight=1 - sensitivity) == -1
        df_filtered["anomaly"] = df["anomaly"][df_filtered.index]

        def root_cause(row):
            if row["hot_day"]: return "Hot day"
            elif row["football_match"]: return "Football"
            elif row["precipitation"] > 10: return "Rainy"
            elif row["promotion"]: return "Promotion"
            elif row["is_weekend"]: return "Weekend"
            elif row["lead_time"] > 5: return "Delay"
            elif row["competitor_promotion"]: return "Comp Promo"
            elif row["supply_chain_disruption"]: return "Disruption"
            return "Unexplained"

        df["root_cause_hint"] = df.apply(lambda row: root_cause(row) if row["anomaly"] else "", axis=1)
        df_filtered["root_cause_hint"] = df["root_cause_hint"][df_filtered.index]

        urban_buffer = np.where(df["region_Urban"] == 1, 1.2, 1.0)
        disruption_buffer = np.where(df["supply_chain_disruption"] == 1, 1.5, 1.0)
        df["reorder_quantity"] = (df["predicted"] * (df["lead_time"] + 1) * urban_buffer * disruption_buffer - df["stock_level"]).clip(0)
        df_filtered["reorder_quantity"] = df["reorder_quantity"][df_filtered.index]

        with tabs[2]:  # Feature Importance
            st.markdown('<div class="card"><h3>📊 Feature Importance</h3><p>Shows the relative impact of each feature on sales predictions. Higher bars indicate stronger influence. Interpret by category (e.g., Weather, Historical) to understand key drivers.</p></div>', unsafe_allow_html=True)
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=importance_df, x="importance", y="feature", hue="category", ax=ax)
                ax.set_title("Feature Importance")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(True)
                st.pyplot(fig)
                with st.expander("Table"): st.dataframe(importance_df)
            except Exception as e: st.error(f"Feature importance error: {str(e)}")

        with tabs[3]:  # Correlation Matrix
            st.markdown('<div class="card"><h3>🔗 Correlation Matrix</h3><p>Displays correlations between sales and top features. Values close to 1 or -1 indicate strong positive or negative relationships. Use the threshold to filter weak correlations.</p></div>', unsafe_allow_html=True)
            try:
                top_features = importance_df["feature"].head(5).tolist()
                corr_features = ["units_sold"] + top_features
                corr_features = [f for f in corr_features if f in df_filtered.columns]
                threshold = st.slider("Corr Threshold", 0.0, 1.0, 0.3, 0.1)
                corr_matrix = compute_correlation_matrix(df_filtered, corr_features, threshold)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0, fmt=".2f", ax=ax)
                ax.set_title("Correlation Matrix")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(True)
                st.pyplot(fig)
                strong_corr = corr_matrix["units_sold"].drop("units_sold")[abs(corr_matrix["units_sold"].drop("units_sold")) > 0.5]
                if not strong_corr.empty:
                    st.write("**Strong Correlations (>0.5)**")
                    st.dataframe(pd.DataFrame({"Feature": strong_corr.index, "Value": strong_corr.values}))
            except Exception as e: st.error(f"Correlation error: {str(e)}")

        with tabs[4]:  # Actual vs Predicted
            st.markdown('<div class="card"><h3>📈 Actual vs Predicted</h3><p>Compares actual sales (blue) with predicted sales (orange dashed). Use to assess model accuracy; close alignment indicates good predictions, while deviations suggest areas for improvement.</p></div>', unsafe_allow_html=True)
            try:
                if df_filtered["units_sold"].isna().any() or df_filtered["predicted"].isna().any():
                    df_filtered["units_sold"] = df_filtered["units_sold"].fillna(df_filtered["units_sold"].mean())
                    df_filtered["predicted"] = df_filtered["predicted"].fillna(df_filtered["predicted"].mean())
                fig, ax = plt.subplots(figsize=(14, 4))
                sns.lineplot(data=df_filtered, x="date", y="units_sold", label="Actual", ax=ax, color="#1f77b4")
                sns.lineplot(data=df_filtered, x="date", y="predicted", label="Predicted", ax=ax, color="#ff7f0e", linestyle="--")
                ax.set_title(f"Actual vs Predicted ({region_filter})")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e: st.error(f"Forecast error: {str(e)}")

        with tabs[5]:  # Anomalies
            st.markdown('<div class="card"><h3>🚨 Anomalies</h3><p>Highlights unusual sales spikes (red dots) against actual sales (blue). Filter by cause to identify events like promotions or disruptions; interpret as outliers needing investigation.</p></div>', unsafe_allow_html=True)
            anomalies = df_filtered[df_filtered["anomaly"]]
            causes = sorted(anomalies["root_cause_hint"].unique())
            selected_causes = st.multiselect("Filter Causes", causes, default=causes)
            filtered_anomalies = anomalies[anomalies["root_cause_hint"].isin(selected_causes)]
            if not filtered_anomalies.empty:
                try:
                    fig, ax = plt.subplots(figsize=(14, 4))
                    sns.lineplot(data=df_filtered, x="date", y="units_sold", label="Actual", ax=ax, color="#1f77b4")
                    sns.scatterplot(data=filtered_anomalies, x="date", y="units_sold", color="red", label="Anomaly", ax=ax)
                    ax.set_title(f"Anomalies ({region_filter})")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.grid(True)
                    st.pyplot(fig)
                    st.dataframe(filtered_anomalies[["date", "units_sold", "predicted", "root_cause_hint"]])
                except Exception as e: st.error(f"Anomaly plot error: {str(e)}")

        with tabs[6]:  # Stock vs Demand
            st.markdown('<div class="card"><h3>📦 Stock vs Demand</h3><p>Shows actual sales (blue), predicted sales (orange dashed), and stock levels (green). Use to identify stock shortages (when stock falls below predicted demand) and plan inventory.</p></div>', unsafe_allow_html=True)
            try:
                fig, ax = plt.subplots(figsize=(14, 4))
                sns.lineplot(data=df_filtered, x="date", y="units_sold", label="Actual", ax=ax, color="#1f77b4")
                sns.lineplot(data=df_filtered, x="date", y="predicted", label="Predicted", ax=ax, color="#ff7f0e", linestyle="--")
                sns.lineplot(data=df_filtered, x="date", y="stock_level", label="Stock", ax=ax, color="#2ca02c")
                ax.set_title(f"Stock vs Demand ({region_filter})")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e: st.error(f"Stock plot error: {str(e)}")

        with tabs[7]:  # Reorder Recommendations
            st.markdown('<div class="card"><h3>📦 Reorder Recommendations</h3><p>Lists predicted sales, current stock, and suggested reorder quantities. A positive reorder value indicates the amount needed to meet demand; interpret as an inventory action plan.</p></div>', unsafe_allow_html=True)
            st.dataframe(df_filtered[["date", "predicted", "stock_level", "reorder_quantity"]])

        with tabs[8]:  # Model Equation
            st.markdown('<div class="card"><h3>🧮 Model Equation</h3><p>Provides a simplified linear approximation of the prediction model based on top features. Use to understand which factors most affect sales; "others" represent minor contributors.</p></div>', unsafe_allow_html=True)
            try:
                top_features = importance_df.head(5)[["feature", "importance"]]
                equation = "Predicted ≈ " + " + ".join([f"{imp:.3f}*{feat}" for feat, imp in zip(top_features["feature"], top_features["importance"])]) + " + others"
                st.write(f"Model: {model_type} (n_estimators={n_estimators}, depth={max_depth}, reg_alpha={reg_alpha})")
                st.write(equation)
                with st.expander("Table"): st.dataframe(top_features)
            except Exception as e: st.error(f"Equation error: {str(e)}")

        with tabs[9]:  # Forecast & Simulation (parent tab)
            forecast_tabs = st.tabs(["🔮 Future Predictions", "🔍 What-If Analysis"])
            
            with forecast_tabs[0]:  # Future Predictions
                st.markdown('<div class="card"><h3>🔮 Future Predictions</h3><p>Upload future data to predict sales. Results show predicted units with 95% prediction intervals. Interpret higher predictions as potential demand increases, adjusted by region.</p></div>', unsafe_allow_html=True)
                sample_data = pd.DataFrame({
                    "date": ["2025-07-18", "2025-07-19"], "is_weekend": [0, 1], "temperature": [25.0, 28.0],
                    "football_match": [0, 1], "holiday": [0, 0], "season": ["Summer", "Summer"],
                    "precipitation": [0.0, 5.0], "lead_time": [3, 3], "beer_type": ["Lager", "IPA"],
                    "promotion": [0, 1], "stock_level": [100, 120], "customer_sentiment": [0.0, 0.5],
                    "competitor_promotion": [0, 0], "region": ["Urban", "Rural"], "supply_chain_disruption": [0, 0],
                    "units_sold_30d_avg": [150.0, 150.0]
                })
                st.download_button("📥 Sample CSV", data=sample_data.to_csv(index=False).encode(), file_name="sample_future.csv", mime="text/csv")

                future_file = st.file_uploader("📤 Upload future data", type=["csv"], key="future_upload")
                if future_file:
                    try:
                        future_df = load_and_process_data(future_file, is_future=True)
                        if future_df is not None:
                            combined = pd.concat([df[["date", "units_sold"]], future_df.assign(units_sold=np.nan)])
                            combined["units_sold_lag1"] = combined["units_sold"].shift(1).fillna(df["units_sold"].mean())
                            combined["units_sold_7d_avg"] = combined["units_sold"].rolling(7, min_periods=1).mean().fillna(df["units_sold"].mean())
                            future_df = future_df.merge(combined[["date", "units_sold_lag1", "units_sold_7d_avg"]], on="date")
                            future_df = align_features(future_df, df, features)

                            # Bootstrapped prediction intervals
                            n_bootstraps = 100
                            bootstrapped_preds = np.zeros((n_bootstraps, len(future_df)))
                            for i in range(n_bootstraps):
                                sample_idx = np.random.randint(0, len(df), len(df))
                                sample_X = X.iloc[sample_idx]
                                sample_y = y.iloc[sample_idx]
                                boot_model = train_model(model_type, n_estimators, max_depth, reg_alpha, sample_X, sample_y)
                                bootstrapped_preds[i] = boot_model.predict(future_df[features])
                            lower_bound = np.percentile(bootstrapped_preds, 2.5, axis=0)
                            upper_bound = np.percentile(bootstrapped_preds, 97.5, axis=0)
                            future_df["predicted"] = model.predict(future_df[features])
                            future_df["lower_bound"] = lower_bound
                            future_df["upper_bound"] = upper_bound

                            future_region = st.selectbox("Future Region", ["All", "Urban", "Suburban", "Rural"], key="future_region")
                            future_df_filtered = future_df if future_region == "All" else future_df[future_df[f"region_{future_region}"] == 1]
                            st.write("**Predictions with 95% Intervals**")
                            st.dataframe(future_df_filtered[["date", "predicted", "lower_bound", "upper_bound"] + [c for c in future_df.columns if c in ["temperature", "football_match", "promotion"]]])
                            st.download_button("📥 Predictions", data=future_df_filtered.to_csv(index=False).encode(), file_name=f"future_{future_region.lower()}.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Future data error: {str(e)}")

            with forecast_tabs[1]:  # What-If Analysis
                with st.container():
                    st.markdown('<h3>🔍 What-If Analysis</h3>', unsafe_allow_html=True)
                    st.markdown('<p>Simulate sales for multiple days. Adjust inputs (e.g., weather, promotions) and click "Predict Sales" to see results with 95% prediction intervals. Interpret predictions as estimates with uncertainty.</p>', unsafe_allow_html=True)
                    with st.form(key="what_if_form_v6"):
                        # Temporal Factors
                        st.markdown('<div class="what-if-section"><h4>🗓️ Temporal Factors</h4></div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input("Start Date", value=pd.to_datetime("2025-07-17"), key="wi_start_date")
                            num_days = st.number_input("Number of Days", 1, 30, 1, key="wi_num_days")
                        with col2:
                            is_weekend = st.checkbox("Weekend", key="wi_weekend")
                            holiday = st.checkbox("Holiday", key="wi_holiday")
                            football = st.checkbox("Football Match", key="wi_football")

                        # Weather Conditions
                        st.markdown('<div class="what-if-section"><h4>🌡️ Weather Conditions</h4></div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            temp = st.slider("Temperature (°C)", 0.0, 40.0, 20.0, key="wi_temp")
                        with col2:
                            precip = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, key="wi_precip")

                        # Inventory & Logistics
                        st.markdown('<div class="what-if-section"><h4>📦 Inventory & Logistics</h4></div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            stock = st.number_input("Stock", 0, 1000, 100, key="wi_stock")
                            lead = st.number_input("Lead Time", 0, 10, 3, key="wi_lead")
                        with col2:
                            disruption = st.checkbox("Supply Disruption", key="wi_disrupt")

                        # Marketing & Market
                        st.markdown('<div class="what-if-section"><h4>📢 Marketing & Market</h4></div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            promo = st.checkbox("Promotion", key="wi_promo")
                            comp_promo = st.checkbox("Competitor Promo", key="wi_comp")
                        with col2:
                            sentiment = st.slider("Customer Sentiment", -1.0, 1.0, 0.0, key="wi_sent")

                        # Product Context
                        st.markdown('<div class="what-if-section"><h4>🍺 Product Context</h4></div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            beer = st.selectbox("Beer Type", df["beer_type"].unique() if "beer_type" in df.columns else ["Lager"], key="wi_beer")
                            region = st.selectbox("Region", ["Urban", "Suburban", "Rural"], key="wi_region")
                        with col2:
                            season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], key="wi_season")

                        # Sales History Input
                        st.markdown('<div class="what-if-section"><h4>📊 Sales History Input</h4></div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            avg_sales = st.number_input("30d Avg", 0.0, 1000.0, df["units_sold"].mean(), key="wi_avg")
                        with col2:
                            st.write("Lag1 & 7d Avg calculated internally")

                        submitted = st.form_submit_button("Predict Sales")
                        
                        if submitted:
                            try:
                                lag1_mean = df["units_sold_lag1"].mean() if "units_sold_lag1" in df.columns else df["units_sold"].mean()
                                avg7d_mean = df["units_sold_7d_avg"].mean() if "units_sold_7d_avg" in df.columns else df["units_sold"].mean()
                                dates = [start_date + pd.Timedelta(days=i) for i in range(num_days)]
                                scenario = pd.DataFrame({
                                    "date": dates, "is_weekend": [1 if (d.weekday() >= 5 or is_weekend) else 0 for d in dates],
                                    "temperature": [temp] * num_days, "football_match": [1 if football else 0] * num_days,
                                    "holiday": [1 if holiday else 0] * num_days, "season": [season] * num_days,
                                    "precipitation": [precip] * num_days, "lead_time": [lead] * num_days, "beer_type": [beer] * num_days,
                                    "promotion": [1 if promo else 0] * num_days, "stock_level": [stock] * num_days,
                                    "customer_sentiment": [sentiment] * num_days, "competitor_promotion": [1 if comp_promo else 0] * num_days,
                                    "region": [region] * num_days, "supply_chain_disruption": [1 if disruption else 0] * num_days,
                                    "units_sold_30d_avg": [avg_sales] * num_days, "units_sold_lag1": [lag1_mean] * num_days,
                                    "units_sold_7d_avg": [avg7d_mean] * num_days
                                })
                                scenario = load_and_process_data(io.StringIO(scenario.to_csv(index=False)), is_future=True)
                                if scenario is not None:
                                    scenario = align_features(scenario, df, features)

                                    # Bootstrapped prediction intervals
                                    n_bootstraps = 100
                                    bootstrapped_preds = np.zeros((n_bootstraps, len(scenario)))
                                    for i in range(n_bootstraps):
                                        sample_idx = np.random.randint(0, len(df), len(df))
                                        sample_X = X.iloc[sample_idx]
                                        sample_y = y.iloc[sample_idx]
                                        boot_model = train_model(model_type, n_estimators, max_depth, reg_alpha, sample_X, sample_y)
                                        bootstrapped_preds[i] = boot_model.predict(scenario[features])
                                    lower_bound = np.percentile(bootstrapped_preds, 2.5, axis=0)
                                    upper_bound = np.percentile(bootstrapped_preds, 97.5, axis=0)
                                    scenario["predicted"] = model.predict(scenario[features])
                                    scenario["lower_bound"] = lower_bound
                                    scenario["upper_bound"] = upper_bound

                                    st.success("Multi-Day Predictions")
                                    st.dataframe(scenario[["date", "predicted", "lower_bound", "upper_bound"]])
                            except Exception as e:
                                st.error(f"Scenario error: {str(e)}")
                        else:
                            st.info("Click 'Predict Sales' to see results")

        with tabs[10]:  # Download Historical Data
            st.markdown('<div class="card"><h3>📥 Download Historical Data</h3><p>Download the filtered historical data with predictions. Use to export results for further analysis; includes all columns shown in the dashboard.</p></div>', unsafe_allow_html=True)
            st.download_button("Download Forecast", data=df_filtered.to_csv(index=False).encode(), file_name=f"forecast_{region_filter.lower()}.csv", mime="text/csv")

        # Display model metrics
        with st.expander("Model Metrics History"):
            st.dataframe(pd.DataFrame(st.session_state.model_metrics))

    except Exception as e:
        st.error(f"App error: {str(e)}")
else:
    st.info("Upload or place raw_beer_sales_data.csv to start.")