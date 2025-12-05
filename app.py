import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

# Configuration & Setup
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Nashik Air Quality Forecasting App",
    page_icon="nashik-air-logo.png",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #fbfbfb;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .subtle-text {
        color: #7f8c8d;
        font-size: 0.9em;
    }
    .stFileUploader {
        padding: 10px;
        border: 2px dashed #bdc3c7;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions

def calc_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle division by zero for MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = np.nan_to_num(mape) # Replace nan/inf with 0 if any
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae, mape

def create_lag_features(series, n_lags=14):
    """Create lag features for supervised learning."""
    df_lag = pd.DataFrame({'y': series})
    for lag in range(1, n_lags + 1):
        df_lag[f'lag_{lag}'] = df_lag['y'].shift(lag)
    df_lag = df_lag.dropna()
    return df_lag

@st.cache_data
def load_and_preprocess_data(file):
    """
    Load data, handle missing values, outliers, and standardization.
    """
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # Check necessary columns
    required_cols = ['dt_time', 'pm2.5cnc', 'pm10cnc']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Excel sheet must contain columns: {required_cols}")
        return None

    # Standardize Date Column
    df['dt_time'] = pd.to_datetime(df['dt_time'], errors='coerce')
    df = df.dropna(subset=['dt_time'])
    df = df.set_index('dt_time').sort_index()

    # Select Targets
    df = df[['pm2.5cnc', 'pm10cnc']].copy()

    # 1. FIX MISSING DATES
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx)

    # 2. IMPUTE MISSING VALUES (Linear Interpolation)
    df = df.interpolate(method='linear')

    # 3. HANDLE OUTLIERS (Rolling Z-Score)
    for col in ['pm2.5cnc', 'pm10cnc']:
        rolling_mean = df[col].rolling(window=7, center=True).mean()
        rolling_std = df[col].rolling(window=7, center=True).std()
        
        # Fill NaN in rolling stats (edges)
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')

        outlier_mask = np.abs(df[col] - rolling_mean) > 3 * rolling_std
        if outlier_mask.any():
            df.loc[outlier_mask, col] = rolling_mean[outlier_mask]

    # Final clean
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def train_models(series, target_name=None, n_lags=14, test_size=30):
    """
    Trains ARIMA, Prophet, RF, XGB, LSTM and returns metrics and predictions.
    """
    if target_name:
        st.info(f"Training models for {target_name}...")

    results = []
    predictions = pd.DataFrame(index=series.index[-test_size:])
    predictions['Actual'] = series.iloc[-test_size:].values

    # Split Data
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
    
    # ARIMA
    try:
        model_arima = ARIMA(train_series, order=(2, 1, 2))
        model_fit = model_arima.fit()
        fc_arima = model_fit.forecast(steps=test_size)
        
        rmse, mae, mape = calc_metrics(test_series, fc_arima)
        results.append({'Model': 'ARIMA', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        predictions['ARIMA'] = fc_arima.values
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")

    # Prophet
    try:
        df_prophet = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=test_size, freq='D')
        forecast = m.predict(future)
        fc_prophet = forecast['yhat'].iloc[-test_size:].values
        
        rmse, mae, mape = calc_metrics(test_series, fc_prophet)
        results.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        predictions['Prophet'] = fc_prophet
    except Exception as e:
        st.warning(f"Prophet failed: {e}")

    # ML Data Preparation
    df_lag = create_lag_features(series, n_lags)
    X = df_lag.drop('y', axis=1).values
    y = df_lag['y'].values
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        fc_rf = rf.predict(X_test)
        
        rmse, mae, mape = calc_metrics(y_test, fc_rf)
        results.append({'Model': 'Random Forest', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        predictions['Random Forest'] = fc_rf
    except Exception as e:
        st.warning(f"Random Forest failed: {e}")

    # XGBoost
    try:
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
        xgb.fit(X_train, y_train)
        fc_xgb = xgb.predict(X_test)
        
        rmse, mae, mape = calc_metrics(y_test, fc_xgb)
        results.append({'Model': 'XGBoost', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        predictions['XGBoost'] = fc_xgb
    except Exception as e:
        st.warning(f"XGBoost failed: {e}")

    # LSTM
    try:
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        tf.random.set_seed(42)
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, input_shape=(n_lags, 1)))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_train_lstm, y_train, epochs=30, batch_size=16, verbose=0)
        
        fc_lstm = model_lstm.predict(X_test_lstm, verbose=0).flatten()
        
        rmse, mae, mape = calc_metrics(y_test, fc_lstm)
        results.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        predictions['LSTM'] = fc_lstm
    except Exception as e:
        st.warning(f"LSTM failed: {e}")

    results_df = pd.DataFrame(results).sort_values(by='RMSE').reset_index(drop=True)
    return results_df, predictions


# Main Header
col1, col2 = st.columns([0.8, 5])
with col1:
    st.image("nashik-air-logo.png", width=80)
with col2:
    st.title("Nashik Air Quality Forecasting App")

# Navigation Tabs
tab_names = [
    "Forecasting", 
    "About Nashik Air", 
    "About Nashik", 
    "About Us", 
    "Air Pollution", 
    "AQI Info", 
    "Particulate Matter"
]

tabs = st.tabs(tab_names)

# Tab 1: Forecasting App
with tabs[0]:
    st.write("Advanced AI models to forecast PM2.5 and PM10 levels.")

    # Upload Section
    st.header("Data Input")

    with st.container(border=True):
        
        st.subheader("Upload Your Excel Data")
        st.markdown("Upload your daily air quality Excel file (`.xlsx`) here.")
        
        uploaded_file = st.file_uploader("Choose file", type=['xlsx'], label_visibility="collapsed")

        # Format Guide
        st.subheader("Data Format Guide")
        st.markdown("Ensure your uploaded file columns match the required format below. The app handles date parsing and cleaning automatically.")
        
        # Example
        example_data = pd.DataFrame({
            'dt_time': [pd.Timestamp('2021-07-01'), pd.Timestamp('2021-07-02'), pd.Timestamp('2021-07-03')],
            'pm2.5cnc': [16.66, 15.85, 15.26],
            'pm10cnc': [19.50, 19.39, 18.37]
        })
        
        # Display table
        st.dataframe(
            example_data, 
            hide_index=True, 
            use_container_width=True, 
            height=150,
            column_config={
                "dt_time": st.column_config.DateColumn("Date (dt_time)", format="YYYY-MM-DD"),
                "pm2.5cnc": st.column_config.NumberColumn("PM 2.5 (pm2.5cnc)", format="%.2f"),
                "pm10cnc": st.column_config.NumberColumn("PM 10 (pm10cnc)", format="%.2f"),
            }
        )

    st.divider()

    # Processing Logic
    if uploaded_file is not None:
        # Data Loading
        with st.spinner("Processing and Cleaning data..."):
            df = load_and_preprocess_data(uploaded_file)

        if df is not None:
            st.success(f"Data loaded successfully! Total records: {len(df)}")

            # Historical Data Overview
            st.header("Analysis & Forecasting")
            st.subheader("Historical Data Overview")
            
            subtab1, subtab2 = st.tabs(["Historical Graph", "Raw Data"])
            
            with subtab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['pm2.5cnc'], mode='lines', name='PM2.5'))
                fig.add_trace(go.Scatter(x=df.index, y=df['pm10cnc'], mode='lines', name='PM10'))
                fig.update_layout(
                    title='Concentration Over Time',
                    xaxis_title='Date',
                    yaxis_title='Concentration',
                    hovermode='x unified',
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                st.dataframe(df.tail(10), use_container_width=True)

            st.markdown("---")
            
            # Model Training & Forecasting
            st.subheader("Model Training & Forecasting (Last 30 Days)")
            
            pollutant_tabs = st.tabs(["PM 2.5 Forecasting", "PM 10 Forecasting"])
            
            # PM 2.5 Logic
            with pollutant_tabs[0]:
                st.markdown("### PM 2.5 Analysis")
                if st.button("Train Models for PM 2.5", key="train_pm25"):
                    with st.spinner("Training models (ARIMA, Prophet, RF, XGBoost, LSTM)..."):
                        metrics_pm25, preds_pm25 = train_models(df['pm2.5cnc'], target_name="PM2.5")
                    
                    st.write("#### Accuracy Metrics")
                    st.table(metrics_pm25.style.format({
                        'RMSE': "{:.4f}", 
                        'MAE': "{:.4f}", 
                        'MAPE': "{:.4f}"
                    }))
                    
                    st.write("#### Forecast vs Actual")
                    fig_pm25 = px.line(preds_pm25, x=preds_pm25.index, y=preds_pm25.columns,
                                        title="PM 2.5: Actual vs Predicted",
                                        labels={"value": "Concentration", "index": "Date", "variable": "Model"})
                    fig_pm25.update_layout(hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig_pm25, use_container_width=True)

                    st.write("#### Model Predictions Table")
                    st.dataframe(preds_pm25.style.format("{:.4f}"), use_container_width=True)
                else:
                    st.info("Click the button above to start training models for PM 2.5.")

            # PM 10 Logic
            with pollutant_tabs[1]:
                st.markdown("### PM 10 Analysis")
                if st.button("Train Models for PM 10", key="train_pm10"):
                    with st.spinner("Training models (ARIMA, Prophet, RF, XGBoost, LSTM)..."):
                        metrics_pm10, preds_pm10 = train_models(df['pm10cnc'], target_name="PM10")
                    
                    st.write("#### Accuracy Metrics")
                    st.table(metrics_pm10.style.format({
                        'RMSE': "{:.4f}", 
                        'MAE': "{:.4f}", 
                        'MAPE': "{:.4f}"
                    }))
                    
                    st.write("#### Forecast vs Actual")
                    fig_pm10 = px.line(preds_pm10, x=preds_pm10.index, y=preds_pm10.columns,
                                        title="PM 10: Actual vs Predicted",
                                        labels={"value": "Concentration", "index": "Date", "variable": "Model"})
                    fig_pm10.update_layout(hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig_pm10, use_container_width=True)

                    st.write("#### Model Predictions Table")
                    st.dataframe(preds_pm10.style.format("{:.4f}"), use_container_width=True)
                else:
                    st.info("Click the button above to start training models for PM 10.")

# Other Info Tabs
info_pages_map = {
    1: "home.md",
    2: "about-nashik.md",
    3: "about-us.md",
    4: "air-pollution.md",
    5: "aqi.md",
    6: "particulate-matter.md"
}

for i in range(1, len(tab_names)):
    with tabs[i]:
        filename = info_pages_map.get(i)
        if filename:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            except FileNotFoundError:
                st.error(f"File {filename} not found.")