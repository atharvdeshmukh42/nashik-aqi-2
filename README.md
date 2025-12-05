# Nashik Air Quality Forecasting App

A Streamlit-based AI application developed to analyze and forecast PM2.5 and PM10 air quality levels in Nashik, India. The tool integrates multiple machine learning and deep learning models to generate accurate predictions using historical data.

---

## Key Features

### Multi-Model Forecasting
The application compares and utilizes predictions from the following models:
- ARIMA (AutoRegressive Integrated Moving Average)
- Prophet (Facebook’s time-series forecasting tool)
- Random Forest Regressor
- XGBoost Regressor
- LSTM (Long Short-Term Memory Neural Network)

### Automated Data Cleaning
- Automatically handles missing dates
- Imputes missing values using linear interpolation
- Removes outliers using rolling Z-score filtering

### Interactive Visualizations
- Forecast comparison charts
- Historical trends using Plotly-based interactive graphs

### Built-in Educational Resources
Includes informational content regarding:
- Air pollution causes and health effects
- National Air Quality Index (AQI) standards
- Comparison of PM2.5 and PM10 particulate matter
- Nashik city and its air quality context

---

## Technology Stack

| Component | Technology |
|----------|-------------|
| User Interface | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-Learn, XGBoost, TensorFlow (Keras) |
| Time-Series Analysis | Prophet, Statsmodels |

---

## Project Structure

```

├── app.py                  # Main application logic
├── requirements.txt        # Python dependencies
├── nashik-air-logo.png     # Logo asset
├── home.md                 # Content for Home tab
├── about-nashik.md         # Content for About Nashik tab
├── about-us.md             # Content for About Us tab
├── air-pollution.md        # Content for Air Pollution tab
├── aqi.md                  # Content for AQI Info tab
└── particulate-matter.md   # Content for PM Info tab

````

---

## Installation and Setup

### Step 1: Clone or Download the Repository
Ensure `app.py` and all `.md` resource files are located in the same folder.

### Step 2 (Optional): Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
````

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

---

## Data Format Guide

To enable forecasting, upload an Excel (.xlsx) file containing daily air quality data with the following columns:

| Column Name | Description                   | Example    |
| ----------- | ----------------------------- | ---------- |
| `dt_time`   | Date in YYYY-MM-DD format     | 2023-01-01 |
| `pm2.5cnc`  | PM2.5 Concentration (numeric) | 45.2       |
| `pm10cnc`   | PM10 Concentration (numeric)  | 88.5       |

### Example Dataset

| dt_time    | pm2.5cnc | pm10cnc |
| ---------- | -------- | ------- |
| 2023-01-01 | 45.2     | 88.5    |
| 2023-01-02 | 42.1     | 85.0    |
| 2023-01-03 | 50.5     | 92.1    |

Note: The application automatically handles missing dates and small data gaps.

---