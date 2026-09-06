# Pearls AQI Predictor

A 100% serverless, end-to-end machine learning system that forecasts a city's Air Quality Index (AQI) for the **next 3 days** (24h, 48h, 72h horizons). 

Built with Hopsworks Feature Store, GitHub Actions scheduled pipelines, Scikit-learn, TensorFlow, Flask API, and an interactive dual-view Streamlit dashboard.

---

## 🏛️ System Architecture

```
                      ┌─────────────────────────┐
                      │   AQICN / OpenWeather    │
                      │        APIs              │
                      └────────────┬─────────────┘
                                   │ hourly (GitHub Actions cron)
                                   ▼
                      ┌─────────────────────────┐
                      │   Feature Pipeline        │
                      │  (fetch → engineer →      │
                      │   write to Feature Store) │
                      └────────────┬─────────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │   Hopsworks Feature Store │
                      │  (aqi_features FG)        │
                      └───────┬─────────┬─────────┘
                 daily (cron) │         │ on-demand (serverless)
                               ▼         ▼
               ┌───────────────────┐   ┌────────────────────────┐
               │ Training Pipeline  │   │ Inference & UI Layer   │
               │ (3 models, SHAP,   │   │ (Streamlit App +       │
               │  model registry)   │──▶│  Flask REST API)       │
               └────────┬───────────┘   └────────────────────────┘
                        ▼
               ┌───────────────────┐
               │ Hopsworks Model    │
               │ Registry           │
               └───────────────────┘
```

---

## 🖥️ Dashboard Architecture & Dual-View Routing

The application delivers an enterprise-grade visualization layer designed for both executive presentation and developer modularity:

### 1. Executive Glassmorphism Dashboard (Default View)
- **Location:** [`web/`](file:///d:/Pearls%20aqi%20predictor/web/) (`index.html`, `style.css`, `app.js`) rendered via Streamlit's component engine (`streamlit.components.v1.html`).
- **Features:** 
  - Dynamic mathematical SVG circular gauge with responsive color-shifting (Good, Moderate, Sensitive, Unhealthy).
  - EPA color-coded 24-hour diurnal trend line with automatic headroom calculation.
  - Live pollutant telemetry matrix (PM2.5, PM10, NO₂, O₃) with mini sparklines.
  - Interactive SHAP feature importance rankings.
  - Model Performance Championship benchmarks with clickable interactive metric switchers (R² Parity Fit, MAE Horizon Bars, RMSE Residual Distribution).
  - Multi-city switcher: **Islamabad**, **Karachi**, and **Lahore**.
  - WHO & EPA Health Advisories with PDF/print bulletin generation.
- **Data Ingestion:** 
  - On **Streamlit Community Cloud**, Python loads live telemetry and model predictions directly from Hopsworks and injects them server-side into `window.__SERVER_DATA__`.
  - When running locally alongside Flask, client-side asynchronous fetches query `http://127.0.0.1:5000/predict` and `/explain`.
  - Truth-in-labeling status badge dynamically reports `LIVE FEATURE STORE`, `LIVE FLASK INFERENCE`, or `OFFLINE BENCHMARK`.

### 2. Modular Streamlit Component Inspector (`?view=modular`)
- **Route:** Access by appending `?view=modular` to the dashboard URL (e.g. `http://localhost:8501/?view=modular`).
- **Purpose:** Renders the underlying native Streamlit modular components:
  - [`src/dashboard/components/forecast_view.py`](file:///d:/Pearls%20aqi%20predictor/src/dashboard/components/forecast_view.py): Metric cards with EPA color tags.
  - [`src/dashboard/components/eda_view.py`](file:///d:/Pearls%20aqi%20predictor/src/dashboard/components/eda_view.py): 4 distinct Plotly charts (7-day trend, hourly diurnal curve, pollutant correlation matrix, distribution histogram).
  - [`src/dashboard/components/shap_view.py`](file:///d:/Pearls%20aqi%20predictor/src/dashboard/components/shap_view.py): Model feature attribution visualizer.
  - [`src/dashboard/components/alert_banner.py`](file:///d:/Pearls%20aqi%20predictor/src/dashboard/components/alert_banner.py): Hazardous AQI alert trigger.

---

## 🚀 Quick Start

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/pearls-aqi-predictor.git
cd pearls-aqi-predictor

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and provide your API keys:

```env
AQICN_API_KEY=your_aqicn_token
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_hopsworks_project
```

- **AQICN Token:** [https://aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/)
- **OpenWeather Key:** [https://openweathermap.org/api](https://openweathermap.org/api)
- **Hopsworks Account:** [https://hopsworks.ai/](https://hopsworks.ai/)

---

## 🏃 Running Pipelines

All pipelines can be executed standalone from the root directory:

```bash
# 1. Historical Backfill (generate and store historical feature data)
python -m src.backfill.run_backfill --days 60

# 2. Hourly Feature Pipeline (fetch live pollutants & weather, store features)
python -m src.feature_pipeline.run_feature_pipeline

# 3. Daily Training Pipeline (train Ridge, RF, TF models, evaluate, and register)
python -m src.training_pipeline.run_training_pipeline
```

---

## 🌐 Serving & Dashboards

### Option A: Streamlit Dashboard (Standalone / Cloud)
```bash
streamlit run src/dashboard/app.py
```
- Open **`http://localhost:8501`** for the Executive UI.
- Open **`http://localhost:8501/?view=modular`** for the Modular Component Inspector.

### Option B: Flask Inference API (Microservice)
```bash
python -m src.inference.api
```
Exposes REST endpoints on port 5000:
- `GET /health`: Health check (`{"status": "healthy"}`)
- `GET /predict?city=islamabad`: 3-day multi-horizon forecast & alert status
- `GET /explain?city=islamabad`: Top SHAP feature importances

---

## 🧪 Testing

Run unit and integration test suites:
```bash
pytest
```
- Tests cover feature engineering, API client retry logic, model evaluation, and inference alerts.
