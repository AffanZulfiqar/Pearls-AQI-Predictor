# Pearls AQI Predictor: Project Submission Report

## 1. Executive Summary
The Pearls AQI Predictor is a 100% serverless, end-to-end machine learning platform designed to forecast the Air Quality Index (AQI) of major cities for up to 72 hours in the future. The project successfully implements a fully automated MLOps pipeline covering automated data ingestion, feature engineering, model training, and real-time prediction serving. The system is designed with no long-running servers, utilizing GitHub Actions for scheduled compute, Hopsworks for the Feature Store and Model Registry, and a modern Streamlit/Flask stack for the user interface.

## 2. Project Requirements Compliance
This project was meticulously designed to fulfill all technical requirements of the assignment:
- **Programming Language:** Python (3.11/3.12).
- **Machine Learning Frameworks:** Scikit-learn (Ridge Regression, Random Forest) and TensorFlow (Deep Neural Networks).
- **Data Management:** Hopsworks (Serverless Feature Store and Model Registry).
- **CI/CD & Orchestration:** GitHub Actions (Automated hourly feature pipelines and daily training pipelines).
- **Web Interface:** Streamlit (Frontend Dashboard) communicating with a Flask (Backend REST API).
- **Data Ingestion:** Live integration with both AQICN and OpenWeather APIs.
- **Explainable AI:** SHAP (Shapley Additive Explanations) integrated for feature importance visualization.
- **Version Control:** Git & GitHub.

## 3. Data Strategy & Ingestion Layer
To achieve high-accuracy predictions, the system relies on a dual-source data ingestion strategy:
1. **Pollutant Telemetry:** Fetched from the AQICN API, providing ground-truth measurements of PM2.5, PM10, NO₂, SO₂, CO, and O₃.
2. **Atmospheric Covariates:** Fetched from the OpenWeather API, providing critical weather factors that influence smog dispersion (Temperature, Humidity, Wind Speed, Pressure, and Precipitation).

**Feature Engineering:**
The raw data is processed hourly to engineer advanced predictive features. These include:
- **Time-based indicators:** Hour of day, day of week, and weekend flags to capture human activity cycles (e.g., morning rush hour).
- **Lag and Rolling Statistics:** 24-hour and 3-hour rolling averages, along with rate-of-change metrics, to capture the inertia of particulate accumulation.

## 4. Machine Learning & Modeling Strategy
The training pipeline is fully automated and experiments with multiple model architectures to find the best fit for the time-series forecasting task:
- **Baseline Model (Ridge Regression):** Provides a linear benchmark to prevent overfitting and ensure mathematical stability.
- **Ensemble Model (Random Forest):** Captures non-linear relationships and complex interactions between weather covariates and pollutant levels.
- **Deep Learning Model (TensorFlow Neural Network):** Analyzes deep patterns in the atmospheric data.

**Evaluation & Model Registry:**
Every 24 hours, the training pipeline splits the data chronologically (preventing data leakage), trains all candidate models for three separate horizons (24h, 48h, 72h), and evaluates them using **RMSE, MAE, and R²**. The system programmatically selects the highest-performing model (minimizing RMSE) and automatically registers it into the Hopsworks Model Registry for live serving.

## 5. Explainable AI & Public Health Focus
Rather than providing a "black box" number, the system focuses heavily on Explainable AI (XAI) and actionable health advisories:
- **SHAP Integration:** The dashboard calculates SHAP values in real-time, displaying exactly which meteorological or pollutant features are driving the current prediction (e.g., high humidity trapping PM2.5).
- **Automated Advisories:** Based on WHO and EPA breakpoints, the system automatically translates raw predicted AQI values into actionable recommendations (e.g., advising sensitive individuals to wear N95 masks when the forecast crosses the 150 AQI threshold).

## 6. Dashboard Architecture & Deployment
The user interface is an enterprise-grade Streamlit application designed for both public presentation and deep analytical inspection:
- **Visuals:** Uses HTML/CSS injection to render a beautiful, responsive, glassmorphism-style UI.
- **Serverless Hosting:** The Streamlit frontend is configured for deployment on Streamlit Community Cloud, while the Flask API is containerized (via `Dockerfile`) for deployment on free serverless platforms (e.g., Hugging Face Spaces, Railway, or Render).
- **Error Handling:** Implements robust asynchronous fetching and UI fallback states to gracefully handle "Cold Starts" inherent to serverless API environments.

## 7. Conclusion
The Pearls AQI Predictor successfully demonstrates the complete lifecycle of a modern Machine Learning operation. By leveraging a serverless architecture, the system achieves enterprise-grade automation and reliability with zero infrastructure overhead.
