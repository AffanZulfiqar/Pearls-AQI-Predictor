# Pearls AQI Predictor — Project Specification & Implementation Blueprint

**Document type:** Handoff spec for AI coding agent (Antigravity)
**Prepared by:** Senior Software Architect / TPM review of raw project notes
**Status:** v1.0 — Ready for implementation

---

## 1. Project Overview & Objectives

### 1.1 Summary
Pearls AQI Predictor is a **100% serverless, end-to-end machine learning system** that forecasts a city's Air Quality Index (AQI) for the **next 3 days**. The system continuously ingests live weather and pollutant data, engineers features, stores them in a feature store, retrains forecasting models on a schedule, and serves predictions through an interactive web dashboard — with no long-running servers to manage.

### 1.2 Why
Air quality directly affects public health, and forecasts are often coarse, delayed, or not localized. This project builds a fully automated pipeline that keeps itself up to date (hourly feature refresh, daily retraining) and gives an interpretable, explorable forecast rather than a black-box number.

### 1.3 Core Value Proposition
- **Automated & self-updating**: no manual intervention needed once deployed — data collection, feature engineering, and retraining all run on schedules.
- **Explainable**: SHAP-based feature importance shows *why* the model predicts what it predicts, not just the number.
- **Actionable**: hazardous-AQI alerting turns a forecast into a warning.
- **Serverless & low-cost**: built entirely on managed/serverless components (feature store, scheduled functions, hosted dashboard) — no infrastructure to provision or patch.

### 1.4 Target User
- Primary: an individual or small team who wants a live, self-maintaining AQI forecast dashboard for one or more cities (portfolio/demo project, but built to production-grade practices).
- Secondary: anyone evaluating the project as a reference implementation of a serverless MLOps pipeline (feature store → training pipeline → model registry → inference app).

### 1.5 Explicit Out-of-Scope (v1)
- Multi-tenant user accounts / auth-gated dashboards.
- Mobile app (web dashboard only).
- Forecasting horizons beyond 3 days.
- Support for arbitrary/unbounded number of cities in v1 (start with 1–3 configurable cities; architecture should not preclude scaling later).

---

## 2. Tech Stack & Architecture

### 2.1 Recommended Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | **Python 3.11+** | Single language across all pipelines for simplicity |
| Data source | **AQICN API** (primary), OpenWeather API (secondary/supplemental) | AQICN gives direct AQI + pollutant readings; OpenWeather supplies weather covariates (temp, humidity, wind, pressure) |
| Feature Store | **Hopsworks (Serverless/Hopsworks.ai free tier)** | Chosen over Vertex AI Feature Store for zero-cost serverless tier and native feature-group/training-dataset abstractions well suited to this pipeline shape. (Swap-in note: if GCP is preferred, Vertex AI Feature Store is a drop-in architectural substitute — see §2.4.) |
| Orchestration / CI-CD | **GitHub Actions** (scheduled workflows) | Chosen over Airflow: no server/orchestrator to host — pure serverless via `cron` schedule triggers. Airflow remains a valid substitute if a persistent orchestrator is preferred. |
| ML libraries | **Scikit-learn** (Random Forest, Ridge Regression), **TensorFlow** (optional deep learning model, e.g. small dense NN or LSTM) | Multiple model classes trained and compared |
| Model Registry | **Hopsworks Model Registry** | Co-located with feature store for a single integrated MLOps backend |
| Explainability | **SHAP** | TreeExplainer for tree models, KernelExplainer/DeepExplainer fallback for TF model |
| Backend/API | **Flask** (or FastAPI — see note) | Serves prediction endpoint(s) consumed by the dashboard; FastAPI is an acceptable substitute if the coding agent prefers async + auto-docs, but keep to one framework |
| Frontend/Dashboard | **Streamlit** | Fastest path to an interactive, deployable dashboard (Streamlit Community Cloud is free serverless hosting) |
| Version control | **Git / GitHub** | Also serves as the CI/CD trigger source |
| Hosting | GitHub Actions (compute/schedules) + Streamlit Community Cloud (dashboard) + Hopsworks (data/model storage) | No servers owned by the project |

### 2.2 Architecture Diagram (logical)

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
                     │  (Feature Groups)          │
                     └───────┬─────────┬─────────┘
                daily (cron) │         │ on-demand (read)
                              ▼         ▼
              ┌───────────────────┐   ┌────────────────────────┐
              │ Training Pipeline  │   │ Inference / App Layer   │
              │ (train, evaluate,  │   │ (Flask API + Streamlit  │
              │  register model)   │──▶│  dashboard)             │
              └────────┬───────────┘   └────────────────────────┘
                       ▼
              ┌───────────────────┐
              │ Hopsworks Model    │
              │ Registry           │
              └───────────────────┘
```

### 2.3 High-Level Folder Structure

```
pearls-aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml       # hourly cron
│       ├── training_pipeline.yml      # daily cron
│       └── backfill.yml               # manual/dispatch trigger
├── src/
│   ├── config/
│   │   ├── settings.py                # city list, API keys via env, constants
│   │   └── logging_config.py
│   ├── data_ingestion/
│   │   ├── aqicn_client.py
│   │   └── openweather_client.py
│   ├── feature_pipeline/
│   │   ├── fetch_raw_data.py
│   │   ├── feature_engineering.py     # time features, lags, rolling stats, AQI change rate
│   │   ├── feature_store_writer.py
│   │   └── run_feature_pipeline.py    # entrypoint invoked by GH Actions
│   ├── backfill/
│   │   └── run_backfill.py            # historical backfill entrypoint
│   ├── training_pipeline/
│   │   ├── dataset_builder.py         # pulls training dataset from feature store
│   │   ├── models/
│   │   │   ├── ridge_model.py
│   │   │   ├── random_forest_model.py
│   │   │   └── tf_model.py
│   │   ├── evaluate.py                # RMSE, MAE, R²
│   │   ├── explain.py                 # SHAP explainability artifacts
│   │   ├── model_registry.py
│   │   └── run_training_pipeline.py   # entrypoint invoked by GH Actions
│   ├── inference/
│   │   ├── predictor.py               # loads model + latest features, produces 3-day forecast
│   │   ├── alerts.py                  # hazardous AQI threshold checks
│   │   └── api.py                     # Flask app exposing /predict, /health, /explain
│   └── dashboard/
│       ├── app.py                     # Streamlit entrypoint
│       ├── components/
│       │   ├── forecast_view.py
│       │   ├── eda_view.py
│       │   ├── shap_view.py
│       │   └── alert_banner.py
│       └── utils.py
├── notebooks/
│   └── eda.ipynb                      # exploratory analysis, ad hoc experimentation
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_data_ingestion.py
│   ├── test_training_pipeline.py
│   └── test_inference.py
├── requirements.txt
├── .env.example
├── README.md
└── pyproject.toml / setup.cfg
```

### 2.4 Architecture Notes for the Coding Agent
- Treat `AQICN_API_KEY`, `OPENWEATHER_API_KEY`, and `HOPSWORKS_API_KEY` as required environment variables / GitHub Actions secrets — **never hardcode**.
- All pipeline entrypoints (`run_feature_pipeline.py`, `run_training_pipeline.py`, `run_backfill.py`) must be runnable standalone via `python -m ...` so GitHub Actions can invoke them directly, and so they're independently testable.
- Feature Store and Model Registry access should be wrapped behind thin interface modules (`feature_store_writer.py`, `model_registry.py`) so the underlying provider (Hopsworks vs. Vertex AI) can be swapped without touching pipeline logic.
- The Flask API and Streamlit dashboard are logically separate; the dashboard should call the API (not re-implement inference logic) so the prediction logic has one source of truth.

---

## 3. Core Features & Functional Requirements

### 3.1 Feature Pipeline Development
**User/system action:** On an hourly schedule, the system fetches current weather + pollutant readings and produces model-ready features.

- Fetch raw pollutant data (PM2.5, PM10, NO2, SO2, CO, O3, current AQI) from AQICN for each configured city.
- Fetch raw weather data (temperature, humidity, wind speed, pressure, precipitation) from OpenWeather for the same city/coordinates.
- Compute time-based features: hour of day, day of week, month, is_weekend.
- Compute derived features: AQI change rate (delta vs. previous reading), rolling averages (e.g. 3h, 24h mean AQI/PM2.5), lag features (AQI at t-1, t-24).
- Write the resulting feature row(s) to the Feature Store, keyed by city + timestamp.

**Acceptance criteria:**
- [ ] Given valid API keys, running the feature pipeline for a configured city produces exactly one new row per city in the feature store for the current hour, with no null values in required fields.
- [ ] All raw API failures (timeout, rate limit, malformed response) are caught, logged, and do not crash the whole pipeline run; a failed city is skipped and logged, others continue.
- [ ] Derived features (AQI change rate, rolling averages, lags) are unit-tested against a fixture of known input sequences and produce mathematically correct output.
- [ ] Pipeline is idempotent: re-running it for the same city/hour does not create duplicate feature-store rows (upsert or dedup logic in place).

### 3.2 Historical Data Backfill
**User/system action:** On demand, generate a historical feature dataset spanning past N days for initial model training.

- Accept a configurable date range (start date, end date) and city list.
- Reuse the same feature-engineering logic as the live pipeline (no duplicated logic) applied to historical raw data pulled from the APIs (or a bulk historical export if the API supports it).
- Write backfilled rows into the same feature store feature group as the live pipeline.

**Acceptance criteria:**
- [ ] Running backfill for a specified date range produces one row per city per hour (or per available granularity) covering that full range, with gaps logged explicitly (not silently dropped).
- [ ] Backfilled data is schema-identical to live-pipeline output (same columns, types) — no downstream training code needs to special-case backfilled vs. live rows.
- [ ] Backfill is resumable: if interrupted, re-running does not duplicate already-written rows.

### 3.3 Training Pipeline Implementation
**User/system action:** On a daily schedule, retrain candidate models on the latest available feature history and promote the best one.

- Pull historical features + target (AQI, for prediction horizons of +24h/+48h/+72h) from the Feature Store as a training dataset.
- Split into train/validation/test sets using **time-based splitting** (no random shuffling — this is a time series).
- Train and compare at least three model types: Random Forest, Ridge Regression, and one TensorFlow model (dense NN minimum; LSTM acceptable stretch goal).
- Evaluate each model using RMSE, MAE, and R² on the held-out test set, per forecast horizon (24h/48h/72h).
- Generate SHAP feature-importance artifacts for the best-performing model.
- Register the best-performing model (by RMSE, tie-break by R²) to the Model Registry, versioned, with its metrics attached as metadata.

**Acceptance criteria:**
- [ ] Training pipeline run produces a metrics report (RMSE, MAE, R²) for every model type, per forecast horizon, saved as a build artifact (e.g. JSON/CSV) and logged.
- [ ] The model selected for registry promotion is chosen programmatically (not manually) based on a documented, deterministic metric comparison rule.
- [ ] A new model version is only registered if it is created — a failed training run must not silently register a broken/empty model.
- [ ] SHAP values are computed and stored (e.g. as a summary plot image + raw values) for the promoted model.
- [ ] Model registry entry includes: model type, training date, feature schema/version used, and evaluation metrics.

### 3.4 Automated CI/CD Pipeline
**User/system action:** No manual action — pipelines run themselves on schedule.

- Feature pipeline triggered hourly via GitHub Actions scheduled workflow (`cron`).
- Training pipeline triggered daily via GitHub Actions scheduled workflow.
- Backfill pipeline available as a manually-dispatched workflow (`workflow_dispatch`), not on a schedule.
- All workflows must fail loudly (non-zero exit + visible GitHub Actions failure status) on unrecoverable errors, and must not silently swallow exceptions.

**Acceptance criteria:**
- [ ] `.github/workflows/feature_pipeline.yml` runs on a valid hourly cron expression and calls the feature pipeline entrypoint.
- [ ] `.github/workflows/training_pipeline.yml` runs on a valid daily cron expression and calls the training pipeline entrypoint.
- [ ] `.github/workflows/backfill.yml` is dispatch-only (no schedule) and accepts start/end date + city as workflow inputs.
- [ ] Secrets (API keys) are referenced via GitHub Actions `secrets.*` context, never committed to the repo.
- [ ] A deliberately broken pipeline run (e.g. bad API key) causes the workflow run to show as failed in GitHub Actions.

### 3.5 Web Application Dashboard
**User/system action:** A visitor opens the dashboard and sees the current 3-day AQI forecast for a selected city.

- Backend (Flask) loads the latest promoted model from the Model Registry and the most recent features from the Feature Store, and computes a prediction for the next 3 days (24h/48h/72h horizons) on request.
- Frontend (Streamlit) lets the user select a city (from the configured list), displays the 3-day forecast (numeric AQI + category label — Good/Moderate/Unhealthy/etc.), and shows a trend chart of recent historical AQI leading into the forecast.
- Dashboard calls the Flask API rather than duplicating inference logic.

**Acceptance criteria:**
- [ ] Selecting a city in the Streamlit UI displays a forecast for +24h, +48h, +72h, each with a numeric AQI value and its standard EPA/AQICN category label + color.
- [ ] If the API/model/feature data is unavailable, the dashboard shows a clear error state instead of crashing or displaying stale data unlabeled.
- [ ] A historical trend chart (at least the last 7 days) renders alongside the forecast for context.
- [ ] `/predict`, `/health` endpoints exist on the Flask API and are documented (request/response shape) in the README.

### 3.6 Advanced Analytics Features
**User/system action:** A visitor explores *why* the forecast looks the way it does, and receives a warning if hazardous levels are predicted.

- EDA view: summary charts of historical AQI/pollutant trends (e.g. by hour-of-day, day-of-week, seasonal pattern) — either precomputed and stored, or computed on demand in the dashboard.
- SHAP explainability view: show top contributing features for the current prediction (bar chart or waterfall from SHAP values).
- Hazardous AQI alert: if any of the 3-day forecast values cross a configurable "unhealthy" threshold (e.g. AQI > 150), display a prominent alert banner in the dashboard.
- Model comparison: dashboard or a dedicated view shows metrics (RMSE/MAE/R²) for all trained model types, not just the promoted one, so the user can see why the current model was picked.

**Acceptance criteria:**
- [ ] EDA view renders at least 3 distinct charts (e.g. AQI by hour-of-day, AQI trend over time, pollutant correlation) without requiring the user to run any code.
- [ ] SHAP view displays the top N (e.g. 5–10) contributing features for the currently displayed prediction, matching the model actually used for that prediction (not a stale/mismatched explanation).
- [ ] Alert threshold is defined in one configurable location (not hardcoded in multiple places) and triggers a visibly distinct UI element when crossed.
- [ ] Model comparison metrics displayed match the latest training pipeline run's output exactly (single source of truth — the metrics artifact from §3.3).

---

## 4. Data Models & Database Schema

> Note: "Database schema" here maps to **Hopsworks Feature Group schemas** (the closest equivalent to tables in this serverless feature-store architecture), plus the Model Registry's metadata schema.

### 4.1 Feature Group: `aqi_features`
Grain: one row per `(city_id, timestamp)`.

| Field | Type | Description |
|---|---|---|
| `city_id` | string (PK part) | Identifier for the city, e.g. `"karachi"` |
| `timestamp` | timestamp (PK part, event time) | UTC hour of the reading |
| `latitude` | float | City coordinate (for weather API lookup) |
| `longitude` | float | City coordinate |
| `aqi` | float | Current AQI reading (target basis) |
| `pm2_5` | float | Raw pollutant reading |
| `pm10` | float | Raw pollutant reading |
| `no2` | float | Raw pollutant reading |
| `so2` | float | Raw pollutant reading |
| `co` | float | Raw pollutant reading |
| `o3` | float | Raw pollutant reading |
| `temperature` | float | From OpenWeather |
| `humidity` | float | From OpenWeather |
| `wind_speed` | float | From OpenWeather |
| `pressure` | float | From OpenWeather |
| `precipitation` | float | From OpenWeather |
| `hour` | int | 0–23, derived |
| `day_of_week` | int | 0–6, derived |
| `month` | int | 1–12, derived |
| `is_weekend` | bool | derived |
| `aqi_change_rate` | float | (aqi_t − aqi_t-1) / aqi_t-1, derived |
| `aqi_rolling_3h` | float | 3-hour rolling mean AQI, derived |
| `aqi_rolling_24h` | float | 24-hour rolling mean AQI, derived |
| `aqi_lag_1h` | float | AQI value 1 hour prior, derived |
| `aqi_lag_24h` | float | AQI value 24 hours prior, derived |
| `data_source_version` | string | Tag for schema/pipeline version (supports safe evolution) |

### 4.2 Training Dataset View: `aqi_training_dataset`
Derived from `aqi_features`, with forward-looking targets joined in:

| Field | Type | Description |
|---|---|---|
| *(all fields from `aqi_features`)* | — | Feature columns as of time `t` |
| `target_aqi_24h` | float | Actual AQI at `t + 24h` (label) |
| `target_aqi_48h` | float | Actual AQI at `t + 48h` (label) |
| `target_aqi_72h` | float | Actual AQI at `t + 72h` (label) |

### 4.3 Model Registry Metadata: `aqi_model_registry`

| Field | Type | Description |
|---|---|---|
| `model_id` | string | Unique model version identifier |
| `model_type` | enum | `ridge`, `random_forest`, `tensorflow_nn` (extendable) |
| `forecast_horizon` | enum | `24h`, `48h`, `72h` |
| `training_date` | timestamp | When this version was trained |
| `feature_schema_version` | string | Matches `data_source_version` used for training |
| `rmse` | float | Test-set metric |
| `mae` | float | Test-set metric |
| `r2` | float | Test-set metric |
| `is_promoted` | bool | Whether this is the currently-served version |
| `shap_summary_ref` | string | Pointer/path to stored SHAP artifact |

### 4.4 Relationships
- `aqi_features` is the single source of truth for both live inference input and (via `aqi_training_dataset`) historical training data — no separate/duplicate storage.
- `aqi_model_registry` entries reference `feature_schema_version` to guarantee a model is only ever served against a feature schema it was trained on (prevents silent train/serve skew).
- One `aqi_model_registry` row exists per `(model_type, forecast_horizon, training_date)` combination; exactly one row per `forecast_horizon` has `is_promoted = true` at any time.

---

## 5. Step-by-Step Implementation Plan

> Each phase should be completed and verified (acceptance criteria met, tests passing) before the next phase begins. This order minimizes rework: nothing downstream is built against a moving target upstream.

### Phase 1 — Project Setup & Foundations
1. Initialize Git repository with the folder structure in §2.3.
2. Set up `requirements.txt` / dependency management (Python 3.11+, pinned versions for scikit-learn, tensorflow, hopsworks, streamlit, flask, shap, requests).
3. Create `.env.example` documenting required secrets: `AQICN_API_KEY`, `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`.
4. Implement `src/config/settings.py` — city list (id, name, lat/lon), AQI alert threshold, all config centralized here (no magic numbers elsewhere).
5. Set up logging config used consistently across all pipeline modules.
6. Write README skeleton documenting setup, environment variables, and how to run each pipeline locally.

**Exit criteria:** repo installs cleanly (`pip install -r requirements.txt`), config loads without error, no pipeline logic yet.

### Phase 2 — Data Ingestion Layer
1. Implement `aqicn_client.py`: authenticated client to fetch current AQI + pollutants for a city.
2. Implement `openweather_client.py`: authenticated client to fetch current weather for a city.
3. Add retry/backoff and error handling for both clients (network failures, rate limits, malformed JSON).
4. Write unit tests using mocked API responses (fixtures) — no live API calls in tests.

**Exit criteria:** both clients return normalized Python objects/dicts for a given city; tests pass without network access.

### Phase 3 — Feature Engineering & Feature Store Integration
1. Implement `feature_engineering.py`: time features, AQI change rate, rolling averages, lags (per §4.1 schema).
2. Implement `feature_store_writer.py`: Hopsworks connection, feature group creation (if not exists) matching the `aqi_features` schema, upsert-safe write logic.
3. Implement `run_feature_pipeline.py` as the orchestrating entrypoint: fetch → engineer → write, for each configured city.
4. Unit test feature engineering functions against known input/output fixtures.

**Exit criteria:** running the feature pipeline locally (with valid keys) writes correctly-shaped, deduplicated rows into the Hopsworks feature group.

### Phase 4 — Historical Backfill
1. Implement `run_backfill.py`, reusing `feature_engineering.py` logic against historical raw data (via API history endpoints where available, or a documented alternative if the API doesn't support deep history).
2. Add resumability (checkpointing or upsert-based idempotency) so partial failures don't require a full restart.
3. Backfill at least 60–90 days of history for each configured city to give the training pipeline a workable dataset.

**Exit criteria:** feature store contains a continuous (gaps logged, not silent) historical time series per city, schema-identical to live pipeline output.

### Phase 5 — Training Pipeline
1. Implement `dataset_builder.py`: pulls `aqi_training_dataset` view (features + forward targets) from the feature store, performs time-based train/val/test split.
2. Implement each model module (`ridge_model.py`, `random_forest_model.py`, `tf_model.py`) with a consistent `train(X, y) -> model` / `predict(model, X) -> y_pred` interface.
3. Implement `evaluate.py`: RMSE/MAE/R² computation per model per horizon, output as a structured artifact (JSON).
4. Implement `explain.py`: SHAP value computation and summary artifact generation for the winning model.
5. Implement `model_registry.py`: registration logic, promotion rule (best RMSE per horizon), metadata write per §4.3 schema.
6. Implement `run_training_pipeline.py` as the orchestrating entrypoint.

**Exit criteria:** local run of the training pipeline (against backfilled data) produces a metrics report for all model types/horizons, registers a promoted model per horizon, and generates SHAP artifacts.

### Phase 6 — Automated CI/CD
1. Write `.github/workflows/feature_pipeline.yml` (hourly cron), `.github/workflows/training_pipeline.yml` (daily cron), `.github/workflows/backfill.yml` (manual dispatch).
2. Add all required secrets to the GitHub repository settings (documented in README, not committed).
3. Validate each workflow with a manual trigger before relying on the schedule.

**Exit criteria:** all three workflows run successfully via manual dispatch in GitHub Actions; scheduled ones are confirmed to trigger correctly at least once on schedule.

### Phase 7 — Inference API
1. Implement `predictor.py`: loads promoted model(s) + latest feature row per city from Hopsworks, produces 24h/48h/72h AQI predictions.
2. Implement `alerts.py`: threshold check against `settings.py` config, returns alert flag/level.
3. Implement `api.py` (Flask): `/predict?city=...`, `/health`, `/explain?city=...` endpoints, with clear JSON response schemas documented in the README.

**Exit criteria:** hitting `/predict` for a configured city returns a valid 3-horizon forecast + alert flag in under a defined latency budget (e.g. <5s), using the actual promoted model from the registry.

### Phase 8 — Dashboard (Streamlit)
1. Implement `dashboard/app.py`: city selector, calls Flask API, renders forecast + category labels/colors.
2. Implement `forecast_view.py` and a historical trend chart component.
3. Implement `eda_view.py`: precomputed or on-demand EDA charts.
4. Implement `shap_view.py`: renders SHAP explanation for the current prediction, sourced from `/explain`.
5. Implement `alert_banner.py`: prominent UI element triggered by the alert flag from the API.
6. Implement model comparison view sourcing directly from the latest training pipeline's metrics artifact.

**Exit criteria:** all acceptance criteria in §3.5 and §3.6 pass via manual QA; dashboard runs both locally (`streamlit run`) and deploys to Streamlit Community Cloud.

### Phase 9 — Testing, Hardening & Documentation
1. Expand test coverage across `tests/` for feature engineering, ingestion, training, and inference modules.
2. Add end-to-end smoke test: run feature pipeline → training pipeline → inference on a small synthetic/sandboxed dataset.
3. Finalize README: architecture diagram, setup instructions, environment variables, how each pipeline runs and how to trigger manually, known limitations.
4. Review error handling across all entrypoints — confirm no silent failures anywhere in the pipeline chain.

**Exit criteria:** project can be cloned fresh, configured via `.env`, and brought fully online (backfill → train → serve → dashboard) by following only the README.

### 5.1 Execution Order Summary
```
Phase 1 (Setup)
   → Phase 2 (Ingestion)
      → Phase 3 (Feature Pipeline)
         → Phase 4 (Backfill)
            → Phase 5 (Training Pipeline)
               → Phase 6 (CI/CD Automation)
                  → Phase 7 (Inference API)
                     → Phase 8 (Dashboard)
                        → Phase 9 (Testing/Docs/Hardening)
```
This order is deliberate: ingestion must exist before features can be engineered; features must be flowing (live + backfilled) before there's anything to train on; training must produce a registered model before there's anything to serve; the API must exist before the dashboard can call it; and hardening/documentation is a final pass across a working system rather than a parallel, moving-target effort.

---

## 6. Open Items for the Coding Agent to Flag (Not Decisions to Make Silently)
- Confirm final choice: Hopsworks vs. Vertex AI feature store (this spec defaults to Hopsworks for serverless/free-tier fit — flag if this needs to change).
- Confirm final choice: Flask vs. FastAPI for the inference API (this spec defaults to Flask per original notes).
- Confirm AQI category thresholds/labels to use (standard EPA breakpoints assumed unless told otherwise).
- Confirm initial list of cities to configure (not specified in source notes).
- Confirm hazardous-AQI alert threshold value (default suggestion: AQI > 150, "Unhealthy," per EPA convention — confirm before hardcoding).
