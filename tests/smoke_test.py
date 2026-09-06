"""
End-to-end smoke test using synthetic data.
Tests the full pipeline: feature engineering → model training → evaluation.
Does NOT require Hopsworks or API keys — runs entirely in-memory.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def test_smoke_full_pipeline():
    """
    Smoke test: generate synthetic data, compute features, train all 3 models,
    evaluate metrics, and verify predictions are produced.
    """
    from src.feature_pipeline.feature_engineering import compute_features
    from src.training_pipeline.models import ridge_model, random_forest_model, tf_model
    from src.training_pipeline.evaluate import evaluate_model
    
    # 1. Generate synthetic data (mimicking backfill output)
    np.random.seed(42)
    n_hours = 200
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    data = {
        "city_id": ["test_city"] * n_hours,
        "timestamp": [base_time + timedelta(hours=i) for i in range(n_hours)],
        "aqi": 80 + np.sin(np.arange(n_hours) * np.pi / 12) * 20 + np.random.normal(0, 5, n_hours),
        "pm2_5": np.random.uniform(20, 80, n_hours),
        "pm10": np.random.uniform(30, 100, n_hours),
        "no2": np.random.uniform(5, 40, n_hours),
        "so2": np.random.uniform(1, 10, n_hours),
        "co": np.random.uniform(0.1, 1.5, n_hours),
        "o3": np.random.uniform(10, 60, n_hours),
        "temperature": 25 + np.random.normal(0, 3, n_hours),
        "humidity": 60 + np.random.normal(0, 5, n_hours),
        "wind_speed": np.abs(np.random.normal(5, 2, n_hours)),
        "pressure": 1012 + np.random.normal(0, 2, n_hours),
        "precipitation": np.where(np.random.random(n_hours) > 0.9, np.random.uniform(0.1, 5, n_hours), 0),
        "latitude": [33.6844] * n_hours,
        "longitude": [73.0479] * n_hours,
    }
    df = pd.DataFrame(data)
    
    # 2. Compute features
    df_featured = compute_features(df)
    
    # Verify feature columns exist
    assert "hour" in df_featured.columns
    assert "day_of_week" in df_featured.columns
    assert "is_weekend" in df_featured.columns
    assert "aqi_lag_1h" in df_featured.columns
    assert "aqi_lag_24h" in df_featured.columns
    assert "aqi_change_rate" in df_featured.columns
    assert "aqi_rolling_3h" in df_featured.columns
    assert "aqi_rolling_24h" in df_featured.columns
    assert "data_source_version" in df_featured.columns
    
    # 3. Create targets
    df_featured["target_aqi_24h"] = df_featured["aqi"].shift(-24)
    df_featured = df_featured.dropna(subset=["target_aqi_24h", "aqi_lag_24h"])
    
    assert len(df_featured) > 50, f"Not enough rows after dropna: {len(df_featured)}"
    
    # 4. Split
    exclude = ["city_id", "timestamp", "target_aqi_24h", "data_source_version"]
    feature_cols = [c for c in df_featured.columns if c not in exclude]
    
    split = int(len(df_featured) * 0.7)
    X_train = df_featured[feature_cols].values[:split]
    y_train = df_featured["target_aqi_24h"].values[:split]
    X_test = df_featured[feature_cols].values[split:]
    y_test = df_featured["target_aqi_24h"].values[split:]
    
    # 5. Train and evaluate all 3 models
    models = {
        "ridge": ridge_model,
        "random_forest": random_forest_model,
        "tensorflow_nn": tf_model,
    }
    
    for name, module in models.items():
        if name == "tensorflow_nn":
            model = module.train(X_train, y_train)  # No val set for smoke test
        else:
            model = module.train(X_train, y_train)
        
        y_pred = module.predict(model, X_test)
        metrics = evaluate_model(y_test, y_pred)
        
        # Basic sanity checks
        assert y_pred.shape == y_test.shape, f"{name}: prediction shape mismatch"
        assert not np.any(np.isnan(y_pred)), f"{name}: NaN predictions"
        assert metrics["rmse"] >= 0, f"{name}: negative RMSE"
        assert metrics["mae"] >= 0, f"{name}: negative MAE"
        # R² can be negative for very bad models, but should be finite
        assert np.isfinite(metrics["r2"]), f"{name}: non-finite R²"
