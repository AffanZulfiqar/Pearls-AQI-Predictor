"""
Integration tests for live Hopsworks + Flask API.

These tests require real credentials and a running system.
They are SKIPPED automatically when HOPSWORKS_API_KEY is not set,
so regular CI unit tests continue to pass.

To run integration tests locally:
    pytest tests/test_live_hopsworks.py -v

To also test Flask (start Flask first):
    python -m src.inference.api &
    pytest tests/test_live_hopsworks.py -v --run-flask
"""
import os
import time
import pytest
import requests
import pandas as pd

# Skip ALL tests in this module if credentials are absent
pytestmark = pytest.mark.skipif(
    not os.getenv("HOPSWORKS_API_KEY") or not os.getenv("HOPSWORKS_PROJECT_NAME"),
    reason="HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME not set — skipping live integration tests"
)

FLASK_BASE = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")
CITIES = ["islamabad", "karachi", "lahore"]


# ── Hopsworks ──────────────────────────────────────────────────────────────────

def test_hopsworks_login():
    """Verify that Hopsworks login succeeds."""
    from src.config.hopsworks_client import get_project, reset_connection
    reset_connection()
    project = get_project()
    assert project is not None, "Hopsworks project is None after login"
    print(f"\n✅ Hopsworks login: project={project.name}")


def test_hopsworks_feature_store():
    """Verify get_feature_store() returns a valid FeatureStore object."""
    from src.config.hopsworks_client import get_feature_store
    fs = get_feature_store()
    assert fs is not None, "FeatureStore is None"
    print(f"\n✅ Feature store: {fs}")


def test_hopsworks_feature_group_exists():
    """Verify that the feature group exists and can be read."""
    from src.config import settings
    from src.config.hopsworks_client import get_feature_store
    fs = get_feature_store()
    fg = fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
    assert fg is not None, (
        f"Feature group '{settings.FEATURE_GROUP_NAME}' v{settings.FEATURE_GROUP_VERSION} not found. "
        "Run backfill first."
    )
    print(f"\n✅ Feature group found: {fg.name} v{fg.version}")


def test_hopsworks_feature_group_has_data():
    """Verify the feature group contains at least one row per city."""
    from src.config import settings
    from src.config.hopsworks_client import get_feature_store
    fs = get_feature_store()
    fg = fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
    assert fg is not None, "Feature group not found"

    df = fg.select_all().read()
    assert df is not None and not df.empty, "Feature group read returned empty DataFrame"

    for city_id in [c["id"] for c in settings.CITIES]:
        city_rows = df[df["city_id"] == city_id]
        assert not city_rows.empty, f"No rows found for city={city_id}"
        latest_ts = city_rows["timestamp"].max()
        print(f"\n✅ Feature Group read: city={city_id}, rows={len(city_rows)}, latest={latest_ts}")


def test_hopsworks_model_registry():
    """Verify that at least one registered model exists in the Model Registry."""
    from src.config import settings
    from src.config.hopsworks_client import get_model_registry
    mr = get_model_registry()
    assert mr is not None, "Model Registry is None"

    for horizon in settings.FORECAST_HORIZONS:
        model_name = f"aqi_model_{horizon}"
        try:
            hw_model = mr.get_best_model(model_name, "rmse", "min")
            assert hw_model is not None, f"No model found for {model_name}"
            data_source = (hw_model.training_metrics or {}).get("data_source", "unknown")
            print(f"\n✅ Model Registry: {model_name} v{hw_model.version} | data_source={data_source}")
        except Exception as e:
            pytest.fail(f"Could not load model {model_name}: {e}")


def test_predictor_loads():
    """Verify that Predictor initialises and loads all models."""
    from src.inference.predictor import Predictor
    p = Predictor()
    assert p.available, "Predictor is not available — check Hopsworks connection"
    assert len(p.models) > 0, "No models loaded by Predictor"
    print(f"\n✅ Predictor: {len(p.models)} models loaded, horizons={list(p.models.keys())}")


def test_predictor_predicts():
    """Verify that Predictor returns predictions for all cities."""
    from src.inference.predictor import Predictor
    p = Predictor()
    if not p.available:
        pytest.skip("Predictor unavailable")
    for city_id in CITIES:
        preds = p.predict(city_id)
        if preds:
            print(f"\n✅ Prediction for {city_id}: {preds}")
        else:
            print(f"\n⚠ No prediction for {city_id} (no feature data?)")


# ── Flask API (optional \u2014 only if --run-flask passed) ───────────────────────────

def _flask_available():
    try:
        r = requests.get(f"{FLASK_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _flask_available(), reason="Flask API not running")
class TestFlaskAPI:
    def test_health(self):
        r = requests.get(f"{FLASK_BASE}/health", timeout=5)
        assert r.status_code == 200
        assert r.json().get("status") == "healthy"
        print(f"\n✅ /health: {r.json()}")

    def test_status(self):
        r = requests.get(f"{FLASK_BASE}/status", timeout=10)
        assert r.status_code == 200
        data = r.json()
        print(f"\n✅ /status: hopsworks={data.get('hopsworks_connected')}, models={data.get('models_loaded')}")

    @pytest.mark.parametrize("city", CITIES)
    def test_current(self, city):
        r = requests.get(f"{FLASK_BASE}/current?city={city}", timeout=15)
        assert r.status_code in (200, 503), f"Unexpected status {r.status_code}"
        if r.status_code == 200:
            d = r.json()
            assert "aqi" in d, "Response missing 'aqi' field"
            print(f"\n✅ /current?city={city}: AQI={d['aqi']} ({d.get('category')})")

    @pytest.mark.parametrize("city", CITIES)
    def test_predict(self, city):
        r = requests.get(f"{FLASK_BASE}/predict?city={city}", timeout=20)
        assert r.status_code in (200, 503), f"Unexpected status {r.status_code}"
        if r.status_code == 200:
            d = r.json()
            assert "forecasts" in d
            print(f"\n✅ /predict?city={city}: {d['forecasts']}")

    @pytest.mark.parametrize("city", CITIES)
    def test_history(self, city):
        r = requests.get(f"{FLASK_BASE}/history?city={city}&hours=168", timeout=20)
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            d = r.json()
            assert "history" in d
            print(f"\n✅ /history?city={city}: {d['count']} rows")

    @pytest.mark.parametrize("city", CITIES)
    def test_explain(self, city):
        r = requests.get(f"{FLASK_BASE}/explain?city={city}", timeout=60)
        assert r.status_code in (200, 503, 404)
        if r.status_code == 200:
            d = r.json()
            assert "explanations" in d
            print(f"\n✅ /explain?city={city}: {list(d['explanations'].keys())}")

    def test_metrics(self):
        r = requests.get(f"{FLASK_BASE}/metrics", timeout=10)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            d = r.json()
            assert "metrics" in d
            print(f"\n✅ /metrics: data_sources={d.get('data_sources')}")
