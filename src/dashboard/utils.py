"""
Dashboard server-side utilities.

In the live architecture, the dashboard does NOT connect to Hopsworks directly.
All data flows through:
    Streamlit → Flask API → Hopsworks → Models

This module only provides:
  - The Flask API URL (from settings/environment)
  - A server health check against the Flask API
  - The payload structure that Streamlit injects into the HTML bundle

The actual data (AQI, forecasts, history, SHAP, metrics) is fetched
client-side in app.js from the Flask API endpoints.
"""
import logging
import requests
from datetime import datetime, timezone
from src.config import settings

logger = logging.getLogger(__name__)


def check_api_health() -> dict:
    """
    Checks whether the Flask API is reachable.
    Returns a status dict — never raises.
    """
    try:
        resp = requests.get(f"{settings.FLASK_API_URL}/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "api_online":             True,
                "hopsworks_connected":    data.get("hopsworks_connected", False),
                "feature_store_available": data.get("feature_store_available", False),
                "models_loaded":          data.get("models_loaded", {}),
                "model_data_sources":     data.get("model_data_sources", {}),
                "latest_data":            data.get("latest_data", {}),
            }
        return {"api_online": False, "error": f"HTTP {resp.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"api_online": False, "error": "Connection refused — Flask API not running"}
    except Exception as e:
        return {"api_online": False, "error": str(e)}


def get_dashboard_payload() -> dict:
    """
    Assembles the minimal server-side payload injected into the HTML bundle.

    The frontend (app.js) receives:
      - flask_api_url  → used for all API calls (never hardcoded in JS)
      - is_live        → whether the Flask API is reachable
      - status         → system health details
      - generated_at   → server timestamp

    All actual data (AQI, forecasts, history, SHAP, metrics) is fetched
    client-side from the Flask API. No Hopsworks connection here.
    """
    status = check_api_health()
    is_live = status.get("api_online", False)

    return {
        "flask_api_url": settings.FLASK_API_URL,
        "is_live":       is_live,
        "status":        status,
        "cities":        [c["id"] for c in settings.CITIES],
        "generated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
