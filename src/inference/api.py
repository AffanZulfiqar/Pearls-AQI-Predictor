"""
Flask REST API for the Pearls AQI Predictor.

Endpoints:
  GET /health      — simple liveness probe
  GET /status      — detailed system status (Hopsworks, models, latest data)
  GET /current     — latest feature row for a city (real Hopsworks data)
  GET /predict     — 24h/48h/72h forecasts for a city
  GET /history     — up to 168h of historical AQI for a city
  GET /explain     — SHAP feature importance for a city
  GET /metrics     — latest training metrics from artifacts/training_metrics.json

When Hopsworks is unavailable, /predict returns HTTP 503 (not fake data).
"""
import json
import logging
import os
import traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config.logging_config import setup_logging
from src.inference.alerts import check_alert, get_aqi_category
from src.inference.predictor import Predictor

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialise predictor once at startup
predictor = Predictor()


# ── /health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": _now()}), 200


# ── /status ────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """Detailed system status — useful for debugging deployment."""
    models_loaded = {h: (h in predictor.models) for h in ["24h", "48h", "72h"]}
    data_sources  = {
        h: predictor.model_metadata.get(h, {}).get("data_source", "unknown")
        for h in ["24h", "48h", "72h"]
    }

    latest_data = {}
    if predictor.available and predictor.fs:
        try:
            from src.config import settings
            fg = predictor.fs.get_feature_group(
                settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION
            )
            if fg is not None:
                df = fg.select_all().read()
                for city_id in ["islamabad", "karachi", "lahore"]:
                    city_df = df[df["city_id"] == city_id] if not df.empty else None
                    if city_df is not None and not city_df.empty:
                        ts = city_df["timestamp"].max()
                        latest_data[city_id] = str(ts)
        except Exception as e:
            latest_data["error"] = str(e)

    return jsonify({
        "status":                 "ok",
        "hopsworks_connected":    predictor.available,
        "feature_store_available": predictor.fs is not None,
        "models_loaded":          models_loaded,
        "model_data_sources":     data_sources,
        "latest_data":            latest_data,
        "timestamp":              _now(),
    }), 200


# ── /current ───────────────────────────────────────────────────────────────────

@app.route("/current", methods=["GET"])
def current():
    """Latest AQI observation for a city from the Hopsworks Feature Store."""
    city = _require_city()
    if isinstance(city, tuple):   # error response
        return city

    if not predictor.available:
        return _unavailable("Hopsworks feature store unavailable")

    row = predictor.get_latest_row(city)
    if row is None:
        return jsonify({"error": f"No data found for city '{city}'"}), 404

    aqi = float(row.get("aqi", 0))
    ts  = row.get("timestamp")
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()

    return jsonify({
        "city":      city,
        "timestamp": str(ts),
        "aqi":       round(aqi, 1),
        "pm25":      _safe_float(row.get("pm2_5")),
        "pm10":      _safe_float(row.get("pm10")),
        "no2":       _safe_float(row.get("no2")),
        "o3":        _safe_float(row.get("o3")),
        "so2":       _safe_float(row.get("so2")),
        "co":        _safe_float(row.get("co")),
        "temperature": _safe_float(row.get("temperature")),
        "humidity":    _safe_float(row.get("humidity")),
        "wind_speed":  _safe_float(row.get("wind_speed")),
        "pressure":    _safe_float(row.get("pressure")),
        "category":  get_aqi_category(aqi),
    }), 200


# ── /predict ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET"])
def predict():
    """24h / 48h / 72h AQI forecasts for a city."""
    city = _require_city()
    if isinstance(city, tuple):
        return city

    if not predictor.available:
        return _unavailable("Hopsworks or models unavailable — cannot generate predictions")

    try:
        predictions = predictor.predict(city)
    except Exception:
        logger.exception("Error in /predict for city=%s", city)
        return jsonify({"error": "Prediction failed", "city": city}), 500

    if not predictions:
        return jsonify({"error": f"No predictions available for city '{city}'"}), 404

    forecasts = {
        h: {"value": round(v, 2), "category": get_aqi_category(v)}
        for h, v in predictions.items()
    }
    alert = check_alert(predictions)

    return jsonify({
        "city":      city,
        "forecasts": forecasts,
        "alert":     alert,
        "timestamp": _now(),
    }), 200


# ── /history ───────────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def history():
    """Historical AQI readings for a city from the Hopsworks Feature Store."""
    city  = _require_city()
    if isinstance(city, tuple):
        return city

    try:
        hours = int(request.args.get("hours", 168))
        hours = max(1, min(hours, 720))   # cap at 30 days
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid 'hours' parameter"}), 400

    if not predictor.available:
        return _unavailable("Hopsworks feature store unavailable")

    data = predictor.get_history(city, hours=hours)
    if data is None:
        return jsonify({"error": f"Could not retrieve history for '{city}'"}), 500

    return jsonify({
        "city":      city,
        "hours":     hours,
        "count":     len(data),
        "history":   data,
        "timestamp": _now(),
    }), 200


# ── /explain ───────────────────────────────────────────────────────────────────

@app.route("/explain", methods=["GET"])
def explain():
    """SHAP feature importance for a city's current prediction."""
    city = _require_city()
    if isinstance(city, tuple):
        return city

    if not predictor.available:
        return _unavailable("Hopsworks or models unavailable")

    try:
        explanations = predictor.explain(city)
    except Exception:
        logger.exception("Error in /explain for city=%s", city)
        return jsonify({"error": "SHAP explanation failed"}), 500

    if not explanations:
        return jsonify({"error": f"No explanations available for '{city}'"}), 404

    return jsonify({
        "city":         city,
        "explanations": explanations,
        "timestamp":    _now(),
    }), 200


# ── /metrics ───────────────────────────────────────────────────────────────────

@app.route("/metrics", methods=["GET"])
def metrics():
    """Latest training metrics from artifacts/training_metrics.json."""
    metrics_path = os.path.join("artifacts", "training_metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({
            "error": "No training metrics found. Run the training pipeline first.",
            "path":  metrics_path,
        }), 404

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)

        # Also read data_source from model metadata if available
        data_source_info = {
            h: predictor.model_metadata.get(h, {}).get("data_source", "unknown")
            for h in ["24h", "48h", "72h"]
        }

        return jsonify({
            "metrics":      data,
            "data_sources": data_source_info,
            "timestamp":    _now(),
        }), 200
    except Exception:
        logger.exception("Error reading training_metrics.json")
        return jsonify({"error": "Failed to read training metrics"}), 500


# ── helpers ────────────────────────────────────────────────────────────────────

def _require_city():
    city = request.args.get("city", "").strip().lower()
    if not city:
        return jsonify({"error": "Missing required query parameter: city"}), 400
    from src.config import settings
    valid = [c["id"] for c in settings.CITIES]
    if city not in valid:
        return jsonify({"error": f"Unknown city '{city}'. Valid cities: {valid}"}), 400
    return city


def _unavailable(reason: str):
    return jsonify({
        "error":  "Prediction service unavailable",
        "reason": reason,
    }), 503


def _safe_float(val, default=0.0):
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return default


def _now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
