import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.config import settings
from src.inference.alerts import get_aqi_category

logger = logging.getLogger(__name__)

# Fallback realistic benchmarks when Hopsworks credentials or offline
DEFAULT_FALLBACK_DATA = {
    "islamabad": {
        "name": "Islamabad",
        "country": "Pakistan",
        "aqi": 61,
        "status": "Moderate",
        "substatus": "Fair",
        "alertText": "<strong>MODERATE</strong> | Sensitive individuals should limit prolonged outdoor exertion.",
        "pollutants": { "pm25": 61, "pm10": 28, "no2": 1.2, "o3": 3.5 },
        "forecasts": {
            "24h": "AQI 58 - Good",
            "48h": "AQI 66 - Moderate",
            "72h": "AQI 74 - Moderate"
        },
        "hourly": [48, 50, 52, 55, 63, 71, 74, 72, 66, 61, 56, 52, 50, 49, 53, 58, 62, 66, 68, 65, 60, 56, 52, 49],
        "extended": [52, 55, 58, 60, 59, 63, 66, 68, 70, 72, 74, 73, 70, 67, 64]
    },
    "karachi": {
        "name": "Karachi",
        "country": "Pakistan",
        "aqi": 142,
        "status": "Unhealthy for Sensitive",
        "substatus": "Poor",
        "alertText": "<strong>HIGH SMOG WARNING</strong> | Reduce prolonged or heavy outdoor exertion.",
        "pollutants": { "pm25": 58, "pm10": 92, "no2": 38, "o3": 42 },
        "forecasts": {
            "24h": "AQI 135 - Sensitive",
            "48h": "AQI 145 - Sensitive",
            "72h": "AQI 152 - Unhealthy"
        },
        "hourly": [115, 120, 125, 132, 140, 148, 155, 158, 152, 145, 138, 132, 130, 134, 140, 146, 150, 154, 156, 148, 142, 135, 128, 122],
        "extended": [125, 130, 135, 138, 142, 145, 148, 150, 152, 150, 146, 142, 138, 135, 130]
    },
    "lahore": {
        "name": "Lahore",
        "country": "Pakistan",
        "aqi": 168,
        "status": "Unhealthy",
        "substatus": "Poor",
        "alertText": "<strong>SMOG WARNING (LAHORE)</strong> | Active unhealthy particulate smog; outdoor exertion strictly restricted.",
        "pollutants": { "pm25": 88, "pm10": 142, "no2": 46, "o3": 38 },
        "forecasts": {
            "24h": "AQI 165 - Unhealthy",
            "48h": "AQI 174 - Unhealthy",
            "72h": "AQI 182 - Unhealthy"
        },
        "hourly": [140, 144, 150, 158, 168, 178, 186, 182, 172, 165, 158, 152, 148, 150, 156, 164, 172, 180, 184, 176, 168, 160, 152, 145],
        "extended": [155, 160, 165, 168, 172, 175, 178, 180, 182, 185, 180, 175, 170, 165, 160]
    }
}


def get_dashboard_payload():
    """
    Assembles server-side dashboard telemetry directly from Hopsworks Feature Store
    and Predictor, with automatic graceful fallback to calibrated benchmarks.
    
    This ensures that when deployed to Streamlit Community Cloud, the app displays
    real feature store data without requiring a localhost Flask API process.
    """
    cities_payload = json.loads(json.dumps(DEFAULT_FALLBACK_DATA))
    is_live = False
    source = "OFFLINE BENCHMARK"

    # 1. Attempt connection to Hopsworks Feature Store
    full_df = None
    if settings.HOPSWORKS_API_KEY and settings.HOPSWORKS_PROJECT_NAME:
        try:
            import hopsworks
            project = hopsworks.login(
                project=settings.HOPSWORKS_PROJECT_NAME,
                api_key_value=settings.HOPSWORKS_API_KEY
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
            full_df = fg.select_all().read()
            if full_df is not None and not full_df.empty:
                is_live = True
                source = "LIVE FEATURE STORE (HOPSWORKS)"
        except Exception as e:
            logger.info(f"Hopsworks connection skipped or unavailable: {e}")
            full_df = None

    # 2. Attempt loading Predictor models
    predictor = None
    if is_live:
        try:
            from src.inference.predictor import Predictor
            predictor = Predictor()
        except Exception as e:
            logger.info(f"Predictor initialization skipped: {e}")
            predictor = None

    # 3. Populate live data per city if available
    if is_live and full_df is not None and not full_df.empty:
        for c in settings.CITIES:
            cid = c["id"]
            if cid not in cities_payload:
                continue

            city_df = full_df[full_df["city_id"] == cid].sort_values("timestamp")
            if not city_df.empty:
                latest = city_df.iloc[-1]
                
                # Real current AQI
                aqi_val = int(round(float(latest.get("aqi", cities_payload[cid]["aqi"]))))
                cities_payload[cid]["aqi"] = aqi_val
                
                cat = get_aqi_category(aqi_val)
                cities_payload[cid]["status"] = cat
                
                if aqi_val <= 50:
                    cities_payload[cid]["substatus"] = "Good"
                elif aqi_val <= 100:
                    cities_payload[cid]["substatus"] = "Fair"
                else:
                    cities_payload[cid]["substatus"] = "Poor"

                # Real Pollutants
                cities_payload[cid]["pollutants"] = {
                    "pm25": round(float(latest.get("pm2_5", cities_payload[cid]["pollutants"]["pm25"])), 1),
                    "pm10": round(float(latest.get("pm10", cities_payload[cid]["pollutants"]["pm10"])), 1),
                    "no2": round(float(latest.get("no2", cities_payload[cid]["pollutants"]["no2"])), 1),
                    "o3": round(float(latest.get("o3", cities_payload[cid]["pollutants"]["o3"])), 1),
                }

                # Real 24h Hourly Trend from Feature Store
                recent_24 = city_df.tail(24)
                if len(recent_24) >= 8:
                    cities_payload[cid]["hourly"] = [
                        int(round(float(val))) for val in recent_24["aqi"].tolist()
                    ]

                # Real Predictions from Predictor
                if predictor:
                    try:
                        preds = predictor.predict(cid)
                        if preds:
                            for h in settings.FORECAST_HORIZONS:
                                if h in preds:
                                    val = int(round(preds[h]))
                                    cities_payload[cid]["forecasts"][h] = f"AQI {val} - {get_aqi_category(val)}"
                    except Exception as e:
                        logger.warning(f"Failed to generate live prediction for {cid}: {e}")

                # Real SHAP from Predictor
                if predictor:
                    try:
                        exps = predictor.explain(cid)
                        if exps and "24h" in exps:
                            cities_payload[cid]["shap"] = exps["24h"]
                    except Exception as e:
                        pass

    # 4. Check for training metrics artifact
    training_metrics = {}
    metrics_path = os.path.join("artifacts", "training_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                training_metrics = json.load(f)
        except Exception:
            pass

    return {
        "cities": cities_payload,
        "is_live": is_live,
        "source": source,
        "metrics": training_metrics,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    }
