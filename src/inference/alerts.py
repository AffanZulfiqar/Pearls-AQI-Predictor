from src.config import settings

def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def check_alert(predictions):
    for horizon, aqi in predictions.items():
        if aqi > settings.HAZARDOUS_AQI_THRESHOLD:
            return {
                "alert": True,
                "level": get_aqi_category(aqi),
                "horizon": horizon
            }
    return {"alert": False}
