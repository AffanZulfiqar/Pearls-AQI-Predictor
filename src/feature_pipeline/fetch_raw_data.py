"""
Fetches raw AQI and weather data for a given city.
Separated from the pipeline runner per spec §2.3 for single-responsibility.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from src.data_ingestion.aqicn_client import AQICNClient
from src.data_ingestion.openweather_client import OpenWeatherClient

logger = logging.getLogger(__name__)


def fetch_for_city(
    aqicn_client: AQICNClient,
    weather_client: OpenWeatherClient,
    city: Dict[str, Any],
    timestamp: datetime
) -> Optional[Dict[str, Any]]:
    """
    Fetches and merges raw AQI + weather data for a single city at a given timestamp.
    Returns None if either API call fails (the caller should skip and continue).
    """
    city_id = city["id"]
    city_name = city["name"]
    lat = city["lat"]
    lon = city["lon"]

    aqi_data = aqicn_client.get_city_data(city_name, lat, lon)
    weather_data = weather_client.get_city_weather(city_name, lat, lon)

    if not aqi_data or not weather_data:
        logger.error(f"Failed to fetch complete data for {city_id}, skipping.")
        return None

    return {
        "city_id": city_id,
        "timestamp": timestamp,
        "latitude": lat,
        "longitude": lon,
        "data_source_type": "real",
        **aqi_data,
        **weather_data
    }
