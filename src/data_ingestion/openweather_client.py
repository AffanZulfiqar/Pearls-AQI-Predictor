"""
OpenWeather API client.

Provides:
  - get_city_weather(): current weather (temperature, humidity, wind, pressure, precipitation)
  - get_air_pollution_history(): FREE historical air pollution data (AQI + pollutants)
    available from 2020-11-27 onwards with hourly granularity.
    Endpoint: http://api.openweathermap.org/data/2.5/air_pollution/history
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _create_session() -> requests.Session:
    """Creates a requests session with retry/backoff strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,  # 1s, 2s, 4s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class OpenWeatherClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.air_pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
        self.air_pollution_current_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.session = _create_session()

    def get_city_weather(self, city_name: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Fetches current weather data for a given city via lat/lon.
        Includes automatic retry with exponential backoff on transient failures.
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_response(data)

        except requests.exceptions.RequestException as e:
            logger.error("Network error fetching OpenWeather data for %s: %s", city_name, e)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching OpenWeather data for %s: %s", city_name, e)
            return None

    def get_air_pollution_history(
        self,
        city_name: str,
        lat: float,
        lon: float,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetches historical air pollution data from OpenWeather's FREE Air Pollution History API.
        Available from 2020-11-27 onwards, with hourly granularity.

        Returns a list of dicts, one per hour, each containing:
          aqi, pm2_5, pm10, no2, so2, co, o3, timestamp, temperature (None — from weather API),
          humidity (None), wind_speed (None), pressure (None), precipitation (None)

        Raises on API failure (caller should handle).
        """
        start_unix = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
        end_unix = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

        params = {
            "lat": lat,
            "lon": lon,
            "start": start_unix,
            "end": end_unix,
            "appid": self.api_key,
        }

        logger.info(
            "Fetching OpenWeather air pollution history for %s from %s to %s...",
            city_name, start_dt.date(), end_dt.date()
        )

        response = self.session.get(self.air_pollution_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        rows = []
        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc).replace(tzinfo=None)
            components = item.get("components", {})
            # OpenWeather AQI is 1-5 scale; convert to EPA-style 0-500 for consistency
            ow_aqi = item.get("main", {}).get("aqi", 1)
            aqi_approx = self._ow_aqi_to_epa(ow_aqi, components)

            rows.append({
                "timestamp": dt,
                "aqi": aqi_approx,
                "pm2_5": components.get("pm2_5", 0.0),
                "pm10": components.get("pm10", 0.0),
                "no2": components.get("no2", 0.0),
                "so2": components.get("so2", 0.0),
                "co": components.get("co", 0.0),
                "o3": components.get("o3", 0.0),
                # Weather fields not available in pollution history — will be 0 for backfill
                "temperature": 0.0,
                "humidity": 0.0,
                "wind_speed": 0.0,
                "pressure": 1013.0,
                "precipitation": 0.0,
            })

        logger.info(
            "Fetched %d historical hourly pollution records for %s.", len(rows), city_name
        )
        return rows

    def _ow_aqi_to_epa(self, ow_aqi_index: int, components: dict) -> float:
        """
        Converts OpenWeather's 1-5 AQI index to an approximate EPA AQI value
        using PM2.5 concentration breakpoints (the primary driver of AQI).

        OpenWeather AQI scale:
          1 = Good       (PM2.5 0-10 µg/m³)    → EPA ~0-50
          2 = Fair       (PM2.5 10-25 µg/m³)   → EPA ~51-100
          3 = Moderate   (PM2.5 25-50 µg/m³)   → EPA ~101-150
          4 = Poor       (PM2.5 50-75 µg/m³)   → EPA ~151-200
          5 = Very Poor  (PM2.5 75+ µg/m³)     → EPA ~201-300
        """
        pm25 = components.get("pm2_5", 0.0)
        # Use PM2.5 breakpoints for EPA AQI calculation (NowCast approximation)
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= pm25 <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
                return round(aqi, 1)
        # Fallback to index-based mapping if PM2.5 is unavailable
        fallback_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
        return float(fallback_map.get(ow_aqi_index, 50))

    def _parse_weather_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the response into a flat dictionary.
        """
        main = data.get("main", {})
        wind = data.get("wind", {})

        rain = data.get("rain", {}).get("1h", 0.0)
        snow = data.get("snow", {}).get("1h", 0.0)
        precipitation = rain + snow

        return {
            "temperature": float(main.get("temp", 0)),
            "humidity": float(main.get("humidity", 0)),
            "wind_speed": float(wind.get("speed", 0)),
            "pressure": float(main.get("pressure", 0)),
            "precipitation": float(precipitation)
        }
