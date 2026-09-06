import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Dict, Any, Optional

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
            "units": "metric" # to get Celsius
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching OpenWeather data for {city_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching OpenWeather data for {city_name}: {e}")
            return None

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the response into a flat dictionary.
        """
        main = data.get("main", {})
        wind = data.get("wind", {})
        
        # Precipitation might be in rain or snow
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
