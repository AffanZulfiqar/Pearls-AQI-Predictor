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


class AQICNClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.waqi.info/feed"
        self.session = _create_session()

    def get_city_data(self, city_name: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Fetches current AQI and pollutant data for a given city via lat/lon.
        Uses geo:lat;lon endpoint to be more precise if city name mapping is ambiguous.
        Includes automatic retry with exponential backoff on transient failures.
        """
        url = f"{self.base_url}/geo:{lat};{lon}/?token={self.api_key}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"AQICN API error for {city_name}: {data.get('data')}")
                return None
                
            return self._parse_response(data["data"])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching AQICN data for {city_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching AQICN data for {city_name}: {e}")
            return None

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the response into a flat dictionary.
        """
        iaqi = data.get("iaqi", {})
        
        # Safely extract values, defaulting to None if missing
        def get_val(key):
            return iaqi.get(key, {}).get("v")

        return {
            "aqi": float(data.get("aqi", 0)),
            "pm2_5": get_val("pm25"),
            "pm10": get_val("pm10"),
            "no2": get_val("no2"),
            "so2": get_val("so2"),
            "co": get_val("co"),
            "o3": get_val("o3"),
        }
