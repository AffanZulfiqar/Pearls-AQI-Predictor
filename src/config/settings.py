import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API Keys
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Flask API URL — configurable for local vs production deployment
# Local:      http://127.0.0.1:5000
# Production: https://<your-deployed-flask-service>
FLASK_API_URL = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000").rstrip("/")

# App Config
HAZARDOUS_AQI_THRESHOLD = 150

# Supported Cities
CITIES = [
    {
        "id": "islamabad",
        "name": "Islamabad",
        "lat": 33.6844,
        "lon": 73.0479
    },
    {
        "id": "karachi",
        "name": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011
    },
    {
        "id": "lahore",
        "name": "Lahore",
        "lat": 31.5204,
        "lon": 74.3587
    }
]

# Feature Store Config
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_REGISTRY_NAME = "aqi_model_registry"

# Data versioning — prevents train/serve skew
DATA_SOURCE_VERSION = "v1.0"

# Model Config
FORECAST_HORIZONS = ["24h", "48h", "72h"]
MODEL_TYPES = ["ridge", "random_forest", "tensorflow_nn"]


def validate_env():
    """
    Validates that all required environment variables are set.
    Logs YES/NO for each — never logs the actual secret values.
    Raises EnvironmentError if any required variable is missing.
    """
    required = {
        "HOPSWORKS_API_KEY": HOPSWORKS_API_KEY,
        "HOPSWORKS_PROJECT_NAME": HOPSWORKS_PROJECT_NAME,
        "AQICN_API_KEY": AQICN_API_KEY,
        "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
    }

    missing = []
    for name, value in required.items():
        status = "YES" if value else "NO"
        logger.info("%s configured: %s", name, status)
        if not value:
            missing.append(name)

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or GitHub Actions secrets."
        )

    logger.info("Hopsworks project configured: YES (project=%s)", HOPSWORKS_PROJECT_NAME)
    logger.info("Flask API URL: %s", FLASK_API_URL)
