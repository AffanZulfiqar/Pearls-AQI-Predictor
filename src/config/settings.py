import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

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
