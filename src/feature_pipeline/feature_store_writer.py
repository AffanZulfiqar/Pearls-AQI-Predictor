import hopsworks
import pandas as pd
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class FeatureStoreWriter:
    def __init__(self):
        self.project = hopsworks.login(
            project=settings.HOPSWORKS_PROJECT_NAME,
            api_key_value=settings.HOPSWORKS_API_KEY
        )
        self.fs = self.project.get_feature_store()

    def get_or_create_feature_group(self):
        try:
            return self.fs.get_feature_group(
                name=settings.FEATURE_GROUP_NAME,
                version=settings.FEATURE_GROUP_VERSION
            )
        except Exception as e:
            logger.info(f"Feature group not found, creating it: {e}")
            return self.fs.create_feature_group(
                name=settings.FEATURE_GROUP_NAME,
                version=settings.FEATURE_GROUP_VERSION,
                description="AQI and weather features",
                primary_key=["city_id", "timestamp"],
                event_time="timestamp",
                online_enabled=True # Needed for inference
            )

    def write_features(self, df: pd.DataFrame):
        """
        Writes the dataframe to the Hopsworks feature group.
        """
        fg = self.get_or_create_feature_group()
        
        # Hopsworks expects timestamp columns to be timezone-naive or properly formatted
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # Ensure we convert any tz-aware to tz-naive UTC for Hopsworks compatibility
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Write to feature store (upsert by default)
        logger.info(f"Writing {len(df)} rows to feature store...")
        fg.insert(df, write_options={"wait_for_job": False})
        logger.info("Successfully initiated write to feature store.")
        
    def get_recent_history(self, city_id: str, hours: int = 48) -> pd.DataFrame:
        """
        Fetches the recent history for a city to compute rolling/lag features for new data.
        """
        try:
            fg = self.fs.get_feature_group(
                name=settings.FEATURE_GROUP_NAME,
                version=settings.FEATURE_GROUP_VERSION
            )
            
            # Simple query to get history
            query = fg.select_all()
            df = query.read()
            
            # Filter in pandas (for small datasets this is fine; for large ones we'd push down)
            df = df[df["city_id"] == city_id]
            df = df.sort_values(by="timestamp").tail(hours)
            return df
        except Exception as e:
            logger.warning(f"Could not fetch history (maybe first run?): {e}")
            # Return empty dataframe with expected columns if we can't read
            return pd.DataFrame()
