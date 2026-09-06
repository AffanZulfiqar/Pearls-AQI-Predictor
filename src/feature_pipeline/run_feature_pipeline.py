import logging
import pandas as pd
from datetime import datetime, timezone
from src.config import settings
from src.config.logging_config import setup_logging
from src.data_ingestion.aqicn_client import AQICNClient
from src.data_ingestion.openweather_client import OpenWeatherClient
from src.feature_pipeline.feature_store_writer import FeatureStoreWriter
from src.feature_pipeline.feature_engineering import compute_features
from src.feature_pipeline.fetch_raw_data import fetch_for_city

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    logger.info("Starting feature pipeline...")

    try:
        aqicn = AQICNClient(settings.AQICN_API_KEY)
        weather = OpenWeatherClient(settings.OPENWEATHER_API_KEY)
        fs_writer = FeatureStoreWriter()
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise e

    current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    for city in settings.CITIES:
        city_id = city["id"]
        logger.info(f"Processing {city_id}...")
        
        # 1. Fetch raw data (using dedicated fetch module)
        raw_row = fetch_for_city(aqicn, weather, city, current_time)
        if raw_row is None:
            continue
        
        df_new = pd.DataFrame([raw_row])
        
        # 2. Get history for context
        df_history = fs_writer.get_recent_history(city_id, hours=48)
        
        if not df_history.empty:
            df_combined = pd.concat([df_history, df_new], ignore_index=True)
            # Deduplicate just in case
            df_combined = df_combined.drop_duplicates(subset=["city_id", "timestamp"], keep="last")
        else:
            df_combined = df_new
            
        # 3. Compute features
        df_featured = compute_features(df_combined)
        
        # 4. Extract just the new row to write
        df_to_write = df_featured[df_featured["timestamp"] == current_time]
        
        # If there was no history, the lag/rolling features will be NaNs.
        # Hopsworks can handle NaNs, but for ML we'll need to impute or drop them later.
        
        # 5. Write to Feature Store
        fs_writer.write_features(df_to_write)
        logger.info(f"Successfully processed {city_id}.")

    logger.info("Feature pipeline finished.")

if __name__ == "__main__":
    main()
