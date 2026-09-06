import logging
import pandas as pd
from datetime import datetime, timezone
from src.config import settings
from src.config.settings import validate_env
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

    # Validate all required env vars before doing anything else
    validate_env()

    # Initialize clients — any failure here raises immediately
    try:
        aqicn = AQICNClient(settings.AQICN_API_KEY)
        weather = OpenWeatherClient(settings.OPENWEATHER_API_KEY)
        fs_writer = FeatureStoreWriter()
    except Exception:
        logger.exception("Failed to initialize clients")
        raise

    current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    pipeline_failures = []

    for city in settings.CITIES:
        city_id = city["id"]
        logger.info("Processing %s...", city_id)

        try:
            # 1. Fetch raw data
            raw_row = fetch_for_city(aqicn, weather, city, current_time)
            if raw_row is None:
                raise RuntimeError(f"API data fetch returned None for {city_id} — check AQICN/OpenWeather APIs")

            df_new = pd.DataFrame([raw_row])

            # 2. Get history for lag/rolling features
            df_history = fs_writer.get_recent_history(city_id, hours=48)

            if not df_history.empty:
                df_combined = pd.concat([df_history, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(
                    subset=["city_id", "timestamp"], keep="last"
                )
            else:
                df_combined = df_new

            # 3. Compute features (NaN handling built-in for first run)
            df_featured = compute_features(df_combined)

            # 4. Extract just the new row to write
            # The timestamp in df_featured is timezone-naive (UTC) as required by Hopsworks, 
            # so we must strip the timezone from current_time before comparing.
            current_time_naive = current_time.replace(tzinfo=None)
            df_to_write = df_featured[df_featured["timestamp"] == current_time_naive]

            if df_to_write.empty:
                raise RuntimeError(
                    f"Feature engineering produced no row for timestamp={current_time_naive} city={city_id}"
                )

            # 5. Write to Feature Store (raises on failure)
            fs_writer.write_features(df_to_write)
            logger.info("Successfully processed %s at %s.", city_id, current_time)

        except Exception:
            logger.exception("Pipeline failed for city=%s", city_id)
            pipeline_failures.append(city_id)

    if pipeline_failures:
        raise RuntimeError(
            f"Feature pipeline failed for cities: {pipeline_failures}. "
            "See logs above for details."
        )

    logger.info("Feature pipeline finished successfully.")


if __name__ == "__main__":
    main()
