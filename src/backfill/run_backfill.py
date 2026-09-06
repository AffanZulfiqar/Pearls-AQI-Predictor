import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
import argparse
from src.config import settings
from src.config.logging_config import setup_logging
from src.feature_pipeline.feature_store_writer import FeatureStoreWriter
from src.feature_pipeline.feature_engineering import compute_features

logger = logging.getLogger(__name__)

def generate_synthetic_history(city, start_date, end_date):
    """
    Since free tiers of AQICN and OpenWeather often lack deep historical endpoints,
    we simulate historical data for backfill purposes for demo/training,
    or we would replace this with a bulk historical CSV load.
    
    NOTE: This produces synthetic (fake) data. Models trained on this data
    will learn synthetic patterns, not real-world AQI dynamics. For production
    use, replace with actual historical data from APIs or CSV exports.
    """
    # Create hourly range
    date_range = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
    
    import numpy as np
    np.random.seed(42)
    
    n = len(date_range)
    # Simulate a daily seasonal pattern for AQI + random noise
    time_of_day_effect = np.sin(date_range.hour * np.pi / 12) * 20
    base_aqi = 80 + time_of_day_effect + np.random.normal(0, 10, n)
    
    data = []
    for i, dt in enumerate(date_range):
        aqi = max(0, base_aqi[i]) # Ensure non-negative
        data.append({
            "city_id": city["id"],
            "timestamp": dt,
            "latitude": city["lat"],
            "longitude": city["lon"],
            "aqi": aqi,
            "pm2_5": aqi * 0.4,
            "pm10": aqi * 0.6,
            "no2": np.random.uniform(5, 40),
            "so2": np.random.uniform(1, 10),
            "co": np.random.uniform(0.1, 1.5),
            "o3": np.random.uniform(10, 60),
            "temperature": 25 + np.sin(dt.hour * np.pi / 12) * 5 + np.random.normal(0, 2),
            "humidity": 60 + np.random.normal(0, 5),
            "wind_speed": max(0, np.random.normal(5, 2)),
            "pressure": 1012 + np.random.normal(0, 2),
            "precipitation": 0 if np.random.random() > 0.1 else np.random.uniform(0.1, 5.0)
        })
        
    return pd.DataFrame(data)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Backfill historical AQI data.")
    parser.add_argument("--days", type=int, default=None, help="Number of days to backfill (shorthand for start/end)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--city", type=str, default=None, help="City ID to backfill (default: all cities)")
    args = parser.parse_args()
    
    # Determine date range
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.days:
        start_date = end_date - timedelta(days=args.days)
    else:
        # Default: 60 days
        start_date = end_date - timedelta(days=60)
    
    # Determine which cities to backfill
    if args.city:
        cities = [c for c in settings.CITIES if c["id"] == args.city.lower()]
        if not cities:
            logger.error(f"City '{args.city}' not found. Available: {[c['id'] for c in settings.CITIES]}")
            return
    else:
        cities = settings.CITIES
    
    logger.info(f"Starting historical backfill from {start_date} to {end_date} for {[c['id'] for c in cities]}...")
    
    fs_writer = FeatureStoreWriter()
    all_featured_data = []
    
    for city in cities:
        logger.info(f"Backfilling {city['id']} from {start_date} to {end_date}...")
        
        # 1. Fetch historical raw data (using synthetic generation for this demo)
        df_raw = generate_synthetic_history(city, start_date, end_date)
        
        # 2. Compute features on the entire historical dataset
        df_featured = compute_features(df_raw)
        
        # Log any gaps
        expected_hours = int((end_date - start_date).total_seconds() / 3600) + 1
        actual_hours = len(df_featured)
        if actual_hours < expected_hours:
            logger.warning(f"Gap detected for {city['id']}: expected {expected_hours} rows, got {actual_hours}")
        
        all_featured_data.append(df_featured)
        
    if all_featured_data:
        final_df = pd.concat(all_featured_data, ignore_index=True)
        # Drop rows with NaNs in lagged columns caused by the start of the series
        final_df = final_df.dropna(subset=["aqi_lag_24h"])
        
        logger.info(f"Writing {len(final_df)} backfilled rows to feature store...")
        fs_writer.write_features(final_df)
        logger.info("Backfill finished.")

if __name__ == "__main__":
    main()
