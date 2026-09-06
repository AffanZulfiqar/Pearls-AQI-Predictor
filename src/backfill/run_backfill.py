"""
Backfill pipeline: REAL historical AQI data from OpenWeather Air Pollution History API.

The OpenWeather Air Pollution History API is FREE and provides REAL hourly data
going back to November 27, 2020. This gives us years of real training data.

API: http://api.openweathermap.org/data/2.5/air_pollution/history
     ?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={key}

Default: 1 year of real data (no synthetic needed!)
Maximum: back to 2020-11-27

Weather fields (temperature, humidity, wind_speed, pressure, precipitation)
from the historical pollution API return 0. These are secondary features;
the primary AQI signals and pollutant concentrations are real.

Synthetic data is available via --allow-synthetic ONLY as a last resort
(e.g., for features not available in the free API). By default we use real data.

Usage:
  # 1 year real data (default):
  python -m src.backfill.run_backfill

  # 3 years real data:
  python -m src.backfill.run_backfill --days 1095

  # Custom date range:
  python -m src.backfill.run_backfill --start-date 2022-01-01 --end-date 2023-01-01

  # Extend with calibrated synthetic beyond available real data:
  python -m src.backfill.run_backfill --allow-synthetic --days 1500
"""
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.config import settings
from src.config.settings import validate_env
from src.config.logging_config import setup_logging
from src.data_ingestion.openweather_client import OpenWeatherClient
from src.feature_pipeline.feature_engineering import compute_features
from src.feature_pipeline.feature_store_writer import FeatureStoreWriter

logger = logging.getLogger(__name__)

# OpenWeather Air Pollution History API is free back to this date
POLLUTION_HISTORY_START = datetime(2020, 11, 27, tzinfo=timezone.utc)

# Default training window: 1 year of real data
DEFAULT_DAYS = 365


def fetch_real_pollution_history(client: OpenWeatherClient, city: dict,
                                 start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetches REAL historical air pollution data from OpenWeather.
    
    FREE tier, available from 2020-11-27 to present, hourly granularity.
    Returns AQI, PM2.5, PM10, NO2, SO2, CO, O3 — all REAL measurements.
    
    Note: temperature/humidity/wind/pressure are 0 for historical rows
    since OpenWeather's historical weather API requires a paid subscription.
    These are secondary features; the primary pollution signals are real.
    """
    # Enforce the API's start date boundary
    actual_start = max(start_dt, POLLUTION_HISTORY_START)
    if actual_start > start_dt:
        logger.info(
            "Adjusting start date from %s to %s (OpenWeather history limit).",
            start_dt.date(), actual_start.date()
        )

    # API can only handle ~1 month per request due to data volume — chunk it
    rows = []
    chunk_start = actual_start
    chunk_size  = timedelta(days=30)

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + chunk_size, end_dt)
        logger.info(
            "Fetching %s: %s → %s...",
            city["name"], chunk_start.date(), chunk_end.date()
        )
        chunk_rows = client.get_air_pollution_history(
            city_name=city["name"],
            lat=city["lat"],
            lon=city["lon"],
            start_dt=chunk_start,
            end_dt=chunk_end,
        )
        rows.extend(chunk_rows)
        chunk_start = chunk_end + timedelta(hours=1)

    if not rows:
        raise RuntimeError(
            f"OpenWeather returned 0 records for {city['name']} "
            f"({actual_start.date()} → {end_dt.date()}). Check API key."
        )

    df = pd.DataFrame(rows)
    df["city_id"]          = city["id"]
    df["latitude"]         = city["lat"]
    df["longitude"]        = city["lon"]
    df["data_source_type"] = "real"
    return df


def generate_calibrated_synthetic(city: dict, start_dt: datetime, end_dt: datetime,
                                   real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates CALIBRATED SYNTHETIC data based on real statistics.
    Only used when --allow-synthetic is passed AND the requested date range
    extends before 2020-11-27 (beyond OpenWeather's free history).
    Clearly marked as synthetic in the data and model metadata.
    """
    logger.warning(
        "Generating calibrated synthetic data for %s (%s → %s). "
        "Statistics derived from real observations. "
        "Model will be marked data_source='mixed'.",
        city["id"], start_dt.date(), end_dt.date()
    )

    real_aqi = real_df["aqi"].dropna()
    base_aqi = float(real_aqi.mean()) if len(real_aqi) > 0 else 80.0
    aqi_std  = float(real_aqi.std())  if len(real_aqi) > 1 else 15.0
    pm25_ratio = float(real_df["pm2_5"].mean() / real_aqi.mean()) if real_aqi.mean() > 0 else 0.35
    pm10_ratio = float(real_df["pm10"].mean() / real_aqi.mean()) if real_aqi.mean() > 0 else 0.55

    date_range = pd.date_range(start=start_dt, end=end_dt, freq="h").tz_localize(None)
    n = len(date_range)
    np.random.seed(42)

    seasonal   = np.sin(np.arange(n) * 2 * np.pi / (24 * 30)) * (aqi_std * 0.5)
    diurnal    = np.sin(np.arange(n) % 24 * np.pi / 12) * 10
    noise      = np.random.normal(0, aqi_std * 0.4, n)
    aqi_series = np.clip(base_aqi + seasonal + diurnal + noise, 5, 400)

    rows = []
    for i, dt in enumerate(date_range):
        aqi = float(aqi_series[i])
        rows.append({
            "city_id": city["id"], "timestamp": dt,
            "latitude": city["lat"], "longitude": city["lon"],
            "aqi": round(aqi, 1),
            "pm2_5": round(max(0, aqi * pm25_ratio + np.random.normal(0, 2)), 1),
            "pm10":  round(max(0, aqi * pm10_ratio + np.random.normal(0, 3)), 1),
            "no2":   round(max(0, np.random.normal(float(real_df["no2"].mean()), 5)), 1),
            "so2":   round(max(0, np.random.normal(float(real_df["so2"].mean()), 2)), 1),
            "co":    round(max(0, np.random.normal(float(real_df["co"].mean()),  0.2)), 2),
            "o3":    round(max(0, np.random.normal(float(real_df["o3"].mean()),  8)), 1),
            "temperature": round(25 + np.sin(dt.hour * np.pi / 12) * 6, 1),
            "humidity":    round(float(np.clip(60 + np.random.normal(0, 8), 20, 100)), 1),
            "wind_speed":  round(max(0, float(np.random.normal(4, 2))), 1),
            "pressure":    1013.0,
            "precipitation": 0.0,
            "data_source_type": "synthetic",
        })
    return pd.DataFrame(rows)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Backfill AQI data using OpenWeather Air Pollution History API "
            "(FREE, real data from 2020-11-27 onwards)."
        )
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Days of history to backfill (default: {DEFAULT_DAYS}). Real data available since 2020-11-27."
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date",   type=str, default=None, help="End date YYYY-MM-DD.")
    parser.add_argument("--city",       type=str, default=None, help="City ID (default: all).")
    parser.add_argument(
        "--allow-synthetic", action="store_true", default=False,
        help=(
            "Allow calibrated synthetic extension when the requested date range "
            "goes before 2020-11-27. Calibrated to real city statistics. "
            "Models marked data_source='mixed'."
        )
    )
    args = parser.parse_args()

    validate_env()

    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date   = datetime.strptime(args.end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_date = end_date - timedelta(days=args.days)

    # Check if we need synthetic extension
    needs_synthetic = start_date < POLLUTION_HISTORY_START
    if needs_synthetic and not args.allow_synthetic:
        logger.warning(
            "Requested start date %s is before OpenWeather history start %s. "
            "Capping at %s. Pass --allow-synthetic to extend further with calibrated synthetic data.",
            start_date.date(), POLLUTION_HISTORY_START.date(), POLLUTION_HISTORY_START.date()
        )
        start_date = POLLUTION_HISTORY_START

    cities = settings.CITIES
    if args.city:
        cities = [c for c in settings.CITIES if c["id"] == args.city.lower()]
        if not cities:
            raise ValueError(f"Unknown city: '{args.city}'")

    logger.info(
        "Backfill: %s → %s | cities=%s | real_data=YES | allow_synthetic=%s",
        start_date.date(), end_date.date(),
        [c["id"] for c in cities], args.allow_synthetic
    )
    logger.info(
        "Using OpenWeather Air Pollution History API (FREE, real data since 2020-11-27)."
    )

    weather_client = OpenWeatherClient(settings.OPENWEATHER_API_KEY)
    fs_writer      = FeatureStoreWriter()
    all_featured   = []

    for city in cities:
        logger.info("=== Processing %s ===", city["id"])
        parts = []

        # Real data window
        real_start = max(start_date, POLLUTION_HISTORY_START)
        df_real = fetch_real_pollution_history(weather_client, city, real_start, end_date)
        logger.info(
            "Real data: %d rows (%s → %s)",
            len(df_real), real_start.date(), end_date.date()
        )
        parts.append(df_real)

        # Synthetic extension if before 2020-11-27
        if args.allow_synthetic and start_date < POLLUTION_HISTORY_START:
            synth_end = POLLUTION_HISTORY_START - timedelta(hours=1)
            df_synth  = generate_calibrated_synthetic(city, start_date, synth_end, df_real)
            logger.info(
                "Calibrated synthetic (pre-2020): %d rows (%s → %s)",
                len(df_synth), start_date.date(), synth_end.date()
            )
            parts.append(df_synth)

        df_city = (pd.concat(parts, ignore_index=True)
                     .sort_values("timestamp")
                     .drop_duplicates(subset=["timestamp"]))

        df_featured = compute_features(df_city)

        if "data_source_type" in df_featured.columns:
            src_counts = df_featured["data_source_type"].value_counts()
            logger.info("Data sources for %s:\n%s", city["id"], src_counts.to_string())

        all_featured.append(df_featured)

    if not all_featured:
        raise RuntimeError("No data collected. Backfill aborted.")

    final_df    = pd.concat(all_featured, ignore_index=True)
    before_drop = len(final_df)
    # Drop first 24 rows of each city series (lag_24h is NaN)
    final_df    = final_df.dropna(subset=["aqi_lag_24h"])
    logger.info(
        "Total: %d rows → after lag-NaN drop: %d rows. Writing to Hopsworks...",
        before_drop, len(final_df)
    )

    fs_writer.write_features(final_df)
    logger.info(
        "Backfill complete. %d REAL hourly observations written to Hopsworks.", len(final_df)
    )


if __name__ == "__main__":
    main()
