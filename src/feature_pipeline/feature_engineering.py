import pandas as pd
import numpy as np
from src.config import settings


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-based and derived features for an AQI dataframe.
    Assumes df has columns: 'timestamp', 'city_id', 'aqi', and other raw readings.
    'timestamp' must be a datetime column.
    
    If df contains a history of readings, it computes rolling and lag features.
    Returns the dataframe with new feature columns added.
    """
    # Ensure sorted by time
    df = df.sort_values(by=["city_id", "timestamp"]).copy()
    
    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Derived features (computed per city)
    # We group by city in case the dataframe contains multiple cities
    
    # Lags
    df["aqi_lag_1h"] = df.groupby("city_id")["aqi"].shift(1)
    df["aqi_lag_24h"] = df.groupby("city_id")["aqi"].shift(24)
    
    # AQI change rate: (aqi_t - aqi_t-1) / aqi_t-1
    # Add a small epsilon to avoid division by zero if aqi_t-1 is 0
    epsilon = 1e-5
    df["aqi_change_rate"] = (df["aqi"] - df["aqi_lag_1h"]) / (df["aqi_lag_1h"] + epsilon)
    
    # Rolling averages
    # Assuming the data is hourly, rolling(3) is 3 hours
    df["aqi_rolling_3h"] = df.groupby("city_id")["aqi"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["aqi_rolling_24h"] = df.groupby("city_id")["aqi"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    
    # Data source version tag — prevents train/serve skew (§4.1)
    df["data_source_version"] = settings.DATA_SOURCE_VERSION
    
    return df
