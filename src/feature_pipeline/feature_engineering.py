import pandas as pd
import numpy as np
from src.config import settings


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-based and derived features for an AQI dataframe.
    Assumes df has columns: 'timestamp', 'city_id', 'aqi', and other raw readings.
    'timestamp' must be a datetime column.

    For first-run scenarios where only one row exists (no history), lag and rolling
    features that would be NaN are filled with the available AQI value so that
    inference never receives NaN model inputs.

    Returns the dataframe with new feature columns added.
    """
    df = df.sort_values(by=["city_id", "timestamp"]).copy()

    # Ensure timestamp is datetime (handles case where Hopsworks returns string or mixed types)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # utc=True safely parses both strings and naive datetimes into UTC datetimes
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # --- Time-based features ---
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # --- Lag features (per city) ---
    df["aqi_lag_1h"] = df.groupby("city_id")["aqi"].shift(1)
    df["aqi_lag_24h"] = df.groupby("city_id")["aqi"].shift(24)

    # --- AQI change rate ---
    epsilon = 1e-5
    df["aqi_change_rate"] = (df["aqi"] - df["aqi_lag_1h"]) / (df["aqi_lag_1h"] + epsilon)

    # --- Rolling averages (min_periods=1 ensures no NaN on first rows) ---
    df["aqi_rolling_3h"] = df.groupby("city_id")["aqi"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["aqi_rolling_24h"] = df.groupby("city_id")["aqi"].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )

    # --- First-run NaN fill ---
    # When there is insufficient history (e.g. first run), lag features are NaN.
    # Fill them with the current AQI value so no NaN reaches the model.
    # This is a conservative estimate (assumes no change), which is acceptable
    # for a cold-start scenario. Real data will replace this within 24h.
    for lag_col in ["aqi_lag_1h", "aqi_lag_24h"]:
        df[lag_col] = df[lag_col].fillna(df["aqi"])

    # Change rate is 0 when there is no prior value (no change assumption)
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0.0)

    # Fill any remaining NaN numeric columns with column median (per city)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df.groupby("city_id")[col].transform(
                lambda x: x.fillna(x.median() if not x.isna().all() else 0.0)
            )
        # Final fallback: global 0 if still NaN
        df[col] = df[col].fillna(0.0)

    # --- Data source version tag (prevents train/serve skew) ---
    df["data_source_version"] = settings.DATA_SOURCE_VERSION

    return df
