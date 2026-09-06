import pandas as pd
import hopsworks
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self):
        self.project = hopsworks.login(
            project=settings.HOPSWORKS_PROJECT_NAME,
            api_key_value=settings.HOPSWORKS_API_KEY
        )
        self.fs = self.project.get_feature_store()

    def build_training_dataset(self):
        """
        Pulls features, computes forward-looking targets, and splits into
        train/validation/test using time-based splitting (no random shuffle).
        
        Returns:
            train_df, val_df, test_df, feature_cols
        """
        logger.info("Fetching features from Hopsworks...")
        fg = self.fs.get_feature_group(
            name=settings.FEATURE_GROUP_NAME,
            version=settings.FEATURE_GROUP_VERSION
        )
        df = fg.select_all().read()
        
        # Ensure sorted chronologically per city
        df = df.sort_values(by=["city_id", "timestamp"]).copy()
        
        # Create targets (forward shifting)
        logger.info("Computing forward targets...")
        df["target_aqi_24h"] = df.groupby("city_id")["aqi"].shift(-24)
        df["target_aqi_48h"] = df.groupby("city_id")["aqi"].shift(-48)
        df["target_aqi_72h"] = df.groupby("city_id")["aqi"].shift(-72)
        
        # Drop rows where target is NaN (the last 72 hours of data)
        df = df.dropna(subset=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h", "aqi_lag_24h", "aqi_rolling_24h"])
        
        # Time-based 3-way split: train (60%) / validation (20%) / test (20%)
        # Sorting globally by timestamp to ensure test is the most recent data
        logger.info("Splitting into train / validation / test sets...")
        df = df.sort_values(by="timestamp")
        
        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Define features to use (exclude ids, targets, and data_source_version)
        exclude_cols = [
            "city_id", "timestamp",
            "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
            "data_source_version"
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df, feature_cols
