"""
Builds training/validation/test datasets from the Hopsworks Feature Store.

Also saves a feature_schema.json artifact so the inference service uses the
exact same feature columns in the same order (prevents train/serve skew).
"""
import json
import os
import logging
import pandas as pd
from src.config import settings
from src.config.hopsworks_client import get_feature_store

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_PATH = "artifacts/feature_schema.json"


class DatasetBuilder:
    def __init__(self):
        self.fs = get_feature_store()

    def build_training_dataset(self):
        """
        Pulls features from Hopsworks, computes forward-looking targets, and splits into
        train/validation/test using time-based splitting (no random shuffle).

        Returns:
            train_df, val_df, test_df, feature_cols
        """
        logger.info("Fetching features from Hopsworks feature store...")
        fg = self.fs.get_feature_group(
            name=settings.FEATURE_GROUP_NAME,
            version=settings.FEATURE_GROUP_VERSION
        )
        if fg is None:
            raise RuntimeError(
                f"Feature group '{settings.FEATURE_GROUP_NAME}' v{settings.FEATURE_GROUP_VERSION} "
                "not found in Hopsworks. Run the backfill pipeline first."
            )

        df = fg.select_all().read()

        if df is None or df.empty:
            raise RuntimeError(
                "Feature group read returned empty DataFrame. "
                "Run backfill before training."
            )

        logger.info("Read %d rows from feature store.", len(df))

        # Sort chronologically per city
        df = df.sort_values(by=["city_id", "timestamp"]).copy()

        # Create forward-looking targets
        logger.info("Computing forward targets...")
        df["target_aqi_24h"] = df.groupby("city_id")["aqi"].shift(-24)
        df["target_aqi_48h"] = df.groupby("city_id")["aqi"].shift(-48)
        df["target_aqi_72h"] = df.groupby("city_id")["aqi"].shift(-72)

        # Drop rows where any target is NaN (the last 72 rows of each city)
        required_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
        df = df.dropna(subset=required_cols)

        if df.empty:
            raise RuntimeError(
                "After dropping rows with NaN targets, dataset is empty. "
                "You need at least 72+ hours of data per city. Run backfill."
            )

        # Time-based 3-way split: train (60%) / val (20%) / test (20%)
        logger.info("Splitting into train / validation / test (60/20/20 time-based)...")
        df = df.sort_values(by="timestamp")

        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        # Define features: exclude ids, targets, metadata columns
        exclude_cols = [
            "city_id", "timestamp",
            "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
            "data_source_version", "data_source_type",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        logger.info(
            "Dataset: train=%d, val=%d, test=%d, features=%d",
            len(train_df), len(val_df), len(test_df), len(feature_cols)
        )
        logger.info("Feature columns: %s", feature_cols)

        # Check if any data is synthetic
        data_source = "real"
        if "data_source_type" in df.columns:
            sources = df["data_source_type"].unique().tolist()
            if "synthetic" in sources:
                data_source = "synthetic" if len(sources) == 1 else "mixed"
                logger.warning(
                    "Training dataset contains %s data (sources: %s). "
                    "Registered models will be marked data_source='%s'.",
                    data_source, sources, data_source
                )

        # Save feature schema for train/serve parity
        self._save_feature_schema(feature_cols, data_source)

        return train_df, val_df, test_df, feature_cols, data_source

    def _save_feature_schema(self, feature_cols: list, data_source: str):
        """
        Saves the feature column list and data source metadata to a JSON file.
        The inference service loads this to ensure exact column matching.
        """
        os.makedirs("artifacts", exist_ok=True)
        schema = {
            "feature_cols": feature_cols,
            "data_source": data_source,
            "feature_group_name": settings.FEATURE_GROUP_NAME,
            "feature_group_version": settings.FEATURE_GROUP_VERSION,
            "data_source_version": settings.DATA_SOURCE_VERSION,
        }
        with open(FEATURE_SCHEMA_PATH, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info("Feature schema saved to %s", FEATURE_SCHEMA_PATH)
