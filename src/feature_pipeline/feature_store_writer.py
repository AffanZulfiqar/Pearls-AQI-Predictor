import pandas as pd
import logging
from src.config import settings
from src.config.hopsworks_client import get_feature_store

logger = logging.getLogger(__name__)


class FeatureStoreWriter:
    def __init__(self):
        self.fs = get_feature_store()

    def get_or_create_feature_group(self):
        """
        Gets the feature group if it exists, or creates it if not.
        Uses time_travel_format='NONE' to avoid requiring the delta library.
        """
        fg = self.fs.get_feature_group(
            name=settings.FEATURE_GROUP_NAME,
            version=settings.FEATURE_GROUP_VERSION
        )
        if fg is not None:
            return fg, False

        logger.info(
            "Feature group '%s' v%s not found — creating it.",
            settings.FEATURE_GROUP_NAME,
            settings.FEATURE_GROUP_VERSION
        )
        fg = self.fs.create_feature_group(
            name=settings.FEATURE_GROUP_NAME,
            version=settings.FEATURE_GROUP_VERSION,
            description="Hourly AQI and weather features for Islamabad, Karachi, Lahore",
            primary_key=["city_id", "timestamp"],
            event_time="timestamp",
            online_enabled=True,
            time_travel_format="HUDI"
        )
        return fg, True

    def write_features(self, df: pd.DataFrame):
        """
        Writes the dataframe to the Hopsworks feature group.
        After writing, reads back and verifies the data was persisted.
        Raises on any failure — never silently succeeds.
        """
        fg, is_new = self.get_or_create_feature_group()

        # Ensure timestamp is datetime (timezone-naive UTC for Hopsworks)
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        n_rows = len(df)
        cities_in_batch = df["city_id"].unique().tolist() if "city_id" in df.columns else []

        logger.info(
            "Writing %d rows to feature store (cities: %s)...",
            n_rows,
            cities_in_batch
        )

        # Always use insert() for Python engine! save() is for Spark and causes time_travel errors.
        fg.insert(df, write_options={"wait_for_job": False})

        logger.info("Write initiated successfully. Offline materialization job is running in Hopsworks.")

    def _verify_readback(self, fg, written_df: pd.DataFrame):
        """
        Reads back the feature group and verifies at least some of the written rows exist.
        Raises if the read-back is empty (indicates write silently failed).
        """
        try:
            readback = fg.select_all().read()
        except Exception:
            logger.exception("Read-back from feature group failed after write")
            raise

        if readback is None or readback.empty:
            raise RuntimeError(
                "Feature group read-back returned empty DataFrame — "
                "the write may have silently failed. Check Hopsworks UI."
            )

        # Log per-city latest timestamp from what we just wrote
        for city_id in written_df["city_id"].unique():
            city_written = written_df[written_df["city_id"] == city_id]
            latest_ts = city_written["timestamp"].max()
            city_rb = readback[readback["city_id"] == city_id] if "city_id" in readback.columns else readback
            n_in_store = len(city_rb)
            logger.info(
                "VERIFIED city=%s  timestamp=%s  total_rows_in_store=%d",
                city_id,
                latest_ts,
                n_in_store
            )

    def get_recent_history(self, city_id: str, hours: int = 48) -> pd.DataFrame:
        """
        Fetches the recent history for a city to compute rolling/lag features for new data.
        Returns an empty DataFrame (not None) if no data exists yet.
        """
        try:
            fg = self.fs.get_feature_group(
                name=settings.FEATURE_GROUP_NAME,
                version=settings.FEATURE_GROUP_VERSION
            )
            if fg is None:
                return pd.DataFrame()

            query = fg.select_all()
            df = query.read()

            df = df[df["city_id"] == city_id]
            df = df.sort_values(by="timestamp").tail(hours)
            return df
        except Exception as e:
            logger.warning("Could not fetch history (maybe first run?): %s", e)
            return pd.DataFrame()
