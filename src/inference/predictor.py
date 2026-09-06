"""
Inference Predictor.

Loads registered models from Hopsworks Model Registry.
Uses feature_schema.json stored with each model to guarantee exact feature
column order and prevent train/serve skew.

If Hopsworks is unavailable, predictor.available = False and the Flask API
returns HTTP 503 — never fake predictions.
"""
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.config.hopsworks_client import get_feature_store, get_model_registry

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self):
        self.fs          = None
        self.mr          = None
        self.models      = {}          # horizon -> model object
        self.model_types = {}          # horizon -> model_type string
        self.feature_cols = {}         # horizon -> list of feature col names
        self.model_metadata = {}       # horizon -> full metadata dict
        self.available   = False

        try:
            self.fs = get_feature_store()
            self.mr = get_model_registry()
            self._load_models()
            self.available = len(self.models) > 0
        except Exception:
            logger.exception(
                "Failed to connect to Hopsworks during Predictor init. "
                "Predictor will return unavailable."
            )
            self.available = False

    def _load_models(self):
        """Loads the best registered model for each horizon from Hopsworks."""
        for horizon in settings.FORECAST_HORIZONS:
            model_name = f"aqi_model_{horizon}"
            try:
                hw_model   = self.mr.get_best_model(model_name, "rmse", "min")
                model_dir  = hw_model.download()

                # Load feature schema — required for train/serve parity
                schema_path = os.path.join(model_dir, "feature_schema.json")
                if not os.path.exists(schema_path):
                    logger.warning(
                        "No feature_schema.json found in model dir for %s. "
                        "Feature column order may not match training.", horizon
                    )
                    feature_cols = None
                else:
                    with open(schema_path) as f:
                        schema = json.load(f)
                    feature_cols = schema.get("feature_cols")
                    logger.info(
                        "Loaded feature schema for %s: %d features, data_source=%s",
                        horizon, len(feature_cols), schema.get("data_source", "unknown")
                    )

                self.feature_cols[horizon]    = feature_cols
                self.model_metadata[horizon]  = dict(hw_model.training_metrics or {})
                model_type = self.model_metadata[horizon].get("model_type", "unknown")
                self.model_types[horizon]     = model_type

                # Load model artifact
                keras_path = os.path.join(model_dir, "model.keras")
                pkl_path   = os.path.join(model_dir, "model.pkl")
                if os.path.exists(keras_path):
                    import tensorflow as tf
                    self.models[horizon] = tf.keras.models.load_model(keras_path)
                elif os.path.exists(pkl_path):
                    self.models[horizon] = joblib.load(pkl_path)
                else:
                    raise FileNotFoundError(
                        f"No model artifact found in {model_dir} for {horizon}"
                    )

                logger.info("Loaded model for %s: %s v%s", horizon, model_name, hw_model.version)

            except Exception as e:
                logger.warning("Could not load model for %s: %s", horizon, e)

    def _get_feature_row(self, city_id: str):
        """
        Fetches the latest feature row for a city from the Hopsworks Feature Store.
        Returns (df_row, feature_cols) or (None, None) if unavailable.
        """
        if not self.fs:
            return None, None

        try:
            fg  = self.fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
            if fg is None:
                logger.error("Feature group not found for inference.")
                return None, None

            df  = fg.select_all().read()
            df  = df[df["city_id"] == city_id].sort_values(by="timestamp").tail(1)

            if df.empty:
                logger.warning("No feature data found for city %s", city_id)
                return None, None

            return df, None   # feature_cols loaded per-horizon from schema

        except Exception:
            logger.exception("Failed to fetch features for %s", city_id)
            return None, None

    def _prepare_features(self, df: pd.DataFrame, horizon: str):
        """
        Selects and orders feature columns according to the stored feature schema.
        Raises ValueError on missing features.
        """
        required = self.feature_cols.get(horizon)

        if required is None:
            # Fallback: exclude standard non-feature cols
            exclude = {
                "city_id", "timestamp",
                "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
                "data_source_version", "data_source_type",
            }
            cols = [c for c in df.columns if c not in exclude]
            logger.warning(
                "No stored feature schema for %s — using dynamic column selection (%d cols). "
                "This may cause train/serve skew.", horizon, len(cols)
            )
            return df[cols].values

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Inference feature row for {horizon} is missing required columns: {missing}. "
                "Feature group schema may not match training schema."
            )

        # Use exact training order; ignore extra columns
        return df[required].values

    def predict(self, city_id: str) -> dict:
        """
        Returns {horizon: float} predictions or raises if unavailable.
        """
        df, _ = self._get_feature_row(city_id)
        if df is None:
            return None

        predictions = {}
        for horizon, model in self.models.items():
            try:
                X    = self._prepare_features(df, horizon)
                pred = model.predict(X)
                if hasattr(pred, "flatten"):
                    pred = pred.flatten()
                predictions[horizon] = float(pred[0])
            except Exception:
                logger.exception("Prediction failed for %s/%s", city_id, horizon)

        return predictions if predictions else None

    def explain(self, city_id: str) -> dict:
        """
        Returns SHAP feature importance per horizon, or empty dict on failure.
        """
        df, _ = self._get_feature_row(city_id)
        if df is None:
            return None

        explanations = {}
        for horizon, model in self.models.items():
            try:
                import shap
                X          = self._prepare_features(df, horizon)
                model_type = self.model_types.get(horizon, "unknown")
                feature_names = self.feature_cols.get(horizon) or [f"f{i}" for i in range(X.shape[1])]

                if model_type == "random_forest":
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                elif model_type == "ridge":
                    ridge_step  = model.named_steps["ridge"]
                    scaler      = model.named_steps["scaler"]
                    X_scaled    = scaler.transform(X)
                    explainer   = shap.LinearExplainer(ridge_step, X_scaled)
                    shap_values = explainer.shap_values(X_scaled)
                else:
                    explainer   = shap.KernelExplainer(model.predict, X)
                    shap_values = explainer.shap_values(X)

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                importance = np.abs(shap_values).flatten()
                ranked = sorted(
                    [{"feature": feature_names[i], "importance": float(importance[i])}
                     for i in range(len(feature_names))],
                    key=lambda x: x["importance"],
                    reverse=True
                )
                explanations[horizon] = ranked[:10]

            except Exception as e:
                logger.error("SHAP explanation failed for %s/%s: %s", city_id, horizon, e)
                explanations[horizon] = []

        return explanations

    def get_latest_row(self, city_id: str) -> dict:
        """Returns the latest feature row as a dict, for the /current endpoint."""
        df, _ = self._get_feature_row(city_id)
        if df is None or df.empty:
            return None
        row = df.iloc[-1].to_dict()
        return row

    def get_history(self, city_id: str, hours: int = 168) -> list:
        """Returns up to `hours` recent rows as a list of dicts, for the /history endpoint."""
        if not self.fs:
            return None
        try:
            fg = self.fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
            if fg is None:
                return None
            df = fg.select_all().read()
            df = df[df["city_id"] == city_id].sort_values("timestamp").tail(hours)
            if df.empty:
                return []
            # Return timestamp + aqi only for the chart
            result = []
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                result.append({"timestamp": str(ts), "aqi": float(row["aqi"])})
            return result
        except Exception:
            logger.exception("Failed to fetch history for %s", city_id)
            return None
