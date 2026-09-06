import hopsworks
from src.config import settings
import logging
import joblib
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        try:
            self.project = hopsworks.login(
                project=settings.HOPSWORKS_PROJECT_NAME,
                api_key_value=settings.HOPSWORKS_API_KEY
            )
            self.fs = self.project.get_feature_store()
            self.mr = self.project.get_model_registry()
            self.models = {}
            self.model_types = {}  # Track which model type won per horizon
            self.feature_cols = None
            self._load_models()
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks during Predictor init: {e}")
            self.fs = None
        
    def _load_models(self):
        for horizon in settings.FORECAST_HORIZONS:
            model_name = f"aqi_model_{horizon}"
            try:
                hw_model = self.mr.get_best_model(model_name, "rmse", "min")
                model_dir = hw_model.download()
                
                # Track model type from metadata
                model_type = hw_model.metrics.get("model_type", "unknown")
                self.model_types[horizon] = model_type
                
                if os.path.exists(f"{model_dir}/model.keras"):
                    import tensorflow as tf
                    self.models[horizon] = tf.keras.models.load_model(f"{model_dir}/model.keras")
                else:
                    self.models[horizon] = joblib.load(f"{model_dir}/model.pkl")
            except Exception as e:
                logger.warning(f"Could not load model for {horizon}: {e}")

    def _get_feature_row(self, city_id):
        """Fetch the latest feature row for a city from the feature store."""
        if not self.fs:
            return None, None
        
        try:
            fg = self.fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
            df = fg.select_all().read()
            df = df[df["city_id"] == city_id].sort_values(by="timestamp").tail(1)
            
            if df.empty:
                logger.warning(f"No feature data found for city {city_id}")
                return None, None
            
            # Define exact columns used in training (excluding targets, ids, version tag)
            exclude_cols = [
                "city_id", "timestamp",
                "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
                "data_source_version"
            ]
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            self.feature_cols = feature_cols
            
            return df, feature_cols
        except Exception as e:
            logger.error(f"Failed to fetch features for {city_id}: {e}")
            return None, None

    def predict(self, city_id):
        df, feature_cols = self._get_feature_row(city_id)
        if df is None:
            return None
        
        X = df[feature_cols].values
        
        predictions = {}
        for horizon, model in self.models.items():
            if hasattr(model, "predict"):
                pred = model.predict(X)
                if hasattr(pred, "flatten"):
                    pred = pred.flatten()
                predictions[horizon] = float(pred[0])
        return predictions

    def explain(self, city_id):
        """
        Generate SHAP feature importance for the current prediction.
        Returns a dict mapping horizon -> list of {feature, importance} sorted by |importance|.
        """
        df, feature_cols = self._get_feature_row(city_id)
        if df is None or not feature_cols:
            return None
        
        X = df[feature_cols].values
        explanations = {}
        
        for horizon, model in self.models.items():
            try:
                import shap
                model_type = self.model_types.get(horizon, "unknown")
                
                if model_type == "random_forest":
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                elif model_type == "ridge":
                    ridge_model = model.named_steps['ridge']
                    scaler = model.named_steps['scaler']
                    X_scaled = scaler.transform(X)
                    explainer = shap.LinearExplainer(ridge_model, X_scaled)
                    shap_values = explainer.shap_values(X_scaled)
                else:
                    # TF or unknown — use KernelExplainer with X as background
                    predict_fn = model.predict if not hasattr(model, 'model') else model.predict
                    explainer = shap.KernelExplainer(predict_fn, X)
                    shap_values = explainer.shap_values(X)
                
                # Build sorted feature importance list
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                importance = np.abs(shap_values).flatten()
                feature_importance = [
                    {"feature": feature_cols[i], "importance": float(importance[i])}
                    for i in range(len(feature_cols))
                ]
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                explanations[horizon] = feature_importance[:10]  # Top 10 features
                
            except Exception as e:
                logger.error(f"SHAP explanation failed for {horizon}: {e}")
                explanations[horizon] = []
        
        return explanations
