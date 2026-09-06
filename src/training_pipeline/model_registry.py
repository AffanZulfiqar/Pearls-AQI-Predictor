"""
Registers trained models to the Hopsworks Model Registry.

Each registered model includes:
  - The model artifact (model.pkl or model.keras)
  - feature_schema.json — exact feature column list and order for inference
  - data_source metadata — 'real', 'synthetic', or 'mixed'
  - Full training metrics (RMSE, MAE, R2)
  - SHAP summary plot (if generated)
"""
import json
import logging
import os
import shutil
import joblib
from datetime import datetime, timezone

from src.config import settings
from src.config.hopsworks_client import get_model_registry

logger = logging.getLogger(__name__)


class ModelRegistryWriter:
    def __init__(self):
        self.mr = get_model_registry()

    def register_model(
        self,
        model,
        model_type: str,
        horizon: str,
        metrics: dict,
        feature_cols: list,
        data_source: str = "real",
        shap_path: str = None,
    ):
        """
        Registers the best trained model for a given horizon.
        Stores feature schema alongside the model to prevent train/serve skew.
        Marks data_source so dashboards and engineers know whether synthetic data was used.

        Raises on any failure — never silently succeeds.
        """
        model_dir = f"artifacts/{model_type}_{horizon}"
        os.makedirs(model_dir, exist_ok=True)

        # 1. Save model artifact
        if model_type == "tensorflow_nn":
            model.model.save(f"{model_dir}/model.keras")
        else:
            joblib.dump(model, f"{model_dir}/model.pkl")

        # 2. Save feature schema alongside model (critical for train/serve parity)
        feature_schema = {
            "feature_cols": feature_cols,
            "data_source": data_source,
            "feature_group_name": settings.FEATURE_GROUP_NAME,
            "feature_group_version": settings.FEATURE_GROUP_VERSION,
            "data_source_version": settings.DATA_SOURCE_VERSION,
        }
        with open(f"{model_dir}/feature_schema.json", "w") as f:
            json.dump(feature_schema, f, indent=2)

        # 3. Copy SHAP plot if available
        if shap_path and os.path.exists(shap_path):
            shutil.copy(shap_path, f"{model_dir}/shap_summary.png")

        # 4. Hopsworks 'metrics' argument strictly accepts ONLY numbers (float/int).
        # We must ensure no strings are passed here to avoid Avro/Hopsworks ValueError.
        numeric_metrics = {
            "rmse": float(metrics.get("rmse", 0)),
            "mae":  float(metrics.get("mae", 0)),
            "r2":   float(metrics.get("r2", 0)),
            "feature_count": float(len(feature_cols))
        }

        if data_source in ("synthetic", "mixed"):
            logger.warning(
                "Registering model for %s trained on %s data. "
                "data_source='%s' stored in model metadata.",
                horizon, data_source, data_source
            )

        # 5. Register in Hopsworks Model Registry
        model_name = f"aqi_model_{horizon}"
        logger.info("Registering %s to Hopsworks Model Registry...", model_name)

        hw_model = self.mr.python.create_model(
            name=model_name,
            metrics=numeric_metrics,
            description=(
                f"{model_type} model for {horizon} AQI forecast | "
                f"data_source={data_source} | "
                f"schema_v={settings.DATA_SOURCE_VERSION} | "
                f"trained {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
        )
        hw_model.save(model_dir)

        logger.info(
            "Registered: %s v%s | RMSE=%.2f | MAE=%.2f | R2=%.3f | data_source=%s",
            hw_model.name, hw_model.version,
            metrics.get("rmse", 0), metrics.get("mae", 0), metrics.get("r2", 0),
            data_source
        )
        return hw_model
