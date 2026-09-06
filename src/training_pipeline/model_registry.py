import hopsworks
from src.config import settings
import logging
import os
import joblib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ModelRegistryWriter:
    def __init__(self):
        self.project = hopsworks.login(
            project=settings.HOPSWORKS_PROJECT_NAME,
            api_key_value=settings.HOPSWORKS_API_KEY
        )
        self.mr = self.project.get_model_registry()
        
    def register_model(self, model, model_type, horizon, metrics, shap_path=None):
        """
        Registers a trained model to the Hopsworks Model Registry.
        Includes full metadata per spec §4.3: model_type, training_date,
        feature_schema_version, metrics, and SHAP artifact reference.
        """
        model_dir = f"artifacts/{model_type}_{horizon}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if model_type == "tensorflow_nn":
            model.model.save(f"{model_dir}/model.keras")
        else:
            joblib.dump(model, f"{model_dir}/model.pkl")
            
        # Copy SHAP plot into model dir if available
        if shap_path and os.path.exists(shap_path):
            import shutil
            shutil.copy(shap_path, f"{model_dir}/shap_summary.png")
        
        # Build metrics with full metadata per spec §4.3
        full_metrics = {
            **metrics,
            "model_type": model_type,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "feature_schema_version": settings.DATA_SOURCE_VERSION,
            "forecast_horizon": horizon,
        }
        
        # Create model in Hopsworks
        logger.info(f"Registering model for {horizon} to Hopsworks...")
        hw_model = self.mr.python.create_model(
            name=f"aqi_model_{horizon}",
            metrics=full_metrics,
            description=f"{model_type} model for {horizon} AQI forecast (trained {datetime.now(timezone.utc).strftime('%Y-%m-%d')})"
        )
        
        hw_model.save(model_dir)
        logger.info(f"Successfully registered model {hw_model.name} version {hw_model.version}")
        return hw_model
