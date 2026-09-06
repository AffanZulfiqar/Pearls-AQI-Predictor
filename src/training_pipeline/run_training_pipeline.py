import logging
import json
import os
from src.config import settings
from src.config.logging_config import setup_logging
from src.training_pipeline.dataset_builder import DatasetBuilder
from src.training_pipeline.models import ridge_model, random_forest_model, tf_model
from src.training_pipeline.evaluate import evaluate_model
from src.training_pipeline.explain import generate_shap_artifacts
from src.training_pipeline.model_registry import ModelRegistryWriter

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    logger.info("Starting training pipeline...")
    
    try:
        builder = DatasetBuilder()
        registry = ModelRegistryWriter()
    except Exception as e:
        logger.error(f"Failed to initialize Hopsworks clients: {e}")
        return
        
    train_df, val_df, test_df, feature_cols = builder.build_training_dataset()
    if len(train_df) == 0:
        logger.error("Training dataset is empty. Run backfill first.")
        return
        
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    models_to_train = {
        "ridge": ridge_model,
        "random_forest": random_forest_model,
        "tensorflow_nn": tf_model
    }
    
    all_metrics = {}
    
    for horizon in settings.FORECAST_HORIZONS:
        logger.info(f"=== Training models for {horizon} horizon ===")
        target_col = f"target_aqi_{horizon}"
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        best_model = None
        best_model_name = None
        best_rmse = float('inf')
        best_metrics = None
        
        horizon_metrics = {}
        
        for name, module in models_to_train.items():
            logger.info(f"Training {name}...")
            
            # TF model accepts validation data for early stopping
            if name == "tensorflow_nn":
                model = module.train(X_train, y_train, X_val, y_val)
            else:
                model = module.train(X_train, y_train)
            
            y_pred = module.predict(model, X_test)
            metrics = evaluate_model(y_test, y_pred)
            
            logger.info(f"{name} metrics: {metrics}")
            horizon_metrics[name] = metrics
            
            # Best model selection: lowest RMSE, tie-break by highest R²
            if metrics["rmse"] < best_rmse or (
                metrics["rmse"] == best_rmse and 
                best_metrics and metrics["r2"] > best_metrics["r2"]
            ):
                best_rmse = metrics["rmse"]
                best_model = model
                best_model_name = name
                best_metrics = metrics
                
        all_metrics[horizon] = horizon_metrics
        logger.info(f"Best model for {horizon}: {best_model_name} (RMSE: {best_rmse})")
        
        # Generate SHAP
        logger.info("Generating SHAP explanations...")
        shap_path = generate_shap_artifacts(
            best_model, X_train, X_test, feature_cols, best_model_name, output_dir="artifacts"
        )
        
        # Register best model
        registry.register_model(best_model, best_model_name, horizon, best_metrics, shap_path)
        
    # Save all metrics artifact
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    logger.info("Training pipeline finished.")

if __name__ == "__main__":
    main()
