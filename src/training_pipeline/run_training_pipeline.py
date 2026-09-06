import logging
import json
import os
from src.config import settings
from src.config.settings import validate_env
from src.config.logging_config import setup_logging
from src.training_pipeline.dataset_builder import DatasetBuilder
from src.training_pipeline.models import ridge_model
from src.training_pipeline.models import random_forest_model
from src.training_pipeline.models import tf_model
from src.training_pipeline.evaluate import evaluate_model
from src.training_pipeline.explain import generate_shap_artifacts
from src.training_pipeline.model_registry import ModelRegistryWriter

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    logger.info("Starting training pipeline...")

    validate_env()

    # Initialize — raises on failure
    try:
        builder = DatasetBuilder()
        registry = ModelRegistryWriter()
    except Exception:
        logger.exception("Failed to initialize Hopsworks clients")
        raise

    train_df, val_df, test_df, feature_cols, data_source = builder.build_training_dataset()

    if len(train_df) == 0:
        raise RuntimeError(
            "Training dataset is empty after build. "
            "Ensure backfill has been run and feature group has data."
        )

    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    models_to_train = {
        "ridge": ridge_model,
        "random_forest": random_forest_model,
        "tensorflow_nn": tf_model,
    }

    all_metrics = {}

    for horizon in settings.FORECAST_HORIZONS:
        logger.info("=== Training models for %s horizon ===", horizon)
        target_col = f"target_aqi_{horizon}"

        y_train = train_df[target_col].values
        y_val   = val_df[target_col].values
        y_test  = test_df[target_col].values

        best_model      = None
        best_model_name = None
        best_rmse       = float("inf")
        best_metrics    = None
        horizon_metrics = {}

        for name, module in models_to_train.items():
            logger.info("Training %s for %s...", name, horizon)
            try:
                if name == "tensorflow_nn":
                    model = module.train(X_train, y_train, X_val, y_val)
                else:
                    model = module.train(X_train, y_train)

                y_pred  = module.predict(model, X_test)
                metrics = evaluate_model(y_test, y_pred)
                logger.info("%s metrics for %s: %s", name, horizon, metrics)
                horizon_metrics[name] = metrics

                if metrics["rmse"] < best_rmse or (
                    metrics["rmse"] == best_rmse
                    and best_metrics
                    and metrics["r2"] > best_metrics["r2"]
                ):
                    best_rmse       = metrics["rmse"]
                    best_model      = model
                    best_model_name = name
                    best_metrics    = metrics

            except Exception:
                logger.exception("Failed to train %s for %s", name, horizon)
                raise

        all_metrics[horizon] = horizon_metrics
        logger.info(
            "Best model for %s: %s (RMSE=%.2f, MAE=%.2f, R2=%.3f)",
            horizon, best_model_name,
            best_metrics["rmse"], best_metrics["mae"], best_metrics["r2"]
        )

        # Generate SHAP artifacts
        logger.info("Generating SHAP explanations for %s...", horizon)
        shap_path = generate_shap_artifacts(
            best_model, X_train, X_test, feature_cols, best_model_name,
            output_dir="artifacts"
        )

        # Register best model with full metadata — raises on failure
        registry.register_model(
            model=best_model,
            model_type=best_model_name,
            horizon=horizon,
            metrics=best_metrics,
            feature_cols=feature_cols,
            data_source=data_source,
            shap_path=shap_path,
        )

    # Save all-horizons metrics artifact
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Training pipeline finished. All models registered in Hopsworks.")


if __name__ == "__main__":
    main()
