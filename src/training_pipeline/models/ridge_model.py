"""
Ridge Regression model for AQI forecasting.

Uses GridSearchCV to tune regularization strength over the training set.
Includes StandardScaler inside a Pipeline to prevent data leakage.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Trains a Ridge Regression model with hyperparameter tuning.
    
    Args:
        X: Training features [n_samples, n_features]
        y: Target AQI values [n_samples]
    
    Returns:
        Fitted sklearn Pipeline (StandardScaler + Ridge)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge())
    ])

    param_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}

    # Time-series-safe CV: use 5 folds, no shuffle (preserves temporal order)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)

    best_alpha = grid.best_params_["ridge__alpha"]
    best_rmse  = -grid.best_score_
    logger.info("Ridge best alpha=%.1f, CV RMSE=%.3f", best_alpha, best_rmse)

    return grid.best_estimator_


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
