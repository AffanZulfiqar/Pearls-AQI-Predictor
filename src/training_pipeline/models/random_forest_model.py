"""
Random Forest Regressor for AQI forecasting.

Uses RandomizedSearchCV for efficient hyperparameter tuning across
n_estimators, max_depth, min_samples_split, and max_features.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import logging

logger = logging.getLogger(__name__)


def train(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor with randomized hyperparameter search.
    
    Args:
        X: Training features [n_samples, n_features]
        y: Target AQI values [n_samples]
    
    Returns:
        Best fitted RandomForestRegressor found by RandomizedSearchCV
    """
    param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [None, 8, 16, 24, 32],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":     ["sqrt", "log2", 0.5, 0.8],
    }

    base = RandomForestRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=20,                         # 20 random combos
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X, y)

    best = search.best_params_
    best_rmse = -search.best_score_
    logger.info(
        "RandomForest best params: n_estimators=%s, max_depth=%s, CV RMSE=%.3f",
        best["n_estimators"], best["max_depth"], best_rmse
    )

    return search.best_estimator_


def predict(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
