import pytest
import numpy as np
from src.training_pipeline.evaluate import evaluate_model


def test_evaluate_model():
    y_true = np.array([100, 150, 200])
    y_pred = np.array([110, 140, 210])
    
    metrics = evaluate_model(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_evaluate_model_perfect():
    """Perfect predictions should give RMSE=0, MAE=0, R²=1."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_ridge_model_train_predict():
    """Test Ridge model trains and produces predictions of correct shape."""
    from src.training_pipeline.models import ridge_model
    
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = X_train[:, 0] * 50 + np.random.normal(0, 5, 100)
    X_test = np.random.rand(20, 5)
    
    model = ridge_model.train(X_train, y_train)
    y_pred = ridge_model.predict(model, X_test)
    
    assert y_pred.shape == (20,)
    assert not np.any(np.isnan(y_pred))


def test_random_forest_train_predict():
    """Test Random Forest model trains and produces predictions of correct shape."""
    from src.training_pipeline.models import random_forest_model
    
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = X_train[:, 0] * 50 + np.random.normal(0, 5, 100)
    X_test = np.random.rand(20, 5)
    
    model = random_forest_model.train(X_train, y_train)
    y_pred = random_forest_model.predict(model, X_test)
    
    assert y_pred.shape == (20,)
    assert not np.any(np.isnan(y_pred))


def test_tf_model_train_predict():
    """Test TensorFlow model trains and produces predictions of correct shape."""
    from src.training_pipeline.models import tf_model
    
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = X_train[:, 0] * 50 + np.random.normal(0, 5, 100)
    X_val = np.random.rand(20, 5)
    y_val = X_val[:, 0] * 50 + np.random.normal(0, 5, 20)
    X_test = np.random.rand(20, 5)
    
    model = tf_model.train(X_train, y_train, X_val, y_val)
    y_pred = tf_model.predict(model, X_test)
    
    assert y_pred.shape == (20,)
    assert not np.any(np.isnan(y_pred))


def test_tf_model_train_without_validation():
    """TF model should also work without validation data."""
    from src.training_pipeline.models import tf_model
    
    np.random.seed(42)
    X_train = np.random.rand(50, 3)
    y_train = np.random.rand(50)
    
    model = tf_model.train(X_train, y_train)
    y_pred = tf_model.predict(model, X_train[:5])
    
    assert y_pred.shape == (5,)
