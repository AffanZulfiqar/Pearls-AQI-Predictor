"""
TensorFlow Neural Network for AQI forecasting.

Architecture:
  - Dense(128) → BatchNorm → Dropout(0.3)
  - Dense(64)  → BatchNorm → Dropout(0.2)
  - Dense(32)  → Dense(1)

Training:
  - Adam optimizer with ReduceLROnPlateau
  - EarlyStopping (patience=10, restore best weights)
  - Up to 100 epochs, batch size 64

The model and scaler are bundled into TFModelWrapper so that:
  1. Inference always applies the same scaler used at training time.
  2. joblib.dump/load works cleanly (wrapper stores scaler state).
"""
import numpy as np
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TFModelWrapper:
    """Wraps Keras model + fitted StandardScaler for clean save/load."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        X_scaled = self.scaler.fit_transform(X)

        n_features = X_scaled.shape[1]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            # Block 1
            tf.keras.layers.Dense(128, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            # Block 2
            tf.keras.layers.Dense(64, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            # Block 3
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",        # robust to AQI outliers vs MSE
            metrics=["mae"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled   = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)

        history = self.model.fit(
            X_scaled, y,
            epochs=100,
            batch_size=64,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        final_loss = history.history["loss"][-1]
        epochs_run = len(history.history["loss"])
        logger.info(
            "TF model trained: %d epochs, final loss=%.4f", epochs_run, final_loss
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()


def train(X: np.ndarray, y: np.ndarray,
          X_val: np.ndarray = None, y_val: np.ndarray = None) -> TFModelWrapper:
    """
    Trains the TF neural network.
    
    Args:
        X: Training features
        y: Training targets
        X_val: Validation features (optional, used for EarlyStopping)
        y_val: Validation targets (optional)
    
    Returns:
        Fitted TFModelWrapper instance
    """
    wrapper = TFModelWrapper()
    wrapper.fit(X, y, X_val, y_val)
    return wrapper


def predict(model: TFModelWrapper, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
