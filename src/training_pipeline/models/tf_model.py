import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

class TFModelWrapper:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Build callbacks
        callbacks = []
        validation_data = None
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ))
        
        self.model.fit(
            X_scaled, y,
            epochs=50,  # increased max since early stopping will prevent overfit
            batch_size=32,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

def train(X, y, X_val=None, y_val=None):
    wrapper = TFModelWrapper()
    wrapper.fit(X, y, X_val, y_val)
    return wrapper

def predict(model, X):
    return model.predict(X)
