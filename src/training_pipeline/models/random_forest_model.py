from sklearn.ensemble import RandomForestRegressor

def train(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)
