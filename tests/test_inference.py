import pytest
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.alerts import get_aqi_category, check_alert


def test_get_aqi_category():
    assert get_aqi_category(40) == "Good"
    assert get_aqi_category(50) == "Good"
    assert get_aqi_category(51) == "Moderate"
    assert get_aqi_category(100) == "Moderate"
    assert get_aqi_category(120) == "Unhealthy for Sensitive Groups"
    assert get_aqi_category(150) == "Unhealthy for Sensitive Groups"
    assert get_aqi_category(180) == "Unhealthy"
    assert get_aqi_category(250) == "Very Unhealthy"
    assert get_aqi_category(350) == "Hazardous"


def test_check_alert():
    predictions = {"24h": 100, "48h": 160, "72h": 90}
    alert = check_alert(predictions)
    assert alert["alert"] == True
    assert alert["horizon"] == "48h"
    
    safe_predictions = {"24h": 100, "48h": 100, "72h": 90}
    assert check_alert(safe_predictions)["alert"] == False


def test_check_alert_all_hazardous():
    """First hazardous horizon should be reported."""
    predictions = {"24h": 200, "48h": 300, "72h": 400}
    alert = check_alert(predictions)
    assert alert["alert"] == True
    assert alert["horizon"] == "24h"


def test_flask_app_endpoints():
    """Test Flask API endpoints using the test client (no live server needed)."""
    # Import the Flask app — this will try to init Predictor which needs Hopsworks.
    # We mock the Predictor to avoid needing real credentials.
    from unittest.mock import patch, MagicMock
    
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "24h": 85.0,
        "48h": 92.0,
        "72h": 78.0
    }
    mock_predictor.explain.return_value = {
        "24h": [{"feature": "aqi", "importance": 45.0}],
        "48h": [{"feature": "pm2_5", "importance": 30.0}],
        "72h": [{"feature": "temperature", "importance": 20.0}]
    }
    
    with patch("src.inference.api.predictor", mock_predictor):
        from src.inference.api import app
        
        client = app.test_client()
        
        # Test /health
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        
        # Test /predict with city
        response = client.get("/predict?city=islamabad")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["city"] == "islamabad"
        assert "forecasts" in data
        assert "24h" in data["forecasts"]
        assert data["forecasts"]["24h"]["value"] == 85.0
        assert data["forecasts"]["24h"]["category"] == "Moderate"
        assert data["alert"]["alert"] == False
        
        # Test /predict without city
        response = client.get("/predict")
        assert response.status_code == 400
        
        # Test /explain
        response = client.get("/explain?city=islamabad")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "explanations" in data
        assert data["explanations"]["24h"][0]["feature"] == "aqi"
        
        # Test /explain without city
        response = client.get("/explain")
        assert response.status_code == 400
