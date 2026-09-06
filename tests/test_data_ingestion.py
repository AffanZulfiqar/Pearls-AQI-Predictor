import pytest
from unittest.mock import patch, MagicMock
from src.data_ingestion.aqicn_client import AQICNClient
from src.data_ingestion.openweather_client import OpenWeatherClient

@pytest.fixture
def aqicn_client():
    return AQICNClient(api_key="test_key")

@pytest.fixture
def openweather_client():
    return OpenWeatherClient(api_key="test_key")

@patch("src.data_ingestion.aqicn_client.requests.get")
def test_aqicn_client_success(mock_get, aqicn_client):
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "ok",
        "data": {
            "aqi": 120,
            "iaqi": {
                "pm25": {"v": 45.2},
                "no2": {"v": 12.1}
            }
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Test
    result = aqicn_client.get_city_data("Karachi", 24.8607, 67.0011)

    # Asserts
    assert result is not None
    assert result["aqi"] == 120.0
    assert result["pm2_5"] == 45.2
    assert result["no2"] == 12.1
    assert result["pm10"] is None
    
    mock_get.assert_called_once()
    assert "geo:24.8607;67.0011" in mock_get.call_args[0][0]

@patch("src.data_ingestion.aqicn_client.requests.get")
def test_aqicn_client_api_error(mock_get, aqicn_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "error",
        "data": "Unknown city"
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = aqicn_client.get_city_data("FakeCity", 0.0, 0.0)
    assert result is None

@patch("src.data_ingestion.openweather_client.requests.get")
def test_openweather_client_success(mock_get, openweather_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {
            "temp": 28.5,
            "humidity": 65,
            "pressure": 1012
        },
        "wind": {
            "speed": 5.2
        },
        "rain": {
            "1h": 1.2
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = openweather_client.get_city_weather("Karachi", 24.8607, 67.0011)

    assert result is not None
    assert result["temperature"] == 28.5
    assert result["humidity"] == 65.0
    assert result["wind_speed"] == 5.2
    assert result["pressure"] == 1012.0
    assert result["precipitation"] == 1.2
    
    mock_get.assert_called_once()
    assert mock_get.call_args[1]["params"]["lat"] == 24.8607

@patch("src.data_ingestion.openweather_client.requests.get")
def test_openweather_client_network_error(mock_get, openweather_client):
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection Failed")

    result = openweather_client.get_city_weather("Karachi", 24.8607, 67.0011)
    assert result is None
