import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_pipeline.feature_engineering import compute_features

def test_compute_features():
    # Create synthetic data for 4 hours
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    
    data = {
        "city_id": ["test_city"] * 4,
        "timestamp": [
            base_time,
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=2),
            base_time + timedelta(hours=3),
        ],
        "aqi": [100.0, 110.0, 120.0, 130.0],
    }
    df = pd.DataFrame(data)
    
    # Run feature engineering
    result = compute_features(df)
    
    # Assert time features
    assert result["hour"].iloc[0] == 12
    assert result["day_of_week"].iloc[0] == 6 # Sunday
    assert result["is_weekend"].iloc[0] == 1
    
    # Assert lags
    assert pd.isna(result["aqi_lag_1h"].iloc[0])
    assert result["aqi_lag_1h"].iloc[1] == 100.0
    assert result["aqi_lag_1h"].iloc[2] == 110.0
    
    # Assert change rate
    # aqi_t = 110, lag_1h = 100 => rate = 10 / 100 = 0.1
    assert pytest.approx(result["aqi_change_rate"].iloc[1], 0.001) == 0.1
    # aqi_t = 120, lag_1h = 110 => rate = 10 / 110 = 0.0909
    assert pytest.approx(result["aqi_change_rate"].iloc[2], 0.001) == 0.0909
    
    # Assert rolling means
    # 3h rolling mean at idx 0: [100] -> 100
    assert result["aqi_rolling_3h"].iloc[0] == 100.0
    # 3h rolling mean at idx 1: [100, 110] -> 105
    assert result["aqi_rolling_3h"].iloc[1] == 105.0
    # 3h rolling mean at idx 2: [100, 110, 120] -> 110
    assert result["aqi_rolling_3h"].iloc[2] == 110.0
    # 3h rolling mean at idx 3: [110, 120, 130] -> 120 (drops the 100)
    assert result["aqi_rolling_3h"].iloc[3] == 120.0
