from flask import Flask, request, jsonify
from flask_cors import CORS
from src.inference.predictor import Predictor
from src.inference.alerts import check_alert, get_aqi_category
import logging
from src.config.logging_config import setup_logging
import traceback

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Streamlit

predictor = Predictor()

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    
    Response: {"status": "healthy"}
    """
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["GET"])
def predict():
    """
    Generate AQI forecast for a city.
    
    Request:  GET /predict?city=islamabad
    Response: {
        "city": "islamabad",
        "forecasts": {
            "24h": {"value": 85.2, "category": "Moderate"},
            "48h": {"value": 92.1, "category": "Moderate"},
            "72h": {"value": 78.5, "category": "Moderate"}
        },
        "alert": {"alert": false}
    }
    """
    try:
        city = request.args.get("city")
        if not city:
            return jsonify({"error": "City parameter is required"}), 400
            
        predictions = predictor.predict(city)
        if not predictions:
            return jsonify({"error": "Data or models not available for this city"}), 404
            
        forecasts = {}
        for h, v in predictions.items():
            forecasts[h] = {
                "value": round(v, 2),
                "category": get_aqi_category(v)
            }
            
        alert = check_alert(predictions)
        
        return jsonify({
            "city": city,
            "forecasts": forecasts,
            "alert": alert
        }), 200
    except Exception as e:
        logger.error(f"Error serving /predict: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/explain", methods=["GET"])
def explain():
    """
    Get SHAP-based feature importance for the current prediction.
    
    Request:  GET /explain?city=islamabad
    Response: {
        "city": "islamabad",
        "explanations": {
            "24h": [
                {"feature": "aqi", "importance": 45.2},
                {"feature": "pm2_5", "importance": 12.8},
                ...
            ],
            "48h": [...],
            "72h": [...]
        }
    }
    """
    try:
        city = request.args.get("city")
        if not city:
            return jsonify({"error": "City parameter is required"}), 400
        
        explanations = predictor.explain(city)
        if not explanations:
            return jsonify({"error": "Explanations not available for this city"}), 404
        
        return jsonify({
            "city": city,
            "explanations": explanations
        }), 200
    except Exception as e:
        logger.error(f"Error serving /explain: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
