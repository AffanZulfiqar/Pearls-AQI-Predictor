import os
import sys
import argparse
from dotenv import load_dotenv
from src.config import settings
from src.data_ingestion.aqicn_client import AQICNClient
from src.data_ingestion.openweather_client import OpenWeatherClient

def get_aqi_category(aqi: float) -> tuple:
    if aqi <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable; however, sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Fetch live AQI and weather for a city.")
    parser.add_argument("--city", type=str, default="islamabad", help="City name or id (e.g. islamabad, karachi, london)")
    parser.add_argument("--aqicn-key", type=str, default=None, help="AQICN (waqi.info) API token (optional if set in .env)")
    parser.add_argument("--openweather-key", type=str, default=None, help="OpenWeather API key (optional if set in .env)")
    
    args = parser.parse_args()
    
    city_target = args.city.lower().strip()
    city_info = next((c for c in settings.CITIES if c["id"] == city_target or c["name"].lower() == city_target), None)
    
    if not city_info:
        if "islamabad" in city_target:
            city_info = {"id": "islamabad", "name": "Islamabad", "lat": 33.6844, "lon": 73.0479}
        else:
            print(f"[Error] City '{args.city}' not recognized. Available: {[c['name'] for c in settings.CITIES]}")
            sys.exit(1)
            
    aqicn_key = args.aqicn_key or os.getenv("AQICN_API_KEY")
    ow_key = args.openweather_key or os.getenv("OPENWEATHER_API_KEY")
    
    print("=" * 60)
    print(f"  LIVE AIR QUALITY & WEATHER REPORT: {city_info['name'].upper()}")
    print("=" * 60)
    
    # 1. Fetch AQI data
    if aqicn_key and aqicn_key != "your_aqicn_api_key_here":
        client = AQICNClient(api_key=aqicn_key)
        aqi_data = client.get_city_data(city_info["name"], city_info["lat"], city_info["lon"])
        if aqi_data:
            aqi = aqi_data.get("aqi", 0)
            category, advice = get_aqi_category(aqi)
            print(f"\n[Air Quality Index]")
            print(f"  AQI Level    : {aqi:.1f} ({category})")
            print(f"  Health Advice: {advice}")
            print(f"\n[Pollutants]")
            print(f"  PM2.5        : {aqi_data.get('pm2_5')} ug/m3")
            print(f"  PM10         : {aqi_data.get('pm10')} ug/m3")
            print(f"  NO2          : {aqi_data.get('no2')} ppb")
            print(f"  SO2          : {aqi_data.get('so2')} ppb")
            print(f"  CO           : {aqi_data.get('co')} ppm")
            print(f"  O3           : {aqi_data.get('o3')} ppb")
        else:
            print(f"\n[!] Unable to retrieve AQICN data for {city_info['name']}. Please verify your AQICN key.")
    else:
        print("\n[!] AQICN_API_KEY is not set. Provide --aqicn-key <KEY> or set AQICN_API_KEY in .env.")
        
    # 2. Fetch Weather data
    if ow_key and ow_key != "your_openweather_api_key_here":
        ow_client = OpenWeatherClient(api_key=ow_key)
        weather_data = ow_client.get_city_weather(city_info["name"], city_info["lat"], city_info["lon"])
        if weather_data:
            print(f"\n[Meteorological Conditions]")
            print(f"  Temperature  : {weather_data.get('temperature')} C")
            print(f"  Humidity     : {weather_data.get('humidity')} %")
            print(f"  Wind Speed   : {weather_data.get('wind_speed')} m/s")
            print(f"  Pressure     : {weather_data.get('pressure')} hPa")
            print(f"  Precipitation: {weather_data.get('precipitation')} mm")
        else:
            print(f"\n[!] Unable to retrieve OpenWeather data for {city_info['name']}. Please verify your OpenWeather key.")
    else:
        print("\n[!] OPENWEATHER_API_KEY is not set. Provide --openweather-key <KEY> or set OPENWEATHER_API_KEY in .env.")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
