import os
import csv 
import io 
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import requests
from pathlib import Path
import json
from animal import get_latest_animal_location

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

ONE_CALL_BASE = "https://api.openweathermap.org/data/2.5/onecall"
AIR_POLLUTION_BASE = "https://api.openweathermap.org/data/2.5/air_pollution"


def get_current_weather(lat, lon): 
    response = requests.get(
        ONE_CALL_BASE, 
        params = { 
            "lat": lat,
            "lon": lon, 
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    current = data["current"]

    return {
        "temperature_c": data["main"]["temp"],
        "feels_like_c": current["feels_like"],
        "humidity_percent": current["humidity"],
        "pressure_hpa": current["pressure"],
        "wind_speed_mps": current["wind_speed"],
        "wind_direction_deg": current["wind_deg"],
        "cloud_cover_percent": current["clouds"],
        "uv_index": current.get("uvi"),
        "weather": current["weather"][0]["description"],
        "weather_time_unix": current["dt"],
    }

def get_air_quality(lat, lon):
    response = requests.get(
        AIR_POLLUTION_BASE,
        params={
            "lat": lat, 
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["list"][0]

    return { 
        "aqi": data["main"]["aqi"],
        "pm2_5": data["components"]["pm2_5"],
        "pm10": data["components"]["pm10"],
        "no2": data["components"]["no2"],
        "o3": data["components"]["o3"],
        "co": data["components"]["co"],
        "so2": data["components"]["so2"],
        "nh3": data["components"]["nh3"],
    }

def get_animal_environment(): 
    animal_info = get_latest_animal_location()

    lat = animal_info["latitude"]
    lon = animal_info["longitude"]

    weather = get_current_weather(lat, lon)
    air_quality = get_air_quality(lat, lon)

    return { 
        "animal": animal_info,
        "weather": weather, 
        "air_quality": air_quality, 
    }


if __name__ == "__main__":
    result = get_animal_environment()
    print(json.dumps(result, indent=2))