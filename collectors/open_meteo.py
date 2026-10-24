"""
open_meteo.py

Retrieves current and forecast outdoor weather and air-quality data for  
home location and selected locations around London.

"""

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

FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"


# Minimal WMO weather-code -> text map (add more if you like).
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 61: "light rain",
    63: "rain", 65: "heavy rain", 71: "snow", 80: "rain showers",
    95: "thunderstorm",
}


def get_current_weather(lat, lon):
    response = requests.get(
        FORECAST_BASE,
        params={
            "latitude": lat,
            "longitude": lon,
            # ask for exactly the current variables we want:
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "surface_pressure,wind_speed_10m,wind_direction_10m,"
                       "cloud_cover,uv_index,weather_code",
            "wind_speed_unit": "ms",   # so it matches our wind_speed_mps column
        },
        timeout=30,
    )
    response.raise_for_status()
    cur = response.json()["current"]   # all current readings live here

    # Return the SAME keys as before, so storage.save_weather() is unchanged.
    return {
        "temperature_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity_percent": cur.get("relative_humidity_2m"),
        "pressure_hpa": cur.get("surface_pressure"),
        "wind_speed_mps": cur.get("wind_speed_10m"),
        "wind_direction_deg": cur.get("wind_direction_10m"),
        "cloud_cover_percent": cur.get("cloud_cover"),
        "uv_index": cur.get("uv_index"),
        "weather": _WMO.get(cur.get("weather_code"), str(cur.get("weather_code"))),
        "weather_time_unix": cur.get("time"),   # Open-Meteo gives an ISO string
    }


def get_air_quality(lat, lon):
    response = requests.get(
        AIR_QUALITY_BASE,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "pm2_5,pm10,nitrogen_dioxide,ozone,"
                       "carbon_monoxide,sulphur_dioxide,ammonia,european_aqi",
        },
        timeout=30,
    )
    response.raise_for_status()
    cur = response.json()["current"]

    # Same keys as before. Note: aqi is now european_aqi (0..100+ scale).
    return {
        "aqi": cur.get("european_aqi"),
        "pm2_5": cur.get("pm2_5"),
        "pm10": cur.get("pm10"),
        "no2": cur.get("nitrogen_dioxide"),
        "o3": cur.get("ozone"),
        "co": cur.get("carbon_monoxide"),
        "so2": cur.get("sulphur_dioxide"),
        "nh3": cur.get("ammonia"),
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