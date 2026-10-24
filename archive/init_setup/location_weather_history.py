"""
For every day/location in animal_location_history.csv, fetch that day's weather
and air quality AT that location and write one merged row per day.

Uses Open-Meteo's ARCHIVE endpoints (past data), asks for HOURLY values and
aggregates them to daily numbers. Column names match collection/weather.py.

Safe to re-run: days already present in the output are skipped.

Run:  python init_setup/location_weather_history.py
Out:  init_setup/location_weather_history.csv
"""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "collection"))
from weather import _WMO

IN = HERE / "animal_location_history.csv"
OUT = HERE / "location_weather_history.csv"

ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
AIR_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOURLY_WEATHER = ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                  "surface_pressure,wind_speed_10m,wind_direction_10m,"
                  "cloud_cover,weather_code,precipitation")
HOURLY_AIR = ("pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,"
              "sulphur_dioxide,ammonia,european_aqi")

FIELDS = [
    "date", "lat", "lon", "n_fixes", "daily_path_km",
    "temp_c", "temp_min_c", "temp_max_c", "feels_like_c",
    "humidity_percent", "pressure_hpa",
    "wind_speed_mps", "wind_speed_max_mps", "wind_direction_deg",
    "cloud_cover_percent", "precipitation_mm", "weather",
    "aqi", "aqi_max", "pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3",
]


def clean(values):
    return [v for v in (values or []) if v is not None]


def mean(values):
    vals = clean(values)
    return round(sum(vals) / len(vals), 2) if vals else None


def lo(values):
    vals = clean(values)
    return round(min(vals), 2) if vals else None


def hi(values):
    vals = clean(values)
    return round(max(vals), 2) if vals else None


def total(values):
    vals = clean(values)
    return round(sum(vals), 2) if vals else None


def fetch_weather(lat, lon, date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "hourly": HOURLY_WEATHER,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_BASE, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    winds = hourly.get("wind_speed_10m") or []
    dirs = hourly.get("wind_direction_10m") or []
    codes = clean(hourly.get("weather_code"))

    # wind direction is only meaningful alongside a speed, so report the
    # direction recorded at the windiest hour of the day
    gust_dir = None
    if clean(winds):
        peak = max(range(len(winds)), key=lambda i: (winds[i] is not None, winds[i]))
        if peak < len(dirs):
            gust_dir = dirs[peak]

    code = Counter(codes).most_common(1)[0][0] if codes else None

    return {
        "temp_c": mean(hourly.get("temperature_2m")),
        "temp_min_c": lo(hourly.get("temperature_2m")),
        "temp_max_c": hi(hourly.get("temperature_2m")),
        "feels_like_c": mean(hourly.get("apparent_temperature")),
        "humidity_percent": mean(hourly.get("relative_humidity_2m")),
        "pressure_hpa": mean(hourly.get("surface_pressure")),
        "wind_speed_mps": mean(winds),
        "wind_speed_max_mps": hi(winds),
        "wind_direction_deg": gust_dir,
        "cloud_cover_percent": mean(hourly.get("cloud_cover")),
        "precipitation_mm": total(hourly.get("precipitation")),
        "weather": _WMO.get(code, str(code)) if code is not None else None,
    }


def fetch_air(lat, lon, date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "hourly": HOURLY_AIR,
        "timezone": "UTC",
    }
    response = requests.get(AIR_BASE, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    return {
        "aqi": mean(hourly.get("european_aqi")),
        "aqi_max": hi(hourly.get("european_aqi")),
        "pm2_5": mean(hourly.get("pm2_5")),
        "pm10": mean(hourly.get("pm10")),
        "no2": mean(hourly.get("nitrogen_dioxide")),
        "o3": mean(hourly.get("ozone")),
        "co": mean(hourly.get("carbon_monoxide")),
        "so2": mean(hourly.get("sulphur_dioxide")),
        "nh3": mean(hourly.get("ammonia")),
    }


def already_done():
    if not OUT.exists():
        return set()
    with open(OUT, newline="", encoding="utf-8") as f:
        return {row["date"] for row in csv.DictReader(f)}


def main():
    with open(IN, newline="", encoding="utf-8") as f:
        days = list(csv.DictReader(f))
    print(f"{len(days)} days to enrich from {IN.name}")

    done = already_done()
    if done:
        print(f"{len(done)} already done, skipping those")

    new_file = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for i, day in enumerate(days, 1):
            if day["date"] in done:
                continue
            lat, lon, date = float(day["lat"]), float(day["lon"]), day["date"]
            try:
                row = {
                    "date": date, "lat": lat, "lon": lon,
                    "n_fixes": day["n_fixes"],
                    "daily_path_km": day["daily_path_km"],
                }
                row.update(fetch_weather(lat, lon, date))
                row.update(fetch_air(lat, lon, date))
                writer.writerow(row)
                f.flush()
                print(f"[{i}/{len(days)}] {date}  {row['temp_c']}C  aqi {row['aqi']}")
            except Exception as e:
                print(f"[{i}/{len(days)}] {date}  FAILED: {e}")
            time.sleep(0.3)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
