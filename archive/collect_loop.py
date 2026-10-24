"""
collect_loop.py — runs the data collection automatically on a loop.

Every INTERVAL seconds it collects THREE streams, each in its own try/except so
one failure never kills the loop or the other streams:
  1. Marina's latest position        -> animal_position
  2. weather + air quality there      -> weather_reading
  3. YOUR HRV from the Ultrahuman ring -> heart_reading

HOW TO RUN (from the project root):
    python collect_loop.py
Leave the terminal open (or a standalone PowerShell window so you can close
VS Code). A line prints each cycle. Press Ctrl+C to stop.
"""

import os
import sys
import time

# --- make our modules importable no matter where python is launched from ----
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "collection"))
sys.path.insert(0, os.path.join(ROOT, "core"))

from animal import get_latest_animal_location              # noqa: E402
from weather import get_current_weather, get_air_quality   # noqa: E402
from heart import (get_daily_metrics, extract_hrv_series,   # noqa: E402
                   extract_hr_series, extract_respiratory_series,
                   extract_movement_series, extract_spo2_series)
import storage                                             # noqa: E402

# How often to collect, in seconds. 1800 = every 30 minutes.
INTERVAL = 1800


def one_cycle():
    """Collect one round of all three streams and save it."""
    ts = time.time()                      # single 'now' for this cycle (UTC seconds)

    # 1) ANIMAL — fetch and save first, on its own.
    animal = get_latest_animal_location()
    storage.save_animal(ts, animal)
    lat = float(animal["latitude"])
    lon = float(animal["longitude"])
    print(f"[{time.strftime('%H:%M:%S')}] animal saved ({lat:.3f}, {lon:.3f})")

    # 2) WEATHER + AIR — its own try so a failure doesn't lose the animal row.
    try:
        weather = get_current_weather(lat, lon)
        air = get_air_quality(lat, lon)
        storage.save_weather(ts, lat, lon, weather, air)
        print("   weather + air saved")
    except Exception as e:
        print(f"   weather failed: {e}")

    # 3) HEART — fetch today's Ultrahuman metrics ONCE, then save HRV + HR + SpO2.
    #    Each metric is saved in its own try, so a wrong key or a missing metric
    #    (e.g. SpO2 only recorded during sleep) never blocks the others.
    try:
        metrics = get_daily_metrics()        # today's Ultrahuman metrics
    except Exception as e:
        print(f"   heart fetch failed: {e}")
        metrics = None

    if metrics is not None:
        try:
            hrv = extract_hrv_series(metrics)             # {timestamp_utc, hrv_ms}
            print(f"   HRV:  {len(hrv)} pts, {storage.save_hrv_series(hrv)} new")
        except Exception as e:
            print(f"   HRV failed: {e}")
        try:
            hr = extract_hr_series(metrics)               # {timestamp_utc, value}
            print(f"   HR:   {len(hr)} pts, {storage.save_series('hr_bpm', hr)} new")
        except Exception as e:
            print(f"   HR failed: {e}")
        try:
            spo2 = extract_spo2_series(metrics)           # {timestamp_utc, value}
            print(f"   SpO2: {len(spo2)} pts, {storage.save_series('spo2', spo2)} new")
        except Exception as e:
            print(f"   SpO2 failed: {e}")


def main():
    storage.init()
    print("Collection started. Interval:", INTERVAL, "seconds. Ctrl+C to stop.")
    print("Starting row counts:", storage.counts())

    while True:
        try:
            one_cycle()
        except Exception as e:
            # The animal fetch itself failed (network / Movebank). Skip this cycle.
            print(f"[{time.strftime('%H:%M:%S')}] cycle failed: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
