"""
settings.py

Non-secret configuration: locations, polling intervals, backends and thresholds.
Secrets (API tokens, Thames Water credentials) stay in .env and are read by the
collector that needs them. Anything here can be overridden by an environment
variable of the same name.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DB_PATH = Path(os.getenv("SIOT_DB", BASE_DIR / "data" / "siot.db"))
SAMPLES_DIR = BASE_DIR / "data" / "samples"

# --- Locations -------------------------------------------------------------
# Default is Imperial (South Kensington). Set HOME_LAT / HOME_LON in .env to
# the real flat/room so outdoor data matches the indoor sensor.
HOME_LAT = float(os.getenv("HOME_LAT", 51.4988))
HOME_LON = float(os.getenv("HOME_LON", -0.1749))
LOCATIONS = {"home": (HOME_LAT, HOME_LON)}

# --- Carbon Intensity API --------------------------------------------------
CARBON_BASE = "https://api.carbonintensity.org.uk"
CARBON_REGION_ID = 13          # UKPN London
CARBON_REGION_NAME = "London"

# --- Open-Meteo ------------------------------------------------------------
OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_HORIZON_H = 24        # forecast rows stored per poll

# --- Ultrahuman ------------------------------------------------------------
ULTRAHUMAN_URL = "https://partner.ultrahuman.com/api/v1/partner/daily_metrics"
USER_TZ = "Europe/London"

# --- Thames Water (needs THAMES_CLIENT_ID / THAMES_CLIENT_SECRET in .env) --
THAMES_API_ROOT = "https://prod-tw-opendata-app.uk-e1.cloudhub.io/data/STE/v1"

# --- SCD40 -----------------------------------------------------------------
# backend: "adafruit" (I2C on a Raspberry Pi), "serial" (Pico/Arduino/ESP over
# USB, CSV lines "co2,temp_c,rh"), or "sim" (synthetic, for development).
SCD40_BACKEND = os.getenv("SCD40_BACKEND", "http")   # "sim" must be chosen explicitly
SCD40_SERIAL_PORT = os.getenv("SCD40_SERIAL_PORT", "COM3")
INGEST_PORT = int(os.getenv("INGEST_PORT", 8765))   # backend "http": ESP32 posts here; token in .env as INGEST_TOKEN
SCD40_WINDOW_S = 60           # one stored reading = mean of the sensor samples in this window
SCD40_TEMP_OFFSET_C = float(os.getenv("SCD40_TEMP_OFFSET_C", 4.0))  # self-heating correction; calibrate against a reference
SCD40_LOCATION = "room"

# --- Polling intervals (seconds) -------------------------------------------
INTERVALS = {
    "scd40": SCD40_WINDOW_S,   # the job itself blocks for the averaging window
    "open_meteo": 1800,        # hourly model output; 30 min guarantees freshness
    "grid_carbon": 900,        # NESO updates each half hour
    "ultrahuman": 1800,        # ring syncs via phone, so sub-30-min polling gains nothing
    "thames_water": 600,
}

# --- Dashboard (localhost only: it exposes health data) ---------------------
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8050))
# nominal sample period (s) per metric, used for the data-completeness figures
EXPECTED_PERIOD_S = {"co2_ppm": 60, "carbon_actual_gco2_kwh": 1800, "outdoor_temp_c": 3600,
                     "pm2_5": 3600, "hr_bpm": 420}
WEEK_TARGET_DAYS = 7

# --- Staleness limits (used by `main.py status` and later by analysis) -----
STALE_AFTER_S = {
    "scd40": 300,
    "open_meteo": 3 * 3600,
    "grid_carbon": 2 * 3600,   # 'actual' trails real time by ~1 h
    "ultrahuman": 6 * 3600,
    "thames_water": 3600,
}
