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

# --- LAQN (London Air Quality Network; no API key) --------------------------
# Reference analysers run by Imperial's ERG. Open-Meteo's air quality is a ~11 km
# CAMS model cell, so these measured sites are what local air actually looks like.
LAQN_API_ROOT = "https://api.erg.ic.ac.uk/AirQuality"
# Site codes to pull; `python -m collectors.laqn` lists the sites nearest home.
# KC1 North Kensington is the urban-background reference (has PM2.5);
# KC5 Earl's Court is kerbside (NO2 and PM10 only) and much closer.
LAQN_SITES = [c.strip().upper() for c in os.getenv("LAQN_SITES", "KC1,KC5").split(",") if c.strip()]

# --- SCD40 -----------------------------------------------------------------
# backend: "adafruit" (I2C on a Raspberry Pi), "serial" (Pico/Arduino/ESP over
# USB, CSV lines "co2,temp_c,rh"), or "sim" (synthetic, for development).
SCD40_BACKEND = os.getenv("SCD40_BACKEND", "http")   # "sim" must be chosen explicitly
SCD40_SERIAL_PORT = os.getenv("SCD40_SERIAL_PORT", "COM3")
INGEST_PORT = int(os.getenv("INGEST_PORT", 8765))   # backend "http": ESP32 posts here; token in .env as INGEST_TOKEN
SCD40_WINDOW_S = 60           # one stored reading = mean of the sensor samples in this window
SCD40_TEMP_OFFSET_C = float(os.getenv("SCD40_TEMP_OFFSET_C", 4.0))  # self-heating correction; calibrate against a reference
SCD40_LOCATION = "desk"       # default room for a node that does not say which room it is in
# Each node names its room in the upload ("room": "bedroom"); the room becomes the
# observation location, so every sensor keeps its own series.

# --- Pet display -----------------------------------------------------------
# The faces the pet can show. One list for the whole system: the host picks a face,
# the dashboard draws it, and the node's OLED renders it (firmware/esp32_scd40).
EXPRESSIONS = ["neutral", "happy", "concerned", "alert", "sleepy", "sad"]

# --- Which collectors run ---------------------------------------------------
# Everything stays importable and testable; this only decides what the scheduler
# starts. grid_carbon and thames_water are off because the current question is
# indoor air and the body's response to it, and neither feeds that. Both APIs
# serve history, so re-enabling and backfilling loses nothing.
COLLECTORS_ENABLED = [c.strip() for c in
                      os.getenv("COLLECTORS", "scd40,open_meteo,laqn,ultrahuman").split(",")
                      if c.strip()]

# Ultrahuman metrics worth storing for the air-quality question: the overnight
# response variables, plus steps and heart rate as the activity confounder and a
# presence hint. skin_temp_c is dropped (mostly off-finger artefacts) along with
# the general activity indices. The API serves history, so a backfill brings any
# of them back if the analysis later needs them.
ULTRAHUMAN_METRICS = [c.strip() for c in os.getenv(
    "ULTRAHUMAN_METRICS",
    "hr_bpm,hrv_ms,steps,daily_sleep_hrv_ms,daily_sleep_rhr_bpm,daily_recovery_index"
).split(",") if c.strip()]

# --- Polling intervals (seconds) -------------------------------------------
INTERVALS = {
    "scd40": SCD40_WINDOW_S,   # the job itself blocks for the averaging window
    "open_meteo": 1800,        # hourly model output; 30 min guarantees freshness
    "grid_carbon": 900,        # NESO updates each half hour
    "ultrahuman": 1800,        # ring syncs via phone, so sub-30-min polling gains nothing
    "thames_water": 600,
    "laqn": 1800,              # hourly analyser means, revised as they are ratified
}

# --- Analysis --------------------------------------------------------------
ANALYSIS_INTERVAL_S = 300     # one pass per grid step; it only reads stored data
ANALYSIS_WINDOW_DAYS = 14     # how far back each re-fit looks

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
    "ultrahuman": 12 * 3600,   # the ring syncs only when the phone app opens, so a
                               # half-day lag is normal, not a fault. A real failure
                               # shows up as a collector error, not as stale data.
    "thames_water": 3600,
    "laqn": 4 * 3600,      # ratification lag means the newest hour often trails
}
