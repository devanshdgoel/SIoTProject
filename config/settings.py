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

# --- AURN (DEFRA reference analysers; no API key) ---------------------------
# Measured outdoor air, against Open-Meteo's ~11 km CAMS model cell. Each pollutant
# at a site is its own station id; `python -m collectors.aurn` lists the nearest.
# 786074 Westminster and 962 North Kensington are urban background; 1033 Marylebone
# Road is kerbside, which brackets a flat on a busy road between the two characters.
AURN_API_ROOT = "https://uk-air.defra.gov.uk/sos-ukair/api/v1"
AURN_STATIONS = [s.strip() for s in os.getenv("AURN_STATIONS", "786074,962,1033").split(",") if s.strip()]

# --- Breathe London (GLA/Imperial Clarity nodes; API key required) ---------
# Worth having for distance, not for accuracy. Measured from HOME_LAT/HOME_LON:
#   CLDP0317 Warwick Road          0.71 km  roadside
#   CLDP0675 Christ Church School  1.80 km  urban background
#   nearest AURN analyser          ~3 km
#   Open-Meteo CAMS cell           ~11 x 7 km, the whole of inner London
# Both are configured on purpose rather than just the closest: a roadside node at
# 700 m measures a road, and the air that infiltrates a bedroom is closer in
# character to urban background. Having both brackets the flat instead of
# assuming which one it resembles. Low-cost optical sensors, so a disagreement
# with AURN is a finding rather than automatically their error.
# Site selection needs no key - `python -m collectors.breathe_london` lists the
# nearest nodes from the GLA's open inventory, with their siting class.
BREATHE_LONDON_ROOT = "https://api.breathelondon-communities.org/api"
BREATHE_LONDON_KEY = os.getenv("BREATHE_LONDON_KEY", "").strip()
BREATHE_LONDON_SITES = [s.strip() for s in
                        os.getenv("BREATHE_LONDON_SITES", "CLDP0317,CLDP0675").split(",")
                        if s.strip()]

# --- Which collectors run ---------------------------------------------------
# Everything stays importable and testable; this only decides what the scheduler
# starts. The city-scale collectors (grid carbon, Thames overflows, LAQN) live in
# archive/london_side with a note on why and how to bring them back; the question
# here is indoor air and the body's response to it, and none of them feed that.
COLLECTORS_ENABLED = [c.strip() for c in
                      os.getenv("COLLECTORS", "scd40,open_meteo,ultrahuman,aurn").split(",")
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
    "ultrahuman": 1800,
    "aurn": 1800,             # hourly analyser means, published with a lag        # ring syncs via phone, so sub-30-min polling gains nothing
    "breathe_london": 1800,   # hourly averages, same as AURN
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
    "aurn": 6 * 3600,         # ratification lag means the newest hour often trails
    "breathe_london": 6 * 3600,
    "ultrahuman": 12 * 3600,   # the ring syncs only when the phone app opens, so a
                               # half-day lag is normal, not a fault. A real failure
                               # shows up as a collector error, not as stale data.
}
