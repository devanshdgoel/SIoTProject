"""
storage.py — saves and loads the project's time-series data in one SQLite file.

THREE tables, one per stream (they update at different rates):
  - animal_position : the stork's GPS fixes (changes slowly, every few hours)
  - weather_reading : weather + air quality AT the animal, fetched each cycle
  - heart_reading   : YOUR physiological readings from the Ultrahuman ring

All share a time axis (UTC unix seconds) so you can line them up for the
analysis later. Keeping everything in ONE .db file keeps it portable.
"""

import sqlite3
import time
import os
from datetime import datetime      # used to turn Ultrahuman's ISO times into unix

# The database file lives in data/ next to this core/ folder.
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "siot.db")


def _connect():
    """Open a connection; create data/ if missing; let us read rows by name."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row      # so we can do row["pm2_5"] not row[13]
    return conn


# ---------------------------------------------------------------------------
# SCHEMA — "IF NOT EXISTS" makes init() safe to run every startup.
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS animal_position (
    ts            REAL NOT NULL,   -- when WE recorded it (UTC unix seconds)
    lat           REAL,            -- latitude
    lon           REAL,            -- longitude
    animal_ts     TEXT,            -- the GPS fix's own timestamp (from Movebank)
    individual_id TEXT             -- which bird
);
CREATE INDEX IF NOT EXISTS idx_animal_ts ON animal_position(ts);

CREATE TABLE IF NOT EXISTS weather_reading (
    ts            REAL NOT NULL,   -- when WE recorded it (UTC unix seconds)
    lat           REAL,            -- location these readings apply to
    lon           REAL,
    -- weather (metric units) --
    temp_c         REAL,           -- air temperature, deg C
    feels_like_c   REAL,
    humidity       REAL,           -- %
    pressure_hpa   REAL,
    wind_speed_mps REAL,           -- metres per second
    wind_deg       REAL,
    cloud_pct      REAL,
    uv_index       REAL,
    weather_desc   TEXT,           -- e.g. "clear sky"
    -- air quality --
    aqi            INTEGER,        -- air quality index
    pm2_5          REAL,           -- ug/m3
    pm10           REAL,
    no2            REAL,
    o3             REAL,
    co             REAL,
    so2            REAL,
    nh3            REAL
);
CREATE INDEX IF NOT EXISTS idx_weather_ts ON weather_reading(ts);

-- Heart / physiological readings in LONG format: one row per (time, metric).
-- Why long format? Ultrahuman gives several metrics (HRV, resting HR, SpO2,
-- breathing rate...) each as its own time-series with its OWN timestamps and
-- cadence. A long table stores them all without a messy wide table full of
-- NULLs, and adding a new metric later needs NO schema change — you just save
-- with a new `metric` name.
CREATE TABLE IF NOT EXISTS heart_reading (
    measured_ts REAL NOT NULL,     -- the reading's OWN time (UTC unix seconds)
    metric      TEXT NOT NULL,     -- 'hrv_ms' (later 'rhr_bpm','spo2','resp_rate'...)
    value       REAL,
    UNIQUE(measured_ts, metric)    -- re-pulling the day won't duplicate points
);
CREATE INDEX IF NOT EXISTS idx_heart_ts ON heart_reading(measured_ts);
"""


def init():
    """Create all tables/indexes if they don't exist. Call once at startup."""
    conn = _connect()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SAVE — one function per stream.
# ---------------------------------------------------------------------------
def save_animal(ts, animal):
    """Save one animal position (dict from animal.get_latest_animal_location())."""
    conn = _connect()
    conn.execute(
        """INSERT INTO animal_position (ts, lat, lon, animal_ts, individual_id)
           VALUES (?, ?, ?, ?, ?)""",
        (
            ts,
            float(animal["latitude"]),
            float(animal["longitude"]),
            animal.get("timestamp"),
            animal.get("individual_id"),
        ),
    )
    conn.commit()
    conn.close()


def save_weather(ts, lat, lon, weather, air):
    """Save one weather+air row (dicts from weather.get_current_weather/get_air_quality)."""
    conn = _connect()
    conn.execute(
        """INSERT INTO weather_reading
           (ts, lat, lon, temp_c, feels_like_c, humidity, pressure_hpa,
            wind_speed_mps, wind_deg, cloud_pct, uv_index, weather_desc,
            aqi, pm2_5, pm10, no2, o3, co, so2, nh3)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ts, lat, lon,
            weather.get("temperature_c"),
            weather.get("feels_like_c"),
            weather.get("humidity_percent"),
            weather.get("pressure_hpa"),
            weather.get("wind_speed_mps"),
            weather.get("wind_direction_deg"),
            weather.get("cloud_cover_percent"),
            weather.get("uv_index"),
            weather.get("weather"),
            air.get("aqi"),
            air.get("pm2_5"),
            air.get("pm10"),
            air.get("no2"),
            air.get("o3"),
            air.get("co"),
            air.get("so2"),
            air.get("nh3"),
        ),
    )
    conn.commit()
    conn.close()


def save_hrv_series(points):
    """
    Save a list of HRV points from heart.extract_hrv_series():
    each point is {timestamp_utc: <ISO string>, hrv_ms: <float>}.

    Each point is stored under measured_ts (its own time, as unix seconds) with
    metric='hrv_ms'. INSERT OR IGNORE + the UNIQUE constraint means re-pulling
    today's data only adds genuinely NEW points — no duplicates.

    Returns how many NEW rows were actually inserted (nice for the loop print).
    """
    conn = _connect()
    before = conn.total_changes
    for p in points:
        # Ultrahuman gives an ISO string like "2026-10-17T09:05:00+00:00".
        # fromisoformat parses it; .timestamp() converts to unix seconds (UTC).
        measured_ts = datetime.fromisoformat(p["timestamp_utc"]).timestamp()
        conn.execute(
            """INSERT OR IGNORE INTO heart_reading (measured_ts, metric, value)
               VALUES (?, ?, ?)""",
            (measured_ts, "hrv_ms", p["hrv_ms"]),
        )
    conn.commit()
    inserted = conn.total_changes - before
    conn.close()
    return inserted


# A generic saver for the OTHER Ultrahuman metrics (HR, SpO2, steps...). Takes
# the SAME point format the heart.extract_series() helpers produce:
#   list of {"timestamp_utc": <ISO string>, "value": <float>}
# No schema change needed — just pass a different `metric` name.
def save_series(metric, points):
    """
    Save a list of {timestamp_utc, value} points under `metric`
    (e.g. 'hr_bpm', 'spo2'). INSERT OR IGNORE + the UNIQUE(measured_ts, metric)
    constraint means re-pulling the day only adds new points.
    Returns the number of NEW rows inserted.
    """
    conn = _connect()
    before = conn.total_changes
    for p in points:
        measured_ts = datetime.fromisoformat(p["timestamp_utc"]).timestamp()
        conn.execute(
            """INSERT OR IGNORE INTO heart_reading (measured_ts, metric, value)
               VALUES (?, ?, ?)""",
            (measured_ts, metric, p["value"]),
        )
    conn.commit()
    inserted = conn.total_changes - before
    conn.close()
    return inserted


# ---------------------------------------------------------------------------
# READ — simple helpers.
# ---------------------------------------------------------------------------
def recent_animal(hours=24):
    return _recent("animal_position", "ts", hours)


def recent_weather(hours=24):
    return _recent("weather_reading", "ts", hours)


def recent_heart(hours=24, metric=None):
    """Heart readings in the last `hours`; optionally filter to one metric."""
    cutoff = time.time() - hours * 3600
    conn = _connect()
    if metric:
        rows = conn.execute(
            "SELECT * FROM heart_reading WHERE measured_ts >= ? AND metric = ? "
            "ORDER BY measured_ts ASC", (cutoff, metric),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM heart_reading WHERE measured_ts >= ? "
            "ORDER BY measured_ts ASC", (cutoff,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _recent(table, time_col, hours):
    cutoff = time.time() - hours * 3600
    conn = _connect()
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE {time_col} >= ? ORDER BY {time_col} ASC",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def counts():
    """Quick row counts so you can confirm data is piling up."""
    conn = _connect()
    a = conn.execute("SELECT COUNT(*) FROM animal_position").fetchone()[0]
    w = conn.execute("SELECT COUNT(*) FROM weather_reading").fetchone()[0]
    h = conn.execute("SELECT COUNT(*) FROM heart_reading").fetchone()[0]
    conn.close()
    return {"animal_position": a, "weather_reading": w, "heart_reading": h}


if __name__ == "__main__":
    init()
    print("DB ready at:", os.path.abspath(DB_PATH))
    print("row counts:", counts())
