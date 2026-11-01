-- Storage schema. All timestamps are UTC unix seconds.

-- One narrow table for every time series (any source, any metric). Adding a
-- signal later (e.g. renewable generation) needs no schema change.
--   kind: measured  - physical sensor / wearable reading
--         modelled  - provider-computed estimate (Open-Meteo, NESO "actual")
--         forecast  - prediction; issued_at records when it was made
--   ts:   the time the value APPLIES to (period start for interval data)
CREATE TABLE IF NOT EXISTS observations (
    source       TEXT    NOT NULL,
    metric       TEXT    NOT NULL,
    location     TEXT    NOT NULL,
    ts           INTEGER NOT NULL,
    kind         TEXT    NOT NULL CHECK (kind IN ('measured','modelled','forecast')),
    issued_at    INTEGER NOT NULL DEFAULT 0,   -- 0 unless kind='forecast'
    value        REAL,
    unit         TEXT,
    quality      TEXT    NOT NULL DEFAULT 'ok', -- ok | partial | suspect
    retrieved_at INTEGER NOT NULL,
    PRIMARY KEY (source, metric, location, ts, kind, issued_at)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_obs_metric_ts ON observations (metric, location, ts);

-- One row per collector execution: latency, success and volume. Gives the
-- performance / reliability numbers for the report for free.
CREATE TABLE IF NOT EXISTS collector_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collector   TEXT    NOT NULL,
    started_at  INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('ok','error','skipped')),
    n_rows      INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs ON collector_runs (collector, started_at);

-- Latest raw payload per (source, key): lets us re-parse after a bug fix and
-- keeps a reproducible record of what each API actually returned.
CREATE TABLE IF NOT EXISTS raw_responses (
    source       TEXT    NOT NULL,
    key          TEXT    NOT NULL,
    retrieved_at INTEGER NOT NULL,
    payload      TEXT    NOT NULL,
    PRIMARY KEY (source, key)
);

-- Ultrahuman sleep session per wake-date (defines the "overnight" window).
CREATE TABLE IF NOT EXISTS sleep_sessions (
    day             TEXT PRIMARY KEY,      -- local wake date YYYY-MM-DD
    bedtime_start   INTEGER,
    bedtime_end     INTEGER,
    total_sleep_s   INTEGER,
    time_in_bed_s   INTEGER,
    deep_s          INTEGER,
    light_s         INTEGER,
    rem_s           INTEGER,
    efficiency_pct  REAL,
    sleep_score     REAL,
    avg_body_temp_c REAL,
    toss_turns      INTEGER,
    stages_json     TEXT,                  -- [{start,end,type}, ...]
    retrieved_at    INTEGER NOT NULL
);

-- Thames Water overflow status: one row per status TRANSITION per asset.
CREATE TABLE IF NOT EXISTS overflow_status (
    location_name TEXT    NOT NULL,
    alert_status  TEXT    NOT NULL,        -- Discharging | Not discharging | Offline
    status_change INTEGER NOT NULL,        -- when the source says it changed (0 if unknown)
    last_start    INTEGER,
    last_stop     INTEGER,
    water_course  TEXT,
    lat           REAL,
    lon           REAL,
    first_seen_at INTEGER NOT NULL,
    PRIMARY KEY (location_name, alert_status, status_change)
);

-- Thames Water discharge alert stream (Start / Stop / Offline ...).
CREATE TABLE IF NOT EXISTS overflow_alerts (
    location_name TEXT    NOT NULL,
    alert_type    TEXT    NOT NULL,
    event_ts      INTEGER NOT NULL,
    PRIMARY KEY (location_name, alert_type, event_ts)
);

-- Filled in week 2 by the analysis scripts; created now so it is stable.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,          -- air_quality | river_discharge | grid_carbon | indoor_co2
    dedupe_key  TEXT    NOT NULL,
    detected_at INTEGER NOT NULL,
    event_start INTEGER NOT NULL,
    event_end   INTEGER,
    severity    TEXT    NOT NULL,
    confidence  REAL,
    location    TEXT,
    evidence    TEXT    NOT NULL,          -- JSON: inputs, thresholds, source timestamps
    UNIQUE (event_type, dedupe_key)
);

-- Push-style sensor nodes (ESP32): last contact and ingest counters.
CREATE TABLE IF NOT EXISTS device_status (
    device          TEXT PRIMARY KEY,
    last_seen       INTEGER NOT NULL,
    remote_addr     TEXT,
    accepted_total  INTEGER NOT NULL DEFAULT 0,
    rejected_total  INTEGER NOT NULL DEFAULT 0,
    last_batch      INTEGER,
    oldest_age_s    INTEGER              -- age of the oldest reading in the last batch (backlog depth)
);

-- What the pet is (or was told to be) showing. The newest row is the current display.
-- mode: live = written by the pet controller, test = written from the dashboard.
CREATE TABLE IF NOT EXISTS display_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         INTEGER NOT NULL,
    expression TEXT    NOT NULL,
    message    TEXT    NOT NULL,
    task       TEXT,
    event_id   INTEGER,
    mode       TEXT    NOT NULL DEFAULT 'live'
);

-- One row per execution of an analysis module (detectors, personal analysis).
CREATE TABLE IF NOT EXISTS analysis_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    module      TEXT    NOT NULL,
    started_at  INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('ok','error')),
    n_events    INTEGER NOT NULL DEFAULT 0,
    note        TEXT
);
