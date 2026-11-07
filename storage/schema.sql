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
    status      TEXT    NOT NULL CHECK (status IN ('ok','empty','error','skipped')),
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

-- ---------------------------------------------------------------- analysis layer

-- Ground truth the user types in: "the bedroom window was open 14:00-15:00".
-- Labels calibrate the learned room parameters and score the system's own guesses.
-- They are needed only for the first few days; after that the system infers these
-- states itself and the labels become the test set.
CREATE TABLE IF NOT EXISTS labels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    location   TEXT    NOT NULL,
    kind       TEXT    NOT NULL,          -- window | door | occupancy | cooking
    value      TEXT    NOT NULL,          -- open | closed | present | absent | yes
    people     INTEGER,                     -- how many, when the label is an occupancy
    start_ts   INTEGER NOT NULL,
    end_ts     INTEGER NOT NULL,
    note       TEXT,
    created_at INTEGER NOT NULL,
    UNIQUE (location, kind, start_ts, end_ts)
);
CREATE INDEX IF NOT EXISTS idx_labels ON labels (location, kind, start_ts);

-- What the system has worked out about each room: one row per parameter, carrying
-- the sample count and standard error so a recommendation can say how sure it is.
-- Below MIN_N observations the analysis falls back to a published default instead.
CREATE TABLE IF NOT EXISTS room_params (
    location  TEXT    NOT NULL,
    param     TEXT    NOT NULL,           -- c_out | lambda_sealed | lambda_open | source_ppm_h | volume_m3 | tau_t_h
    value     REAL,
    stderr    REAL,
    n         INTEGER NOT NULL DEFAULT 0,
    unit      TEXT,
    fitted_at INTEGER NOT NULL,
    detail    TEXT,                       -- JSON fit diagnostics, kept for the report
    PRIMARY KEY (location, param)
);

-- One row per automatically detected CO2 decay: the data points behind the
-- air-change-rate model, each stamped with the weather it happened under.
CREATE TABLE IF NOT EXISTS decay_events (
    location  TEXT    NOT NULL,
    start_ts  INTEGER NOT NULL,
    end_ts    INTEGER NOT NULL,
    lambda_h  REAL    NOT NULL,           -- air changes per hour
    stderr    REAL,
    r2        REAL,
    c_start   REAL,
    c_end     REAL,
    c_out     REAL,
    c_inf     REAL,                       -- what the decay was actually heading for
    asym_note TEXT,                       -- why that is, or is not, outdoor air
    delta_t_c REAL,                       -- indoor minus outdoor at the time
    wind_ms   REAL,
    labelled  TEXT,                       -- window state from labels, if one covers it
    inferred  TEXT,                       -- sealed | ventilated, from the fitted value
    method    TEXT,                       -- departure (standardised) | window (threshold-found)
    PRIMARY KEY (location, start_ts)
);

-- Every forecast is written before its outcome is known and scored later. The
-- rolling error over these rows is the evidence that the system is improving.
CREATE TABLE IF NOT EXISTS predictions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    issued_at INTEGER NOT NULL,
    location  TEXT    NOT NULL,
    metric    TEXT    NOT NULL,
    target_ts INTEGER NOT NULL,
    horizon_s INTEGER NOT NULL,
    predicted REAL    NOT NULL,
    lo        REAL,
    hi        REAL,
    basis     TEXT,                        -- learned | default: which parameters were used
    naive     REAL,                         -- persistence: the room's CO2 when this was issued
    actual    REAL,
    scored_at INTEGER,
    UNIQUE (location, metric, issued_at, target_ts)
);
CREATE INDEX IF NOT EXISTS idx_pred_open ON predictions (scored_at, target_ts);

-- Things the system believes happened, before anybody has confirmed them. The user
-- answers each one yes, no or not sure; a yes becomes an ordinary label, a no becomes
-- evidence the other way, and a not-sure is recorded and then ignored. The verdicts
-- are also what the detector's accuracy is scored against, so the system can report
-- how often it was right rather than asserting that it improves.
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    location    TEXT    NOT NULL,
    kind        TEXT    NOT NULL,          -- window | occupancy | cooking | door
    value       TEXT    NOT NULL,          -- open | present | yes
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL,
    confidence  REAL,
    evidence    TEXT,                      -- JSON: what led it to think so
    detected_at INTEGER NOT NULL,
    verdict     TEXT,                      -- confirmed | denied | unsure
    verdict_at  INTEGER,
    people      INTEGER,                   -- supplied with a confirmation, when relevant
    UNIQUE (location, kind, start_ts)
);
CREATE INDEX IF NOT EXISTS idx_episodes_open ON episodes (verdict, start_ts);
