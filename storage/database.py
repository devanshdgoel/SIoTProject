"""
database.py

Provides the common interface for storing and retrieving sensor readings,
API responses, derived features, environmental events, recommendations and task history.

SQLite in WAL mode; a short-lived connection per call keeps it safe to use
from the one-thread-per-collector scheduler in main.py.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from config import settings

SCHEMA = Path(__file__).with_name("schema.sql")


def _insert(c, table, row, verb="INSERT OR IGNORE"):
    cols = ",".join(row)
    c.execute(f"{verb} INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))


class Database:
    def __init__(self, path=None):
        self.path = Path(path or settings.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA.read_text(encoding="utf-8"))

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---- writes ----------------------------------------------------------
    def save_result(self, r):
        """Persist a collectors.base.CollectResult. Idempotent: re-running a
        collector or a backfill over the same period never duplicates rows."""
        with self.conn() as c:
            c.executemany(
                """INSERT INTO observations
                   (source,metric,location,ts,kind,issued_at,value,unit,quality,retrieved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source,metric,location,ts,kind,issued_at)
                   DO UPDATE SET value=excluded.value, quality=excluded.quality,
                                 retrieved_at=excluded.retrieved_at""",
                [(o.source, o.metric, o.location, o.ts, o.kind, o.issued_at,
                  o.value, o.unit, o.quality, o.retrieved_at) for o in r.observations])
            for key, payload in r.raw.items():
                c.execute("INSERT OR REPLACE INTO raw_responses VALUES (?,?,?,?)",
                          (r.source, key, int(time.time()), json.dumps(payload)))
            for s in r.sleep_sessions:
                _insert(c, "sleep_sessions", s, "INSERT OR REPLACE")
            for s in r.overflow_status:
                _insert(c, "overflow_status", s)
            for a in r.overflow_alerts:
                _insert(c, "overflow_alerts", a)
        return len(r.observations)

    def log_run(self, collector, started_at, duration_ms, status, n_rows=0, error=None):
        with self.conn() as c:
            c.execute("INSERT INTO collector_runs (collector,started_at,duration_ms,status,n_rows,error) "
                      "VALUES (?,?,?,?,?,?)",
                      (collector, int(started_at), int(duration_ms), status, n_rows, error))

    def touch_device(self, device, accepted, rejected, batch, oldest_age_s, addr=None):
        with self.conn() as c:
            c.execute("""INSERT INTO device_status
                         (device,last_seen,remote_addr,accepted_total,rejected_total,last_batch,oldest_age_s)
                         VALUES (?,?,?,?,?,?,?)
                         ON CONFLICT(device) DO UPDATE SET last_seen=excluded.last_seen,
                           remote_addr=excluded.remote_addr,
                           accepted_total=accepted_total+excluded.accepted_total,
                           rejected_total=rejected_total+excluded.rejected_total,
                           last_batch=excluded.last_batch, oldest_age_s=excluded.oldest_age_s""",
                      (device, int(time.time()), addr, accepted, rejected, batch, oldest_age_s))

    def set_display(self, expression, message, task=None, event_id=None, mode="live"):
        """Record what the pet is showing (the pet controller calls this on every change)."""
        with self.conn() as c:
            c.execute("INSERT INTO display_log (ts,expression,message,task,event_id,mode) VALUES (?,?,?,?,?,?)",
                      (int(time.time()), expression, message, task, event_id, mode))

    def log_analysis_run(self, module, started_at, duration_ms, status, n_events=0, note=None):
        with self.conn() as c:
            c.execute("INSERT INTO analysis_runs (module,started_at,duration_ms,status,n_events,note) "
                      "VALUES (?,?,?,?,?,?)", (module, int(started_at), int(duration_ms), status, n_events, note))

    # ---- reads -----------------------------------------------------------
    def latest(self, metric, location=None):
        """Most recent non-forecast observation of a metric (a Row or None)."""
        q = "SELECT * FROM observations WHERE metric=? AND kind!='forecast'"
        args = [metric]
        if location:
            q += " AND location=?"
            args.append(location)
        with self.conn() as c:
            return c.execute(q + " ORDER BY ts DESC LIMIT 1", args).fetchone()

    def max_alert_ts(self):
        with self.conn() as c:
            return c.execute("SELECT MAX(event_ts) FROM overflow_alerts").fetchone()[0]

    def read_df(self, metric, location=None, start=None, end=None, kind=None):
        """Time series as a pandas DataFrame indexed by UTC datetime
        (columns: value, kind, issued_at, quality, location)."""
        import pandas as pd
        q = "SELECT ts, value, kind, issued_at, quality, location FROM observations WHERE metric=?"
        args = [metric]
        for col, val, op in (("location", location, "="), ("kind", kind, "="),
                             ("ts", start, ">="), ("ts", end, "<")):
            if val is not None:
                q += f" AND {col}{op}?"
                args.append(val)
        with self.conn() as c:
            df = pd.read_sql_query(q + " ORDER BY ts", c, params=args)
        df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
        return df

    def status_summary(self):
        with self.conn() as c:
            obs = c.execute("""SELECT source, metric, location, kind, COUNT(*) n,
                                      MIN(ts) first_ts, MAX(ts) last_ts
                               FROM observations WHERE kind!='forecast'
                               GROUP BY source, metric, location, kind ORDER BY source, metric""").fetchall()
            runs = c.execute("""SELECT collector, COUNT(*) n,
                                       SUM(status='ok') ok, SUM(status='error') err,
                                       SUM(status='skipped') skipped,
                                       ROUND(AVG(CASE WHEN status='ok' THEN duration_ms END)) avg_ms,
                                       MAX(started_at) last_run
                                FROM collector_runs GROUP BY collector""").fetchall()
        return obs, runs
