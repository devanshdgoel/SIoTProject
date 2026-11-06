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
            self._migrate(c)

    @staticmethod
    def _migrate(c):
        """Add columns that CREATE TABLE IF NOT EXISTS cannot add to an existing table.

        Additive only: a column that is missing is appended, nothing is dropped or
        rewritten, so running against an older database never loses anything."""
        for table, column, decl in (("labels", "people", "INTEGER"),):
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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

    def latest_display(self):
        """Newest pet display state (a Row or None): what the node should be showing."""
        with self.conn() as c:
            return c.execute("SELECT expression, message FROM display_log "
                             "ORDER BY id DESC LIMIT 1").fetchone()

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

    # ---- analysis layer --------------------------------------------------
    def add_label(self, location, kind, value, start_ts, end_ts, note=None, people=None):
        """Record user ground truth ('bedroom window open 14:00-15:00'). Idempotent.

        `people` matters for occupancy: two people fill a room twice as fast, so a
        source term fitted without knowing the headcount is wrong for both cases and
        the room volume it implies comes out at half its true size."""
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO labels "
                      "(location,kind,value,people,start_ts,end_ts,note,created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (location, kind, value, None if people is None else int(people),
                       int(start_ts), int(end_ts), note, int(time.time())))

    def labels(self, location=None, kind=None, start=None, end=None):
        q, args = "SELECT * FROM labels WHERE 1=1", []
        for col, val, op in (("location", location, "="), ("kind", kind, "="),
                             ("end_ts", start, ">="), ("start_ts", end, "<=")):
            if val is not None:
                q += f" AND {col}{op}?"
                args.append(val)
        with self.conn() as c:
            return c.execute(q + " ORDER BY start_ts", args).fetchall()

    def set_param(self, location, param, value, stderr=None, n=0, unit=None, detail=None):
        """Store one learned room parameter, replacing any earlier fit."""
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO room_params "
                      "(location,param,value,stderr,n,unit,fitted_at,detail) VALUES (?,?,?,?,?,?,?,?)",
                      (location, param, value, stderr, int(n), unit, int(time.time()),
                       json.dumps(detail) if detail is not None else None))

    def params(self, location):
        """Learned parameters for one room as {param: Row}."""
        with self.conn() as c:
            return {r["param"]: r for r in
                    c.execute("SELECT * FROM room_params WHERE location=?", (location,))}

    def save_decays(self, rows):
        with self.conn() as c:
            for r in rows:
                _insert(c, "decay_events", r, "INSERT OR REPLACE")
        return len(rows)

    def decays(self, location=None, min_r2=None):
        q, args = "SELECT * FROM decay_events WHERE 1=1", []
        if location:
            q, args = q + " AND location=?", args + [location]
        if min_r2 is not None:
            q, args = q + " AND r2>=?", args + [min_r2]
        with self.conn() as c:
            return c.execute(q + " ORDER BY start_ts", args).fetchall()

    def add_predictions(self, rows):
        """Write forecasts before their outcome is known; scored later by score_due()."""
        with self.conn() as c:
            for r in rows:
                _insert(c, "predictions", r, "INSERT OR REPLACE")
        return len(rows)

    def score_due(self, now=None, tol_s=600):
        """Fill in `actual` for any prediction whose target time has passed.

        The measurement nearest the target time wins, provided it is within tol_s;
        otherwise the row stays unscored rather than being matched to a distant value."""
        now = int(time.time()) if now is None else int(now)
        scored = 0
        with self.conn() as c:
            due = c.execute("SELECT id, location, metric, target_ts FROM predictions "
                            "WHERE scored_at IS NULL AND target_ts<=?", (now,)).fetchall()
            for p in due:
                row = c.execute("""SELECT value FROM observations
                                   WHERE metric=? AND location=? AND kind!='forecast'
                                     AND ts BETWEEN ? AND ? AND value IS NOT NULL
                                   ORDER BY ABS(ts-?) LIMIT 1""",
                                (p["metric"], p["location"], p["target_ts"] - tol_s,
                                 p["target_ts"] + tol_s, p["target_ts"])).fetchone()
                if row is None:
                    continue
                c.execute("UPDATE predictions SET actual=?, scored_at=? WHERE id=?",
                          (row["value"], now, p["id"]))
                scored += 1
        return scored

    def prediction_error(self, location=None, metric="co2_ppm", days=7):
        """Mean absolute error of scored forecasts per day: the learning curve."""
        q = ["SELECT DATE(issued_at,'unixepoch') day, horizon_s, COUNT(*) n,"
             " ROUND(AVG(ABS(actual-predicted)),1) mae, ROUND(AVG(actual-predicted),1) bias",
             "FROM predictions WHERE actual IS NOT NULL AND metric=?"]
        args = [metric]
        if location:
            q.append("AND location=?")
            args.append(location)
        q.append("AND issued_at>=? GROUP BY day, horizon_s ORDER BY day")
        args.append(int(time.time()) - days * 86400)
        with self.conn() as c:
            return c.execute(" ".join(q), args).fetchall()
