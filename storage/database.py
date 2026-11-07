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
        for table, column, decl in (("labels", "people", "INTEGER"),
                                    ("decay_events", "method", "TEXT"),
                                    ("decay_events", "c_inf", "REAL"),
                                    ("decay_events", "asym_note", "TEXT"),
                                    ("predictions", "naive", "REAL")):
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

        # collector_runs gained an 'empty' status. SQLite cannot alter a CHECK
        # constraint in place, so an older database has to be rebuilt - additively,
        # keeping every row, because the run history is the record of how the
        # system behaved and is the only place an outage is visible after the fact.
        sql = c.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                        "AND name='collector_runs'").fetchone()
        if sql and "'empty'" not in (sql[0] or ""):
            c.execute("ALTER TABLE collector_runs RENAME TO collector_runs_old")
            c.executescript(SCHEMA.read_text(encoding="utf-8"))
            cols = [r[1] for r in c.execute("PRAGMA table_info(collector_runs_old)")]
            names = ",".join(cols)
            c.execute(f"INSERT INTO collector_runs ({names}) "
                      f"SELECT {names} FROM collector_runs_old")
            c.execute("DROP TABLE collector_runs_old")

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

    def backup_to(self, path):
        """A consistent single-file copy, safe to take while the collector writes.

        The three ad-hoc backups in data/ are plain file copies of a WAL-mode
        database taken without the -wal sidecar, so each holds only what had been
        checkpointed into the main file at that instant: 19, 31 and 60 CO2 rows
        against the live 49,000. They open, they pass integrity_check, and
        restoring from the newest would silently lose six days. integrity_check
        verifies b-tree structure, not that the data is there, which is why nobody
        noticed.

        sqlite3's own backup API reads through the WAL and takes a proper snapshot.
        The row count is returned so a caller can verify by content rather than by
        the file existing."""
        import sqlite3 as _sq
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as src:
            dst = _sq.connect(path)
            try:
                src.backup(dst)
                n = dst.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            finally:
                dst.close()
        live = 0
        with self.conn() as c:
            live = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        return {"path": str(path), "rows": int(n), "live_rows": int(live),
                "complete": int(n) == int(live)}

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

    def decays(self, location=None, min_r2=None, start=None, end=None):
        """Fitted decays, optionally restricted to a window.

        The window matters: `learn()` only deletes rows inside the period it
        re-fitted, so older rows survive carrying the lambda they were fitted with
        against an OLDER c_out. This room's c_out moved 107 ppm in one refinement,
        which is about 20% on every lambda, so pooling across that boundary mixes
        two different measurements of two different things."""
        q, args = "SELECT * FROM decay_events WHERE 1=1", []
        if location:
            q, args = q + " AND location=?", args + [location]
        if min_r2 is not None:
            q, args = q + " AND r2>=?", args + [min_r2]
        if start is not None:
            q, args = q + " AND start_ts>=?", args + [int(start)]
        if end is not None:
            q, args = q + " AND start_ts<=?", args + [int(end)]
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
        """Mean absolute error of scored forecasts per day: the learning curve.

        `naive_mae` is the same error for "assume no change", which is the forecast
        a reader should compare against before believing any of this. A model that
        cannot beat persistence has not been shown to know anything about the room."""
        q = ["SELECT DATE(issued_at,'unixepoch') day, horizon_s, COUNT(*) n,"
             " ROUND(AVG(ABS(actual-predicted)),1) mae, ROUND(AVG(actual-predicted),1) bias,"
             " ROUND(AVG(ABS(actual-naive)),1) naive_mae,"
             " SUM(naive IS NOT NULL) n_naive",
             "FROM predictions WHERE actual IS NOT NULL AND metric=?"]
        args = [metric]
        if location:
            q.append("AND location=?")
            args.append(location)
        q.append("AND issued_at>=? GROUP BY day, horizon_s ORDER BY day")
        args.append(int(time.time()) - days * 86400)
        with self.conn() as c:
            return c.execute(" ".join(q), args).fetchall()

    def forecast_skill(self, location=None, metric="co2_ppm", days=3650):
        """Model error against persistence, split by horizon and by basis.

        Split by basis because that is the comparison the learning claim rests on:
        if the error with learned parameters is no better than with the published
        defaults, the room taught the model nothing. Grouping only by day - which is
        what the figure used to do - cannot show that."""
        q = ["SELECT location, horizon_s, basis, COUNT(*) n,"
             " ROUND(AVG(ABS(actual-predicted)),1) mae,"
             " ROUND(AVG(ABS(actual-naive)),1) naive_mae,"
             " ROUND(AVG(actual-predicted),1) bias",
             "FROM predictions WHERE actual IS NOT NULL AND naive IS NOT NULL",
             "AND metric=?"]
        args = [metric]
        if location:
            q.append("AND location=?")
            args.append(location)
        q.append("AND issued_at>=? GROUP BY location, horizon_s, basis"
                 " ORDER BY location, horizon_s, basis")
        args.append(int(time.time()) - days * 86400)
        with self.conn() as c:
            rows = [dict(r) for r in c.execute(" ".join(q), args).fetchall()]
        for r in rows:
            r["skill"] = (None if not r["naive_mae"] else
                          round(1 - r["mae"] / r["naive_mae"], 3))
        return rows

    # ---- episodes awaiting a verdict --------------------------------------
    def add_episodes(self, rows):
        """Store detections. Idempotent on (location, kind, start_ts), and a row that
        has already been judged is never overwritten by a fresh detection."""
        kept = 0
        with self.conn() as c:
            for r in rows:
                judged = c.execute("SELECT verdict FROM episodes WHERE location=? AND kind=? "
                                   "AND start_ts=?", (r["location"], r["kind"],
                                                      r["start_ts"])).fetchone()
                if judged and judged["verdict"]:
                    continue
                _insert(c, "episodes", r, "INSERT OR REPLACE")
                kept += 1
        return kept

    def episodes(self, verdict="__open__", limit=50, since=None):
        """Detections, newest first. By default only those nobody has judged yet."""
        q, args = "SELECT * FROM episodes WHERE 1=1", []
        if verdict == "__open__":
            q += " AND verdict IS NULL"
        elif verdict is not None:
            q, args = q + " AND verdict=?", args + [verdict]
        if since is not None:
            q, args = q + " AND start_ts>=?", args + [int(since)]
        with self.conn() as c:
            return c.execute(q + " ORDER BY start_ts DESC LIMIT ?", args + [int(limit)]).fetchall()

    def judge_episode(self, episode_id, verdict, people=None):
        """Record the user's answer, and turn a confirmation into a real label.

        A denial is not discarded: it is stored as the opposite label, because
        "this was not a window opening" is just as useful to the threshold that
        separates ventilated decays from sealed ones."""
        opposite = {"open": "closed", "present": "absent", "yes": "no"}
        with self.conn() as c:
            row = c.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
            if row is None:
                return None
            c.execute("UPDATE episodes SET verdict=?, verdict_at=?, people=? WHERE id=?",
                      (verdict, int(time.time()), people, episode_id))
        if verdict == "confirmed":
            self.add_label(row["location"], row["kind"], row["value"], row["start_ts"],
                           row["end_ts"], "confirmed from a detection", people)
        elif verdict == "denied":
            flip = opposite.get(row["value"])
            if flip:
                self.add_label(row["location"], row["kind"], flip, row["start_ts"],
                               row["end_ts"], "detection denied")
        return dict(row)

    def episode_accuracy(self, days=60):
        """How often the detector was right, by kind - the learning curve's numerator."""
        with self.conn() as c:
            return c.execute(
                "SELECT kind, COUNT(*) n, SUM(verdict='confirmed') right_, "
                "SUM(verdict='denied') wrong, SUM(verdict='unsure') unsure "
                "FROM episodes WHERE verdict IS NOT NULL AND verdict_at>=? GROUP BY kind",
                (int(time.time()) - days * 86400,)).fetchall()
