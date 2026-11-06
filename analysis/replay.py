"""
replay.py

Re-run the system at past moments using only what it knew then.

The project's headline claim is that its forecasts improve as it learns. A sceptic
has an obvious objection: the model re-fits over a fourteen-day window, so a
prediction scored for Tuesday may have been made with parameters fitted to Friday's
data. If that is happening, the improving curve means nothing - it is just a model
reading the answers.

This settles it. For a chosen moment T it builds a database holding only rows that
had actually arrived by T, judged on `retrieved_at` for measurements and `issued_at`
for forecasts - not on the timestamp the value applies to, which is the distinction
that matters, because an hour's weather is often stored some hours after it happened.
Derived tables are left empty, so the model has to re-learn from scratch at every
step rather than inheriting anything.

It is built incrementally. T only ever moves forward, so each step copies just the
rows that arrived since the last one; a week-long replay therefore costs about one
pass over the data rather than one pass per step.

The second use is less obvious but more valuable day to day: changing a detector and
re-running a fortnight takes seconds, instead of waiting a fortnight to find out.
"""

import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from storage.database import Database

# Source tables and the column saying when each row became known. Anything derived -
# room_params, decay_events, predictions, events, display_log, analysis_runs - is
# deliberately absent: it has to be recomputed, never inherited.
SOURCE_TABLES = {
    "observations": "retrieved_at",
    "sleep_sessions": "retrieved_at",
    "labels": "created_at",
    "raw_responses": "retrieved_at",
}


class AsOf:
    """A database that only ever contains what had arrived by some moment.

    Use it as a context manager; the scratch file is removed on exit."""

    def __init__(self, source, scratch=None):
        self.source_path = Path(source.path if isinstance(source, Database) else source)
        self.scratch = Path(scratch or Path(tempfile.gettempdir()) /
                            f"replay-{int(time.time() * 1000)}.db")
        self.db = Database(self.scratch)
        self._cursor = 0          # rows up to here have been copied already

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.scratch) + suffix).unlink(missing_ok=True)

    def advance_to(self, ts):
        """Bring the view forward to `ts`, copying only what arrived since last time.

        Forecasts are judged on when they were issued, not on the hour they describe,
        so a forecast of tomorrow made this morning is visible and one made tomorrow
        is not."""
        ts = int(ts)
        if ts < self._cursor:
            raise ValueError("replay only moves forward; build a new AsOf to go back")
        copied = 0
        # A dedicated autocommit connection: ATTACH cannot run inside a transaction,
        # and Database.conn() opens one as soon as it is used.
        dest = sqlite3.connect(self.scratch, timeout=30, isolation_level=None)
        dest.row_factory = sqlite3.Row
        try:
            dest.execute("ATTACH DATABASE ? AS src", (str(self.source_path),))
            try:
                for table, column in SOURCE_TABLES.items():
                    cols = [r[1] for r in dest.execute(f"PRAGMA table_info({table})")]
                    names = ",".join(cols)
                    extra = ""
                    if table == "observations":
                        # a forecast is known from when it was issued
                        extra = " AND (kind!='forecast' OR (issued_at>? AND issued_at<=?))"
                    sql = (f"INSERT OR REPLACE INTO {table} ({names}) SELECT {names} "
                           f"FROM src.{table} WHERE {column}>? AND {column}<=?{extra}")
                    args = [self._cursor, ts] + ([self._cursor, ts] if extra else [])
                    copied += dest.execute(sql, args).rowcount
            finally:
                dest.execute("DETACH DATABASE src")
        finally:
            dest.close()
        self._cursor = ts
        return copied


def walk(source, start, end, step_s=3600, fn=None, scratch=None):
    """Step through time, running `fn(db, now)` at each point.

    `fn` receives a database containing only what was known, and the moment it is
    standing at. Whatever it returns is collected and handed back with its timestamp."""
    out = []
    with AsOf(source, scratch) as view:
        for now in range(int(start), int(end) + 1, int(step_s)):
            added = view.advance_to(now)
            result = fn(view.db, now) if fn else None
            out.append({"at": now, "rows_known": added, "result": result})
    return out


def replay_forecasts(source, start, end, step_s=3600, horizons=(1800, 7200)):
    """Re-forecast the past honestly, then score it against what happened.

    At each step the model is re-fitted from scratch on data that had arrived by
    then, asked for a prediction, and the prediction is compared with the value the
    real database eventually recorded. Any improvement across the window is therefore
    the model learning, not the model peeking."""
    from analysis import analysis_human, features
    from analysis.room_model import RoomModel, learn, rooms

    truth = Database(source.path if isinstance(source, Database) else source)
    scored = []

    def step(db, now):
        made = []
        for room in rooms(db):
            learn(db, room, days=14, now=now)
            model = RoomModel(db, room)
            state = analysis_human.room_state(db, room, model, now, 1.0)
            if state.get("stale"):
                continue
            # The prior has to be rebuilt from the same restricted view, or the
            # replay would be handing the forecast a pattern learned from the future.
            profile = features.occupancy_prior(db, room, now - 14 * 86400, now) or {}
            for horizon in horizons:
                path = model.forward(state["co2"], horizon / 60.0, state.get("occupied", False),
                                     prior=profile.get("prior"), now=now,
                                     persistence_min=profile.get("persistence_min", 60.0))
                if not path:
                    continue
                made.append({"room": room, "horizon_s": horizon, "target_ts": now + horizon,
                             "predicted": path[-1][1], "basis": "learned" if model.learned
                             else "default"})
        return made

    for entry in walk(source, start, end, step_s, step):
        for made in entry["result"] or []:
            with truth.conn() as c:
                row = c.execute(
                    "SELECT value FROM observations WHERE metric='co2_ppm' AND location=? "
                    "AND kind!='forecast' AND ts BETWEEN ? AND ? ORDER BY ABS(ts-?) LIMIT 1",
                    (made["room"], made["target_ts"] - 600, made["target_ts"] + 600,
                     made["target_ts"])).fetchone()
            if row is None or row["value"] is None:
                continue
            scored.append({**made, "issued_at": entry["at"], "actual": row["value"],
                           "error": made["predicted"] - row["value"]})
    return scored


def summarise(scored):
    """Mean absolute error and bias per room, horizon and day issued."""
    import statistics
    buckets = {}
    for s in scored:
        day = time.strftime("%Y-%m-%d", time.gmtime(s["issued_at"]))
        buckets.setdefault((s["room"], s["horizon_s"], day), []).append(s["error"])
    out = []
    for (room, horizon, day), errors in sorted(buckets.items()):
        out.append({"room": room, "horizon_s": horizon, "day": day, "n": len(errors),
                    "mae": round(statistics.fmean(abs(e) for e in errors), 1),
                    "bias": round(statistics.fmean(errors), 1),
                    "basis": "mixed"})
    return out
