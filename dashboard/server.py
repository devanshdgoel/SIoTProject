"""
server.py

Local monitoring dashboard: a read-mostly JSON API over the project database plus
the single-page UI in index.html. Runs as its own process (WAL lets it read while
`main.py collect` writes). Binds to 127.0.0.1 only because it exposes health data.

  GET  /                     the page
  GET  /api/state            collectors, storage, devices, analysis, display, config
  GET  /api/series           downsampled time series for one metric
  POST /api/run/<collector>  run one collector now (not scd40)
  POST /api/display          write a TEST display state (pet preview / hardware test)

POSTs require the header X-Requested-With: dashboard, which a cross-site form or
fetch cannot send without a CORS preflight that this server never approves.
"""

import ast
import json
import os
import statistics
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from collectors.base import run_collector
from config import settings

INDEX = Path(__file__).with_name("index.html")
CHART_JS = Path(__file__).with_name("chart.min.js")   # Chart.js 4.4.1, bundled so the page works offline
EXPRESSIONS = settings.EXPRESSIONS      # shared with the node firmware
LABEL_KINDS = ["window", "door", "occupancy", "cooking"]
SECRETS = ["ULTRAHUMAN_TOKEN", "THAMES_CLIENT_ID", "THAMES_CLIENT_SECRET", "INGEST_TOKEN"]
COMPONENT_DIRS = ["collectors", "analysis", "pet", "actions"]


def _rows(c, q, *args):
    return [dict(r) for r in c.execute(q, args).fetchall()]


# ---------------------------------------------------------------- state
def collectors_state(c, collectors, now):
    newest = {r["source"]: r["t"] for r in c.execute(
        "SELECT source, MAX(ts) t FROM observations WHERE kind!='forecast' GROUP BY source")}
    out = []
    for name, module in collectors.items():
        recent = _rows(c, "SELECT started_at, duration_ms, status, n_rows, error FROM collector_runs "
                          "WHERE collector=? ORDER BY started_at DESC LIMIT 200", name)
        last = recent[0] if recent else None
        day = [r for r in recent if r["started_at"] >= now - 86400]
        ok_ms = sorted(r["duration_ms"] for r in day if r["status"] == "ok")
        last_ok = next((r["started_at"] for r in recent if r["status"] == "ok"), None)
        last_err = next((r for r in recent if r["status"] == "error"), None)
        streak = 0
        for r in recent:
            if r["status"] != "error":
                break
            streak += 1
        data_ts = newest.get(getattr(module, "SOURCE", name))
        limit = settings.STALE_AFTER_S.get(name, 3600)
        age = now - data_ts if data_ts else None
        if last is None:
            health, why = "idle", "never run"
        elif last["status"] == "skipped":
            health, why = "idle", last["error"] or "not configured"
        elif last["status"] == "error":
            health, why = "bad", last["error"]
        elif data_ts is None:
            health, why = "warn", "running, no data yet"
        elif age > limit:
            health, why = "warn", f"newest data is {int(age // 60)} min old (limit {limit // 60} min)"
        else:
            health, why = "good", "ok"
        out.append({
            "name": name, "interval_s": settings.INTERVALS.get(name), "health": health, "why": why,
            "last_run": last["started_at"] if last else None, "last_ok": last_ok,
            "runs_24h": len(day), "ok_24h": sum(r["status"] == "ok" for r in day),
            "err_24h": sum(r["status"] == "error" for r in day),
            "p50_ms": int(statistics.median(ok_ms)) if ok_ms else None,
            "p95_ms": ok_ms[min(len(ok_ms) - 1, int(len(ok_ms) * 0.95))] if ok_ms else None,
            "newest_data": data_ts, "data_age_s": age, "stale_limit_s": limit, "fail_streak": streak,
            "last_error": {"at": last_err["started_at"], "msg": last_err["error"]} if last_err else None,
            "can_run": name != "scd40",
        })
    return out


def storage_state(c, now, db_path):
    db_path = Path(db_path)
    tot = c.execute("""SELECT SUM(kind!='forecast') obs, SUM(kind='forecast') fc,
                              SUM(quality='partial') partial, SUM(quality='suspect') suspect
                       FROM observations""").fetchone()
    counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
              ("raw_responses", "sleep_sessions", "overflow_status", "overflow_alerts",
               "collector_runs", "events", "display_log", "analysis_runs")}
    metrics = _rows(c, """SELECT o.source, o.metric, o.location, o.kind, COUNT(*) n, MIN(o.ts) first_ts,
                                 MAX(o.ts) last_ts, MAX(o.unit) unit
                          FROM observations o WHERE o.kind!='forecast'
                          GROUP BY o.source, o.metric, o.location, o.kind ORDER BY o.source, o.metric""")
    for m in metrics:
        m["last_value"] = c.execute(
            "SELECT value FROM observations WHERE source=? AND metric=? AND location=? AND kind=? "
            "ORDER BY ts DESC LIMIT 1", (m["source"], m["metric"], m["location"], m["kind"])).fetchone()[0]
        period = settings.EXPECTED_PERIOD_S.get(m["metric"])
        if period and m["last_ts"] > m["first_ts"]:
            expected = (m["last_ts"] - m["first_ts"]) / period + 1
            m["completeness"] = round(min(1.0, m["n"] / expected), 3)
    per_day = _rows(c, """SELECT source, ts/86400 d, COUNT(*) n FROM observations
                          WHERE kind!='forecast' AND ts>=? GROUP BY source, d ORDER BY d""", now - 14 * 86400)
    co2 = c.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM observations WHERE metric='co2_ppm'").fetchone()
    gaps = _rows(c, """SELECT prev_ts gap_start, ts gap_end, ts-prev_ts gap_s FROM (
                         SELECT ts, LAG(ts) OVER (ORDER BY ts) prev_ts FROM observations
                         WHERE metric='co2_ppm' AND ts>=?) WHERE ts-prev_ts>300
                       ORDER BY gap_s DESC LIMIT 10""", now - 14 * 86400)
    size = sum(p.stat().st_size for p in db_path.parent.glob(db_path.name + "*") if p.is_file())
    return {
        "db_path": str(db_path), "db_bytes": size, "observations": tot["obs"] or 0, "forecast_rows": tot["fc"] or 0,
        "partial": tot["partial"] or 0, "suspect": tot["suspect"] or 0, "tables": counts, "metrics": metrics,
        "per_day": per_day, "gaps": gaps,
        "week": {"first": co2[0], "last": co2[1], "rows": co2[2], "target_days": settings.WEEK_TARGET_DAYS,
                 "days": round((co2[1] - co2[0]) / 86400, 2) if co2[0] else 0},
    }


def component_state():
    base = Path(settings.BASE_DIR)
    out = {}
    for d in COMPONENT_DIRS:
        items = []
        for f in sorted((base / d).glob("*.py")):
            if f.name == "__init__.py":
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
                n = sum(isinstance(x, (ast.FunctionDef, ast.ClassDef)) for x in ast.walk(tree))
            except SyntaxError:
                n = -1
            items.append({"file": f.name, "defs": n, "modified": int(f.stat().st_mtime),
                          "status": "error" if n < 0 else "implemented" if n else "stub"})
        out[d] = items
    return out


def full_state(db, collectors):
    now = int(time.time())
    with db.conn() as c:
        cols = collectors_state(c, collectors, now)
        return {
            "now": now, "collectors": cols, "storage": storage_state(c, now, db.path),
            "devices": _rows(c, "SELECT * FROM device_status ORDER BY last_seen DESC"),
            "events": _rows(c, "SELECT * FROM events ORDER BY event_start DESC LIMIT 30"),
            "analysis_runs": _rows(c, "SELECT * FROM analysis_runs ORDER BY started_at DESC LIMIT 20"),
            "display": _rows(c, "SELECT * FROM display_log ORDER BY id DESC LIMIT 15"),
            "alerts": _rows(c, "SELECT * FROM overflow_alerts ORDER BY event_ts DESC LIMIT 10"),
            "errors": _rows(c, "SELECT collector, started_at, error FROM collector_runs "
                               "WHERE status='error' ORDER BY started_at DESC LIMIT 10"),
            "body": body_state(c, now),
            "components": component_state(),
            "config": {
                "scd40_backend": settings.SCD40_BACKEND, "ingest_port": settings.INGEST_PORT,
                "intervals_s": settings.INTERVALS, "locations": settings.LOCATIONS,
                "collectors_enabled": settings.COLLECTORS_ENABLED,
                "ultrahuman_metrics": settings.ULTRAHUMAN_METRICS,
                "laqn_sites": settings.LAQN_SITES,
                "stale_after_s": settings.STALE_AFTER_S,
                "secrets_present": {k: bool(os.getenv(k)) for k in SECRETS},
            },
            "expressions": EXPRESSIONS,
            "label_kinds": LABEL_KINDS,
            "params": _rows(c, "SELECT location, param, value, stderr, n, unit, fitted_at "
                               "FROM room_params ORDER BY location, param"),
            "labels": _rows(c, "SELECT location, kind, value, start_ts, end_ts, note "
                               "FROM labels ORDER BY start_ts DESC LIMIT 12"),
            "decay_events": _rows(c, "SELECT location, start_ts, end_ts, lambda_h, stderr, r2, "
                                     "c_start, c_end, c_out, delta_t_c, wind_ms, inferred, labelled "
                                     "FROM decay_events ORDER BY start_ts DESC LIMIT 80"),
            "decays": _rows(c, "SELECT location, COUNT(*) n, "
                               "ROUND(AVG(lambda_h), 2) mean_lambda, inferred "
                               "FROM decay_events WHERE r2>=0.9 GROUP BY location, inferred"),
            "forecast_error": _rows(c, "SELECT DATE(issued_at,'unixepoch') day, horizon_s, "
                                       "COUNT(*) n, ROUND(AVG(ABS(actual-predicted)),1) mae "
                                       "FROM predictions WHERE actual IS NOT NULL "
                                       "GROUP BY day, horizon_s ORDER BY day DESC LIMIT 14"),
        }


def body_state(c, now):
    """The wearable in one object: today's scalars against the user's own baseline,
    the last few nights, and how long ago the ring last reached the cloud.

    The baseline is a median of this person's own recent days rather than any
    population figure - it is what the strictness multiplier is measured against."""
    import statistics
    out = {"latest": {}, "baseline": {}, "n_days": {}}
    for metric in ("daily_recovery_index", "daily_sleep_hrv_ms", "daily_sleep_rhr_bpm"):
        rows = c.execute("SELECT ts, value FROM observations WHERE source='ultrahuman' "
                         "AND metric=? AND value IS NOT NULL AND ts>? ORDER BY ts",
                         (metric, now - 30 * 86400)).fetchall()
        if not rows:
            continue
        values = [r["value"] for r in rows]
        out["latest"][metric] = {"value": values[-1], "ts": rows[-1]["ts"]}
        out["baseline"][metric] = round(statistics.median(values), 1)
        out["n_days"][metric] = len(values)
    newest = c.execute("SELECT MAX(ts) t FROM observations WHERE source='ultrahuman' "
                       "AND metric='hr_bpm'").fetchone()["t"]
    out["newest_sample"] = newest
    out["sync_age_s"] = None if not newest else now - newest
    today = c.execute("SELECT SUM(value) s FROM observations WHERE source='ultrahuman' "
                      "AND metric='steps' AND ts>?", (now - now % 86400,)).fetchone()["s"]
    out["steps_today"] = int(today) if today else 0
    out["sleep"] = _rows(c, "SELECT day, bedtime_start, bedtime_end, total_sleep_s, deep_s, "
                            "rem_s, efficiency_pct, sleep_score, avg_body_temp_c, toss_turns "
                            "FROM sleep_sessions ORDER BY day DESC LIMIT 8")
    return out


def rooms_state(db):
    """Live per-room snapshot: what the sensors say, what the model believes, and
    where it thinks the room is heading. Computed on request rather than stored,
    so the page always reflects the parameters as they stand right now."""
    from analysis import analysis_human as ah
    from analysis.room_model import RoomModel, rooms

    now = int(time.time())
    quiet = ah.quiet_now(now)
    strict, z = ah.strictness(db, now)
    out = {"now": now, "quiet": quiet, "strictness": strict, "recovery_z": z,
           "thresholds": ah.THRESH, "rooms": []}
    for name in rooms(db):
        model = RoomModel(db, name)
        st = ah.room_state(db, name, model, now, strict)
        expr, msg, sev = ah.decide(dict(st, quiet=quiet))
        traj = [] if st.get("stale") else model.forward(
            st["co2"], 180, st.get("occupied", False))
        out["rooms"].append({**st, "expression": expr, "message": msg, "severity": sev,
                             "trajectory": traj, "learned": model.learned,
                             "volume_m3": model.volume_m3,
                             "params": {k: {"value": v, "stderr": model.stderr[k],
                                            "n": model.n[k], "basis": model.basis[k]}
                                        for k, v in model.value.items()}})
    return out


def nights_state(db, limit=14):
    """Overnight exposure per night, over the sleep window the ring actually
    measured, paired with that morning's recovery figures."""
    import pandas as pd

    from analysis import features

    now = int(time.time())
    step_h = features.GRID_S / 3600.0
    out = []
    for start, end, day in features.night_windows(db, now - 40 * 86400, now)[-limit:]:
        seg = features.series(db, "co2_ppm", "bedroom", start - 600, end + 600)
        seg = seg.loc[pd.Timestamp(start, unit="s", tz="UTC"):
                      pd.Timestamp(end, unit="s", tz="UTC")].dropna()
        if seg.size < 6:
            continue
        with db.conn() as c:
            def daily(metric):
                row = c.execute("SELECT value FROM observations WHERE metric=? AND "
                                "location='user' AND DATE(ts,'unixepoch')=? LIMIT 1",
                                (metric, day)).fetchone()
                return row["value"] if row else None
            hrv, recovery = daily("daily_sleep_hrv_ms"), daily("daily_recovery_index")
        out.append({"day": day, "start": start, "end": end,
                    "hours": round((end - start) / 3600, 1),
                    "mean": round(float(seg.mean())), "peak": round(float(seg.max())),
                    "rise": round(float(seg.iloc[-1] - seg.iloc[0])),
                    "above1000_h": round(float((seg > 1000).sum()) * step_h, 1),
                    "above1400_h": round(float((seg > 1400).sum()) * step_h, 1),
                    "dose": round(float((seg - 800).clip(lower=0).sum()) * step_h),
                    "hrv": hrv, "recovery": recovery})
    return out


def series(db, q):
    metric = q["metric"][0]
    hours = min(float(q.get("hours", ["24"])[0]), 24 * 30)
    ahead = min(float(q.get("ahead", ["0"])[0]), 72)
    kind = q.get("kind", ["observed"])[0]
    now = time.time()
    sql = "SELECT ts, value, issued_at FROM observations WHERE metric=? AND ts>=? AND ts<=?"
    args = [metric, now - hours * 3600, now + ahead * 3600]
    if kind == "forecast":
        sql += " AND kind='forecast'"
    else:
        sql += " AND kind!='forecast'"
    if q.get("location"):
        sql += " AND location=?"
        args.append(q["location"][0])
    with db.conn() as c:
        rows = c.execute(sql + " ORDER BY issued_at, ts", args).fetchall()
    pts = {r["ts"]: r["value"] for r in rows if r["value"] is not None}   # latest forecast issue wins
    items = sorted(pts.items())
    n_max = 500
    if len(items) > n_max:                       # mean-downsample into equal time buckets
        t0, t1 = items[0][0], items[-1][0]
        width = max(1, (t1 - t0) // n_max + 1)
        buckets = {}
        for t, v in items:
            buckets.setdefault((t - t0) // width, []).append((t, v))
        items = [(sum(t for t, _ in b) // len(b), sum(v for _, v in b) / len(b))
                 for _, b in sorted(buckets.items())]
    return {"metric": metric, "points": [[t, round(v, 3)] for t, v in items]}


# ---------------------------------------------------------------- http
def make_handler(db, collectors):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/chart.min.js":
                    return self._send(200, CHART_JS.read_bytes(), "application/javascript")
                if u.path == "/api/state":
                    return self._send(200, full_state(db, collectors))
                if u.path == "/api/rooms":
                    return self._send(200, rooms_state(db))
                if u.path == "/api/nights":
                    return self._send(200, {"nights": nights_state(db)})
                if u.path == "/api/series":
                    return self._send(200, series(db, parse_qs(u.query)))
            except KeyError:
                return self._send(400, {"error": "missing parameter"})
            except Exception as e:  # noqa: BLE001 - never crash the dashboard on one bad query
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.headers.get("X-Requested-With") != "dashboard":
                return self._send(403, {"error": "forbidden"})
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if u.path.startswith("/api/run/"):
                    name = u.path.rsplit("/", 1)[1]
                    if name not in collectors or name == "scd40":
                        return self._send(404, {"error": "unknown or non-runnable collector"})
                    status, rows, err = run_collector(name, collectors[name].collect, db)
                    return self._send(200, {"status": status, "rows": rows, "error": err})
                if u.path == "/api/label":
                    try:
                        start, end = int(body["start_ts"]), int(body["end_ts"])
                    except (KeyError, TypeError, ValueError):
                        return self._send(400, {"error": "start_ts/end_ts required"})
                    kind, value = str(body.get("kind", "")), str(body.get("value", ""))[:40]
                    if kind not in LABEL_KINDS or not value or end <= start:
                        return self._send(400, {"error": "bad label"})
                    db.add_label(str(body.get("location", ""))[:40], kind, value,
                                 start, end, str(body.get("note", ""))[:200] or None)
                    return self._send(200, {"ok": True})
                if u.path == "/api/analyse":
                    from analysis import analysis_human
                    out = analysis_human.run(db)
                    return self._send(200, {"decision": out.get("decision"),
                                            "fits": out.get("fits"),
                                            "strictness": out.get("strictness")})
                if u.path == "/api/display":
                    expr, msg = str(body.get("expression", "")), str(body.get("message", ""))[:120]
                    if expr not in EXPRESSIONS or not msg:
                        return self._send(400, {"error": "bad expression/message"})
                    db.set_display(expr, msg, mode="test")
                    return self._send(200, {"ok": True})
            except (ValueError, TypeError):
                return self._send(400, {"error": "malformed"})
            self._send(404, {"error": "not found"})

    return Handler


def create_server(db, collectors, port=None):
    return ThreadingHTTPServer(("127.0.0.1", settings.DASHBOARD_PORT if port is None else port),
                               make_handler(db, collectors))


def serve(db, collectors, port=None):
    srv = create_server(db, collectors, port)
    print(f"dashboard: http://127.0.0.1:{srv.server_address[1]}  (db: {db.path})  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
