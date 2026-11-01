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
            "components": component_state(),
            "config": {
                "scd40_backend": settings.SCD40_BACKEND, "ingest_port": settings.INGEST_PORT,
                "intervals_s": settings.INTERVALS, "locations": settings.LOCATIONS,
                "stale_after_s": settings.STALE_AFTER_S,
                "secrets_present": {k: bool(os.getenv(k)) for k in SECRETS},
            },
            "expressions": EXPRESSIONS,
        }


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
