"""
dashboard/server.py — a tiny LIVE status dashboard for the whole project.

WHY A LOCAL SERVER (and not a web page on its own)?
  The data lives in your SQLite file on THIS computer. A plain HTML file can't
  read a database, and a hosted page can't reach your machine. So we run a
  very small web server HERE that reads the DB and hands the page fresh JSON.
  It uses ONLY Python's standard library — nothing to pip install.

HOW TO RUN (from the project root, in its own terminal):
    python dashboard/server.py
Then open the link it prints (http://127.0.0.1:8000) in your browser.
The page refreshes itself every 15 seconds. Press Ctrl+C here to stop it.

This is READ-ONLY: it never writes to the database, so it's completely safe
to run at the same time as collect_loop.py.
"""

import os
import sys
import json
import math
import time
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- make our own modules importable (core/ holds storage.py + analysis.py) --
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, os.path.join(ROOT, "core"))

import storage      # reuse the same DB path + helpers the collector uses
import analysis     # the habitat-stress "brain"

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "index.html")
PORT = 8000
COLLECT_INTERVAL = 1800   # must match collect_loop.py, used only to judge "is it alive?"


# ---------------------------------------------------------------------------
# DB helpers — all read-only. We open our own connection per request (simple
# and safe; SQLite handles concurrent readers fine).
# ---------------------------------------------------------------------------
def _conn():
    c = sqlite3.connect(storage.DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def animal_status(c, now):
    row = c.execute(
        "SELECT ts, lat, lon, animal_ts, individual_id "
        "FROM animal_position ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None

    # distinct GPS fixes (the bird's OWN timestamps) tell us real movement,
    # separate from how often WE polled.
    fixes = c.execute(
        "SELECT DISTINCT animal_ts, lat, lon FROM animal_position "
        "WHERE animal_ts IS NOT NULL ORDER BY animal_ts"
    ).fetchall()
    last_move = None
    if len(fixes) >= 2:
        a, b = fixes[-2], fixes[-1]
        last_move = round(_haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]), 1)

    return {
        "lat": row["lat"],
        "lon": row["lon"],
        "fix_time": row["animal_ts"],          # the stork's own GPS timestamp (string)
        "recorded_age_s": round(now - row["ts"], 1),
        "individual_id": row["individual_id"],
        "distinct_fixes": len(fixes),
        "last_move_m": last_move,
    }


def weather_status(c, now):
    latest = c.execute(
        "SELECT * FROM weather_reading ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if not latest:
        return None, None

    d = dict(latest)
    d["recorded_age_s"] = round(now - latest["ts"], 1)

    # a short recent series for the sparkline (oldest -> newest)
    series = c.execute(
        "SELECT ts, temp_c, aqi FROM weather_reading ORDER BY ts DESC LIMIT 48"
    ).fetchall()
    series = [{"ts": r["ts"], "temp_c": r["temp_c"], "aqi": r["aqi"]}
              for r in reversed(series)]
    return d, series


def heart_status(c):
    """Latest value of each physiological metric. The table may not exist yet
    (if the collector hasn't been restarted since the heart code was added),
    so we fail softly and tell the page that."""
    try:
        rows = c.execute(
            # for each metric, grab its most recent reading
            "SELECT metric, value, measured_ts FROM heart_reading h "
            "WHERE measured_ts = (SELECT MAX(measured_ts) FROM heart_reading "
            "                     WHERE metric = h.metric) "
        ).fetchall()
    except sqlite3.OperationalError:
        return {"available": False, "metrics": {}}

    metrics = {}
    for r in rows:
        metrics[r["metric"]] = {"value": r["value"], "ts": r["measured_ts"]}
    return {"available": len(metrics) > 0, "metrics": metrics}


def build_status():
    """Assemble the whole live picture into one JSON-able dict."""
    now = time.time()
    c = _conn()
    try:
        animal = animal_status(c, now)
        weather, wseries = weather_status(c, now)
        heart = heart_status(c)
        counts = storage.counts()
    finally:
        c.close()

    # The headline backend number: habitat stress from the latest weather.
    stress = analysis.habitat_stress(weather) if weather else None

    # Is the collector actually running? Judge by how stale the newest row is.
    freshest_age = None
    for part in (animal, weather):
        if part and part.get("recorded_age_s") is not None:
            freshest_age = (part["recorded_age_s"] if freshest_age is None
                            else min(freshest_age, part["recorded_age_s"]))
    collector_alive = (freshest_age is not None
                       and freshest_age < COLLECT_INTERVAL * 2)

    return {
        "now_utc": now,
        "collector": {
            "alive": collector_alive,
            "freshest_age_s": freshest_age,
            "interval_s": COLLECT_INTERVAL,
        },
        "counts": counts,
        "animal": animal,
        "weather": weather,
        "weather_series": wseries,
        "heart": heart,
        "indoor": {"available": False,
                   "note": "SCD40 indoor sensing starts late October."},
        "analysis": stress,
    }


# ---------------------------------------------------------------------------
# The web server: '/' serves the page, '/api/status' serves live JSON.
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")   # always fresh
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            try:
                payload = json.dumps(build_status()).encode("utf-8")
                self._send(200, payload, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(),
                           "application/json")
            return

        if self.path in ("/", "/index.html"):
            try:
                with open(PAGE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"index.html not found", "text/plain")
            return

        self._send(404, b"not found", "text/plain")

    def log_message(self, *args):
        pass   # keep the terminal quiet; comment this out to see each request


def main():
    storage.init()   # harmless if tables already exist; makes sure they do
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Kindred dashboard running -> http://127.0.0.1:{PORT}")
    print("Open that link in your browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
