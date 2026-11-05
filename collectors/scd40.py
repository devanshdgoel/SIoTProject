"""
scd40.py

Reads indoor CO2, temperature and relative humidity from the physical SCD40 sensor.

The sensor measures every 5 s; one stored reading is the mean of the samples in a
SCD40_WINDOW_S (60 s) window. Indoor CO2 evolves over minutes (occupancy build-up,
window-open decay), so 1/min comfortably satisfies Nyquist while averaging out the
sensor's +-(40 ppm + 5 %) noise and keeping a week to ~10k rows. A window with fewer
than half its expected samples is stored with quality='partial'.

Backends (settings.SCD40_BACKEND):
  adafruit - SCD40 on I2C of a Raspberry Pi (adafruit-circuitpython-scd4x)
  serial   - microcontroller streaming "co2,temp_c,rh" lines over USB serial
  http     - ESP32 over WiFi POSTs buffered minute readings to this host (preferred)
  sim      - synthetic bedroom, so the pipeline and analysis can be developed
             without hardware. Use a separate database for simulated data.

Notes for the real deployment:
  * Disable automatic self-calibration (assumes weekly exposure to 400 ppm air; a
    continuously occupied bedroom would drift low) and instead run a forced
    recalibration once outdoors.
  * Temperature reads high from self-heating; SCD40_TEMP_OFFSET_C compensates and
    should be tuned against a reference thermometer.
  * Ambient pressure from Open-Meteo is fed to the sensor so CO2 is pressure-compensated.
"""

import hmac
import json
import math
import os
import queue
import random
import re
import threading
import time
from pathlib import Path

from collectors.base import CollectResult, CollectorError, NotConfigured, Observation
from config import settings

SOURCE = "scd40"
SAMPLE_PERIOD_S = 5
ROOM_RE = re.compile(r"[a-z0-9_-]{1,20}")   # room names become observation locations


# --------------------------------------------------------------------------
# Backends: each yields (co2_ppm, temp_c, rh_pct) tuples via read_window()
# --------------------------------------------------------------------------
class AdafruitBackend:
    def __init__(self):
        try:
            import adafruit_scd4x
            import board
        except ImportError as e:
            raise NotConfigured(f"adafruit backend needs a Raspberry Pi + adafruit-circuitpython-scd4x ({e})")
        self.dev = adafruit_scd4x.SCD4X(board.I2C())
        self.dev.stop_periodic_measurement()
        self.dev.self_calibration_enabled = False
        self.dev.temperature_offset = settings.SCD40_TEMP_OFFSET_C
        self.dev.start_periodic_measurement()
        self.pressure_set_at = 0

    def set_pressure(self, hpa):
        if time.time() - self.pressure_set_at > 3600 and 800 < hpa < 1100:
            self.dev.ambient_pressure = int(hpa)
            self.pressure_set_at = time.time()

    def read_window(self, seconds):
        out, end = [], time.time() + seconds
        while time.time() < end:
            if self.dev.data_ready:
                out.append((self.dev.CO2, self.dev.temperature, self.dev.relative_humidity))
            time.sleep(1)
        return out


class SerialBackend:
    def __init__(self):
        try:
            import serial
        except ImportError as e:
            raise NotConfigured(f"pyserial missing ({e})")
        self.port = serial.Serial(settings.SCD40_SERIAL_PORT, 115200, timeout=2)

    def set_pressure(self, hpa):
        pass

    def read_window(self, seconds):
        out, end = [], time.time() + seconds
        while time.time() < end:
            line = self.port.readline().decode("ascii", "ignore").strip()
            try:
                co2, t, rh = (float(x) for x in line.split(",")[:3])
                out.append((co2, t, rh))
            except ValueError:
                continue                      # banner/garbage line
        return out


class SimBackend:
    """Synthetic bedroom: occupied 23:00-07:30 with the window shut (CO2 climbs),
    ventilated by a short window opening at 07:30 and occasionally during the evening.
    Deterministic per timestamp (seeded by the minute), so reruns are reproducible."""

    OUT_PPM, GEN_PPM_MIN, ACH_CLOSED, ACH_OPEN = 420.0, 9.0, 0.35, 6.0

    def set_pressure(self, hpa):
        pass

    def _step(self, c, t):
        """Advance the mass balance by one minute starting at unix time t."""
        h = time.localtime(t)
        hour = h.tm_hour + h.tm_min / 60
        occupied = hour >= 23 or hour < 7.5
        window_open = (7.5 <= hour < 7.75) or (18.0 <= hour < 18.25 and h.tm_wday % 2 == 0)
        ach = self.ACH_OPEN if window_open else self.ACH_CLOSED
        return c + (self.OUT_PPM - c) * ach / 60 + (self.GEN_PPM_MIN if occupied else 0)

    def window_at(self, ts, seconds):
        """Raw samples for the window ending at ts (integrates 12 h of history first)."""
        c = 600.0
        for k in range(720):
            c = self._step(c, ts - 12 * 3600 + 60 * k)
        return self._samples(c, ts, seconds)

    def _samples(self, c, ts, seconds):
        rng = random.Random(ts // 60)
        hour = time.localtime(ts).tm_hour
        temp = 19.5 + 2.0 * math.sin((hour - 15) / 24 * 2 * math.pi) + 0.3 * (c - 420) / 1000
        n = max(1, seconds // SAMPLE_PERIOD_S)
        return [(c + rng.gauss(0, 15), temp + rng.gauss(0, 0.1), 52 + rng.gauss(0, 0.5)) for _ in range(n)]

    def series(self, start_ts, minutes):
        """Yield (ts, samples) once per minute in a single pass (for bulk generation)."""
        c = 600.0
        for k in range(minutes):
            ts = start_ts + 60 * k
            c = self._step(c, ts)
            yield ts, self._samples(c, ts, 60)

    def read_window(self, seconds):
        return self.window_at(int(time.time()), seconds)


def _clean(s, n):
    """Printable ASCII only, no quotes or backslashes: the node parses the reply by hand."""
    return re.sub(r'[^ -~]|["\\]', "", str(s))[:n]


def _display_for_node(row):
    """db.latest_display() row -> the pet state sent back in the /ingest reply."""
    return None if row is None else {"expression": _clean(row["expression"], 15),
                                     "message": _clean(row["message"], 40)}


class HttpBackend:
    """ESP32 (or any node) pushes minute-averaged readings over WiFi:

        POST /ingest   header X-Auth-Token: <INGEST_TOKEN>
        {"device": "esp32-1", "room": "desk", "readings": [{"ts": 1793313660, "co2": 812.4,
                                            "co2_max": 840, "t": 21.3, "rh": 49.8, "n": 12}, ...]}

    The node timestamps readings (NTP) and buffers them while this host is offline, then
    uploads the backlog; storage is idempotent on (metric, location, ts), so retries are
    safe. The reply carries the host's decisions back on the same connection:

        {"accepted": 2, "pressure_hpa": 1012.3,
         "display": {"expression": "concerned", "message": "open a window"}}

    pressure_hpa lets the node pressure-compensate the sensor; display is the newest
    display_log row, which the node with the OLED renders (null until one exists)."""

    MAX_AGE_S, MAX_FUTURE_S = 3 * 86400, 300

    def __init__(self, db=None, port=None, token=None):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        token = token or os.getenv("INGEST_TOKEN")
        if not token:
            raise NotConfigured("INGEST_TOKEN missing from .env (shared secret for the ESP32)")
        self.queue, self.db, self.pressure, self.display = queue.Queue(), db, None, None
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path != "/ingest":
                    return self._reply(404, {"error": "not found"})
                if not hmac.compare_digest(self.headers.get("X-Auth-Token", ""), token):
                    return self._reply(401, {"error": "bad token"})
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    if not 0 < n <= 200_000:
                        raise ValueError("size")
                    body = json.loads(self.rfile.read(n))
                    readings = body["readings"]
                    room = str(body.get("room") or settings.SCD40_LOCATION)
                    if not ROOM_RE.fullmatch(room):
                        raise ValueError("room")
                    accepted = sum(backend.accept(r, room) for r in readings)
                except (ValueError, KeyError, TypeError):
                    return self._reply(400, {"error": "malformed"})
                if backend.db is not None and readings:
                    oldest = min((int(r["ts"]) for r in readings if isinstance(r, dict) and "ts" in r), default=None)
                    backend.db.touch_device(str(body.get("device", "unknown"))[:40], accepted,
                                            len(readings) - accepted, len(readings),
                                            int(time.time() - oldest) if oldest else None,
                                            self.client_address[0])
                self._reply(200, {"accepted": accepted, "pressure_hpa": backend.pressure,
                                  "display": backend.display})

        self.server = ThreadingHTTPServer(("0.0.0.0", settings.INGEST_PORT if port is None else port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True, name="ingest").start()

    def accept(self, r, room=settings.SCD40_LOCATION):
        """Validate one reading and queue it; returns 1 if kept (bad/old/future rows are dropped)."""
        try:
            ts, co2, t, rh = int(r["ts"]), float(r["co2"]), float(r["t"]), float(r["rh"])
            co2_max, n = float(r.get("co2_max", co2)), int(r.get("n", 12))
        except (KeyError, TypeError, ValueError):
            return 0
        now = time.time()
        if not (now - self.MAX_AGE_S <= ts <= now + self.MAX_FUTURE_S):
            return 0
        # The CO2 floor is 250, not 350. These sensors run on their factory
        # calibration with self-calibration off, and this pair reads roughly 60 ppm
        # below true outdoor air - so a fully ventilated room lands near 350 and a
        # 350 floor censors exactly the readings the outdoor baseline is measured
        # from. It silently deleted 51 minutes of the first window-open test and
        # pinned the lowest stored value at precisely 350.0. A genuinely faulty
        # reading is 0 or negative, which 250 still rejects.
        if not (250 <= co2 <= 10000 and -10 <= t <= 60 and 0 <= rh <= 100):
            return 0
        self.queue.put((ts, co2, co2_max, t, rh, n, room))
        return 1

    def drain(self):
        """Queued readings for storage; also refresh what the next reply tells the node.

        Both DB reads happen here, on the collector thread, so the request handler
        never touches SQLite."""
        if self.db is not None:
            p = self.db.latest("pressure_hpa")
            self.pressure = round(p["value"], 1) if p is not None and p["value"] else None
            self.display = _display_for_node(self.db.latest_display())
        rows = []
        while True:
            try:
                rows.append(self.queue.get_nowait())
            except queue.Empty:
                return rows


_backend = None


def _get_backend(db=None):
    global _backend
    if _backend is None:
        name = settings.SCD40_BACKEND
        cls = {"adafruit": AdafruitBackend, "serial": SerialBackend, "sim": SimBackend,
               "http": lambda: HttpBackend(db)}.get(name)
        if cls is None:
            raise NotConfigured(f"unknown SCD40_BACKEND '{name}'")
        if name == "sim" and db is not None and Path(db.path).resolve() == Path(settings.BASE_DIR / "data" / "siot.db").resolve():
            raise NotConfigured("SCD40_BACKEND=sim refuses to write synthetic data into the real database "
                                "(use --db data/sim.db, or set SCD40_BACKEND=http)")
        _backend = cls()
    return _backend


def reading_to_obs(ts, co2, co2_max, t, rh, n, location=settings.SCD40_LOCATION):
    """One node-averaged reading -> Observations ('partial' if under half the samples arrived)."""
    q = "ok" if n >= 6 else "partial"
    mk = lambda m, v, u: Observation(SOURCE, m, location, ts, round(v, 2), "measured", u, quality=q)  # noqa: E731
    return [mk("co2_ppm", co2, "ppm"), mk("co2_max_ppm", co2_max, "ppm"),
            mk("indoor_temp_c", t, "C"), mk("indoor_rh_pct", rh, "%")]


def summarise(samples, ts, window_s=settings.SCD40_WINDOW_S, location=settings.SCD40_LOCATION):
    """Samples -> Observations (mean of the window, plus CO2 max). Pure function."""
    expected = max(1, window_s // SAMPLE_PERIOD_S)
    good = [s for s in samples if 350 <= s[0] <= 10000 and -10 <= s[1] <= 60 and 0 <= s[2] <= 100]
    if not good:
        raise CollectorError(f"no valid SCD40 samples in window ({len(samples)} received)")
    q = "ok" if len(good) >= expected / 2 else "partial"
    mean = lambda i: sum(s[i] for s in good) / len(good)  # noqa: E731
    mk = lambda metric, v, unit: Observation(SOURCE, metric, location, ts, round(v, 2),  # noqa: E731
                                              "measured", unit, quality=q)
    return [mk("co2_ppm", mean(0), "ppm"), mk("co2_max_ppm", max(s[0] for s in good), "ppm"),
            mk("indoor_temp_c", mean(1), "C"), mk("indoor_rh_pct", mean(2), "%")]


def collect(db=None):
    backend = _get_backend(db)
    if hasattr(backend, "drain"):                 # push-style node: store whatever arrived
        result = CollectResult(SOURCE)
        for row in backend.drain():
            result.observations += reading_to_obs(*row)
        return result
    if db is not None:
        p = db.latest("pressure_hpa")
        if p is not None and p["value"]:
            backend.set_pressure(p["value"])
    window = settings.SCD40_WINDOW_S
    samples = backend.read_window(window)
    ts = int(time.time()) // window * window      # window end aligned to the minute
    result = CollectResult(SOURCE)
    result.observations = summarise(samples, ts, window)
    return result


if __name__ == "__main__":
    for o in collect().observations:
        print(o.metric, o.value, o.unit, o.quality)
