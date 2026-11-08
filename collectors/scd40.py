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
import logging
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

log = logging.getLogger(__name__)

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


def _display_for_node(row, info=None):
    """db.latest_display() row -> the pet state sent back in the /ingest reply."""
    if row is None and not info:
        return None
    out = {"expression": _clean(row["expression"], 15) if row is not None else "",
           "message": _clean(row["message"], 40) if row is not None else ""}
    if info:
        out["info"] = info
    return out


# The info screen is recomputed on this interval rather than every minute: the
# purge search evaluates 144 start times against the forecast, which is cheap but
# not free, and the answer does not change in sixty seconds.
INFO_EVERY_S = 300


def _info_lines(db, location):
    """Four short lines for the node's double-tap screen, or [] if nothing to say.

    The node renders these and interprets none of them, which is the right split:
    deciding what is worth saying needs the room model, the weather forecast and
    the air-quality feed, and none of those belong on a microcontroller. The node's
    job is to show whatever the host last said, and to keep showing it when the
    host goes quiet.

    Kept to 22 characters so nothing is clipped at the 6x10 font's 21 columns.
    """
    import datetime as _dt

    lines = []
    try:
        from analysis.room_model import RoomModel
        from analysis.ventilation import plan_purge

        aqi = db.latest("european_aqi", "home")
        pm = db.latest("pm2_5", "home")
        bits = []
        if aqi is not None and aqi["value"] is not None:
            bits.append(f"AQI {aqi['value']:.0f}")
        if pm is not None and pm["value"] is not None:
            bits.append(f"PM2.5 {pm['value']:.0f}")
        lines.append("out: " + (", ".join(bits) if bits else "no data"))

        co2 = db.latest("co2_ppm", location)
        if co2 is not None and co2["value"] is not None:
            lines.append(f"{location[:8]} {co2['value']:.0f} ppm")

        plan = plan_purge(db, location)
        best = plan.get("best") if plan else None
        if best:
            when = _dt.datetime.fromtimestamp(best["at"], _dt.UTC).astimezone()
            lines.append(f"air at {when:%H:%M} for {best['duration_min']}m")
            lines.append(f"-{best['clears_ppm']}ppm {best['heat_cost_c']:+.1f}C")
        else:
            lines.append("no good window found")
    except Exception as e:                           # noqa: BLE001
        # The info screen must never be able to break ingest. A node that shows a
        # stale screen is a much smaller problem than a node whose readings stop
        # being accepted because the advice could not be computed.
        log.debug("info lines unavailable: %s", e)
        return []
    return [_clean(ln, 22) for ln in lines if ln][:4]


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
        self._info, self._info_at = [], 0.0
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body):
                # Compact, with no space after the colons. The node parses this
                # reply by hand - it has no JSON library and should not need one
                # for four fields - and its lookups are literal: it searches for
                # `"expression":"` and `"info":[`. json.dumps defaults to `: `
                # separators, so every one of those lookups failed and had been
                # failing since the feature was written. The face fell back to
                # local CO2 and looked stuck; the info screen stayed empty.
                #
                # Fixing it here rather than in the firmware means it takes effect
                # without reflashing either node, and the bytes are smaller too.
                # The node's parser is also being made whitespace-tolerant so that
                # this cannot bite again, but that needs a flash and this does not.
                data = json.dumps(body, separators=(",", ":")).encode()
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
                    # Whatever health fields the node volunteered. Older firmware
                    # sends none and that must stay fine, so every one is optional.
                    health = {k: body.get(k) for k in
                              ("up_s", "heap", "heap_max_block", "reset")
                              if body.get(k) is not None}
                    backend.db.touch_device(str(body.get("device", "unknown"))[:40], accepted,
                                            len(readings) - accepted, len(readings),
                                            int(time.time() - oldest) if oldest else None,
                                            self.client_address[0], health=health)
                self._reply(200, {"accepted": accepted, "pressure_hpa": backend.pressure,
                                  "display": backend.display})

        class Ingest(ThreadingHTTPServer):
            # Windows treats SO_REUSEADDR as permission to share a listening port, so
            # a second collector binds 8765 alongside the first instead of failing.
            # Nothing errors; the nodes' readings are then split between two processes
            # and land in whichever database each one writes to. Refusing the second
            # bind turns a silent data loss into an immediate, obvious failure.
            allow_reuse_address = False

        self.server = Ingest(("0.0.0.0", settings.INGEST_PORT if port is None else port), Handler)
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
        self.queue.put((ts, co2, co2_max, t, rh, n, room, _particles(r)))
        return 1

    def drain(self):
        """Queued readings for storage; also refresh what the next reply tells the node.

        Both DB reads happen here, on the collector thread, so the request handler
        never touches SQLite."""
        if self.db is not None:
            p = self.db.latest("pressure_hpa")
            self.pressure = round(p["value"], 1) if p is not None and p["value"] else None
            now = time.time()
            if now - self._info_at >= INFO_EVERY_S:
                self._info = _info_lines(self.db, settings.SCD40_LOCATION)
                self._info_at = now
            self.display = _display_for_node(self.db.latest_display(), self._info)
        rows = []
        while True:
            try:
                rows.append(self.queue.get_nowait())
            except queue.Empty:
                return rows


# PMS5003 fields, present only on the minutes the fan actually ran. A node without
# the sensor sends none of them, and a node with it sends them every fifth minute,
# so their absence is normal and must never cost the CO2 reading that carries them.
PM_MASS = (("pm1", "pm1_0"), ("pm25", "pm2_5"), ("pm10", "pm10"))
PM_COUNT = (("n03", "pc0_3"), ("n05", "pc0_5"), ("n10", "pc1_0"),
            ("n25", "pc2_5"), ("n50", "pc5_0"), ("n100", "pc10_0"))


def _particles(r):
    """Pull the particle fields out of one posted reading, or None if it has none.

    Everything is validated and the whole set is dropped together if any part of it
    fails. A frame that survived the sensor's own checksum but arrives with one
    impossible field is not a frame to take three quarters of.
    """
    if not isinstance(r, dict) or "pm25" not in r:
        return None
    out = {}
    try:
        for key, metric in PM_MASS:
            v = float(r[key])
            if not 0 <= v <= 1000:          # ug/m3; above this the sensor is saturated
                return None
            out[metric] = v
        for key, metric in PM_COUNT:
            v = float(r[key])
            if not 0 <= v <= 65535:         # per 0.1 L, the wire format's own ceiling
                return None
            out[metric] = v
    except (KeyError, TypeError, ValueError):
        return None
    # Counts are cumulative - every particle above 0.5 um is also above 0.3 - so a
    # set that rises with size is impossible and means the frame was misaligned.
    counts = [out[m] for _k, m in PM_COUNT]
    if any(b > a for a, b in zip(counts, counts[1:])):
        return None
    return out


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


# One floor, used by both paths. 350 censored 51 minutes of a window-open
# test - the room went BELOW it - and that is exactly the data c_out is
# measured from. The fix was once applied to the HTTP path only, leaving the
# serial and simulate paths able to reintroduce the bug.
CO2_FLOOR_PPM = 250


def reading_to_obs(ts, co2, co2_max, t, rh, n, location=settings.SCD40_LOCATION, pm=None):
    """One node-averaged reading -> Observations ('partial' if under half the samples arrived)."""
    q = "ok" if n >= 6 else "partial"
    mk = lambda m, v, u: Observation(SOURCE, m, location, ts, round(v, 2), "measured", u, quality=q)  # noqa: E731
    obs = [mk("co2_ppm", co2, "ppm"), mk("co2_max_ppm", co2_max, "ppm"),
           mk("indoor_temp_c", t, "C"), mk("indoor_rh_pct", rh, "%")]
    if pm:
        # Masses use the same metric names as the outdoor feeds, distinguished by
        # location alone, so an indoor/outdoor ratio is a join rather than a mapping.
        # The quality flag comes from the CO2 window and does not describe the fan,
        # so particles carry their own "ok": the node only sends a set it was able
        # to take a median over.
        for _k, metric in PM_MASS:
            obs.append(Observation(SOURCE, metric, location, ts, round(pm[metric], 1),
                                   "measured", "ug/m3", quality="ok"))
        for _k, metric in PM_COUNT:
            obs.append(Observation(SOURCE, metric, location, ts, pm[metric],
                                   "measured", "per 0.1 L", quality="ok"))
    return obs


def summarise(samples, ts, window_s=settings.SCD40_WINDOW_S, location=settings.SCD40_LOCATION):
    """Samples -> Observations (mean of the window, plus CO2 max). Pure function."""
    expected = max(1, window_s // SAMPLE_PERIOD_S)
    good = [s for s in samples if CO2_FLOOR_PPM <= s[0] <= 10000 and -10 <= s[1] <= 60 and 0 <= s[2] <= 100]
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
