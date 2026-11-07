#!/usr/bin/env python3
"""Notice when the collector has stopped, without depending on the collector.

On 7 November the collector died at 00:10 and nobody knew for ten hours. Nothing
crashed and nothing logged: the process simply stopped storing readings while the
terminal scrolled on, and the only reason it came to light is that a human happened
to ask why the overnight data looked thin. The ESP32 ring buffers happened to hold
the night and replayed it on reconnect, so nothing was lost that time - but the
bedroom node only buffers 24 hours, so a weekend would have cost a weekend.

The rule this is built on: a watchdog must not share a failure mode with the thing
it watches. So it is a separate process, started by the OS rather than by the app,
and it depends on nothing but the database FILE - the one artefact that outlives
whatever died and carries a timestamp. Standard library only, no project imports,
read-only connection, so it cannot block the writer, cannot corrupt anything, and
cannot itself be broken by an edit to the analysis code.

Run it from Windows Task Scheduler every five minutes:

    schtasks /create /sc minute /mo 5 /tn SIoTWatchdog ^
      /tr "pythonw <repo>\\scripts\\watchdog.py"

or from cron:

    */5 * * * * /usr/bin/python3 <repo>/scripts/watchdog.py

Exit status is 0 when everything is fresh and 1 when anything is stale, so a task
runner that reports failures will surface it without any of the below.
"""

import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "siot.db"
STATE = ROOT / "data" / "watchdog.state.json"
LOG = ROOT / "data" / "watchdog.log"

# Three questions, deliberately separate, because they have different answers.
#   readings  - is anything arriving from the sensors?
#   collector - is the collect process alive at all?
#   analysis  - is the analysis thread alive, while collection carries on?
# The third is the one nothing else could ever notice: parameters would simply
# stop being re-fitted and keep serving their last values for ever.
CHECKS = (
    ("collector", 600,
     "SELECT CAST(MAX(started_at) AS INTEGER) FROM collector_runs",
     "the collect process has not run"),
    ("analysis", 1800,
     "SELECT CAST(MAX(started_at) AS INTEGER) FROM analysis_runs",
     "the analysis pass has not run"),
)

# Set these to be told on a phone. Without them the watchdog still logs and still
# exits non-zero; it just cannot reach you while the laptop is closed.
NTFY_TOPIC = os.environ.get("SIOT_NTFY_TOPIC", "").strip()
NTFY_URL = os.environ.get("SIOT_NTFY_URL", "https://ntfy.sh").rstrip("/")


SENSOR_STALE_S = 900        # a node posting every minute is late after fifteen


def _per_sensor(conn):
    """(location, seconds since its newest reading) for every indoor sensor.

    Locations are discovered from the data rather than configured, so a node added
    later is watched automatically and one removed stops being watched once its
    last row ages out of the window below.
    """
    import time as _t
    now = int(_t.time())
    try:
        rows = conn.execute(
            "SELECT location, CAST(MAX(ts) AS INTEGER) FROM observations "
            "WHERE source='scd40' AND ts > ? GROUP BY location",
            (now - 7 * 86400,)).fetchall()
    except sqlite3.Error:
        return []
    return [(r[0], now - int(r[1])) for r in rows if r[1] is not None]


def age_of(conn, sql):
    """Seconds since the newest row this query finds, or None if there are none."""
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error:
        return None
    if not row or row[0] is None:
        return None
    return int(time.time()) - int(row[0])


def notify(title, body):
    """One push. Failure to notify must never be fatal - the log is the fallback."""
    if not NTFY_TOPIC:
        return False
    try:
        req = urllib.request.Request(
            f"{NTFY_URL}/{NTFY_TOPIC}", data=body.encode("utf-8"),
            headers={"Title": title, "Priority": "high"})
        urllib.request.urlopen(req, timeout=10).read()
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main():
    now = int(time.time())
    lines, alarms = [], {}

    if not DB.exists():
        lines.append(f"database missing at {DB}")
        alarms["database"] = "missing"
    else:
        # mode=ro: this process can never write, lock or vacuum the live database.
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
        try:
            # Each sensor separately. The first version of this took a global
            # MAX(ts) over every scd40 row, which one healthy node keeps fresh for
            # ever - so the bedroom node went quiet for half an hour and the
            # watchdog reported everything fine, which is precisely the failure it
            # exists to catch. A node is only as watched as its own newest row.
            for loc, age in _per_sensor(conn):
                if age is None:
                    continue
                if age > SENSOR_STALE_S:
                    lines.append(f"sensor {loc}: STALE - nothing for "
                                 f"{age // 60} min (limit {SENSOR_STALE_S // 60})")
                    alarms[f"sensor:{loc}"] = f"{age // 60}min"
                else:
                    lines.append(f"sensor {loc}: ok ({age // 60} min)")

            for name, limit, sql, phrase in CHECKS:
                age = age_of(conn, sql)
                if age is None:
                    lines.append(f"{name}: no rows at all")
                    alarms[name] = "empty"
                elif age > limit:
                    lines.append(f"{name}: STALE - {phrase} for {age // 60} min "
                                 f"(limit {limit // 60} min)")
                    alarms[name] = f"{age // 60}min"
                else:
                    lines.append(f"{name}: ok ({age // 60} min)")
        finally:
            conn.close()

    # Alarm only on a CHANGE of state. A watchdog that repeats itself every five
    # minutes is muted within a day, and a muted watchdog is worse than none
    # because it looks like cover.
    try:
        was = json.loads(STATE.read_text(encoding="utf-8")).get("alarms", {})
    except (OSError, ValueError):
        was = {}
    changed = set(alarms) != set(was)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {'ALARM' if alarms else 'ok'} | "
                     + "; ".join(lines) + "\n")
        STATE.write_text(json.dumps({"at": now, "alarms": alarms}), encoding="utf-8")
    except OSError:
        pass

    if alarms and changed:
        notify("SIoT collector", "\n".join(ln for ln in lines if "STALE" in ln
                                           or "no rows" in ln or "missing" in ln))
    elif was and not alarms:
        notify("SIoT collector", "recovered: everything is reporting again")

    print(f"{stamp} " + "; ".join(lines))
    return 1 if alarms else 0


if __name__ == "__main__":
    sys.exit(main())
