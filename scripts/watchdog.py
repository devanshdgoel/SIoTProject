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

A log nobody opens is not monitoring, so it can also push to a phone - ntfy, email,
or both. Settings live in .env (a scheduled task inherits no shell, so they cannot
live in the environment); `watchdog.py --test` proves the route works before a real
failure has to. See scripts/README-watchdog.md.
"""

import io
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

def _dotenv(path):
    """KEY=VALUE pairs from a .env file.

    A scheduled task inherits no shell, so os.environ here holds almost nothing -
    which is why the notification settings cannot live in the environment alone.
    They are read from the same .env the rest of the project uses, and a real
    environment variable still wins so a one-off test can override it.
    """
    out = {}
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


CONF = _dotenv(ROOT / ".env")
CONF.update({k: v for k, v in os.environ.items() if k.startswith("SIOT_")})


def conf(key, default=""):
    return str(CONF.get(key, default)).strip()


# Route one: ntfy. A topic name is the whole configuration - no account, no key,
# no SMTP server that silently starts refusing logins. Subscribe to the same topic
# in the ntfy app and the push arrives on the phone.
NTFY_TOPIC = conf("SIOT_NTFY_TOPIC")
NTFY_URL = conf("SIOT_NTFY_URL", "https://ntfy.sh").rstrip("/")

# Route two: email, including the address-to-text gateways some carriers run. Use
# an app password, never the account password.
MAIL_TO = conf("SIOT_MAIL_TO")
MAIL_FROM = conf("SIOT_MAIL_FROM") or MAIL_TO
MAIL_HOST = conf("SIOT_MAIL_HOST", "smtp.gmail.com")
MAIL_PORT = int(conf("SIOT_MAIL_PORT", "587") or 587)
MAIL_USER = conf("SIOT_MAIL_USER") or MAIL_FROM
MAIL_PASS = conf("SIOT_MAIL_PASS")

# A watchdog that only ever speaks up on failure is indistinguishable from a
# watchdog that has itself died. One message a day saying all is well makes the
# silence mean something. Empty disables it; otherwise it is an hour, 0-23.
HEARTBEAT_HOUR = conf("SIOT_HEARTBEAT_HOUR")


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


def _push_ntfy(title, body, priority):
    if not NTFY_TOPIC:
        return None
    req = urllib.request.Request(
        f"{NTFY_URL}/{NTFY_TOPIC}", data=body.encode("utf-8"),
        headers={"Title": title, "Priority": priority,
                 "Tags": "warning" if priority == "high" else "white_check_mark"})
    urllib.request.urlopen(req, timeout=10).read()
    return "ntfy"


def _push_mail(title, body, priority):
    if not (MAIL_TO and MAIL_PASS):
        return None
    import smtplib                      # imported late: unused on the ntfy path
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = title, MAIL_FROM, MAIL_TO
    msg.set_content(body)
    with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(MAIL_USER, MAIL_PASS)
        smtp.send_message(msg)
    return "email"


def routes():
    """Which notification routes are configured, for --test to report."""
    out = []
    if NTFY_TOPIC:
        out.append(f"ntfy -> {NTFY_URL}/{NTFY_TOPIC}")
    if MAIL_TO and MAIL_PASS:
        out.append(f"email -> {MAIL_TO} via {MAIL_HOST}:{MAIL_PORT}")
    return out


def notify(title, body, priority="high"):
    """Send on every configured route. Returns the ones that worked.

    Every route is tried even after one succeeds, because the point of having two
    is that either may be the one that fails - a phone with no data connection
    still receives email over wifi later, and ntfy still arrives when an SMTP
    login has expired. And failing to notify is never fatal: the exit status and
    the log remain, so a broken notifier degrades to the behaviour before it
    existed rather than taking the watchdog down with it.
    """
    sent = []
    for fn in (_push_ntfy, _push_mail):
        try:
            got = fn(title, body, priority)
            if got:
                sent.append(got)
        except Exception as exc:                            # noqa: BLE001
            # Deliberately broad: smtplib alone raises a dozen unrelated types,
            # and none of them is worth crashing a monitor over.
            try:
                with open(LOG, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"NOTIFY-FAILED {fn.__name__}: "
                             f"{type(exc).__name__}: {exc}\n")
            except OSError:
                pass
    return sent


def _heartbeat_due(state, now):
    """True once per day, in the configured hour, so silence is informative."""
    if not HEARTBEAT_HOUR.isdigit():
        return False
    if time.localtime(now).tm_hour != int(HEARTBEAT_HOUR):
        return False
    last = state.get("heartbeat", 0)
    return (time.strftime("%Y-%m-%d", time.localtime(last))
            != time.strftime("%Y-%m-%d", time.localtime(now)))


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
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    was = state.get("alarms", {})
    changed = set(alarms) != set(was)
    beat = (not alarms) and _heartbeat_due(state, now)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {'ALARM' if alarms else 'ok'} | "
                     + "; ".join(lines) + "\n")
        STATE.write_text(json.dumps({
            "at": now, "alarms": alarms,
            "heartbeat": now if beat else state.get("heartbeat", 0),
        }), encoding="utf-8")
    except OSError:
        pass

    if alarms and changed:
        notify("Air monitor: something stopped",
               "\n".join(ln for ln in lines if "STALE" in ln
                          or "no rows" in ln or "missing" in ln))
    elif was and not alarms:
        notify("Air monitor: recovered",
               "Everything is reporting again.\n\n" + "\n".join(lines),
               priority="default")
    elif beat:
        notify("Air monitor: all well", "\n".join(lines), priority="low")

    print(f"{stamp} " + "; ".join(lines))
    return 1 if alarms else 0


def selftest():
    """`watchdog.py --test` - prove a notification actually reaches the phone.

    Worth having as a command of its own: a notifier is only ever exercised by the
    failure it exists to report, so without this the first time it runs for real is
    also the first time anyone finds out it was misconfigured.
    """
    found = routes()
    if not found:
        print("No notification route configured.\n"
              "Put SIOT_NTFY_TOPIC (easiest) or SIOT_MAIL_TO + SIOT_MAIL_PASS\n"
              f"into {ROOT / '.env'} - see scripts/README-watchdog.md")
        return 1
    for r in found:
        print("configured:", r)
    sent = notify("Air monitor: test",
                  "This is a test from the watchdog. If you are reading it on "
                  "your phone, the alerting works.", priority="default")
    print("delivered via:", ", ".join(sent) if sent else
          "NOTHING - see data/watchdog.log for the reason")
    return 0 if sent else 1


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
