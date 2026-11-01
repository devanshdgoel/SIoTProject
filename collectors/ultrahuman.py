"""
ultrahuman.py

Retrieves personal health and recovery data from the Ultrahuman partner API.
The API returns a whole local day per call (no "now" endpoint), so each poll
fetches today and yesterday and relies on idempotent storage: intraday series
(HR, HRV, skin temperature, steps; ~6 min resolution) become observations,
the night's sleep session goes to its own table (it defines the "overnight"
window used by the analysis), and daily scalars (resting HR, recovery index...)
are stored against the start of their day.

The ring syncs via the phone, so the newest points can lag real time; the
collector never assumes a reading is "now".
"""

import json
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from collectors.base import CollectResult, CollectorError, NotConfigured, Observation, get_json
from config import settings

SOURCE = "ultrahuman"
LOCATION = "user"

SERIES = {            # metric type in API -> (our metric, unit)
    "hr": ("hr_bpm", "bpm"),
    "hrv": ("hrv_ms", "ms"),
    "temp": ("skin_temp_c", "C"),
    "steps": ("steps", "count"),
}
PLAUSIBLE = {"skin_temp_c": (25, 42), "hr_bpm": (30, 220), "hrv_ms": (3, 300)}   # outside = ring off/poor contact
DAILY = {             # metric type in API -> our metric (value lives at object.value)
    "avg_sleep_hrv": "daily_sleep_hrv_ms",
    "sleep_rhr": "daily_sleep_rhr_bpm",
    "recovery_index": "daily_recovery_index",
    "movement_index": "daily_movement_index",
    "active_minutes": "daily_active_minutes",
}


def _token():
    tok = os.getenv("ULTRAHUMAN_TOKEN")
    if not tok:
        raise NotConfigured("ULTRAHUMAN_TOKEN missing from .env")
    return tok


def fetch_day(day):
    """Raw API payload for one local date (YYYY-MM-DD)."""
    payload = get_json(settings.ULTRAHUMAN_URL, params={"date": day},
                       headers={"Authorization": _token(), "Accept": "application/json"})
    if payload.get("error") or payload.get("status") not in (None, 200):
        raise CollectorError(f"Ultrahuman error: {payload.get('error')}")
    return payload


def _sleep_row(day, obj, now):
    seg = lambda t: next((s["stage_time"] for s in obj.get("sleep_stages") or []  # noqa: E731
                          if s.get("type") == t), None)
    graph = (obj.get("sleep_graph") or {}).get("data") or []
    return {
        "day": day,
        "bedtime_start": obj.get("bedtime_start"),
        "bedtime_end": obj.get("bedtime_end"),
        "total_sleep_s": (obj.get("total_sleep") or {}).get("seconds"),
        "time_in_bed_s": (obj.get("time_in_bed") or {}).get("minutes", 0) * 60 or None,
        "deep_s": seg("deep_sleep"), "light_s": seg("light_sleep"), "rem_s": seg("rem_sleep"),
        "efficiency_pct": (obj.get("sleep_efficiency") or {}).get("percentage"),
        "sleep_score": (obj.get("sleep_score") or {}).get("score"),
        "avg_body_temp_c": (obj.get("average_body_temperature") or {}).get("celsius"),
        "toss_turns": (obj.get("tosses_and_turns") or {}).get("count"),
        "stages_json": json.dumps([{"start": g["start"], "end": g["end"], "type": g["type"]}
                                   for g in graph]),
        "retrieved_at": now,
    }


def parse_day(day, payload, now=None):
    """One day's API payload -> CollectResult (pure function; unit-testable offline)."""
    now = now or int(time.time())
    result = CollectResult(SOURCE)
    metrics = ((payload.get("data") or {}).get("metrics") or {}).get(day) or []
    for m in metrics:
        typ, obj = m.get("type"), m.get("object") or {}
        if typ in SERIES:
            metric, unit = SERIES[typ]
            for p in obj.get("values") or []:
                if p.get("value") is not None and p.get("timestamp"):
                    lo, hi = PLAUSIBLE.get(metric, (None, None))
                    v = float(p["value"])
                    suspect = lo is not None and not (lo <= v <= hi)
                    result.observations.append(Observation(
                        SOURCE, metric, LOCATION, int(p["timestamp"]), v, "measured", unit,
                        quality="suspect" if suspect else "ok", retrieved_at=now))
        elif typ in DAILY and obj.get("value") is not None:
            result.observations.append(Observation(
                SOURCE, DAILY[typ], LOCATION, int(obj["day_start_timestamp"]), float(obj["value"]),
                "measured", "", retrieved_at=now))
        elif typ == "sleep" and obj.get("bedtime_start"):
            result.sleep_sessions.append(_sleep_row(day, obj, now))
    return result


def collect(db=None, days_back=1):
    tz = ZoneInfo(settings.USER_TZ)
    today = datetime.now(tz).date()
    result = CollectResult(SOURCE)
    for i in range(days_back + 1):
        day = (today - timedelta(days=i)).isoformat()
        payload = fetch_day(day)
        part = parse_day(day, payload)
        result.observations += part.observations
        result.sleep_sessions += part.sleep_sessions
        result.raw[day] = payload
    return result


def backfill(days, db=None):
    return collect(db, days_back=int(days))


if __name__ == "__main__":
    r = collect()
    by = {}
    for o in r.observations:
        by[o.metric] = by.get(o.metric, 0) + 1
    print(by)
    print(r.sleep_sessions)
