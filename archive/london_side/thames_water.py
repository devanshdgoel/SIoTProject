"""
thames_water.py

Retrieves river storm-overflow/discharge data, including event times,
locations and status where provided by the source.

Endpoints (Thames Water Open Data, header auth with client_id / client_secret from
the Thames Water API portal; set THAMES_CLIENT_ID and THAMES_CLIENT_SECRET in .env):
  GET {root}/DischargeCurrentStatus   live status per monitored overflow
  GET {root}/DischargeAlerts          alert stream (Start/Stop/Offline...), filterable by
                                      ?col_1=DateTime&operand_1=gt&value_1=YYYY-MM-DD,
                                      max 1000 records per call

Field names were taken from a third-party write-up of the API and the portal's docs
could not be fetched without an account, so parsing is deliberately tolerant
(case/underscore-insensitive keys) and the raw payload of every poll is stored.
Verify the schema against the first real response (data/samples/) before trusting it.

A record means "the monitor reports the overflow is discharging". It is not a
measurement of pollution volume or concentration.
"""

import os
import time
from datetime import datetime, timedelta, timezone

from collectors.base import CollectResult, CollectorError, NotConfigured, Observation, get_json, iso_to_unix
from config import settings

SOURCE = "thames_water"
LOCATION = "thames_region"
PAGE_LIMIT = 1000

_transformer = None


def _headers():
    cid, sec = os.getenv("THAMES_CLIENT_ID"), os.getenv("THAMES_CLIENT_SECRET")
    if not cid or not sec:
        raise NotConfigured("THAMES_CLIENT_ID / THAMES_CLIENT_SECRET missing from .env "
                            "(register at the Thames Water API portal)")
    return {"client_id": cid, "client_secret": sec, "Accept": "application/json"}


def _norm(rec):
    """Lower-case keys with non-alphanumerics removed: 'Location_Name' -> 'locationname'."""
    return {"".join(ch for ch in k.lower() if ch.isalnum()): v for k, v in rec.items()}


def _ts(v):
    try:
        return iso_to_unix(v) if v else None
    except ValueError:
        return None


def _latlon(x, y):
    """British National Grid (EPSG:27700) -> WGS84, or (None, None)."""
    global _transformer
    try:
        if _transformer is None:
            from pyproj import Transformer
            _transformer = Transformer.from_crs(27700, 4326, always_xy=True)
        lon, lat = _transformer.transform(float(x), float(y))
        return round(lat, 5), round(lon, 5)
    except Exception:  # noqa: BLE001 - missing/garbled coordinates are non-fatal
        return None, None


def _items(payload):
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise CollectorError(f"unexpected Thames Water payload shape: {type(payload).__name__}")
    return items


def parse_status(payload, now=None):
    """DischargeCurrentStatus payload -> CollectResult (pure; unit-testable offline)."""
    now = now or int(time.time())
    result = CollectResult(SOURCE)
    counts = {"discharging": 0, "offline": 0, "not discharging": 0}
    for raw in _items(payload):
        r = _norm(raw)
        status = str(r.get("alertstatus") or "").strip()
        name = r.get("locationname")
        if not status or not name:
            continue
        counts[status.lower()] = counts.get(status.lower(), 0) + 1
        lat, lon = _latlon(r.get("x"), r.get("y"))
        result.overflow_status.append({
            "location_name": name, "alert_status": status,
            "status_change": _ts(r.get("statuschange")) or 0,
            "last_start": _ts(r.get("mostrecentdischargealertstart")),
            "last_stop": _ts(r.get("mostrecentdischargealertstop")),
            "water_course": r.get("receivingwatercourse"),
            "lat": lat, "lon": lon, "first_seen_at": now,
        })
    total = sum(counts.values())
    if total == 0:
        raise CollectorError("Thames Water status payload contained no usable records")
    for metric, key in (("overflows_discharging", "discharging"), ("overflows_offline", "offline"),
                        ("overflows_monitored", None)):
        value = counts.get(key, 0) if key else total
        result.observations.append(Observation(SOURCE, metric, LOCATION, now, float(value),
                                               "measured", "count", retrieved_at=now))
    return result


def parse_alerts(items):
    out = []
    for raw in items:
        r = _norm(raw)
        ts = _ts(r.get("datetime"))
        if r.get("locationname") and r.get("alerttype") and ts:
            out.append({"location_name": r["locationname"], "alert_type": r["alerttype"], "event_ts": ts})
    return out


def fetch_alerts_since(since_ts, max_pages=10):
    """Page through DischargeAlerts newer than since_ts (1000 records per call)."""
    out, cursor = [], int(since_ts)
    for _ in range(max_pages):
        day = datetime.fromtimestamp(cursor, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        payload = get_json(f"{settings.THAMES_API_ROOT}/DischargeAlerts", headers=_headers(),
                           params={"col_1": "DateTime", "operand_1": "gt", "value_1": day})
        items = _items(payload)
        page = parse_alerts(items)
        out += page
        if len(items) < PAGE_LIMIT or not page:
            break
        newest = max(a["event_ts"] for a in page)
        if newest <= cursor:
            break
        cursor = newest
    return out


def collect(db=None, alert_days=7):
    headers = _headers()
    payload = get_json(f"{settings.THAMES_API_ROOT}/DischargeCurrentStatus", headers=headers)
    result = parse_status(payload)
    result.raw["status"] = payload if len(str(payload)) < 5_000_000 else {"truncated": True}
    last = db.max_alert_ts() if db is not None else None
    since = last if last else int((datetime.now(timezone.utc) - timedelta(days=alert_days)).timestamp())
    result.overflow_alerts = fetch_alerts_since(since)
    return result


def backfill(days, db=None):
    result = CollectResult(SOURCE)
    result.overflow_alerts = fetch_alerts_since(
        int((datetime.now(timezone.utc) - timedelta(days=int(days))).timestamp()), max_pages=50)
    return result


if __name__ == "__main__":
    r = collect()
    print(r.observations)
    print(len(r.overflow_status), "assets,", len(r.overflow_alerts), "alerts")
