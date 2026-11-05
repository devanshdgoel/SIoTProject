"""
laqn.py

London Air Quality Network: reference-grade analyser measurements from the network
run by Imperial College London's Environmental Research Group. No API key needed.

Why this sits next to open_meteo. Open-Meteo's air quality is CAMS model output on a
0.1 degree grid, which at London's latitude is a cell about 11 km north-south by
7 km east-west, so a whole swathe of the city shares one value. LAQN gives real
readings from named sites a few hundred metres away. Both are stored under the same
metric names and differ only in `location` and `kind` ('measured' here, 'modelled'
there), so the analysis can compare them directly and quantify the model's local bias.

Sites are chosen in settings.LAQN_SITES (site codes such as "KC1"); each site's data
is stored under location=<site code>. Run `python -m collectors.laqn` to list the
sites nearest to HOME_LAT / HOME_LON and check the feed is reachable.
"""

import math
import time
from datetime import datetime, timedelta, timezone

from collectors.base import (CollectResult, CollectorError, NotConfigured,
                             Observation, get_json, iso_to_unix)
from config import settings

SOURCE = "laqn"

# LAQN species code -> (our metric, unit). Names match open_meteo on purpose.
# CO is deliberately absent: LAQN reports it in mg/m3 while CAMS uses ug/m3, and
# mixing the two under one metric name would silently corrupt any comparison.
SPECIES = {
    "NO2": ("no2", "ug/m3"),
    "PM10": ("pm10", "ug/m3"),
    "PM25": ("pm2_5", "ug/m3"),
    "O3": ("o3", "ug/m3"),
    "SO2": ("so2", "ug/m3"),
}
# Stored either way, but flagged 'suspect' outside these bounds. Small negatives are
# normal analyser zero-drift near clean air, so the lower bound is not zero.
PLAUSIBLE = {"no2": (-10, 1000), "pm10": (-10, 1000), "pm2_5": (-10, 500),
             "o3": (-10, 500), "so2": (-10, 1000)}


def _norm(rec):
    """Lower-case keys with non-alphanumerics dropped: '@MeasurementDateGMT' -> 'measurementdategmt'."""
    return {"".join(ch for ch in k.lower() if ch.isalnum()): v for k, v in rec.items()}


def _records(payload, name):
    """Pull a named record list out of the response whatever it is wrapped in.

    The service is a .NET XML API rendered as JSON: keys carry an '@' prefix, the
    wrapper is called AirQualityData or RawAQData depending on the endpoint, and a
    lone record arrives as an object instead of a one-element list."""
    node = payload
    for _ in range(4):
        if not isinstance(node, dict):
            break
        got = _norm(node).get(name)
        if got is not None:
            return [got] if isinstance(got, dict) else list(got)
        inner = [v for v in node.values() if isinstance(v, dict)]
        if not inner:
            break
        node = inner[0]
    raise CollectorError(f"no '{name}' records in the LAQN response")


def _parse(rows, site, now):
    """Hourly analyser rows -> Observations. Blank values mean the analyser was down."""
    out = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        r = _norm(raw)
        metric_unit = SPECIES.get(str(r.get("speciescode") or "").upper())
        if metric_unit is None or r.get("value") in (None, ""):
            continue
        try:
            value = float(r["value"])
            ts = iso_to_unix(r.get("measurementdategmt"))
        except (KeyError, TypeError, ValueError):
            continue
        if ts is None or ts > now + 3600:
            continue
        metric, unit = metric_unit
        lo, hi = PLAUSIBLE[metric]
        out.append(Observation(SOURCE, metric, site, ts, value, "measured", unit,
                               quality="ok" if lo <= value <= hi else "suspect",
                               retrieved_at=now))
    return out


def collect(db=None, days=2):
    """One request per site. The recent window is re-requested every poll because
    LAQN rows are revised as they are ratified; storage upserts, so this self-heals."""
    sites = settings.LAQN_SITES
    if not sites:
        raise NotConfigured("LAQN_SITES is empty; set it in .env, e.g. LAQN_SITES=KC1,KC5 "
                            "(run `python -m collectors.laqn` to list sites near home)")
    now = int(time.time())
    today = datetime.fromtimestamp(now, timezone.utc)
    start = (today - timedelta(days=int(days))).strftime("%Y-%m-%d")
    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")   # EndDate is exclusive
    result = CollectResult(SOURCE)
    unparsed = set()
    for site in sites:
        payload = get_json(f"{settings.LAQN_API_ROOT}/Data/Site/SiteCode={site}"
                           f"/StartDate={start}/EndDate={end}/Json")
        rows = _records(payload, "data")
        obs = _parse(rows, site, now)
        if rows and not obs:
            unparsed |= {k for r in rows[:5] if isinstance(r, dict) for k in r}
        result.observations += obs
        result.raw[f"site_{site}"] = payload
    if unparsed and not result.observations:
        raise CollectorError(f"LAQN returned rows but none parsed; keys seen: {sorted(unparsed)}")
    return result


def backfill(days, db=None):
    """LAQN serves years of history; one request per site covers the whole window."""
    return collect(db, days=min(int(days), 365))


def _km(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def nearest(lat=None, lon=None, n=12, open_only=True):
    """Sites closest to a point as (km, code, name, type) - use this to pick LAQN_SITES."""
    lat = settings.HOME_LAT if lat is None else lat
    lon = settings.HOME_LON if lon is None else lon
    out = []
    for raw in _records(get_json(f"{settings.LAQN_API_ROOT}/Information/MonitoringSites"
                                 f"/GroupName=London/Json"), "site"):
        if not isinstance(raw, dict):
            continue
        r = _norm(raw)
        if open_only and r.get("dateclosed"):
            continue
        try:
            slat, slon = float(r["latitude"]), float(r["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if slat == 0:                       # the feed uses 0,0 for sites with no fix
            continue
        out.append((_km(lat, lon, slat, slon), r.get("sitecode"), r.get("sitename"),
                    r.get("sitetype")))
    out.sort()
    return out[:n]


if __name__ == "__main__":
    print(f"sites nearest {settings.HOME_LAT:.4f}, {settings.HOME_LON:.4f}:")
    for dist, code, name, kind in nearest():
        print(f"  {dist:5.2f} km  {code:6s} {kind or '?':18s} {name or ''}")
    print(f"\nconfigured LAQN_SITES: {settings.LAQN_SITES}")
    r = collect()
    print(f"{len(r.observations)} observations")
    for o in r.observations[:3]:
        print(o)
