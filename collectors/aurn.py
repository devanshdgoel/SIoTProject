"""
aurn.py

DEFRA's Automatic Urban and Rural Network: reference-grade analysers, read through
the Sensor Observation Service at uk-air.defra.gov.uk. No API key.

Why this and not the two obvious alternatives. LAQN refused every connection for
days from two separate networks. Breathe London moved host and now requires a key
(api.breathelondon-communities.org, free from their developer page) - worth adding
later for its density, since it has hundreds of nodes against AURN's handful. AURN
answers today, needs nothing, and reaches the same London reference sites, including
North Kensington.

What it is for. Open-Meteo's air quality is CAMS model output on a 0.1 degree grid,
about 11 km by 7 km at London's latitude, so a whole swathe of the city shares one
number. These are instruments. Storing both under the same metric names, differing
only by location and kind ('measured' here, 'modelled' there), makes the comparison
a query rather than a project.

One quirk worth knowing: this service reports coordinates as [latitude, longitude],
not the [longitude, latitude] GeoJSON specifies, and each pollutant at a site is its
own station. `python -m collectors.aurn` lists the nearest stations so the ids in
settings.AURN_STATIONS can be chosen deliberately.
"""

import math
import time

from collectors.base import CollectResult, CollectorError, NotConfigured, Observation, get_json
from config import settings

SOURCE = "aurn"

# Substrings of the station label -> (our metric, unit). Names match open_meteo so a
# measured series and a modelled one differ only by location.
LABELS = (
    ("particulate matter less than 2.5", ("pm2_5", "ug/m3")),
    ("non-volatile pm2.5", ("pm2_5", "ug/m3")),
    ("particulate matter less than 10", ("pm10", "ug/m3")),
    ("non-volatile pm10", ("pm10", "ug/m3")),
    ("nitrogen dioxide", ("no2", "ug/m3")),
    ("ozone", ("o3", "ug/m3")),
    ("sulphur dioxide", ("so2", "ug/m3")),
)
PLAUSIBLE = {"pm2_5": (-5, 500), "pm10": (-5, 1000), "no2": (-5, 1000),
             "o3": (-5, 500), "so2": (-5, 1000)}


def _metric(label):
    low = (label or "").lower()
    for needle, pair in LABELS:
        if needle in low:
            return pair
    return None


def _site(label):
    """'London Westminster-Particulate matter...' -> 'london_westminster'."""
    head = (label or "").split("-")[0].strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in head).strip("_")[:40] or "aurn"


def _timeseries_for(station):
    """The one timeseries a station carries, as (id, label, unit)."""
    rows = get_json(f"{settings.AURN_API_ROOT}/timeseries", params={"station": station})
    if not rows:
        return None                       # retired or not publishing; skip, do not fail
    first = rows[0]
    return (str(first["id"]), first.get("label", ""),
            (first.get("station") or {}).get("properties", {}).get("label", ""))


def collect(db=None, hours=48):
    """One pair of small requests per configured station: resolve, then fetch."""
    stations = settings.AURN_STATIONS
    if not stations:
        raise NotConfigured("AURN_STATIONS is empty; set it in .env, e.g. AURN_STATIONS=786074 "
                            "(run `python -m collectors.aurn` to list sites near home)")
    now = int(time.time())
    end = time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime(now))
    result = CollectResult(SOURCE)
    unparsed = []
    for station in stations:
        found = _timeseries_for(station)
        if found is None:
            # One dead station must not take the whole collector down with it.
            unparsed.append(f"{station} (no timeseries)")
            continue
        ts_id, _label, site_label = found
        pair = _metric(site_label)
        if pair is None:
            unparsed.append(site_label)
            continue
        metric, unit = pair
        location = _site(site_label)
        payload = get_json(f"{settings.AURN_API_ROOT}/timeseries/{ts_id}/getData",
                           params={"timespan": f"PT{int(hours)}H/{end}"})
        values = (payload or {}).get("values") or []
        for point in values:
            try:
                when = int(point["timestamp"]) // 1000
                value = float(point["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if when > now + 3600:
                continue
            lo, hi = PLAUSIBLE[metric]
            result.observations.append(Observation(
                SOURCE, metric, location, when, value, "measured", unit,
                quality="ok" if lo <= value <= hi else "suspect", retrieved_at=now))
        result.raw[f"{station}"] = {"timeseries": ts_id, "label": site_label,
                                    "n": len(values)}
    if unparsed and not result.observations:
        raise CollectorError(f"no recognised pollutant in: {unparsed}")
    return result


def backfill(days, db=None):
    """The service serves years; one request per station covers the whole window."""
    return collect(db, hours=min(int(days) * 24, 24 * 90))


def _km(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def nearest(lat=None, lon=None, n=12, metric=None):
    """Stations closest to a point, as (km, id, label) - use it to pick AURN_STATIONS."""
    lat = settings.HOME_LAT if lat is None else lat
    lon = settings.HOME_LON if lon is None else lon
    out = []
    for feature in get_json(f"{settings.AURN_API_ROOT}/stations"):
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        props = feature.get("properties") or {}
        if len(coords) < 2:
            continue
        # This service emits [lat, lon]; GeoJSON says [lon, lat]. Accept either by
        # keeping whichever ordering lands inside the British Isles.
        a, b = float(coords[0]), float(coords[1])
        slat, slon = (a, b) if (49 < a < 61 and -11 < b < 3) else (b, a)
        if not (49 < slat < 61 and -11 < slon < 3):
            continue
        label = props.get("label", "")
        if metric and (_metric(label) or ("", ""))[0] != metric:
            continue
        out.append((_km(lat, lon, slat, slon), props.get("id"), label))
    out.sort()
    return out[:n]


if __name__ == "__main__":
    print(f"AURN stations nearest {settings.HOME_LAT:.4f}, {settings.HOME_LON:.4f}")
    for dist, sid, label in nearest(metric="pm2_5"):
        print(f"  {dist:5.2f} km  id={sid:<8} {label}")
    print(f"\nconfigured AURN_STATIONS: {settings.AURN_STATIONS}")
    if settings.AURN_STATIONS:
        r = collect()
        print(f"{len(r.observations)} observations")
        for o in r.observations[-3:]:
            print(" ", o)
