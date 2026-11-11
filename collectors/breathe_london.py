"""
breathe_london.py

Breathe London: the GLA/Imperial network of Clarity low-cost nodes. Hundreds of
them, against AURN's handful, and the reason it is worth having is distance.

    Breathe London CLDP0444 (Science Museum)   0.17 km from the flat
    Breathe London CLDP0353 (Serpentine)       0.66 km
    AURN London N. Kensington                  ~3 km
    Open-Meteo CAMS grid cell                  ~11 x 7 km, the whole of inner London

Every outdoor number in this project has so far been borrowed from kilometres
away. A node 170 m down the road is a different class of evidence.

The trade-off is instrument quality, and it runs the other way: these are optical
low-cost sensors, calibrated against reference sites but not reference-grade, so
they are stored as 'measured' like AURN but their disagreements with AURN are
interesting rather than automatically their fault. Keeping all four sources under
the same metric names, separated only by location and kind, is what makes
`python main.py outdoor` a query rather than a project.

API, from www.breathelondon-communities.org/developers:

    GET /api/ListSensors?key=...
    GET /api/getClarityData/{SiteCode}/{species}/{start}/{end}/{averaging}?key=...
        species   IPM25 | INO2
        averaging Hourly
        dates     'Mon 11 Apr 2022 11:00:00 GMT'

Note the host moved: api.breathelondon.org now answers 403 with a pointer to
api.breathelondon-communities.org.

Site selection does NOT need the key. The GLA publishes the node inventory as an
open ArcGIS layer, so `python -m collectors.breathe_london` lists the nodes
nearest home - with their siting classification, which matters more here than it
does for AURN. 'Roadside' at 170 m is measuring a road, not the air in a bedroom;
'Urban Background' at 660 m may be the better comparison despite being further.
"""

import math
import time
import urllib.parse

from collectors.base import (CollectResult, CollectorError, NotConfigured,
                             Observation, get_json)
from config import settings

SOURCE = "breathe_london"

# Their species code -> (our metric, unit). Names match open_meteo and aurn.
SPECIES = {
    "IPM25": ("pm2_5", "ug/m3"),
    "INO2": ("no2", "ug/m3"),
}
PLAUSIBLE = {"pm2_5": (-5, 500), "no2": (-5, 1000)}

# The open inventory, no key required. Coordinates come back in whatever the
# layer is projected in, so ask for WGS84 explicitly.
GLA_LAYER = ("https://gis.london.gov.uk/arcgis/rest/services/apps/"
             "breathe_london_sensors/FeatureServer/0/query")


def _stamp(unix):
    """Their date format: 'Mon 11 Apr 2022 11:00:00 GMT'."""
    return time.strftime("%a %d %b %Y %H:%M:%S GMT", time.gmtime(unix))


def _site_key(code, name=None):
    """'CLDP0444' + 'Science Museum' -> 'bl_cldp0444_science_museum'."""
    bits = [str(code or "").lower()]
    if name:
        bits.append("".join(ch if ch.isalnum() else "_" for ch in name.lower()))
    return ("bl_" + "_".join(b.strip("_") for b in bits if b))[:60]


def collect(db=None, hours=48):
    """One request per site per species. Small windows, so no paging to worry about."""
    key = settings.BREATHE_LONDON_KEY
    if not key:
        raise NotConfigured(
            "BREATHE_LONDON_KEY is not set; register at "
            "www.breathelondon-communities.org/developers and put the key in .env")
    sites = settings.BREATHE_LONDON_SITES
    if not sites:
        raise NotConfigured(
            "BREATHE_LONDON_SITES is empty; run `python -m collectors.breathe_london` "
            "to list the nodes near home and choose by distance AND classification")

    now = int(time.time())
    start, end = _stamp(now - int(hours) * 3600), _stamp(now)
    result = CollectResult(SOURCE)
    refused = []

    for site in sites:
        for species, (metric, unit) in SPECIES.items():
            path = "/".join(urllib.parse.quote(str(p)) for p in
                            (site, species, start, end, "Hourly"))
            payload = get_json(f"{settings.BREATHE_LONDON_ROOT}/getClarityData/{path}",
                               params={"key": key})
            # The API answers 200 with {"error": ...} rather than a status code, so
            # a rejected key looks exactly like an empty result unless we look.
            if isinstance(payload, dict) and payload.get("error"):
                refused.append(f"{site}/{species}: {payload['error']}")
                continue
            rows = payload if isinstance(payload, list) else []
            location = _site_key(site)
            kept = 0
            for row in rows:
                try:
                    when = int(_iso(row["DateTime"]))
                    value = float(row["ScaledValue"])
                except (KeyError, TypeError, ValueError):
                    continue
                if when > now + 3600:
                    continue
                lo, hi = PLAUSIBLE[metric]
                result.observations.append(Observation(
                    SOURCE, metric, location, when, value, "measured", unit,
                    quality="ok" if lo <= value <= hi else "suspect",
                    retrieved_at=now))
                kept += 1
            result.raw[f"{site}/{species}"] = {"n": kept}

    if refused and not result.observations:
        raise CollectorError("; ".join(refused[:3]))
    return result


def _iso(text):
    """Their DateTime is ISO 8601 UTC; tolerate a trailing Z or an offset."""
    from collectors.base import iso_to_unix
    return iso_to_unix(text)


def backfill(days, db=None):
    return collect(db, hours=min(int(days) * 24, 24 * 90))


def _km(lat1, lon1, lat2, lon2):
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def nearest(lat=None, lon=None, n=12):
    """Nodes closest to a point, from the OPEN inventory - no API key needed.

    Returns (km, sensor_id, site_name, classification, pm25_2022). The annual
    mean is the layer's own 2022 figure: not live, but a useful sanity check on
    whether a node's long-run level is credible before trusting its hourly data.
    """
    lat = settings.HOME_LAT if lat is None else lat
    lon = settings.HOME_LON if lon is None else lon
    payload = get_json(GLA_LAYER, params={
        "where": "1=1", "outSR": "4326", "f": "json",
        "outFields": "sensor_id,site_name,site_type,site_classification,pm25_2022,no2_2022"})
    out = []
    for feature in (payload or {}).get("features", []):
        geom, attrs = feature.get("geometry") or {}, feature.get("attributes") or {}
        if geom.get("y") is None or geom.get("x") is None:
            continue
        out.append((_km(lat, lon, float(geom["y"]), float(geom["x"])),
                    attrs.get("sensor_id"), attrs.get("site_name"),
                    attrs.get("site_classification"), attrs.get("pm25_2022")))
    out.sort(key=lambda row: row[0])
    return out[:n]


def verify():
    """Is the key live yet? Prints a verdict, and says what it has ruled out.

    Worth being a command rather than a note in a commit message, because the
    answer changes without anything here changing: a key that their service has
    not activated yet becomes a key that works, with no edit on this side. Run it
    again rather than re-deriving the diagnosis.

    It sends exactly one request, by the transport their documentation specifies.
    The earlier investigation established what is NOT worth re-testing: the key is
    39 characters with Google's usual key prefix (well formed, not truncated); the
    rejection carries `x-powered-by: Express`, so it comes from Breathe London's
    own application rather than Google's gateway; the same message comes back for
    a request with no key at all, so the text says nothing about transport; and
    `key`, `apikey`, `api_key`, `API_Key` and an `x-api-key` header were all
    refused identically. That leaves the key not being active on their side.
    """
    key = settings.BREATHE_LONDON_KEY
    if not key:
        print("no BREATHE_LONDON_KEY in .env")
        return False
    print(f"key: {len(key)} chars, starts {key[:6]!r} "
          f"({'well formed' if len(key) == 39 else 'UNEXPECTED LENGTH - 39 expected'})")
    payload = get_json(f"{settings.BREATHE_LONDON_ROOT}/ListSensors", params={"key": key})
    if isinstance(payload, list) and payload:
        print(f"LIVE: {len(payload)} nodes returned. Add breathe_london to COLLECTORS "
              f"in .env and run `python main.py once breathe_london`.")
        return True
    err = payload.get("error") if isinstance(payload, dict) else payload
    print(f"REFUSED: {err}")
    print("Their own application is rejecting the key value - transport is correct and")
    print("every alternative was tried. Nothing on this side can fix it. Either the key")
    print("is not activated yet (new registrations are not always immediate), or it was")
    print("issued for something other than this API. If it is still refused after a day,")
    print("helpdesk@breathelondon.org, quoting the first six characters only and the URL")
    print(f"  {settings.BREATHE_LONDON_ROOT}/ListSensors?key=...")
    return False


if __name__ == "__main__":
    print("--- is the key live? ---")
    verify()
    print()
    print(f"Breathe London nodes nearest {settings.HOME_LAT:.4f}, {settings.HOME_LON:.4f}")
    print(f"{'km':>6}  {'id':<11} {'site':<32} {'classification':<18} {'pm25 2022':>9}")
    for dist, sid, name, cls, pm in nearest():
        pm = f"{pm:.1f}" if isinstance(pm, (int, float)) else "-"
        print(f"{dist:6.2f}  {str(sid):<11} {str(name)[:32]:<32} {str(cls)[:18]:<18} {pm:>9}")
    print(f"\nconfigured BREATHE_LONDON_SITES: {settings.BREATHE_LONDON_SITES}")
    print(f"key present: {bool(settings.BREATHE_LONDON_KEY)}")
    if settings.BREATHE_LONDON_KEY and settings.BREATHE_LONDON_SITES:
        try:
            r = collect(hours=24)
            print(f"{len(r.observations)} observations")
            for o in r.observations[-3:]:
                print(" ", o)
        except (CollectorError, NotConfigured) as exc:
            print("collect failed:", exc)
