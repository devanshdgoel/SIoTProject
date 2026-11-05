"""
grid_carbon.py

Retrieves current and forecast GB electricity carbon intensity (national and
the London DNO region) plus the national generation mix, from the NESO Carbon
Intensity API (no authentication).

Half-hourly periods are stored against the period START. NESO's "actual" is
computed from outturn generation, so it is labelled 'modelled' rather than
'measured'. Every poll stores the 48 h forecast with its issue time so forecast
error can be quantified later.
"""

import time
from datetime import datetime, timedelta, timezone

from collectors.base import CollectResult, CollectorError, Observation, get_json, iso_to_unix
from config import settings

SOURCE = "carbon_intensity"
FMT = "%Y-%m-%dT%H:%MZ"


def _fmt(dt):
    return dt.astimezone(timezone.utc).strftime(FMT)


def _periods(payload):
    """Flatten the national ({data:[{from,to,intensity}]}) and regional
    ({data:[{...,data:[{from,to,intensity}]}]} or {data:{...,data:[...]}}) shapes."""
    data = payload.get("data")
    if isinstance(data, dict):
        data = [data]
    out = []
    for item in data or []:
        out += item["data"] if "data" in item and isinstance(item["data"], list) else [item]
    return out


def _intensity_obs(periods, location, now, forecast_only=False):
    obs = []
    issued = now - now % 1800
    for p in periods:
        ts = iso_to_unix(p["from"])
        inten = p.get("intensity") or {}
        if inten.get("actual") is not None and not forecast_only:
            obs.append(Observation(SOURCE, "carbon_actual_gco2_kwh", location, ts,
                                   float(inten["actual"]), "modelled", "gCO2/kWh", retrieved_at=now))
        if inten.get("forecast") is not None and ts + 1800 > now:   # only periods not yet over
            obs.append(Observation(SOURCE, "carbon_forecast_gco2_kwh", location, ts,
                                   float(inten["forecast"]), "forecast", "gCO2/kWh",
                                   issued_at=issued, retrieved_at=now))
    return obs


def collect(db=None):
    now = int(time.time())
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    result = CollectResult(SOURCE)

    # National: last 24 h (actuals fill any gap) + next 48 h forecast
    past = get_json(f"{settings.CARBON_BASE}/intensity/{_fmt(start - timedelta(hours=23))}/{_fmt(start)}")
    fwd = get_json(f"{settings.CARBON_BASE}/intensity/{_fmt(start)}/fw48h")
    result.observations += _intensity_obs(_periods(past), "GB", now)
    result.observations += _intensity_obs(_periods(fwd), "GB", now)

    # London region: forecast only (regions carry no 'actual')
    reg = get_json(f"{settings.CARBON_BASE}/regional/intensity/{_fmt(start)}/fw48h/regionid/{settings.CARBON_REGION_ID}")
    result.observations += _intensity_obs(_periods(reg), settings.CARBON_REGION_NAME, now)

    # Generation mix, national, current half hour
    gen = get_json(f"{settings.CARBON_BASE}/generation")
    g = gen.get("data")
    g = g[0] if isinstance(g, list) else g
    if not g or "generationmix" not in g:
        raise CollectorError("generation response missing generationmix")
    ts = iso_to_unix(g["from"])
    for mix in g["generationmix"]:
        result.observations.append(Observation(SOURCE, f"gen_{mix['fuel']}_pct", "GB", ts,
                                               float(mix["perc"]), "modelled", "%", retrieved_at=now))
    result.raw["national_forecast"] = fwd
    result.raw["london_forecast"] = reg
    result.raw["generation"] = gen
    return result


def backfill(days, db=None):
    """Actual national intensity for the last `days` (API limit: 14 days per call)."""
    now = int(time.time())
    end = datetime.now(timezone.utc)
    result = CollectResult(SOURCE)
    remaining = min(int(days), 60)
    while remaining > 0:
        step = min(remaining, 13)
        start = end - timedelta(days=step)
        payload = get_json(f"{settings.CARBON_BASE}/intensity/{_fmt(start)}/{_fmt(end)}")
        result.observations += [o for o in _intensity_obs(_periods(payload), "GB", now)
                                if o.kind != "forecast"]
        end, remaining = start, remaining - step
    return result


if __name__ == "__main__":
    r = collect()
    kinds = {}
    for o in r.observations:
        kinds[(o.metric, o.location, o.kind)] = kinds.get((o.metric, o.location, o.kind), 0) + 1
    for k, v in sorted(kinds.items()):
        print(v, k)
