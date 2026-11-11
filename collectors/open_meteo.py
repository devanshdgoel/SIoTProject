"""
open_meteo.py

Retrieves outdoor weather and air-quality data from Open-Meteo for each configured
location. One poll returns hourly arrays covering the last two days (model
analysis, kind='modelled') and the next FORECAST_HORIZON_H hours (kind='forecast',
stamped with the poll hour as issued_at). Re-fetching the past window every poll
makes the collector self-healing after downtime, and keeping issued forecasts
allows forecast-vs-actual verification in the evaluation week.

Air-quality values are CAMS model output, not station measurements.
"""

import time

from collectors.base import CollectResult, Observation, get_json
from config import settings

SOURCE = "open_meteo"

# open-meteo variable -> (our metric, unit)
WEATHER_VARS = {
    "temperature_2m": ("outdoor_temp_c", "C"),
    "relative_humidity_2m": ("outdoor_rh_pct", "%"),
    "dew_point_2m": ("outdoor_dewpoint_c", "C"),
    "precipitation": ("precip_mm", "mm"),
    "surface_pressure": ("pressure_hpa", "hPa"),
    "wind_speed_10m": ("wind_ms", "m/s"),
    "wind_direction_10m": ("wind_dir_deg", "deg"),
    "cloud_cover": ("cloud_pct", "%"),
}
AIR_VARS = {
    "pm2_5": ("pm2_5", "ug/m3"),
    "pm10": ("pm10", "ug/m3"),
    "nitrogen_dioxide": ("no2", "ug/m3"),
    "ozone": ("o3", "ug/m3"),
    "sulphur_dioxide": ("so2", "ug/m3"),
    "carbon_monoxide": ("co", "ug/m3"),
    # Modelled outdoor CO2, in ppm. Collected for one reason: it is the only
    # independent check on the outdoor baseline the room's own decays fit to.
    # That baseline came out at 358 ppm, which is below the global atmospheric
    # background and 87 ppm below the LOWEST hour this model has ever reported
    # here - so one of the two is wrong, and until this series existed there was
    # no way to tell. It is a model on an 11 km cell, not a measurement, so its
    # floor is the useful part rather than its hour-to-hour value.
    "carbon_dioxide": ("co2_outdoor_ppm", "ppm"),
    "european_aqi": ("european_aqi", "index"),
}


def _parse(payload, varmap, location, now):
    """Hourly arrays -> Observations. Past/current hour = modelled, future = forecast."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    issued = now - now % 3600
    horizon = issued + settings.FORECAST_HORIZON_H * 3600
    out = []
    for var, (metric, unit) in varmap.items():
        for t, v in zip(times, hourly.get(var) or []):
            if v is None or t > horizon:
                continue
            fc = t > issued
            out.append(Observation(SOURCE, metric, location, int(t), float(v),
                                   "forecast" if fc else "modelled", unit,
                                   issued_at=issued if fc else 0, retrieved_at=now))
    return out


def collect(db=None, past_days=2):
    now = int(time.time())
    result = CollectResult(SOURCE)
    for name, (lat, lon) in settings.LOCATIONS.items():
        common = {"latitude": lat, "longitude": lon, "timeformat": "unixtime",
                  "past_days": past_days, "forecast_days": 2}
        weather = get_json(settings.OPEN_METEO_WEATHER, {
            **common, "hourly": ",".join(WEATHER_VARS), "wind_speed_unit": "ms"})
        air = get_json(settings.OPEN_METEO_AIR, {**common, "hourly": ",".join(AIR_VARS)})
        result.observations += _parse(weather, WEATHER_VARS, name, now)
        result.observations += _parse(air, AIR_VARS, name, now)
        result.raw[f"weather_{name}"] = weather
        result.raw[f"air_{name}"] = air
    return result


def backfill(days, db=None):
    """Open-Meteo serves up to ~3 months of history in one call."""
    return collect(db, past_days=min(int(days), 92))


if __name__ == "__main__":
    r = collect()
    print(f"{len(r.observations)} observations")
    for o in r.observations[:3]:
        print(o)
