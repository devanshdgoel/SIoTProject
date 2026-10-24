"""
Fetch the animal's GPS history from Movebank and reduce it to ONE ROW PER DAY.

Run:  python init_setup/animal_location_history.py [individual_id] [months_back]
Out:  init_setup/animal_location_history.csv
"""

import csv
import io
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "collection"))
import animal

OUT = HERE / "animal_location_history.csv"

INDIVIDUAL_ID = sys.argv[1] if len(sys.argv) > 1 else animal.ANIMAL_ID
MONTHS_BACK = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_events(start, end):
    params = {
        "entity_type": "event",
        "study_id": animal.STUDY_ID,
        "individual_id": INDIVIDUAL_ID,
        "sensor_type": "gps",
        "timestamp_start": start.strftime("%Y%m%d%H%M%S") + "000",
        "timestamp_end": end.strftime("%Y%m%d%H%M%S") + "000",
        "api-token": animal.get_movebank_token(),
    }
    response = requests.get(animal.MOVEBANK_BASE, params=params, timeout=180)
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows:
        raise ValueError("Movebank returned no GPS events for this individual/window.")
    return rows


def to_points(rows, start, end):
    """Keep only rows that have a real position and a timestamp inside the window."""
    ts_field = next((f for f in rows[0] if "timestamp" in f.lower()), None)
    points = []
    for row in rows:
        try:
            lat = float(row["location_lat"])
            lon = float(row["location_long"])
        except (KeyError, TypeError, ValueError):
            continue
        raw = (row.get(ts_field) or "")[:19]
        try:
            when = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if start <= when <= end:
            points.append((when, lat, lon))
    points.sort()
    return points


def to_daily(points):
    """One row per UTC day: mean position, fix count, and the day's travelled path."""
    days = {}
    for when, lat, lon in points:
        days.setdefault(when.date(), []).append((when, lat, lon))

    rows = []
    for day in sorted(days):
        fixes = days[day]
        lats = [f[1] for f in fixes]
        lons = [f[2] for f in fixes]
        path_km = sum(
            haversine_km(fixes[i - 1][1], fixes[i - 1][2], fixes[i][1], fixes[i][2])
            for i in range(1, len(fixes))
        )
        rows.append({
            "date": day.isoformat(),
            "lat": round(sum(lats) / len(lats), 6),
            "lon": round(sum(lons) / len(lons), 6),
            "n_fixes": len(fixes),
            "daily_path_km": round(path_km, 3),
            "first_ts": fixes[0][0].isoformat(),
            "last_ts": fixes[-1][0].isoformat(),
        })
    return rows


def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * MONTHS_BACK)
    print(f"individual {INDIVIDUAL_ID}, window {start.date()} .. {end.date()}")

    rows = fetch_events(start, end)
    print(f"raw event rows: {len(rows)}")

    points = to_points(rows, start, end)
    print(f"usable GPS fixes: {len(points)}")
    if not points:
        raise ValueError("No fixes with positions in this window.")

    daily = to_daily(points)
    print(f"days covered: {len(daily)}  ({daily[0]['date']} .. {daily[-1]['date']})")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(daily[0].keys()))
        writer.writeheader()
        writer.writerows(daily)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
