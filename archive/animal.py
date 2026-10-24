import os
import csv 
import io 
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import requests
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

MOVEBANK_USERNAME = os.getenv("MOVEBANK_USERNAME")
MOVEBANK_PASSWORD = os.getenv("MOVEBANK_PASSWORD")

STUDY_ID = "24442409" #Chosen Study - LifeTrack White Stork Bavaria - https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study24442409
ANIMAL_ID = "4003322822" # Marina

MOVEBANK_BASE = "https://www.movebank.org/movebank/service/direct-read"


def get_movebank_token(): 
    response = requests.get(
        f"{MOVEBANK_BASE}?service=request-token",
        auth=(MOVEBANK_USERNAME, MOVEBANK_PASSWORD),
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return data["api-token"]

def get_latest_animal_location():
    params = {
        "entity_type": "event",
        "study_id": STUDY_ID,
        "individual_id": "4003322822",
        "sensor_type": "gps",
        "api-token": get_movebank_token(),
    }

    response = requests.get(MOVEBANK_BASE, params=params, timeout=60)
    response.raise_for_status()

    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows: 
        raise ValueError("No GPS events were returned for this animal or study")

    timestamp_field = next(
        (field for field in rows[0] if "timestamp" in field.lower()),
        None
    )

    if timestamp_field:
        rows.sort(key=lambda row: row[timestamp_field] or "")

    latest = rows[-1]

    return {
        "latitude": (latest["location_lat"]),
        "longitude": (latest["location_long"]),
        "timestamp": latest.get(timestamp_field),
        "individual_id": latest.get("individual_local_identifier", ANIMAL_ID),
    }

if __name__ == "__main__":
    print(get_latest_animal_location())