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

STUDY_ID = "24442409"
ANIMAL_ID = ""

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

def inspect_movebank_response(token):
    params = {
        "entity_type": "event",
        "study_id": STUDY_ID,
        "individual_local_identifier": ANIMAL_ID,
        "sensor_type": "gps",
        "api-token": token,
    }

    response = requests.get(MOVEBANK_BASE, params=params, timeout = 60)
    response.raise_for_status()

    print("STATUS: ", response.status_code)
    print ("CONTEXT-TYPE:", response.headers.get("Content-Type"))
    print("FIRST 3000 CHARACTERS: ")
    print(response.text[:3000])

def get_animal_id_from_tag(token, tag_id): 
    params ={ 
        "entity_type": "deployment", 
        "study_id": STUDY_ID,
        "api-token": token, 
    }

    response = requests.get(MOVEBANK_BASE, params=params, timeout=60)
    response.raise_for_status()
    
    print("RAW RESPONSE: ")
    print(response.text[:3000])

    rows = list(csv.DictReader(io.StringIO(response.text)))

    if not rows:
        raise ValueError(f"No deployment found for tag ID {tag_id}.")

    print("Deployment columns:", list(rows[0].keys()))

    for row in rows:
        print(row)

        animal_id = (
            row.get("individual_id")
            or row.get("individual_local_identifier")
            or row.get("local_identifier")
        )

        if animal_id:
            print(f"Tag {tag_id} belongs to animal ID: {animal_id}")
            return str(animal_id)

    raise ValueError("Deployment row did not contain an individual ID.")

def list_individuals(token):
    params = {
        "entity_type": "individual",
        "study_id": STUDY_ID,
        "api-token": token,
    }
    response = requests.get(MOVEBANK_BASE, params=params, timeout=60)
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    for row in rows:
        # the animal's name is in local_identifier; its numeric id in id
        print(row.get("id"), "|", row.get("local_identifier"))
    return rows



def find_live_individuals(token, days=2):
    """List individuals reporting in the last `days`, newest first, with names."""

    # --- 1. get a NAME map: individual_id -> readable name --------------------
    # (names live in the 'individual' table, not in event records)
    ind_params = {
        "entity_type": "individual",
        "study_id": STUDY_ID,
        "api-token": token,
    }
    ind_rows = list(csv.DictReader(io.StringIO(
        requests.get(MOVEBANK_BASE, params=ind_params, timeout=60).text
    )))
    name_of = {row["id"]: row.get("local_identifier", "") for row in ind_rows}

    # --- 2. get RECENT events, keyed by numeric individual_id ----------------
    start = datetime.now(timezone.utc) - timedelta(days=days)
    ev_params = {
        "entity_type": "event",
        "study_id": STUDY_ID,
        "timestamp_start": start.strftime("%Y%m%d%H%M%S000"),
        "attributes": "individual_id,timestamp,location_lat,location_long",
        "api-token": token,
    }
    ev_rows = list(csv.DictReader(io.StringIO(
        requests.get(MOVEBANK_BASE, params=ev_params, timeout=120).text
    )))

    # --- 3. keep the newest point per individual_id --------------------------
    latest = {}
    for row in ev_rows:
        iid = row.get("individual_id")
        ts = row.get("timestamp")
        if not iid or not ts:
            continue
        if iid not in latest or ts > latest[iid]["timestamp"]:
            latest[iid] = {"timestamp": ts,
                           "lat": row.get("location_lat"),
                           "lon": row.get("location_long")}

    # --- 4. print newest first, with the name joined in ----------------------
    print(f"\n{len(latest)} individuals reporting in the last {days} days:\n")
    for iid, d in sorted(latest.items(), key=lambda kv: kv[1]["timestamp"], reverse=True):
        name = name_of.get(iid, "?")
        print(f"{d['timestamp']}  id={iid}  {name}  ({d['lat']}, {d['lon']})")

    return latest

def get_latest_animal_location(token):
    params = {
        "entity_type": "event",
        "study_id": STUDY_ID,
        "individual_id": "4003322822",
        "sensor_type": "gps",
        "api-token": token,
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

#print ("TOKEN=" + get_movebank_token())
#print(get_latest_animal_location(get_movebank_token()))

if __name__ == "__main__":
    token = get_movebank_token()

    #animal_id = get_animal_id_from_tag(token, "560050660")

    #print("ANIMAL ID:" , animal_id)

    #ANIMAL_ID = animal_id

    #print("Indivudials: ", list_individuals(token))
    #print(find_live_individuals(token))
    print(get_latest_animal_location(token))
