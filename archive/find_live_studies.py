"""
find_live_studies.py

This script will help find studies on Movebank that are: 
a) downloadable by me,
(b) still reporting live recent data
(c) migratory birds as they would move aorund — then prints a ranked shortlist.

"""

import os
import csv
import io
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- load credentials from .env (sits at the project root) ------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
USERNAME = os.getenv("MOVEBANK_USERNAME")
PASSWORD = os.getenv("MOVEBANK_PASSWORD")

MOVEBANK_BASE = "https://www.movebank.org/movebank/service/direct-read"

# How recent must the study's last location be, to count as "live"?
RECENT_DAYS = 60

# Migratory-bird genera to match in the taxon_ids column. Add/remove freely.
BIRD_GENERA = [
    "Ciconia",   # storks
    "Cuculus",   # cuckoos
    "Pandion",   # ospreys
    "Circus",    # harriers
    "Grus",      # cranes
    "Anser", "Branta",   # geese
    "Milvus",    # kites
    "Falco",     # falcons
    "Aquila",    # eagles
    "Numenius",  # curlews
    "Larus",     # gulls
    "Calidris",  # sandpipers
]


def get_token():
    """Request a short-lived API token using your username/password."""
    r = requests.get(
        f"{MOVEBANK_BASE}?service=request-token",
        auth=(USERNAME, PASSWORD),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["api-token"]


def get_all_studies(token):
    """
    Ask Movebank for EVERY study it knows about (one row each). Returns a list
    of dict rows. This is a big download (thousands of studies) but one call.
    """
    params = {"entity_type": "study", "api-token": token}
    r = requests.get(MOVEBANK_BASE, params=params, timeout=180)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    print(f"Movebank returned {len(rows)} studies total.")
    # Show the available columns once, so you can see what you can filter on.
    if rows:
        print("Columns available:", list(rows[0].keys()))
    return rows


def parse_ts(text):
    """
    Parse Movebank's timestamp format 'yyyy-MM-dd HH:mm:ss.SSS' into a
    timezone-aware datetime (UTC). Returns None if empty/unparseable, so the
    filter can simply skip studies with no last-location date.
    """
    if not text:
        return None
    # Movebank sometimes omits the milliseconds; try both forms.
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def is_bird(taxon_ids):
    """True if the study's taxon list contains any of our migratory genera."""
    if not taxon_ids:
        return False
    return any(genus.lower() in taxon_ids.lower() for genus in BIRD_GENERA)


def main():
    token = get_token()
    studies = get_all_studies(token)

    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    candidates = []

    for s in studies:
        # 1) must be downloadable by you
        if s.get("i_have_download_access", "").lower() != "true":
            continue
        # 2) must have a recent last location
        last = parse_ts(s.get("timestamp_last_deployed_location", ""))
        if last is None or last < cutoff:
            continue
        # 3) must be a migratory bird
        if not is_bird(s.get("taxon_ids", "")):
            continue
        candidates.append((last, s))

    # newest first
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    print(f"\n=== {len(candidates)} LIVE migratory-bird studies you can access ===\n")
    for last, s in candidates[:25]:
        print(f"[{last.date()}]  id={s.get('id')}  "
              f"n={s.get('number_of_deployed_locations')}  "
              f"{s.get('taxon_ids')}")
        print(f"            {s.get('name')}")
        print()

    if not candidates:
        print("None matched. Try: raise RECENT_DAYS, widen BIRD_GENERA,")
        print("or check that your account has accepted some study licences.")


if __name__ == "__main__":
    main()
