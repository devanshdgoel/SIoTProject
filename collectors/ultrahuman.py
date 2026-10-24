"""
ultrahuman.py

Retrieves personal health and recovery data, such as sleep, resting heart rate and HRV, 
where available through the API. Handles authentication, API errors and timestamps.

"""


import os
import json
from datetime import date, datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

ULTRAHUMAN_TOKEN = os.getenv("ULTRAHUMAN_TOKEN")

ULTRHUMAN_DAILY_METRICS_URL = ("https://partner.ultrahuman.com/api/v1/partner/daily_metrics")


def get_daily_metrics(target_date=None):
    """
    Fetch Ultrahuman metrics for a date (default: today, UTC). The API returns
    the WHOLE day's data in one call — there is no "just now" endpoint — so to
    get a "now" reading we take the most recent point of each series (see
    __main__ below and the extract_* helpers).
    """
    if not ULTRAHUMAN_TOKEN:
        raise ValueError("Ultrhuman Token is missing")

    if target_date is None:
        request_date = datetime.now(timezone.utc).date()
    elif isinstance(target_date, date):
        request_date = target_date
    else:
        request_date = date.fromisoformat(target_date)

    headers = {
        "Authorization": ULTRAHUMAN_TOKEN,
        "Accept": "application/json",
    }
    params = {"date": request_date.isoformat()}

    response = requests.get(
        ULTRHUMAN_DAILY_METRICS_URL,
        headers=headers,
        params=params,
        timeout=30,
    )
    print("Ultrahuman status: ", response.status_code)
    if response.status_code != 200:
        print("Response: ", response.text[:1000])
        response.raise_for_status()

    return {"date": request_date.isoformat(), "raw_response": response.json()}


# ===========================================================================
# GRAPH EXTRACTION
# ===========================================================================
# Confirmed graph keys from the real response (via inspect_graph_keys):
HRV_GRAPH_KEY = "hrv_graph"
HR_GRAPH_KEY = "hr_graph"                     # heart rate
RESPIRATORY_GRAPH_KEY = "respiratory_graph"   # breathing rate (air-quality relevant)
MOVEMENT_GRAPH_KEY = "movement_graph"         # movement (confounder control)
SPO2_GRAPH_KEY = "spo2_graph"                 # best guess; NOT seen in responses so far
# (No spo2 graph has appeared in daily_metrics yet. Other keys: temp_graph, sleep_graph.)


def find_graphs(node, graph_key):
    """Recursively find every object stored under `graph_key`."""
    graphs = []
    if isinstance(node, dict):
        if graph_key in node:
            graphs.append(node[graph_key])
        for value in node.values():
            graphs.extend(find_graphs(value, graph_key))
    elif isinstance(node, list):
        for item in node:
            graphs.extend(find_graphs(item, graph_key))
    return graphs


def extract_series(metrics_response, graph_key, value_name="value"):
    """
    Pull a metric graph into a list of {"timestamp_utc": <ISO>, <value_name>: <float>},
    time-sorted. Returns [] if the metric isn't present (never raises), so a
    missing metric never breaks the collection loop.
    """
    data = metrics_response["raw_response"]
    points = []
    for graph in find_graphs(data, graph_key):
        if not isinstance(graph, dict):        # skip a null/empty graph (e.g. respiratory today)
            continue
        for point in (graph.get("data") or []):  # "data" can also be null
            if not isinstance(point, dict):
                continue
            value = point.get("value")
            timestamp = point.get("timestamp")
            if value is None or timestamp is None:
                continue
            points.append({
                "timestamp_utc": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                value_name: float(value),
            })
    points.sort(key=lambda p: p["timestamp_utc"])
    return points


def extract_hrv_series(metrics_response):
    """HRV series as {timestamp_utc, hrv_ms}. Kept with its own value name for
    compatibility with storage.save_hrv_series()."""
    pts = extract_series(metrics_response, HRV_GRAPH_KEY, value_name="hrv_ms")
    if not pts:
        raise ValueError("No HRV points found in the Ultrahuman response.")
    return pts


def extract_hr_series(metrics_response):
    """Heart rate series as {timestamp_utc, value}."""
    return extract_series(metrics_response, HR_GRAPH_KEY)


def extract_respiratory_series(metrics_response):
    """Respiratory (breathing) rate series as {timestamp_utc, value}."""
    return extract_series(metrics_response, RESPIRATORY_GRAPH_KEY)


def extract_movement_series(metrics_response):
    """Movement series as {timestamp_utc, value} (confounder control)."""
    return extract_series(metrics_response, MOVEMENT_GRAPH_KEY)


def extract_spo2_series(metrics_response):
    """
    SpO2 (blood-oxygen %) as {timestamp_utc, value}, time-sorted.

    Ultrahuman's daily_metrics has NOT returned any SpO2 in the responses we've
    inspected, so this tries TWO shapes and FAILS SOFT (returns []):
      1) a 'spo2_graph' like the other metrics, if one ever appears;
      2) a single daily scalar (e.g. {"spo2": 97} or {"type":"spo2","value":97}),
         wrapped as a one-point series stamped 'now' so storage treats it like
         every other metric.
    Run inspect_graph_keys() after a fetch to check whether a real SpO2 key
    has started to show up.
    """
    # 1) graph form (preferred, if present)
    pts = extract_series(metrics_response, SPO2_GRAPH_KEY)
    if pts:
        return pts
    # 2) scalar fallback -> wrap the single value as one 'now' point
    hits = find_metric_values(metrics_response["raw_response"],
                              ["spo2", "oxygen", "saturation"])
    if hits:
        _, value = hits[0]
        return [{
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "value": float(value),
        }]
    return []


# ===========================================================================
# DISCOVERY HELPERS
# ===========================================================================
def inspect_graph_keys(metrics_response):
    """Print every key in the response that looks like a metric graph."""
    data = metrics_response["raw_response"]
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if "graph" in str(k).lower():
                    found.add(k)
                walk(v)
        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(data)
    print("Graph keys found in response:", sorted(found))
    return sorted(found)


def find_metric_values(node, substrings):
    """
    Search the whole response for a metric's single VALUE (for metrics that are
    numbers rather than graphs, like resting HR or an SpO2 average).
    Catches {"resting_heart_rate": 54} AND {"type":"spo2","value":97}.
    Returns a list of (label, value). Empty = genuinely not present.
    """
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            if any(s in str(k).lower() for s in substrings) and isinstance(v, (int, float)):
                hits.append((k, v))
        label = str(node.get("type") or node.get("name") or node.get("title") or "").lower()
        if any(s in label for s in substrings):
            for vk in ("value", "last_value", "avg", "average", "last_reading", "today", "latest"):
                if isinstance(node.get(vk), (int, float)):
                    hits.append((f"{label}.{vk}", node[vk]))
        for v in node.values():
            hits.extend(find_metric_values(v, substrings))
    elif isinstance(node, list):
        for item in node:
            hits.extend(find_metric_values(item, substrings))
    return hits


# ===========================================================================
# Run directly -> print the LATEST ("now") reading of each metric.
# ===========================================================================
if __name__ == "__main__":
    metrics = get_daily_metrics()
    raw = metrics["raw_response"]

    print("\n=== Latest readings (most recent point = 'now') ===")

    # HRV
    try:
        hrv = extract_hrv_series(metrics)
        p = hrv[-1]
        print(f"HRV : {p['hrv_ms']:.0f} ms    at {p['timestamp_utc']}")
    except Exception as e:
        print("HRV : no data —", e)

    # Heart rate
    hr = extract_hr_series(metrics)
    if hr:
        p = hr[-1]
        print(f"HR  : {p['value']:.0f} bpm   at {p['timestamp_utc']}")
    else:
        print("HR  : no data")

    # Respiratory rate
    resp = extract_respiratory_series(metrics)
    if resp:
        p = resp[-1]
        print(f"Resp: {p['value']:.1f} /min  at {p['timestamp_utc']}")
    else:
        print("Resp: no data")

    # Movement
    mov = extract_movement_series(metrics)
    if mov:
        p = mov[-1]
        print(f"Move: {p['value']}         at {p['timestamp_utc']}")
    else:
        print("Move: no data")

    # SpO2 — via the extractor (graph if present, else a daily scalar). Expect
    # "no series" for now: the ring hasn't returned SpO2 in daily_metrics yet.
    spo2_series = extract_spo2_series(metrics)
    if spo2_series:
        p = spo2_series[-1]
        print(f"SpO2: {p['value']:.0f} %     at {p['timestamp_utc']}")
    else:
        print("SpO2: no series (not present in daily_metrics response)")

    # RHR and SpO2 are single daily values (if present) — search for them.
    rhr = find_metric_values(raw, ["resting", "rhr"])
    print("RHR :", rhr if rhr else "not found in response")
    spo2 = find_metric_values(raw, ["spo2", "oxygen", "saturation"])
    print("SpO2:", spo2 if spo2 else "not found in response")

    print("\n--- available graph keys ---")
    inspect_graph_keys(metrics)
