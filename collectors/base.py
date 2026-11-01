"""
base.py

Shared collector contract. Every collector exposes `collect(db=None) -> CollectResult`
(and optionally `backfill(days, db=None)`), so main.py, tests and the analysis
layer treat all sources identically.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests


class CollectorError(Exception):
    """Source reachable but the response was unusable."""


class NotConfigured(Exception):
    """Credentials / hardware missing; the run is logged as 'skipped', not as a failure."""


@dataclass
class Observation:
    source: str
    metric: str
    location: str
    ts: int                      # time the value applies to (UTC unix s)
    value: float | None
    kind: str                    # measured | modelled | forecast
    unit: str = ""
    issued_at: int = 0           # forecasts only
    quality: str = "ok"
    retrieved_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class CollectResult:
    source: str
    observations: list = field(default_factory=list)
    sleep_sessions: list = field(default_factory=list)
    overflow_status: list = field(default_factory=list)
    overflow_alerts: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def n_rows(self):
        return (len(self.observations) + len(self.sleep_sessions)
                + len(self.overflow_status) + len(self.overflow_alerts))


def get_json(url, params=None, headers=None, retries=3, timeout=30):
    """GET with exponential backoff on network errors / 5xx / 429. Other 4xx raise immediately."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last = e
            code = e.response.status_code if e.response is not None else 0
            if 400 <= code < 500 and code != 429:
                raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        time.sleep(2 ** attempt)
    raise CollectorError(f"{url} failed after {retries} attempts: {last}")


def iso_to_unix(s):
    """Parse '2026-10-29T22:00Z' / ISO-8601 (naive = UTC) to unix seconds; None passthrough."""
    if not s:
        return None
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def run_collector(name, fn, db):
    """Run one collection, persist it, and log latency/outcome. Never raises."""
    t0 = time.time()
    ms = lambda: (time.time() - t0) * 1000  # noqa: E731
    try:
        result = fn(db)
        db.save_result(result)
        db.log_run(name, t0, ms(), "ok", result.n_rows())
        return "ok", result.n_rows(), None
    except NotConfigured as e:
        db.log_run(name, t0, ms(), "skipped", 0, str(e))
        return "skipped", 0, str(e)
    except Exception as e:  # noqa: BLE001 - a collector must never kill the scheduler
        err = f"{type(e).__name__}: {e}"[:500]
        db.log_run(name, t0, ms(), "error", 0, err)
        return "error", 0, err
