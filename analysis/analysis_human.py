"""
analysis_human.py

Turns the learned room model into the two things the pet actually does: an
expression and one short suggestion.

The chain each run is the same:
    score yesterday's forecasts -> re-fit the rooms -> read the present ->
    predict forward -> decide -> write the display, the events and the new forecasts

Two design decisions are worth knowing about.

Forecasts are written down before their outcome is known. That costs nothing now and
buys the only honest evidence that the system is improving: the mean absolute error
of those rows, grouped by the day they were issued.

Health data changes how strict the thresholds are, never what the physics says. A
poor recovery score does not mean CO2 caused it; it means the cost of a stuffy room
tonight is higher than usual, so be more cautious. The adjustment is bounded, so one
noisy ring reading cannot swing a recommendation.
"""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analysis import features
from analysis.room_model import RoomModel, learn, rooms
from config import settings

MODULE = "analysis_human"

# Comfort thresholds. 1000 and 1400 ppm are the conventional indoor-air action
# points (CIBSE / ASHRAE guidance, and the levels used in the cognitive-performance
# literature); they are anchors, not personal findings.
THRESH = {"fresh": 800.0, "warn": 1000.0, "high": 1400.0}
PM25_PURGE = 25.0        # above this, prefer a short purge to a long open window
COLD_C = 8.0             # below this outdoors, a long open window is not realistic
PURGE_MIN = 10
STALE_S = 900            # no indoor reading for this long and the pet admits it
HORIZONS_S = (1800, 7200)
QUIET_START, QUIET_END = 23, 7
STRICTNESS_K, STRICTNESS_CLAMP = 0.1, 0.15


def quiet_now(now, tz=None):
    """True during the hours when a nudge would wake the user rather than help them."""
    local = datetime.fromtimestamp(now, ZoneInfo(tz or settings.USER_TZ))
    return local.hour >= QUIET_START or local.hour < QUIET_END


def strictness(db, now, days=30):
    """Threshold multiplier from how the body is doing, clamped to +/-15%.

    Below the user's own rolling baseline the multiplier drops, which lowers the CO2
    at which the pet speaks up. The baseline is personal: a rolling median of this
    user's own recovery scores, not a population norm."""
    df = db.read_df("daily_recovery_index", "user", now - days * 86400, now)
    vals = df["value"].astype(float).dropna() if not df.empty else []
    if len(vals) < 5:
        return 1.0, None
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    if mad <= 0:                       # too few distinct scores for a robust spread
        mad = float(np.std(vals))
    if mad <= 0:
        return 1.0, None
    z = (float(vals.iloc[-1]) - med) / mad
    return float(np.clip(1.0 + STRICTNESS_K * z, 1.0 - STRICTNESS_CLAMP,
                         1.0 + STRICTNESS_CLAMP)), round(z, 2)


def advice(s):
    """The one short line the pet shows. Terse because it has to fit a 128 px screen."""
    clear = s.get("clear_minutes")
    cost = s.get("purge_cost")
    tail = ""
    if clear is not None:
        tail = f" {clear:.0f}min"
        if cost:
            tail += f" -{cost:.1f}C"
    if s.get("pm25") is not None and s["pm25"] > PM25_PURGE:
        return f"brief purge {PURGE_MIN}min, air outside is poor"[:40]
    if s.get("t_out") is not None and s["t_out"] < COLD_C:
        return f"purge {PURGE_MIN}min, its cold out{tail}"[:40]
    return f"open a window{tail}"[:40]


def decide(s):
    """State -> (expression, message, severity). Pure, so it is testable without data."""
    if s.get("stale"):
        return "sad", f"no reading from the {s['room']}", "info"
    if s.get("quiet"):
        return "sleepy", "quiet hours, report in the morning", "info"

    co2 = s["co2"]
    warn = THRESH["warn"] * s.get("strictness", 1.0)
    high = THRESH["high"] * s.get("strictness", 1.0)
    soon = s.get("minutes_to_warn")

    if co2 >= high:
        return "alert", advice(s), "high"
    if co2 >= warn:
        return "concerned", advice(s), "warn"
    if soon is not None and soon <= 30:
        return "concerned", f"stuffy in {soon:.0f}min, air it soon"[:40], "warn"
    if co2 <= THRESH["fresh"]:
        return "happy", f"air is good, {co2:.0f}ppm"[:40], "info"
    return "neutral", f"{co2:.0f}ppm and holding"[:40], "info"


def room_state(db, location, model, now, strict):
    """Everything the decision needs about one room, right now."""
    start = now - 6 * 3600
    co2 = features.series(db, "co2_ppm", location, start, now)
    clean = co2.dropna()
    state = {"room": location, "strictness": strict, "stale": True, "learned": model.learned}
    if clean.empty or now - clean.index[-1].timestamp() > STALE_S:
        return state

    latest = float(clean.iloc[-1])
    recent = clean[clean.index >= clean.index[-1] - pd.Timedelta(minutes=45)]
    trend = 0.0
    if recent.size >= 3:
        hours = (recent.index - recent.index[0]).total_seconds().to_numpy() / 3600.0
        trend = float(np.polyfit(hours, recent.to_numpy(float), 1)[0])

    t_in = features.at(features.series(db, "indoor_temp_c", location, start, now), now)
    t_out = features.at(features.series(db, "outdoor_temp_c", "home", start, now), now)
    pm25 = features.at(features.series(db, "pm2_5", "home", start, now), now)
    occupied = trend > 20.0                       # CO2 climbing means someone is in it

    state.update({
        "stale": False, "co2": latest, "trend_ppm_h": round(trend, 1), "occupied": occupied,
        "t_in": t_in, "t_out": t_out, "pm25": pm25,
        "minutes_to_warn": model.time_to(THRESH["warn"] * strict, latest, occupied),
        "minutes_to_high": model.time_to(THRESH["high"] * strict, latest, occupied),
        "clear_minutes": model.time_to(THRESH["fresh"], latest, False, ventilated=True),
        "purge_cost": model.purge_cost_c(t_in, t_out, PURGE_MIN),
    })
    return state


def active_room(db, names, quiet):
    """Which room the pet should be speaking about: the bedroom overnight, else the desk."""
    if not names:
        return None
    if quiet and "bedroom" in names:
        return "bedroom"
    return "desk" if "desk" in names else names[0]


def forecast_rows(model, state, now):
    """Forecasts for the fixed horizons, written before the outcome is known."""
    if state.get("stale"):
        return []
    out = []
    for horizon in HORIZONS_S:
        path = model.forward(state["co2"], horizon / 60.0, state["occupied"])
        if not path:
            continue
        value = path[-1][1]
        err = model.stderr.get("lambda_sealed")
        spread = abs(value - state["co2"]) * (0.5 if err is None else
                                              min(0.5, 2.0 * err / model.value["lambda_sealed"]))
        out.append({"issued_at": now, "location": state["room"], "metric": "co2_ppm",
                    "target_ts": now + horizon, "horizon_s": horizon, "predicted": value,
                    "lo": round(value - spread, 1), "hi": round(value + spread, 1),
                    "basis": "learned" if model.learned else "default"})
    return out


def run(db, now=None, relearn_days=14):
    """One full analysis pass. Returns a summary dict; never raises on thin data."""
    now = int(time.time()) if now is None else int(now)
    t0 = time.time()
    scored = db.score_due(now)
    quiet = quiet_now(now)
    strict, recovery_z = strictness(db, now)
    names = rooms(db)
    out = {"scored": scored, "quiet": quiet, "strictness": round(strict, 3),
           "recovery_z": recovery_z, "rooms": {}, "fits": {}}

    models, states = {}, {}
    for name in names:
        out["fits"][name] = learn(db, name, relearn_days, now)
        models[name] = RoomModel(db, name)
        states[name] = room_state(db, name, models[name], now, strict)
        out["rooms"][name] = states[name]
        db.add_predictions(forecast_rows(models[name], states[name], now))

    focus = active_room(db, names, quiet)
    if focus is None:
        db.log_analysis_run(MODULE, t0, (time.time() - t0) * 1000, "ok", 0, "no rooms yet")
        return out

    state = dict(states[focus], quiet=quiet)
    expression, message, severity = decide(state)
    out["decision"] = {"room": focus, "expression": expression, "message": message,
                       "severity": severity, "basis": models[focus].describe()}

    previous = db.latest_display()
    if previous is None or previous["expression"] != expression or previous["message"] != message:
        db.set_display(expression, message, task=f"{focus}:{severity}", mode="live")
        out["display_changed"] = True

    n_events = 0
    if severity in ("warn", "high") and not state.get("stale"):
        hour = now - now % 3600
        evidence = json.dumps({k: v for k, v in state.items()
                               if not isinstance(v, (list, dict))})
        with db.conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO events (event_type,dedupe_key,detected_at,"
                            "event_start,severity,confidence,location,evidence) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            ("indoor_co2", f"{focus}:{hour}", now, hour, severity,
                             0.9 if models[focus].learned else 0.6, focus, evidence))
            n_events = cur.rowcount if cur.rowcount > 0 else 0
    db.log_analysis_run(MODULE, t0, (time.time() - t0) * 1000, "ok", n_events,
                        f"{focus}:{expression}")
    return out


def night_associations(db, days=60):
    """Overnight CO2 exposure against the next morning's recovery and HRV.

    Deliberately exploratory. With a few dozen nights only a strong rank correlation
    (|rho| above roughly 0.45) is detectable at all, so the result is reported with n
    and a bootstrap interval and is never allowed to drive a recommendation."""
    from scipy import stats

    now = int(time.time())
    nights = features.night_windows(db, now - days * 86400, now)
    co2 = features.series(db, "co2_ppm", "bedroom", now - days * 86400, now)
    out = {"nights": 0, "results": {}}
    if co2.dropna().empty or not nights:
        return out

    exposure, labels = [], []
    for start, end, day in nights:
        seg = co2.loc[pd.Timestamp(start, unit="s", tz="UTC"):
                      pd.Timestamp(end, unit="s", tz="UTC")].dropna()
        if seg.size < 12:
            continue
        exposure.append(float(seg.mean()))
        labels.append(day)
    out["nights"] = len(exposure)
    if len(exposure) < 6:
        return out

    with db.conn() as c:
        for metric in ("daily_recovery_index", "daily_sleep_hrv_ms"):
            pairs = []
            for value, day in zip(exposure, labels):
                row = c.execute("SELECT value FROM observations WHERE metric=? AND location='user'"
                                " AND DATE(ts,'unixepoch')=? LIMIT 1", (metric, day)).fetchone()
                if row and row["value"] is not None:
                    pairs.append((value, float(row["value"])))
            if len(pairs) < 6:
                continue
            xs, ys = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            rho, p = stats.spearmanr(xs, ys)
            rng = np.random.default_rng(0)        # fixed seed: the interval is reproducible
            boot = []
            for _ in range(1000):
                pick = rng.integers(0, len(xs), len(xs))
                boot.append(stats.spearmanr(xs[pick], ys[pick])[0])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            out["results"][metric] = {"n": len(pairs), "rho": round(float(rho), 3),
                                      "p": round(float(p), 4),
                                      "ci95": [round(float(lo), 3), round(float(hi), 3)],
                                      "note": "exploratory; underpowered below |rho|~0.45"}
    return out
