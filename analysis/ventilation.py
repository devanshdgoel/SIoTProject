"""
ventilation.py

Turns the room model and the outdoor forecast into the two questions worth asking:
what will tonight be like under each option, and when is the cheapest moment to air
a room out.

Both answers come from the same forward model. Over a step where the parameters hold
constant the mass balance is exactly exponential, so a candidate plan is evaluated by
stepping through it in closed form - no solver, and fast enough to try every option.

What the room is heading for matters more than where it starts. The ceiling is
outdoor + source/lambda, and a bedroom with a time constant near three hours that is
slept in for eight passes two or three of them, so the starting concentration has
decayed to under a tenth of its influence by morning. That is why airing a room
before bed achieves almost nothing and why the rate during the night is everything.
"""

import math
import time

from analysis import features
from analysis.room_model import RoomModel
from config import settings

STEP_MIN = 5
CO2_TARGETS = (1000.0, 1400.0, 2000.0)
PM25_LIMIT = 25.0          # above this, prefer a short purge to a long opening
COMFORT_FLOOR_C = 16.0     # below this indoors the heat cost stops being acceptable

# Air-change rates to compare, in air changes per hour. "shut" and "open" come from
# the room's own fitted values; "ajar" has never been measured here, so it carries the
# figure the model says is *needed* rather than one observed, and says so.
def options(model, wind=None):
    """The three configurations to compare, at the wind the opening would happen in.

    A wide-open window is not one number: what it delivers scales with wind speed, so
    the same window is worth five times as much on a breezy night as on a still one.
    Passing the forecast wind is what makes the comparison honest."""
    shut = model.value["lambda_sealed"]
    wide, basis = model.lambda_open_at(wind)
    return [("door and window shut", shut, model.basis["lambda_sealed"]),
            ("window ajar", None, "unmeasured"),
            ("window wide open", wide, basis)]


def trajectory(c0, minutes, lam, source, c_out, occupied):
    """CO2 from now, in closed-form steps of STEP_MIN."""
    ceiling = c_out + (source / lam if occupied else 0.0)
    decay = math.exp(-lam * STEP_MIN / 60.0)
    out, c = [], float(c0)
    for step in range(1, int(minutes / STEP_MIN) + 1):
        c = ceiling + (c - ceiling) * decay
        out.append((step * STEP_MIN, c))
    return out


def summarise(path, c_out):
    """Peak, final value and time spent over each target, from one trajectory."""
    values = [v for _m, v in path]
    hours = STEP_MIN / 60.0
    return {"peak": max(values), "final": values[-1],
            "hours_over": {int(t): round(sum(1 for v in values if v > t) * hours, 1)
                           for t in CO2_TARGETS}}


def lambda_needed(source, c_out, target):
    """The air-change rate that would hold a room at `target` while occupied."""
    headroom = target - c_out
    return None if headroom <= 0 else source / headroom


def _extra_cooling(model, t_in, t_out, minutes, lam):
    """Degrees lost by ventilating at `lam`, over and above leaving the room shut.

    The absolute figure is not the useful one. A sealed unheated room also drifts
    toward outdoor temperature, so charging an open window with the whole drop
    double-counts what would have happened anyway, and this model carries no heating
    term to put back. What a decision needs is the marginal cost, so the loss at the
    sealed rate is subtracted and a shut window correctly costs nothing.

    Air exchange is what carries the heat out, so the thermal time constant shortens
    in proportion to the air-change rate."""
    if t_in is None or t_out is None or t_in <= t_out:
        return 0.0
    sealed = model.value["lambda_sealed"]
    tau = model.value["tau_t_h"]
    loss = lambda rate: (t_in - t_out) * (1.0 - math.exp(
        -(minutes / 60.0) / (tau * sealed / max(rate, 1e-6))))   # noqa: E731
    return round(max(0.0, loss(lam) - loss(sealed)), 1)


def plan_night(db, location="bedroom", now=None, hours=8.0):
    """What tonight looks like under each ventilation option.

    Returns the options with their predicted peak and time over each threshold, plus
    the air-change rate each target would require. The honest centre of this is
    `needed`: the model can say what rate is wanted long before it has measured what
    a partly-open window actually delivers."""
    now = int(time.time()) if now is None else int(now)
    model = RoomModel(db, location)
    co2 = features.series(db, "co2_ppm", location, now - 6 * 3600, now).dropna()
    if co2.empty:
        return {"error": f"no recent CO2 for {location}"}

    c_out = model.value["c_out"]
    # Per PERSON: that is the stored unit, and RoomModel.steady_state multiplies by
    # the headcount while this did not. Latent only because this bedroom sleeps one
    # - a two-person room got a ceiling exactly half the truth from the CLI while
    # the pet, reading the same parameter, got it right.
    source = model.value["source_ppm_h"] * model.occupants
    start = float(co2.iloc[-1])
    t_in = features.at(features.series(db, "indoor_temp_c", location, now - 6 * 3600, now), now)
    t_out = _forecast_mean(db, "outdoor_temp_c", now, hours)
    pm25 = _forecast_mean(db, "pm2_5", now, hours)
    wind = _forecast_mean(db, "wind_ms", now, hours)

    out = {"location": location, "now": now, "start_ppm": round(start),
           "hours": hours, "c_out": round(c_out), "source_ppm_h": round(source),
           "source_basis": model.basis["source_ppm_h"],
           "t_in": t_in, "t_out_forecast": t_out, "pm25_forecast": pm25,
           "wind_forecast": None if wind is None else round(wind, 1),
           "volume_m3": model.volume_m3, "options": [],
           "needed": {int(t): round(lambda_needed(source, c_out, t), 2)
                      for t in CO2_TARGETS if lambda_needed(source, c_out, t)}}

    for name, lam, basis in options(model, wind):
        if lam is None:
            out["options"].append({"name": name, "lambda_h": None, "basis": basis,
                                   "note": "never measured in this room - open it once "
                                           "and the next fit will have it"})
            continue
        path = trajectory(start, hours * 60, lam, source, c_out, occupied=True)
        row = {"name": name, "lambda_h": round(lam, 3), "basis": basis,
               "ceiling": round(c_out + source / lam), **summarise(path, c_out)}
        row["peak"] = round(row["peak"])
        row["final"] = round(row["final"])
        row["heat_cost_c"] = _extra_cooling(model, t_in, t_out, hours * 60, lam)
        out["options"].append(row)
    return out


def plan_purge(db, location, now=None, horizon_h=12.0, durations=(10, 20, 30, 45, 60)):
    """The cheapest moment in the next `horizon_h` to air the room out.

    Every start time on the grid is paired with every duration and scored; there are
    only a couple of thousand combinations and each is a closed-form integration, so
    the whole space is searched exactly rather than approximately.

    The open-window rate is taken from the wind forecast at each candidate start, not
    held constant, so the search trades three things off against each other: a breezy
    hour clears the room faster, a cold one costs more heat, and a dirty one brings in
    more particles. The same window really is worth several times as much at one hour
    as at another, and that is now in the arithmetic rather than in a caveat.

    Still approximate: the law is a single coefficient fitted through the origin over
    the wind speeds this room has actually been aired in, and it is held inside that
    range rather than extrapolated."""
    now = int(time.time()) if now is None else int(now)
    model = RoomModel(db, location)
    co2 = features.series(db, "co2_ppm", location, now - 6 * 3600, now).dropna()
    if co2.empty:
        return {"error": f"no recent CO2 for {location}"}
    c_out = model.value["c_out"]
    t_in = features.at(features.series(db, "indoor_temp_c", location, now - 6 * 3600, now), now)
    start_ppm = float(co2.iloc[-1])

    best, scored = None, 0
    for offset in range(0, int(horizon_h * 60), STEP_MIN):
        when = now + offset * 60
        t_out = _forecast_at(db, "outdoor_temp_c", when)
        pm25 = _forecast_at(db, "pm2_5", when)
        if t_out is None:
            continue
        lam_open, lam_basis = model.lambda_open_at(_forecast_at(db, "wind_ms", when))
        for minutes in durations:
            scored += 1
            cleared = start_ppm - (c_out + (start_ppm - c_out)
                                   * math.exp(-lam_open * minutes / 60.0))
            heat = _extra_cooling(model, t_in, t_out, minutes, lam_open)
            if t_in is not None and t_in - heat < COMFORT_FLOOR_C:
                continue                       # would leave the room uncomfortably cold
            # Outdoor particles enter on the same clock as CO2 leaves, which is why a
            # short purge wins on a bad-air day without anyone writing that rule.
            intake = (pm25 or 0.0) * (1.0 - math.exp(-lam_open * minutes / 60.0)) * minutes / 60.0
            cost = -cleared / 100.0 + heat + intake / 5.0
            if best is None or cost < best["cost"]:
                best = {"at": when, "in_minutes": offset, "duration_min": minutes,
                        "clears_ppm": round(cleared), "heat_cost_c": round(heat, 1),
                        "t_out": round(t_out, 1), "pm25": None if pm25 is None else round(pm25, 1),
                        "lambda_open": round(lam_open, 3), "basis": lam_basis,
                        "cost": round(cost, 3)}
    pooled, pooled_basis = model.lambda_open_at(None)
    return {"location": location, "evaluated": scored, "start_ppm": round(start_ppm),
            "lambda_open": None if best is None else best["lambda_open"],
            "basis": pooled_basis if best is None else best["basis"],
            "lambda_open_pooled": round(pooled, 3),
            "best": best}


def _forecast_at(db, metric, ts, location="home"):
    """Newest forecast for a metric at a time, falling back to what was measured."""
    with db.conn() as c:
        row = c.execute("SELECT value FROM observations WHERE metric=? AND location=? "
                        "AND ts BETWEEN ? AND ? ORDER BY issued_at DESC, ABS(ts-?) LIMIT 1",
                        (metric, location, ts - 5400, ts + 5400, ts)).fetchone()
    return None if row is None else row["value"]


def _forecast_mean(db, metric, now, hours, location="home"):
    with db.conn() as c:
        row = c.execute("SELECT AVG(value) v FROM observations WHERE metric=? AND location=? "
                        "AND ts BETWEEN ? AND ?",
                        (metric, location, now, now + int(hours * 3600))).fetchone()
    return None if row is None or row["v"] is None else round(row["v"], 1)
