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
# PM25_LIMIT lived here with a comment describing a rule that does not exist:
# nothing branched on it, and the real behaviour is the intake term in the
# purge cost below. A constant that documents imaginary behaviour is worse
# than no constant. analysis_human.PM25_PURGE is the one that is actually used.
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


# The three terms of the purge cost are in different units - ppm cleared, degrees
# lost, and microgram-hours of particles breathed - so combining them needs weights,
# and weights chosen by hand are an opinion wearing a number's clothes. These are
# that opinion, named and in one place rather than inline: a hundred ppm of CO2 is
# treated as worth one degree of cooling, and so is five microgram-hours of PM2.5.
#
# They are not defensible from first principles and nothing here pretends otherwise.
# What CAN be done is to check whether the recommendation actually depends on them,
# which is what `stability` below measures: the same candidates re-ranked under
# weights two times larger and smaller, counting how often the answer survives. An
# answer that holds across that range is one the weights did not decide.
COST_WEIGHTS = {"ppm_per_degree": 100.0, "pm_per_degree": 5.0}
WEIGHT_SWEEP = (0.5, 1.0, 2.0)
SAME_START_MIN = 15             # two starts this close are the same advice
MATERIAL_GAP_MIN = 45           # ...and this far apart are genuinely different


def _purge_cost(cand, ppm_per_degree, pm_per_degree):
    """Combine the three terms. Lower is better; the units are degrees-equivalent."""
    return (-cand["cleared"] / ppm_per_degree
            + cand["heat"]
            + cand["intake"] / pm_per_degree)


def plan_purge(db, location, now=None, horizon_h=12.0, durations=(10, 20, 30, 45, 60)):
    """The cheapest moment in the next `horizon_h` to air the room out, and how
    firmly that is the answer.

    Every start time on the grid is paired with every duration and scored; there are
    only a couple of thousand combinations and each is a closed-form integration, so
    the whole space is searched exactly rather than approximately. The trade-off is
    real: a breezy hour clears the room faster, a cold one costs more heat, and a
    dirty one brings in more particles.

    This used to return the single winner and nothing else, which made the advice
    look far more decided than it was. Dozens of candidates sit within a few percent
    of each other, the forecast shifts every half hour, and the "best" moment would
    hop between them - so the recommendation appeared to change its mind when really
    it had never had one. Keeping the ranked set lets the caller say which it is.

    Returns, besides `best`:
      alternatives  the next genuinely different options, not near-identical starts
      terms         what the winning cost is made of, so a reader can see what drove it
      margin        how much better the best is than the nearest materially different
                    option, in the same degrees-equivalent units as the cost
      stability     the fraction of weightings (see COST_WEIGHTS) that pick the same
                    moment. Below about 0.7, the weights are choosing, not the physics.
    """
    now = int(time.time()) if now is None else int(now)
    model = RoomModel(db, location)
    co2 = features.series(db, "co2_ppm", location, now - 6 * 3600, now).dropna()
    if co2.empty:
        return {"error": f"no recent CO2 for {location}"}
    c_out = model.value["c_out"]
    t_in = features.at(features.series(db, "indoor_temp_c", location, now - 6 * 3600, now), now)
    start_ppm = float(co2.iloc[-1])

    cands, scored, too_cold, warmest = [], 0, 0, None
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
                too_cold += 1
                # Remember the least-bad one. "Nothing is warm enough" is a true
                # answer and a useless one: it does not say whether the best option
                # misses the floor by a tenth of a degree or by five, and those call
                # for completely different decisions. It is also where the advice
                # becomes unstable, because a candidate sitting on the boundary
                # flips in and out with every forecast refresh.
                landed = t_in - heat
                if warmest is None or landed > warmest[0]:
                    warmest = (landed, minutes, offset, round(heat, 1))
                continue
            # Outdoor particles enter on the same clock as CO2 leaves, which is why a
            # short purge wins on a bad-air day without anyone writing that rule.
            intake = (pm25 or 0.0) * (1.0 - math.exp(-lam_open * minutes / 60.0)) * minutes / 60.0
            cands.append({"at": when, "in_minutes": offset, "duration_min": minutes,
                          "cleared": cleared, "heat": heat, "intake": intake,
                          "t_out": round(t_out, 1),
                          "pm25": None if pm25 is None else round(pm25, 1),
                          "lambda_open": round(lam_open, 3), "basis": lam_basis})

    def shaped(cand, cost):
        return {"at": cand["at"], "in_minutes": cand["in_minutes"],
                "duration_min": cand["duration_min"],
                "clears_ppm": round(cand["cleared"]), "heat_cost_c": round(cand["heat"], 1),
                "intake_ug_h": round(cand["intake"], 1), "t_out": cand["t_out"],
                "pm25": cand["pm25"], "lambda_open": cand["lambda_open"],
                "basis": cand["basis"], "cost": round(cost, 3)}

    best = None
    out = {"location": location, "evaluated": scored, "start_ppm": round(start_ppm),
           "rejected_too_cold": too_cold}
    if warmest is not None and not cands:
        landed, minutes, offset, heat = warmest
        out["all_too_cold"] = {
            "best_case_c": round(landed, 1), "floor_c": COMFORT_FLOOR_C,
            "short_by_c": round(COMFORT_FLOOR_C - landed, 1),
            "duration_min": minutes, "in_minutes": offset, "heat_cost_c": heat,
            "note": (f"every one of {too_cold} options would take the room below "
                     f"{COMFORT_FLOOR_C:.0f} C. The warmest is {minutes} min "
                     f"starting in {offset} min, landing at {landed:.1f} C - short "
                     f"by {COMFORT_FLOOR_C - landed:.1f} C. At {model.value['lambda_open']:.1f} "
                     f"air changes an hour a window here is not a gentle thing, so "
                     f"the real choice is a colder room or a staler one, not a free "
                     f"lunch.")}
    if cands:
        w = COST_WEIGHTS
        ranked = sorted(cands, key=lambda c: _purge_cost(c, w["ppm_per_degree"],
                                                         w["pm_per_degree"]))
        top = ranked[0]
        best = shaped(top, _purge_cost(top, w["ppm_per_degree"], w["pm_per_degree"]))

        # What the winning cost is made of, in the units the cost is measured in.
        # Printing this is the difference between "air it at 03:10" and "air it at
        # 03:10 because that is the warmest hour, not because it clears more".
        best_cost = _purge_cost(top, w["ppm_per_degree"], w["pm_per_degree"])
        out["terms"] = {
            "co2_benefit": round(-top["cleared"] / w["ppm_per_degree"], 3),
            "heat_cost": round(top["heat"], 3),
            "pm_cost": round(top["intake"] / w["pm_per_degree"], 3),
            "total": round(best_cost, 3),
            "dominant": max((("clearing CO2", abs(top["cleared"] / w["ppm_per_degree"])),
                             ("losing heat", abs(top["heat"])),
                             ("letting particles in",
                              abs(top["intake"] / w["pm_per_degree"]))),
                            key=lambda kv: kv[1])[0]}

        # Genuinely different alternatives: skip anything starting within a quarter
        # hour of one already listed, since that is the same advice with a rounding
        # difference and listing it only makes the answer look more varied than it is.
        alts, seen = [], [top["in_minutes"]]
        for cand in ranked[1:]:
            if all(abs(cand["in_minutes"] - s_) >= SAME_START_MIN for s_ in seen):
                alts.append(shaped(cand, _purge_cost(cand, w["ppm_per_degree"],
                                                     w["pm_per_degree"])))
                seen.append(cand["in_minutes"])
            if len(alts) >= 3:
                break
        out["alternatives"] = alts

        far = [c for c in ranked
               if abs(c["in_minutes"] - top["in_minutes"]) >= MATERIAL_GAP_MIN]
        if far:
            out["margin"] = round(
                _purge_cost(far[0], w["ppm_per_degree"], w["pm_per_degree"]) - best_cost, 3)
            out["runner_up_in_minutes"] = far[0]["in_minutes"]

        # Sensitivity. The weights are an opinion; this says whether it mattered.
        agree, total = 0, 0
        for fa in WEIGHT_SWEEP:
            for fb in WEIGHT_SWEEP:
                total += 1
                pick = min(cands, key=lambda c: _purge_cost(
                    c, w["ppm_per_degree"] * fa, w["pm_per_degree"] * fb))
                if abs(pick["in_minutes"] - top["in_minutes"]) < SAME_START_MIN:
                    agree += 1
        out["stability"] = round(agree / total, 2)
        # Weight-agreement alone is not confidence, and saying so was misleading.
        # A sweep can report that every weighting picks the same minute while the
        # runner-up an hour away is worse by one percent - the weights did not
        # decide it because NOTHING did, and "firm" was the wrong word for a tie.
        # So the verdict needs the margin as well: how much better this moment is
        # than a materially different one, relative to the cost itself.
        rel = (abs(out["margin"]) / max(abs(best_cost), 1e-9)
               if out.get("margin") is not None else None)
        out["margin_rel"] = None if rel is None else round(rel, 3)
        if rel is not None and rel < 0.05:
            out["timing"] = "interchangeable"
            out["timing_note"] = (
                f"several start times score within {rel:.0%} of each other, so the "
                f"exact minute carries no information - any of them is the same "
                f"advice, and a recommendation that moves between them has not "
                f"changed its mind")
        elif out["stability"] >= 0.8:
            out["timing"] = "firm"
            out["timing_note"] = ("this moment is clearly better and survives every "
                                  "weighting tried")
        elif out["stability"] >= 0.5:
            out["timing"] = "soft"
            out["timing_note"] = ("the cost weights are doing some of the deciding; "
                                  "treat the direction as the advice, not the minute")
        else:
            out["timing"] = "arbitrary"
            out["timing_note"] = ("the hand-chosen cost weights are choosing this, "
                                  "not the physics")
        out["stability_note"] = (
            "the same moment under every weighting tried, so the weights are not "
            "choosing it" if agree == total else
            f"only {agree} of {total} weightings pick this moment - the hand-chosen "
            f"cost weights are doing much of the deciding, so treat the exact timing "
            f"as soft and the direction as the real advice")

    pooled, pooled_basis = model.lambda_open_at(None)
    out.update({"lambda_open": None if best is None else best["lambda_open"],
                "basis": pooled_basis if best is None else best["basis"],
                "lambda_open_pooled": round(pooled, 3),
                "best": best})
    return out


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
