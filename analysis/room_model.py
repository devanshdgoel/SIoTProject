"""
room_model.py

What the system knows about each room, and how it works it out.

Everything here comes from one equation, the mass balance for a well-mixed room:

    dC/dt = S - lambda * (C - C_out)

    C       indoor CO2 (ppm)          C_out  outdoor CO2 (ppm)
    S       source term (ppm/h)       lambda air changes per hour

Read it in three directions and it yields three different parameters:
  * nobody in the room, so S = 0, and the solution is a pure exponential decay whose
    rate is lambda. Every decay the record contains is one measurement of it.
  * somebody in the room long enough for CO2 to level off, so dC/dt = 0 and
    S = lambda * excess. That is the source term.
  * S is known physiology (a seated adult exhales about 18.7 L/h of CO2), so the
    room volume falls out of it. Comparing that against a tape measure validates
    the whole model in a single number.

Nothing here is machine learning. It is parameter estimation on a physical model,
which is why it can work from a handful of observations rather than thousands, and
why every number it produces has a unit and a confidence interval.
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from analysis import features

# Published fallbacks, used until the room has been watched enough to do better.
# A recommendation always says which of the two it is standing on.
DEFAULTS = {
    "c_out": (420.0, "ppm"),
    "lambda_sealed": (0.5, "1/h"),
    "lambda_open": (3.0, "1/h"),
    "source_ppm_h": (500.0, "ppm/h"),
    "tau_t_h": (6.0, "h"),
}
# Observations required before a learned value displaces its default.
MIN_N = {"c_out": 12, "lambda_sealed": 2, "lambda_open": 2, "source_ppm_h": 2, "tau_t_h": 2}

CO2_PER_PERSON_LH = 18.7        # seated adult at rest, litres of CO2 per hour

# Parameters that came from outside the sensors. `learn()` clears everything it is
# about to re-derive, and these are the ones it cannot re-derive, so they survive it.
MEASURED_PARAMS = ("volume_measured_m3",)


def measured_volume(db, location):
    """A tape-measured volume, if the user has given one. None otherwise.

    Stored under its own name so that `learn()` can never overwrite it. Everything
    else here is inferred and may be re-inferred; this one is the only number in the
    system that came from outside the sensors, which makes it the only thing capable
    of checking them."""
    row = db.params(location).get("volume_measured_m3")
    return float(row["value"]) if row and row["value"] else None


# What an adult actually exhales, per hour, asleep. The 18.7 L/h above is a textbook
# figure for a SEATED adult at about 1.2 MET; sleep is nearer 0.8 MET, and CO2 output
# scales with metabolic rate, so the nights this source term is fitted from should sit
# well below it. The band is wide because body mass and fitness move it several
# litres either way - it is there to be failed by a real discrepancy, not by biology.
SLEEP_CO2_LH = (11.0, 15.0)


def volume_check(db, location, source_per_person, n, source_stderr=None):
    """Compare the inferred volume with the measured one, and invert the comparison.

    volume = CO2_PER_PERSON_LH / source, so measuring the volume independently turns
    the relation round and solves for the CO2 the occupant actually produces:

        L/h = source_ppm_h * volume_m3 / 1000

    That is worth more than the agreement itself, because it is a quantity with a
    known answer that nothing in this system was fitted to. The room stops being a
    way of assuming a person and becomes a way of measuring one.

    The test has to be TWO-SIDED or it is not a test. An earlier version of this
    noted only that sleeping output should be lower than the seated figure, so any
    value below 18.7 read as confirmation - which is the same "cannot fail" structure
    `fit_open_wind` was criticised for, written by the same hand a day later. A result
    above the band is as informative as one below it, and here the result IS above it.
    """
    truth = measured_volume(db, location)
    if truth is None or not source_per_person:
        return None
    inferred = CO2_PER_PERSON_LH * 1000.0 / source_per_person
    implied_lh = source_per_person * truth / 1000.0
    err = (inferred - truth) / truth

    # The inferred volume is 18700/S, so its relative error is exactly the source's.
    # Reporting the volume without it invites an 8% gap to be read as agreement when
    # the interval is wider than the gap.
    rel = (source_stderr / source_per_person) if source_stderr else None
    lo = hi = None
    if rel:
        lo, hi = inferred * (1 - rel), inferred * (1 + rel)

    band_lo, band_hi = SLEEP_CO2_LH
    implied_err = implied_lh * rel if rel else 0.0
    agrees = (implied_lh - implied_err) <= band_hi and (implied_lh + implied_err) >= band_lo

    out = {"measured_m3": truth, "inferred_m3": round(inferred, 1),
           "inferred_lo": None if lo is None else round(lo, 1),
           "inferred_hi": None if hi is None else round(hi, 1),
           "error_pct": round(err * 100, 1), "n": n,
           "implied_co2_lh": round(implied_lh, 2),
           "implied_co2_lh_err": round(implied_err, 2),
           "sleep_band_lh": list(SLEEP_CO2_LH),
           "assumed_co2_lh": CO2_PER_PERSON_LH,
           "agrees": bool(agrees)}

    span = (f" (+/-{rel * 100:.0f}%, {lo:.1f}-{hi:.1f})" if rel else
            " (no interval: the source term carries no stderr)")
    if agrees:
        out["note"] = (
            f"source from {n} night(s) implies {inferred:.1f} m3{span} against "
            f"{truth:.1f} m3 measured, {err * 100:+.0f}%. Inverted, the occupant "
            f"produces {implied_lh:.1f}+/-{implied_err:.1f} L/h of CO2, inside the "
            f"{band_lo:.0f}-{band_hi:.0f} L/h expected of an adult asleep.")
    else:
        # Say which way it failed and what that implies, because the direction is
        # the diagnostic: too much CO2 per hour for the measured room means the
        # source term is too big, which makes the inferred volume too small.
        way = "above" if implied_lh > band_hi else "below"
        out["note"] = (
            f"source from {n} night(s) implies {inferred:.1f} m3{span} against "
            f"{truth:.1f} m3 measured, {err * 100:+.0f}%. But inverted it FAILS: the "
            f"occupant would have to produce {implied_lh:.1f}+/-{implied_err:.1f} L/h "
            f"of CO2, {way} the {band_lo:.0f}-{band_hi:.0f} L/h an adult asleep "
            f"actually does. So the {err * 100:+.0f}% volume agreement is luck, not "
            f"accuracy - the source term is about "
            f"{abs(1 - implied_lh / ((band_lo + band_hi) / 2)) * 100:.0f}% too "
            f"{'high' if way == 'above' else 'low'} and the volume error happens to "
            f"be small because both are wrong in the same direction.")
    return out


MIN_R2 = 0.90                   # a decay fit below this was not a clean decay
VENT_RATIO = 2.0                # an open window must at least double lambda to count


def rooms(db):
    """Rooms that have indoor data, taken from the data rather than from config."""
    with db.conn() as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT location FROM observations WHERE source='scd40' ORDER BY location")]


def _robust(values):
    """Median and the standard error of the median, via the MAD.

    The median is used instead of the mean because a single bad fit among a dozen
    decays would otherwise drag the answer; 1.4826 rescales the MAD to a standard
    deviation for normal data, and 1.253 converts that to the error of a median."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return None, None, 0
    med = float(np.median(a))
    if a.size < 2:
        return med, None, 1
    sd = 1.4826 * float(np.median(np.abs(a - med)))
    return med, float(1.253 * sd / np.sqrt(a.size)), int(a.size)


def split_point(values, default=1.5):
    """Where the sealed and ventilated clusters of lambda divide.

    Opening a window multiplies the air-change rate several-fold, so the histogram
    of fitted values is bimodal. On sorted one-dimensional data the optimal
    two-cluster split is simply the gap that minimises within-group variance, so no
    clustering library is needed. Returns None when the data is not convincingly
    bimodal, which is the honest answer for a room that was never opened."""
    v = np.sort(np.log(np.asarray(list(values), dtype=float)))
    if v.size < 6:
        return default
    best, cut, groups = None, None, None
    for i in range(2, v.size - 1):
        a, b = v[:i], v[i:]
        wss = a.var() * a.size + b.var() * b.size
        if best is None or wss < best:
            best, cut = wss, float(np.exp((a[-1] + b[0]) / 2.0))
            groups = (float(np.exp(np.median(a))), float(np.exp(np.median(b))))
    if groups is None or groups[1] < groups[0] * VENT_RATIO:
        return None                      # one cluster: the window stayed shut all record
    return cut


def classify(lambdas, labelled, cut):
    """sealed / ventilated per decay. A user label always wins over the inferred cut."""
    out = []
    for lam, label in zip(lambdas, labelled):
        if label in ("open", "closed"):
            out.append("ventilated" if label == "open" else "sealed")
        elif cut is None:
            out.append("sealed")
        else:
            out.append("ventilated" if lam >= cut else "sealed")
    return out


def learn(db, location, days=14, now=None):
    """Re-fit every parameter for one room from the last `days` of record.

    Returns a summary dict. Safe to run repeatedly: it is a fresh fit each time, so
    nothing drifts through repeated self-application."""
    now = int(time.time()) if now is None else int(now)
    start = now - days * 86400
    raw = features.series(db, "co2_ppm", location, start, now)
    co2, spikes = features.despike(raw)
    summary = {"location": location, "points": int(co2.dropna().size), "spikes": spikes,
               "decays": 0, "plateaus": 0}
    if co2.dropna().size < 12:
        return summary

    # Outdoor CO2 as this sensor reads it. The low percentile of the record is only a
    # starting guess: in a room that is rarely left empty for long it sits well above
    # the truth. Refine it to the asymptote the deep decays are actually heading for.
    labels = db.labels(location=location, start=start, end=now)

    # room_params is derived, like decay_events: each run recomputes every parameter
    # from scratch over its window, so last run's values are cleared rather than left
    # behind. Without this a parameter that can no longer be measured keeps its old
    # number for ever - which is how a bedroom went on claiming 325 cubic metres,
    # from a source term measured before occupancy was required.
    # ...except for anything that was MEASURED rather than fitted. The wipe exists so
    # that a parameter which can no longer be derived stops being served; a tape
    # measure cannot stop being derivable, and re-deriving it is exactly what it is
    # here to make unnecessary. Without this exclusion the user records a volume, the
    # cross-check prints, and the next analysis pass five minutes later silently
    # deletes the only external calibration in the system.
    with db.conn() as conn:
        conn.execute(
            "DELETE FROM room_params WHERE location=? AND param NOT IN (%s)"
            % ",".join("?" * len(MEASURED_PARAMS)),
            (location, *MEASURED_PARAMS))

    # Best evidence first. An empty room with the window open has equilibrated with
    # the air outside, so what the sensor reads then IS outdoor CO2 as that sensor
    # reads it - calibration offset included, which is the number the model wants.
    # Nothing is fitted; it is simply observed. The first half hour is skipped while
    # the room is still flushing.
    direct, unfinished = [], 0
    for lb in labels:
        if lb["kind"] != "window" or lb["value"] != "open":
            continue
        seg = co2.loc[pd.Timestamp(lb["start_ts"] + 1800, unit="s", tz="UTC"):
                      pd.Timestamp(lb["end_ts"], unit="s", tz="UTC")].dropna()
        if seg.size < 6:
            continue
        level = features.settled_baseline(seg, features.outdoor_baseline(co2))
        if level is None:
            unfinished += 1      # still flushing: its floor is an upper bound only
        else:
            direct.append(level)

    c_out = features.outdoor_baseline(co2)
    method = "5th percentile of this sensor's own record"
    tails = []
    for a, b in features.decay_windows(co2, c_out, min_drop=250.0):
        trimmed = features.best_decay_fit(co2.loc[a:b], c_out)
        if trimmed is None:
            continue
        fit = features.fit_asymptote(co2.loc[trimmed["start"]:trimmed["end"]], c_out)
        if fit is not None and fit["r2"] >= 0.95:
            tails.append(fit["c_inf"])
    refined = len(tails) >= 2
    if direct:
        c_out, refined = float(np.median(direct)), True
        method = f"measured directly during {len(direct)} labelled window-open period(s)"
    elif refined:
        c_out = float(np.median(tails))
        method = f"asymptote of {len(tails)} deep decays"
    if unfinished:
        method += (f"; {unfinished} further opening(s) ignored for the baseline, "
                   f"still flushing when the window shut")
    # If the percentile had to be clamped and no decay settled, the room was never
    # observed near outdoor air. The clamp is then a guess wearing a measurement's
    # clothes, so it is stored with n=0 and the published default keeps being used.
    # One usable asymptote is not a refinement: the branch above needs two before it
    # replaces the percentile, so a single tail must not make the clamp look trusted.
    clamped = not 250.0 <= float(co2.dropna().quantile(0.05)) <= 700.0
    trusted = 0 if (clamped and not refined) else int(co2.dropna().size)
    db.set_param(location, "c_out", c_out, None, trusted, "ppm",
                 {"method": method, "clamped": clamped})
    summary["c_out"] = round(c_out, 1)
    summary["c_out_trusted"] = bool(trusted)

    temp_out = features.series(db, "outdoor_temp_c", "home", start, now)
    wind = features.series(db, "wind_ms", "home", start, now)
    temp_in = features.series(db, "indoor_temp_c", location, start, now)

    def record(fit, a, b, method):
        mid = int((a.timestamp() + b.timestamp()) / 2)
        t_in, t_out = features.at(temp_in, mid), features.at(temp_out, mid)
        seg = co2.loc[a:b]
        # The asymptote was already chosen where the curve was fitted - it has to be,
        # because which asymptote is right changes which stretch of the curve fits
        # best, so it cannot be a correction applied afterwards. Here it is only read
        # off and recorded, so that a lambda can always be traced to what the room was
        # actually emptying into.
        asym = fit.get("asym") or {}
        c_inf = fit.get("c_inf", c_out)
        if asym.get("free"):
            method += " +free asymptote"
        return {"location": location, "start_ts": int(a.timestamp()),
                "end_ts": int(b.timestamp()), "lambda_h": fit["lambda_h"],
                "stderr": fit.get("stderr"), "r2": fit["r2"],
                "c_start": float(seg.iloc[0]), "c_end": float(seg.iloc[-1]),
                "c_out": c_out, "c_inf": c_inf, "asym_note": asym.get("reason"),
                "delta_t_c": None if t_in is None or t_out is None else t_in - t_out,
                "wind_ms": features.at(wind, mid), "method": method,
                "labelled": features.label_over(labels, int(a.timestamp()),
                                                int(b.timestamp()), "window")}

    rows, lams, marks = [], [], []

    # Standardised measurements first. Every one of these starts the same delay after
    # somebody left and runs for the same span, so they are comparable with each other
    # in a way that threshold-found windows are not - which is what makes them usable
    # for tracking how the rate moves with the weather.
    taken = []
    for when in features.departures(co2):
        fit = features.departure_decay(co2, c_out, when)
        if fit is None or fit["r2"] < MIN_R2:
            continue
        rows.append(record(fit, fit["start"], fit["end"], "departure"))
        taken.append((fit["start"], fit["end"]))
        lams.append(fit["lambda_h"])
        marks.append(rows[-1]["labelled"])

    for a, b in features.decay_windows(co2, c_out):
        if any(a <= end and start <= b for start, end in taken):
            continue                             # already measured, and measured better
        fit = features.best_decay_fit(co2.loc[a:b], c_out, later=co2)
        if fit is None or fit["r2"] < MIN_R2:
            continue
        a, b = fit["start"], fit["end"]          # the trimmed span the fit actually used
        rows.append(record(fit, a, b, "window"))
        lams.append(fit["lambda_h"])
        marks.append(rows[-1]["labelled"])

    # decay_events is derived, not a log: every fit in it must share the c_out that
    # produced it. Rows from an earlier run used a different baseline, so re-fitting
    # replaces them instead of accumulating a mixture nobody can interpret.
    with db.conn() as conn:
        conn.execute("DELETE FROM decay_events WHERE location=? AND start_ts>=?",
                     (location, start))
    if rows:
        cut = split_point(lams)
        for row, state in zip(rows, classify(lams, marks, cut)):
            row["inferred"] = state
        db.save_decays(rows)
        summary["decays"] = len(rows)
        summary["split_at"] = cut
        for state, param in (("sealed", "lambda_sealed"), ("ventilated", "lambda_open")):
            vals = [r["lambda_h"] for r in rows if r["inferred"] == state]
            med, err, n = _robust(vals)
            if med is not None:
                db.set_param(location, param, med, err, n, "1/h",
                             {"from": "CO2 decay fits", "split_at": cut,
                              "labelled": sum(1 for r in rows
                                              if r["inferred"] == state and r["labelled"])})
                summary[param] = (round(med, 3), n)

    # Source term from occupied rises. A room filling toward its ceiling gives the
    # source and the air-change rate from one curve, and unlike a plateau it does not
    # have to reach steady state first - which this bedroom never does overnight.
    rises = []
    for a, b in features.occupied_windows(db, location, labels, start, now):
        if features.label_at(labels, (a + b) // 2, "window") == "open":
            continue
        seg = co2.loc[pd.Timestamp(a, unit="s", tz="UTC"):
                      pd.Timestamp(b, unit="s", tz="UTC")].dropna()
        fit = features.fit_rise(seg, c_out)
        if fit is not None and fit["r2"] >= 0.90:
            # Divide by the headcount before averaging. Two people fill a room twice
            # as fast, so a source fitted across a mixture of one- and two-person
            # spells is wrong for both, and the volume it implies comes out halved.
            fit["people"] = features.people_at(labels, (a + b) // 2, default=1)
            fit["per_person"] = fit["source_ppm_h"] / fit["people"]
            fit["lambda_usable"] = (fit["covered"] >= MIN_RISE_COVER
                                    and fit["r2"] >= MIN_RISE_R2)
            rises.append(fit)
    if rises:
        med, err, n = _robust([r["per_person"] for r in rises])
        counts = [r["people"] for r in rises]
        typical = int(round(float(np.median(counts))))
        db.set_param(location, "source_ppm_h", med, err, n, "ppm/h per person",
                     {"from": f"{n} occupied rise(s)", "people_seen": counts,
                      "ceilings": [round(r["ceiling"]) for r in rises],
                      "lambda_from_rise": [round(r["lambda_h"], 3) for r in rises]})
        db.set_param(location, "typical_people", typical, None, n, "people",
                     {"from": "median headcount across the labelled occupied rises"})
        volume = CO2_PER_PERSON_LH * 1000.0 / med
        # V = 18700/S exactly, so V's relative error IS S's. Storing None here was
        # claiming a precision the source term does not have, and it is the number a
        # reader compares against a tape measure.
        v_err = volume * (err / med) if err else None
        db.set_param(location, "volume_m3", volume, v_err, n, "m3",
                     {"assumes": "one seated adult at 18.7 L/h; compare with a tape measure",
                      "stderr_from": "the source term's own spread; the 18.7 L/h "
                                     "constant carries a further systematic this "
                                     "does not include"})
        summary["rises"] = n
        summary["source_ppm_h"] = (round(med, 1), n)
        summary["typical_people"] = typical
        summary["volume_m3"] = round(volume, 1)
        # The one number in the system that did not come from the sensors. It is a
        # check on them, so it is reported whether it agrees or not.
        check = volume_check(db, location, med, n, source_stderr=err)
        if check:
            summary["volume_check"] = check

    # A rise measures the same air-change rate a decay does. The loss term of the
    # mass balance is lambda*(C - C_out) whether the room is filling or emptying, so
    # an occupied night with the window shut is a measurement of lambda_sealed - and
    # in this bedroom a far longer one than any decay: nine hours against thirty
    # minutes. It is also the only route available in a room that is occupied nearly
    # every night and so rarely produces a clean decay at all.
    #
    # They are pooled rather than one replacing the other, because they fail in
    # opposite directions and agreeing is the evidence. A decay is fitted on a short,
    # steep, well-conditioned stretch and is vulnerable to the asymptote being wrong;
    # a rise is long and gently curved and is vulnerable to lambda trading against
    # its ceiling. Nothing about the room makes them agree, so when they do, the
    # number is the room's and not the method's.
    rise_lams = [r["lambda_h"] for r in rises if r["lambda_usable"]]
    if rise_lams:
        decay_lams = [r["lambda_h"] for r in rows if r.get("inferred") == "sealed"]
        med, err, n = _robust(decay_lams + rise_lams)
        if med is not None:
            d_med, _e, _n = _robust(decay_lams)
            r_med, _e, _n = _robust(rise_lams)
            detail = {"from": "CO2 decays and occupied rises pooled",
                      "decays": len(decay_lams), "rises": len(rise_lams),
                      "decay_median": None if d_med is None else round(d_med, 3),
                      "rise_median": round(r_med, 3),
                      "skipped_rises": sum(1 for r in rises if not r["lambda_usable"])}
            # Two independent routes to one number, so say how far apart they landed.
            # Agreement is the claim being made; a reader should not have to take it
            # on trust, and a later disagreement should be loud rather than averaged.
            if d_med is not None:
                gap = abs(r_med - d_med) / max(d_med, 1e-9)
                detail["agreement"] = (
                    f"emptying gives {d_med:.3f}/h, filling gives {r_med:.3f}/h: "
                    f"{gap * 100:.0f}% apart from two methods that share no "
                    f"assumption but the mass balance itself")
                detail["disagree"] = gap > RISE_DECAY_GAP
            db.set_param(location, "lambda_sealed", med, err, n, "1/h", detail)
            summary["lambda_sealed"] = (round(med, 3), n)
            summary["lambda_sealed_agreement"] = detail.get("agreement")

    lam_sealed = db.params(location).get("lambda_sealed")
    lam_for_source = float(lam_sealed["value"]) if lam_sealed else DEFAULTS["lambda_sealed"][0]
    # A plateau only measures a source if somebody was producing one. An empty room
    # holding steady is not at an occupied equilibrium - it is sitting at whatever
    # the rest of the flat is at, with its door open - and reading a source off that
    # invented a 325 cubic metre bedroom before this check existed. With no occupancy
    # labels at all, the honest answer is to measure nothing.
    sources = []
    for a, b in ([] if rises else features.plateau_windows(co2, c_out)):
        mid = int((a.timestamp() + b.timestamp()) / 2)
        if features.label_at(labels, mid, "occupancy") != "present":
            continue
        if features.label_at(labels, mid, "window") == "open":
            continue                      # an open-window plateau would use the wrong lambda
        sources.append(lam_for_source * (float(co2.loc[a:b].mean()) - c_out))
    med, err, n = _robust(sources)
    summary["plateaus"] = len(sources)
    if med is not None and med > 0:
        db.set_param(location, "source_ppm_h", med, err, n, "ppm/h",
                     {"from": "occupied plateaus", "lambda_used": lam_for_source})
        volume = CO2_PER_PERSON_LH * 1000.0 / med      # L/h -> m3/h -> ppm/h inverts to m3
        db.set_param(location, "volume_m3", volume, None, n, "m3",
                     {"assumes": "one seated adult at 18.7 L/h; compare with a tape measure"})
        summary["source_ppm_h"] = (round(med, 1), n)
        summary["volume_m3"] = round(volume, 1)

    taus = []
    for row in rows:
        if row.get("inferred") != "ventilated" or row["delta_t_c"] is None:
            continue
        seg = temp_in.loc[pd.Timestamp(row["start_ts"], unit="s", tz="UTC"):
                          pd.Timestamp(row["end_ts"], unit="s", tz="UTC")].dropna()
        t_out = features.at(temp_out, row["start_ts"])
        if seg.size < 4 or t_out is None:
            continue
        fit = features.fit_decay(seg, t_out)          # same exponential, different quantity
        if fit is not None and fit["r2"] >= 0.8 and fit["lambda_h"] > 0:
            taus.append(1.0 / fit["lambda_h"])
    med, err, n = _robust(taus)
    if med is not None:
        db.set_param(location, "tau_t_h", med, err, n, "h",
                     {"from": "indoor temperature decay while ventilating"})
        summary["tau_t_h"] = (round(med, 2), n)

    # Both weather relationships are re-fitted here, against the decays this run has
    # just written. They have to run after save_decays and after room_params is
    # cleared, so that neither reads a rate measured against an older baseline nor
    # survives as a stale row once the evidence for it is gone.
    summary["weather"] = fit_weather(db, location)
    summary["open_wind"] = fit_open_wind(db, location)
    return summary


MIN_WEATHER_FITS = 6          # below this the three coefficients are not identifiable


def fit_weather(db, location, min_r2=0.90):
    """How this room's air-change rate moves with the weather.

    Infiltration is driven by two pressures - the stack effect from the indoor to
    outdoor temperature difference, and wind on the outside of the building. They add
    in quadrature rather than linearly, because pressures combine as energies and flow
    goes as the square root of pressure. So the form is

        lambda = sqrt(a + b*|dT| + c*v^2)

    which squares into an ordinary linear regression of lambda^2 on |dT| and v^2. The
    functional form is derived, not chosen to fit.

    Only standardised, departure-anchored measurements are used. Threshold-found
    windows differ in length and in how far into a decay they begin, so pooling them
    would put the measurement geometry into the residuals alongside the weather.

    Returns None - and stores nothing - until there are enough of them. With three
    coefficients and a handful of points the fit would be meaningless however good its
    r-squared looked."""
    usable = [r for r in db.decays(location, min_r2=min_r2)
              if (r["method"] == "departure" and r["inferred"] == "sealed"
                  and r["wind_ms"] is not None and r["delta_t_c"] is not None)]
    if len(usable) < MIN_WEATHER_FITS:
        return {"n": len(usable), "need": MIN_WEATHER_FITS,
                "note": "not enough standardised measurements yet"}

    lam = np.array([r["lambda_h"] for r in usable], dtype=float)
    wind = np.array([r["wind_ms"] for r in usable], dtype=float)
    dT = np.abs(np.array([r["delta_t_c"] for r in usable], dtype=float))
    design = np.column_stack([np.ones_like(lam), dT, wind ** 2])
    coef, *_ = np.linalg.lstsq(design, lam ** 2, rcond=None)
    predicted = design @ coef
    resid = lam ** 2 - predicted
    dof = max(1, len(lam) - 3)
    cov = float(np.sum(resid ** 2)) / dof * np.linalg.pinv(design.T @ design)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    total = float(np.sum((lam ** 2 - np.mean(lam ** 2)) ** 2))
    out = {"n": len(lam), "a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]),
           "stderr": [float(e) for e in err],
           "r2": 1 - float(np.sum(resid ** 2)) / total if total > 0 else 0.0,
           "wind_range": [round(float(wind.min()), 1), round(float(wind.max()), 1)],
           "dT_range": [round(float(dT.min()), 1), round(float(dT.max()), 1)]}
    db.set_param(location, "lambda_wind_c", out["c"], out["stderr"][2], len(lam),
                 "(1/h)^2 per (m/s)^2", out)
    return out


def predicted_lambda(fit, wind, delta_t):
    """The air-change rate the fitted relationship expects, or None without a fit."""
    if not fit or "c" not in fit:
        return None
    value = fit["a"] + fit["b"] * abs(delta_t) + fit["c"] * wind ** 2
    return float(np.sqrt(value)) if value > 0 else None


MIN_OPEN_FITS = 2               # one point cannot give a slope

# A rise has to have bent before its rate can be read off it. Below this fraction of
# the climb to its own fitted ceiling the curve is still near-linear, and lambda and
# the ceiling are trading against each other rather than being measured.
MIN_RISE_COVER = 0.6
MIN_RISE_R2 = 0.95
RISE_DECAY_GAP = 0.5            # how far the two routes may differ before it is news


def fit_open_wind(db, location, min_r2=MIN_R2):
    """How fast an OPEN window exchanges air, as a function of wind speed.

    The quadrature in `fit_weather` still applies - stack pressure and wind pressure
    add as energies - but a wide opening is a different regime from an envelope leak.
    The hypothesis tested here is that the opening is large enough for wind pressure
    to dominate the stack term, which would collapse the three-coefficient form to

        lambda^2 = c*v^2    ->    lambda = k*v

    one parameter, fitted through the origin. One parameter is what makes it testable
    at all: the two-parameter form lambda = lambda0 + k*v fits two measurements
    exactly and leaves no residual to judge it by, while k alone leaves one. The form
    is derived and then tested, and on this room's record it fails the test - see the
    r2 guard below. A negative r2 is reported rather than suppressed, because "the
    opening is buoyancy-driven, not wind-driven" is the finding.

    Only openings the user confirmed with a label count. An inferred "ventilated"
    decay may be a window, a door or a trickle vent, and this bedroom's own record
    has a corridor-door flush sitting numerically between its two window tests - so
    pooling all three yields a median that describes none of them.

    Returns a note - and stores nothing - below MIN_OPEN_FITS confirmed openings."""
    usable = [r for r in db.decays(location, min_r2=min_r2)
              if r["labelled"] == "open" and r["wind_ms"] is not None]
    if len(usable) < MIN_OPEN_FITS:
        return {"n": len(usable), "need": MIN_OPEN_FITS,
                "note": "not enough confirmed window openings yet"}

    lam = np.array([r["lambda_h"] for r in usable], dtype=float)
    wind = np.array([r["wind_ms"] for r in usable], dtype=float)
    if float(np.sum(wind ** 2)) <= 0:
        return {"n": len(usable), "note": "every opening was measured in dead-still air"}

    k = float(np.sum(lam * wind) / np.sum(wind ** 2))        # least squares through 0
    resid = lam - k * wind
    dof = max(1, len(lam) - 1)
    stderr = float(np.sqrt(float(np.sum(resid ** 2)) / dof / float(np.sum(wind ** 2))))
    total = float(np.sum((lam - np.mean(lam)) ** 2))
    out = {"n": len(lam), "k": k, "stderr": stderr,
           "r2": 1 - float(np.sum(resid ** 2)) / total if total > 0 else 0.0,
           "wind_range": [round(float(wind.min()), 2), round(float(wind.max()), 2)],
           "lambda_range": [round(float(lam.min()), 3), round(float(lam.max()), 3)],
           "spread": round(float(lam.max() / max(lam.min(), 1e-9)), 1)}

    # An opening whose decay settled above outdoor air was not exchanging purely with
    # outdoors: part of its make-up air came from the rest of the home, through a
    # door or a corridor, and that path has no wind pressure across it. Wind therefore
    # drives only a fraction of such an event's lambda, and k fitted through it is
    # biased - which way depends on whether the still openings or the windy ones are
    # the mixed ones. The count has to travel with the coefficient.
    def _house_air(r):
        try:
            return r["c_inf"] is not None and r["c_inf"] - r["c_out"] >= features.ASYM_MIN_ABOVE
        except (IndexError, KeyError):
            return False

    mixed = [r for r in usable if _house_air(r)]
    if mixed:
        out["mixed"] = len(mixed)
        levels = ", ".join("%.0f" % r["c_inf"] for r in mixed)
        out["mixed_note"] = (
            f"{len(mixed)} of {len(lam)} opening(s) settled above outdoor CO2 "
            f"(at {levels} ppm), so some of their make-up air came from inside the "
            f"flat, where wind has no pressure to apply. k is not a pure outdoor "
            f"coupling.")
    # With two openings and one coefficient there is a single residual degree of
    # freedom, so a high r-squared is close to guaranteed: a line through the origin
    # between two widely separated points will always look excellent, and the stated
    # standard error is correspondingly optimistic. What would actually test the law
    # is an opening at an intermediate wind speed, which no amount of arithmetic on
    # the existing two can substitute for. Say so wherever the number is shown.
    if len(lam) < 3:
        out["provisional"] = (f"{len(lam)} openings, one coefficient: the fit cannot "
                              f"fail yet. Needs an opening between "
                              f"{out['wind_range'][0]} and {out['wind_range'][1]} m/s "
                              f"to be tested.")

    # A law has to beat the thing it replaces. r2 below zero means the line through the
    # origin predicts these openings worse than their own mean does, so the pooled
    # median is the better model and the coefficient is not stored. That is a result,
    # not a failure: proportionality through the origin assumed wind pressure dominates
    # the stack term, and a still-air opening that still exchanges several air changes
    # an hour says buoyancy is carrying it. lambda = lambda0 + k*v is the form that
    # implies, and two points fit two parameters exactly - so it cannot be fitted here
    # either, and what is needed is a third opening, not a different regression.
    if out["r2"] < 0:
        out["refuted"] = (
            f"lambda is not proportional to wind: fitted through the origin it "
            f"explains less than the mean does (r2={out['r2']:.2f}). The still-air "
            f"opening still gave {float(lam[np.argmin(wind)]):.2f}/h at "
            f"{float(wind.min()):.2f} m/s, where the law expects "
            f"{k * float(wind.min()):.2f}/h - so buoyancy, not wind, drives most of "
            f"it. Falling back to the pooled median until a third opening can "
            f"separate the two terms.")
        db.set_param(location, "lambda_open_k", None, None, 0, "(1/h) per (m/s)", out)
        return out

    db.set_param(location, "lambda_open_k", k, stderr, len(lam), "(1/h) per (m/s)", out)
    return out


def predicted_lambda_open(fit, wind, floor=None):
    """The open-window rate the fitted law expects at a wind speed.

    Wind is held inside the range actually measured rather than extrapolated. A law
    fitted over 0.8 to 3.7 m/s says nothing about a gale, and the failure is not
    symmetric: over-promising air leads the system to recommend a window that will
    not deliver, so the conservative edge is the right one to fall off.

    `floor` is a floor, not a fit. However still the air, an open window cannot
    exchange less than the same room with the window shut."""
    if not fit or "k" not in fit or wind is None:
        return None
    lo, hi = fit["wind_range"]
    held = min(max(float(wind), lo), hi)
    lam = fit["k"] * held
    return max(lam, floor) if floor is not None else lam


class RoomModel:
    """The current parameters for one room, with published defaults where the record
    is still too thin. `basis` records which is which so advice can be honest."""

    def __init__(self, db, location):
        self.location = location
        self.value, self.stderr, self.basis, self.n = {}, {}, {}, {}
        stored = db.params(location)
        for name, (default, _unit) in DEFAULTS.items():
            row = stored.get(name)
            if row is not None and row["value"] is not None and row["n"] >= MIN_N[name]:
                self.value[name] = float(row["value"])
                self.stderr[name] = row["stderr"]
                self.n[name] = int(row["n"])
                self.basis[name] = "learned"
            else:
                self.value[name] = default
                self.stderr[name] = None
                self.n[name] = 0 if row is None else int(row["n"])
                self.basis[name] = "default"
        # A measured volume beats an inferred one everywhere it is used, and the
        # inferred value stays on the record beside it as the thing being checked.
        self.volume_m3 = stored["volume_m3"]["value"] if "volume_m3" in stored else None
        self.volume_inferred_m3 = self.volume_m3
        if "volume_measured_m3" in stored and stored["volume_measured_m3"]["value"]:
            self.volume_m3 = float(stored["volume_measured_m3"]["value"])
        self.volume_basis = ("measured" if "volume_measured_m3" in stored
                             and stored["volume_measured_m3"]["value"] else "inferred")
        # How many people usually occupy it, when it is occupied at all.
        row = stored.get("typical_people")
        self.occupants = max(1, int(row["value"])) if row and row["value"] else 1
        # The open-window rate is a function of wind here, not a constant. See
        # lambda_open_at; the stored pooled median remains the fallback.
        row = stored.get("lambda_open_k")
        # The fit is kept whatever its verdict, because a law that was tested and
        # failed is a result worth showing; only a law that PASSED may be used to
        # predict, so the two are read into separate attributes.
        detail = json.loads(row["detail"]) if row is not None and row["detail"] else None
        self.open_wind_fit = detail
        self.open_wind = (detail if detail and row["n"] >= MIN_OPEN_FITS
                          and not detail.get("refuted") else None)

    @property
    def learned(self):
        """True once the parameters that drive a prediction are the room's own."""
        return all(self.basis[k] == "learned" for k in ("c_out", "lambda_sealed", "source_ppm_h"))

    def lambda_open_at(self, wind=None):
        """The open-window air-change rate at a wind speed, and how it was arrived at.

        A wind law is used only where one has survived its own test. The bedroom's
        two confirmed openings give 4.55/h in 3.7 m/s and 2.58/h in 0.75 m/s: a
        factor of 1.8 for a factor of five in wind, which proportionality through the
        origin cannot produce - it would have the still-air opening at 0.98/h, a
        third of what was measured. Most of an open window's exchange here is
        buoyancy, and the wind adds to it rather than being it.

        So `fit_open_wind` refuses to store a coefficient whose r2 is negative, and
        this falls back to the pooled median. That fallback is the safer error: it
        under-promises on a windy night rather than over-promising on a still one,
        and over-promising is the one that reaches the user as bad advice."""
        fit = getattr(self, "open_wind", None)
        if fit and wind is not None:
            lam = predicted_lambda_open(fit, wind, floor=self.value["lambda_sealed"])
            if lam is not None:
                lo, hi = fit["wind_range"]
                held = "" if lo <= float(wind) <= hi else f", wind held inside {lo}-{hi} m/s"
                mark = " (provisional)" if fit.get("provisional") else ""
                return lam, (f"wind law, k={fit['k']:.2f} per m/s "
                             f"from {fit['n']} openings{held}{mark}")
        return self.value["lambda_open"], self.basis["lambda_open"]

    def lam(self, ventilated, wind=None):
        """Whichever air-change rate applies, shut or open."""
        if not ventilated:
            return self.value["lambda_sealed"]
        return self.lambda_open_at(wind)[0]

    def steady_state(self, occupied, ventilated=False, wind=None):
        """Where the room is heading. The source is stored per person, so presence
        alone is not enough - a living room shared by two fills twice as fast."""
        lam = self.lam(ventilated, wind)
        source = self.value["source_ppm_h"] * self.occupants
        return self.value["c_out"] + (source / lam if occupied else 0.0)

    def forward(self, c0, minutes, occupied, ventilated=False, step_min=5,
                prior=None, now=None, persistence_min=60.0, wind=None):
        """CO2 trajectory as [(minutes_from_now, ppm), ...].

        Closed form per step: over a step where lambda and the source are constant
        the equation is exactly exponential, so no solver is needed.

        With a `prior` - 24 hourly occupancy probabilities from
        features.occupancy_prior - the assumption that whoever is here now stays here
        is dropped. Belief in the present decays towards the long-run pattern with a
        time constant of `persistence_min`, so a short horizon trusts what the room
        is doing and a long one trusts what usually happens at that hour.

        Taking the expected source rather than sampling is exact, not an
        approximation: the equation is linear in the source, so the response to the
        average source is the average of the responses."""
        import time as _time

        lam = self.lam(ventilated, wind)
        source = self.value["source_ppm_h"] * self.occupants
        c_out = self.value["c_out"]
        decay = np.exp(-lam * step_min / 60.0)
        belief = 1.0 if occupied else 0.0
        now = int(_time.time()) if now is None else int(now)

        hour_of = None
        if prior:
            from zoneinfo import ZoneInfo

            from config import settings
            zone = ZoneInfo(settings.USER_TZ)

            def hour_of(ts):
                return datetime.fromtimestamp(ts, zone).hour

        out, c = [], float(c0)
        for step in range(1, int(minutes / step_min) + 1):
            ahead = step * step_min
            if hour_of is None:
                occupancy = belief
            else:
                weight = float(np.exp(-ahead / persistence_min))
                occupancy = weight * belief + (1 - weight) * prior[hour_of(now + ahead * 60)]
            ss = c_out + source * occupancy / lam
            c = ss + (c - ss) * decay
            out.append((ahead, round(c, 1)))
        return out

    def time_to(self, threshold, c0, occupied, ventilated=False, wind=None):
        """Minutes until CO2 reaches a threshold, or None if it never gets there."""
        lam = self.lam(ventilated, wind)
        ss = self.steady_state(occupied, ventilated, wind)
        denom = float(c0) - ss
        if abs(denom) < 1e-9:
            return None
        ratio = (threshold - ss) / denom
        if not 0.0 < ratio <= 1.0:
            return None                      # the threshold lies beyond the steady state
        return float(-np.log(ratio) / lam * 60.0)

    def purge_cost_c(self, t_in, t_out, minutes):
        """Degrees lost by holding a window open, from the room's thermal time constant."""
        if t_in is None or t_out is None or t_in <= t_out:
            return 0.0
        return float((t_in - t_out) * (1.0 - np.exp(-minutes / 60.0 / self.value["tau_t_h"])))

    def describe(self):
        return json.dumps({k: {"value": round(v, 3), "basis": self.basis[k], "n": self.n[k]}
                           for k, v in self.value.items()})
