"""
episodes.py

Reading the record back as a list of things that happened, so they can be confirmed.

The system already infers states continuously - it has to, in order to fit anything -
but those inferences are invisible and unchecked. This turns them into discrete,
reviewable claims: "the bedroom window was open 09:00 to 10:10", "somebody was in the
living room 19:30 to midnight". Each one carries the evidence that produced it and a
confidence, and waits for a yes, a no or a not-sure.

Three detectors, each reading a different physical signature:

  window / ventilation  an air-change rate several times the room's sealed figure.
                        A window is not subtle - it multiplies lambda by three to
                        fifteen - so this is a threshold on a measured quantity, not
                        a pattern that has to be recognised.

  occupancy             CO2 climbing faster than leakage alone can account for. A
                        sealed room left alone can only fall towards outdoor air; if
                        it is rising, something in it is producing CO2.

  cooking               CO2, temperature and humidity all rising together. Each alone
                        is ambiguous - a person raises all three a little - but a hob
                        raises heat and water vapour far out of proportion to its CO2,
                        and that ratio is what separates them.

Nothing here is trained. Each detector is an arithmetic test on a physical quantity,
and the thresholds it uses are themselves measured from the room rather than chosen.
What the verdicts change is the measurements, not any model weights: see `learn_from`
at the bottom.
"""

import json
import time

import numpy as np
import pandas as pd

from analysis import features

MIN_EPISODE_S = 900          # shorter than this and there is nothing to confirm
COOK_CO2_PPM_H = 150.0       # all three must be rising together
COOK_TEMP_C_H = 0.8
COOK_RH_PCT_H = 1.5


def _span(seg):
    return int(seg.index[0].timestamp()), int(seg.index[-1].timestamp())


def _runs_from_mask(index, mask, bridge=3):
    """Contiguous True stretches as (start_ts, end_ts), bridging brief interruptions."""
    out, start, last, missed = [], None, None, 0
    for ts, flag in zip(index, mask):
        if flag:
            if start is None:
                start = ts
            last, missed = ts, 0
        elif start is not None:
            missed += 1
            if missed > bridge:
                out.append((int(start.timestamp()), int(last.timestamp())))
                start, missed = None, 0
    if start is not None:
        out.append((int(start.timestamp()), int(last.timestamp())))
    return out


def detect_ventilation(db, location, model, start, end):
    """Decays whose air-change rate is far above this room's sealed figure.

    The comparison is against the room's own measured sealed rate, so a draughty flat
    and a sealed one are judged on their own terms rather than against a constant."""
    sealed = model.value["lambda_sealed"]
    out = []
    for row in db.decays(location, min_r2=0.90):
        if not (start <= row["start_ts"] <= end):
            continue
        if row["end_ts"] - row["start_ts"] < MIN_EPISODE_S:
            continue
        ratio = row["lambda_h"] / max(sealed, 1e-6)
        if ratio < 2.0:
            continue
        out.append({
            "location": location, "kind": "window", "value": "open",
            "start_ts": row["start_ts"], "end_ts": row["end_ts"],
            "confidence": round(min(0.95, 0.35 + 0.1 * ratio), 2),
            "evidence": json.dumps({
                "lambda_h": round(row["lambda_h"], 3),
                "sealed_lambda_h": round(sealed, 3),
                "times_faster": round(ratio, 1),
                "co2_fell": [round(row["c_start"]), round(row["c_end"])],
                "r2": round(row["r2"], 4),
                "why": f"cleared {ratio:.1f}x faster than this room does when shut"}),
        })
    return out


def detect_occupancy(db, location, model, co2, start, end):
    """Stretches where CO2 rose faster than leakage can explain.

    An empty sealed room can only fall toward outdoor air. A sustained rise therefore
    means a source, and in a home that means a person."""
    if co2.dropna().size < 12:
        return []
    lam, c_out = model.value["lambda_sealed"], model.value["c_out"]
    smooth = co2.rolling(3, center=True, min_periods=1).median()
    slope = smooth.diff() * 3600.0 / features.GRID_S
    # what the room would be doing with nobody in it
    expected_empty = -lam * (smooth - c_out)
    excess = (slope - expected_empty).fillna(0.0)
    mask = (excess > 120.0).to_numpy()

    out = []
    for a, b in _runs_from_mask(co2.index, mask):
        if b - a < MIN_EPISODE_S:
            continue
        seg = co2.loc[pd.Timestamp(a, unit="s", tz="UTC"):
                      pd.Timestamp(b, unit="s", tz="UTC")].dropna()
        if seg.size < 3:
            continue                       # the run was all gaps
        produced = float(np.nanmean(excess.reindex(seg.index)))
        if not np.isfinite(produced):
            continue
        people = max(1, int(round(produced / max(model.value["source_ppm_h"], 1.0))))
        out.append({
            "location": location, "kind": "occupancy", "value": "present",
            "start_ts": a, "end_ts": b,
            "confidence": round(min(0.95, 0.4 + produced / 2000.0), 2),
            "evidence": json.dumps({
                "co2_rose": [round(float(seg.iloc[0])), round(float(seg.iloc[-1]))],
                "production_ppm_h": round(produced),
                "one_person_is_about": round(model.value["source_ppm_h"]),
                "people_guess": people,
                "why": "CO2 climbed faster than this room can leak"}),
        })
    return out


def detect_cooking(db, location, co2, start, end):
    """CO2, temperature and humidity rising together.

    A person raises all three slightly. A hob raises heat and water vapour far out of
    proportion to the CO2 it adds, and it is that disproportion the test looks for,
    which is why it needs all three channels rather than any one of them."""
    temp = features.series(db, "indoor_temp_c", location, start, end)
    rh = features.series(db, "indoor_rh_pct", location, start, end)
    if temp.dropna().size < 12 or rh.dropna().size < 12 or co2.dropna().size < 12:
        return []
    per_h = lambda s: s.rolling(3, center=True, min_periods=1).median().diff() * 3600.0 / features.GRID_S  # noqa: E731
    frame = pd.DataFrame({"co2": per_h(co2), "t": per_h(temp), "rh": per_h(rh)}).dropna()
    if frame.empty:
        return []
    mask = ((frame["co2"] > COOK_CO2_PPM_H) & (frame["t"] > COOK_TEMP_C_H)
            & (frame["rh"] > COOK_RH_PCT_H)).to_numpy()

    out = []
    for a, b in _runs_from_mask(frame.index, mask, bridge=2):
        if b - a < 600:                      # cooking episodes are shorter than the rest
            continue
        win = frame.loc[pd.Timestamp(a, unit="s", tz="UTC"):pd.Timestamp(b, unit="s", tz="UTC")]
        out.append({
            "location": location, "kind": "cooking", "value": "yes",
            "start_ts": a, "end_ts": b,
            "confidence": round(min(0.9, 0.3 + float(win["t"].mean()) / 6.0), 2),
            "evidence": json.dumps({
                "co2_rate_ppm_h": round(float(win["co2"].mean())),
                "temp_rate_c_h": round(float(win["t"].mean()), 2),
                "humidity_rate_pct_h": round(float(win["rh"].mean()), 2),
                "why": "heat and moisture rose far out of proportion to the CO2"}),
        })
    return out


def detect(db, days=3, now=None):
    """Run every detector over every room and store what has not been judged already."""
    from analysis.room_model import RoomModel, rooms

    now = int(time.time()) if now is None else int(now)
    start = now - days * 86400
    found = []
    for location in rooms(db):
        model = RoomModel(db, location)
        co2, _ = features.despike(features.series(db, "co2_ppm", location, start, now))
        found += detect_ventilation(db, location, model, start, now)
        found += detect_occupancy(db, location, model, co2, start, now)
        found += detect_cooking(db, location, co2, start, now)
    for row in found:
        row["detected_at"] = now
    # Anything the user has already told us about needs no confirming.
    out = []
    for row in found:
        labels = db.labels(location=row["location"], kind=row["kind"],
                           start=row["start_ts"], end=row["end_ts"])
        if not any(lb["note"] != "confirmed from a detection" for lb in labels):
            out.append(row)
    return db.add_episodes(out), len(found)


def learn_from(db, location=None):
    """What the verdicts actually change. No weights are updated anywhere.

    A confirmation becomes an ordinary label, and labels enter the estimates in four
    places, each of them arithmetic:

      1. the sealed / ventilated split. Decays are separated by the gap that minimises
         within-group variance; a labelled decay is placed by the label rather than by
         that cut, which moves where the cut falls for everything else.
      2. the outdoor baseline. A confirmed window-open period with nobody in the room
         IS outdoor air as this sensor reads it, so c_out stops being inferred from a
         low percentile and becomes an observation.
      3. the source term. A confirmed occupancy, with its headcount, says how much one
         person produces - and therefore how big the room is.
      4. the occupancy pattern, which is a count of labelled hours, not a fit.

    A denial is equally useful: it moves a decay from the ventilated group into the
    sealed one, which shifts the split the other way. A not-sure is recorded and then
    excluded from all four, because an ambiguous answer is not evidence.

    So "learning" here is re-estimating physical constants from a growing set of
    observations. The same arithmetic runs on one label or a thousand; more of them
    only narrows the error bars."""
    from analysis.room_model import learn, rooms

    return {name: learn(db, name) for name in ([location] if location else rooms(db))}
