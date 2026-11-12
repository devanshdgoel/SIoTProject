"""
holdout.py - leave-one-night-out validation of the room model.

Everything else in this project checks the model against itself. The decay route
and the rise route are compared and found to be some percent apart; the fitted
volume is compared with a tape measure; the two sensors are asked whether they
agree about outdoor air. Those are consistency checks, and a consistency check can
be passed by a model that is wrong in a way both routes share - they use the same
c_out, the same despiking, the same mass balance.

This asks the different question, the one a consistency check cannot reach: given
only what was known BEFORE a night happened, how close does the model come to what
the room actually did? One night is held out, every parameter is re-fitted without
it, and the night is then predicted forward from its first sample and scored
against what the sensor recorded.

    for each night N:
        fit lambda and S from every decay and rise that is NOT in N
        predict N forward from its opening concentration
        score against what happened, and against persistence

Persistence - "the room stays where it is" - is the baseline, because it is the
honest rival. A model that cannot beat it has not earned the arithmetic, and a
skill score is the only form in which an error in ppm means anything: 180 ppm of
error is excellent for a night that moved 900 ppm and useless for one that moved 200.

WHAT IS AND IS NOT HELD OUT, because this is where validations quietly cheat:

  held out   every decay fit and every occupied rise overlapping the night, so the
             lambda and the source used to predict it never saw it.
  NOT held   c_out, the outdoor baseline. It comes from labelled window-open periods
             and deep-decay asymptotes across the whole record, and holding it out
             would usually leave nothing to hold. It is a shared constant and a
             shared risk: if it is wrong - and `analysis/outdoor.py` argues it is
             about 65 ppm low - every night here is wrong together, and this
             validation would not notice.
  NOT held   the occupancy labels for the night itself. Who was in the room is an
             INPUT to the prediction, not part of what is being predicted. The
             model is being asked "given that someone slept here, what happened to
             the CO2", which is the question it exists to answer.

So this is validation of lambda and S out of sample, not of the whole pipeline.
That is worth stating plainly rather than letting a skill score imply more.
"""

import math
import time

import numpy as np

from analysis import features
from analysis.room_model import (MIN_R2, MIN_RISE_COVER, MIN_RISE_R2, _robust,
                                 _uncertainty)
from analysis.ventilation import STEP_MIN, trajectory

# A night has to be long enough that the room's time constant matters. Below about
# three hours a bedroom has barely begun to bend, and any model at all scores well
# against persistence for the wrong reason.
MIN_NIGHT_H = 3.0
MIN_NIGHT_SAMPLES = 24


def _overlaps(a0, a1, b0, b1):
    return a0 <= b1 and b0 <= a1


def _rises(db, location, labels, co2, start, end, exclude=None):
    """Occupied rises in a window, optionally dropping those inside one night.

    Recomputed rather than read back, because rises are not stored as events - only
    their pooled result is - and a leave-one-out needs the population itself.
    """
    out = []
    for a, b in features.occupied_windows(db, location, labels, start, end):
        if exclude and _overlaps(a, b, exclude[0], exclude[1]):
            continue
        if features.label_at(labels, (a + b) // 2, "window") == "open":
            continue
        import pandas as pd
        seg = co2.loc[pd.Timestamp(a, unit="s", tz="UTC"):
                      pd.Timestamp(b, unit="s", tz="UTC")].dropna()
        fit = features.fit_rise(seg, _c_out(db, location))
        if fit is None or fit["r2"] < 0.90:
            continue
        people = features.people_at(labels, (a + b) // 2, default=1)
        fit["per_person"] = fit["source_ppm_h"] / max(people, 1)
        fit["usable"] = (fit["covered"] >= MIN_RISE_COVER
                         and fit["r2"] >= MIN_RISE_R2)
        out.append(fit)
    return out


def _c_out(db, location):
    row = db.params(location).get("c_out")
    return float(row["value"]) if row is not None else 420.0


def _fit_without(db, location, labels, co2, start, end, exclude):
    """lambda and the source, from every event that is not inside `exclude`."""
    decays = [r for r in db.decays(location, min_r2=MIN_R2, start=start, end=end)
              if r["inferred"] == "sealed"
              and not (exclude and _overlaps(int(r["start_ts"]), int(r["end_ts"]),
                                             exclude[0], exclude[1]))]
    rises = _rises(db, location, labels, co2, start, end, exclude=exclude)
    usable = [r for r in rises if r["usable"]] or rises

    lam_values = ([r["lambda_h"] for r in decays]
                  + [r["lambda_h"] for r in rises if r["usable"]])
    lam, lam_err, n_lam = _robust(lam_values)
    src, src_err, n_src = _robust([r["per_person"] for r in usable])
    return {"lambda_h": lam, "lambda_stderr": lam_err, "n_lambda": n_lam,
            "source_ppm_h": src, "source_stderr": src_err, "n_source": n_src,
            "n_decays": len(decays), "n_rises": len(usable)}


def _score(actual, predicted):
    """MAE of a prediction against what happened, in ppm."""
    both = min(len(actual), len(predicted))
    if both == 0:
        return None
    a = np.asarray(actual[:both], float)
    p = np.asarray(predicted[:both], float)
    return float(np.mean(np.abs(a - p)))


def one_night_out(db, location, days=21, now=None):
    """Predict each night from a model that never saw it.

    Returns a dict with one entry per night and a pooled summary. Nights with too
    little data, or for which the remaining events cannot support a fit, are
    reported with the reason rather than dropped silently - a validation that
    quietly discards the nights it cannot handle is choosing its own exam paper.
    """
    now = int(time.time()) if now is None else int(now)
    start = now - int(days) * 86400
    raw = features.series(db, "co2_ppm", location, start, now)
    co2, _ = features.despike(raw)
    labels = db.labels(location=location, start=start, end=now)
    c_out = _c_out(db, location)

    report = {"location": location, "c_out": round(c_out, 1), "nights": [],
              "skipped": [], "days": days}

    for a, b, day in features.night_windows(db, start, now):
        hours = (b - a) / 3600.0
        if hours < MIN_NIGHT_H:
            report["skipped"].append({"day": day, "why": "%.1f h is too short" % hours})
            continue
        import pandas as pd
        seg = co2.loc[pd.Timestamp(a, unit="s", tz="UTC"):
                      pd.Timestamp(b, unit="s", tz="UTC")].dropna()
        if seg.size < MIN_NIGHT_SAMPLES:
            report["skipped"].append(
                {"day": day, "why": "only %d samples in the room that night" % seg.size})
            continue

        fit = _fit_without(db, location, labels, co2, start, now, exclude=(a, b))
        if fit["lambda_h"] is None or fit["source_ppm_h"] is None:
            report["skipped"].append(
                {"day": day, "why": "no usable events left once this night was removed "
                                    "(%d decays, %d rises)"
                                    % (fit["n_decays"], fit["n_rises"])})
            continue

        actual = [float(v) for v in seg.to_numpy(float)]
        minutes = int(round((seg.index[-1] - seg.index[0]).total_seconds() / 60.0))
        people = features.people_at(labels, (a + b) // 2, default=1)
        lam, source = fit["lambda_h"], fit["source_ppm_h"] * people
        steps = trajectory(actual[0], minutes, lam, source, c_out, occupied=True)
        # trajectory steps every STEP_MIN; the series is on the feature grid, so
        # compare on the coarser of the two rather than interpolating a prediction
        # into looking better resolved than it is.
        every = max(1, int(STEP_MIN * 60 / features.GRID_S))
        stamps = [int(t.timestamp()) for t in seg.index][::every][1:]
        on_grid = actual[::every][1:]
        predicted = [c for _m, c in steps]

        # Score only while the room is believed shut. The last sample of a night is
        # very often the morning airing - 2971 ppm to 2154 in a quarter of an hour on
        # 3 Nov - and a model told nothing about an opening cannot be marked against
        # one. Left in, it turned a steady +500 ppm over-prediction into an apparent
        # +1334 and made a metric out of an event the model never saw.
        keep = [i for i, ts in enumerate(stamps)
                if features.label_at(labels, ts, "window") != "open"]
        dropped = len(on_grid) - len(keep)
        on_grid = [on_grid[i] for i in keep]
        predicted = [predicted[i] for i in keep if i < len(predicted)]
        mae = _score(on_grid, predicted)
        naive = _score(on_grid, [actual[0]] * len(on_grid))
        if mae is None or naive in (None, 0) or len(on_grid) < 12:
            report["skipped"].append(
                {"day": day, "why": "too little of the night was shut to score"})
            continue

        # Signed error over the last third, rather than at one final sample. A single
        # point is at the mercy of whatever happened in the last five minutes; the
        # drift over the closing hours is the thing that matters for a model used to
        # answer "what will this room be like by morning".
        n = min(len(on_grid), len(predicted))
        tail = max(1, n // 3)
        late = float(np.mean([predicted[i] - on_grid[i] for i in range(n - tail, n)]))
        entry = {"day": day, "hours": round(hours, 1), "people": people,
                 "n_points": n, "dropped_open": dropped,
                 "start_ppm": round(actual[0], 1),
                 "peak_ppm": round(max(on_grid), 1),
                 "lambda_used": round(lam, 3),
                 "source_used": round(fit["source_ppm_h"], 1),
                 # The ceiling is what the two fitted numbers jointly claim the room
                 # is heading for, and it is the most interrogable thing here: a
                 # bedroom that never passes 3000 ppm cannot be heading for 4000.
                 "ceiling_ppm": round(c_out + source / lam),
                 "fitted_from": "%d decays + %d rises, none of them tonight"
                                % (fit["n_decays"], fit["n_rises"]),
                 "mae_ppm": round(mae, 1), "naive_mae_ppm": round(naive, 1),
                 "skill": round(1.0 - mae / naive, 3),
                 "late_bias_ppm": round(late, 1)}
        report["nights"].append(entry)

    nights = report["nights"]
    if nights:
        skills = [e["skill"] for e in nights]
        errs = [e["late_bias_ppm"] for e in nights]
        report["summary"] = {
            "n_nights": len(nights),
            "median_skill": round(float(np.median(skills)), 3),
            "worst_skill": round(float(min(skills)), 3),
            "best_skill": round(float(max(skills)), 3),
            "beat_persistence": sum(1 for s in skills if s > 0),
            "median_mae_ppm": round(float(np.median([e["mae_ppm"] for e in nights])), 1),
            "median_late_bias_ppm": round(float(np.median(errs)), 1),
            "late_bias_spread_ppm": round(float(np.std(errs, ddof=1)), 1)
            if len(errs) > 1 else None,
            "nights_biased_high": sum(1 for e in errs if e > 0),
            "median_ceiling_ppm": round(float(np.median(
                [e["ceiling_ppm"] for e in nights]))),
            "highest_observed_ppm": round(float(max(e["peak_ppm"] for e in nights))),
            "lambda_across_folds": _uncertainty([e["lambda_used"] for e in nights])}
        # The bias is the more damning of the two numbers. Scatter is the weather;
        # a median sitting well off zero, in the same direction on night after
        # night, says the model is systematically wrong - and no amount of further
        # data will average that away.
        s = report["summary"]
        # Whether the late bias is a BIAS or just scatter decides what to do about
        # it, and they look identical in a median. A systematic offset is a wrong
        # parameter and is worth chasing; scatter of this size over seven nights is
        # the weather, and chasing it would be fitting noise. So the median is put
        # against its own standard error before any claim is made either way.
        spread = s["late_bias_spread_ppm"]
        se = (spread / math.sqrt(len(errs))) if spread else None
        med = s["median_late_bias_ppm"]
        systematic = bool(se and abs(med) > 2 * se)
        s["bias_is_systematic"] = systematic
        s["late_bias_se_ppm"] = round(se, 1) if se else None
        if systematic:
            bias_note = ("It runs %s by %+.0f ppm (+/-%.0f) through the last third of "
                         "the night, on %d of %d nights - a real offset, not scatter"
                         % ("HOT" if med > 0 else "COLD", med, se,
                            s["nights_biased_high"] if med > 0
                            else len(nights) - s["nights_biased_high"], len(nights)))
        else:
            bias_note = ("No systematic bias: %+.0f ppm late in the night against a "
                         "standard error of %.0f, %d nights high and %d low. The "
                         "model is not leaning one way - it is noisy, by about "
                         "%.0f ppm by morning"
                         % (med, se or 0, s["nights_biased_high"],
                            len(nights) - s["nights_biased_high"], spread or 0))
        report["summary"]["verdict"] = (
            "beats 'it stays where it is' on %d of %d nights, median skill %+.0f%%, "
            "typical error %.0f ppm. %s"
            % (s["beat_persistence"], len(nights), s["median_skill"] * 100,
               s["median_mae_ppm"], bias_note))
        if s["median_ceiling_ppm"] > s["highest_observed_ppm"] * 1.25:
            report["summary"]["ceiling_warning"] = (
                "the fitted parameters claim this room is heading for %d ppm, and it "
                "has never been observed above %d. lambda and the source trade "
                "against each other in every rise fit, so a source too high and a "
                "lambda too low reproduce the early climb and then overshoot - which "
                "is exactly the signature above"
                % (s["median_ceiling_ppm"], s["highest_observed_ppm"]))
    return report
