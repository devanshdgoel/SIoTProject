"""
particles.py

What the particle counter is for, which is not "a second air-quality number".

CO2 and particles are carried by the same air exchange and lost in different ways,
and that difference is the only genuinely independent measurement of the air-change
rate this project can make.

    CO2:        dC/dt = -lambda (C - C_out)
    particles:  dP/dt = lambda * p * P_out - (lambda + k) P + S/V

CO2 is a gas. It leaves only with the air, so a decay measures lambda and nothing
else. A particle also settles out on the floor, the walls and the duvet, at a
deposition rate k, so a particle decay measures lambda + k. Subtract one from the
other over the same event and you have k: a second physical property of the room,
from a sensor that shares no baseline, no calibration and no arithmetic with the
CO2 path.

That matters here specifically. `estimators.compare` offered three routes to lambda
and two of them subtract the same c_out, so an error there moves them together and
their agreement is partly tautological; the third turned out to measure the shape
of the occupancy cycle rather than the room. Particles have none of those couplings.

THE FALSIFICATION. Deposition is not a free parameter to be fitted and admired: it
is strongly size-dependent, and the direction is not in doubt. Bigger particles
settle faster, so

    k(PM1) < k(PM2.5) < k(PM10)

always, in every room, by gravity. The sensor reports all three from the same air
at the same instant, so this ordering is a prediction the hardware cannot help but
test. If the fitted k rise with size, the method is measuring deposition. If they
come out in some other order, it is fitting noise and says so - which is the whole
reason to compute all three rather than just PM2.5.

PENETRATION. With the window shut, particles arriving from outside are filtered by
the gaps they come through, so only a fraction p survives. At steady state with no
indoor source,

    P_in / P_out = lambda * p / (lambda + k)

so once lambda and k are known the floor a decay settles to gives p. That turns
"open the window to clear the CO2" into "open it, and here is the PM it will let
in", which is the trade-off `ventilation.plan_purge` currently weighs with a
hand-picked divisor and no measurement behind it.

WHAT THIS NEEDS FROM YOU. CO2 events happen because you breathe. Particle events do
not: there has to be an indoor source and then an end to it. One deliberate cooking
event with the door shut gives every number on this page.
"""

import numpy as np

from analysis import features

# The three masses the PMS5003 reports, with the size each stands for. The order is
# load-bearing: it is what the deposition ordering is checked against.
PM_SIZES = (("pm1_0", 1.0), ("pm2_5", 2.5), ("pm10", 10.0))

MIN_PM_DROP = 0.5          # a decay must lose half its excess to be worth fitting
MIN_PM_MINUTES = 40        # and run long enough for the curvature to be real
MIN_PM_PEAK = 15.0         # ug/m3 above the floor; below this it is sensor noise
MIN_PM_R2 = 0.85           # particle data is noisier than CO2; do not demand 0.99


def pm_decays(series, min_minutes=MIN_PM_MINUTES, min_drop=MIN_PM_DROP,
              min_peak=MIN_PM_PEAK):
    """Stretches where particles fall steadily: a source stopped and the room cleared.

    Deliberately simpler than `features.decay_windows`. A particle event has a
    sharp peak and a long tail rather than CO2's slow build, and the floor it
    settles to is set by outdoor air leaking in rather than by a known baseline, so
    the test is "falling from a real peak for long enough" and the floor is left to
    the fit.
    """
    s = series.dropna()
    if s.size < 8:
        return []
    smooth = s.rolling(3, center=True, min_periods=1).median()
    # Bridge the wobble. Particle counts are far noisier than CO2 - the counter
    # samples a tenth of a litre and Poisson noise alone moves it several percent
    # - so a real decay ticks upward every few samples. Splitting the run there
    # turns one eight-hour clearance into forty fragments, none long enough to
    # fit, which is exactly what happened to the first night of data this saw.
    falling = (smooth.diff() <= 0).fillna(False)
    out = [(s.index.get_loc(a), s.index.get_loc(b))
           for a, b in features._runs(falling, bridge=3)]

    keep = []
    for a, b in out:
        seg = s.iloc[a:b + 1]
        if seg.size < 6:
            continue
        minutes = (seg.index[-1] - seg.index[0]).total_seconds() / 60.0
        peak, floor = float(seg.iloc[0]), float(seg.min())
        if minutes < min_minutes or peak - floor < min_peak:
            continue
        if (peak - float(seg.iloc[-1])) < min_drop * (peak - floor):
            continue
        keep.append((seg.index[0], seg.index[-1]))
    return keep


def fit_pm_decay(seg):
    """Total loss rate (lambda + k) from one particle decay, with its own floor free.

    The asymptote has to be free. A particle decay does not head for zero - it heads
    for whatever outdoor air is leaking in and surviving the journey - and forcing
    it to zero would do here exactly what forcing CO2 to c_out did there: buy the
    misfit off with a rate that is too low.
    """
    seg = seg.dropna()
    if seg.size < 6:
        return None
    fit = features.fit_decay_free(seg, 0.0)
    if fit is None or fit["r2"] < MIN_PM_R2:
        return None
    return {"total_loss_h": fit["lambda_h"], "floor": fit["c_inf"],
            "r2": fit["r2"], "stderr": fit.get("stderr"),
            "n": int(seg.size), "n_eff": fit.get("n_eff"),
            "start": seg.index[0], "end": seg.index[-1],
            "peak": float(seg.iloc[0])}


def _lambda_during(db, location, start_ts, end_ts, model=None):
    """The air-change rate that applied during a window, and where it came from.

    A measured CO2 decay overlapping the same minutes is worth far more than the
    room's pooled average, because deposition is the DIFFERENCE between two rates
    and a difference of two averages taken at different times is mostly noise. The
    pooled value is accepted as a fallback but labelled as one.
    """
    best, overlap = None, 0.0
    for r in db.decays(location):
        lo, hi = int(r["start_ts"]), int(r["end_ts"])
        seen = min(end_ts, hi) - max(start_ts, lo)
        if seen > overlap and seen > 0:
            best, overlap = r, seen
    span = max(float(end_ts - start_ts), 1.0)
    if best is not None and overlap / span >= 0.5:
        return float(best["lambda_h"]), "measured CO2 decay over the same minutes"

    # No decay, but a particle event does not need the room to be empty - and the
    # best one so far happened while someone slept through it. An occupied rise
    # over the same hours measures the same lambda and is often the better curve:
    # last night's was r2 = 0.999 over five hours against a pooled value standing
    # on eleven mixed events.
    import pandas as _pd

    co2 = features.series(db, "co2_ppm", location,
                          start_ts - 1800, end_ts + 1800).dropna()
    seg = co2.loc[_pd.Timestamp(start_ts, unit="s", tz="UTC"):
                  _pd.Timestamp(end_ts, unit="s", tz="UTC")]
    # The RISING LIMB of the window, not the whole of it. A particle decay can
    # easily outlast the occupancy that started it - last night's ran from 02:35
    # to 11:25 while the sleeper left around 08:00 - so the window contains a rise
    # and then a fall, and fitting one exponential across both describes neither.
    # The climb to the peak is the part that measures lambda.
    # ...and only the OCCUPIED part of that limb. fit_rise assumes a source running
    # from the first sample, so a flat lead-in before anyone came in breaks it: last
    # night the window opened at 02:35 and the sleeper arrived near 03:30, and
    # fitting from 02:35 gave lambda 0.068 with a 9611 ppm ceiling - a fit the
    # `covered` gate correctly refused. Intersecting with the occupancy the room
    # itself reports turns the same data into r2 = 0.999.
    labels = db.labels(location=location, start=start_ts - 86400, end=end_ts + 3600)
    for occ_a, occ_b in features.occupied_windows(db, location, labels,
                                                  start_ts - 3600, end_ts + 3600):
        a = max(occ_a, start_ts)
        b = min(occ_b, end_ts)
        if b - a < 3 * 3600:
            continue
        limb = co2.loc[_pd.Timestamp(a, unit="s", tz="UTC"):
                       _pd.Timestamp(b, unit="s", tz="UTC")]
        if limb.size < 24:
            continue
        rise = features.fit_rise(limb, model.value["c_out"] if model else 400.0)
        if rise and rise["r2"] >= 0.95 and rise.get("covered", 0) >= 0.5:
            return float(rise["lambda_h"]), "CO2 rise of the occupancy inside it"

    # Last resort, and the one that actually fires: the room's own climb, with the
    # lead-in trimmed by searching start offsets exactly as best_decay_fit does.
    #
    # This is needed more often than it should be. Occupancy is taken from labels
    # and from ring sleep sessions, and the ring publishes a night only after it
    # syncs - so the morning after the best particle event this project has
    # recorded, the system had no idea anyone had been in the room, while the CO2
    # trace went from 553 to 3469 ppm. A rise like that is not ambiguous evidence
    # of occupancy; it just was not being read as any.
    limb = seg.loc[:seg.idxmax()]
    best = None
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4):
        part = limb.iloc[int(len(limb) * frac):]
        if part.size < 24:
            continue
        rise = features.fit_rise(part, model.value["c_out"] if model else 400.0)
        if rise is None or rise["r2"] < 0.95 or rise.get("covered", 0) < 0.5:
            continue
        if best is None or rise["covered"] > best["covered"]:
            best = rise
    if best is not None:
        return float(best["lambda_h"]), "CO2 rise inside the same window"
    if model is not None:
        return float(model.value["lambda_sealed"]), "the room's pooled lambda_sealed"
    return None, "no CO2 rate available"


def deposition(db, location, days=14, now=None, model=None):
    """Deposition rate per particle size, and whether it behaves like deposition.

    For each size: find particle decays, fit the total loss rate, subtract the CO2
    air-change rate measured over the same minutes, and keep what is left.

    Returns a dict per size plus an `ordered` verdict on the size dependence, which
    is the test that makes this physics rather than curve fitting.
    """
    import time as _time

    now = int(_time.time()) if now is None else int(now)
    start = now - days * 86400
    out = {"location": location, "sizes": {}, "events": 0}

    # One window for all three sizes.
    #
    # Detecting a decay separately per size is wrong for the one comparison that
    # matters. The sizes are noisier at the coarse end, so their runs start and end
    # at different times, and the first version of this fitted PM2.5 over 530
    # minutes and PM10 over 375 - then compared the rates as if they described the
    # same event. They did not, and the "wrong" ordering it produced was an artefact
    # of the windows rather than a finding about the room.
    #
    # PM2.5 picks the window: it is the best-conditioned of the three, high enough
    # above the counter's noise to give a clean run and fine enough not to be
    # dominated by a few coarse counts.
    pivot = features.series(db, "pm2_5", location, start, now)
    windows = pm_decays(pivot) if pivot.dropna().size >= 12 else []

    for metric, micron in PM_SIZES:
        s_ = features.series(db, metric, location, start, now)
        if s_.dropna().size < 12:
            continue
        ks, rows = [], []
        for a, b in windows:
            seg = s_.loc[a:b].dropna()
            if seg.size < 6:
                continue
            fit = fit_pm_decay(seg)
            if fit is None:
                continue
            lam, basis = _lambda_during(db, location, int(a.timestamp()),
                                        int(b.timestamp()), model)
            if lam is None:
                continue
            k = fit["total_loss_h"] - lam
            rows.append({"total_loss_h": round(fit["total_loss_h"], 3),
                         "lambda_h": round(lam, 3), "k_h": round(k, 3),
                         "floor": round(fit["floor"], 1), "r2": round(fit["r2"], 4),
                         "peak": round(fit["peak"], 1), "basis": basis,
                         "minutes": round((b - a).total_seconds() / 60),
                         "start_ts": int(a.timestamp())})
            # A negative k means particles left SLOWER than the air did, which is
            # impossible: deposition can only add to the loss. Kept in the record
            # because a run of them says the CO2 rate is wrong, not the particles.
            if k > 0:
                ks.append(k)
        if rows:
            med, err, n = _robust_k(ks)
            out["sizes"][metric] = {"micron": micron, "events": len(rows),
                                    "k_h": med, "stderr": err, "n_usable": n,
                                    "negative": len(rows) - n, "fits": rows}
            out["events"] += len(rows)
    out["window_minutes"] = [round((b - a).total_seconds() / 60) for a, b in windows]

    power = _ordering_power(db, location, windows)
    if power:
        out["coarse_share"] = power["coarse_share"]
    out.update(_check_ordering(out["sizes"], power))
    return out


def _robust_k(values):
    if not values:
        return None, None, 0
    a = np.asarray(values, dtype=float)
    med = float(np.median(a))
    if a.size < 2:
        return round(med, 3), None, 1
    sd = 1.4826 * float(np.median(np.abs(a - med)))
    sd = max(sd, 0.05 * abs(med))
    return round(med, 3), round(float(1.253 * sd / np.sqrt(a.size)), 3), int(a.size)


# Below this the three masses are all measuring the same fine aerosol and no
# ordering can be resolved, however good the fits are. Expressed as the share of
# PM10 mass that is coarse (above 2.5 um).
MIN_COARSE_SHARE = 0.25


def _ordering_power(db, location, windows):
    """Is there enough coarse material for the size test to mean anything?

    The falsification only works if the three masses describe genuinely different
    particles. A combustion or cooking aerosol is almost entirely sub-micron, so
    PM1, PM2.5 and PM10 are three views of one population and their decay rates
    MUST come out nearly equal - whatever the room is doing.

    Last night is the case in point: 16% of PM10 mass was coarse and 0.56% of the
    counts were above 2.5 um. The rates duly came out 0.32 and 0.29 with PM10 a
    shade SLOWER, which would have been reported as "deposition does not rise with
    size, so these are fits to noise". That verdict would have been wrong. The test
    had no power to begin with, and a test that reports failure when it cannot
    discriminate is as misleading as one that cannot fail.
    """
    import pandas as _pd

    if not windows:
        return None
    fine = features.series(db, "pm2_5", location)
    coarse = features.series(db, "pm10", location)
    a, b = windows[0][0], windows[-1][1]
    j = _pd.DataFrame({"f": fine, "c": coarse}).dropna().loc[a:b]
    if j.empty or float(j["c"].median()) <= 0:
        return None
    share = float(((j["c"] - j["f"]) / j["c"]).median())
    return {"coarse_share": round(share, 3), "enough": share >= MIN_COARSE_SHARE}


def _check_ordering(sizes, power=None):
    if power is not None and not power.get("enough", True):
        return {"ordered": None, "power": power,
                "ordering_note": (
                    f"only {power['coarse_share'] * 100:.0f}% of PM10 mass is coarse, "
                    f"so all three sizes are measuring the same fine aerosol and "
                    f"their rates must come out nearly equal whatever the room does. "
                    f"The size test has no power here - that is not the same as "
                    f"failing it. It needs an event with real coarse material: "
                    f"vacuuming, a made bed, something dusty rather than something "
                    f"burnt.")}
    return _check_ordering_rates(sizes)


def _check_ordering_rates(sizes):
    """Does fitted deposition rise with particle size, as gravity requires?

    This is the falsification and it costs nothing: the sensor reports all three
    masses from the same air at the same instant, so no extra experiment is needed
    and no tuning can rescue a wrong answer. Two sizes can only be compared, not
    confirmed - a pair agrees by chance half the time - so the verdict stays
    provisional until all three are in.
    """
    have = [(m, d["k_h"]) for m, d in sizes.items()
            if d.get("k_h") is not None]
    order = [m for m, _mic in PM_SIZES if m in dict(have)]
    ks = [dict(have)[m] for m in order]
    if len(ks) < 2:
        return {"ordered": None,
                "ordering_note": "needs at least two sizes with a fitted rate"}
    rising = all(b > a for a, b in zip(ks, ks[1:]))
    note = " < ".join(f"k({m})={v:.2f}" for m, v in zip(order, ks))
    if rising and len(ks) >= 3:
        return {"ordered": True,
                "ordering_note": (f"deposition rises with size as gravity requires: "
                                  f"{note}. Three sizes from one sensor in one air "
                                  f"sample, so this is a real test and it passed.")}
    if rising:
        return {"ordered": True,
                "ordering_note": (f"{note}, which is the right direction - but two "
                                  f"sizes agree by chance half the time. Provisional "
                                  f"until the third is fitted.")}
    return {"ordered": False,
            "ordering_note": (f"deposition does NOT rise with particle size: {note}. "
                              f"Gravity does not permit that, so these are fits to "
                              f"noise rather than measurements of deposition, and "
                              f"no k should be believed from them.")}


# Where to look for "outdoor", best first. AURN is real instruments; Open-Meteo is
# a model on a ~11 km grid and was measuring a third of what the instruments read
# on the night this was written, so taking whichever is present is not good enough.
def outdoor_series(db, metric, start, now):
    """Outdoor particles from the best source available, and which one it was."""
    for src, kind in (("aurn", "measured"), ("open_meteo", "modelled")):
        with db.conn() as c:
            locs = [r[0] for r in c.execute(
                "SELECT DISTINCT location FROM observations WHERE source=? "
                "AND metric=? AND quality!='suspect' AND ts>?",
                (src, metric, start))]
        best = None
        for loc in locs:
            s_ = features.series(db, metric, loc, start, now).dropna()
            if best is None or s_.size > best[1].size:
                best = (loc, s_)
        if best and best[1].size >= 24:
            return best[1], f"{best[0]} ({kind})"
    return None, None


def calibration_bound(db, location, k_h, lambda_h, days=14, now=None,
                      metric="pm2_5"):
    """How much the counter over-reads, bounded by a law it cannot break.

    With the window shut and no indoor source, the steady state is

        P_in / P_out = lambda * p / (lambda + k)

    and p, the fraction of outdoor particles surviving the journey in, cannot
    exceed 1 - nothing indoors creates particles out of filtered air. So the
    measured ratio implies a value for p, and any p above 1 is not a discovery
    about the building, it is an inconsistency in the inputs.

    That makes it a calibration. Against a reference instrument a few kilometres
    away, with lambda and k measured in this room from two independent tracers, the
    excess over p = 1 is what the counter is reading high by:

        over-read factor = ratio * (lambda + k) / lambda,   when that exceeds 1

    Low-cost optical counters are known to read high against reference instruments,
    because they infer size from scattered light and humidity swells the particles.
    This puts a number on it for THIS unit without co-locating it with anything.

    The alternatives have to be said, because the arithmetic cannot separate them:
    a residual indoor source during the supposedly quiet hours, a genuinely dirtier
    street than the reference site, or lambda being underestimated would each
    produce the same excess. It is an upper bound on the over-read, not a
    measurement of it.
    """
    import time as _time

    now = int(_time.time()) if now is None else int(now)
    start = now - days * 86400
    inside = features.series(db, metric, location, start, now).dropna()
    outside, src = outdoor_series(db, metric, start, now)
    if inside.size < 24 or outside is None or not lambda_h:
        return None

    pair = inside.to_frame("in").join(outside.to_frame("out"), how="inner").dropna()
    pair = pair[pair["out"] > 2.0]
    if pair.shape[0] < 12:
        return None
    pair["ratio"] = pair["in"] / pair["out"]
    quiet = pair[pair["ratio"] <= pair["ratio"].quantile(0.25)]
    ratio = float(quiet["ratio"].median()) if not quiet.empty else float(
        pair["ratio"].median())
    implied_p = ratio * (lambda_h + k_h) / lambda_h

    out = {"metric": metric, "outdoor_source": src, "n": int(pair.shape[0]),
           "ratio_in_out": round(ratio, 3), "implied_penetration": round(implied_p, 3),
           "lambda_h": round(lambda_h, 3), "k_h": round(k_h, 3)}
    if implied_p <= 1.0:
        out["over_read"] = None
        out["note"] = (f"indoors sits at {ratio:.2f} of {src} {metric}; with "
                       f"lambda={lambda_h:.2f}/h and k={k_h:.2f}/h that implies "
                       f"{implied_p:.2f} of outdoor particles survive the way in, "
                       f"which is physically possible and needs no explaining away")
    else:
        out["over_read"] = round(implied_p, 2)
        out["min_true_outdoor"] = round(
            float(inside.tail(24).median()) * (lambda_h + k_h) / lambda_h, 1)
        out["note"] = (
            f"indoors sits at {ratio:.2f} of {src} {metric}, which with "
            f"lambda={lambda_h:.2f}/h and k={k_h:.2f}/h demands a penetration of "
            f"{implied_p:.2f}. Penetration cannot exceed 1, so the inputs disagree "
            f"by {(implied_p - 1) * 100:.0f}%. The likeliest single cause is this "
            f"counter reading high against the reference instrument - optical "
            f"sensors infer size from scattered light and humidity swells the "
            f"particles - but a residual indoor source during the quiet hours, a "
            f"dirtier street than the reference site, or an underestimated lambda "
            f"would each look identical. Treat {implied_p:.2f} as an upper bound on "
            f"the over-read, not a calibration factor to divide by.")
    return out


def penetration(db, location, k_h, lambda_h, days=14, now=None, metric="pm2_5"):
    """Fraction of outdoor particles that survive the journey indoors.

    At steady state with no indoor source, P_in/P_out = lambda*p/(lambda + k), so

        p = (P_in / P_out) * (lambda + k) / lambda

    Measured on the quietest indoor stretches available, because the derivation
    assumes the indoor source is zero and anything else biases p upwards.
    """
    import time as _time

    now = int(_time.time()) if now is None else int(now)
    start = now - days * 86400
    inside = features.series(db, metric, location, start, now).dropna()
    outside, _src = outdoor_series(db, metric, start, now)
    if inside.size < 24 or outside is None or outside.size < 24 or not lambda_h:
        return None

    pair = inside.to_frame("in").join(outside.to_frame("out"), how="inner").dropna()
    pair = pair[pair["out"] > 2.0]                 # a ratio to nothing is not a ratio
    if pair.shape[0] < 24:
        return None
    # The quietest quarter: lowest indoor level for the outdoor level at the time.
    pair["ratio"] = pair["in"] / pair["out"]
    quiet = pair[pair["ratio"] <= pair["ratio"].quantile(0.25)]
    if quiet.empty:
        return None
    ratio = float(quiet["ratio"].median())
    p = ratio * (lambda_h + k_h) / lambda_h
    return {"metric": metric, "ratio_in_out": round(ratio, 3),
            "penetration": round(min(p, 1.5), 3), "n": int(quiet.shape[0]),
            "capped": bool(p > 1.5),
            "note": (f"indoors sits at {ratio:.2f} of outdoor {metric} when nothing "
                     f"is happening inside; with lambda={lambda_h:.2f}/h and "
                     f"k={k_h:.2f}/h that implies {min(p, 1.5):.2f} of outdoor "
                     f"particles survive the way in"
                     + (". Capped at 1.5: a penetration above 1 means the model is "
                        "wrong, most likely an indoor source during the 'quiet' "
                        "hours." if p > 1.5 else "."))}


def learn_particles(db, location, days=14, now=None):
    """Fit everything the particle counter can say, and store what survives its test.

    Nothing is stored unless the size ordering passes, because a deposition rate
    that does not rise with size is not a deposition rate.
    """
    from analysis.room_model import RoomModel

    model = RoomModel(db, location)
    out = deposition(db, location, days=days, now=now, model=model)
    if not out["sizes"]:
        out["note"] = "no usable particle decays yet - needs an indoor source event"
        return out

    fine = out["sizes"].get("pm2_5")
    # Store unless the ordering test actively FAILED. `None` means it could not
    # discriminate - an all-fine aerosol gives the three masses nothing to differ
    # about - and refusing on that would throw away a good measurement for want of
    # a test that was never applicable. `False` is a real contradiction and still
    # blocks. The distinction is recorded with the value so nobody has to guess
    # which kind of silence it was.
    if out.get("ordered") is not False and fine and fine["k_h"]:
        db.set_param(location, "deposition_k_pm2_5", fine["k_h"], fine["stderr"],
                     fine["n_usable"], "1/h",
                     {"from": f"{fine['events']} particle decay(s) minus the CO2 "
                              f"rate over the same minutes",
                      "size_test": ("passed" if out.get("ordered")
                                    else "no power on this aerosol"),
                      "coarse_share": out.get("coarse_share"),
                      "ordering": out["ordering_note"]})
        # Penetration, measured against real instruments rather than a model.
        # Which outdoor source is used is not a detail: the same indoor data
        # implies p = 1.5 against Open-Meteo's grid and p = 0.79 against the AURN
        # stations, and only one of those is a number rather than a contradiction.
        lam = model.value["lambda_sealed"]
        chk = calibration_bound(db, location, fine["k_h"], lam, days=days, now=now)
        if chk:
            out["calibration"] = chk
            if chk["implied_penetration"] <= 1.0:
                db.set_param(location, "penetration_pm2_5",
                             chk["implied_penetration"], None, chk["n"], "fraction",
                             chk)
                out["penetration"] = chk["implied_penetration"]
    return out
