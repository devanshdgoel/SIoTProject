"""
features.py

Shared feature layer: everything above this file works on clean, regularly sampled
series instead of raw rows.

Three jobs.
  1. Put four sources sampled at 1 min, 7 min, 30 min and 60 min onto one 5-minute
     grid. 5 minutes is well inside Nyquist for room CO2 dynamics, whose time
     constant (1/lambda) runs from about 20 minutes to several hours.
  2. Remove spikes without smearing them (Hampel), and refuse to invent data across
     gaps where the sensor was simply absent.
  3. Find the stretches of record that carry physical information: decays, from which
     the air-change rate follows, and plateaus, from which the CO2 source term does.
"""

import datetime as dt

import numpy as np
import pandas as pd
from scipy import optimize, stats

GRID_S = 300

# How each metric reaches the common grid.
#   "agg"  collapses the samples inside one bin
#   "fill" bridges bins the source never sampled at all (the hourly feeds)
# mean is not cosmetic: averaging the 1-minute sensors is the anti-alias filter.
# Taking one sample in five would fold sensor noise down into the band of interest.
# max exists because peak exposure must not be averaged away.
# interp suits a continuous physical field, which is near-linear across an hour;
# an interval total such as rainfall cannot be interpolated and is held instead.
RESAMPLE = {
    "co2_ppm": ("mean", None),
    "co2_max_ppm": ("max", None),
    "indoor_temp_c": ("mean", None),
    "indoor_rh_pct": ("mean", None),
    "outdoor_temp_c": ("mean", "interp"),
    "outdoor_rh_pct": ("mean", "interp"),
    "outdoor_dewpoint_c": ("mean", "interp"),
    "pressure_hpa": ("mean", "interp"),
    "wind_ms": ("mean", "interp"),
    "pm2_5": ("mean", "interp"),
    "pm10": ("mean", "interp"),
    "no2": ("mean", "interp"),
    "o3": ("mean", "interp"),
    "european_aqi": ("mean", "interp"),
    "precip_mm": ("sum", "ffill"),
    "hr_bpm": ("mean", "interp"),
    "hrv_ms": ("mean", "interp"),
    "steps": ("sum", None),
}
# Longest gap each metric may be bridged across. The ring comes off the finger, and
# interpolating through that would manufacture a heart rate that was never measured.
MAX_GAP_S = {"hr_bpm": 1800, "hrv_ms": 1800, "skin_temp_c": 1800}
DEFAULT_GAP_S = 3 * 3600


def series(db, metric, location=None, start=None, end=None, step=GRID_S, kind=None):
    """One metric as a regularly sampled pandas Series (UTC index), suspect rows dropped."""
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    df = db.read_df(metric, location, start, end, kind)
    if df.empty:
        return empty
    good = df[df["quality"] != "suspect"]
    if good.empty:
        return empty
    agg, fill = RESAMPLE.get(metric, ("mean", None))
    out = getattr(good["value"].astype(float).resample(f"{step}s"), agg)()
    if fill == "interp":
        limit = max(1, MAX_GAP_S.get(metric, DEFAULT_GAP_S) // step)
        out = out.interpolate(limit=limit, limit_area="inside")
    elif fill == "ffill":
        out = out.fillna(0.0)
    return out


def despike(s, k=4.0):
    """Remove single-sample excursions; returns (cleaned, n_replaced).

    A spike is a reading that disagrees with both of its neighbours while they agree
    with each other, so the test is made against the straight line through them. The
    scale is the median absolute residual, which is robust: one spike cannot inflate
    the threshold that is meant to catch it.

    This is used in preference to a plain Hampel filter because Hampel assumes the
    signal is locally constant. A ventilated room empties with a time constant of
    around twenty minutes, and across a median window that steep exponential looks
    exactly like a run of outliers - so Hampel deletes the very events the air-change
    model is trying to measure. A point lying between its two neighbours can never
    trip the test below, however fast the room is changing.

    A low-pass filter is not an alternative: a spike is broadband, so filtering
    smears it across its neighbours rather than removing it."""
    if s.dropna().size < 5:
        return s, 0
    prev, nxt = s.shift(1), s.shift(-1)
    straight = (prev + nxt) / 2.0
    resid = s - straight
    scale = 1.4826 * float(resid.abs().median(skipna=True) or 0.0)
    span = (nxt - prev).abs()
    bad = ((resid.abs() > max(k * scale, 1e-9))
           & ((s - prev).abs() > span) & ((s - nxt).abs() > span)).fillna(False)
    return s.mask(bad, straight), int(bad.sum())


def outdoor_baseline(co2, q=0.05, lo=250.0, hi=700.0):
    """Outdoor CO2 as this particular sensor sees it: the low tail of its own record.

    An empty or open room tends towards outdoor air, so the 5th percentile is that
    value. Using the sensor's own tail rather than a textbook 420 ppm also absorbs
    its calibration offset, which is what makes two sensors comparable."""
    v = co2.dropna()
    if v.empty:
        return None
    return float(min(max(float(v.quantile(q)), lo), hi))


def _runs(mask, bridge=0):
    """True stretches of a boolean Series as (start_index, end_index).

    `bridge` absorbs short interruptions. A genuine decay is occasionally broken by
    one sample that fails the test - a door held open for a moment, or noise where
    the curve has gone flat - and splitting there turns a single long measurement
    into several poor short ones. The twelve-hour decay of an empty room is worth
    far more than the six fragments it would otherwise become."""
    out, start, last_true, misses = [], None, None, 0
    for ts, ok in mask.items():
        if ok:
            if start is None:
                start = ts
            last_true, misses = ts, 0
        elif start is not None:
            misses += 1
            if misses > bridge:
                out.append((start, last_true))
                start, misses = None, 0
    if start is not None:
        out.append((start, last_true))
    return out


def decay_windows(co2, c_out, min_minutes=20, min_drop=100.0, min_excess=150.0,
                  min_rate_h=0.15, step=GRID_S):
    """Stretches where CO2 falls steadily, i.e. the room emptied.

    With no occupants the source term is zero and the mass balance collapses to a
    pure exponential, which is the only condition under which lambda can be read off
    directly. The thresholds keep the logarithm well conditioned: a shallow decay
    from a low starting point fits noise, not physics."""
    if co2.dropna().size < 4:
        return []
    smooth = co2.rolling(3, center=True, min_periods=1).median()
    # "Falling" has to mean falling at a rate a decay would produce, not merely not
    # rising. A room sitting at its occupied steady state drifts downward by a ppm or
    # two, and admitting that flat tail into the window wrecks the exponential fit.
    ceiling = -min_rate_h * (smooth - c_out).clip(lower=1.0) * step / 3600.0
    falling = (smooth.diff() <= ceiling).fillna(False)
    keep = []
    for a, b in _runs(falling, bridge=2):      # ride out up to ten minutes of wobble
        seg = co2.loc[a:b].dropna()
        if seg.size < 4 or seg.size * step < min_minutes * 60:
            continue
        if seg.iloc[0] - seg.iloc[-1] < min_drop or seg.iloc[0] - c_out < min_excess:
            continue
        keep.append((a, b))
    return keep


def plateau_windows(co2, c_out, min_minutes=30, max_slope_ppm_h=40.0,
                    min_excess=150.0, step=GRID_S):
    """Stretches where an occupied room has levelled off.

    At steady state production balances losses, so source = lambda * excess. That is
    a far more robust way to get the source term than fitting the rising transient."""
    if co2.dropna().size < 4:
        return []
    smooth = co2.rolling(3, center=True, min_periods=1).median()
    slope = smooth.diff().abs() * 3600.0 / step
    flat = slope.rolling(5, center=True, min_periods=2).mean() <= max_slope_ppm_h
    flat = (flat & ((co2 - c_out) >= min_excess)).fillna(False)
    keep = []
    for a, b in _runs(flat):
        seg = co2.loc[a:b].dropna()
        if seg.size >= 4 and seg.size * step >= min_minutes * 60:
            keep.append((a, b))
    return keep


def fit_decay(seg, c_out):
    """Air-change rate from one decay, in air changes per hour, or None if unusable.

    ln(C - C_out) is linear in time with slope -lambda, so a straight line gives it.
    The log transform makes the residuals heteroscedastic and over-weights the small
    values in the tail, so the linear answer seeds a nonlinear fit on the exponential
    itself; reporting both shows how large that bias actually is."""
    seg = seg.dropna()
    if seg.size < 4:
        return None
    hours = (seg.index - seg.index[0]).total_seconds().to_numpy() / 3600.0
    excess = seg.to_numpy(dtype=float) - c_out
    if float(np.min(excess)) <= 1.0:              # the log needs headroom above baseline
        return None
    fit = stats.linregress(hours, np.log(excess))
    lam = -float(fit.slope)
    if not np.isfinite(lam) or lam <= 0:
        return None
    out = {"lambda_h": lam, "stderr": float(fit.stderr), "r2": float(fit.rvalue) ** 2}

    def model(t, amp, rate):
        return amp * np.exp(-rate * t)

    try:
        popt, pcov = optimize.curve_fit(model, hours, excess, p0=[excess[0], lam], maxfev=5000)
        if popt[1] > 0 and np.all(np.isfinite(pcov)):
            out["lambda_nls"] = float(popt[1])
            out["stderr_nls"] = float(np.sqrt(pcov[1, 1]))
    except (RuntimeError, ValueError, TypeError):
        pass                                      # the log-linear answer still stands
    return out


def best_decay_fit(seg, c_out, offsets=(0.0, 0.15, 0.30, 0.45), min_excess=15.0):
    """Fit a decay, allowing its start to be trimmed, and keep the best-conditioned one.

    A detected window often begins before the room really started emptying: the flat
    tail of a plateau, or the minutes before someone opened the window. A single
    exponential drawn through such a regime change fits neither part of it. Trying a
    few start offsets and keeping the highest R-squared recovers the decay that is
    actually in the data, and reports the segment it was measured over."""
    # Trim the tail first. The logarithm needs headroom above the baseline, so once a
    # decay has arrived within a few ppm of it the remaining samples carry no rate
    # information and one of them going slightly negative discards the whole event.
    # That is how the first deliberate window-open test was nearly lost: the room
    # settled below the baseline the model was still assuming.
    thin = (seg - c_out) <= min_excess
    if bool(thin.any()):
        cut = seg.index.get_loc(thin.idxmax())
        seg = seg.iloc[:cut]
    if seg.size < 6:
        return None

    best = None
    for frac in offsets:
        part = seg.iloc[int(len(seg) * frac):]
        if part.size < 6:
            continue
        fit = fit_decay(part, c_out)
        if fit is None:
            continue
        fit["start"], fit["end"] = part.index[0], part.index[-1]
        if best is None or fit["r2"] > best["r2"]:
            best = fit
    return best


def fit_asymptote(seg, c_out_guess, min_drop=50.0, settle=0.25):
    """Fit C(t) = C_inf + A exp(-rate t) with the asymptote left free.

    A decaying room settles towards whatever air is flushing it, so C_inf is outdoor
    CO2 as this sensor reads it. Measuring it this way is better than taking a low
    percentile of the record, which is biased upward in a room that is rarely left
    empty long enough to actually get there."""
    seg = seg.dropna()
    if seg.size < 8:
        return None
    hours = (seg.index - seg.index[0]).total_seconds().to_numpy() / 3600.0
    y = seg.to_numpy(dtype=float)
    if float(y[0] - y[-1]) < min_drop:
        return None

    def model(t, c_inf, amp, rate):
        return c_inf + amp * np.exp(-rate * t)

    try:
        popt, _ = optimize.curve_fit(model, hours, y,
                                     p0=[c_out_guess, max(y[0] - c_out_guess, 1.0), 1.0],
                                     bounds=([300.0, 1.0, 0.01], [800.0, 20000.0, 20.0]),
                                     maxfev=10000)
    except (RuntimeError, ValueError, TypeError):
        return None
    c_inf = float(popt[0])
    if not 300.0 < c_inf < 800.0:            # the solver pinned a bound: not a measurement
        return None
    # An asymptote the room never approached cannot be measured. Insist the decay ran
    # far enough that what remains is a small fraction of where it started.
    span = y[0] - c_inf
    if span <= 0 or (y[-1] - c_inf) / span > settle:
        return None
    total = float(np.sum((y - y.mean()) ** 2))
    resid = float(np.sum((y - model(hours, *popt)) ** 2))
    return {"c_inf": c_inf, "lambda_h": float(popt[2]),
            "r2": 1.0 - resid / total if total > 0 else 0.0}


def fit_rise(seg, c_out):
    """Fit an occupied room filling up, with the ceiling left free.

    C(t) = C_inf + (C0 - C_inf)*exp(-lambda*t), so one curve yields both the air-change
    rate and the level the room is heading for; the source follows as
    lambda*(C_inf - C_out).

    This is strictly better than waiting for a plateau. A bedroom with a time constant
    of three or four hours and a sleeper in it for eight never actually reaches steady
    state - it is still climbing at dawn - so the plateau method measures nothing,
    while the rise is the richest curve in the whole record."""
    seg = seg.dropna()
    if seg.size < 24:
        return None
    hours = (seg.index - seg.index[0]).total_seconds().to_numpy() / 3600.0
    y = seg.to_numpy(dtype=float)
    if y[-1] - y[0] < 200.0:
        return None

    def model(t, c_inf, c0, rate):
        return c_inf + (c0 - c_inf) * np.exp(-rate * t)

    try:
        popt, pcov = optimize.curve_fit(
            model, hours, y, p0=[max(y[-1] * 1.3, y[-1] + 200), y[0], 0.3],
            # The starting value is bounded by what was actually seen, not by c_out.
            # Tying it to the baseline meant that whenever the baseline was still
            # overestimated - which is exactly when a room has not been aired - the
            # true start fell outside the bounds and the fit could not run at all.
            bounds=([y[-1], float(np.min(y)) - 100.0, 0.02],
                    [20000.0, y[0] + 200.0, 10.0]), maxfev=20000)
    except (RuntimeError, ValueError, TypeError):
        return None
    c_inf, _, rate = popt
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - model(hours, *popt)) ** 2)) / total if total > 0 else 0.0
    source = rate * (c_inf - c_out)
    if source <= 0 or not np.isfinite(source):
        return None
    return {"lambda_h": float(rate), "stderr": float(np.sqrt(pcov[2][2])),
            "ceiling": float(c_inf), "source_ppm_h": float(source), "r2": r2}


def occupied_windows(db, location, labels, start, end):
    """Periods the room is believed occupied, as (start_ts, end_ts).

    Two sources of evidence, both measured rather than assumed: spans the user
    labelled `occupancy=present`, and the sleep windows the ring recorded. A sleep
    window is only a candidate - the room still has to show a real rise before it is
    used - so a night spent elsewhere simply fails the test instead of inventing one."""
    spans = [(lb["start_ts"], lb["end_ts"]) for lb in labels
             if lb["kind"] == "occupancy" and lb["value"] == "present"]
    spans += [(a, b) for a, b, _day in night_windows(db, start, end)]
    return sorted(set(spans))


def occupancy_prior(db, location, start, end, step_s=GRID_S):
    """How often this room is occupied, by hour of the local day, and for how long.

    Returns {"prior": 24 probabilities, "persistence_min": how long a belief about
    the present should be trusted, ...}. Evidence is measured, never assumed: spans the user
    labelled, sleep windows the ring recorded and the room corroborated with a real
    rise, and otherwise the room's own testimony - CO2 climbing faster than leakage
    alone can explain means somebody is producing it.

    This exists so a forecast can stop assuming that whoever is in the room now will
    still be there in two hours. People leave; the long-run pattern knows roughly
    when."""
    from zoneinfo import ZoneInfo

    from config import settings

    co2 = series(db, "co2_ppm", location, start, end, step_s).dropna()
    if co2.size < 48:
        return None
    labels = db.labels(location=location, start=start, end=end)
    secs = co2.index.view("int64") // 1_000_000_000
    values = co2.to_numpy(float)
    occ = np.full(len(values), np.nan)

    # weakest evidence first, so stronger evidence overwrites it
    rise = np.full(len(values), np.nan)
    rise[1:] = np.diff(values) * 3600.0 / step_s
    occ[rise > 60.0] = 1.0
    occ[rise < -30.0] = 0.0
    # A room resting at its own outdoor level is empty, whatever its slope is doing.
    # Somebody in it would be pushing it up. Without this, every long flat stretch
    # after a room has finished clearing counts as no evidence at all, and the hours
    # a room spends genuinely empty fall back to the average instead of to zero.
    floor = float(np.nanpercentile(values, 5))
    occ[(values < floor + 120.0) & ~(rise > 60.0)] = 0.0

    for a, b, _day in night_windows(db, start, end):
        inside = (secs >= a) & (secs <= b)
        if inside.sum() >= 12 and values[inside][-1] - values[inside][0] > 200.0:
            occ[inside] = 1.0
    for lb in labels:
        if lb["kind"] != "occupancy" or lb["value"] not in ("present", "absent"):
            continue
        occ[(secs >= lb["start_ts"]) & (secs <= lb["end_ts"])] = (
            1.0 if lb["value"] == "present" else 0.0)

    zone = ZoneInfo(settings.USER_TZ)
    hours = np.array([dt.datetime.fromtimestamp(int(t), zone).hour for t in secs])
    prior = []
    for hour in range(24):
        seen = occ[(hours == hour) & ~np.isnan(occ)]
        prior.append(float(seen.mean()) if seen.size >= 3 else None)
    known = [p for p in prior if p is not None]
    fallback = float(np.mean(known)) if known else 0.5
    prior = [fallback if p is None else p for p in prior]

    # How long does somebody stay? Measured, not chosen. Occupied runs are found in
    # the same evidence, and for a spell of known length the expected time still to
    # run is half of it - which is the quantity a forecast actually needs.
    # Bridge short interruptions first. Occupancy evidence flickers - CO2 does not
    # climb on every single step even with somebody sitting there - and counting raw
    # runs turns one night's sleep into dozens of ten-minute spells.
    bridge = max(1, 1800 // step_s)
    runs, current, missed = [], 0, 0
    for flag in occ:
        if flag == 1.0:
            current += 1 + missed
            missed = 0
        elif current:
            missed += 1
            if missed > bridge:
                runs.append(current); current, missed = 0, 0
    if current:
        runs.append(current)
    spell_min = (float(np.median(runs)) * step_s / 60.0) if runs else 120.0
    persistence = float(min(360.0, max(20.0, spell_min / 2.0)))
    return {"prior": prior, "persistence_min": persistence,
            "spell_min": round(spell_min), "runs": len(runs)}


SETTLE_PPM = 25.0             # residual excess a sensor cannot tell from its own noise


def settled_baseline(seg, c_out_guess):
    """Outdoor CO2 read off an aired room - but only once the airing has finished.

    A room approaches outdoor air exponentially: after a time t at rate lambda the
    excess that remains is exp(-lambda*t) of the excess it started with. The sensor
    reads outdoor air only once that remainder has fallen below what it can resolve.
    A short opening, or one in still air where lambda is small, stops well short of
    that, and its lowest reading is then an UPPER BOUND on outdoor CO2 rather than a
    measurement of it. Treating the bound as a measurement biases the baseline up,
    which shrinks every excess the model computes and so inflates every air-change
    rate fitted against it - one unfinished flush is enough to corrupt the room.

    So the segment has to earn it: either it is already flat, or its own fitted
    asymptote has to sit within sensor noise of where it actually ended up.

    Returns the settled level in ppm, or None if the trace was still falling.
    """
    seg = seg.dropna()
    if seg.size < 6:
        return None
    if float(seg.max() - seg.min()) <= SETTLE_PPM:
        return float(seg.median())          # flat throughout: already equilibrated
    fit = fit_asymptote(seg, c_out_guess)
    if fit is None or fit["r2"] < 0.9:
        return None                         # no usable curve, so no usable asymptote
    # Judge the settling on the median of the tail, not on the final sample. The
    # label's end is a rounded, self-reported time and the decimation window there
    # straddles it, so the last point picks up whatever the room did next - being
    # reoccupied, usually, which pushes it back up. That one point is the most
    # contaminated in the segment and the worst possible thing to test.
    tail = max(3, seg.size // 4)
    level = float(seg.tail(tail).median())
    if level - fit["c_inf"] > SETTLE_PPM:
        return None                         # still descending towards its asymptote
    return level


DEPARTURE_OFFSET_MIN = 20     # let the room settle after the door closes
DEPARTURE_WINDOW_MIN = 60     # then measure over a fixed span, every time


def departures(co2, rise_ppm_h=60.0, fall_ppm_h=-40.0, min_gap_s=5400):
    """Moments a room stopped filling and began clearing - i.e. the last person left.

    Anchoring on this rather than on a threshold crossing gives every measurement the
    same geometry: the same delay after the event, over the same span. Threshold-found
    windows vary in length and in how far into the decay they start, so their results
    are not strictly comparable with each other. These are."""
    if co2.dropna().size < 24:
        return []
    smooth = co2.rolling(5, center=True, min_periods=2).median()
    slope = smooth.diff() * 3600.0 / GRID_S
    rising = (slope > rise_ppm_h).to_numpy()
    falling = (slope < fall_ppm_h).to_numpy()
    out = []
    for i in range(6, len(co2) - 6):
        if rising[i - 6:i].sum() >= 4 and falling[i:i + 6].sum() >= 4:
            when = co2.index[i]
            if not out or (when - out[-1]).total_seconds() > min_gap_s:
                out.append(when)
    return out


def departure_decay(co2, c_out, when, offset_min=DEPARTURE_OFFSET_MIN,
                    window_min=DEPARTURE_WINDOW_MIN):
    """Air-change rate measured over a fixed span after a departure, or None.

    The offset exists because the minute somebody walks out the room is still mixing;
    starting immediately measures the mixing as well as the leak."""
    start = when + pd.Timedelta(minutes=offset_min)
    seg = co2.loc[start:start + pd.Timedelta(minutes=window_min)].dropna()
    if seg.size < 8:
        return None
    fit = fit_decay(seg, c_out)
    if fit is None:
        return None
    fit["start"], fit["end"] = seg.index[0], seg.index[-1]
    return fit


def at(s, ts, tol_s=5400):
    """Value of a series nearest a timestamp, or None if nothing is close enough."""
    if s is None:
        return None
    clean = s.dropna()
    if clean.empty:
        return None
    want = pd.Timestamp(int(ts), unit="s", tz="UTC")
    idx = clean.index.get_indexer([want], method="nearest")[0]
    if abs((clean.index[idx] - want).total_seconds()) > tol_s:
        return None
    return float(clean.iloc[idx])


def night_windows(db, start=None, end=None):
    """Real sleep periods from the ring as (start_ts, end_ts, day).

    Using measured sleep rather than an assumed bedtime is what makes overnight
    exposure a personal number instead of a guess."""
    q = ("SELECT day, bedtime_start, bedtime_end FROM sleep_sessions "
         "WHERE bedtime_start IS NOT NULL AND bedtime_end IS NOT NULL")
    args = []
    if start is not None:
        q, args = q + " AND bedtime_end>=?", args + [int(start)]
    if end is not None:
        q, args = q + " AND bedtime_start<=?", args + [int(end)]
    with db.conn() as c:
        rows = c.execute(q + " ORDER BY day", args).fetchall()
    return [(int(r["bedtime_start"]), int(r["bedtime_end"]), r["day"])
            for r in rows if r["bedtime_end"] > r["bedtime_start"]]


def people_at(labels, ts, default=1):
    """How many people the user says were in the room, or `default` if unstated.

    One is the right default: it is what an unannotated occupancy label has always
    meant, and it keeps a single-occupant room - which is where the measurements are
    cleanest - behaving exactly as before."""
    for lb in labels:
        if lb["kind"] == "occupancy" and lb["value"] == "present" and                 lb["start_ts"] <= ts <= lb["end_ts"]:
            try:
                return max(1, int(lb["people"]))
            except (KeyError, TypeError, ValueError, IndexError):
                return default
    return default


def label_at(labels, ts, kind):
    """The user's stated state of `kind` covering a timestamp, or None."""
    for lb in labels:
        if lb["kind"] == kind and lb["start_ts"] <= ts <= lb["end_ts"]:
            return lb["value"]
    return None


LABEL_SLACK_S = 15 * 60       # people report times to the nearest quarter hour
LABEL_COVER = 0.5             # the label must account for most of the decay


def label_over(labels, start_ts, end_ts, kind, slack=LABEL_SLACK_S):
    """The user's stated state of `kind` spanning a decay, matched by overlap.

    `label_at` asks whether one instant sits inside a label, which is too brittle
    for a fitted decay: the fit's own edges come from the data, the label's come
    from somebody remembering what time they opened a window, and those are rounded
    to the nearest five or fifteen minutes. A fast flush is over in half an hour, so
    a quarter-hour of rounding is enough to push its midpoint outside the label and
    throw the measurement away - the cleanest events, the fastest ones, are exactly
    the ones lost that way.

    So the label is widened by `slack` at both ends and has to cover at least half
    the decay. Where several labels of the kind qualify, the longest overlap wins.
    """
    best, best_overlap = None, 0.0
    span = max(float(end_ts - start_ts), 1.0)
    for lb in labels:
        if lb["kind"] != kind:
            continue
        lo, hi = lb["start_ts"] - slack, lb["end_ts"] + slack
        overlap = min(end_ts, hi) - max(start_ts, lo)
        if overlap / span >= LABEL_COVER and overlap > best_overlap:
            best, best_overlap = lb["value"], overlap
    return best
