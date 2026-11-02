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


def outdoor_baseline(co2, q=0.05, lo=380.0, hi=650.0):
    """Outdoor CO2 as this particular sensor sees it: the low tail of its own record.

    An empty or open room tends towards outdoor air, so the 5th percentile is that
    value. Using the sensor's own tail rather than a textbook 420 ppm also absorbs
    its calibration offset, which is what makes two sensors comparable."""
    v = co2.dropna()
    if v.empty:
        return None
    return float(min(max(float(v.quantile(q)), lo), hi))


def _runs(mask):
    """Contiguous True stretches of a boolean Series as (start_index, end_index)."""
    out, start, prev = [], None, None
    for ts, ok in mask.items():
        if ok and start is None:
            start = ts
        elif not ok and start is not None:
            out.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        out.append((start, prev))
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
    for a, b in _runs(falling):
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


def best_decay_fit(seg, c_out, offsets=(0.0, 0.15, 0.30, 0.45)):
    """Fit a decay, allowing its start to be trimmed, and keep the best-conditioned one.

    A detected window often begins before the room really started emptying: the flat
    tail of a plateau, or the minutes before someone opened the window. A single
    exponential drawn through such a regime change fits neither part of it. Trying a
    few start offsets and keeping the highest R-squared recovers the decay that is
    actually in the data, and reports the segment it was measured over."""
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


def label_at(labels, ts, kind):
    """The user's stated state of `kind` covering a timestamp, or None."""
    for lb in labels:
        if lb["kind"] == kind and lb["start_ts"] <= ts <= lb["end_ts"]:
            return lb["value"]
    return None
