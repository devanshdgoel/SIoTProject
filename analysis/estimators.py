"""
estimators.py

Three independent ways to measure the same physical quantity: lambda, the number of
times per hour a room exchanges its air with somewhere else.

Lambda is the only thing that distinguishes one room from another in this model.
Everything the system predicts - the overnight ceiling, how long a purge takes, what
a window is worth - is a consequence of it. So it is worth measuring more than once,
by routes that fail in different ways.

  DECAY      Fit an exponential to each empty-room decay.
             Direct and easy to defend, but it only looks at the minutes when the
             room happened to be empty and falling steadily: a few percent of the
             record. One number per event, and events are rare.

  INTEGRAL   Integrate the mass balance over many windows and regress.
             Over any window at all,
                 change in CO2 = source x (occupied hours) - lambda x (area of excess)
             Both right-hand terms are counted straight off the data, so every window
             gives one equation in the same two unknowns and they are solved together.
             No derivatives, so sensor noise is not amplified; no need for the room to
             be empty, so occupied data counts too. Uses orders of magnitude more of
             the record than DECAY does.

  SPECTRUM   Read lambda off the shape of the power spectrum.
             A room is a first-order low-pass filter: slow changes pass through, fast
             ones are smoothed away, and the corner between the two sits at
             lambda / 2*pi. Fitting that corner uses the whole record at once and
             never looks at a single decay. It needs a long record - the corner for a
             sealed bedroom sits near a thirty-hour rhythm, so segments have to be
             longer than that before it can be seen at all.

They are not interchangeable. DECAY is the one to run continuously: it is cheap,
local in time, and tells you about one event. INTEGRAL is the one to trust for a
headline number.

SPECTRUM was described here as the independent check, on the grounds that it shares
no arithmetic with the other two. The arithmetic claim is true and the conclusion
drawn from it was not: a statistic can be independent and still be uninformative,
and this one is. The room's output spectrum is dominated by the shape of its
occupancy drive rather than by its own time constant, so the corner it fits moves
by 10% when the true rate moves by a factor of ten. It is retained as a TEST of
whether the room behaves as first order - the roll-off exponent is falsifiable and
is currently failing - and it no longer returns a rate unless it passes. Agreement
between it and the other two was a coincidence, not evidence.

DECAY and INTEGRAL are also less independent than they look: both subtract the same
c_out, so an error there moves them together.
"""

import numpy as np
from scipy import optimize, signal

from analysis import features

STEP_H = features.GRID_S / 3600.0


def _occupancy(db, location, labels, series, start, end):
    """Occupied / empty / unknown per grid sample, from measured evidence only.

    Labels the user typed and sleep windows the ring recorded. Anything else is left
    unknown and excluded rather than guessed, because a wrong occupancy flag feeds
    straight into the source term."""
    occ = np.full(len(series), np.nan)
    secs = series.index.view("int64") // 1_000_000_000
    for lb in labels:
        if lb["kind"] != "occupancy" or lb["value"] not in ("present", "absent"):
            continue
        inside = (secs >= lb["start_ts"]) & (secs <= lb["end_ts"])
        occ[inside] = 1.0 if lb["value"] == "present" else 0.0
    # A sleep window says the user was asleep somewhere, not that they were asleep
    # in this room. Marking the living room occupied all night made the regression
    # look for a source while the room was quietly emptying, and it returned a
    # negative one. So a sleep window only counts where the room's own CO2 actually
    # rose through it - the room has to corroborate the claim.
    values = series.to_numpy(float)
    for a, b, _day in features.night_windows(db, start, end):
        inside = (secs >= a) & (secs <= b)
        if inside.sum() < 12:
            continue
        seen = values[inside]
        if seen[-1] - seen[0] > 200.0:
            occ[inside] = 1.0
    return occ


def by_decay(db, location, days=14, now=None):
    """Exponential fits to empty-room decays - what the system already does."""
    rows = db.decays(location, min_r2=0.90)
    sealed = [r["lambda_h"] for r in rows if r["inferred"] == "sealed"]
    if not sealed:
        return {"method": "decay", "n": 0}
    med = float(np.median(sealed))
    mad = 1.4826 * float(np.median(np.abs(np.array(sealed) - med)))
    return {"method": "decay", "lambda_h": med, "n": len(sealed),
            "stderr": float(1.253 * mad / np.sqrt(len(sealed))) if len(sealed) > 1 else None,
            "samples_used": int(sum((r["end_ts"] - r["start_ts"]) / features.GRID_S
                                    for r in rows if r["inferred"] == "sealed")),
            "values": [round(v, 3) for v in sorted(sealed)]}


def by_integral(db, location, c_out, days=14, now=None, window_h=3.0, overlap=0.5):
    """Integral-form regression: lambda and the source together, from every window
    where occupancy is known.

    Integrating dC/dt = S*occ - lambda*(C - C_out) across a window removes the
    derivative entirely, which matters because differencing a noisy signal multiplies
    its noise by the sample rate. What remains is linear in both unknowns."""
    import time as _time

    now = int(_time.time()) if now is None else int(now)
    start = now - days * 86400
    raw = features.series(db, "co2_ppm", location, start, now)
    co2, _ = features.despike(raw)
    co2 = co2.dropna()
    if co2.size < 60:
        return {"method": "integral", "n": 0}
    labels = db.labels(location=location, start=start, end=now)
    occ = _occupancy(db, location, labels, co2, start, now)

    values = co2.to_numpy(float)
    width = max(6, int(window_h / STEP_H))
    step = max(1, int(width * (1 - overlap)))
    rows_dc, rows_a, rows_b = [], [], []
    for i in range(0, len(values) - width, step):
        occ_win = occ[i:i + width]
        if np.isnan(occ_win).any():
            continue                       # occupancy unknown: excluded, not assumed
        chunk = values[i:i + width]
        rows_dc.append(chunk[-1] - chunk[0])
        rows_a.append(float(np.sum(occ_win)) * STEP_H)
        rows_b.append(float(np.sum(chunk - c_out)) * STEP_H)
    if len(rows_dc) < 8:
        return {"method": "integral", "n": len(rows_dc),
                "note": "not enough windows with known occupancy"}

    design = np.column_stack([np.array(rows_a), -np.array(rows_b)])
    target = np.array(rows_dc)
    (source, lam), residuals, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ np.array([source, lam])
    ss_res = float(np.sum((target - predicted) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    dof = max(1, len(target) - 2)
    cov = ss_res / dof * np.linalg.pinv(design.T @ design)
    return {"method": "integral", "lambda_h": float(lam), "source_ppm_h": float(source),
            "stderr": float(np.sqrt(cov[1, 1])), "source_stderr": float(np.sqrt(cov[0, 0])),
            "n": len(target), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "samples_used": int(len(values))}


# A first-order room rolls off at -20 dB/decade in amplitude, which is -2 decades
# of power per decade on a log-log PSD. Fitting the exponent instead of assuming it
# is what turns this from an estimator into a test.
FIRST_ORDER_SLOPE = -2.0
SLOPE_TOL = 0.35                # how far from -2 still counts as first order


def by_spectrum(db, location, days=14, now=None, nperseg_h=48.0):
    """What the power spectrum says about this room - including that it cannot say much.

    The idea is sound in principle: a first-order room is a low-pass filter, slow
    rhythms pass and fast ones are smoothed away, and the corner sits at lambda/2pi.
    Fitting the corner uses the whole record at once and never looks at one event.

    In practice this room's spectrum cannot deliver lambda, for two reasons that
    took a controlled experiment to separate, and the honest thing is to report
    that rather than the plausible number the broken version produced.

    FIRST, THE DRIVE IS NOT BROADBAND. |H(f)|^2 = A/(1+(f/fc)^2) is the response to
    white forcing. The forcing here is one person asleep for eight hours a night:
    a near-periodic square wave whose spectrum is a line at 1/24 h and its
    harmonics, not a flat floor. Fitting a corner to the OUTPUT spectrum then
    measures the drive's own shape. Simulated, with everything else held fixed, a
    tenfold change in the true lambda moved the old estimate by 10%:

        true 0.20 -> 0.311     true 1.00 -> 0.328
        true 0.50 -> 0.314     true 2.00 -> 0.341

    It returned 0.48/h for the desk and 0.33/h for the bedroom, which look right,
    and that is exactly what made it dangerous.

    SECOND, THE WEIGHTING WAS WRONG. Welch PSD bins are chi-squared, so their error
    is MULTIPLICATIVE - the same relative spread at every bin. Ordinary least
    squares on raw PSD values spanning seven decades is therefore a fit to the three
    lowest-frequency bins and nothing else; the fitted noise floor collapsed to
    1e-15 and the whole high-frequency half contributed nothing. Taking logs makes
    the error additive and homoscedastic, which is the condition OLS needs.

    So: fit log PSD, free the exponent, and report the slope as the test. The slope
    is the falsifiable part - first order predicts -2.0 decades of power per decade.

    Measured here: -2.65 (bedroom) and -2.83 (desk). Both reject first order, and
    the reason is NOT settled. A control run alongside this - a simulated
    first-order room, same sampling, same noise - gives -1.64 to -1.75 under white
    forcing and -1.82 to -2.19 under a daily sleep square wave. So drive shape does
    steepen the slope, by about 0.4 decades, and that is a real part of the answer;
    but it does not reach -2.65, so it is not the whole answer.

    The leading candidate for the rest is that the room is not first order. A
    bedroom coupled to a corridor which is coupled to outdoors is a TWO-compartment
    system, and second order rolls off at -4; a mixture sits between -2 and -4,
    which is where both rooms are. That is not idle speculation here - the decay
    asymptotes give independent evidence for exactly that coupling, which is why
    "coupled" is now a class of its own in room_model. Sensor response and the
    despike filter also contribute some high-frequency attenuation and have not
    been separated out.

    What can be said with confidence is the negative: the corner of this spectrum
    is not lambda, and the agreement the old version showed with the decay estimate
    was a coincidence. So `lambda_h` is returned only when the slope is consistent
    with first order, and otherwise the caller is told why not.
    """
    import time as _time

    now = int(_time.time()) if now is None else int(now)
    start = now - days * 86400
    co2, _ = features.despike(features.series(db, "co2_ppm", location, start, now))
    co2 = co2.interpolate(limit_area="inside").dropna()
    if co2.size < 288:
        return {"method": "spectrum", "n": 0, "note": "need at least a day of samples"}

    x = co2.to_numpy(float)
    x = x - x.mean()
    fs = 1.0 / STEP_H                                   # samples per hour
    nperseg = min(len(x), int(nperseg_h / STEP_H))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                              window="hann", detrend="linear")
    keep = (freqs > 0) & (psd > 0)
    freqs, psd = freqs[keep], psd[keep]
    if freqs.size < 8:
        return {"method": "spectrum", "n": int(freqs.size),
                "note": "too few spectral bins to fit"}

    # Welch averages K half-overlapping segments; K sets the error on every bin and
    # therefore everything that can be concluded, so it is reported rather than
    # buried. K = 5 on a 6-day record means 45% relative spread per bin.
    segs = max(1, int(2 * len(x) / nperseg) - 1)

    def log_model(f, log_amp, fc, n_order):
        return log_amp - np.log10(1.0 + (f / fc) ** (2.0 * n_order))

    y = np.log10(psd)
    try:
        popt, pcov = optimize.curve_fit(
            log_model, freqs, y,
            p0=[float(y[0]), 0.05, 1.0],
            bounds=([-30.0, 1e-4, 0.3], [30.0, 5.0, 4.0]), maxfev=40000)
    except (RuntimeError, ValueError):
        return {"method": "spectrum", "n": len(freqs), "note": "fit did not converge"}
    log_amp, fc, n_order = (float(v) for v in popt)
    try:
        n_err = float(np.sqrt(pcov[2, 2]))
        fc_err = float(np.sqrt(pcov[1, 1]))
    except (ValueError, IndexError):
        n_err = fc_err = float("nan")

    slope = -2.0 * n_order                      # decades of power per decade
    resid = y - log_model(freqs, *popt)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    first_order = abs(slope - FIRST_ORDER_SLOPE) <= SLOPE_TOL

    out = {"method": "spectrum", "corner_hz_per_h": fc,
           "corner_period_h": 1.0 / fc if fc else None,
           "order": n_order, "order_stderr": n_err,
           "slope_dec_per_dec": round(slope, 2),
           "first_order": bool(first_order),
           "segments": segs, "welch_K": segs,
           "bin_rel_sd": round(float(1.0 / np.sqrt(max(segs, 1))), 2),
           "n": len(freqs), "samples_used": int(len(x)),
           "r2_log": 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0,
           "spectrum": {"freqs": freqs.tolist(), "psd": psd.tolist(),
                        "fit": (10.0 ** log_model(freqs, *popt)).tolist()}}

    if first_order:
        out["lambda_h"] = float(2 * np.pi * fc)
        out["stderr"] = float(2 * np.pi * fc_err)
        out["note"] = (f"roll-off {slope:.2f} decades/decade is consistent with "
                       f"first order, so the corner is a rate")
    else:
        out["lambda_h"] = None
        out["note"] = (
            f"roll-off is {slope:.2f} decades/decade, not the {FIRST_ORDER_SLOPE:.1f} "
            f"a first-order room gives, so the corner is NOT lambda and no rate is "
            f"reported from it. A simulated first-order room driven by a daily sleep "
            f"cycle reaches about -2.1, so the occupancy drive explains part of the "
            f"excess but not all of it; a room coupled to the rest of the flat is "
            f"second order, which would explain the remainder and which the decay "
            f"asymptotes independently support.")
    return out


def compare(db, location, days=14, now=None):
    """All three, side by side, on the same record."""
    from analysis.room_model import RoomModel

    c_out = RoomModel(db, location).value["c_out"]
    return {"location": location, "c_out": c_out,
            "decay": by_decay(db, location, days, now),
            "integral": by_integral(db, location, c_out, days, now),
            "spectrum": by_spectrum(db, location, days, now)}
