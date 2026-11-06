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
headline number. SPECTRUM is the independent check - it shares no arithmetic with
the other two, so agreement between them is real evidence rather than a tautology.
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


def by_spectrum(db, location, days=14, now=None, nperseg_h=48.0):
    """Lambda from the corner of the power spectrum.

    Fits |H(f)|^2 = A / (1 + (f/fc)^2), the response of a first-order system, and
    returns lambda = 2*pi*fc. Shares no arithmetic with the other two methods: it
    never looks at an individual event, only at which rhythms the room let through."""
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
    keep = freqs > 0
    freqs, psd = freqs[keep], psd[keep]

    def model(f, amp, fc, floor):
        return amp / (1.0 + (f / fc) ** 2) + floor

    try:
        popt, pcov = optimize.curve_fit(
            model, freqs, psd, p0=[psd[0], 0.05, float(np.median(psd[-10:]))],
            bounds=([0, 1e-4, 0], [np.inf, 5.0, np.inf]), maxfev=20000)
    except (RuntimeError, ValueError):
        return {"method": "spectrum", "n": len(freqs), "note": "fit did not converge"}
    amp, fc, floor = popt
    resid = psd - model(freqs, *popt)
    ss_tot = float(np.sum((psd - psd.mean()) ** 2))
    return {"method": "spectrum", "lambda_h": float(2 * np.pi * fc),
            "stderr": float(2 * np.pi * np.sqrt(pcov[1, 1])),
            "corner_hz_per_h": float(fc), "corner_period_h": float(1 / fc),
            "segments": int(len(x) / (nperseg / 2)) - 1, "n": len(freqs),
            "r2": 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0,
            "samples_used": int(len(x)),
            "spectrum": {"freqs": freqs.tolist(), "psd": psd.tolist(),
                         "fit": model(freqs, *popt).tolist()}}


def compare(db, location, days=14, now=None):
    """All three, side by side, on the same record."""
    from analysis.room_model import RoomModel

    c_out = RoomModel(db, location).value["c_out"]
    return {"location": location, "c_out": c_out,
            "decay": by_decay(db, location, days, now),
            "integral": by_integral(db, location, c_out, days, now),
            "spectrum": by_spectrum(db, location, days, now)}
