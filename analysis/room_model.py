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
    if refined:
        c_out = float(np.median(tails))
        method = f"asymptote of {len(tails)} deep decays"
    # If the percentile had to be clamped and no decay settled, the room was never
    # observed near outdoor air. The clamp is then a guess wearing a measurement's
    # clothes, so it is stored with n=0 and the published default keeps being used.
    # One usable asymptote is not a refinement: the branch above needs two before it
    # replaces the percentile, so a single tail must not make the clamp look trusted.
    clamped = not 380.0 <= float(co2.dropna().quantile(0.05)) <= 650.0
    trusted = 0 if (clamped and not refined) else int(co2.dropna().size)
    db.set_param(location, "c_out", c_out, None, trusted, "ppm",
                 {"method": method, "clamped": clamped})
    summary["c_out"] = round(c_out, 1)
    summary["c_out_trusted"] = bool(trusted)

    temp_out = features.series(db, "outdoor_temp_c", "home", start, now)
    wind = features.series(db, "wind_ms", "home", start, now)
    temp_in = features.series(db, "indoor_temp_c", location, start, now)
    labels = db.labels(location=location, start=start, end=now)

    rows, lams, marks = [], [], []
    for a, b in features.decay_windows(co2, c_out):
        fit = features.best_decay_fit(co2.loc[a:b], c_out)
        if fit is None or fit["r2"] < MIN_R2:
            continue
        a, b = fit["start"], fit["end"]          # the trimmed span the fit actually used
        mid = int((a.timestamp() + b.timestamp()) / 2)
        t_in, t_out = features.at(temp_in, mid), features.at(temp_out, mid)
        rows.append({"location": location, "start_ts": int(a.timestamp()),
                     "end_ts": int(b.timestamp()), "lambda_h": fit["lambda_h"],
                     "stderr": fit.get("stderr"), "r2": fit["r2"],
                     "c_start": float(co2.loc[a:b].iloc[0]), "c_end": float(co2.loc[a:b].iloc[-1]),
                     "c_out": c_out,
                     "delta_t_c": None if t_in is None or t_out is None else t_in - t_out,
                     "wind_ms": features.at(wind, mid),
                     "labelled": features.label_at(labels, mid, "window")})
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

    lam_sealed = db.params(location).get("lambda_sealed")
    lam_for_source = float(lam_sealed["value"]) if lam_sealed else DEFAULTS["lambda_sealed"][0]
    sources = []
    for a, b in features.plateau_windows(co2, c_out):
        if features.label_at(labels, int(a.timestamp()), "window") == "open":
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
    return summary


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
        self.volume_m3 = stored["volume_m3"]["value"] if "volume_m3" in stored else None

    @property
    def learned(self):
        """True once the parameters that drive a prediction are the room's own."""
        return all(self.basis[k] == "learned" for k in ("c_out", "lambda_sealed", "source_ppm_h"))

    def steady_state(self, occupied, ventilated=False):
        lam = self.value["lambda_open" if ventilated else "lambda_sealed"]
        return self.value["c_out"] + (self.value["source_ppm_h"] / lam if occupied else 0.0)

    def forward(self, c0, minutes, occupied, ventilated=False, step_min=5):
        """CO2 trajectory as [(minutes_from_now, ppm), ...].

        Solved in closed form per step rather than numerically: over a step where
        lambda and the source are constant the equation is exactly exponential."""
        lam = self.value["lambda_open" if ventilated else "lambda_sealed"]
        ss = self.steady_state(occupied, ventilated)
        decay = np.exp(-lam * step_min / 60.0)
        out, c = [], float(c0)
        for step in range(1, int(minutes / step_min) + 1):
            c = ss + (c - ss) * decay
            out.append((step * step_min, round(c, 1)))
        return out

    def time_to(self, threshold, c0, occupied, ventilated=False):
        """Minutes until CO2 reaches a threshold, or None if it never gets there."""
        lam = self.value["lambda_open" if ventilated else "lambda_sealed"]
        ss = self.steady_state(occupied, ventilated)
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
