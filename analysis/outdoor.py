"""
outdoor.py - how well do four claims about 'outside' actually agree?

Every indoor result in this project is measured against an outdoor reference:
the excess CO2 that defines occupancy, the penetration factor, the purge cost in
micrograms. Until now 'outside' has been whichever source happened to be in the
database, and nobody had asked whether they agree.

Four sources, in order of how directly they measure the air at this flat:

    the sensor itself, window open   0 km    the actual air, for an hour at a time
    Breathe London                  0.7 km  low-cost optical, dense
    AURN                            ~3 km   reference-grade analysers
    Open-Meteo (CAMS)               ~11 km  a model, not a measurement

Distance and instrument quality run in opposite directions, so there is no single
best source - which is the point. This module pairs them on a common hourly grid
and reports bias, scatter and correlation, so a disagreement becomes a number.

It exists because of one specific contradiction. The bedroom's decays fit an
outdoor CO2 baseline of 358 ppm. The global atmospheric background is about
423 ppm and rising; Open-Meteo's model for this postcode has never reported an
hour below 445. A room cannot decay towards air that is cleaner than the planet's,
so either the sensor reads low or the asymptote is biased - and that question was
unanswerable until the modelled series was collected to ask it against.
"""

import time

import numpy as np

from config import settings

# NOAA global mean marine surface CO2, extrapolated to late 2026 at the recent
# ~2.5 ppm/yr trend. Deliberately a floor, not a target: a UK city is always at
# or above background, never below it, so any fitted outdoor baseline below this
# figure is wrong regardless of how well the curve fitted.
GLOBAL_CO2_FLOOR_PPM = 423.0

# How long after a window opens the room can be treated as being outdoor air.
# Short enough that the hour is really ventilated, long enough to settle: at the
# measured open rate of 3.56/h, 20 minutes is 1.2 time constants, which closes
# about 70% of the gap.
OPEN_SETTLE_MIN = 20


def _hourly(db, metric, location=None, kind=None, days=14):
    """One source's series, averaged to the hour so sources are comparable."""
    start = int(time.time()) - days * 86400
    df = db.read_df(metric, location=location, start=start, kind=kind)
    if df.empty:
        return None
    return df["value"].resample("1h").mean().dropna()


def _pair(a, b):
    """Bias, scatter and correlation on the hours both sources covered."""
    joined = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    if len(joined) < 6:
        return None
    x, y = joined["a"].to_numpy(float), joined["b"].to_numpy(float)
    d = y - x
    out = {"n": len(joined), "bias": float(np.mean(d)),
           "rms": float(np.sqrt(np.mean(d ** 2))),
           "mean_a": float(np.mean(x)), "mean_b": float(np.mean(y))}
    if np.std(x) > 0 and np.std(y) > 0:
        out["r"] = float(np.corrcoef(x, y)[0, 1])
    # Ratio is the useful figure for particles: a model that is 45% low is a
    # scaling error, which matters more than an offset for penetration work.
    out["ratio"] = float(np.mean(y) / np.mean(x)) if np.mean(x) else float("nan")
    return out


def pm_sources(db, days=14):
    """Every PM2.5 series available, hour-aligned, plus pairwise comparisons."""
    series, labels = {}, {}
    model = _hourly(db, "pm2_5", location="home", kind="modelled", days=days)
    if model is not None:
        series["open_meteo"] = model
        labels["open_meteo"] = "Open-Meteo CAMS, ~11 km cell (modelled)"
    df = db.read_df("pm2_5", start=int(time.time()) - days * 86400)
    for loc in sorted(set(df["location"])) if not df.empty else []:
        if loc == "home":
            continue
        s = _hourly(db, "pm2_5", location=loc, days=days)
        if s is None or len(s) < 6:
            continue
        series[loc] = s
        if loc.startswith("bl_"):
            labels[loc] = "Breathe London node (measured, low-cost)"
        elif loc.startswith("london_"):
            labels[loc] = "AURN reference analyser (measured)"
        else:
            labels[loc] = "this flat, indoors (measured)"
    return series, labels


def open_window_co2(db, location=None, days=30):
    """What the sensor itself reads when the window is open - the 0 km reference.

    Two routes, because labels are scarce:
      labelled  a window-open label, taking the settled part of it
      fitted    the asymptote of a decay the classifier called ventilated
    They answer slightly different questions - the first is what the air was, the
    second is what the room was heading towards - so both are reported rather
    than averaged.
    """
    location = location or settings.SCD40_LOCATION
    out = {"labelled": [], "fitted": []}
    for lab in (dict(r) for r in db.labels(location=location, kind="window")):
        if str(lab.get("value", "")).lower() not in ("open", "ajar"):
            continue
        a = int(lab["start_ts"]) + OPEN_SETTLE_MIN * 60
        b = int(lab["end_ts"])
        if b - a < 300:
            continue
        df = db.read_df("co2_ppm", location=location, start=a, end=b)
        if df.empty:
            continue
        out["labelled"].append({
            "from": a, "to": b, "minutes": (b - a) // 60,
            "min": float(df["value"].min()),
            "settled": float(df["value"].tail(10).mean())})
    for d in (dict(r) for r in db.decays(location=location)):
        if (d.get("inferred") or "") != "ventilated":
            continue
        if d.get("c_inf") is None:
            continue
        out["fitted"].append({"from": int(d["start_ts"]), "lambda_h": d["lambda_h"],
                              "c_inf": float(d["c_inf"]), "c_end": d.get("c_end")})
    return out


def co2_baseline_check(db, location=None, days=14):
    """Is the fitted outdoor CO2 baseline physically possible?"""
    location = location or settings.SCD40_LOCATION
    params = db.params(location) or {}
    row = params.get("c_out")
    fitted = row["value"] if row is not None else None
    # How c_out was arrived at decides what a contradiction means. If it was
    # measured during a labelled window-open period then it IS the 0 km reading,
    # and no amount of curve-fitting scepticism explains it away - the sensor is
    # the only remaining suspect.
    how = None
    if row is not None and row["detail"]:
        try:
            import json
            how = (json.loads(row["detail"]) or {}).get("method")
        except (ValueError, TypeError):
            how = None

    model = _hourly(db, "co2_outdoor_ppm", location="home", kind="modelled", days=days)
    indoor = _hourly(db, "co2_ppm", location=location, days=days)

    report = {"location": location, "fitted_c_out": fitted, "how": how,
              "global_floor": GLOBAL_CO2_FLOOR_PPM, "open_window": open_window_co2(db, location)}
    if model is not None and len(model):
        report["model"] = {"n": int(len(model)), "min": float(model.min()),
                           "median": float(model.median()), "max": float(model.max())}
    if indoor is not None and len(indoor):
        report["indoor_min"] = float(indoor.min())

    verdicts = []
    if fitted is not None:
        if fitted < GLOBAL_CO2_FLOOR_PPM:
            verdicts.append(
                f"the fitted outdoor baseline {fitted:.0f} ppm is BELOW the global "
                f"background ({GLOBAL_CO2_FLOOR_PPM:.0f} ppm). No air anywhere is that "
                f"clean, so this is an instrument or fitting bias of at least "
                f"{GLOBAL_CO2_FLOOR_PPM - fitted:.0f} ppm, not a property of the street.")
        if "model" in report and fitted < report["model"]["min"]:
            verdicts.append(
                f"it is also {report['model']['min'] - fitted:.0f} ppm below the lowest "
                f"hour the model has reported here ({report['model']['min']:.0f} ppm over "
                f"{report['model']['n']} hours) - two independent reasons, same direction.")
        if report.get("indoor_min") is not None and report["indoor_min"] < GLOBAL_CO2_FLOOR_PPM:
            verdicts.append(
                f"the sensor itself has read as low as {report['indoor_min']:.0f} ppm "
                f"indoors, which points at the sensor rather than the fit: an asymptote "
                f"cannot explain a raw reading.")
        else:
            verdicts.append(
                "the sensor's own lowest raw reading is above the global floor, so the "
                "bias is more likely in the extrapolated asymptote than in the sensor.")
        if how and "measured directly" in how:
            verdicts.append(
                "and this baseline was MEASURED, not extrapolated - it is what the "
                "sensor read with the window open, which is the 0 km reference. A "
                "fitting artefact cannot produce it, so the SCD40 is reading low. "
                "The SCD40's automatic self-calibration assumes the lowest value it "
                "sees over its window is 400 ppm fresh air; in a bedroom whose true "
                "floor is higher than that, ASC drags every reading down by the "
                "difference, which is exactly this signature.")
    report["verdicts"] = verdicts

    # The correction is deliberately NOT applied anywhere. Differences carry it
    # away: an offset shared by every reading cancels in C - c_out, so lambda and
    # the source term are unaffected, while absolute ppm and the 1000/1400/2000
    # thresholds are not. Saying which results move is more useful than silently
    # shifting the series.
    if fitted is not None and fitted < GLOBAL_CO2_FLOOR_PPM:
        report["affected"] = {
            "unchanged": ["lambda_sealed", "lambda_open", "source_ppm_h",
                          "forecast skill", "decay classification"],
            "shifted_low_by_ppm": round(GLOBAL_CO2_FLOOR_PPM - fitted),
            "affected": ["absolute CO2 shown to the user", "hours above 1400 ppm",
                         "the ventilation thresholds", "any CO2 comparison with "
                         "an outdoor source"]}
    return report


def rooms(db):
    """Indoor locations that have CO2, newest first - so both rooms get checked.

    The default used to be settings.SCD40_LOCATION, which is the desk: the room
    with no particle counter, no measured volume and a baseline already known to
    be house air rather than outdoor. The interesting case was silently skipped.
    """
    import time as _t
    with db.conn() as c:
        got = c.execute(
            "SELECT location, MAX(ts) t FROM observations WHERE metric='co2_ppm' "
            "AND ts > ? GROUP BY location ORDER BY t DESC",
            (int(_t.time()) - 30 * 86400,)).fetchall()
    return [r[0] for r in got]


def compare(db, days=14):
    """Everything: the PM cross-source table and the CO2 baseline check."""
    series, labels = pm_sources(db, days=days)
    pairs = {}
    names = sorted(series)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            got = _pair(series[a], series[b])
            if got:
                pairs[f"{a} vs {b}"] = got
    return {"pm": {"sources": {k: {"n": int(len(v)), "mean": float(v.mean()),
                                   "min": float(v.min()), "max": float(v.max()),
                                   "label": labels.get(k, "")}
                               for k, v in series.items()},
                   "pairs": pairs},
            "co2": [co2_baseline_check(db, location=r, days=days) for r in rooms(db)],
            "missing": [n for n in ("breathe_london",)
                        if not any(k.startswith("bl_") for k in series)]}
