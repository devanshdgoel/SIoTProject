"""
figures.py

Regenerates every figure the project has evidence for, from the live database.

Run it with no arguments to write PNGs into a figures/ directory next to the
database, or pass a destination:

    python -m analysis.figures [outdir]

Each function takes the database and an axis, so the same code can later produce
the report plates and the dashboard thumbnails without diverging.
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

from analysis import features              # noqa: E402
from storage.database import Database      # noqa: E402

BG, PANEL, INK, MUTE, GRID = "#0c1116", "#141b23", "#e6edf3", "#8b9bab", "#243140"
ROOM = {"desk": "#4f93e8", "bedroom": "#b07ae0"}
GOOD, WARN, BAD = "#3fb97f", "#e2a23b", "#e2574c"
BAND = {"present": ("#4f93e8", .13), "colocated": ("#3fb97f", .16),
        "open": ("#b07ae0", .18), "extractor": ("#e2a23b", .15)}


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": MUTE, "axes.edgecolor": GRID,
        "xtick.color": MUTE, "ytick.color": MUTE, "grid.color": GRID,
        "grid.alpha": .45, "axes.grid": True, "axes.titlecolor": INK,
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "600",
        "legend.frameon": False, "legend.labelcolor": MUTE,
        "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    })


def ts(t):
    return pd.Timestamp(int(t), unit="s", tz="UTC")


def timeaxis(ax, days=False):
    fmt = "%d %b %H:%M" if days else "%H:%M"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=None))
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)


def shade_labels(ax, labels, kinds=("occupancy", "window", "cooking"), lo=None, hi=None):
    """Paint the user's ground truth behind a trace."""
    seen = set()
    for lb in labels:
        if lb["kind"] not in kinds:
            continue
        colour, alpha = BAND.get(lb["value"], (None, 0))
        if colour is None:
            continue
        a, b = ts(lb["start_ts"]), ts(lb["end_ts"])
        if lo is not None and b < lo or hi is not None and a > hi:
            continue
        ax.axvspan(a, b, color=colour, alpha=alpha, lw=0,
                   label=None if lb["value"] in seen else lb["value"])
        seen.add(lb["value"])


# ---------------------------------------------------------------- panels
def fig_overview(db, out):
    """Both rooms across the whole record, with the labels behind them."""
    now = int(time.time())
    fig, ax = plt.subplots(figsize=(13, 5.2))
    labels = db.labels(start=now - 5 * 86400)
    shade_labels(ax, labels)
    lo = hi = None
    for room, colour in ROOM.items():
        s = features.series(db, "co2_ppm", room, now - 5 * 86400, now)
        if s.dropna().empty:
            continue
        # plotted with the NaN bins left in, so a stretch where the node was off
        # shows as a break rather than a straight line through nothing
        seen = s.dropna()
        lo = seen.index[0] if lo is None else min(lo, seen.index[0])
        hi = seen.index[-1] if hi is None else max(hi, seen.index[-1])
        ax.plot(s.index, s.values, color=colour, lw=1.5, label=room)
    for level, colour, name in ((1000, WARN, "1000 ppm"), (1400, BAD, "1400 ppm")):
        ax.axhline(level, color=colour, ls=":", lw=1, alpha=.7)
        ax.text(hi, level + 25, name, color=colour, fontsize=8, ha="right")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_title("Indoor CO$_2$, both rooms — shading is the ground truth you recorded")
    ax.legend(loc="upper left", ncol=6, fontsize=8)
    ax.set_xlim(lo, hi)
    timeaxis(ax, days=True)
    for lab in ax.get_xticklabels():
        lab.set_rotation(18)
        lab.set_ha("right")
    fig.tight_layout()
    fig.savefig(out / "01_overview.png")
    plt.close(fig)


def fig_colocation(db, out, window=None):
    """The two sensors in the same air: how far apart they really read."""
    labels = [lb for lb in db.labels(kind="occupancy") if lb["value"] == "colocated"]
    if not labels:
        return None
    lb = labels[-1] if window is None else window
    a, b = ts(lb["start_ts"]), ts(lb["end_ts"])
    metrics = [("co2_ppm", "CO$_2$ (ppm)", "ppm"),
               ("indoor_temp_c", "temperature (°C)", "°C"),
               ("indoor_rh_pct", "humidity (%)", "%")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), height_ratios=[2, 1])
    for col, (metric, ylabel, unit) in enumerate(metrics):
        top, bot = axes[0][col], axes[1][col]
        ser = {}
        for room, colour in ROOM.items():
            s = features.series(db, metric, room, lb["start_ts"] - 600,
                                lb["end_ts"] + 600).dropna()
            s = s.loc[a:b]
            ser[room] = s
            if not s.empty:
                top.plot(s.index, s.values, color=colour, lw=1.6, label=room)
        top.set_title(ylabel, fontsize=10)
        top.legend(fontsize=8, loc="best")
        timeaxis(top)
        if len(ser) == 2 and not ser["desk"].empty and not ser["bedroom"].empty:
            d = (ser["desk"] - ser["bedroom"]).dropna()
            med = float(np.median(d.values))
            bot.plot(d.index, d.values, color="#c9d6e2", lw=1.2)
            bot.axhline(med, color=GOOD, ls="--", lw=1.2)
            bot.axhline(0, color=MUTE, lw=.8, alpha=.5)
            bot.fill_between(d.index, d.values, med, color=GOOD, alpha=.12)
            bot.set_ylabel(f"desk − bedroom\n({unit})", fontsize=8)
            bot.text(.02, .86, f"median {med:+.2f} {unit}", transform=bot.transAxes,
                     color=GOOD, fontsize=9, fontweight="600")
        timeaxis(bot)
    fig.suptitle(f"Sensor co-location — empty, closed bedroom, "
                 f"{a:%d %b %H:%M} to {b:%H:%M} UTC", y=.99)
    fig.tight_layout()
    fig.savefig(out / "02_colocation.png")
    plt.close(fig)
    return True


def fig_decays(db, out):
    """Every fitted decay as a straight line in log space.

    The whole air-change measurement rests on one claim: with nobody in the room
    the excess CO2 falls exponentially, so its logarithm falls linearly and the
    slope is lambda. Plotting it this way makes the claim checkable by eye."""
    rows = [dict(r) for r in db.decays()]
    if not rows:
        return None
    rows.sort(key=lambda r: r["start_ts"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, room in zip(axes, ROOM):
        here = [r for r in rows if r["location"] == room]
        ax.set_title(f"{room} — {len(here)} fitted decays")
        ax.set_xlabel("hours from the start of the decay")
        ax.set_ylabel("ln(CO$_2$ − outdoor baseline)")
        for r in here:
            s = features.series(db, "co2_ppm", room, r["start_ts"] - 60,
                                r["end_ts"] + 60).dropna()
            s = s.loc[ts(r["start_ts"]):ts(r["end_ts"])]
            excess = s.values - r["c_out"]
            keep = excess > 1
            if keep.sum() < 3:
                continue
            hrs = (s.index - s.index[0]).total_seconds().to_numpy() / 3600.0
            vent = r["inferred"] == "ventilated"
            colour = WARN if vent else ROOM[room]
            ax.plot(hrs[keep], np.log(excess[keep]), lw=1.8, color=colour,
                    alpha=.9, ls="--" if vent else "-")
            ax.annotate(f"λ={r['lambda_h']:.2f}  r²={r['r2']:.3f}",
                        (hrs[keep][-1], np.log(excess[keep])[-1]),
                        textcoords="offset points", xytext=(6, -2),
                        fontsize=7.5, color=colour)
        ax.plot([], [], color=ROOM[room], lw=1.8, label="sealed")
        ax.plot([], [], color=WARN, lw=1.8, ls="--", label="ventilated")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Air-change rate: a straight line here is an exponential decay, "
                 "and its slope is λ", y=.99)
    fig.tight_layout()
    fig.savefig(out / "03_decays.png")
    plt.close(fig)
    return True


def fig_occupancy(db, out):
    """Rate of change against the labelled truth: the occupancy signature."""
    now = int(time.time())
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    labels = db.labels(start=now - 3 * 86400)
    for ax, room in zip(axes, ROOM):
        s = features.series(db, "co2_ppm", room, now - 3 * 86400, now)
        if s.dropna().empty:
            continue
        rate = s.diff() * 3600.0 / features.GRID_S
        rate = rate.rolling(3, center=True, min_periods=1).median()
        seen = s.dropna()
        shade_labels(ax, [lb for lb in labels if lb["location"] == room],
                     lo=seen.index[0], hi=seen.index[-1])
        ax.fill_between(rate.index, 0, rate.values,
                        where=rate.values > 0, color=BAD, alpha=.55, lw=0)
        ax.fill_between(rate.index, 0, rate.values,
                        where=rate.values <= 0, color=GOOD, alpha=.5, lw=0)
        ax.axhline(0, color=MUTE, lw=.8)
        ax.set_xlim(seen.index[0], seen.index[-1])
        # Carrying a sensor between rooms makes a step of several thousand ppm/h that
        # is an artefact of the move, not room dynamics. Clipped so the real signal,
        # which lives within about a thousand, is actually visible.
        ax.set_ylim(-1500, 1500)
        inside = rate.where(rate.abs() < 1500)
        peak = float(np.nanmax(inside.values))
        at = inside.index[int(np.nanargmax(inside.values))]
        ax.annotate(f"+{peak:.0f} ppm/h", (at, peak), textcoords="offset points",
                    xytext=(8, -4), color=BAD, fontsize=9, fontweight="600")
        ax.set_ylabel(f"{room}\ndCO$_2$/dt (ppm/h)")
        ax.legend(fontsize=8, loc="lower left", ncol=4)
    axes[0].set_title("Occupancy signature — red is a room filling, green is it clearing "
                      "(moves between rooms clipped)")
    timeaxis(axes[1], days=True)
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(18)
        lab.set_ha("right")
    fig.tight_layout()
    fig.savefig(out / "04_occupancy.png")
    plt.close(fig)


def fig_forecast(db, out):
    """Forecast error against the forecast that assumes nothing changes.

    The old version of this plotted model MAE by day issued, which cannot answer
    the only question a reader should ask: compared with WHAT? Every forecast is
    now scored against persistence - "the room stays where it is" - recorded at the
    same instant from the same information, and split by whether the model was
    running on this room's learned parameters or on published defaults.

    On this record the model loses at every room, horizon and basis. That is drawn
    rather than hidden, because it is the most informative result the project has.
    The diagnosis is in the bias column: with default parameters the bedroom is
    under-predicted by 200-615 ppm, which is a wrong room rather than a wrong
    method, while with learned parameters the bias nearly vanishes at 30 minutes
    (-15 ppm) and the error is almost all variance. So learning fixed the bias and
    did not buy accuracy - over half an hour a room with a 3.4-hour time constant
    barely moves, and "no change" is very hard to beat.

    What this does NOT say is that the model is useless. Persistence cannot answer
    the question the system is actually for - what happens if I open the window -
    because it has no counterfactual. The honest claim is narrower than the one the
    project started with: the model is for comparing options, not for extrapolating.
    """
    rows = db.forecast_skill()
    if not rows:
        return None
    rooms = sorted({r["location"] for r in rows})
    fig, axes = plt.subplots(1, max(1, len(rooms)), figsize=(6.6 * len(rooms), 4.8),
                             squeeze=False)
    for ax, room in zip(axes[0], rooms):
        mine = [r for r in rows if r["location"] == room]
        mine.sort(key=lambda r: (r["horizon_s"], r["basis"]))
        labels = [f"{r['horizon_s'] // 60} min\n{r['basis']}\nn={r['n']}" for r in mine]
        x = np.arange(len(mine))
        ax.bar(x - 0.2, [r["mae"] for r in mine], 0.4, label="physical model",
               color=ROOM.get(room, "#4f93e8"), alpha=.9)
        ax.bar(x + 0.2, [r["naive_mae"] for r in mine], 0.4,
               label="persistence (assume no change)", color=MUTE, alpha=.75)
        for xi, r in zip(x, mine):
            ax.text(xi, max(r["mae"], r["naive_mae"]),
                    f"skill {r['skill']:+.0%}", ha="center", va="bottom",
                    fontsize=8, color=WARN if r["skill"] < 0 else INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("mean absolute error (ppm)")
        ax.set_title(room)
        ax.legend(fontsize=8)
    fig.suptitle("Every forecast against the one that assumes nothing changes "
                 "— negative skill means the model lost", y=.99)
    fig.tight_layout()
    fig.savefig(out / "05_forecast_error.png")
    plt.close(fig)
    return True


def fig_nights(db, out):
    """Overnight exposure against the morning that followed it."""
    now = int(time.time())
    nights = features.night_windows(db, now - 40 * 86400, now)
    rows = []
    step_h = features.GRID_S / 3600.0
    for start, end, day in nights:
        s = features.series(db, "co2_ppm", "bedroom", start - 600, end + 600)
        s = s.loc[ts(start):ts(end)].dropna()
        if s.size < 6:
            continue
        with db.conn() as c:
            def daily(metric):
                r = c.execute("SELECT value FROM observations WHERE metric=? AND "
                              "location='user' AND DATE(ts,'unixepoch')=? LIMIT 1",
                              (metric, day)).fetchone()
                return r["value"] if r else np.nan
            rows.append({"day": day, "mean": float(s.mean()), "peak": float(s.max()),
                         "h1400": float((s > 1400).sum()) * step_h,
                         "hrv": daily("daily_sleep_hrv_ms"),
                         "recovery": daily("daily_recovery_index")})
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             width_ratios=[max(2, len(rows)), 3])
    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - .2, [r["mean"] for r in rows], .4, color=ROOM["bedroom"], label="mean")
    ax.bar(x + .2, [r["peak"] for r in rows], .4, color=BAD, alpha=.8, label="peak")
    ax.axhline(1400, color=WARN, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["day"][5:] for r in rows], fontsize=8)
    # keep the bars a sensible width even on the first night, so one night does not
    # fill the panel and read as more evidence than it is
    ax.set_xlim(-0.8, max(len(rows), 5) - 0.2)
    ax.set_ylabel("bedroom CO$_2$ (ppm)")
    ax.set_title("Overnight exposure, over the sleep window the ring recorded")
    for xi, r in zip(x, rows):
        ax.text(xi, r["peak"] + 40, f"{r['h1400']:.1f} h\n>1400",
                ha="center", fontsize=7, color=MUTE)
    ax.legend(fontsize=8)

    ax = axes[1]
    ok = [r for r in rows if not np.isnan(r["hrv"])]
    if len(ok) >= 2:
        sizes = [40 + 2.2 * r["h1400"] * 10 for r in ok]
        sc = ax.scatter([r["mean"] for r in ok], [r["hrv"] for r in ok], s=sizes,
                        c=[r["recovery"] for r in ok], cmap="viridis",
                        edgecolor=INK, linewidth=.6, zorder=3)
        plt.colorbar(sc, ax=ax, label="recovery index")
        for r in ok:
            ax.annotate(r["day"][5:], (r["mean"], r["hrv"]), fontsize=7,
                        color=MUTE, textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("overnight mean CO$_2$ (ppm)")
        ax.set_ylabel("sleep HRV (ms)")
        ax.set_title(f"Exploratory only — n={len(ok)} nights, far too few to conclude")
    else:
        ax.text(.5, .5, f"only {len(ok)} night(s) pair CO$_2$ with the ring yet",
                transform=ax.transAxes, ha="center", color=MUTE)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "06_nights.png")
    plt.close(fig)
    return True


def fig_cooking(db, out):
    """Cooking with the extractor on: mechanical ventilation, seen in the data."""
    cook = [lb for lb in db.labels(kind="cooking")]
    if not cook:
        return None
    lb = cook[-1]
    a, b = lb["start_ts"] - 3600, min(lb["end_ts"] + 600, int(time.time()))
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax = axes[0]
    for room, colour in ROOM.items():
        s = features.series(db, "co2_ppm", room, a, b).dropna()
        if not s.empty:
            ax.plot(s.index, s.values, color=colour, lw=1.8, label=room)
    ax.axvspan(ts(lb["start_ts"]), ts(lb["end_ts"]), color=WARN, alpha=.15, lw=0,
               label="hob + extractor")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_title("Cooking with the extractor running — forced ventilation, "
                 "and the sensor being carried back to the desk")
    ax.legend(fontsize=8, ncol=3)

    ax = axes[1]
    for metric, colour, label in (("indoor_temp_c", BAD, "temperature (°C)"),
                                  ("indoor_rh_pct", GOOD, "humidity (%)")):
        s = features.series(db, metric, "desk", a, b).dropna()
        if s.empty:
            continue
        twin = ax if metric == "indoor_temp_c" else ax.twinx()
        twin.plot(s.index, s.values, color=colour, lw=1.6, label=label)
        twin.set_ylabel(label, color=colour)
        twin.tick_params(axis="y", colors=colour)
        if twin is not ax:
            twin.grid(False)
    ax.axvspan(ts(lb["start_ts"]), ts(lb["end_ts"]), color=WARN, alpha=.15, lw=0)
    timeaxis(ax)
    fig.tight_layout()
    fig.savefig(out / "07_cooking.png")
    plt.close(fig)
    return True


def main(outdir=None):
    style()
    db = Database()
    out = Path(outdir) if outdir else Path(db.path).parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in (("overview", fig_overview), ("colocation", fig_colocation),
                     ("decays", fig_decays), ("occupancy", fig_occupancy),
                     ("forecast", fig_forecast), ("nights", fig_nights),
                     ("cooking", fig_cooking), ("estimators", fig_estimators),
                     ("openings", fig_openings)):
        try:
            fn(db, out)
            made.append(name)
        except Exception as e:                        # noqa: BLE001
            print(f"  {name}: skipped ({type(e).__name__}: {e})")
    print(f"wrote {len(made)} figures to {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} kB")



def fig_estimators(db, out):
    """Three independent routes to the same lambda, and what each one looks at.

    The top row shows which samples each method consumes: decay fitting uses only
    the handful of windows where the room happened to be empty and falling, while
    the other two use the whole record. The bottom row shows the spectrum each
    frequency estimate is read from, and the three answers side by side."""
    from analysis import estimators
    style()
    rooms = ["bedroom", "desk"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11),
                             gridspec_kw={"height_ratios": [1.1, 1.0, 0.9]})

    for col, room in enumerate(rooms):
        res = estimators.compare(db, room, days=14)
        c_out = res["c_out"]
        now = int(time.time())
        co2, _ = features.despike(features.series(db, "co2_ppm", room, now - 14 * 86400, now))
        co2 = co2.dropna()

        # --- what each method looks at -------------------------------------
        ax = axes[0][col]
        ax.plot(co2.index, co2.values, lw=0.8, color="#4f93e8" if room == "desk" else "#b07ae0",
                label="every sample (integral, spectrum)")
        used = 0
        for r in db.decays(room, min_r2=0.90):
            if r["inferred"] != "sealed":
                continue
            a = pd.Timestamp(r["start_ts"], unit="s", tz="UTC")
            b = pd.Timestamp(r["end_ts"], unit="s", tz="UTC")
            seg = co2.loc[a:b]
            used += len(seg)
            ax.plot(seg.index, seg.values, lw=2.4, color="#e2574c",
                    label="empty-room decays (decay fit)" if used == len(seg) else None)
        ax.axhline(c_out, ls=":", lw=1, color="#8b9bab")
        ax.set_title(f"{room} - decay fitting sees {100*used/max(len(co2),1):.0f}% of the record",
                     loc="left")
        ax.set_ylabel("CO$_2$ ppm")
        timeaxis(ax, days=True)
        ax.legend(loc="upper right", fontsize=8)

        # --- the spectrum the frequency estimate comes from -----------------
        ax = axes[1][col]
        sp = res["spectrum"].get("spectrum")
        if sp:
            f_, p_, fit = np.array(sp["freqs"]), np.array(sp["psd"]), np.array(sp["fit"])
            ax.loglog(f_, p_, lw=0.7, color="#8b9bab", label="measured spectrum")
            spec = res["spectrum"]
            slope = spec.get("slope_dec_per_dec")
            ax.loglog(f_, fit, lw=2.0, color="#3fb97f",
                      label=f"fitted roll-off {slope} dec/dec"
                            if slope is not None else "fit")
            fc = spec["corner_hz_per_h"]
            # The corner is only a rate if the roll-off says the room is first
            # order. Here it does not, so the line is drawn as a corner and
            # explicitly NOT labelled with a lambda - putting one there is what
            # made this figure claim three independent measurements of one
            # quantity when it had two.
            if spec.get("lambda_h"):
                tag = (f"corner {spec['corner_period_h']:.0f} h"
                       rf"  ->  $\lambda$={spec['lambda_h']:.2f}/h")
            else:
                tag = (f"corner {spec['corner_period_h']:.0f} h — not a rate: "
                       f"roll-off {slope} ≠ −2.0, so not first order")
            ax.axvline(fc, color="#3fb97f", ls="--", lw=1.2, label=tag)
            if res["decay"].get("lambda_h"):
                ax.axvline(res["decay"]["lambda_h"] / (2 * np.pi), color="#e2574c", ls="--",
                           lw=1.2, label=f"where decay fitting says it is")
            ax.axvline(1 / 24, color="#e2a23b", ls=":", lw=1, label="daily rhythm")
        ax.set_xlabel("cycles per hour"); ax.set_ylabel("power")
        ax.set_title("the room as a filter: fast rhythms are smoothed away", loc="left")
        ax.legend(fontsize=8)

        # --- the three answers ---------------------------------------------
        ax = axes[2][col]
        names, vals, errs, cols, ns = [], [], [], [], []
        for key, colour in (("decay", "#e2574c"), ("integral", "#e2a23b"),
                            ("spectrum", "#3fb97f")):
            r = res[key]
            if r.get("lambda_h") is None:
                continue
            names.append(key); vals.append(r["lambda_h"])
            errs.append(r.get("stderr") or 0.0); cols.append(colour)
            ns.append(f"n={r['n']}, {r.get('samples_used', 0)} samples")
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=errs, color=cols, height=0.55, capsize=5, alpha=.85)
        ax.set_yticks(y); ax.set_yticklabels(names)
        for i, (v, note) in enumerate(zip(vals, ns)):
            ax.text(v + (errs[i] or 0) + 0.02, i, f"{v:.3f}/h   {note}",
                    va="center", fontsize=8, color="#c7d3de")
        ax.set_xlim(0, max(vals + [0.1]) * 1.9)
        ax.set_xlabel("air changes per hour")
        # Not "three routes to the same number": the spectral route declines to
        # give one at all (its roll-off rejects first order), and the two that do
        # disagree by more than their error bars. Titling a disagreement as
        # agreement is the failure mode this whole figure exists to prevent.
        ax.set_title("what each route can measure, and how far apart they land",
                     loc="left")

    fig.suptitle("Three routes to the air-change rate — one of them refuses, "
                 "and the other two disagree", y=.995)
    fig.tight_layout()
    path = Path(out) / "08_estimators.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  {path.name}  {path.stat().st_size//1024} kB")
    return path


PRE_MIN, POST_MIN = 20, 25       # context either side of the opening
SEARCH_MIN = 45                  # how far from a reported time to hunt the peak
MIN_EXCESS_PPM = 40              # below this there is no excess worth scaling by


def _minutes_from(series, origin_ts):
    """Index as minutes since `origin_ts`, so two events on different days overlay."""
    secs = series.index.astype("int64") // 10 ** 9
    return (secs - int(origin_ts)) / 60.0


def fig_openings(db, out, location="bedroom"):
    """Every deliberate window opening on one clock, each curve named by its wind.

    The two bedroom tests are the same window in the same room: 4.55 air changes an
    hour in 3.7 m/s against 2.58 in 0.75 m/s. A factor of 1.8 for a factor of five in
    wind, which is the measurement that refuted proportionality through the origin -
    and the lower panel says why. The still night had the LARGER indoor-outdoor
    difference, 9.5 C against 6.5, so buoyancy was working harder on the night the
    wind was doing nothing. An open window here is a stack-driven opening that wind
    adds to, not a wind-driven one.

    CO2 is drawn as excess above the outdoor baseline, as a percentage of the excess
    at the peak. That is not cosmetic. A first-order room obeys

        C(t) = C_out + (C_0 - C_out) e^(-lambda t)

    so subtracting the baseline and dividing by the starting excess leaves e^(-lambda
    t) exactly - a curve that depends on the air-change rate and on nothing else. The
    two openings started 475 ppm apart, which otherwise shows up as a vertical gap
    that has no bearing on how fast either room cleared. Normalised, the only thing
    left between the curves is the quantity being compared. The absolute levels are
    kept in the legend so nothing is hidden by the transform.

    Zero is the start of the fitted decay, and both curves are normalised to their
    excess there. The peak is the wrong reference, which the normalisation itself is
    what revealed: anchored at the peak the two curves very nearly overlap for the
    first half hour and cross 1/e within six minutes of each other, which cannot be
    reconciled with air-change rates five times apart.

    The 6 Nov curve does not reach zero, and that is the second finding rather than
    a defect in the measurement. It flattens at 29% of its starting excess, which in
    ppm is 479 - far above the 358 ppm outside, and close to where the desk room
    sits when nobody has aired it. With the window open on a still night the air replacing
    the bedroom's was mostly coming through the door from the rest of the flat, not
    in through the window. Each curve is therefore drawn against the asymptote its
    own fit found, marked where it differs from outdoor air.

    This was originally read as a two-phase decay with a breath plume in front of it.
    It is not: the sensor is far from where the occupant sleeps, and fitting the same
    trace with the asymptote left free gives one clean exponential at r2 = 0.999. The
    arc that looked like a second phase was the residual of forcing the curve down to
    a floor the room was never heading for. Lead-in that the fit trims is still drawn
    faintly, because how much had to be trimmed is worth seeing.
    """
    labels = db.labels(location=location)
    events = sorted((lb for lb in labels
                     if lb["kind"] == "window" and lb["value"] == "open"),
                    key=lambda lb: lb["start_ts"])
    if not events:
        raise ValueError(f"no labelled window openings for {location}")

    fits = [r for r in db.decays(location, min_r2=0.9) if r["labelled"] == "open"]
    stored = db.params(location)
    c_out = float(stored["c_out"]["value"]) if "c_out" in stored else None

    fig, (ax_c, ax_t) = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True,
                                     gridspec_kw={"height_ratios": [1.3, 1]})
    palette = ["#4f93e8", "#e2a23b", "#3fb97f", "#b07ae0", "#e2574c"]
    shut_marks = []

    for i, ev in enumerate(events):
        colour = palette[i % len(palette)]
        fit = next((r for r in fits
                    if ev["start_ts"] - 1800 <= r["start_ts"] <= ev["end_ts"]), None)
        wide = features.series(db, "co2_ppm", location,
                               ev["start_ts"] - 3600, ev["end_ts"] + POST_MIN * 60).dropna()
        if wide.empty:
            continue
        # The peak, kept for the legend: it says what the room actually reached, and
        # how much of the record the fit had to discard before it behaved.
        near = wide.loc[ts(ev["start_ts"] - SEARCH_MIN * 60):
                        ts(ev["start_ts"] + SEARCH_MIN * 60)]
        peak_ts = int((near.idxmax() if not near.empty else wide.index[0]).timestamp())
        origin = int(fit["start_ts"]) if fit is not None else peak_ts
        a, b = min(origin, peak_ts) - PRE_MIN * 60, ev["end_ts"] + POST_MIN * 60

        co2 = wide.loc[ts(a):ts(b)]
        t_in = features.series(db, "indoor_temp_c", location, a, b).dropna()
        t_out = features.series(db, "outdoor_temp_c", "home", a, b).dropna()
        if co2.empty:
            continue

        when = ts(peak_ts).strftime("%d %b %H:%M")
        gust = (float(fit["wind_ms"]) if fit is not None and fit["wind_ms"] is not None
                else float("nan"))
        peak = float(co2.loc[ts(peak_ts)]) if ts(peak_ts) in co2.index else float(co2.max())
        trimmed = (origin - peak_ts) / 60.0
        name = f"{when} — wind {gust:.1f} m/s,  peak {peak:.0f} ppm"
        if fit is not None:
            name += f",  λ = {fit['lambda_h']:.2f}/h"
            if trimmed >= 10:
                name += f"\n        (fit starts {trimmed:.0f} min after the peak)"

        # Excess above the baseline, as a share of the excess where the fit begins.
        # Guard the division: a room already sitting on outdoor air has no excess to
        # scale by, and dividing by near-zero turns sensor noise into a signal.
        base = c_out if c_out is not None else float(co2.min())
        at_origin = (float(co2.loc[ts(origin)]) if ts(origin) in co2.index
                     else float(co2.iloc[0]))
        excess0 = at_origin - base
        if excess0 < MIN_EXCESS_PPM:
            continue
        x_all = _minutes_from(co2, origin)
        y_all = (co2.values - base) / excess0 * 100.0
        # Split rather than overlay: a translucent line drawn on top of a solid one
        # still reads as solid, and the point of the lead-in is that it is excluded.
        keep, drop = x_all >= 0, x_all <= 0
        ax_c.plot(x_all[keep], y_all[keep], color=colour, lw=1.9, marker="o",
                  ms=2.6, label=name, zorder=3)
        if drop.sum() > 1:
            ax_c.plot(x_all[drop], y_all[drop], color=colour, lw=1.4, alpha=.3,
                      ls=(0, (2, 2)), zorder=2.5)

        # The fitted exponential over its own span, normalised the same way, to show
        # the model IS the curve rather than a line drawn near it.
        if fit is not None:
            span = co2.loc[ts(fit["start_ts"]):ts(fit["end_ts"])]
            if span.size >= 3:
                x = _minutes_from(span, origin)
                # Each curve is drawn against the asymptote its own fit found, not
                # against outdoor air. That is the whole correction: an opening that
                # settles on house air is not a slow approach to outdoors, and a
                # model curve aimed at zero would hide exactly the discrepancy the
                # axis is here to show.
                c_inf = (float(fit["c_inf"]) if fit["c_inf"] is not None else base)
                model = (c_inf - base) + (at_origin - c_inf) * np.exp(
                    -fit["lambda_h"] * x / 60.0)
                ax_c.plot(x, model / excess0 * 100.0, color=colour, lw=1.0,
                          ls=(0, (4, 3)), alpha=.9, zorder=4)
                floor = (c_inf - base) / excess0 * 100.0
                if floor > 4.0:
                    ax_c.axhline(floor, color=colour, lw=1.0, ls=(0, (1, 2)),
                                 alpha=.65, zorder=1.5)
                    ax_c.annotate(f"settles at {c_inf:.0f} ppm — house air, "
                                  f"not outdoors",
                                  (0.985, floor), xycoords=("axes fraction", "data"),
                                  xytext=(0, 5), textcoords="offset points",
                                  ha="right", color=colour, fontsize=7.5)

        shut = (ev["end_ts"] - origin) / 60.0
        shut_marks.append((shut, colour))

        # --- temperature: indoors only ---------------------------------------
        # Outdoor sits around 9-11 C against an indoor 15-19, so plotting both on one
        # axis squeezes the indoor trace into a band too thin to read the cooling off.
        # The outdoor figure matters as the DIFFERENCE driving that cooling, so it
        # belongs in the legend as dT rather than as a second line.
        if not t_in.empty:
            held = t_in.loc[ts(origin):ts(ev["end_ts"])]
            outside = float(t_out.mean()) if not t_out.empty else float("nan")
            if held.size >= 2:
                # The deepest cooling, not first-minus-last: the 4 Nov opening ran
                # long enough for the heating to come back on, which would otherwise
                # score a three-hour airing as a temperature GAIN.
                lost = float(held.iloc[0] - held.min())
                dT = float(held.mean()) - outside
                xt = _minutes_from(t_in, origin)
                k, d = xt >= 0, xt <= 0
                ax_t.plot(xt[k], t_in.values[k], color=colour, lw=1.9, marker="o",
                          ms=2.6, zorder=3,
                          label=f"{when} — outdoors {outside:.1f} °C, "
                                f"ΔT {dT:.1f} °C,  fell {lost:.1f} °C")
                if d.sum() > 1:
                    ax_t.plot(xt[d], t_in.values[d], color=colour, lw=1.4,
                              alpha=.3, ls=(0, (2, 2)), zorder=2.5)

    ax_c.axhline(0, color=MUTE, lw=1.1, ls="--", alpha=.75, zorder=1)
    ax_c.annotate(f"outdoor baseline, {c_out:.0f} ppm" if c_out is not None
                  else "outdoor baseline",
                  (0.015, 0), xycoords=("axes fraction", "data"),
                  xytext=(0, 5), textcoords="offset points",
                  color=MUTE, fontsize=7.5)
    # Where a curve crosses 1/e it has run for exactly one time constant, 1/lambda.
    # The horizontal distance between the two crossings IS the ratio of the rates,
    # which is the whole comparison, readable off the axis without arithmetic.
    ax_c.axhline(100 / math.e, color=MUTE, lw=1.0, ls=":", alpha=.65, zorder=1)
    ax_c.annotate("1/e — one time constant, 1/λ",
                  (0.015, 100 / math.e), xycoords=("axes fraction", "data"),
                  xytext=(0, 5), textcoords="offset points",
                  color=MUTE, fontsize=7.5)

    # Each opening ran for a different length, so the close is marked per event, at
    # the foot of the axis where it cannot collide with the traces or the legend.
    for ax in (ax_c, ax_t):
        ax.axvline(0, color=INK, lw=1.0, alpha=.35, zorder=1)
        for shut, colour in shut_marks:
            ax.axvline(shut, color=colour, lw=1.0, ls=":", alpha=.6, zorder=1)
    lo, hi = ax_c.get_ylim()
    for shut, colour in shut_marks:
        ax_c.annotate(f"shut, {shut:.0f} min", (shut, lo), xytext=(4, 6),
                      textcoords="offset points", color=colour, fontsize=7.5)

    ax_c.set_ylim(-6, 128)      # the trimmed plume leaves the top of the panel
    ax_c.set_title(f"the same {location} window, opened twice "
                   f"(CO₂ normalised where each fit begins)")
    ax_c.set_ylabel("CO₂ excess above baseline\n(% of excess at t = 0)")
    ax_c.legend(loc="upper right", fontsize=8)
    ax_t.set_title("indoor temperature over the same openings")
    ax_t.set_ylabel("temperature (°C)")
    ax_t.set_xlabel("minutes since the fitted decay begins (0 = measured, not reported)")
    ax_t.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "09_openings.png", dpi=140)
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
