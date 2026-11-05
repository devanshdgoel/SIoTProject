"""
figures.py

Regenerates every figure the project has evidence for, from the live database.

Run it with no arguments to write PNGs into a figures/ directory next to the
database, or pass a destination:

    python -m analysis.figures [outdir]

Each function takes the database and an axis, so the same code can later produce
the report plates and the dashboard thumbnails without diverging.
"""

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
    """Forecast error by horizon: the evidence the model is or is not improving."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    any_data = False
    for ax, room in zip(axes, ROOM):
        rows = [dict(r) for r in db.prediction_error(room, days=60)]
        ax.set_title(f"{room}")
        ax.set_ylabel("mean absolute error (ppm)")
        if not rows:
            ax.text(.5, .5, "no scored forecasts yet", transform=ax.transAxes,
                    ha="center", color=MUTE)
            ax.set_axis_off()
            continue
        any_data = True
        horizons = sorted({r["horizon_s"] for r in rows})
        days = sorted({r["day"] for r in rows})
        width = .8 / max(1, len(horizons))
        for i, h in enumerate(horizons):
            vals = [next((r["mae"] for r in rows
                          if r["day"] == d and r["horizon_s"] == h), np.nan) for d in days]
            ax.bar(np.arange(len(days)) + i * width, vals, width,
                   color=[ROOM[room], WARN][i % 2], alpha=.85,
                   label=f"{h // 60} min ahead")
            for x, v in zip(np.arange(len(days)) + i * width, vals):
                if not np.isnan(v):
                    ax.text(x, v, f"{v:.0f}", ha="center", va="bottom",
                            fontsize=8, color=INK)
        ax.set_xticks(np.arange(len(days)) + width * (len(horizons) - 1) / 2)
        ax.set_xticklabels(days, fontsize=8)
        ax.legend(fontsize=8)
    if not any_data:
        plt.close(fig)
        return None
    fig.suptitle("Forecast error — every prediction was written down before its "
                 "outcome was known", y=.99)
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
                     ("cooking", fig_cooking)):
        try:
            fn(db, out)
            made.append(name)
        except Exception as e:                        # noqa: BLE001
            print(f"  {name}: skipped ({type(e).__name__}: {e})")
    print(f"wrote {len(made)} figures to {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} kB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
