"""
Analyse the animal's daily movement against the weather / air quality it met.

Reads  init_setup/location_weather_history.csv
Writes init_setup/fig1..fig5 PNGs and prints a correlation table.

Needs: pandas, numpy, matplotlib   (pip install pandas numpy matplotlib)
Run:   python init_setup/animal_history_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "location_weather_history.csv"

MOVE_THRESHOLD_KM = 5.0      # a day counts as "moving" above this
ENV = ["temp_c", "temp_min_c", "temp_max_c", "wind_speed_max_mps",
       "precipitation_mm", "humidity_percent", "pressure_hpa",
       "cloud_cover_percent", "aqi", "pm2_5", "pm10", "no2", "o3"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def spearman(a, b):
    """Spearman = Pearson on ranks. Done by hand so scipy isn't needed."""
    return a.rank().corr(b.rank())


def load():
    df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # net step between one day's mean position and the next
    df["step_km"] = haversine_km(df["lat"].shift(), df["lon"].shift(),
                                 df["lat"], df["lon"]).round(3)
    # straight-line distance from where the track starts
    df["net_km"] = haversine_km(df["lat"].iloc[0], df["lon"].iloc[0],
                                df["lat"], df["lon"]).round(3)
    df["cum_path_km"] = df["daily_path_km"].cumsum().round(1)
    df["path_7d"] = df["daily_path_km"].rolling(7, min_periods=3).mean()
    df["moving"] = df["daily_path_km"] > MOVE_THRESHOLD_KM
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def correlations(df, target="daily_path_km"):
    """Pearson (linear) and Spearman (rank/monotonic) against each variable,
    for the same day and for the previous day's weather (lag 1)."""
    out = []
    for col in ENV:
        if col not in df:
            continue
        for lag in (0, 1):
            pair = pd.DataFrame({
                "y": df[target],
                "x": df[col].shift(lag),
            }).dropna()
            if len(pair) < 10:
                continue
            out.append({
                "variable": col,
                "lag_days": lag,
                "n": len(pair),
                "pearson": round(pair["y"].corr(pair["x"]), 3),
                "spearman": round(spearman(pair["y"], pair["x"]), 3),
            })
    table = pd.DataFrame(out)
    return table.reindex(table["spearman"].abs().sort_values(ascending=False).index)


def fig_track(df):
    fig, ax = plt.subplots(figsize=(7, 7))
    days = (df["date"] - df["date"].min()).dt.days
    ax.plot(df["lon"], df["lat"], color="0.8", lw=0.8, zorder=1)
    pts = ax.scatter(df["lon"], df["lat"], c=days, cmap="viridis", s=22, zorder=2)
    ax.scatter(df["lon"].iloc[0], df["lat"].iloc[0], marker="o", s=120,
               facecolor="none", edgecolor="green", lw=2, label="start", zorder=3)
    ax.scatter(df["lon"].iloc[-1], df["lat"].iloc[-1], marker="*", s=220,
               color="crimson", label="latest", zorder=3)
    fig.colorbar(pts, ax=ax, label="days since start")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Track of daily mean positions")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig1_track.png", dpi=150)


def fig_timeseries(df):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(df["date"], df["daily_path_km"], width=1.0, color="#3b6fb0",
           alpha=0.55, label="daily path (km)")
    ax.plot(df["date"], df["path_7d"], color="#13243a", lw=2, label="7-day mean")
    ax.set_ylabel("distance travelled (km/day)")
    ax.set_xlabel("date")

    ax2 = ax.twinx()
    ax2.plot(df["date"], df["temp_c"], color="#d6556b", lw=1.2, alpha=0.9,
             label="mean temp (C)")
    ax2.set_ylabel("mean temperature (C)", color="#d6556b")
    ax2.tick_params(axis="y", colors="#d6556b")

    lines = ax.get_legend_handles_labels()
    extra = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + extra[0], lines[1] + extra[1], loc="upper left", fontsize=8)
    ax.set_title("Daily movement and temperature over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_timeseries.png", dpi=150)


def fig_scatters(df):
    picks = [c for c in ["temp_c", "wind_speed_max_mps", "aqi", "pm2_5"] if c in df]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), picks):
        pair = df[[col, "daily_path_km"]].dropna()
        x, y = pair[col].values, pair["daily_path_km"].values
        ax.scatter(x, y, s=20, alpha=0.6, color="#0f9b8e", edgecolor="none")
        # Movement is very skewed: lots of near-zero days, a few long flights.
        # A symlog axis shows both, and the trend is fitted in log space so the
        # line is not dragged around by the handful of big days.
        ax.set_yscale("symlog", linthresh=0.1)
        if len(pair) > 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, np.log1p(y), 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.expm1(slope * xs + intercept), color="#c8871b", lw=1.8)
        r = pair["daily_path_km"].corr(pair[col])
        rho = spearman(pair["daily_path_km"], pair[col])
        ax.set_title(f"{col}   r={r:.2f}  rho={rho:.2f}  n={len(pair)}", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("daily path (km)")
        ax.grid(alpha=0.3)
    fig.suptitle("Daily distance travelled vs environment", y=1.0)
    fig.tight_layout()
    fig.savefig(HERE / "fig3_scatters.png", dpi=150)


def fig_heatmap(df):
    cols = ["daily_path_km", "step_km"] + [c for c in ENV if c in df]
    corr = df[cols].rank().corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Rank correlation matrix")
    fig.tight_layout()
    fig.savefig(HERE / "fig4_heatmap.png", dpi=150)


def fig_monthly(df):
    groups = [g["daily_path_km"].dropna().values for _, g in df.groupby("month")]
    labels = [m for m, _ in df.groupby("month")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.set_ylabel("daily path (km)")
    ax.set_xlabel("month")
    ax.set_title("Distribution of daily movement by month")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "fig5_monthly.png", dpi=150)


def main():
    df = load()

    print(f"\ndays: {len(df)}   {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"fixes per day: median {df['n_fixes'].median():.0f}")
    print(f"total path travelled: {df['daily_path_km'].sum():.1f} km")
    print(f"daily path: mean {df['daily_path_km'].mean():.2f} km, "
          f"median {df['daily_path_km'].median():.2f} km, "
          f"max {df['daily_path_km'].max():.1f} km "
          f"on {df.loc[df['daily_path_km'].idxmax(), 'date'].date()}")
    print(f"moving days (>{MOVE_THRESHOLD_KM} km): "
          f"{df['moving'].sum()} of {len(df)} ({100 * df['moving'].mean():.0f}%)")
    print(f"furthest from start: {df['net_km'].max():.1f} km")

    table = correlations(df)
    print("\ncorrelation with daily distance travelled "
          "(sorted by |spearman|):")
    print(table.to_string(index=False))

    fig_track(df)
    fig_timeseries(df)
    fig_scatters(df)
    fig_heatmap(df)
    fig_monthly(df)
    print(f"\nsaved 5 figures to {HERE}")

    df.to_csv(HERE / "animal_history_metrics.csv", index=False)
    table.to_csv(HERE / "animal_history_correlations.csv", index=False)
    print("saved animal_history_metrics.csv and animal_history_correlations.csv")
    plt.show()


if __name__ == "__main__":
    main()
