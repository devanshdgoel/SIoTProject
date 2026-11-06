"""
main.py

Starts the application, schedules data collection and analysis,
handles errors, and connects the analysis outputs to the pet controller.

Week 1 commands (collection only):
  python main.py collect            run all collectors forever (one thread each)
  python main.py once [name ...]    run collectors once and print the outcome
  python main.py backfill --days 7  pull history for sources that offer it
  python main.py status             row counts, freshness and reliability per source
  python main.py analyse            one pass: re-fit the rooms, decide, forecast
  python main.py label bedroom window open --last 60
                                    tell the system what was true (calibrates it)
  python main.py params             what the system has learned about each room
  python main.py dashboard          monitoring UI at http://127.0.0.1:8050 (separate process)
  python main.py simulate --days 7 --db data/sim.db
                                    synthetic SCD40 week for developing analysis
"""

import argparse
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis import analysis_human
from collectors import aurn, open_meteo, scd40, ultrahuman
from collectors.base import run_collector
from config import settings
from storage.database import Database

# Every collector the project has. What actually runs is the subset named in
# settings.COLLECTORS_ENABLED, so a source can be parked without deleting it.
ALL_COLLECTORS = {
    "scd40": scd40,
    "open_meteo": open_meteo,
    "ultrahuman": ultrahuman,
    "aurn": aurn,
}
COLLECTORS = {n: m for n, m in ALL_COLLECTORS.items() if n in settings.COLLECTORS_ENABLED}
PARKED = [n for n in ALL_COLLECTORS if n not in COLLECTORS]
log = logging.getLogger("siot")


def _job_loop(name, module, db, stop):
    interval = settings.INTERVALS[name]
    while not stop.is_set():
        t0 = time.time()
        status, n, err = run_collector(name, module.collect, db)
        log.info("%-13s %-7s rows=%-5d %.1fs %s", name, status, n, time.time() - t0, err or "")
        # fixed cadence without drift; SCD40 blocks for its own window so waits ~0
        stop.wait(max(0.0, interval - (time.time() - t0)))


def _analysis_loop(db, stop):
    """Analysis runs on its own clock; it only ever reads what the collectors stored."""
    while not stop.is_set():
        t0 = time.time()
        try:
            d = analysis_human.run(db).get("decision")
            log.info("%-13s %-7s %s", "analysis", "ok",
                     f"{d['room']} {d['expression']} - {d['message']}" if d else "no rooms yet")
        except Exception as e:  # noqa: BLE001 - analysis must never kill collection
            log.warning("analysis failed: %s: %s", type(e).__name__, e)
            db.log_analysis_run(analysis_human.MODULE, t0, (time.time() - t0) * 1000,
                                "error", 0, f"{type(e).__name__}: {e}"[:300])
        stop.wait(max(0.0, settings.ANALYSIS_INTERVAL_S - (time.time() - t0)))


def _catch_up(db):
    """On start, pull whatever happened while this host was off.

    Every API here serves history, so a gap only costs you data if nobody asks for
    it. A normal poll looks back about two days, which covers an overnight sleep but
    not a weekend, so the first request after a gap is widened to match. The width
    comes from what is already stored rather than from a guess, and storage is
    idempotent, so an unnecessary catch-up is harmless.

    scd40 has no backfill and needs none: the node holds 48 h in RAM and uploads the
    backlog itself once this host answers again."""
    obs, _ = db.status_summary()
    newest = {}
    for r in obs:
        newest[r["source"]] = max(newest.get(r["source"], 0), r["last_ts"] or 0)
    now = time.time()
    for name, module in COLLECTORS.items():
        if not hasattr(module, "backfill"):
            continue
        last = newest.get(getattr(module, "SOURCE", name))
        gap_h = (now - last) / 3600 if last else None
        if gap_h is not None and gap_h <= 3:
            continue                                  # a normal poll already covers this
        days = 7 if gap_h is None else min(30, int(gap_h // 24) + 2)
        status, n, err = run_collector(name, lambda d, m=module, k=days: m.backfill(k, d), db)
        log.info("catch-up %-12s %-7s rows=%-6d (%s, %d day window) %s", name, status, n,
                 "no data yet" if gap_h is None else f"{gap_h:.1f} h gap", days, err or "")


def cmd_collect(args, db):
    if not args.no_catchup:
        _catch_up(db)
    stop = threading.Event()
    threads = [threading.Thread(target=_job_loop, args=(n, m, db, stop), name=n, daemon=True)
               for n, m in COLLECTORS.items() if not args.only or n in args.only]
    if not args.no_analysis:
        threads.append(threading.Thread(target=_analysis_loop, args=(db, stop),
                                        name="analysis", daemon=True))
    for t in threads:
        t.start()
    log.info("collecting into %s (Ctrl+C to stop)", db.path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        log.info("stopping")


def cmd_once(args, db):
    for name in args.names or COLLECTORS:
        t0 = time.time()
        status, n, err = run_collector(name, COLLECTORS[name].collect, db)
        print(f"{name:13s} {status:7s} rows={n:<5d} {time.time() - t0:5.1f}s {err or ''}")


def cmd_backfill(args, db):
    for name, module in COLLECTORS.items():
        if not hasattr(module, "backfill") or (args.only and name not in args.only):
            continue
        t0 = time.time()
        status, n, err = run_collector(name, lambda d, m=module: m.backfill(args.days, d), db)
        print(f"{name:13s} {status:7s} rows={n:<6d} {time.time() - t0:5.1f}s {err or ''}")


def _fmt(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%d %H:%M") if ts else "-"


def cmd_status(args, db):
    obs, runs = db.status_summary()
    now = time.time()
    print(f"DB: {db.path}\n\n{'source':17s}{'metric':26s}{'loc':8s}{'kind':9s}{'n':>7s}  {'first (UTC)':12s}{'last (UTC)':12s}")
    for r in obs:
        print(f"{r['source']:17s}{r['metric']:26s}{r['location']:8s}{r['kind']:9s}{r['n']:7d}  "
              f"{_fmt(r['first_ts']):12s}{_fmt(r['last_ts']):12s}")
    # staleness = age of the newest DATA from that source (a push node can fail while its job still runs)
    newest = {}
    for r in obs:
        newest[r["source"]] = max(newest.get(r["source"], 0), r["last_ts"])
    print(f"\n{'collector':14s}{'runs':>6s}{'ok':>6s}{'err':>5s}{'skip':>6s}{'avg ms':>8s}  {'last run':13s}{'newest data':12s}")
    for r in runs:
        mod = ALL_COLLECTORS.get(r["collector"])   # parked ones still have stored data
        last = newest.get(getattr(mod, "SOURCE", r["collector"]))
        stale = " STALE" if last and now - last > settings.STALE_AFTER_S.get(r["collector"], 3600) else ""
        print(f"{r['collector']:14s}{r['n']:6d}{r['ok']:6d}{r['err']:5d}{r['skipped']:6d}"
              f"{(r['avg_ms'] or 0):8.0f}  {_fmt(r['last_run']):13s}{_fmt(last):12s}{stale}")
    _parked_note()


def _when(text, tz):
    """Parse 14:00 (today), 2026-11-02 14:00, or now -> unix seconds."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(tz)
    if text in (None, "now"):
        return int(time.time())
    today = datetime.now(zone)
    for fmt, dated in (("%Y-%m-%d %H:%M", True), ("%H:%M", False)):
        try:
            got = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if not dated:
            got = got.replace(year=today.year, month=today.month, day=today.day)
        return int(got.replace(tzinfo=zone).timestamp())
    raise SystemExit("cannot read time " + repr(text) + "; use HH:MM or YYYY-MM-DD HH:MM")


def cmd_label(args, db):
    """Ground truth from the user. A handful of these in the first days are what turn
    the published defaults into this room's own measured numbers."""
    if args.last:
        end = int(time.time())
        start = end - args.last * 60
    else:
        start, end = _when(args.start, settings.USER_TZ), _when(args.end, settings.USER_TZ)
    if end <= start:
        raise SystemExit("that label ends before it starts")
    db.add_label(args.location, args.kind, args.value, start, end, args.note, args.people)
    who = f" ({args.people} people)" if args.people else ""
    print(f"labelled {args.location} {args.kind}={args.value}{who}  "
          f"{_fmt(start)} -> {_fmt(end)} UTC  ({(end - start) / 60:.0f} min)")


def cmd_params(args, db):
    """What the system currently believes about each room, and how sure it is."""
    from analysis.room_model import RoomModel, rooms
    for name in rooms(db):
        model = RoomModel(db, name)
        print()
        print(f"{name}  ({'learned' if model.learned else 'still on defaults'})")
        for key, value in model.value.items():
            err = model.stderr.get(key)
            print(f"  {key:16s}{value:9.3f} {('+/-' + format(err, '.3f')) if err else '':10s}"
                  f" n={model.n[key]:<4d} {model.basis[key]}")
        if model.volume_m3:
            print(f"  volume_m3       {model.volume_m3:9.1f}            inferred; check it with a tape measure")
        rows = db.decays(name, min_r2=0.9)
        if rows:
            vent = sum(1 for r in rows if r["inferred"] == "ventilated")
            print(f"  {len(rows)} usable decays: {len(rows) - vent} sealed, {vent} ventilated")


def cmd_analyse(args, db):
    out = analysis_human.run(db)
    print(f"scored {out['scored']} past forecasts | strictness {out['strictness']} "
          f"(recovery z={out['recovery_z']}) | quiet hours={out['quiet']}")
    for name, fit in out["fits"].items():
        extra = " ".join(f"{k}={v}" for k, v in fit.items()
                         if k in ("c_out", "lambda_sealed", "lambda_open",
                                  "source_ppm_h", "volume_m3"))
        print(f"  {name:10s} points={fit['points']:<6d} spikes={fit['spikes']:<4d} "
              f"decays={fit['decays']:<3d} plateaus={fit['plateaus']:<3d} {extra}")
    d = out.get("decision")
    print()
    print(f"pet: [{d['expression']}] {d['message']}   (the {d['room']}, {d['severity']})"
          if d else "no indoor data yet, so the pet has nothing to say")
    if args.associations:
        print()
        print("overnight CO2 vs body (exploratory):",
              analysis_human.night_associations(db))


def _parked_note():
    if PARKED:
        print()
        print(f"parked, not collecting: {', '.join(PARKED)}"
              "   (set COLLECTORS=... in .env to change)")


def cmd_preflight(args, db):
    """Check the things that silently break a flash or an overnight run.

    Each one has cost real debugging time: a token that drifted out of step with
    .env, an IP the laptop no longer has, a missing display library that stops the
    sketch compiling at all."""
    import re
    import socket
    sec = settings.BASE_DIR / "firmware" / "esp32_scd40" / "secrets.h"
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label:38s}{detail}")

    print(f"host: {db.path}")
    check("SCD40 backend is http", settings.SCD40_BACKEND == "http", settings.SCD40_BACKEND)
    check("INGEST_TOKEN set in .env", bool(os.getenv("INGEST_TOKEN")))
    if not sec.exists():
        check("firmware secrets.h exists", False, "copy secrets.example.h to secrets.h")
        return
    text = sec.read_text(encoding="utf-8")
    grab = lambda k: (re.search(r'#define\s+' + k + r'\s+"([^"]*)"', text) or [None, None])[1]  # noqa: E731
    node = re.search(r"#define\s+NODE\s+(\d+)", text)
    check("secrets.h NODE switch present", bool(node), f"NODE {node.group(1)}" if node else "")
    check("firmware token matches .env", grab("INGEST_TOKEN") == os.getenv("INGEST_TOKEN"))
    url = grab("INGEST_URL") or ""
    host = (re.search(r"http://([^:/]+)", url) or [None, None])[1]
    mine = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None)}
    check("ingest IP is one of ours", host in mine, f"{host}  (this laptop: {sorted(mine)})")
    libs = Path(os.path.expanduser("~/Documents/Arduino/libraries"))
    have = [d.name for d in libs.iterdir()] if libs.exists() else []
    needs_oled = "#define HAS_OLED    1" in text
    check("U8g2 installed (display nodes)", any("u8g2" in d.lower() for d in have) or not needs_oled)
    print()
    print(f"collecting: {', '.join(COLLECTORS)}")
    if PARKED:
        print(f"parked:     {', '.join(PARKED)}")
    print("ready to flash" if ok else "fix the FAIL lines above before flashing")


def cmd_offset(args, db):
    """Compare two sensors standing in the same air.

    Two SCD40s typically disagree by tens of ppm, which is within their spec but
    not within the precision this project needs: without this number a difference
    between rooms cannot be told apart from a difference between units. The median
    is used rather than the mean so a single disturbed minute cannot move it, and
    the first `--settle` minutes are dropped because a sensor that has just been
    carried into a different room is still equilibrating."""
    import statistics
    end = int(time.time())
    start = end - args.last * 60
    # The most recent co-location wins. Narrowing the window across every label in
    # turn, as this once did, leaves `start` on the newest and `end` on the oldest,
    # which is an empty range and reports no paired minutes at all.
    colocated = [lb for lb in db.labels(kind="occupancy") if lb["value"] == "colocated"]
    if colocated:
        last = colocated[-1]                      # db.labels() returns them in time order
        start, end = last["start_ts"], min(last["end_ts"], end)
    start += args.settle * 60
    with db.conn() as c:
        pairs = c.execute(
            "SELECT a.ts, a.value, b.value FROM observations a JOIN observations b "
            "ON a.ts=b.ts AND b.source='scd40' AND b.metric=a.metric AND b.location=? "
            "WHERE a.source='scd40' AND a.metric=? AND a.location=? AND a.ts BETWEEN ? AND ? "
            "ORDER BY a.ts", (args.b, args.metric, args.a, start, end)).fetchall()
    if len(pairs) < 10:
        print(f"only {len(pairs)} paired minutes after the settling window - leave them "
              f"side by side longer, or lower --settle")
        return
    diffs = sorted(r[1] - r[2] for r in pairs)
    med = statistics.median(diffs)
    q1, q3 = diffs[len(diffs) // 4], diffs[3 * len(diffs) // 4]
    print(f"{args.metric}: {args.a} minus {args.b}, {_fmt(pairs[0]['ts'])} -> {_fmt(pairs[-1]['ts'])} UTC")
    print(f"  n={len(diffs)} paired minutes")
    print(f"  median {med:+.1f}   IQR {q1:+.1f} to {q3:+.1f}   range {diffs[0]:+.1f} to {diffs[-1]:+.1f}")
    # Is the difference still travelling? Compare the first third of the window with
    # the last, in time order. A real offset is steady; a sensor that has just been
    # carried into a different room drifts in one direction while it equilibrates.
    ordered = [r[1] - r[2] for r in pairs]
    third = max(3, len(ordered) // 3)
    drift = statistics.median(ordered[-third:]) - statistics.median(ordered[:third])
    print(f"  drift across the window {drift:+.1f} (first third to last)")
    if q3 - q1 > 25:
        print("  the spread is wide: they are probably not yet in the same air")
    elif abs(drift) > 15:
        print("  still equilibrating: leave them together longer before trusting this")
    else:
        print(f"  usable: subtract {med:+.1f} from {args.a} to put it on the same scale as {args.b}")


def cmd_ventilate(args, db):
    """What tonight looks like under each option, and the cheapest moment to air out."""
    from analysis import ventilation
    plan = ventilation.plan_night(db, args.room, hours=args.hours)
    if "error" in plan:
        raise SystemExit(plan["error"])
    print(f"{args.room}: now {plan['start_ppm']} ppm, outdoor baseline {plan['c_out']} ppm")
    print(f"  one sleeper puts out {plan['source_ppm_h']} ppm/h into "
          f"{plan['volume_m3'] and round(plan['volume_m3'])} m3 ({plan['source_basis']})")
    print(f"  next {plan['hours']:.0f} h outdoors: {plan['t_out_forecast']} C, "
          f"PM2.5 {plan['pm25_forecast']}")
    print()
    print(f"  {'option':24s}{'rate/h':>9s}{'heads to':>10s}{'peak':>7s}"
          f"{'h>1400':>8s}{'heat':>7s}")
    for o in plan["options"]:
        if o["lambda_h"] is None:
            print(f"  {o['name']:24s}{'?':>9s}   {o['note']}")
            continue
        print(f"  {o['name']:24s}{o['lambda_h']:9.2f}{o['ceiling']:10d}{o['peak']:7d}"
              f"{o['hours_over'][1400]:8.1f}"
              f"{('-' + str(o['heat_cost_c']) + 'C') if o['heat_cost_c'] else '-':>7s}")
    print()
    print("  to hold the night below:  " + "   ".join(
        f"{t} ppm needs {v}/h" for t, v in plan["needed"].items()))

    purge = ventilation.plan_purge(db, args.room, horizon_h=args.horizon)
    b = purge.get("best")
    print()
    if b:
        print(f"  cheapest airing in the next {args.horizon:.0f} h "
              f"({purge['evaluated']} options tried):")
        print(f"    in {b['in_minutes']} min, for {b['duration_min']} min "
              f"-> clears {b['clears_ppm']} ppm, costs {b['heat_cost_c']} C "
              f"(outdoor {b['t_out']} C, PM2.5 {b['pm25']})")
        print(f"    using the open-window rate {purge['lambda_open']}/h ({purge['basis']})")
    else:
        print("  no airing in the next window would leave the room warm enough")


def cmd_dashboard(args, db):
    from dashboard.server import serve
    serve(db, COLLECTORS, args.port)


def cmd_simulate(args, db):
    sim = scd40.SimBackend()
    end = int(time.time()) // 60 * 60
    start = end - args.days * 86400
    result = scd40.CollectResult("scd40")
    for ts, samples in sim.series(start, args.days * 1440):
        result.observations += scd40.summarise(samples, ts + 60)
    db.save_result(result)
    print(f"wrote {len(result.observations)} simulated SCD40 rows to {db.path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="database path (default: settings.DB_PATH)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect"); p.add_argument("--only", nargs="*")
    p.add_argument("--no-analysis", action="store_true", help="collect only, skip the analysis loop")
    p.add_argument("--no-catchup", action="store_true", help="skip the gap backfill on start")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("once"); p.add_argument("names", nargs="*"); p.set_defaults(fn=cmd_once)
    p = sub.add_parser("backfill"); p.add_argument("--days", type=int, default=7)
    p.add_argument("--only", nargs="*"); p.set_defaults(fn=cmd_backfill)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("analyse"); p.add_argument("--associations", action="store_true")
    p.set_defaults(fn=cmd_analyse)
    p = sub.add_parser("label", help="record ground truth: label bedroom window open --last 60")
    p.add_argument("location"); p.add_argument("kind", choices=["window", "door", "occupancy", "cooking"])
    p.add_argument("value"); p.add_argument("--last", type=int, metavar="MIN")
    p.add_argument("--from", dest="start"); p.add_argument("--to", dest="end", default="now")
    p.add_argument("--note")
    p.add_argument("--people", type=int, help="how many, for an occupancy label")
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("params"); p.set_defaults(fn=cmd_params)
    p = sub.add_parser("ventilate", help="tonight under each option, and when to air out")
    p.add_argument("--room", default="bedroom"); p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--horizon", type=float, default=12.0); p.set_defaults(fn=cmd_ventilate)
    p = sub.add_parser("preflight"); p.set_defaults(fn=cmd_preflight)
    p = sub.add_parser("offset", help="compare two sensors standing in the same air")
    p.add_argument("--a", default="desk"); p.add_argument("--b", default="bedroom")
    p.add_argument("--metric", default="co2_ppm"); p.add_argument("--last", type=int, default=120)
    p.add_argument("--settle", type=int, default=20, help="minutes to drop after a move")
    p.set_defaults(fn=cmd_offset)
    p = sub.add_parser("dashboard"); p.add_argument("--port", type=int); p.set_defaults(fn=cmd_dashboard)
    p = sub.add_parser("simulate"); p.add_argument("--days", type=int, default=7); p.set_defaults(fn=cmd_simulate)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.fn(args, Database(args.db))


if __name__ == "__main__":
    main()
