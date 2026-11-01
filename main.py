"""
main.py

Starts the application, schedules data collection and analysis,
handles errors, and connects the analysis outputs to the pet controller.

Week 1 commands (collection only):
  python main.py collect            run all collectors forever (one thread each)
  python main.py once [name ...]    run collectors once and print the outcome
  python main.py backfill --days 7  pull history for sources that offer it
  python main.py status             row counts, freshness and reliability per source
  python main.py dashboard          monitoring UI at http://127.0.0.1:8050 (separate process)
  python main.py simulate --days 7 --db data/sim.db
                                    synthetic SCD40 week for developing analysis
"""

import argparse
import logging
import threading
import time
from datetime import datetime, timezone

from collectors import grid_carbon, open_meteo, scd40, thames_water, ultrahuman
from collectors.base import run_collector
from config import settings
from storage.database import Database

COLLECTORS = {
    "scd40": scd40,
    "open_meteo": open_meteo,
    "grid_carbon": grid_carbon,
    "ultrahuman": ultrahuman,
    "thames_water": thames_water,
}
log = logging.getLogger("siot")


def _job_loop(name, module, db, stop):
    interval = settings.INTERVALS[name]
    while not stop.is_set():
        t0 = time.time()
        status, n, err = run_collector(name, module.collect, db)
        log.info("%-13s %-7s rows=%-5d %.1fs %s", name, status, n, time.time() - t0, err or "")
        # fixed cadence without drift; SCD40 blocks for its own window so waits ~0
        stop.wait(max(0.0, interval - (time.time() - t0)))


def cmd_collect(args, db):
    stop = threading.Event()
    threads = [threading.Thread(target=_job_loop, args=(n, m, db, stop), name=n, daemon=True)
               for n, m in COLLECTORS.items() if not args.only or n in args.only]
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
        mod = COLLECTORS.get(r["collector"])
        last = newest.get(getattr(mod, "SOURCE", r["collector"]))
        stale = " STALE" if last and now - last > settings.STALE_AFTER_S.get(r["collector"], 3600) else ""
        print(f"{r['collector']:14s}{r['n']:6d}{r['ok']:6d}{r['err']:5d}{r['skipped']:6d}"
              f"{(r['avg_ms'] or 0):8.0f}  {_fmt(r['last_run']):13s}{_fmt(last):12s}{stale}")


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
    p = sub.add_parser("collect"); p.add_argument("--only", nargs="*"); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("once"); p.add_argument("names", nargs="*"); p.set_defaults(fn=cmd_once)
    p = sub.add_parser("backfill"); p.add_argument("--days", type=int, default=7)
    p.add_argument("--only", nargs="*"); p.set_defaults(fn=cmd_backfill)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("dashboard"); p.add_argument("--port", type=int); p.set_defaults(fn=cmd_dashboard)
    p = sub.add_parser("simulate"); p.add_argument("--days", type=int, default=7); p.set_defaults(fn=cmd_simulate)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.fn(args, Database(args.db))


if __name__ == "__main__":
    main()
