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
import json
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
    # One collector per database. A second one does not share the work, it
    # duplicates it: two ingest servers racing for the port, two analysis loops
    # interleaving their rewrites of room_params, and two copies of every API call.
    # See single_instance for why the lock is a socket and not a file.
    import single_instance

    lock = single_instance.acquire(db.path)
    if lock is None:
        print(f"a collector is already running for {db.path}")
        print(f"  held by {single_instance.describe(db.path)}")
        print("  stop that one first, or point this at another database with --db")
        raise SystemExit(1)
    single_instance.record(db.path, f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    code_at_start = single_instance.code_fingerprint()

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
        warned = False
        while True:
            time.sleep(1)
            # Say so, once, when the code on disk has moved past what is running.
            # Nothing can be done about it from in here - the modules are already
            # imported - but a line in the log is the difference between "the fix
            # did not work" and "the fix is not running".
            if not warned and single_instance.stale_by(code_at_start) > 0:
                warned = True
                log.warning("the project's code has changed since this collector "
                            "started - it is still running the OLD version. "
                            "Restart it (Ctrl+C, then `python main.py collect`) "
                            "or anything you just fixed is not in effect here.")
    except KeyboardInterrupt:
        stop.set()
        log.info("stopping")
    finally:
        single_instance.release(lock)


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


def cmd_volume(args, db):
    """Record a volume that was measured rather than inferred.

    This is the only ground truth the system has. Everything else it believes about a
    room it worked out from the same CO2 trace it is trying to explain, so a number
    from a tape measure is the one thing capable of catching a self-consistent error.
    It is stored under its own name so that re-running the analysis cannot overwrite
    it."""
    from analysis.room_model import RoomModel, volume_check
    if not 2.0 <= args.m3 <= 2000.0:
        raise SystemExit(f"{args.m3} m3 is not a room")
    db.set_param(args.room, "volume_measured_m3", args.m3, None, 1, "m3",
                 {"from": "tape measure"})
    src = db.params(args.room).get("source_ppm_h")
    print(f"{args.room}: volume measured at {args.m3} m3")
    chk = volume_check(db, args.room, float(src["value"]), int(src["n"]),
                       src["stderr"]) if src else None
    print("  " + chk["note"] if chk else
          "  no source term fitted yet, so nothing to check it against")


def cmd_nodes(args, db):
    """Each sensor node: is it up, has it rebooted, and why.

    A node that reboots and one that loses WiFi both simply stop arriving, which
    is why "it keeps stopping" went undiagnosed for a whole evening. The node's
    own uptime separates them - back lower than it went away means the board
    restarted - and the ESP32's reset reason says which kind of restart."""
    import time as _t

    now = int(_t.time())
    with db.conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM device_status ORDER BY last_seen DESC")]
    if not rows:
        print("no node has ever checked in")
        return
    import single_instance
    note = single_instance.describe(db.path)
    if any(r.get("up_s") is None for r in rows):
        print("Note: nodes reporting no uptime are either on older firmware, or "
              "the collector storing their uploads is.")
        print(f"      the collector holding the lock says: {note}")
        print()
    for r in rows:
        age = (now - r["last_seen"]) // 60
        up = r.get("up_s")
        state = "LIVE" if age < 3 else f"silent {age} min"
        print(f"{r['device']:10s} {state:16s} accepted={r['accepted_total']} "
              f"rejected={r['rejected_total']}")
        if up is None:
            print("           (older firmware: no uptime reported, so a reboot "
                  "cannot be told from a dropout)")
            continue
        print(f"           uptime {up // 3600}h{up % 3600 // 60:02d}m   "
              f"reboots seen {r.get('reboots') or 0}   "
              f"last reset: {r.get('reset_reason') or '?'}")
        if r.get("heap") is not None:
            print(f"           heap {r['heap']} free, largest block "
                  f"{r.get('heap_max_block')}")
        reason = (r.get("reset_reason") or "").lower()
        if reason == "brownout":
            print("           -> BROWNOUT: the supply sagged. Thicker cable, a "
                  "2 A adapter, or power the PMS straight from the adapter.")
        elif reason in ("panic", "task_wdt", "int_wdt", "wdt"):
            print("           -> the FIRMWARE crashed or hung, not the power. "
                  "The serial log at the moment of reset has the backtrace.")


def cmd_particles(args, db):
    """Is the particle counter working, and are its numbers physically possible?

    "Is there data" is the easy half. The PMS5003 can also sit on 3V3 and report
    plausible-looking numbers from an airflow too weak to be a sample, which no
    freshness check can catch - so this tests the internal consistency the sensor
    cannot fake: the three masses must be ordered, the six counts must fall with
    size because they are cumulative, and a fan that is actually running produces
    variance rather than a flat line.
    """
    import time as _t

    from analysis import features
    from analysis.particles import PM_SIZES

    now = int(_t.time())
    from analysis.room_model import rooms
    locs = rooms(db)

    found = False
    for loc in locs:
        series = {m: features.series(db, m, loc).dropna() for m, _u in PM_SIZES}
        if not any(s_.size for s_ in series.values()):
            continue
        found = True
        print(f"\n{loc}")
        newest = max((s_.index[-1] for s_ in series.values() if s_.size), default=None)
        age = (now - int(newest.timestamp())) // 60 if newest is not None else None
        print(f"  newest sample {age} min ago")

        # 1. masses must be ordered: PM10 counts everything PM2.5 counts, and more
        bad_order = 0
        pm1, pm25, pm10 = (series["pm1_0"], series["pm2_5"], series["pm10"])
        joined = pm1.to_frame("a").join(pm25.to_frame("b"), how="inner") \
                    .join(pm10.to_frame("c"), how="inner").dropna()
        if joined.shape[0]:
            bad_order = int(((joined["a"] > joined["b"] + 0.01)
                             | (joined["b"] > joined["c"] + 0.01)).sum())
            print(f"  mass ordering PM1<=PM2.5<=PM10: "
                  f"{joined.shape[0] - bad_order}/{joined.shape[0]} ok"
                  + ("" if not bad_order else f"  <-- {bad_order} IMPOSSIBLE"))

        # 2. the counter must actually vary; a flat line is a fan that is not moving
        for name, s_ in (("PM2.5", pm25),):
            if s_.size >= 12:
                spread = float(s_.max() - s_.min())
                print(f"  {name} range over the record: {spread:.1f} ug/m3"
                      + ("" if spread > 1.0 else
                         "  <-- FLAT: check VCC is on 5V, not 3V3"))

        # 3. cumulative counts must fall with size
        bins = ["pc0_3", "pc0_5", "pc1_0", "pc2_5", "pc5_0", "pc10_0"]
        cnt = {b: features.series(db, b, loc).dropna() for b in bins}
        have = [b for b in bins if cnt[b].size]
        if len(have) >= 2:
            df = None
            for b in have:
                f_ = cnt[b].to_frame(b)
                df = f_ if df is None else df.join(f_, how="inner")
            df = df.dropna()
            bad = 0
            for a, b in zip(have, have[1:]):
                bad += int((df[a] < df[b]).sum())
            print(f"  count bins fall with size: {len(df) * (len(have) - 1) - bad}"
                  f"/{len(df) * (len(have) - 1)} ok"
                  + ("" if not bad else f"  <-- {bad} IMPOSSIBLE"))

        # 4. duty cycle, measured only over the span the counter has existed.
        # Comparing its sample count against the whole CO2 record reported 0.00
        # for a sensor that had been running for six minutes, which looks like a
        # fault and is just arithmetic over the wrong window.
        if pm25.size >= 2:
            co2 = features.series(db, "co2_ppm", loc).dropna()
            since = co2.loc[pm25.index[0]:pm25.index[-1]]
            if since.size:
                share = pm25.size / since.size
                print(f"  particle samples per CO2 sample since the counter "
                      f"started: {share:.2f} (expect ~1.0 on the 5-minute grid)")
            gaps = pm25.index.to_series().diff().dt.total_seconds().dropna()
            if gaps.size:
                print(f"  interval between samples: {gaps.median() / 60:.0f} min "
                      f"(expect 5)")

        dep = db.params(loc).get("deposition_k_pm2_5")
        if dep and dep["value"]:
            print(f"  deposition k = {dep['value']:.3f}/h from {dep['n']} event(s)")
        else:
            print("  no deposition rate yet: needs an indoor source that then stops "
                  "(cook something with the door shut)")

    if not found:
        print("No particle data from any room.")
        print()
        print("If the bedroom node is flashed and wired, check in this order:")
        print("  1. Is it reporting at all?   python main.py status")
        print("  2. Serial at 115200 should show, within a minute of boot:")
        print("       [pms] PMS5003 on UART2, RX=GPIO25 TX=GPIO26, 9600 baud")
        print("     If that line is missing, HAS_PMS is 0 - check NODE is 2.")
        print("  3. After the first 30 s warm-up it should show a reading. If it")
        print("     says 'only 0 good frame(s)', TXD is not on GPIO25 or VCC is")
        print("     not on 5V. Those are the two failures that look identical.")


def cmd_limits(args, db):
    """What this system does not know, measured from its own record.

    Written as a command rather than a document because a document goes stale the
    day after it is written and this has already been wrong twice. Everything below
    is read from the database at the moment it is printed, so it cannot drift from
    what the code actually did. What a reader needs is the line between what was
    measured in THIS flat and what is a property of the method.
    """
    import json as _json

    from analysis import features
    from analysis.room_model import (RoomModel, colocation_check,
                                     measured_volume, rooms)

    print("WHAT THIS SYSTEM DOES NOT KNOW")
    print()

    names = rooms(db)
    print("Record length")
    for name in names:
        s_ = features.series(db, "co2_ppm", name).dropna()
        if s_.empty:
            continue
        span = (s_.index[-1] - s_.index[0]).total_seconds() / 86400
        gaps = int((s_.index.to_series().diff().dt.total_seconds() > 1800).sum())
        print(f"  {name:9s} {span:.1f} days, {s_.size} samples on the 5-min grid, "
              f"{gaps} gap(s) over 30 min")
    print("  Every rate here rests on days, not seasons. Nothing in it has seen")
    print("  a cold snap, a gale, or a summer, and the weather terms say so.")
    print()

    print("One flat, one occupant, two sensors")
    print("  Rooms are hard-coded as 'bedroom' and 'desk' in analysis_human,")
    print("  figures and night_associations, so a third room is collected and")
    print("  then silently not analysed.")
    for name in names:
        row = db.params(name).get("typical_people")
        if row and row["value"]:
            print(f"  {name}: typical occupancy {int(row['value'])}. The source term is")
            print(f"    stored per person but has only ever been fitted at that count.")
    print("  Constants chosen against this flat and not validated elsewhere:")
    print("    DEPARTURE_OFFSET_MIN=20 (its mixing time), SPIKE_IMPOSSIBLE_PPM=400")
    print("    (its volume and occupancy), COMFORT_FLOOR_C=16.")
    print()

    print("Ground truth: what was measured rather than inferred")
    any_truth = False
    for name in names:
        v = measured_volume(db, name)
        if v:
            any_truth = True
            print(f"  {name}: volume {v} m3, by tape measure. Everything else this")
            print(f"    system believes about {name} it worked out from the same CO2")
            print(f"    trace it is trying to explain.")
    if not any_truth:
        print("  None. Every parameter is self-derived, so a self-consistent error")
        print("  cannot be detected. `main.py volume <room> <m3>` fixes that.")
    print()

    chk = colocation_check(db)
    if chk:
        print("Do the two sensors agree about outdoor air?")
        print(f"  {chk['note']}")
        print()

    print("Fits that were tested and failed")
    failed = 0
    for name in names:
        for key, what in (("lambda_open_k", "open-window wind law"),
                          ("lambda_wind_c", "infiltration weather law")):
            row = db.params(name).get(key)
            if not row or not row["detail"]:
                continue
            d = _json.loads(row["detail"])
            if d.get("refuted"):
                failed += 1
                print(f"  {name} {what}: {d['refuted'][:160]}")
    from analysis import estimators
    for name in names:
        try:
            sp = estimators.by_spectrum(db, name)
        except Exception:                            # noqa: BLE001
            continue
        if sp.get("lambda_h") is None and sp.get("slope_dec_per_dec") is not None:
            failed += 1
            print(f"  {name} spectral estimator: roll-off "
                  f"{sp['slope_dec_per_dec']} dec/dec rejects first order, so it "
                  f"returns no rate.")
    if not failed:
        print("  None recorded.")
    print()

    print("The forecast")
    sk = db.forecast_skill()
    if sk:
        best = max(sk, key=lambda r: r["skill"])
        print(f"  Scored against 'assume no change' on {sum(r['n'] for r in sk)} "
              f"predictions. Best skill {best['skill']:+.0%} "
              f"({best['location']}, {best['horizon_s'] // 60} min, {best['basis']}).")
        if best["skill"] < 0:
            print("  The model loses to persistence at every room, horizon and basis.")
            print("  It is not an extrapolator and should not be read as one. What")
            print("  persistence cannot do is compare options - 'what if I open the")
            print("  window' has no persistence answer - and that is what it is for.")
    else:
        print("  No scored forecasts with a baseline yet.")
    print()

    print("Known and not fixed")
    print("  - Per-fit stderrs are corrected for lag-1 autocorrelation only; where")
    print("    residuals are structural rather than noisy that is still a floor.")
    print("  - The purge cost adds ppm, degrees and ug.h/m3 with hand-picked")
    print("    weights (ventilation.py), and no sensitivity sweep has been run.")
    print("  - decay_events rows are deleted only inside the refit window, so a")
    print("    longer window can still mix two eras of c_out.")
    print("  - Episode verdicts: the accuracy metric exists but has n=1.")


def cmd_backup(args, db):
    """Take a consistent copy. Verified by row count, not by the file appearing."""
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    target = args.path or str(Path(db.path).parent / f"backup-{stamp}.db")
    out = db.backup_to(target)
    print(f"{out['path']}: {out['rows']} observations "
          f"({'complete' if out['complete'] else 'INCOMPLETE'} "
          f"against {out['live_rows']} live)")
    if not out["complete"]:
        raise SystemExit("the copy is short - do not trust it")


def cmd_skill(args, db):
    """Forecast error against the forecast that assumes nothing changes.

    Compared with what? is the first question any error figure has to answer, and
    for a long time this project could not. Persistence is the right rival: it uses
    the same information at the same instant and costs nothing, so beating it is
    the minimum evidence that the physics is doing work."""
    rows = db.forecast_skill(days=args.days)
    if not rows:
        print("no scored forecasts with a persistence baseline yet")
        return
    print(f"{'room':9s}{'horizon':>8s}  {'basis':8s}{'n':>5s}{'model':>9s}"
          f"{'persist':>9s}{'skill':>8s}{'bias':>9s}")
    for r in rows:
        print(f"{r['location']:9s}{r['horizon_s'] // 60:6d}m  {r['basis']:8s}"
              f"{r['n']:5d}{r['mae']:9.1f}{r['naive_mae']:9.1f}"
              f"{r['skill']:+8.0%}{r['bias']:+9.1f}")
    worst = min(rows, key=lambda r: r["skill"])
    best = max(rows, key=lambda r: r["skill"])
    print()
    if best["skill"] < 0:
        print("The model loses to persistence everywhere. Read the bias column for "
              "why: a large bias is a wrong room, a small one with a large error is "
              "variance the model is adding rather than removing.")
        print(f"Best it manages is {best['skill']:+.0%} ({best['location']} "
              f"{best['horizon_s'] // 60} min, {best['basis']}); "
              f"worst {worst['skill']:+.0%}.")
        print("Persistence cannot answer 'what if I open the window', which is what "
              "this model is actually for. Extrapolation is not its job and it "
              "should not be sold as one.")
    else:
        print(f"Best {best['skill']:+.0%}, worst {worst['skill']:+.0%}.")


def cmd_params(args, db):
    """What the system currently believes about each room, and how sure it is."""
    from analysis.room_model import RoomModel, colocation_check, rooms

    # Before any per-room numbers: do the two sensors even agree about outdoor air?
    # If they do not, the rooms are not comparable and nothing below is.
    chk = colocation_check(db)
    if chk:
        flag = "ok" if chk["explained_by_sensors"] else "MISMATCH"
        print(f"\nsensor cross-check  {flag}")
        print(f"  {chk['note']}")

    for name in rooms(db):
        model = RoomModel(db, name)
        print()
        print(f"{name}  ({'learned' if model.learned else 'still on defaults'})")
        for key, value in model.value.items():
            err = model.stderr.get(key)
            print(f"  {key:16s}{value:9.3f} {('+/-' + format(err, '.3f')) if err else '':10s}"
                  f" n={model.n[key]:<4d} {model.basis[key]}")
        if model.volume_m3:
            print(f"  volume_m3       {model.volume_m3:9.1f}            {model.volume_basis}"
                  + ("" if model.volume_basis == "measured"
                     else "; check it with a tape measure"))
        if model.volume_basis == "measured" and model.volume_inferred_m3:
            from analysis.room_model import volume_check
            src = db.params(name).get("source_ppm_h")
            chk = volume_check(db, name, float(src["value"]), int(src["n"]),
                                   src["stderr"]) if src else None
            if chk:
                print(f"  volume            cross-check: {chk['note']}")
        rows = db.decays(name, min_r2=0.9)
        if rows:
            kinds = {}
            for r in rows:
                kinds[r["inferred"] or "unclassified"] = kinds.get(r["inferred"] or "unclassified", 0) + 1
            shown = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
            print(f"  {len(rows)} usable decays: {shown}")
            if kinds.get("coupled"):
                print(f"  {'':18s}'coupled' settled on house air, not outdoor air, so they "
                      f"measure a door rather than the envelope and are kept out of lambda_sealed")

        # lambda_sealed is measured twice over, by a room emptying and by the same
        # room filling. They share no assumption but the mass balance, so how far
        # apart they land is the best single check the system has on itself.
        stored = db.params(name).get("lambda_sealed")
        detail = json.loads(stored["detail"]) if stored and stored["detail"] else {}
        if detail.get("agreement"):
            flag = "DISAGREE - " if detail.get("disagree") else "cross-check: "
            print(f"  lambda_sealed     {flag}{detail['agreement']}")
            print(f"  {'':18s}{detail['decays']} decays + {detail['rises']} rises"
                  + (f", {detail['skipped_rises']} rise(s) too straight to read a "
                     f"rate off" if detail.get("skipped_rises") else ""))

        # Particles, where the counter has produced anything. This is the only
        # route to lambda that shares no arithmetic with the CO2 path, so it is
        # shown next to it rather than tucked away.
        dep = db.params(name).get("deposition_k_pm2_5")
        if dep and dep["value"]:
            d = json.loads(dep["detail"]) if dep["detail"] else {}
            err = f" +/-{dep['stderr']:.3f}" if dep["stderr"] else ""
            print(f"  deposition_k      {dep['value']:9.3f}{err} 1/h (PM2.5), "
                  f"n={dep['n']}")
            if d.get("ordering"):
                print(f"  {'':18s}{d['ordering']}")
            lam = model.value["lambda_sealed"]
            print(f"  {'':18s}particles leave at {lam + dep['value']:.2f}/h against "
                  f"the air's {lam:.2f}/h - the difference is settling, measured "
                  f"by a sensor that shares no baseline with the CO2")
        pen = db.params(name).get("penetration_pm2_5")
        if pen and pen["value"]:
            pd_ = json.loads(pen["detail"]) if pen["detail"] else {}
            print(f"  penetration       {pen['value']:9.3f}            of outdoor "
                  f"PM2.5 survives the way in")
            if pd_.get("note"):
                print(f"  {'':18s}{pd_['note']}")

        # The weather law gets the same treatment as the wind law: shown whether it
        # passed or failed, because a refutation is a result and a silent parameter
        # nobody reads is not.
        wrow = db.params(name).get("lambda_wind_c")
        wd = json.loads(wrow["detail"]) if wrow and wrow["detail"] else None
        if wd and wd.get("refuted"):
            print(f"  weather law       REFUTED: lambda^2 = a + b|dT| + c*v^2 over "
                  f"n={wd['n']}, r2_adj={wd.get('r2_adj') and round(wd['r2_adj'], 2)}")
            for reason in wd["refuted"].split("; "):
                print(f"  {'':18s}- {reason}")
        elif wd and wd.get("usable"):
            print(f"  weather law       lambda^2 = {wd['a']:.3f} + {wd['b']:.4f}|dT| "
                  f"+ {wd['c']:.4f}v^2   r2_adj={wd['r2_adj']:.2f} n={wd['n']}")

        # An open window is not one number - what it delivers scales with wind - so
        # the pooled lambda_open above is only the fallback. Show the law and what it
        # predicts across the wind speeds this room was actually aired in.
        law = model.open_wind_fit
        if law and law.get("refuted"):
            print(f"  open-window law   REFUTED by this room's own openings")
            print(f"  {'':18s}{law['refuted']}")
            print(f"  {'':18s}tried lambda = {law['k']:.3f} x wind, r2={law['r2']:.4f}, "
                  f"n={law['n']}; using the pooled median instead")
            if law.get("mixed_note"):
                print(f"  {'':18s}{law['mixed_note']}")
        elif law and "k" in law:
            lo, hi = law["wind_range"]
            print(f"  open-window law   lambda = {law['k']:.3f} +/-{law['stderr']:.3f} "
                  f"x wind   r2={law['r2']:.4f}  n={law['n']}")
            print(f"  {'':18s}measured {lo}-{hi} m/s, "
                  f"lambda {law['lambda_range'][0]}-{law['lambda_range'][1]}/h "
                  f"({law['spread']}x)")
            for v in (lo, (lo + hi) / 2, hi):
                print(f"  {'':18s}wind {v:4.1f} m/s -> {model.lambda_open_at(v)[0]:5.2f}/h")
            if law.get("provisional"):
                print(f"  {'':18s}provisional: {law['provisional']}")
            if law.get("mixed_note"):
                print(f"  {'':18s}{law['mixed_note']}")
        else:
            print("  open-window law   not yet: needs two labelled window openings "
                  "at different wind speeds")


def cmd_analyse(args, db):
    out = analysis_human.run(db)
    if out.get("skipped"):
        print(out["skipped"] + " - nothing done here")
        return
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

    # Both pollutants, which move in opposite directions through the same opening.
    # Until the particle counter arrived this half did not exist and the CO2 half
    # was presented as if it were the whole decision.
    try:
        from analysis.particles import air_trade
        trade = air_trade(db, args.room, minutes=60)
    except Exception:                                   # noqa: BLE001
        trade = None
    if trade and not trade.get("error"):
        print()
        print(f"  over the next hour, both pollutants "
              f"(indoor PM2.5 {trade['pm_now']}, outdoor {trade['pm_outdoor']} "
              f"from {trade['outdoor_source']}):")
        print(f"  {'option':24s}{'CO2 ends':>10s}{'change':>8s}"
              f"{'PM ends':>9s}{'change':>8s}{'PM dose':>9s}")
        for r in trade["options"]:
            if r.get("unmeasured"):
                continue
            print(f"  {r['option']:24s}{r['co2_end']:10d}{r['co2_change']:+8d}"
                  f"{r['pm_end']:9.1f}{r['pm_change']:+8.1f}{r['pm_dose_ug_h']:9.1f}")
        shut = next((r for r in trade["options"] if "shut" in r["option"]), None)
        wide = next((r for r in trade["options"] if "wide" in r["option"]), None)
        if shut and wide:
            dpm = wide["pm_dose_ug_h"] - shut["pm_dose_ug_h"]
            dco2 = shut["co2_end"] - wide["co2_end"]
            if dpm <= 1.0:
                print(f"    opening costs nothing in particles today "
                      f"({dpm:+.1f} ug.h/m3) and saves {dco2} ppm - the trade is "
                      f"one-sided, so open it")
            else:
                print(f"    opening saves {dco2} ppm of CO2 and costs "
                      f"{dpm:+.1f} ug.h/m3 of PM2.5 - that is the trade, and it is "
                      f"measured rather than assumed")
        print(f"    using k={trade['k_h']}/h and penetration {trade['penetration_shut']} "
              f"measured in this room")

    purge = ventilation.plan_purge(db, args.room, horizon_h=args.horizon)
    b = purge.get("best")
    print()

    def _when(mins):
        return time.strftime("%H:%M", time.localtime(time.time() + mins * 60))

    if b:
        print(f"  cheapest airing in the next {args.horizon:.0f} h "
              f"({purge['evaluated']} options tried, "
              f"{purge.get('rejected_too_cold', 0)} rejected as too cold):")
        print(f"    {_when(b['in_minutes'])} (in {b['in_minutes']} min) for "
              f"{b['duration_min']} min")
        print(f"      clears {b['clears_ppm']} ppm | costs {b['heat_cost_c']} C | "
              f"lets in {b['intake_ug_h']} ug.h/m3 of PM2.5")
        print(f"      conditions then: {b['t_out']} C out, PM2.5 {b['pm25']}, "
              f"rate {b['lambda_open']}/h ({b['basis']})")

        # What actually decided it. Three terms in different units, combined with
        # weights that are an opinion; naming the dominant one stops the number
        # being read as more objective than it is.
        t = purge.get("terms")
        if t:
            print(f"      the choice turns on {t['dominant']} "
                  f"(CO2 {t['co2_benefit']:+.2f}, heat {t['heat_cost']:+.2f}, "
                  f"PM {t['pm_cost']:+.2f}, total {t['total']:+.2f})")

        # How firm it is. This is the part that was missing while the advice
        # appeared to change its mind: dozens of candidates sit within a hair of
        # each other and the winner hops between them as the forecast shifts.
        if purge.get("timing"):
            rel = purge.get("margin_rel")
            extra = f", margin {rel:.0%}" if rel is not None else ""
            print(f"    timing: {purge['timing']} "
                  f"({purge.get('stability', 0):.0%} of cost weightings agree{extra})")
            print(f"      {purge['timing_note']}")
        if purge.get("margin") is not None:
            print(f"      next materially different option is "
                  f"{_when(purge['runner_up_in_minutes'])}, worse by "
                  f"{purge['margin']:.2f}")
        for a in (purge.get("alternatives") or []):
            print(f"    also: {_when(a['in_minutes'])} for {a['duration_min']} min "
                  f"-> {a['clears_ppm']} ppm, {a['heat_cost_c']} C "
                  f"(cost {a['cost']:+.2f})")
    elif purge.get("all_too_cold"):
        a = purge["all_too_cold"]
        print(f"  nothing worth doing in the next {args.horizon:.0f} h:")
        print(f"    {a['note']}")
        print(f"    if you will accept {a['best_case_c']} C, that is "
              f"{a['duration_min']} min at {_when(a['in_minutes'])}.")
    else:
        print("  no airing options could be scored (no outdoor forecast?)")


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
    p = sub.add_parser("nodes", help="node uptime, reboots and why")
    p.set_defaults(fn=cmd_nodes)
    p = sub.add_parser("particles", help="is the particle counter working")
    p.set_defaults(fn=cmd_particles)
    p = sub.add_parser("limits", help="what this system does not know")
    p.set_defaults(fn=cmd_limits)
    p = sub.add_parser("backup", help="consistent copy of the database")
    p.add_argument("path", nargs="?"); p.set_defaults(fn=cmd_backup)
    p = sub.add_parser("skill", help="forecast error vs 'assume no change'")
    p.add_argument("--days", type=int, default=3650); p.set_defaults(fn=cmd_skill)
    p = sub.add_parser("volume", help="record a tape-measured room volume: volume bedroom 18")
    p.add_argument("room"); p.add_argument("m3", type=float)
    p.set_defaults(fn=cmd_volume)
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
