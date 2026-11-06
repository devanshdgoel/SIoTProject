"""Offline tests: parsers and storage behaviour, no network or hardware needed."""

import pytest

from collectors import open_meteo, scd40, ultrahuman
from collectors.base import CollectorError, CollectResult, NotConfigured, Observation, run_collector
from storage.database import Database

NOW = 1_793_313_600  # 2026-10-29 22:40 UTC


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.db")


def test_open_meteo_splits_modelled_and_forecast():
    t0 = NOW - NOW % 3600
    payload = {"hourly": {"time": [t0 - 3600, t0, t0 + 3600, t0 + 30 * 3600],
                          "temperature_2m": [10.0, 11.0, 12.0, 13.0]}}
    obs = open_meteo._parse(payload, {"temperature_2m": ("outdoor_temp_c", "C")}, "home", NOW)
    kinds = {o.ts - t0: (o.kind, o.issued_at) for o in obs}
    assert kinds == {-3600: ("modelled", 0), 0: ("modelled", 0), 3600: ("forecast", t0)}  # +30h beyond horizon


def test_ultrahuman_parse_day():
    payload = {"data": {"metrics": {"2026-10-29": [
        {"type": "hr", "object": {"values": [{"value": 60, "timestamp": 1000}, {"value": None, "timestamp": 1060}]}},
        {"type": "recovery_index", "object": {"value": 57, "day_start_timestamp": 900}},
        {"type": "sleep", "object": {"bedtime_start": 100, "bedtime_end": 500,
                                     "total_sleep": {"seconds": 300}, "time_in_bed": {"minutes": 7},
                                     "sleep_stages": [{"type": "deep_sleep", "stage_time": 60}],
                                     "sleep_graph": {"data": [{"start": 100, "end": 200, "type": "awake"}]}}}]}}}
    r = ultrahuman.parse_day("2026-10-29", payload, now=NOW)
    assert [(o.metric, o.ts, o.value) for o in r.observations] == [
        ("hr_bpm", 1000, 60.0), ("daily_recovery_index", 900, 57.0)]
    s = r.sleep_sessions[0]
    assert (s["day"], s["total_sleep_s"], s["deep_s"], s["time_in_bed_s"]) == ("2026-10-29", 300, 60, 420)




def test_scd40_summarise_filters_and_flags():
    good = [(800 + i, 21.0, 50.0) for i in range(12)]
    obs = {o.metric: o for o in scd40.summarise(good + [(0, 21, 50)], NOW)}   # CO2=0 is a sensor fault
    assert obs["co2_ppm"].quality == "ok" and abs(obs["co2_ppm"].value - 805.5) < 0.01
    assert obs["co2_max_ppm"].value == 811
    partial = {o.metric: o for o in scd40.summarise(good[:3], NOW)}
    assert partial["co2_ppm"].quality == "partial"
    with pytest.raises(CollectorError):
        scd40.summarise([(0, 0, 0)], NOW)


def test_sim_backend_shows_overnight_buildup():
    sim = scd40.SimBackend()
    values = {h: sum(s[0] for s in sim.window_at(NOW - NOW % 86400 + h * 3600, 60)) / 12 for h in (6, 14)}
    assert values[6] > values[14] + 300          # occupied closed room at 06:00 vs empty at 14:00


def test_storage_is_idempotent_and_keeps_forecast_issues(db):
    def result(value, issued):
        r = CollectResult("x")
        r.observations.append(Observation("x", "m", "home", 1000, value, "forecast" if issued else "modelled",
                                          issued_at=issued))
        return r
    db.save_result(result(1.0, 0))
    db.save_result(result(2.0, 0))             # revision of the same hour overwrites
    db.save_result(result(3.0, 500))           # forecast issued at a different time is a separate row
    db.save_result(result(4.0, 600))
    df = db.read_df("m")
    assert len(df) == 3 and df[df.kind == "modelled"].value.iloc[0] == 2.0
    assert db.latest("m")["value"] == 2.0      # latest() ignores forecasts


def test_run_collector_logs_ok_skipped_and_error(db):
    def ok(_):
        r = CollectResult("x"); r.observations.append(Observation("x", "m", "l", 1, 1.0, "measured")); return r
    def skip(_): raise NotConfigured("no creds")
    def boom(_): raise RuntimeError("down")
    assert run_collector("a", ok, db)[0] == "ok"
    assert run_collector("b", skip, db)[0] == "skipped"
    status, _, err = run_collector("c", boom, db)
    assert status == "error" and "down" in err
    _, runs = db.status_summary()
    assert {r["collector"]: (r["ok"], r["err"], r["skipped"]) for r in runs} == {
        "a": (1, 0, 0), "b": (0, 0, 1), "c": (0, 1, 0)}


def test_http_ingest_roundtrip_auth_and_validation(db):
    import json, time, requests
    b = scd40.HttpBackend(db, port=0, token="secret")
    url = f"http://127.0.0.1:{b.port}/ingest"
    now = int(time.time()) // 60 * 60
    good = {"ts": now, "co2": 900, "co2_max": 950, "t": 21.5, "rh": 48, "n": 12}
    batch = {"readings": [good, {**good, "ts": now - 60, "n": 3},
                          {**good, "ts": now - 10 * 86400},          # too old
                          {**good, "co2": 0}]}                        # sensor fault
    assert requests.post(url, json=batch).status_code == 401
    assert requests.post(url, headers={"X-Auth-Token": "wrong"}, json=batch).status_code == 401
    assert requests.post(url, headers={"X-Auth-Token": "secret"}, data="not json").status_code == 400
    r = requests.post(url, headers={"X-Auth-Token": "secret"}, json=batch)
    assert r.status_code == 200 and r.json()["accepted"] == 2
    rows = b.drain()
    assert len(rows) == 2 and b.drain() == []
    obs = [o for row in rows for o in scd40.reading_to_obs(*row)]
    assert {o.quality for o in obs if o.ts == now - 60} == {"partial"}
    b.server.shutdown()


def test_http_ingest_keeps_rooms_separate(db):
    import time, requests
    b = scd40.HttpBackend(db, port=0, token="secret")
    url, hdr = f"http://127.0.0.1:{b.port}/ingest", {"X-Auth-Token": "secret"}
    now = int(time.time()) // 60 * 60
    r = {"ts": now, "co2": 700, "co2_max": 720, "t": 20.0, "rh": 50, "n": 12}
    assert requests.post(url, headers=hdr, json={"room": "bedroom", "readings": [r]}).json()["accepted"] == 1
    assert requests.post(url, headers=hdr, json={"readings": [r]}).json()["accepted"] == 1     # no room -> default
    assert requests.post(url, headers=hdr, json={"room": "Bad Room!", "readings": [r]}).status_code == 400
    locs = {scd40.reading_to_obs(*row)[0].location for row in b.drain()}
    assert locs == {"bedroom", "desk"}
    b.server.shutdown()


def test_ingest_reply_carries_pet_display_to_the_node(db):
    import time, requests
    b = scd40.HttpBackend(db, port=0, token="secret")
    url, hdr = f"http://127.0.0.1:{b.port}/ingest", {"X-Auth-Token": "secret"}
    now = int(time.time()) // 60 * 60
    r = {"ts": now, "co2": 1200, "co2_max": 1250, "t": 21.0, "rh": 50, "n": 12}
    assert requests.post(url, headers=hdr, json={"readings": [r]}).json()["display"] is None
    db.set_display("concerned", 'open a window "now"\x01', mode="test")
    b.drain()                                    # the collector thread refreshes the reply payload
    d = requests.post(url, headers=hdr, json={"readings": [r]}).json()["display"]
    assert d == {"expression": "concerned", "message": "open a window now"}   # quotes/controls stripped
    b.server.shutdown()


def test_http_backend_requires_token(monkeypatch):
    monkeypatch.delenv("INGEST_TOKEN", raising=False)
    with pytest.raises(NotConfigured):
        scd40.HttpBackend(port=0)


def test_sim_backend_refuses_real_database(db, monkeypatch):
    from config import settings
    monkeypatch.setattr(scd40, "_backend", None)
    monkeypatch.setattr(settings, "SCD40_BACKEND", "sim")
    real = Database.__new__(Database)
    real.path = settings.BASE_DIR / "data" / "siot.db"
    with pytest.raises(NotConfigured):
        scd40._get_backend(real)
    assert isinstance(scd40._get_backend(db), scd40.SimBackend)      # temp DB is fine
    monkeypatch.setattr(scd40, "_backend", None)


def test_ultrahuman_flags_implausible_readings():
    payload = {"data": {"metrics": {"d": [{"type": "temp", "object": {"values": [
        {"value": 33.0, "timestamp": 1}, {"value": 14.8, "timestamp": 2}]}}]}}}
    keep = ["skin_temp_c"]      # explicit: this metric is not in the default set
    q = {o.ts: o.quality for o in ultrahuman.parse_day("d", payload, keep=keep).observations}
    assert q == {1: "ok", 2: "suspect"}


def test_ultrahuman_stores_only_the_metrics_we_asked_for():
    """The ring returns far more than the air-quality question needs. Anything
    skipped can be backfilled later, because the API serves history."""
    payload = {"data": {"metrics": {"d": [
        {"type": "hr", "object": {"values": [{"value": 62, "timestamp": 1}]}},
        {"type": "temp", "object": {"values": [{"value": 33.0, "timestamp": 1}]}},
        {"type": "recovery_index", "object": {"value": 71, "day_start_timestamp": 0}},
        {"type": "movement_index", "object": {"value": 55, "day_start_timestamp": 0}},
    ]}}}
    keep = ["hr_bpm", "daily_recovery_index"]
    got = {o.metric for o in ultrahuman.parse_day("d", payload, keep=keep).observations}
    assert got == {"hr_bpm", "daily_recovery_index"}        # temp and movement dropped


def test_two_nodes_posting_at_once_both_land_in_their_own_room(db):
    """Both ESP32s share one ingest port. Each request gets its own handler thread,
    so the queue and the device table have to cope with them arriving together."""
    import threading, time, requests
    b = scd40.HttpBackend(db, port=0, token="secret")
    url, hdr = f"http://127.0.0.1:{b.port}/ingest", {"X-Auth-Token": "secret"}
    now = int(time.time()) // 60 * 60
    ok = []

    def post(device, room, base):
        batch = [{"ts": now - i * 60, "co2": base + i, "co2_max": base + i + 10,
                  "t": 21.0, "rh": 50, "n": 12} for i in range(30)]
        r = requests.post(url, headers=hdr, json={"device": device, "room": room,
                                                  "readings": batch})
        ok.append(r.json()["accepted"])

    threads = [threading.Thread(target=post, args=("esp32-1", "desk", 700)),
               threading.Thread(target=post, args=("esp32-2", "bedroom", 900))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ok == [30, 30]

    rows = [r for row in b.drain() for r in scd40.reading_to_obs(*row)]
    per_room = {}
    for o in rows:
        per_room[o.location] = per_room.get(o.location, 0) + 1
    assert per_room == {"desk": 120, "bedroom": 120}        # 30 readings x 4 metrics each
    with db.conn() as c:
        seen = dict(c.execute("SELECT device, accepted_total FROM device_status"))
    assert seen == {"esp32-1": 30, "esp32-2": 30}
    b.server.shutdown()


def test_a_full_48_hour_backlog_is_accepted_batch_after_batch(db):
    """What happens when the laptop sleeps: the node holds up to 48 h and uploads it
    all on waking. Nothing in that window may be rejected as too old."""
    import time, requests
    b = scd40.HttpBackend(db, port=0, token="secret")
    url, hdr = f"http://127.0.0.1:{b.port}/ingest", {"X-Auth-Token": "secret"}
    now = int(time.time()) // 60 * 60
    oldest = now - 2880 * 60                                  # exactly the buffer depth
    total = 0
    for start in range(0, 2880, 30):                          # as the firmware sends it
        batch = [{"ts": oldest + (start + i) * 60, "co2": 800, "co2_max": 820,
                  "t": 21.0, "rh": 50, "n": 12} for i in range(30)]
        total += requests.post(url, headers=hdr,
                               json={"device": "esp32-2", "room": "bedroom",
                                     "readings": batch}).json()["accepted"]
    assert total == 2880                                      # not one reading refused
    assert len(b.drain()) == 2880
    b.server.shutdown()


def test_catch_up_matches_its_window_to_the_gap(tmp_path, monkeypatch):
    """What happens when the laptop was shut. Every API here serves history, so the
    only thing that loses data is nobody asking for it; the first request after a
    gap has to be as wide as the gap."""
    import time
    import main
    asked = []

    class Fake:
        SOURCE = "fake"

        @staticmethod
        def backfill(days, db=None):
            asked.append(days)
            return CollectResult("fake")

    class Pushed:           # like scd40: no backfill, the node holds its own backlog
        SOURCE = "pushed"

    monkeypatch.setattr(main, "COLLECTORS", {"fake": Fake, "pushed": Pushed})

    def db_last_seen(age_s):
        db = Database(tmp_path / f"c{age_s}.db")
        if age_s is not None:
            r = CollectResult("fake")
            r.observations = [Observation("fake", "m", "l", int(time.time()) - age_s,
                                          1.0, "measured")]
            db.save_result(r)
        return db

    main._catch_up(db_last_seen(None))
    assert asked == [7]                       # nothing stored at all: pull a week
    main._catch_up(db_last_seen(600))
    assert asked == [7]                       # ten minutes old: the normal poll covers it
    main._catch_up(db_last_seen(3 * 86400))
    assert asked == [7, 5]                    # three-day gap: ask for five days
    main._catch_up(db_last_seen(400 * 86400))
    assert asked[-1] == 30                    # capped, so a stale database cannot stall startup


def test_a_well_ventilated_room_is_not_rejected_as_a_fault(db):
    """A sensor on its factory calibration reads tens of ppm below true outdoor air,
    so a fully aired room lands in the 340s. The old 350 floor threw those readings
    away - censoring the very measurements the outdoor baseline comes from - while
    still having to reject a dead sensor reporting zero."""
    import time
    b = scd40.HttpBackend(db, port=0, token="secret")
    now = int(time.time()) // 60 * 60
    base = {"co2_max": 400, "t": 18.0, "rh": 55, "n": 12}
    assert b.accept({**base, "ts": now, "co2": 345}) == 1       # aired room, kept
    assert b.accept({**base, "ts": now - 60, "co2": 255}) == 1  # extreme but plausible
    assert b.accept({**base, "ts": now - 120, "co2": 0}) == 0   # dead sensor, rejected
    assert b.accept({**base, "ts": now - 180, "co2": 240}) == 0
    b.server.shutdown()


def test_a_second_ingest_server_cannot_quietly_share_the_port(db):
    """Windows treats SO_REUSEADDR as permission to share a listening socket, so a
    second collector bound 8765 alongside the first and the nodes' readings were
    split between two databases with nothing reported. The second bind must fail."""
    import pytest as _pytest
    first = scd40.HttpBackend(db, port=0, token="secret")
    try:
        with _pytest.raises(OSError):
            scd40.HttpBackend(db, port=first.port, token="secret")
    finally:
        first.server.shutdown()
