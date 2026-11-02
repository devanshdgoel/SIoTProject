"""Offline tests: parsers and storage behaviour, no network or hardware needed."""

import pytest

from collectors import grid_carbon, laqn, open_meteo, scd40, thames_water, ultrahuman
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


def test_carbon_period_shapes_and_forecast_only_for_future():
    national = {"data": [{"from": "2026-10-29T22:00Z", "to": "2026-10-29T22:30Z",
                          "intensity": {"forecast": 200, "actual": 210, "index": "high"}}]}
    regional = {"data": {"regionid": 13, "data": [{"from": "2026-10-29T23:00Z", "to": "2026-10-29T23:30Z",
                                                   "intensity": {"forecast": 250, "index": "high"}}]}}
    nat = grid_carbon._intensity_obs(grid_carbon._periods(national), "GB", NOW)
    reg = grid_carbon._intensity_obs(grid_carbon._periods(regional), "London", NOW)
    assert [(o.metric, o.kind) for o in nat] == [("carbon_actual_gco2_kwh", "modelled")]   # period already over
    assert [(o.metric, o.kind, o.issued_at) for o in reg] == [("carbon_forecast_gco2_kwh", "forecast", NOW - NOW % 1800)]


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


THAMES = {"items": [
    {"LocationName": "Site A", "AlertStatus": "Discharging", "StatusChange": "2026-10-29T20:00:00",
     "MostRecentDischargeAlertStart": "2026-10-29T20:00:00", "MostRecentDischargeAlertStop": None,
     "ReceivingWaterCourse": "River Thames", "X": 529000, "Y": 180000},
    {"LocationName": "Site B", "AlertStatus": "Not discharging", "StatusChange": "2026-10-28T10:00:00",
     "X": 530000, "Y": 181000},
    {"LocationName": "Site C", "AlertStatus": "Offline"},
    {"AlertStatus": "Discharging"},                      # no name -> ignored
]}


def test_thames_parse_status_counts_and_coordinates():
    r = thames_water.parse_status(THAMES, now=NOW)
    counts = {o.metric: o.value for o in r.observations}
    assert counts == {"overflows_discharging": 1.0, "overflows_offline": 1.0, "overflows_monitored": 3.0}
    a = r.overflow_status[0]
    assert a["alert_status"] == "Discharging" and a["last_stop"] is None
    assert 51.3 < a["lat"] < 51.7 and -0.4 < a["lon"] < 0.1       # central London from BNG


def test_thames_rejects_unexpected_shape():
    with pytest.raises(CollectorError):
        thames_water.parse_status({"items": "nope"})
    with pytest.raises(CollectorError):
        thames_water.parse_status({"items": []})


def test_thames_alerts_parse():
    a = thames_water.parse_alerts([{"LocationName": "A", "AlertType": "Start", "DateTime": "2026-10-29T20:00:00"},
                                   {"LocationName": "B", "AlertType": "Stop", "DateTime": None}])
    assert len(a) == 1 and a[0]["alert_type"] == "Start"


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
    q = {o.ts: o.quality for o in ultrahuman.parse_day("d", payload).observations}
    assert q == {1: "ok", 2: "suspect"}


def test_laqn_unwraps_dotnet_json_and_maps_species():
    # the service renders XML as JSON: '@' prefixes, a named wrapper, hourly GMT stamps
    payload = {"AirQualityData": {"@SiteCode": "KC1", "Data": [
        {"@MeasurementDateGMT": "2026-10-29 21:00:00", "@SpeciesCode": "PM25", "@Value": "12.4"},
        {"@MeasurementDateGMT": "2026-10-29 22:00:00", "@SpeciesCode": "NO2", "@Value": "41.0"},
        {"@MeasurementDateGMT": "2026-10-29 22:00:00", "@SpeciesCode": "PM25", "@Value": ""},
        {"@MeasurementDateGMT": "2026-10-29 22:00:00", "@SpeciesCode": "CO", "@Value": "0.3"},
        {"@MeasurementDateGMT": "2026-10-29 22:00:00", "@SpeciesCode": "PM10", "@Value": "4000"},
    ]}}
    obs = laqn._parse(laqn._records(payload, "data"), "KC1", NOW)
    got = {(o.metric, o.ts): (o.value, o.kind, o.unit, o.quality, o.location) for o in obs}
    assert got == {
        ("pm2_5", NOW - 6000): (12.4, "measured", "ug/m3", "ok", "KC1"),
        ("no2", NOW - 2400): (41.0, "measured", "ug/m3", "ok", "KC1"),
        ("pm10", NOW - 2400): (4000.0, "measured", "ug/m3", "suspect", "KC1"),
    }   # blank value skipped (analyser down), CO skipped (mg/m3, would not compare with CAMS)


def test_laqn_accepts_a_lone_record_and_rejects_a_shapeless_response():
    one = {"RawAQData": {"Data": {"@MeasurementDateGMT": "2026-10-29 22:00:00",
                                  "@SpeciesCode": "O3", "@Value": "55"}}}
    assert [o.metric for o in laqn._parse(laqn._records(one, "data"), "KC1", NOW)] == ["o3"]
    with pytest.raises(CollectorError):
        laqn._records({"Nope": {"Other": 1}}, "data")


def test_laqn_skipped_when_no_sites_configured(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "LAQN_SITES", [])
    with pytest.raises(NotConfigured):
        laqn.collect()


def test_laqn_nearest_ranks_sites_by_distance(monkeypatch):
    feed = {"Sites": {"Site": [
        {"@SiteCode": "FAR", "@SiteName": "Far", "@SiteType": "Roadside",
         "@Latitude": "51.60", "@Longitude": "-0.10", "@DateClosed": ""},
        {"@SiteCode": "NEAR", "@SiteName": "Near", "@SiteType": "Urban Background",
         "@Latitude": "51.4890", "@Longitude": "-0.1900", "@DateClosed": ""},
        {"@SiteCode": "SHUT", "@SiteName": "Shut", "@SiteType": "Kerbside",
         "@Latitude": "51.4861", "@Longitude": "-0.1893", "@DateClosed": "2019-01-01"},
        {"@SiteCode": "NOFIX", "@SiteName": "No fix", "@SiteType": "Kerbside",
         "@Latitude": "0", "@Longitude": "0", "@DateClosed": ""},
    ]}}
    monkeypatch.setattr(laqn, "get_json", lambda *a, **k: feed)
    out = laqn.nearest(51.486114, -0.1893498)
    assert [r[1] for r in out] == ["NEAR", "FAR"]      # closed site and 0,0 placeholder dropped
    assert out[0][0] < 1 and out[1][0] > 5             # km, nearest first


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
