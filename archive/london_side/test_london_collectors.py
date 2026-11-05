"""Unit tests for the parked city-scale collectors. Kept beside the code they
cover so reviving one brings its tests with it. pytest.ini limits collection to
tests/, so these do not run as part of the normal suite."""

import pytest

from collectors import grid_carbon, laqn, thames_water
from collectors.base import CollectorError, NotConfigured

NOW = 1_793_313_600

THAMES = {"items": [
    {"LocationName": "Site A", "AlertStatus": "Discharging", "StatusChange": "2026-10-29T20:00:00",
     "MostRecentDischargeAlertStart": "2026-10-29T20:00:00", "MostRecentDischargeAlertStop": None,
     "ReceivingWaterCourse": "River Thames", "X": 529000, "Y": 180000},
    {"LocationName": "Site B", "AlertStatus": "Not discharging", "StatusChange": "2026-10-28T10:00:00",
     "X": 530000, "Y": 181000},
    {"LocationName": "Site C", "AlertStatus": "Offline"},
    {"AlertStatus": "Discharging"},                      # no name -> ignored
]}


def test_carbon_period_shapes_and_forecast_only_for_future():
    national = {"data": [{"from": "2026-10-29T22:00Z", "to": "2026-10-29T22:30Z",
                          "intensity": {"forecast": 200, "actual": 210, "index": "high"}}]}
    regional = {"data": {"regionid": 13, "data": [{"from": "2026-10-29T23:00Z", "to": "2026-10-29T23:30Z",
                                                   "intensity": {"forecast": 250, "index": "high"}}]}}
    nat = grid_carbon._intensity_obs(grid_carbon._periods(national), "GB", NOW)
    reg = grid_carbon._intensity_obs(grid_carbon._periods(regional), "London", NOW)
    assert [(o.metric, o.kind) for o in nat] == [("carbon_actual_gco2_kwh", "modelled")]   # period already over
    assert [(o.metric, o.kind, o.issued_at) for o in reg] == [("carbon_forecast_gco2_kwh", "forecast", NOW - NOW % 1800)]


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
