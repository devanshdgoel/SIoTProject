"""Dashboard API tests: real server on an ephemeral port, temporary database, fake collectors."""

import threading
import time

import pytest
import requests

from collectors.base import CollectResult, Observation
from dashboard.server import create_server
from storage.database import Database


class FakeCollector:
    SOURCE = "fake"

    @staticmethod
    def collect(db=None):
        r = CollectResult("fake")
        r.observations.append(Observation("fake", "m", "l", int(time.time()), 1.0, "measured"))
        return r


@pytest.fixture
def server(tmp_path):
    db = Database(tmp_path / "d.db")
    srv = create_server(db, {"fake": FakeCollector, "scd40": FakeCollector}, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield db, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


H = {"X-Requested-With": "dashboard"}


def test_state_series_and_run(server):
    db, url = server
    s = requests.get(url + "/api/state").json()
    assert {c["name"]: c["health"] for c in s["collectors"]} == {"fake": "idle", "scd40": "idle"}
    assert requests.get(url + "/").text.startswith("<!doctype html>")
    assert requests.post(url + "/api/run/fake", headers=H).json()["status"] == "ok"
    s = requests.get(url + "/api/state").json()
    assert s["collectors"][0]["health"] == "good" and s["storage"]["observations"] == 1
    pts = requests.get(url + "/api/series", params={"metric": "m", "location": "l", "hours": 1}).json()["points"]
    assert len(pts) == 1 and pts[0][1] == 1.0


def test_post_guards_and_display(server):
    db, url = server
    assert requests.post(url + "/api/run/fake").status_code == 403                    # no CSRF header
    assert requests.post(url + "/api/run/scd40", headers=H).status_code == 404        # not runnable
    assert requests.post(url + "/api/run/nope", headers=H).status_code == 404
    assert requests.post(url + "/api/display", headers=H, json={"expression": "x", "message": "hi"}).status_code == 400
    assert requests.post(url + "/api/display", headers=H, json={"expression": "happy", "message": "hi"}).status_code == 200
    d = requests.get(url + "/api/state").json()["display"][0]
    assert (d["expression"], d["mode"]) == ("happy", "test")
    assert requests.get(url + "/api/series").status_code == 400                       # missing metric
