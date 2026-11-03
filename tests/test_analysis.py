"""Analysis tests. The room is simulated from known physics, so the fits can be
checked against the truth that generated them rather than against themselves."""

import math

import numpy as np
import pandas as pd
import pytest

from analysis import analysis_human, features
from analysis.room_model import RoomModel, classify, learn, split_point
from collectors.base import CollectResult, Observation
from storage.database import Database

T0 = 1_793_000_000 - 1_793_000_000 % 86400      # midnight UTC, for tidy hour maths
TRUE = {"lam_sealed": 0.5, "lam_open": 3.0, "source": 600.0, "c_out": 430.0}


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.db")


def simulate(db, location="bedroom", days=3, seed=0, **truth):
    """A room obeying dC/dt = S - lambda(C - C_out), sampled every minute.

    Occupied 23-07 and 09-12 and 14-18; the window is open 12-14 while empty. That
    gives both sealed decays (after 07 and after 18) and ventilated ones (12-14),
    which is what the classifier has to separate."""
    t = dict(TRUE, **truth)
    rng = np.random.default_rng(seed)
    step, obs = 60, []
    c = t["c_out"]
    for i in range(days * 86400 // step):
        ts = T0 + i * step
        hour = (ts // 3600) % 24
        occupied = hour >= 23 or hour < 7 or 9 <= hour < 12 or 14 <= hour < 18
        ventilated = 12 <= hour < 14
        lam = t["lam_open"] if ventilated else t["lam_sealed"]
        steady = t["c_out"] + (t["source"] / lam if occupied else 0.0)
        c = steady + (c - steady) * math.exp(-lam * step / 3600.0)
        obs.append(Observation("scd40", "co2_ppm", location, ts,
                               round(c + rng.normal(0, 2), 2), "measured", "ppm"))
        if i % 60 == 0:                       # hourly outdoor context, as the API gives it
            obs.append(Observation("open_meteo", "outdoor_temp_c", "home", ts,
                                   8.0, "modelled", "C"))
            obs.append(Observation("open_meteo", "wind_ms", "home", ts, 3.0, "modelled", "m/s"))
    result = CollectResult("scd40")
    result.observations = obs
    db.save_result(result)
    return T0 + days * 86400


# ---------------------------------------------------------------- learning
def test_learns_the_physics_that_generated_the_room(db):
    now = simulate(db, days=3)
    out = learn(db, "bedroom", days=4, now=now)

    assert out["decays"] >= 6, out
    assert abs(out["c_out"] - TRUE["c_out"]) < 30

    model = RoomModel(db, "bedroom")
    assert model.learned, model.describe()
    # within 20% of truth is a real test: nothing here was told the answer
    assert model.value["lambda_sealed"] == pytest.approx(TRUE["lam_sealed"], rel=0.20)
    assert model.value["lambda_open"] == pytest.approx(TRUE["lam_open"], rel=0.30)
    assert model.value["source_ppm_h"] == pytest.approx(TRUE["source"], rel=0.25)
    # the inferred volume should be a believable bedroom, not a cupboard or a hall
    assert 10 < model.volume_m3 < 120


def test_a_room_that_is_never_opened_does_not_invent_a_second_cluster(db):
    now = simulate(db, location="desk", days=3, lam_open=0.5)   # "open" is no different
    learn(db, "desk", days=4, now=now)
    rows = db.decays("desk")
    assert rows and all(r["inferred"] == "sealed" for r in rows)


def test_split_point_separates_bimodal_and_refuses_unimodal():
    assert split_point([0.4, 0.45, 0.5, 2.8, 3.0, 3.2]) == pytest.approx(1.2, abs=0.6)
    assert split_point([0.40, 0.44, 0.47, 0.51, 0.55, 0.58]) is None   # one cluster only


def test_a_user_label_overrides_the_inferred_split():
    lams = [0.4, 0.5, 3.0, 3.1]
    assert classify(lams, [None, None, None, None], 1.5) == \
        ["sealed", "sealed", "ventilated", "ventilated"]
    # the user says the window was shut, so a high lambda is a draught, not a window
    assert classify(lams, [None, None, "closed", None], 1.5)[2] == "sealed"


# ---------------------------------------------------------------- prediction
def test_forward_and_time_to_agree_with_each_other(db):
    simulate(db, days=1)
    db.set_param("bedroom", "c_out", 430.0, None, 99, "ppm")
    db.set_param("bedroom", "lambda_sealed", 0.5, 0.05, 9, "1/h")
    db.set_param("bedroom", "source_ppm_h", 600.0, 50.0, 4, "ppm/h")
    model = RoomModel(db, "bedroom")

    minutes = model.time_to(1000.0, 600.0, occupied=True)
    assert 10 < minutes < 120
    path = model.forward(600.0, minutes + 5, occupied=True, step_min=1)
    reached = next(m for m, v in path if v >= 1000.0)
    assert reached == pytest.approx(minutes, abs=2)          # same curve, two readings of it

    assert model.time_to(1000.0, 600.0, occupied=False) is None   # empty room never gets there
    assert model.time_to(800.0, 1600.0, occupied=False, ventilated=True) < 60


def test_predictions_are_scored_against_what_actually_happened(db):
    db.add_predictions([{"issued_at": T0, "location": "bedroom", "metric": "co2_ppm",
                         "target_ts": T0 + 1800, "horizon_s": 1800, "predicted": 900.0,
                         "basis": "learned"}])
    assert db.score_due(now=T0 + 3600) == 0                  # nothing measured yet
    simulate(db, days=1)
    assert db.score_due(now=T0 + 3600) == 1
    rows = db.prediction_error("bedroom", days=99999)
    assert rows and rows[0]["n"] == 1 and rows[0]["mae"] is not None


# ---------------------------------------------------------------- cleaning
def test_despike_removes_a_spike_and_leaves_its_neighbours_alone():
    s = pd.Series([500.0] * 20, index=pd.date_range("2026-11-01", periods=20, freq="5min", tz="UTC"))
    s.iloc[10] = 5000.0
    clean, n = features.despike(s)
    assert n == 1 and clean.iloc[10] == 500.0
    assert (clean.drop(clean.index[10]) == 500.0).all()


def test_the_ring_is_not_interpolated_across_a_long_gap(db):
    rows = [Observation("ultrahuman", "hr_bpm", "user", T0 + i * 420, 60.0, "measured", "bpm")
            for i in range(3)]
    rows.append(Observation("ultrahuman", "hr_bpm", "user", T0 + 6 * 3600, 62.0, "measured", "bpm"))
    r = CollectResult("ultrahuman"); r.observations = rows
    db.save_result(r)
    s = features.series(db, "hr_bpm", "user", T0 - 60, T0 + 7 * 3600)
    # 30 minutes either side may be bridged; the rest of the 6-hour hole stays empty
    assert s.dropna().size <= 10
    assert s.isna().sum() > 50


# ---------------------------------------------------------------- decisions
@pytest.mark.parametrize("state,expected", [
    ({"room": "desk", "stale": True}, "sad"),
    ({"room": "bedroom", "quiet": True, "co2": 1800.0}, "sleepy"),
    ({"room": "desk", "co2": 1600.0}, "alert"),
    ({"room": "desk", "co2": 1100.0}, "concerned"),
    ({"room": "desk", "co2": 900.0, "minutes_to_warn": 15.0}, "concerned"),
    ({"room": "desk", "co2": 900.0, "minutes_to_warn": 300.0}, "neutral"),
    ({"room": "desk", "co2": 600.0}, "happy"),
])
def test_decide_covers_each_case(state, expected):
    expression, message, _severity = analysis_human.decide(state)
    assert expression == expected
    assert 0 < len(message) <= 40            # it has to fit the OLED


def test_quiet_hours_never_nag_even_when_the_air_is_bad():
    _expr, msg, severity = analysis_human.decide(
        {"room": "bedroom", "quiet": True, "co2": 1900.0})
    assert severity == "info" and "window" not in msg


def test_poor_recovery_makes_the_pet_stricter(db):
    """The threshold moves, the physics does not: a low recovery score only means the
    cost of a stuffy room is higher tonight."""
    scores = [78.0, 82.0, 80.0, 85.0, 76.0, 81.0, 79.0, 84.0, 77.0, 83.0] * 2
    base = [Observation("ultrahuman", "daily_recovery_index", "user", T0 + d * 86400,
                        v, "measured", "index") for d, v in enumerate(scores)]
    base.append(Observation("ultrahuman", "daily_recovery_index", "user",
                            T0 + 20 * 86400, 40.0, "measured", "index"))
    r = CollectResult("ultrahuman"); r.observations = base
    db.save_result(r)
    strict, z = analysis_human.strictness(db, T0 + 21 * 86400, days=40)
    assert z is not None and z < 0 and 0.85 <= strict < 1.0


# ---------------------------------------------------------------- end to end
def test_full_pass_writes_a_face_a_forecast_and_an_event(db):
    now = simulate(db, days=3)
    out = analysis_human.run(db, now=now - 60)

    assert "bedroom" in out["fits"] and out["fits"]["bedroom"]["decays"] >= 6
    assert out["decision"]["room"] == "bedroom"
    assert out["decision"]["expression"] in ("happy", "neutral", "concerned", "alert", "sleepy")

    assert db.latest_display() is not None
    with db.conn() as c:
        assert c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] >= 1
        assert c.execute("SELECT COUNT(*) FROM analysis_runs WHERE status='ok'").fetchone()[0] == 1


def test_a_pass_with_no_indoor_data_is_quiet_rather_than_broken(db):
    out = analysis_human.run(db, now=T0)
    assert out["rooms"] == {} and "decision" not in out


def test_a_clamped_baseline_is_not_passed_off_as_measured(db):
    """A room that never reaches outdoor air gives a 5th percentile pinned at the
    clamp. That is a guess, not a measurement, so it must keep using the default
    until enough decays actually settle - one settled decay is not enough."""
    import time
    now = int(time.time()) // 300 * 300
    rows = []
    for i in range(60):                       # an hour of stuffy air, never airing out
        rows.append(Observation("scd40", "co2_ppm", "stuffy", now - (60 - i) * 60,
                                1500.0 + (i % 5), "measured", "ppm"))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)

    learn(db, "stuffy", days=1, now=now)
    stored = db.params("stuffy")["c_out"]
    assert stored["value"] == 650.0           # pinned at the clamp
    assert stored["n"] == 0                   # ...and therefore not trusted
    assert RoomModel(db, "stuffy").basis["c_out"] == "default"


def test_a_decay_broken_by_one_noisy_sample_stays_one_measurement():
    """A twelve-hour decay interrupted by a single wobble is worth far more whole
    than as the fragments it would otherwise become, which disagree with each other."""
    idx = pd.date_range("2026-11-03", periods=60, freq="5min", tz="UTC")
    clean = pd.Series(430 + 2000 * np.exp(-0.35 * np.arange(60) * 5 / 60.0), index=idx)
    nudged = clean.copy()
    nudged.iloc[30] += 25.0                     # someone walks past, one sample up

    whole = features.decay_windows(clean, 430.0)
    split = features.decay_windows(nudged, 430.0)
    assert len(whole) == 1 and len(split) == 1  # the wobble does not end the run
    assert (split[0][1] - split[0][0]).total_seconds() > 4 * 3600

    fit = features.best_decay_fit(nudged.loc[split[0][0]:split[0][1]], 430.0)
    assert fit["lambda_h"] == pytest.approx(0.35, rel=0.1)


def test_a_long_interruption_does_end_the_run():
    idx = pd.date_range("2026-11-03", periods=60, freq="5min", tz="UTC")
    s = pd.Series(430 + 2000 * np.exp(-0.35 * np.arange(60) * 5 / 60.0), index=idx)
    s.iloc[25:35] = s.iloc[25]                  # half an hour flat: the room was reoccupied
    runs = features.decay_windows(s, 430.0)
    assert len(runs) == 2
