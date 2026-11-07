"""Analysis tests. The room is simulated from known physics, so the fits can be
checked against the truth that generated them rather than against themselves."""

import datetime as dt
import json
import math
import time

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


def simulate(db, location="bedroom", days=3, seed=0, label=True, **truth):
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
    # The simulation knows when the room was occupied and when the window was open,
    # so it supplies that as ground truth the way the user would. Without occupancy
    # labels the source term is deliberately not measurable.
    for d in range(days) if label else ():
        base = T0 + d * 86400
        for h0, h1 in ((0, 7), (9, 12), (14, 18), (23, 24)):
            db.add_label(location, "occupancy", "present", base + h0 * 3600, base + h1 * 3600)
        db.add_label(location, "occupancy", "absent", base + 7 * 3600, base + 9 * 3600)
        db.add_label(location, "window", "open", base + 12 * 3600, base + 14 * 3600)
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
    # No labels here on purpose: this exercises the inference path, where the split
    # has to be read out of the data alone and must refuse to find one.
    now = simulate(db, location="desk", days=3, lam_open=0.5, label=False)
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
    assert stored["value"] == 700.0           # pinned at the clamp
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


def test_offset_uses_the_most_recent_co_location(db, monkeypatch):
    """Narrowing the window across every co-location label in turn left `start` on
    the newest and `end` on the oldest, which is an empty range reporting no paired
    minutes — exactly when a fresh measurement had just been taken."""
    import argparse
    import time

    import main

    now = int(time.time()) // 60 * 60
    db.add_label("bedroom", "occupancy", "colocated", now - 40 * 3600, now - 39 * 3600, "old")
    db.add_label("bedroom", "occupancy", "colocated", now - 3600, now - 60, "new")
    rows = []
    for i in range(60):
        ts = now - 3600 + i * 60
        rows.append(Observation("scd40", "co2_ppm", "desk", ts, 700.0 + i * 0.1, "measured", "ppm"))
        rows.append(Observation("scd40", "co2_ppm", "bedroom", ts, 720.0 + i * 0.1, "measured", "ppm"))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    main.cmd_offset(argparse.Namespace(a="desk", b="bedroom", metric="co2_ppm",
                                       last=120, settle=0), db)
    out = "\n".join(printed)
    assert "n=60 paired minutes" in out          # not the empty range the bug gave
    assert "-20.0" in out            # desk reads 20 ppm below bedroom, by construction


def test_where_a_night_ends_up_barely_depends_on_where_it_starts(db):
    """The point the window test made: the overnight ceiling is outdoor + source/lambda,
    and a room slept in for two or three time constants forgets its starting value."""
    from analysis import ventilation

    c_out, source, lam = 356.0, 900.0, 0.26
    high = ventilation.trajectory(835, 9 * 60, lam, source, c_out, occupied=True)
    low = ventilation.trajectory(379, 9 * 60, lam, source, c_out, occupied=True)
    gap_at_dawn = high[-1][1] - low[-1][1]
    assert 450 > gap_at_dawn > 0                     # they started 456 ppm apart
    assert gap_at_dawn < 60                          # and arrive within 60 of each other
    assert abs(high[-1][1] - (c_out + source / lam)) < 500   # both heading for the ceiling


def test_the_rate_a_target_needs_is_the_inverse_of_the_headroom(db):
    from analysis import ventilation

    need = ventilation.lambda_needed(900.0, 356.0, 1400.0)
    assert need == pytest.approx(900.0 / (1400.0 - 356.0), rel=1e-6)
    # holding a room at its own outdoor level would take infinite exchange
    assert ventilation.lambda_needed(900.0, 356.0, 300.0) is None


def test_a_shut_window_is_charged_no_heat(db):
    from analysis import ventilation

    db.set_param("x", "lambda_sealed", 0.3, None, 9, "1/h")
    db.set_param("x", "tau_t_h", 6.0, None, 4, "h")
    model = RoomModel(db, "x")
    assert ventilation._extra_cooling(model, 21.0, 9.0, 480, 0.3) == 0.0
    assert ventilation._extra_cooling(model, 21.0, 9.0, 480, 3.0) > 1.0
    assert ventilation._extra_cooling(model, 15.0, 18.0, 480, 3.0) == 0.0   # warmer outside


# ---------------------------------------------------------------- event rules
def _series(now, values, step=300):
    return [(now - (len(values) - 1 - i) * step, v) for i, v in enumerate(values)]


def test_a_single_spike_is_not_an_event():
    from analysis import event_rules as er
    rule = er.Rule("co2", "co2_ppm", enter=1000, exit_at=850, persistence_s=900)
    now = 1_793_000_000
    a = er.assess(rule, _series(now, [700, 700, 1500, 700, 700]), now)
    assert not a.on and "not yet persistent" in a.reason


def test_it_fires_only_once_the_condition_has_held():
    from analysis import event_rules as er
    rule = er.Rule("co2", "co2_ppm", enter=1000, exit_at=850, persistence_s=900)
    now = 1_793_000_000
    a = er.assess(rule, _series(now, [1100] * 6), now)
    assert a.on and a.fire and 0 < a.confidence <= 1


def test_hysteresis_leaves_no_value_to_chatter_on():
    from analysis import event_rules as er
    rule = er.Rule("co2", "co2_ppm", enter=1000, exit_at=850, persistence_s=900)
    now = 1_793_000_000
    # sitting between the two levels: once on, it stays on. It was announced ten
    # minutes ago, so it holds the state quietly rather than repeating itself.
    held = er.assess(rule, _series(now, [920] * 6), now, was_on=True, last_fired_at=now - 600)
    assert held.on and not held.fire
    # and from off, that same value does not turn it on
    assert not er.assess(rule, _series(now, [920] * 6), now, was_on=False).on
    # only falling past the release level clears it
    assert not er.assess(rule, _series(now, [800] * 6), now, was_on=True).on


def test_a_cooldown_stops_it_repeating_itself():
    from analysis import event_rules as er
    rule = er.Rule("co2", "co2_ppm", enter=1000, exit_at=850, persistence_s=900,
                   cooldown_s=3600)
    now = 1_793_000_000
    quiet = er.assess(rule, _series(now, [1100] * 6), now, was_on=True, last_fired_at=now - 600)
    assert quiet.on and not quiet.fire and "said" in quiet.reason
    later = er.assess(rule, _series(now, [1100] * 6), now, was_on=False, last_fired_at=now - 7200)
    assert later.fire


def test_a_dead_sensor_produces_nothing():
    from analysis import event_rules as er
    rule = er.Rule("co2", "co2_ppm", enter=1000, exit_at=850, stale_after_s=900)
    now = 1_793_000_000
    stale = er.assess(rule, _series(now - 7200, [1800] * 6), now)
    assert stale.stale and not stale.on and not stale.fire


def test_confidence_rises_with_margin_and_completeness():
    from analysis import event_rules as er
    rule = er.Rule("co2", "co2_ppm", enter=1000, exit_at=850, persistence_s=900)
    now = 1_793_000_000
    barely = er.assess(rule, _series(now, [1005] * 6), now)
    clearly = er.assess(rule, _series(now, [2000] * 6), now)
    assert clearly.confidence > barely.confidence


# ---------------------------------------------------------------- replay
def test_replay_shows_only_what_had_arrived(db, tmp_path):
    """The whole point: a view of the database as it stood at some past moment,
    judged on when each row was stored rather than the time it describes."""
    import time as _time
    from analysis import replay

    now = int(_time.time()) // 300 * 300
    rows = []
    for age_h in (5, 3, 1):
        rows.append(Observation("scd40", "co2_ppm", "bedroom", now - age_h * 3600,
                                700.0 + age_h, "measured", "ppm",
                                retrieved_at=now - age_h * 3600))
    # a value about the past that was only stored just now: it must stay hidden
    rows.append(Observation("scd40", "co2_ppm", "bedroom", now - 4 * 3600, 999.0,
                            "measured", "ppm", retrieved_at=now))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)

    with replay.AsOf(db, tmp_path / "replay.db") as view:
        view.advance_to(now - 2 * 3600)
        with view.db.conn() as c:
            seen = [x[0] for x in c.execute("SELECT value FROM observations ORDER BY ts")]
        assert 999.0 not in seen          # stored later, so invisible at that moment
        assert len(seen) == 2             # only the 5 h and 3 h old rows had arrived

        view.advance_to(now)
        with view.db.conn() as c:
            seen_now = [x[0] for x in c.execute("SELECT value FROM observations")]
        assert 999.0 in seen_now and len(seen_now) == 4


def test_replay_refuses_to_run_backwards(db, tmp_path):
    import time as _time
    from analysis import replay

    now = int(_time.time())
    with replay.AsOf(db, tmp_path / "r2.db") as view:
        view.advance_to(now)
        with pytest.raises(ValueError):
            view.advance_to(now - 3600)


def test_replay_leaves_derived_tables_empty(db, tmp_path):
    """Learned parameters must be re-derived at each step, never inherited, or the
    replay would be handing the model answers it had not yet earned."""
    import time as _time
    from analysis import replay

    now = int(_time.time())
    db.set_param("bedroom", "lambda_sealed", 0.3, None, 9, "1/h")
    db.save_decays([{"location": "bedroom", "start_ts": now - 7200,
                     "end_ts": now - 3600, "lambda_h": 0.3, "r2": 0.99}])
    with replay.AsOf(db, tmp_path / "r3.db") as view:
        view.advance_to(now)
        with view.db.conn() as c:
            assert c.execute("SELECT COUNT(*) FROM room_params").fetchone()[0] == 0
            assert c.execute("SELECT COUNT(*) FROM decay_events").fetchone()[0] == 0


def test_a_forecast_stops_assuming_you_never_leave(db):
    """The occupancy bias: the old forecast saw CO2 rising, decided the room was
    occupied, and then assumed it stayed occupied for the whole horizon. It
    over-predicted by more than a thousand ppm in the same direction every time,
    which makes an improving-error curve meaningless."""
    db.set_param("room", "c_out", 400.0, None, 99, "ppm")
    db.set_param("room", "lambda_sealed", 0.25, 0.02, 9, "1/h")
    db.set_param("room", "source_ppm_h", 900.0, 50.0, 5, "ppm/h")
    model = RoomModel(db, "room")

    # a room that is in fact almost always empty at every hour
    prior = [0.05] * 24
    stays = model.forward(800, 120, True)
    leaves = model.forward(800, 120, True, prior=prior, now=1_793_000_000)
    assert stays[-1][1] > leaves[-1][1] + 300          # the assumption was costing this much
    assert leaves[0][1] == pytest.approx(stays[0][1], rel=0.02)   # agree at the first step

    # a room that is in fact always occupied: the prior should barely change anything
    always = model.forward(800, 120, True, prior=[1.0] * 24, now=1_793_000_000)
    assert always[-1][1] == pytest.approx(stays[-1][1], rel=0.02)


def test_the_occupancy_prior_is_learned_not_assumed(db):
    """It comes from the room's own testimony - CO2 climbing faster than leakage can
    explain - plus labels and corroborated sleep windows. Never from a guess about
    when people are usually in."""
    from zoneinfo import ZoneInfo

    from config import settings

    zone = ZoneInfo(settings.USER_TZ)
    now = int(time.time()) // 300 * 300
    rows, value = [], 500.0
    for i in range(288 * 4):                       # four days at five minutes
        ts = now - (288 * 4 - i) * 300
        hour = dt.datetime.fromtimestamp(ts, zone).hour
        # occupied 01:00-07:00: filling. Otherwise empty: clearing.
        value += 55.0 if 1 <= hour < 7 else -35.0
        value = min(2600.0, max(450.0, value))
        rows.append(Observation("scd40", "co2_ppm", "room", ts, value, "measured", "ppm"))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)

    profile = features.occupancy_prior(db, "room", now - 6 * 86400, now)
    assert profile is not None and len(profile["prior"]) == 24
    assert profile["prior"][3] > 0.8               # deep in the occupied stretch
    assert profile["prior"][14] < 0.2              # mid-afternoon, clearing
    # and it measures how long a spell lasts rather than assuming a number
    assert 20.0 <= profile["persistence_min"] <= 360.0
    assert profile["spell_min"] > 60               # six hours of filling, not minutes


def test_two_people_do_not_halve_the_room(db):
    """A source fitted without the headcount is wrong for one person and for two,
    and the volume it implies comes out at half the real size. Storing it per person
    keeps the physics straight whoever is in the room."""
    now = int(time.time()) // 300 * 300
    # A room obeying the mass balance with two people in it: source 1600 ppm/h, so
    # 800 each. It has to be a real saturating rise, not a straight line, because the
    # curvature is what carries the information the fit reads.
    c_out, lam, source = 450.0, 0.4, 1600.0
    ceiling = c_out + source / lam
    rows = []
    for i in range(96):
        ts = now - (96 - i) * 300
        hours = i * 300 / 3600.0
        value = ceiling + (500.0 - ceiling) * math.exp(-lam * hours)
        rows.append(Observation("scd40", "co2_ppm", "pair", ts, round(value, 1),
                                "measured", "ppm"))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)
    db.add_label("pair", "occupancy", "present", now - 96 * 300, now, "two of us", people=2)

    learn(db, "pair", days=2, now=now)
    stored = db.params("pair")
    assert "source_ppm_h" in stored
    per_person = stored["source_ppm_h"]["value"]
    assert stored["source_ppm_h"]["unit"] == "ppm/h per person"
    assert per_person == pytest.approx(800.0, rel=0.25)   # half of 1600, not all of it
    assert int(stored["typical_people"]["value"]) == 2

    # the model puts the people back when it predicts
    model = RoomModel(db, "pair")
    assert model.occupants == 2
    # One rise is below the count at which a fitted source displaces the default, so
    # the model is still on its default here - but it must multiply whatever source
    # it is using by the headcount, which is the behaviour under test.
    assert model.steady_state(occupied=True) == pytest.approx(
        model.value["c_out"]
        + model.value["source_ppm_h"] * 2 / model.value["lambda_sealed"], rel=1e-6)
    db.set_param("pair", "typical_people", 1, None, 9, "people")
    assert RoomModel(db, "pair").steady_state(occupied=True) < model.steady_state(occupied=True)


def test_an_unlabelled_room_still_assumes_one_person(db):
    """One is what an occupancy label has always meant, so nothing changes for the
    bedroom, which is where the measurements are cleanest."""
    from analysis import features
    labels = [{"kind": "occupancy", "value": "present", "people": None,
               "start_ts": 0, "end_ts": 100}]
    assert features.people_at(labels, 50) == 1
    labels[0]["people"] = 3
    assert features.people_at(labels, 50) == 3
    assert features.people_at(labels, 500) == 1          # outside the label


# ---------------------------------------------------------------- episodes
def test_a_confirmation_becomes_a_label_and_a_denial_becomes_its_opposite(db):
    """The whole learning loop in one assertion: answering a detection writes the
    same label the user would have typed, and saying no is evidence too - it moves
    that decay into the sealed group rather than being thrown away."""
    now = int(time.time())
    db.add_episodes([{"location": "bedroom", "kind": "window", "value": "open",
                      "start_ts": now - 3600, "end_ts": now - 1800, "confidence": 0.8,
                      "evidence": "{}", "detected_at": now},
                     {"location": "bedroom", "kind": "window", "value": "open",
                      "start_ts": now - 7200, "end_ts": now - 5400, "confidence": 0.6,
                      "evidence": "{}", "detected_at": now}])
    open_now = db.episodes()
    assert len(open_now) == 2

    db.judge_episode(open_now[0]["id"], "confirmed")
    db.judge_episode(open_now[1]["id"], "denied")
    values = {(lb["kind"], lb["value"]) for lb in db.labels(location="bedroom")}
    assert ("window", "open") in values          # the yes
    assert ("window", "closed") in values        # the no, kept as its opposite
    assert db.episodes() == []                   # nothing left waiting


def test_not_sure_records_the_answer_but_creates_no_label(db):
    now = int(time.time())
    db.add_episodes([{"location": "desk", "kind": "cooking", "value": "yes",
                      "start_ts": now - 3600, "end_ts": now - 3000, "confidence": 0.5,
                      "evidence": "{}", "detected_at": now}])
    db.judge_episode(db.episodes()[0]["id"], "unsure")
    assert db.labels(location="desk") == []      # ambiguity is not evidence
    assert db.episodes(verdict="unsure")         # but it is remembered


def test_a_confirmed_headcount_reaches_the_label(db):
    now = int(time.time())
    db.add_episodes([{"location": "desk", "kind": "occupancy", "value": "present",
                      "start_ts": now - 3600, "end_ts": now - 1800, "confidence": 0.7,
                      "evidence": "{}", "detected_at": now}])
    db.judge_episode(db.episodes()[0]["id"], "confirmed", people=2)
    assert db.labels(location="desk")[0]["people"] == 2


def test_an_answered_episode_is_not_re_asked(db):
    """Re-running the detector must not undo a verdict already given."""
    now = int(time.time())
    row = {"location": "bedroom", "kind": "window", "value": "open",
           "start_ts": now - 3600, "end_ts": now - 1800, "confidence": 0.8,
           "evidence": "{}", "detected_at": now}
    db.add_episodes([row])
    db.judge_episode(db.episodes()[0]["id"], "confirmed")
    assert db.add_episodes([row]) == 0            # detected again, but not re-opened
    assert db.episodes() == []


def test_the_detector_finds_a_night_in_the_bedroom(db):
    """A sealed room cannot rise on its own, so a sustained climb is somebody in it."""
    from analysis import episodes
    from analysis.room_model import RoomModel

    now = int(time.time()) // 300 * 300
    db.set_param("bedroom", "c_out", 400.0, None, 99, "ppm")
    db.set_param("bedroom", "lambda_sealed", 0.25, 0.02, 9, "1/h")
    db.set_param("bedroom", "source_ppm_h", 900.0, 50.0, 5, "ppm/h per person")
    rows, value = [], 450.0
    for i in range(120):                          # ten hours, rising for the first six
        ts = now - (120 - i) * 300
        value += 45.0 if i < 72 else -25.0
        rows.append(Observation("scd40", "co2_ppm", "bedroom", ts, value, "measured", "ppm"))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)

    co2 = features.series(db, "co2_ppm", "bedroom", now - 86400, now)
    found = episodes.detect_occupancy(db, "bedroom", RoomModel(db, "bedroom"),
                                      co2, now - 86400, now)
    assert found, "a six-hour climb should be reported as somebody being there"
    assert found[0]["end_ts"] - found[0]["start_ts"] > 3 * 3600


# ---------------------------------------------------------------- standardised fits
def test_a_departure_is_where_filling_turns_into_clearing():
    """Anchoring on the event rather than on a threshold crossing is what makes two
    measurements comparable: same delay after the event, same span."""
    idx = pd.date_range("2026-11-03", periods=72, freq="5min", tz="UTC")
    up = 500 + np.arange(36) * 40.0                       # three hours filling
    down = up[-1] - np.arange(36) * 30.0                  # three hours clearing
    s = pd.Series(np.concatenate([up, down]), index=idx)

    found = features.departures(s)
    assert len(found) == 1
    assert abs((found[0] - idx[35]).total_seconds()) <= 900   # within a few samples


def test_the_standardised_fit_always_has_the_same_geometry(db):
    idx = pd.date_range("2026-11-03", periods=96, freq="5min", tz="UTC")
    hours = np.arange(96) * 5 / 60.0
    rise = 500 + np.arange(24) * 50.0
    peak_at = idx[23]
    decay = 400 + (rise[-1] - 400) * np.exp(-0.5 * (hours[24:] - hours[24]))
    s = pd.Series(np.concatenate([rise, decay]), index=idx)

    fit = features.departure_decay(s, 400.0, peak_at)
    assert fit is not None
    assert fit["lambda_h"] == pytest.approx(0.5, rel=0.15)
    # the window starts exactly the configured delay after the departure
    offset = (fit["start"] - peak_at).total_seconds() / 60
    assert features.DEPARTURE_OFFSET_MIN <= offset < features.DEPARTURE_OFFSET_MIN + 10
    span = (fit["end"] - fit["start"]).total_seconds() / 60
    assert span == pytest.approx(features.DEPARTURE_WINDOW_MIN, abs=10)


def test_the_weather_fit_refuses_until_it_has_enough(db):
    from analysis.room_model import fit_weather

    db.save_decays([{"location": "r", "start_ts": 100 * i, "end_ts": 100 * i + 50,
                     "lambda_h": 0.4, "r2": 0.99, "method": "departure",
                     "inferred": "sealed", "wind_ms": 3.0, "delta_t_c": 10.0}
                    for i in range(3)])
    out = fit_weather(db, "r")
    assert "c" not in out and out["n"] == 3        # three points, three coefficients


def test_the_weather_fit_recovers_coefficients_it_was_given(db):
    """lambda^2 = a + b*dT + c*v^2 is linear in the coefficients, so a room built to
    obey it should hand them back."""
    from analysis.room_model import fit_weather, predicted_lambda

    a, b, c = 0.05, 0.004, 0.012
    rows = []
    for i, (wind, dT) in enumerate([(1.0, 8.0), (2.5, 11.0), (4.0, 9.0), (5.5, 12.0),
                                    (3.0, 10.0), (6.5, 8.5), (1.5, 13.0), (5.0, 9.5)]):
        lam = math.sqrt(a + b * dT + c * wind ** 2)
        rows.append({"location": "r", "start_ts": 1000 * i, "end_ts": 1000 * i + 500,
                     "lambda_h": lam, "r2": 0.99, "method": "departure",
                     "inferred": "sealed", "wind_ms": wind, "delta_t_c": dT})
    db.save_decays(rows)

    out = fit_weather(db, "r")
    assert out["n"] == 8 and out["r2"] > 0.99
    assert out["c"] == pytest.approx(c, rel=0.05)      # the wind term, recovered
    assert out["b"] == pytest.approx(b, abs=0.004)
    assert predicted_lambda(out, 4.0, 10.0) == pytest.approx(
        math.sqrt(a + b * 10 + c * 16), rel=0.05)


def test_threshold_windows_do_not_duplicate_a_standardised_one(db):
    """One physical event must produce one row, measured the better way."""
    now = int(time.time()) // 300 * 300
    rows, value = [], 500.0
    for i in range(144):
        ts = now - (144 - i) * 300
        value = value + 55.0 if i < 48 else 400 + (value - 400) * math.exp(-0.5 * 5 / 60)
        rows.append(Observation("scd40", "co2_ppm", "solo", ts, round(value, 1),
                                "measured", "ppm"))
    r = CollectResult("scd40"); r.observations = rows
    db.save_result(r)
    learn(db, "solo", days=2, now=now)

    spans = [(d["start_ts"], d["end_ts"], d["method"]) for d in db.decays("solo")]
    for i, (a1, b1, _m1) in enumerate(spans):
        for a2, b2, _m2 in spans[i + 1:]:
            assert not (a1 <= b2 and a2 <= b1), f"overlapping rows {spans}"


# --- an unfinished flush must not be mistaken for a baseline measurement -------

def _flush(minutes, lam, c0=1150.0, c_out=356.0, step=5):
    """Synthetic open-window decay towards c_out at rate lam, sampled every `step`."""
    n = minutes // step + 1
    idx = pd.date_range("2026-11-06 22:00", periods=n, freq=f"{step}min", tz="UTC")
    t = np.arange(n) * (step / 60.0)
    return pd.Series(c_out + (c0 - c_out) * np.exp(-lam * t), index=idx)


def test_settled_baseline_accepts_a_flush_that_equilibrated():
    # Three hours at 4.5/h leaves the room sitting on outdoor air.
    got = features.settled_baseline(_flush(180, 4.5), 400.0)
    assert got is not None
    assert abs(got - 356.0) < features.SETTLE_PPM


def test_settled_baseline_rejects_a_flush_still_falling():
    # One hour at 0.87/h - the 6 Nov still-air test - stops far short, and its
    # floor is an upper bound on outdoor CO2 rather than a measurement of it.
    seg = _flush(60, 0.87)
    assert seg.iloc[-1] - 356.0 > features.SETTLE_PPM     # genuinely unfinished
    assert features.settled_baseline(seg, 400.0) is None


def test_settled_baseline_accepts_an_already_flat_segment():
    idx = pd.date_range("2026-11-06 22:00", periods=10, freq="5min", tz="UTC")
    flat = pd.Series(np.full(10, 357.0) + np.array([0, 2, -1, 1, 0, -2, 1, 0, -1, 1]),
                     index=idx)
    got = features.settled_baseline(flat, 400.0)
    assert got is not None and abs(got - 357.0) < 3.0


def test_unfinished_flush_does_not_inflate_the_baseline_or_lambda(db):
    """The regression this guard exists for.

    A still-air opening that is over before the room reaches outdoor air used to set
    c_out from its lowest reading. On the real bedroom that lifted the baseline from
    358 to 440 ppm, and because every excess is measured against the baseline, the
    sealed air-change rate came out at 0.42/h instead of 0.27 - one unfinished flush
    inflating the room's defining parameter by more than half.
    """
    now = simulate(db, days=3, lam_open=0.9)      # too slow to finish in the 2h airing
    out = learn(db, "bedroom", days=4, now=now)
    note = json.loads(db.params("bedroom")["c_out"]["detail"])["method"]

    assert "still flushing" in note, note
    assert out["c_out"] < TRUE["c_out"] + 120      # not dragged up to the flush floor
    assert abs(out["lambda_sealed"][0] - TRUE["lam_sealed"]) < 0.2, out


def test_a_finished_flush_is_still_used_as_a_direct_measurement(db):
    # The guard must not cost us the measurement when the room really did settle:
    # two hours at 3/h leaves a couple of ppm of excess, and that IS outdoor air.
    now = simulate(db, days=3)
    out = learn(db, "bedroom", days=4, now=now)
    note = json.loads(db.params("bedroom")["c_out"]["detail"])["method"]
    assert "measured directly" in note, note
    assert abs(out["c_out"] - TRUE["c_out"]) < 15


# --- rounded self-reported times must still match their decay -----------------

def test_label_over_matches_a_decay_whose_midpoint_falls_outside():
    """The 4 Nov window test: fit 19:40-20:10, label reported as 20:00-23:30.

    The fit's edges come from the data, the label's from somebody rounding to the
    hour, so the midpoint lands five minutes short and strict containment loses the
    fastest, cleanest ventilation measurement in the record.
    """
    S = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    labels = [{"kind": "window", "value": "open",
               "start_ts": S("2026-11-04 20:00"), "end_ts": S("2026-11-04 23:30")}]
    a, b = S("2026-11-04 19:40"), S("2026-11-04 20:10")
    assert features.label_at(labels, (a + b) // 2, "window") is None   # the old test
    assert features.label_over(labels, a, b, "window") == "open"       # the new one


def test_label_over_ignores_a_label_that_barely_clips_the_decay():
    # The 3 Nov morning door flush: a window=closed label ends just as it starts,
    # so slack alone must not sweep it up.
    S = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    labels = [{"kind": "window", "value": "closed",
               "start_ts": S("2026-11-03 00:15"), "end_ts": S("2026-11-03 08:40")}]
    a, b = S("2026-11-03 08:45"), S("2026-11-03 09:10")
    assert features.label_over(labels, a, b, "window") is None


def test_label_over_prefers_the_longer_overlap():
    S = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    labels = [{"kind": "window", "value": "closed",
               "start_ts": S("2026-11-04 18:00"), "end_ts": S("2026-11-04 19:50")},
              {"kind": "window", "value": "open",
               "start_ts": S("2026-11-04 19:50"), "end_ts": S("2026-11-04 23:30")}]
    a, b = S("2026-11-04 19:45"), S("2026-11-04 20:45")
    assert features.label_over(labels, a, b, "window") == "open"


def test_one_unfinished_opening_cannot_drag_a_measured_baseline(db):
    """The real bedroom: two labelled openings, only one of which finished.

    4 Nov ran three and a half hours at 4.5/h and reached outdoor air. 6 Nov ran one
    hour in still air at 0.87/h and stopped 150 ppm short. Pooling both put the
    baseline at 440 instead of 358. The finished one has to carry it alone.
    """
    now = simulate(db, days=3)
    base = T0 + 2 * 86400
    # Label an ordinary sealed morning decay as an opening. At 0.5/h it is nowhere
    # near outdoor air by the time it ends, which is exactly the shape of a still-air
    # airing that was shut too early.
    db.add_label("bedroom", "window", "open", base + 7 * 3600, base + 8 * 3600)
    out = learn(db, "bedroom", days=4, now=now)
    note = json.loads(db.params("bedroom")["c_out"]["detail"])["method"]

    assert "measured directly during 3" in note, note
    assert "1 further opening(s) ignored" in note, note
    assert abs(out["c_out"] - TRUE["c_out"]) < 15, out
    assert abs(out["lambda_sealed"][0] - TRUE["lam_sealed"]) < 0.15, out


# --- an open window is not one number ----------------------------------------

def _opening(i, lam, wind, labelled="open"):
    return {"location": "r", "start_ts": 10_000 * i, "end_ts": 10_000 * i + 3600,
            "lambda_h": lam, "r2": 0.99, "method": "window", "inferred": "ventilated",
            "wind_ms": wind, "delta_t_c": 8.0, "labelled": labelled}


def test_the_open_window_law_refuses_on_a_single_opening(db):
    from analysis.room_model import fit_open_wind

    db.save_decays([_opening(0, 4.5, 3.7)])
    out = fit_open_wind(db, "r")
    assert "k" not in out and out["n"] == 1 and out["need"] == 2


def test_the_open_window_law_recovers_the_slope_it_was_given(db):
    from analysis.room_model import fit_open_wind, predicted_lambda_open

    k = 1.25
    db.save_decays([_opening(i, k * v, v) for i, v in
                    enumerate([0.8, 1.6, 2.4, 3.2, 4.0])])
    out = fit_open_wind(db, "r")
    assert out["n"] == 5
    assert out["k"] == pytest.approx(k, rel=0.01)
    assert out["r2"] > 0.99
    assert "provisional" not in out                 # five points can actually fail
    assert predicted_lambda_open(out, 2.0) == pytest.approx(k * 2.0, rel=0.01)


def test_the_law_uses_only_openings_the_user_confirmed(db):
    """The bedroom's corridor-door flush sits numerically between its two window
    tests, so pooling every inferred 'ventilated' decay would bend the slope towards
    an event that was not a window at all."""
    from analysis.room_model import fit_open_wind

    rows = [_opening(0, 2.575, 0.76), _opening(1, 4.547, 3.72),
            _opening(2, 1.197, 4.20, labelled=None)]      # the door, high wind, low rate
    db.save_decays(rows)
    out = fit_open_wind(db, "r")
    assert out["n"] == 2, "an unconfirmed opening entered the fit"


def test_two_openings_are_marked_as_untestable(db):
    from analysis.room_model import fit_open_wind

    db.save_decays([_opening(0, 1.0, 0.8), _opening(1, 4.6, 3.7)])
    out = fit_open_wind(db, "r")
    assert out["r2"] > 0.99                         # and yet
    assert "provisional" in out, "a fit that cannot fail must say so"


def test_this_rooms_own_openings_refute_proportionality_to_wind(db):
    """The measurement that killed the law, kept as the record of it.

    4.55/h in 3.7 m/s and 2.58/h in 0.75 m/s is a factor of 1.8 for a factor of five
    in wind. A line through the origin cannot do that: it has to put the still-air
    opening near 1/h, a third of what was measured, and it ends up explaining LESS
    than the mean of the two does. An open window here is mostly buoyancy-driven, and
    a law that has failed must not be stored as if it had passed."""
    from analysis.room_model import RoomModel, fit_open_wind

    db.save_decays([_opening(0, 2.575, 0.75), _opening(1, 4.547, 3.72)])
    out = fit_open_wind(db, "r")
    assert out["r2"] < 0, out
    assert "refuted" in out
    assert "buoyancy" in out["refuted"]
    # Stored with n=0, so nothing downstream can predict from it.
    assert db.params("r")["lambda_open_k"]["n"] == 0
    assert db.params("r")["lambda_open_k"]["value"] is None


def test_a_refuted_law_is_shown_but_never_used(db):
    from analysis.room_model import RoomModel, fit_open_wind

    simulate(db, days=3)
    db.save_decays([dict(_opening(0, 2.575, 0.75), location="bedroom"),
                    dict(_opening(1, 4.547, 3.72), location="bedroom")])
    fit_open_wind(db, "bedroom")
    model = RoomModel(db, "bedroom")

    assert model.open_wind is None, "a refuted law must not be available to predict"
    assert model.open_wind_fit["refuted"]            # but it is still on the record
    for wind in (0.8, 2.0, 3.7):
        lam, basis = model.lambda_open_at(wind)
        assert lam == pytest.approx(model.value["lambda_open"])
        assert "wind law" not in basis


def test_the_law_is_held_inside_the_winds_it_was_measured_in(db):
    from analysis.room_model import fit_open_wind, predicted_lambda_open

    db.save_decays([_opening(i, 1.25 * v, v) for i, v in enumerate([1.0, 2.0, 3.0])])
    out = fit_open_wind(db, "r")
    # A law fitted to 1-3 m/s says nothing about a gale, and over-promising air is
    # the error that reaches the user, so extrapolation is clamped rather than trusted.
    assert predicted_lambda_open(out, 30.0) == pytest.approx(1.25 * 3.0, rel=0.01)
    assert predicted_lambda_open(out, 0.0) == pytest.approx(1.25 * 1.0, rel=0.01)


def test_an_open_window_is_never_worse_than_a_shut_one(db):
    from analysis.room_model import fit_open_wind, predicted_lambda_open

    db.save_decays([_opening(i, 0.02 * v, v) for i, v in enumerate([0.5, 1.0, 1.5])])
    out = fit_open_wind(db, "r")
    assert predicted_lambda_open(out, 0.5, floor=0.27) == pytest.approx(0.27)


def test_lambda_open_follows_the_wind_and_falls_back_without_it(db):
    from analysis.room_model import RoomModel, fit_open_wind

    simulate(db, days=3)                            # gives the room its other params
    db.save_decays([dict(_opening(0, 0.95, 0.76), location="bedroom"),
                    dict(_opening(1, 4.55, 3.72), location="bedroom")])
    fit_open_wind(db, "bedroom")
    model = RoomModel(db, "bedroom")

    still, basis_still = model.lambda_open_at(0.8)
    windy, basis_windy = model.lambda_open_at(3.7)
    assert windy > 4 * still, (still, windy)
    assert "wind law" in basis_still and "wind law" in basis_windy
    # With no wind to feed it there is nothing to condition on, so the pooled
    # median is what remains - and it is honest about being that.
    pooled, basis = model.lambda_open_at(None)
    assert pooled == pytest.approx(model.value["lambda_open"])
    assert "wind law" not in basis


def test_the_night_plan_changes_with_the_forecast_wind(db):
    """The end of it: the same window, two different nights, different advice."""
    from analysis.room_model import fit_open_wind
    from analysis.ventilation import options

    simulate(db, days=3)
    db.save_decays([dict(_opening(0, 0.95, 0.76), location="bedroom"),
                    dict(_opening(1, 4.55, 3.72), location="bedroom")])
    fit_open_wind(db, "bedroom")
    model = RoomModel(db, "bedroom")

    calm = {name: lam for name, lam, _b in options(model, wind=0.8)}
    breezy = {name: lam for name, lam, _b in options(model, wind=3.7)}
    assert breezy["window wide open"] > 4 * calm["window wide open"]
    assert calm["door and window shut"] == breezy["door and window shut"]


# --- the opening-comparison figure -------------------------------------------

def test_the_openings_figure_refuses_without_a_labelled_opening(db, tmp_path):
    from analysis import figures

    simulate(db, days=1, label=False)
    with pytest.raises(ValueError, match="no labelled window openings"):
        figures.fig_openings(db, tmp_path, location="bedroom")


def test_the_openings_figure_falls_back_to_the_peak_without_a_fit(db, tmp_path):
    """With no fitted decay to anchor on, the figure finds the turn itself.

    The room here peaks a quarter of an hour before the time the user reported, the
    way a real one does, so anchoring on the label would start the curve part-way
    down its own decay. Where a fit DOES exist the fit's start wins instead - see
    test_the_openings_figure_prefers_the_fit_start - because the peak can sit inside
    a mixing transient that obeys no air-change rate.
    """
    from analysis import figures

    t0 = int(pd.Timestamp("2026-11-06 21:00", tz="UTC").timestamp())
    peak_at = t0 + 30 * 60
    top = 600 + 20 * 6
    obs = []
    for i in range(48):                       # rise for 30 min, then decay
        ts_i = t0 + i * 300
        if ts_i <= peak_at:
            value = 600 + 20 * i
        else:
            value = 360 + (top - 360) * math.exp(-1.2 * (ts_i - peak_at) / 3600)
        obs.append(Observation("scd40", "co2_ppm", "bedroom", ts_i,
                               round(value, 1), "measured", "ppm"))
        obs.append(Observation("scd40", "indoor_temp_c", "bedroom", ts_i,
                               18.5, "measured", "C"))
    result = CollectResult("scd40")
    result.observations = obs
    db.save_result(result)
    # reported a quarter of an hour late, as people do
    db.add_label("bedroom", "window", "open", peak_at + 900, t0 + 47 * 300)

    figures.style()
    figures.fig_openings(db, tmp_path, location="bedroom")
    assert (tmp_path / "09_openings.png").exists()

    # the anchor itself, checked directly rather than through the picture
    wide = features.series(db, "co2_ppm", "bedroom", t0, t0 + 48 * 300).dropna()
    near = wide.loc[figures.ts(peak_at + 900 - figures.SEARCH_MIN * 60):
                    figures.ts(peak_at + 900 + figures.SEARCH_MIN * 60)]
    assert abs(int(near.idxmax().timestamp()) - peak_at) <= 300


def test_minutes_from_puts_two_days_on_one_axis(db):
    from analysis import figures

    idx = pd.date_range("2026-11-04 19:30", periods=4, freq="10min", tz="UTC")
    got = figures._minutes_from(pd.Series([1.0] * 4, index=idx),
                                int(idx[0].timestamp()))
    assert list(got) == [0.0, 10.0, 20.0, 30.0]


def test_the_openings_figure_prefers_the_fit_start(db, tmp_path):
    """A peak can sit inside a plume, so the fit's start is the better anchor.

    The 6 Nov bedroom opening falls steeply for 35 minutes as a breath plume
    disperses past the sensor, then settles into the slow decay the fit describes.
    Normalised at the peak, that opening very nearly overlays a night with five
    times its air-change rate; normalised where the fit begins, it does not.
    """
    from analysis import figures

    t0 = int(pd.Timestamp("2026-11-06 21:00", tz="UTC").timestamp())
    peak_at = t0 + 30 * 60
    fit_from = peak_at + 35 * 60
    obs = []
    for i in range(60):
        ts_i = t0 + i * 300
        if ts_i <= peak_at:
            value = 600 + 20 * i
        elif ts_i < fit_from:
            value = 720 - 3.0 * (ts_i - peak_at) / 60.0        # the plume, steep
        else:
            value = 615 * math.exp(-0.87 * (ts_i - fit_from) / 3600) + 358
        obs.append(Observation("scd40", "co2_ppm", "bedroom", ts_i,
                               round(value, 1), "measured", "ppm"))
        obs.append(Observation("scd40", "indoor_temp_c", "bedroom", ts_i,
                               18.5, "measured", "C"))
    result = CollectResult("scd40")
    result.observations = obs
    db.save_result(result)
    db.add_label("bedroom", "window", "open", peak_at, t0 + 59 * 300)
    db.save_decays([{"location": "bedroom", "start_ts": fit_from,
                     "end_ts": t0 + 59 * 300, "lambda_h": 0.87, "r2": 0.99,
                     "method": "window", "inferred": "ventilated",
                     "wind_ms": 0.76, "delta_t_c": 9.3, "labelled": "open"}])

    figures.style()
    figures.fig_openings(db, tmp_path, location="bedroom")
    assert (tmp_path / "09_openings.png").exists()

    # The anchor the figure uses: the fit's start, 35 minutes after the peak.
    fits = [r for r in db.decays("bedroom", min_r2=0.9) if r["labelled"] == "open"]
    assert len(fits) == 1
    assert int(fits[0]["start_ts"]) - peak_at == 35 * 60


# --- free asymptotes: what was the room actually emptying into? ----------------

def _decay(c_inf, c0, lam, minutes=60, step=5, start="2026-11-06 22:00", noise=0.0):
    """A synthetic exponential approach to c_inf, sampled on the 5-minute grid."""
    n = minutes // step + 1
    idx = pd.date_range(start, periods=n, freq=f"{step}min", tz="UTC")
    t = np.arange(n) * step / 60.0
    y = c_inf + (c0 - c_inf) * np.exp(-lam * t)
    if noise:
        y = y + np.random.default_rng(0).normal(0, noise, n)
    return pd.Series(y, index=idx)


def test_a_decay_towards_house_air_recovers_its_own_asymptote():
    """The whole point: a room emptying into the flat is not emptying into outdoors."""
    seg = _decay(c_inf=480.0, c0=1140.0, lam=3.0, minutes=70)
    got = features.fit_decay_free(seg, c_out=358.0)
    assert got is not None
    assert got["c_inf"] == pytest.approx(480.0, abs=5.0)
    assert got["lambda_h"] == pytest.approx(3.0, rel=0.05)


def test_forcing_the_wrong_asymptote_understates_the_rate():
    """Why this matters, stated as a measurement rather than an argument.

    The same curve fitted against outdoor air comes out far too slow, because the
    only way an exponential aimed at 358 ppm can pass through a trace that stops at
    480 is to be much shallower than the trace really is."""
    seg = _decay(c_inf=480.0, c0=1140.0, lam=3.0, minutes=70)
    forced = features.fit_decay(seg, c_out=358.0)
    assert forced["lambda_h"] < 0.6 * 3.0


def test_a_real_leak_to_outdoors_keeps_its_forced_asymptote():
    """The guard has to be two-sided or it is just a licence to raise every rate."""
    seg = _decay(c_inf=358.0, c0=1100.0, lam=1.2, minutes=180, noise=3.0)
    got = features.asymptote_test(seg, 358.0, 1.2)
    assert got["free"] is False
    assert "noise" in got["reason"] or "fits as well" in got["reason"]


def test_a_knee_is_not_an_asymptote():
    """Two regimes in one window look exactly like one fast decay that has arrived.

    This is the 3 November bedroom morning: a door-open flush from 2970 to 1305 in
    forty minutes, then a slow crawl once it shut. A one-hour slice of that fits as
    lambda=4.5 settling at 1246 ppm, and the only thing in the record that refutes it
    is the room walking straight through 1246 afterwards."""
    fast = _decay(c_inf=1240.0, c0=2970.0, lam=4.5, minutes=60)
    slow = _decay(c_inf=360.0, c0=1300.0, lam=0.25, minutes=180,
                  start="2026-11-03 09:50")
    whole = pd.concat([fast, slow])
    free = features.fit_decay_free(fast, c_out=358.0)
    assert free["c_inf"] > 1000.0                 # the bogus fit is available...
    got = features.asymptote_test(fast, 358.0, 1.0, later=whole)
    assert got["free"] is False                   # ...and the later record kills it
    assert "later reached" in got["reason"]


def test_the_asymptote_competes_during_the_offset_search_not_after_it():
    """The bug this was built to fix, as a regression test.

    Choosing the start offset by R-squared UNDER THE FORCED ASYMPTOTE silently
    discards the informative part of the curve: no offset can make an exponential
    aimed at 358 ppm fit a steep drop that stops at 480, so the best-conditioned
    forced fit is always the shallow far tail - and the reported rate comes out
    several times too low from a fit that looks excellent."""
    seg = _decay(c_inf=480.0, c0=1140.0, lam=3.0, minutes=70)
    best = features.best_decay_fit(seg, c_out=358.0)
    assert best is not None
    assert best["c_inf"] == pytest.approx(480.0, abs=10.0)
    assert best["lambda_h"] == pytest.approx(3.0, rel=0.1)
    # and it used the whole curve rather than retreating into the tail
    assert best["start"] == seg.index[0]


def test_an_asymptote_within_sensor_noise_of_outdoors_is_not_worth_a_parameter():
    seg = _decay(c_inf=368.0, c0=1100.0, lam=1.5, minutes=150, noise=2.0)
    got = features.asymptote_test(seg, 358.0, 1.5)
    assert got["free"] is False
    assert "sensor noise" in got["reason"]


# --- a rise measures the same lambda a decay does -----------------------------

def _rise(c_inf, c0, lam, hours, step=5, start="2026-11-07 01:00"):
    """A room filling towards c_inf, sampled on the 5-minute grid."""
    n = int(hours * 60 / step) + 1
    idx = pd.date_range(start, periods=n, freq=f"{step}min", tz="UTC")
    t = np.arange(n) * step / 60.0
    return pd.Series(c_inf + (c0 - c_inf) * np.exp(-lam * t), index=idx)


def test_a_rise_reports_how_far_up_its_own_curve_it_got():
    """Lambda and the ceiling trade off, and coverage is what says whether they can
    be told apart. A room an hour into a nine-hour climb is drawing a straight line,
    and a straight line is equally well a slow climb to a low ceiling or a fast one
    to a high one."""
    far = features.fit_rise(_rise(5200.0, 620.0, 0.25, hours=9), 358.0)
    near = features.fit_rise(_rise(5200.0, 620.0, 0.25, hours=2), 358.0)
    assert far["covered"] > 0.85
    assert near["covered"] < 0.45


def test_a_full_nights_rise_recovers_the_rate_it_was_built_from():
    fit = features.fit_rise(_rise(5200.0, 620.0, 0.25, hours=9), 358.0)
    assert fit["lambda_h"] == pytest.approx(0.25, rel=0.05)
    assert fit["ceiling"] == pytest.approx(5200.0, rel=0.05)


def test_emptying_and_filling_agree_and_are_pooled(db):
    """The cross-check the whole model rests on.

    A decay and a rise share no assumption but the mass balance: one is short, steep
    and vulnerable to the asymptote being wrong, the other long, gently curved and
    vulnerable to lambda trading against its ceiling. Nothing makes them agree, so
    when they do the number belongs to the room rather than to the method."""
    now = simulate(db, days=6)
    learn(db, "bedroom", days=7, now=now)
    detail = json.loads(db.params("bedroom")["lambda_sealed"]["detail"])
    assert detail.get("rises"), "the simulated nights produced no usable rise"
    if detail.get("rises"):
        assert detail["decays"] and detail["rises"]
        assert detail["disagree"] is False, detail["agreement"]
        assert "mass balance" in detail["agreement"]
        assert db.params("bedroom")["lambda_sealed"]["n"] == (
            detail["decays"] + detail["rises"])
        # Both routes have to land on the rate the room was actually built with, or
        # agreeing means only that they are wrong in the same direction.
        assert detail["decay_median"] == pytest.approx(TRUE["lam_sealed"], rel=0.1)
        assert detail["rise_median"] == pytest.approx(TRUE["lam_sealed"], rel=0.1)


def test_a_rise_too_straight_to_read_is_counted_but_not_used(db):
    """Being excluded has to be visible, or a quietly dropped measurement looks the
    same as one that was never taken."""
    from analysis.room_model import MIN_RISE_COVER

    rises = [{"lambda_h": 0.25, "r2": 0.99, "covered": 0.9},
             {"lambda_h": 9.90, "r2": 0.99, "covered": MIN_RISE_COVER - 0.2}]
    usable = [r["lambda_h"] for r in rises
              if r["covered"] >= MIN_RISE_COVER and r["r2"] >= 0.95]
    assert usable == [0.25], "a near-linear rise reached the pooled rate"


# --- the one number that did not come from the sensors ------------------------

def test_a_measured_volume_survives_relearning(db):
    """learn() clears what it is about to re-derive. It cannot re-derive a tape measure.

    This failed in the field, not in theory: the volume was recorded, the cross-check
    printed, and the collector's next analysis pass five minutes later deleted it.
    The delete was scoped to the room rather than to the parameters being rewritten,
    so it took the external calibration with it and the room quietly went back to
    checking itself. Nothing logged anything."""
    from analysis.room_model import measured_volume

    now = simulate(db, days=4)
    learn(db, "bedroom", days=5, now=now)
    db.set_param("bedroom", "volume_measured_m3", 18.0, None, 1, "m3",
                 {"from": "tape measure"})
    assert measured_volume(db, "bedroom") == 18.0

    learn(db, "bedroom", days=5, now=now)
    assert measured_volume(db, "bedroom") == 18.0, "relearning deleted the tape measure"
    assert "c_out" in db.params("bedroom")      # and the rest is still re-derived


def test_a_measured_volume_is_what_the_model_uses(db):
    now = simulate(db, days=4)
    learn(db, "bedroom", days=5, now=now)
    inferred = RoomModel(db, "bedroom").volume_m3
    db.set_param("bedroom", "volume_measured_m3", 18.0, None, 1, "m3", {})
    model = RoomModel(db, "bedroom")
    assert model.volume_m3 == 18.0
    assert model.volume_basis == "measured"
    assert model.volume_inferred_m3 == inferred     # kept, as the thing being checked


def test_the_volume_cross_check_can_actually_fail(db):
    """A one-sided check is not a check.

    The first version of this only observed that a sleeping rate should come out
    below the seated 18.7 L/h, so every value in (0, 18.7) read as confirmation -
    the same cannot-fail shape this codebase already criticises in fit_open_wind,
    written a day later by the same hand. The real bedroom fails it: 960 ppm/h in
    18 m3 demands 17.3 L/h, and an adult asleep produces 11-15."""
    from analysis.room_model import SLEEP_CO2_LH, volume_check

    db.set_param("r", "volume_measured_m3", 18.0, None, 1, "m3", {})

    bad = volume_check(db, "r", 960.2, 6, source_stderr=90.4)
    assert bad["agrees"] is False
    assert bad["implied_co2_lh"] == pytest.approx(17.3, abs=0.1)
    assert "FAILS" in bad["note"]

    # A source implying a rate inside the band passes, so the test has two sides.
    mid = (SLEEP_CO2_LH[0] + SLEEP_CO2_LH[1]) / 2 * 1000.0 / 18.0
    good = volume_check(db, "r", mid, 6, source_stderr=mid * 0.05)
    assert good["agrees"] is True
    assert "FAILS" not in good["note"]


def test_the_inferred_volume_carries_the_sources_error(db):
    """volume = 18700/S exactly, so its relative error IS the source's. Storing None
    invited an 8% gap to be read as agreement when the interval is wider than it."""
    from analysis.room_model import volume_check

    db.set_param("r", "volume_measured_m3", 18.0, None, 1, "m3", {})
    chk = volume_check(db, "r", 1000.0, 6, source_stderr=100.0)
    assert chk["inferred_lo"] == pytest.approx(18.7 * 0.9, abs=0.1)
    assert chk["inferred_hi"] == pytest.approx(18.7 * 1.1, abs=0.1)


# --- a cut that any sample would produce is not a measurement -----------------

def test_a_split_must_beat_one_cluster():
    """split_point used to find a split in essentially any sample.

    Minimising within-group variance over every cut ALWAYS returns something, and
    the ratio-of-medians guard does not catch it because a heavy right tail clears
    a factor of two on its own. Drawn from a SINGLE log-normal with this room's
    observed spread, the old version called the sample bimodal 95% of the time at
    n=6 and 99% at n=11 - it was detecting dispersion, not windows. And it decided
    lambda_sealed."""
    rng = np.random.default_rng(1)
    false = sum(split_point(np.exp(rng.normal(-1.0, 1.03, 11))) is not None
                for _ in range(60))
    assert false / 60 < 0.15, f"one cluster declared bimodal {false}/60 times"


def test_a_real_split_is_still_found():
    """The guard has to be two-sided or it just disables the feature."""
    sealed = [0.28, 0.30, 0.31, 0.33, 0.29, 0.27]
    opened = [2.9, 3.1, 3.0, 3.2]
    cut = split_point(sealed + opened)
    assert cut is not None
    assert max(sealed) < cut < min(opened)


def test_this_rooms_lambdas_are_one_cluster(db):
    """The bedroom's own fitted values, which the old cut split at 0.615/h."""
    real = [0.147, 0.198, 0.211, 0.348, 0.381, 0.390, 0.471, 0.804, 1.197,
            2.575, 4.547]
    assert split_point(real) is None, "a heavy tail was mistaken for a second mode"


def test_robust_never_claims_infinite_precision():
    """A zero MAD is reachable with four values and wrote stderr=0.0 to the database."""
    from analysis.room_model import _robust

    med, err, n = _robust([0.5, 0.5, 0.5, 2.0])
    assert med == 0.5 and n == 4
    assert err is not None and err > 0.0
    med, err, _n = _robust([0.3, 0.3, 0.31])
    assert err > 0.0


# --- sealed must mean sealed --------------------------------------------------

def test_a_decay_that_settled_on_house_air_is_not_sealed(db):
    """lambda_sealed is the envelope leaking to OUTDOOR air; that is what every
    ceiling is built from and what fit_weather regresses on weather. A decay whose
    own fit says it was heading for 1029 ppm against an outdoor 358 measured a door.
    Nine of eighteen desk decays were such rows and one was the bedroom's median."""
    from analysis import features

    now = simulate(db, days=4)
    learn(db, "bedroom", days=5, now=now)
    rows = [dict(r) for r in db.decays("bedroom")]
    assert rows
    for r in rows:
        if r["inferred"] == "sealed":
            assert (r["c_inf"] is None
                    or r["c_inf"] - r["c_out"] < features.ASYM_MIN_ABOVE), r


def test_c_out_counts_baseline_observations_not_co2_samples(db):
    """n said 1169 for a c_out measured in a single labelled airing, which is the
    opposite of what the n column is for."""
    now = simulate(db, days=4)
    learn(db, "bedroom", days=5, now=now)
    row = db.params("bedroom")["c_out"]
    samples = len(features.series(db, "co2_ppm", "bedroom").dropna())
    assert row["n"] < samples, "n is still the CO2 sample count"
    assert row["n"] <= 20


# --- fits that cannot be believed are not stored ------------------------------

def test_fit_weather_refuses_an_impossible_intercept(db):
    """a is still-air infiltration SQUARED, so a < 0 is impossible rather than weak.
    The desk stored a = -0.709 with every coefficient inside its own stderr."""
    from analysis.room_model import fit_weather

    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        rows.append({"location": "r", "start_ts": 10_000 * i,
                     "end_ts": 10_000 * i + 3600,
                     "lambda_h": float(0.4 + 0.02 * rng.normal()), "r2": 0.99,
                     "method": "departure", "inferred": "sealed",
                     "wind_ms": 3.0 + 0.1 * i,       # almost no span
                     "delta_t_c": 9.5 + 0.2 * (i % 2),   # 0.2 C of span
                     "labelled": None})
    db.save_decays(rows)
    out = fit_weather(db, "r")
    assert out.get("refuted"), out
    assert "dT" in out["refuted"] or "t = " in out["refuted"]
    assert db.params("r")["lambda_wind_c"]["n"] == 0
    assert db.params("r")["lambda_wind_c"]["value"] is None


def test_decays_can_be_restricted_to_a_window(db):
    """learn() only deletes rows inside the period it re-fitted, so older rows keep
    the lambda they were fitted against an older c_out - 107 ppm away, here."""
    db.save_decays([{"location": "r", "start_ts": 1000, "end_ts": 2000,
                     "lambda_h": 0.3, "r2": 0.99, "method": "window",
                     "inferred": "sealed", "labelled": None},
                    {"location": "r", "start_ts": 900_000, "end_ts": 901_000,
                     "lambda_h": 0.4, "r2": 0.99, "method": "window",
                     "inferred": "sealed", "labelled": None}])
    assert len(db.decays("r")) == 2
    assert len(db.decays("r", start=500_000)) == 1
    assert len(db.decays("r", end=500_000)) == 1


# --- compared with what? ------------------------------------------------------

def test_every_forecast_records_the_rival_that_assumes_no_change(db):
    """Persistence is written at issue time, not reconstructed at scoring time, so
    it is a genuine out-of-sample forecast rather than a retrospective one."""
    now = simulate(db, days=3)
    analysis_human.run(db, now=now)
    with db.conn() as c:
        rows = c.execute("SELECT predicted, naive FROM predictions "
                         "WHERE naive IS NOT NULL").fetchall()
    assert rows, "no forecast carried a persistence baseline"
    assert all(r["naive"] > 0 for r in rows)


def test_skill_is_reported_against_persistence(db):
    """The comparison the project's headline claim needs and did not have."""
    base = {"location": "r", "metric": "co2_ppm", "horizon_s": 1800,
            "basis": "learned"}
    t0 = int(time.time()) - 3600
    rows = []
    for i in range(6):
        rows.append(dict(base, issued_at=t0 + i * 100, target_ts=t0 + 1800 + i * 100,
                         predicted=900.0, naive=800.0, actual=810.0,
                         scored_at=t0 + 3600))
    db.add_predictions(rows)
    got = db.forecast_skill()
    assert len(got) == 1
    r = got[0]
    assert r["mae"] == pytest.approx(90.0)        # model is 90 out
    assert r["naive_mae"] == pytest.approx(10.0)  # persistence is 10 out
    assert r["skill"] < 0, "a model this much worse must score negative skill"
