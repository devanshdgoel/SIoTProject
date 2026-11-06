"""
event_rules.py

Turning a measurement into an event worth acting on.

A bare threshold is useless in practice. Sitting at 1,001 ppm it fires, at 999 it
clears, and a pet driven straight off that flickers all evening. Worse, it fires on
a single spike that was never real, and keeps firing on a sensor that stopped
reporting an hour ago because its last reading happened to be high.

Five guards, each answering a different failure:

  persistence  the condition has to hold for a while before it counts, so one odd
               sample is not an event
  hysteresis   it starts at one level and stops at a lower one, so there is no value
               the signal can sit on and chatter
  freshness    stale data produces nothing, because the last thing a dead sensor
               said is not evidence about now
  confidence   how complete the window was and how far past the line it went, so an
               event can be reported with how sure it is rather than as a bare fact
  cooldown     having said a thing, wait before saying it again

Everything here is a pure function of its arguments. Nothing reads the database or
the clock, so the whole rule set is testable against invented series, and the same
code decides live and during a replay.
"""

from dataclasses import dataclass

GRID_S = 300


@dataclass(frozen=True)
class Rule:
    """One condition worth watching.

    `enter` and `exit_at` deliberately differ: a band between them is where nothing
    changes, which is what stops the flapping. For a rule that fires when a value
    falls rather than rises, set below=True and put `exit_at` above `enter`."""

    name: str
    metric: str
    enter: float
    exit_at: float
    persistence_s: int = 900
    cooldown_s: int = 3600
    stale_after_s: int = 900
    severity: str = "warn"
    below: bool = False

    def breached(self, value):
        return value <= self.enter if self.below else value >= self.enter

    def released(self, value):
        return value >= self.exit_at if self.below else value <= self.exit_at


@dataclass
class Assessment:
    """What a rule makes of the data right now."""

    rule: str
    on: bool                  # is the condition considered true
    fire: bool                # should anything be said about it now
    reason: str
    confidence: float = 0.0
    held_s: int = 0
    margin: float = 0.0
    completeness: float = 1.0
    stale: bool = False


def _trailing(samples, now, span_s):
    return [(t, v) for t, v in samples if now - span_s <= t <= now and v is not None]


def confidence(rule, window, margin, expected):
    """How much to believe this event.

    Three things pull it down: a window with samples missing, a value that only just
    crossed the line, and a short record. A recommendation carrying 0.4 should be
    phrased differently from one carrying 0.95, which is why this is a number and
    not a boolean."""
    completeness = min(1.0, len(window) / max(1, expected))
    reach = min(1.0, abs(margin) / max(1.0, abs(rule.enter - rule.exit_at)))
    return round(max(0.0, min(1.0, 0.45 * completeness + 0.45 * reach + 0.10)), 3)


def assess(rule, samples, now, was_on=False, last_fired_at=None, step_s=GRID_S):
    """Apply one rule to a series.

    `samples` is [(unix_ts, value), ...] oldest first. `was_on` and `last_fired_at`
    carry the only state there is, so the caller can keep it wherever it likes - in
    this project, in the events table."""
    fresh = [(t, v) for t, v in samples if v is not None]
    if not fresh:
        return Assessment(rule.name, False, False, "no data at all", stale=True)

    newest_t, newest_v = fresh[-1]
    if now - newest_t > rule.stale_after_s:
        # A dead sensor's last reading is not evidence about the present moment.
        return Assessment(rule.name, False, False,
                          f"newest reading is {int((now - newest_t) / 60)} min old",
                          stale=True)

    window = _trailing(fresh, now, rule.persistence_s)
    expected = max(1, rule.persistence_s // step_s)
    margin = newest_v - rule.enter if not rule.below else rule.enter - newest_v

    if was_on:
        # Already on: it takes a drop past the lower bound to turn it off.
        if window and all(rule.released(v) for _t, v in window):
            return Assessment(rule.name, False, False, "fell back past the release level",
                              held_s=0, margin=margin, completeness=len(window) / expected)
        # Still on. Worth repeating, but only once the cooldown has run - a room that
        # stays bad all evening deserves an occasional reminder, not a running commentary.
        quiet = last_fired_at is not None and now - last_fired_at < rule.cooldown_s
        return Assessment(rule.name, True, not quiet,
                          f"on, but said {int((now - last_fired_at) / 60)} min ago" if quiet
                          else "still on, and due a reminder",
                          confidence(rule, window, margin, expected),
                          held_s=int(now - window[0][0]) if window else 0, margin=margin,
                          completeness=len(window) / expected)

    held = [(t, v) for t, v in window if rule.breached(v)]
    if len(held) < expected or len(window) < expected:
        return Assessment(rule.name, False, False,
                          f"breached for {len(held)} of {expected} samples - not yet persistent",
                          held_s=len(held) * step_s, margin=margin,
                          completeness=len(window) / expected)

    if last_fired_at is not None and now - last_fired_at < rule.cooldown_s:
        # True, but recently said. Hold the state without speaking again.
        return Assessment(rule.name, True, False,
                          f"on, but said {int((now - last_fired_at) / 60)} min ago",
                          confidence(rule, window, margin, expected),
                          held_s=rule.persistence_s, margin=margin,
                          completeness=len(window) / expected)

    return Assessment(rule.name, True, True,
                      f"held past {rule.enter:.0f} for {rule.persistence_s // 60} min",
                      confidence(rule, window, margin, expected),
                      held_s=rule.persistence_s, margin=margin,
                      completeness=len(window) / expected)


# The rules this project actually runs. Thresholds are the conventional indoor-air
# action points; the release levels sit far enough below to leave a dead band the
# signal cannot chatter across.
INDOOR_CO2 = (
    Rule("co2_high", "co2_ppm", enter=1400.0, exit_at=1150.0,
         persistence_s=900, cooldown_s=3600, severity="high"),
    Rule("co2_warn", "co2_ppm", enter=1000.0, exit_at=850.0,
         persistence_s=1200, cooldown_s=5400, severity="warn"),
)
OUTDOOR_PM = (
    Rule("pm25_high", "pm2_5", enter=35.0, exit_at=25.0,
         persistence_s=3600, cooldown_s=6 * 3600, severity="warn"),
)


def assess_all(rules, samples, now, state=None, step_s=GRID_S):
    """Run several rules over one series; `state` is {rule_name: (was_on, last_fired)}."""
    state = state or {}
    out = []
    for rule in rules:
        was_on, last = state.get(rule.name, (False, None))
        out.append(assess(rule, samples, now, was_on, last, step_s))
    return out
