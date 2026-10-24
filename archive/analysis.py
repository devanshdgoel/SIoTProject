"""
analysis.py — the project's "brain", kept separate from collection and storage.

For now it holds ONE real computation: a habitat-stress score for Marina's
location, derived from the latest weather + air-quality reading. This is the
number that will eventually decide how "sick" the pet looks.

It is deliberately simple and TRANSPARENT: the overall score is a weighted
blend of named sub-scores (cold, wind, storm, air), and we return every
sub-score so the dashboard can show the breakdown rather than a magic number.

Later this will grow to:
  - use a TREND (is it getting worse over hours?) not just the latest point,
  - fuse in YOUR indoor air (SCD40) and physiology (HRV/HR) for the local side.
Those hooks are left as clearly-marked placeholders at the bottom.
"""

# --- tiny helpers -----------------------------------------------------------

def _clamp(x, lo=0.0, hi=100.0):
    """Keep a score inside [lo, hi]."""
    return max(lo, min(hi, x))


def _ramp(value, good, bad):
    """
    Map a raw reading onto 0..100.
      value <= good  -> 0   (no stress)
      value >= bad   -> 100 (maximum stress)
      in between     -> linear
    Works whether 'bad' is higher than 'good' (e.g. wind) or lower (e.g. temp),
    because we handle the direction with a simple check.
    """
    if good == bad:
        return 0.0
    frac = (value - good) / (bad - good)
    return _clamp(frac * 100.0)


# --- the sub-scores ---------------------------------------------------------
# Each returns 0 (fine) .. 100 (harsh). All take plain numbers so they're easy
# to unit-test on their own.

def cold_stress(feels_like_c):
    """Cold is the main habitat stressor for an overwintering stork.
    Comfortable at ~15C+, harsh by ~5C, severe by -5C."""
    if feels_like_c is None:
        return 0.0
    return _ramp(feels_like_c, good=15.0, bad=-5.0)   # lower temp -> higher stress


def wind_stress(wind_speed_mps):
    """Calm (<2 m/s) = 0; a strong gale (~15 m/s) = 100."""
    if wind_speed_mps is None:
        return 0.0
    return _ramp(wind_speed_mps, good=2.0, bad=15.0)


def storm_stress(weather_desc):
    """Rough weather read straight off the text description.
    A lookup keeps it readable and easy to extend."""
    if not weather_desc:
        return 0.0
    d = weather_desc.lower()
    table = [
        ("thunderstorm", 100),
        ("snow", 90),
        ("freezing", 90),
        ("hail", 85),
        ("rain", 60),
        ("drizzle", 35),
        ("fog", 25),
        ("overcast", 15),
        ("cloud", 8),      # "partly cloudy" / "overcast" partial match
    ]
    for key, score in table:
        if key in d:
            return float(score)
    return 0.0   # clear / sunny


def air_stress(european_aqi, pm2_5):
    """Open-Meteo European AQI is already a 0..100+ band scale, so we use it
    directly and just cap it. We also nudge it up if PM2.5 is high, as a
    second opinion."""
    aqi_score = _clamp(european_aqi if european_aqi is not None else 0.0)
    pm_score = _ramp(pm2_5, good=5.0, bad=50.0) if pm2_5 is not None else 0.0
    return max(aqi_score, pm_score)   # whichever says "worse"


# --- the overall blend ------------------------------------------------------
# Weights say how much each stressor matters. They sum to 1.0 so the overall
# score stays on a 0..100 scale. Cold + storm dominate for a wintering bird.
WEIGHTS = {
    "cold":  0.40,
    "storm": 0.30,
    "wind":  0.20,
    "air":   0.10,
}


def habitat_stress(weather_row):
    """
    Take one weather_reading row (a dict) and return the habitat-stress result:

        {
          "overall": 0..100,         # the headline "sickness" driver
          "label":   "healthy" | "uneasy" | "unwell" | "sick",
          "components": { cold, storm, wind, air },   # each 0..100
          "weights": {...},          # so the UI can explain the blend
        }

    Returns None if there's no weather row to score.
    """
    if not weather_row:
        return None

    comp = {
        "cold":  round(cold_stress(weather_row.get("feels_like_c")), 1),
        "storm": round(storm_stress(weather_row.get("weather_desc")), 1),
        "wind":  round(wind_stress(weather_row.get("wind_speed_mps")), 1),
        "air":   round(air_stress(weather_row.get("aqi"), weather_row.get("pm2_5")), 1),
    }

    overall = sum(comp[k] * WEIGHTS[k] for k in WEIGHTS)
    overall = round(_clamp(overall), 1)

    # A friendly label band for the face / headline.
    if overall < 20:
        label = "healthy"
    elif overall < 45:
        label = "uneasy"
    elif overall < 70:
        label = "unwell"
    else:
        label = "sick"

    return {
        "overall": overall,
        "label": label,
        "components": comp,
        "weights": WEIGHTS,
    }


# ===========================================================================
# PLACEHOLDERS — the local (you) side of the story. Wired in later.
# ===========================================================================
def indoor_air_health(indoor_row):
    """TODO (after SCD40 arrives, ~late Oct): score YOUR desk air from CO2 /
    temp / humidity, and detect an open window from a CO2 drop + temp shift."""
    return None


def personal_sensitivity(heart_series, air_series, movement_series):
    """TODO: correlate HR/HRV/respiration against air quality, CONTROLLING for
    movement, so activity isn't mistaken for a pollution response."""
    return None
