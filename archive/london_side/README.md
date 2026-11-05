# London-scale side, parked

Collectors and analysis for the city around the flat, as opposed to the air inside
it. Parked on 4 Nov 2026 so the project could focus on one coherent question:
indoor air and the body's response to it.

| file | what it did | state when parked |
|---|---|---|
| `grid_carbon.py` | NESO Carbon Intensity, GB and London, with a 48 h forecast | worked; 17 days collected |
| `thames_water.py` | Thames Water storm-overflow discharges | never ran; no credentials, schema unverified |
| `laqn.py` | London Air Quality Network reference analysers | written and unit-tested; the API was unreachable from this network for days, 46 consecutive SSL failures |
| `analysis_london.py` | event detection across the three | docstring only |

## Why each one stopped

**grid_carbon** and **thames_water** measure the user's *impact* on the world.
The indoor work measures the world's *exposure* on the user. Both are legitimate,
but only the second is what the analysis layer was built around, and six pages of
report will not hold both.

**laqn** is the one worth reviving. Open-Meteo's air quality is CAMS model output
on a 0.1 degree grid, about 11 km by 7 km at London's latitude, so the whole
borough shares one value and the home and Imperial fall in the same cell. LAQN
would have given real analyser readings a few hundred metres away, and comparing
measured against modelled would have quantified the model's local bias. The API
simply would not answer: the TLS handshake died before a certificate was offered,
from two independent networks over several days.

## Bringing them back

The collectors still satisfy the `collect(db=None) -> CollectResult` contract, so
reviving one means moving the file back into `collectors/`, re-adding it to
`ALL_COLLECTORS` in `main.py` and naming it in `COLLECTORS_ENABLED`. Their unit
tests are in `test_london_collectors.py` here, and in git history up to 2e0ee18.

Before reviving LAQN, check the API answers at all:

    curl -sS https://api.erg.ic.ac.uk/AirQuality/Information/Species/Json | head

If it still refuses, DEFRA's AURN feed at uk-air.defra.gov.uk was reachable and
carries the same London reference sites, including North Kensington.

## Data already collected

Nothing was deleted. `observations` still holds 17 days of `carbon_intensity`
rows, and the `overflow_status` and `overflow_alerts` tables remain in the schema
(empty). Re-enabling a collector resumes into the same series.
