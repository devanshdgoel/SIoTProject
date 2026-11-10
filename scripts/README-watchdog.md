# Watchdog alerts on a phone

The watchdog runs every five minutes from Windows Task Scheduler and checks the
database file — never the collector itself, so it cannot share a failure mode with
the thing it watches. It writes `data/watchdog.log` and exits non-zero on trouble.
That is only useful if somebody looks at it, so it can also push.

## Route 1 — ntfy (set up, recommended)

No account, no API key, nothing to expire. A topic name *is* the configuration.

1. Install **ntfy** on the phone (App Store / Play Store / F-Droid).
2. Tap **+**, subscribe to the topic in `.env` under `SIOT_NTFY_TOPIC`.
3. Check it:

```bash
python scripts/watchdog.py --test
```

A notification should arrive within a second or two.

The topic is a random string because **ntfy.sh topics are public to anyone who
knows the name** — it is unguessable rather than authenticated. The messages say
only which sensor is late and by how long; there is no data and no location in
them. If that is still too much, run your own ntfy server and set `SIOT_NTFY_URL`,
or use email below instead.

## Route 2 — email, and phone SMS gateways

Add to `.env`:

```
SIOT_MAIL_TO=you@example.com
SIOT_MAIL_FROM=you@gmail.com
SIOT_MAIL_HOST=smtp.gmail.com
SIOT_MAIL_PORT=587
SIOT_MAIL_PASS=<16-character Gmail app password>
```

Use an **app password** (Google Account → Security → 2-Step Verification → App
passwords), never the account password. `SIOT_MAIL_TO` can be a carrier
email-to-SMS gateway if you want a text message rather than a push.

Both routes are tried on every alert, because the point of having two is that
either may be the one that fails.

## What you will actually receive

| When | Priority |
|---|---|
| A sensor, the collector or the analysis pass goes stale | high — breaks through Do Not Disturb |
| Everything comes back | default |
| Once a day at `SIOT_HEARTBEAT_HOUR`, all well | low — silent, just a record |

Alerts fire **only on a change of state**, not every five minutes: a watchdog that
repeats itself is muted within a day, and a muted watchdog is worse than none
because it looks like cover.

The daily heartbeat exists for the opposite reason. A monitor that only ever
speaks up on failure is indistinguishable from a monitor that has itself died —
with the heartbeat, no message by mid-morning is itself the alarm. Set
`SIOT_HEARTBEAT_HOUR=` (blank) to turn it off.

Failing to notify is never fatal: the reason is appended to `data/watchdog.log`
as `NOTIFY-FAILED`, and the log and exit status behave exactly as before.

## The scheduled task

```bash
schtasks /Query /TN SIoTWatchdog
```

It reads settings from `.env`, not from the environment, because a scheduled task
inherits no shell. Editing `.env` is enough; the task does not need recreating.
