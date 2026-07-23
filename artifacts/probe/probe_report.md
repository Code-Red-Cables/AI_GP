# Simulator Probe Report

Measured over 30s of wall clock.

## Measured rates

| stream | mean Hz | mean dt (ms) | std dt (ms) | p95 dt (ms) | count |
|---|---|---|---|---|---|
| camera | 30.00 | 33.34 | 2.98 | 36.33 | 900 |
| attitude | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| highres_imu | 114.23 | 8.75 | 0.73 | 10.01 | 3400 |
| actions_sent | 48.47 | 20.63 | 0.38 | 21.19 | 1450 |

## Camera health

- frames completed: 34006
- incomplete frames dropped: 0
- duplicate frame ids: 33106

## Privileged / race state

- race_status present: True
- track gates received: 0
- position telemetry available (VQ2 expects False): False

## Notes
- Compare camera mean Hz against the spec's claimed 30 Hz.
- If std dt is large, the decision rate / watchdogs must tolerate jitter.
- `position_available=True` would enable privileged dense-progress reward.
