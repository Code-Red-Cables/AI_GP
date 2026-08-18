# Reference session artifacts

Human / classical practice artifacts from a finished full-course session.
They are **not** the timed policy submission and are not a scoreboard.

| File | Contents |
|---|---|
| `events_best.txt` | Events log for the session |
| `pilot_best.csv` | Pilot tuning CSV |
| `attitude_pb.json` / `attitude_best.json` | Open-loop attitude tape of one finished lap |
| `best_run.json` | Scorer metadata |

Open-loop tape replay is not a reliable full-lap path on this sim (wind,
density, and spawn vary). Use a tape as a supervised prefix only — see
[`MANUAL_INSTRUCTIONS.md`](../../MANUAL_INSTRUCTIONS.md).

The timed client is `FLIGHT_MODE=policy` with
`models/policy_seed_17.pt`.
