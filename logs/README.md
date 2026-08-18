# Logs

Flight CSVs and event files were omitted from the team zip (they are large
and session-specific). Recreate the folders locally:

| Folder | What belongs here |
|---|---|
| `seed/` | Finished all-human laps (gates 0–17 only) |
| `coach/` | Finished HG-DAgger laps (gates 0–17 only) |
| `best/` | Human PB archive |

Do not dump DNFs, pad sits, or early quits into `seed/` or `coach/`. The
trainer globs every file in those folders. See the main README.
