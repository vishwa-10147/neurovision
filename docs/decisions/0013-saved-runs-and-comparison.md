# ADR 0013: Saved external runs, run comparison and metric export

Status: accepted

## Context

External runs (ADR 0006) lived only in the backend's memory, so a restart
lost every run's history, including runs that had finished hours earlier.
Milestone 8 asks for experiment metadata and metrics history, and users who
tune a model want to put two runs side by side and take the numbers
elsewhere. Architecture projects already persist as versioned JSON files
(ADR 0012); there are still no user accounts.

## Decision

- Save each run as one versioned JSON file (`schema_version: 1`) in
  `NEUROVISION_RUNS_DIR` (default `data/runs`), written atomically through a
  temporary file and `os.replace`, like projects.
- A file holds the run's metadata (`RunInfo` fields that describe the run) and
  its bounded replay history: `training.started`, `training.step`,
  `run.alert` and the terminal event. `run.command` events are left out
  because no script can apply them after a restart. The history keeps the
  existing bounds: 10,000 steps, activations and visuals only on the latest
  step that had them, per-layer health on the latest 200.
- The write token is never stored, in any form. A reloaded run has no token
  digest, so every write is refused (`409`, saying the run is read-only).
  Its `model_id` is cleared: imported models live in memory, so the linked
  graph is gone after a restart.
- A running run is saved when it starts, at most every 5 seconds while it
  logs (writing the whole history on every batch would cost O(n²) for long
  runs), when it finishes or fails, and when the backend shuts down. A save
  that fails (full or read-only disk) is logged and does not stop the run.
- On startup, every readable file is validated (the stored events with the
  same event schema as live ones) and loaded oldest first. A run whose status
  was still `running` gets a `training.failed` event saying the backend
  restarted, and is saved again so this happens once. Unreadable or invalid
  files are skipped. Retention stays at the 32 most recent finished runs; a
  run dropped from memory also has its file deleted.
- Comparison happens in the browser: the Live Runs list has a **Compare**
  checkbox per run (up to 6). The comparison opens one event stream per
  selected run (the existing WebSocket, which replays finished runs in full)
  and overlays the curves in the existing `RunMetricCharts`, by step, one
  color per run, with validation variants dashed.
- `GET /runs/{run_id}/metrics.csv` exports one row per logged step: `step`,
  `epoch`, then each metric. Metric names that a spreadsheet would read as a
  formula are prefixed with `'`.

## Alternatives considered

- **SQLite or PostgreSQL:** better for querying across runs, but the access
  model (accounts, ownership) still does not exist, and one file per run is
  enough for 32 runs. Same reasoning as ADR 0012.
- **Append-only JSON Lines per run:** cheaper writes, but the history is
  rewritten in place (activations, visuals and older health details are
  dropped as new steps arrive), so the file would either grow past the
  in-memory bounds or need compaction. A throttled full write keeps the file
  identical to what is replayed.
- **Keeping running runs as `running` after a restart:** the script's token
  died with the old process, so it could never log again; the run would only
  time out after 5 minutes with a less accurate message.
- **A JSON endpoint for comparison data:** the WebSocket replay already gives
  every step, and also keeps a running run's curves live in the comparison.

## Consequences

- Runs streamed by a script are still listed and replayable after the
  backend restarts; a run that was training at the time shows why it ended.
- After a crash (no shutdown), up to the last 5 seconds of steps of a running
  run can be missing.
- A finished run's file is at most a few MB (10,000 steps with metrics, health
  and system numbers), so the directory stays small with 32 runs.
- Like projects, saved runs are readable by anyone who can reach the backend;
  authentication remains a prerequisite for a shared deployment.
