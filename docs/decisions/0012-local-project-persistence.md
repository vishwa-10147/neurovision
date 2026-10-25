# ADR 0012: Versioned local project persistence

Status: accepted

## Context

Architecture Studio definitions were held only in browser memory and the
backend's in-memory model repository. A backend restart therefore lost the
definition, even though the configuration is small, portable and easy to
represent as JSON. Multi-user ownership and authentication are not implemented
yet, so introducing PostgreSQL would add deployment complexity before the
access model exists.

## Decision

- Store architecture projects as versioned JSON documents in a configurable
  local directory, `NEUROVISION_PROJECTS_DIR`, defaulting to `data/projects`.
- Expose `GET /projects`, `POST /projects`, and `GET /projects/{project_id}`.
- Validate project configuration through the existing strict
  `DenseNetworkConfig` schema.
- Write files atomically through a temporary file and `os.replace`.
- Keep list responses small by returning project summaries without the full
  configuration; reopen returns the complete document.
- Use an explicit `schema_version` field from the first version so migrations
  can be added before the format changes.

## Alternatives considered

- **PostgreSQL now:** premature without authentication, ownership or multi-user
  requirements.
- **Browser localStorage only:** does not survive browser/profile changes and
  cannot be reopened by scripts or future backend features.
- **Reuse the in-memory model repository:** model records contain runtime
  PyTorch state and are not a stable project format.

## Consequences

- Architecture definitions survive backend restarts on the same machine.
- The stored JSON is human-readable and easy to back up.
- Projects are not protected from other users who can access the backend;
  authentication and ownership are required before exposing this endpoint on a
  shared or deployed service.
- Experiment metrics, run comparison and model binary storage remain later
  slices of Milestone 8.
