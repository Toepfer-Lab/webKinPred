# Multi-Target Job Progress Redesign

## Problem Summary

The current progress system is job-global, not target-stage-aware.

- Every target run resets shared `Job` counters (`molecules_processed`, `total_predictions`, etc.).
- Embedding progress is also single-slot per job and is overwritten by the next target.
- The Track Job page therefore shows one target at a time and appears to "reset" when moving to the next target.

This is why multi-target jobs are hard to understand in the current UI.

## Goals

1. Show progress for **all selected targets** under **Results**.
2. Keep finished target progress visible (do not reset visually).
3. Show per-target **Protein Embeddings** progress as well.
4. Remove the separate **Live Progress** section (duplicate information).
5. Keep API compatibility during rollout.

## Non-Goals

1. Changing prediction execution order (still sequential by canonical target order).
2. Rewriting prediction methods themselves.
3. Removing legacy job-level counters immediately.

## Root Causes in Current Code

1. Shared job counters are reset per method run:
   - `api/prediction_engines/generic_subprocess.py` (`_initialise_job_progress`)
   - Same reset pattern in `dlkcat.py`, `unikp.py`, `kinform.py`, `turnup.py`, `eitlem.py`
2. Subprocess progress writes only to job-global fields:
   - `api/prediction_engines/subprocess_runner.py`
3. Embedding tracker is explicitly active-method-only, keyed only by job:
   - `api/services/embedding_progress_service.py` (`_TRACKERS` by `job_public_id`, Redis key `job_embedding_progress:<job>`)
4. Track page renders one flattened metrics object:
   - `frontend/src/components/JobStatus.jsx`

## Proposed System: Stage-Based Progress

Represent each target as a **stage** with its own prediction and embedding progress.

Example stage list for a 3-target job:

1. `kcat` with method X
2. `Km` with method Y
3. `kcat/Km` with method Z

Each stage has independent lifecycle and counters:

- `pending` -> `running` -> `completed` (or `failed`)
- Prediction counters
- Embedding counters/status

Completed stages are immutable snapshots, so the UI never "resets" when the next stage starts.

## Backend Data Model

Add a new model (recommended):

`JobProgressStage`

- `job` (FK to `Job`)
- `stage_index` (execution order)
- `target` (`kcat`, `Km`, `kcat/Km`)
- `method_key`
- `method_display_name`
- `status` (`pending`, `running`, `completed`, `failed`, `skipped`)
- `started_at`, `completed_at`, `updated_at`
- Prediction fields:
  - `molecules_total`, `molecules_processed`, `invalid_rows`
  - `predictions_total`, `predictions_made`
- Embedding fields:
  - `embedding_status` (`not_required`, `pending`, `running`, `completed`, `failed`)
  - `embedding_total`, `embedding_cached_already`
  - `embedding_need_computation`, `embedding_computed`, `embedding_remaining`
- Optional: `message` (stage-level errors/warnings)

Keep existing `Job` counters for compatibility; treat them as "active stage mirror" during transition.

## Backend Orchestration Changes

### 1. Initialize Stages at Job Start

In `run_multi_prediction`:

- Build canonical ordered targets.
- Resolve method descriptors.
- Create one `JobProgressStage` row per target with `status=pending`.

### 2. Stage Lifecycle in Multi-Target Loop

Before each target execution:

- Mark stage `running`.
- Set stage start time.

During execution:

- Route progress updates to the active stage (prediction + embedding).

After target completes:

- Snapshot final counters into stage row.
- Mark stage `completed`.

On failure:

- Mark active stage `failed`.
- Preserve completed stages.

### 3. Progress Update Abstraction

Introduce a small progress service layer (instead of writing directly to `Job` everywhere):

- `start_stage(job_id, stage_index, ...)`
- `update_validation(...)`
- `update_prediction_progress(...)`
- `update_embedding_progress(...)`
- `complete_stage(...)`
- `fail_stage(...)`

Then adapt subprocess/engine updates to use this layer.

### 4. Embedding Tracking

Change embedding progress from single job slot to stage-aware updates.

Two valid implementation patterns:

1. Directly write embedding counters onto the active `JobProgressStage`.
2. Keep Redis tracking but key by `(job_public_id, stage_index)` and persist final snapshot to DB stage row.

Either way, each target keeps its own embedding history and status.

## API Contract Changes

### New status payload shape (additive)

Expose per-stage progress in both:

- `/api/job-status/<public_id>/`
- `/api/v1/status/<public_id>/`

Add:

`progress_stages` (or `progress.stages`) array:

- Stage identity: `index`, `target`, `methodKey`, `methodName`
- Stage status
- Prediction counters
- Embedding counters/status

Also add:

- `active_stage_index`
- `completed_stage_count`
- `total_stage_count`

### Backward compatibility

Continue returning existing legacy fields (`molecules_processed`, `predictions_made`, etc.) as mirrors of the active stage until frontend migration is fully deployed.

## Frontend Track Page Redesign

File: `frontend/src/components/JobStatus.jsx`

### Results Section

Replace single aggregate cards with per-stage rows/cards:

- `kcat - MethodName` : `predictions_made / predictions_total` + progress bar + status badge
- `Km - MethodName` : same
- `kcat/Km - MethodName` : same

Show completed stages with checkmark and frozen 100%.
Show running stage with animated indicator.
Show pending stages as queued.

### Protein Embeddings Section

Render per-stage embedding cards:

- Cached
- Need computation
- Computed / Need computation
- Status badge (`Not required`, `Running`, `Completed`, `Failed`)

### Remove Live Progress Section

Delete the standalone "Live Progress" block and related CSS. Progress is now fully represented in Results + Protein Embeddings.

## Rollout Plan

1. Add `JobProgressStage` model + migration.
2. Add progress service and wire `run_multi_prediction` stage lifecycle.
3. Wire engine/subprocess/embedding updates through stage-aware progress service.
4. Extend status endpoints with additive stage payload.
5. Update `JobStatus.jsx` to consume stage payload and remove Live Progress.
6. Update tests:
   - multi-target stage progression
   - per-stage embedding snapshots
   - backward compatibility fields
7. After one release, consider deprecating legacy job-global progress fields.

## Test Strategy

1. Unit tests:
   - stage transitions across 2-3 targets
   - target failure preserves earlier completed stages
   - embedding stage isolation (no overwrite across stages)
2. API tests:
   - status payload includes all stages in canonical order
   - running stage updates while completed stages stay unchanged
3. Frontend tests:
   - rendering for pending/running/completed/failed stage combinations
   - no Live Progress section rendered

## UX Outcome

The user sees a clear timeline:

- which target is running now,
- which targets are already done,
- what embedding work each target needed,
- and where a failure happened if one occurs.

No more confusing progress resets.
