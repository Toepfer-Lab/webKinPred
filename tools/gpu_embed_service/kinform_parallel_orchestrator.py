#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import pickle
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_T5_FAMILY = "t5"
_ESM2_FAMILY = "esm2"
_ESMC_FAMILY = "esmc"
_PSEQ_WORKER = "pseq2sites"

_FAMILY_ROOTS: dict[str, tuple[str, ...]] = {
    _T5_FAMILY: ("prot_t5_layer_19", "prot_t5_last"),
    _ESM2_FAMILY: ("esm2_layer_26", "esm2_layer_29"),
    _ESMC_FAMILY: ("esmc_layer_24", "esmc_layer_32"),
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _log_level_value(name: str) -> int:
    mapping = {
        "debug": 10,
        "info": 20,
        "warn": 30,
        "warning": 30,
        "error": 40,
        "quiet": 100,
    }
    return mapping.get(name.strip().lower(), 20)


def _log(
    env: dict[str, str],
    level: str,
    message: str,
    *,
    job_id: str | None = None,
) -> None:
    configured = _log_level_value(str(env.get("KINFORM_PARALLEL_LOG_LEVEL", "info")))
    current = _log_level_value(level)
    if current < configured:
        return
    job = job_id or "unknown"
    print(f"KINFORM_PARALLEL_{level.upper()} job_id={job} {message}")


def _artifact_path(media_path: Path, root: str, kind: str, seq_id: str) -> Path:
    return (media_path / "sequence_info" / root / f"{kind}_vecs" / f"{seq_id}.npy").resolve()


def _read_binding_site_scores(binding_sites_path: Path) -> dict[str, np.ndarray]:
    if not binding_sites_path.exists():
        return {}

    out: dict[str, np.ndarray] = {}
    try:
        with binding_sites_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                return {}
            key_col = "PDB" if "PDB" in reader.fieldnames else reader.fieldnames[0]
            score_col = (
                "Pred_BS_Scores"
                if "Pred_BS_Scores" in reader.fieldnames
                else (reader.fieldnames[1] if len(reader.fieldnames) > 1 else "")
            )
            if not score_col:
                return {}
            for row in reader:
                seq_id = str(row.get(key_col, "")).strip()
                score_text = str(row.get(score_col, "")).strip()
                if not seq_id or not score_text:
                    continue
                try:
                    weights = np.fromiter((float(x) for x in score_text.split(",")), dtype=np.float64)
                except ValueError:
                    continue
                if weights.size:
                    out[seq_id] = weights
    except Exception:
        return {}
    return out


class BindingSiteScoreCache:
    def __init__(self, binding_sites_path: Path) -> None:
        self.binding_sites_path = binding_sites_path
        self._mtime_ns: int | None = None
        self._scores: dict[str, np.ndarray] = {}

    def read(self) -> dict[str, np.ndarray]:
        if not self.binding_sites_path.exists():
            self._mtime_ns = None
            self._scores = {}
            return {}
        stat = self.binding_sites_path.stat()
        if self._mtime_ns != stat.st_mtime_ns:
            self._scores = _read_binding_site_scores(self.binding_sites_path)
            self._mtime_ns = stat.st_mtime_ns
        return self._scores


def weighted_mean_from_residue(residue_embedding: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if residue_embedding.ndim != 2:
        raise ValueError(f"Expected 2D residue embedding array, got shape {residue_embedding.shape}")
    if weights.ndim != 1:
        raise ValueError(f"Expected 1D weights array, got shape {weights.shape}")
    if residue_embedding.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Weight length ({weights.shape[0]}) != residue length ({residue_embedding.shape[0]})"
        )
    denom = float(np.sum(weights))
    if denom <= 0.0:
        raise ValueError("Binding-site weights sum to zero; cannot normalize.")
    normalized = weights.astype(np.float64) / denom
    vec = (residue_embedding.astype(np.float64) * normalized[:, None]).sum(axis=0)
    return vec.astype(np.float32)


def _save_array_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _remove_path_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


@dataclass
class ArtifactTargets:
    seq_ids: list[str]
    media_path: Path
    binding_sites_path: Path
    weighted_targets: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    mean_targets: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    binding_site_targets: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for family, roots in _FAMILY_ROOTS.items():
            for root in roots:
                key = (family, root)
                weighted_missing = set()
                mean_missing = set()
                for seq_id in self.seq_ids:
                    weighted_path = _artifact_path(self.media_path, root, "weighted", seq_id)
                    mean_path = _artifact_path(self.media_path, root, "mean", seq_id)
                    if not weighted_path.exists():
                        weighted_missing.add(seq_id)
                    if not mean_path.exists():
                        mean_missing.add(seq_id)
                self.weighted_targets[key] = weighted_missing
                self.mean_targets[key] = mean_missing

        seen = _read_binding_site_scores(self.binding_sites_path)
        self.binding_site_targets = {sid for sid in self.seq_ids if sid not in seen}

    def all_done(self, bs_scores: dict[str, np.ndarray]) -> bool:
        for (family, root), targets in self.weighted_targets.items():
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "weighted", seq_id).exists():
                    return False
        for (family, root), targets in self.mean_targets.items():
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "mean", seq_id).exists():
                    return False
        for seq_id in self.binding_site_targets:
            if seq_id not in bs_scores:
                return False
        return True

    def missing_weighted_count(self, family: str) -> tuple[int, int]:
        missing = 0
        total = 0
        for (fam, root), targets in self.weighted_targets.items():
            if fam != family:
                continue
            total += len(targets)
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "weighted", seq_id).exists():
                    missing += 1
        return missing, total

    def missing_mean_count(self, family: str) -> tuple[int, int]:
        missing = 0
        total = 0
        for (fam, root), targets in self.mean_targets.items():
            if fam != family:
                continue
            total += len(targets)
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "mean", seq_id).exists():
                    missing += 1
        return missing, total


def _derive_mean_if_ready(
    *,
    targets: ArtifactTargets,
    family: str,
    root: str,
    seq_id: str,
) -> bool:
    if seq_id not in targets.mean_targets.get((family, root), set()):
        return False
    mean_path = _artifact_path(targets.media_path, root, "mean", seq_id)
    if mean_path.exists():
        return False
    residue_path = _artifact_path(targets.media_path, root, "residue", seq_id)
    if not residue_path.exists():
        return False
    residue = np.load(residue_path)
    _save_array_atomic(mean_path, residue.mean(axis=0).astype(np.float32))
    return True


def _derive_weighted_if_ready(
    *,
    targets: ArtifactTargets,
    family: str,
    root: str,
    seq_id: str,
    bs_scores: dict[str, np.ndarray],
    weighted_retry_errors: dict[tuple[str, str, str], int],
) -> bool:
    if seq_id not in targets.weighted_targets.get((family, root), set()):
        return False
    weighted_path = _artifact_path(targets.media_path, root, "weighted", seq_id)
    if weighted_path.exists():
        return False
    residue_path = _artifact_path(targets.media_path, root, "residue", seq_id)
    if not residue_path.exists():
        return False
    weights = bs_scores.get(seq_id)
    if weights is None:
        return False
    key = (family, root, seq_id)
    try:
        residue = np.load(residue_path)
        weighted_vec = weighted_mean_from_residue(residue, weights)
        _save_array_atomic(weighted_path, weighted_vec)
        _remove_path_if_exists(residue_path)
        weighted_retry_errors.pop(key, None)
        return True
    except Exception:
        weighted_retry_errors[key] = weighted_retry_errors.get(key, 0) + 1
        if weighted_retry_errors[key] > 1:
            raise
        return False


def _to_seq_subset(seq_id_to_seq: dict[str, str], seq_ids: set[str]) -> dict[str, str]:
    return {sid: seq_id_to_seq[sid] for sid in seq_id_to_seq if sid in seq_ids}


def _write_worker_inputs(seq_id_to_seq: dict[str, str]) -> tuple[Path, Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="kinform_parallel_worker_"))
    seq_file = tmp_dir / "seq_ids.txt"
    id_to_seq_pkl = tmp_dir / "id_to_seq.pkl"
    seq_map_json = tmp_dir / "seq_id_to_seq.json"

    with seq_file.open("w", encoding="utf-8") as handle:
        for seq_id in seq_id_to_seq:
            handle.write(f"{seq_id}\n")

    with id_to_seq_pkl.open("wb") as handle:
        pickle.dump(seq_id_to_seq, handle, protocol=4)

    seq_map_json.write_text(json.dumps(seq_id_to_seq), encoding="utf-8")
    return seq_file, id_to_seq_pkl, seq_map_json


def _start_worker(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    print("+", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.Popen(cmd, env=env)


def _needs_t5_worker_seq_ids(
    *,
    targets: ArtifactTargets,
    bs_scores: dict[str, np.ndarray],
) -> set[str]:
    out: set[str] = set()
    t5_last_root = "prot_t5_last"
    for seq_id in targets.seq_ids:
        needs = False
        for root in _FAMILY_ROOTS[_T5_FAMILY]:
            if seq_id in targets.mean_targets[(_T5_FAMILY, root)] and not _artifact_path(
                targets.media_path, root, "mean", seq_id
            ).exists():
                needs = True
                break
            if seq_id in targets.weighted_targets[(_T5_FAMILY, root)]:
                weighted_exists = _artifact_path(targets.media_path, root, "weighted", seq_id).exists()
                residue_exists = _artifact_path(targets.media_path, root, "residue", seq_id).exists()
                if not weighted_exists and not residue_exists:
                    needs = True
                    break

        if not needs and seq_id in targets.binding_site_targets and seq_id not in bs_scores:
            if not _artifact_path(targets.media_path, t5_last_root, "residue", seq_id).exists():
                needs = True
        if needs:
            out.add(seq_id)
    return out


def _needs_esm_worker_seq_ids(
    *,
    family: str,
    targets: ArtifactTargets,
) -> set[str]:
    out: set[str] = set()
    for seq_id in targets.seq_ids:
        needs = False
        for root in _FAMILY_ROOTS[family]:
            if seq_id in targets.mean_targets[(family, root)] and not _artifact_path(
                targets.media_path, root, "mean", seq_id
            ).exists():
                needs = True
                break
            if seq_id in targets.weighted_targets[(family, root)]:
                weighted_exists = _artifact_path(targets.media_path, root, "weighted", seq_id).exists()
                residue_exists = _artifact_path(targets.media_path, root, "residue", seq_id).exists()
                if not weighted_exists and not residue_exists:
                    needs = True
                    break
        if needs:
            out.add(seq_id)
    return out


def _needs_pseq_worker_seq_ids(
    *,
    targets: ArtifactTargets,
    bs_scores: dict[str, np.ndarray],
) -> set[str]:
    return {seq_id for seq_id in targets.binding_site_targets if seq_id not in bs_scores}


@dataclass
class WorkerState:
    name: str
    attempts: int = 0
    process: subprocess.Popen | None = None
    tmp_inputs_dir: Path | None = None
    active_seq_ids: set[str] = field(default_factory=set)

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


def _terminate_worker(state: WorkerState) -> None:
    if state.process is None:
        return
    if state.process.poll() is None:
        state.process.terminate()
        try:
            state.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            state.process.kill()
            state.process.wait(timeout=10)
    state.process = None


def _cleanup_worker_inputs(state: WorkerState) -> None:
    if state.tmp_inputs_dir is None:
        return
    tmp_dir = state.tmp_inputs_dir
    state.tmp_inputs_dir = None
    try:
        for child in tmp_dir.glob("*"):
            child.unlink(missing_ok=True)
        tmp_dir.rmdir()
    except OSError:
        pass


def run_kinform_parallel_pipeline(
    *,
    env: dict[str, str],
    repo_root: Path,
    media_path: Path,
    seq_id_to_seq: dict[str, str],
    job_id: str | None = None,
) -> None:
    seq_ids = list(seq_id_to_seq.keys())
    if not seq_ids:
        _log(env, "info", "No sequence IDs provided to KinForm parallel pipeline.", job_id=job_id)
        return

    binding_sites_path = (media_path / "pseq2sites" / "binding_sites_all.tsv").resolve()
    targets = ArtifactTargets(
        seq_ids=seq_ids,
        media_path=media_path,
        binding_sites_path=binding_sites_path,
    )
    score_cache = BindingSiteScoreCache(binding_sites_path)
    weighted_retry_errors: dict[tuple[str, str, str], int] = {}

    # Fast-path: nothing missing for this batch.
    if targets.all_done(score_cache.read()):
        _log(env, "info", "All KinForm artifacts already exist; skipping parallel pipeline.", job_id=job_id)
        return

    t5_script = (repo_root / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py").resolve()
    prot_script = (repo_root / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py").resolve()
    pseq_stream_script = (
        repo_root
        / "models"
        / "KinForm"
        / "code"
        / "pseq2sites"
        / "pseq2sites_stream_worker.py"
    ).resolve()

    workers: dict[str, WorkerState] = {
        _T5_FAMILY: WorkerState(name=_T5_FAMILY),
        _ESM2_FAMILY: WorkerState(name=_ESM2_FAMILY),
        _ESMC_FAMILY: WorkerState(name=_ESMC_FAMILY),
        _PSEQ_WORKER: WorkerState(name=_PSEQ_WORKER),
    }

    def needed_ids(worker_name: str, bs_scores: dict[str, np.ndarray]) -> set[str]:
        if worker_name == _T5_FAMILY:
            return _needs_t5_worker_seq_ids(targets=targets, bs_scores=bs_scores)
        if worker_name == _ESM2_FAMILY:
            return _needs_esm_worker_seq_ids(family=_ESM2_FAMILY, targets=targets)
        if worker_name == _ESMC_FAMILY:
            return _needs_esm_worker_seq_ids(family=_ESMC_FAMILY, targets=targets)
        if worker_name == _PSEQ_WORKER:
            return _needs_pseq_worker_seq_ids(targets=targets, bs_scores=bs_scores)
        return set()

    def build_cmd(worker_name: str, seq_subset: dict[str, str], seq_file: Path, id_to_seq_pkl: Path, seq_map_json: Path) -> list[str]:
        if worker_name == _T5_FAMILY:
            return [
                env["KINFORM_T5_PATH"],
                str(t5_script),
                "--seq_file",
                str(seq_file),
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                "1",
                "--setting",
                "residue+mean",
                "--layers",
                "19",
                "None",
            ]
        if worker_name == _ESM2_FAMILY:
            return [
                env["KINFORM_ESM_PATH"],
                str(prot_script),
                "--seq_file",
                str(seq_file),
                "--models",
                "esm2",
                "--layers",
                "26",
                "29",
                "--setting",
                "residue+mean",
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                "1",
            ]
        if worker_name == _ESMC_FAMILY:
            return [
                env["KINFORM_ESMC_PATH"],
                str(prot_script),
                "--seq_file",
                str(seq_file),
                "--models",
                "esmc",
                "--layers",
                "24",
                "32",
                "--setting",
                "residue+mean",
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                "1",
            ]
        if worker_name == _PSEQ_WORKER:
            return [
                env["KINFORM_PSEQ2SITES_PATH"],
                str(pseq_stream_script),
                "--seq-id-to-seq-file",
                str(seq_map_json),
                "--binding-sites-path",
                str(binding_sites_path),
                "--poll-interval-seconds",
                "0.5",
                "--batch-size",
                "8",
            ]
        raise RuntimeError(f"Unknown KinForm worker '{worker_name}'.")

    def launch_worker(worker_name: str, bs_scores: dict[str, np.ndarray]) -> bool:
        state = workers[worker_name]
        seq_ids_to_run = needed_ids(worker_name, bs_scores)
        if not seq_ids_to_run:
            return False
        if state.attempts >= 2:
            raise RuntimeError(
                f"{worker_name} exhausted retries with remaining seq_ids={sorted(seq_ids_to_run)}"
            )
        seq_subset = _to_seq_subset(seq_id_to_seq, seq_ids_to_run)
        seq_file, id_to_seq_pkl, seq_map_json = _write_worker_inputs(seq_subset)
        state.tmp_inputs_dir = seq_file.parent
        cmd = build_cmd(worker_name, seq_subset, seq_file, id_to_seq_pkl, seq_map_json)
        state.process = _start_worker(cmd, env)
        state.active_seq_ids = set(seq_ids_to_run)
        state.attempts += 1
        _log(
            env,
            "info",
            f"launched worker={worker_name} attempt={state.attempts} seq_count={len(seq_ids_to_run)}",
            job_id=job_id,
        )
        return True

    def poll_worker(worker_name: str, bs_scores: dict[str, np.ndarray]) -> bool:
        state = workers[worker_name]
        if state.process is None:
            return False
        rc = state.process.poll()
        if rc is None:
            return False
        _cleanup_worker_inputs(state)
        state.process = None
        remaining = needed_ids(worker_name, bs_scores)
        if rc == 0 and not remaining:
            _log(env, "info", f"worker={worker_name} completed.", job_id=job_id)
            return True
        if rc != 0:
            _log(
                env,
                "warn",
                f"worker={worker_name} failed rc={rc}; remaining_seq_count={len(remaining)}",
                job_id=job_id,
            )
        else:
            _log(
                env,
                "warn",
                f"worker={worker_name} exited but artifacts still missing; remaining_seq_count={len(remaining)}",
                job_id=job_id,
            )
        if remaining and state.attempts < 2:
            launch_worker(worker_name, bs_scores)
            return True
        if remaining:
            raise RuntimeError(
                f"worker={worker_name} failed after retry; remaining seq_ids={sorted(remaining)}"
            )
        return True

    # Initial launch.
    scores = score_cache.read()
    for name in workers:
        launch_worker(name, scores)

    poll_interval_seconds = 0.5
    progress_interval_seconds = 10.0
    last_progress_ts = 0.0

    try:
        while True:
            scores = score_cache.read()

            derived_weighted = 0
            derived_mean = 0
            for family, roots in _FAMILY_ROOTS.items():
                for root in roots:
                    for seq_id in seq_ids:
                        if _derive_mean_if_ready(
                            targets=targets,
                            family=family,
                            root=root,
                            seq_id=seq_id,
                        ):
                            derived_mean += 1
                        if _derive_weighted_if_ready(
                            targets=targets,
                            family=family,
                            root=root,
                            seq_id=seq_id,
                            bs_scores=scores,
                            weighted_retry_errors=weighted_retry_errors,
                        ):
                            derived_weighted += 1

            if derived_weighted or derived_mean:
                _log(
                    env,
                    "debug",
                    f"derived mean={derived_mean} weighted={derived_weighted} this iteration",
                    job_id=job_id,
                )

            scores = score_cache.read()
            if targets.all_done(scores):
                _log(env, "info", "all target artifacts are complete.", job_id=job_id)
                break

            had_activity = bool(derived_weighted or derived_mean)
            for name in workers:
                if poll_worker(name, scores):
                    had_activity = True

            # Launch any worker that is currently idle and still needed.
            for name, state in workers.items():
                if state.process is None:
                    if launch_worker(name, scores):
                        had_activity = True

            if all(state.process is None for state in workers.values()) and not had_activity:
                # No worker can make progress and artifacts are still missing.
                missing_fragments = []
                for family in (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY):
                    mw, tw = targets.missing_weighted_count(family)
                    mm, tm = targets.missing_mean_count(family)
                    missing_fragments.append(f"{family}:weighted={mw}/{tw},mean={mm}/{tm}")
                bs_missing = len(_needs_pseq_worker_seq_ids(targets=targets, bs_scores=scores))
                missing_fragments.append(f"binding_sites_missing={bs_missing}/{len(targets.binding_site_targets)}")
                raise RuntimeError("KinForm parallel pipeline stalled: " + " ".join(missing_fragments))

            now = time.monotonic()
            if now - last_progress_ts >= progress_interval_seconds:
                last_progress_ts = now
                bs_ready = len(targets.binding_site_targets) - len(
                    _needs_pseq_worker_seq_ids(targets=targets, bs_scores=scores)
                )
                bs_total = len(targets.binding_site_targets)
                t5_w_missing, t5_w_total = targets.missing_weighted_count(_T5_FAMILY)
                esm2_w_missing, esm2_w_total = targets.missing_weighted_count(_ESM2_FAMILY)
                esmc_w_missing, esmc_w_total = targets.missing_weighted_count(_ESMC_FAMILY)
                _log(
                    env,
                    "info",
                    (
                        f"progress bs={bs_ready}/{bs_total} "
                        f"weighted_t5={t5_w_total - t5_w_missing}/{t5_w_total} "
                        f"weighted_esm2={esm2_w_total - esm2_w_missing}/{esm2_w_total} "
                        f"weighted_esmc={esmc_w_total - esmc_w_missing}/{esmc_w_total}"
                    ),
                    job_id=job_id,
                )

            time.sleep(poll_interval_seconds)
    finally:
        for state in workers.values():
            _terminate_worker(state)
            _cleanup_worker_inputs(state)
