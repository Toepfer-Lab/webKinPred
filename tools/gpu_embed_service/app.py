from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class EmbedJobRequest(BaseModel):
    method_key: str
    target: str
    profile: str | None = None
    step_work: dict[str, list[str]] = Field(default_factory=dict)
    seq_id_to_seq: dict[str, str] = Field(default_factory=dict)


class EmbedJobStatus(BaseModel):
    job_id: str
    status: str
    method_key: str
    target: str
    profile: str | None = None
    step_work: dict[str, list[str]] = Field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


@dataclass
class _JobState:
    request: EmbedJobRequest
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


app = FastAPI(title="webKinPred GPU Embedding Service", version="0.1.0")

_jobs: dict[str, _JobState] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gpu_health_snapshot() -> dict[str, Any]:
    # Prefer NVIDIA's own CLI because it's present whenever drivers are installed.
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                # Example line: "NVIDIA RTX A4500, 17519, 20470"
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    free_mb = _parse_float(parts[1])
                    total_mb = _parse_float(parts[2])
                    free_gb = (free_mb / 1024.0) if free_mb is not None else None
                    total_gb = (total_mb / 1024.0) if total_mb is not None else None
                    return {
                        "online": True,
                        "gpu_name": gpu_name or "unknown",
                        "free_vram_gb": round(free_gb, 2) if free_gb is not None else None,
                        "total_vram_gb": round(total_gb, 2) if total_gb is not None else None,
                    }
    except Exception:
        pass

    # Fallback if nvidia-smi is unavailable in path.
    fallback_name = str(os.environ.get("GPU_NAME", "unknown")).strip() or "unknown"
    return {
        "online": True,
        "gpu_name": fallback_name,
        "free_vram_gb": None,
        "total_vram_gb": None,
    }


def _token_is_valid(auth_header: str | None) -> bool:
    token = str(os.environ.get("GPU_EMBED_SERVICE_TOKEN", "")).strip()
    if not token:
        return True
    if not auth_header:
        return False
    parts = auth_header.split(" ", 1)
    return len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip() == token


def _require_auth(authorization: str | None) -> None:
    if not _token_is_valid(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")


def _run_command(cmd: str) -> None:
    proc = subprocess.run(cmd, shell=True, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"step command failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _execute_step(step_key: str, seq_ids: list[str], seq_id_to_seq: dict[str, str]) -> None:
    # Optional step override command. If absent, this is a no-op by default.
    # Example env key: GPU_EMBED_STEP_CMD_KINFORM_ESM2_LAYERS
    env_key = f"GPU_EMBED_STEP_CMD_{step_key.upper()}"
    cmd = str(os.environ.get(env_key, "")).strip()
    if not cmd:
        return

    seq_ids_arg = ",".join(seq_ids)
    seq_count = len(seq_ids)
    cmd = cmd.format(step_key=step_key, seq_ids=seq_ids_arg, seq_count=seq_count)
    _run_command(cmd)


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        state = _jobs[job_id]
        state.status = "running"
        state.started_at = time.time()

    req = state.request

    try:
        for step_key, seq_ids in req.step_work.items():
            if not seq_ids:
                continue
            _execute_step(step_key, seq_ids, req.seq_id_to_seq)

        with _jobs_lock:
            state.status = "done"
            state.finished_at = time.time()
    except Exception as exc:
        with _jobs_lock:
            state.status = "failed"
            state.error = str(exc)
            state.finished_at = time.time()


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            _run_job(job_id)
        finally:
            _job_queue.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_worker_loop, name="gpu-embed-worker", daemon=True)
        thread.start()
        _worker_started = True


def _status_payload(job_id: str, state: _JobState) -> EmbedJobStatus:
    return EmbedJobStatus(
        job_id=job_id,
        status=state.status,
        method_key=state.request.method_key,
        target=state.request.target,
        profile=state.request.profile,
        step_work=state.request.step_work,
        started_at=state.started_at,
        finished_at=state.finished_at,
        error=state.error,
    )


@app.on_event("startup")
def _on_startup() -> None:
    _ensure_worker()


@app.get("/health")
def health() -> dict:
    gpu = _gpu_health_snapshot()
    return {
        "online": bool(gpu.get("online", True)),
        "gpu_name": gpu.get("gpu_name"),
        "free_vram_gb": gpu.get("free_vram_gb"),
        "total_vram_gb": gpu.get("total_vram_gb"),
        "active_jobs": sum(1 for s in _jobs.values() if s.status == "running"),
        "queued_jobs": _job_queue.qsize(),
    }


@app.post("/embed/jobs")
def submit_embed_job(payload: EmbedJobRequest, authorization: str | None = Header(default=None)) -> dict:
    _require_auth(authorization)
    _ensure_worker()

    if not payload.step_work:
        raise HTTPException(status_code=400, detail="step_work is required")

    job_id = uuid.uuid4().hex
    state = _JobState(request=payload)
    with _jobs_lock:
        _jobs[job_id] = state
    _job_queue.put(job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/embed/jobs/{job_id}")
def get_embed_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_auth(authorization)
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        payload = _status_payload(job_id, state)
    return payload.model_dump()
