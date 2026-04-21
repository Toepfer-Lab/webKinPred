from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from api.services.embedding_plan_service import build_embedding_plan, gpu_step_work
from api.services.embedding_progress_service import start_embedding_tracking
from api.services.gpu_precompute_status_service import record_gpu_precompute_result


_DEFAULT_HEALTH_TTL = 10
_DEFAULT_JOB_TIMEOUT = 1200
_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_HTTP_TIMEOUT = 5.0

_status_cache_lock = threading.Lock()
_status_cache_payload: dict | None = None
_status_cache_ts = 0.0


@dataclass(frozen=True)
class GpuPrecomputeResult:
    attempted: bool
    used_gpu: bool
    completed: bool
    failed: bool
    reason: str | None = None


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _base_url() -> str:
    return str(os.environ.get("GPU_EMBED_SERVICE_URL", "")).strip().rstrip("/")


def _auth_header() -> dict[str, str]:
    token = str(os.environ.get("GPU_EMBED_SERVICE_TOKEN", "")).strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = _DEFAULT_HTTP_TIMEOUT) -> dict:
    body: bytes | None = None
    headers = {"Accept": "application/json", **_auth_header()}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, method=method.upper(), data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw) if raw else {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def get_gpu_status(*, force_refresh: bool = False) -> dict:
    global _status_cache_payload, _status_cache_ts

    base = _base_url()
    if not base:
        return {
            "configured": False,
            "online": False,
            "mode": "cpu",
            "reason": "not_configured",
        }

    ttl = _env_int("GPU_EMBED_HEALTH_TTL_SECONDS", _DEFAULT_HEALTH_TTL)
    now = time.time()

    if not force_refresh:
        with _status_cache_lock:
            if _status_cache_payload is not None and (now - _status_cache_ts) < ttl:
                return dict(_status_cache_payload)

    url = f"{base}/health"
    try:
        remote = _http_json("GET", url)
        payload = {
            "configured": True,
            "online": bool(remote.get("online", True)),
            "mode": "gpu" if bool(remote.get("online", True)) else "cpu",
            "reason": None if bool(remote.get("online", True)) else "remote_offline",
            "gpu_name": remote.get("gpu_name") or remote.get("name"),
            "free_vram_gb": remote.get("free_vram_gb"),
            "total_vram_gb": remote.get("total_vram_gb"),
            "active_jobs": int(remote.get("active_jobs", 0) or 0),
            "raw": remote,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        payload = {
            "configured": True,
            "online": False,
            "mode": "cpu",
            "reason": f"unreachable: {exc}",
        }

    with _status_cache_lock:
        _status_cache_payload = dict(payload)
        _status_cache_ts = now

    return dict(payload)


def _poll_job(base: str, gpu_job_id: str) -> dict:
    timeout_secs = _env_int("GPU_EMBED_JOB_TIMEOUT_SECONDS", _DEFAULT_JOB_TIMEOUT)
    deadline = time.monotonic() + float(timeout_secs)

    while True:
        status = _http_json("GET", f"{base}/embed/jobs/{urllib.parse.quote(gpu_job_id)}")
        state = str(status.get("status", "")).strip().lower()
        if state in {"done", "completed"}:
            return {"status": "done", "detail": status}
        if state in {"failed", "error"}:
            return {"status": "failed", "detail": status}

        if time.monotonic() >= deadline:
            return {"status": "timeout", "detail": status}

        time.sleep(_DEFAULT_POLL_INTERVAL)


def run_gpu_precompute_if_available(
    *,
    job_public_id: str,
    method_key: str,
    target: str,
    valid_sequences: list[str],
    env: dict | None,
) -> GpuPrecomputeResult:
    fail_closed = _env_bool("GPU_EMBED_FAIL_CLOSED", default=False)

    def _record(result: GpuPrecomputeResult) -> None:
        try:
            record_gpu_precompute_result(
                job_public_id=job_public_id,
                method_key=method_key,
                target=target,
                attempted=result.attempted,
                used_gpu=result.used_gpu,
                completed=result.completed,
                failed=result.failed,
                reason=result.reason,
            )
        except Exception:
            # Telemetry is best-effort; never break prediction flow/tests.
            pass

    def _finish(result: GpuPrecomputeResult) -> GpuPrecomputeResult:
        _record(result)
        if fail_closed and not result.completed:
            raise RuntimeError(
                f"GPU precompute required but incomplete for {method_key}/{target}: "
                f"{result.reason or 'unknown_reason'}"
            )
        return result

    if not valid_sequences:
        return _finish(GpuPrecomputeResult(False, False, False, False, "no_valid_sequences"))

    env_data = env or {}
    try:
        plan = build_embedding_plan(
            method_key=method_key,
            target=target,
            sequences=valid_sequences,
            env=env_data,
        )
    except Exception as exc:
        return _finish(GpuPrecomputeResult(False, False, False, False, f"plan_failed: {exc}"))

    if plan.need_computation <= 0:
        return _finish(GpuPrecomputeResult(False, False, True, False, "cache_complete"))
    if not plan.gpu_supported:
        # Keep unsupported methods fail-open even in strict mode.
        unsupported = GpuPrecomputeResult(
            False, False, False, False, plan.gpu_reason or "unsupported"
        )
        _record(unsupported)
        return unsupported

    base = _base_url()
    if not base:
        return _finish(GpuPrecomputeResult(False, False, False, False, "not_configured"))

    status = get_gpu_status()
    if not status.get("online"):
        return _finish(
            GpuPrecomputeResult(True, False, False, True, status.get("reason") or "offline")
        )

    step_work = gpu_step_work(plan)
    if not step_work:
        return _finish(GpuPrecomputeResult(True, False, True, False, "no_missing_gpu_steps"))

    missing_seq_ids = sorted({seq_id for seq_ids in step_work.values() for seq_id in seq_ids})
    seq_id_to_seq = {seq_id: plan.seq_id_to_seq[seq_id] for seq_id in missing_seq_ids if seq_id in plan.seq_id_to_seq}

    # Start tracker before GPU writes begin so inotify observes remote file creation.
    start_embedding_tracking(
        job_public_id=job_public_id,
        method_key=method_key,
        target=target,
        valid_sequences=valid_sequences,
        env=env_data,
    )

    payload = {
        "method_key": method_key,
        "target": target,
        "profile": plan.profile,
        "step_work": step_work,
        "seq_id_to_seq": seq_id_to_seq,
    }

    try:
        response = _http_json("POST", f"{base}/embed/jobs", payload=payload)
        gpu_job_id = str(response.get("job_id", "")).strip()
        if not gpu_job_id:
            return _finish(GpuPrecomputeResult(True, True, False, True, "missing_job_id"))

        polled = _poll_job(base, gpu_job_id)
        if polled["status"] == "done":
            # Remote service may report "done" while writing no cache files
            # (for example during smoke/no-op command wiring). Re-check local
            # cache state before treating GPU precompute as completed.
            try:
                post_plan = build_embedding_plan(
                    method_key=method_key,
                    target=target,
                    sequences=valid_sequences,
                    env=env_data,
                )
            except Exception as exc:
                return _finish(
                    GpuPrecomputeResult(True, True, False, True, f"postcheck_failed: {exc}")
                )

            if post_plan.need_computation > 0:
                return _finish(
                    GpuPrecomputeResult(
                        True,
                        True,
                        False,
                        True,
                        f"incomplete_remote_outputs:{post_plan.need_computation}",
                    )
                )

            return _finish(GpuPrecomputeResult(True, True, True, False, "done"))
        return _finish(GpuPrecomputeResult(True, True, False, True, polled["status"]))
    except RuntimeError:
        # Preserve strict-mode failures as-is (do not wrap them again).
        raise
    except Exception as exc:
        return _finish(GpuPrecomputeResult(True, True, False, True, f"gpu_request_failed: {exc}"))
