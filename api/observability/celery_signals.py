"""Celery lifecycle logging and correlation binding."""

from __future__ import annotations

import logging
import time
from typing import Any

from celery.signals import task_failure, task_postrun, task_prerun

from api.observability.context import bind_log_context, reset_log_context

_log = logging.getLogger("api.observability.celery")


def _task_context(task: Any, task_id: str | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    job_public_id = kwargs.get("public_id")
    method_key = kwargs.get("method_key")
    target = kwargs.get("target")

    if job_public_id is None and args:
        job_public_id = args[0]

    task_name = getattr(task, "name", "") or ""
    if task_name.endswith("run_prediction") and len(args) >= 3:
        method_key = method_key or args[1]
        target = target or args[2]

    return {
        "job_public_id": job_public_id,
        "celery_task_id": task_id,
        "method_key": method_key,
        "target": target,
    }


@task_prerun.connect(weak=False)
def log_task_prerun(task_id=None, task=None, args=None, kwargs=None, **_extra):
    args = tuple(args or ())
    kwargs = dict(kwargs or {})
    context = _task_context(task, task_id, args, kwargs)
    token = bind_log_context(**context)
    if task is not None:
        task.request._observability_token = token
        task.request._observability_started_at = time.monotonic()
    _log.info(
        "Celery task started",
        extra={
            "event": "celery.task.started",
            "task_name": getattr(task, "name", None),
            **context,
        },
    )


@task_postrun.connect(weak=False)
def log_task_postrun(task_id=None, task=None, retval=None, state=None, args=None, kwargs=None, **_extra):
    started_at = getattr(getattr(task, "request", None), "_observability_started_at", None)
    duration_ms = int((time.monotonic() - started_at) * 1000) if started_at else None
    args = tuple(args or ())
    kwargs = dict(kwargs or {})
    context = _task_context(task, task_id, args, kwargs)
    _log.info(
        "Celery task finished",
        extra={
            "event": "celery.task.finished",
            "task_name": getattr(task, "name", None),
            "state": state,
            "duration_ms": duration_ms,
            **context,
        },
    )
    token = getattr(getattr(task, "request", None), "_observability_token", None)
    if token is not None:
        reset_log_context(token)


@task_failure.connect(weak=False)
def log_task_failure(
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    sender=None,
    **_extra,
):
    task = sender
    args = tuple(args or ())
    kwargs = dict(kwargs or {})
    context = _task_context(task, task_id, args, kwargs)
    _log.error(
        "Celery task failed",
        extra={
            "event": "celery.task.failed",
            "task_name": getattr(task, "name", None),
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None,
            **context,
        },
        exc_info=einfo.exc_info if einfo is not None else None,
    )
