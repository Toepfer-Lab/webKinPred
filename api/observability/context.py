"""Request/task correlation context for structured logs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_CONTEXT_KEYS = (
    "request_id",
    "job_public_id",
    "celery_task_id",
    "method_key",
    "target",
)

_context: ContextVar[dict[str, Any]] = ContextVar("webkinpred_log_context", default={})


def get_log_context() -> dict[str, Any]:
    """Return the current logging context with only non-empty values."""
    return {key: value for key, value in _context.get().items() if value not in (None, "")}


def bind_log_context(**values: Any) -> Token[dict[str, Any]]:
    """Bind correlation fields for the current context and return a reset token."""
    current = dict(_context.get())
    for key, value in values.items():
        if key not in _CONTEXT_KEYS:
            continue
        if value in (None, ""):
            current.pop(key, None)
        else:
            current[key] = value
    return _context.set(current)


def reset_log_context(token: Token[dict[str, Any]]) -> None:
    _context.reset(token)


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Temporarily bind logging correlation fields."""
    token = bind_log_context(**values)
    try:
        yield
    finally:
        reset_log_context(token)
