"""JSON logging primitives used by Django, Celery, and subprocess runners."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from api.observability.context import get_log_context

try:
    from pythonjsonlogger import jsonlogger

    _BaseJsonFormatter = jsonlogger.JsonFormatter
except ImportError:  # pragma: no cover - production images install python-json-logger.
    class _BaseJsonFormatter(logging.Formatter):
        def add_fields(
            self,
            log_record: dict[str, Any],
            record: logging.LogRecord,
            message_dict: dict[str, Any],
        ) -> None:
            log_record.update(message_dict)

        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {}
            self.add_fields(payload, record, {})
            return json.dumps(payload, default=str)

_DEFAULT_FIELDS: dict[str, Any] = {
    "request_id": None,
    "job_public_id": None,
    "celery_task_id": None,
    "method_key": None,
    "target": None,
}

_RESERVED_LOG_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__)


class CorrelationFilter(logging.Filter):
    """Attach service metadata and contextvars correlation fields to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = os.environ.get("WEBKINPRED_SERVICE", "webkinpred")
        for key, default in _DEFAULT_FIELDS.items():
            if not hasattr(record, key):
                setattr(record, key, default)
        for key, value in get_log_context().items():
            setattr(record, key, value)
        return True


class JsonLogFormatter(_BaseJsonFormatter):
    """Stable JSON formatter with ISO timestamps and useful exception fields."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["message"] = record.getMessage()
        log_record["service"] = getattr(record, "service", os.environ.get("WEBKINPRED_SERVICE", "webkinpred"))

        for key, default in _DEFAULT_FIELDS.items():
            log_record.setdefault(key, getattr(record, key, default))

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
            log_record["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key in log_record:
                continue
            if key.startswith("_"):
                continue
            log_record[key] = value
