"""Django request correlation middleware."""

from __future__ import annotations

import logging
import time
import uuid

from api.observability.context import bind_log_context, reset_log_context

_REQUEST_HEADER = "HTTP_X_REQUEST_ID"
_RESPONSE_HEADER = "X-Request-ID"
_log = logging.getLogger(__name__)


class RequestIDMiddleware:
    """Create or propagate a request ID and expose it to structured logs."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = (request.META.get(_REQUEST_HEADER) or "").strip() or uuid.uuid4().hex
        request.request_id = request_id
        started_at = time.monotonic()
        token = bind_log_context(request_id=request_id)
        try:
            response = self.get_response(request)
        except Exception:
            _log.exception(
                "HTTP request failed",
                extra={
                    "event": "http.request_failed",
                    "method": getattr(request, "method", None),
                    "path": getattr(request, "path", None),
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                    "remote_addr": request.META.get("REMOTE_ADDR"),
                    "user_agent": request.META.get("HTTP_USER_AGENT"),
                },
            )
            raise
        else:
            response[_RESPONSE_HEADER] = request_id
            path = getattr(request, "path", "") or ""
            log_level = logging.DEBUG if path.startswith(("/api/health/", "/django_static/")) else logging.INFO
            _log.log(
                log_level,
                "HTTP request completed",
                extra={
                    "event": "http.request",
                    "method": getattr(request, "method", None),
                    "path": path,
                    "status_code": getattr(response, "status_code", None),
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                    "remote_addr": request.META.get("REMOTE_ADDR"),
                    "user_agent": request.META.get("HTTP_USER_AGENT"),
                },
            )
            return response
        finally:
            reset_log_context(token)
