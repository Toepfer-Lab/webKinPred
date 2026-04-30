import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from api.services.progress_service import push_line, finish_session
from api.services.similarity_service import analyze_sequence_similarity
from api.utils.http_utils import (
    validate_post_request_similarity,
    extract_csv_file_from_request,
    extract_validation_session_id,
)

_log = logging.getLogger(__name__)


@csrf_exempt
def sequence_similarity_summary(request):
    """Analyze protein sequence similarity against target databases."""
    _log.info(
        "Sequence similarity summary requested",
        extra={"event": "similarity.summary_requested"},
    )

    # Validate request method
    method_error = validate_post_request_similarity(request)
    if method_error:
        return method_error

    # Extract session ID
    session_id = extract_validation_session_id(request)

    try:
        # Extract CSV file from request
        csv_file, file_error = extract_csv_file_from_request(request)
        if file_error:
            return file_error

        # Start similarity analysis
        push_line(session_id, "==> Starting MMseqs2 similarity analysis")

        # Perform similarity analysis
        result = analyze_sequence_similarity(csv_file, session_id=session_id)

        push_line(session_id, "==> Similarity histograms computed successfully")
        return JsonResponse(result, status=200)

    except ValueError as ve:
        _log.warning(
            "Sequence similarity validation failed",
            extra={
                "event": "similarity.validation_failed",
                "session_id": session_id,
                "exception_type": type(ve).__name__,
            },
        )
        push_line(session_id, f"[VALIDATION ERROR] {ve}")
        return JsonResponse({"error": str(ve)}, status=400)

    except Exception as e:
        _log.exception(
            "Sequence similarity failed",
            extra={
                "event": "similarity.failed",
                "session_id": session_id,
                "exception_type": type(e).__name__,
            },
        )
        push_line(session_id, f"[EXCEPTION] {e}")
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        finish_session(session_id)
