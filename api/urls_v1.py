"""
URL configuration for the public REST API — version 1.

All routes here are mounted under /api/v1/ in webKinPred/urls.py.

Endpoints:

    GET  /api/v1/health/              — Service health check (no auth)
    GET  /api/v1/methods/             — List available methods (no auth)
    GET  /api/v1/quota/               — Check remaining quota (auth required)
    POST /api/v1/validate/            — Validate CSV without submitting (auth required)
                                        Add runSimilarity=true for MMseqs2 analysis.
    POST /api/v1/submit/              — Submit a prediction job (auth required)
    GET  /api/v1/status/<public_id>/  — Poll job status (auth required)
    GET  /api/v1/result/<public_id>/  — Download results (auth required)
                                        Add ?format=json for JSON output.
"""

from django.urls import path

from api.views import v1_views

urlpatterns = [
    # -------------------------------------------------------------------------
    # No authentication required
    # -------------------------------------------------------------------------
    path(
        "health/",
        v1_views.api_health,
        name="api_v1_health",
    ),
    path(
        "methods/",
        v1_views.api_list_methods,
        name="api_v1_methods",
    ),

    # -------------------------------------------------------------------------
    # Authentication required (Bearer token)
    # -------------------------------------------------------------------------
    path(
        "quota/",
        v1_views.api_quota,
        name="api_v1_quota",
    ),
    path(
        "validate/",
        v1_views.api_validate,
        name="api_v1_validate",
    ),
    path(
        "submit/",
        v1_views.api_submit_job,
        name="api_v1_submit",
    ),
    path(
        "status/<slug:public_id>/",
        v1_views.api_job_status,
        name="api_v1_status",
    ),
    path(
        "result/<slug:public_id>/",
        v1_views.api_download_result,
        name="api_v1_result",
    ),
]
