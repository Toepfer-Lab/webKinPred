"""
Public REST API — version 1 endpoints.

All endpoints live under the /api/v1/ prefix (see api/urls_v1.py).

Authentication
--------------
Most endpoints require an API key delivered as a Bearer token:

    Authorization: Bearer ak_<your_key>

The two exceptions are /health/ and /methods/, which are intentionally public
so that clients can discover available methods and check server status without
needing credentials.

Quota
-----
Quota accounting uses the IP address stored on the ApiUser record associated
with the API key (not the request's source IP).  This keeps quota consistent
regardless of where the script is run.
"""

import io
import json

import pandas as pd
from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from api.models import Job
from api.services.job_service import process_job_submission_from_params
from api.services.validation_service import validate_input_file
from api.services.similarity_service import analyze_sequence_similarity
from api.utils.api_auth import require_api_key
from api.utils.quotas import get_quota_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_error(message: str, status: int = 400) -> JsonResponse:
    """Return a consistently formatted error response."""
    return JsonResponse({"error": message}, status=status)


def _quota_dict(ip_address: str) -> dict:
    """Return quota data in the camelCase shape expected by API clients."""
    usage = get_quota_usage(ip_address)
    return {
        "limit": usage["limit"],
        "used": usage["used"],
        "remaining": usage["remaining"],
        "resetsInSeconds": usage["reset_in_seconds"],
    }


# ---------------------------------------------------------------------------
# GET /api/v1/health/
# ---------------------------------------------------------------------------

@csrf_exempt
def api_health(request):
    """
    Health check endpoint — no authentication required.

    Returns a simple JSON object confirming the service is up.
    """
    return JsonResponse(
        {
            "status": "ok",
            "service": "KineticXPredictor API",
            "version": "1",
            "timestamp": timezone.now().isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/v1/methods/
# ---------------------------------------------------------------------------

@csrf_exempt
def api_list_methods(request):
    """
    List all available prediction methods and their requirements.

    No authentication required — this endpoint acts as living documentation
    so clients can discover what methods exist and what CSV columns they need
    before writing any code.
    """
    methods = {
        "kcat": [
            {
                "id": "DLKcat",
                "name": "DLKcat",
                "description": (
                    "Deep-learning model for kcat prediction using enzyme sequence "
                    "and substrate structure."
                ),
                "maxSequenceLength": None,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
            {
                "id": "TurNup",
                "name": "TurNup",
                "description": (
                    "Machine-learning model optimised for natural wild-type reactions "
                    "with multi-substrate / multi-product support."
                ),
                "maxSequenceLength": 1024,
                "requiredColumns": ["Protein Sequence", "Substrates", "Products"],
                "substrateFormat": "Semicolon-separated SMILES or InChI strings",
            },
            {
                "id": "EITLEM",
                "name": "EITLEM",
                "description": (
                    "Ensemble deep-learning model that predicts kcat and KM "
                    "simultaneously from enzyme sequence and substrate."
                ),
                "maxSequenceLength": 1024,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
            {
                "id": "UniKP",
                "name": "UniKP",
                "description": (
                    "Unified kinetic parameter predictor for kcat and KM, "
                    "based on protein language model embeddings."
                ),
                "maxSequenceLength": 1000,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
            {
                "id": "KinForm-H",
                "name": "KinForm-H",
                "description": (
                    "KinForm variant recommended for enzymes with high sequence "
                    "similarity to the training set."
                ),
                "maxSequenceLength": 1500,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
            {
                "id": "KinForm-L",
                "name": "KinForm-L",
                "description": (
                    "KinForm variant recommended for enzymes with low sequence "
                    "similarity to the training set."
                ),
                "maxSequenceLength": 1500,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
        ],
        "Km": [
            {
                "id": "EITLEM",
                "name": "EITLEM",
                "description": "Predicts KM from enzyme sequence and substrate.",
                "maxSequenceLength": 1024,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
            {
                "id": "UniKP",
                "name": "UniKP",
                "description": "Predicts KM from enzyme sequence and substrate.",
                "maxSequenceLength": 1000,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
            {
                "id": "KinForm-H",
                "name": "KinForm-H",
                "description": (
                    "Predicts KM — recommended when the enzyme has high similarity "
                    "to the training data."
                ),
                "maxSequenceLength": 1500,
                "requiredColumns": ["Protein Sequence", "Substrate"],
                "substrateFormat": "SMILES or InChI",
            },
        ],
    }

    return JsonResponse(
        {
            "predictionTypes": ["kcat", "Km", "both"],
            "methods": methods,
            "longSequenceOptions": {
                "truncate": "Shorten sequences that exceed the model's maximum length.",
                "skip": "Omit rows where the sequence exceeds the model's maximum length.",
            },
            "notes": {
                "both": (
                    "When predictionType is 'both', you must specify both kcatMethod "
                    "and kmMethod.  KinForm-L is not available as a Km method."
                ),
                "quota": "Each row in your CSV counts as one prediction against your daily quota.",
            },
        }
    )


# ---------------------------------------------------------------------------
# GET /api/v1/quota/
# ---------------------------------------------------------------------------

@require_api_key
def api_quota(request):
    """
    Return the current quota status for the authenticated key owner.

    Quota is tracked per-day (resetting at midnight UTC) and per-user
    (keyed by the IP address stored on the ApiUser record).
    """
    return JsonResponse(_quota_dict(request.api_ip))


# ---------------------------------------------------------------------------
# POST /api/v1/submit/
# ---------------------------------------------------------------------------

@csrf_exempt
@require_api_key
def api_submit_job(request):
    """
    Submit a new prediction job.

    Accepts two content types:

    1. multipart/form-data — upload a CSV file plus form fields:
         file                  (required) — the CSV file
         predictionType        (required) — "kcat", "Km", or "both"
         kcatMethod            (required if predictionType is "kcat" or "both")
         kmMethod              (required if predictionType is "Km" or "both")
         handleLongSequences   (optional, default "truncate") — "truncate" or "skip"
         useExperimental       (optional, default "false")   — "true" or "false"

    2. application/json — send data directly as a JSON body:
         {
           "predictionType": "kcat",
           "kcatMethod": "DLKcat",
           "handleLongSequences": "truncate",
           "useExperimental": false,
           "data": [
             {"Protein Sequence": "MKTL...", "Substrate": "CC(=O)O"},
             ...
           ]
         }

    Both paths call the same underlying job-submission service, so validation,
    quota accounting, and task dispatch behave identically.

    On success, returns a JSON object with:
      - jobId       — use this to poll /status/ and download /result/
      - statusUrl   — convenience URL for polling
      - resultUrl   — convenience URL for downloading results
      - quota       — your remaining quota after this submission
    """
    if request.method != "POST":
        return _json_error("This endpoint only accepts POST requests.", 405)

    content_type = request.content_type or ""

    if "application/json" in content_type:
        csv_file, params, error = _parse_json_body(request)
    else:
        csv_file, params, error = _parse_multipart_body(request)

    if error:
        return error

    error_response, success_data = process_job_submission_from_params(
        params, csv_file, request.api_ip
    )

    if error_response:
        return error_response

    public_id = success_data["public_id"]

    return JsonResponse(
        {
            "jobId": public_id,
            "status": "Pending",
            "statusUrl": f"/api/v1/status/{public_id}/",
            "resultUrl": f"/api/v1/result/{public_id}/",
            "quota": _quota_dict(request.api_ip),
        },
        status=201,
    )


def _parse_multipart_body(request):
    """
    Extract CSV file and parameters from a multipart/form-data request.

    Returns (csv_file, params, error).  On success, error is None.
    """
    csv_file = request.FILES.get("file")

    if not csv_file:
        return None, None, _json_error(
            "No file provided. Include 'file' as a multipart field."
        )

    if not csv_file.name.lower().endswith(".csv"):
        return None, None, _json_error("The uploaded file must have a .csv extension.")

    params = {
        "prediction_type": request.POST.get("predictionType"),
        "kcat_method": request.POST.get("kcatMethod"),
        "km_method": request.POST.get("kmMethod"),
        "handle_long_sequences": request.POST.get("handleLongSequences", "truncate"),
        "use_experimental": request.POST.get("useExperimental", "false").lower() == "true",
    }

    return csv_file, params, None


def _parse_json_body(request):
    """
    Extract CSV data and parameters from an application/json request body.

    The caller supplies the data as an array of row objects under the key
    "data".  We convert this to an in-memory CSV file so the same downstream
    validation and processing code can handle it without modification.

    Returns (csv_file, params, error).  On success, error is None.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None, _json_error("Request body is not valid JSON.")

    rows = body.get("data")

    if not rows or not isinstance(rows, list):
        return None, None, _json_error(
            "'data' must be a non-empty array of row objects. "
            "Each object should map column names to values, e.g. "
            '{"Protein Sequence": "MKTL...", "Substrate": "CC(=O)O"}.'
        )

    if len(rows) > 10_000:
        return None, None, _json_error(
            f"JSON body submission is limited to 10,000 rows. "
            f"You submitted {len(rows):,}.  Use CSV file upload for larger datasets."
        )

    # Convert the list of dicts to an in-memory CSV file-like object.
    try:
        df = pd.DataFrame(rows)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        csv_bytes = io.BytesIO(buffer.getvalue().encode("utf-8"))
        csv_bytes.name = "input.csv"  # downstream code may inspect .name
    except Exception as e:
        return None, None, _json_error(f"Could not convert 'data' to CSV: {e}")

    params = {
        "prediction_type": body.get("predictionType"),
        "kcat_method": body.get("kcatMethod"),
        "km_method": body.get("kmMethod"),
        "handle_long_sequences": body.get("handleLongSequences", "truncate"),
        "use_experimental": bool(body.get("useExperimental", False)),
    }

    return csv_bytes, params, None


# ---------------------------------------------------------------------------
# GET /api/v1/status/<public_id>/
# ---------------------------------------------------------------------------

@require_api_key
def api_job_status(request, public_id):
    """
    Return the current status of a prediction job.

    Poll this endpoint until status is "Completed" or "Failed", then
    fetch your results from the resultUrl.

    Possible status values:
      Pending    — job is queued and waiting for a worker
      Processing — worker is running predictions
      Completed  — predictions are done; results are ready to download
      Failed     — something went wrong; see the 'error' field for details
    """
    try:
        job = Job.objects.get(public_id=public_id)
    except Job.DoesNotExist:
        return _json_error(f"No job found with id '{public_id}'.", status=404)

    elapsed = (
        int((job.completion_time - job.submission_time).total_seconds())
        if job.completion_time
        else int((timezone.now() - job.submission_time).total_seconds())
    )

    data = {
        "jobId": job.public_id,
        "status": job.status,
        "submittedAt": job.submission_time.isoformat(),
        "elapsedSeconds": max(0, elapsed),
        "progress": {
            "moleculesTotal": job.total_molecules,
            "moleculesProcessed": job.molecules_processed,
            "predictionsTotal": job.total_predictions,
            "predictionsMade": job.predictions_made,
            "invalidMolecules": job.invalid_molecules,
        },
    }

    if job.status == "Completed":
        data["completedAt"] = job.completion_time.isoformat()
        data["resultUrl"] = f"/api/v1/result/{public_id}/"

    if job.status == "Failed":
        data["error"] = job.error_message or "An unknown error occurred."

    return JsonResponse(data)


# ---------------------------------------------------------------------------
# GET /api/v1/result/<public_id>/
# ---------------------------------------------------------------------------

@require_api_key
def api_download_result(request, public_id):
    """
    Download the prediction results for a completed job.

    By default, returns a CSV file attachment.  Add ?format=json to receive
    the same data as a JSON object instead — useful when you want to parse
    results directly without writing an intermediate file.

    The response includes all columns from the input CSV plus:
      - A prediction column (e.g. "kcat (1/s)" or "KM (mM)")
      - A "Source" column indicating whether the value came from a model
        or from the experimental database
      - An "Extra Info" column with additional details

    Returns 409 Conflict if the job has not yet completed.
    Returns 404 if the job does not exist.
    Returns 500 if the output file is missing (should not normally occur).
    """
    try:
        job = Job.objects.get(public_id=public_id)
    except Job.DoesNotExist:
        return _json_error(f"No job found with id '{public_id}'.", status=404)

    if job.status != "Completed":
        return _json_error(
            f"Results are not yet available — job status is '{job.status}'. "
            "Poll /status/ until the job is Completed.",
            status=409,
        )

    if not job.output_file:
        return _json_error(
            "The output file for this job is missing. Please contact support.",
            status=500,
        )

    output_path = job.output_file.path

    # --- JSON format ----------------------------------------------------------
    if request.GET.get("format") == "json":
        try:
            df = pd.read_csv(output_path)
        except Exception as e:
            return _json_error(f"Could not read output file: {e}", status=500)

        return JsonResponse(
            {
                "jobId": public_id,
                "columns": list(df.columns),
                "rowCount": len(df),
                "data": df.to_dict(orient="records"),
            }
        )

    # --- Default: CSV download ------------------------------------------------
    try:
        file_handle = open(output_path, "rb")
    except OSError as e:
        return _json_error(f"Could not open output file: {e}", status=500)

    filename = f"webkinpred-{slugify(public_id)}-results.csv"
    return FileResponse(
        file_handle,
        as_attachment=True,
        filename=filename,
        content_type="text/csv",
    )


# ---------------------------------------------------------------------------
# POST /api/v1/validate/
# ---------------------------------------------------------------------------

@csrf_exempt
@require_api_key
def api_validate(request):
    """
    Validate a CSV file (or inline JSON data) without submitting a prediction
    job.  Checks substrate SMILES/InChI strings, protein amino-acid sequences,
    and sequence-length limits for every available model.

    Optionally runs MMseqs2 sequence similarity analysis against each method's
    training database when runSimilarity=true.  This is a synchronous call that
    blocks until the analysis is complete — it may take several minutes for
    large inputs.  No quota is consumed by this endpoint.

    Accepts the same two content-type formats as /submit/:

    1. multipart/form-data
         file           (required) — the CSV file
         runSimilarity  (optional, default "false") — "true" to include similarity

    2. application/json
         {
           "data":          [...],      // required — array of row objects
           "runSimilarity": false       // optional
         }

    Response fields:
      rowCount          — total number of data rows in the input
      invalidSubstrates — list of rows with unparseable SMILES/InChI strings
      invalidProteins   — list of rows with invalid amino-acid sequences
      lengthViolations  — per-model violation counts and sequence-length limits
      similarity        — null when not requested; otherwise a dict keyed by
                          method name, each containing histogram_max,
                          histogram_mean, average_max_similarity,
                          average_mean_similarity, count_max, count_mean
    """
    if request.method != "POST":
        return _json_error("This endpoint only accepts POST requests.", 405)

    content_type = request.content_type or ""

    if "application/json" in content_type:
        csv_file, run_similarity, error = _parse_validate_json_body(request)
    else:
        csv_file, run_similarity, error = _parse_validate_multipart_body(request)

    if error:
        return error

    # Count rows (fast peek — parse once, then reset file pointer)
    try:
        csv_file.seek(0)
        row_count = len(pd.read_csv(csv_file))
        csv_file.seek(0)
    except Exception as e:
        return _json_error(f"Could not parse CSV: {e}")

    # Run substrate + protein + length validation
    validation_result = validate_input_file(csv_file)

    if "error" in validation_result:
        return _json_error(
            validation_result["error"],
            status=validation_result.get("status_code", 400),
        )

    # Optionally run MMseqs2 sequence similarity analysis
    similarity = None
    if run_similarity:
        try:
            csv_file.seek(0)
            similarity = analyze_sequence_similarity(csv_file, session_id="api_validate")
        except Exception as e:
            # Surface the error as a structured field rather than a 500 —
            # the validation results are still useful even if similarity fails.
            similarity = {"error": str(e)}

    return JsonResponse({
        "rowCount": row_count,
        "invalidSubstrates": validation_result["invalid_substrates"],
        "invalidProteins": validation_result["invalid_proteins"],
        "lengthViolations": validation_result["length_violations"],
        "similarity": similarity,
    })


def _parse_validate_multipart_body(request):
    """
    Extract CSV file and runSimilarity flag from a multipart/form-data request.

    Returns (csv_file, run_similarity, error).  On success, error is None.
    """
    csv_file = request.FILES.get("file")

    if not csv_file:
        return None, False, _json_error(
            "No file provided. Include 'file' as a multipart field."
        )

    if not csv_file.name.lower().endswith(".csv"):
        return None, False, _json_error("The uploaded file must have a .csv extension.")

    run_similarity = request.POST.get("runSimilarity", "false").lower() == "true"
    return csv_file, run_similarity, None


def _parse_validate_json_body(request):
    """
    Extract inline data and runSimilarity flag from an application/json request.

    Returns (csv_file, run_similarity, error).  On success, error is None.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, False, _json_error("Request body is not valid JSON.")

    rows = body.get("data")

    if not rows or not isinstance(rows, list):
        return None, False, _json_error(
            "'data' must be a non-empty array of row objects. "
            "Each object should map column names to values, e.g. "
            '{"Protein Sequence": "MKTL...", "Substrate": "CC(=O)O"}.'
        )

    if len(rows) > 10_000:
        return None, False, _json_error(
            f"JSON body submission is limited to 10,000 rows. "
            f"You submitted {len(rows):,}.  Use CSV file upload for larger datasets."
        )

    try:
        df = pd.DataFrame(rows)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        csv_bytes = io.BytesIO(buffer.getvalue().encode("utf-8"))
        csv_bytes.name = "input.csv"
    except Exception as e:
        return None, False, _json_error(f"Could not convert 'data' to CSV: {e}")

    run_similarity = bool(body.get("runSimilarity", False))
    return csv_bytes, run_similarity, None
