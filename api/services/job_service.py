"""
Job service that orchestrates job submission and management workflows.
"""
from typing import Any, Dict, Optional, Tuple

from django.http import JsonResponse

from api.models import Job
from api.tasks import (
    run_both_predictions,
    run_dlkcat_predictions,
    run_eitlem_predictions,
    run_kinform_h_predictions,
    run_kinform_l_predictions,
    run_turnup_predictions,
    run_unikp_predictions,
)
from api.utils.job_utils import (
    create_job_directory,
    create_job_status_response_data,
    create_rate_limit_headers,
    determine_required_columns,
    get_experimental_results,
    save_job_input_file,
    validate_prediction_parameters,
    validate_sequence_handling_option,
)
from api.utils.quotas import (
    DAILY_LIMIT,
    get_client_ip,
    get_or_create_user,
    reserve_or_reject,
)
from api.utils.validation_utils import (
    parse_csv_file,
    validate_column_emptiness,
    validate_required_columns,
)


def process_job_submission(
    request, file
) -> Tuple[Optional[JsonResponse], Optional[Dict[str, Any]]]:
    """
    Process a job submission from the web UI.

    Extracts parameters from the Django request object (form fields + IP) and
    delegates to the shared ``process_job_submission_from_params`` function.

    Args:
        request: Django HTTP request (POST with multipart form data).
        file:    Uploaded CSV file extracted from request.FILES.

    Returns:
        Tuple of (error_response, success_data).
        On success: (None, {"message": ..., "public_id": ...})
        On failure: (JsonResponse with error, None)
    """
    from api.utils.job_utils import extract_job_parameters_from_request

    params = extract_job_parameters_from_request(request)
    ip_address = get_client_ip(request)
    return process_job_submission_from_params(params, file, ip_address)


def process_job_submission_from_params(
    params: Dict[str, Any], file, ip_address: str
) -> Tuple[Optional[JsonResponse], Optional[Dict[str, Any]]]:
    """
    Core job-submission logic, decoupled from the HTTP request object.

    This function is called by both the web-UI view (via
    ``process_job_submission``) and the public API v1 submit endpoint.  It
    accepts an explicit params dict and IP address so it can be used
    regardless of how the caller obtained those values.

    Args:
        params:     Dict with keys:
                      prediction_type      – "kcat", "Km", or "both"
                      kcat_method          – e.g. "DLKcat", "EITLEM", …
                      km_method            – e.g. "EITLEM", "UniKP", …
                      handle_long_sequences– "truncate" or "skip"
                      use_experimental     – bool
        file:       A file-like object (Django InMemoryUploadedFile or
                    io.BytesIO) containing the CSV data.
        ip_address: The IP address to charge quota against.

    Returns:
        Tuple of (error_response, success_data).
        On success: (None, {"message": ..., "public_id": ...})
        On failure: (JsonResponse with error, None)
    """
    # --- Validate parameters ---------------------------------------------------

    param_error = validate_prediction_parameters(
        params["prediction_type"],
        params["kcat_method"],
        params["km_method"],
    )
    if param_error:
        return JsonResponse({"error": param_error}, status=400), None

    seq_handling_error = validate_sequence_handling_option(
        params["handle_long_sequences"]
    )
    if seq_handling_error:
        return JsonResponse({"error": seq_handling_error}, status=400), None

    # --- Parse and validate the CSV --------------------------------------------

    try:
        dataframe = parse_csv_file(file)
    except Exception as e:
        return JsonResponse({"error": f"Could not read CSV file: {e}"}, status=400), None

    required_columns = determine_required_columns(
        params["prediction_type"],
        params["kcat_method"],
        params["km_method"],
    )

    column_error = validate_required_columns(dataframe, required_columns)
    if column_error:
        return JsonResponse({"error": column_error}, status=400), None

    # Ensure key columns are not mostly empty.
    substrate_col = "Substrate" if "Substrate" in dataframe.columns else None
    if substrate_col:
        emptiness_error = validate_column_emptiness(dataframe, substrate_col)
        if emptiness_error:
            return JsonResponse({"error": emptiness_error}, status=400), None

    sequence_error = validate_column_emptiness(dataframe, "Protein Sequence")
    if sequence_error:
        return JsonResponse({"error": sequence_error}, status=400), None

    # --- Quota -----------------------------------------------------------------

    quota_response = handle_quota_validation(ip_address, len(dataframe))
    if quota_response:
        return quota_response, None

    # --- Create job record and dispatch task -----------------------------------

    try:
        user = get_or_create_user(ip_address)
    except Exception as e:
        print(f"Warning: could not create/update ApiUser for {ip_address}: {e}")
        user = None

    experimental_results = get_experimental_results(
        params["use_experimental"],
        params["kcat_method"],
        dataframe,
        params["prediction_type"],
    )

    job = create_job_record(params, ip_address, len(dataframe), user)

    job_dir = create_job_directory(job.public_id)
    save_job_input_file(file, job_dir)

    dispatch_prediction_task(job.public_id, params, experimental_results)

    return None, {
        "message": "Job submitted successfully",
        "public_id": job.public_id,
    }


def handle_quota_validation(ip_address: str, requested_rows: int) -> Optional[JsonResponse]:
    """
    Handle quota validation and return error response if quota exceeded.
    
    Args:
        ip_address: Client IP address
        requested_rows: Number of rows being requested
        
    Returns:
        JsonResponse with error if quota exceeded, None if allowed
    """
    allowed, remaining, ttl = reserve_or_reject(ip_address, requested_rows)

    rate_headers = create_rate_limit_headers(DAILY_LIMIT, remaining, ttl)
    
    if not allowed:
        error_response = JsonResponse({
            "error": (
                f"Upload rejected: daily limit exceeded. "
                f"{remaining} predictions remaining today; this upload requires {requested_rows}."
            )
        }, status=429)
        
        # Add rate limiting headers
        for key, value in rate_headers.items():
            error_response[key] = value
        
        return error_response
    
    return None


def create_job_record(params: Dict[str, Any], ip_address: str, requested_rows: int, user) -> Job:
    """
    Create and save a new job record.
    
    Args:
        params: Job parameters dictionary
        ip_address: Client IP address
        requested_rows: Number of rows in the request
        user: User model instance
        
    Returns:
        Created Job instance
    """
    job = Job(
        prediction_type=params['prediction_type'],
        kcat_method=params['kcat_method'],
        km_method=params['km_method'],
        status="Pending",
        handle_long_sequences=params['handle_long_sequences'],
        ip_address=ip_address,
        requested_rows=requested_rows,
        user=user,
    )
    job.save()
    print("Saved Job:", job.public_id)
    return job


def dispatch_prediction_task(public_id: str, params: Dict[str, Any], experimental_results: Optional[Dict]) -> None:
    """
    Dispatch appropriate prediction task based on parameters.
    
    Args:
        public_id: Job public ID
        params: Job parameters
        experimental_results: Experimental results if available
    """
    prediction_type = params['prediction_type']
    kcat_method = params['kcat_method']
    km_method = params['km_method']
    print(f"DEBUG: Dispatching task: {prediction_type}, {kcat_method}, {km_method}")
    if prediction_type == "both":
        run_both_predictions.delay(public_id, experimental_results)
    elif prediction_type == "kcat":
        method_to_func = {
            "DLKcat": run_dlkcat_predictions,
            "TurNup": run_turnup_predictions,
            "EITLEM": run_eitlem_predictions,
            "UniKP": run_unikp_predictions,
            "KinForm-H": run_kinform_h_predictions,
            "KinForm-L": run_kinform_l_predictions,
        }
        pred_func = method_to_func.get(kcat_method)
        if pred_func:
            pred_func.delay(public_id, experimental_results)
        else:
            print("No valid prediction function found for the given method.")
    elif prediction_type == "Km":
        method_to_func = {
            "EITLEM": run_eitlem_predictions,
            "UniKP": run_unikp_predictions,
            "KinForm-H": run_kinform_h_predictions,
        }
        pred_func = method_to_func.get(km_method)
        if pred_func:
            pred_func.delay(public_id, experimental_results)
            print("Dispatching task to Celery:", prediction_type, kcat_method, km_method)
        else:
            print("No valid prediction function found for the given method.")


def get_job_status_data(job: Job) -> Dict[str, Any]:
    """
    Get formatted job status data.
    
    Args:
        job: Job model instance
        
    Returns:
        Dictionary containing job status information
    """
    return create_job_status_response_data(job)
