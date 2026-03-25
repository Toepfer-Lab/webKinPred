"""
Job-specific utility functions for job submission and management.
"""
import json
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from django.conf import settings
from django.utils import timezone
from api.utils import get_experimental

TARGET_ORDER = ["kcat", "Km", "kcat/Km"]
VALID_TARGETS = set(TARGET_ORDER)


def coerce_bool_param(value: Any, default: bool = False) -> bool:
    """
    Coerce common HTML/JSON boolean representations to a Python bool.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def canonicalise_targets(targets: List[str]) -> List[str]:
    """
    Return deduplicated targets in canonical display/execution order.
    """
    out: List[str] = []
    for target in TARGET_ORDER:
        if target in targets and target not in out:
            out.append(target)
    return out


def canonical_prediction_type(targets: List[str]) -> str:
    """
    Build a compact, human-readable prediction_type label for persisted jobs.
    """
    ordered = canonicalise_targets(targets)
    return "+".join(ordered)


def validate_prediction_parameters(
    targets: List[str],
    methods: Dict[str, str],
) -> Optional[str]:
    """
    Validate target and method parameters against the method registry.

    Returns an error message if validation fails, None if valid.
    """
    from api.methods.registry import all_methods

    if not isinstance(targets, list) or not targets:
        return (
            'Invalid targets. Expected a non-empty list with values from '
            '"kcat", "Km", "kcat/Km".'
        )

    if not isinstance(methods, dict):
        return (
            'Invalid methods payload. Expected an object mapping target names '
            "to method keys."
        )

    invalid_targets = [t for t in targets if t not in VALID_TARGETS]
    if invalid_targets:
        return (
            "Invalid target(s): "
            + ", ".join(map(str, invalid_targets))
            + '. Allowed targets: "kcat", "Km", "kcat/Km".'
        )

    extra_method_keys = [k for k in methods.keys() if k not in VALID_TARGETS]
    if extra_method_keys:
        return (
            "Invalid method mapping keys: "
            + ", ".join(map(str, extra_method_keys))
            + '. Allowed keys: "kcat", "Km", "kcat/Km".'
        )

    registry = all_methods()
    for target in canonicalise_targets(targets):
        method_key = methods.get(target)
        if not isinstance(method_key, str) or not method_key.strip():
            return f"Missing method selection for target '{target}'."

        desc = registry.get(method_key)
        if desc is None or target not in desc.supports:
            valid = sorted(k for k, d in registry.items() if target in d.supports)
            return (
                f"Invalid method '{method_key}' for target '{target}'. "
                f"Available {target} methods: {', '.join(valid)}."
            )

    return None


def validate_sequence_handling_option(handle_long_seq: str) -> Optional[str]:
    """
    Validate the sequence handling option parameter.

    Returns an error message if invalid, None if valid.
    """
    if handle_long_seq not in ("truncate", "skip"):
        return 'Invalid handleLongSeq value. Expected "truncate" or "skip".'
    return None


def determine_required_columns(
    targets: List[str],
    methods: Dict[str, str],
) -> List[str]:
    """
    Determine strict required columns for the selected target/method set.

    The result always includes "Protein Sequence".  Additional columns are
    derived from each selected descriptor's col_to_kwarg mapping.
    """
    from api.methods.registry import get

    required: set[str] = {"Protein Sequence"}

    for target in canonicalise_targets(targets):
        method_key = methods.get(target)
        if not method_key:
            continue
        try:
            desc = get(method_key)
            required.update(desc.col_to_kwarg.keys())
        except KeyError:
            pass

    return list(required)


def validate_required_columns_for_methods(
    dataframe: pd.DataFrame,
    targets: List[str],
    methods: Dict[str, str],
) -> Optional[str]:
    """
    Validate CSV columns for selected methods with multi-substrate bridge support.

    Methods requiring "Substrate" are considered valid when either:
    - "Substrate" exists, or
    - both "Substrates" and "Products" exist (bridge path).
    """
    from api.methods.registry import get

    missing: set[str] = set()
    if "Protein Sequence" not in dataframe.columns:
        missing.add("Protein Sequence")

    has_substrate = "Substrate" in dataframe.columns
    has_multi = {"Substrates", "Products"}.issubset(set(dataframe.columns))
    needs_single_substrate = False

    for target in canonicalise_targets(targets):
        method_key = methods.get(target)
        if not method_key:
            continue
        try:
            desc = get(method_key)
        except KeyError:
            continue

        for col in desc.col_to_kwarg.keys():
            if col == "Substrate":
                needs_single_substrate = True
                continue
            if col not in dataframe.columns:
                missing.add(col)

    if needs_single_substrate and not (has_substrate or has_multi):
        missing.add("Substrate")

    if not missing:
        return None

    ordered = sorted(missing, key=lambda c: (c != "Protein Sequence", c))
    if ordered == ["Substrate"]:
        return (
            "Missing required columns: Substrate (or provide both Substrates and "
            "Products for multi-substrate bridge mode)."
        )
    return f'Missing required columns: {", ".join(ordered)}'


def create_job_directory(public_id: str) -> str:
    """
    Create directory structure for a job.

    Returns the path to the created job directory.
    """
    job_dir = os.path.join(settings.MEDIA_ROOT, "jobs", str(public_id))
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def save_job_input_file(file, job_dir: str) -> str:
    """
    Save the input CSV file to the job directory.

    Returns the path to the saved file.
    """
    file_path = os.path.join(job_dir, "input.csv")
    file.seek(0)
    input_df = pd.read_csv(file)
    input_df.dropna(how="all", inplace=True)
    input_df.to_csv(file_path, index=False)
    return file_path


def get_experimental_results(
    use_experimental: bool,
    methods: Dict[str, str],
    targets: List[str],
    dataframe: pd.DataFrame,
) -> Optional[Dict[str, list]]:
    """
    Look up experimental kinetic values when the user has opted in.

    Experimental lookup is skipped for multi-substrate methods (TurNup) since
    the experimental database is indexed by single substrates.

    Returns a dict keyed by target ("kcat", "Km"), or None.
    """
    if not use_experimental:
        return None

    if "Substrate" not in dataframe.columns:
        return None

    selected = set(targets)
    out: Dict[str, list] = {}

    if "kcat" in selected:
        kcat_method = methods.get("kcat")
        if kcat_method:
            try:
                from api.methods.registry import get

                desc = get(kcat_method)
                if desc.input_format != "multi":
                    out["kcat"] = get_experimental.lookup_experimental(
                        dataframe["Protein Sequence"].tolist(),
                        dataframe["Substrate"].tolist(),
                        param_type="kcat",
                    )
            except KeyError:
                pass

    if "Km" in selected:
        out["Km"] = get_experimental.lookup_experimental(
            dataframe["Protein Sequence"].tolist(),
            dataframe["Substrate"].tolist(),
            param_type="Km",
        )

    return out or None


def extract_job_parameters_from_request(request) -> Dict[str, Any]:
    """
    Extract job parameters from an HTTP request.

    Returns a parameters dictionary used by process_job_submission_from_params.
    """
    parse_error = ""
    targets: Any = request.POST.get("targets", "[]")
    methods: Any = request.POST.get("methods", "{}")

    try:
        if isinstance(targets, str):
            targets = json.loads(targets) if targets.strip() else []
    except json.JSONDecodeError:
        parse_error = (
            "Invalid 'targets' format. Expected a JSON array, for example: "
            '["kcat", "Km"].'
        )
        targets = []

    try:
        if isinstance(methods, str):
            methods = json.loads(methods) if methods.strip() else {}
    except json.JSONDecodeError:
        parse_error = (
            "Invalid 'methods' format. Expected a JSON object, for example: "
            '{"kcat":"DLKcat","Km":"UniKP"}.'
        )
        methods = {}

    return {
        "use_experimental": coerce_bool_param(
            request.POST.get("useExperimental"),
            default=False,
        ),
        "canonicalize_substrates": coerce_bool_param(
            request.POST.get("canonicalizeSubstrates"),
            default=True,
        ),
        "targets": targets if isinstance(targets, list) else [],
        "methods": methods if isinstance(methods, dict) else {},
        "handle_long_sequences": request.POST.get("handleLongSequences", "truncate"),
        "_parse_error": parse_error,
    }


def create_rate_limit_headers(
    daily_limit: int, remaining: int, ttl: int
) -> Dict[str, str]:
    """
    Create standard rate-limiting headers for HTTP responses.
    """
    return {
        "X-RateLimit-Limit": str(daily_limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(ttl),
    }


def create_job_status_response_data(job) -> Dict[str, Any]:
    """
    Create a response data dictionary for the job-status endpoint.
    """
    return {
        "public_id": job.public_id,
        "status": job.status,
        "submission_time": job.submission_time,
        "completion_time": job.completion_time,
        "server_time": timezone.now(),
        "elapsed_seconds": (
            int(max(0, (job.completion_time - job.submission_time).total_seconds()))
            if job.completion_time
            else int(max(0, (timezone.now() - job.submission_time).total_seconds()))
        ),
        "error_message": job.error_message,
        "total_molecules": job.total_molecules,
        "molecules_processed": job.molecules_processed,
        "invalid_rows": job.invalid_rows,
        "total_predictions": job.total_predictions,
        "predictions_made": job.predictions_made,
    }
