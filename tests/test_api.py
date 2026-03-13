"""
test_api.py — Comprehensive API test suite for the KineticXPredictor REST API.

Tests every endpoint, every expected success response, and every expected error
response.  Designed to run against a local Django dev server.

Usage:
    python tests/test_api.py                        # uses the default key below
    python tests/test_api.py --key ak_yourkey       # pass a key on the command line
    python tests/test_api.py --base http://host:8000/api/v1   # different server
"""

import argparse
import io
import json
import sys
import textwrap
import time

import requests

# ---------------------------------------------------------------------------
# Configuration — edit or pass via CLI flags
# ---------------------------------------------------------------------------

DEFAULT_BASE = "http://127.0.0.1:8000/api/v1"
DEFAULT_KEY  = "ak_17f90e7c1f6ac3f5fc861d8cec4667a2b888c358a333bb81f75b631a9b50066b"

# ---------------------------------------------------------------------------
# Known method IDs (normalised to lowercase for comparison)
# ---------------------------------------------------------------------------

# kcat-capable methods
KCAT_METHOD_IDS = ["DLKcat", "TurNup", "EITLEM", "UniKP", "KinForm-H", "KinForm-L"]
# Km-capable methods
KM_METHOD_IDS   = ["EITLEM", "UniKP", "KinForm-H"]
# All recognised method IDs (de-duplicated, lowercase)
ALL_METHOD_IDS  = sorted({m.lower() for m in KCAT_METHOD_IDS + KM_METHOD_IDS})


def sel(methods: set, *names: str) -> bool:
    """Return True if *any* of the given method names appear in the selected set."""
    return any(n.lower() in methods for n in names)


# ---------------------------------------------------------------------------
# Tiny inline CSV fixtures
# ---------------------------------------------------------------------------

# Standard single-substrate CSV (used by DLKcat, EITLEM, UniKP, KinForm-*)
# Substrates use simple SMILES strings (no commas) to avoid CSV parsing issues.
SINGLE_SUBSTRATE_CSV = textwrap.dedent("""\
    Protein Sequence,Substrate
    MAAAALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O
    MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTKAGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDELPALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYLGEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQPGYDHSYYFIASFIEDHLRFHAQYLLK,c1ccccc1
    MCTAITLNGNSNYFGRNLDLDFSYGEEVIITPAEYEFKFRKEKAIKNHKSLIGVGIVANDYPLYFDAINEDGLGMAGLNFPGNAYYSDALENDKDNITPFEFIPWILGQCSDVNEARNLVEKINLINLSFSEQLPLAGLHWLIADREKSIVVEVTKSGVHIYDNPIGILTNNPEFNYQMYNLNKYRNLSISTPQNTFSDSVDLKVDGTGFGGIGLPGDVSPESRFVRATFSKLNSSKGMTVEEDITQFFHILGTVEQIKGVNKTESGKEEYTVYSNCYDLDNKTLYYTTYENRQIVAVTLNKDKDGNRLVTYPFERKQIINKLN,OCC(O)CO
""")

# Multi-substrate CSV (for TurNup only — uses Substrates + Products columns)
MULTI_SUBSTRATE_CSV = textwrap.dedent("""\
    Protein Sequence,Substrates,Products
    MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O;O,CC(O)=O;[H+]
    MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTKAGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDELPALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYLGEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQPGYDHSYYFIASFIEDHLRFHAQYLLK,C1CCCCC1;O,OC1CCCCC1;[H+]
""")

# CSV missing the required "Substrate" column — to test validation errors
MISSING_COLUMN_CSV = textwrap.dedent("""\
    Protein Sequence,Extra Column
    MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,foo
""")

# CSV with one valid row and one row with an invalid substrate + invalid protein.
# Used to verify that /validate/ correctly detects invalid content.
INVALID_CONTENT_CSV = textwrap.dedent("""\
    Protein Sequence,Substrate
    MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O
    !!!NOT_A_VALID_AMINO_ACID_SEQUENCE_123!!!,NOT_A_VALID_SMILES_OR_INCHI_STRING
""")

# A plain text file that is NOT a CSV — for extension validation
NOT_A_CSV_BYTES = b"this is not a csv file"

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []   # (name, passed, detail)


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Record and print a single assertion."""
    status = "PASS" if condition else "FAIL"
    colour = "\033[32m" if condition else "\033[31m"
    reset  = "\033[0m"
    pad    = "." * max(0, 65 - len(name))
    print(f"  {colour}{status}{reset}  {name}{pad} {detail}")
    _results.append((name, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def csv_file(content: str, filename: str = "input.csv"):
    """Return a (name, file-object, mime-type) tuple for requests.post files=."""
    return (filename, io.BytesIO(content.encode("utf-8")), "text/csv")


def submit(base: str, headers: dict, csv_content: str,
           prediction_type: str, kcat_method: str = None,
           km_method: str = None, handle_long: str = "truncate",
           use_experimental: str = "false") -> requests.Response:
    """Helper: POST to /submit/ and return the response."""
    data = {
        "predictionType":      prediction_type,
        "handleLongSequences": handle_long,
        "useExperimental":     use_experimental,
    }
    if kcat_method:
        data["kcatMethod"] = kcat_method
    if km_method:
        data["kmMethod"] = km_method

    return requests.post(
        f"{base}/submit/",
        headers=headers,
        files={"file": csv_file(csv_content)},
        data=data,
    )


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

def test_health(base: str) -> None:
    section("GET /health/ — no auth required")
    r = requests.get(f"{base}/health/")
    check("status 200",       r.status_code == 200,  f"got {r.status_code}")
    j = r.json()
    check("has status=ok",    j.get("status") == "ok")
    check("has service key",  "service" in j)
    check("has version key",  "version" in j)
    check("has timestamp",    "timestamp" in j)


def test_methods(base: str, methods: set) -> None:
    section("GET /methods/ — no auth required")
    r = requests.get(f"{base}/methods/")
    check("status 200",              r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("has predictionTypes",     "predictionTypes" in j)
    check("kcat in predictionTypes", "kcat" in j.get("predictionTypes", []))
    check("Km in predictionTypes",   "Km"   in j.get("predictionTypes", []))
    check("both in predictionTypes", "both" in j.get("predictionTypes", []))
    check("has methods.kcat",        isinstance(j.get("methods", {}).get("kcat"), list))
    check("has methods.Km",          isinstance(j.get("methods", {}).get("Km"), list))
    kcat_ids = {m["id"] for m in j.get("methods", {}).get("kcat", [])}
    if sel(methods, "DLKcat"):   check("DLKcat in kcat methods",  "DLKcat"    in kcat_ids)
    if sel(methods, "TurNup"):   check("TurNup in kcat methods",  "TurNup"    in kcat_ids)
    if sel(methods, "EITLEM"):   check("EITLEM in kcat methods",  "EITLEM"    in kcat_ids)
    if sel(methods, "UniKP"):    check("UniKP in kcat methods",   "UniKP"     in kcat_ids)
    if sel(methods, "KinForm-H"): check("KinForm-H in kcat",      "KinForm-H" in kcat_ids)
    if sel(methods, "KinForm-L"): check("KinForm-L in kcat",      "KinForm-L" in kcat_ids)
    km_ids = {m["id"] for m in j.get("methods", {}).get("Km", [])}
    if sel(methods, "EITLEM"):   check("EITLEM in Km methods",    "EITLEM"    in km_ids)
    if sel(methods, "UniKP"):    check("UniKP in Km methods",     "UniKP"     in km_ids)
    if sel(methods, "KinForm-H"): check("KinForm-H in Km methods", "KinForm-H" in km_ids)
    check("has longSequenceOptions", "longSequenceOptions" in j)


def test_auth(base: str, key: str) -> None:
    section("Authentication — invalid / missing keys")

    # No Authorization header at all
    r = requests.get(f"{base}/quota/")
    check("no header → 401",   r.status_code == 401, f"got {r.status_code}")
    check("error key present", "error" in r.json())

    # Wrong Authorization format (not "Bearer ...")
    r = requests.get(f"{base}/quota/", headers={"Authorization": "Basic abc123"})
    check("non-Bearer scheme → 401", r.status_code == 401, f"got {r.status_code}")

    # Bearer prefix present but key does not exist in DB
    r = requests.get(f"{base}/quota/", headers={"Authorization": "Bearer ak_doesnotexist"})
    check("fake key → 401",    r.status_code == 401, f"got {r.status_code}")

    # Valid key
    r = requests.get(f"{base}/quota/", headers={"Authorization": f"Bearer {key}"})
    check("valid key → 200",   r.status_code == 200, f"got {r.status_code}")


def test_quota(base: str, headers: dict) -> None:
    section("GET /quota/")
    r = requests.get(f"{base}/quota/", headers=headers)
    check("status 200",           r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("has limit",            "limit"          in j)
    check("has used",             "used"           in j)
    check("has remaining",        "remaining"      in j)
    check("has resetsInSeconds",  "resetsInSeconds" in j)
    check("limit > 0",            j.get("limit", 0) > 0)
    check("remaining = limit−used",
          j.get("remaining") == j.get("limit", 0) - j.get("used", 0))


def test_submit_success(base: str, headers: dict, methods: set) -> dict | None:
    """
    Submit a valid job using the first available kcat method from *methods*.
    Returns the job dict so later tests can poll its status and attempt to
    download its result.  Returns None if no kcat method is selected.
    """
    # Pick the first kcat method that the user asked to test.
    kcat_method = next(
        (m for m in KCAT_METHOD_IDS if m.lower() in methods), None
    )
    if kcat_method is None:
        print("\n  (skipping submit/status/result tests — no kcat method selected)")
        return None

    # TurNup requires a different CSV format; fall back if it's the only option.
    csv_content = MULTI_SUBSTRATE_CSV if kcat_method == "TurNup" else SINGLE_SUBSTRATE_CSV

    section(f"POST /submit/ — valid CSV file upload (kcat / {kcat_method})")
    r = submit(base, headers, csv_content, "kcat", kcat_method=kcat_method)
    check("status 201",          r.status_code == 201, f"got {r.status_code}")
    j = r.json()
    check("has jobId",           "jobId"     in j)
    check("status=Pending",      j.get("status") == "Pending")
    check("has statusUrl",       "statusUrl" in j)
    check("has resultUrl",       "resultUrl" in j)
    check("has quota",           "quota"     in j)
    q = j.get("quota", {})
    check("quota has remaining", "remaining" in q)
    check("quota has limit",     "limit"     in q)
    remaining = q.get('remaining', '?')
    rem_str = f"{remaining:,}" if isinstance(remaining, int) else str(remaining)
    print(f"         → Job ID: {j.get('jobId')}  |  Method: {kcat_method}  |  Quota remaining: {rem_str}")
    return j


def test_submit_km(base: str, headers: dict, methods: set) -> None:
    # Pick the first Km method that the user asked to test.
    km_method = next(
        (m for m in KM_METHOD_IDS if m.lower() in methods), None
    )
    if km_method is None:
        print("\n  (skipping Km submit test — no Km-capable method selected)")
        return
    section(f"POST /submit/ — valid CSV file upload (Km / {km_method})")
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "Km", km_method=km_method)
    check("status 201",     r.status_code == 201, f"got {r.status_code}")
    check("has jobId",      "jobId" in r.json())


def test_submit_both(base: str, headers: dict, methods: set) -> None:
    kcat_method = next(
        (m for m in KCAT_METHOD_IDS if m.lower() in methods), None
    )
    km_method = next(
        (m for m in KM_METHOD_IDS if m.lower() in methods), None
    )
    if kcat_method is None or km_method is None:
        print("\n  (skipping 'both' submit test — need at least one kcat and one Km method selected)")
        return
    section(f"POST /submit/ — valid CSV file upload (both / {kcat_method} + {km_method})")
    csv_content = MULTI_SUBSTRATE_CSV if kcat_method == "TurNup" else SINGLE_SUBSTRATE_CSV
    r = submit(base, headers, csv_content, "both",
               kcat_method=kcat_method, km_method=km_method)
    check("status 201",     r.status_code == 201, f"got {r.status_code}")
    check("has jobId",      "jobId" in r.json())


def test_submit_turnup(base: str, headers: dict, methods: set) -> None:
    if not sel(methods, "TurNup"):
        print("\n  (skipping TurNup submit test — not in selected methods)")
        return
    section("POST /submit/ — multi-substrate CSV (kcat / TurNup)")
    r = submit(base, headers, MULTI_SUBSTRATE_CSV, "kcat", kcat_method="TurNup")
    check("status 201",     r.status_code == 201, f"got {r.status_code}")
    check("has jobId",      "jobId" in r.json())


def test_submit_json_body(base: str, headers: dict, methods: set) -> None:
    kcat_method = next(
        (m for m in KCAT_METHOD_IDS if m.lower() in methods and m != "TurNup"), None
    )  # TurNup needs multi-substrate format; pick any other kcat method
    if kcat_method is None:
        print("\n  (skipping JSON body submit test — no non-TurNup kcat method selected)")
        return
    section(f"POST /submit/ — JSON body (inline data, no CSV file) [{kcat_method}]")
    json_headers = {**headers, "Content-Type": "application/json"}
    payload = {
        "predictionType":      "kcat",
        "kcatMethod":          kcat_method,
        "handleLongSequences": "truncate",
        "useExperimental":     False,
        "data": [
            {
                "Protein Sequence": (
                    "MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAEL"
                    "IEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKS"
                    "GHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYF"
                    "YPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIER"
                    "WPGMPE"
                ),
                "Substrate": "CC(=O)O",
            },
            {
                "Protein Sequence": (
                    "MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTK"
                    "AGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDEL"
                    "PALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYL"
                    "GEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQ"
                    "PGYDHSYYFIASFIEDHLRFHAQYLLK"
                ),
                "Substrate": "C1CCCCC1",
            },
        ],
    }
    r = requests.post(f"{base}/submit/", headers=json_headers, json=payload)
    check("status 201",     r.status_code == 201, f"got {r.status_code}")
    j = r.json()
    check("has jobId",      "jobId"  in j)
    check("status=Pending", j.get("status") == "Pending")


def test_submit_errors(base: str, headers: dict) -> None:
    section("POST /submit/ — validation error cases (all should fail cleanly)")

    # No file attached
    r = requests.post(f"{base}/submit/", headers=headers, data={
        "predictionType": "kcat", "kcatMethod": "DLKcat",
        "handleLongSequences": "truncate",
    })
    check("no file → 400",              r.status_code == 400, f"got {r.status_code}")
    check("error key present",          "error" in r.json())

    # File with wrong extension (.txt instead of .csv)
    r = requests.post(f"{base}/submit/", headers=headers,
                      files={"file": ("input.txt", io.BytesIO(NOT_A_CSV_BYTES), "text/plain")},
                      data={"predictionType": "kcat", "kcatMethod": "DLKcat",
                            "handleLongSequences": "truncate"})
    check("non-.csv extension → 400",   r.status_code == 400, f"got {r.status_code}")

    # Invalid predictionType
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "invalid_type")
    check("bad predictionType → 400",   r.status_code == 400, f"got {r.status_code}")

    # Missing kcatMethod when predictionType=kcat
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat", kcat_method="NOTAMETHOD")
    check("bad kcatMethod → 400",       r.status_code == 400, f"got {r.status_code}")

    # Valid predictionType but wrong method for it (KinForm-L is not a Km method)
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "Km", km_method="KinForm-L")
    check("KinForm-L for Km → 400",     r.status_code == 400, f"got {r.status_code}")

    # handleLongSequences with invalid value
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat",
               kcat_method="DLKcat", handle_long="invalid_value")
    check("bad handleLongSeq → 400",    r.status_code == 400, f"got {r.status_code}")

    # CSV missing required "Substrate" column
    r = submit(base, headers, MISSING_COLUMN_CSV, "kcat", kcat_method="DLKcat")
    check("missing column → 400",       r.status_code == 400, f"got {r.status_code}")

    # TurNup with single-substrate CSV (missing Substrates + Products columns)
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat", kcat_method="TurNup")
    check("TurNup+wrong CSV → 400",     r.status_code == 400, f"got {r.status_code}")

    # JSON body with empty data array
    r = requests.post(
        f"{base}/submit/",
        headers={**headers, "Content-Type": "application/json"},
        json={"predictionType": "kcat", "kcatMethod": "DLKcat",
              "handleLongSequences": "truncate", "data": []},
    )
    check("empty JSON data → 400",      r.status_code == 400, f"got {r.status_code}")

    # JSON body exceeding 10,000 row limit
    big_data = [{"Protein Sequence": "M" * 10, "Substrate": "C"} for _ in range(10_001)]
    r = requests.post(
        f"{base}/submit/",
        headers={**headers, "Content-Type": "application/json"},
        json={"predictionType": "kcat", "kcatMethod": "DLKcat",
              "handleLongSequences": "truncate", "data": big_data},
    )
    check("10001-row JSON body → 400",  r.status_code == 400, f"got {r.status_code}")


def test_status(base: str, headers: dict, job: dict | None) -> None:
    if job is None:
        return
    section("GET /status/<jobId>/ — job status polling")
    job_id = job["jobId"]

    # Valid status request
    r = requests.get(f"{base}/status/{job_id}/", headers=headers)
    check("status 200",            r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("jobId matches",         j.get("jobId") == job_id)
    check("status field present",  "status" in j)
    check("status is known value",
          j.get("status") in {"Pending", "Processing", "Completed", "Failed"})
    check("submittedAt present",   "submittedAt" in j)
    check("elapsedSeconds ≥ 0",    j.get("elapsedSeconds", -1) >= 0)
    check("progress present",      "progress" in j)
    prog = j.get("progress", {})
    check("progress.moleculesTotal",    "moleculesTotal"    in prog)
    check("progress.predictionsTotal",  "predictionsTotal"  in prog)

    # Non-existent job
    r = requests.get(f"{base}/status/NOTAREALIDXXX/", headers=headers)
    check("fake jobId → 404",      r.status_code == 404, f"got {r.status_code}")
    check("error key present",     "error" in r.json())


def test_result_not_ready(base: str, headers: dict, job: dict | None) -> None:
    if job is None:
        return
    """
    Unless the job completed instantly (unlikely in tests), result should
    return 409 Conflict because the job is still Pending/Processing.
    """
    section("GET /result/<jobId>/ — result before job completes")
    job_id = job["jobId"]

    r = requests.get(f"{base}/result/{job_id}/", headers=headers)
    # The job might have completed if a worker picked it up; handle both cases.
    if r.status_code == 200:
        check("result available (job already done)",  True)
    else:
        check("pending job → 409",  r.status_code == 409, f"got {r.status_code}")
        check("error key present",  "error" in r.json())

    # Non-existent job
    r = requests.get(f"{base}/result/NOTAREALIDXXX/", headers=headers)
    check("fake jobId → 404",  r.status_code == 404, f"got {r.status_code}")


def test_result_completed(base: str, headers: dict, job: dict | None,
                          poll_timeout: int = 1000) -> None:
    """
    Poll the job submitted by test_submit_success until it completes, then
    download its results in both CSV and JSON formats.

    Falls back to a local DB lookup only when running against localhost and
    no submitted job is available (e.g. worker not running).
    poll_timeout — seconds to wait for the job to finish (default 1000 s / 16.67 min).
    """
    section("GET /result/<jobId>/ — downloading a completed job")

    job_id: str | None = job.get("jobId") if job else None

    if job_id:
        # Poll the status endpoint until the job is done (or we time out).
        print(f"         → Polling job {job_id} until completion "
              f"(timeout {poll_timeout}s)…")
        deadline = time.time() + poll_timeout
        interval = 5
        while time.time() < deadline:
            r = requests.get(f"{base}/status/{job_id}/", headers=headers)
            if r.status_code != 200:
                break
            status = r.json().get("status", "")
            if status in ("Completed", "Failed"):
                break
            time.sleep(interval)
            interval = min(interval * 1.5, 30)  # back off up to 30 s
        else:
            print(f"         (skipped — job did not complete within {poll_timeout}s)")
            return

        # Re-check final status
        r = requests.get(f"{base}/status/{job_id}/", headers=headers)
        if r.status_code == 200 and r.json().get("status") == "Failed":
            check("job completed successfully (not Failed)", False,
                  f"job {job_id} ended in Failed state")
            return
        if r.status_code != 200 or r.json().get("status") != "Completed":
            print(f"         (skipped — job {job_id} not yet completed; "
                  f"status={r.json().get('status', '?')})")
            return

        print(f"         → Job {job_id} completed — downloading result")
    else:
        # No submitted job available — attempt a local DB fallback (dev only).
        import os as _os
        import sys as _sys
        _sys.path.insert(0, "/home/saleh/webKinPred")
        _os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
        try:
            import django
            django.setup()
            from api.models import Job as _Job
            completed = _Job.objects.filter(
                status="Completed", output_file__isnull=False
            ).exclude(output_file="").order_by("-completion_time").first()
        except Exception as e:
            print(f"         (skipped — no submitted job and could not query DB: {e})")
            return
        if not completed:
            print("         (skipped — no completed jobs in the database)")
            return
        job_id = completed.public_id
        print(f"         → Using DB-found completed job: {job_id}")

    # CSV download
    r = requests.get(f"{base}/result/{job_id}/", headers=headers)
    check("CSV status 200",           r.status_code == 200, f"got {r.status_code}")
    check("content-type is text/csv",
          "text/csv" in r.headers.get("Content-Type", ""),
          r.headers.get("Content-Type"))
    check("non-empty body",           len(r.content) > 0)

    # JSON format
    r = requests.get(f"{base}/result/{job_id}/?format=json", headers=headers)
    check("JSON status 200",          r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("JSON has jobId",           j.get("jobId") == job_id)
    check("JSON has columns list",    isinstance(j.get("columns"), list))
    check("JSON has rowCount",        isinstance(j.get("rowCount"), int))
    check("JSON has data list",       isinstance(j.get("data"), list))
    check("rowCount matches data",    j.get("rowCount") == len(j.get("data", [])))
    if j.get("data"):
        first_row = j["data"][0]
        check("row has Protein Sequence", "Protein Sequence" in first_row)


def test_validate(base: str, headers: dict) -> None:
    """
    POST /validate/ — input validation without running predictions.

    Covers:
      - Valid CSV → all-clear response
      - Invalid content CSV → invalid substrates + proteins detected
      - JSON body submission
      - Auth errors
      - Input errors (no file, bad extension, missing columns)
    """
    section("POST /validate/ — input validation (no similarity)")

    # ── Valid CSV — expect no issues ─────────────────────────────────────────
    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(SINGLE_SUBSTRATE_CSV)},
        data={"runSimilarity": "false"},
    )
    if check("valid CSV → 200",             r.status_code == 200, f"got {r.status_code}"):
        j = r.json()
        check("has rowCount",                "rowCount"          in j)
        check("rowCount = 3",                j.get("rowCount")   == 3)
        check("has invalidSubstrates",       "invalidSubstrates" in j)
        check("has invalidProteins",         "invalidProteins"   in j)
        check("has lengthViolations",        "lengthViolations"  in j)
        check("has similarity key",          "similarity"        in j)
        check("similarity is null",          j.get("similarity") is None)
        check("no invalid substrates",       len(j.get("invalidSubstrates", [1])) == 0)
        check("no invalid proteins",         len(j.get("invalidProteins",   [1])) == 0)

    # ── Invalid content — one valid row + one invalid row ────────────────────
    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(INVALID_CONTENT_CSV)},
        data={"runSimilarity": "false"},
    )
    if check("invalid CSV → 200",           r.status_code == 200, f"got {r.status_code}"):
        j = r.json()
        check("rowCount = 2",                j.get("rowCount") == 2)
        check("detects invalid substrate",   len(j.get("invalidSubstrates", [])) > 0)
        check("detects invalid protein",     len(j.get("invalidProteins",   [])) > 0)

    # ── JSON body ────────────────────────────────────────────────────────────
    r = requests.post(
        f"{base}/validate/",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "data": [
                {
                    "Protein Sequence": (
                        "MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAEL"
                        "IEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKS"
                        "GHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYF"
                        "YPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIER"
                        "WPGMPE"
                    ),
                    "Substrate": "CC(=O)O",
                },
            ],
            "runSimilarity": False,
        },
    )
    if check("JSON body → 200",             r.status_code == 200, f"got {r.status_code}"):
        j = r.json()
        check("JSON rowCount = 1",           j.get("rowCount") == 1)
        check("JSON similarity = null",      j.get("similarity") is None)

    # ── Auth errors ──────────────────────────────────────────────────────────
    r = requests.post(f"{base}/validate/",
                      files={"file": csv_file(SINGLE_SUBSTRATE_CSV)})
    check("no auth → 401",               r.status_code == 401, f"got {r.status_code}")

    # ── Input errors ─────────────────────────────────────────────────────────
    r = requests.post(f"{base}/validate/", headers=headers)
    check("no file → 400",               r.status_code == 400, f"got {r.status_code}")

    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": ("input.txt", io.BytesIO(NOT_A_CSV_BYTES), "text/plain")},
    )
    check("non-.csv extension → 400",    r.status_code == 400, f"got {r.status_code}")

    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(MISSING_COLUMN_CSV)},
    )
    check("missing columns → 400",       r.status_code == 400, f"got {r.status_code}")

    r = requests.post(
        f"{base}/validate/",
        headers={**headers, "Content-Type": "application/json"},
        json={"data": [], "runSimilarity": False},
    )
    check("empty JSON data → 400",       r.status_code == 400, f"got {r.status_code}")


def test_validate_similarity(base: str, headers: dict) -> None:
    """
    POST /validate/ with runSimilarity=true.

    This is a blocking synchronous call — the HTTP response is only returned
    once MMseqs2 has finished analysing all sequences.  Allow up to 10 minutes.
    Skipped gracefully if mmseqs2 is unavailable on this server.
    """
    section("POST /validate/ — with runSimilarity=true (blocking, may be slow)")
    print("         → sending request and waiting for MMseqs2 to complete…")

    try:
        r = requests.post(
            f"{base}/validate/",
            headers=headers,
            files={"file": csv_file(SINGLE_SUBSTRATE_CSV)},
            data={"runSimilarity": "true"},
            timeout=600,
        )
    except requests.exceptions.Timeout:
        print("         (skipped — request timed out after 10 minutes)")
        return

    if not check("status 200",           r.status_code == 200, f"got {r.status_code}"):
        return

    j = r.json()
    sim = j.get("similarity")
    check("similarity is not null",      sim is not None)
    if sim is None:
        return

    # The similarity service may return {"error": "..."} if mmseqs2 is unavailable
    if "error" in sim:
        check("similarity succeeded (mmseqs2 available)",
              False, sim["error"])
        return

    check("similarity is a dict",        isinstance(sim, dict))
    check("similarity has method keys",  len(sim) > 0)

    # Verify structure of at least one method's result
    for method_name, method_data in sim.items():
        if not isinstance(method_data, dict):
            continue
        check(f"{method_name} has histogram_max",         "histogram_max"          in method_data)
        check(f"{method_name} has histogram_mean",        "histogram_mean"         in method_data)
        check(f"{method_name} has average_max_similarity","average_max_similarity" in method_data)
        check(f"{method_name} has count_max",             "count_max"              in method_data)
        hist = method_data.get("histogram_max", [])
        check(f"{method_name} histogram has 101 bins",    len(hist) == 101)
        break


def test_wrong_methods(base: str, headers: dict) -> None:
    """Make sure method-not-allowed cases are handled (submit must be POST)."""
    section("HTTP method errors")
    # GET on submit/ should not work the same way as POST
    r = requests.get(f"{base}/submit/", headers=headers)
    # Django returns 405 for wrong method on csrf_exempt views, or our own 405.
    check("GET /submit/ → 4xx",  r.status_code >= 400, f"got {r.status_code}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KineticXPredictor API test suite")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Base URL including /api/v1 (default: %(default)s)")
    parser.add_argument("--key",  default=DEFAULT_KEY,
                        help="API Bearer token (default: hardcoded test key)")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip the slow runSimilarity=true test (requires MMseqs2)")
    parser.add_argument("--poll-timeout", type=int, default=1000, metavar="SECONDS",
                        help="Seconds to wait for a submitted job to complete before "
                             "skipping the result-download test (default: 1000)")
    parser.add_argument(
        "--methods",
        default="all",
        metavar="METHOD[,METHOD…]",
        help=(
            "Comma-separated list of prediction methods to test. "
            f"Recognised values (case-insensitive): {', '.join(ALL_METHOD_IDS)}. "
            "Use 'all' (default) to test every method."
        ),
    )
    args = parser.parse_args()

    base    = args.base.rstrip("/")
    key     = args.key
    headers = {"Authorization": f"Bearer {key}"}

    # Build the normalised set of selected methods.
    if args.methods.strip().lower() == "all":
        methods: set = {m.lower() for m in ALL_METHOD_IDS}
    else:
        methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
        unknown = methods - {m.lower() for m in ALL_METHOD_IDS}
        if unknown:
            parser.error(
                f"Unknown method(s): {', '.join(sorted(unknown))}. "
                f"Valid choices: {', '.join(ALL_METHOD_IDS)}"
            )

    print("=" * 70)
    print("  KineticXPredictor API Test Suite")
    print(f"  Base URL : {base}")
    print(f"  API Key  : {key[:15]}…")
    print(f"  Methods  : {', '.join(sorted(methods))}")
    print("=" * 70)

    # Run all test groups
    test_health(base)
    test_methods(base, methods)
    test_auth(base, key)
    test_quota(base, headers)

    # Submit a real job — we'll use its ID for status / result tests
    job = test_submit_success(base, headers, methods)

    # Other valid submission variants
    test_submit_km(base, headers, methods)
    test_submit_both(base, headers, methods)
    test_submit_turnup(base, headers, methods)
    test_submit_json_body(base, headers, methods)

    # All the ways submission can fail
    test_submit_errors(base, headers)

    # Status polling
    test_status(base, headers, job)

    # Result endpoint (job not done yet)
    test_result_not_ready(base, headers, job)

    # Result endpoint (poll the submitted job until done, then download)
    test_result_completed(base, headers, job, poll_timeout=args.poll_timeout)

    # Validate endpoint — fast checks (no similarity)
    test_validate(base, headers)

    # Validate endpoint — with MMseqs2 similarity (slow, blocking)
    if args.skip_similarity:
        print("\n  (skipping similarity test — pass without --skip-similarity to run it)")
    else:
        test_validate_similarity(base, headers)

    # Method-not-allowed sanity check
    test_wrong_methods(base, headers)

    # ── Summary ─────────────────────────────────────────────────────────────
    total   = len(_results)
    passed  = sum(1 for _, ok, _ in _results if ok)
    failed  = total - passed

    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
    else:
        print("  — all tests passed!")
    print("=" * 70)

    if failed:
        print("\nFailed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  ✗  {name}  {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
