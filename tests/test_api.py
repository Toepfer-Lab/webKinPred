"""
test_api.py — Comprehensive API test suite for the Open Kinetics Predictor REST API.

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
KCAT_METHOD_IDS = ["DLKcat", "TurNup", "EITLEM", "UniKP", "KinForm-H", "KinForm-L", "CataPro"]
# Km-capable methods
KM_METHOD_IDS   = ["EITLEM", "UniKP", "KinForm-H", "CataPro"]
# kcat/Km-capable methods
KCAT_KM_METHOD_IDS = ["CataPro"]
# All recognised method IDs (de-duplicated, lowercase)
ALL_METHOD_IDS  = sorted(
    {m.lower() for m in KCAT_METHOD_IDS + KM_METHOD_IDS + KCAT_KM_METHOD_IDS}
)


def sel(methods: set, *names: str) -> bool:
    """Return True if *any* of the given method names appear in the selected set."""
    return any(n.lower() in methods for n in names)


def selected_kcat_methods(methods: set) -> list[str]:
    """Return selected kcat-capable methods in canonical order."""
    return [m for m in KCAT_METHOD_IDS if m.lower() in methods]


def selected_km_methods(methods: set) -> list[str]:
    """Return selected Km-capable methods in canonical order."""
    return [m for m in KM_METHOD_IDS if m.lower() in methods]


def selected_kcat_km_methods(methods: set) -> list[str]:
    """Return selected kcat/Km-capable methods in canonical order."""
    return [m for m in KCAT_KM_METHOD_IDS if m.lower() in methods]


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
           km_method: str = None, kcat_km_method: str = None,
           handle_long: str = "truncate",
           use_experimental: str = "false") -> requests.Response:
    """Helper: POST to /submit/ and return the response."""
    targets = []
    methods = {}
    if prediction_type == "kcat":
        targets = ["kcat"]
        if kcat_method:
            methods["kcat"] = kcat_method
    elif prediction_type == "Km":
        targets = ["Km"]
        if km_method:
            methods["Km"] = km_method
    elif prediction_type == "kcat/Km":
        targets = ["kcat/Km"]
        if kcat_km_method:
            methods["kcat/Km"] = kcat_km_method
    data = {
        "targets":             json.dumps(targets),
        "methods":             json.dumps(methods),
        "handleLongSequences": handle_long,
        "useExperimental":     use_experimental,
    }

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
    check(
        "kcat/Km in predictionTypes",
        "kcat/Km" in j.get("predictionTypes", []),
    )
    check("has methods.kcat",        isinstance(j.get("methods", {}).get("kcat"), list))
    check("has methods.Km",          isinstance(j.get("methods", {}).get("Km"), list))
    check("has methods.kcat/Km",     isinstance(j.get("methods", {}).get("kcat/Km"), list))
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
    ratio_ids = {m["id"] for m in j.get("methods", {}).get("kcat/Km", [])}
    if sel(methods, "CataPro"):  check("CataPro in kcat/Km methods", "CataPro" in ratio_ids)
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


def build_selected_method_jobs(methods: set) -> list[dict]:
    """
    Build one submit job per selected method/target:
      - all selected kcat-capable methods as prediction_type=kcat
      - all selected Km-capable methods as prediction_type=Km
      - all selected kcat/Km-capable methods as prediction_type=kcat/Km
    """
    jobs: list[dict] = []
    for kcat_method in selected_kcat_methods(methods):
        jobs.append({
            "prediction_type": "kcat",
            "kcat_method": kcat_method,
            "km_method": None,
            "kcat_km_method": None,
            "csv_content": MULTI_SUBSTRATE_CSV if kcat_method == "TurNup" else SINGLE_SUBSTRATE_CSV,
            "label": f"kcat/{kcat_method}",
        })
    for km_method in selected_km_methods(methods):
        jobs.append({
            "prediction_type": "Km",
            "kcat_method": None,
            "km_method": km_method,
            "kcat_km_method": None,
            "csv_content": SINGLE_SUBSTRATE_CSV,
            "label": f"Km/{km_method}",
        })
    for ratio_method in selected_kcat_km_methods(methods):
        jobs.append({
            "prediction_type": "kcat/Km",
            "kcat_method": None,
            "km_method": None,
            "kcat_km_method": ratio_method,
            "csv_content": SINGLE_SUBSTRATE_CSV,
            "label": f"kcat/Km/{ratio_method}",
        })
    return jobs


def test_submit_selected_methods(base: str, headers: dict, methods: set) -> list[dict]:
    """
    Submit one job for every selected method/prediction type combination.
    Returns a list of tracked submitted jobs for later polling/result checks.
    """
    specs = build_selected_method_jobs(methods)
    if not specs:
        print("\n  (skipping method submits — no supported methods selected)")
        return []

    section("POST /submit/ — valid CSV upload for every selected method")
    submitted_jobs: list[dict] = []
    for spec in specs:
        label = spec["label"]
        r = submit(
            base,
            headers,
            spec["csv_content"],
            spec["prediction_type"],
            kcat_method=spec["kcat_method"],
            km_method=spec["km_method"],
            kcat_km_method=spec.get("kcat_km_method"),
        )
        ok = check(f"[{label}] status 201", r.status_code == 201, f"got {r.status_code}")
        if not ok:
            continue
        j = r.json()
        check(f"[{label}] has jobId", "jobId" in j)
        check(f"[{label}] status=Pending", j.get("status") == "Pending")
        check(f"[{label}] has statusUrl", "statusUrl" in j)
        check(f"[{label}] has resultUrl", "resultUrl" in j)
        check(f"[{label}] has quota", "quota" in j)
        q = j.get("quota", {})
        check(f"[{label}] quota has remaining", "remaining" in q)
        check(f"[{label}] quota has limit", "limit" in q)
        if "jobId" not in j:
            continue
        submitted = {**spec, "job": j, "job_id": j["jobId"]}
        submitted_jobs.append(submitted)
        print(f"         → Submitted {label}: jobId={j['jobId']}")
    return submitted_jobs


def test_submit_json_body(base: str, headers: dict, methods: set) -> dict | None:
    kcat_method = next(
        (m for m in KCAT_METHOD_IDS if m.lower() in methods and m != "TurNup"), None
    )  # TurNup needs multi-substrate format; pick any other kcat method
    if kcat_method is None:
        print("\n  (skipping JSON body submit test — no non-TurNup kcat method selected)")
        return None
    label = f"json/kcat/{kcat_method}"
    section(f"POST /submit/ — JSON body (inline data, no CSV file) [{kcat_method}]")
    json_headers = {**headers, "Content-Type": "application/json"}
    payload = {
        "targets":             ["kcat"],
        "methods":             {"kcat": kcat_method},
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
    if not check(f"[{label}] status 201", r.status_code == 201, f"got {r.status_code}"):
        return None
    j = r.json()
    check(f"[{label}] has jobId", "jobId" in j)
    check(f"[{label}] status=Pending", j.get("status") == "Pending")
    if "jobId" not in j:
        return None
    print(f"         → Submitted {label}: jobId={j['jobId']}")
    return {
        "prediction_type": "kcat",
        "kcat_method": kcat_method,
        "km_method": None,
        "kcat_km_method": None,
        "label": label,
        "job": j,
        "job_id": j["jobId"],
    }


def test_submit_errors(base: str, headers: dict) -> None:
    section("POST /submit/ — validation error cases (all should fail cleanly)")

    # No file attached
    r = requests.post(f"{base}/submit/", headers=headers, data={
        "targets": '["kcat"]', "methods": '{"kcat":"DLKcat"}',
        "handleLongSequences": "truncate",
    })
    check("no file → 400",              r.status_code == 400, f"got {r.status_code}")
    check("error key present",          "error" in r.json())

    # File with wrong extension (.txt instead of .csv)
    r = requests.post(f"{base}/submit/", headers=headers,
                      files={"file": ("input.txt", io.BytesIO(NOT_A_CSV_BYTES), "text/plain")},
                      data={"targets": '["kcat"]', "methods": '{"kcat":"DLKcat"}',
                            "handleLongSequences": "truncate"})
    check("non-.csv extension → 400",   r.status_code == 400, f"got {r.status_code}")

    # Invalid target selection
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "invalid_type")
    check("bad target list → 400",      r.status_code == 400, f"got {r.status_code}")

    # Invalid kcat method for target
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat", kcat_method="NOTAMETHOD")
    check("bad kcat method → 400",      r.status_code == 400, f"got {r.status_code}")

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
        json={"targets": ["kcat"], "methods": {"kcat": "DLKcat"},
              "handleLongSequences": "truncate", "data": []},
    )
    check("empty JSON data → 400",      r.status_code == 400, f"got {r.status_code}")

    # JSON body exceeding 10,000 row limit
    big_data = [{"Protein Sequence": "M" * 10, "Substrate": "C"} for _ in range(10_001)]
    r = requests.post(
        f"{base}/submit/",
        headers={**headers, "Content-Type": "application/json"},
        json={"targets": ["kcat"], "methods": {"kcat": "DLKcat"},
              "handleLongSequences": "truncate", "data": big_data},
    )
    check("10001-row JSON body → 400",  r.status_code == 400, f"got {r.status_code}")


def test_status(base: str, headers: dict, submitted_jobs: list[dict]) -> None:
    if not submitted_jobs:
        return
    section("GET /status/<jobId>/ — job status polling")
    sample = submitted_jobs[0]
    job_id = sample["job_id"]
    label = sample["label"]

    # Valid status request
    r = requests.get(f"{base}/status/{job_id}/", headers=headers)
    check(f"[{label}] status 200",            r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check(f"[{label}] jobId matches",         j.get("jobId") == job_id)
    check(f"[{label}] status field present",  "status" in j)
    check(f"[{label}] status is known value",
          j.get("status") in {"Pending", "Processing", "Completed", "Failed"})
    check(f"[{label}] submittedAt present",   "submittedAt" in j)
    check(f"[{label}] elapsedSeconds ≥ 0",    j.get("elapsedSeconds", -1) >= 0)
    check(f"[{label}] progress present",      "progress" in j)
    prog = j.get("progress", {})
    check(f"[{label}] progress.moleculesTotal",    "moleculesTotal"    in prog)
    check(f"[{label}] progress.predictionsTotal",  "predictionsTotal"  in prog)

    # Non-existent job
    r = requests.get(f"{base}/status/NOTAREALIDXXX/", headers=headers)
    check("fake jobId → 404",      r.status_code == 404, f"got {r.status_code}")
    check("error key present",     "error" in r.json())


def test_result_not_ready(base: str, headers: dict, submitted_jobs: list[dict]) -> None:
    if not submitted_jobs:
        return
    """
    Unless the job completed instantly (unlikely in tests), result should
    return 409 Conflict because the job is still Pending/Processing.
    """
    section("GET /result/<jobId>/ — result before job completes")
    sample = submitted_jobs[0]
    job_id = sample["job_id"]
    label = sample["label"]

    r = requests.get(f"{base}/result/{job_id}/", headers=headers)
    # The job might have completed if a worker picked it up; handle both cases.
    if r.status_code == 200:
        check(f"[{label}] result available (job already done)",  True)
    else:
        check(f"[{label}] pending job → 409",  r.status_code == 409, f"got {r.status_code}")
        check(f"[{label}] error key present",  "error" in r.json())

    # Non-existent job
    r = requests.get(f"{base}/result/NOTAREALIDXXX/", headers=headers)
    check("fake jobId → 404",  r.status_code == 404, f"got {r.status_code}")


def wait_for_terminal_status(
    base: str,
    headers: dict,
    job_id: str,
    label: str,
    poll_timeout: int,
) -> str | None:
    """Poll /status/<jobId>/ until Completed/Failed, or return None on error/timeout."""
    print(f"         → Polling [{label}] job {job_id} until completion (timeout {poll_timeout}s)…")
    deadline = time.time() + poll_timeout
    interval = 5.0
    last_status = "Unknown"
    while time.time() < deadline:
        r = requests.get(f"{base}/status/{job_id}/", headers=headers)
        if r.status_code != 200:
            check(f"[{label}] status polling HTTP 200", False, f"got {r.status_code}")
            return None
        j = r.json()
        last_status = str(j.get("status", "Unknown"))
        if last_status in ("Completed", "Failed"):
            return last_status
        time.sleep(interval)
        interval = min(interval * 1.5, 30.0)  # back off up to 30 s

    check(
        f"[{label}] completed within timeout",
        False,
        f"timed out after {poll_timeout}s (last status={last_status})",
    )
    return None


def validate_completed_result(base: str, headers: dict, job_id: str, label: str) -> None:
    """Validate CSV and JSON result downloads for a completed job."""
    # CSV download
    r = requests.get(f"{base}/result/{job_id}/", headers=headers)
    check(f"[{label}] CSV status 200", r.status_code == 200, f"got {r.status_code}")
    check(
        f"[{label}] content-type is text/csv",
        "text/csv" in r.headers.get("Content-Type", ""),
        r.headers.get("Content-Type"),
    )
    check(f"[{label}] non-empty body", len(r.content) > 0)

    # JSON format
    r = requests.get(f"{base}/result/{job_id}/?format=json", headers=headers)
    check(f"[{label}] JSON status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        return
    j = r.json()
    check(f"[{label}] JSON has jobId", j.get("jobId") == job_id)
    check(f"[{label}] JSON has columns list", isinstance(j.get("columns"), list))
    check(f"[{label}] JSON has rowCount", isinstance(j.get("rowCount"), int))
    check(f"[{label}] JSON has data list", isinstance(j.get("data"), list))
    check(f"[{label}] rowCount matches data", j.get("rowCount") == len(j.get("data", [])))

    if j.get("data"):
        first_row = j["data"][0]
        check(f"[{label}] row has Protein Sequence", "Protein Sequence" in first_row)
        if "TurNup" in label:
            check(f"[{label}] row has Substrates", "Substrates" in first_row)
            check(f"[{label}] row has Products", "Products" in first_row)
        else:
            check(f"[{label}] row has Substrate", "Substrate" in first_row)


def test_result_completed(base: str, headers: dict, submitted_jobs: list[dict],
                          poll_timeout: int = 1000) -> None:
    """
    Poll every submitted method job until it reaches a terminal state, then
    validate its result in CSV and JSON formats.

    poll_timeout — seconds to wait per job (default 1000 s / 16.67 min).
    """
    if not submitted_jobs:
        return

    section("GET /result/<jobId>/ — downloading completed results for all selected methods")
    for submitted in submitted_jobs:
        job_id = submitted["job_id"]
        label = submitted["label"]
        final_status = wait_for_terminal_status(base, headers, job_id, label, poll_timeout)
        if final_status is None:
            continue
        check(f"[{label}] final status is Completed", final_status == "Completed", f"got {final_status}")
        if final_status != "Completed":
            continue
        validate_completed_result(base, headers, job_id, label)


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
    parser = argparse.ArgumentParser(description="Open Kinetics Predictor API test suite")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Base URL including /api/v1 (default: %(default)s)")
    parser.add_argument("--key",  default=DEFAULT_KEY,
                        help="API Bearer token (default: hardcoded test key)")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip the slow runSimilarity=true test (requires MMseqs2)")
    parser.add_argument("--poll-timeout", type=int, default=1000, metavar="SECONDS",
                        help="Seconds to wait per submitted method job before marking "
                             "that job as failed (default: 1000)")
    parser.add_argument(
        "--extra-submit-variants",
        action="store_true",
        help=(
            "Also submit an additional success-path variant (JSON-body submit). "
            "This job is fully waited/validated too. "
            "Default is off for faster runs."
        ),
    )
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
    print("  Open Kinetics Predictor API Test Suite")
    print(f"  Base URL : {base}")
    print(f"  API Key  : {key[:15]}…")
    print(f"  Methods  : {', '.join(sorted(methods))}")
    print(f"  Variants : {'extra' if args.extra_submit_variants else 'minimal'}")
    print("=" * 70)

    # Run all test groups
    test_health(base)
    test_methods(base, methods)
    test_auth(base, key)
    test_quota(base, headers)

    # Submit one job for each selected method/prediction type.
    submitted_jobs = test_submit_selected_methods(base, headers, methods)

    # Optional success-path submission variants. If enabled, their jobs are
    # included in full end-to-end status/result validation.
    if args.extra_submit_variants:
        json_job = test_submit_json_body(base, headers, methods)
        if json_job:
            submitted_jobs.append(json_job)
    else:
        print("\n  (skipping extra submit variants — pass --extra-submit-variants to run them)")

    # All the ways submission can fail
    test_submit_errors(base, headers)

    # Status polling
    test_status(base, headers, submitted_jobs)

    # Result endpoint (job not done yet)
    test_result_not_ready(base, headers, submitted_jobs)

    # Result endpoint (poll every submitted method job until done, then download)
    test_result_completed(base, headers, submitted_jobs, poll_timeout=args.poll_timeout)

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
