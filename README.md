# WebKinPred

WebKinPred is a production web interface for predicting enzyme kinetic parameters (kcat and KM) from protein sequence and substrate SMILES. It consolidates several state‑of‑the‑art machine learning / deep learning models behind a unified, asynchronous job API so you can submit sequences and retrieve structured predictions.

**Live service:** [https://kineticxpredictor.humanmetabolism.org/](https://kineticxpredictor.humanmetabolism.org/)

## Prediction Engines

| Engine | Input needed | Output | Citation |
|--------|--------------|--------|----------|
| KinForm-H | Protein sequence + substrate SMILES | kcat or KM | [Alwer & Fleming, Arxiv](https://arxiv.org/abs/2507.14639) ([GitHub](https://github.com/Digital-Metabolic-Twin-Centre/KinForm)) |
| KinForm-L | Protein sequence + substrate SMILES | kcat or KM | [Alwer & Fleming, Arxiv](https://arxiv.org/abs/2507.14639) ([GitHub](https://github.com/Digital-Metabolic-Twin-Centre/KinForm)) |
| UniKP | Protein sequence + substrate SMILES | kcat or KM | [Yu et al., Nat Commun 2023](https://www.nature.com/articles/s41467-023-44113-1) ([GitHub](https://github.com/Luo-SynBioLab/UniKP)) |
| DLKcat | Protein sequence + substrate SMILES | kcat | [Li et al., Nat Catal 2022](https://www.nature.com/articles/s41929-022-00798-z) ([GitHub](https://github.com/SysBioChalmers/DLKcat)) |
| TurNup | Protein sequence + substrates list + products list | kcat | [Kroll et al., Nat Commun 2023](https://www.nature.com/articles/s41467-023-39840-4) ([GitHub](https://github.com/AlexanderKroll/Kcat_prediction)) |
| EITLEM | Protein sequence + substrate SMILES | kcat or KM | [Shen et al., Biotechnol Adv 2024](https://www.sciencedirect.com/science/article/pii/S2667109324002665) ([GitHub](https://github.com/XvesS/EITLEM-Kinetics)) |

Each model is loaded with its published weights/code (see `api/prediction_engines/`) and invoked through a standard internal interface so new engines can be added with minimal wiring.

## Features

* Batch submission of sequences and substrates.
* Long‑running inference handled asynchronously (Celery + Redis) with progress tracking.
* Sequence similarity distribution of input data vs mehtods' training data (Using mmseq2).
* Caching sequence embeddings.

## Tech Stack

### Frontend

* React 18 + Vite (fast dev + ESM build)
* Bootstrap / React‑Bootstrap for layout & components
* Axios for API calls; Chart.js for result visualisation

### Backend

* Django 5.1 (REST-style endpoints under `api/`)
* Celery workers for queued prediction tasks (`api/tasks.py`)
* Redis as Celery broker
* SQLite
* PyTorch, scikit-learn, RDKit, pandas for model computation & cheminformatics

## High-Level Flow

1. User submits a job (sequence + substrate(s) [+ products/mutant context if required]) via the frontend.
2. Backend validates input (`api/services/validation_service.py`).
3. A Celery task is enqueued; Redis broker stores the task message.
4. Worker loads the selected model wrapper (e.g. `prediction_engines/kinform.py`) and executes inference.
5. Results & intermediate status are persisted; cached for repeated identical queries.
6. Frontend polls job status endpoint to update progress and results.

## API Access

KineticXPredictor provides a REST API for programmatic access. Submit prediction
jobs, poll their status, and download results — no web browser required.

**Base URL:** `https://kineticxpredictor.humanmetabolism.org/api/v1`

> **Full interactive documentation** is also available on the live site at
> [`/api-docs`](https://kineticxpredictor.humanmetabolism.org/api-docs).

---

### Getting an API Key

API keys are issued on request. Contact the administrators with your use case to
receive a Bearer token. Keys look like `ak_7f8a9b2c3d4e5f6a…` and are sent in
the `Authorization` header of every authenticated request.

If you are self-hosting, generate a key via the management command:

```bash
python manage.py create_api_key --ip 1.2.3.4 --label "My Lab Script"
```

---

### Endpoint Overview

| Method | Endpoint | Auth | Description |
| ------ | -------- | ---- | ----------- |
| `GET` | `/health/` | No | Service health check |
| `GET` | `/methods/` | No | List available methods and required columns |
| `GET` | `/quota/` | Yes | Check remaining daily quota |
| `POST` | `/validate/` | Yes | Validate input data without submitting a job |
| `POST` | `/submit/` | Yes | Submit a prediction job |
| `GET` | `/status/<jobId>/` | Yes | Poll job status and progress |
| `GET` | `/result/<jobId>/` | Yes | Download results (CSV or `?format=json`) |

---

### Quick Start — Python

```python
import requests
import time

API_KEY = "ak_your_key_here"
BASE    = "https://kineticxpredictor.humanmetabolism.org/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# 1. Submit a job
with open("input.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/submit/",
        headers=HEADERS,
        files={"file": f},
        data={
            "predictionType":      "kcat",
            "kcatMethod":          "DLKcat",
            "handleLongSequences": "truncate",
            "useExperimental":     "true",
        },
    )
resp.raise_for_status()
job = resp.json()
print(f"Job ID: {job['jobId']}  |  Quota remaining: {job['quota']['remaining']:,}")

# 2. Poll until complete
while True:
    status = requests.get(f"{BASE}/status/{job['jobId']}/", headers=HEADERS).json()
    print(f"  {status['status']} ({status['elapsedSeconds']}s)")
    if status["status"] == "Completed":
        break
    if status["status"] == "Failed":
        raise RuntimeError(f"Job failed: {status.get('error')}")
    time.sleep(5)

# 3. Download results
result = requests.get(f"{BASE}/result/{job['jobId']}/", headers=HEADERS)
with open("output.csv", "wb") as f:
    f.write(result.content)
print("Saved to output.csv")
```

---

### Quick Start — curl

```bash
API_KEY="ak_your_key_here"
BASE="https://kineticxpredictor.humanmetabolism.org/api/v1"

# 1. Submit
JOB=$(curl -s -X POST "$BASE/submit/" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@input.csv" \
  -F "predictionType=kcat" \
  -F "kcatMethod=DLKcat" \
  -F "handleLongSequences=truncate")

JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['jobId'])")
echo "Submitted: $JOB_ID"

# 2. Poll
while true; do
  STATE=$(curl -s "$BASE/status/$JOB_ID/" \
    -H "Authorization: Bearer $API_KEY" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  $STATE"
  [ "$STATE" = "Completed" ] && break
  [ "$STATE" = "Failed" ]    && { echo "Job failed"; exit 1; }
  sleep 5
done

# 3. Download
curl -s "$BASE/result/$JOB_ID/" \
  -H "Authorization: Bearer $API_KEY" \
  -o output.csv
```

---

### JSON Body Submission (no CSV file needed)

For small datasets (≤ 10,000 rows) you can send data directly as JSON:

```python
requests.post(
    f"{BASE}/submit/",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "predictionType":      "kcat",
        "kcatMethod":          "DLKcat",
        "handleLongSequences": "truncate",
        "useExperimental":     True,
        "data": [
            {"Protein Sequence": "MKTLLIFAG...", "Substrate": "CC(=O)O"},
            {"Protein Sequence": "MGSSHHHHH...", "Substrate": "C1CCCCC1"},
        ],
    },
)
```

---

### Validating Input Before Submission

Use `/validate/` to check substrate SMILES/InChI strings, protein sequences,
and per-model length limits **without consuming any quota or running predictions**.
This is equivalent to the validation step available in the web interface.

```python
import requests

API_KEY = "ak_your_key_here"
BASE    = "https://kineticxpredictor.humanmetabolism.org/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Basic validation (fast)
with open("input.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/validate/",
        headers=HEADERS,
        files={"file": f},
        data={"runSimilarity": "false"},
    )

result = resp.json()
print(f"Rows: {result['rowCount']}")
print(f"Invalid substrates: {len(result['invalidSubstrates'])}")
print(f"Invalid proteins:   {len(result['invalidProteins'])}")
print(f"Length violations:  {result['lengthViolations']}")
```

Set `runSimilarity=true` to also run MMseqs2 sequence similarity analysis
against each method's training database. The request **blocks synchronously**
until the analysis is complete (can take several minutes for large inputs):

```python
with open("input.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/validate/",
        headers=HEADERS,
        files={"file": f},
        data={"runSimilarity": "true"},
        timeout=600,
    )

similarity = resp.json()["similarity"]
for method, data in similarity.items():
    print(f"{method}: avg max identity = {data['average_max_similarity']:.1f}%")
```

The `similarity` field in the response is a dict keyed by method name. Each entry
contains `histogram_max`, `histogram_mean` (10-bin arrays, 0–100% identity),
`average_max_similarity`, `average_mean_similarity`, `count_max`, and `count_mean`.

---

### CSV Format

| Method | Predicts | Required columns | Max sequence length |
| ------ | -------- | ---------------- | ------------------- |
| DLKcat | kcat | `Protein Sequence`, `Substrate` | No limit |
| TurNup | kcat | `Protein Sequence`, `Substrates`, `Products` | 1,022 residues |
| EITLEM | kcat or KM | `Protein Sequence`, `Substrate` | 1,022 residues |
| UniKP | kcat or KM | `Protein Sequence`, `Substrate` | 1,000 residues |
| KinForm-H | kcat or KM | `Protein Sequence`, `Substrate` | 1,000 residues |
| KinForm-L | kcat only | `Protein Sequence`, `Substrate` | 1,000 residues |

Substrates must be SMILES or InChI strings. For TurNup, separate multiple
substrates/products with semicolons: `CC(=O)O;C1CCCCC1`.

---

### Rate Limits

* **20,000 predictions/day** per API key (default; custom limits available).
* Counter resets at midnight UTC.
* Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
* HTTP `429` is returned when the quota is exceeded.

---

### Error Format

All errors return JSON with a single `error` key:

```json
{ "error": "A human-readable description of what went wrong." }
```

| Status | Meaning |
| ------ | ------- |
| 400 | Invalid parameters, missing CSV columns, or bad data |
| 401 | Missing or invalid API key |
| 403 | Account suspended |
| 404 | Job not found |
| 409 | Results not ready yet |
| 429 | Quota exceeded |
| 500 | Internal server error |

---

## Adding a New Model

... todo ...
Create a new module in `api/prediction_engines/` following existing wrappers.

## Attribution

Please cite the original publications when using predictions from a specific engine. When publishing aggregated results produced through WebKinPred, cite all underlying sources plus this platform.

## Contact

For questions or collaboration: open an issue or reach out to the authors of the respective model.

## Funding

... todo ...
