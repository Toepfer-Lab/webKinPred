# Contributing a New Prediction Method

Start with the `MethodDescriptor`.  
It is the contract between your method and the rest of the platform.

## 1. Define the Descriptor First

Create `api/methods/your_method.py` first, then implement the method behind it.

```python
from api.methods.base import MethodDescriptor, SubprocessEngineConfig
from api.prediction_engines.your_method import your_method_predictions  # only for Path 2

descriptor = MethodDescriptor(
    key="YourMethod",                  # unique ID used in API/UI
    display_name="Your Method",        # human-readable name
    description="One-line summary.",
    authors="Author A, Author B",
    publication_title="Paper title",
    citation_url="https://doi.org/...",
    repo_url="https://github.com/...",

    supports=["kcat"],                 # ["kcat"], ["Km"], or ["kcat", "Km"]
    input_format="single",             # "single" or "multi"
    output_cols={"kcat": "kcat (1/s)"},
    max_seq_len=1024,

    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={"kcat": {}},

    # Choose one path below:
    # subprocess=SubprocessEngineConfig(...),   # Path 1
    # pred_func=your_method_predictions,        # Path 2

    embeddings_used=[],
)
```

### What these fields mean

- `supports`: which targets your method predicts.
- `input_format`: CSV shape expected from users.
- `col_to_kwarg`: maps CSV columns to kwargs passed into your method runtime.
- `target_kwargs`: per-target switches (for shared kcat/Km scripts).
- `subprocess` or `pred_func`: choose one implementation path.

## 2. Choose an Implementation Path

You can implement your method in either of these two ways.

## Path 1: Script + Shared Engine

Use this if your model can run as one subprocess call.

You write:
- One prediction script
- `subprocess=SubprocessEngineConfig(...)` in descriptor

The shared engine handles:
- Row validation (sequence + substrate/product chemistry)
- Temporary input/output files
- Subprocess execution
- Progress parsing (`Progress: x/y`)
- Output parsing and row mapping

Your script must support:

```bash
python your_script.py --input <input.json> --output <output.json>
```

Input JSON:

```json
{
  "method": "YourMethod",
  "target": "kcat",
  "public_id": "abc1234",
  "rows": [
    {"sequence": "MKT...", "substrates": "CC(=O)O"}
  ],
  "params": {
    "kinetics_type": "KCAT"
  }
}
```

Output JSON:

```json
{
  "predictions": [12.3],
  "invalid_indices": []
}
```

Rules:
- `predictions` length must equal `rows` length.
- `invalid_indices` is optional and is relative to `rows`.
- Use `null` for missing predictions.

Path config example:

```python
subprocess=SubprocessEngineConfig(
    python_path_key="YourMethod",
    script_key="YourMethod",
    data_path_env={"YOUR_METHOD_DATA": "YourMethod"},
)
```

## Path 2: Script + Custom Engine

Use this if you need custom behavior not covered by the shared engine.

Examples:
- Special validation rules
- Non-standard file contracts
- Multi-stage orchestration
- Extra Python-side preprocessing/caching

You write:
- `api/prediction_engines/your_method.py`
- `pred_func=your_method_predictions` in descriptor

Expected engine signature:

```python
def your_method_predictions(
    sequences: list[str],
    public_id: str,
    **kwargs,
) -> tuple[list, list[int]]:
    ...
```

Return:
- `predictions`: one value per input row
- `invalid_indices`: failed row indices relative to input list

## 3. Register Runtime Paths

If your method needs a new Python environment, you must update the full worker image `Dockerfile` (not `Dockerfile.web`).

1. Add a requirements file:

```text
docker-requirements/your_method_requirements.txt
```

2. Add an env layer in `Dockerfile` (same pattern as current methods):

```dockerfile
COPY docker-requirements/your_method_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n your_method_env python=3.10 -c conda-forge -y \
    && conda run -n your_method_env pip install -r docker-requirements/your_method_requirements.txt \
    && conda clean -afy
```

If your method needs extra conda packages (example: RDKit, XGBoost), install them in this same layer before `pip install` (see `DLKcat` and `TurNup` blocks in `Dockerfile`).

3. Add runtime keys in:
- `webKinPred/config_docker.py`
- `webKinPred/config_local.py` (for local development)

```python
PYTHON_PATHS["YourMethod"] = "/opt/conda/envs/your_method_env/bin/python"
PREDICTION_SCRIPTS["YourMethod"] = "/app/api/YourMethod/predict.py"
DATA_PATHS["YourMethod"] = "/app/api/YourMethod/data"
```

If your method can reuse an existing env, skip steps 1-2 and only add the config keys.

## 4. Embeddings: Sequence ID Mapping and Cache Reuse

If your method uses protein embeddings, follow this pattern.

### 4.1 Resolve sequence IDs first (once per batch)

Do not name embedding files by raw sequence.  
Resolve sequence IDs with `tools/seqmap/main.py` and use those IDs as filenames.

Preferred pattern:
- Call `batch-get-or-create` once for all input sequences in a job.
- Keep returned IDs in input order (duplicates are allowed and expected).
- Use each `seq_id` as the cache key for embedding files.

`KinForm`, `UniKP`, and `EITLEM` already use batch ID resolution. `TurNup` still does extra per-call resolution in parts of its legacy script, but new methods should follow the single-batch pattern above.

### 4.2 Use load/generate/save with those IDs

For each sequence:
1. Build the expected cache path from `seq_id`.
2. If file exists, load it.
3. If missing, generate embedding.
4. Save to the same shared cache path using `seq_id`.

Minimal pattern:

```python
seq_ids = resolve_seq_ids_via_cli(sequences)  # one call, ordered
for seq, seq_id in zip(sequences, seq_ids):
    vec_path = f"{EMB_DIR}/{seq_id}.npy"
    if os.path.exists(vec_path):
        vec = np.load(vec_path)
    else:
        vec = compute_embedding(seq)
        np.save(vec_path, vec)
```

### 4.3 Reuse existing embedding models and directories

Check `api/embeddings/registry.py` first, then reuse existing paths.

- `prot_t5` (shared): Python path key `t5`; cached under `media/sequence_info/prot_t5_*` (for example `prot_t5_last/mean_vecs/{seq_id}.npy`).
- `esm2` (shared): Python path key `esm2`; cached under `media/sequence_info/esm2_layer_26` and `esm2_layer_29`.
- `esmc` (shared): Python path key `esmc`; cached under `media/sequence_info/esmc_layer_24` and `esmc_layer_32`.
- `pseq2sites` (shared support model): Python path key `pseq2sites`; binding-site cache at `media/pseq2sites/binding_sites_all.tsv` keyed by `PDB == seq_id`.

Method-specific examples in current codebase:
- `UniKP` reuses `media/sequence_info/prot_t5_last/mean_vecs/{seq_id}.npy`.
- `EITLEM` stores ESM-1v embeddings in `media/sequence_info/esm1v/{seq_id}.npy`.
- `TurNup` stores ESM-1b embeddings in `media/sequence_info/esm1b_turnup/{seq_id}.npy`.

If your method needs ProtT5/ESM2/ESMC/Pseq2Sites, reuse the existing model env and cache directories above. Do not create a parallel duplicate cache for the same embedding.

### 4.4 If you need a new embedding model

1. Add a new env in `Dockerfile` (or reuse an existing one if possible).
2. Add a Python path key in `config_docker.py` and `config_local.py`.
3. Add an entry in `api/embeddings/registry.py`.
4. Choose one stable cache directory under `media/sequence_info/<your_embedding_name>/...` and save by `seq_id`.
5. List the embedding key in your method descriptor `embeddings_used`.

If the embedding is not yet installed platform-wide, set `implemented: False` in `api/embeddings/registry.py` until its env is added.

## 5. Test Your Integration End-to-End

Use:

```bash
python tools/test_method_integration.py --method YourMethod
```

What it tests:
- method registry discovery
- descriptor validity (runnable config checks)
- direct prediction execution through backend task helpers
- output CSV generation and output-shape checks
- all targets your method supports (`kcat`, `Km`, and `both` when applicable)
- optional DLKcat sanity check first

Useful flags:

```bash
python tools/test_method_integration.py --method YourMethod --skip-dlkcat-sanity
python tools/test_method_integration.py --method YourMethod --allow-empty-predictions
python tools/test_method_integration.py --method YourMethod --keep-artifacts
```

## 6. Local Run Setup for Testing

Use two Python runtimes:
- Runner runtime: the Python you use for `manage.py` and `tools/test_method_integration.py`.
- Method runtime: the interpreter referenced by `PYTHON_PATHS["YourMethod"]` in `webKinPred/config_local.py` (used by Path 1 subprocess methods).

Runner runtime setup:

```bash
pip install -r requirements.txt
python manage.py migrate
```

Then run:

```bash
python tools/test_method_integration.py --method YourMethod
```

If you use Path 1 (`subprocess=SubprocessEngineConfig(...)`), make sure `PYTHON_PATHS["YourMethod"]` points to a real env with your method dependencies installed.
