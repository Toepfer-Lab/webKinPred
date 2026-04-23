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
    authors="Author A, Author B",
    publication_title="Paper title",
    citation_url="https://doi.org/...",
    repo_url="https://github.com/...",

    supports=["kcat"],                 # e.g. ["kcat"], ["Km"], ["kcat/Km"], or combinations
    input_format="single",             # "single", "multi", or "full reaction" ("full reaction" uses "multi" internally)
    output_cols={"kcat": "kcat (1/s)"},
    max_seq_len=1024,

    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={"kcat": {}},

    # Engine selection rule:
    # - Use subprocess=SubprocessEngineConfig(...) by default.
    # - Use pred_func=your_method_predictions only when custom orchestration is required.

    embeddings_used=[],
)
```

### What these fields mean

- `supports`: which targets your method predicts.
- `input_format`: CSV shape expected from users.
  Use `single`, `multi`, or `full reaction` in user-facing docs.
  In descriptors, `full reaction` is represented by `multi` (`Substrates` + `Products`).
- `col_to_kwarg`: maps CSV columns to kwargs passed into your method runtime.
- `target_kwargs`: per-target switches (for shared kcat/Km scripts).
- `subprocess` or `pred_func`: set exactly one.
  Use `subprocess` by default.
  Use `pred_func` only when the shared subprocess engine cannot support your runtime flow.

## 2. Implement Your Method's Predictor

Use this decision rule:

1. Use the shared subprocess engine by default.
2. Use a custom engine only when required by method-specific behaviour.

Source code of your method should be added to "models/YourMethod/" (this can be a Git submodule).

General batching best practice:
- Batching is fine, but keep batch sizes realistic to avoid RAM spikes (generally no more than 32-64 rows/sequences per batch).

## Path 1: Script + Shared Engine (default)

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
- If your script uses PyTorch, handle both GPU and CPU runtimes:
  use CUDA only when `torch.cuda.is_available()` is `True`, and keep a CPU fallback.

Path config example:

```python
subprocess=SubprocessEngineConfig(
    python_path_key="YourMethod",
    script_key="YourMethod",
    data_path_env={"YOUR_METHOD_DATA": "YourMethod"},
)
```

## Path 2: Script + Custom Engine (only when required)

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

If your method needs a new Python environment, you must update the full worker image `Dockerfile.envs`.

1. Add a requirements file:

```text
docker-requirements/your_method_requirements.txt
```

2. Add a parallel env stage in `Dockerfile.envs`.

The Dockerfile uses multi-stage builds so all envs are built in parallel by BuildKit. Add two things:

**a) A new `FROM base AS env-your_method` stage** (alongside the other `env-*` stages):

```dockerfile
# ── YourMethod ────────────────────────────────────────────────────────────────
FROM base AS env-your_method
COPY docker-requirements/your_method_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py310,target=/root/.cache/pip,sharing=locked \
    mamba create -n your_method_env python=3.10 -c conda-forge -y \
    && conda run -n your_method_env pip install -r docker-requirements/your_method_requirements.txt
```

If your method needs extra conda packages (e.g. RDKit, XGBoost), install them before `pip install` (see `env-dlkcat` and `env-turnup` stages for examples).

**b) A `COPY --from` line in the `final` stage** (alongside the other env copies):

```dockerfile
COPY --from=env-your_method /opt/conda/envs/your_method_env /opt/conda/envs/your_method_env
```

3. Add runtime keys in:
- `webKinPred/config_docker.py`
- `webKinPred/config_local.py` (for local development)
Both inherit common path shape from `webKinPred/config_base.py`.

```python
PYTHON_PATHS["YourMethod"] = "/opt/conda/envs/your_method_env/bin/python"
PREDICTION_SCRIPTS["YourMethod"] = "/app/models/YourMethod/predict.py"
DATA_PATHS["YourMethod"] = "/app/models/YourMethod/data"
```

If your method can reuse an existing env, skip steps 1-2 and only add the config keys.

## 4. PLM Embeddings (Optional)

If your method uses PLM embeddings, read [PLM_EMBEDDING_CACHE.md](PLM_EMBEDDING_CACHE.md) before you start.

### 4.1 Why this layer exists

- PLM inference is often the slowest part of prediction.
- The cache is file based under `media/sequence_info`.
- Cache keys use `seq_id`, not raw sequence text.
- `seq_id` is resolved through `tools/seqmap/main.py`, then reused across methods.
- The GPU host is remote. Your code must separate embedding generation from prediction inference.

### 4.2 Current profiles in `embedding_plan_service.py`

- `KinForm-H`, `KinForm-L` -> `kinform_full`
- `CataPro`, `UniKP` -> `prot_t5_mean`
- `TurNup` -> `turnup_esm1b`
- `CatPred` -> `catpred_embed`
- `EITLEM` -> `eitlem_esm1v`

### 4.3 Current method behaviours

- KinForm-H and KinForm-L
  - Engine calls GPU precompute before subprocess.
  - Sparse step logic runs `kinform_t5_full`, `kinform_esm2_layers`, `kinform_esmc_layers` only for missing `seq_id`s.
- CataPro
  - Uses the generic subprocess engine.
  - Generic engine runs GPU precompute.
  - Reuses `prot_t5_last/mean_vecs`.
- UniKP
  - Custom engine runs GPU precompute.
  - Reuses `prot_t5_last/mean_vecs`.
- TurNup
  - Custom engine runs GPU precompute.
  - Uses `esm1b_turnup`.
- CatPred
  - Uses the generic subprocess engine, so GPU precompute runs.
  - Uses target specific step keys `catpred_embed_kcat` or `catpred_embed_km`.
  - Caches checkpoint specific pooled tensors in `catpred_esm2/{parameter}/{model_key}/{seq_id}.pt`.
- EITLEM
  - Planner profile and GPU step runner support `eitlem_esm1v`.
  - Current configured prediction script in `config_base.py` is `eitlem_prediction_script_batch.py`.
  - `models/EITLEM/Code/eitlem_prediction_script.py` shows the preferred ephemeral full-matrix cleanup pattern.

### 4.4 Required execution pattern

1. Resolve valid sequences.
2. Resolve `seq_id` values.
3. Compute expected cache files for this method and target.
4. Build sparse step work with missing files only.
5. Start embedding tracker before remote submission.
6. Submit GPU job only when GPU health is online.
7. Poll GPU job status until done or failed.
8. Re-check cache completeness.
9. Continue to prediction subprocess.
10. Keep fail-open fallback unless you have a strict fail-fast requirement.

### 4.5 Decision rule for new methods

Use this rule when adding PLM support for a method:

1. If your method uses a PLM and artefact type we already support, you must reuse the existing embedding family ([4.6](#46-reuse-an-existing-embedding-family-required-when-applicable)).
2. If your method introduces a new PLM or a new artefact type, you must add a new embedding family ([4.7](#47-add-a-new-embedding-family-only-when-required)).

Existing PLM caches:

- `prot_t5_last/mean_vecs/{seq_id}.npy`: mean embedding of the last layer from `Rostlab/prot_t5_xl_uniref50` (shared by CataPro and UniKP, also used by KinForm).
- `prot_t5_last/weighted_vecs/{seq_id}.npy`: weighted average of last-layer `Rostlab/prot_t5_xl_uniref50` residue embeddings, with weights from Pseq2Sites binding-site probabilities in `media/pseq2sites/binding_sites_all.tsv` (KinForm).
- `prot_t5_layer_19/mean_vecs/{seq_id}.npy`: mean embedding of layer 19 from `Rostlab/prot_t5_xl_uniref50` (KinForm).
- `prot_t5_layer_19/weighted_vecs/{seq_id}.npy`: weighted average of layer 19 `Rostlab/prot_t5_xl_uniref50` residue embeddings, with Pseq2Sites binding-site probabilities (KinForm).
- `esm2_layer_26/mean_vecs/{seq_id}.npy`: mean embedding from `esm2_t33_650M_UR50D` layer 26 (KinForm).
- `esm2_layer_26/weighted_vecs/{seq_id}.npy`: weighted average of `esm2_t33_650M_UR50D` layer 26 residue embeddings, with Pseq2Sites binding-site probabilities (KinForm).
- `esm2_layer_29/mean_vecs/{seq_id}.npy`: mean embedding from `esm2_t33_650M_UR50D` layer 29 (KinForm).
- `esm2_layer_29/weighted_vecs/{seq_id}.npy`: weighted average of `esm2_t33_650M_UR50D` layer 29 residue embeddings, with Pseq2Sites binding-site probabilities (KinForm).
- `esmc_layer_24/mean_vecs/{seq_id}.npy`: mean embedding from `esmc_600m` layer 24 (KinForm).
- `esmc_layer_24/weighted_vecs/{seq_id}.npy`: weighted average of `esmc_600m` layer 24 residue embeddings, with Pseq2Sites binding-site probabilities (KinForm).
- `esmc_layer_32/mean_vecs/{seq_id}.npy`: mean embedding from `esmc_600m` layer 32 (KinForm).
- `esmc_layer_32/weighted_vecs/{seq_id}.npy`: weighted average of `esmc_600m` layer 32 residue embeddings, with Pseq2Sites binding-site probabilities (KinForm).
- `esm1b_turnup/{seq_id}.npy`: TurNup protein vector derived from `esm1b_t33_650M_UR50S` plus the TurNup fine-tuned checkpoint (`model_ESM_binary_A100_epoch_1_new_split.pkl`).
- `catpred_esm2/{kcat|km}/{model_key}/{seq_id}.pt`: CatPred checkpoint-specific pooled tensor, generated from `esm2_t33_650M_UR50D` residue features and attentive pooling.

### 4.6 Reuse an existing embedding family (required when applicable)

1. Reuse an existing cache artefact layout in `media/sequence_info`.
2. Map your method key in `_profile_for_method`.
3. Update `expected_paths_by_seq` only if your artefact list differs.
4. Ensure your prediction script reads cache first, then computes missing files only.
5. If you use a custom engine, call `run_gpu_precompute_if_available(...)` before `run_prediction_subprocess(...)`.
6. If you use `SubprocessEngineConfig`, precompute is already called by `run_generic_subprocess_prediction(...)`.
7. Add planner tests for mixed cache states.
8. Add orchestration tests for failed-job fallback behaviour.

### 4.7 Add a new embedding family (only when required)

1. Define a stable artefact path keyed by `seq_id`.
2. Add expected file mapping in `expected_paths_by_seq`.
3. Add a new profile mapping in `_profile_for_method`.
4. Add sparse step partitioning in `_step_plans_for_profile`.
5. Add step execution in `tools/gpu_embed_service/run_step.py`.
6. Add optional override command in `tools/gpu_embed_service/gpu_service.env`.
7. Pass required env paths to subprocess scripts through `data_path_env` or engine `env`.
8. Add tests for planner sparsity, orchestration, and API status.

### 4.8 Full-matrix policy

- Default rule
  - Do not persist full residue matrices.
  - Persist reduced artefacts when reduction is deterministic and substrate independent.
- CatPred example
  - CatPred needs ESM2 residue signals plus checkpoint attentive pooling.
  - The pooled tensor is deterministic for `(seq_id, checkpoint_key)`.
  - Persist `catpred_esm2/{parameter}/{model_key}/{seq_id}.pt`.
- EITLEM example
  - EITLEM consumes a full residue matrix and is not sequence-deterministic.
  - Preferred pattern is ephemeral files, run prediction, then delete touched files.
  - See `models/EITLEM/Code/eitlem_prediction_script.py` for this pattern.

### 4.9 Tracking rules

- Embedding progress is file based, not sequence based.
- Tracker must start before GPU submission.
- Inotify drives progress updates when files appear.
- Reconciliation polling handles missed events or non-Linux environments.
## 5. Add MMseqs Similarity Dataset (Optional)

If you want to include your method's training data in the sequence-similarity validation, read:
- [MMSEQS_SIMILARITY_DATASETS.md](MMSEQS_SIMILARITY_DATASETS.md)

This includes:
- reusing an existing dataset by extending its label (for example `DLKcat/UniKP/YourMethod`)
- adding a new FASTA + DB dataset

## 6. Test Your Integration End-to-End

Setup:

```bash
pip install -r requirements.txt
python manage.py migrate
```

Run:

```bash
python tools/test_method_integration.py --method YourMethod
```

What it tests:
- method registry discovery
- descriptor validity (runnable config checks)
- direct prediction execution through backend task helpers
- output CSV generation and output-shape checks
- all targets your method supports (`kcat`, `Km`, and/or `kcat/Km`)
- optional DLKcat sanity check first

If you use Path 1 (`subprocess=SubprocessEngineConfig(...)`), do this before testing:
- create/install your method environment
- set `PYTHON_PATHS["YourMethod"]` in `webKinPred/config_local.py` to that environment's Python executable
