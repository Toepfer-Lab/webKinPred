# GPU Embedding Service

Minimal FastAPI service for remote GPU embedding offload.

## Endpoints

- `GET /health`
- `POST /embed/jobs`
- `GET /embed/jobs/{job_id}`

## Request Shape

`POST /embed/jobs`

```json
{
  "method_key": "KinForm-H",
  "target": "kcat",
  "profile": "kinform_full",
  "step_work": {
    "kinform_esm2_layers": ["seq_1", "seq_2"],
    "kinform_prott5_layers": ["seq_2"]
  },
  "seq_id_to_seq": {
    "seq_1": "MPE...",
    "seq_2": "MQA..."
  }
}
```

## Optional Environment Variables

- `GPU_EMBED_SERVICE_TOKEN`: optional bearer token expected by clients.
- `GPU_NAME`: value surfaced by `/health`.
- `GPU_EMBED_STEP_CMD_<STEP_KEY>`: optional shell command per step.
  - Example key: `GPU_EMBED_STEP_CMD_KINFORM_ESM2_LAYERS`
  - Command template supports:
    - `{step_key}`
    - `{seq_ids}` (comma-separated)
    - `{seq_count}`

If no step command is configured for a step, it is treated as a no-op.

### Real Step Commands (GPU host)

Use the bundled step runner so jobs write real cache artifacts:

```bash
export GPU_EMBED_STEP_CMD_KINFORM_PSEQ2SITES="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_pseq2sites --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_KINFORM_ESM2_LAYERS="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_esm2_layers --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_KINFORM_ESMC_LAYERS="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_esmc_layers --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_KINFORM_PROTT5_LAYERS="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step kinform_prott5_layers --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_PROT_T5_MEAN="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step prot_t5_mean --seq-ids '{seq_ids}'"
export GPU_EMBED_STEP_CMD_TURNUP_ESM1B="/usr/bin/python3 /path/to/webKinPred/tools/gpu_embed_service/run_step.py --step turnup_esm1b --seq-ids '{seq_ids}'"
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```
