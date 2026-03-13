# PLM Embedding Cache Guide

Use this guide if your method uses protein language model embeddings.

Goal: reuse the interface cache so repeated sequences do not recompute embeddings.

## 1. Cache Key Rule

Always cache by `seq_id`, not by raw sequence text.

Get IDs once per batch using `seqmap`, then use `{seq_id}.npy` files.

## 2. Shared PLM Cache Directories

Use these model directories (last-layer mean only):

- `prot_t5`: `media/sequence_info/prot_t5_last/mean_vecs/{seq_id}.npy`
- `esm2`: `media/sequence_info/esm2_last/mean_vecs/{seq_id}.npy`
- `esmc`: `media/sequence_info/esmc_last/mean_vecs/{seq_id}.npy`

## 3. Reuse Pattern (Load or Generate)

```python
import os
from pathlib import Path
import numpy as np

def get_cached_or_compute(seq_ids, sequences, emb_dir):
    emb_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for seq_id, seq in zip(seq_ids, sequences):
        fp = emb_dir / f"{seq_id}.npy"
        if fp.exists():
            vec = np.load(fp)
        else:
            vec = compute_last_layer_mean(seq)  # your model code
            np.save(fp, vec)
        out.append(vec)
    return out
```

## 4. End-to-End Example

```python
media = Path(os.environ["YOUR_METHOD_MEDIA_PATH"])
seq_ids = resolve_seq_ids_via_cli(sequences)  # one batch call

prot_t5_dir = media / "sequence_info" / "prot_t5_last" / "mean_vecs"
t5_vectors = get_cached_or_compute(seq_ids, sequences, prot_t5_dir)
```

## 5. Adding a New PLM Cache

If your PLM is not listed above:

1. Add runtime env support (Dockerfile + config paths).
2. Add an entry in `api/embeddings/registry.py`.
3. Store last-layer mean vectors under:
   `media/sequence_info/<model_name>_last/mean_vecs/{seq_id}.npy`
4. Add the embedding key to your method descriptor `embeddings_used`.
