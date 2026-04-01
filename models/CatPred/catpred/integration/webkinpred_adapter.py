from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from catpred.inference import PredictionRequest, run_prediction_pipeline

_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
_TARGET_TO_PARAMETER = {
    "kcat": "kcat",
    "Km": "km",
    "km": "km",
    "ki": "ki",
    "Ki": "ki",
}
_PREDICTION_COLUMN = {
    "kcat": "Prediction_(s^(-1))",
    "km": "Prediction_(mM)",
    "ki": "Prediction_(mM)",
}
_INFERENCE_PROGRESS_CHUNK_SIZE = 50
_AA_TO_INDEX = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}


def _repo_root() -> Path:
    return _REPO_ROOT


def _contains_model_checkpoints(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.rglob("model.pt"))


def _discover_checkpoint_root(repo_root: Path) -> Path:
    production_root = (repo_root / ".e2e-assets" / "pretrained" / "production").resolve()
    checkpoints_root = (repo_root / "checkpoints").resolve()
    if _contains_model_checkpoints(production_root):
        return production_root
    return checkpoints_root


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default.resolve()


def _resolve_parameter(payload: dict[str, Any]) -> str:
    target = payload.get("target")
    if isinstance(target, str) and target in _TARGET_TO_PARAMETER:
        return _TARGET_TO_PARAMETER[target]

    params = payload.get("params") or {}
    kinetics_type = str(params.get("kinetics_type", "")).strip().upper()
    if kinetics_type == "KCAT":
        return "kcat"
    if kinetics_type == "KM":
        return "km"
    if kinetics_type == "KI":
        return "ki"
    raise RuntimeError(f"Unsupported target in payload: {target!r}")


def _stable_seq_id(sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]
    return f"seq_{digest}"


def _resolve_seq_ids(sequences: list[str], tools_path: Path, media_path: Path) -> list[str]:
    seqmap_cli = tools_path / "seqmap" / "main.py"
    seqmap_db = media_path / "sequence_info" / "seqmap.sqlite3"
    if not seqmap_cli.exists() or not seqmap_db.exists():
        return [_stable_seq_id(sequence) for sequence in sequences]

    seqmap_python = os.environ.get("CATPRED_SEQMAP_PYTHON", sys.executable)
    payload = "\n".join(sequences) + "\n"
    cmd = [
        seqmap_python,
        str(seqmap_cli),
        "--db",
        str(seqmap_db),
        "batch-get-or-create",
        "--stdin",
    ]
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    seq_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(seq_ids) != len(sequences):
        raise RuntimeError(f"seqmap returned {len(seq_ids)} ids for {len(sequences)} sequences")
    return seq_ids


def _checkpoint_cache_key(checkpoint_path: Path) -> str:
    checkpoint_path = checkpoint_path.resolve()
    parent = checkpoint_path.parent.name
    grandparent = checkpoint_path.parent.parent.name
    return f"{grandparent}__{parent}" if grandparent else parent


def _discover_checkpoint_models(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    checkpoint_files = sorted(checkpoint_dir.rglob("model.pt"))
    if not checkpoint_files:
        raise RuntimeError(f"No CatPred model checkpoints found under: {checkpoint_dir}")

    entries: list[tuple[str, Path]] = []
    seen_keys: set[str] = set()
    for checkpoint_path in checkpoint_files:
        key = _checkpoint_cache_key(checkpoint_path)
        if key in seen_keys:
            raise RuntimeError(f"Duplicate checkpoint cache key '{key}' under {checkpoint_dir}")
        seen_keys.add(key)
        entries.append((key, checkpoint_path.resolve()))
    return entries


def _cache_path_for_model_seq(
    *,
    cache_root: Path,
    parameter: str,
    model_key: str,
    seq_id: str,
) -> Path:
    return (cache_root / parameter / model_key / f"{seq_id}.pt").resolve()


def _sequence_to_tensor(sequence: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor([_AA_TO_INDEX[aa] for aa in sequence], device=device, dtype=torch.long).unsqueeze(0)


def _compute_seq_pooled_output(
    model: Any,
    sequence: str,
    seq_id: str,
    esm_feature: torch.Tensor,
) -> torch.Tensor:
    seq_arr = _sequence_to_tensor(sequence, model.device)
    esm_feature_arr = esm_feature.to(model.device).unsqueeze(0)

    if seq_arr.shape[1] != esm_feature_arr.shape[1]:
        common_len = min(seq_arr.shape[1], esm_feature_arr.shape[1])
        seq_arr = seq_arr[:, :common_len]
        esm_feature_arr = esm_feature_arr[:, :common_len]

    seq_outs = model.seq_embedder(seq_arr)
    q = model.rotary_embedder.rotate_queries_or_keys(seq_outs, seq_dim=1)
    k = model.rotary_embedder.rotate_queries_or_keys(seq_outs, seq_dim=1)
    seq_outs, _ = model.multihead_attn(q, k, seq_outs)

    if model.args.add_esm_feats:
        seq_outs = torch.cat([esm_feature_arr, seq_outs], dim=-1)

    if not model.args.skip_attentive_pooling:
        seq_pooled_outs, _ = model.attentive_pooler(seq_outs)
    else:
        seq_pooled_outs = seq_outs.mean(dim=1)

    if model.args.add_pretrained_egnn_feats:
        if seq_id in model.pretrained_egnn_feats_dict:
            pretrained_egnn = model.pretrained_egnn_feats_dict[seq_id].to(model.device).unsqueeze(0)
        else:
            pretrained_egnn = model.pretrained_egnn_feats_avg.to(model.device).unsqueeze(0)
        seq_pooled_outs = torch.cat([seq_pooled_outs, pretrained_egnn], dim=-1)

    return seq_pooled_outs.squeeze(0).detach().cpu()


def _prepare_seq_pooled_cache(
    *,
    rows: list[dict[str, Any]],
    seq_ids: list[str],
    parameter: str,
    media_path: Path,
    checkpoint_dir: Path,
) -> dict[str, dict[str, Path]]:
    if parameter not in {"kcat", "km"}:
        return {seq_id: {} for seq_id in seq_ids}

    cache_root = media_path / "sequence_info" / "catpred_esm2"
    checkpoint_models = _discover_checkpoint_models(checkpoint_dir)
    sequence_by_id = {
        seq_id: str(row.get("sequence", "")).strip()
        for row, seq_id in zip(rows, seq_ids)
    }

    cache_paths: dict[str, dict[str, Path]] = {}
    missing_by_model: dict[str, list[str]] = {}
    any_missing = False
    for seq_id in seq_ids:
        seq_cache_paths: dict[str, Path] = {}
        for model_key, _ in checkpoint_models:
            path = _cache_path_for_model_seq(
                cache_root=cache_root,
                parameter=parameter,
                model_key=model_key,
                seq_id=seq_id,
            )
            seq_cache_paths[model_key] = path
            if not path.exists():
                any_missing = True
                missing_by_model.setdefault(model_key, []).append(seq_id)
        cache_paths[seq_id] = seq_cache_paths

    if not any_missing:
        return cache_paths

    os.environ.setdefault("PROTEIN_EMBED_USE_CPU", "1")
    from catpred.data.esm_utils import get_single_esm_repr
    from catpred.utils import load_checkpoint

    missing_seq_ids = sorted({seq_id for values in missing_by_model.values() for seq_id in values})
    esm_by_seq_id: dict[str, torch.Tensor] = {}
    for seq_id in missing_seq_ids:
        esm_by_seq_id[seq_id] = get_single_esm_repr(sequence_by_id[seq_id]).cpu()

    for model_key, checkpoint_path in checkpoint_models:
        pending_seq_ids = sorted(set(missing_by_model.get(model_key, [])))
        if not pending_seq_ids:
            continue

        model = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            for seq_id in pending_seq_ids:
                output_path = cache_paths[seq_id][model_key]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pooled = _compute_seq_pooled_output(
                    model=model,
                    sequence=sequence_by_id[seq_id],
                    seq_id=seq_id,
                    esm_feature=esm_by_seq_id[seq_id],
                )
                torch.save(pooled, output_path)

    return cache_paths


def _build_input_dataframe(rows: list[dict[str, Any]], seq_ids: list[str]) -> pd.DataFrame:
    formatted_rows = []
    for row, seq_id in zip(rows, seq_ids):
        substrate = row.get("substrates", row.get("substrate", row.get("Substrate", "")))
        if isinstance(substrate, list):
            if len(substrate) != 1:
                raise RuntimeError("CatPred expects exactly one substrate per row.")
            substrate = substrate[0]
        substrate = str(substrate).strip()
        sequence = str(row.get("sequence", "")).strip()
        formatted_rows.append(
            {
                "SMILES": substrate,
                "sequence": sequence,
                "pdbpath": seq_id,
            }
        )
    return pd.DataFrame(formatted_rows)


def _write_protein_records(
    rows: list[dict[str, Any]],
    seq_ids: list[str],
    parameter: str,
    media_path: Path,
    checkpoint_dir: Path,
    out_path: Path,
) -> None:
    records: dict[str, dict[str, Any]] = {}
    pooled_cache_paths = _prepare_seq_pooled_cache(
        rows=rows,
        seq_ids=seq_ids,
        parameter=parameter,
        media_path=media_path,
        checkpoint_dir=checkpoint_dir,
    )

    for row, seq_id in zip(rows, seq_ids):
        sequence = str(row.get("sequence", "")).strip()
        record: dict[str, Any] = {"name": seq_id, "seq": sequence}
        if parameter in {"kcat", "km"}:
            record["seq_pooled_outs_paths"] = {
                key: str(path)
                for key, path in pooled_cache_paths[seq_id].items()
            }
        records[seq_id] = record

    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)


def _predict_rows_chunk(
    *,
    rows: list[dict[str, Any]],
    seq_ids: list[str],
    parameter: str,
    repo_root: Path,
    media_path: Path,
    checkpoint_root: Path,
) -> list[float]:
    with tempfile.TemporaryDirectory(prefix="catpred_webkinpred_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str).resolve()
        input_csv = tmp_dir / "input.csv"
        protein_records = tmp_dir / "protein_records.json.gz"
        results_dir = tmp_dir / "results"

        _build_input_dataframe(rows, seq_ids).to_csv(input_csv, index=False)
        _write_protein_records(
            rows=rows,
            seq_ids=seq_ids,
            parameter=parameter,
            media_path=media_path,
            checkpoint_dir=(checkpoint_root / parameter).resolve(),
            out_path=protein_records,
        )

        request = PredictionRequest(
            parameter=parameter,
            input_file=str(input_csv),
            checkpoint_dir=str((checkpoint_root / parameter).resolve()),
            use_gpu=False,
            repo_root=str(repo_root),
            python_executable=sys.executable,
            protein_records_file=str(protein_records),
        )
        output_file = run_prediction_pipeline(request=request, results_dir=str(results_dir))
        output_df = pd.read_csv(output_file)
        value_col = _PREDICTION_COLUMN[parameter]
        if value_col not in output_df.columns:
            raise RuntimeError(f"CatPred output is missing expected column: {value_col}")

        raw_preds = output_df[value_col].tolist()
        if len(raw_preds) != len(rows):
            raise RuntimeError(
                f"CatPred produced {len(raw_preds)} predictions for {len(rows)} rows."
            )
        return [float(pred) for pred in raw_preds]


def run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise RuntimeError("'rows' must be a list in input payload.")

    valid_rows: list[dict[str, Any]] = []
    invalid_indices: list[int] = []
    for idx, row in enumerate(rows):
        sequence = str(row.get("sequence", "")).strip()
        substrate = row.get("substrates", row.get("substrate", row.get("Substrate", "")))
        if isinstance(substrate, list):
            substrate = substrate[0] if len(substrate) == 1 else ""
        substrate = str(substrate).strip()
        if not sequence or not substrate or not set(sequence).issubset(_VALID_AAS):
            invalid_indices.append(idx)
            continue
        valid_rows.append(row)

    predictions: list[float | None] = [None] * len(rows)
    if not valid_rows:
        for idx in range(len(rows)):
            print(f"Progress: {idx + 1}/{len(rows)}", flush=True)
        return {"predictions": predictions, "invalid_indices": invalid_indices}

    repo_root = _env_path("CATPRED_REPO_ROOT", _repo_root())
    media_path = _env_path("CATPRED_MEDIA_PATH", repo_root / "media")
    tools_path = _env_path("CATPRED_TOOLS_PATH", repo_root / "tools")
    checkpoint_root = _env_path("CATPRED_CHECKPOINT_ROOT", _discover_checkpoint_root(repo_root))
    parameter = _resolve_parameter(payload)

    seq_ids = _resolve_seq_ids(
        [str(row.get("sequence", "")).strip() for row in valid_rows],
        tools_path=tools_path,
        media_path=media_path,
    )

    valid_predictions: list[float] = []
    total = len(rows)
    base_done = len(invalid_indices)
    processed_valid = 0
    for start in range(0, len(valid_rows), _INFERENCE_PROGRESS_CHUNK_SIZE):
        chunk_rows = valid_rows[start : start + _INFERENCE_PROGRESS_CHUNK_SIZE]
        chunk_seq_ids = seq_ids[start : start + _INFERENCE_PROGRESS_CHUNK_SIZE]
        chunk_preds = _predict_rows_chunk(
            rows=chunk_rows,
            seq_ids=chunk_seq_ids,
            parameter=parameter,
            repo_root=repo_root,
            media_path=media_path,
            checkpoint_root=checkpoint_root,
        )
        valid_predictions.extend(chunk_preds)
        processed_valid += len(chunk_rows)
        done = min(total, base_done + processed_valid)
        print(f"Progress: {done}/{total}", flush=True)

    if len(valid_predictions) != len(valid_rows):
        raise RuntimeError(
            f"CatPred produced {len(valid_predictions)} predictions for {len(valid_rows)} rows."
        )

    valid_iter = iter(valid_predictions)
    invalid_set = set(invalid_indices)
    for idx in range(total):
        if idx in invalid_set:
            continue
        predictions[idx] = float(next(valid_iter))

    return {
        "predictions": predictions,
        "invalid_indices": sorted(set(invalid_indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatPred via the webKinPred subprocess contract.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = run_from_payload(payload)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[CatPred] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
