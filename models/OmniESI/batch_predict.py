#!/usr/bin/env python3
"""
OmniESI prediction wrapper for webKinPred generic subprocess engine.

Contract:
  python batch_predict.py --input <input.json> --output <output.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)
from tqdm import tqdm


def _resolve_code_root() -> Path:
    """Resolve OmniESI code root from env or repository layout."""
    script_dir = Path(__file__).resolve().parent
    env_root = Path(os.environ.get("OmniESI_ROOT", "/OmniESI")).resolve()
    candidates = [
        script_dir / "code",
        script_dir,
        env_root,
        env_root / "code",
    ]
    for candidate in candidates:
        if (candidate / "configs.py").exists() and (candidate / "models.py").exists():
            return candidate
    # Keep compatibility with container layouts where /OmniESI itself is code root.
    return env_root


ROOT_DIR = _resolve_code_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import OmniESI  # noqa: E402
from utils import set_seed  # noqa: E402
from configs import get_cfg_defaults  # noqa: E402
from scripts.embedding import ESM_model  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED_LIST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def _additional_data_dir() -> Path:
    env_value = os.environ.get("OmniESI_ADDITIONAL_DATA")
    if env_value:
        return Path(env_value).resolve()
    candidates = [
        ROOT_DIR / "OmniESI_additional_data" / "additional_data",
        ROOT_DIR.parent / "OmniESI_additional_data" / "additional_data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _safe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Input payload is malformed: 'rows' must be a list")

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Input payload is malformed: row {i} is not an object")
        out.append(row)
    return out


def _substrate_to_smiles(raw: Any) -> str | None:
    if isinstance(raw, list):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        text = str(raw).strip()
        if not text:
            return None
        tokens = [tok.strip() for tok in text.split(";") if tok.strip()]

    if len(tokens) != 1:
        return None
    return tokens[0]


def _kinetics_type_from_payload(payload: dict[str, Any]) -> str:
    params = payload.get("params") or {}
    ktype = str(params.get("kinetics_type", "")).upper().strip()
    if ktype in {"KCAT", "KM"}:
        return ktype

    target = str(payload.get("target", "")).strip()
    if target == "kcat":
        return "KCAT"
    if target == "Km":
        return "KM"

    raise RuntimeError("Missing kinetics type and unknown target in input payload.")


def prepare_inputs(smiles: str, protein_seq: str, embedder: ESM_model):
    try:
        atom_featurizer = CanonicalAtomFeaturizer()
        bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        fc = partial(smiles_to_bigraph, add_self_loop=True)
        v_d = fc(
            smiles=smiles,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
        )

        actual_node_feats = v_d.ndata.pop("h")
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = 0
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata["h"] = actual_node_feats

        virtual_node_feat = torch.cat(
            (torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)),
            1,
        )
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        v_d = dgl.batch([v_d])
        v_d_mask = torch.zeros(num_actual_nodes, dtype=torch.bool).unsqueeze(0)

        v_p = embedder([protein_seq])
        v_p = v_p[:, 1 : len(protein_seq) + 1, :]
        v_p_mask = torch.zeros(v_p.shape[1], dtype=torch.bool).unsqueeze(0)

        return v_d, v_p, v_d_mask, v_p_mask
    except Exception:
        return None


def _discover_available_seeds(weight_folder: Path) -> list[int]:
    available: list[int] = []
    for seed in SEED_LIST:
        weight_path = weight_folder / f"OmniESI_ensemble_{seed}" / "best_model_epoch.pth"
        if weight_path.exists():
            available.append(seed)
    return available


def predict_kinetic_parameter_ensemble(
    valid_df: pd.DataFrame,
    kinetics_type: str,
    esm_model: ESM_model,
) -> list[float | None]:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(ROOT_DIR / "configs" / "model" / "OmniESI_ensemble.yaml"))
    set_seed(42)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    additional_data_dir = _additional_data_dir()
    if kinetics_type == "KCAT":
        weight_folder = additional_data_dir / "results" / "CatPred_kcat" / "fold_OmniESI"
    elif kinetics_type == "KM":
        weight_folder = additional_data_dir / "results" / "CatPred_km" / "fold_OmniESI"
    else:
        raise RuntimeError(f"Unsupported kinetics type: {kinetics_type}")

    available_seeds = _discover_available_seeds(weight_folder)
    if not available_seeds:
        raise RuntimeError(f"No OmniESI checkpoints found under: {weight_folder}")

    n_rows = len(valid_df)
    total_ops = n_rows * len(available_seeds)
    ops_done = 0
    progress_done = -1

    predictions_per_seed: list[list[float | None]] = []

    for seed in available_seeds:
        model = OmniESI(**cfg)
        weight_path = weight_folder / f"OmniESI_ensemble_{seed}" / "best_model_epoch.pth"
        model.load_state_dict(torch.load(str(weight_path), map_location=device))
        model.to(device)
        model.eval()

        seed_predictions: list[float | None] = []
        torch.backends.cudnn.benchmark = True

        with torch.no_grad():
            iterator = tqdm(valid_df.iterrows(), total=n_rows, desc=f"Seed {seed}", leave=False)
            for _, row in iterator:
                smiles = row["smiles"]
                protein_seq = row["sequence"]

                result = prepare_inputs(smiles, protein_seq, esm_model)
                if result is None:
                    seed_predictions.append(None)
                else:
                    v_d, v_p, v_d_mask, v_p_mask = result
                    v_d = v_d.to(device)
                    v_p = v_p.to(device)
                    v_d_mask = v_d_mask.to(device)
                    v_p_mask = v_p_mask.to(device)
                    try:
                        output = model(v_d, v_p, v_d_mask, v_p_mask)
                        y = output[-1]
                        y_value = float(y.cpu().numpy().flatten()[0])
                        seed_predictions.append(y_value)
                    except Exception:
                        seed_predictions.append(None)

                ops_done += 1
                done = int((ops_done / max(total_ops, 1)) * n_rows)
                if done > progress_done:
                    progress_done = done
                    print(f"Progress: {done}/{n_rows}", flush=True)

        predictions_per_seed.append(seed_predictions)

    ensemble_predictions: list[float | None] = []
    for i in range(n_rows):
        sample_predictions = [
            seed_preds[i] for seed_preds in predictions_per_seed if seed_preds[i] is not None
        ]
        if sample_predictions:
            ensemble_predictions.append(float(np.mean(sample_predictions)))
        else:
            ensemble_predictions.append(None)

    if progress_done < n_rows:
        print(f"Progress: {n_rows}/{n_rows}", flush=True)

    return ensemble_predictions


def run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _safe_rows(payload)
    kinetics_type = _kinetics_type_from_payload(payload)

    predictions: list[float | None] = [None] * len(rows)
    invalid_indices: list[int] = []
    valid_indices: list[int] = []
    valid_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        seq = str(row.get("sequence", "")).strip()
        smiles = _substrate_to_smiles(row.get("substrates", row.get("substrate", "")))
        if not seq or smiles is None:
            invalid_indices.append(idx)
            continue
        valid_indices.append(idx)
        valid_rows.append({"sequence": seq, "smiles": smiles})

    if not valid_indices:
        total = len(rows)
        for i in range(total):
            print(f"Progress: {i + 1}/{total}", flush=True)
        return {"predictions": predictions, "invalid_indices": sorted(set(invalid_indices))}

    torch.cuda.empty_cache()
    try:
        esm_model = ESM_model()
        esm_model.device = device
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize ESM model: {exc}") from exc

    valid_df = pd.DataFrame(valid_rows)
    valid_preds = predict_kinetic_parameter_ensemble(
        valid_df=valid_df,
        kinetics_type=kinetics_type,
        esm_model=esm_model,
    )

    for local_idx, pred in enumerate(valid_preds):
        global_idx = valid_indices[local_idx]
        if pred is None:
            invalid_indices.append(global_idx)
        predictions[global_idx] = pred

    return {
        "predictions": predictions,
        "invalid_indices": sorted(set(invalid_indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniESI webKinPred batch adapter")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = run_from_payload(payload)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[OmniESI] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
