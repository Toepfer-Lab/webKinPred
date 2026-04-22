#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def _read_binding_site_rows(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return {}
        key_col = "PDB" if "PDB" in reader.fieldnames else reader.fieldnames[0]
        value_col = (
            "Pred_BS_Scores"
            if "Pred_BS_Scores" in reader.fieldnames
            else (reader.fieldnames[1] if len(reader.fieldnames) > 1 else "")
        )
        if not value_col:
            return {}
        for row in reader:
            seq_id = str(row.get(key_col, "")).strip()
            scores = str(row.get(value_col, "")).strip()
            if seq_id and scores:
                out[seq_id] = scores
    return out


def merge_binding_site_rows_atomic(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_binding_site_rows(path)
    ordered_ids = list(existing.keys())
    for seq_id, scores in updates.items():
        if seq_id not in existing:
            ordered_ids.append(seq_id)
        existing[seq_id] = scores

    fd, tmp_name = None, None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fd = None
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["PDB", "Pred_BS_Scores"])
            for seq_id in ordered_ids:
                value = existing.get(seq_id)
                if value:
                    writer.writerow([seq_id, value])
        os.replace(tmp_name, path)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_name and Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


def _prepare_runtime():
    here = Path(__file__).resolve().parent
    pseq_root = (here / "Pseq2Sites").resolve()
    import sys

    if str(pseq_root) not in sys.path:
        sys.path.insert(0, str(pseq_root))

    from modules.TrainIters import Pseq2SitesTrainIter  # type: ignore
    from modules.data import Dataloader, PocketDataset  # type: ignore
    from modules.helpers import prepare_prots_input  # type: ignore
    from modules.utils import load_cfg  # type: ignore

    return pseq_root, load_cfg, Pseq2SitesTrainIter, PocketDataset, Dataloader, prepare_prots_input


def _load_model_once(config: dict, model_path: Path, TrainIterCls):
    import torch

    trainiter = TrainIterCls(config)
    checkpoint_path = model_path / "Pseq2Sites.pth"
    checkpoint = torch.load(checkpoint_path, map_location=trainiter.device)
    trainiter.model.load_state_dict(checkpoint["state_dict"])
    trainiter.model.eval()
    return trainiter


def _resolve_features(residue_dir: Path, seq_id: str, sequence: str) -> np.ndarray | None:
    parts = str(sequence).split(",")
    if len(parts) <= 1:
        seq_path = residue_dir / f"{seq_id}.npy"
        if not seq_path.exists():
            return None
        return np.load(seq_path).astype(np.float32, copy=False)

    # Multi-chain fallback:
    # Prefer chain-specific files when present; otherwise use <seq_id>.npy.
    chain_features: list[np.ndarray] = []
    for idx, _ in enumerate(parts):
        chain_path = residue_dir / f"{seq_id}__c{idx}.npy"
        if chain_path.exists():
            chain_features.append(np.load(chain_path).astype(np.float32, copy=False))
        else:
            full_seq_path = residue_dir / f"{seq_id}.npy"
            if not full_seq_path.exists():
                return None
            return np.load(full_seq_path).astype(np.float32, copy=False)
    return np.concatenate(chain_features, axis=0).astype(np.float32, copy=False)


def _predict_binding_scores(
    *,
    trainiter,
    config: dict,
    seq_id_to_seq: dict[str, str],
    seq_id_to_feats: dict[str, np.ndarray],
    PocketDatasetCls,
    DataloaderFn,
    prepare_prots_input_fn,
    batch_size: int,
) -> dict[str, str]:
    import torch

    seq_ids = list(seq_id_to_feats.keys())
    feats = [seq_id_to_feats[sid] for sid in seq_ids]
    seqs = [seq_id_to_seq[sid] for sid in seq_ids]
    dataset = PocketDatasetCls(seq_ids, feats, seqs)
    loader = DataloaderFn(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pred_rows: dict[str, str] = {}
    with torch.no_grad():
        for batch in loader:
            aa_feats, prot_feats, prot_masks, position_ids, chain_idx = prepare_prots_input_fn(
                config,
                batch,
                training=False,
                device=trainiter.device,
            )
            _, pred_bs, _ = trainiter.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)
            pred_bs = pred_bs * prot_masks
            probs = torch.sigmoid(pred_bs).detach().cpu().numpy()
            for item, score_arr in zip(batch, probs):
                seq_id = str(item[0])
                seq_len = int(seq_id_to_feats[seq_id].shape[0])
                values = ",".join(f"{float(x):.6f}" for x in score_arr[:seq_len])
                pred_rows[seq_id] = values
    return pred_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming Pseq2Sites GPU worker")
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--binding-sites-path", required=True, type=str)
    parser.add_argument("--poll-interval-seconds", default=0.5, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()

    seq_id_to_seq = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
    if not seq_id_to_seq:
        print("PSEQ_STREAM no sequence IDs provided.")
        return 0

    media_path = Path(os.environ["KINFORM_MEDIA_PATH"]).resolve()
    binding_sites_path = Path(args.binding_sites_path).resolve()
    residue_dir = (media_path / "sequence_info" / "prot_t5_last" / "residue_vecs").resolve()
    residue_dir.mkdir(parents=True, exist_ok=True)

    pseq_root, load_cfg, TrainIterCls, PocketDatasetCls, DataloaderFn, prepare_prots_input_fn = _prepare_runtime()
    config_path = (pseq_root / "configuration_temp.yml").resolve()
    config = load_cfg(str(config_path))
    model_path = (pseq_root / "results" / "model").resolve()
    config["paths"]["model_path"] = str(model_path)
    config["train"]["batch_size"] = max(1, int(args.batch_size))

    print(f"PSEQ_STREAM loading model from {model_path}")
    trainiter = _load_model_once(config, model_path, TrainIterCls)
    print(f"PSEQ_STREAM model loaded on device={trainiter.device}")

    existing = _read_binding_site_rows(binding_sites_path)
    pending: dict[str, str] = {
        sid: seq for sid, seq in seq_id_to_seq.items() if sid not in existing
    }
    if not pending:
        print("PSEQ_STREAM all sequence IDs already present in binding-site cache.")
        return 0

    print(f"PSEQ_STREAM pending_count={len(pending)}")
    while pending:
        ready_feats: dict[str, np.ndarray] = {}
        for seq_id, sequence in pending.items():
            feats = _resolve_features(residue_dir, seq_id, sequence)
            if feats is not None:
                ready_feats[seq_id] = feats

        if not ready_feats:
            time.sleep(max(0.05, float(args.poll_interval_seconds)))
            continue

        ready_map = {sid: pending[sid] for sid in ready_feats}
        pred_rows = _predict_binding_scores(
            trainiter=trainiter,
            config=config,
            seq_id_to_seq=ready_map,
            seq_id_to_feats=ready_feats,
            PocketDatasetCls=PocketDatasetCls,
            DataloaderFn=DataloaderFn,
            prepare_prots_input_fn=prepare_prots_input_fn,
            batch_size=max(1, int(args.batch_size)),
        )
        merge_binding_site_rows_atomic(binding_sites_path, pred_rows)
        for seq_id in pred_rows:
            pending.pop(seq_id, None)
        print(
            f"PSEQ_STREAM wrote={len(pred_rows)} remaining={len(pending)} "
            f"cache={binding_sites_path}"
        )

    print("PSEQ_STREAM completed all pending sequence IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
