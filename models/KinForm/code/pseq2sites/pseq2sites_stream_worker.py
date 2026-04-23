#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.gpu_embed_service.kinform_stream_ipc import StreamClient


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


def _decode_array_payload(header: dict, payload: bytes) -> np.ndarray:
    dtype = np.dtype(str(header.get("dtype", "float32")))
    shape_raw = header.get("shape")
    if not isinstance(shape_raw, list) or not shape_raw:
        raise ValueError("Invalid stream shape header")
    shape = tuple(int(x) for x in shape_raw)
    arr = np.frombuffer(payload, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Payload element mismatch: got={arr.size} expected={expected}")
    return arr.reshape(shape).astype(np.float32, copy=False)


def _score_text_to_float_array(score_text: str) -> np.ndarray:
    vals = [float(x) for x in score_text.split(",") if x.strip()]
    return np.asarray(vals, dtype=np.float32)


def _run_legacy_polling(
    *,
    seq_id_to_seq: dict[str, str],
    binding_sites_path: Path,
    poll_interval_seconds: float,
    batch_size: int,
    trainiter,
    config: dict,
    PocketDatasetCls,
    DataloaderFn,
    prepare_prots_input_fn,
) -> int:
    media_path = Path(os.environ["KINFORM_MEDIA_PATH"]).resolve()
    residue_dir = (media_path / "sequence_info" / "prot_t5_last" / "residue_vecs").resolve()
    residue_dir.mkdir(parents=True, exist_ok=True)

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
            time.sleep(max(0.05, float(poll_interval_seconds)))
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
            batch_size=max(1, int(batch_size)),
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


def _run_stream_mode(
    *,
    seq_id_to_seq: dict[str, str],
    binding_sites_path: Path,
    batch_size: int,
    stream_socket: str,
    stream_job_id: str,
    worker_name: str,
    trainiter,
    config: dict,
    PocketDatasetCls,
    DataloaderFn,
    prepare_prots_input_fn,
) -> int:
    stream = StreamClient(stream_socket)
    stream.send(
        {
            "type": "PSEQ_REGISTER",
            "worker": worker_name,
            "job_id": stream_job_id,
        },
        b"",
    )

    existing = _read_binding_site_rows(binding_sites_path)
    pending: dict[str, str] = {
        sid: seq for sid, seq in seq_id_to_seq.items() if sid not in existing
    }
    if not pending:
        stream.send(
            {
                "type": "WORKER_DONE",
                "worker": worker_name,
                "job_id": stream_job_id,
            },
            b"",
        )
        print("PSEQ_STREAM all sequence IDs already present in binding-site cache.")
        stream.close()
        return 0

    print(f"PSEQ_STREAM pending_count={len(pending)}")

    queue_size_raw = str(os.environ.get("KINFORM_PARALLEL_PSEQ_STREAM_QUEUE_SIZE", "")).strip()
    if queue_size_raw:
        try:
            queue_size = max(4, int(queue_size_raw))
        except Exception:
            queue_size = max(32, int(batch_size) * 8)
    else:
        queue_size = max(32, int(batch_size) * 8)

    recv_queue: queue.Queue = queue.Queue(maxsize=queue_size)
    recv_stop = threading.Event()
    receiver_exc: list[str] = []

    def _queue_put(item) -> None:
        while not recv_stop.is_set():
            try:
                recv_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _receiver_loop() -> None:
        try:
            while not recv_stop.is_set():
                try:
                    header, payload = stream.recv(timeout_seconds=0.2)
                except TimeoutError:
                    continue
                except EOFError:
                    _queue_put(("finish", "", "", None))
                    return

                evt_type = str(header.get("type", "")).strip().upper()
                if evt_type == "PSEQ_RESIDUE":
                    seq_id = str(header.get("seq_id", "")).strip()
                    if not seq_id:
                        continue
                    sequence = str(header.get("sequence", ""))
                    arr = _decode_array_payload(header, payload)
                    _queue_put(("residue", seq_id, sequence, arr))
                    continue
                if evt_type == "PSEQ_FINISH":
                    _queue_put(("finish", "", "", None))
                    return
        except Exception as exc:
            receiver_exc.append(str(exc))
            _queue_put(("error", "", "", str(exc)))

    receiver_thread = threading.Thread(
        target=_receiver_loop,
        name=f"{worker_name}-stream-recv",
        daemon=True,
    )
    receiver_thread.start()

    buffer_feats: dict[str, np.ndarray] = {}
    buffer_seqs: dict[str, str] = {}
    finish_received = False
    last_buffer_add = time.monotonic()
    idle_flush_seconds = max(0.05, float(os.environ.get("KINFORM_PARALLEL_PSEQ_STREAM_IDLE_FLUSH_SECONDS", "0.2")))
    persist_every_rows = max(
        1,
        int(os.environ.get("KINFORM_PARALLEL_PSEQ_STREAM_PERSIST_EVERY_ROWS", "64")),
    )
    persist_every_seconds = max(
        0.2,
        float(os.environ.get("KINFORM_PARALLEL_PSEQ_STREAM_PERSIST_EVERY_SECONDS", "15.0")),
    )
    pending_persist_rows: dict[str, str] = {}
    last_persist = time.monotonic()

    def persist_rows(*, force: bool) -> int:
        nonlocal pending_persist_rows, last_persist
        if not pending_persist_rows:
            return 0
        now = time.monotonic()
        if not force:
            if len(pending_persist_rows) < persist_every_rows and (now - last_persist) < persist_every_seconds:
                return 0
        started = time.monotonic()
        merge_binding_site_rows_atomic(binding_sites_path, pending_persist_rows)
        wrote = len(pending_persist_rows)
        pending_persist_rows = {}
        last_persist = time.monotonic()
        print(
            f"PSEQ_STREAM persisted={wrote} cache={binding_sites_path} "
            f"persist_s={last_persist - started:.3f}"
        )
        return wrote

    def flush_buffer() -> int:
        nonlocal buffer_feats, buffer_seqs, pending_persist_rows
        if not buffer_feats:
            return 0
        ready_map = {sid: buffer_seqs[sid] for sid in buffer_feats}
        infer_started = time.monotonic()
        pred_rows = _predict_binding_scores(
            trainiter=trainiter,
            config=config,
            seq_id_to_seq=ready_map,
            seq_id_to_feats=buffer_feats,
            PocketDatasetCls=PocketDatasetCls,
            DataloaderFn=DataloaderFn,
            prepare_prots_input_fn=prepare_prots_input_fn,
            batch_size=max(1, int(batch_size)),
        )
        infer_elapsed = time.monotonic() - infer_started
        pending_persist_rows.update(pred_rows)
        persist_rows(force=False)
        for seq_id, score_text in pred_rows.items():
            pending.pop(seq_id, None)
            arr = _score_text_to_float_array(score_text)
            stream.send(
                {
                    "type": "BS_READY",
                    "worker": worker_name,
                    "job_id": stream_job_id,
                    "seq_id": seq_id,
                    "dtype": "float32",
                    "shape": [int(arr.shape[0])],
                },
                arr.tobytes(order="C"),
            )
        wrote = len(pred_rows)
        buffer_feats = {}
        buffer_seqs = {}
        print(
            f"PSEQ_STREAM wrote={wrote} remaining={len(pending)} cache={binding_sites_path} "
            f"infer_s={infer_elapsed:.3f}"
        )
        return wrote

    def handle_queue_item(item) -> None:
        nonlocal finish_received, last_buffer_add
        kind, seq_id, sequence, payload_obj = item
        if kind == "residue":
            if seq_id not in pending:
                return
            if not isinstance(payload_obj, np.ndarray):
                raise RuntimeError(f"Invalid residue payload for seq_id={seq_id}")
            buffer_feats[seq_id] = payload_obj
            buffer_seqs[seq_id] = sequence or pending[seq_id]
            last_buffer_add = time.monotonic()
            return
        if kind == "finish":
            finish_received = True
            return
        if kind == "error":
            raise RuntimeError(f"Pseq stream receiver failed: {payload_obj}")
        raise RuntimeError(f"Unknown receiver queue item kind={kind}")

    try:
        while pending:
            got_data = False
            try:
                item = recv_queue.get(timeout=0.2)
                handle_queue_item(item)
                got_data = True
            except queue.Empty:
                pass

            # Drain available queue items quickly to keep socket backpressure low.
            while True:
                try:
                    item = recv_queue.get_nowait()
                except queue.Empty:
                    break
                handle_queue_item(item)
                got_data = True
                if len(buffer_feats) >= max(1, int(batch_size)):
                    break

            if receiver_exc:
                raise RuntimeError(f"Pseq stream receiver thread error: {receiver_exc[-1]}")

            if len(buffer_feats) >= max(1, int(batch_size)):
                flush_buffer()
                continue

            if buffer_feats and (
                finish_received
                or (
                    (not got_data)
                    and (time.monotonic() - last_buffer_add >= idle_flush_seconds)
                )
            ):
                flush_buffer()
                continue

            if finish_received and not buffer_feats:
                if pending:
                    missing = sorted(pending.keys())
                    raise RuntimeError(f"Received PSEQ_FINISH with pending sequence IDs: {missing}")
                break

        persist_rows(force=True)
        stream.send(
            {
                "type": "WORKER_DONE",
                "worker": worker_name,
                "job_id": stream_job_id,
            },
            b"",
        )
        print("PSEQ_STREAM completed all pending sequence IDs.")
        return 0
    except Exception as exc:
        try:
            stream.send(
                {
                    "type": "WORKER_ERROR",
                    "worker": worker_name,
                    "job_id": stream_job_id,
                    "message": str(exc),
                },
                b"",
            )
        except Exception:
            pass
        raise
    finally:
        try:
            persist_rows(force=True)
        except Exception:
            pass
        recv_stop.set()
        if receiver_thread.is_alive():
            receiver_thread.join(timeout=1.0)
        stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming Pseq2Sites GPU worker")
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--binding-sites-path", required=True, type=str)
    parser.add_argument("--poll-interval-seconds", default=0.5, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--stream-mode", action="store_true")
    parser.add_argument("--stream-socket", default="", type=str)
    parser.add_argument("--stream-job-id", default="", type=str)
    parser.add_argument("--worker-name", default="pseq2sites", type=str)
    args = parser.parse_args()

    seq_id_to_seq = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
    if not seq_id_to_seq:
        print("PSEQ_STREAM no sequence IDs provided.")
        return 0

    binding_sites_path = Path(args.binding_sites_path).resolve()

    pseq_root, load_cfg, TrainIterCls, PocketDatasetCls, DataloaderFn, prepare_prots_input_fn = _prepare_runtime()
    config_path = (pseq_root / "configuration_temp.yml").resolve()
    config = load_cfg(str(config_path))
    model_path = (pseq_root / "results" / "model").resolve()
    config["paths"]["model_path"] = str(model_path)
    config["train"]["batch_size"] = max(1, int(args.batch_size))

    print(f"PSEQ_STREAM loading model from {model_path}")
    trainiter = _load_model_once(config, model_path, TrainIterCls)
    print(f"PSEQ_STREAM model loaded on device={trainiter.device}")

    if args.stream_mode:
        if not args.stream_socket:
            raise RuntimeError("--stream-socket is required when --stream-mode is enabled")
        return _run_stream_mode(
            seq_id_to_seq=seq_id_to_seq,
            binding_sites_path=binding_sites_path,
            batch_size=max(1, int(args.batch_size)),
            stream_socket=args.stream_socket,
            stream_job_id=args.stream_job_id,
            worker_name=args.worker_name,
            trainiter=trainiter,
            config=config,
            PocketDatasetCls=PocketDatasetCls,
            DataloaderFn=DataloaderFn,
            prepare_prots_input_fn=prepare_prots_input_fn,
        )

    return _run_legacy_polling(
        seq_id_to_seq=seq_id_to_seq,
        binding_sites_path=binding_sites_path,
        poll_interval_seconds=max(0.05, float(args.poll_interval_seconds)),
        batch_size=max(1, int(args.batch_size)),
        trainiter=trainiter,
        config=config,
        PocketDatasetCls=PocketDatasetCls,
        DataloaderFn=DataloaderFn,
        prepare_prots_input_fn=prepare_prots_input_fn,
    )


if __name__ == "__main__":
    raise SystemExit(main())
