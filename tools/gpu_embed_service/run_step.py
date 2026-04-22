#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


STEP_CHOICES = (
    "kinform_t5_full",
    "kinform_esm2_layers",
    "kinform_esmc_layers",
    "prot_t5_mean",
    "turnup_esm1b",
    "eitlem_esm1v",
    "catpred_embed_kcat",
    "catpred_embed_km",
    # Deprecated: superseded by kinform_t5_full
    "kinform_pseq2sites",
    "kinform_prott5_layers",
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_in_home_env(env_name: str) -> str:
    return str((Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python").resolve())


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)


def _parse_seq_ids(raw: str) -> list[str]:
    return [sid.strip() for sid in raw.split(",") if sid.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _load_seq_id_to_seq(seq_ids: list[str], seqmap_db: Path) -> dict[str, str]:
    _ensure_exists(seqmap_db, "seqmap DB")
    if not seq_ids:
        return {}

    placeholders = ",".join(["?"] * len(seq_ids))
    sql = f"SELECT id, seq FROM sequences WHERE id IN ({placeholders})"
    # Use read-only URI mode to avoid WAL/lock file creation on NFS mounts.
    uri = f"file:{seqmap_db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        rows = con.execute(sql, seq_ids).fetchall()
    found = {str(row[0]): str(row[1]) for row in rows}
    missing = [sid for sid in seq_ids if sid not in found]
    if missing:
        raise RuntimeError(f"Missing seq IDs in seqmap DB: {','.join(missing)}")
    return {sid: found[sid] for sid in seq_ids}


def _write_temp_inputs(seq_id_to_seq: dict[str, str]) -> tuple[Path, Path, Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="gpu_embed_step_"))
    seq_file = tmp_dir / "seq_ids.txt"
    id_to_seq_pkl = tmp_dir / "id_to_seq.pkl"
    seq_map_json = tmp_dir / "seq_id_to_seq.json"

    with seq_file.open("w", encoding="utf-8") as handle:
        for seq_id in seq_id_to_seq.keys():
            handle.write(f"{seq_id}\n")

    with id_to_seq_pkl.open("wb") as handle:
        pickle.dump(seq_id_to_seq, handle, protocol=4)

    seq_map_json.write_text(json.dumps(seq_id_to_seq), encoding="utf-8")
    return tmp_dir, seq_file, id_to_seq_pkl, seq_map_json


def _kinform_env(repo_root: Path, media_path: Path, tools_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("KINFORM_MEDIA_PATH", str(media_path))
    env.setdefault("KINFORM_TOOLS_PATH", str(tools_path))
    env.setdefault("KINFORM_DATA", str((repo_root / "models" / "KinForm" / "results").resolve()))
    env.setdefault("KINFORM_ESM_PATH", _python_in_home_env("esm"))
    env.setdefault("KINFORM_ESMC_PATH", _python_in_home_env("esmc"))
    env.setdefault("KINFORM_T5_PATH", _python_in_home_env("prot_t5"))
    env.setdefault("KINFORM_PSEQ2SITES_PATH", _python_in_home_env("pseq2sites"))
    t5_model_default = (
        repo_root / "models" / "UniKP-main" / "models" / "protT5_xl" / "prot_t5_xl_uniref50"
    ).resolve()
    if t5_model_default.exists():
        env.setdefault("KINFORM_T5_MODEL_PATH", str(t5_model_default))
    # GPU embed service should run embedding workloads on CUDA, not CPU fallback.
    env.setdefault("KINFORM_REQUIRE_CUDA", "1")
    env.setdefault("GPU_REPO_ROOT", str(repo_root))
    return env


def _run_kinform_pseq2sites(env: dict[str, str], seq_map_json: Path) -> None:
    py = env["KINFORM_PSEQ2SITES_PATH"]
    code = r"""
import json
import os
import sys
from pathlib import Path
import pandas as pd

repo_root = Path(os.environ["GPU_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root / "models" / "KinForm" / "code"))

from pseq2sites.get_sites import get_sites
from config import BS_PRED_PATH

seq_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bs_path = Path(BS_PRED_PATH)
bs_path.parent.mkdir(parents=True, exist_ok=True)
if bs_path.exists():
    bs_df = pd.read_csv(bs_path, sep="\t")
else:
    bs_df = pd.DataFrame(columns=["PDB", "Pred_BS_Scores"])
get_sites(seq_map, bs_df, save_path=str(bs_path), return_prot_t5=False)
"""
    _run([py, "-c", code, str(seq_map_json)], env)


def _run_kinform_esm2(env: dict[str, str], seq_file: Path, id_to_seq_pkl: Path) -> None:
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    _run(
        [
            env["KINFORM_ESM_PATH"],
            str(script),
            "--seq_file",
            str(seq_file),
            "--models",
            "esm2",
            "--setting",
            "mean+weighted",
            "--weights_file",
            str(bs_pred),
            "--id_to_seq_file",
            str(id_to_seq_pkl),
            "--batch_size",
            "1",
        ],
        env,
    )


def _run_kinform_esmc(env: dict[str, str], seq_file: Path, id_to_seq_pkl: Path) -> None:
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    _run(
        [
            env["KINFORM_ESMC_PATH"],
            str(script),
            "--seq_file",
            str(seq_file),
            "--models",
            "esmc",
            "--setting",
            "mean+weighted",
            "--weights_file",
            str(bs_pred),
            "--id_to_seq_file",
            str(id_to_seq_pkl),
            "--batch_size",
            "1",
        ],
        env,
    )


def _run_kinform_t5(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    *,
    mean_only: bool,
) -> None:
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    cmd = [
        env["KINFORM_T5_PATH"],
        str(script),
        "--seq_file",
        str(seq_file),
        "--id_to_seq_file",
        str(id_to_seq_pkl),
        "--batch_size",
        "1",
    ]
    if mean_only:
        cmd.extend(["--setting", "mean", "--layers", "None"])
    else:
        cmd.extend(
            [
                "--setting",
                "mean+weighted",
                "--weights_file",
                str(bs_pred),
                "--layers",
                "19",
                "None",
            ]
        )
    _run(cmd, env)


def _run_kinform_t5_full_legacy_only(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    seq_map_json: Path,
) -> None:
    """Single T5 load path with early progress-visible cache writes.

    Flow:
    1) residue+mean for layers 19 + last (single model load)
    2) binding-site prediction from saved last-layer residue
    3) weighted vectors derived from residue files (no T5 reload)
    """
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    base_cmd = [
        env["KINFORM_T5_PATH"],
        str(script),
        "--seq_file", str(seq_file),
        "--id_to_seq_file", str(id_to_seq_pkl),
        "--batch_size", "1",
    ]

    # Phase 1: extract residue + mean for layers 19 + last in one model load.
    # Writing mean_vecs here lets embedding_progress track real progress while
    # weighted vectors wait for binding-site prediction.
    _run(base_cmd + ["--setting", "residue+mean", "--layers", "19", "None"], env)

    # Phase 2: predict binding sites — last-layer residue is already on disk, T5 is a no-op
    _run_kinform_pseq2sites(env, seq_map_json)

    # Phase 3: derive weighted from saved residue (Smart Reuse, no T5 reload).
    _run(
        base_cmd + [
            "--setting", "weighted",
            "--weights_file", str(bs_pred),
            "--layers", "19", "None",
        ],
        env,
    )


def _run_kinform_t5_full_legacy_with_esm(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    seq_map_json: Path,
) -> None:
    _run_kinform_t5_full_legacy_only(env, seq_file, id_to_seq_pkl, seq_map_json)
    _run_kinform_esm2(env, seq_file, id_to_seq_pkl)
    _run_kinform_esmc(env, seq_file, id_to_seq_pkl)


def _run_kinform_t5_full(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    seq_map_json: Path,
    *,
    seq_id_to_seq: dict[str, str],
    job_id: str | None = None,
) -> None:
    parallel_enabled = _env_bool("KINFORM_PARALLEL_EMBED_ENABLE", True)
    allow_fallback = _env_bool("KINFORM_PARALLEL_EMBED_ALLOW_LEGACY_FALLBACK", True)

    if not parallel_enabled:
        _run_kinform_t5_full_legacy_only(env, seq_file, id_to_seq_pkl, seq_map_json)
        return

    from kinform_parallel_orchestrator import run_kinform_parallel_pipeline

    try:
        run_kinform_parallel_pipeline(
            env=env,
            repo_root=Path(env["GPU_REPO_ROOT"]).resolve(),
            media_path=Path(env["KINFORM_MEDIA_PATH"]).resolve(),
            seq_id_to_seq=seq_id_to_seq,
            job_id=job_id,
        )
        return
    except Exception as exc:
        if not allow_fallback:
            raise
        print(
            "KINFORM_PARALLEL_FALLBACK "
            f"job_id={job_id or 'unknown'} "
            f"reason={exc.__class__.__name__}:{exc}"
        )
        _run_kinform_t5_full_legacy_with_esm(env, seq_file, id_to_seq_pkl, seq_map_json)


def _run_turnup(env: dict[str, str], seq_map_json: Path) -> None:
    turnup_env = dict(env)
    turnup_env.setdefault("TURNUP_MEDIA_PATH", env["KINFORM_MEDIA_PATH"])
    turnup_env.setdefault("TURNUP_TOOLS_PATH", env["KINFORM_TOOLS_PATH"])
    turnup_env.setdefault(
        "TURNUP_DATA_PATH",
        str((Path(env["GPU_REPO_ROOT"]) / "models" / "TurNup" / "data").resolve()),
    )
    turnup_python = (
        os.environ.get("TURNUP_EMBED_PYTHON")
        or os.environ.get("KINFORM_ESM_PATH")
        or _python_in_home_env("esm")
    )
    # seq_map_json is {seq_id: sequence} — use seq_ids directly, no seqmap CLI needed.
    code = r"""
import json, os, sys
import numpy as np
from pathlib import Path
import torch

seq_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))  # {seq_id: sequence}

media = os.environ.get("TURNUP_MEDIA_PATH") or os.environ.get("KINFORM_MEDIA_PATH")
vec_dir = Path(media) / "sequence_info" / "esm1b_turnup"
vec_dir.mkdir(parents=True, exist_ok=True)

missing = {sid: seq for sid, seq in seq_map.items() if not (vec_dir / f"{sid}.npy").exists()}
if not missing:
    print("All TurNup ESM1b embeddings already on disk — skipping model load.")
    sys.exit(0)

print(f"Loading ESM1b for {len(missing)} sequence(s)...")
import esm
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

for sid, seq in missing.items():
    seq_t = seq[:1022]  # ESM1b token limit
    _, _, tokens = batch_converter([(sid, seq_t)])
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
    rep = out["representations"][33][0, 0].cpu().numpy()
    np.save(vec_dir / f"{sid}.npy", rep)

print(f"Saved {len(missing)} TurNup ESM1b embedding(s).")
"""
    _run([turnup_python, "-c", code, str(seq_map_json)], turnup_env)


def _run_eitlem_esm1v(env: dict[str, str], seq_map_json: Path) -> None:
    """Compute ESM1v layer-33 per-residue representations for EITLEM on GPU.

    Saves the full residue matrix (seq_len × 1280, float32) to
    <media>/sequence_info/esm1v/<seq_id>.npy — the same format and path
    the EITLEM prediction script expects.  Files are ephemeral: the
    prediction script deletes them after all predictions are complete.
    """
    eitlem_python = (
        os.environ.get("EITLEM_EMBED_PYTHON")
        or os.environ.get("KINFORM_ESM_PATH")  # esm conda env includes ESM1v
        or _python_in_home_env("eitlem_env")
    )
    repo_root_str = env.get("GPU_REPO_ROOT", "")
    code = r"""
import json, os, sys
import numpy as np
from pathlib import Path
import torch

seq_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

media = os.environ.get("EITLEM_MEDIA_PATH") or os.environ.get("KINFORM_MEDIA_PATH")
if not media:
    raise RuntimeError("Neither EITLEM_MEDIA_PATH nor KINFORM_MEDIA_PATH is set.")

emb_dir = Path(media) / "sequence_info" / "esm1v"
emb_dir.mkdir(parents=True, exist_ok=True)

missing = {sid: seq for sid, seq in seq_map.items()
           if not (emb_dir / f"{sid}.npy").exists()}
if not missing:
    print(f"All {len(seq_map)} EITLEM ESM1v representation(s) already on disk — skipping.")
    sys.exit(0)

model_path = os.environ.get("EITLEM_MODEL_PATH")
if not model_path:
    repo_root = os.environ.get("GPU_REPO_ROOT", "")
    if not repo_root:
        raise RuntimeError(
            "EITLEM_MODEL_PATH is not set and GPU_REPO_ROOT is unavailable to derive it."
        )
    model_path = str(
        (Path(repo_root) / "models" / "EITLEM" / "Weights" / "esm1v"
         / "esm1v_t33_650M_UR90S_1.pt").resolve()
    )

print(f"Loading ESM1v 650M for {len(missing)} sequence(s) from {model_path} ...")
import esm as esm_lib
esm_model, alphabet = esm_lib.pretrained.load_model_and_alphabet_local(model_path)
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)
print(f"ESM1v running on {device}.")

for sid, seq in missing.items():
    # ESM1v max context is 1024 tokens (incl. <cls>/<eos>) → seq ≤ 1022.
    if len(seq) > 1022:
        seq = seq[:500] + seq[-500:]

    _, _, batch_tokens = batch_converter([("protein", seq)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_repr = results["representations"][33]
    tokens_len = batch_lens[0]
    # Strip <cls> (position 0) and <eos> (position tokens_len-1).
    # Shape: (seq_len, 1280) — full per-residue matrix, as the EITLEM model requires.
    residue_repr = token_repr[0, 1 : tokens_len - 1].cpu().numpy()
    np.save(emb_dir / f"{sid}.npy", residue_repr)

print(f"Saved {len(missing)} EITLEM ESM1v representation(s) to {emb_dir}.")
"""
    eitlem_env = dict(env)
    eitlem_env.setdefault("GPU_REPO_ROOT", repo_root_str)
    _run([eitlem_python, "-c", code, str(seq_map_json)], eitlem_env)


def _run_catpred_embed(parameter: str, env: dict[str, str], seq_map_json: Path) -> None:
    """Compute CatPred ESM2 + attention-pooled embeddings on GPU.

    Calls the standalone catpred_embed_gpu.py script which handles ESM2
    inference and per-checkpoint attentive pooling, saving pooled .pt tensors
    to the shared cache so the production adapter can skip embedding entirely.
    """
    catpred_python = (
        os.environ.get("CATPRED_EMBED_PYTHON")
        or _python_in_home_env("catpred_env")
    )
    repo_root = Path(env.get("GPU_REPO_ROOT", str(_default_repo_root()))).resolve()
    script = (
        repo_root / "models" / "CatPred" / "catpred" / "integration" / "catpred_embed_gpu.py"
    ).resolve()
    _ensure_exists(script, "catpred_embed_gpu.py")

    checkpoint_root = (
        os.environ.get("CATPRED_CHECKPOINT_ROOT")
        or str((repo_root / "models" / "CatPred" / ".e2e-assets" / "pretrained" / "production").resolve())
    )
    media_path = (
        env.get("CATPRED_MEDIA_PATH")
        or env.get("KINFORM_MEDIA_PATH")
        or os.environ.get("CATPRED_MEDIA_PATH")
        or os.environ.get("KINFORM_MEDIA_PATH", "")
    )
    cache_root = (
        os.environ.get("CATPRED_CACHE_ROOT")
        or (str((Path(media_path) / "sequence_info" / "catpred_esm2").resolve()) if media_path else "")
    )
    if not cache_root:
        raise RuntimeError(
            "Cannot determine CatPred cache root: set CATPRED_CACHE_ROOT or "
            "CATPRED_MEDIA_PATH / KINFORM_MEDIA_PATH."
        )

    catpred_env = dict(env)
    # Ensure the checkpoint root is trusted for torch.load deserialization.
    existing_roots = catpred_env.get("CATPRED_TRUSTED_DESERIALIZATION_ROOTS", "")
    trusted = [r for r in existing_roots.split(os.pathsep) if r.strip()]
    if checkpoint_root not in trusted:
        trusted.append(checkpoint_root)
    catpred_env["CATPRED_TRUSTED_DESERIALIZATION_ROOTS"] = os.pathsep.join(trusted)
    catpred_env.setdefault("CATPRED_ALLOW_UNSAFE_DESERIALIZATION", "1")

    _run(
        [
            catpred_python,
            str(script),
            "--seq-id-to-seq-file", str(seq_map_json),
            "--parameter", parameter,
            "--checkpoint-root", checkpoint_root,
            "--cache-root", cache_root,
        ],
        catpred_env,
    )


def run_step(
    step: str,
    seq_ids: list[str],
    repo_root: Path,
    media_path: Path,
    tools_path: Path,
    *,
    seq_id_to_seq: dict[str, str] | None = None,
    job_id: str | None = None,
) -> None:
    if not seq_ids:
        print("No sequence IDs provided; nothing to do.")
        return

    if seq_id_to_seq is None:
        seqmap_db = (media_path / "sequence_info" / "seqmap.sqlite3").resolve()
        seq_id_to_seq = _load_seq_id_to_seq(seq_ids, seqmap_db)
    else:
        missing = [sid for sid in seq_ids if sid not in seq_id_to_seq]
        if missing:
            raise RuntimeError(f"Missing seq IDs in provided map: {','.join(missing)}")

    tmp_dir, seq_file, id_to_seq_pkl, seq_map_json = _write_temp_inputs(seq_id_to_seq)
    env = _kinform_env(repo_root=repo_root, media_path=media_path, tools_path=tools_path)

    try:
        if step == "kinform_t5_full":
            _run_kinform_t5_full(
                env,
                seq_file,
                id_to_seq_pkl,
                seq_map_json,
                seq_id_to_seq=seq_id_to_seq,
                job_id=job_id,
            )
        elif step == "kinform_esm2_layers":
            _run_kinform_esm2(env, seq_file, id_to_seq_pkl)
        elif step == "kinform_esmc_layers":
            _run_kinform_esmc(env, seq_file, id_to_seq_pkl)
        elif step == "prot_t5_mean":
            _run_kinform_t5(env, seq_file, id_to_seq_pkl, mean_only=True)
        elif step == "turnup_esm1b":
            _run_turnup(env, seq_map_json)
        elif step == "eitlem_esm1v":
            _run_eitlem_esm1v(env, seq_map_json)
        elif step == "catpred_embed_kcat":
            _run_catpred_embed("kcat", env, seq_map_json)
        elif step == "catpred_embed_km":
            _run_catpred_embed("km", env, seq_map_json)
        elif step == "kinform_pseq2sites":
            _run_kinform_pseq2sites(env, seq_map_json)
        elif step == "kinform_prott5_layers":
            _run_kinform_t5(env, seq_file, id_to_seq_pkl, mean_only=False)
        else:
            raise RuntimeError(f"Unsupported step: {step}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one GPU embedding step for a seq-id batch.")
    parser.add_argument("--step", required=True, choices=STEP_CHOICES)
    parser.add_argument("--seq-ids", required=True, help="Comma-separated sequence IDs")
    parser.add_argument("--repo-root", default=os.environ.get("GPU_EMBED_REPO_ROOT", ""))
    parser.add_argument("--media-path", default=os.environ.get("KINFORM_MEDIA_PATH", ""))
    parser.add_argument("--tools-path", default=os.environ.get("KINFORM_TOOLS_PATH", ""))
    parser.add_argument(
        "--seq-id-to-seq-json",
        default="",
        help="JSON string mapping seq_id→sequence (skips seqmap DB lookup)",
    )
    parser.add_argument(
        "--seq-id-to-seq-file",
        default="",
        help="Path to JSON file mapping seq_id→sequence (skips seqmap DB lookup)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _default_repo_root().resolve()
    media_path = Path(args.media_path).resolve() if args.media_path else Path("/mnt/webkinpred/media").resolve()
    tools_path = Path(args.tools_path).resolve() if args.tools_path else (repo_root / "tools").resolve()
    seq_ids = _parse_seq_ids(args.seq_ids)

    seq_id_to_seq: dict[str, str] | None = None
    if args.seq_id_to_seq_file:
        seq_id_to_seq = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
    elif args.seq_id_to_seq_json:
        seq_id_to_seq = json.loads(args.seq_id_to_seq_json)

    run_step(
        step=args.step,
        seq_ids=seq_ids,
        repo_root=repo_root,
        media_path=media_path,
        tools_path=tools_path,
        seq_id_to_seq=seq_id_to_seq,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
