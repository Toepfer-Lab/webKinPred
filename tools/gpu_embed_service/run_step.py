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
    "kinform_pseq2sites",
    "kinform_esm2_layers",
    "kinform_esmc_layers",
    "kinform_prott5_layers",
    "prot_t5_mean",
    "turnup_esm1b",
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


def _load_seq_id_to_seq(seq_ids: list[str], seqmap_db: Path) -> dict[str, str]:
    _ensure_exists(seqmap_db, "seqmap DB")
    if not seq_ids:
        return {}

    placeholders = ",".join(["?"] * len(seq_ids))
    sql = f"SELECT id, seq FROM sequences WHERE id IN ({placeholders})"
    with sqlite3.connect(str(seqmap_db)) as con:
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
    code = r"""
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["GPU_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root / "models" / "TurNup" / "code"))
from enzyme_representations import calcualte_esm1b_ts_vectors

seq_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seqs = list(dict.fromkeys(seq_map.values()))
calcualte_esm1b_ts_vectors(seqs)
"""
    _run([turnup_python, "-c", code, str(seq_map_json)], turnup_env)


def run_step(step: str, seq_ids: list[str], repo_root: Path, media_path: Path, tools_path: Path) -> None:
    if not seq_ids:
        print("No sequence IDs provided; nothing to do.")
        return

    seqmap_db = (media_path / "sequence_info" / "seqmap.sqlite3").resolve()
    seq_id_to_seq = _load_seq_id_to_seq(seq_ids, seqmap_db)
    tmp_dir, seq_file, id_to_seq_pkl, seq_map_json = _write_temp_inputs(seq_id_to_seq)
    env = _kinform_env(repo_root=repo_root, media_path=media_path, tools_path=tools_path)

    try:
        if step == "kinform_pseq2sites":
            _run_kinform_pseq2sites(env, seq_map_json)
        elif step == "kinform_esm2_layers":
            _run_kinform_esm2(env, seq_file, id_to_seq_pkl)
        elif step == "kinform_esmc_layers":
            _run_kinform_esmc(env, seq_file, id_to_seq_pkl)
        elif step == "kinform_prott5_layers":
            _run_kinform_t5(env, seq_file, id_to_seq_pkl, mean_only=False)
        elif step == "prot_t5_mean":
            _run_kinform_t5(env, seq_file, id_to_seq_pkl, mean_only=True)
        elif step == "turnup_esm1b":
            _run_turnup(env, seq_map_json)
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
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _default_repo_root().resolve()
    media_path = Path(args.media_path).resolve() if args.media_path else Path("/mnt/webkinpred/media").resolve()
    tools_path = Path(args.tools_path).resolve() if args.tools_path else (repo_root / "tools").resolve()
    seq_ids = _parse_seq_ids(args.seq_ids)

    run_step(
        step=args.step,
        seq_ids=seq_ids,
        repo_root=repo_root,
        media_path=media_path,
        tools_path=tools_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
