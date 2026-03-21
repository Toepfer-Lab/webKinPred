#!/usr/bin/env python3
"""
Build MMseqs target databases for configured similarity datasets.

Examples
--------
Build all configured datasets:
    python tools/build_similarity_dbs.py --all

Build one dataset:
    python tools/build_similarity_dbs.py --dataset "TurNup"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_conda_path() -> str | None:
    env_override = os.environ.get("WEBKINPRED_CONDA_PATH")
    if env_override:
        return env_override
    default = Path.home() / "anaconda3" / "bin" / "conda"
    return str(default) if default.exists() else None


def _load_similarity_registry() -> dict[str, dict[str, str]]:
    registry_path = REPO_ROOT / "webKinPred" / "similarity_dataset_registry.py"
    spec = importlib.util.spec_from_file_location("similarity_dataset_registry", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load registry file: {registry_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = getattr(module, "SIMILARITY_DATASET_REGISTRY", {})
    if not isinstance(registry, dict):
        raise RuntimeError("SIMILARITY_DATASET_REGISTRY must be a dict.")
    return registry


CONDA_PATH = _default_conda_path()
FASTAS_DIR = Path(os.environ.get("WEBKINPRED_FASTAS_DIR", str(REPO_ROOT / "fastas"))).resolve()


def _mmseqs_cmd(*args: str) -> list[str]:
    if CONDA_PATH:
        return [CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", *args]
    return ["mmseqs", *args]


def _datasets() -> dict[str, dict[str, str]]:
    registry = _load_similarity_registry()
    datasets: dict[str, dict[str, str]] = {}
    for label, meta in registry.items():
        fasta_filename = meta.get("fasta_filename")
        db_name = meta.get("db_name")
        if not fasta_filename or not db_name:
            raise ValueError(f"Dataset '{label}' is missing fasta_filename or db_name in registry.")
        datasets[label] = {
            "label": label,
            "fasta": str(FASTAS_DIR / fasta_filename),
            "target_db": str(FASTAS_DIR / "dbs" / db_name),
        }
    return datasets


def _build_one(label: str, dataset: dict) -> None:
    fasta = dataset.get("fasta")
    target_db = dataset.get("target_db")
    if not fasta or not target_db:
        raise ValueError(f"Dataset '{label}' is missing fasta or target_db path.")
    if not os.path.exists(fasta):
        raise FileNotFoundError(f"FASTA not found for '{label}': {fasta}")

    os.makedirs(os.path.dirname(target_db), exist_ok=True)
    cmd = _mmseqs_cmd("createdb", fasta, target_db)
    print(f"\n==> Building: {label}")
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] Built DB: {target_db}")


def parse_args() -> argparse.Namespace:
    datasets = _datasets()
    parser = argparse.ArgumentParser(description="Build MMseqs DBs for configured similarity datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(datasets.keys()),
        help="Dataset label to build (can be passed multiple times).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all configured datasets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured datasets and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = _datasets()

    if args.list:
        for label, ds in datasets.items():
            print(f"- {label}")
            print(f"  fasta: {ds.get('fasta')}")
            print(f"  target_db: {ds.get('target_db')}")
        return 0

    selected_labels: list[str]
    if args.all:
        selected_labels = list(datasets.keys())
    elif args.dataset:
        selected_labels = args.dataset
    else:
        print("Specify --all or at least one --dataset.", file=sys.stderr)
        return 2

    for label in selected_labels:
        _build_one(label, datasets[label])

    print("\nAll selected similarity databases were built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
